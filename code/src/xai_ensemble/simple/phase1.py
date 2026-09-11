"""Phase 1: generate and publish complete signed attribution maps once."""

from __future__ import annotations

import gc
import os
import posixpath
import random
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase1.explainers import (
    EXPLAINER_SPECS,
    build_explainer,
    compute_attribution,
)
from xai_ensemble.phase1.relprop import (
    relprop_attribution_provider,
    relprop_required,
)

from .artifacts import (
    PHASE1_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    phase1_artifact_root,
    write_phase1_shard,
    write_phase2_shard,
)
from .config import Phase1Task, SimpleExperiment
from .data import (
    BatchedTensor,
    fixed_baselines,
    load_model,
    load_raw_dataset_mean,
    load_relprop_model,
    load_split,
    materialize_clean_model_inputs,
    materialize_model_inputs,
)
from .io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from .manifest_identity import dataset_manifest_identity_sha256
from .profiler import captum_params, load_profile
from .rank_ready import (
    build_rank_ready_tensors,
    compact_rank_input_fp32_equivalence,
    simpleavg_spatial_field,
    supported_patch_sizes,
    write_rank_ready_sidecar,
)
from .relprop_equivalence import ensure_relprop_equivalence_certificate
from .runtime import emit_gpu_release_signal
from .spool import SpoolQuota, SpoolReservation, UploadGate
from .telemetry import GpuUtilizationSampler, StageTimings

CLEAN_MODEL_TARGET_POLICY = "clean_model_fp32_prediction"
FULL_REFERENCE_CLEAN_TARGET_POLICY = "full_reference_clean_fp32_prediction"
TASK_MODEL_OUTPUT_SOURCE = "task_model_condition_fp32"
FULL_REFERENCE_MODEL_OUTPUT_SOURCE = "full_reference_model_condition_fp32"


def _model_output_field_descriptions(source: str) -> tuple[str, str]:
    labels = {
        TASK_MODEL_OUTPUT_SOURCE: "task model",
        FULL_REFERENCE_MODEL_OUTPUT_SOURCE: "full-reference model",
    }
    try:
        label = labels[source]
    except KeyError as error:
        raise ValueError(f"Unknown Phase 1 model output source: {source}") from error
    return (
        f"int64[N] argmax predictions from the {label} FP32 condition forward",
        f"float32[N,num_classes] logits from the {label} FP32 condition forward",
    )


def _resize_to_input(attributions: Any, height: int, width: int) -> Any:
    if getattr(attributions, "ndim", None) != 4:
        raise ValueError(f"Explainer must return [N,C,H,W], got {tuple(attributions.shape)}")
    if tuple(attributions.shape[-2:]) == (height, width):
        return attributions
    import torch.nn.functional as functional

    return functional.interpolate(
        attributions,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


def _seed_attribution(seed: int, *, device: Any) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _digest_path(path: Path) -> str:
    path = resolve_full_matrix_runtime_path(path)
    if path.is_file():
        return file_sha256(path)
    manifest = path / "manifest.json"
    if manifest.is_file():
        return file_sha256(manifest)
    raise FileNotFoundError(path)


def _source_identity(
    experiment: SimpleExperiment,
    task: Phase1Task,
    store: ArtifactStore,
) -> Mapping[str, str]:
    checkpoint = (
        object_sha256(
            {
                "init_mode": task.model.init_mode,
                "model": task.model.model_key,
                "num_classes": task.model.num_classes,
                "class_index_map": task.model.class_index_map,
            }
        )
        if task.model.checkpoint_path is None
        else file_sha256(resolve_full_matrix_runtime_path(task.model.checkpoint_path))
    )
    values = {
        "dataset_manifest_sha256": dataset_manifest_identity_sha256(task.dataset.manifest_path),
        "checkpoint_sha256": checkpoint,
        "mean_artifact_digest": _digest_path(task.model.mean_path),
    }
    attribution_provider = relprop_attribution_provider(
        task.family,
        task.model.architecture,
    )
    values.update(
        {f"attribution_provider_{key}": value for key, value in attribution_provider.items()}
    )
    if task.condition.kind == "adversarial":
        from .adversarial import adversarial_source_binding

        attack_task = experiment.adversarial_task_for(
            dataset_id=task.dataset.dataset_id,
            model_id=task.model.model_id,
            split=task.split,
            condition_id=task.condition.condition_id,
        )
        values.update(adversarial_source_binding(experiment, attack_task, store=store))
    return {**values, "digest": object_sha256(values)}


def _validate_completed_source(
    manifest: Mapping[str, Any],
    *,
    variant: Any,
    source_identity: Mapping[str, str],
    target_policy: str,
    model_output_source: str,
    allow_legacy_model_output_source: bool = False,
) -> None:
    method = manifest.get("method")
    dataset = manifest.get("dataset")
    model = manifest.get("model")
    baseline = manifest.get("baseline")
    fields = manifest.get("safetensors_fields")
    manifest_source_identity = manifest.get("source_identity")
    prediction_description, logit_description = _model_output_field_descriptions(
        model_output_source
    )
    checks = {
        "variant_digest": (
            method.get("variant_digest") if isinstance(method, Mapping) else None,
            variant.digest,
        ),
        "dataset_manifest_sha256": (
            dataset.get("manifest_sha256") if isinstance(dataset, Mapping) else None,
            source_identity["dataset_manifest_sha256"],
        ),
        "checkpoint_sha256": (
            model.get("checkpoint_sha256") if isinstance(model, Mapping) else None,
            source_identity["checkpoint_sha256"],
        ),
        "mean_artifact_digest": (
            baseline.get("mean_artifact_digest") if isinstance(baseline, Mapping) else None,
            source_identity["mean_artifact_digest"],
        ),
        "source_identity_digest": (
            manifest.get("source_identity_digest"),
            source_identity["digest"],
        ),
        "target_policy": (manifest.get("target_policy"), target_policy),
    }
    observed_output_source = manifest.get("model_output_source")
    legacy_output_metadata = (
        allow_legacy_model_output_source
        and model_output_source == TASK_MODEL_OUTPUT_SOURCE
        and observed_output_source is None
    )
    if not legacy_output_metadata:
        checks.update(
            {
                "model_output_source": (observed_output_source, model_output_source),
                "safetensors_fields.predictions": (
                    fields.get("predictions") if isinstance(fields, Mapping) else None,
                    prediction_description,
                ),
                "safetensors_fields.logits": (
                    fields.get("logits") if isinstance(fields, Mapping) else None,
                    logit_description,
                ),
            }
        )
    for name, expected in (
        ("target_policy", target_policy),
        ("model_output_source", model_output_source),
    ):
        identity_value = source_identity.get(name)
        if identity_value is None:
            continue
        if identity_value != expected:
            raise ValueError(
                f"Phase 1 {name} contradicts the current source identity: "
                f"{expected!r} != {identity_value!r}"
            )
        checks[f"source_identity.{name}"] = (
            manifest_source_identity.get(name)
            if isinstance(manifest_source_identity, Mapping)
            else None,
            identity_value,
        )
    expected_provider = relprop_attribution_provider(
        variant.family,
        variant.architecture,
    )
    if expected_provider:
        observed_provider = (
            model.get("attribution_provider") if isinstance(model, Mapping) else None
        )
        for key, expected in expected_provider.items():
            checks[f"attribution_provider.{key}"] = (
                observed_provider.get(key) if isinstance(observed_provider, Mapping) else None,
                expected,
            )
    mismatches = {
        name: {"artifact": observed, "current": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise ArtifactError(f"Completed artifact source identity changed: {mismatches}")


def _predict(
    model: Any,
    inputs: Any,
    *,
    device: Any,
    batch_size: int,
) -> tuple[Any, Any]:
    import torch

    logits_chunks = []
    prediction_chunks = []
    with torch.inference_mode():
        for start in range(0, int(inputs.shape[0]), batch_size):
            stop = min(int(inputs.shape[0]), start + batch_size)
            batch = inputs[start:stop].to(device, dtype=torch.float32, non_blocking=True)
            logits = model(batch)
            if logits.ndim != 2 or int(logits.shape[0]) != stop - start:
                raise ValueError("Classifier must return [N,classes] logits")
            logits_cpu = logits.detach().to("cpu", dtype=torch.float32)
            if not bool(torch.isfinite(logits_cpu).all()):
                raise ValueError("Classifier logits contain NaN or infinite values")
            logits_chunks.append(logits_cpu)
            prediction_chunks.append(logits_cpu.argmax(dim=1).to(dtype=torch.int64))
    return torch.cat(logits_chunks), torch.cat(prediction_chunks)


def _shard_bounds(count: int, shard_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(count, start + shard_size)) for start in range(0, count, shard_size))


def _shard_names(index: int) -> tuple[str, str]:
    stem = f"shard-{index:05d}"
    return f"shards/{stem}.safetensors", f"shards/{stem}.json"


def _estimated_shard_bytes(
    *,
    indices: Any,
    labels: Any,
    predictions: Any,
    logits: Any,
    targets: Any,
    attributions: Any,
    patch_sizes: Sequence[int] = (),
) -> int:
    integer_elements = sum(int(value.numel()) for value in (indices, labels, predictions, targets))
    float_elements = int(logits.numel()) + int(attributions.numel())
    # Safetensors has a small JSON header. One MiB is deliberately conservative
    # for this fixed six-field schema and prevents an on-disk quota overrun.
    count = int(indices.shape[0])
    height, width = (int(value) for value in attributions.shape[-2:])
    patch_elements = sum((height // patch) * (width // patch) for patch in patch_sizes)
    # The sidecar retains one FP32 normalized spatial map for exact SimpleAvg
    # operation order, plus one int32 rank and one FP32 patch score per p.
    sidecar_bytes = (
        integer_elements * 8
        + int(logits.numel()) * 4
        + count * height * width * 4
        + count * patch_elements * 8
        + 2**20
        if patch_sizes
        else 0
    )
    return integer_elements * 8 + float_elements * 4 + sidecar_bytes + 2**20


class _Phase1Publisher:
    """Stage shards in tmpfs and publish them on bounded background I/O threads."""

    def __init__(
        self,
        experiment: SimpleExperiment,
        task: Phase1Task,
        store: ArtifactStore,
        *,
        compact_patch_size: int | None = None,
        artifact_schema_version: int = PHASE1_SCHEMA_VERSION,
        timings: StageTimings | None = None,
    ) -> None:
        self.experiment = experiment
        self.task = task
        self.store = store
        self.compact_patch_size = compact_patch_size
        self.artifact_schema_version = artifact_schema_version
        self.timings = timings or StageTimings()
        if compact_patch_size is not None and compact_patch_size <= 0:
            raise ValueError("compact_patch_size must be positive")
        self.quota = SpoolQuota(
            experiment.storage.spool_root,
            max_bytes=experiment.storage.spool_max_bytes,
            min_free_bytes=experiment.storage.spool_min_free_bytes,
        )
        self.input_quota = SpoolQuota(
            experiment.storage.spool_root / "input-prefetch",
            max_bytes=int(experiment.runtime.phase1_prefetch_max_gib * 2**30),
            min_free_bytes=int(experiment.runtime.phase1_prefetch_min_free_gib * 2**30),
        )
        self.work_directory = self.quota.root / "workers" / f"{os.getpid()}-{uuid.uuid4().hex}"
        self.work_directory.mkdir(parents=True, exist_ok=False)
        self.upload_gate = UploadGate(
            self.quota.root,
            max_slots=experiment.runtime.phase1_upload_global_limit,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=experiment.runtime.phase1_upload_workers,
            thread_name_prefix="phase1-publisher",
        )
        self.object_executor = ThreadPoolExecutor(
            max_workers=max(2, experiment.runtime.phase1_upload_workers * 2),
            thread_name_prefix="phase1-object-publisher",
        )
        self.stage_executor = ThreadPoolExecutor(
            max_workers=experiment.runtime.phase1_stage_workers,
            thread_name_prefix="phase1-stager",
        )
        self.finalizer_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="phase1-finalizer",
        )
        self.futures: list[Future[Any]] = []
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._upload_stats_lock = threading.Lock()
        self._upload_operations = 0
        self._uploaded_bytes = 0
        self._peak_upload_slots = 0

    def _record_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error

    def check(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise ArtifactError("Background Phase 1 publication failed") from failure

    def _publish_remote(
        self,
        local_path: Path,
        relative_path: str,
        *,
        write_receipt: bool = True,
        existing_payload_equivalence: Any = None,
    ) -> Any:
        started = time.monotonic()
        slot = self.upload_gate.acquire()
        self.timings.add("upload_queue_wait", time.monotonic() - started)
        try:
            with self._upload_stats_lock:
                self._upload_operations += 1
                self._peak_upload_slots = max(
                    self._peak_upload_slots,
                    self.upload_gate.active_slots(),
                )
            with self.timings.measure("upload_remote_publish"):
                published = self.store.publish(
                    local_path,
                    relative_path,
                    write_receipt=write_receipt,
                    existing_payload_equivalence=existing_payload_equivalence,
                )
            with self._upload_stats_lock:
                self._uploaded_bytes += int(published.size_bytes)
            return published
        finally:
            self.upload_gate.release(slot)

    def publication_summary(self) -> Mapping[str, int]:
        with self._upload_stats_lock:
            return {
                "upload_operations": self._upload_operations,
                "uploaded_bytes": self._uploaded_bytes,
                "peak_upload_slots": self._peak_upload_slots,
            }

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _publish_staged_shard(
        self,
        *,
        reservation: SpoolReservation,
        local_payload: Path,
        local_rank_ready: Path | None,
        rank_ready_relative_path: str | None,
        rank_ready_identity_digest: str | None,
        task_id: str,
        relative_payload: str,
        relative_record: str,
        record_values: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        local_record = local_payload.with_suffix(".json")
        try:
            self.check()
            payload_future = self.object_executor.submit(
                self._publish_remote,
                local_payload,
                relative_payload,
                existing_payload_equivalence=(
                    compact_rank_input_fp32_equivalence
                    if self.compact_patch_size is not None
                    else None
                ),
            )
            rank_ready_payload = None
            rank_ready_future = None
            if local_rank_ready is not None:
                if rank_ready_relative_path is None or rank_ready_identity_digest is None:
                    raise RuntimeError("Rank-ready staging metadata is incomplete")
                rank_ready_future = self.object_executor.submit(
                    self._publish_remote,
                    local_rank_ready,
                    rank_ready_relative_path,
                )
            published = payload_future.result()
            if rank_ready_future is not None:
                compact = rank_ready_future.result()
                rank_ready_payload = {
                    "relative_path": compact.relative_path,
                    "sha256": compact.sha256,
                    "size_bytes": compact.size_bytes,
                    "identity_digest": rank_ready_identity_digest,
                }
            record: Mapping[str, Any] = {
                **record_values,
                "payload": {
                    "relative_path": published.relative_path,
                    "sha256": published.sha256,
                    "size_bytes": published.size_bytes,
                },
                **({} if rank_ready_payload is None else {"rank_ready": rank_ready_payload}),
            }
            atomic_write_json(local_record, record)
            self.object_executor.submit(
                self._publish_remote, local_record, relative_record
            ).result()
            print(
                f"PHASE1_PUBLISHED task={task_id} shard={record['shard_index']}",
                flush=True,
            )
            return record
        except BaseException as error:
            self._record_failure(error)
            raise
        finally:
            self._unlink(local_record)
            if local_rank_ready is not None:
                self._unlink(local_rank_ready)
            self._unlink(local_payload)
            self.quota.release(reservation)

    def submit_shard(
        self,
        variant: Any,
        *,
        root: str,
        shard_index: int,
        start: int,
        stop: int,
        attributions: Any,
        labels: Any,
        indices: Any,
        predictions: Any,
        logits: Any,
        targets: Any,
        profile_id: str,
        batch_size: int,
        source_identity_digest: str,
    ) -> Future[Mapping[str, Any]]:
        self.check()
        payload_name, record_name = _shard_names(shard_index)
        shard_indices = indices[start:stop]
        shard_labels = labels[start:stop]
        shard_predictions = predictions[start:stop]
        shard_logits = logits[start:stop]
        shard_targets = targets[start:stop]
        if self.compact_patch_size is None:
            patch_sizes = supported_patch_sizes(
                int(attributions.shape[-2]),
                int(attributions.shape[-1]),
            )
            byte_count = _estimated_shard_bytes(
                indices=shard_indices,
                labels=shard_labels,
                predictions=shard_predictions,
                logits=shard_logits,
                targets=shard_targets,
                attributions=attributions,
                patch_sizes=patch_sizes,
            )
        else:
            patch_sizes = (self.compact_patch_size,)
            height, width = (int(value) for value in attributions.shape[-2:])
            if height % self.compact_patch_size or width % self.compact_patch_size:
                raise ValueError(
                    f"Attribution shape {(height, width)} is not divisible by "
                    f"p={self.compact_patch_size}"
                )
            fixed_bytes = sum(
                int(value.numel()) * int(value.element_size())
                for value in (
                    shard_indices,
                    shard_labels,
                    shard_predictions,
                    shard_logits,
                    shard_targets,
                )
            )
            patch_count = (height // self.compact_patch_size) * (width // self.compact_patch_size)
            transient_bytes = int(attributions.numel()) * int(attributions.element_size())
            compact_bytes = fixed_bytes + int(attributions.shape[0]) * patch_count * 8
            # The queued staging closure owns the full CPU attribution until
            # reduction finishes, so account for it even though it is never
            # written or uploaded.
            byte_count = transient_bytes + compact_bytes + 2**20
        reservation = self.quota.acquire(
            byte_count,
            work_directory=self.work_directory,
        )
        local_payload = self.work_directory / (f"{variant.digest[:12]}-{Path(payload_name).name}")
        local_rank_ready = (
            self.work_directory
            / f"{variant.digest[:12]}-{Path(payload_name).stem}.rank-ready.safetensors"
            if patch_sizes and self.compact_patch_size is None
            else None
        )
        current_task = self.task
        record_values: Mapping[str, Any] = {
            "schema_version": self.artifact_schema_version,
            "task_digest": current_task.digest,
            "variant_digest": variant.digest,
            "profile_id": profile_id,
            "batch_size": batch_size,
            "source_identity_digest": source_identity_digest,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "count": stop - start,
            "artifact_representation": (
                "full-signed-attribution"
                if self.compact_patch_size is None
                else "p16-rank-and-simpleavg-patch-score"
            ),
            **(
                {
                    "attribution_shape": list(attributions.shape),
                    "attribution_dtype": "float32",
                }
                if self.compact_patch_size is None
                else {
                    "transient_attribution_shape": list(attributions.shape),
                    "transient_attribution_dtype": "float32",
                    "patch_size": self.compact_patch_size,
                    "patch_count": (int(attributions.shape[-2]) // self.compact_patch_size)
                    * (int(attributions.shape[-1]) // self.compact_patch_size),
                }
            ),
        }
        result: Future[Mapping[str, Any]] = Future()

        def stage_then_publish() -> None:
            rank_ready_relative_path = None
            rank_ready_identity_digest = None
            try:
                self.check()
                staging_started = time.monotonic()
                if self.compact_patch_size is None:
                    write_phase1_shard(
                        local_payload,
                        indices=shard_indices,
                        labels=shard_labels,
                        predictions=shard_predictions,
                        logits=shard_logits,
                        targets=shard_targets,
                        attributions=attributions,
                        metadata={
                            "schema_version": str(self.artifact_schema_version),
                            "task_id": current_task.task_id,
                            "task_digest": current_task.digest,
                            "variant_digest": variant.digest,
                            "source_identity_digest": source_identity_digest,
                            "method": variant.family,
                            "variant": variant.variant,
                            "precision": "fp32",
                            "signed_attribution": "true",
                        },
                    )
                else:
                    compact = dict(
                        build_rank_ready_tensors(
                            {
                                "indices": shard_indices,
                                "labels": shard_labels,
                                "predictions": shard_predictions,
                                "logits": shard_logits,
                                "targets": shard_targets,
                                "attributions": attributions,
                            },
                            simpleavg_normalization=self.experiment.phase2.simpleavg_normalization,
                            patch_sizes=(self.compact_patch_size,),
                        )
                    )
                    compact.pop(simpleavg_spatial_field())
                    write_phase2_shard(
                        local_payload,
                        tensors=compact,
                        metadata={
                            "schema": "simple-assumptions-source-rank-input-v1",
                            "schema_version": str(self.artifact_schema_version),
                            "task_id": current_task.task_id,
                            "task_digest": current_task.digest,
                            "variant_digest": variant.digest,
                            "source_identity_digest": source_identity_digest,
                            "method": variant.family,
                            "variant": variant.variant,
                            "precision": "fp32",
                            "patch_size": str(self.compact_patch_size),
                            "rank_base": "0",
                            "full_attribution_retained": "false",
                            "simpleavg_normalization": self.experiment.phase2.simpleavg_normalization,
                        },
                    )
                if local_rank_ready is not None:
                    source_payload_sha256 = file_sha256(local_payload)
                    rank_ready = write_rank_ready_sidecar(
                        local_rank_ready,
                        fields={
                            "indices": shard_indices,
                            "labels": shard_labels,
                            "predictions": shard_predictions,
                            "logits": shard_logits,
                            "targets": shard_targets,
                            "attributions": attributions,
                        },
                        source_payload_sha256=source_payload_sha256,
                        simpleavg_normalization=self.experiment.phase2.simpleavg_normalization,
                        patch_sizes=patch_sizes,
                    )
                    rank_ready_relative_path = rank_ready.relative_path
                    rank_ready_identity_digest = rank_ready.identity_digest
                self.timings.add("staging", time.monotonic() - staging_started)
                staged_total = local_payload.stat().st_size + (
                    0 if local_rank_ready is None else local_rank_ready.stat().st_size
                )
                if staged_total > reservation.byte_count:
                    raise RuntimeError(
                        "Staged safetensors exceeded its conservative spool reservation"
                    )
                print(
                    f"PHASE1_STAGED task={current_task.task_id} shard={shard_index} "
                    f"spool_mib={staged_total / 2**20:.1f}",
                    flush=True,
                )
                enqueue_started = time.monotonic()
                published = self.executor.submit(
                    self._publish_staged_shard,
                    reservation=reservation,
                    local_payload=local_payload,
                    local_rank_ready=local_rank_ready,
                    rank_ready_relative_path=rank_ready_relative_path,
                    rank_ready_identity_digest=rank_ready_identity_digest,
                    task_id=current_task.task_id,
                    relative_payload=posixpath.join(root, payload_name),
                    relative_record=posixpath.join(root, record_name),
                    record_values=record_values,
                )
                self.timings.add("upload_enqueue", time.monotonic() - enqueue_started)

                def resolve(future: Future[Mapping[str, Any]]) -> None:
                    try:
                        result.set_result(future.result())
                    except BaseException as error:
                        result.set_exception(error)

                published.add_done_callback(resolve)
            except BaseException as error:
                self._record_failure(error)
                if local_rank_ready is not None:
                    self._unlink(local_rank_ready)
                self._unlink(local_payload)
                self.quota.release(reservation)
                result.set_exception(error)

        try:
            self.stage_executor.submit(stage_then_publish)
        except BaseException:
            self._unlink(local_payload)
            self.quota.release(reservation)
            raise
        self.futures.append(result)
        print(
            f"PHASE1_STAGE_QUEUED task={current_task.task_id} shard={shard_index}",
            flush=True,
        )
        return result

    def submit_finalizer(
        self,
        function: Callable[[], Mapping[str, Any]],
    ) -> Future[Mapping[str, Any]]:
        def guarded() -> Mapping[str, Any]:
            try:
                self.check()
                return function()
            except BaseException as error:
                self._record_failure(error)
                raise

        future = self.finalizer_executor.submit(guarded)
        self.futures.append(future)
        return future

    def shutdown(self) -> None:
        self.stage_executor.shutdown(wait=True)
        self.finalizer_executor.shutdown(wait=True)
        self.executor.shutdown(wait=True)
        self.object_executor.shutdown(wait=True)
        shutil.rmtree(self.work_directory, ignore_errors=True)


def _existing_shards(
    store: ArtifactStore,
    root: str,
    *,
    task: Phase1Task,
    schema_version: int = PHASE1_SCHEMA_VERSION,
    variant_digest: str,
    profile_id: str,
    batch_size: int,
    source_identity_digest: str,
    bounds: tuple[tuple[int, int], ...],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, (start, stop) in enumerate(bounds):
        payload_name, record_name = _shard_names(index)
        relative_record = posixpath.join(root, record_name)
        if not store.exists(relative_record):
            continue
        record = store.read_json(relative_record)
        expected = {
            "schema_version": schema_version,
            "task_digest": task.digest,
            "variant_digest": variant_digest,
            "profile_id": profile_id,
            "batch_size": batch_size,
            "source_identity_digest": source_identity_digest,
            "shard_index": index,
            "start": start,
            "stop": stop,
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ArtifactError(
                f"Partial shard record contradicts the task at {relative_record}: {mismatches}"
            )
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or payload.get("relative_path") != posixpath.join(
            root, payload_name
        ):
            raise ArtifactError(f"Malformed partial shard record: {relative_record}")
        if not store.exists(f"{payload['relative_path']}.receipt.json"):
            raise ArtifactError(f"Shard record has no verified payload receipt: {relative_record}")
        result[index] = record
    return result


def _manifest_value(
    experiment: SimpleExperiment,
    task: Phase1Task,
    variant: Any,
    *,
    records: list[Mapping[str, Any]],
    input_shape: tuple[int, int, int],
    checkpoint_sha256: str,
    mean_digest: str,
    source_identity: Mapping[str, str],
    dataset_manifest_sha256: str,
    profile_id: str,
    batch_size: int,
    attribution_provider: Mapping[str, Any],
    target_policy: str,
    model_output_source: str,
) -> Mapping[str, Any]:
    prediction_description, logit_description = _model_output_field_descriptions(
        model_output_source
    )
    return {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment.experiment_id,
        "experiment_digest": experiment.digest,
        "phase1_experiment_digest": experiment.phase1_digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "source_identity_digest": source_identity["digest"],
        "source_identity": dict(source_identity),
        "dataset": {
            "id": task.dataset.dataset_id,
            "registry_key": task.dataset.registry_key,
            "provider_factory": task.dataset.provider_factory,
            "manifest_sha256": dataset_manifest_sha256,
            "split": task.split,
            "condition": task.condition.condition_id,
        },
        "model": {
            "id": task.model.model_id,
            "model_key": task.model.model_key,
            "architecture": task.model.architecture,
            "checkpoint_sha256": checkpoint_sha256,
            "attribution_provider": dict(attribution_provider),
        },
        "method": {
            "family": variant.family,
            "artifact_name": variant.artifact_name,
            "variant": variant.variant,
            "variant_digest": variant.digest,
            "params": dict(variant.params),
        },
        "precision": "fp32",
        "runtime_profile": {
            "profile_id": profile_id,
            "batch_size": batch_size,
        },
        "target_policy": target_policy,
        "model_output_source": model_output_source,
        "attribution_semantics": {
            "signed": True,
            "absolute_value_applied": False,
            "channel_reduction_applied": False,
            "rank_generated": False,
            "spatial_shape": list(input_shape[-2:]),
        },
        "baseline": {
            "mean_artifact_digest": mean_digest,
            "zero_space": "model_input",
            "gaussian_space": "model_input",
            "gaussian_distribution": "N(0,1)",
            "gaussian_seed": experiment.runtime.seed,
            "distribution_members": ["zero", "gaussian", "dataset_mean"],
        },
        "safetensors_fields": {
            "indices": "int64[N] provider row indices",
            "labels": "int64[N]",
            "predictions": prediction_description,
            "logits": logit_description,
            "targets": "int64[N] fixed explanation targets",
            "attributions": "float32[N,C,H,W] full signed attribution",
        },
        "sample_count": sum(int(record["stop"]) - int(record["start"]) for record in records),
        "shard_size": experiment.runtime.shard_size,
        "shards": records,
    }


def _generate_variant(
    experiment: SimpleExperiment,
    task: Phase1Task,
    variant: Any,
    *,
    store: ArtifactStore,
    publisher: _Phase1Publisher,
    model: Any,
    model_inputs: Any,
    labels: Any,
    indices: Any,
    predictions: Any,
    logits: Any,
    targets: Any,
    baseline: Any,
    distribution: Any,
    device: Any,
    batch_size: int,
    profile_id: str,
    source_identity: Mapping[str, str],
    checkpoint_sha256: str,
    mean_digest: str,
    attribution_provider: Mapping[str, Any],
    target_policy: str,
    model_output_source: str,
    allow_legacy_model_output_source: bool = False,
    artifact_root: str | None = None,
    artifact_schema_version: int = PHASE1_SCHEMA_VERSION,
    completed_validator: Callable[[Mapping[str, Any]], None] | None = None,
    manifest_builder: Callable[..., Mapping[str, Any]] | None = None,
) -> Future[Mapping[str, Any]]:
    import torch

    root = artifact_root or phase1_artifact_root(task, variant.artifact_name)
    complete = completed_manifest(
        store,
        root,
        expected_task_digest=task.digest,
        expected_schema_version=artifact_schema_version,
    )
    if complete is not None:
        if completed_validator is None:
            _validate_completed_source(
                complete,
                variant=variant,
                source_identity=source_identity,
                target_policy=target_policy,
                model_output_source=model_output_source,
                allow_legacy_model_output_source=allow_legacy_model_output_source,
            )
        else:
            completed_validator(complete)
        future: Future[Mapping[str, Any]] = Future()
        future.set_result(complete)
        return future
    count = int(model_inputs.shape[0])
    input_shape = tuple(int(item) for item in model_inputs.shape[1:])
    bounds = _shard_bounds(count, experiment.runtime.shard_size)
    records = _existing_shards(
        store,
        root,
        task=task,
        schema_version=artifact_schema_version,
        variant_digest=variant.digest,
        profile_id=profile_id,
        batch_size=batch_size,
        source_identity_digest=source_identity["digest"],
        bounds=bounds,
    )
    missing = set(range(len(bounds))) - set(records)
    publication_futures: dict[int, Future[Mapping[str, Any]]] = {}
    if missing:
        explainer = build_explainer(
            model,
            variant.family,
            architecture=task.model.architecture,
        )
        mean_baseline = distribution[2:3]
        selected_baseline = (
            baseline
            if variant.params.get("baseline") == "zero"
            else mean_baseline
            if variant.params.get("baseline") == "dataset_mean"
            else None
        )
        selected_distribution = (
            distribution if variant.params.get("baseline_distribution") else None
        )
        buffers: dict[int, list[Any]] = {index: [] for index in missing}
        buffered_counts = {index: 0 for index in missing}
        differentiable = EXPLAINER_SPECS[variant.family].differentiable_input
        prefetch_items = []
        input_elements = int(np.prod(input_shape, dtype=np.int64))
        for candidate_start in range(0, count, batch_size):
            candidate_stop = min(count, candidate_start + batch_size)
            if any(
                bounds[index][0] < candidate_stop and bounds[index][1] > candidate_start
                for index in missing
            ):
                prefetch_items.append(
                    PrefetchItem(
                        key=candidate_start,
                        byte_count=max(
                            1,
                            input_elements * 4 * (candidate_stop - candidate_start) + 2**20,
                        ),
                        load=lambda _directory, start=candidate_start, stop=candidate_stop: (
                            model_inputs[start:stop]
                            if isinstance(model_inputs, BatchedTensor)
                            else model_inputs[start:stop].clone()
                        ),
                    )
                )
        prefetcher = ByteBoundedPrefetcher(
            publisher.input_quota,
            prefetch_items,
            # The streaming adversarial source owns a one-shard read cache;
            # keep its loader serialized while still overlapping that read
            # with the GPU attribution loop.
            workers=(
                1
                if task.condition.kind in {"adversarial", "factory"}
                else experiment.runtime.phase1_prefetch_workers
            ),
            namespace=f"phase1-input-{task.task_id[:24]}-{variant.digest[:12]}",
            priority=20,
        )
        try:
            for batch_start in range(0, count, batch_size):
                publisher.check()
                batch_stop = min(count, batch_start + batch_size)
                overlaps = [
                    index
                    for index in missing
                    if bounds[index][0] < batch_stop and bounds[index][1] > batch_start
                ]
                if not overlaps:
                    continue
                prefetched = prefetcher.get(batch_start)
                publisher.timings.add("input_wait", prefetched.wait_seconds)
                publisher.timings.add("input_load", prefetched.load_seconds)
                inputs_cpu = prefetched.value
                with publisher.timings.measure("h2d"):
                    inputs = inputs_cpu.to(device, dtype=torch.float32, non_blocking=True)
                inputs.requires_grad_(differentiable)
                batch_targets = targets[batch_start:batch_stop].to(device, non_blocking=True)
                seed = int(
                    object_sha256(
                        {
                            "base": experiment.runtime.seed,
                            "task": task.digest,
                            "variant": variant.digest,
                            "batch_start": batch_start,
                        }
                    )[:15],
                    16,
                )
                _seed_attribution(seed, device=device)
                with publisher.timings.measure("attribution"):
                    attribution = compute_attribution(
                        explainer,
                        variant.family,
                        inputs,
                        batch_targets,
                        params=captum_params(variant.params),
                        baseline=selected_baseline,
                        baseline_distribution=selected_distribution,
                    )
                attribution = _resize_to_input(
                    attribution, int(inputs.shape[-2]), int(inputs.shape[-1])
                ).detach()
                if int(attribution.shape[0]) != batch_stop - batch_start:
                    raise RuntimeError("Explainer changed the attribution batch dimension")
                if not bool(torch.isfinite(attribution).all()):
                    raise ValueError(
                        f"{variant.artifact_name} produced NaN or infinite attribution values"
                    )
                attribution_cpu = attribution.to("cpu", dtype=torch.float32)
                # The input reservation is no longer needed once attribution
                # is on CPU. Release it before acquiring an output reservation
                # so a full spool cannot deadlock publication behind lookahead.
                prefetcher.release(batch_start)
                for shard_index in overlaps:
                    shard_start, shard_stop = bounds[shard_index]
                    overlap_start = max(shard_start, batch_start)
                    overlap_stop = min(shard_stop, batch_stop)
                    piece = attribution_cpu[
                        overlap_start - batch_start : overlap_stop - batch_start
                    ]
                    buffers[shard_index].append(piece)
                    buffered_counts[shard_index] += overlap_stop - overlap_start
                    if buffered_counts[shard_index] == shard_stop - shard_start:
                        full = torch.cat(buffers.pop(shard_index), dim=0)
                        with publisher.timings.measure("output_enqueue"):
                            publication_futures[shard_index] = publisher.submit_shard(
                                variant,
                                root=root,
                                shard_index=shard_index,
                                start=shard_start,
                                stop=shard_stop,
                                attributions=full,
                                labels=labels,
                                indices=indices,
                                predictions=predictions,
                                logits=logits,
                                targets=targets,
                                profile_id=profile_id,
                                batch_size=batch_size,
                                source_identity_digest=source_identity["digest"],
                            )
                        buffered_counts.pop(shard_index)
                        print(
                            f"PHASE1_COMPUTED shard={shard_index + 1}/{len(bounds)} "
                            f"task={task.task_id} variant={variant.artifact_name}",
                            flush=True,
                        )
                        del full
                model.zero_grad(set_to_none=True)
                del inputs, inputs_cpu, batch_targets, attribution, attribution_cpu
        finally:
            prefetcher.close()
        if buffers or buffered_counts:
            raise RuntimeError(f"Incomplete attribution shard buffers: {buffered_counts}")
        del explainer

    def finalize() -> Mapping[str, Any]:
        resolved = dict(records)
        for shard_index, future in publication_futures.items():
            resolved[shard_index] = future.result()
        ordered = [resolved[index] for index in range(len(bounds))]
        if sum(int(item["count"]) for item in ordered) != count:
            raise RuntimeError("Published shard counts do not cover the configured split")
        builder = manifest_builder or _manifest_value
        manifest = builder(
            experiment,
            task,
            variant,
            records=ordered,
            input_shape=input_shape,
            checkpoint_sha256=checkpoint_sha256,
            mean_digest=mean_digest,
            source_identity=source_identity,
            dataset_manifest_sha256=source_identity["dataset_manifest_sha256"],
            profile_id=profile_id,
            batch_size=batch_size,
            attribution_provider=attribution_provider,
            target_policy=target_policy,
            model_output_source=model_output_source,
        )
        local_manifest = publisher.work_directory / (f"{variant.digest[:12]}-manifest.json")
        try:
            atomic_write_json(local_manifest, manifest)
            publisher._publish_remote(
                local_manifest,
                posixpath.join(root, "manifest.json"),
                write_receipt=False,
            )
        finally:
            publisher._unlink(local_manifest)
        return manifest

    return publisher.submit_finalizer(finalize)


def _write_phase1_telemetry(
    experiment: SimpleExperiment,
    task: Phase1Task,
    *,
    timings: StageTimings,
    gpu: Mapping[str, Any] | None,
    publication: Mapping[str, int],
) -> None:
    """Persist execution-only timing data outside immutable artifact manifests."""

    destination = experiment.runtime.log_directory / "telemetry" / "phase1"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        destination / f"{task.task_id}.json",
        {
            "schema_version": 1,
            "kind": "phase1-runtime-telemetry",
            "task_id": task.task_id,
            "task_digest": task.digest,
            "experiment_id": experiment.experiment_id,
            "stages": timings.summary(),
            "gpu": gpu,
            "publication": dict(publication),
            "execution_controls": {
                "upload_workers": experiment.runtime.phase1_upload_workers,
                "upload_global_limit": experiment.runtime.phase1_upload_global_limit,
                "stage_workers": experiment.runtime.phase1_stage_workers,
                "prefetch_workers": experiment.runtime.phase1_prefetch_workers,
                "prefetch_max_gib": experiment.runtime.phase1_prefetch_max_gib,
                "prefetch_min_free_gib": experiment.runtime.phase1_prefetch_min_free_gib,
            },
        },
    )


def run_phase1_task(
    experiment: SimpleExperiment,
    task: Phase1Task,
    *,
    device: str = "cuda:0",
    batch_size_override: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Generate every variant for one method-level task, with shared model/data."""

    import torch

    store = ArtifactStore(experiment)
    source_identity = _source_identity(experiment, task, store)
    completed = []
    all_complete = True
    for variant in task.variants:
        root = phase1_artifact_root(task, variant.artifact_name)
        manifest = completed_manifest(
            store,
            root,
            expected_task_digest=task.digest,
            expected_schema_version=PHASE1_SCHEMA_VERSION,
        )
        if manifest is None:
            all_complete = False
            break
        _validate_completed_source(
            manifest,
            variant=variant,
            source_identity=source_identity,
            target_policy=CLEAN_MODEL_TARGET_POLICY,
            model_output_source=TASK_MODEL_OUTPUT_SOURCE,
            allow_legacy_model_output_source=True,
        )
        completed.append(manifest)
    if all_complete:
        return tuple(completed)

    from .runtime_dependencies import require_relprop_runtime

    require_relprop_runtime(((task.family, task.model.architecture),))

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 1 requires an available CUDA device")
    timings = StageTimings()
    publisher = _Phase1Publisher(experiment, task, store, timings=timings)
    gpu_sampler = (
        GpuUtilizationSampler(
            requested_device=str(target_device),
            interval_seconds=experiment.runtime.phase1_telemetry_interval_seconds,
        ).start()
        if target_device.type == "cuda"
        else None
    )
    gpu_telemetry: Mapping[str, Any] | None = None
    telemetry_written = False
    loaded_model = None
    attribution_model = None
    relprop_model = None
    attribution_provider: Mapping[str, Any] = {"kind": "timm"}
    bundle = None
    model_inputs = None
    logits = None
    predictions = None
    targets = None
    raw_mean = None
    zero = None
    distribution = None
    clean_inputs = None
    adversarial_inputs = None
    manifest_futures: list[Future[Mapping[str, Any]]] = []
    try:
        try:
            loaded_model = load_model(
                task.model,
                device=target_device,
                include_checkpoint=True,
            )
            with timings.measure("input_load"):
                bundle = load_split(
                    task.dataset,
                    loaded_model,
                    split=task.split,
                    workers=experiment.runtime.dataloader_workers,
                    shared_cache_root=experiment.storage.spool_root / "shared-cache",
                )
            if task.condition.kind == "adversarial":
                from .adversarial import materialize_adversarial_model_inputs

                with timings.measure("input_materialize"):
                    adversarial_inputs = materialize_adversarial_model_inputs(
                        experiment,
                        task,
                        bundle,
                        loaded_model,
                        store=store,
                    )
                model_inputs = adversarial_inputs.model_inputs
                logits = adversarial_inputs.adversarial_logits
                predictions = adversarial_inputs.predictions
                targets = adversarial_inputs.targets
                if not torch.equal(predictions, targets):
                    raise ArtifactError(
                        "Formal adversarial inputs no longer preserve their clean predictions"
                    )
            else:
                with timings.measure("input_materialize"):
                    model_inputs = materialize_model_inputs(
                        bundle,
                        loaded_model,
                        task.condition,
                        device=target_device,
                        batch_size=experiment.runtime.prediction_batch_size,
                        seed=experiment.runtime.seed,
                        shared_cache_root=experiment.storage.spool_root / "shared-cache",
                    )
                with timings.measure("model_forward"):
                    logits, predictions = _predict(
                        loaded_model.model,
                        model_inputs,
                        device=target_device,
                        batch_size=experiment.runtime.prediction_batch_size,
                    )
                if task.condition.kind == "clean":
                    targets = predictions.clone()
                else:
                    clean_inputs = materialize_clean_model_inputs(
                        bundle,
                        loaded_model,
                        batch_size=experiment.runtime.prediction_batch_size,
                    )
                    with timings.measure("clean_target_forward"):
                        _, targets = _predict(
                            loaded_model.model,
                            clean_inputs,
                            device=target_device,
                            batch_size=experiment.runtime.prediction_batch_size,
                        )
                    clean_inputs = None
            raw_mean = load_raw_dataset_mean(
                task.model,
                input_size=int(loaded_model.preprocessing["input_size"]),
            )
            zero, distribution = fixed_baselines(
                raw_mean,
                loaded_model,
                device=target_device,
                seed=experiment.runtime.seed,
            )
            attribution_model = loaded_model.model
            if relprop_required(task.family, task.model.architecture):
                relprop_model = load_relprop_model(
                    task.model,
                    method=task.family,
                    device=target_device,
                )
                if object_sha256(relprop_model.preprocessing) != object_sha256(
                    loaded_model.preprocessing
                ):
                    raise ArtifactError(
                        "RelProp preprocessing differs from the reference timm model"
                    )
                equivalence = ensure_relprop_equivalence_certificate(
                    experiment,
                    task.model,
                    method=task.family,
                    reference_model=loaded_model.model,
                    relprop_model=relprop_model.model,
                    preprocessing=loaded_model.preprocessing,
                    checkpoint_sha256=source_identity["checkpoint_sha256"],
                    device=target_device,
                )
                attribution_model = relprop_model.model
                attribution_provider = {
                    **relprop_attribution_provider(task.family, task.model.architecture),
                    "equivalence_certificate": equivalence,
                }
                # Prediction targets and conditioned inputs are already fixed.
                # Retain only the RelProp implementation during attribution.
                loaded_model = relprop_model
                gc.collect()
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
            checkpoint_digest = source_identity["checkpoint_sha256"]
            mean_digest = source_identity["mean_artifact_digest"]
            for variant in task.variants:
                profile = next(
                    item
                    for item in experiment.profiles()
                    if item.profile_id in task.profile_ids and item.method.digest == variant.digest
                )
                locked_profile = load_profile(experiment, profile)
                if batch_size_override is None and locked_profile is None:
                    # Profiling is a resource measurement that never enters
                    # the task digest; without it (e.g. CPU-only runs), fall
                    # back to the configured fixed batch size.
                    batch_size = int(experiment.runtime.prediction_batch_size)
                    print(
                        f"WARNING missing batch profile {profile.profile_id}; "
                        "falling back to runtime.prediction_batch_size="
                        f"{batch_size}",
                        flush=True,
                    )
                else:
                    batch_size = (
                        int(batch_size_override)
                        if batch_size_override is not None
                        else int(locked_profile.selected_batch_size)  # type: ignore[union-attr]
                    )
                manifest_futures.append(
                    _generate_variant(
                        experiment,
                        task,
                        variant,
                        store=store,
                        publisher=publisher,
                        model=attribution_model,
                        model_inputs=model_inputs,
                        labels=bundle.labels,
                        indices=bundle.indices,
                        predictions=predictions,
                        logits=logits,
                        targets=targets,
                        baseline=zero,
                        distribution=distribution,
                        device=target_device,
                        batch_size=batch_size,
                        profile_id=profile.profile_id,
                        source_identity=source_identity,
                        checkpoint_sha256=checkpoint_digest,
                        mean_digest=mean_digest,
                        attribution_provider=attribution_provider,
                        target_policy=CLEAN_MODEL_TARGET_POLICY,
                        model_output_source=TASK_MODEL_OUTPUT_SOURCE,
                        allow_legacy_model_output_source=True,
                    )
                )
                gc.collect()
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            clean_inputs = None
            adversarial_inputs = None
            model_inputs = None
            logits = None
            predictions = None
            targets = None
            raw_mean = None
            zero = None
            distribution = None
            bundle = None
            attribution_model = None
            relprop_model = None
            loaded_model = None
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
            emit_gpu_release_signal()
            print(
                f"PHASE1_GPU_RELEASED task={task.task_id} "
                f"pending_publications={sum(not future.done() for future in publisher.futures)}",
                flush=True,
            )
        manifests = tuple(future.result() for future in manifest_futures)
        publisher.check()
        return manifests
    finally:
        try:
            publisher.shutdown()
        finally:
            if gpu_sampler is not None:
                gpu_telemetry = gpu_sampler.stop()
            if not telemetry_written:
                _write_phase1_telemetry(
                    experiment,
                    task,
                    timings=timings,
                    gpu=gpu_telemetry,
                    publication=publisher.publication_summary(),
                )
                telemetry_written = True


__all__ = ["run_phase1_task"]
