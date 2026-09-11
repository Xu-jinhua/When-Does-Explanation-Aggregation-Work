"""Formal, reusable adversarial datasets for the simple experiment."""

from __future__ import annotations

import fcntl
import gc
import json
import os
import posixpath
import shutil
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_ensemble.core.cache import CacheError, ShardCache
from xai_ensemble.core.hashing import file_sha256, object_sha256, stable_seed
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path

from .artifacts import (
    ADVERSARIAL_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    _canonical_safetensors_header,
    adversarial_artifact_root,
    completed_manifest,
    load_safetensors,
)
from .config import AdversarialTask, SimpleExperiment
from .data import (
    BatchedTensor,
    DatasetBundle,
    LoadedModel,
    hot_cache_bytes,
    hot_cache_root,
    load_model,
    load_split,
    streaming_inputs_enabled,
)
from .manifest_identity import dataset_manifest_identity_sha256
from .runtime import emit_gpu_release_signal
from .spool import SpoolQuota, SpoolReservation


@dataclass(frozen=True, slots=True)
class LoadedAdversarialInputs:
    model_inputs: Any
    clean_logits: Any
    adversarial_logits: Any
    predictions: Any
    targets: Any


class _StreamingAdversarialSource:
    """Load and retain only the adversarial shard currently consumed by Phase 1."""

    def __init__(
        self,
        experiment: SimpleExperiment,
        task: AdversarialTask,
        bundle: DatasetBundle,
        loaded_model: LoadedModel,
        store: ArtifactStore,
        manifest: Mapping[str, Any],
    ) -> None:
        self.experiment = experiment
        self.task = task
        self.bundle = bundle
        self.loaded_model = loaded_model
        self.store = store
        self.manifest = manifest
        self.records = tuple(sorted(manifest["shards"], key=lambda item: int(item["shard_index"])))
        self._cached_index: int | None = None
        self._cached_fields: Mapping[str, Any] | None = None

    def clear_cached_shard(self) -> None:
        """Release the last full adversarial shard retained by the loader."""

        self._cached_index = None
        self._cached_fields = None

    def _load(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        shard_index = int(record["shard_index"])
        if self._cached_index == shard_index and self._cached_fields is not None:
            return self._cached_fields
        start, stop = int(record["start"]), int(record["stop"])
        raw, labels, indices = self.bundle.batch(start, stop)
        fields = load_adversarial_shard(
            self.experiment,
            self.task,
            shard_index,
            expected_indices=indices,
            expected_labels=labels,
            clean_images=raw,
            store=self.store,
            manifest=self.manifest,
        )
        self._cached_index = shard_index
        self._cached_fields = fields
        return fields

    def model_inputs(self, start: int, stop: int) -> Any:
        import torch

        pieces = []
        for record in self.records:
            shard_start, shard_stop = int(record["start"]), int(record["stop"])
            overlap_start = max(start, shard_start)
            overlap_stop = min(stop, shard_stop)
            if overlap_start >= overlap_stop:
                continue
            fields = self._load(record)
            # Slice before normalizing. A normalized full shard is hundreds of
            # MiB while the requested batch is only a few dozen rows, and the
            # Phase 1 prefetcher accounts each prefetched batch as batch-sized
            # bytes. Returning a view into the full-shard tensor would pin the
            # whole shard for the lifetime of every queued batch, letting the
            # loader race ahead of a slow explainer (e.g. IntegratedGradients)
            # until the host runs out of memory (OOM-killed, exit -9).
            rows = fields["adversarial_images"][
                overlap_start - shard_start : overlap_stop - shard_start
            ]
            pieces.append(
                self.loaded_model.normalize(rows).to(dtype=torch.float32, device="cpu")
            )
        if not pieces:
            raise IndexError((start, stop))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _streaming_adversarial_cache(experiment: SimpleExperiment) -> ShardCache:
    """Return the shared bounded cache used for immutable adversarial shards."""

    return ShardCache(
        hot_cache_root(experiment.storage.spool_root / "shared-cache" / "bounded-hot-cache"),
        max_bytes=hot_cache_bytes(),
    )


def _streaming_adversarial_cache_key(
    task: AdversarialTask,
    shard_index: int,
    digest: str,
    size: int,
) -> str:
    return f"{task.digest}:{shard_index}:{digest}:{size}.safetensors"


def _checkpoint_digest(task: AdversarialTask) -> str:
    if task.model.checkpoint_path is not None:
        return file_sha256(resolve_full_matrix_runtime_path(task.model.checkpoint_path))
    return object_sha256(
        {
            "init_mode": task.model.init_mode,
            "model": task.model.model_key,
            "num_classes": task.model.num_classes,
            "class_index_map": task.model.class_index_map,
        }
    )


def adversarial_source_identity(task: AdversarialTask) -> Mapping[str, str]:
    return {
        "dataset_manifest_sha256": dataset_manifest_identity_sha256(task.dataset.manifest_path),
        "checkpoint_sha256": _checkpoint_digest(task),
    }


def _shard_bounds(count: int, shard_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(count, start + shard_size)) for start in range(0, count, shard_size))


def _shard_names(index: int) -> tuple[str, str]:
    stem = f"shard-{index:05d}"
    return f"shards/{stem}.safetensors", f"shards/{stem}.json"


def adversarial_artifact_digest(
    task: AdversarialTask,
    source_identity: Mapping[str, str],
    records: Sequence[Mapping[str, Any]],
) -> str:
    return object_sha256(
        {
            "schema": "simple-adversarial-artifact-v1",
            "task_digest": task.digest,
            "source_identity": dict(source_identity),
            "shards": [
                {
                    "shard_index": int(record["shard_index"]),
                    "start": int(record["start"]),
                    "stop": int(record["stop"]),
                    "row_indices_digest": str(record["row_indices_digest"]),
                    "sha256": str(record["payload"]["sha256"]),
                    "size_bytes": int(record["payload"]["size_bytes"]),
                }
                for record in records
            ],
        }
    )


def _validate_manifest_structure(
    task: AdversarialTask,
    manifest: Mapping[str, Any],
    *,
    source_identity: Mapping[str, str],
) -> None:
    expected = {
        "task_id": task.task_id,
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "algorithm": task.algorithm,
        "source_method": task.source_method,
        "precision": "fp32",
        "source_identity": dict(source_identity),
    }
    mismatches = {
        key: {"artifact": manifest.get(key), "current": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Adversarial manifest identity changed: {mismatches}")
    records_value = manifest.get("shards")
    if not isinstance(records_value, Sequence) or isinstance(records_value, (str, bytes)):
        raise ArtifactError("Adversarial manifest shards must be a sequence")
    if not records_value or int(manifest.get("sample_count", 0)) <= 0:
        raise ArtifactError("Adversarial manifest must cover at least one sample")
    records = []
    expected_start = 0
    for expected_index, value in enumerate(records_value):
        if not isinstance(value, Mapping):
            raise ArtifactError("Adversarial shard record must be a mapping")
        try:
            shard_index = int(value["shard_index"])
            start = int(value["start"])
            stop = int(value["stop"])
            count = int(value["count"])
            payload = value["payload"]
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError("Malformed adversarial shard record") from error
        if (
            shard_index != expected_index
            or start != expected_start
            or stop <= start
            or count != stop - start
            or not isinstance(payload, Mapping)
            or not payload.get("sha256")
            or int(payload.get("size_bytes", 0)) <= 0
        ):
            raise ArtifactError(f"Invalid adversarial shard layout at index {expected_index}")
        expected_start = stop
        records.append(value)
    if expected_start != int(manifest.get("sample_count", -1)):
        raise ArtifactError("Adversarial shards do not cover the declared sample count")
    observed_digest = adversarial_artifact_digest(task, source_identity, records)
    if manifest.get("artifact_digest") != observed_digest:
        raise ArtifactError("Adversarial artifact digest does not match its shard records")


def completed_adversarial_manifest(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    artifact_store = store or ArtifactStore(experiment)
    root = adversarial_artifact_root(task)
    manifest = completed_manifest(
        artifact_store,
        root,
        expected_task_digest=task.digest,
        expected_schema_version=ADVERSARIAL_SCHEMA_VERSION,
    )
    if manifest is None:
        return None
    source_identity = adversarial_source_identity(task)
    _validate_manifest_structure(task, manifest, source_identity=source_identity)
    return manifest


def adversarial_source_binding(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, str]:
    manifest = completed_adversarial_manifest(experiment, task, store=store)
    if manifest is None:
        raise FileNotFoundError(
            f"Adversarial dataset is incomplete: {adversarial_artifact_root(task)}"
        )
    return {
        "adversarial_task_digest": task.digest,
        "adversarial_artifact_digest": str(manifest["artifact_digest"]),
    }


def _cpu_tensor(value: Any) -> Any:
    import torch

    return torch.as_tensor(value).detach().to("cpu").contiguous().clone()


def _validate_shard_tensors(
    task: AdversarialTask,
    tensors: Mapping[str, Any],
    *,
    expected_indices: Any | None = None,
    expected_labels: Any | None = None,
    clean_images: Any | None = None,
    expected_targets: Any | None = None,
    expected_adversarial_logits: Any | None = None,
) -> None:
    import torch

    required = {
        "indices",
        "labels",
        "targets",
        "clean_logits",
        "adversarial_logits",
        "adversarial_images",
        "deltas",
        "best_steps",
    }
    missing = required - set(tensors)
    if missing:
        raise ArtifactError(f"Adversarial shard is missing fields: {sorted(missing)}")
    count = int(tensors["indices"].shape[0])
    if count <= 0 or any(int(tensors[name].shape[0]) != count for name in required):
        raise ArtifactError("Adversarial shard fields are not aligned")
    for name in ("indices", "labels", "targets", "best_steps"):
        if tensors[name].dtype != torch.int64 or tensors[name].ndim != 1:
            raise ArtifactError(f"Adversarial field {name} must be int64[N]")
    for name in ("clean_logits", "adversarial_logits"):
        value = tensors[name]
        if value.dtype != torch.float32 or value.ndim != 2 or not bool(torch.isfinite(value).all()):
            raise ArtifactError(f"Adversarial field {name} must be finite float32[N,K]")
    images = tensors["adversarial_images"]
    deltas = tensors["deltas"]
    if (
        images.dtype != torch.float32
        or deltas.dtype != torch.float32
        or images.ndim != 4
        or deltas.shape != images.shape
        or int(images.shape[1]) != 3
        or not bool(torch.isfinite(images).all())
        or not bool(torch.isfinite(deltas).all())
    ):
        raise ArtifactError("Adversarial images and deltas must be finite aligned float32 NCHW")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ArtifactError("Saved adversarial images escaped raw [0,1] space")
    if float(deltas.abs().max()) > task.epsilon + 1e-6:
        raise ArtifactError("Saved adversarial deltas escaped the configured Linf bound")
    targets = tensors["targets"]
    if not torch.equal(tensors["clean_logits"].argmax(dim=1), targets):
        raise ArtifactError("Clean attack logits do not match fixed targets")
    if not torch.equal(tensors["adversarial_logits"].argmax(dim=1), targets):
        raise ArtifactError("Adversarial attack logits are not prediction-preserving")
    reconstructed_clean = images - deltas
    if float(reconstructed_clean.min()) < -1e-6 or float(reconstructed_clean.max()) > 1.0 + 1e-6:
        raise ArtifactError("Saved adversarial images and deltas reconstruct invalid clean images")
    comparisons = (
        ("indices", expected_indices),
        ("labels", expected_labels),
        ("targets", expected_targets),
        ("adversarial_logits", expected_adversarial_logits),
    )
    for name, expected in comparisons:
        expected_value = (
            None
            if expected is None
            else torch.as_tensor(expected).detach().to("cpu", dtype=tensors[name].dtype)
        )
        if expected_value is not None and not torch.equal(tensors[name], expected_value):
            raise ArtifactError(f"Saved adversarial {name} do not match the consuming phase")
    if clean_images is not None and not torch.allclose(
        reconstructed_clean,
        torch.as_tensor(clean_images).detach().to("cpu", dtype=torch.float32),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ArtifactError("Saved adversarial image/delta pairs do not match the source dataset")


def write_adversarial_shard(
    path: str | Path,
    *,
    task: AdversarialTask,
    tensors: Mapping[str, Any],
) -> Path:
    from safetensors.torch import save_file

    from .adversarial_pilot import attack_source_semantic_name

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "indices": _cpu_tensor(tensors["indices"]),
        "labels": _cpu_tensor(tensors["labels"]),
        "targets": _cpu_tensor(tensors["targets"]),
        "clean_logits": _cpu_tensor(tensors["clean_logits"]),
        "adversarial_logits": _cpu_tensor(tensors["adversarial_logits"]),
        "adversarial_images": _cpu_tensor(tensors["adversarial_images"]),
        "deltas": _cpu_tensor(tensors["deltas"]),
        "best_steps": _cpu_tensor(tensors["best_steps"]),
    }
    import torch

    for name in ("indices", "labels", "targets", "best_steps"):
        values[name] = values[name].to(dtype=torch.int64)
    for name in ("clean_logits", "adversarial_logits", "adversarial_images", "deltas"):
        values[name] = values[name].to(dtype=torch.float32)
    _validate_shard_tensors(task, values)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".safetensors", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            values,
            str(temporary),
            metadata={
                "schema_version": str(ADVERSARIAL_SCHEMA_VERSION),
                "task_id": task.task_id,
                "task_digest": task.digest,
                "algorithm": task.algorithm,
                "source_method": task.source_method,
                "source_method_semantics": attack_source_semantic_name(task.source_method),
                "precision": "fp32",
                "image_space": "raw_[0,1]",
                "reconstruction": "clean=adversarial_images-deltas",
            },
        )
        _canonical_safetensors_header(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _estimated_shard_bytes(tensors: Mapping[str, Any]) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for value in tensors.values()) + 2**20


def _cache_paths(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    shard_index: int,
) -> tuple[Path, Path, Path]:
    root = experiment.storage.spool_root / "adversarial-cache" / task.digest
    payload = root / f"shard-{shard_index:05d}.safetensors"
    return payload, payload.with_suffix(".receipt.json"), payload.with_suffix(".lock")


def _check_cache_capacity(
    experiment: SimpleExperiment,
    destination: Path,
    *,
    size: int,
    reserved_credit: int = 0,
) -> None:
    cache_root = experiment.storage.spool_root / "adversarial-cache"
    existing_size = destination.stat().st_size if destination.is_file() else 0
    cache_bytes = sum(
        path.stat().st_size for path in cache_root.rglob("*.safetensors") if path.is_file()
    )
    quota = SpoolQuota(
        experiment.storage.spool_root,
        max_bytes=experiment.storage.spool_max_bytes,
        min_free_bytes=experiment.storage.spool_min_free_bytes,
    )
    reserved = max(0, quota.reserved_bytes() - reserved_credit)
    projected = cache_bytes - existing_size + size + reserved
    if projected > experiment.storage.spool_max_bytes:
        raise ArtifactError(
            "Adversarial read cache plus active spool reservations would exceed "
            f"the configured {experiment.storage.spool_max_bytes} byte limit"
        )
    free = shutil.disk_usage(experiment.storage.spool_root).free
    additional = max(0, size - existing_size)
    if free - additional < experiment.storage.spool_min_free_bytes:
        raise ArtifactError("Adversarial read cache would violate the spool free-space floor")


def _cache_receipt_valid(path: Path, receipt_path: Path, *, digest: str, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return receipt.get("sha256") == digest and int(receipt.get("size_bytes", -1)) == size


def _write_cache_receipt(path: Path, *, digest: str, size: int) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "sha256": digest,
            "size_bytes": size,
            "verified_utc": datetime.now(UTC).isoformat(),
        },
    )


def _install_cache_payload(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    shard_index: int,
    source: Path,
    *,
    digest: str,
    size: int,
) -> None:
    cache, receipt, lock = _cache_paths(experiment, task, shard_index)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if _cache_receipt_valid(cache, receipt, digest=digest, size=size):
            source.unlink(missing_ok=True)
            return
        global_lock = experiment.storage.spool_root / "adversarial-cache" / ".cache.lock"
        with global_lock.open("a+b") as global_handle:
            fcntl.flock(global_handle.fileno(), fcntl.LOCK_EX)
            _check_cache_capacity(experiment, cache, size=size, reserved_credit=size)
            if cache.is_file() and file_sha256(cache) == digest and cache.stat().st_size == size:
                source.unlink(missing_ok=True)
            else:
                os.replace(source, cache)
            _write_cache_receipt(receipt, digest=digest, size=size)


def _install_streaming_cache_payload(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    shard_index: int,
    source: Path,
    *,
    digest: str,
    size: int,
) -> None:
    """Keep a published shard only in the shared bounded execution cache."""

    cache = _streaming_adversarial_cache(experiment)
    key = _streaming_adversarial_cache_key(task, shard_index, digest, size)

    def fetch(destination: Path) -> None:
        shutil.copyfile(source, destination)

    try:
        cache.get(
            key,
            fetch,
            expected_sha256=digest,
            expected_size=size,
        )
    except (CacheError, OSError) as error:
        # The remote payload and record are already immutable and verified.
        # A rebuildable performance cache must not turn that publication into
        # a failed scientific task.
        print(
            "ADVERSARIAL_HOT_CACHE_SKIPPED "
            f"task={task.task_id} shard={shard_index} error={type(error).__name__}",
            flush=True,
        )


class _AdversarialPublisher:
    def __init__(
        self,
        experiment: SimpleExperiment,
        task: AdversarialTask,
        store: ArtifactStore,
    ) -> None:
        self.experiment = experiment
        self.task = task
        self.store = store
        self.quota = SpoolQuota(
            experiment.storage.spool_root,
            max_bytes=experiment.storage.spool_max_bytes,
            min_free_bytes=experiment.storage.spool_min_free_bytes,
        )
        self.work_directory = (
            self.quota.root / "workers" / f"adversarial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        self.work_directory.mkdir(parents=True, exist_ok=False)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="adversarial-publisher"
        )
        self.futures: list[Future[Mapping[str, Any]]] = []
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()

    def _record_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error

    def check(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise ArtifactError("Background adversarial publication failed") from failure

    def _publish(
        self,
        *,
        reservation: SpoolReservation,
        local_payload: Path,
        relative_payload: str,
        relative_record: str,
        record_values: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        local_record = local_payload.with_suffix(".json")
        try:
            self.check()
            published = self.store.publish(local_payload, relative_payload)
            record: Mapping[str, Any] = {
                **record_values,
                "payload": {
                    "relative_path": published.relative_path,
                    "sha256": published.sha256,
                    "size_bytes": published.size_bytes,
                },
            }
            atomic_write_json(local_record, record)
            self.store.publish(local_record, relative_record)
            if streaming_inputs_enabled():
                _install_streaming_cache_payload(
                    self.experiment,
                    self.task,
                    int(record["shard_index"]),
                    local_payload,
                    digest=published.sha256,
                    size=published.size_bytes,
                )
            else:
                _install_cache_payload(
                    self.experiment,
                    self.task,
                    int(record["shard_index"]),
                    local_payload,
                    digest=published.sha256,
                    size=published.size_bytes,
                )
            print(
                f"ADVERSARIAL_PUBLISHED task={self.task.task_id} shard={record['shard_index']}",
                flush=True,
            )
            return record
        except BaseException as error:
            self._record_failure(error)
            raise
        finally:
            local_record.unlink(missing_ok=True)
            local_payload.unlink(missing_ok=True)
            self.quota.release(reservation)

    def submit_shard(
        self,
        *,
        root: str,
        shard_index: int,
        start: int,
        stop: int,
        tensors: Mapping[str, Any],
        source_identity_digest: str,
    ) -> Future[Mapping[str, Any]]:
        import torch

        self.check()
        payload_name, record_name = _shard_names(shard_index)
        byte_count = _estimated_shard_bytes(tensors)
        reservation = self.quota.acquire(byte_count, work_directory=self.work_directory)
        local_payload = self.work_directory / f"{self.task.digest[:12]}-{Path(payload_name).name}"
        try:
            write_adversarial_shard(local_payload, task=self.task, tensors=tensors)
            if local_payload.stat().st_size > reservation.byte_count:
                raise RuntimeError("Adversarial shard exceeded its spool reservation")
            staged_size = local_payload.stat().st_size
        except BaseException:
            local_payload.unlink(missing_ok=True)
            self.quota.release(reservation)
            raise
        targets = torch.as_tensor(tensors["targets"])
        adversarial_logits = torch.as_tensor(tensors["adversarial_logits"])
        best_steps = torch.as_tensor(tensors["best_steps"])
        deltas = torch.as_tensor(tensors["deltas"])
        indices = torch.as_tensor(tensors["indices"], dtype=torch.int64)
        record_values: Mapping[str, Any] = {
            "schema_version": ADVERSARIAL_SCHEMA_VERSION,
            "task_digest": self.task.digest,
            "source_identity_digest": source_identity_digest,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "count": stop - start,
            "row_indices_digest": object_sha256([int(value) for value in indices.tolist()]),
            "image_shape": list(tensors["adversarial_images"].shape),
            "image_dtype": "float32",
            "delta_linf_max": float(deltas.abs().max()),
            "prediction_preserved_count": int(adversarial_logits.argmax(dim=1).eq(targets).sum()),
            "unchanged_candidate_count": int(best_steps.eq(-1).sum()),
        }
        try:
            future = self.executor.submit(
                self._publish,
                reservation=reservation,
                local_payload=local_payload,
                relative_payload=posixpath.join(root, payload_name),
                relative_record=posixpath.join(root, record_name),
                record_values=record_values,
            )
        except BaseException:
            local_payload.unlink(missing_ok=True)
            self.quota.release(reservation)
            raise
        self.futures.append(future)
        print(
            f"ADVERSARIAL_STAGED task={self.task.task_id} shard={shard_index} "
            f"spool_mib={staged_size / 2**20:.1f}",
            flush=True,
        )
        return future

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
        shutil.rmtree(self.work_directory, ignore_errors=True)


def _existing_shards(
    store: ArtifactStore,
    root: str,
    *,
    task: AdversarialTask,
    source_identity_digest: str,
    bounds: Sequence[tuple[int, int]],
    indices: Any,
) -> dict[int, Mapping[str, Any]]:
    result = {}
    for index, (start, stop) in enumerate(bounds):
        payload_name, record_name = _shard_names(index)
        relative_record = posixpath.join(root, record_name)
        if not store.exists(relative_record):
            continue
        record = store.read_json(relative_record)
        expected = {
            "schema_version": ADVERSARIAL_SCHEMA_VERSION,
            "task_digest": task.digest,
            "source_identity_digest": source_identity_digest,
            "shard_index": index,
            "start": start,
            "stop": stop,
            "count": stop - start,
            "row_indices_digest": object_sha256(
                [int(value) for value in indices[start:stop].tolist()]
            ),
        }
        mismatches = {
            key: {"record": record.get(key), "current": value}
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ArtifactError(f"Contradictory adversarial shard record: {mismatches}")
        payload = record.get("payload")
        expected_payload = posixpath.join(root, payload_name)
        if not isinstance(payload, Mapping) or payload.get("relative_path") != expected_payload:
            raise ArtifactError(f"Malformed adversarial shard record: {relative_record}")
        receipt_path = f"{expected_payload}.receipt.json"
        if not store.exists(expected_payload) or not store.exists(receipt_path):
            raise ArtifactError(f"Unverified adversarial shard payload: {expected_payload}")
        receipt = store.read_json(receipt_path)
        if receipt.get("sha256") != payload.get("sha256") or int(
            receipt.get("size_bytes", -1)
        ) != int(payload.get("size_bytes", -2)):
            raise ArtifactError(f"Adversarial shard receipt mismatch: {receipt_path}")
        result[index] = record
    return result


def _raw_input_model(classifier: Any, preprocessing: Mapping[str, Any]) -> Any:
    import torch

    class RawInputModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = classifier
            self.register_buffer(
                "input_mean",
                torch.as_tensor(preprocessing["mean"], dtype=torch.float32).reshape(1, -1, 1, 1),
            )
            self.register_buffer(
                "input_std",
                torch.as_tensor(preprocessing["std"], dtype=torch.float32).reshape(1, -1, 1, 1),
            )

        def forward(self, raw_images: Any) -> Any:
            return self.classifier((raw_images - self.input_mean) / self.input_std)

    return RawInputModel().eval()


def _attack_config(task: AdversarialTask) -> Any:
    from .adversarial_pilot import SaraAttackConfig

    return SaraAttackConfig(
        epsilon=task.epsilon,
        steps=task.steps,
        learning_rate=task.learning_rate,
        classification_weight=task.classification_weight,
        top_fraction=task.top_fraction,
    )


def run_adversarial_task(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    """Generate one complete attacked split and publish it exactly once."""

    import torch

    store = ArtifactStore(experiment)
    complete = completed_adversarial_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal adversarial generation requires CUDA")
    root = adversarial_artifact_root(task)
    source_identity = adversarial_source_identity(task)
    source_identity_digest = object_sha256(source_identity)
    publisher = _AdversarialPublisher(experiment, task, store)
    publication_futures: dict[int, Future[Mapping[str, Any]]] = {}
    loaded = None
    bundle = None
    raw_model = None
    sample_count = 0
    try:
        try:
            loaded = load_model(task.model, device=target_device, include_checkpoint=True)
            for parameter in loaded.model.parameters():
                parameter.requires_grad_(False)
            raw_model = (
                _raw_input_model(loaded.model, loaded.preprocessing).to(target_device).eval()
            )
            bundle = load_split(
                task.dataset,
                loaded,
                split=task.split,
                workers=experiment.runtime.dataloader_workers,
                shared_cache_root=experiment.storage.spool_root / "shared-cache",
            )
            sample_count = len(bundle)
            bounds = _shard_bounds(len(bundle), experiment.runtime.shard_size)
            records = _existing_shards(
                store,
                root,
                task=task,
                source_identity_digest=source_identity_digest,
                bounds=bounds,
                indices=bundle.indices,
            )
            from .adversarial_pilot import sara_attack_batch

            config = _attack_config(task)
            for shard_index, (start, stop) in enumerate(bounds):
                if shard_index in records:
                    continue
                publisher.check()
                raw_shard, shard_labels, shard_indices = bundle.batch(start, stop)
                pieces: dict[str, list[Any]] = {
                    "targets": [],
                    "clean_logits": [],
                    "adversarial_logits": [],
                    "adversarial_images": [],
                    "deltas": [],
                    "best_steps": [],
                }
                for batch_start in range(0, stop - start, task.batch_size):
                    batch_stop = min(stop - start, batch_start + task.batch_size)
                    batch_indices = shard_indices[batch_start:batch_stop]
                    seeds = tuple(
                        int(
                            stable_seed(
                                experiment.runtime.seed,
                                "sara-adversarial",
                                task.model.model_id,
                                int(index),
                            )
                        )
                        for index in batch_indices.tolist()
                    )
                    clean_batch = raw_shard[batch_start:batch_stop].to(
                        target_device, dtype=torch.float32, non_blocking=True
                    )
                    result = sara_attack_batch(
                        raw_model,
                        clean_batch,
                        source_method=task.source_method,
                        config=config,
                        sample_seeds=seeds,
                    )
                    if not torch.equal(result.clean_images, clean_batch):
                        raise RuntimeError("Attack changed the clean source tensor")
                    for name in pieces:
                        pieces[name].append(
                            getattr(result, name).detach().to("cpu").contiguous().clone()
                        )
                    del result, clean_batch
                tensors: Mapping[str, Any] = {
                    "indices": shard_indices.to(dtype=torch.int64),
                    "labels": shard_labels.to(dtype=torch.int64),
                    **{name: torch.cat(values, dim=0) for name, values in pieces.items()},
                }
                _validate_shard_tensors(
                    task,
                    tensors,
                    expected_indices=shard_indices,
                    expected_labels=shard_labels,
                    clean_images=raw_shard,
                )
                publication_futures[shard_index] = publisher.submit_shard(
                    root=root,
                    shard_index=shard_index,
                    start=start,
                    stop=stop,
                    tensors=tensors,
                    source_identity_digest=source_identity_digest,
                )
                print(
                    f"ADVERSARIAL_COMPUTED task={task.task_id} "
                    f"shard={shard_index + 1}/{len(bounds)}",
                    flush=True,
                )
                del raw_shard, shard_labels, shard_indices, pieces, tensors
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            raw_model = None
            bundle = None
            loaded = None
            torch.cuda.synchronize(target_device)
            gc.collect()
            torch.cuda.empty_cache()
            emit_gpu_release_signal()
            print(
                f"ADVERSARIAL_GPU_RELEASED task={task.task_id} "
                f"pending_publications={sum(not future.done() for future in publisher.futures)}",
                flush=True,
            )

        resolved = dict(records)
        for shard_index, future in publication_futures.items():
            resolved[shard_index] = future.result()
        publisher.check()
        ordered = [resolved[index] for index in range(len(bounds))]
        artifact_digest = adversarial_artifact_digest(task, source_identity, ordered)
        from .adversarial_pilot import attack_source_semantic_name

        manifest: Mapping[str, Any] = {
            "schema_version": ADVERSARIAL_SCHEMA_VERSION,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "experiment_id": experiment.experiment_id,
            "experiment_digest": experiment.digest,
            "phase1_experiment_digest": experiment.phase1_digest,
            "task_id": task.task_id,
            "task_digest": task.digest,
            "artifact_digest": artifact_digest,
            "dataset": task.dataset.dataset_id,
            "model": task.model.model_id,
            "split": task.split,
            "condition": task.condition.condition_id,
            "algorithm": task.algorithm,
            "source_method": task.source_method,
            "source_method_semantics": attack_source_semantic_name(task.source_method),
            "precision": "fp32",
            "source_identity": dict(source_identity),
            "attack": {
                "epsilon": task.epsilon,
                "norm": "Linf",
                "steps": task.steps,
                "learning_rate": task.learning_rate,
                "learning_rate_schedule": "cosine_to_one_tenth",
                "classification_weight": task.classification_weight,
                "top_fraction": task.top_fraction,
                "target_policy": "clean_model_fp32_prediction",
                "candidate_policy": "per_sample_best_prediction_preserving",
                "image_space": "raw_[0,1]",
                "batch_size": task.batch_size,
                "seed_policy": "stable_seed(base,sara-adversarial,model_id,row_index)",
            },
            "sample_count": sample_count,
            "shard_size": experiment.runtime.shard_size,
            "safetensors_fields": {
                "indices": "int64[N] provider row indices",
                "labels": "int64[N]",
                "targets": "int64[N] clean FP32 predictions",
                "clean_logits": "float32[N,num_classes]",
                "adversarial_logits": "float32[N,num_classes]",
                "adversarial_images": "float32[N,3,H,W] raw [0,1] attacked images",
                "deltas": "float32[N,3,H,W] raw-pixel adversarial_images-clean_images",
                "best_steps": "int64[N] selected optimizer step; -1 means clean candidate",
            },
            "shards": ordered,
        }
        local_manifest = publisher.work_directory / "manifest.json"
        atomic_write_json(local_manifest, manifest)
        try:
            store.publish(
                local_manifest,
                posixpath.join(root, "manifest.json"),
                write_receipt=False,
            )
        finally:
            local_manifest.unlink(missing_ok=True)
        print(
            f"ADVERSARIAL_COMPLETE task={task.task_id} samples={manifest['sample_count']} "
            f"artifact_digest={artifact_digest}",
            flush=True,
        )
        return manifest
    finally:
        publisher.shutdown()


def _materialize_cached_payload(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    store: ArtifactStore,
    *,
    shard_index: int,
    relative_path: str,
    digest: str,
    size: int,
) -> Path:
    cache, receipt, lock = _cache_paths(experiment, task, shard_index)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if _cache_receipt_valid(cache, receipt, digest=digest, size=size):
            return cache
        global_lock = experiment.storage.spool_root / "adversarial-cache" / ".cache.lock"
        with global_lock.open("a+b") as global_handle:
            fcntl.flock(global_handle.fileno(), fcntl.LOCK_EX)
            _check_cache_capacity(experiment, cache, size=size)
            store.materialize(relative_path, cache, expected_sha256=digest)
            if cache.stat().st_size != size:
                raise ArtifactError("Cached adversarial shard size differs from its manifest")
            _write_cache_receipt(receipt, digest=digest, size=size)
    return cache


def load_adversarial_shard(
    experiment: SimpleExperiment,
    task: AdversarialTask,
    shard_index: int,
    *,
    expected_indices: Any | None = None,
    expected_labels: Any | None = None,
    clean_images: Any | None = None,
    expected_targets: Any | None = None,
    expected_adversarial_logits: Any | None = None,
    store: ArtifactStore | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    artifact_store = store or ArtifactStore(experiment)
    complete = manifest or completed_adversarial_manifest(experiment, task, store=artifact_store)
    if complete is None:
        raise FileNotFoundError(
            f"Adversarial dataset is incomplete: {adversarial_artifact_root(task)}"
        )
    try:
        record = next(
            value for value in complete["shards"] if int(value["shard_index"]) == shard_index
        )
    except (KeyError, StopIteration, TypeError) as error:
        raise ArtifactError(f"Adversarial artifact has no shard {shard_index}") from error
    payload = record["payload"]
    relative_path = str(payload["relative_path"])
    digest = str(payload["sha256"])
    size = int(payload["size_bytes"])
    if streaming_inputs_enabled():
        cache = _streaming_adversarial_cache(experiment)
        key = f"{task.digest}:{shard_index}:{digest}:{size}.safetensors"

        def fetch(destination: Path) -> None:
            artifact_store.materialize(
                relative_path,
                destination,
                expected_sha256=digest,
            )

        with cache.use(
            key,
            fetch,
            expected_sha256=digest,
            expected_size=size,
        ) as local:
            values = dict(load_safetensors(local))
    else:
        local = _materialize_cached_payload(
            experiment,
            task,
            artifact_store,
            shard_index=shard_index,
            relative_path=relative_path,
            digest=digest,
            size=size,
        )
        values = dict(load_safetensors(local))
    _validate_shard_tensors(
        task,
        values,
        expected_indices=expected_indices,
        expected_labels=expected_labels,
        clean_images=clean_images,
        expected_targets=expected_targets,
        expected_adversarial_logits=expected_adversarial_logits,
    )
    if object_sha256([int(value) for value in values["indices"].tolist()]) != record.get(
        "row_indices_digest"
    ):
        raise ArtifactError("Adversarial shard row-index digest differs from its record")
    return values


def materialize_adversarial_model_inputs(
    experiment: SimpleExperiment,
    phase_task: Any,
    bundle: DatasetBundle,
    loaded_model: LoadedModel,
    *,
    store: ArtifactStore | None = None,
) -> LoadedAdversarialInputs:
    import torch

    task = experiment.adversarial_task_for(
        dataset_id=phase_task.dataset.dataset_id,
        model_id=phase_task.model.model_id,
        split=phase_task.split,
        condition_id=phase_task.condition.condition_id,
    )
    artifact_store = store or ArtifactStore(experiment)
    manifest = completed_adversarial_manifest(experiment, task, store=artifact_store)
    if manifest is None:
        raise FileNotFoundError(
            f"Adversarial dataset is incomplete: {adversarial_artifact_root(task)}"
        )

    if streaming_inputs_enabled():
        source = _StreamingAdversarialSource(
            experiment,
            task,
            bundle,
            loaded_model,
            artifact_store,
            manifest,
        )
        input_shape = (
            3,
            int(loaded_model.preprocessing["input_size"]),
            int(loaded_model.preprocessing["input_size"]),
        )
        count = len(bundle)
        clean_logits: list[Any] = []
        adversarial_logits: list[Any] = []
        targets: list[Any] = []
        for record in source.records:
            fields = source._load(record)
            clean_logits.append(fields["clean_logits"].clone())
            adversarial_logits.append(fields["adversarial_logits"].clone())
            targets.append(fields["targets"].clone())
        source.clear_cached_shard()
        combined_adversarial_logits = torch.cat(adversarial_logits, dim=0)
        combined_targets = torch.cat(targets, dim=0)
        result = LoadedAdversarialInputs(
            model_inputs=BatchedTensor(
                (count, *input_shape),
                source.model_inputs,
            ),
            clean_logits=torch.cat(clean_logits, dim=0),
            adversarial_logits=combined_adversarial_logits,
            predictions=combined_adversarial_logits.argmax(dim=1).to(dtype=torch.int64),
            targets=combined_targets,
        )
        if any(
            int(values.shape[0]) != count
            for values in (
                result.clean_logits,
                result.adversarial_logits,
                result.predictions,
                result.targets,
            )
        ):
            raise ArtifactError("Adversarial metadata row count differs from the dataset split")
        return result

    input_chunks = []
    clean_logits = []
    adversarial_logits = []
    targets = []
    for record in manifest["shards"]:
        shard_index = int(record["shard_index"])
        start, stop = int(record["start"]), int(record["stop"])
        raw, labels, indices = bundle.batch(start, stop)
        fields = load_adversarial_shard(
            experiment,
            task,
            shard_index,
            expected_indices=indices,
            expected_labels=labels,
            clean_images=raw,
            store=artifact_store,
            manifest=manifest,
        )
        input_chunks.append(
            loaded_model.normalize(fields["adversarial_images"]).to(dtype=torch.float32)
        )
        clean_logits.append(fields["clean_logits"])
        adversarial_logits.append(fields["adversarial_logits"])
        targets.append(fields["targets"])
        del fields
    combined_adversarial_logits = torch.cat(adversarial_logits, dim=0)
    result = LoadedAdversarialInputs(
        model_inputs=torch.cat(input_chunks, dim=0),
        clean_logits=torch.cat(clean_logits, dim=0),
        adversarial_logits=combined_adversarial_logits,
        predictions=combined_adversarial_logits.argmax(dim=1).to(dtype=torch.int64),
        targets=torch.cat(targets, dim=0),
    )
    if int(result.model_inputs.shape[0]) != len(bundle):
        raise ArtifactError("Adversarial shards do not cover the configured split")
    return result


__all__ = [
    "LoadedAdversarialInputs",
    "adversarial_artifact_digest",
    "adversarial_source_binding",
    "adversarial_source_identity",
    "completed_adversarial_manifest",
    "load_adversarial_shard",
    "materialize_adversarial_model_inputs",
    "run_adversarial_task",
    "write_adversarial_shard",
]
