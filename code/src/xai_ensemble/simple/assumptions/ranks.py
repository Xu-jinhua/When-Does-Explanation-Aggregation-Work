"""GPU rank construction for IND, matched-NAIVE, and Oracle NOISE."""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.phase2.torch_aggregation import aggregate_rankings_torch

from ..artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    load_safetensors,
    phase2_artifact_root,
)
from ..io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from ..methods import PATCH_METHODS
from ..phase2 import (
    _aligned,
    _source_manifests,
    _summarize_aggregation_statistics,
)
from ..rank_ready import (
    RankReadyPublisher,
    ensure_rank_ready_sidecar,
    rank_field,
    simpleavg_score_field,
    simpleavg_spatial_field,
)
from ..runtime import emit_gpu_release_signal
from ..spool import SpoolQuota
from ..telemetry import GpuUtilizationSampler, StageTimings
from .artifacts import (
    RANK_SCHEMA_VERSION,
    AssumptionShardPublisher,
    completed_rank_manifest,
    completed_selection_manifest,
    existing_shard_records,
    output_store,
    publish_manifest,
    task_spool_path,
)
from .config import AssumptionExperiment, RankTask
from .phase1 import (
    SOURCE_RANK_INPUT_REPRESENTATION,
    SOURCE_RANK_INPUT_SCHEMA,
    SOURCE_RANK_INPUT_SCHEMA_VERSION,
    _generation_experiment,
    compact_artifact_root,
)
from .training import ensure_checkpoint

_CUDA_WORKSPACE_BYTES = 2 * 2**30


@dataclass(frozen=True, slots=True)
class AttributionSource:
    label: str
    family: str
    experiment: Any
    store: ArtifactStore
    root: str
    manifest: Mapping[str, Any]
    source_id: str


@dataclass(frozen=True, slots=True)
class RankShardInput:
    reference: Mapping[str, Any]
    ballots_by_method: Mapping[str, np.ndarray]
    simple_scores: np.ndarray
    sidecars: tuple[Mapping[str, Any], ...]


def _layout(manifest: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
        for item in manifest["shards"]
    )


def _load_attribution_shard(
    experiment: AssumptionExperiment,
    source: AttributionSource,
    *,
    shard_index: int,
) -> Mapping[str, Any]:
    record = next(
        item for item in source.manifest["shards"] if int(item["shard_index"]) == shard_index
    )
    payload = record["payload"]
    cache = experiment.storage.spool_root / "attribution-cache" / str(os.getpid())
    local = cache / f"{object_sha256({'root': source.root})[:16]}--{shard_index:05d}.safetensors"
    source.store.materialize(
        str(payload["relative_path"]),
        local,
        expected_sha256=str(payload["sha256"]),
    )
    try:
        return dict(load_safetensors(local))
    finally:
        local.unlink(missing_ok=True)


def _source_record(source: AttributionSource, shard_index: int) -> Mapping[str, Any]:
    return next(
        item for item in source.manifest["shards"] if int(item["shard_index"]) == shard_index
    )


def _source_scope(
    experiment: AssumptionExperiment,
    task: RankTask,
    source_id: str,
) -> Any:
    matches = [
        scope
        for scope in experiment.source_phase1_tasks()
        if scope.cell.cell_id == task.cell.cell_id
        and scope.source_id == source_id
        and (task.family_id is None or scope.family_id == task.family_id)
        and scope.condition.condition_id == task.condition.condition_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source Phase 1 scope for {task.cell.cell_id}/{source_id}/"
            f"{task.condition.condition_id}; found {len(matches)}"
        )
    return matches[0]


def _source_attributions(
    experiment: AssumptionExperiment,
    task: RankTask,
) -> tuple[AttributionSource, ...]:
    result = []
    reference_layout = None
    for source_id, family in task.source_method_pairs:
        scope = _source_scope(experiment, task, source_id)
        training = experiment.find_training_task(scope.training_task_id)
        checkpoint, _ = ensure_checkpoint(experiment, training)
        generation = _generation_experiment(experiment, scope, checkpoint)
        producers = {
            producer.family: producer
            for producer in experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
        }
        producer = producers[family]
        artifact_name = f"{family}__p16" if family in PATCH_METHODS else family
        variants = {variant.artifact_name: variant for variant in producer.variants}
        if artifact_name not in variants:
            raise ArtifactError(f"Source Phase 1 has no p=16 {family} variant")
        store = ArtifactStore(generation)
        root = compact_artifact_root(producer, artifact_name)
        manifest = completed_manifest(
            store,
            root,
            expected_task_digest=producer.digest,
            expected_schema_version=SOURCE_RANK_INPUT_SCHEMA_VERSION,
        )
        if manifest is None:
            raise FileNotFoundError(f"Compact source rank input is incomplete: {root}")
        if (
            manifest.get("schema") != SOURCE_RANK_INPUT_SCHEMA
            or manifest.get("representation") != SOURCE_RANK_INPUT_REPRESENTATION
            or manifest.get("full_attribution_retained") is not False
        ):
            raise ArtifactError(f"Source rank input has contradictory semantics: {root}")
        current_layout = _layout(manifest)
        if reference_layout is None:
            reference_layout = current_layout
        elif current_layout != reference_layout:
            raise ArtifactError("Source-model Phase 1 shard layouts are not aligned")
        label = family if task.setting == "matched-naive" else f"{source_id}__{family}"
        result.append(
            AttributionSource(label, family, generation, store, root, manifest, source_id)
        )
    return tuple(result)


def _selected_methods(
    experiment: AssumptionExperiment,
    task: RankTask,
) -> tuple[str, ...]:
    if task.setting != "oracle-noise":
        return task.methods
    assert task.selection_task_id is not None
    selection_task = experiment.find_selection_task(task.selection_task_id)
    manifest = completed_selection_manifest(experiment, selection_task)
    if manifest is None:
        raise FileNotFoundError(f"Oracle NOISE selection is incomplete: {selection_task.task_id}")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ArtifactError("Oracle NOISE selection manifest is malformed")
    selected = tuple(str(item) for item in selection["selected_methods"])
    if not selected or not set(selected) <= set(task.cell.methods):
        raise ArtifactError("Oracle NOISE selected an invalid method roster")
    return selected


def _base_attributions(
    experiment: AssumptionExperiment,
    task: RankTask,
    methods: Sequence[str],
) -> tuple[AttributionSource, ...]:
    base_task = experiment.base_phase2_task(task.cell, task.condition.condition_id)
    store = ArtifactStore(experiment.base)
    sources = _source_manifests(experiment.base, base_task, store, methods)
    return tuple(
        AttributionSource(
            family,
            family,
            experiment.base,
            store,
            sources[family][0],
            sources[family][1],
            "reference-full",
        )
        for family in methods
    )


def _base_rank_manifest(
    experiment: AssumptionExperiment,
    task: RankTask,
) -> tuple[str, Mapping[str, Any]]:
    base_task = experiment.base_phase2_task(task.cell, task.condition.condition_id)
    store = ArtifactStore(experiment.base)
    root = phase2_artifact_root(base_task)
    manifest = completed_manifest(
        store,
        root,
        expected_task_digest=base_task.digest,
        expected_schema_version=PHASE2_SCHEMA_VERSION,
    )
    if manifest is None:
        raise FileNotFoundError(f"NAIVE p=16 rank source is incomplete: {root}")
    return root, manifest


def _base_single_ranks(
    experiment: AssumptionExperiment,
    task: RankTask,
    manifest: Mapping[str, Any],
    *,
    shard_index: int,
    methods: Sequence[str],
    work_directory: Any | None = None,
) -> Mapping[str, np.ndarray]:
    record = next(item for item in manifest["shards"] if int(item["shard_index"]) == shard_index)
    labels = record.get("rule_labels")
    if not isinstance(labels, Mapping):
        raise ArtifactError("NAIVE rank shard is missing rule labels")
    by_name = {str(name): str(field_id) for field_id, name in labels.items()}
    payload = record["payload"]
    local = (
        task_spool_path(
            experiment,
            namespace="naive-rank-cache",
            task_digest=task.digest,
            relative_path=str(payload["relative_path"]),
        )
        if work_directory is None
        else work_directory / f"naive-{payload['sha256']}.safetensors"
    )
    store = ArtifactStore(experiment.base)
    store.materialize(str(payload["relative_path"]), local, expected_sha256=str(payload["sha256"]))
    try:
        fields = load_safetensors(local)
        result = {
            family: fields[f"rank__{by_name[f'single__{family}']}"]
            .numpy()
            .astype(np.int64, copy=False)
            .copy()
            for family in methods
        }
    finally:
        local.unlink(missing_ok=True)
    return result


def _load_rank_shard_input(
    experiment: AssumptionExperiment,
    task: RankTask,
    sources: Sequence[AttributionSource],
    methods: Sequence[str],
    *,
    shard_index: int,
    work_directory: Any,
    base_rank_manifest: Mapping[str, Any] | None,
    rank_ready_publisher: RankReadyPublisher | None,
) -> RankShardInput:
    import torch

    reference = None
    computed_ballots: dict[str, np.ndarray] = {}
    simple_spatial_sum = None
    simple_patch_sum = None
    sidecars = []
    for source_index, source in enumerate(sources):
        record = _source_record(source, shard_index)
        payload = record["payload"]
        count = int(record["stop"]) - int(record["start"])
        if source.manifest.get("schema") == SOURCE_RANK_INPUT_SCHEMA:
            local = work_directory / f"source-{source_index:03d}" / Path(
                str(payload["relative_path"])
            ).name
            source.store.materialize(
                str(payload["relative_path"]),
                local,
                expected_sha256=str(payload["sha256"]),
            )
            try:
                fields = dict(load_safetensors(local))
            finally:
                local.unlink(missing_ok=True)
            required = {
                "indices",
                "labels",
                "predictions",
                "logits",
                "targets",
                rank_field(experiment.patch_size),
                simpleavg_score_field(experiment.patch_size),
            }
            if set(fields) != required:
                raise ArtifactError("Compact source rank input fields differ from its schema")
            if any(int(value.shape[0]) != count for value in fields.values()):
                raise ArtifactError("Compact source rank input fields are not shard-aligned")
            if fields["logits"].ndim != 2 or not bool(
                torch.isfinite(fields["logits"]).all()
            ):
                raise ArtifactError("Compact source rank input logits are invalid")
            if not torch.equal(fields["predictions"], fields["logits"].argmax(dim=1)):
                raise ArtifactError("Compact source predictions differ from argmax(logits)")
            rank = fields[rank_field(experiment.patch_size)]
            score = fields[simpleavg_score_field(experiment.patch_size)]
            expected_items = int(source.manifest.get("patch_count", -1))
            if expected_items != (224 // experiment.patch_size) ** 2:
                raise ArtifactError("Compact source manifest has an invalid patch count")
            if rank.shape != (count, expected_items) or score.shape != rank.shape:
                raise ArtifactError("Compact source patch fields have invalid shapes")
            if rank.dtype != torch.int32 or score.dtype != torch.float32:
                raise ArtifactError("Compact source patch fields have invalid dtypes")
            expected_rank = torch.arange(expected_items, dtype=torch.int32).expand(count, -1)
            if not torch.equal(torch.sort(rank, dim=1).values, expected_rank):
                raise ArtifactError("Compact source ranks are not strict zero-based permutations")
            if not bool(torch.isfinite(score).all()):
                raise ArtifactError("Compact source SimpleAvg scores are not finite")
            compact_descriptor = {
                "schema": SOURCE_RANK_INPUT_SCHEMA,
                "representation": SOURCE_RANK_INPUT_REPRESENTATION,
                "payload": dict(payload),
                "generated_from_legacy_source": False,
                "materialize_seconds": None,
            }
        else:
            if rank_ready_publisher is None:
                raise RuntimeError("Legacy attribution source requires a rank-ready publisher")
            compact = ensure_rank_ready_sidecar(
                source.store,
                source_payload=payload,
                work_directory=work_directory / f"source-{source_index:03d}",
                lock_root=experiment.storage.spool_root / "rank-ready-locks",
                simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
                count=count,
                recorded_sidecar=(
                    record.get("rank_ready")
                    if isinstance(record.get("rank_ready"), Mapping)
                    else None
                ),
                publisher=rank_ready_publisher,
            )
            fields = compact.fields
            compact_descriptor = {
                "schema": "simple-rank-ready-v2",
                "representation": "legacy-full-attribution-derived-sidecar",
                "identity_digest": compact.descriptor.identity_digest,
                "payload": dict(compact.payload),
                "generated_from_legacy_source": compact.generated,
                "materialize_seconds": compact.elapsed_seconds,
            }
        if reference is None:
            reference = {
                key: fields[key].clone()
                for key in ("indices", "labels", "predictions", "logits", "targets")
            }
        else:
            _aligned(reference, fields, source.label)
        if task.setting != "oracle-noise":
            computed_ballots[source.label] = (
                fields[rank_field(experiment.patch_size)]
                .numpy()
                .astype(np.int64, copy=False)
                .copy()
            )
        if source.manifest.get("schema") == SOURCE_RANK_INPUT_SCHEMA:
            patch_scores = (
                fields[simpleavg_score_field(experiment.patch_size)]
                .numpy()
                .astype(np.float32, copy=False)
            )
            simple_patch_sum = (
                patch_scores.copy()
                if simple_patch_sum is None
                else simple_patch_sum + patch_scores
            )
        else:
            spatial = fields[simpleavg_spatial_field()].numpy().astype(np.float32, copy=False)
            simple_spatial_sum = (
                spatial.copy() if simple_spatial_sum is None else simple_spatial_sum + spatial
            )
        sidecars.append(
            {
                "label": source.label,
                **compact_descriptor,
            }
        )
    if reference is None or (simple_patch_sum is None and simple_spatial_sum is None):
        raise ArtifactError("Rank shard has no compact explanation inputs")
    if simple_patch_sum is not None and simple_spatial_sum is not None:
        raise ArtifactError("Rank shard mixes incompatible SimpleAvg input representations")
    if task.setting == "oracle-noise":
        if base_rank_manifest is None:
            raise RuntimeError("Oracle NOISE rank source is missing")
        ballots = _base_single_ranks(
            experiment,
            task,
            base_rank_manifest,
            shard_index=shard_index,
            methods=methods,
            work_directory=work_directory,
        )
    else:
        ballots = computed_ballots
    if simple_patch_sum is not None:
        simple_scores = simple_patch_sum / float(len(sources))
    else:
        assert simple_spatial_sum is not None
        averaged = simple_spatial_sum / float(len(sources))
        height, width = averaged.shape[-2:]
        grid_h, grid_w = height // experiment.patch_size, width // experiment.patch_size
        simple_scores = averaged.reshape(
            averaged.shape[0],
            grid_h,
            experiment.patch_size,
            grid_w,
            experiment.patch_size,
        ).mean(axis=(2, 4), dtype=np.float32)
    return RankShardInput(
        reference=reference,
        ballots_by_method=ballots,
        simple_scores=simple_scores,
        sidecars=tuple(sidecars),
    )


def _aggregate(
    experiment: AssumptionExperiment,
    task: RankTask,
    ballots: np.ndarray,
    simple_scores: np.ndarray,
    indices: np.ndarray,
    *,
    device: Any,
) -> Any:
    seeds = tuple(
        int(object_sha256({"rank_task": task.digest, "row_index": int(index)})[:15], 16)
        for index in indices
    )
    return aggregate_rankings_torch(
        ballots,
        simple_scores,
        requested=("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"),
        rrf_c=experiment.base.phase2.rrf_c,
        kemeny_starts=experiment.base.phase2.kemeny_starts,
        kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
        seeds=seeds,
        device=device,
        workspace_bytes=_CUDA_WORKSPACE_BYTES,
    )


def run_rank_task(
    experiment: AssumptionExperiment,
    task: RankTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    import torch

    output = output_store(experiment)
    complete = completed_rank_manifest(experiment, task, store=output)
    if complete is not None:
        return complete
    methods = _selected_methods(experiment, task)
    sources = (
        _base_attributions(experiment, task, methods)
        if task.setting == "oracle-noise"
        else _source_attributions(experiment, task)
    )
    if not sources:
        raise ArtifactError("Rank construction has no attribution sources")
    layout = _layout(sources[0].manifest)
    if any(_layout(source.manifest) != layout for source in sources[1:]):
        raise ArtifactError("Attribution source layouts differ")
    base_rank_root = None
    base_rank_manifest = None
    if task.setting == "oracle-noise":
        base_rank_root, base_rank_manifest = _base_rank_manifest(experiment, task)
        if _layout(base_rank_manifest) != layout:
            raise ArtifactError("NAIVE rank and Phase 1 shard layouts differ")
    records = existing_shard_records(
        output,
        task.artifact_root,
        task_digest=task.digest,
        schema_version=RANK_SCHEMA_VERSION,
        source_layout=layout,
    )
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Assumption rank construction requires CUDA")

    missing_layout = tuple(item for item in layout if item[0] not in records)
    quota = SpoolQuota(
        experiment.storage.spool_root,
        max_bytes=experiment.storage.spool_max_bytes,
        min_free_bytes=experiment.storage.spool_min_free_bytes,
    )
    rank_ready_publisher = RankReadyPublisher(
        spool_root=experiment.storage.spool_root,
        spool_max_bytes=experiment.storage.spool_max_bytes,
        spool_min_free_bytes=experiment.storage.spool_min_free_bytes,
        namespace=task.digest[:16],
    )
    prefetch_items = []
    for shard_index, _, _ in missing_layout:
        estimated = 0
        for source in sources:
            record = _source_record(source, shard_index)
            compact = record.get("rank_ready")
            payload = compact if isinstance(compact, Mapping) else record["payload"]
            estimated += int(payload["size_bytes"])
        if base_rank_manifest is not None:
            base_record = next(
                item
                for item in base_rank_manifest["shards"]
                if int(item["shard_index"]) == shard_index
            )
            estimated += int(base_record["payload"]["size_bytes"])
        prefetch_items.append(
            PrefetchItem(
                key=shard_index,
                byte_count=max(1, estimated),
                load=lambda directory, index=shard_index: _load_rank_shard_input(
                    experiment,
                    task,
                    sources,
                    methods,
                    shard_index=index,
                    work_directory=directory,
                    base_rank_manifest=base_rank_manifest,
                    rank_ready_publisher=rank_ready_publisher,
                ),
            )
        )
    publication_futures: dict[int, Future[Mapping[str, Any]]] = {}
    publisher = AssumptionShardPublisher(experiment, output, task_id=task.task_id)
    ballot_labels = (
        tuple(methods)
        if task.setting == "oracle-noise"
        else tuple(source.label for source in sources)
    )
    compute_complete = False
    timings = StageTimings()
    gpu_sampler = GpuUtilizationSampler(requested_device=str(target_device)).start()
    gpu_telemetry: Mapping[str, Any] | None = None
    try:
        with ByteBoundedPrefetcher(
            quota,
            prefetch_items,
            workers=experiment.runtime.cpu_workers,
            namespace=f"rank-{task.digest[:16]}",
        ) as prefetcher:
            for shard_index, start, stop in missing_layout:
                prefetched = prefetcher.get(shard_index)
                timings.add("input_wait", prefetched.wait_seconds)
                timings.add("input_materialize", prefetched.load_seconds)
                shard = prefetched.value
                reference = shard.reference
                ballots_by_method = shard.ballots_by_method
                ballots = np.stack([ballots_by_method[label] for label in ballot_labels], axis=1)
                indices = reference["indices"].numpy().astype(np.int64, copy=False)
                with timings.measure("gpu_aggregation"):
                    aggregation = _aggregate(
                        experiment,
                        task,
                        ballots,
                        shard.simple_scores,
                        indices,
                        device=target_device,
                    )
                rules = dict(aggregation)
                for label in ballot_labels:
                    rules[f"single__{label}"] = ballots_by_method[label]
                rule_labels = {f"r{index:03d}": name for index, name in enumerate(rules)}
                tensors = {
                    "indices": reference["indices"],
                    "labels": reference["labels"],
                    "targets": reference["targets"],
                    "unmasked_predictions": reference["predictions"],
                    **{
                        f"rank__{field_id}": torch.from_numpy(rules[name]).to(torch.int32)
                        for field_id, name in rule_labels.items()
                    },
                }
                with timings.measure("output_enqueue"):
                    publication_futures[shard_index] = publisher.submit_shard(
                        root=task.artifact_root,
                        task_digest=task.digest,
                        schema_version=RANK_SCHEMA_VERSION,
                        shard_index=shard_index,
                        start=start,
                        stop=stop,
                        tensors=tensors,
                        metadata={
                            "rank_base": "0",
                            "patch_size": "16",
                            "precision": "fp32",
                        },
                        record_fields={
                            "rule_labels": rule_labels,
                            "aggregation_statistics": aggregation.statistics,
                            "rank_ready_sources": list(shard.sidecars),
                        },
                    )
                print(
                    "ASSUMPTIONS_RANK "
                    f"shard={shard_index + 1}/{len(layout)} "
                    f"prefetch_wait_seconds={prefetched.wait_seconds:.3f} "
                    f"prefetch_load_seconds={prefetched.load_seconds:.3f} "
                    f"sidecars_generated={sum(bool(item['generated_from_legacy_source']) for item in shard.sidecars)} "
                    f"task={task.task_id}",
                    flush=True,
                )
                del shard, reference, ballots_by_method, ballots
                del aggregation, rules, tensors
                prefetcher.release(shard_index)
                publisher.check()
                rank_ready_publisher.check()
        compute_complete = True
    finally:
        gpu_telemetry = gpu_sampler.stop()
        utilization = gpu_telemetry.get("utilization_percent")
        print(
            "ASSUMPTIONS_GPU_TELEMETRY "
            f"kind=rank samples={gpu_telemetry['sample_count']} "
            f"mean_utilization={None if utilization is None else utilization['mean']} "
            f"task={task.task_id}",
            flush=True,
        )
        torch.cuda.synchronize(target_device)
        gc.collect()
        torch.cuda.empty_cache()
        emit_gpu_release_signal()
        if not compute_complete:
            try:
                publisher.shutdown()
            finally:
                rank_ready_publisher.shutdown()
    try:
        for shard_index, future in publication_futures.items():
            records[shard_index] = future.result()
        publisher.check()
    finally:
        try:
            publisher.shutdown()
        finally:
            rank_ready_publisher.shutdown()

    ordered = [records[index] for index, _, _ in layout]
    sample_count = sum(int(record["count"]) for record in ordered)
    statistics = _summarize_aggregation_statistics(ordered, require_kemeny=True)
    source_records = [
        {
            "label": source.label,
            "source_id": source.source_id,
            "method": source.family,
            "root": source.root,
            "task_digest": source.manifest["task_digest"],
        }
        for source in sources
    ]
    value: Mapping[str, Any] = {
        "schema_version": RANK_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "setting": task.setting,
        "oracle": task.setting == "oracle-noise",
        "cell": task.cell.cell_id,
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "split": experiment.split,
        "condition": task.condition.condition_id,
        "source_id": task.source_id,
        "family_id": task.family_id,
        "distance_model": task.distance_model,
        "patch_size": experiment.patch_size,
        "rank_base": 0,
        "tie_break": "stable_row_major_patch_index",
        "paper_rank_semantics": "mean_over_patch_and_channels(abs(full_attribution))",
        "target_policy": "full_reference_clean_fp32_prediction",
        "methods": list(methods),
        "ballot_labels": list(ballot_labels),
        "simpleavg_semantics": {
            "channel_reduction": "mean(abs(attribution), channels)",
            "per_method_spatial_normalization": experiment.base.phase2.simpleavg_normalization,
            "method_reduction": "arithmetic_mean",
            "patch_reduction": "arithmetic_mean",
        },
        "rules": ["SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"],
        "rule_parameters": {
            "rrf_c": experiment.base.phase2.rrf_c,
            "kemeny_starts": experiment.base.phase2.kemeny_starts,
            "kemeny_max_passes": experiment.base.phase2.kemeny_max_passes,
        },
        "aggregation_backend": "torch-cuda-semantic-parity-v1",
        "aggregation_statistics": statistics,
        "runtime_telemetry": {
            "stages": timings.summary(),
            "gpu": gpu_telemetry,
        },
        "sample_count": sample_count,
        "attribution_sources": source_records,
        "naive_rank_source": (
            None
            if base_rank_manifest is None
            else {
                "root": base_rank_root,
                "task_digest": base_rank_manifest["task_digest"],
                "semantics": "exact_saved_single_method_ranks",
            }
        ),
        "selection_task_id": task.selection_task_id,
        "shards": ordered,
    }
    publish_manifest(
        experiment, output, root=task.artifact_root, task_id=task.task_id, manifest=value
    )
    return value


__all__ = ["run_rank_task"]
