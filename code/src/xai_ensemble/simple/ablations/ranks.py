"""NAIVE rank construction, independent of mask-game evaluation settings."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.phase2.torch_aggregation import aggregate_rankings_torch

from ..artifacts import (
    PHASE1_SCHEMA_VERSION,
    ArtifactError,
    completed_manifest,
    phase1_artifact_root,
)
from ..methods import PATCH_METHODS
from ..phase2 import (
    _aligned,
    _load_method_rank_ready_shard,
    _summarize_aggregation_statistics,
)
from ..rank_ready import rank_field, simpleavg_spatial_field
from .artifacts import (
    RANK_SCHEMA_VERSION,
    completed_rank_manifest,
    existing_shard_records,
    output_store,
    publish_manifest,
    publish_shard,
)
from .config import AblationExperiment, RankConstructionTask

_CUDA_WORKSPACE_BYTES = 2 * 2**30
_CPU_WORKSPACE_BYTES = 64 * 2**20


def _phase1_sources(
    experiment: AblationExperiment,
    task: RankConstructionTask,
) -> Mapping[str, tuple[str, Mapping[str, Any]]]:
    store = output_store(experiment)
    by_id = {item.task_id: item for item in experiment.phase1_tasks()}
    result = {}
    layout = None
    for task_id in task.phase1_task_ids:
        producer = by_id[task_id]
        artifact_name = (
            f"{producer.family}__p{task.construction.patch_size}"
            if producer.family in PATCH_METHODS
            else producer.family
        )
        variants = {variant.artifact_name: variant for variant in producer.variants}
        if artifact_name not in variants:
            raise ArtifactError(f"No p={task.construction.patch_size} {producer.family} variant")
        root = phase1_artifact_root(producer, artifact_name)
        manifest = completed_manifest(
            store,
            root,
            expected_task_digest=producer.digest,
            expected_schema_version=PHASE1_SCHEMA_VERSION,
        )
        if manifest is None:
            raise FileNotFoundError(f"Phase 1 source is incomplete: {root}")
        method = manifest.get("method")
        if (
            not isinstance(method, Mapping)
            or method.get("variant_digest") != variants[artifact_name].digest
        ):
            raise ArtifactError(f"Phase 1 method identity mismatch: {root}")
        current_layout = tuple(
            (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
            for item in manifest["shards"]
        )
        if layout is None:
            layout = current_layout
        elif current_layout != layout:
            raise ArtifactError(f"Phase 1 shard layout differs for {producer.family}")
        result[producer.family] = (root, manifest)
    if tuple(result) != task.construction.methods:
        raise ArtifactError("Phase 1 source roster differs from the construction identity")
    return result


def _aggregate(
    ballots: np.ndarray,
    simple_scores: np.ndarray,
    *,
    experiment: AblationExperiment,
    task: RankConstructionTask,
    indices: np.ndarray,
    device: Any,
) -> Any:
    seeds = tuple(
        int(object_sha256({"rank_task": task.digest, "row_index": int(index)})[:15], 16)
        for index in indices
    )
    return aggregate_rankings_torch(
        ballots,
        simple_scores,
        requested=task.construction.rules,
        rrf_c=experiment.base.phase2.rrf_c,
        kemeny_starts=experiment.base.phase2.kemeny_starts,
        kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
        seeds=seeds,
        device=device,
        workspace_bytes=(
            _CUDA_WORKSPACE_BYTES
            if getattr(device, "type", str(device)) == "cuda"
            else _CPU_WORKSPACE_BYTES
        ),
    )


def run_rank_task(
    experiment: AblationExperiment,
    task: RankConstructionTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    """Aggregate aligned Phase 1 attributions into a reusable rank bank."""

    import torch

    store = output_store(experiment)
    complete = completed_rank_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    generation = experiment.generation_experiment()
    sources = _phase1_sources(experiment, task)
    first_manifest = next(iter(sources.values()))[1]
    layout = tuple(
        (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
        for item in first_manifest["shards"]
    )
    records = existing_shard_records(
        store,
        task.artifact_root,
        task_digest=task.digest,
        schema_version=RANK_SCHEMA_VERSION,
        source_layout=layout,
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU rank construction requires CUDA")

    for shard_index, start, stop in layout:
        if shard_index in records:
            continue
        reference: Mapping[str, Any] | None = None
        ballots_by_method = {}
        simple_sum = None
        height = width = 0
        for family in task.construction.methods:
            _, manifest = sources[family]
            fields = _load_method_rank_ready_shard(
                generation,
                store,
                family=family,
                manifest=manifest,
                shard_index=shard_index,
            )
            if reference is None:
                reference = {
                    key: fields[key]
                    for key in ("indices", "labels", "predictions", "logits", "targets")
                }
            else:
                _aligned(reference, fields, family)
            ballots_by_method[family] = (
                fields[rank_field(task.construction.patch_size)]
                .numpy()
                .astype(np.int64, copy=False)
                .copy()
            )
            spatial = fields[simpleavg_spatial_field()].numpy().astype(np.float32, copy=False)
            simple_sum = spatial.copy() if simple_sum is None else simple_sum + spatial
            height, width = spatial.shape[-2:]
            del fields
        assert reference is not None and simple_sum is not None
        indices = reference["indices"].numpy().astype(np.int64, copy=False)
        ballots = np.stack(
            [ballots_by_method[family] for family in task.construction.methods], axis=1
        )
        averaged = simple_sum / float(len(task.construction.methods))
        grid_h = height // task.construction.patch_size
        grid_w = width // task.construction.patch_size
        simple_scores = averaged.reshape(
            averaged.shape[0],
            grid_h,
            task.construction.patch_size,
            grid_w,
            task.construction.patch_size,
        ).mean(axis=(2, 4), dtype=np.float32)
        if target_device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        aggregation = _aggregate(
            ballots,
            simple_scores,
            experiment=experiment,
            task=task,
            indices=indices,
            device=target_device,
        )
        rules = dict(aggregation)
        for family in task.construction.methods:
            rules[f"single__{family}"] = ballots_by_method[family]
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
        records[shard_index] = publish_shard(
            experiment,
            store,
            root=task.artifact_root,
            task_id=task.task_id,
            task_digest=task.digest,
            schema_version=RANK_SCHEMA_VERSION,
            shard_index=shard_index,
            start=start,
            stop=stop,
            tensors=tensors,
            metadata={
                "rank_base": "0",
                "patch_size": str(task.construction.patch_size),
                "precision": "fp32",
            },
            record_fields={
                "rule_labels": rule_labels,
                "aggregation_statistics": aggregation.statistics,
            },
        )
        print(
            f"ABLATION_RANK shard={shard_index + 1}/{len(layout)} task={task.task_id}",
            flush=True,
        )
        del (
            reference,
            ballots_by_method,
            ballots,
            simple_sum,
            simple_scores,
            aggregation,
            rules,
            tensors,
        )
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    ordered = [records[index] for index, _, _ in layout]
    sample_count = sum(int(item["count"]) for item in ordered)
    statistics = _summarize_aggregation_statistics(
        ordered,
        require_kemeny="Kemeny" in task.construction.rules,
    )
    manifest: Mapping[str, Any] = {
        "schema_version": RANK_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "construction": asdict(task.construction),
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "split": experiment.split,
        "condition": task.condition.condition_id,
        "patch_size": task.construction.patch_size,
        "rank_base": 0,
        "tie_break": "stable_row_major_patch_index",
        "paper_rank_semantics": "mean_over_patch_and_channels(abs(full_attribution))",
        "simpleavg_semantics": {
            "channel_reduction": "mean(abs(attribution), channels)",
            "per_method_spatial_normalization": experiment.base.phase2.simpleavg_normalization,
            "method_reduction": "arithmetic_mean",
            "patch_reduction": "arithmetic_mean",
        },
        "rule_parameters": {
            "rrf_c": experiment.base.phase2.rrf_c,
            "kemeny_starts": experiment.base.phase2.kemeny_starts,
            "kemeny_max_passes": experiment.base.phase2.kemeny_max_passes,
        },
        "aggregation_backend": f"torch-{target_device.type}-semantic-parity-v1",
        "aggregation_statistics": statistics,
        "methods": list(task.construction.methods),
        "sample_count": sample_count,
        "source_manifests": {
            family: {"root": root, "task_digest": source["task_digest"]}
            for family, (root, source) in sources.items()
        },
        "shards": ordered,
    }
    publish_manifest(
        experiment,
        store,
        root=task.artifact_root,
        task_id=task.task_id,
        manifest=manifest,
    )
    return manifest


__all__ = ["run_rank_task"]
