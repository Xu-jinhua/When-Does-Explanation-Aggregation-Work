"""Phase 2: paper-defined patch ranks, aggregation, and masking evaluation."""

from __future__ import annotations

import gc
import os
import posixpath
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase1.relprop import relprop_attribution_provider
from xai_ensemble.phase2.evaluator import FillReference, evaluate_reference_model_bank
from xai_ensemble.phase2.metrics import QUALITY_METRICS
from xai_ensemble.phase2.torch_aggregation import (
    AggregationResult,
    aggregate_rankings_torch,
)

from .artifacts import (
    PHASE1_SCHEMA_VERSION,
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    load_safetensors,
    phase1_artifact_root,
    phase2_artifact_root,
    write_phase2_shard,
)
from .config import Phase1Task, Phase2Task, SimpleExperiment
from .data import apply_condition, load_model, load_raw_dataset_mean, load_split
from .manifest_identity import dataset_manifest_identity_sha256
from .methods import PATCH_METHODS
from .rank_ready import (
    attribution_to_patch_scores,
    ensure_rank_ready_sidecar,
    rank_field,
    scores_to_ranks,
    simpleavg_spatial_field,
)

_MIN_AGGREGATION_WORKSPACE_BYTES = 64 * 2**20
_MAX_AGGREGATION_WORKSPACE_BYTES = 2 * 2**30
_AGGREGATION_RESERVATION_SAFETY_BYTES = 256 * 2**20


def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Compatibility alias for callers that historically imported this helper."""

    return scores_to_ranks(scores)


def _digest_path(path: Path) -> str:
    path = resolve_full_matrix_runtime_path(path)
    if path.is_file():
        return file_sha256(path)
    manifest = path / "manifest.json"
    if manifest.is_file():
        return file_sha256(manifest)
    raise FileNotFoundError(path)


def _current_source_identity(
    experiment: SimpleExperiment,
    task: Phase2Task,
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
    if task.condition.kind == "adversarial":
        from .adversarial import adversarial_source_binding

        attack_task = experiment.adversarial_task_for(
            dataset_id=task.dataset.dataset_id,
            model_id=task.model.model_id,
            split=task.split,
            condition_id=task.condition.condition_id,
        )
        values.update(adversarial_source_binding(experiment, attack_task, store=store))
    return values


def _method_names(experiment: SimpleExperiment, task: Phase2Task) -> tuple[str, ...]:
    available = tuple(
        item.family for item in experiment.methods.for_architecture(task.model.architecture)
    )
    configured = task.ensemble.methods
    if configured == "architecture_default":
        return available
    unknown = set(configured) - set(available)
    if unknown:
        raise ValueError(
            f"Ensemble {task.ensemble.ensemble_id} requests methods unavailable for "
            f"{task.model.architecture}: {sorted(unknown)}"
        )
    return tuple(configured)


def _phase1_task(
    experiment: SimpleExperiment,
    task: Phase2Task,
    family: str,
) -> Phase1Task:
    for candidate in experiment.phase1_tasks():
        if (
            candidate.dataset.dataset_id == task.dataset.dataset_id
            and candidate.model.model_id == task.model.model_id
            and candidate.split == task.split
            and candidate.condition.condition_id == task.condition.condition_id
            and candidate.family == family
        ):
            return candidate
    raise KeyError(f"No Phase 1 producer for Phase 2 method {family}")


def _matching_clean_task(
    experiment: SimpleExperiment,
    task: Phase2Task,
) -> Phase2Task:
    matches = [
        candidate
        for candidate in experiment.phase2_tasks()
        if candidate.condition.kind == "clean"
        and candidate.dataset.dataset_id == task.dataset.dataset_id
        and candidate.model.model_id == task.model.model_id
        and candidate.split == task.split
        and candidate.ensemble.ensemble_id == task.ensemble.ensemble_id
        and candidate.patch_size == task.patch_size
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching clean Phase 2 task, found {len(matches)}")
    return matches[0]


def _source_manifests(
    experiment: SimpleExperiment,
    task: Phase2Task,
    store: ArtifactStore,
    methods: Sequence[str],
) -> Mapping[str, tuple[str, Mapping[str, Any]]]:
    result = {}
    reference_layout = None
    for family in methods:
        producer = _phase1_task(experiment, task, family)
        artifact_name = f"{family}__p{task.patch_size}" if family in PATCH_METHODS else family
        variants = {item.artifact_name: item for item in producer.variants}
        if artifact_name not in variants:
            raise ValueError(
                f"Phase 2 p={task.patch_size} has no matching {family} artifact variant"
            )
        root = phase1_artifact_root(producer, artifact_name)
        manifest = completed_manifest(
            store,
            root,
            expected_task_digest=producer.digest,
            expected_schema_version=PHASE1_SCHEMA_VERSION,
        )
        if manifest is None:
            raise FileNotFoundError(f"Phase 1 artifact is incomplete: {root}")
        method = manifest.get("method")
        if (
            not isinstance(method, Mapping)
            or method.get("variant_digest") != variants[artifact_name].digest
        ):
            raise ArtifactError(f"Phase 1 method identity mismatch: {root}")
        expected_provider = relprop_attribution_provider(family, task.model.architecture)
        if expected_provider:
            model = manifest.get("model")
            observed_provider = (
                model.get("attribution_provider") if isinstance(model, Mapping) else None
            )
            mismatches = {
                key: {
                    "artifact": (
                        observed_provider.get(key)
                        if isinstance(observed_provider, Mapping)
                        else None
                    ),
                    "current": expected,
                }
                for key, expected in expected_provider.items()
                if not isinstance(observed_provider, Mapping)
                or observed_provider.get(key) != expected
            }
            if mismatches:
                raise ArtifactError(
                    f"Phase 1 attribution provider mismatch for {root}: {mismatches}"
                )
        layout = tuple(
            (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
            for item in manifest["shards"]
        )
        if reference_layout is None:
            reference_layout = layout
        elif layout != reference_layout:
            raise ArtifactError(f"Phase 1 shard layout is not aligned for method {family}")
        result[family] = (root, manifest)
    return result


def _load_method_shard(
    experiment: SimpleExperiment,
    store: ArtifactStore,
    *,
    family: str,
    root: str,
    manifest: Mapping[str, Any],
    shard_index: int,
) -> Mapping[str, Any]:
    record = manifest["shards"][shard_index]
    payload = record["payload"]
    scratch = experiment.storage.scratch_root / "phase2" / "source-cache" / str(os.getpid())
    local = scratch / f"{family}--{shard_index:05d}.safetensors"
    store.materialize(
        str(payload["relative_path"]),
        local,
        expected_sha256=str(payload["sha256"]),
    )
    try:
        values = dict(load_safetensors(local))
    finally:
        if local.exists():
            local.unlink()
    return values


def _load_method_rank_ready_shard(
    experiment: SimpleExperiment,
    store: ArtifactStore,
    *,
    family: str,
    manifest: Mapping[str, Any],
    shard_index: int,
) -> Mapping[str, Any]:
    record = next(item for item in manifest["shards"] if int(item["shard_index"]) == shard_index)
    compact = ensure_rank_ready_sidecar(
        store,
        source_payload=record["payload"],
        work_directory=(
            experiment.storage.spool_root
            / "phase2-rank-ready"
            / str(os.getpid())
            / family
            / f"{shard_index:05d}"
        ),
        lock_root=experiment.storage.spool_root / "rank-ready-locks",
        simpleavg_normalization=experiment.phase2.simpleavg_normalization,
        count=int(record["stop"]) - int(record["start"]),
        recorded_sidecar=(
            record.get("rank_ready") if isinstance(record.get("rank_ready"), Mapping) else None
        ),
    )
    return compact.fields


def _aligned(reference: Mapping[str, Any], candidate: Mapping[str, Any], family: str) -> None:
    import torch

    for field in ("indices", "labels", "predictions", "logits", "targets"):
        if field not in candidate or not torch.equal(reference[field], candidate[field]):
            raise ArtifactError(f"Phase 1 {field} is not aligned for method {family}")


def _aggregate_rules(
    ballots: np.ndarray,
    simple_scores: np.ndarray | None,
    *,
    task: Phase2Task,
    experiment: SimpleExperiment,
    indices: np.ndarray,
    device: Any,
    workspace_bytes: int,
) -> AggregationResult:
    seeds = tuple(
        int(
            object_sha256(
                {
                    "task": task.digest,
                    "row_index": int(row_index),
                }
            )[:15],
            16,
        )
        for row_index in indices
    )
    return aggregate_rankings_torch(
        ballots,
        simple_scores,
        requested=task.ensemble.rules,
        rrf_c=experiment.phase2.rrf_c,
        kemeny_starts=experiment.phase2.kemeny_starts,
        kemeny_max_passes=experiment.phase2.kemeny_max_passes,
        seeds=seeds,
        device=device,
        workspace_bytes=workspace_bytes,
    )


def _aggregation_workspace_bytes(
    experiment: SimpleExperiment,
    task: Phase2Task,
    *,
    device: Any,
) -> int:
    import torch

    if device.type != "cuda":
        return _MIN_AGGREGATION_WORKSPACE_BYTES
    from .profiler import load_phase2_profile

    profile = experiment.phase2_profile_for_model(task.model)
    measurement = load_phase2_profile(experiment, profile)
    if measurement is None:
        raise RuntimeError(
            f"Phase 2 GPU aggregation requires completed profile {profile.profile_id}"
        )
    resident = int(torch.cuda.memory_allocated(device))
    available = measurement.reservation_bytes - resident - _AGGREGATION_RESERVATION_SAFETY_BYTES
    workspace = min(_MAX_AGGREGATION_WORKSPACE_BYTES, available // 2)
    if workspace < _MIN_AGGREGATION_WORKSPACE_BYTES:
        raise RuntimeError(
            f"Phase 2 profile {profile.profile_id} leaves no bounded aggregation workspace"
        )
    return workspace


def _conditioned_raw(
    raw: Any,
    labels: Any,
    indices: Any,
    *,
    experiment: SimpleExperiment,
    task: Phase2Task,
    store: ArtifactStore,
    shard_index: int,
    targets: Any,
    adversarial_logits: Any,
    model: Any,
    normalize: Any,
    device: Any,
    batch_size: int,
    seed: int,
) -> Any:
    import torch

    if task.condition.kind == "clean":
        return raw.to(dtype=torch.float32)
    if task.condition.kind == "adversarial":
        from .adversarial import load_adversarial_shard

        attack_task = experiment.adversarial_task_for(
            dataset_id=task.dataset.dataset_id,
            model_id=task.model.model_id,
            split=task.split,
            condition_id=task.condition.condition_id,
        )
        fields = load_adversarial_shard(
            experiment,
            attack_task,
            shard_index,
            expected_indices=indices,
            expected_labels=labels,
            clean_images=raw,
            expected_targets=targets,
            expected_adversarial_logits=adversarial_logits,
            store=store,
        )
        return fields["adversarial_images"]
    chunks = []
    for start in range(0, int(raw.shape[0]), batch_size):
        stop = min(int(raw.shape[0]), start + batch_size)
        batch = raw[start:stop].to(device, dtype=torch.float32, non_blocking=True)
        conditioned = apply_condition(
            task.condition,
            batch,
            labels=labels[start:stop].to(device),
            indices=indices[start:stop].to(device),
            model=model,
            normalize=normalize,
            seed=seed,
        )
        chunks.append(conditioned.detach().cpu())
    return torch.cat(chunks)


def _output_names(index: int) -> tuple[str, str]:
    stem = f"shard-{index:05d}"
    return f"shards/{stem}.safetensors", f"shards/{stem}.json"


def _existing_output_records(
    store: ArtifactStore,
    root: str,
    *,
    task: Phase2Task,
    source_layout: Sequence[tuple[int, int, int]],
) -> dict[int, Mapping[str, Any]]:
    result = {}
    for shard_index, start, stop in source_layout:
        payload_name, record_name = _output_names(shard_index)
        relative_record = posixpath.join(root, record_name)
        if not store.exists(relative_record):
            continue
        record = store.read_json(relative_record)
        expected = {
            "schema_version": PHASE2_SCHEMA_VERSION,
            "task_digest": task.digest,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ArtifactError(f"Contradictory Phase 2 shard record: {relative_record}")
        payload = record.get("payload")
        expected_payload = posixpath.join(root, payload_name)
        if not isinstance(payload, Mapping) or payload.get("relative_path") != expected_payload:
            raise ArtifactError(f"Malformed Phase 2 shard record: {relative_record}")
        if not store.exists(f"{expected_payload}.receipt.json"):
            raise ArtifactError(f"Unverified Phase 2 shard payload: {expected_payload}")
        result[shard_index] = record
    return result


def _publish_output_shard(
    experiment: SimpleExperiment,
    store: ArtifactStore,
    task: Phase2Task,
    *,
    root: str,
    shard_index: int,
    start: int,
    stop: int,
    tensors: Mapping[str, Any],
    rule_labels: Mapping[str, str],
    metric_sums: Mapping[str, Mapping[str, float]],
    aggregation_statistics: Mapping[str, Mapping[str, int | float]],
) -> Mapping[str, Any]:
    payload_name, record_name = _output_names(shard_index)
    scratch = experiment.storage.scratch_root / "phase2" / task.task_id
    scratch.mkdir(parents=True, exist_ok=True)
    local_payload = scratch / Path(payload_name).name
    write_phase2_shard(
        local_payload,
        tensors=tensors,
        metadata={
            "schema_version": str(PHASE2_SCHEMA_VERSION),
            "task_id": task.task_id,
            "task_digest": task.digest,
            "rank_base": "0",
            "patch_size": str(task.patch_size),
            "rules": ",".join(rule_labels),
            "precision": "fp32",
        },
    )
    published = store.publish(local_payload, posixpath.join(root, payload_name))
    local_payload.unlink()
    record: Mapping[str, Any] = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "task_digest": task.digest,
        "shard_index": shard_index,
        "start": start,
        "stop": stop,
        "count": stop - start,
        "rule_labels": dict(rule_labels),
        "metric_sums": {
            rule: {metric: float(value) for metric, value in sums.items()}
            for rule, sums in metric_sums.items()
        },
        "aggregation_statistics": {
            rule: {str(key): value for key, value in values.items()}
            for rule, values in aggregation_statistics.items()
        },
        "payload": {
            "relative_path": published.relative_path,
            "sha256": published.sha256,
            "size_bytes": published.size_bytes,
        },
    }
    local_record = scratch / Path(record_name).name
    atomic_write_json(local_record, record)
    store.publish(local_record, posixpath.join(root, record_name))
    local_record.unlink()
    return record


def _metric_sums(trace: Any) -> Mapping[str, float]:
    contributions = trace.stats.contributions()
    return {
        metric: float(np.sum(contributions[metric], dtype=np.float64)) for metric in QUALITY_METRICS
    }


def _summarize_aggregation_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    require_kemeny: bool,
) -> Mapping[str, Mapping[str, int | float]]:
    rows: list[Mapping[str, Any]] = []
    for record in records:
        statistics = record.get("aggregation_statistics")
        if not isinstance(statistics, Mapping):
            raise ArtifactError("Phase 2 shard is missing aggregation statistics")
        kemeny = statistics.get("kemeny")
        if kemeny is None:
            if require_kemeny:
                raise ArtifactError("Phase 2 Kemeny shard is missing search statistics")
            continue
        if not isinstance(kemeny, Mapping):
            raise ArtifactError("Malformed Phase 2 Kemeny search statistics")
        if int(kemeny.get("sample_count", -1)) != int(record["count"]):
            raise ArtifactError("Phase 2 Kemeny statistics do not match shard size")
        rows.append(kemeny)

    if not rows:
        return {}
    starts = {int(row["starts_requested"]) for row in rows}
    pass_caps = {int(row["max_passes"]) for row in rows}
    if len(starts) != 1 or len(pass_caps) != 1:
        raise ArtifactError("Phase 2 shards used inconsistent Kemeny search budgets")

    summed_keys = (
        "sample_count",
        "search_instance_count",
        "total_moves",
        "cap_hit_count",
        "converged_count",
        "borda_objective_sum",
        "final_objective_sum",
        "objective_improvement_sum",
    )
    totals = {key: sum(int(row[key]) for row in rows) for key in summed_keys}
    if totals["cap_hit_count"] + totals["converged_count"] != totals["search_instance_count"]:
        raise ArtifactError("Contradictory Phase 2 Kemeny convergence statistics")
    if (
        totals["borda_objective_sum"] - totals["final_objective_sum"]
        != totals["objective_improvement_sum"]
        or totals["objective_improvement_sum"] < 0
    ):
        raise ArtifactError("Kemeny objective statistics violate the Borda guarantee")
    instances = totals["search_instance_count"]
    if instances <= 0:
        raise ArtifactError("Phase 2 Kemeny statistics contain no search instances")
    return {
        "kemeny": {
            **totals,
            "starts_requested": starts.pop(),
            "max_passes": pass_caps.pop(),
            "max_moves": max(int(row["max_moves"]) for row in rows),
            "converged_fraction": float(totals["converged_count"] / instances),
        }
    }


def run_phase2_task(
    experiment: SimpleExperiment,
    task: Phase2Task,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    """Read aligned attribution shards, aggregate ranks, and evaluate each rule."""

    import torch

    store = ArtifactStore(experiment)
    root = phase2_artifact_root(task)
    current_source = _current_source_identity(experiment, task, store)
    complete = completed_manifest(
        store,
        root,
        expected_task_digest=task.digest,
        expected_schema_version=PHASE2_SCHEMA_VERSION,
    )
    if complete is not None:
        if complete.get("source_identity") != dict(current_source):
            raise ArtifactError(
                "Completed Phase 2 artifact source identity differs from current inputs"
            )
        return complete
    methods = _method_names(experiment, task)
    sources = _source_manifests(experiment, task, store, methods)
    for family, (_, source_manifest) in sources.items():
        dataset_identity = source_manifest.get("dataset")
        model_identity = source_manifest.get("model")
        baseline_identity = source_manifest.get("baseline")
        recorded_source = source_manifest.get("source_identity")
        if isinstance(recorded_source, Mapping):
            observed = {key: recorded_source.get(key) for key in current_source}
        else:
            observed = {
                "dataset_manifest_sha256": (
                    dataset_identity.get("manifest_sha256")
                    if isinstance(dataset_identity, Mapping)
                    else None
                ),
                "checkpoint_sha256": (
                    model_identity.get("checkpoint_sha256")
                    if isinstance(model_identity, Mapping)
                    else None
                ),
                "mean_artifact_digest": (
                    baseline_identity.get("mean_artifact_digest")
                    if isinstance(baseline_identity, Mapping)
                    else None
                ),
            }
        if observed != current_source:
            raise ArtifactError(
                f"Current Phase 2 inputs differ from {family} Phase 1 sources: "
                f"artifact={observed}, current={current_source}"
            )
    first_manifest = next(iter(sources.values()))[1]
    source_layout = tuple(
        (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
        for item in first_manifest["shards"]
    )
    output_records = _existing_output_records(
        store,
        root,
        task=task,
        source_layout=source_layout,
    )
    target_device = torch.device(device)
    loaded_model = load_model(task.model, device=target_device, include_checkpoint=True)
    aggregation_workspace = _aggregation_workspace_bytes(
        experiment,
        task,
        device=target_device,
    )
    print(
        "PHASE2_AGGREGATION "
        f"backend=torch-{target_device.type} "
        f"workspace_gib={aggregation_workspace / 2**30:.2f} "
        f"task={task.task_id}"
    )
    bundle = load_split(
        task.dataset,
        loaded_model,
        split=task.split,
        workers=experiment.runtime.dataloader_workers,
        shared_cache_root=experiment.storage.spool_root / "shared-cache",
    )
    raw_mean = load_raw_dataset_mean(
        task.model, input_size=int(loaded_model.preprocessing["input_size"])
    )
    fill_id = object_sha256(
        {
            "path": str(task.model.mean_path),
            "digest": (
                file_sha256(resolve_full_matrix_runtime_path(task.model.mean_path))
                if resolve_full_matrix_runtime_path(task.model.mean_path).is_file()
                else file_sha256(
                    resolve_full_matrix_runtime_path(task.model.mean_path / "manifest.json")
                )
            ),
        }
    )
    fill = FillReference(
        values=raw_mean.squeeze(0).numpy(),
        source_split="train",
        artifact_id=fill_id,
    )

    for shard_index, start, stop in source_layout:
        if shard_index in output_records:
            continue
        reference_fields: Mapping[str, Any] | None = None
        ballots_by_method: dict[str, np.ndarray] = {}
        simple_sum: np.ndarray | None = None
        height = width = 0
        for family in methods:
            _, source_manifest = sources[family]
            fields = _load_method_rank_ready_shard(
                experiment,
                store,
                family=family,
                manifest=source_manifest,
                shard_index=shard_index,
            )
            if reference_fields is None:
                reference_fields = {
                    key: fields[key]
                    for key in ("indices", "labels", "predictions", "logits", "targets")
                }
            else:
                _aligned(reference_fields, fields, family)
            ballots_by_method[family] = (
                fields[rank_field(task.patch_size)].numpy().astype(np.int64, copy=False).copy()
            )
            if "SimpleAvg" in task.ensemble.rules:
                spatial = fields[simpleavg_spatial_field()].numpy().astype(np.float32, copy=False)
                simple_sum = spatial.copy() if simple_sum is None else simple_sum + spatial
                height, width = spatial.shape[-2:]
            del fields
        assert reference_fields is not None
        indices = reference_fields["indices"].numpy().astype(np.int64, copy=False)
        labels = reference_fields["labels"].numpy().astype(np.int64, copy=False)
        predictions = reference_fields["predictions"].numpy().astype(np.int64, copy=False)
        logits = reference_fields["logits"]
        if logits.ndim != 2 or int(logits.shape[0]) != len(indices):
            raise ArtifactError("Phase 1 logits do not have [N,num_classes] shape")
        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise ArtifactError("Phase 1 logits must be finite FP32 values")
        if not np.array_equal(
            predictions,
            torch.argmax(logits, dim=1).numpy().astype(np.int64, copy=False),
        ):
            raise ArtifactError("Phase 1 predictions do not equal argmax(logits)")
        targets = reference_fields["targets"].numpy().astype(np.int64, copy=False)
        ballots = np.stack([ballots_by_method[family] for family in methods], axis=1)
        simple_patch_scores = None
        if simple_sum is not None:
            averaged = simple_sum / float(len(methods))
            grid_h, grid_w = height // task.patch_size, width // task.patch_size
            simple_patch_scores = averaged.reshape(
                averaged.shape[0],
                grid_h,
                task.patch_size,
                grid_w,
                task.patch_size,
            ).mean(axis=(2, 4), dtype=np.float32)
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
            torch.cuda.synchronize(target_device)
        aggregation_started = time.monotonic()
        aggregation_result = _aggregate_rules(
            ballots,
            simple_patch_scores,
            task=task,
            experiment=experiment,
            indices=indices,
            device=target_device,
            workspace_bytes=aggregation_workspace,
        )
        rules = dict(aggregation_result)
        aggregation_statistics = aggregation_result.statistics
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        aggregation_seconds = time.monotonic() - aggregation_started
        aggregation_peak = (
            max(
                int(torch.cuda.max_memory_allocated(target_device)),
                int(torch.cuda.max_memory_reserved(target_device)),
            )
            if target_device.type == "cuda"
            else 0
        )
        kemeny_statistics = aggregation_statistics.get("kemeny")
        kemeny_log = (
            ""
            if kemeny_statistics is None
            else (
                f" kemeny_max_moves={int(kemeny_statistics['max_moves'])}"
                f" kemeny_cap_hits={int(kemeny_statistics['cap_hit_count'])}"
            )
        )
        print(
            "PHASE2_AGGREGATED "
            f"shard={shard_index + 1}/{len(source_layout)} "
            f"samples={len(indices)} patches={ballots.shape[2]} "
            f"seconds={aggregation_seconds:.3f} "
            f"peak_gib={aggregation_peak / 2**30:.2f} "
            f"task={task.task_id}{kemeny_log}"
        )
        if task.ensemble.include_singles:
            for family in methods:
                rules[f"single__{family}"] = ballots_by_method[family]
        raw_images, observed_labels = bundle.rows(indices)
        if not torch.equal(observed_labels.to(dtype=torch.int64), reference_fields["labels"]):
            raise ArtifactError("Dataset labels changed after Phase 1 publication")
        conditioned = _conditioned_raw(
            raw_images,
            reference_fields["labels"],
            reference_fields["indices"],
            experiment=experiment,
            task=task,
            store=store,
            shard_index=shard_index,
            targets=reference_fields["targets"],
            adversarial_logits=reference_fields["logits"],
            model=loaded_model.model,
            normalize=loaded_model.normalize,
            device=target_device,
            batch_size=experiment.phase2.inference_batch_size,
            seed=experiment.runtime.seed,
        )
        output_tensors: dict[str, Any] = {
            "indices": reference_fields["indices"],
            "labels": reference_fields["labels"],
            "targets": reference_fields["targets"],
            "unmasked_predictions": reference_fields["predictions"],
        }
        traces = evaluate_reference_model_bank(
            loaded_model.model,
            conditioned,
            rules,
            true_labels=labels,
            target_labels=targets,
            fill_reference=fill,
            reference_model_id=task.model.model_id,
            sample_ids=indices,
            patch_size=task.patch_size,
            k=experiment.phase2.k,
            index_base=0,
            batch_size=experiment.phase2.inference_batch_size,
            device=str(target_device),
            autocast=False,
            normalize=loaded_model.normalize,
            require_target_matches_clean=task.condition.kind == "clean",
            clean_predictions=predictions,
        )
        metric_sums = {}
        rule_labels = {}
        for rule_index, (rule_name, ranks) in enumerate(rules.items()):
            field_id = f"r{rule_index:03d}"
            rule_labels[field_id] = rule_name
            trace = traces[rule_name]
            if not np.array_equal(trace.clean_predictions, predictions):
                raise RuntimeError("Evaluator did not retain the supplied Phase 1 predictions")
            output_tensors[f"rank__{field_id}"] = torch.from_numpy(ranks).to(torch.int32)
            output_tensors[f"removed_predictions__{field_id}"] = torch.from_numpy(
                trace.removed_predictions
            )
            output_tensors[f"retained_predictions__{field_id}"] = torch.from_numpy(
                trace.retained_predictions
            )
            metric_sums[field_id] = _metric_sums(trace)
        output_records[shard_index] = _publish_output_shard(
            experiment,
            store,
            task,
            root=root,
            shard_index=shard_index,
            start=start,
            stop=stop,
            tensors=output_tensors,
            rule_labels=rule_labels,
            metric_sums=metric_sums,
            aggregation_statistics=aggregation_statistics,
        )
        print(f"PHASE2 shard={shard_index + 1}/{len(source_layout)} task={task.task_id}")
        del (
            reference_fields,
            ballots_by_method,
            ballots,
            aggregation_result,
            aggregation_statistics,
            rules,
            traces,
            conditioned,
            output_tensors,
        )

    ordered = [output_records[index] for index, _, _ in source_layout]
    total_count = sum(int(item["count"]) for item in ordered)
    aggregation_statistics = _summarize_aggregation_statistics(
        ordered,
        require_kemeny="Kemeny" in task.ensemble.rules,
    )
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {metric: 0.0 for metric in QUALITY_METRICS}
    )
    labels_by_field: dict[str, str] = {}
    for record in ordered:
        labels_by_field.update({str(k): str(v) for k, v in record["rule_labels"].items()})
        for field_id, values in record["metric_sums"].items():
            for metric in QUALITY_METRICS:
                sums[str(field_id)][metric] += float(values[metric])
    metrics = {
        labels_by_field[field_id]: {metric: value / total_count for metric, value in values.items()}
        for field_id, values in sums.items()
    }
    robustness = None
    if task.condition.kind != "clean":
        clean_task = _matching_clean_task(experiment, task)
        clean_root = phase2_artifact_root(clean_task)
        clean_manifest = completed_manifest(
            store,
            clean_root,
            expected_task_digest=clean_task.digest,
            expected_schema_version=PHASE2_SCHEMA_VERSION,
        )
        if clean_manifest is None:
            raise FileNotFoundError(
                f"Perturbed evaluation requires its completed clean counterpart: {clean_root}"
            )
        if (
            clean_manifest.get("sample_count") != total_count
            or clean_manifest.get("methods") != list(methods)
            or clean_manifest.get("patch_size") != task.patch_size
        ):
            raise ArtifactError("Clean and perturbed Phase 2 manifests are not aligned")
        clean_metrics = clean_manifest.get("metrics")
        if not isinstance(clean_metrics, Mapping) or set(clean_metrics) != set(metrics):
            raise ArtifactError("Clean and perturbed rule sets are not aligned")
        directions = {"F": "max", "Fbar": "min", "C": "max", "Cbar": "min"}
        robustness = {}
        for rule, current_values in metrics.items():
            clean_values = clean_metrics[rule]
            robustness[rule] = {
                "absolute": {
                    metric: abs(float(clean_values[metric]) - float(current_values[metric]))
                    for metric in QUALITY_METRICS
                },
                "signed_degradation": {
                    metric: (
                        float(clean_values[metric]) - float(current_values[metric])
                        if directions[metric] == "max"
                        else float(current_values[metric]) - float(clean_values[metric])
                    )
                    for metric in QUALITY_METRICS
                },
            }
    manifest: Mapping[str, Any] = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment.experiment_id,
        "experiment_digest": experiment.digest,
        "phase1_experiment_digest": experiment.phase1_digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "ensemble": task.ensemble.ensemble_id,
        "methods": list(methods),
        "patch_size": task.patch_size,
        "patch_setting": (
            "primary" if task.patch_size == experiment.phase2.primary_patch_size else "additional"
        ),
        "k": experiment.phase2.k,
        "precision": "fp32",
        "target_policy": "clean_model_fp32_prediction",
        "unmasked_prediction_source": "phase1_fp32_logits_argmax",
        "inference_batch_size": experiment.phase2.inference_batch_size,
        "inference_batch_semantics": "maximum_actual_model_forward_batch",
        "aggregation_backend": f"torch-{target_device.type}-semantic-parity-v1",
        "aggregation_workspace_bytes": aggregation_workspace,
        "aggregation_statistics": aggregation_statistics,
        "source_identity": dict(current_source),
        "rank_base": 0,
        "tie_break": "stable_row_major_patch_index",
        "paper_rank_semantics": "mean_over_patch_and_channels(abs(full_attribution))",
        "simpleavg_semantics": {
            "channel_reduction": "mean(abs(attribution), channels)",
            "per_method_spatial_normalization": experiment.phase2.simpleavg_normalization,
            "method_reduction": "arithmetic_mean",
            "patch_reduction": "arithmetic_mean",
        },
        "rule_parameters": {
            "rrf_c": experiment.phase2.rrf_c,
            "kemeny_starts": experiment.phase2.kemeny_starts,
            "kemeny_max_passes": experiment.phase2.kemeny_max_passes,
        },
        "source_manifests": {
            family: {
                "root": root_value,
                "task_digest": manifest_value["task_digest"],
                "variant_digest": manifest_value["method"]["variant_digest"],
            }
            for family, (root_value, manifest_value) in sources.items()
        },
        "sample_count": total_count,
        "metrics": metrics,
        "robustness": robustness,
        "shards": ordered,
    }
    scratch = experiment.storage.scratch_root / "phase2" / task.task_id
    local_manifest = scratch / "manifest.json"
    atomic_write_json(local_manifest, manifest)
    store.publish(local_manifest, posixpath.join(root, "manifest.json"), write_receipt=False)
    del bundle, raw_mean, loaded_model
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


__all__ = ["attribution_to_patch_scores", "run_phase2_task"]
