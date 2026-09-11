"""GPU aggregation and mask-game evaluation for every Fidelity prefix."""

from __future__ import annotations

import gc
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from datetime import UTC, datetime
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.phase2.evaluator import evaluate_reference_model_bank
from xai_ensemble.phase2.metrics import QUALITY_METRICS
from xai_ensemble.phase2.torch_aggregation import aggregate_rankings_torch

from ..artifacts import ArtifactError
from ..assumptions.evaluator import _conditioned_images, _fill
from ..data import load_model, load_split, materialize_shared_conditioned_raw_images
from ..io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from ..phase2 import _metric_sums
from ..runtime import emit_gpu_release_signal
from ..spool import SpoolQuota
from ..telemetry import GpuUtilizationSampler, StageTimings
from .artifacts import (
    EVALUATION_SCHEMA_VERSION,
    PrefixShardPublisher,
    completed_evaluation_manifest,
    completed_shards,
    output_store,
    publish_manifest,
)
from .config import RULE_IDS, NoisePrefixExperiment, PrefixEvaluationTask
from .inputs import (
    catalog_task_input,
    input_byte_count,
    load_input_catalog,
    load_prefix_shard,
    prepare_input_catalog,
    source_layout,
)


def _rule_key(q: int, rule: str) -> str:
    return f"q{q:02d}__{rule.lower()}"


def _prefix_seeds(task: PrefixEvaluationTask, q: int, indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(
            object_sha256({"task": task.digest, "q": int(q), "row_index": int(row_index)})[:15],
            16,
        )
        for row_index in indices
    )


def aggregate_prefix_bank(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
    *,
    ballots: np.ndarray,
    simple_scores_by_q: Mapping[int, np.ndarray],
    indices: np.ndarray,
    device: Any,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Mapping[str, int | float]]]:
    """Build the q=2..10 rule bank without retaining intermediate GPU tensors."""

    bank = {}
    statistics = {}
    for q in experiment.q_values[:-1]:
        result = aggregate_rankings_torch(
            ballots[:, :q],
            simple_scores_by_q[q],
            requested=experiment.rules,
            rrf_c=experiment.base.phase2.rrf_c,
            kemeny_starts=experiment.base.phase2.kemeny_starts,
            kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
            seeds=_prefix_seeds(task, q, indices),
            device=device,
            workspace_bytes=experiment.runtime.aggregation_workspace_bytes,
        )
        for rule, ranks in result.items():
            bank[_rule_key(q, rule)] = ranks
        for rule, values in result.statistics.items():
            statistics[_rule_key(q, rule)] = dict(values)
    expected = {_rule_key(q, rule) for q in experiment.q_values[:-1] for rule in RULE_IDS}
    if set(bank) != expected:
        raise RuntimeError(
            f"Prefix aggregation rule bank is incomplete: {sorted(expected - set(bank))}"
        )
    return bank, statistics


def _summarize_kemeny(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, int | float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        values = record.get("aggregation_statistics")
        if not isinstance(values, Mapping):
            raise ArtifactError("Prefix shard is missing aggregation statistics")
        for key, row in values.items():
            if not isinstance(row, Mapping):
                raise ArtifactError(f"Malformed Kemeny statistics for {key}")
            grouped[str(key)].append(row)
    expected = {_rule_key(q, "kemeny") for q in range(2, 11)}
    if set(grouped) != expected:
        raise ArtifactError("Prefix shards do not cover every q-specific Kemeny search")

    result = {}
    summed = (
        "sample_count",
        "search_instance_count",
        "total_moves",
        "cap_hit_count",
        "converged_count",
        "borda_objective_sum",
        "final_objective_sum",
        "objective_improvement_sum",
    )
    for key, rows in grouped.items():
        starts = {int(row["starts_requested"]) for row in rows}
        pass_caps = {int(row["max_passes"]) for row in rows}
        if len(starts) != 1 or len(pass_caps) != 1:
            raise ArtifactError(f"Inconsistent Kemeny budget for {key}")
        totals = {name: sum(int(row[name]) for row in rows) for name in summed}
        instances = totals["search_instance_count"]
        if instances <= 0 or (totals["cap_hit_count"] + totals["converged_count"] != instances):
            raise ArtifactError(f"Contradictory Kemeny convergence totals for {key}")
        if (
            totals["borda_objective_sum"] - totals["final_objective_sum"]
            != totals["objective_improvement_sum"]
            or totals["objective_improvement_sum"] < 0
        ):
            raise ArtifactError(f"Kemeny violated its Borda guarantee for {key}")
        result[key] = {
            **totals,
            "starts_requested": starts.pop(),
            "max_passes": pass_caps.pop(),
            "max_moves": max(int(row["max_moves"]) for row in rows),
            "converged_fraction": float(totals["converged_count"] / instances),
        }
    return result


def _metrics_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
) -> Mapping[str, Mapping[str, float]]:
    labels = {}
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {metric: 0.0 for metric in QUALITY_METRICS}
    )
    for record in records:
        current_labels = record.get("rule_labels")
        current_sums = record.get("metric_sums")
        if not isinstance(current_labels, Mapping) or not isinstance(current_sums, Mapping):
            raise ArtifactError("Prefix shard lacks rule labels or metric sums")
        for field_id, name in current_labels.items():
            field = str(field_id)
            label = str(name)
            if field in labels and labels[field] != label:
                raise ArtifactError(f"Prefix field {field} changed labels across shards")
            labels[field] = label
        for field_id, values in current_sums.items():
            if not isinstance(values, Mapping):
                raise ArtifactError(f"Malformed metric sums for {field_id}")
            for metric in QUALITY_METRICS:
                sums[str(field_id)][metric] += float(values[metric])
    if set(labels) != set(sums):
        raise ArtifactError("Prefix metric fields do not align with rule labels")
    return {
        labels[field_id]: {metric: float(value / sample_count) for metric, value in values.items()}
        for field_id, values in sums.items()
    }


def _q11_robustness(
    q11_reference: Mapping[str, Any],
    *,
    clean_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = q11_reference.get("robustness")
    if not isinstance(source, Mapping):
        raise ArtifactError("Perturbed q=11 NAIVE reference has no robustness summary")
    result = {}
    for rule in RULE_IDS:
        key = _rule_key(11, rule)
        row = source.get(rule)
        if not isinstance(row, Mapping) or not isinstance(row.get("absolute"), Mapping):
            raise ArtifactError(f"Malformed q=11 robustness reference for {rule}")
        absolute = {metric: float(row["absolute"][metric]) for metric in QUALITY_METRICS}
        for metric in QUALITY_METRICS:
            derived = abs(float(clean_metrics[key][metric]) - float(current_metrics[key][metric]))
            if not math.isclose(absolute[metric], derived, rel_tol=0.0, abs_tol=1e-12):
                raise ArtifactError(
                    f"q=11 robustness reference disagrees with its frozen metrics: {rule}/{metric}"
                )
        result[key] = {"absolute": absolute}
    return result


def _load_or_prepare_input_catalog(experiment: NoisePrefixExperiment) -> Mapping[str, Any]:
    """Load the immutable catalog, repairing only legacy missing catalogs."""

    if not experiment.runtime.input_catalog_path.is_file():
        prepare_input_catalog(experiment)
    return load_input_catalog(experiment)


def run_evaluation_task(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    import torch

    store = output_store(experiment)
    complete = completed_evaluation_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    # Older full-matrix scopes may have a successful sidecar readiness marker
    # from before the catalog was part of that gate. Recreate the catalog with
    # its normal lock and immutable-input validation before evaluating; this is
    # idempotent and leaves existing valid catalogs untouched.
    catalog = _load_or_prepare_input_catalog(experiment)
    task_row = catalog_task_input(experiment, task, catalog=catalog)
    layout = source_layout(task_row)
    records = completed_shards(
        experiment,
        task,
        source_layout=layout,
        store=store,
    )
    missing_layout = tuple(item for item in layout if item[0] not in records)
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NOISE prefix evaluation requires CUDA")

    fill = _fill(experiment, task)  # type: ignore[arg-type]
    publisher = PrefixShardPublisher(experiment, store, task_id=task.task_id)
    publication_futures: dict[int, Future[Mapping[str, Any]]] = {}
    timings = StageTimings()
    gpu_sampler = GpuUtilizationSampler(requested_device=str(target_device)).start()
    gpu_telemetry: Mapping[str, Any] | None = None
    loaded = None
    bundle = None
    shared_conditioned = None
    compute_complete = False
    try:
        if missing_layout:
            loaded = load_model(
                task.cell.reference_model,
                device=target_device,
                include_checkpoint=True,
            )
            bundle = load_split(
                task.cell.dataset,
                loaded,
                split=experiment.split,
                workers=experiment.base.runtime.dataloader_workers,
                shared_cache_root=experiment.runtime.shared_cache_root,
            )
            if task.condition.kind == "factory":
                with timings.measure("shared_condition_cache"):
                    shared_conditioned = materialize_shared_conditioned_raw_images(
                        bundle,
                        loaded,
                        task.condition,
                        device=target_device,
                        batch_size=experiment.runtime.inference_batch_size,
                        seed=experiment.base.runtime.seed,
                        shared_cache_root=experiment.runtime.shared_cache_root,
                    )
            quota = SpoolQuota(
                experiment.storage.spool_root,
                max_bytes=experiment.storage.spool_max_bytes,
                min_free_bytes=experiment.storage.spool_min_free_bytes,
            )
            prefetch_items = [
                PrefetchItem(
                    key=shard_index,
                    byte_count=input_byte_count(task_row, shard_index),
                    load=lambda directory, index=shard_index: load_prefix_shard(
                        experiment,
                        task_row,
                        shard_index=index,
                        work_directory=directory,
                    ),
                )
                for shard_index, _, _ in missing_layout
            ]
            with ByteBoundedPrefetcher(
                quota,
                prefetch_items,
                workers=experiment.runtime.cpu_workers,
                namespace=f"prefix-{task.digest[:16]}",
            ) as prefetcher:
                for shard_index, start, stop in missing_layout:
                    prefetched = prefetcher.get(shard_index)
                    timings.add("input_wait", prefetched.wait_seconds)
                    timings.add("input_materialize", prefetched.load_seconds)
                    shard = prefetched.value
                    fixed = shard.reference
                    indices = fixed["indices"].numpy().astype(np.int64, copy=False)
                    positions = [bundle.row_to_position[int(value)] for value in indices]
                    if shared_conditioned is None:
                        raw, observed_labels = bundle.rows(indices)
                    else:
                        raw = None
                        observed_labels = bundle.labels[positions]
                    if not torch.equal(
                        observed_labels.to(torch.int64), fixed["labels"].to(torch.int64)
                    ):
                        raise ArtifactError("Dataset labels changed after rank-ready publication")
                    with timings.measure("condition_materialize"):
                        if shared_conditioned is None:
                            conditioned = _conditioned_images(
                                experiment,  # type: ignore[arg-type]
                                task,  # type: ignore[arg-type]
                                shard_index=shard_index,
                                raw=raw,
                                labels=fixed["labels"],
                                indices=fixed["indices"],
                                targets=fixed["targets"],
                                model=loaded.model,
                                normalize=loaded.normalize,
                                device=target_device,
                            )
                        else:
                            conditioned = shared_conditioned[positions]
                    with timings.measure("gpu_aggregation"):
                        bank, aggregation_statistics = aggregate_prefix_bank(
                            experiment,
                            task,
                            ballots=shard.ballots,
                            simple_scores_by_q=shard.simple_scores_by_q,
                            indices=indices,
                            device=target_device,
                        )
                    labels = fixed["labels"].numpy().astype(np.int64, copy=False)
                    targets = fixed["targets"].numpy().astype(np.int64, copy=False)
                    predictions = fixed["predictions"].numpy().astype(np.int64, copy=False)
                    with timings.measure("gpu_rule_bank_evaluation"):
                        traces = evaluate_reference_model_bank(
                            loaded.model,
                            conditioned,
                            bank,
                            true_labels=labels,
                            target_labels=targets,
                            fill_reference=fill,
                            reference_model_id=task.cell.reference_model.model_id,
                            sample_ids=indices,
                            patch_size=experiment.patch_size,
                            k=experiment.k,
                            index_base=0,
                            batch_size=experiment.runtime.inference_batch_size,
                            device=str(target_device),
                            autocast=False,
                            normalize=loaded.normalize,
                            require_target_matches_clean=task.condition.kind == "clean",
                            clean_predictions=predictions,
                        )
                    tensors = {
                        "indices": fixed["indices"],
                        "labels": fixed["labels"],
                        "targets": fixed["targets"],
                        "unmasked_predictions": fixed["predictions"],
                    }
                    rule_labels = {}
                    metric_sums = {}
                    for rule_index, (rule_name, trace) in enumerate(traces.items()):
                        if not np.array_equal(trace.clean_predictions, predictions):
                            raise RuntimeError("Evaluator changed the registered predictions")
                        field_id = f"r{rule_index:03d}"
                        rule_labels[field_id] = rule_name
                        tensors[f"top_patch_indices__{field_id}"] = torch.from_numpy(
                            trace.selected_patch_indices
                        )
                        tensors[f"removed_predictions__{field_id}"] = torch.from_numpy(
                            trace.removed_predictions
                        )
                        tensors[f"retained_predictions__{field_id}"] = torch.from_numpy(
                            trace.retained_predictions
                        )
                        metric_sums[field_id] = _metric_sums(trace)
                    with timings.measure("output_enqueue"):
                        publication_futures[shard_index] = publisher.submit_shard(
                            root=task.artifact_root,
                            task_digest=task.digest,
                            schema_version=EVALUATION_SCHEMA_VERSION,
                            shard_index=shard_index,
                            start=start,
                            stop=stop,
                            tensors=tensors,
                            metadata={
                                "patch_size": str(experiment.patch_size),
                                "k": str(experiment.k),
                                "fill": "dataset_mean",
                                "precision": "fp32",
                                "rank_payload": "top_k_only",
                            },
                            record_fields={
                                "rule_labels": rule_labels,
                                "metric_sums": metric_sums,
                                "aggregation_statistics": aggregation_statistics,
                                "rank_ready_sidecars": list(shard.sidecars),
                            },
                        )
                    print(
                        "NOISE_PREFIX_EVALUATION "
                        f"shard={shard_index + 1}/{len(layout)} "
                        f"prefetch_wait_seconds={prefetched.wait_seconds:.3f} "
                        f"prefetch_load_seconds={prefetched.load_seconds:.3f} "
                        f"rules={len(traces)} task={task.task_id}",
                        flush=True,
                    )
                    del raw, conditioned, bank, traces, tensors, shard
                    prefetcher.release(shard_index)
                    publisher.check()
        compute_complete = True
    finally:
        if loaded is not None:
            torch.cuda.synchronize(target_device)
        gpu_telemetry = gpu_sampler.stop()
        loaded = None
        bundle = None
        shared_conditioned = None
        gc.collect()
        torch.cuda.empty_cache()
        emit_gpu_release_signal()
        if not compute_complete:
            publisher.shutdown()

    try:
        for shard_index, future in publication_futures.items():
            records[shard_index] = future.result()
        publisher.check()
    finally:
        publisher.shutdown()

    ordered_records = [records[index] for index, _, _ in layout]
    sample_count = sum(int(record["count"]) for record in ordered_records)
    if sample_count != int(task_row["q11_reference"]["sample_count"]):
        raise ArtifactError("Prefix sample count differs from the q=11 NAIVE reference")
    metrics = dict(_metrics_from_records(ordered_records, sample_count=sample_count))
    for rule in RULE_IDS:
        metrics[_rule_key(11, rule)] = {
            metric: float(task_row["q11_reference"]["metrics"][rule][metric])
            for metric in QUALITY_METRICS
        }
    expected_rules = {_rule_key(q, rule) for q in experiment.q_values for rule in RULE_IDS}
    if set(metrics) != expected_rules:
        raise ArtifactError("Final prefix metrics do not cover q=2..11 and all rules")

    robustness = None
    clean_source = None
    if task.condition.kind != "clean":
        clean_task = experiment.clean_task(task)
        clean = completed_evaluation_manifest(experiment, clean_task, store=store)
        if clean is None:
            raise FileNotFoundError(f"Clean prefix evaluation is incomplete: {clean_task.task_id}")
        clean_metrics = clean.get("metrics")
        if not isinstance(clean_metrics, Mapping) or set(clean_metrics) != set(metrics):
            raise ArtifactError("Clean and perturbed prefix rules are not aligned")
        robustness = {
            key: {
                "absolute": {
                    metric: abs(float(clean_metrics[key][metric]) - float(metrics[key][metric]))
                    for metric in QUALITY_METRICS
                }
            }
            for key in metrics
            if not key.startswith("q11__")
        }
        robustness.update(
            _q11_robustness(
                task_row["q11_reference"],
                clean_metrics=clean_metrics,
                current_metrics=metrics,
            )
        )
        clean_source = {"task_id": clean_task.task_id, "task_digest": clean_task.digest}

    value: Mapping[str, Any] = {
        "schema": "simple-noise-prefix-evaluation-v1",
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell.cell_id,
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "split": experiment.split,
        "condition": task.condition.condition_id,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "fill": "dataset_mean",
        "fill_artifact_id": fill.artifact_id,
        "precision": "fp32",
        "autocast": False,
        "target_policy": "full_reference_clean_fp32_prediction",
        "unmasked_prediction_source": "rank_ready_full_reference_fp32_logits_argmax",
        "fidelity_scope": "complete_test_set",
        "fidelity_direction": "descending",
        "fidelity_tie_break": "method_id_ascending",
        "fidelity": dict(task_row["fidelity"]),
        "ordered_methods": list(task_row["ordered_methods"]),
        "method_prefixes": dict(task_row["method_prefixes"]),
        "q_values": list(experiment.q_values),
        "computed_q_values": list(experiment.q_values[:-1]),
        "q11_policy": "exact_immutable_naive_p16_reference",
        "q11_reference": dict(task_row["q11_reference"]),
        "noise_selection_provenance": dict(catalog["noise_selections"][task.cell.cell_id]),
        "input_catalog_digest": catalog["catalog_digest"],
        "inference_batch_size": experiment.runtime.inference_batch_size,
        "inference_batch_semantics": "maximum_actual_model_forward_batch",
        "rank_payload": "top_k_patch_indices_only_no_full_consensus_rank",
        "sample_count": sample_count,
        "runtime_telemetry": {
            "stages": timings.summary(),
            "gpu": gpu_telemetry,
        },
        "aggregation_statistics": _summarize_kemeny(ordered_records),
        "metrics": metrics,
        "robustness": robustness,
        "clean_metric_source": clean_source,
        "shards": ordered_records,
    }
    publish_manifest(
        experiment,  # type: ignore[arg-type]
        store,
        root=task.artifact_root,
        task_id=task.task_id,
        manifest=value,
    )
    return value


__all__ = ["aggregate_prefix_bank", "run_evaluation_task"]
