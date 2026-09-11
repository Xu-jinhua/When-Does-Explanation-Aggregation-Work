"""Phase-2 evaluation for frozen noise-consistent random subsets."""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from datetime import UTC, datetime
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.phase2.evaluator import evaluate_reference_model_bank
from xai_ensemble.phase2.metrics import QUALITY_METRICS
from xai_ensemble.phase2.torch_aggregation import aggregate_rankings_torch

from ..artifacts import ArtifactError
from ..assumptions.evaluator import _conditioned_images, _fill
from ..data import load_model, load_split, materialize_shared_conditioned_raw_images
from ..io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from ..noise_prefix.artifacts import completed_evaluation_manifest as completed_prefix_manifest
from ..noise_prefix.inputs import (
    catalog_task_input,
    input_byte_count,
    load_input_catalog,
    load_prefix_shard,
    source_layout,
)
from ..phase2 import _metric_sums
from ..runtime import emit_gpu_release_signal
from ..spool import SpoolQuota
from ..telemetry import GpuUtilizationSampler, StageTimings
from .artifacts import (
    NoiseSubsetShardPublisher,
    completed_evaluation_manifest,
    completed_evaluation_shards,
    completed_selection_manifest,
    output_store,
    publish_manifest,
)
from .config import (
    ANCHORED_CONTROL_MODE,
    GEOMETRY_RULES,
    NoiseSubsetEvaluationTask,
    NoiseSubsetExperiment,
)


def _candidate_rule_key(selection_position: int, rule: str) -> str:
    return f"random_{selection_position:02d}__{rule.lower()}"


def _anchored_candidate_rule_key(candidate_position: int, rule: str) -> str:
    return f"candidate_{candidate_position:03d}__{rule.lower()}"


def _candidate_seeds(
    task: NoiseSubsetEvaluationTask,
    *,
    candidate_digest: str,
    indices: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        stable_seed(
            "simple-noise-random-subset-evaluation-v1",
            task.digest,
            candidate_digest,
            int(row_index),
        )
        for row_index in indices
    )


def aggregate_subset_bank(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
    *,
    ballots: np.ndarray,
    method_patch_scores: np.ndarray,
    ordered_methods: Sequence[str],
    selected_candidates: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    device: Any,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Mapping[str, int | float]]]:
    if ballots.ndim != 3 or method_patch_scores.shape != ballots.shape:
        raise ValueError("ballots and method patch scores must share [N,M,P]")
    method_positions = {str(method): position for position, method in enumerate(ordered_methods)}
    if len(method_positions) != ballots.shape[1]:
        raise ValueError("ordered method roster differs from the rank bank")
    bank = {}
    statistics = {}
    for candidate in selected_candidates:
        selection_position = int(candidate["selection_position"])
        methods = tuple(str(value) for value in candidate["methods"])
        positions = tuple(int(value) for value in candidate["positions"])
        expected_positions = tuple(method_positions[method] for method in methods)
        if positions != expected_positions or len(positions) != int(candidate["q"]):
            raise ArtifactError("frozen random subset method positions are contradictory")
        result = aggregate_rankings_torch(
            ballots[:, positions, :],
            method_patch_scores[:, positions, :].mean(axis=1, dtype=np.float32),
            requested=experiment.rules,
            rrf_c=experiment.base.phase2.rrf_c,
            kemeny_starts=experiment.base.phase2.kemeny_starts,
            kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
            seeds=_candidate_seeds(
                task,
                candidate_digest=str(candidate["candidate_digest"]),
                indices=indices.tolist(),
            ),
            device=device,
            workspace_bytes=experiment.runtime.aggregation_workspace_bytes,
        )
        for rule, ranks in result.items():
            bank[_candidate_rule_key(selection_position, rule)] = ranks
        for rule, row in result.statistics.items():
            statistics[_candidate_rule_key(selection_position, rule)] = dict(row)
    expected = {
        _candidate_rule_key(int(candidate["selection_position"]), rule)
        for candidate in selected_candidates
        for rule in experiment.rules
    }
    if set(bank) != expected:
        raise RuntimeError(
            f"random-subset aggregation bank is incomplete: {sorted(expected - set(bank))}"
        )
    return bank, statistics


def aggregate_anchored_candidate_bank(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
    *,
    ballots: np.ndarray,
    method_patch_scores: np.ndarray,
    ordered_methods: Sequence[str],
    geometry_selection: Mapping[str, Any],
    indices: np.ndarray,
    device: Any,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Mapping[str, int | float]]]:
    if ballots.ndim != 3 or method_patch_scores.shape != ballots.shape:
        raise ValueError("ballots and method patch scores must share [N,M,P]")
    candidates = geometry_selection.get("candidates")
    selected = geometry_selection.get("selected")
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not isinstance(selected, Sequence)
        or isinstance(selected, (str, bytes))
    ):
        raise ArtifactError("anchored random-order candidate bank is malformed")
    method_positions = {str(method): position for position, method in enumerate(ordered_methods)}
    selected_digests = {str(row["candidate_digest"]) for row in selected}
    center_rule = GEOMETRY_RULES[task.geometry]
    canonical_rules = {rule.lower(): rule for rule in experiment.rules}
    bank = {}
    statistics = {}
    expected = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ArtifactError("anchored random-order candidate is malformed")
        candidate_position = int(candidate["candidate_position"])
        methods = tuple(str(value) for value in candidate["methods"])
        positions = tuple(int(value) for value in candidate["positions"])
        expected_positions = tuple(method_positions[method] for method in methods)
        if positions != expected_positions or len(positions) != int(candidate["q"]):
            raise ArtifactError("anchored candidate method positions are contradictory")
        requested = (
            experiment.rules
            if str(candidate["candidate_digest"]) in selected_digests
            else (canonical_rules[center_rule],)
        )
        result = aggregate_rankings_torch(
            ballots[:, positions, :],
            method_patch_scores[:, positions, :].mean(axis=1, dtype=np.float32),
            requested=requested,
            rrf_c=experiment.base.phase2.rrf_c,
            kemeny_starts=experiment.base.phase2.kemeny_starts,
            kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
            seeds=_candidate_seeds(
                task,
                candidate_digest=str(candidate["candidate_digest"]),
                indices=indices.tolist(),
            ),
            device=device,
            workspace_bytes=experiment.runtime.aggregation_workspace_bytes,
        )
        for rule, ranks in result.items():
            key = _anchored_candidate_rule_key(candidate_position, rule)
            bank[key] = ranks
            expected.add(key)
        for rule, row in result.statistics.items():
            statistics[_anchored_candidate_rule_key(candidate_position, rule)] = dict(row)
    if set(bank) != expected:
        raise RuntimeError("anchored random-order aggregation bank is incomplete")
    return bank, statistics


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
        rule_labels = record.get("rule_labels")
        metric_sums = record.get("metric_sums")
        if not isinstance(rule_labels, Mapping) or not isinstance(metric_sums, Mapping):
            raise ArtifactError("random-subset shard lacks rule labels or metric sums")
        for field_id, label in rule_labels.items():
            field = str(field_id)
            if field in labels and labels[field] != str(label):
                raise ArtifactError("random-subset rule labels changed across shards")
            labels[field] = str(label)
        for field_id, values in metric_sums.items():
            if not isinstance(values, Mapping):
                raise ArtifactError("random-subset metric sums are malformed")
            for metric in QUALITY_METRICS:
                sums[str(field_id)][metric] += float(values[metric])
    if set(labels) != set(sums):
        raise ArtifactError("random-subset metric labels do not align with sums")
    return {
        labels[field_id]: {metric: float(value / sample_count) for metric, value in values.items()}
        for field_id, values in sorted(sums.items())
    }


def _selection_input(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    selection_task = experiment.find_selection_task(task.selection_task_id)
    manifest = completed_selection_manifest(experiment, selection_task)
    if manifest is None:
        raise FileNotFoundError(f"random NOISE selection is incomplete: {selection_task.task_id}")
    expected = {
        "study_digest": experiment.digest,
        "task_digest": selection_task.digest,
        "cell": task.cell.cell_id,
        "fit_policy": experiment.fit_policy,
        "candidate_policy": experiment.candidate_policy,
        "selection_policy": experiment.selection_policy,
    }
    if experiment.control_mode == ANCHORED_CONTROL_MODE:
        expected.update(
            {
                "schema": "simple-noise-random-order-anchored-selection-v1",
                "schema_version": experiment.artifact_schema_version,
                "control_mode": experiment.control_mode,
                "q_values": list(experiment.q_values),
                "random_order_count": experiment.random_order_count,
            }
        )
    contradictions = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if contradictions:
        raise ArtifactError(f"random NOISE selection identity changed: {contradictions}")
    geometries = manifest.get("geometries")
    if not isinstance(geometries, Mapping) or task.geometry not in geometries:
        raise ArtifactError("random NOISE selection lacks the requested geometry")
    geometry = geometries[task.geometry]
    if not isinstance(geometry, Mapping):
        raise ArtifactError("random NOISE geometry selection is malformed")
    selected = geometry.get("selected")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise ArtifactError("random NOISE selected bank is malformed")
    rows = tuple(row for row in selected if isinstance(row, Mapping))
    expected_count = (
        experiment.random_order_count
        if experiment.control_mode == ANCHORED_CONTROL_MODE
        else experiment.minimum_accepted_candidates
    )
    if len(rows) != len(selected) or len(rows) < expected_count:
        raise ArtifactError("random NOISE selected bank is incomplete")
    if [int(row["selection_position"]) for row in rows] != list(range(len(rows))):
        raise ArtifactError("random NOISE selected positions are not contiguous")
    if experiment.control_mode == ANCHORED_CONTROL_MODE:
        candidates = geometry.get("candidates")
        orders = geometry.get("orders")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not isinstance(orders, Sequence)
            or isinstance(orders, (str, bytes))
            or len(orders) != experiment.random_order_count
        ):
            raise ArtifactError("anchored random-order candidate coverage is incomplete")
        candidate_rows = tuple(row for row in candidates if isinstance(row, Mapping))
        if len(candidate_rows) != len(candidates) or [
            int(row["candidate_position"]) for row in candidate_rows
        ] != list(range(len(candidate_rows))):
            raise ArtifactError("anchored candidate positions are not contiguous")
        return manifest, candidate_rows
    return manifest, rows


def _reference_metrics(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
    *,
    q: int,
) -> tuple[Mapping[str, Mapping[str, float]], Mapping[str, Any]]:
    prefix_task = experiment.prefix_task(task.cell, task.condition.condition_id)
    manifest = completed_prefix_manifest(experiment.prefix, prefix_task)
    if manifest is None:
        raise FileNotFoundError(f"prefix reference is incomplete: {prefix_task.task_id}")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ArtifactError("prefix reference has no metrics")
    result = {}
    for rule in experiment.rules:
        key = f"q{q:02d}__{rule.lower()}"
        row = metrics.get(key)
        if not isinstance(row, Mapping) or set(row) != set(QUALITY_METRICS):
            raise ArtifactError(f"prefix reference metric coverage is incomplete: {key}")
        result[rule.lower()] = {metric: float(row[metric]) for metric in QUALITY_METRICS}
    return result, {
        "task_id": prefix_task.task_id,
        "task_digest": prefix_task.digest,
        "manifest_content_digest": object_sha256(manifest),
    }


def _reference_metrics_by_q(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
) -> tuple[Mapping[str, Mapping[str, Mapping[str, float]]], Mapping[str, Any]]:
    by_q = {}
    source = None
    for q in experiment.q_values:
        metrics, current_source = _reference_metrics(experiment, task, q=q)
        if source is not None and current_source != source:
            raise ArtifactError("prefix reference source changed across q values")
        source = current_source
        by_q[str(q)] = metrics
    assert source is not None
    return by_q, source


def run_evaluation_task(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    import torch

    store = output_store(experiment)
    complete = completed_evaluation_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    selection, selected_candidates = _selection_input(experiment, task)
    geometry = selection["geometries"][task.geometry]
    q = None if experiment.control_mode == ANCHORED_CONTROL_MODE else int(geometry["q"])
    ordered_methods = tuple(str(value) for value in selection["ordered_methods"])
    catalog = load_input_catalog(experiment.prefix)
    prefix_task = experiment.prefix_task(task.cell, task.condition.condition_id)
    task_row = catalog_task_input(experiment.prefix, prefix_task, catalog=catalog)
    if task_row.get("ordered_methods") != list(ordered_methods):
        raise ArtifactError("random-subset evaluation method order changed")
    layout = source_layout(task_row)
    records = completed_evaluation_shards(
        experiment,
        task,
        source_layout=layout,
        store=store,
    )
    missing_layout = tuple(item for item in layout if item[0] not in records)
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("random NOISE subset evaluation requires CUDA")

    publisher = NoiseSubsetShardPublisher(experiment, store, task_id=task.task_id)  # type: ignore[arg-type]
    publication_futures: dict[int, Future[Mapping[str, Any]]] = {}
    timings = StageTimings()
    sampler = GpuUtilizationSampler(requested_device=str(target_device)).start()
    telemetry: Mapping[str, Any] | None = None
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
            fill = _fill(experiment, task)  # type: ignore[arg-type]
            quota = SpoolQuota(
                experiment.storage.spool_root,
                max_bytes=experiment.storage.spool_max_bytes,
                min_free_bytes=experiment.storage.spool_min_free_bytes,
            )
            items = [
                PrefetchItem(
                    key=shard_index,
                    byte_count=input_byte_count(task_row, shard_index),
                    load=lambda directory, index=shard_index: load_prefix_shard(
                        experiment.prefix,
                        task_row,
                        shard_index=index,
                        work_directory=directory,
                    ),
                )
                for shard_index, _, _ in missing_layout
            ]
            with ByteBoundedPrefetcher(
                quota,
                items,
                workers=experiment.runtime.cpu_workers,
                namespace=f"noise-random-evaluate-{task.digest[:16]}",
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
                        raise ArtifactError("dataset labels changed after rank-ready publication")
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
                        if experiment.control_mode == ANCHORED_CONTROL_MODE:
                            bank, aggregation_statistics = aggregate_anchored_candidate_bank(
                                experiment,
                                task,
                                ballots=shard.ballots,
                                method_patch_scores=shard.method_patch_scores,
                                ordered_methods=ordered_methods,
                                geometry_selection=geometry,
                                indices=indices,
                                device=target_device,
                            )
                        else:
                            bank, aggregation_statistics = aggregate_subset_bank(
                                experiment,
                                task,
                                ballots=shard.ballots,
                                method_patch_scores=shard.method_patch_scores,
                                ordered_methods=ordered_methods,
                                selected_candidates=selected_candidates,
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
                            raise RuntimeError("evaluator changed registered predictions")
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
                    publication_futures[shard_index] = publisher.submit_shard(
                        root=task.artifact_root,
                        task_digest=task.digest,
                        schema_version=experiment.artifact_schema_version,
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
                            "geometry": task.geometry,
                        },
                        record_fields={
                            "rule_labels": rule_labels,
                            "metric_sums": metric_sums,
                            "aggregation_statistics": aggregation_statistics,
                            "rank_ready_sidecars": list(shard.sidecars),
                        },
                    )
                    print(
                        "NOISE_RANDOM_EVALUATION "
                        f"geometry={task.geometry} shard={shard_index + 1}/{len(layout)} "
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
        telemetry = sampler.stop()
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
        raise ArtifactError("random-subset sample count differs from the q=11 reference")
    metrics = _metrics_from_records(ordered_records, sample_count=sample_count)
    if experiment.control_mode == ANCHORED_CONTROL_MODE:
        selected_digests = {
            str(candidate["candidate_digest"]) for candidate in geometry["selected"]
        }
        center_rule = GEOMETRY_RULES[task.geometry]
        expected_rules = {
            _anchored_candidate_rule_key(
                int(candidate["candidate_position"]),
                rule,
            )
            for candidate in selected_candidates
            for rule in (
                experiment.rules
                if str(candidate["candidate_digest"]) in selected_digests
                else (center_rule,)
            )
        }
    else:
        expected_rules = {
            _candidate_rule_key(int(candidate["selection_position"]), rule)
            for candidate in selected_candidates
            for rule in experiment.rules
        }
    if set(metrics) != expected_rules:
        raise ArtifactError("random-subset final metric coverage is incomplete")
    if experiment.control_mode == ANCHORED_CONTROL_MODE:
        reference_metrics_by_q, reference_source = _reference_metrics_by_q(experiment, task)
        reference_metrics = None
    else:
        assert q is not None
        reference_metrics, reference_source = _reference_metrics(
            experiment,
            task,
            q=q,
        )
        reference_metrics_by_q = None
    value: Mapping[str, Any] = {
        "schema": (
            "simple-noise-random-order-anchored-evaluation-v1"
            if experiment.control_mode == ANCHORED_CONTROL_MODE
            else "simple-noise-random-subset-evaluation-v1"
        ),
        "schema_version": experiment.artifact_schema_version,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "selection_task_id": task.selection_task_id,
        "selection_task_digest": experiment.find_selection_task(task.selection_task_id).digest,
        "selection_manifest_content_digest": object_sha256(selection),
        "cell": task.cell.cell_id,
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "split": experiment.split,
        "condition": task.condition.condition_id,
        "geometry": task.geometry,
        "center_rule": GEOMETRY_RULES[task.geometry],
        **(
            {"q_values": list(experiment.q_values)}
            if experiment.control_mode == ANCHORED_CONTROL_MODE
            else {"q": q}
        ),
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "fill": "dataset_mean",
        "precision": "fp32",
        "target_policy": "full_reference_clean_fp32_prediction",
        "input_catalog_digest": catalog["catalog_digest"],
        "ordered_methods": list(ordered_methods),
        "selected_candidates": list(geometry["selected"]),
        **(
            {
                "candidate_bank": list(selected_candidates),
                "random_orders": list(geometry["orders"]),
                "candidate_evaluation_policy": geometry["candidate_evaluation_policy"],
            }
            if experiment.control_mode == ANCHORED_CONTROL_MODE
            else {}
        ),
        "rules": list(experiment.rules),
        "sample_count": sample_count,
        "metrics": metrics,
        **(
            {"reference_prefix_metrics_by_q": reference_metrics_by_q}
            if experiment.control_mode == ANCHORED_CONTROL_MODE
            else {"reference_prefix_metrics": reference_metrics}
        ),
        "reference_prefix_source": reference_source,
        "runtime_telemetry": {"stages": timings.summary(), "gpu": telemetry},
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


__all__ = ["aggregate_subset_bank", "run_evaluation_task"]
