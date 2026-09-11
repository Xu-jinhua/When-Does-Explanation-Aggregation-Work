"""Common full-reference mask-game evaluation for all assumption settings."""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import Future
from datetime import UTC, datetime
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase0.models import get_model_definition
from xai_ensemble.phase2.evaluator import FillReference, evaluate_reference_model_bank
from xai_ensemble.phase2.metrics import QUALITY_METRICS

from ..adversarial import load_adversarial_shard
from ..artifacts import ArtifactError, ArtifactStore, load_safetensors
from ..data import (
    apply_condition,
    load_model,
    load_raw_dataset_mean,
    load_split,
    materialize_shared_conditioned_raw_images,
)
from ..io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from ..phase2 import _metric_sums
from ..runtime import emit_gpu_release_signal
from ..spool import SpoolQuota
from ..telemetry import GpuUtilizationSampler, StageTimings
from .artifacts import (
    EVALUATION_SCHEMA_VERSION,
    AssumptionShardPublisher,
    completed_evaluation_manifest,
    completed_rank_manifest,
    existing_shard_records,
    output_store,
    publish_manifest,
    task_spool_path,
)
from .config import AssumptionExperiment, EvaluationTask, RankTask


def _rank_task(experiment: AssumptionExperiment, task: EvaluationTask) -> RankTask:
    return experiment.find_rank_task(task.rank_task_id)


def _layout(manifest: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
        for item in manifest["shards"]
    )


def _load_rank_shard(
    experiment: AssumptionExperiment,
    task: EvaluationTask,
    rank_manifest: Mapping[str, Any],
    shard_index: int,
    work_directory: Any | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    record = next(
        item for item in rank_manifest["shards"] if int(item["shard_index"]) == shard_index
    )
    payload = record["payload"]
    local = (
        task_spool_path(
            experiment,
            namespace="rank-cache",
            task_digest=task.digest,
            relative_path=str(payload["relative_path"]),
        )
        if work_directory is None
        else work_directory / str(payload["sha256"])
    )
    store = output_store(experiment)
    store.materialize(str(payload["relative_path"]), local, expected_sha256=str(payload["sha256"]))
    try:
        fields = load_safetensors(local)
        labels = record.get("rule_labels")
        if not isinstance(labels, Mapping):
            raise ArtifactError("Assumption rank shard has no rule labels")
        ranks = {
            str(name): fields[f"rank__{field_id}"].numpy().astype(np.int64, copy=False).copy()
            for field_id, name in labels.items()
        }
        fixed = {
            "indices": fields["indices"].clone(),
            "labels": fields["labels"].clone(),
            "targets": fields["targets"].clone(),
            "predictions": fields["unmasked_predictions"].clone(),
        }
    finally:
        local.unlink(missing_ok=True)
    return fixed, ranks


def _fill(experiment: AssumptionExperiment, task: EvaluationTask) -> FillReference:
    definition = get_model_definition(task.cell.reference_model.model_key)
    values = load_raw_dataset_mean(
        task.cell.reference_model, input_size=definition.input_size
    ).squeeze(0)
    mean_path = resolve_full_matrix_runtime_path(task.cell.reference_model.mean_path)
    manifest = mean_path / "manifest.json" if mean_path.is_dir() else mean_path
    identity = object_sha256(
        {
            "kind": "dataset_mean",
            "path": str(task.cell.reference_model.mean_path),
            "digest": file_sha256(manifest),
        }
    )
    return FillReference(values=values, source_split="train", artifact_id=identity)


def _conditioned_images(
    experiment: AssumptionExperiment,
    task: EvaluationTask,
    *,
    shard_index: int,
    raw: Any,
    labels: Any,
    indices: Any,
    targets: Any,
    model: Any,
    normalize: Any,
    device: Any,
) -> Any:
    import torch

    condition = task.condition
    if condition.kind == "clean":
        return raw.to(dtype=torch.float32)
    if condition.kind == "adversarial":
        attack = experiment.base.adversarial_task_for(
            dataset_id=task.cell.dataset.dataset_id,
            model_id=task.cell.reference_model.model_id,
            split=experiment.split,
            condition_id=condition.condition_id,
        )
        fields = load_adversarial_shard(
            experiment.base,
            attack,
            shard_index,
            expected_indices=indices,
            expected_labels=labels,
            clean_images=raw,
            expected_targets=targets,
            store=ArtifactStore(experiment.base),
        )
        return fields["adversarial_images"]
    chunks = []
    source_batch = max(1, experiment.runtime.inference_batch_size // 2)
    for start in range(0, int(raw.shape[0]), source_batch):
        stop = min(int(raw.shape[0]), start + source_batch)
        batch = raw[start:stop].to(device, dtype=torch.float32, non_blocking=True)
        chunks.append(
            apply_condition(
                condition,
                batch,
                labels=labels[start:stop].to(device),
                indices=indices[start:stop].to(device),
                model=model,
                normalize=normalize,
                seed=experiment.base.runtime.seed,
            )
            .detach()
            .cpu()
        )
    return torch.cat(chunks)


def _matching_clean_task(
    experiment: AssumptionExperiment,
    task: EvaluationTask,
) -> EvaluationTask:
    matches = [
        candidate
        for candidate in experiment.evaluation_tasks()
        if candidate.cell.cell_id == task.cell.cell_id
        and candidate.setting == task.setting
        and candidate.source_id == task.source_id
        and candidate.distance_model == task.distance_model
        and candidate.condition.kind == "clean"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one clean evaluation dependency; found {len(matches)}")
    return matches[0]


def run_evaluation_task(
    experiment: AssumptionExperiment,
    task: EvaluationTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    import torch

    store = output_store(experiment)
    complete = completed_evaluation_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    rank_task = _rank_task(experiment, task)
    rank_manifest = completed_rank_manifest(experiment, rank_task, store=store)
    if rank_manifest is None:
        raise FileNotFoundError(f"Rank artifact is incomplete: {rank_task.task_id}")
    layout = _layout(rank_manifest)
    records = existing_shard_records(
        store,
        task.artifact_root,
        task_digest=task.digest,
        schema_version=EVALUATION_SCHEMA_VERSION,
        source_layout=layout,
    )
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Assumption evaluation requires CUDA")
    loaded = load_model(task.cell.reference_model, device=target_device, include_checkpoint=True)
    bundle = load_split(
        task.cell.dataset,
        loaded,
        split=experiment.split,
        workers=experiment.base.runtime.dataloader_workers,
        shared_cache_root=experiment.storage.spool_root / "shared-cache",
    )
    fill = _fill(experiment, task)
    missing_layout = tuple(item for item in layout if item[0] not in records)
    publication_futures: dict[int, Future[Mapping[str, Any]]] = {}
    publisher = AssumptionShardPublisher(experiment, store, task_id=task.task_id)
    quota = SpoolQuota(
        experiment.storage.spool_root,
        max_bytes=experiment.storage.spool_max_bytes,
        min_free_bytes=experiment.storage.spool_min_free_bytes,
    )
    prefetch_items = []
    for shard_index, _, _ in missing_layout:
        record = next(
            item for item in rank_manifest["shards"] if int(item["shard_index"]) == shard_index
        )
        payload = record["payload"]
        prefetch_items.append(
            PrefetchItem(
                key=shard_index,
                byte_count=max(1, int(payload["size_bytes"])),
                load=lambda directory, index=shard_index: _load_rank_shard(
                    experiment,
                    task,
                    rank_manifest,
                    index,
                    work_directory=directory,
                ),
            )
        )
    compute_complete = False
    timings = StageTimings()
    gpu_sampler = GpuUtilizationSampler(requested_device=str(target_device)).start()
    gpu_telemetry: Mapping[str, Any] | None = None
    shared_conditioned = None
    try:
        if task.condition.kind == "factory":
            with timings.measure("shared_condition_cache"):
                shared_conditioned = materialize_shared_conditioned_raw_images(
                    bundle,
                    loaded,
                    task.condition,
                    device=target_device,
                    batch_size=experiment.runtime.inference_batch_size,
                    seed=experiment.base.runtime.seed,
                    shared_cache_root=experiment.storage.spool_root / "shared-cache",
                )
        with ByteBoundedPrefetcher(
            quota,
            prefetch_items,
            workers=experiment.runtime.cpu_workers,
            namespace=f"evaluation-{task.digest[:16]}",
        ) as prefetcher:
            for shard_index, start, stop in missing_layout:
                prefetched = prefetcher.get(shard_index)
                timings.add("input_wait", prefetched.wait_seconds)
                timings.add("input_materialize", prefetched.load_seconds)
                fixed, ranks = prefetched.value
                positions = [
                    bundle.row_to_position[int(value)] for value in fixed["indices"].tolist()
                ]
                if shared_conditioned is None:
                    raw, observed_labels = bundle.rows(fixed["indices"].numpy())
                else:
                    raw = None
                    observed_labels = bundle.labels[positions]
                if not torch.equal(
                    observed_labels.to(torch.int64), fixed["labels"].to(torch.int64)
                ):
                    raise ArtifactError("Dataset labels changed after rank publication")
                with timings.measure("condition_materialize"):
                    if shared_conditioned is None:
                        conditioned = _conditioned_images(
                            experiment,
                            task,
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
                labels = fixed["labels"].numpy().astype(np.int64, copy=False)
                targets = fixed["targets"].numpy().astype(np.int64, copy=False)
                predictions = fixed["predictions"].numpy().astype(np.int64, copy=False)
                tensors = {
                    "indices": fixed["indices"],
                    "labels": fixed["labels"],
                    "targets": fixed["targets"],
                    "unmasked_predictions": fixed["predictions"],
                }
                with timings.measure("gpu_rule_bank_evaluation"):
                    traces = evaluate_reference_model_bank(
                        loaded.model,
                        conditioned,
                        ranks,
                        true_labels=labels,
                        target_labels=targets,
                        fill_reference=fill,
                        reference_model_id=task.cell.reference_model.model_id,
                        sample_ids=fixed["indices"].numpy(),
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
                rule_labels = {}
                metric_sums = {}
                for rule_index, (rule_name, trace) in enumerate(traces.items()):
                    if not np.array_equal(trace.clean_predictions, predictions):
                        raise RuntimeError("Evaluator changed the registered reference predictions")
                    field_id = f"r{rule_index:03d}"
                    rule_labels[field_id] = rule_name
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
                            "patch_size": "16",
                            "k": "20",
                            "fill": "dataset_mean",
                            "precision": "fp32",
                        },
                        record_fields={
                            "rule_labels": rule_labels,
                            "metric_sums": metric_sums,
                        },
                    )
                print(
                    "ASSUMPTIONS_EVALUATION "
                    f"shard={shard_index + 1}/{len(layout)} "
                    f"prefetch_wait_seconds={prefetched.wait_seconds:.3f} "
                    f"prefetch_load_seconds={prefetched.load_seconds:.3f} "
                    f"rules={len(traces)} task={task.task_id}",
                    flush=True,
                )
                del fixed, ranks, raw, conditioned, tensors, traces
                prefetcher.release(shard_index)
                publisher.check()
        compute_complete = True
    finally:
        gpu_telemetry = gpu_sampler.stop()
        utilization = gpu_telemetry.get("utilization_percent")
        print(
            "ASSUMPTIONS_GPU_TELEMETRY "
            f"kind=evaluation samples={gpu_telemetry['sample_count']} "
            f"mean_utilization={None if utilization is None else utilization['mean']} "
            f"task={task.task_id}",
            flush=True,
        )
        loaded = None
        bundle = None
        shared_conditioned = None
        torch.cuda.synchronize(target_device)
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

    ordered = [records[index] for index, _, _ in layout]
    sample_count = sum(int(record["count"]) for record in ordered)
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {metric: 0.0 for metric in QUALITY_METRICS}
    )
    labels_by_field = {}
    for record in ordered:
        labels_by_field.update(
            {str(key): str(value) for key, value in record["rule_labels"].items()}
        )
        for field_id, values in record["metric_sums"].items():
            for metric in QUALITY_METRICS:
                sums[str(field_id)][metric] += float(values[metric])
    metrics = {
        labels_by_field[field_id]: {
            metric: value / sample_count for metric, value in values.items()
        }
        for field_id, values in sums.items()
    }
    robustness = None
    clean_source = None
    if task.condition.kind != "clean":
        clean_task = _matching_clean_task(experiment, task)
        clean = completed_evaluation_manifest(experiment, clean_task, store=store)
        if clean is None:
            raise FileNotFoundError(f"Clean evaluation is incomplete: {clean_task.task_id}")
        clean_metrics = clean.get("metrics")
        if not isinstance(clean_metrics, Mapping) or set(clean_metrics) != set(metrics):
            raise ArtifactError("Clean and perturbed assumption rules are not aligned")
        robustness = {
            rule: {
                "absolute": {
                    metric: abs(float(clean_metrics[rule][metric]) - float(current[metric]))
                    for metric in QUALITY_METRICS
                }
            }
            for rule, current in metrics.items()
        }
        clean_source = {"task_id": clean["task_id"], "task_digest": clean["task_digest"]}
    value: Mapping[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "rank_task_id": rank_task.task_id,
        "rank_task_digest": rank_task.digest,
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
        "k": experiment.k,
        "fill": "dataset_mean",
        "fill_artifact_id": fill.artifact_id,
        "target_policy": "full_reference_clean_fp32_prediction",
        "unmasked_prediction_source": "rank_source_reference_fp32_logits_argmax",
        "inference_batch_size": experiment.runtime.inference_batch_size,
        "inference_batch_semantics": "maximum_actual_model_forward_batch",
        "sample_count": sample_count,
        "runtime_telemetry": {
            "stages": timings.summary(),
            "gpu": gpu_telemetry,
        },
        "metrics": metrics,
        "robustness": robustness,
        "clean_metric_source": clean_source,
        "shards": ordered,
    }
    publish_manifest(
        experiment, store, root=task.artifact_root, task_id=task.task_id, manifest=value
    )
    return value


__all__ = ["run_evaluation_task"]
