"""Mask-game evaluation over construction-independent rank sources."""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase0.models import get_model_definition
from xai_ensemble.phase2.evaluator import (
    ClassMeanFillReference,
    FillReference,
    evaluate_reference_model_bank,
)
from xai_ensemble.phase2.metrics import QUALITY_METRICS

from ..adversarial import load_adversarial_shard
from ..artifacts import ArtifactError, ArtifactStore
from ..data import (
    apply_condition,
    load_model,
    load_raw_class_means,
    load_raw_dataset_mean,
    load_split,
)
from ..phase2 import _metric_sums
from .artifacts import (
    EVALUATION_SCHEMA_VERSION,
    completed_evaluation_manifest,
    existing_shard_records,
    output_store,
    publish_manifest,
    publish_shard,
)
from .config import AblationExperiment, EvaluationSpec
from .rank_source import load_rank_manifest, load_rank_shard, source_layout


def _fill_reference(
    experiment: AblationExperiment,
    task: EvaluationSpec,
) -> FillReference | ClassMeanFillReference:
    definition = get_model_definition(experiment.model.model_key)
    mean_path = resolve_full_matrix_runtime_path(experiment.model.mean_path)
    mean_manifest = mean_path / "manifest.json" if mean_path.is_dir() else mean_path
    identity = object_sha256(
        {
            "path": str(experiment.model.mean_path),
            "digest": file_sha256(mean_manifest),
            "kind": task.fill,
            "class_policy": (
                "fixed_clean_explanation_target" if task.fill == "class_mean" else None
            ),
        }
    )
    if task.fill == "dataset_mean":
        values = load_raw_dataset_mean(experiment.model, input_size=definition.input_size).squeeze(
            0
        )
        return FillReference(values=values, source_split="train", artifact_id=identity)
    means, counts = load_raw_class_means(experiment.model, input_size=definition.input_size)
    return ClassMeanFillReference(
        values=means,
        class_counts=counts,
        source_split="train",
        artifact_id=identity,
    )


def _condition_experiment(experiment: AblationExperiment, task: EvaluationSpec) -> Any:
    return (
        experiment.base if task.rank_source.store == "base" else experiment.generation_experiment()
    )


def _conditioned_images(
    experiment: AblationExperiment,
    task: EvaluationSpec,
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

    condition_experiment = _condition_experiment(experiment, task)
    condition = task.condition
    if condition.kind == "clean":
        return raw.to(dtype=torch.float32)
    if condition.kind == "adversarial":
        attack_task = condition_experiment.adversarial_task_for(
            dataset_id=experiment.dataset_id,
            model_id=experiment.model_id,
            split=experiment.split,
            condition_id=condition.condition_id,
        )
        fields = load_adversarial_shard(
            condition_experiment,
            attack_task,
            shard_index,
            expected_indices=indices,
            expected_labels=labels,
            clean_images=raw,
            expected_targets=targets,
            store=ArtifactStore(condition_experiment),
        )
        return fields["adversarial_images"]
    chunks = []
    source_batch = max(1, experiment.runtime.inference_batch_size // 2)
    for start in range(0, int(raw.shape[0]), source_batch):
        stop = min(int(raw.shape[0]), start + source_batch)
        batch = raw[start:stop].to(device, dtype=torch.float32, non_blocking=True)
        conditioned = apply_condition(
            condition,
            batch,
            labels=labels[start:stop].to(device),
            indices=indices[start:stop].to(device),
            model=model,
            normalize=normalize,
            seed=condition_experiment.runtime.seed,
        )
        chunks.append(conditioned.detach().cpu())
    return torch.cat(chunks)


def _matching_clean_evaluation(
    experiment: AblationExperiment,
    task: EvaluationSpec,
) -> EvaluationSpec | None:
    matches = [
        candidate
        for candidate in experiment.evaluation_tasks()
        if candidate.table_id == task.table_id
        and candidate.parameter_value == task.parameter_value
        and candidate.condition.kind == "clean"
        and candidate.patch_size == task.patch_size
        and candidate.k == task.k
        and candidate.fill == task.fill
    ]
    if len(matches) > 1:
        raise RuntimeError("Multiple clean ablation evaluations match one task")
    return matches[0] if matches else None


def clean_metrics_manifest(
    experiment: AblationExperiment,
    task: EvaluationSpec,
) -> Mapping[str, Any]:
    clean_task = _matching_clean_evaluation(experiment, task)
    if clean_task is not None:
        manifest = completed_evaluation_manifest(experiment, clean_task)
        if manifest is None:
            raise FileNotFoundError(
                f"Perturbed evaluation requires clean task {clean_task.task_id}"
            )
        return manifest
    if task.k != 20 or task.fill != "dataset_mean":
        raise RuntimeError("No clean metric source exists for this evaluation setting")
    clean = next(
        condition for condition in experiment.base_conditions() if condition.kind == "clean"
    )
    manifest = load_rank_manifest(experiment, experiment.rank_source(clean))
    if not isinstance(manifest.get("metrics"), Mapping):
        raise ArtifactError("The immutable clean Phase 2 source has no metrics")
    return manifest


def run_evaluation_task(
    experiment: AblationExperiment,
    task: EvaluationSpec,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    """Evaluate one rank source for one scientific p/k/fill setting."""

    import torch

    store = output_store(experiment)
    complete = completed_evaluation_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    rank_manifest = load_rank_manifest(experiment, task.rank_source)
    layout = source_layout(rank_manifest)
    records = existing_shard_records(
        store,
        task.artifact_root,
        task_digest=task.digest,
        schema_version=EVALUATION_SCHEMA_VERSION,
        source_layout=layout,
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Ablation evaluation requires CUDA")
    loaded = load_model(experiment.model, device=target_device, include_checkpoint=True)
    bundle = load_split(
        experiment.dataset,
        loaded,
        split=experiment.split,
        workers=experiment.base.runtime.dataloader_workers,
        shared_cache_root=experiment.storage.spool_root / "shared-cache",
    )
    fill = _fill_reference(experiment, task)

    for shard_index, start, stop in layout:
        if shard_index in records:
            continue
        source = load_rank_shard(experiment, task.rank_source, rank_manifest, shard_index)
        raw, observed_labels = bundle.rows(source.indices.numpy().astype(np.int64, copy=False))
        if not torch.equal(observed_labels.to(torch.int64), source.labels):
            raise ArtifactError("Dataset labels changed after rank publication")
        conditioned = _conditioned_images(
            experiment,
            task,
            shard_index=shard_index,
            raw=raw,
            labels=source.labels,
            indices=source.indices,
            targets=source.targets,
            model=loaded.model,
            normalize=loaded.normalize,
            device=target_device,
        )
        labels = source.labels.numpy().astype(np.int64, copy=False)
        targets = source.targets.numpy().astype(np.int64, copy=False)
        predictions = source.predictions.numpy().astype(np.int64, copy=False)
        output = {
            "indices": source.indices,
            "labels": source.labels,
            "targets": source.targets,
            "unmasked_predictions": source.predictions,
        }
        traces = evaluate_reference_model_bank(
            loaded.model,
            conditioned,
            source.ranks,
            true_labels=labels,
            target_labels=targets,
            fill_reference=fill,
            reference_model_id=experiment.model_id,
            sample_ids=source.indices.numpy(),
            patch_size=task.patch_size,
            k=task.k,
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
        for rule_index, rule_name in enumerate(source.ranks):
            field_id = f"r{rule_index:03d}"
            rule_labels[field_id] = rule_name
            trace = traces[rule_name]
            if not np.array_equal(trace.clean_predictions, predictions):
                raise RuntimeError("Evaluator did not retain the registered predictions")
            output[f"removed_predictions__{field_id}"] = torch.from_numpy(trace.removed_predictions)
            output[f"retained_predictions__{field_id}"] = torch.from_numpy(
                trace.retained_predictions
            )
            metric_sums[field_id] = _metric_sums(trace)
        records[shard_index] = publish_shard(
            experiment,
            store,
            root=task.artifact_root,
            task_id=task.task_id,
            task_digest=task.digest,
            schema_version=EVALUATION_SCHEMA_VERSION,
            shard_index=shard_index,
            start=start,
            stop=stop,
            tensors=output,
            metadata={
                "patch_size": str(task.patch_size),
                "k": str(task.k),
                "fill": task.fill,
                "precision": "fp32",
            },
            record_fields={"rule_labels": rule_labels, "metric_sums": metric_sums},
        )
        print(
            f"ABLATION_EVALUATION shard={shard_index + 1}/{len(layout)} task={task.task_id}",
            flush=True,
        )
        del source, raw, conditioned, output

    ordered = [records[index] for index, _, _ in layout]
    sample_count = sum(int(record["count"]) for record in ordered)
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {metric: 0.0 for metric in QUALITY_METRICS}
    )
    labels_by_field = {}
    for record in ordered:
        labels_by_field.update({str(k): str(v) for k, v in record["rule_labels"].items()})
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
        clean_manifest = clean_metrics_manifest(experiment, task)
        clean_metrics = clean_manifest.get("metrics")
        if not isinstance(clean_metrics, Mapping) or set(clean_metrics) != set(metrics):
            raise ArtifactError("Clean and perturbed evaluation rules are not aligned")
        robustness = {
            rule: {
                "absolute": {
                    metric: abs(float(clean_metrics[rule][metric]) - float(current[metric]))
                    for metric in QUALITY_METRICS
                }
            }
            for rule, current in metrics.items()
        }
        clean_source = {
            "task_id": clean_manifest["task_id"],
            "task_digest": clean_manifest["task_digest"],
        }
    manifest: Mapping[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "table": task.table_id,
        "parameter": task.parameter,
        "parameter_value": task.parameter_value,
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "split": experiment.split,
        "condition": task.condition.condition_id,
        "construction_setting": "naive",
        "rank_source": {
            "kind": task.rank_source.kind,
            "store": task.rank_source.store,
            "root": task.rank_source.root,
            "digest": task.rank_source.digest,
        },
        "patch_size": task.patch_size,
        "k": task.k,
        "fill": task.fill,
        "fill_artifact_id": fill.artifact_id,
        "fill_class_policy": (
            "fixed_clean_explanation_target" if task.fill == "class_mean" else None
        ),
        "target_policy": "clean_model_fp32_prediction",
        "unmasked_prediction_source": "rank_source_phase1_fp32_logits_argmax",
        "inference_batch_size": experiment.runtime.inference_batch_size,
        "inference_batch_semantics": "maximum_actual_model_forward_batch",
        "sample_count": sample_count,
        "metrics": metrics,
        "robustness": robustness,
        "clean_metric_source": clean_source,
        "shards": ordered,
    }
    publish_manifest(
        experiment,
        store,
        root=task.artifact_root,
        task_id=task.task_id,
        manifest=manifest,
    )
    del bundle, loaded
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


__all__ = ["clean_metrics_manifest", "run_evaluation_task"]
