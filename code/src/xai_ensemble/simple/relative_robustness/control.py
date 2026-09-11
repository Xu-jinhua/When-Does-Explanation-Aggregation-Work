"""Phase-2-only random top-k controls for relative robustness."""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256, stable_seed
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase0.models import get_model_definition
from xai_ensemble.phase2.evaluator import FillReference, evaluate_reference_model_bank
from xai_ensemble.phase2.metrics import QUALITY_METRICS

from ..adversarial import load_adversarial_shard
from ..artifacts import ArtifactError, ArtifactStore
from ..data import (
    apply_condition,
    load_model,
    load_raw_dataset_mean,
    load_split,
    materialize_shared_conditioned_raw_images,
)
from ..io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from ..noise_prefix.inputs import catalog_task_input, load_input_catalog, source_layout
from ..rank_ready import load_existing_rank_ready_sidecar
from ..runtime import emit_gpu_release_signal
from ..spool import SpoolQuota
from ..telemetry import GpuUtilizationSampler, StageTimings
from .artifacts import (
    CONTROL_SCHEMA_VERSION,
    RandomControlShardPublisher,
    completed_control_manifest,
    completed_control_shards,
    output_store,
    publish_manifest,
)
from .config import CONTROL_SCHEMA, RandomControlTask, RelativeRobustnessExperiment


def _rule_label(seed_position: int) -> str:
    return f"random_seed_{seed_position:02d}"


def _field_id(seed_position: int) -> str:
    return f"r{seed_position:03d}"


def _metric_sums(trace: Any) -> Mapping[str, float]:
    contributions = trace.stats.contributions()
    return {
        metric: float(np.sum(contributions[metric], dtype=np.float64)) for metric in QUALITY_METRICS
    }


def random_rank_bank(
    *,
    indices: np.ndarray,
    cell_id: str,
    patch_count: int,
    seed_bank: Sequence[int],
) -> np.ndarray:
    """Return full strict random ranks shared across conditions for each image.

    The condition is intentionally absent from the seed identity.  Thus a
    clean image and every corruption use exactly the same random top-k masks,
    while distinct seeds give independent uniform patch permutations.
    """

    rows = np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional integer array")
    if patch_count <= 0 or not seed_bank:
        raise ValueError("patch_count and seed_bank must be non-empty")
    ranks = np.empty((rows.size, len(seed_bank), patch_count), dtype=np.int32)
    positions = np.arange(patch_count, dtype=np.int32)
    for row_position, row_index in enumerate(rows.tolist()):
        for seed_position, seed in enumerate(seed_bank):
            generator = np.random.default_rng(
                stable_seed(
                    CONTROL_SCHEMA,
                    "uniform-patch-permutation",
                    cell_id,
                    int(row_index),
                    int(seed),
                )
            )
            permutation = generator.permutation(patch_count)
            ranks[row_position, seed_position, permutation] = positions
    return ranks


def _input_source(
    task_row: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    methods = tuple(str(value) for value in task_row.get("ordered_methods", ()))
    if not methods:
        raise ArtifactError("Relative control input catalog has no ordered methods")
    method = methods[0]
    sources = task_row.get("sources")
    if not isinstance(sources, Mapping) or method not in sources:
        raise ArtifactError("Relative control input catalog has no canonical rank-ready source")
    source = sources[method]
    if not isinstance(source, Mapping) or not isinstance(source.get("shards"), Sequence):
        raise ArtifactError("Relative control canonical source is malformed")
    shards = []
    for row in source["shards"]:
        if not isinstance(row, Mapping):
            raise ArtifactError("Relative control canonical source shard is malformed")
        payload = row.get("source_payload")
        sidecar = row.get("rank_ready")
        if not isinstance(payload, Mapping) or not isinstance(sidecar, Mapping):
            raise ArtifactError("Relative control source shard has no verified sidecar")
        shards.append(
            {
                "shard_index": int(row["shard_index"]),
                "start": int(row["start"]),
                "stop": int(row["stop"]),
                "source_payload_sha256": str(payload["sha256"]),
                "rank_ready_relative_path": str(sidecar["relative_path"]),
                "rank_ready_sha256": str(sidecar["sha256"]),
                "rank_ready_identity_digest": str(sidecar["identity_digest"]),
            }
        )
    q11_reference = task_row.get("q11_reference")
    if not isinstance(q11_reference, Mapping):
        raise ArtifactError("Relative control input catalog lacks q=11 reference provenance")
    return method, {
        "canonical_method": method,
        "source_task_digest": str(source["task_digest"]),
        "source_variant_digest": str(source["variant_digest"]),
        "source_shards": sorted(shards, key=lambda row: int(row["shard_index"])),
        "q11_reference_task_id": str(q11_reference["task_id"]),
        "q11_reference_task_digest": str(q11_reference["task_digest"]),
        "q11_reference_manifest_content_digest": str(q11_reference["manifest_content_digest"]),
    }


def _load_fixed_shard(
    experiment: RelativeRobustnessExperiment,
    task_row: Mapping[str, Any],
    *,
    canonical_method: str,
    shard_index: int,
    work_directory: Path,
) -> Mapping[str, Any]:
    import torch

    sources = task_row["sources"]
    source = sources[canonical_method]
    record = next(item for item in source["shards"] if int(item["shard_index"]) == shard_index)
    count = int(record["stop"]) - int(record["start"])
    sidecar = load_existing_rank_ready_sidecar(
        ArtifactStore(experiment.base),
        source_payload=record["source_payload"],
        recorded_sidecar=record["rank_ready"],
        work_directory=work_directory,
        simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
        count=count,
    )
    fields = sidecar.fields
    fixed = {
        name: torch.as_tensor(fields[name]).detach().to("cpu").contiguous().clone()
        for name in ("indices", "labels", "predictions", "logits", "targets")
    }
    if fixed["logits"].ndim != 2 or not torch.equal(
        fixed["predictions"], fixed["logits"].argmax(dim=1)
    ):
        raise ArtifactError("Relative control rank-ready logits contradict predictions")
    return fixed


def _fill(experiment: RelativeRobustnessExperiment, task: RandomControlTask) -> FillReference:
    model = task.prefix_task.cell.reference_model
    definition = get_model_definition(model.model_key)
    values = load_raw_dataset_mean(model, input_size=definition.input_size).squeeze(0)
    mean_path = resolve_full_matrix_runtime_path(model.mean_path)
    manifest = mean_path / "manifest.json" if mean_path.is_dir() else mean_path
    return FillReference(
        values=values,
        source_split="train",
        artifact_id=object_sha256(
            {
                "kind": "dataset_mean",
                "path": str(model.mean_path),
                "digest": file_sha256(manifest),
            }
        ),
    )


def _conditioned_images(
    experiment: RelativeRobustnessExperiment,
    task: RandomControlTask,
    *,
    shard_index: int,
    raw: Any,
    labels: Any,
    indices: Any,
    targets: Any,
    logits: Any,
    model: Any,
    normalize: Any,
    device: Any,
) -> Any:
    import torch

    condition = task.prefix_task.condition
    if condition.kind == "clean":
        return raw.to(dtype=torch.float32)
    if condition.kind == "adversarial":
        attack = experiment.base.adversarial_task_for(
            dataset_id=task.prefix_task.cell.dataset.dataset_id,
            model_id=task.prefix_task.cell.reference_model.model_id,
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
            expected_adversarial_logits=logits,
            store=ArtifactStore(experiment.base),
        )
        return fields["adversarial_images"]
    pieces = []
    source_batch = max(1, experiment.runtime.inference_batch_size // 2)
    for start in range(0, int(raw.shape[0]), source_batch):
        stop = min(int(raw.shape[0]), start + source_batch)
        piece = apply_condition(
            condition,
            raw[start:stop].to(device, dtype=torch.float32, non_blocking=True),
            labels=labels[start:stop].to(device),
            indices=indices[start:stop].to(device),
            model=model,
            normalize=normalize,
            seed=experiment.base.runtime.seed,
        )
        pieces.append(piece.detach().cpu())
    return torch.cat(pieces)


def _metrics_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
) -> Mapping[str, Mapping[str, float]]:
    labels: dict[str, str] = {}
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {metric: 0.0 for metric in QUALITY_METRICS}
    )
    for record in records:
        rule_labels = record.get("rule_labels")
        metric_sums = record.get("metric_sums")
        if not isinstance(rule_labels, Mapping) or not isinstance(metric_sums, Mapping):
            raise ArtifactError("Relative control shard lacks metric records")
        for field_id, label in rule_labels.items():
            field = str(field_id)
            if field in labels and labels[field] != str(label):
                raise ArtifactError("Relative control seed labels changed across shards")
            labels[field] = str(label)
        for field_id, values in metric_sums.items():
            if not isinstance(values, Mapping):
                raise ArtifactError("Relative control metric sums are malformed")
            for metric in QUALITY_METRICS:
                sums[str(field_id)][metric] += float(values[metric])
    if set(labels) != set(sums):
        raise ArtifactError("Relative control metric labels do not align with sums")
    return {
        labels[field_id]: {metric: float(value / sample_count) for metric, value in values.items()}
        for field_id, values in sorted(sums.items())
    }


def run_control_task(
    experiment: RelativeRobustnessExperiment,
    task: RandomControlTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    """Evaluate one condition's fixed random-mask bank without Phase 1 work."""

    import torch

    catalog = load_input_catalog(experiment.prefix)
    task_row = catalog_task_input(experiment.prefix, task.prefix_task, catalog=catalog)
    canonical_method, input_source = _input_source(task_row)
    store = output_store(experiment)
    complete = completed_control_manifest(experiment, task, store=store)
    if complete is not None:
        if complete.get("input_source") != dict(input_source):
            raise ArtifactError(
                "Completed random control source differs from current rank-ready inputs"
            )
        return complete
    layout = source_layout(task_row)
    records = completed_control_shards(experiment, task, source_layout=layout, store=store)
    missing_layout = tuple(item for item in layout if item[0] not in records)
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Relative robustness random controls require CUDA")

    publisher = RandomControlShardPublisher(experiment, store, task_id=task.task_id)  # type: ignore[arg-type]
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
                task.prefix_task.cell.reference_model,
                device=target_device,
                include_checkpoint=True,
            )
            bundle = load_split(
                task.prefix_task.cell.dataset,
                loaded,
                split=experiment.split,
                workers=experiment.base.runtime.dataloader_workers,
                shared_cache_root=experiment.runtime.shared_cache_root,
            )
            if task.prefix_task.condition.kind == "factory":
                with timings.measure("shared_condition_cache"):
                    shared_conditioned = materialize_shared_conditioned_raw_images(
                        bundle,
                        loaded,
                        task.prefix_task.condition,
                        device=target_device,
                        batch_size=experiment.runtime.inference_batch_size,
                        seed=experiment.base.runtime.seed,
                        shared_cache_root=experiment.runtime.shared_cache_root,
                    )
            fill = _fill(experiment, task)
            quota = SpoolQuota(
                experiment.storage.spool_root,
                max_bytes=experiment.storage.spool_max_bytes,
                min_free_bytes=experiment.storage.spool_min_free_bytes,
            )
            prefetch_items = []
            source_rows = task_row["sources"][canonical_method]["shards"]
            for shard_index, _, _ in missing_layout:
                source_record = next(
                    row for row in source_rows if int(row["shard_index"]) == shard_index
                )
                sidecar = source_record["rank_ready"]
                prefetch_items.append(
                    PrefetchItem(
                        key=shard_index,
                        byte_count=max(1, int(sidecar["size_bytes"])),
                        load=lambda directory, index=shard_index: _load_fixed_shard(
                            experiment,
                            task_row,
                            canonical_method=canonical_method,
                            shard_index=index,
                            work_directory=directory,
                        ),
                    )
                )
            with ByteBoundedPrefetcher(
                quota,
                prefetch_items,
                workers=experiment.runtime.cpu_workers,
                namespace=f"relative-control-{task.digest[:16]}",
            ) as prefetcher:
                for shard_index, start, stop in missing_layout:
                    prefetched = prefetcher.get(shard_index)
                    timings.add("input_wait", prefetched.wait_seconds)
                    timings.add("input_materialize", prefetched.load_seconds)
                    fixed = prefetched.value
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
                                experiment,
                                task,
                                shard_index=shard_index,
                                raw=raw,
                                labels=fixed["labels"],
                                indices=fixed["indices"],
                                targets=fixed["targets"],
                                logits=fixed["logits"],
                                model=loaded.model,
                                normalize=loaded.normalize,
                                device=target_device,
                            )
                        else:
                            conditioned = shared_conditioned[positions]
                    height, width = (int(conditioned.shape[-2]), int(conditioned.shape[-1]))
                    if height % task.patch_size or width % task.patch_size:
                        raise ArtifactError(
                            "Relative control image shape is incompatible with p=16"
                        )
                    with timings.measure("random_rank_generation"):
                        ranks = random_rank_bank(
                            indices=indices,
                            cell_id=task.cell_id,
                            patch_count=(height // task.patch_size) * (width // task.patch_size),
                            seed_bank=experiment.random_seed_bank,
                        )
                        rank_bank = {
                            _rule_label(position): ranks[:, position]
                            for position in range(len(experiment.random_seed_bank))
                        }
                    labels = fixed["labels"].numpy().astype(np.int64, copy=False)
                    targets = fixed["targets"].numpy().astype(np.int64, copy=False)
                    predictions = fixed["predictions"].numpy().astype(np.int64, copy=False)
                    with timings.measure("gpu_random_bank_evaluation"):
                        traces = evaluate_reference_model_bank(
                            loaded.model,
                            conditioned,
                            rank_bank,
                            true_labels=labels,
                            target_labels=targets,
                            fill_reference=fill,
                            reference_model_id=task.prefix_task.cell.reference_model.model_id,
                            sample_ids=indices,
                            patch_size=task.patch_size,
                            k=task.k,
                            index_base=0,
                            batch_size=experiment.runtime.inference_batch_size,
                            device=str(target_device),
                            autocast=False,
                            normalize=loaded.normalize,
                            require_target_matches_clean=task.prefix_task.condition.kind == "clean",
                            clean_predictions=predictions,
                        )
                    tensors: dict[str, Any] = {
                        "indices": fixed["indices"],
                        "labels": fixed["labels"],
                        "targets": fixed["targets"],
                        "unmasked_predictions": fixed["predictions"],
                    }
                    rule_labels = {}
                    metric_sums = {}
                    for seed_position, (label, trace) in enumerate(traces.items()):
                        if not np.array_equal(trace.clean_predictions, predictions):
                            raise RuntimeError(
                                "Random-control evaluator changed registered predictions"
                            )
                        field_id = _field_id(seed_position)
                        rule_labels[field_id] = label
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
                            schema_version=CONTROL_SCHEMA_VERSION,
                            shard_index=shard_index,
                            start=start,
                            stop=stop,
                            tensors=tensors,
                            metadata={
                                "schema": CONTROL_SCHEMA,
                                "patch_size": str(task.patch_size),
                                "k": str(task.k),
                                "fill": "dataset_mean",
                                "precision": "fp32",
                                "rank_payload": "random_top_k_patch_indices_only",
                            },
                            record_fields={
                                "rule_labels": rule_labels,
                                "metric_sums": metric_sums,
                            },
                        )
                    print(
                        "RELATIVE_RANDOM_CONTROL "
                        f"shard={shard_index + 1}/{len(layout)} "
                        f"prefetch_wait_seconds={prefetched.wait_seconds:.3f} "
                        f"prefetch_load_seconds={prefetched.load_seconds:.3f} "
                        f"seeds={len(traces)} task={task.task_id}",
                        flush=True,
                    )
                    del raw, conditioned, ranks, rank_bank, traces, tensors, fixed
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

    ordered = [records[index] for index, _, _ in layout]
    sample_count = sum(int(record["count"]) for record in ordered)
    metrics = _metrics_from_records(ordered, sample_count=sample_count)
    expected_labels = tuple(_rule_label(index) for index in range(len(experiment.random_seed_bank)))
    if tuple(metrics) != expected_labels:
        raise ArtifactError("Random-control metrics do not cover the fixed seed bank")
    mean_metrics = {
        metric: float(np.mean([metrics[label][metric] for label in expected_labels]))
        for metric in QUALITY_METRICS
    }
    manifest: Mapping[str, Any] = {
        "schema": CONTROL_SCHEMA,
        "schema_version": CONTROL_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "prefix_sweep_id": experiment.prefix.sweep_id,
        "prefix_sweep_digest": experiment.prefix.digest,
        "input_catalog_digest": catalog["catalog_digest"],
        "input_source": dict(input_source),
        "cell": task.cell_id,
        "dataset": task.prefix_task.cell.dataset.dataset_id,
        "model": task.prefix_task.cell.reference_model.model_id,
        "split": experiment.split,
        "condition": task.prefix_task.condition.condition_id,
        "condition_kind": task.prefix_task.condition.kind,
        "patch_size": task.patch_size,
        "k": task.k,
        "fill": "dataset_mean",
        "fill_artifact_id": _fill(experiment, task).artifact_id,
        "precision": "fp32",
        "autocast": False,
        "target_policy": "full_reference_clean_fp32_prediction",
        "unmasked_prediction_source": "rank_ready_full_reference_fp32_logits_argmax",
        "random_mask_policy": "uniform_patch_permutation_shared_across_conditions_per_image_seed",
        "random_seed_bank": list(experiment.random_seed_bank),
        "random_seed_labels": list(expected_labels),
        "rank_payload": "top_k_patch_indices_only_no_full_random_rank",
        "inference_batch_size": experiment.runtime.inference_batch_size,
        "inference_batch_semantics": "maximum_actual_model_forward_batch",
        "sample_count": sample_count,
        "runtime_telemetry": {"stages": timings.summary(), "gpu": telemetry},
        "metrics": metrics,
        "mean_metrics": mean_metrics,
        "shards": ordered,
    }
    publish_manifest(
        experiment,  # type: ignore[arg-type]
        store,
        root=task.artifact_root,
        task_id=task.task_id,
        manifest=manifest,
    )
    return manifest


__all__ = ["random_rank_bank", "run_control_task"]
