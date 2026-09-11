"""Test-set Fidelity ordering and Oracle NOISE prefix selection."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.phase2.selection import (
    locked_spearman_family_factory,
    order_methods_by_fidelity,
    select_noise_from_distance_samples,
)
from xai_ensemble.phase2.torch_aggregation import aggregate_rankings_torch

from ..artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    load_safetensors,
    phase2_artifact_root,
)
from ..runtime import emit_gpu_release_signal
from .artifacts import (
    SELECTION_SCHEMA_VERSION,
    completed_selection_manifest,
    output_store,
    publish_manifest,
    task_spool_path,
)
from .config import AssumptionExperiment, SelectionTask
from .prepare import SPEARMAN_ITEMS, spearman_family_config

_CUDA_WORKSPACE_BYTES = 2 * 2**30


def _base_manifest(
    experiment: AssumptionExperiment,
    task: SelectionTask,
) -> tuple[Any, str, Mapping[str, Any]]:
    source_task = experiment.base_phase2_task(task.cell, "clean")
    store = ArtifactStore(experiment.base)
    root = phase2_artifact_root(source_task)
    manifest = completed_manifest(
        store,
        root,
        expected_task_digest=source_task.digest,
        expected_schema_version=PHASE2_SCHEMA_VERSION,
    )
    if manifest is None:
        raise FileNotFoundError(f"NAIVE clean p=16 Phase 2 artifact is incomplete: {root}")
    if manifest.get("patch_size") != 16 or manifest.get("rank_base") != 0:
        raise ArtifactError("Oracle NOISE requires zero-based p=16 NAIVE ranks")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ArtifactError("NAIVE Phase 2 manifest has no metric summary")
    return source_task, root, manifest


def _load_rank_shard(
    experiment: AssumptionExperiment,
    task: SelectionTask,
    *,
    root: str,
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    store = ArtifactStore(experiment.base)
    payload = record["payload"]
    local = task_spool_path(
        experiment,
        namespace="naive-rank-cache",
        task_digest=task.digest,
        relative_path=str(payload["relative_path"]),
    )
    store.materialize(
        str(payload["relative_path"]),
        local,
        expected_sha256=str(payload["sha256"]),
    )
    try:
        fields = load_safetensors(local)
        labels = record.get("rule_labels")
        if not isinstance(labels, Mapping):
            raise ArtifactError("NAIVE Phase 2 shard has no rule label mapping")
        by_name = {str(name): str(field_id) for field_id, name in labels.items()}
        missing = [family for family in task.cell.methods if f"single__{family}" not in by_name]
        if missing:
            raise ArtifactError(f"NAIVE Phase 2 shard lacks individual ranks: {missing}")
        ballots = np.stack(
            [
                fields[f"rank__{by_name[f'single__{family}']}"].numpy().astype(np.int64, copy=False)
                for family in task.cell.methods
            ],
            axis=1,
        ).copy()
        indices = fields["indices"].numpy().astype(np.int64, copy=False).copy()
    finally:
        local.unlink(missing_ok=True)
    return indices, ballots


def _distances(
    ballots: np.ndarray,
    consensus: np.ndarray,
    *,
    distance: str,
    device: Any,
) -> np.ndarray:
    import torch

    values = torch.as_tensor(ballots, device=device, dtype=torch.int64)
    center = torch.as_tensor(consensus, device=device, dtype=torch.int64)
    if distance == "spearman":
        result = ((values - center[:, None, :]) ** 2).sum(dim=2)
        return result.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    if distance != "kendall":
        raise ValueError(distance)
    chunks = []
    for start in range(0, int(values.shape[0]), 16):
        stop = min(int(values.shape[0]), start + 16)
        current = values[start:stop]
        ordering = torch.argsort(center[start:stop], dim=1, stable=True)
        ordered = torch.gather(
            current,
            2,
            ordering[:, None, :].expand(-1, current.shape[1], -1),
        )
        pairwise = ordered[:, :, :, None] > ordered[:, :, None, :]
        inversions = torch.triu(pairwise, diagonal=1).sum(dim=(2, 3))
        chunks.append(inversions.detach().cpu())
    return torch.cat(chunks).numpy().reshape(-1).astype(np.int64, copy=False)


def _serialize_selection(result: Any) -> Mapping[str, Any]:
    return {
        "selected_size": result.selected_size,
        "selected_methods": list(result.selected_methods),
        "ordered_methods": list(result.ordered_methods),
        "selection_rule": result.selection_rule,
        "alpha": result.alpha,
        "forced_fallback": result.forced_fallback,
        "scope": result.scope,
        "evaluations": [
            {
                "size": item.size,
                "methods": list(item.methods),
                "distance": item.distance,
                "n_distances": item.n_distances,
                "mean_distance": item.mean_distance,
                "gof": {
                    "statistic": item.gof.statistic,
                    "p_value": item.gof.p_value,
                    "B": item.gof.B,
                    "model_name": item.gof.model_name,
                    "parameters": dict(item.gof.parameters),
                    "refit": item.gof.refit,
                },
            }
            for item in result.evaluations
        ],
    }


def run_selection_task(
    experiment: AssumptionExperiment,
    task: SelectionTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    import torch

    output = output_store(experiment)
    complete = completed_selection_manifest(experiment, task, store=output)
    if complete is not None:
        return complete
    source_task, source_root, source = _base_manifest(experiment, task)
    metrics = source["metrics"]
    fidelity = {family: float(metrics[f"single__{family}"]["F"]) for family in task.cell.methods}
    ordering, ordered_methods = order_methods_by_fidelity(task.cell.methods, fidelity)
    prefix_sizes = tuple(range(experiment.selection.min_prefix, len(task.cell.methods) + 1))
    accumulated: dict[int, list[np.ndarray]] = {size: [] for size in prefix_sizes}
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Oracle NOISE distance construction requires CUDA")
    sample_count = 0
    try:
        for record in source["shards"]:
            indices, all_ballots = _load_rank_shard(
                experiment, task, root=source_root, record=record
            )
            sample_count += int(indices.size)
            for size in prefix_sizes:
                selected = all_ballots[:, ordering[:size], :]
                seeds = tuple(
                    int(
                        object_sha256(
                            {
                                "selection": task.digest,
                                "prefix": size,
                                "row_index": int(index),
                            }
                        )[:15],
                        16,
                    )
                    for index in indices
                )
                requested = ("Borda",) if task.aggregation == "borda" else ("Kemeny",)
                aggregation = aggregate_rankings_torch(
                    selected,
                    None,
                    requested=requested,
                    rrf_c=experiment.base.phase2.rrf_c,
                    kemeny_starts=experiment.base.phase2.kemeny_starts,
                    kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
                    seeds=seeds,
                    device=target_device,
                    workspace_bytes=_CUDA_WORKSPACE_BYTES,
                )
                consensus = aggregation[task.aggregation]
                accumulated[size].append(
                    _distances(
                        selected,
                        consensus,
                        distance=task.distance_model,
                        device=target_device,
                    )
                )
                del aggregation, consensus
            del all_ballots
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.cuda.synchronize(target_device)
        gc.collect()
        torch.cuda.empty_cache()
        emit_gpu_release_signal()

    distances = {size: np.concatenate(values) for size, values in accumulated.items()}
    family_factory = None
    family_identity = None
    if task.distance_model == "spearman":
        if (
            next(iter(distances.values())).max(initial=0)
            > SPEARMAN_ITEMS * (SPEARMAN_ITEMS * SPEARMAN_ITEMS - 1) // 3
        ):
            raise ArtifactError("Observed Spearman distance exceeds the p=16 support")
        family_identity = dict(spearman_family_config(experiment))
        pilot_digest = str(family_identity.pop("pilot_digest"))

        def family_factory(distance: str, n_items: int) -> Any:
            return locked_spearman_family_factory(
                distance=distance,
                n_items=n_items,
                seed=experiment.assignment_seed,
                pilot_digest=pilot_digest,
                config=family_identity,
            )

    result = select_noise_from_distance_samples(
        distances,
        ordered_methods=ordered_methods,
        distance=task.distance_model,
        n_items=SPEARMAN_ITEMS,
        B=experiment.selection.bootstrap_replicates,
        alpha=experiment.selection.alpha,
        selection_rule=experiment.selection.selection_rule,
        seed=int(task.digest[:15], 16),
        gof_family_factory=family_factory,
        bootstrap_refit=True,
    )
    selection = _serialize_selection(result)
    value: Mapping[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "setting": "Oracle NOISE",
        "oracle_scope": "complete_test_set_in_sample",
        "cell": task.cell.cell_id,
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "split": experiment.split,
        "distance_model": task.distance_model,
        "consensus_for_selection": task.aggregation,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "sample_count": sample_count,
        "fidelity_metric": "F",
        "fidelity_scope": "complete_test_set",
        "fidelity": fidelity,
        "source_phase2": {
            "task_id": source_task.task_id,
            "task_digest": source_task.digest,
            "root": source_root,
        },
        "spearman_family": family_identity,
        "selection": selection,
    }
    publish_manifest(
        experiment, output, root=task.artifact_root, task_id=task.task_id, manifest=value
    )
    return value


__all__ = ["run_selection_task"]
