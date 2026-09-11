"""Materialize active-cell configs and collect existing worker DAGs."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.simple.assumptions.config import load_assumption_experiment
from xai_ensemble.simple.config import load_experiment
from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
from xai_ensemble.simple.scheduler import QueueJob

from .assets import compatibility_complete
from .catalog import MatrixCell
from .config import FullMatrixExperiment

EXECUTION_SCOPE_SCHEMA = "simple-full-matrix-execution-scope-v1"
SCOPED_ACTIVE_CELLS_SCHEMA = "simple-full-matrix-active-cells-v2"


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """A frozen downstream subset derived from completed Phase-0 cells."""

    scope_id: str
    scope_digest: str
    deferred_datasets: tuple[str, ...]
    cell_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPaths:
    """Scope-specific outputs which must never collide with the full matrix."""

    work_root: Path
    generated_root: Path
    scope_manifest_path: Path | None
    active_cells_path: Path
    base_config_path: Path
    assumptions_config_path: Path
    prefix_config_path: Path
    result_root: Path


def _scope_identity(
    experiment: FullMatrixExperiment,
    *,
    deferred_datasets: Sequence[str],
    cell_ids: Sequence[str],
) -> Mapping[str, Any]:
    return {
        "schema": EXECUTION_SCOPE_SCHEMA,
        "matrix_id": experiment.experiment_id,
        "matrix_digest": experiment.digest,
        "protocol_digest": experiment.protocol_digest,
        "deferred_datasets": list(deferred_datasets),
        "planned_cells": list(cell_ids),
        "selection_policy": "reference_training_succeeded_then_strict_compatibility",
    }


def build_execution_scope(
    experiment: FullMatrixExperiment,
    *,
    deferred_datasets: Sequence[str],
    cell_ids: Sequence[str],
    expected_digest: str | None = None,
) -> ExecutionScope:
    """Construct a deterministic, explicitly partial downstream scope.

    Deferred datasets cannot be silently reintroduced through generated Phase
    1/2 configs.  The selected cells are also frozen into the identity so later
    Phase-0 completions cannot change an in-flight partial experiment.
    """

    deferred = tuple(sorted(set(str(dataset) for dataset in deferred_datasets)))
    if not deferred:
        raise ValueError("An execution scope requires at least one deferred dataset")
    unknown_datasets = set(deferred) - set(experiment.dataset_ids)
    if unknown_datasets:
        raise ValueError(f"Execution scope names unknown datasets: {sorted(unknown_datasets)}")

    known_cells = {cell.cell_id: cell for cell in experiment.cells()}
    selected = tuple(sorted(set(str(cell_id) for cell_id in cell_ids)))
    if not selected:
        raise ValueError("An execution scope requires at least one planned cell")
    unknown_cells = set(selected) - set(known_cells)
    if unknown_cells:
        raise ValueError(f"Execution scope names unknown cells: {sorted(unknown_cells)}")
    deferred_cells = [
        cell_id for cell_id in selected if known_cells[cell_id].dataset_id in set(deferred)
    ]
    if deferred_cells:
        raise ValueError(
            f"Execution scope cannot include cells from deferred datasets: {deferred_cells}"
        )

    identity = _scope_identity(
        experiment,
        deferred_datasets=deferred,
        cell_ids=selected,
    )
    digest = object_sha256(identity)
    if expected_digest is not None and expected_digest != digest:
        raise ValueError("Execution scope digest is contradictory")
    return ExecutionScope(
        scope_id=f"ready-{digest[:16]}",
        scope_digest=digest,
        deferred_datasets=deferred,
        cell_ids=selected,
    )


def execution_paths(
    experiment: FullMatrixExperiment,
    scope: ExecutionScope | None = None,
) -> ExecutionPaths:
    if scope is None:
        return ExecutionPaths(
            work_root=experiment.storage.run_root,
            generated_root=experiment.generated_root,
            scope_manifest_path=None,
            active_cells_path=experiment.active_cells_path,
            base_config_path=experiment.base_config_path,
            assumptions_config_path=experiment.assumptions_config_path,
            prefix_config_path=experiment.prefix_config_path,
            result_root=experiment.result_root,
        )
    root = experiment.generated_root / "scopes" / scope.scope_id
    return ExecutionPaths(
        work_root=experiment.storage.run_root / "scopes" / scope.scope_id,
        generated_root=root,
        scope_manifest_path=root / "scope.json",
        active_cells_path=root / "active-cells.json",
        base_config_path=root / "paper-main.yaml",
        assumptions_config_path=root / "paper-assumptions.yaml",
        prefix_config_path=root / "paper-noise-prefix.yaml",
        result_root=experiment.result_root / "scopes" / scope.scope_id,
    )


def _scope_metadata(scope: ExecutionScope) -> Mapping[str, Any]:
    return {
        "schema": EXECUTION_SCOPE_SCHEMA,
        "scope_id": scope.scope_id,
        "scope_digest": scope.scope_digest,
        "deferred_datasets": list(scope.deferred_datasets),
        "planned_cells": list(scope.cell_ids),
        "selection_policy": "reference_training_succeeded_then_strict_compatibility",
    }


def write_execution_scope(
    experiment: FullMatrixExperiment,
    scope: ExecutionScope,
) -> Path:
    """Seal the downstream scope before generating component configs."""

    paths = execution_paths(experiment, scope)
    assert paths.scope_manifest_path is not None
    payload = {
        **_scope_identity(
            experiment,
            deferred_datasets=scope.deferred_datasets,
            cell_ids=scope.cell_ids,
        ),
        **_scope_metadata(scope),
    }
    path = paths.scope_manifest_path
    if path.is_file():
        observed = read_json(path)
        if observed != payload:
            raise ValueError("Existing execution scope contradicts the requested scope")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def load_execution_scope(
    experiment: FullMatrixExperiment,
    path: str | Path,
) -> ExecutionScope:
    target = Path(path).expanduser().resolve()
    value = read_json(target)
    if not isinstance(value, Mapping) or value.get("schema") != EXECUTION_SCOPE_SCHEMA:
        raise ValueError(f"Execution scope is invalid: {target}")
    datasets = value.get("deferred_datasets")
    cells = value.get("planned_cells")
    if not isinstance(datasets, list) or not isinstance(cells, list):
        raise ValueError(f"Execution scope is malformed: {target}")
    scope = build_execution_scope(
        experiment,
        deferred_datasets=tuple(str(item) for item in datasets),
        cell_ids=tuple(str(item) for item in cells),
        expected_digest=str(value.get("scope_digest", "")),
    )
    if value != {
        **_scope_identity(
            experiment,
            deferred_datasets=scope.deferred_datasets,
            cell_ids=scope.cell_ids,
        ),
        **_scope_metadata(scope),
    }:
        raise ValueError(f"Execution scope is contradictory: {target}")
    expected_path = execution_paths(experiment, scope).scope_manifest_path
    if expected_path is None or target != expected_path:
        raise ValueError(f"Execution scope path is contradictory: {target}")
    return scope


def _yaml(path: Path, value: Mapping[str, Any]) -> Path:
    serialized = yaml.safe_dump(dict(value), sort_keys=False, width=100)
    if path.is_file():
        observed = path.read_text(encoding="utf-8")
        if observed != serialized:
            raise ValueError(
                f"Generated component config already exists with different bytes: {path}"
            )
        return path
    return atomic_write_text(path, serialized)


def _active_manifest_without_digest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: item for key, item in value.items() if key != "active_manifest_digest"}


def _active_cell_rows(
    experiment: FullMatrixExperiment,
    cells: Sequence[MatrixCell],
) -> list[Mapping[str, Any]]:
    rows = []
    for cell in cells:
        if not compatibility_complete(experiment, cell):
            raise FileNotFoundError(f"Compatibility gate is incomplete: {cell.cell_id}")
        gate_path = experiment.compatibility_directory(cell) / "gate.json"
        gate = read_json(gate_path)
        rows.append(
            {
                "cell": cell.cell_id,
                "dataset": cell.dataset_id,
                "model": cell.model_key,
                "architecture": cell.architecture,
                "status": "active" if gate["status"] == "passed" else "blocked",
                "gate_digest": gate["gate_digest"],
                "blocked_methods": list(gate["blocked_methods"]),
                "failures": list(gate["failures"]),
                "gate_path": str(gate_path),
            }
        )
    return rows


def resolve_active_cells(experiment: FullMatrixExperiment) -> Mapping[str, Any]:
    cells = _active_cell_rows(experiment, experiment.cells())
    active = [row for row in cells if row["status"] == "active"]
    blocked = [row for row in cells if row["status"] == "blocked"]
    value: dict[str, Any] = {
        "schema": "simple-full-matrix-active-cells-v1",
        "schema_version": 1,
        "status": "complete",
        "matrix_id": experiment.experiment_id,
        "matrix_digest": experiment.digest,
        "policy": {
            "required_methods_per_cell": 11,
            "failed_method_action": "block_complete_cell_without_roster_reduction",
            "swin_action": "block_each_failed_swin_cell_and_report_before_formal_use",
        },
        "counts": {
            "planned_cells": len(cells),
            "active_cells": len(active),
            "blocked_cells": len(blocked),
        },
        "cells": cells,
    }
    value["active_manifest_digest"] = object_sha256(value)
    path = experiment.active_cells_path
    if path.is_file():
        observed = read_json(path)
        if observed != value:
            raise ValueError(
                "Existing active-cell manifest contradicts current compatibility gates; "
                "use a new experiment identity instead of overwriting it"
            )
        return observed
    atomic_write_json(path, value)
    return value


def load_active_cells(experiment: FullMatrixExperiment) -> tuple[MatrixCell, ...]:
    value = read_json(experiment.active_cells_path)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "simple-full-matrix-active-cells-v1"
        or value.get("matrix_digest") != experiment.digest
        or value.get("active_manifest_digest")
        != object_sha256(_active_manifest_without_digest(value))
    ):
        raise ValueError("Active-cell manifest is invalid")
    active_ids = {
        str(row["cell"])
        for row in value["cells"]
        if isinstance(row, Mapping) and row.get("status") == "active"
    }
    cells = tuple(cell for cell in experiment.cells() if cell.cell_id in active_ids)
    if len(cells) != int(value["counts"]["active_cells"]):
        raise ValueError("Active-cell manifest coverage is contradictory")
    return cells


def resolve_scoped_active_cells(
    experiment: FullMatrixExperiment,
    scope: ExecutionScope,
) -> Mapping[str, Any]:
    """Write an active-cell manifest for exactly one immutable partial scope."""

    paths = execution_paths(experiment, scope)
    scoped_cells = tuple(experiment.cell(cell_id) for cell_id in scope.cell_ids)
    cells = _active_cell_rows(experiment, scoped_cells)
    active = [row for row in cells if row["status"] == "active"]
    blocked = [row for row in cells if row["status"] == "blocked"]
    value: dict[str, Any] = {
        "schema": SCOPED_ACTIVE_CELLS_SCHEMA,
        "schema_version": 2,
        "status": "complete",
        "matrix_id": experiment.experiment_id,
        "matrix_digest": experiment.digest,
        "execution_scope": _scope_metadata(scope),
        "policy": {
            "required_methods_per_cell": 11,
            "failed_method_action": "block_complete_cell_without_roster_reduction",
            "swin_action": "block_each_failed_swin_cell_and_report_before_formal_use",
        },
        "counts": {
            "planned_cells": len(cells),
            "active_cells": len(active),
            "blocked_cells": len(blocked),
            "deferred_datasets": len(scope.deferred_datasets),
        },
        "cells": cells,
    }
    value["active_manifest_digest"] = object_sha256(value)
    path = paths.active_cells_path
    if path.is_file():
        observed = read_json(path)
        if observed != value:
            raise ValueError(
                "Existing scoped active-cell manifest contradicts current compatibility gates"
            )
        return observed
    atomic_write_json(path, value)
    return value


def load_scoped_active_cells(
    experiment: FullMatrixExperiment,
    scope: ExecutionScope,
) -> tuple[MatrixCell, ...]:
    value = read_json(execution_paths(experiment, scope).active_cells_path)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != SCOPED_ACTIVE_CELLS_SCHEMA
        or value.get("matrix_digest") != experiment.digest
        or value.get("execution_scope") != _scope_metadata(scope)
        or value.get("active_manifest_digest")
        != object_sha256(_active_manifest_without_digest(value))
    ):
        raise ValueError("Scoped active-cell manifest is invalid")
    active_ids = {
        str(row["cell"])
        for row in value["cells"]
        if isinstance(row, Mapping) and row.get("status") == "active"
    }
    if not active_ids <= set(scope.cell_ids):
        raise ValueError("Scoped active-cell manifest contains an out-of-scope cell")
    cells = tuple(experiment.cell(cell_id) for cell_id in scope.cell_ids if cell_id in active_ids)
    if len(cells) != int(value["counts"]["active_cells"]):
        raise ValueError("Scoped active-cell manifest coverage is contradictory")
    return cells


def _dataset_rows(
    experiment: FullMatrixExperiment,
    active_cells: Sequence[MatrixCell],
) -> list[Mapping[str, Any]]:
    active_datasets = {cell.dataset_id for cell in active_cells}
    return [
        {
            "id": dataset_id,
            "registry_key": dataset_id,
            "manifest_path": str(experiment.manifest_path(dataset_id)),
            "splits": ["test"],
            "cache_directory": str(experiment.cache_directory(dataset_id)),
            "keep_provider_in_memory": False,
            "cache_images_in_ram": True,
        }
        for dataset_id in experiment.dataset_ids
        if dataset_id in active_datasets
    ]


def _model_rows(
    experiment: FullMatrixExperiment,
    active_cells: Sequence[MatrixCell],
) -> list[Mapping[str, Any]]:
    return [
        {
            "id": cell.model_id,
            "dataset": cell.dataset_id,
            "model_key": cell.model_key,
            "architecture": cell.architecture,
            "num_classes": cell.num_classes,
            "init_mode": "checkpoint",
            "checkpoint_path": str(experiment.checkpoint_path(cell)),
            "strict_checkpoint": True,
            "mean_path": str(experiment.mean_path(cell)),
            "mean_key": "dataset_mean",
        }
        for cell in active_cells
    ]


def _conditions() -> list[Mapping[str, Any]]:
    return [
        {"id": "clean", "kind": "clean"},
        {
            "id": "gaussian-0.15",
            "kind": "factory",
            "factory": "xai_ensemble.simple.conditions:natural_corruption",
            "kwargs": {"kind": "gaussian", "severity": 0.15},
        },
        {
            "id": "salt-pepper-0.05",
            "kind": "factory",
            "factory": "xai_ensemble.simple.conditions:natural_corruption",
            "kwargs": {"kind": "salt_pepper", "severity": 0.05},
        },
        {
            "id": "speckle-0.15",
            "kind": "factory",
            "factory": "xai_ensemble.simple.conditions:natural_corruption",
            "kwargs": {"kind": "speckle", "severity": 0.15},
        },
        {
            "id": "adversarial-sara-2-255",
            "kind": "adversarial",
            "kwargs": {
                "algorithm": "sara-repetto-v2",
                "epsilon": 2.0 / 255.0,
                "steps": 100,
                "learning_rate": 0.1,
                "classification_weight": 0.0001,
                "top_fraction": 0.1,
                "source_by_architecture": {
                    "cnn": "DeepLift",
                    "vit": "TransformerAttribution",
                },
                "batch_size_by_architecture": {"cnn": 32, "vit": 16},
            },
        },
    ]


def _component_identifier(
    experiment: FullMatrixExperiment,
    *,
    suffix: str,
    scope: ExecutionScope | None,
) -> str:
    if scope is None:
        return f"{experiment.experiment_id}-{suffix}"
    return f"{experiment.experiment_id}-{scope.scope_id}-{suffix}"


def _scoped_remote_root(root: str, scope: ExecutionScope | None) -> str:
    if scope is None:
        return root
    return posixpath.join(root.rstrip("/"), "scopes", scope.scope_id)


def _full_matrix_metadata(
    experiment: FullMatrixExperiment,
    active_manifest: Mapping[str, Any],
    *,
    scope: ExecutionScope | None,
    include_release_scope: bool,
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "matrix_digest": experiment.digest,
        "active_manifest_digest": active_manifest["active_manifest_digest"],
    }
    if include_release_scope:
        value["scope"] = "github_complete_results_without_parameter_ablations"
    if scope is not None:
        value["execution_scope"] = _scope_metadata(scope)
    return value


def _base_config(
    experiment: FullMatrixExperiment,
    active_cells: Sequence[MatrixCell],
    active_manifest: Mapping[str, Any],
    *,
    paths: ExecutionPaths | None = None,
    scope: ExecutionScope | None = None,
) -> Mapping[str, Any]:
    paths = paths or execution_paths(experiment, scope)
    run_root = paths.work_root
    return {
        "schema_version": 1,
        "experiment_id": _component_identifier(experiment, suffix="naive", scope=scope),
        "precision": "fp32",
        "seed": experiment.seed,
        # The complete release is not a p-ablation. Keep p=8/14 available in
        # the shared catalog for the existing ablation studies, but do not
        # regenerate them across all 112 cells.
        "phase1_patch_sizes": [16],
        "methods_file": str(experiment.methods_path),
        "full_matrix": _full_matrix_metadata(
            experiment,
            active_manifest,
            scope=scope,
            include_release_scope=True,
        ),
        "storage": {
            "remote_root": _scoped_remote_root(experiment.storage.base_remote_root, scope),
            "scratch_root": str(run_root / "base" / "scratch"),
            "rclone_binary": str(experiment.storage.rclone_binary),
            "spool_root": f"/dev/shm/xai-simple/{_component_identifier(experiment, suffix='base', scope=scope)}",
            "spool_max_gib": 160,
            "spool_min_free_gib": 24,
        },
        "runtime": {
            "profile_directory": str(run_root / "base" / "profiles"),
            "database_path": str(run_root / "component-databases" / "base.sqlite3"),
            "log_directory": str(run_root / "base" / "logs"),
            "gpu_ids": list(experiment.runtime.gpu_ids),
            "search_grid": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "default_profile_start": 256,
            "shard_size": 512,
            "prediction_batch_size": 512,
            "dataloader_workers": experiment.training.workers,
            "headroom_fraction": experiment.runtime.headroom_fraction,
            "max_retries": experiment.runtime.max_retries,
        },
        "datasets": _dataset_rows(experiment, active_cells),
        "models": _model_rows(experiment, active_cells),
        "conditions": _conditions(),
        "phase2": {
            "patch_sizes": [16],
            "primary_patch_size": 16,
            "k": 20,
            "inference_batch_size": 512,
            "simpleavg_normalization": "minmax",
            "rrf_c": 60,
            "kemeny_starts": 1,
            "kemeny_max_passes": 1024,
            "ensembles": [
                {
                    "id": "all-paper-methods",
                    "methods": "architecture_default",
                    "rules": ["SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"],
                    "include_singles": True,
                }
            ],
        },
    }


def _assumptions_config(
    experiment: FullMatrixExperiment,
    active_manifest: Mapping[str, Any],
    *,
    paths: ExecutionPaths | None = None,
    scope: ExecutionScope | None = None,
) -> Mapping[str, Any]:
    paths = paths or execution_paths(experiment, scope)
    run_root = paths.work_root
    class_balance = {"default": "none", **dict(experiment.training.class_balance)}
    return {
        "schema_version": 1,
        "assumption_id": _component_identifier(experiment, suffix="full-ind", scope=scope),
        "base_config": str(paths.base_config_path),
        "full_matrix": _full_matrix_metadata(
            experiment,
            active_manifest,
            scope=scope,
            include_release_scope=False,
        ),
        "storage": {
            "remote_root": _scoped_remote_root(experiment.storage.assumptions_remote_root, scope),
            "scratch_root": str(run_root / "assumptions" / "scratch"),
            "rclone_binary": str(experiment.storage.rclone_binary),
            "spool_root": f"/dev/shm/xai-simple/{_component_identifier(experiment, suffix='assumptions', scope=scope)}",
            "spool_max_gib": 160,
            "spool_min_free_gib": 24,
        },
        "runtime": {
            "database_path": str(run_root / "component-databases" / "assumptions.sqlite3"),
            "log_directory": str(run_root / "assumptions" / "logs"),
            "gpu_ids": list(experiment.runtime.gpu_ids),
            "inference_batch_size": 512,
            "headroom_fraction": experiment.runtime.headroom_fraction,
            "max_retries": experiment.runtime.max_retries,
            "cpu_workers": experiment.runtime.max_cpu_jobs,
            "training_reservation_gib": {"cnn": 44, "vit": 44},
            "phase1_reservation_gib": {"cnn": 44, "vit": 44},
            "selection_reservation_gib": 12,
            "rank_reservation_gib": 12,
            "evaluation_reservation_gib": 10,
            "phase1_batch_caps": {
                "IntegratedGradients": 32,
                "FeatureAblation": 1024,
                "Occlusion": 1024,
                "GradientShap": 64,
                "DeepLiftShap": 64,
                "PartialLRP": 128,
                "FullLRP": 128,
            },
        },
        "training": {
            "train_split": "train",
            "validation_split": "validation",
            "epochs": experiment.training.epochs,
            "batch_size": {
                "cnn": experiment.training.batch_size["cnn"],
                "vit": experiment.training.batch_size["vit"],
            },
            "validation_batch_size": {
                "cnn": experiment.training.validation_batch_size["cnn"],
                "vit": experiment.training.validation_batch_size["vit"],
            },
            "precision": experiment.training.precision,
            "class_balance": class_balance,
        },
        "selection": {
            "alpha": 0.05,
            "bootstrap_replicates": 499,
            "min_prefix": 2,
            "selection_rule": "largest_not_rejected",
            "spearman_artifact": str(run_root / "unused" / "spearman-p196.npz"),
            "spearman_calibration_config": str(
                experiment.source_path.parents[2] / "configs" / "pilots" / "noise_gof_cost.yaml"
            ),
        },
        "science": {
            "settings": ["ind", "matched-naive"],
            "split": "test",
            "patch_size": 16,
            "k": 20,
            "assignment_seed": experiment.seed,
            "matched_source_selection": "all",
            "partition_families": 3,
        },
    }


def _prefix_config(
    experiment: FullMatrixExperiment,
    active_manifest: Mapping[str, Any],
    *,
    paths: ExecutionPaths | None = None,
    scope: ExecutionScope | None = None,
) -> Mapping[str, Any]:
    paths = paths or execution_paths(experiment, scope)
    run_root = paths.work_root
    return {
        "schema_version": 1,
        "sweep_id": _component_identifier(experiment, suffix="noise-prefix", scope=scope),
        "assumptions_config": str(paths.assumptions_config_path),
        "full_matrix": _full_matrix_metadata(
            experiment,
            active_manifest,
            scope=scope,
            include_release_scope=False,
        ),
        "storage": {
            "remote_root": _scoped_remote_root(experiment.storage.prefix_remote_root, scope),
            "scratch_root": str(run_root / "noise-prefix" / "scratch"),
            "rclone_binary": str(experiment.storage.rclone_binary),
            "spool_root": f"/dev/shm/xai-simple/{_component_identifier(experiment, suffix='prefix', scope=scope)}",
            "spool_max_gib": 160,
            "spool_min_free_gib": 24,
        },
        "runtime": {
            "database_path": str(run_root / "component-databases" / "prefix.sqlite3"),
            "log_directory": str(run_root / "noise-prefix" / "logs"),
            "input_catalog_path": str(run_root / "noise-prefix" / "input-catalog.json"),
            "shared_cache_root": f"/dev/shm/xai-simple/{_component_identifier(experiment, suffix='shared', scope=scope)}",
            "gpu_ids": list(experiment.runtime.gpu_ids),
            "inference_batch_size": 512,
            "evaluation_reservation_gib": 12,
            "aggregation_workspace_gib": 2,
            "headroom_fraction": experiment.runtime.headroom_fraction,
            "max_retries": experiment.runtime.max_retries,
            "cpu_workers": experiment.runtime.max_cpu_jobs,
        },
        "science": {
            "split": "test",
            "patch_size": 16,
            "k": 20,
            "q_values": list(range(2, 12)),
            "rules": ["SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"],
            "selection_input_mode": "independent_geometry_deferred",
        },
    }


def materialize_component_configs(
    experiment: FullMatrixExperiment,
    *,
    scope: ExecutionScope | None = None,
) -> Mapping[str, Any]:
    """Materialize either the original full matrix or a frozen partial scope."""

    paths = execution_paths(experiment, scope)
    if scope is None:
        active_manifest = resolve_active_cells(experiment)
        active_cells = load_active_cells(experiment)
    else:
        write_execution_scope(experiment, scope)
        active_manifest = resolve_scoped_active_cells(experiment, scope)
        active_cells = load_scoped_active_cells(experiment, scope)
    if not active_cells:
        raise RuntimeError("No cell passed the strict 11-method compatibility gate")
    paths.generated_root.mkdir(parents=True, exist_ok=True)
    _yaml(
        paths.base_config_path,
        _base_config(experiment, active_cells, active_manifest, paths=paths, scope=scope),
    )
    _yaml(
        paths.assumptions_config_path,
        _assumptions_config(experiment, active_manifest, paths=paths, scope=scope),
    )
    _yaml(
        paths.prefix_config_path,
        _prefix_config(experiment, active_manifest, paths=paths, scope=scope),
    )

    base = load_experiment(paths.base_config_path)
    assumptions = load_assumption_experiment(paths.assumptions_config_path)
    prefix = load_noise_prefix_experiment(paths.prefix_config_path)
    expected_cells = len(active_cells)
    if (
        len(base.models) != expected_cells
        or len(assumptions.cells()) != expected_cells
        or len(prefix.cells()) != expected_cells
    ):
        raise RuntimeError("Generated component configs disagree on active-cell coverage")
    result: dict[str, Any] = {
        "status": "complete",
        "active_cells": expected_cells,
        "blocked_cells": active_manifest["counts"]["blocked_cells"],
        "active_manifest": str(paths.active_cells_path),
        "active_manifest_digest": active_manifest["active_manifest_digest"],
        "base_config": str(paths.base_config_path),
        "base_digest": base.digest,
        "assumptions_config": str(paths.assumptions_config_path),
        "assumptions_digest": assumptions.digest,
        "prefix_config": str(paths.prefix_config_path),
        "prefix_digest": prefix.digest,
    }
    if scope is not None:
        result["execution_scope"] = _scope_metadata(scope)
        assert paths.scope_manifest_path is not None
        result["scope_manifest"] = str(paths.scope_manifest_path)
    return result


class JobCollector:
    """Small in-memory SimpleJobStore interface used by component planners."""

    def __init__(self, existing: Sequence[QueueJob] = ()) -> None:
        self._jobs = {job.job_id: job for job in existing}

    def submit(self, job: QueueJob) -> None:
        observed = self._jobs.get(job.job_id)
        if observed is not None and observed != job:
            raise ValueError(f"Collected job identity collision: {job.job_id}")
        self._jobs[job.job_id] = job

    def submit_many(self, jobs: Sequence[QueueJob]) -> Mapping[str, int]:
        before = len(self._jobs)
        for job in jobs:
            self.submit(job)
        return {
            "submitted": len(self._jobs) - before,
            "already_present": len(jobs) - (len(self._jobs) - before),
        }

    def update_reservation(self, job_id: str, reservation_bytes: int) -> None:
        self._jobs[job_id] = replace(self._jobs[job_id], reservation_bytes=reservation_bytes)

    def update_reservations(self, reservations: Mapping[str, int]) -> None:
        for job_id, reservation in reservations.items():
            self.update_reservation(job_id, reservation)

    def jobs(self, *, status: str | None = None) -> tuple[QueueJob, ...]:
        values = tuple(self._jobs[key] for key in sorted(self._jobs))
        return values if status is None else tuple(job for job in values if job.status == status)


def with_dependency(job: QueueJob, dependency: str) -> QueueJob:
    if dependency in job.dependencies:
        return job
    return replace(job, dependencies=(*job.dependencies, dependency))


__all__ = [
    "EXECUTION_SCOPE_SCHEMA",
    "ExecutionPaths",
    "ExecutionScope",
    "JobCollector",
    "build_execution_scope",
    "execution_paths",
    "load_active_cells",
    "load_execution_scope",
    "load_scoped_active_cells",
    "materialize_component_configs",
    "resolve_active_cells",
    "resolve_scoped_active_cells",
    "with_dependency",
    "write_execution_scope",
]
