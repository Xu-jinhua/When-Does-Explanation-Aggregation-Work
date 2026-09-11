"""Command-line entry points for the complete GitHub result matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json, read_json

from .assets import (
    compatibility_variants,
    planned_asset_jobs,
    prepare_compatibility_samples,
    run_compatibility_gate,
    static_compatibility_failures,
    write_static_compatibility_gate,
)
from .catalog import validate_method_rosters
from .config import FullMatrixExperiment, load_full_matrix_experiment
from .planner import (
    ExecutionScope,
    build_execution_scope,
    execution_paths,
    load_execution_scope,
    materialize_component_configs,
)


def _load(args: argparse.Namespace) -> FullMatrixExperiment:
    return load_full_matrix_experiment(args.config)


def _loaded_scope(
    args: argparse.Namespace,
    experiment: FullMatrixExperiment,
) -> ExecutionScope | None:
    scope_manifest = getattr(args, "scope_manifest", None)
    return None if scope_manifest is None else load_execution_scope(experiment, scope_manifest)


def _materialization_scope(
    args: argparse.Namespace,
    experiment: FullMatrixExperiment,
) -> ExecutionScope | None:
    cells = tuple(getattr(args, "scope_cell", None) or ())
    datasets = tuple(getattr(args, "deferred_dataset", None) or ())
    digest = getattr(args, "scope_digest", None)
    if not cells and not datasets and digest is None:
        return None
    if not cells or not datasets or not isinstance(digest, str):
        raise ValueError(
            "Scoped materialization requires --scope-digest, --deferred-dataset, and --scope-cell"
        )
    return build_execution_scope(
        experiment,
        deferred_datasets=datasets,
        cell_ids=cells,
        expected_digest=digest,
    )


def _print(value: Mapping[str, Any]) -> int:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0


def _validate(args: argparse.Namespace) -> int:
    experiment = _load(args)
    from xai_ensemble.simple.runtime_dependencies import require_relprop_runtime

    rosters = validate_method_rosters(experiment.methods)
    runtime_dependencies = require_relprop_runtime(
        (method.family, architecture)
        for architecture in ("cnn", "vit")
        for method in experiment.methods.for_architecture(architecture)
    )
    static_blocks: dict[str, dict[str, Any]] = {}
    for cell in experiment.cells():
        failures = static_compatibility_failures(experiment, cell)
        if failures:
            row = static_blocks.setdefault(
                cell.model_key,
                {"cells": 0, "failures": list(failures)},
            )
            if row["failures"] != list(failures):
                raise RuntimeError(
                    "Static provider support differs across cells with the same model key"
                )
            row["cells"] += 1
    return _print(
        {
            "status": "valid",
            "schema": "simple-full-matrix-v1",
            "experiment_id": experiment.experiment_id,
            "digest": experiment.digest,
            "scheduler_digest": experiment.scheduler_digest,
            "protocol": str(experiment.protocol_path),
            "protocol_digest": experiment.protocol_digest,
            "cells": len(experiment.cells()),
            "datasets": list(experiment.dataset_ids),
            "models": list(experiment.model_keys),
            "method_rosters": {key: list(value) for key, value in rosters.items()},
            "runtime_dependencies": {"relprop": runtime_dependencies},
            "static_provider_blocks": static_blocks,
            "static_provider_blocked_cells": sum(
                int(row["cells"]) for row in static_blocks.values()
            ),
            "runtime": {
                "gpu_ids": list(experiment.runtime.gpu_ids),
                "headroom_fraction": experiment.runtime.headroom_fraction,
                "max_cpu_jobs": experiment.runtime.max_cpu_jobs,
                "max_retries": experiment.runtime.max_retries,
            },
            "frozen_primary_evaluation": {
                "split": "test",
                "patch_size": 16,
                "k": 20,
                "fill": "dataset_mean",
                "conditions": [
                    "clean",
                    "gaussian-0.15",
                    "salt-pepper-0.05",
                    "speckle-0.15",
                    "adversarial-sara-2-255",
                ],
            },
        }
    )


def _projected_counts(experiment: FullMatrixExperiment) -> Mapping[str, Any]:
    # Planning reports the frozen DAG cardinality; completion validation is a
    # scheduler-start concern and may require many mounted-artifact reads.
    assets = planned_asset_jobs(experiment, check_complete=False)
    asset_counts = Counter(job.kind for job in assets)
    conditions = 5
    method_families = 11
    active_cells = tuple(
        cell for cell in experiment.cells() if not static_compatibility_failures(experiment, cell)
    )
    cells = len(active_cells)
    profile_variants = sum(len(compatibility_variants(experiment, cell)) for cell in active_cells)
    return {
        "asset_jobs": len(assets),
        "asset_by_kind": dict(sorted(asset_counts.items())),
        "cells": {
            "declared": len(experiment.cells()),
            "static_provider_blocked": len(experiment.cells()) - cells,
            "eligible_for_real_checkpoint_gate": cells,
        },
        "base": {
            "adversarial_datasets": cells,
            "phase1_method_tasks": cells * conditions * method_families,
            "phase1_patch_sizes": [16],
            "phase2_ensemble_tasks": cells * conditions,
            "profile_variants": profile_variants,
        },
        "noise_prefix": {
            "q_values": list(range(2, 12)),
            "evaluation_tasks": cells * conditions,
            "selector_partials": cells,
            "independent_geometries": 2,
        },
        "full_ind": {
            "partition_families": 3,
            "source_models": cells * 3 * 11,
            "source_method_pairs_per_family_condition": 121,
            "source_method_pairs_per_condition": 3 * 121,
            "partition_jobs": cells * 3,
            "source_method_jobs": cells * conditions * 3 * 121,
            "rank_jobs": cells * conditions * 3 * 12,
            "evaluation_jobs": cells * conditions * 3 * 12,
            "matched_naive_full_sources_per_family": 11,
            "matched_naive_full_sources_per_cell": 33,
        },
    }


def _plan(args: argparse.Namespace) -> int:
    experiment = _load(args)
    value: dict[str, Any] = {
        "schema": "simple-full-matrix-plan-v1",
        "experiment_id": experiment.experiment_id,
        "matrix_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "dynamic_order": [
            "phase0-assets-and-compatibility",
            "materialize-active-cell-configs",
            "naive-phase1-and-phase2",
            "rank-ready-preparation",
            "noise-prefix-q-sweep",
            "independent-geometry-selector",
            "selected-noise-summary",
            "full-ind-and-matched-naive",
            "coverage",
        ],
        "projected": _projected_counts(experiment),
        "run_gate": "simple full-matrix run requires --confirm-full-matrix",
    }
    if experiment.active_cells_path.is_file():
        active = read_json(experiment.active_cells_path)
        value["active_cells"] = active.get("counts", {})
    if args.output:
        atomic_write_json(args.output, value)
        print(f"WROTE {Path(args.output).resolve()}")
        return 0
    return _print(value)


def _prepare_samples(args: argparse.Namespace) -> int:
    value = prepare_compatibility_samples(_load(args), dataset_id=args.dataset)
    return _print(value)


def _train_reference(args: argparse.Namespace) -> int:
    from .assets import train_reference_model

    experiment = _load(args)
    project_root = args.project_root or str(experiment.source_path.parents[3])
    return _print(
        train_reference_model(
            experiment,
            cell_id=args.cell,
            project_root=project_root,
            device=args.device,
        )
    )


def _compatibility(args: argparse.Namespace) -> int:
    value = run_compatibility_gate(_load(args), cell_id=args.cell, device=args.device)
    # A blocked cell is a valid gate result.  The active-cell manifest records
    # it and prevents silent roster reduction without aborting other cells.
    return _print(value)


def _static_compatibility(args: argparse.Namespace) -> int:
    return _print(
        write_static_compatibility_gate(
            _load(args),
            cell_id=args.cell,
        )
    )


def _materialize_config(args: argparse.Namespace) -> int:
    experiment = _load(args)
    return _print(
        materialize_component_configs(experiment, scope=_materialization_scope(args, experiment))
    )


def _prepare_prefix_inputs(args: argparse.Namespace) -> int:
    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
    from xai_ensemble.simple.noise_prefix.inputs import (
        materialize_missing_rank_ready,
        readiness_report,
    )

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    prefix = load_noise_prefix_experiment(paths.prefix_config_path)
    result = materialize_missing_rank_ready(prefix)
    # The full-matrix scheduler launches evaluator jobs directly rather than
    # through the standalone prefix scheduler. Build its immutable input
    # catalog here so a successful readiness marker is a complete gate.
    readiness = readiness_report(prefix)
    payload = {
        "schema": "simple-full-matrix-prefix-ready-v1",
        "matrix_digest": experiment.digest,
        "prefix_digest": prefix.digest,
        "status": "complete",
        "result": result,
        "input_catalog": {
            "path": readiness["catalog_path"],
            "catalog_digest": readiness["catalog_digest"],
            "tasks": readiness["tasks"],
        },
    }
    if scope is not None:
        payload["execution_scope"] = {
            "scope_id": scope.scope_id,
            "scope_digest": scope.scope_digest,
            "deferred_datasets": list(scope.deferred_datasets),
        }
    payload["path"] = str(
        atomic_write_json(paths.result_root / "control" / "prefix-ready.json", payload)
    )
    return _print(payload)


def _selector_cell(args: argparse.Namespace) -> int:
    from .selector import materialize_selector_cell

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    return _print(
        materialize_selector_cell(
            experiment,
            prefix_config=paths.prefix_config_path,
            cell_id=args.cell,
            result_root=paths.result_root,
        )
    )


def _merge_selector(args: argparse.Namespace) -> int:
    from .selector import merge_selector_partials

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    return _print(
        merge_selector_partials(
            experiment,
            prefix_config=paths.prefix_config_path,
            result_root=paths.result_root,
        )
    )


def _summarize_base(args: argparse.Namespace) -> int:
    from xai_ensemble.simple.config import load_experiment
    from xai_ensemble.simple.summary import write_table1_summary

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    result = write_table1_summary(
        load_experiment(paths.base_config_path),
        output_directory=paths.result_root / "base",
    )
    payload = {
        "schema": "simple-full-matrix-base-summary-v1",
        "matrix_digest": experiment.digest,
        "status": "complete",
        "summary_path": result["summary_json"],
        "summary_digest": result["summary_digest"],
        "result": result,
    }
    if scope is not None:
        payload["execution_scope"] = {
            "scope_id": scope.scope_id,
            "scope_digest": scope.scope_digest,
            "deferred_datasets": list(scope.deferred_datasets),
        }
    atomic_write_json(paths.result_root / "base" / "complete.json", payload)
    return _print(payload)


def _summarize_prefix(args: argparse.Namespace) -> int:
    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
    from xai_ensemble.simple.noise_prefix.summary import write_summary

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    return _print(
        write_summary(
            load_noise_prefix_experiment(paths.prefix_config_path),
            output_directory=paths.result_root / "noise-prefix" / "summary",
        )
    )


def _summarize_selected_noise(args: argparse.Namespace) -> int:
    from .results import write_selected_noise_summary

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    return _print(
        write_selected_noise_summary(
            prefix_config=paths.prefix_config_path,
            selector_path=paths.result_root / "noise-prefix" / "selector" / "selector.json",
            q_summary_path=paths.result_root / "noise-prefix" / "summary",
            output_directory=paths.result_root / "noise-prefix" / "selected",
        )
    )


def _summarize_ind(args: argparse.Namespace) -> int:
    from xai_ensemble.simple.assumptions.config import load_assumption_experiment
    from xai_ensemble.simple.assumptions.ind_table_summary import write_ind_table_priority_summary

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    return _print(
        write_ind_table_priority_summary(
            load_assumption_experiment(paths.assumptions_config_path),
            output_directory=paths.result_root / "ind" / "summary",
        )
    )


def _barrier(args: argparse.Namespace) -> int:
    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    stage = args.stage_option or args.stage_positional
    if stage is None:
        raise ValueError("barrier requires a stage via --stage or positional argument")
    if args.stage_option is not None and args.stage_positional is not None:
        if args.stage_option != args.stage_positional:
            raise ValueError("barrier stage arguments disagree")
    payload = {
        "schema": "simple-full-matrix-barrier-v1",
        "matrix_digest": experiment.digest,
        "stage": stage,
        "status": "complete",
    }
    if scope is not None:
        payload["execution_scope"] = {
            "scope_id": scope.scope_id,
            "scope_digest": scope.scope_digest,
            "deferred_datasets": list(scope.deferred_datasets),
        }
    atomic_write_json(paths.result_root / "control" / f"{stage}.json", payload)
    return _print(payload)


def _coverage(args: argparse.Namespace) -> int:
    from .results import write_matrix_coverage

    experiment = _load(args)
    scope = _loaded_scope(args, experiment)
    paths = execution_paths(experiment, scope)
    base = read_json(paths.result_root / "base" / "complete.json")
    if not isinstance(base, Mapping) or not isinstance(base.get("summary_path"), str):
        raise ValueError("The base summary completion marker is invalid")
    return _print(
        write_matrix_coverage(
            experiment_id=experiment.experiment_id,
            matrix_digest=experiment.digest,
            active_manifest_path=paths.active_cells_path,
            component_summaries={
                "naive": base["summary_path"],
                "noise_prefix": paths.result_root / "noise-prefix" / "summary",
                "selected_noise": paths.result_root / "noise-prefix" / "selected",
                "full_ind": paths.result_root / "ind" / "summary",
            },
            output_directory=paths.result_root / "coverage",
            execution_scope=(
                None
                if scope is None
                else {
                    "scope_id": scope.scope_id,
                    "scope_digest": scope.scope_digest,
                    "deferred_datasets": list(scope.deferred_datasets),
                }
            ),
        )
    )


def _run(args: argparse.Namespace) -> int:
    if not args.confirm_full_matrix:
        raise ValueError("Refusing to start the complete matrix without --confirm-full-matrix")
    from .scheduler import run_scheduler

    counts = run_scheduler(_load(args), poll_seconds=args.poll_seconds)
    _print({"status": "terminal", "counts": counts})
    return 0 if not counts.get("failed") and not counts.get("blocked") else 1


def _status(args: argparse.Namespace) -> int:
    from .scheduler import scheduler_status

    return _print(scheduler_status(_load(args)))


def _retry(args: argparse.Namespace) -> int:
    from .scheduler import retry_failed

    return _print(retry_failed(_load(args), args.job_id))


def _requeue(args: argparse.Namespace) -> int:
    from .scheduler import requeue_succeeded

    return _print(requeue_succeeded(_load(args), args.job_id, reason=args.reason))


def _defer_datasets(args: argparse.Namespace) -> int:
    from .scheduler import write_deferred_datasets

    return _print(
        write_deferred_datasets(
            _load(args),
            datasets=args.dataset,
            reason=args.reason,
        )
    )


def _resume_datasets(args: argparse.Namespace) -> int:
    from .scheduler import clear_deferred_datasets

    return _print(clear_deferred_datasets(_load(args)))


def register_subcommands(commands: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    scoped = argparse.ArgumentParser(add_help=False)
    scoped.add_argument("--scope-manifest")

    validate = commands.add_parser("validate", parents=[common])
    validate.set_defaults(handler=_validate)
    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)

    prepare_samples = commands.add_parser("prepare-samples", parents=[common])
    prepare_samples.add_argument("--dataset", required=True)
    prepare_samples.set_defaults(handler=_prepare_samples)
    train_reference = commands.add_parser("train-reference", parents=[common])
    train_reference.add_argument("--cell", required=True)
    train_reference.add_argument("--project-root")
    train_reference.add_argument("--device", default="cuda:0")
    train_reference.set_defaults(handler=_train_reference)
    compatibility = commands.add_parser("compatibility", parents=[common])
    compatibility.add_argument("--cell", required=True)
    compatibility.add_argument("--device", default="cuda:0")
    compatibility.set_defaults(handler=_compatibility)
    static_compatibility = commands.add_parser(
        "static-compatibility",
        parents=[common],
        help="Record a known unavailable provider without training that cell",
    )
    static_compatibility.add_argument("--cell", required=True)
    static_compatibility.set_defaults(handler=_static_compatibility)
    materialize = commands.add_parser("materialize-config", parents=[common])
    materialize.add_argument("--scope-digest")
    materialize.add_argument("--deferred-dataset", action="append")
    materialize.add_argument("--scope-cell", action="append")
    materialize.set_defaults(handler=_materialize_config)
    prefix_inputs = commands.add_parser("prepare-prefix-inputs", parents=[common, scoped])
    prefix_inputs.set_defaults(handler=_prepare_prefix_inputs)
    selector_cell = commands.add_parser("selector-cell", parents=[common, scoped])
    selector_cell.add_argument("--cell", required=True)
    selector_cell.set_defaults(handler=_selector_cell)
    merge_selector = commands.add_parser("merge-selector", parents=[common, scoped])
    merge_selector.set_defaults(handler=_merge_selector)
    for name, handler in (
        ("summarize-base", _summarize_base),
        ("summarize-prefix", _summarize_prefix),
        ("summarize-selected-noise", _summarize_selected_noise),
        ("summarize-ind", _summarize_ind),
        ("coverage", _coverage),
    ):
        parser = commands.add_parser(name, parents=[common, scoped])
        parser.set_defaults(handler=handler)
    barrier = commands.add_parser("barrier", parents=[common, scoped])
    # Accept the positional form emitted by older queue snapshots as well as
    # the current explicit option, so retrying an existing job remains safe.
    barrier.add_argument("stage_positional", nargs="?")
    barrier.add_argument("--stage", dest="stage_option")
    barrier.set_defaults(handler=_barrier)

    run = commands.add_parser("run", parents=[common])
    run.add_argument("--confirm-full-matrix", action="store_true")
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.set_defaults(handler=_run)
    status = commands.add_parser("status", parents=[common])
    status.set_defaults(handler=_status)
    retry = commands.add_parser("retry", parents=[common])
    retry.add_argument("--job-id", action="append", required=True)
    retry.set_defaults(handler=_retry)
    requeue = commands.add_parser(
        "requeue",
        parents=[common],
        help="Requeue succeeded jobs so a dataset contract change re-runs them",
    )
    requeue.add_argument("--job-id", action="append", required=True)
    requeue.add_argument("--reason", required=True)
    requeue.set_defaults(handler=_requeue)
    defer = commands.add_parser(
        "defer-datasets",
        parents=[common],
        help="Pause claiming pending jobs for selected datasets without changing SQLite",
    )
    defer.add_argument("--dataset", action="append", required=True)
    defer.add_argument("--reason", required=True)
    defer.set_defaults(handler=_defer_datasets)
    resume = commands.add_parser(
        "resume-datasets",
        parents=[common],
        help="Remove the runtime dataset deferral control",
    )
    resume.set_defaults(handler=_resume_datasets)


__all__ = ["register_subcommands"]
