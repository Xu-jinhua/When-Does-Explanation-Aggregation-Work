"""CLI for the isolated IND, matched-NAIVE, and Oracle NOISE DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json

from .config import load_assumption_experiment


def _load(args: argparse.Namespace) -> Any:
    return load_assumption_experiment(args.config)


def _validate(args: argparse.Namespace) -> int:
    experiment = _load(args)
    from ..cli import _validate_assets
    from .readiness import assumption_readiness

    assets = _validate_assets(experiment.base)
    readiness = assumption_readiness(experiment)
    if not experiment.selection.spearman_calibration_config.is_file():
        raise FileNotFoundError(experiment.selection.spearman_calibration_config)
    value = {
        "status": "valid",
        "assumption_id": experiment.assumption_id,
        "digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "execution_ready": readiness["ready"],
        "base_inputs": readiness["base_inputs"],
        "runtime_dependencies": readiness["runtime_dependencies"],
        "assets": assets,
        "science": {
            "settings": list(experiment.settings),
            "split": experiment.split,
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "fill": "dataset_mean",
            "target_policy": "full_reference_clean_fp32_prediction",
            "noise_selection_scope": "complete_test_set_in_sample",
            "noise_label": "Oracle NOISE",
        },
        "counts": {
            "cells": len(experiment.cells()),
            "partitions": len(experiment.partition_tasks()),
            "source_models": len(experiment.training_tasks()),
            "source_phase1_scopes": len(experiment.source_phase1_tasks()),
            "source_method_artifacts": sum(
                len(task.cell.methods) for task in experiment.source_phase1_tasks()
            ),
            "selections": len(experiment.selection_tasks()),
            "ranks": len(experiment.rank_tasks()),
            "evaluations": len(experiment.evaluation_tasks()),
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _readiness(args: argparse.Namespace) -> int:
    from .readiness import assumption_readiness

    value = assumption_readiness(_load(args))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["ready"] else 1


def _plan_value(experiment: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "settings": list(experiment.settings),
        "method_assignments": {
            cell.cell_id: [list(pair) for pair in experiment.method_assignment(cell)]
            for cell in experiment.cells()
        },
        "partitions": [
            {"task_id": task.task_id, "digest": task.digest, "cell": task.cell.cell_id}
            for task in experiment.partition_tasks()
        ],
        "training": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "source_id": task.source_id,
            }
            for task in experiment.training_tasks()
        ],
        "source_phase1": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "source_id": task.source_id,
                "condition": task.condition.condition_id,
                "methods": list(task.cell.methods),
            }
            for task in experiment.source_phase1_tasks()
        ],
        "selection": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "distance_model": task.distance_model,
                "aggregation": task.aggregation,
            }
            for task in experiment.selection_tasks()
        ],
        "rank": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "setting": task.setting,
                "source_id": task.source_id,
                "distance_model": task.distance_model,
                "condition": task.condition.condition_id,
            }
            for task in experiment.rank_tasks()
        ],
        "evaluation": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "rank_task_id": task.rank_task_id,
                "cell": task.cell.cell_id,
                "setting": task.setting,
                "condition": task.condition.condition_id,
            }
            for task in experiment.evaluation_tasks()
        ],
    }


def _plan(args: argparse.Namespace) -> int:
    value = _plan_value(_load(args))
    if args.output:
        atomic_write_json(args.output, value)
        print(f"WROTE {Path(args.output).resolve()}")
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    experiment = _load(args)
    if args.task_id == "spearman-p196":
        from .prepare import prepare_spearman_family

        value = prepare_spearman_family(experiment)
    else:
        from .prepare import run_partition_task

        value = run_partition_task(experiment, experiment.find_partition_task(args.task_id))
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0


def _train(args: argparse.Namespace) -> int:
    from .training import run_training_task

    experiment = _load(args)
    value = run_training_task(
        experiment, experiment.find_training_task(args.task_id), device=args.device
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _phase1(args: argparse.Namespace) -> int:
    from .phase1 import run_source_phase1_task

    experiment = _load(args)
    value = run_source_phase1_task(
        experiment, experiment.find_source_phase1_task(args.task_id), device=args.device
    )
    print(json.dumps(value[-1], indent=2, sort_keys=True))
    return 0


def _phase1_method(args: argparse.Namespace) -> int:
    from .phase1 import run_source_phase1_method_task

    experiment = _load(args)
    value = run_source_phase1_method_task(
        experiment,
        experiment.find_source_phase1_task(args.task_id),
        args.family,
        device=args.device,
    )
    print(json.dumps(value[-1], indent=2, sort_keys=True))
    return 0


def _select(args: argparse.Namespace) -> int:
    from .selection import run_selection_task

    experiment = _load(args)
    value = run_selection_task(
        experiment, experiment.find_selection_task(args.task_id), device=args.device
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _rank(args: argparse.Namespace) -> int:
    from .ranks import run_rank_task

    experiment = _load(args)
    value = run_rank_task(experiment, experiment.find_rank_task(args.task_id), device=args.device)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluator import run_evaluation_task

    experiment = _load(args)
    value = run_evaluation_task(
        experiment, experiment.find_evaluation_task(args.task_id), device=args.device
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    from .scheduler import run_scheduler

    counts = run_scheduler(
        _load(args),
        poll_seconds=args.poll_seconds,
        headroom_fraction=args.headroom_fraction,
    )
    print(json.dumps({"status": "terminal", "counts": counts}, indent=2, sort_keys=True))
    return 0 if not counts.get("failed") and not counts.get("blocked") else 1


def _status(args: argparse.Namespace) -> int:
    from .scheduler import scheduler_status

    print(json.dumps(scheduler_status(_load(args)), indent=2, sort_keys=True))
    return 0


def _retry(args: argparse.Namespace) -> int:
    from ..scheduler import SimpleJobStore
    from .scheduler import scheduler_status

    experiment = _load(args)
    store = SimpleJobStore(
        experiment.runtime.database_path, experiment_digest=experiment.scheduler_digest
    )
    store.recover_orphans()
    value = dict(store.retry_failed(tuple(args.job_id)))
    value["status"] = scheduler_status(experiment)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from .summary import write_noise_summary, write_summary

    writer = write_noise_summary if args.scope == "noise" else write_summary
    value = writer(_load(args), output_directory=args.output_directory)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _table_priority_run(args: argparse.Namespace) -> int:
    from .table_priority import run_scheduler

    counts = run_scheduler(
        _load(args),
        root=args.root,
        poll_seconds=args.poll_seconds,
        headroom_fraction=args.headroom_fraction,
    )
    print(json.dumps({"status": "terminal", "counts": counts}, indent=2, sort_keys=True))
    return 0 if not counts.get("failed") and not counts.get("blocked") else 1


def _table_priority_status(args: argparse.Namespace) -> int:
    from .table_priority import scheduler_status

    print(json.dumps(scheduler_status(_load(args), root=args.root), indent=2, sort_keys=True))
    return 0


def _table_priority_retry(args: argparse.Namespace) -> int:
    from .table_priority import retry_failed

    value = retry_failed(_load(args), tuple(args.job_id), root=args.root)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _table_priority_summarize(args: argparse.Namespace) -> int:
    from .ind_table_summary import write_ind_table_priority_summary

    value = write_ind_table_priority_summary(_load(args), output_directory=args.output_directory)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def register_subcommands(commands: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    validate = commands.add_parser("validate", parents=[common])
    validate.set_defaults(handler=_validate)
    readiness = commands.add_parser("readiness", parents=[common])
    readiness.set_defaults(handler=_readiness)
    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)
    prepare = commands.add_parser("prepare", parents=[common])
    prepare.add_argument("--task-id", required=True)
    prepare.set_defaults(handler=_prepare)
    for name, handler in (
        ("train", _train),
        ("phase1", _phase1),
        ("select", _select),
        ("rank", _rank),
        ("evaluate", _evaluate),
    ):
        parser = commands.add_parser(name, parents=[common])
        parser.add_argument("--task-id", required=True)
        parser.add_argument("--device", default="cuda:0")
        parser.set_defaults(handler=handler)
    phase1_method = commands.add_parser("phase1-method", parents=[common])
    phase1_method.add_argument("--task-id", required=True)
    phase1_method.add_argument("--family", required=True)
    phase1_method.add_argument("--device", default="cuda:0")
    phase1_method.set_defaults(handler=_phase1_method)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.add_argument(
        "--headroom-fraction",
        type=float,
        help=(
            "Execution-only GPU admission headroom override; this does not change "
            "experiment or artifact identities"
        ),
    )
    run.set_defaults(handler=_run)
    status = commands.add_parser("status", parents=[common])
    status.set_defaults(handler=_status)
    retry = commands.add_parser("retry", parents=[common])
    retry.add_argument("--job-id", action="append", required=True)
    retry.set_defaults(handler=_retry)
    summarize = commands.add_parser("summarize", parents=[common])
    summarize.add_argument("--output-directory")
    summarize.add_argument("--scope", choices=("all", "noise"), default="all")
    summarize.set_defaults(handler=_summarize)
    table_run = commands.add_parser("table-priority-run", parents=[common])
    table_run.add_argument("--root")
    table_run.add_argument("--poll-seconds", type=float, default=2.0)
    table_run.add_argument("--headroom-fraction", type=float, default=0.05)
    table_run.set_defaults(handler=_table_priority_run)
    table_status = commands.add_parser("table-priority-status", parents=[common])
    table_status.add_argument("--root")
    table_status.set_defaults(handler=_table_priority_status)
    table_retry = commands.add_parser("table-priority-retry", parents=[common])
    table_retry.add_argument("--root")
    table_retry.add_argument("--job-id", action="append", required=True)
    table_retry.set_defaults(handler=_table_priority_retry)
    table_summary = commands.add_parser("table-priority-summarize", parents=[common])
    table_summary.add_argument("--output-directory")
    table_summary.set_defaults(handler=_table_priority_summarize)


__all__ = ["register_subcommands"]
