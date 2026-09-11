"""CLI for the formal NOISE prefix-size sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json

from ..scheduler import SimpleJobStore
from .config import load_noise_prefix_experiment


def _load(args: argparse.Namespace) -> Any:
    return load_noise_prefix_experiment(args.config)


def _validate(args: argparse.Namespace) -> int:
    from ..cli import _validate_assets

    experiment = _load(args)
    value = {
        "status": "valid",
        "sweep_id": experiment.sweep_id,
        "digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "assumption_id": experiment.assumptions.assumption_id,
        "assumption_digest": experiment.assumptions.digest,
        "assets": _validate_assets(experiment.base),
        "science": dict(experiment.raw_science),
        "counts": {
            "cells": len(experiment.cells()),
            "conditions": len(experiment.base.conditions),
            "evaluation_tasks": len(experiment.evaluation_tasks()),
            "q_values": len(experiment.q_values),
            "computed_rule_bank": (len(experiment.q_values) - 1) * len(experiment.rules),
            "q11_references": len(experiment.rules),
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _readiness(args: argparse.Namespace) -> int:
    from .inputs import materialize_missing_rank_ready, readiness_report

    experiment = _load(args)
    preparation = materialize_missing_rank_ready(experiment) if args.materialize_missing else None
    value = readiness_report(
        experiment,
        refresh=args.refresh or args.materialize_missing,
    )
    if preparation is not None:
        value = {**value, "rank_ready_preparation": preparation}
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _plan_value(experiment: Any) -> dict[str, Any]:
    clean_jobs = {
        task.cell.cell_id: task.task_id
        for task in experiment.evaluation_tasks()
        if task.condition.kind == "clean"
    }
    return {
        "schema_version": 1,
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "fidelity_order": {
            "metric": "F",
            "scope": "complete_test_set",
            "direction": "descending",
            "tie_break": "method_id_ascending",
        },
        "q_values": list(experiment.q_values),
        "rules": list(experiment.rules),
        "q11_policy": "exact_immutable_naive_p16_reference",
        "evaluations": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "condition": task.condition.condition_id,
                "dependency": (
                    None if task.condition.kind == "clean" else clean_jobs[task.cell.cell_id]
                ),
                "artifact_root": task.artifact_root,
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


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluator import run_evaluation_task

    experiment = _load(args)
    value = run_evaluation_task(
        experiment,
        experiment.find_evaluation_task(args.task_id),
        device=args.device,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    from .scheduler import run_scheduler

    value = run_scheduler(
        _load(args),
        poll_seconds=args.poll_seconds,
        headroom_fraction=args.headroom_fraction,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _status(args: argparse.Namespace) -> int:
    from .scheduler import scheduler_status

    print(json.dumps(scheduler_status(_load(args)), indent=2, sort_keys=True))
    return 0


def _retry(args: argparse.Namespace) -> int:
    experiment = _load(args)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    value = store.retry_failed(tuple(args.job_id))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from .summary import write_summary

    value = write_summary(_load(args), output_directory=args.output_directory)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def register_subcommands(commands: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    validate = commands.add_parser("validate", parents=[common])
    validate.set_defaults(handler=_validate)
    readiness = commands.add_parser("readiness", parents=[common])
    readiness.add_argument("--refresh", action="store_true")
    readiness.add_argument(
        "--materialize-missing",
        action="store_true",
        help="explicitly derive any missing legacy rank-ready sidecars before validation",
    )
    readiness.set_defaults(handler=_readiness)
    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)
    evaluate = commands.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--task-id", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.set_defaults(handler=_evaluate)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.add_argument(
        "--headroom-fraction",
        type=float,
        help="Execution-only GPU admission override; artifact identity is unchanged",
    )
    run.set_defaults(handler=_run)
    status = commands.add_parser("status", parents=[common])
    status.set_defaults(handler=_status)
    retry = commands.add_parser("retry", parents=[common])
    retry.add_argument("--job-id", action="append", required=True)
    retry.set_defaults(handler=_retry)
    summarize = commands.add_parser("summarize", parents=[common])
    summarize.add_argument("--output-directory")
    summarize.set_defaults(handler=_summarize)


__all__ = ["register_subcommands"]
