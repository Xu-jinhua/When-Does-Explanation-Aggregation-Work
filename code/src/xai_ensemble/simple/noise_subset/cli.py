"""CLI for the noise-consistent random-subset control."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json

from ..noise_prefix.inputs import load_input_catalog
from ..scheduler import SimpleJobStore
from .config import load_noise_subset_experiment


def _load(args: argparse.Namespace) -> Any:
    return load_noise_subset_experiment(args.config)


def _validate(args: argparse.Namespace) -> int:
    from ..cli import _validate_assets

    experiment = _load(args)
    value = {
        "status": "valid",
        "study_id": experiment.study_id,
        "digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "prefix_sweep_id": experiment.prefix.sweep_id,
        "prefix_digest": experiment.prefix.digest,
        "selector_digest": experiment.selector_digest,
        "assets": _validate_assets(experiment.base),
        "science": dict(experiment.raw_science),
        "counts": {
            "cells": len(experiment.cells()),
            "conditions": len(experiment.conditions()),
            "selection_tasks": len(experiment.selection_tasks()),
            "evaluation_tasks": len(experiment.evaluation_tasks()),
            "planned_jobs": len(experiment.selection_tasks()) + len(experiment.evaluation_tasks()),
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _readiness(args: argparse.Namespace) -> int:
    experiment = _load(args)
    catalog = load_input_catalog(experiment.prefix)
    required_prefix_tasks = {
        experiment.prefix_task(cell, condition.condition_id).task_id
        for cell in experiment.cells()
        for condition in experiment.conditions()
    }
    missing = sorted(required_prefix_tasks.difference(catalog["tasks"]))
    if missing:
        raise FileNotFoundError(f"prefix input catalog lacks required tasks: {missing}")
    value = {
        "ready": True,
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "input_catalog": str(experiment.prefix.runtime.input_catalog_path),
        "input_catalog_digest": catalog["catalog_digest"],
        "selector": str(experiment.selector_path),
        "selector_digest": experiment.selector_digest,
        "required_prefix_tasks": sorted(required_prefix_tasks),
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _plan_value(experiment: Any) -> Mapping[str, Any]:
    selections = {task.task_id: task for task in experiment.selection_tasks()}
    return {
        "schema_version": 1,
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "science": dict(experiment.raw_science),
        "selections": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "artifact_root": task.artifact_root,
            }
            for task in experiment.selection_tasks()
        ],
        "evaluations": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell.cell_id,
                "condition": task.condition.condition_id,
                "geometry": task.geometry,
                "selection_dependency": selections[task.selection_task_id].task_id,
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


def _select(args: argparse.Namespace) -> int:
    from .selection import run_selection_task

    experiment = _load(args)
    value = run_selection_task(
        experiment,
        experiment.find_selection_task(args.task_id),
        device=args.device,
    )
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
    readiness.set_defaults(handler=_readiness)
    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)
    select = commands.add_parser("select", parents=[common])
    select.add_argument("--task-id", required=True)
    select.add_argument("--device", default="cuda:0")
    select.set_defaults(handler=_select)
    evaluate = commands.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--task-id", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.set_defaults(handler=_evaluate)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.add_argument("--headroom-fraction", type=float)
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
