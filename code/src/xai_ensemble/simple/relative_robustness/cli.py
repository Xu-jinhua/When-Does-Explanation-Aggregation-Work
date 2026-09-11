"""CLI for fixed random controls and null-anchored relative robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json

from ..scheduler import SimpleJobStore
from .config import load_experiment


def _load(args: argparse.Namespace) -> Any:
    return load_experiment(args.config)


def _validate(args: argparse.Namespace) -> int:
    experiment = _load(args)
    value = {
        "status": "valid",
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "prefix_sweep_id": experiment.prefix.sweep_id,
        "prefix_sweep_digest": experiment.prefix.digest,
        "random_seed_bank": list(experiment.random_seed_bank),
        "controls": len(experiment.control_tasks()),
        "cells": len(experiment.prefix.cells()),
        "independent_noise_summary": str(experiment.independent_noise_summary),
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _plan(args: argparse.Namespace) -> int:
    experiment = _load(args)
    value = {
        "schema_version": 1,
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "random_seed_bank": list(experiment.random_seed_bank),
        "science": {
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "random_mask_policy": "uniform_patch_permutation_shared_across_conditions_per_image_seed",
            "inference_only": True,
        },
        "controls": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "cell": task.cell_id,
                "condition": task.prefix_task.condition.condition_id,
                "artifact_root": task.artifact_root,
            }
            for task in experiment.control_tasks()
        ],
    }
    if args.output:
        atomic_write_json(args.output, value)
        print(f"WROTE {Path(args.output).resolve()}")
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _control(args: argparse.Namespace) -> int:
    from .control import run_control_task

    experiment = _load(args)
    value = run_control_task(
        experiment,
        experiment.find_control_task(args.task_id),
        device=args.device,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    from .scheduler import run_scheduler

    experiment = _load(args)
    value = run_scheduler(
        experiment,
        poll_seconds=args.poll_seconds,
        headroom_fraction=args.headroom_fraction,
    )
    if args.report_output_directory:
        from .report import write_relative_robustness_report

        value = {
            **value,
            "report": write_relative_robustness_report(
                experiment,
                output_directory=args.report_output_directory,
            ),
        }
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


def _report(args: argparse.Namespace) -> int:
    from .report import write_relative_robustness_report

    value = write_relative_robustness_report(
        _load(args),
        output_directory=args.output_directory,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_batch_size=args.bootstrap_batch_size,
        confidence=args.confidence,
        seed=args.seed,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def register_subcommands(commands: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    validate = commands.add_parser("validate", parents=[common])
    validate.set_defaults(handler=_validate)
    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)
    control = commands.add_parser("control", parents=[common])
    control.add_argument("--task-id", required=True)
    control.add_argument("--device", default="cuda:0")
    control.set_defaults(handler=_control)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("--poll-seconds", type=float, default=10.0)
    run.add_argument("--headroom-fraction", type=float)
    run.add_argument("--report-output-directory")
    run.set_defaults(handler=_run)
    status = commands.add_parser("status", parents=[common])
    status.set_defaults(handler=_status)
    retry = commands.add_parser("retry", parents=[common])
    retry.add_argument("--job-id", action="append", required=True)
    retry.set_defaults(handler=_retry)
    report = commands.add_parser("report", parents=[common])
    report.add_argument("--output-directory", required=True)
    report.add_argument("--bootstrap-replicates", type=int)
    report.add_argument("--bootstrap-batch-size", type=int)
    report.add_argument("--confidence", type=float)
    report.add_argument("--seed", type=int, default=0)
    report.set_defaults(handler=_report)


__all__ = ["register_subcommands"]
