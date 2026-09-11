"""CLI for the isolated NAIVE ablation queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.phase0.models import get_model_definition

from ..adversarial import run_adversarial_task
from ..phase1 import run_phase1_task
from ..scheduler import SimpleJobStore
from .config import load_ablation_experiment
from .evaluator import run_evaluation_task
from .rank_source import load_rank_manifest
from .ranks import run_rank_task
from .scheduler import _profile_reservation, run_scheduler, scheduler_status
from .summary import write_summaries


def _load(args: argparse.Namespace) -> Any:
    return load_ablation_experiment(args.config)


def _validate_value(experiment: Any) -> dict[str, Any]:
    if not experiment.output_storage.rclone_binary.is_file():
        raise FileNotFoundError(f"Missing rclone binary: {experiment.output_storage.rclone_binary}")
    scratch_parent = experiment.output_storage.scratch_root
    while not scratch_parent.exists() and scratch_parent != scratch_parent.parent:
        scratch_parent = scratch_parent.parent
    if not scratch_parent.is_dir() or not os.access(scratch_parent, os.W_OK | os.X_OK):
        raise ValueError(f"Ablation scratch is not writable below {scratch_parent}")
    spool = experiment.output_storage.spool_root
    if not spool.is_dir() or not os.access(spool, os.W_OK | os.X_OK):
        raise ValueError(f"Ablation spool must already be writable: {spool}")
    free = shutil.disk_usage(spool).free
    if free <= experiment.output_storage.spool_min_free_bytes:
        raise ValueError("Ablation spool violates its configured free-space floor")
    generation = experiment.generation_experiment()
    if generation.phase1_digest != experiment.base.phase1_digest:
        raise ValueError("Ablation generation changed the accepted Phase 1 semantics")
    reservations = {}
    for task in experiment.phase1_tasks():
        reservations.setdefault(task.family, _profile_reservation(experiment, task))
    base_rank_sources = {}
    for condition in experiment.base_conditions():
        source = experiment.rank_source(condition)
        manifest = load_rank_manifest(experiment, source)
        base_rank_sources[condition.condition_id] = {
            "task_digest": manifest["task_digest"],
            "sample_count": manifest["sample_count"],
        }
    from ..data import load_raw_class_means

    definition = get_model_definition(experiment.model.model_key)
    class_means, class_counts = load_raw_class_means(
        experiment.model, input_size=definition.input_size
    )
    counts = {
        "adversarial": len(experiment.adversarial_tasks()),
        "phase1": len(experiment.phase1_tasks()),
        "rank": len(experiment.rank_tasks()),
        "evaluation": len(experiment.evaluation_tasks()),
    }
    expected = {"adversarial": 2, "phase1": 88, "rank": 8, "evaluation": 28}
    if counts != expected:
        raise ValueError(f"Ablation task matrix differs from the accepted plan: {counts}")
    return {
        "status": "valid",
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "base_phase1_digest": experiment.base.phase1_digest,
        "output_root": experiment.output_storage.remote_root,
        "scope": {
            "dataset": experiment.dataset_id,
            "model": experiment.model_id,
            "split": experiment.split,
            "patch_size": experiment.patch_size,
        },
        "task_counts": counts,
        "phase1_batch_caps": dict(experiment.runtime.phase1_batch_caps),
        "phase1_reservation_gib": {
            key: round(value / 2**30, 3) for key, value in reservations.items()
        },
        "inference_batch_size": experiment.runtime.inference_batch_size,
        "evaluation_reservation_gib": round(
            experiment.runtime.evaluation_reservation_bytes / 2**30, 3
        ),
        "base_rank_sources": base_rank_sources,
        "class_means_shape": list(class_means.shape),
        "class_count_shape": list(class_counts.shape),
        "spool_free_gib": round(free / 2**30, 2),
    }


def _validate(args: argparse.Namespace) -> int:
    print(json.dumps(_validate_value(_load(args)), indent=2, sort_keys=True))
    return 0


def _plan_value(experiment: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "scheduler_digest": experiment.scheduler_digest,
        "reuse": [
            {
                "condition": condition.condition_id,
                "rank_source": experiment.rank_source(condition).root,
            }
            for condition in experiment.base_conditions()
        ],
        "noise_levels": {
            kind: [
                {
                    "severity": level.severity,
                    "condition": level.condition_id,
                    "center_reused": level.center,
                }
                for level in levels
            ]
            for kind, levels in experiment.noise_levels.items()
        },
        "adversarial": [
            {
                "task_id": task.task_id,
                "condition": task.condition.condition_id,
                "epsilon": task.epsilon,
                "batch_size": task.batch_size,
            }
            for task in experiment.adversarial_tasks()
        ],
        "phase1": [
            {
                "task_id": task.task_id,
                "condition": task.condition.condition_id,
                "method": task.family,
                "batch_cap": experiment.runtime.phase1_batch_caps.get(task.family),
                "variants": [variant.artifact_name for variant in task.variants],
            }
            for task in experiment.phase1_tasks()
        ],
        "rank": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "condition": task.condition.condition_id,
                "phase1_dependencies": list(task.phase1_task_ids),
            }
            for task in experiment.rank_tasks()
        ],
        "evaluation": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "table": task.table_id,
                "condition": task.condition.condition_id,
                "rank_source": task.rank_source.source_id,
                "patch_size": task.patch_size,
                "k": task.k,
                "fill": task.fill,
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


def _phase1(args: argparse.Namespace) -> int:
    experiment = _load(args)
    generation = experiment.generation_experiment()
    task = next((item for item in experiment.phase1_tasks() if item.task_id == args.task_id), None)
    if task is None:
        raise KeyError(f"Unknown ablation Phase 1 task {args.task_id!r}")
    manifests = run_phase1_task(
        generation,
        task,
        device=args.device,
        batch_size_override=experiment.runtime.phase1_batch_caps.get(task.family),
    )
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": "complete",
                "artifacts": [value["method"]["artifact_name"] for value in manifests],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _adversarial(args: argparse.Namespace) -> int:
    experiment = _load(args)
    generation = experiment.generation_experiment()
    task = next(
        (item for item in experiment.adversarial_tasks() if item.task_id == args.task_id), None
    )
    if task is None:
        raise KeyError(f"Unknown ablation adversarial task {args.task_id!r}")
    manifest = run_adversarial_task(generation, task, device=args.device)
    print(json.dumps({"task_id": task.task_id, "status": manifest["status"]}, indent=2))
    return 0


def _rank(args: argparse.Namespace) -> int:
    experiment = _load(args)
    task = experiment.find_rank_task(args.task_id)
    manifest = run_rank_task(experiment, task, device=args.device)
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    experiment = _load(args)
    task = experiment.find_evaluation_task(args.task_id)
    manifest = run_evaluation_task(experiment, task, device=args.device)
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "metrics": manifest["metrics"],
                "robustness": manifest["robustness"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    print(json.dumps(write_summaries(_load(args)), indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    experiment = _load(args)
    print("ABLATION_RUN phase=validate status=started", file=sys.stderr, flush=True)
    _validate_value(experiment)
    print("ABLATION_RUN phase=validate status=complete", file=sys.stderr, flush=True)
    counts = run_scheduler(experiment, poll_seconds=args.poll_seconds)
    result: dict[str, Any] = {"status": "terminal", "counts": counts}
    if not counts.get("failed") and not counts.get("blocked"):
        result["summaries"] = write_summaries(experiment)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not counts.get("failed") and not counts.get("blocked") else 1


def _status(args: argparse.Namespace) -> int:
    experiment = _load(args)
    value = dict(scheduler_status(experiment))
    value.update(
        {
            "ablation_id": experiment.ablation_id,
            "ablation_digest": experiment.digest,
            "task_counts": {
                "adversarial": len(experiment.adversarial_tasks()),
                "phase1": len(experiment.phase1_tasks()),
                "rank": len(experiment.rank_tasks()),
                "evaluation": len(experiment.evaluation_tasks()),
            },
        }
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _retry(args: argparse.Namespace) -> int:
    experiment = _load(args)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    store.recover_orphans()
    result = dict(store.retry_failed(tuple(args.job_id)))
    result["status"] = scheduler_status(experiment)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def register_subcommands(commands: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)

    validate = commands.add_parser("validate", parents=[common])
    validate.set_defaults(handler=_validate)

    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)

    run = commands.add_parser("run", parents=[common])
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.set_defaults(handler=_run)

    status = commands.add_parser("status", parents=[common])
    status.set_defaults(handler=_status)

    retry = commands.add_parser("retry", parents=[common])
    retry.add_argument("--job-id", action="append", required=True)
    retry.set_defaults(handler=_retry)

    summarize = commands.add_parser("summarize", parents=[common])
    summarize.set_defaults(handler=_summarize)

    for name, handler in (
        ("adversarial", _adversarial),
        ("phase1", _phase1),
        ("rank", _rank),
        ("evaluate", _evaluate),
    ):
        parser = commands.add_parser(name, parents=[common])
        parser.add_argument("--task-id", required=True)
        parser.add_argument("--device", default="cuda:0")
        parser.set_defaults(handler=handler)


__all__ = ["register_subcommands"]
