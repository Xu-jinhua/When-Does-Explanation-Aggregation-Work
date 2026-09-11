"""CLI entry points for the simple two-stage experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.data import get_dataset_spec, read_manifest
from xai_ensemble.phase0.models import get_model_definition

from .config import (
    find_adversarial_task,
    find_phase1_task,
    find_phase2_profile,
    find_phase2_task,
    find_profile,
    load_experiment,
)


def _load(args: argparse.Namespace) -> Any:
    return load_experiment(args.config)


def _validate_assets(experiment: Any) -> list[Mapping[str, Any]]:
    from .data import load_raw_dataset_mean

    failures: list[str] = []
    manifests = {}
    report: list[Mapping[str, Any]] = []
    if not experiment.storage.rclone_binary.is_file():
        failures.append(f"missing rclone binary: {experiment.storage.rclone_binary}")
    spool_parent = experiment.storage.spool_root
    while not spool_parent.exists() and spool_parent != spool_parent.parent:
        spool_parent = spool_parent.parent
    if not spool_parent.is_dir() or not os.access(spool_parent, os.W_OK | os.X_OK):
        failures.append(f"Phase 1 spool is not writable below {experiment.storage.spool_root}")
    else:
        spool_free = shutil.disk_usage(spool_parent).free
        if spool_free <= experiment.storage.spool_min_free_bytes:
            failures.append(
                f"Phase 1 spool has only {spool_free / 2**30:.2f} GiB free; "
                f"it must preserve {experiment.storage.spool_min_free_bytes / 2**30:.2f} GiB"
            )
        report.append(
            {
                "kind": "phase1_spool",
                "root": str(experiment.storage.spool_root),
                "available_gib": round(spool_free / 2**30, 2),
                "max_pending_gib": round(
                    experiment.storage.spool_max_bytes / 2**30,
                    2,
                ),
                "min_free_gib": round(
                    experiment.storage.spool_min_free_bytes / 2**30,
                    2,
                ),
            }
        )
    for dataset in experiment.datasets:
        manifest_path = resolve_full_matrix_runtime_path(dataset.manifest_path)
        if not manifest_path.is_file():
            failures.append(f"{dataset.dataset_id}: missing manifest {dataset.manifest_path}")
            continue
        try:
            manifest = read_manifest(manifest_path)
            if dataset.registry_key is not None:
                spec = get_dataset_spec(dataset.registry_key)
                if manifest.metadata.dataset_id != spec.dataset_id:
                    raise ValueError(
                        f"dataset id {manifest.metadata.dataset_id!r} != {spec.dataset_id!r}"
                    )
                if manifest.metadata.revision != spec.revision:
                    raise ValueError("dataset revision does not match the registry")
            split_counts = {
                split: len(manifest.records_for_split(split)) for split in dataset.splits
            }
            if any(count == 0 for count in split_counts.values()):
                raise ValueError(f"empty configured split: {split_counts}")
            manifests[dataset.dataset_id] = manifest
            report.append(
                {
                    "kind": "dataset",
                    "id": dataset.dataset_id,
                    "fingerprint": manifest.fingerprint,
                    "splits": split_counts,
                }
            )
        except Exception as error:
            failures.append(f"{dataset.dataset_id}: invalid manifest: {error}")
    for model in experiment.models:
        mean_path = resolve_full_matrix_runtime_path(model.mean_path)
        checkpoint_path = (
            None
            if model.checkpoint_path is None
            else resolve_full_matrix_runtime_path(model.checkpoint_path)
        )
        manifest = manifests.get(model.dataset_id)
        definition = get_model_definition(model.model_key)
        if not mean_path.exists():
            failures.append(f"{model.model_id}: missing mean artifact {model.mean_path}")
        else:
            try:
                mean = load_raw_dataset_mean(model, input_size=definition.input_size)
                mean_manifest_path = mean_path / "manifest.json" if mean_path.is_dir() else None
                if mean_manifest_path is not None and mean_manifest_path.is_file():
                    mean_manifest = json.loads(mean_manifest_path.read_text(encoding="utf-8"))
                    metadata = mean_manifest.get("metadata", {})
                    if metadata.get("model_id") != model.model_key:
                        raise ValueError("mean artifact model_id mismatch")
                    if manifest is not None:
                        if metadata.get("dataset_id") != manifest.metadata.dataset_id:
                            raise ValueError("mean artifact dataset_id mismatch")
                        if metadata.get("dataset_manifest_fingerprint") != manifest.fingerprint:
                            raise ValueError("mean artifact manifest fingerprint mismatch")
                report.append(
                    {
                        "kind": "mean",
                        "id": model.model_id,
                        "shape": list(mean.shape[1:]),
                    }
                )
            except Exception as error:
                failures.append(f"{model.model_id}: invalid mean artifact: {error}")
        if model.checkpoint_path is None:
            report.append({"kind": "model", "id": model.model_id, "init_mode": model.init_mode})
            continue
        assert checkpoint_path is not None
        if not checkpoint_path.is_file():
            failures.append(f"{model.model_id}: missing checkpoint {model.checkpoint_path}")
            continue
        sidecar_path = Path(f"{checkpoint_path}.json")
        if not sidecar_path.is_file():
            failures.append(f"{model.model_id}: missing checkpoint sidecar {sidecar_path}")
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            metadata = sidecar.get("metadata", {})
            expected = {
                "model_key": model.model_key,
                "num_classes": model.num_classes,
                "role": "reference",
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if manifest is not None:
                for key, value in {
                    "dataset_id": manifest.metadata.dataset_id,
                    "dataset_manifest_fingerprint": manifest.fingerprint,
                }.items():
                    if metadata.get(key) != value:
                        mismatches[key] = (metadata.get(key), value)
            if mismatches:
                raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
            observed_sha256 = file_sha256(checkpoint_path)
            if sidecar.get("checkpoint_sha256") != observed_sha256:
                raise ValueError("checkpoint SHA-256 differs from its sidecar")
            report.append(
                {
                    "kind": "model",
                    "id": model.model_id,
                    "checkpoint_sha256": observed_sha256,
                }
            )
        except Exception as error:
            failures.append(f"{model.model_id}: invalid checkpoint: {error}")
    if failures:
        raise ValueError("Asset validation failed:\n- " + "\n- ".join(failures))
    return report


def _validate(args: argparse.Namespace) -> int:
    experiment = _load(args)
    assets = _validate_assets(experiment)
    from .runtime_dependencies import require_relprop_runtime

    runtime = require_relprop_runtime(
        (task.family, task.model.architecture) for task in experiment.phase1_tasks()
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "experiment_id": experiment.experiment_id,
                "digest": experiment.digest,
                "phase1_digest": experiment.phase1_digest,
                "scheduler_digest": experiment.scheduler_digest,
                "assets": assets,
                "runtime_dependencies": {"relprop": runtime},
                "precision": experiment.precision,
                "profiles": len(experiment.profiles()),
                "phase2_profiles": len(experiment.phase2_profiles()),
                "adversarial_tasks": len(experiment.adversarial_tasks()),
                "phase1_tasks": len(experiment.phase1_tasks()),
                "phase2_tasks": len(experiment.phase2_tasks()),
                "method_catalog_digest": experiment.methods.source_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _plan_value(experiment: Any) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "experiment_digest": experiment.digest,
        "phase1_experiment_digest": experiment.phase1_digest,
        "scheduler_digest": experiment.scheduler_digest,
        "precision": "fp32",
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "model_key": profile.model_key,
                "architecture": profile.architecture,
                "method": profile.method.family,
                "variant": profile.method.variant,
                "params": dict(profile.method.params),
                "profile_start": profile.method.profile_start,
            }
            for profile in experiment.profiles()
        ],
        "phase2_profiles": [
            {
                "profile_id": profile.profile_id,
                "model_key": profile.model_key,
                "architecture": profile.architecture,
                "input_shape": [3, profile.input_size, profile.input_size],
                "forward_batch_size": profile.inference_batch_size,
            }
            for profile in experiment.phase2_profiles()
        ],
        "adversarial": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "dataset": task.dataset.dataset_id,
                "model": task.model.model_id,
                "split": task.split,
                "condition": task.condition.condition_id,
                "algorithm": task.algorithm,
                "source_method": task.source_method,
                "batch_size": task.batch_size,
                "epsilon": task.epsilon,
                "steps": task.steps,
            }
            for task in experiment.adversarial_tasks()
        ],
        "phase1": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "dataset": task.dataset.dataset_id,
                "model": task.model.model_id,
                "split": task.split,
                "condition": task.condition.condition_id,
                "method": task.family,
                "variants": [item.artifact_name for item in task.variants],
                "profile_ids": list(task.profile_ids),
            }
            for task in experiment.phase1_tasks()
        ],
        "phase2": [
            {
                "task_id": task.task_id,
                "digest": task.digest,
                "dataset": task.dataset.dataset_id,
                "model": task.model.model_id,
                "split": task.split,
                "condition": task.condition.condition_id,
                "ensemble": task.ensemble.ensemble_id,
                "patch_size": task.patch_size,
                "setting": (
                    "primary"
                    if task.patch_size == experiment.phase2.primary_patch_size
                    else "additional"
                ),
            }
            for task in experiment.phase2_tasks()
        ],
    }


def _plan(args: argparse.Namespace) -> int:
    experiment = _load(args)
    value = _plan_value(experiment)
    if args.output:
        atomic_write_json(args.output, value)
        print(f"WROTE {Path(args.output).resolve()}")
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _profile(args: argparse.Namespace) -> int:
    from .profiler import run_profile

    experiment = _load(args)
    profile = find_profile(experiment, args.profile_id)
    result = run_profile(
        experiment,
        profile,
        device=args.device,
    )
    if result is None:
        print(f"SKIPPED profile={profile.profile_id} reason=no-cuda-device")
        return 0
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _phase2_profile(args: argparse.Namespace) -> int:
    from .profiler import run_phase2_profile

    experiment = _load(args)
    profile = find_phase2_profile(experiment, args.profile_id)
    result = run_phase2_profile(experiment, profile, device=args.device)
    if result is None:
        print(f"SKIPPED phase2_profile={profile.profile_id} reason=no-cuda-device")
        return 0
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _adversarial(args: argparse.Namespace) -> int:
    from .adversarial import run_adversarial_task

    experiment = _load(args)
    task = find_adversarial_task(experiment, args.task_id)
    manifest = run_adversarial_task(experiment, task, device=args.device)
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": manifest["status"],
                "artifact_digest": manifest["artifact_digest"],
                "sample_count": manifest["sample_count"],
                "shards": len(manifest["shards"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _phase1(args: argparse.Namespace) -> int:
    from .phase1 import run_phase1_task

    experiment = _load(args)
    task = find_phase1_task(experiment, args.task_id)
    manifests = run_phase1_task(
        experiment,
        task,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": "complete",
                "artifacts": [
                    {
                        "method": value["method"]["artifact_name"],
                        "sample_count": value["sample_count"],
                        "shards": len(value["shards"]),
                    }
                    for value in manifests
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _phase2(args: argparse.Namespace) -> int:
    from .phase2 import run_phase2_task

    experiment = _load(args)
    task = find_phase2_task(experiment, args.task_id)
    manifest = run_phase2_task(experiment, task, device=args.device)
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "metrics": manifest["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    from .scheduler import run_scheduler

    experiment = _load(args)
    counts = run_scheduler(
        experiment,
        include_phase2=args.include_phase2,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps({"status": "terminal", "counts": counts}, indent=2, sort_keys=True))
    return 0 if not counts.get("failed") and not counts.get("blocked") else 1


def _status(args: argparse.Namespace) -> int:
    from .profiler import missing_phase2_profiles, missing_profiles
    from .scheduler import scheduler_status

    experiment = _load(args)
    value = dict(scheduler_status(experiment))
    value.update(
        {
            "experiment_id": experiment.experiment_id,
            "experiment_digest": experiment.digest,
            "profiles_total": len(experiment.profiles()),
            "profiles_missing": [item.profile_id for item in missing_profiles(experiment)],
            "phase2_profiles_total": len(experiment.phase2_profiles()),
            "phase2_profiles_missing": [
                item.profile_id for item in missing_phase2_profiles(experiment)
            ],
            "adversarial_tasks_total": len(experiment.adversarial_tasks()),
        }
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _retry(args: argparse.Namespace) -> int:
    from .scheduler import SimpleJobStore, scheduler_status

    experiment = _load(args)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    store.recover_orphans()
    result = dict(store.retry_failed(tuple(args.job_id)))
    result["jobs"] = list(args.job_id)
    result["status"] = scheduler_status(experiment)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _summarize(args: argparse.Namespace) -> int:
    if args.table == "table1":
        from .summary import write_table1_summary

        result = write_table1_summary(
            _load(args),
            manifest_source=args.manifest_source,
            output_directory=args.output_directory,
        )
    elif args.table == "metrics":
        from .summary import write_metrics_summary

        result = write_metrics_summary(
            _load(args),
            manifest_source=args.manifest_source,
            output_directory=args.output_directory,
        )
    else:
        raise ValueError(f"Unsupported simple summary table {args.table!r}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _effective_robustness(args: argparse.Namespace) -> int:
    from .effective_robustness import write_effective_robustness

    result = write_effective_robustness(
        input_summary_path=args.input_summary,
        reference_summary_path=args.reference_summary,
        experiment=_load(args),
        manifest_source=args.manifest_source,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _effective_robustness_noise_bootstrap(args: argparse.Namespace) -> int:
    from .effective_robustness_noise import write_noise_vs_naive_bootstrap
    from .noise_prefix.config import load_noise_prefix_experiment

    result = write_noise_vs_naive_bootstrap(
        base_experiment=_load(args),
        prefix_experiment=load_noise_prefix_experiment(args.noise_prefix_config),
        naive_er_summary_path=args.naive_er_summary,
        noise_er_summary_path=args.noise_er_summary,
        output_directory=args.output_directory,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence=args.confidence,
        seed=args.seed,
        bootstrap_batch_size=args.bootstrap_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _quality_retention(args: argparse.Namespace) -> int:
    from .quality_retention import write_quality_retention

    result = write_quality_retention(
        candidate_summary_path=args.candidate_summary,
        reference_summary_path=args.naive_summary,
        experiment=_load(args),
        manifest_source=args.manifest_source,
        temporary_directory=args.temporary_directory,
        download_workers=args.download_workers,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _adversarial_pilot(args: argparse.Namespace) -> int:
    from .adversarial_pilot import SaraAttackConfig, run_adversarial_pilot

    experiment = _load(args)
    report = run_adversarial_pilot(
        experiment,
        model_id=args.model_id,
        split=args.split,
        source_method=args.source_method,
        batch_sizes=args.batch_sizes,
        profile_steps=args.profile_steps,
        analysis_samples=args.analysis_samples,
        attack_config=SaraAttackConfig(
            epsilon=args.epsilon,
            steps=args.steps,
            learning_rate=args.learning_rate,
            classification_weight=args.classification_weight,
            top_fraction=args.top_fraction,
        ),
        patch_sizes=args.patch_sizes,
        top_k=args.top_k,
        device=args.device,
        output_directory=args.output_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def register_subcommands(commands: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)

    validate = commands.add_parser(
        "validate", parents=[common], help="Validate the paper and runtime contract"
    )
    validate.set_defaults(handler=_validate)

    plan = commands.add_parser(
        "plan",
        parents=[common],
        help="Show deterministic profiles, adversarial datasets, and Phase 1/2 tasks",
    )
    plan.add_argument("--output")
    plan.set_defaults(handler=_plan)

    profile = commands.add_parser(
        "profile", parents=[common], help="Run one missing model-method batch search"
    )
    profile.add_argument("--profile-id", required=True)
    profile.add_argument("--device", default="cuda:0")
    profile.set_defaults(handler=_profile)

    phase2_profile = commands.add_parser(
        "phase2-profile",
        parents=[common],
        help="Measure one model/input/forward-batch Phase 2 memory profile",
    )
    phase2_profile.add_argument("--profile-id", required=True)
    phase2_profile.add_argument("--device", default="cuda:0")
    phase2_profile.set_defaults(handler=_phase2_profile)

    adversarial_formal = commands.add_parser(
        "adversarial",
        parents=[common],
        help="Generate one immutable sharded adversarial dataset",
    )
    adversarial_formal.add_argument("--task-id", required=True)
    adversarial_formal.add_argument("--device", default="cuda:0")
    adversarial_formal.set_defaults(handler=_adversarial)

    phase1 = commands.add_parser(
        "phase1", parents=[common], help="Generate one method task as safetensors shards"
    )
    phase1.add_argument("--task-id", required=True)
    phase1.add_argument("--device", default="cuda:0")
    phase1.set_defaults(handler=_phase1)

    phase2 = commands.add_parser(
        "phase2", parents=[common], help="Aggregate and evaluate one aligned artifact task"
    )
    phase2.add_argument("--task-id", required=True)
    phase2.add_argument("--device", default="cuda:0")
    phase2.set_defaults(handler=_phase2)

    run = commands.add_parser(
        "run",
        parents=[common],
        help="Run Phase 1 jobs, optionally followed by Phase 2 jobs",
    )
    run.add_argument(
        "--include-phase2",
        action="store_true",
        help="also schedule Phase 2 jobs already present in or added to the queue",
    )
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.set_defaults(handler=_run)

    status = commands.add_parser(
        "status", parents=[common], help="Report profiles and SQLite queue state"
    )
    status.set_defaults(handler=_status)

    retry = commands.add_parser(
        "retry",
        parents=[common],
        help="Retry explicit failed jobs and unblock their descendants",
    )
    retry.add_argument("--job-id", action="append", required=True)
    retry.set_defaults(handler=_retry)

    summarize = commands.add_parser(
        "summarize",
        parents=[common],
        help="Create deterministic table-ready summaries from completed Phase 2 artifacts",
    )
    summarize.add_argument("--table", choices=("table1", "metrics"), default="table1")
    summarize.add_argument(
        "--manifest-source",
        choices=("auto", "local", "remote"),
        default="auto",
    )
    summarize.add_argument("--output-directory")
    summarize.set_defaults(handler=_summarize)

    effective_robustness = commands.add_parser(
        "effective-robustness",
        parents=[common],
        help="Calculate offline Effective Robustness from completed quality summaries",
    )
    effective_robustness.add_argument(
        "--input-summary",
        required=True,
        help="Completed NAIVE, IND, or NOISE summary containing raw quality values",
    )
    effective_robustness.add_argument(
        "--reference-summary",
        required=True,
        help="Original NAIVE Table 1 summary used to locate single-explainer references",
    )
    effective_robustness.add_argument(
        "--manifest-source",
        choices=("auto", "local", "remote"),
        default="auto",
        help="Where to read the original NAIVE Phase 2 manifests",
    )
    effective_robustness.add_argument(
        "--output-directory",
        required=True,
        help="Directory for JSON, CSV, TeX, and reference-curve outputs",
    )
    effective_robustness.set_defaults(handler=_effective_robustness)

    effective_robustness_noise = commands.add_parser(
        "effective-robustness-noise-bootstrap",
        parents=[common],
        help="Bootstrap fixed-q NOISE ER against the matching q=11 NAIVE rule",
    )
    effective_robustness_noise.add_argument(
        "--noise-prefix-config",
        required=True,
        help="Completed q-prefix NOISE configuration used to locate prediction shards",
    )
    effective_robustness_noise.add_argument(
        "--naive-er-summary",
        required=True,
        help="Completed main q=11 NAIVE Effective Robustness summary",
    )
    effective_robustness_noise.add_argument(
        "--noise-er-summary",
        required=True,
        help="Completed independent-geometry NOISE Effective Robustness summary",
    )
    effective_robustness_noise.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=19999,
        help="Class-stratified paired bootstrap repetitions; 12,799+ are needed for 640-endpoint Holm resolution at alpha=0.05",
    )
    effective_robustness_noise.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Percentile confidence level and family-wise alpha complement",
    )
    effective_robustness_noise.add_argument("--seed", type=int, default=0)
    effective_robustness_noise.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=32,
        help="Replicate batch size for bounded CPU memory",
    )
    effective_robustness_noise.add_argument("--output-directory", required=True)
    effective_robustness_noise.set_defaults(handler=_effective_robustness_noise_bootstrap)

    quality_retention = commands.add_parser(
        "quality-retention",
        parents=[common],
        help="Calculate normalized absolute quality, clean retention, and geometric mean",
    )
    quality_retention.add_argument(
        "--candidate-summary",
        required=True,
        help="Completed independent-geometry NOISE summary",
    )
    quality_retention.add_argument(
        "--naive-summary",
        required=True,
        help="Original q=11 NAIVE Table 1 summary used to locate Phase 2 artifacts",
    )
    quality_retention.add_argument(
        "--manifest-source",
        choices=("auto", "local", "remote"),
        default="auto",
    )
    quality_retention.add_argument(
        "--temporary-directory",
        default="/dev/shm",
        help="Temporary location for one verified Phase 2 condition at a time",
    )
    quality_retention.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Concurrent verified shard downloads within one condition",
    )
    quality_retention.add_argument("--output-directory", required=True)
    quality_retention.set_defaults(handler=_quality_retention)

    adversarial = commands.add_parser(
        "adversarial-pilot",
        parents=[common],
        help="Profile and validate Sara-style explanation attacks",
    )
    adversarial.add_argument("--model-id", required=True)
    adversarial.add_argument("--split", default="test")
    adversarial.add_argument(
        "--source-method",
        choices=(
            "auto",
            "DeepLift",
            "GradientAttentionRollout",
            "TransformerAttribution",
        ),
        default="auto",
    )
    adversarial.add_argument(
        "--batch-sizes",
        type=_csv_ints,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    adversarial.add_argument("--profile-steps", type=int, default=10)
    adversarial.add_argument("--analysis-samples", type=int, default=32)
    adversarial.add_argument("--steps", type=int, default=100)
    adversarial.add_argument("--epsilon", type=float, default=2.0 / 255.0)
    adversarial.add_argument("--learning-rate", type=float, default=0.1)
    adversarial.add_argument("--classification-weight", type=float, default=1e-4)
    adversarial.add_argument("--top-fraction", type=float, default=0.1)
    adversarial.add_argument("--patch-sizes", type=_csv_ints, default=(8, 14, 16))
    adversarial.add_argument("--top-k", type=int, default=20)
    adversarial.add_argument("--device", default="cuda:0")
    adversarial.add_argument("--output-directory")
    adversarial.set_defaults(handler=_adversarial_pilot)

    ablation = commands.add_parser(
        "ablation",
        help="Run isolated NAIVE ablations over immutable main artifacts",
    )
    ablation_commands = ablation.add_subparsers(dest="ablation_command", required=True)
    from .ablations.cli import register_subcommands as register_ablation_subcommands

    register_ablation_subcommands(ablation_commands)

    assumptions = commands.add_parser(
        "assumptions",
        help="Run isolated IND, matched-NAIVE, and Oracle NOISE experiments",
    )
    assumption_commands = assumptions.add_subparsers(dest="assumptions_command", required=True)
    from .assumptions.cli import register_subcommands as register_assumption_subcommands

    register_assumption_subcommands(assumption_commands)

    noise_prefix = commands.add_parser(
        "noise-prefix",
        help="Measure every Fidelity-ordered q=2..11 NOISE prefix",
    )
    noise_prefix_commands = noise_prefix.add_subparsers(dest="noise_prefix_command", required=True)
    from .noise_prefix.cli import register_subcommands as register_noise_prefix_subcommands

    register_noise_prefix_subcommands(noise_prefix_commands)

    full_matrix = commands.add_parser(
        "full-matrix",
        help="Prepare and run the complete dataset-by-model GitHub result matrix",
    )
    full_matrix_commands = full_matrix.add_subparsers(
        dest="full_matrix_command",
        required=True,
    )
    from .full_matrix.cli import register_subcommands as register_full_matrix_subcommands

    register_full_matrix_subcommands(full_matrix_commands)

    noise_subset = commands.add_parser(
        "noise-subset",
        help="Compare random NOISE-consistent subsets with Fidelity prefixes",
    )
    noise_subset_commands = noise_subset.add_subparsers(dest="noise_subset_command", required=True)
    from .noise_subset.cli import register_subcommands as register_noise_subset_subcommands

    register_noise_subset_subcommands(noise_subset_commands)

    relative_robustness = commands.add_parser(
        "relative-robustness",
        help="Evaluate fixed random-mask controls and calculate null-anchored R_rel",
    )
    relative_robustness_commands = relative_robustness.add_subparsers(
        dest="relative_robustness_command",
        required=True,
    )
    from .relative_robustness.cli import (
        register_subcommands as register_relative_robustness_subcommands,
    )

    register_relative_robustness_subcommands(relative_robustness_commands)


__all__ = ["register_subcommands"]
