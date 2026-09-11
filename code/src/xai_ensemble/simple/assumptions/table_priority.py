"""Full 11-by-11 IND and matched-NAIVE execution across three partition families."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, GpuSnapshot, NvidiaSmiProbe
from xai_ensemble.core.hashing import object_sha256

from ..profiler import load_profile
from ..runtime import gpu_release_signal_matches
from ..scheduler import (
    STATUSES,
    QueueJob,
    RunningProcess,
    SimpleJobStore,
    _gpu_release_blocks_admission,
    _launch,
    _outstanding_reservation,
    _ready_jobs,
    _refresh_gpu_release_admission,
    _reservation_fits,
)
from .artifacts import (
    PARTITION_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
    completed_evaluation_manifest,
    completed_rank_manifest,
    completed_task_manifest,
)
from .config import AssumptionExperiment, SourcePhase1Task
from .phase1 import _base_profile, source_method_complete
from .training import checkpoint_paths

TABLE_PRIORITY_SCHEMA = "simple-assumptions-full-ind-v4"
TABLE_PRIORITY_JOB_KINDS = frozenset(
    {"partition", "training", "source-method", "rank", "evaluation"}
)
CPU_RESERVATION_BYTES = 1 * 2**30
MINIMUM_METHOD_RESERVATION_BYTES = 4 * 2**30
METHOD_RUNTIME_OVERHEAD_BYTES = 2 * 2**30
METHOD_RESERVATION_SAFETY_FACTOR = 1.10
COMPLETION_SCAN_WORKERS = 8

# Worker-path VRAM peaks measured 2026-09-03 on the production call path (real
# checkpoints, locked-grid batch sizes, vit-b16, 224x224 inputs), keyed by
# (architecture, method family, batch size).  The stored batch profiles ran
# their target forward on the relprop-wrapped model, whose forward hooks stash
# activations even under no_grad; relprop-family peaks came out inflated
# several-fold (FullLRP: 30.4 GiB probed vs 5.3 GiB observed in production).
# Entries here supersede the stored profile peaks; unlisted combinations keep
# using the profile telemetry.
MEASURED_METHOD_PEAK_BYTES: Mapping[tuple[str, str, int], int] = {
    ("vit", "FullLRP", 128): int(7.76 * 2**30),
    ("vit", "PartialLRP", 128): int(6.32 * 2**30),
    ("vit", "CheferTransformerAttribution", 128): int(13.94 * 2**30),
    ("vit", "Saliency", 256): int(30.12 * 2**30),
    ("vit", "InputXGradient", 256): int(30.12 * 2**30),
    ("vit", "AttentionGradCAM", 128): int(24.43 * 2**30),
    ("vit", "GradientAttentionRollout", 128): int(24.53 * 2**30),
    ("vit", "FeatureAblation", 1024): int(14.46 * 2**30),
    ("vit", "Occlusion", 1024): int(12.15 * 2**30),
}


@dataclass(frozen=True, slots=True)
class TablePriorityPaths:
    root: Path
    database: Path
    logs: Path


def table_priority_paths(
    experiment: AssumptionExperiment,
    *,
    root: str | Path | None = None,
) -> TablePriorityPaths:
    destination = (
        Path(root).expanduser().resolve()
        if root is not None
        else experiment.runtime.database_path.parent.parent
        / f"{experiment.assumption_id}-ind-table-priority"
    )
    return TablePriorityPaths(destination, destination / "jobs.sqlite3", destination / "logs")


def table_priority_digest(
    experiment: AssumptionExperiment,
    paths: TablePriorityPaths,
) -> str:
    return object_sha256({
        "schema": TABLE_PRIORITY_SCHEMA,
        "assumption_digest": experiment.digest,
        "database": str(paths.database),
        "partition_families": experiment.partition_families,
        "matched_source_ids": {key: list(value) for key, value in matched_source_ids(experiment).items()},
        "ind_q": 11,
        "matched_source_count_per_family": 11,
        "source_method_pairs_per_family_condition": 121,
    })

def matched_source_ids(
    experiment: AssumptionExperiment,
) -> Mapping[str, tuple[str, ...]]:
    return {cell.cell_id: experiment.source_ids(cell) for cell in experiment.cells()}

def matched_evaluation_tasks(experiment: AssumptionExperiment) -> tuple[Any, ...]:
    selected = matched_source_ids(experiment)
    tasks = tuple(
        task
        for task in experiment.evaluation_tasks()
        if task.setting == "matched-naive" and task.source_id in selected[task.cell.cell_id]
    )
    expected = (
        len(experiment.cells())
        * experiment.partition_families
        * 11
        * len(experiment.base.conditions)
    )
    if len(tasks) != expected:
        raise RuntimeError(f"Expected {expected} full-grid matched evaluations; found {len(tasks)}")
    return tasks


def require_matched_evaluations_complete(
    experiment: AssumptionExperiment,
) -> Mapping[str, Any]:
    tasks = matched_evaluation_tasks(experiment)

    def inspect(task: Any) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
        manifest = completed_evaluation_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(f"Matched evaluation is incomplete: {task.task_id}")
        expected = {
            "setting": "matched-naive",
            "cell": task.cell.cell_id,
            "source_id": task.source_id,
            "family_id": task.family_id,
            "condition": task.condition.condition_id,
            "target_policy": "full_reference_clean_fp32_prediction",
        }
        mismatches = {
            key: {"artifact": manifest.get(key), "current": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Matched evaluation identity changed: {mismatches}")
        provenance = {
            "task_id": task.task_id,
            "task_digest": task.digest,
            "cell": task.cell.cell_id,
            "source_id": task.source_id,
            "family_id": task.family_id,
            "condition": task.condition.condition_id,
            "artifact_root": task.artifact_root,
            "manifest_content_digest": object_sha256(manifest),
        }
        return task, manifest, provenance

    with ThreadPoolExecutor(
        max_workers=min(COMPLETION_SCAN_WORKERS, len(tasks)),
        thread_name_prefix="ind-table-matched-scan",
    ) as executor:
        inspected = tuple(executor.map(inspect, tasks))

    inputs = []
    sample_counts: dict[tuple[str, str], int] = {}
    for task, manifest, provenance in inspected:
        count_key = (task.cell.cell_id, str(task.source_id))
        sample_count = int(manifest["sample_count"])
        previous = sample_counts.setdefault(count_key, sample_count)
        if previous != sample_count:
            raise ValueError(f"Matched evaluation sample count changed within {count_key}")
        inputs.append(provenance)
    return {
        "status": "complete",
        "selection_policy": "all_sources_in_all_partition_families",
        "source_count_per_cell": experiment.partition_families * 11,
        "matched_source_ids": {
            key: list(value) for key, value in matched_source_ids(experiment).items()
        },
        "evaluation_count": len(inputs),
        "inputs": inputs,
    }


def assigned_method_requirements(
    experiment: AssumptionExperiment,
) -> tuple[tuple[SourcePhase1Task, str], ...]:
    values = tuple(
        (scope, method)
        for scope in experiment.source_phase1_tasks()
        for method in scope.cell.methods
    )
    expected = sum(len(cell.methods) ** 2 for cell in experiment.cells())
    expected *= experiment.partition_families * len(experiment.base.conditions)
    if len(values) != expected:
        raise RuntimeError(f"Expected {expected} full-grid method requirements; found {len(values)}")
    return values

def ind_rank_tasks(experiment: AssumptionExperiment) -> tuple[Any, ...]:
    tasks = tuple(task for task in experiment.rank_tasks() if task.setting == "ind")
    expected = len(experiment.cells()) * experiment.partition_families * len(experiment.base.conditions)
    if len(tasks) != expected:
        raise RuntimeError(f"Expected {expected} IND rank tasks; found {len(tasks)}")
    if any(len(task.source_method_pairs) != 11 for task in tasks):
        raise ValueError("Table-priority IND ranks must retain the exact q=11 assignment")
    return tasks


def ind_evaluation_tasks(experiment: AssumptionExperiment) -> tuple[Any, ...]:
    tasks = tuple(task for task in experiment.evaluation_tasks() if task.setting == "ind")
    expected = len(experiment.cells()) * experiment.partition_families * len(experiment.base.conditions)
    if len(tasks) != expected:
        raise RuntimeError(f"Expected {expected} IND evaluation tasks; found {len(tasks)}")
    return tasks


def planned_rank_tasks(experiment: AssumptionExperiment) -> tuple[Any, ...]:
    tasks = list(ind_rank_tasks(experiment))
    selected = matched_source_ids(experiment)
    tasks.extend(
        task
        for task in experiment.rank_tasks()
        if task.setting == "matched-naive"
        and task.source_id in selected[task.cell.cell_id]
    )
    expected = len(experiment.cells()) * experiment.partition_families * 12 * len(experiment.base.conditions)
    if len(tasks) != expected:
        raise RuntimeError(f"Expected {expected} full-grid rank tasks; found {len(tasks)}")
    return tuple(tasks)


def planned_evaluation_tasks(experiment: AssumptionExperiment) -> tuple[Any, ...]:
    tasks = list(ind_evaluation_tasks(experiment))
    tasks.extend(matched_evaluation_tasks(experiment))
    expected = len(experiment.cells()) * experiment.partition_families * 12 * len(experiment.base.conditions)
    if len(tasks) != expected:
        raise RuntimeError(
            f"Expected {expected} full-grid evaluation tasks; found {len(tasks)}"
        )
    return tuple(tasks)


def _job_id(kind: str, task_id: str) -> str:
    return f"{kind}:{task_id}"


def source_method_job_id(scope: SourcePhase1Task, family: str) -> str:
    return _job_id("source-method", f"{scope.task_id}--{family}")


def planned_job_ids(experiment: AssumptionExperiment) -> frozenset[str]:
    return frozenset(
        [
            *(_job_id("partition", task.task_id) for task in experiment.partition_tasks()),
            *(_job_id("training", task.task_id) for task in experiment.training_tasks()),
            *(
                source_method_job_id(scope, family)
                for scope, family in assigned_method_requirements(experiment)
            ),
            *(_job_id("rank", task.task_id) for task in planned_rank_tasks(experiment)),
            *(
                _job_id("evaluation", task.task_id)
                for task in planned_evaluation_tasks(experiment)
            ),
        ]
    )


def _command(
    experiment: AssumptionExperiment,
    action: str,
    *,
    task_id: str,
    family: str | None = None,
) -> tuple[str, ...]:
    values = [
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "simple",
        "assumptions",
        action,
        "--config",
        str(experiment.source_path),
        "--task-id",
        task_id,
    ]
    if family is not None:
        values.extend(("--family", family))
    return tuple(values)


def _method_reservation_bytes(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    family: str,
) -> int:
    training = experiment.find_training_task(scope.training_task_id)
    checkpoint, _ = checkpoint_paths(experiment, training)
    method_task = next(
        task
        for task in experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
        if task.family == family
    )
    architecture = scope.cell.reference_model.architecture
    measured = []
    for variant in method_task.variants:
        profile_class, selected_batch = _base_profile(experiment, scope, variant)
        measured_peak = MEASURED_METHOD_PEAK_BYTES.get((architecture, family, selected_batch))
        if measured_peak is not None:
            measured.append(measured_peak)
            continue
        profile = load_profile(experiment.base, profile_class)
        if profile is None:
            raise FileNotFoundError(f"Missing Phase 1 profile: {profile_class.profile_id}")
        candidates = [
            item
            for item in profile.measurements
            if item.passed and item.batch_size == selected_batch
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Profile {profile.profile_id} has no unique measurement for batch {selected_batch}"
            )
        item = candidates[0]
        if item.peak_allocated_bytes is None or item.peak_reserved_bytes is None:
            raise ValueError(f"Profile {profile.profile_id} has incomplete CUDA telemetry")
        measured.append(max(item.peak_allocated_bytes, item.peak_reserved_bytes))
    estimate = int(
        (max(measured) + METHOD_RUNTIME_OVERHEAD_BYTES) * METHOD_RESERVATION_SAFETY_FACTOR
    )
    configured_cap = experiment.runtime.phase1_reservation_bytes[architecture]
    return min(configured_cap, max(MINIMUM_METHOD_RESERVATION_BYTES, estimate))


def _complete(experiment: AssumptionExperiment, kind: str, task: Any) -> bool:
    if kind == "partition":
        return (
            completed_task_manifest(experiment, task, schema_version=PARTITION_SCHEMA_VERSION)
            is not None
        )
    if kind == "training":
        return (
            completed_task_manifest(experiment, task, schema_version=TRAINING_SCHEMA_VERSION)
            is not None
        )
    if kind == "source-method":
        scope, family = task
        return source_method_complete(experiment, scope, family)
    if kind == "rank":
        return completed_rank_manifest(experiment, task) is not None
    if kind == "evaluation":
        return completed_evaluation_manifest(experiment, task) is not None
    raise KeyError(kind)


def submit_plan(
    experiment: AssumptionExperiment,
    store: SimpleJobStore,
    *,
    paths: TablePriorityPaths,
) -> Mapping[str, int]:
    jobs: list[QueueJob] = []
    counts = {kind: 0 for kind in TABLE_PRIORITY_JOB_KINDS}
    existing = frozenset(job.job_id for job in store.jobs())
    partitions = experiment.partition_tasks()
    training = experiment.training_tasks()
    method_requirements = assigned_method_requirements(experiment)
    ranks = planned_rank_tasks(experiment)
    evaluations = planned_evaluation_tasks(experiment)

    completion_specs = (
        *((_job_id("partition", task.task_id), "partition", task) for task in partitions),
        *((_job_id("training", task.task_id), "training", task) for task in training),
        *(
            (source_method_job_id(scope, family), "source-method", (scope, family))
            for scope, family in method_requirements
        ),
        *((_job_id("rank", task.task_id), "rank", task) for task in ranks),
        *((_job_id("evaluation", task.task_id), "evaluation", task) for task in evaluations),
    )
    unregistered = tuple(spec for spec in completion_specs if spec[0] not in existing)

    def inspect(spec: tuple[str, str, Any]) -> tuple[str, bool]:
        job_id, kind, task = spec
        return job_id, _complete(experiment, kind, task)

    if unregistered:
        with ThreadPoolExecutor(
            max_workers=min(COMPLETION_SCAN_WORKERS, len(unregistered)),
            thread_name_prefix="ind-table-completion-scan",
        ) as executor:
            completion = dict(executor.map(inspect, unregistered))
    else:
        completion = {}

    def initial_status(job_id: str) -> str:
        if job_id in existing:
            return "pending"
        return "succeeded" if completion[job_id] else "pending"

    for task in partitions:
        job_id = _job_id("partition", task.task_id)
        jobs.append(
            QueueJob(
                job_id,
                "partition",
                _command(experiment, "prepare", task_id=task.task_id),
                (),
                (),
                CPU_RESERVATION_BYTES,
                initial_status(job_id),
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(paths.logs / f"partition--{task.task_id}.log"),
            )
        )
        counts["partition"] += 1
    partition_jobs = {task.task_id: _job_id("partition", task.task_id) for task in partitions}

    for task in training:
        job_id = _job_id("training", task.task_id)
        jobs.append(
            QueueJob(
                job_id,
                "training",
                _command(experiment, "train", task_id=task.task_id),
                (partition_jobs[task.partition_task_id],),
                (),
                experiment.runtime.training_reservation_bytes[
                    task.cell.reference_model.architecture
                ],
                initial_status(job_id),
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(paths.logs / f"training--{task.task_id}.log"),
            )
        )
        counts["training"] += 1
    training_jobs = {task.task_id: _job_id("training", task.task_id) for task in training}

    method_jobs = {}
    for scope, family in method_requirements:
        job_id = source_method_job_id(scope, family)
        jobs.append(
            QueueJob(
                job_id,
                "source-method",
                _command(
                    experiment,
                    "phase1-method",
                    task_id=scope.task_id,
                    family=family,
                ),
                (training_jobs[scope.training_task_id],),
                (),
                _method_reservation_bytes(experiment, scope, family),
                initial_status(job_id),
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(paths.logs / f"source-method--{scope.task_id}--{family}.log"),
            )
        )
        method_jobs[
            scope.cell.cell_id,
            scope.source_id,
            scope.condition.condition_id,
            family,
        ] = job_id
        counts["source-method"] += 1

    for task in ranks:
        dependencies = tuple(
            sorted(
                method_jobs[
                    task.cell.cell_id,
                    source_id,
                    task.condition.condition_id,
                    family,
                ]
                for source_id, family in task.source_method_pairs
            )
        )
        job_id = _job_id("rank", task.task_id)
        jobs.append(
            QueueJob(
                job_id,
                "rank",
                _command(experiment, "rank", task_id=task.task_id),
                dependencies,
                (),
                experiment.runtime.rank_reservation_bytes,
                initial_status(job_id),
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(paths.logs / f"rank--{task.task_id}.log"),
            )
        )
        counts["rank"] += 1
    rank_jobs = {task.task_id: _job_id("rank", task.task_id) for task in ranks}

    clean_jobs = {
        (task.cell.cell_id, task.setting, task.source_id, task.distance_model): _job_id(
            "evaluation", task.task_id
        )
        for task in evaluations
        if task.condition.kind == "clean"
    }
    for task in evaluations:
        dependencies = [rank_jobs[task.rank_task_id]]
        if task.condition.kind != "clean":
            dependencies.append(
                clean_jobs[
                    task.cell.cell_id,
                    task.setting,
                    task.source_id,
                    task.distance_model,
                ]
            )
        job_id = _job_id("evaluation", task.task_id)
        jobs.append(
            QueueJob(
                job_id,
                "evaluation",
                _command(experiment, "evaluate", task_id=task.task_id),
                tuple(dependencies),
                (),
                experiment.runtime.evaluation_reservation_bytes,
                initial_status(job_id),
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(paths.logs / f"evaluation--{task.task_id}.log"),
            )
        )
        counts["evaluation"] += 1

    store.submit_many(jobs)
    store.update_reservations(
        {job.job_id: job.reservation_bytes for job in jobs if job.reservation_bytes is not None}
    )
    return counts


def _requires_exclusive_gpu(job: QueueJob) -> bool:
    return job.kind == "training"


def _execution_priority(job: QueueJob) -> tuple[int, int, str]:
    order = {
        "partition": 0,
        "source-method": 1,
        "training": 2,
        "rank": 3,
        "evaluation": 4,
    }
    return order[job.kind], -int(job.reservation_bytes or 0), job.job_id


def _queue_counts(store: SimpleJobStore, planned: frozenset[str]) -> Mapping[str, int]:
    jobs = tuple(job for job in store.jobs() if job.job_id in planned)
    return {status: sum(job.status == status for job in jobs) for status in STATUSES}


def run_scheduler(
    experiment: AssumptionExperiment,
    *,
    root: str | Path | None = None,
    poll_seconds: float = 2.0,
    headroom_fraction: float = 0.05,
) -> Mapping[str, int]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if not 0.0 <= headroom_fraction < 0.5:
        raise ValueError("headroom_fraction must lie in [0,0.5)")

    from .readiness import require_base_inputs_ready, require_runtime_dependencies

    runtime = require_runtime_dependencies(experiment)
    relprop = runtime["relprop"]
    if relprop["required"]:
        print(
            "RELPROP_RUNTIME_READY "
            f"revision={relprop['revision']} source_digest={relprop['source_digest']} "
            f"repository={relprop['repository']}",
            flush=True,
        )
    readiness = require_base_inputs_ready(experiment)
    print(
        "IND_TABLE_BASE_INPUTS_READY "
        f"base={readiness['base_experiment_id']} "
        f"phase1_digest={readiness['base_phase1_digest']}",
        flush=True,
    )
    print(
        "IND_TABLE_MATCHED_POLICY "
        "selection=all sources_per_family=11 "
        f"partition_families={experiment.partition_families}",
        flush=True,
    )

    paths = table_priority_paths(experiment, root=root)
    store = SimpleJobStore(
        paths.database,
        experiment_digest=table_priority_digest(experiment, paths),
    )
    store.recover_orphans()
    plan_counts = submit_plan(experiment, store, paths=paths)
    planned = planned_job_ids(experiment)
    print(
        "IND_TABLE_PRIORITY_PLAN "
        f"jobs={sum(plan_counts.values())} database={paths.database} "
        f"headroom_fraction={headroom_fraction:.6f}",
        flush=True,
    )

    probe = NvidiaSmiProbe(cache_seconds=0.1)
    gpu_probe_error: GpuProbeError | None = None
    try:
        initial: GpuSnapshot | None = probe.snapshot(force=True)
    except GpuProbeError as error:
        gpu_probe_error = error
        initial = None
    if initial is None:
        # No usable NVIDIA driver/nvidia-smi: degrade to a single serial
        # worker and skip GPU admission control entirely.
        print(
            "WARNING GPU telemetry unavailable "
            f"({gpu_probe_error}); running serially without GPU admission control",
            flush=True,
        )
        oversized: list[str] = []
    else:
        capacities = {
            gpu_id: initial.device(gpu_id).total_bytes
            - int(initial.device(gpu_id).total_bytes * headroom_fraction)
            for gpu_id in experiment.runtime.gpu_ids
        }
        largest_capacity = max(capacities.values())
        oversized = [
            job.job_id
            for job in store.jobs(status="pending")
            if job.job_id in planned
            and job.reservation_bytes is not None
            and job.reservation_bytes > largest_capacity
        ]
    if oversized:
        raise RuntimeError(f"Table-priority reservations cannot fit a GPU: {oversized[:5]}")

    running: dict[str, RunningProcess] = {}
    last_heartbeat = 0.0
    while True:
        for job_id, active in tuple(running.items()):
            exit_code = active.process.poll()
            if exit_code is None:
                if not active.gpu_released and gpu_release_signal_matches(
                    active.release_marker,
                    job_id=active.job.job_id,
                    pid=active.process.pid,
                    token=active.release_token,
                ):
                    active.gpu_released = True
                    store.record_event(
                        job_id,
                        "gpu_released",
                        f"gpu={active.job.gpu_id},admission=awaiting_nvml",
                    )
                    print(
                        f"GPU_RELEASED job={job_id} gpu={active.job.gpu_id} "
                        "publication_tail=active admission=awaiting_nvml",
                        flush=True,
                    )
                continue
            active.log_handle.close()
            active.release_marker.unlink(missing_ok=True)
            store.finish(job_id, exit_code=exit_code)
            del running[job_id]
        store.recover_orphans()
        now = time.monotonic()
        if now - last_heartbeat >= 30.0:
            for job_id in running:
                store.heartbeat(job_id)
            last_heartbeat = now
        store.block_failed_dependents()
        if gpu_probe_error is None:
            _refresh_gpu_release_admission(store, running, probe)
        scoped = tuple(job for job in store.jobs() if job.job_id in planned)
        if not any(job.status in {"pending", "running"} for job in scoped):
            break

        try:
            snapshot = probe.snapshot(force=True)
            configured = {gpu_id: snapshot.device(gpu_id) for gpu_id in experiment.runtime.gpu_ids}
            gpu_probe_error = None
        except GpuProbeError as error:
            if gpu_probe_error is None:
                print(
                    "WARNING GPU telemetry unavailable "
                    f"({error}); running serially without GPU admission control",
                    flush=True,
                )
            gpu_probe_error = error
            configured = {}
        ready = sorted(
            _ready_jobs(
                store,
                runnable_kinds=TABLE_PRIORITY_JOB_KINDS,
                runnable_job_ids=planned,
            ),
            key=_execution_priority,
        )
        launched = False
        for job in ready:
            if job.job_id in running or job.reservation_bytes is None:
                continue
            if gpu_probe_error is not None:
                if running:
                    break
                running[job.job_id] = _launch(
                    store,
                    job,
                    gpu_id=-1,
                    reservation_bytes=job.reservation_bytes,
                    signal_directory=experiment.storage.spool_root / ".signals",
                )
                print(
                    f"SCHEDULED job={job.job_id} gpu=-1 "
                    "reservation_gib=0.00 mode=cpu-serial-fallback",
                    flush=True,
                )
                launched = True
                continue
            snapshot = probe.snapshot(force=True)
            candidates = []
            for gpu_id, device in configured.items():
                active = [item for item in running.values() if item.job.gpu_id == gpu_id]
                if _gpu_release_blocks_admission(active):
                    continue
                holders = [item for item in active if not item.gpu_admission_released]
                if _requires_exclusive_gpu(job) and holders:
                    continue
                if any(_requires_exclusive_gpu(item.job) for item in holders):
                    continue
                outstanding = 0
                for item in active:
                    memory = probe.process_memory_bytes(item.process.pid, (gpu_id,)).get(gpu_id, 0)
                    outstanding += _outstanding_reservation(
                        reservation_bytes=item.reservation_bytes,
                        observed_process_bytes=memory,
                        gpu_admission_released=item.gpu_admission_released,
                    )
                headroom = int(device.total_bytes * headroom_fraction)
                live_free = snapshot.device(gpu_id).free_bytes
                if _reservation_fits(
                    job_kind=job.kind,
                    reservation_bytes=job.reservation_bytes,
                    live_free_bytes=live_free,
                    outstanding_bytes=outstanding,
                    headroom_bytes=headroom,
                ):
                    available = live_free - outstanding - headroom
                    candidates.append((available - job.reservation_bytes, gpu_id))
            if not candidates:
                continue
            _, gpu_id = min(candidates)
            running[job.job_id] = _launch(
                store,
                job,
                gpu_id=gpu_id,
                reservation_bytes=job.reservation_bytes,
                signal_directory=experiment.storage.spool_root / ".signals",
            )
            print(
                f"SCHEDULED job={job.job_id} gpu={gpu_id} "
                f"reservation_gib={job.reservation_bytes / 2**30:.2f}",
                flush=True,
            )
            launched = True
        if not launched:
            time.sleep(poll_seconds)
    return _queue_counts(store, planned)


def scheduler_status(
    experiment: AssumptionExperiment,
    *,
    root: str | Path | None = None,
) -> Mapping[str, Any]:
    paths = table_priority_paths(experiment, root=root)
    if not paths.database.is_file():
        return {"database": str(paths.database), "exists": False, "counts": {}}
    store = SimpleJobStore(
        paths.database,
        experiment_digest=table_priority_digest(experiment, paths),
    )
    planned = planned_job_ids(experiment)
    jobs = tuple(job for job in store.jobs() if job.job_id in planned)
    return {
        "schema": TABLE_PRIORITY_SCHEMA,
        "database": str(paths.database),
        "exists": True,
        "planned_job_count": len(planned),
        "counts": {status: sum(job.status == status for job in jobs) for status in STATUSES},
        "by_kind": {
            kind: {
                status: sum(job.kind == kind and job.status == status for job in jobs)
                for status in STATUSES
            }
            for kind in sorted(TABLE_PRIORITY_JOB_KINDS)
        },
        "running": [
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "pid": job.pid,
                "gpu_id": job.gpu_id,
                "log_path": job.log_path,
            }
            for job in jobs
            if job.status == "running"
        ],
        "failed_or_blocked": [job.job_id for job in jobs if job.status in {"failed", "blocked"}],
    }


def retry_failed(
    experiment: AssumptionExperiment,
    job_ids: tuple[str, ...],
    *,
    root: str | Path | None = None,
) -> Mapping[str, Any]:
    paths = table_priority_paths(experiment, root=root)
    store = SimpleJobStore(
        paths.database,
        experiment_digest=table_priority_digest(experiment, paths),
    )
    store.recover_orphans()
    return {
        **dict(store.retry_failed(job_ids)),
        "status": scheduler_status(experiment, root=root),
    }


__all__ = [
    "TABLE_PRIORITY_JOB_KINDS",
    "TABLE_PRIORITY_SCHEMA",
    "assigned_method_requirements",
    "ind_evaluation_tasks",
    "ind_rank_tasks",
    "matched_evaluation_tasks",
    "matched_source_ids",
    "planned_evaluation_tasks",
    "planned_job_ids",
    "planned_rank_tasks",
    "require_matched_evaluations_complete",
    "retry_failed",
    "run_scheduler",
    "scheduler_status",
    "source_method_job_id",
    "submit_plan",
    "table_priority_digest",
    "table_priority_paths",
]
