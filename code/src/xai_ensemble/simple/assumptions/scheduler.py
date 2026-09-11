"""Independent memory-aware SQLite scheduler for assumption experiments."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, GpuSnapshot, NvidiaSmiProbe
from xai_ensemble.core.hashing import object_sha256

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
    completed_selection_manifest,
    completed_task_manifest,
)
from .config import AssumptionExperiment

JOB_KINDS = frozenset(
    {"partition", "spearman", "training", "source-phase1", "selection", "rank", "evaluation"}
)
CPU_RESERVATION_BYTES = 1 * 2**30


def _effective_headroom_fraction(configured: float, override: float | None) -> float:
    value = configured if override is None else override
    if not 0.0 <= value < 0.5:
        raise ValueError("headroom_fraction must lie in [0,0.5)")
    return value


def _job_id(kind: str, task_id: str) -> str:
    return f"{kind}:{task_id}"


def spearman_job_id(experiment: AssumptionExperiment) -> str:
    """Bind the producer row to the immutable Spearman family identity."""

    from .prepare import spearman_identity

    return _job_id("spearman", f"p196--{object_sha256(spearman_identity(experiment))[:12]}")


def _requires_exclusive_gpu(job: QueueJob) -> bool:
    return job.kind in {"training", "source-phase1"}


def _is_noise_job(job: QueueJob) -> bool:
    return job.kind in {"spearman", "selection"} or "--oracle-noise--" in job.job_id


def _noise_barrier_active(jobs: tuple[QueueJob, ...]) -> bool:
    return any(_is_noise_job(job) and job.status in {"pending", "running"} for job in jobs)


def _failed_noise_jobs(jobs: tuple[QueueJob, ...]) -> tuple[str, ...]:
    return tuple(
        job.job_id for job in jobs if _is_noise_job(job) and job.status in {"failed", "blocked"}
    )


def _execution_priority(job: QueueJob) -> int:
    """Finish reusable Oracle NOISE work before generating IND source banks."""

    if job.kind in {"spearman", "selection"}:
        return 0
    if "--oracle-noise--" in job.job_id:
        return 1
    if _requires_exclusive_gpu(job):
        return 2
    return 3


def _spearman_complete(experiment: AssumptionExperiment) -> bool:
    from .prepare import completed_spearman_family

    return completed_spearman_family(experiment, restore=True) is not None


def _complete(
    experiment: AssumptionExperiment,
    kind: str,
    task: Any,
) -> bool:
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
    if kind == "source-phase1":
        from .phase1 import source_scope_complete

        return source_scope_complete(experiment, task)
    if kind == "selection":
        return completed_selection_manifest(experiment, task) is not None
    if kind == "rank":
        return completed_rank_manifest(experiment, task) is not None
    if kind == "evaluation":
        return completed_evaluation_manifest(experiment, task) is not None
    raise KeyError(kind)


def planned_job_ids(experiment: AssumptionExperiment) -> frozenset[str]:
    values = [
        *(_job_id("partition", task.task_id) for task in experiment.partition_tasks()),
        *(_job_id("training", task.task_id) for task in experiment.training_tasks()),
        *(_job_id("source-phase1", task.task_id) for task in experiment.source_phase1_tasks()),
        *(_job_id("selection", task.task_id) for task in experiment.selection_tasks()),
        *(_job_id("rank", task.task_id) for task in experiment.rank_tasks()),
        *(_job_id("evaluation", task.task_id) for task in experiment.evaluation_tasks()),
    ]
    if experiment.uses_oracle_noise:
        values.append(spearman_job_id(experiment))
    return frozenset(values)


def _command(
    experiment: AssumptionExperiment,
    action: str,
    *,
    task_id: str | None = None,
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
    ]
    if task_id is not None:
        values.extend(("--task-id", task_id))
    return tuple(values)


def submit_plan(
    experiment: AssumptionExperiment,
    store: SimpleJobStore,
    *,
    scan_existing: bool,
) -> Mapping[str, int]:
    counts = {kind: 0 for kind in JOB_KINDS}
    jobs: list[QueueJob] = []
    existing_job_ids = frozenset(job.job_id for job in store.jobs())

    def initial_status(kind: str, task: Any, job_id: str) -> str:
        if not scan_existing or job_id in existing_job_ids:
            return "pending"
        return "succeeded" if _complete(experiment, kind, task) else "pending"

    partitions = experiment.partition_tasks()
    for task in partitions:
        job_id = _job_id("partition", task.task_id)
        job = QueueJob(
            job_id,
            "partition",
            _command(experiment, "prepare", task_id=task.task_id),
            (),
            (),
            CPU_RESERVATION_BYTES,
            initial_status("partition", task, job_id),
            0,
            experiment.runtime.max_retries,
            None,
            None,
            str(experiment.runtime.log_directory / f"partition--{task.task_id}.log"),
        )
        jobs.append(job)
        counts["partition"] += 1

    if experiment.uses_oracle_noise:
        spearman_id = spearman_job_id(experiment)
        jobs.append(
            QueueJob(
                spearman_id,
                "spearman",
                _command(experiment, "prepare", task_id="spearman-p196"),
                (),
                (),
                CPU_RESERVATION_BYTES,
                (
                    "succeeded"
                    if scan_existing
                    and spearman_id not in existing_job_ids
                    and _spearman_complete(experiment)
                    else "pending"
                ),
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(experiment.runtime.log_directory / f"{spearman_id.replace(':', '--')}.log"),
            )
        )
        counts["spearman"] += 1

    partition_jobs = {task.task_id: _job_id("partition", task.task_id) for task in partitions}
    training = experiment.training_tasks()
    for task in training:
        reservation = experiment.runtime.training_reservation_bytes[
            task.cell.reference_model.architecture
        ]
        job_id = _job_id("training", task.task_id)
        job = QueueJob(
            job_id,
            "training",
            _command(experiment, "train", task_id=task.task_id),
            (partition_jobs[task.partition_task_id],),
            (),
            reservation,
            initial_status("training", task, job_id),
            0,
            experiment.runtime.max_retries,
            None,
            None,
            str(experiment.runtime.log_directory / f"training--{task.task_id}.log"),
        )
        jobs.append(job)
        counts["training"] += 1
    training_jobs = {task.task_id: _job_id("training", task.task_id) for task in training}

    source_tasks = experiment.source_phase1_tasks()
    for task in source_tasks:
        reservation = experiment.runtime.phase1_reservation_bytes[
            task.cell.reference_model.architecture
        ]
        job_id = _job_id("source-phase1", task.task_id)
        job = QueueJob(
            job_id,
            "source-phase1",
            _command(experiment, "phase1", task_id=task.task_id),
            (training_jobs[task.training_task_id],),
            (),
            reservation,
            initial_status("source-phase1", task, job_id),
            0,
            experiment.runtime.max_retries,
            None,
            None,
            str(experiment.runtime.log_directory / f"source-phase1--{task.task_id}.log"),
        )
        jobs.append(job)
        counts["source-phase1"] += 1
    source_jobs = {
        (task.cell.cell_id, task.source_id, task.condition.condition_id): _job_id(
            "source-phase1", task.task_id
        )
        for task in source_tasks
    }

    selection = experiment.selection_tasks()
    for task in selection:
        dependencies = (spearman_job_id(experiment),) if task.distance_model == "spearman" else ()
        reservation = experiment.runtime.selection_reservation_bytes
        job_id = _job_id("selection", task.task_id)
        job = QueueJob(
            job_id,
            "selection",
            _command(experiment, "select", task_id=task.task_id),
            dependencies,
            (),
            reservation,
            initial_status("selection", task, job_id),
            0,
            experiment.runtime.max_retries,
            None,
            None,
            str(experiment.runtime.log_directory / f"selection--{task.task_id}.log"),
        )
        jobs.append(job)
        counts["selection"] += 1
    selection_jobs = {task.task_id: _job_id("selection", task.task_id) for task in selection}

    ranks = experiment.rank_tasks()
    for task in ranks:
        dependencies = []
        if task.setting == "oracle-noise":
            assert task.selection_task_id is not None
            dependencies.append(selection_jobs[task.selection_task_id])
        else:
            dependencies.extend(
                source_jobs[task.cell.cell_id, source_id, task.condition.condition_id]
                for source_id in {source_id for source_id, _ in task.source_method_pairs}
            )
        reservation = experiment.runtime.rank_reservation_bytes
        job_id = _job_id("rank", task.task_id)
        job = QueueJob(
            job_id,
            "rank",
            _command(experiment, "rank", task_id=task.task_id),
            tuple(sorted(dependencies)),
            (),
            reservation,
            initial_status("rank", task, job_id),
            0,
            experiment.runtime.max_retries,
            None,
            None,
            str(experiment.runtime.log_directory / f"rank--{task.task_id}.log"),
        )
        jobs.append(job)
        counts["rank"] += 1
    rank_jobs = {task.task_id: _job_id("rank", task.task_id) for task in ranks}

    evaluations = experiment.evaluation_tasks()
    clean_jobs: dict[tuple[str, str, str | None, str | None], str] = {}
    for task in evaluations:
        if task.condition.kind == "clean":
            clean_jobs[(task.cell.cell_id, task.setting, task.source_id, task.distance_model)] = (
                _job_id("evaluation", task.task_id)
            )
    for task in evaluations:
        dependencies = [rank_jobs[task.rank_task_id]]
        if task.condition.kind != "clean":
            dependencies.append(
                clean_jobs[(task.cell.cell_id, task.setting, task.source_id, task.distance_model)]
            )
        reservation = experiment.runtime.evaluation_reservation_bytes
        job_id = _job_id("evaluation", task.task_id)
        job = QueueJob(
            job_id,
            "evaluation",
            _command(experiment, "evaluate", task_id=task.task_id),
            tuple(dependencies),
            (),
            reservation,
            initial_status("evaluation", task, job_id),
            0,
            experiment.runtime.max_retries,
            None,
            None,
            str(experiment.runtime.log_directory / f"evaluation--{task.task_id}.log"),
        )
        jobs.append(job)
        counts["evaluation"] += 1
    store.submit_many(jobs)
    store.update_reservations(
        {job.job_id: job.reservation_bytes for job in jobs if job.reservation_bytes is not None}
    )
    return counts


def run_scheduler(
    experiment: AssumptionExperiment,
    *,
    poll_seconds: float = 2.0,
    headroom_fraction: float | None = None,
) -> Mapping[str, int]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    effective_headroom = _effective_headroom_fraction(
        experiment.runtime.headroom_fraction, headroom_fraction
    )
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
        "ASSUMPTIONS_BASE_INPUTS_READY "
        f"base={readiness['base_experiment_id']} phase1_digest={readiness['base_phase1_digest']}",
        flush=True,
    )
    print(
        "ASSUMPTIONS_SCHEDULER_POLICY "
        f"configured_headroom_fraction={experiment.runtime.headroom_fraction:.6f} "
        f"effective_headroom_fraction={effective_headroom:.6f} "
        f"runtime_override={headroom_fraction is not None} "
        f"settings={','.join(experiment.settings)} "
        "execution_order=noise-first",
        flush=True,
    )
    database_existed = experiment.runtime.database_path.is_file()
    store = SimpleJobStore(
        experiment.runtime.database_path, experiment_digest=experiment.scheduler_digest
    )
    store.recover_orphans()
    submit_plan(experiment, store, scan_existing=database_existed)
    planned = planned_job_ids(experiment)
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
            - int(initial.device(gpu_id).total_bytes * effective_headroom)
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
        raise RuntimeError(
            f"Planned reservations cannot fit any configured GPU after headroom: {oversized[:5]}"
        )
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
        failed_noise = _failed_noise_jobs(scoped)
        if failed_noise and not _noise_barrier_active(scoped):
            raise RuntimeError(
                "Oracle NOISE terminated with failed or blocked jobs; refusing to release the "
                f"IND barrier: {list(failed_noise[:8])}"
            )
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
        ready = list(_ready_jobs(store, runnable_kinds=JOB_KINDS, runnable_job_ids=planned))
        if _noise_barrier_active(scoped):
            ready = [job for job in ready if _is_noise_job(job)]
        ready.sort(
            key=lambda job: (
                _execution_priority(job),
                -int(job.reservation_bytes or 0),
                job.job_id,
            )
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
                exclusive = _requires_exclusive_gpu(job)
                if exclusive and holders:
                    continue
                if any(_requires_exclusive_gpu(item.job) for item in holders):
                    continue
                observed = 0
                outstanding = 0
                for item in active:
                    memory = probe.process_memory_bytes(item.process.pid, (gpu_id,)).get(gpu_id, 0)
                    observed += memory
                    outstanding += _outstanding_reservation(
                        reservation_bytes=item.reservation_bytes,
                        observed_process_bytes=memory,
                        gpu_admission_released=item.gpu_admission_released,
                    )
                headroom = int(device.total_bytes * effective_headroom)
                live_free = snapshot.device(gpu_id).free_bytes
                if _reservation_fits(
                    job_kind=job.kind,
                    reservation_bytes=job.reservation_bytes,
                    live_free_bytes=live_free,
                    outstanding_bytes=outstanding,
                    headroom_bytes=headroom,
                ):
                    available = live_free - outstanding - headroom
                    candidates.append((available - job.reservation_bytes, gpu_id, observed))
            if not candidates:
                continue
            _, gpu_id, _ = min(candidates)
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
    return {
        status: sum(job.status == status for job in store.jobs() if job.job_id in planned)
        for status in STATUSES
    }


def scheduler_status(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {"database": str(experiment.runtime.database_path), "exists": False, "counts": {}}
    store = SimpleJobStore(
        experiment.runtime.database_path, experiment_digest=experiment.scheduler_digest
    )
    planned = planned_job_ids(experiment)
    jobs = tuple(job for job in store.jobs() if job.job_id in planned)
    return {
        "database": str(experiment.runtime.database_path),
        "exists": True,
        "planned_job_count": len(planned),
        "counts": {status: sum(job.status == status for job in jobs) for status in STATUSES},
        "by_kind": {
            kind: {
                status: sum(job.kind == kind and job.status == status for job in jobs)
                for status in STATUSES
            }
            for kind in sorted(JOB_KINDS)
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


__all__ = [
    "JOB_KINDS",
    "planned_job_ids",
    "run_scheduler",
    "scheduler_status",
    "spearman_job_id",
    "submit_plan",
]
