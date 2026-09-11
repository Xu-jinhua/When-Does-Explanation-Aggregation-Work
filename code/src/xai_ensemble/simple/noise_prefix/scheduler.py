"""Independent memory-aware scheduler for the formal prefix sweep."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, GpuSnapshot, NvidiaSmiProbe

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
from .artifacts import completed_evaluation_manifest
from .config import NoisePrefixExperiment, PrefixEvaluationTask

JOB_KINDS = frozenset({"noise-prefix-evaluation"})


def _job_id(task: PrefixEvaluationTask) -> str:
    return f"noise-prefix-evaluation:{task.task_id}"


def planned_job_ids(experiment: NoisePrefixExperiment) -> frozenset[str]:
    return frozenset(_job_id(task) for task in experiment.evaluation_tasks())


def _command(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "simple",
        "noise-prefix",
        "evaluate",
        "--config",
        str(experiment.source_path),
        "--task-id",
        task.task_id,
    )


def submit_plan(
    experiment: NoisePrefixExperiment,
    store: SimpleJobStore,
    *,
    scan_existing: bool,
) -> Mapping[str, int]:
    tasks = experiment.evaluation_tasks()
    existing_job_ids = frozenset(job.job_id for job in store.jobs())
    clean_jobs = {
        task.cell.cell_id: _job_id(task) for task in tasks if task.condition.kind == "clean"
    }
    jobs = []
    for task in tasks:
        job_id = _job_id(task)
        status = "pending"
        if (
            scan_existing
            and job_id not in existing_job_ids
            and completed_evaluation_manifest(experiment, task) is not None
        ):
            status = "succeeded"
        dependencies = () if task.condition.kind == "clean" else (clean_jobs[task.cell.cell_id],)
        jobs.append(
            QueueJob(
                job_id,
                "noise-prefix-evaluation",
                _command(experiment, task),
                dependencies,
                (),
                experiment.runtime.evaluation_reservation_bytes,
                status,
                0,
                experiment.runtime.max_retries,
                None,
                None,
                str(experiment.runtime.log_directory / f"{task.task_id}.log"),
            )
        )
    result = store.submit_many(jobs)
    store.update_reservations(
        {job.job_id: experiment.runtime.evaluation_reservation_bytes for job in jobs}
    )
    return {**result, "planned": len(jobs)}


def _status_counts(store: SimpleJobStore, planned: frozenset[str]) -> Mapping[str, int]:
    return {
        status: sum(job.status == status for job in store.jobs() if job.job_id in planned)
        for status in STATUSES
    }


def run_scheduler(
    experiment: NoisePrefixExperiment,
    *,
    poll_seconds: float = 2.0,
    headroom_fraction: float | None = None,
) -> Mapping[str, int]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    headroom = (
        experiment.runtime.headroom_fraction
        if headroom_fraction is None
        else float(headroom_fraction)
    )
    if not 0.0 <= headroom < 0.5:
        raise ValueError("headroom_fraction must lie in [0,0.5)")

    from .inputs import readiness_report

    readiness = readiness_report(experiment)
    print(
        "NOISE_PREFIX_INPUTS_READY "
        f"catalog_digest={readiness['catalog_digest']} tasks={readiness['tasks']} "
        f"sidecars={readiness['required_unique_sidecars']}",
        flush=True,
    )
    print(
        "NOISE_PREFIX_SCHEDULER_POLICY "
        f"reservation_gib={experiment.runtime.evaluation_reservation_bytes / 2**30:.2f} "
        f"headroom_fraction={headroom:.6f}",
        flush=True,
    )
    database_existed = experiment.runtime.database_path.is_file()
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
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
    else:
        largest_capacity = max(
            initial.device(gpu_id).total_bytes - int(initial.device(gpu_id).total_bytes * headroom)
            for gpu_id in experiment.runtime.gpu_ids
        )
        if experiment.runtime.evaluation_reservation_bytes > largest_capacity:
            raise RuntimeError("Prefix evaluation reservation cannot fit any configured GPU")

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
        _refresh_gpu_release_admission(store, running, probe)
        scoped = tuple(job for job in store.jobs() if job.job_id in planned)
        if not any(job.status in {"pending", "running"} for job in scoped):
            failed = [job.job_id for job in scoped if job.status in {"failed", "blocked"}]
            if failed:
                raise RuntimeError(
                    f"NOISE prefix sweep terminated with failed or blocked jobs: {failed[:8]}"
                )
            break

        ready = sorted(
            _ready_jobs(store, runnable_kinds=JOB_KINDS, runnable_job_ids=planned),
            key=lambda job: job.job_id,
        )
        launched = False
        for job in ready:
            if job.job_id in running:
                continue
            if gpu_probe_error is not None:
                if running:
                    break
                running[job.job_id] = _launch(
                    store,
                    job,
                    gpu_id=-1,
                    reservation_bytes=experiment.runtime.evaluation_reservation_bytes,
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
            for gpu_id in experiment.runtime.gpu_ids:
                device = snapshot.device(gpu_id)
                active = [item for item in running.values() if item.job.gpu_id == gpu_id]
                if _gpu_release_blocks_admission(active):
                    continue
                outstanding = 0
                for item in active:
                    memory = probe.process_memory_bytes(item.process.pid, (gpu_id,)).get(gpu_id, 0)
                    outstanding += _outstanding_reservation(
                        reservation_bytes=item.reservation_bytes,
                        observed_process_bytes=memory,
                        gpu_admission_released=item.gpu_admission_released,
                    )
                headroom_bytes = int(device.total_bytes * headroom)
                if _reservation_fits(
                    job_kind=job.kind,
                    reservation_bytes=experiment.runtime.evaluation_reservation_bytes,
                    live_free_bytes=device.free_bytes,
                    outstanding_bytes=outstanding,
                    headroom_bytes=headroom_bytes,
                ):
                    remaining = (
                        device.free_bytes
                        - outstanding
                        - headroom_bytes
                        - experiment.runtime.evaluation_reservation_bytes
                    )
                    candidates.append((remaining, gpu_id))
            if not candidates:
                continue
            _, gpu_id = min(candidates)
            running[job.job_id] = _launch(
                store,
                job,
                gpu_id=gpu_id,
                reservation_bytes=experiment.runtime.evaluation_reservation_bytes,
                signal_directory=experiment.storage.spool_root / ".signals",
            )
            print(
                f"SCHEDULED job={job.job_id} gpu={gpu_id} "
                f"reservation_gib={experiment.runtime.evaluation_reservation_bytes / 2**30:.2f}",
                flush=True,
            )
            launched = True
        if not launched:
            time.sleep(poll_seconds)
    return _status_counts(store, planned)


def scheduler_status(experiment: NoisePrefixExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {
            "database": str(experiment.runtime.database_path),
            "exists": False,
            "planned_job_count": len(experiment.evaluation_tasks()),
            "counts": {},
        }
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    planned = planned_job_ids(experiment)
    jobs = tuple(job for job in store.jobs() if job.job_id in planned)
    return {
        "database": str(experiment.runtime.database_path),
        "exists": True,
        "planned_job_count": len(planned),
        "counts": {status: sum(job.status == status for job in jobs) for status in STATUSES},
        "running": [
            {
                "job_id": job.job_id,
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
    "submit_plan",
]
