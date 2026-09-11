"""Idle-GPU scheduler for the Phase-2-only random-control bank."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, GpuSnapshot, NvidiaSmiProbe

from ..runtime import (
    GPU_RELEASE_JOB_ENV,
    GPU_RELEASE_PATH_ENV,
    GPU_RELEASE_TOKEN_ENV,
    gpu_release_signal_matches,
)
from ..scheduler import (
    STATUSES,
    QueueJob,
    RunningProcess,
    SimpleJobStore,
    _launch,
    _outstanding_reservation,
    _ready_jobs,
    _refresh_gpu_release_admission,
    _reservation_fits,
)
from .artifacts import completed_control_manifest
from .config import RandomControlTask, RelativeRobustnessExperiment

JOB_KIND = "relative-random-control"


def _job_id(task: RandomControlTask) -> str:
    return f"relative-random-control:{task.task_id}"


def planned_job_ids(experiment: RelativeRobustnessExperiment) -> frozenset[str]:
    return frozenset(_job_id(task) for task in experiment.control_tasks())


def _command(experiment: RelativeRobustnessExperiment, task: RandomControlTask) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "simple",
        "relative-robustness",
        "control",
        "--config",
        str(experiment.source_path),
        "--task-id",
        task.task_id,
    )


def submit_plan(
    experiment: RelativeRobustnessExperiment,
    store: SimpleJobStore,
    *,
    scan_existing: bool,
) -> Mapping[str, int]:
    existing = frozenset(job.job_id for job in store.jobs())
    jobs = []
    for task in experiment.control_tasks():
        job_id = _job_id(task)
        status = "pending"
        if (
            scan_existing
            and job_id not in existing
            and completed_control_manifest(experiment, task)
        ):
            status = "succeeded"
        jobs.append(
            QueueJob(
                job_id=job_id,
                kind=JOB_KIND,
                command=_command(experiment, task),
                dependencies=(),
                resource_ids=(),
                reservation_bytes=experiment.runtime.control_reservation_bytes,
                status=status,
                attempts=0,
                max_retries=experiment.runtime.max_retries,
                pid=None,
                gpu_id=None,
                log_path=str(experiment.runtime.log_directory / f"{task.task_id}.log"),
            )
        )
    result = store.submit_many(jobs)
    store.update_reservations(
        {job.job_id: experiment.runtime.control_reservation_bytes for job in jobs}
    )
    return {**result, "planned": len(jobs)}


def _external_compute_processes(
    snapshot: GpuSnapshot,
    *,
    gpu_id: int,
    running: Mapping[str, RunningProcess],
) -> tuple[int, ...]:
    """Return external CUDA PIDs which may still submit GPU work.

    A Phase 1 worker that has emitted the authenticated release signal is in
    its CloudStorage publication tail: its CUDA tensors are gone and it will
    not submit further GPU work.  Its remaining CUDA context is accounted for
    by NVML's live-free-memory reading, so a control may safely use the device
    once every *nonreleased* external worker is gone.
    """

    device = snapshot.device(gpu_id)
    own_roots = {active.process.pid for active in running.values()}
    return tuple(
        process.pid
        for process in snapshot.processes
        if process.gpu_uuid == device.uuid
        and process.pid not in own_roots
        and not _external_process_released_gpu(process.pid)
    )


def _external_process_released_gpu(pid: int) -> bool:
    """Validate an external simple-worker's optional GPU-release marker."""

    try:
        rows = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    environment: dict[str, str] = {}
    for row in rows:
        if b"=" not in row:
            continue
        key, value = row.split(b"=", 1)
        try:
            environment[key.decode("ascii")] = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
    path = environment.get(GPU_RELEASE_PATH_ENV)
    token = environment.get(GPU_RELEASE_TOKEN_ENV)
    job_id = environment.get(GPU_RELEASE_JOB_ENV)
    if not path or not token or not job_id:
        return False
    return gpu_release_signal_matches(path, job_id=job_id, pid=pid, token=token)


def _status_counts(
    store: SimpleJobStore,
    planned: frozenset[str],
) -> Mapping[str, int]:
    return {
        status: sum(job.status == status for job in store.jobs() if job.job_id in planned)
        for status in STATUSES
    }


def run_scheduler(
    experiment: RelativeRobustnessExperiment,
    *,
    poll_seconds: float = 10.0,
    headroom_fraction: float | None = None,
) -> Mapping[str, int]:
    """Run at most one random-control evaluation on each otherwise idle GPU."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    headroom = (
        experiment.runtime.headroom_fraction if headroom_fraction is None else headroom_fraction
    )
    if not 0.0 <= headroom < 0.5:
        raise ValueError("headroom_fraction must lie in [0,0.5)")
    from ..noise_prefix.inputs import load_input_catalog

    catalog = load_input_catalog(experiment.prefix)
    print(
        "RELATIVE_ROBUSTNESS_INPUTS_READY "
        f"catalog_digest={catalog['catalog_digest']} controls={len(experiment.control_tasks())}",
        flush=True,
    )
    print(
        "RELATIVE_ROBUSTNESS_SCHEDULER_POLICY "
        f"reservation_gib={experiment.runtime.control_reservation_bytes / 2**30:.2f} "
        f"headroom_fraction={headroom:.6f} external_compute_policy=wait_for_idle",
        flush=True,
    )
    existed = experiment.runtime.database_path.is_file()
    store = SimpleJobStore(
        experiment.runtime.database_path, experiment_digest=experiment.scheduler_digest
    )
    store.recover_orphans()
    submit_plan(experiment, store, scan_existing=existed)
    planned = planned_job_ids(experiment)
    probe = NvidiaSmiProbe(cache_seconds=0.1)
    gpu_probe_error: GpuProbeError | None = None
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
                raise RuntimeError(f"Relative robustness controls failed: {failed[:8]}")
            break

        ready = sorted(
            _ready_jobs(store, runnable_kinds=frozenset({JOB_KIND}), runnable_job_ids=planned),
            key=lambda job: job.job_id,
        )
        launched = False
        for job in ready:
            if job.job_id in running:
                continue
            try:
                snapshot = probe.snapshot(force=True)
                gpu_probe_error = None
            except GpuProbeError as error:
                # No usable NVIDIA driver/nvidia-smi: degrade to a single
                # serial worker and skip GPU admission control entirely.
                if gpu_probe_error is None:
                    print(
                        "WARNING GPU telemetry unavailable "
                        f"({error}); running serially without GPU admission control",
                        flush=True,
                    )
                gpu_probe_error = error
                if running:
                    break
                running[job.job_id] = _launch(
                    store,
                    job,
                    gpu_id=-1,
                    reservation_bytes=experiment.runtime.control_reservation_bytes,
                    signal_directory=experiment.storage.spool_root / ".signals",
                )
                print(
                    f"SCHEDULED job={job.job_id} gpu=-1 "
                    "reservation_gib=0.00 mode=cpu-serial-fallback",
                    flush=True,
                )
                launched = True
                continue
            candidates = []
            for gpu_id in experiment.runtime.gpu_ids:
                active = [entry for entry in running.values() if entry.job.gpu_id == gpu_id]
                if any(not entry.gpu_admission_released for entry in active):
                    continue
                external = _external_compute_processes(snapshot, gpu_id=gpu_id, running=running)
                if external:
                    continue
                device = snapshot.device(gpu_id)
                outstanding = 0
                for entry in active:
                    observed = probe.process_memory_bytes(entry.process.pid, (gpu_id,)).get(
                        gpu_id, 0
                    )
                    outstanding += _outstanding_reservation(
                        reservation_bytes=entry.reservation_bytes,
                        observed_process_bytes=observed,
                        gpu_admission_released=entry.gpu_admission_released,
                    )
                if not _reservation_fits(
                    job_kind=JOB_KIND,
                    reservation_bytes=experiment.runtime.control_reservation_bytes,
                    live_free_bytes=device.free_bytes,
                    outstanding_bytes=outstanding,
                    headroom_bytes=int(device.total_bytes * headroom),
                ):
                    continue
                remaining = (
                    device.free_bytes
                    - outstanding
                    - int(device.total_bytes * headroom)
                    - experiment.runtime.control_reservation_bytes
                )
                candidates.append((remaining, gpu_id))
            if not candidates:
                continue
            _, gpu_id = min(candidates)
            running[job.job_id] = _launch(
                store,
                job,
                gpu_id=gpu_id,
                reservation_bytes=experiment.runtime.control_reservation_bytes,
                signal_directory=experiment.storage.spool_root / ".signals",
            )
            print(
                f"SCHEDULED job={job.job_id} gpu={gpu_id} "
                f"reservation_gib={experiment.runtime.control_reservation_bytes / 2**30:.2f}",
                flush=True,
            )
            launched = True
        if not launched:
            time.sleep(poll_seconds)
    return _status_counts(store, planned)


def scheduler_status(experiment: RelativeRobustnessExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {
            "database": str(experiment.runtime.database_path),
            "exists": False,
            "planned_job_count": len(experiment.control_tasks()),
            "counts": {},
        }
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


__all__ = ["planned_job_ids", "run_scheduler", "scheduler_status", "submit_plan"]
