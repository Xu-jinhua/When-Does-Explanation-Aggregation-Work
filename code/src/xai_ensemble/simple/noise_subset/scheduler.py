"""Memory-aware DAG scheduler for random NOISE subset controls."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, NvidiaSmiProbe

from ..noise_prefix.inputs import load_input_catalog
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
from .artifacts import completed_evaluation_manifest, completed_selection_manifest
from .config import (
    NoiseSubsetEvaluationTask,
    NoiseSubsetExperiment,
    NoiseSubsetSelectionTask,
)

SELECTION_KIND = "noise-random-subset-selection"
EVALUATION_KIND = "noise-random-subset-evaluation"
JOB_KINDS = frozenset({SELECTION_KIND, EVALUATION_KIND})


def _selection_job_id(task: NoiseSubsetSelectionTask) -> str:
    return f"{SELECTION_KIND}:{task.task_id}"


def _evaluation_job_id(task: NoiseSubsetEvaluationTask) -> str:
    return f"{EVALUATION_KIND}:{task.task_id}"


def planned_job_ids(experiment: NoiseSubsetExperiment) -> frozenset[str]:
    return frozenset(
        [*(_selection_job_id(task) for task in experiment.selection_tasks())]
        + [*(_evaluation_job_id(task) for task in experiment.evaluation_tasks())]
    )


def _selection_command(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "simple",
        "noise-subset",
        "select",
        "--config",
        str(experiment.source_path),
        "--task-id",
        task.task_id,
    )


def _evaluation_command(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "simple",
        "noise-subset",
        "evaluate",
        "--config",
        str(experiment.source_path),
        "--task-id",
        task.task_id,
    )


def _reservation(experiment: NoiseSubsetExperiment, kind: str) -> int:
    if kind == SELECTION_KIND:
        return experiment.runtime.selection_reservation_bytes
    if kind == EVALUATION_KIND:
        return experiment.runtime.evaluation_reservation_bytes
    raise KeyError(kind)


def submit_plan(
    experiment: NoiseSubsetExperiment,
    store: SimpleJobStore,
    *,
    scan_existing: bool,
) -> Mapping[str, int]:
    existing = frozenset(job.job_id for job in store.jobs())
    jobs = []
    selection_jobs = {}
    for task in experiment.selection_tasks():
        job_id = _selection_job_id(task)
        selection_jobs[task.task_id] = job_id
        status = "pending"
        if (
            scan_existing
            and job_id not in existing
            and completed_selection_manifest(experiment, task) is not None
        ):
            status = "succeeded"
        jobs.append(
            QueueJob(
                job_id=job_id,
                kind=SELECTION_KIND,
                command=_selection_command(experiment, task),
                dependencies=(),
                resource_ids=(),
                reservation_bytes=experiment.runtime.selection_reservation_bytes,
                status=status,
                attempts=0,
                max_retries=experiment.runtime.max_retries,
                pid=None,
                gpu_id=None,
                log_path=str(experiment.runtime.log_directory / f"{task.task_id}.log"),
            )
        )
    for task in experiment.evaluation_tasks():
        job_id = _evaluation_job_id(task)
        status = "pending"
        if (
            scan_existing
            and job_id not in existing
            and completed_evaluation_manifest(experiment, task) is not None
        ):
            status = "succeeded"
        jobs.append(
            QueueJob(
                job_id=job_id,
                kind=EVALUATION_KIND,
                command=_evaluation_command(experiment, task),
                dependencies=(selection_jobs[task.selection_task_id],),
                resource_ids=(),
                reservation_bytes=experiment.runtime.evaluation_reservation_bytes,
                status=status,
                attempts=0,
                max_retries=experiment.runtime.max_retries,
                pid=None,
                gpu_id=None,
                log_path=str(experiment.runtime.log_directory / f"{task.task_id}.log"),
            )
        )
    result = store.submit_many(jobs)
    store.update_reservations({job.job_id: int(job.reservation_bytes or 0) for job in jobs})
    return {**result, "planned": len(jobs)}


def run_scheduler(
    experiment: NoiseSubsetExperiment,
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
    catalog = load_input_catalog(experiment.prefix)
    print(
        "NOISE_RANDOM_INPUTS_READY "
        f"catalog_digest={catalog['catalog_digest']} selector={experiment.selector_digest} "
        f"jobs={len(planned_job_ids(experiment))}",
        flush=True,
    )
    existed = experiment.runtime.database_path.is_file()
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
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
                raise RuntimeError(f"random NOISE subset DAG failed: {failed[:8]}")
            break

        ready = sorted(
            _ready_jobs(store, runnable_kinds=JOB_KINDS, runnable_job_ids=planned),
            key=lambda job: job.job_id,
        )
        launched = False
        for job in ready:
            if job.job_id in running:
                continue
            reservation = _reservation(experiment, job.kind)
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
                    reservation_bytes=reservation,
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
                device = snapshot.device(gpu_id)
                active = [entry for entry in running.values() if entry.job.gpu_id == gpu_id]
                if _gpu_release_blocks_admission(active):
                    continue
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
                headroom_bytes = int(device.total_bytes * headroom)
                if not _reservation_fits(
                    job_kind=job.kind,
                    reservation_bytes=reservation,
                    live_free_bytes=device.free_bytes,
                    outstanding_bytes=outstanding,
                    headroom_bytes=headroom_bytes,
                ):
                    continue
                remaining = device.free_bytes - outstanding - headroom_bytes - reservation
                candidates.append((remaining, gpu_id))
            if not candidates:
                continue
            # This audit has exactly two independent geometry evaluations.
            # Prefer the emptiest admissible device so they occupy both GPUs
            # instead of packing onto one card while the other stays idle.
            _, gpu_id = max(candidates)
            running[job.job_id] = _launch(
                store,
                job,
                gpu_id=gpu_id,
                reservation_bytes=reservation,
                signal_directory=experiment.storage.spool_root / ".signals",
            )
            print(
                f"SCHEDULED job={job.job_id} gpu={gpu_id} "
                f"reservation_gib={reservation / 2**30:.2f}",
                flush=True,
            )
            launched = True
        if not launched:
            time.sleep(poll_seconds)
    return {
        status: sum(job.status == status for job in store.jobs() if job.job_id in planned)
        for status in STATUSES
    }


def scheduler_status(experiment: NoiseSubsetExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {
            "database": str(experiment.runtime.database_path),
            "exists": False,
            "planned_job_count": len(planned_job_ids(experiment)),
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
