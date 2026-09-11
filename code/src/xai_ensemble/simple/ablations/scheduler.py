"""Independent SQLite scheduler for additive ablation jobs."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, NvidiaSmiProbe

from ..adversarial import completed_adversarial_manifest
from ..artifacts import PHASE1_SCHEMA_VERSION, completed_manifest, phase1_artifact_root
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
from .artifacts import completed_evaluation_manifest, completed_rank_manifest, output_store
from .config import AblationExperiment, EvaluationSpec

JOB_KINDS = frozenset({"adversarial", "phase1", "rank", "evaluation"})
EXCLUSIVE_METHODS = frozenset({"FeatureAblation", "Occlusion"})
ADVERSARIAL_RESERVATION_BYTES = 8 * 2**30


def _job_id(kind: str, task_id: str) -> str:
    return f"{kind}:{task_id}"


def _report_plan_scan(kind: str, index: int, total: int) -> None:
    if index == 1 or index == total or index % 10 == 0:
        print(f"ABLATION_PLAN_SCAN kind={kind} progress={index}/{total}", flush=True)


def _requires_exclusive_gpu(job: QueueJob) -> bool:
    return job.kind == "adversarial" or (
        job.kind == "phase1" and bool(set(job.resource_ids) & EXCLUSIVE_METHODS)
    )


def _phase1_complete(experiment: AblationExperiment, task: Any) -> bool:
    store = output_store(experiment)
    for variant in task.variants:
        root = phase1_artifact_root(task, variant.artifact_name)
        if (
            completed_manifest(
                store,
                root,
                expected_task_digest=task.digest,
                expected_schema_version=PHASE1_SCHEMA_VERSION,
            )
            is None
        ):
            return False
    return True


def _profile_reservation(experiment: AblationExperiment, task: Any) -> int:
    generation = experiment.generation_experiment()
    profiles = {profile.profile_id: profile for profile in generation.profiles()}
    cap = experiment.runtime.phase1_batch_caps.get(task.family)
    values = []
    for profile_id in task.profile_ids:
        result = load_profile(generation, profiles[profile_id])
        if result is None:
            raise FileNotFoundError(f"Required existing batch profile is missing: {profile_id}")
        if cap is None:
            values.append(result.reservation_bytes)
            continue
        matches = [
            measurement
            for measurement in result.measurements
            if measurement.batch_size == cap and measurement.passed
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Profile {profile_id} has no unique successful measurement at cap {cap}"
            )
        measurement = matches[0]
        assert measurement.peak_allocated_bytes is not None
        assert measurement.peak_reserved_bytes is not None
        values.append(max(measurement.peak_allocated_bytes, measurement.peak_reserved_bytes))
    floor = experiment.runtime.phase1_reservation_floors.get(task.family, 0)
    return max(*values, floor)


def _clean_dependency(
    experiment: AblationExperiment,
    task: EvaluationSpec,
) -> str | None:
    if task.condition.kind == "clean":
        return None
    matches = [
        candidate
        for candidate in experiment.evaluation_tasks()
        if candidate.table_id == task.table_id
        and candidate.parameter_value == task.parameter_value
        and candidate.condition.kind == "clean"
        and candidate.k == task.k
        and candidate.fill == task.fill
    ]
    if len(matches) > 1:
        raise RuntimeError("Multiple clean evaluation dependencies match one task")
    return None if not matches else _job_id("evaluation", matches[0].task_id)


def planned_job_ids(experiment: AblationExperiment) -> frozenset[str]:
    return frozenset(
        [
            *(_job_id("adversarial", task.task_id) for task in experiment.adversarial_tasks()),
            *(_job_id("phase1", task.task_id) for task in experiment.phase1_tasks()),
            *(_job_id("rank", task.task_id) for task in experiment.rank_tasks()),
            *(_job_id("evaluation", task.task_id) for task in experiment.evaluation_tasks()),
        ]
    )


def submit_plan(
    experiment: AblationExperiment,
    store: SimpleJobStore,
) -> Mapping[str, int]:
    config = str(experiment.source_path)
    generation = experiment.generation_experiment()
    counts = {kind: 0 for kind in JOB_KINDS}
    attack_tasks = experiment.adversarial_tasks()
    attacks = {task.condition.condition_id: task for task in attack_tasks}
    for index, task in enumerate(attack_tasks, start=1):
        _report_plan_scan("adversarial", index, len(attack_tasks))
        complete = completed_adversarial_manifest(generation, task)
        job = QueueJob(
            job_id=_job_id("adversarial", task.task_id),
            kind="adversarial",
            command=(
                sys.executable,
                "-m",
                "xai_ensemble.cli",
                "simple",
                "ablation",
                "adversarial",
                "--config",
                config,
                "--task-id",
                task.task_id,
            ),
            dependencies=(),
            resource_ids=(),
            reservation_bytes=ADVERSARIAL_RESERVATION_BYTES,
            status="succeeded" if complete is not None else "pending",
            attempts=0,
            max_retries=experiment.runtime.max_retries,
            pid=None,
            gpu_id=None,
            log_path=str(experiment.runtime.log_directory / f"adversarial--{task.task_id}.log"),
        )
        store.submit(job)
        store.update_reservation(job.job_id, ADVERSARIAL_RESERVATION_BYTES)
        counts["adversarial"] += 1
    phase1 = experiment.phase1_tasks()
    for index, task in enumerate(phase1, start=1):
        _report_plan_scan("phase1", index, len(phase1))
        dependencies = ()
        if task.condition.kind == "adversarial":
            dependencies = (_job_id("adversarial", attacks[task.condition.condition_id].task_id),)
        reservation = _profile_reservation(experiment, task)
        job = QueueJob(
            job_id=_job_id("phase1", task.task_id),
            kind="phase1",
            command=(
                sys.executable,
                "-m",
                "xai_ensemble.cli",
                "simple",
                "ablation",
                "phase1",
                "--config",
                config,
                "--task-id",
                task.task_id,
            ),
            dependencies=dependencies,
            resource_ids=(task.family,),
            reservation_bytes=reservation,
            status="succeeded" if _phase1_complete(experiment, task) else "pending",
            attempts=0,
            max_retries=experiment.runtime.max_retries,
            pid=None,
            gpu_id=None,
            log_path=str(experiment.runtime.log_directory / f"phase1--{task.task_id}.log"),
        )
        store.submit(job)
        store.update_reservation(job.job_id, reservation)
        counts["phase1"] += 1
    phase1_ids = {task.task_id: _job_id("phase1", task.task_id) for task in phase1}
    rank_tasks = experiment.rank_tasks()
    for index, task in enumerate(rank_tasks, start=1):
        _report_plan_scan("rank", index, len(rank_tasks))
        job = QueueJob(
            job_id=_job_id("rank", task.task_id),
            kind="rank",
            command=(
                sys.executable,
                "-m",
                "xai_ensemble.cli",
                "simple",
                "ablation",
                "rank",
                "--config",
                config,
                "--task-id",
                task.task_id,
            ),
            dependencies=tuple(phase1_ids[value] for value in task.phase1_task_ids),
            resource_ids=(),
            reservation_bytes=experiment.runtime.rank_reservation_bytes,
            status=(
                "succeeded" if completed_rank_manifest(experiment, task) is not None else "pending"
            ),
            attempts=0,
            max_retries=experiment.runtime.max_retries,
            pid=None,
            gpu_id=None,
            log_path=str(experiment.runtime.log_directory / f"rank--{task.task_id}.log"),
        )
        store.submit(job)
        store.update_reservation(job.job_id, experiment.runtime.rank_reservation_bytes)
        counts["rank"] += 1
    rank_ids = {task.condition.condition_id: _job_id("rank", task.task_id) for task in rank_tasks}
    evaluation_tasks = experiment.evaluation_tasks()
    for index, task in enumerate(evaluation_tasks, start=1):
        _report_plan_scan("evaluation", index, len(evaluation_tasks))
        dependencies = []
        if task.rank_source.kind == "constructed":
            dependencies.append(rank_ids[task.condition.condition_id])
        clean_dependency = _clean_dependency(experiment, task)
        if clean_dependency is not None:
            dependencies.append(clean_dependency)
        job = QueueJob(
            job_id=_job_id("evaluation", task.task_id),
            kind="evaluation",
            command=(
                sys.executable,
                "-m",
                "xai_ensemble.cli",
                "simple",
                "ablation",
                "evaluate",
                "--config",
                config,
                "--task-id",
                task.task_id,
            ),
            dependencies=tuple(dependencies),
            resource_ids=(),
            reservation_bytes=experiment.runtime.evaluation_reservation_bytes,
            status=(
                "succeeded"
                if completed_evaluation_manifest(experiment, task) is not None
                else "pending"
            ),
            attempts=0,
            max_retries=experiment.runtime.max_retries,
            pid=None,
            gpu_id=None,
            log_path=str(experiment.runtime.log_directory / f"evaluation--{task.task_id}.log"),
        )
        store.submit(job)
        store.update_reservation(job.job_id, experiment.runtime.evaluation_reservation_bytes)
        counts["evaluation"] += 1
    return counts


def run_scheduler(
    experiment: AblationExperiment,
    *,
    poll_seconds: float = 2.0,
) -> Mapping[str, int]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    store.recover_orphans()
    submit_plan(experiment, store)
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
            # No usable NVIDIA driver/nvidia-smi: degrade to a single serial
            # worker and skip GPU admission control entirely.
            if gpu_probe_error is None:
                print(
                    "WARNING GPU telemetry unavailable "
                    f"({error}); running serially without GPU admission control",
                    flush=True,
                )
            gpu_probe_error = error
            configured = {}
        ready = list(_ready_jobs(store, runnable_kinds=JOB_KINDS, runnable_job_ids=planned))
        ready.sort(
            key=lambda job: (
                0 if _requires_exclusive_gpu(job) else 1,
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
                    signal_directory=experiment.output_storage.spool_root / ".signals",
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
                active = [value for value in running.values() if value.job.gpu_id == gpu_id]
                if _gpu_release_blocks_admission(active):
                    continue
                holders = [value for value in active if not value.gpu_admission_released]
                exclusive = _requires_exclusive_gpu(job)
                if exclusive and holders:
                    continue
                if any(_requires_exclusive_gpu(value.job) for value in holders):
                    continue
                observed = 0
                outstanding = 0
                for value in active:
                    memory = probe.process_memory_bytes(value.process.pid, (gpu_id,)).get(gpu_id, 0)
                    observed += memory
                    outstanding += _outstanding_reservation(
                        reservation_bytes=value.reservation_bytes,
                        observed_process_bytes=memory,
                        gpu_admission_released=value.gpu_admission_released,
                    )
                headroom = int(device.total_bytes * experiment.runtime.headroom_fraction)
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
                signal_directory=experiment.output_storage.spool_root / ".signals",
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


def scheduler_status(experiment: AblationExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {
            "database": str(experiment.runtime.database_path),
            "exists": False,
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
    "planned_job_ids",
    "run_scheduler",
    "scheduler_status",
    "submit_plan",
]
