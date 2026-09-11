"""Small profile-aware SQLite scheduler for the two local GPUs."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, NvidiaSmiProbe

from .config import SimpleExperiment
from .profiler import load_phase2_profile, load_profile
from .runtime import (
    GPU_RELEASE_JOB_ENV,
    GPU_RELEASE_PATH_ENV,
    GPU_RELEASE_TOKEN_ENV,
    gpu_release_signal_matches,
)

STATUSES = ("pending", "running", "succeeded", "failed", "blocked")
PROFILE_JOB_KINDS = frozenset({"profile", "phase2_profile"})
COMPUTE_EXCLUSIVE_JOB_KINDS = frozenset({"adversarial"})
EXCLUSIVE_JOB_KINDS = PROFILE_JOB_KINDS | COMPUTE_EXCLUSIVE_JOB_KINDS
PHASE1_RUN_JOB_KINDS = frozenset({"profile", "adversarial", "phase1"})
PHASE2_RUN_JOB_KINDS = frozenset({"phase2_profile", "phase2"})
# Kept as a public compatibility name for existing scheduler diagnostics/tests.
EXCLUSIVE_PROFILE_KINDS = PROFILE_JOB_KINDS
ADVERSARIAL_RESERVATION_BYTES = {"cnn": 8 * 2**30, "vit": 10 * 2**30}
_EXCLUSIVE_PHASE1_PROFILE_MARKERS = ("--FeatureAblation__", "--Occlusion__")
# A release marker means the worker has finished its CUDA work, but its process
# can still retain a large CUDA allocator pool during asynchronous publication.
# Do not reuse the GPU until NVML has observed that pool at a small baseline in
# two scheduler passes.
# A released worker may retain its CUDA context and allocator pool while its
# publication tail drains.  Treat up to 1 GiB as the measured idle baseline;
# larger residual usage keeps the physical GPU reserved.
GPU_RELEASE_ADMISSION_FLOOR_BYTES = 1 * 2**30
GPU_RELEASE_ADMISSION_CONFIRMATIONS = 2


@dataclass(frozen=True, slots=True)
class QueueJob:
    job_id: str
    kind: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    resource_ids: tuple[str, ...]
    reservation_bytes: int | None
    status: str
    attempts: int
    max_retries: int
    pid: int | None
    gpu_id: int | None
    log_path: str


@dataclass(slots=True)
class RunningProcess:
    job: QueueJob
    process: subprocess.Popen[bytes]
    log_handle: Any
    reservation_bytes: int
    release_marker: Path
    release_token: str
    gpu_released: bool = False
    gpu_admission_released: bool = False
    gpu_release_safe_observations: int = 0

    @property
    def gpu_release_pending_admission(self) -> bool:
        """Whether a publication tail must still reserve its physical GPU."""

        return self.gpu_released and not self.gpu_admission_released


def _process_state(pid: int) -> tuple[str, int] | None:
    """Return the Linux process state and parent PID when procfs can identify it."""

    if pid <= 0:
        return None
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, separator, suffix = value.rpartition(")")
    if not separator:
        return None
    fields = suffix.split()
    if len(fields) < 2:
        return None
    try:
        return fields[0], int(fields[1])
    except ValueError:
        return None


def _process_is_zombie(pid: int) -> bool:
    """Return whether Linux retains ``pid`` only as an unreaped child.

    ``kill(pid, 0)`` succeeds for zombies, even though they cannot execute or
    hold a GPU.  An unreadable or unexpected procfs record is deliberately
    treated as inconclusive so callers retain their existing conservative
    behavior.
    """

    state = _process_state(pid)
    return state is not None and state[0] == "Z"


def _requires_exclusive_gpu(job: QueueJob) -> bool:
    if job.kind in EXCLUSIVE_JOB_KINDS:
        return True
    return job.kind == "phase1" and any(
        marker in profile_id
        for profile_id in job.resource_ids
        for marker in _EXCLUSIVE_PHASE1_PROFILE_MARKERS
    )


class SimpleJobStore:
    def __init__(self, path: str | Path, *, experiment_digest: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    resource_ids_json TEXT NOT NULL,
                    reservation_bytes INTEGER,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL,
                    pid INTEGER,
                    gpu_id INTEGER,
                    log_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    lease_expires_at REAL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    detail TEXT
                );
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='experiment_digest'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('experiment_digest',?)",
                    (experiment_digest,),
                )
            elif row[0] != experiment_digest:
                raise ValueError("Scheduler database belongs to a different experiment digest")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _job(row: sqlite3.Row) -> QueueJob:
        return QueueJob(
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            command=tuple(json.loads(row["command_json"])),
            dependencies=tuple(json.loads(row["dependencies_json"])),
            resource_ids=tuple(json.loads(row["resource_ids_json"])),
            reservation_bytes=(
                None if row["reservation_bytes"] is None else int(row["reservation_bytes"])
            ),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            max_retries=int(row["max_retries"]),
            pid=None if row["pid"] is None else int(row["pid"]),
            gpu_id=None if row["gpu_id"] is None else int(row["gpu_id"]),
            log_path=str(row["log_path"]),
        )

    @staticmethod
    def _matches_plan(observed: QueueJob, planned: QueueJob) -> bool:
        return (
            observed.command == planned.command
            and observed.dependencies == planned.dependencies
            and observed.resource_ids == planned.resource_ids
            and observed.kind == planned.kind
        )

    def submit_many(self, jobs: Sequence[QueueJob]) -> Mapping[str, int]:
        """Validate and insert a complete plan in one SQLite transaction."""

        planned = tuple(jobs)
        if not planned:
            return {"submitted": 0, "already_present": 0}
        job_ids = tuple(job.job_id for job in planned)
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Submitted queue job ids must be unique")
        job_id_set = frozenset(job_ids)

        now = time.time()
        with self.connect() as connection:
            existing = {
                str(row["job_id"]): self._job(row)
                for row in connection.execute("SELECT * FROM jobs").fetchall()
                if str(row["job_id"]) in job_id_set
            }
            for job in planned:
                observed = existing.get(job.job_id)
                if observed is not None and not self._matches_plan(observed, job):
                    raise ValueError(f"Existing queue job contradicts plan: {job.job_id}")

            pending = tuple(job for job in planned if job.job_id not in existing)
            connection.executemany(
                """
                INSERT INTO jobs(
                    job_id,kind,command_json,dependencies_json,resource_ids_json,
                    reservation_bytes,status,attempts,max_retries,pid,gpu_id,
                    log_path,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                tuple(
                    (
                        job.job_id,
                        job.kind,
                        json.dumps(job.command),
                        json.dumps(job.dependencies),
                        json.dumps(job.resource_ids),
                        job.reservation_bytes,
                        job.status,
                        job.attempts,
                        job.max_retries,
                        job.pid,
                        job.gpu_id,
                        job.log_path,
                        now,
                    )
                    for job in pending
                ),
            )
            connection.executemany(
                "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                tuple((job.job_id, "submitted", now, None) for job in pending),
            )
        return {"submitted": len(pending), "already_present": len(planned) - len(pending)}

    def submit(self, job: QueueJob) -> None:
        self.submit_many((job,))

    def update_reservations(self, reservations: Mapping[str, int]) -> None:
        if any(value <= 0 for value in reservations.values()):
            raise ValueError("reservation_bytes must be positive")
        with self.connect() as connection:
            connection.executemany(
                "UPDATE jobs SET reservation_bytes=? WHERE job_id=? AND status='pending'",
                tuple((value, job_id) for job_id, value in reservations.items()),
            )

    def jobs(self, *, status: str | None = None) -> tuple[QueueJob, ...]:
        with self.connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at,job_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at,job_id",
                    (status,),
                ).fetchall()
        return tuple(self._job(row) for row in rows)

    def update_reservation(self, job_id: str, reservation_bytes: int) -> None:
        self.update_reservations({job_id: reservation_bytes})

    def start(self, job_id: str, *, pid: int, gpu_id: int) -> None:
        now = time.time()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs SET status='running',attempts=attempts+1,pid=?,gpu_id=?,
                    started_at=?,finished_at=NULL,lease_expires_at=?,error=NULL
                WHERE job_id=? AND status='pending'
                """,
                (pid, gpu_id, now, now + 120.0, job_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Could not claim pending job {job_id}")
            connection.execute(
                "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                (job_id, "started", now, f"pid={pid},gpu={gpu_id}"),
            )

    def heartbeat(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=? AND status='running'",
                (time.time() + 120.0, job_id),
            )

    def record_event(self, job_id: str, event: str, detail: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                (job_id, event, time.time(), detail),
            )

    def finish(self, job_id: str, *, exit_code: int) -> None:
        now = time.time()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempts,max_retries FROM jobs WHERE job_id=? AND status='running'",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Running job vanished from queue: {job_id}")
            if exit_code == 0:
                status = "succeeded"
            elif int(row["attempts"]) <= int(row["max_retries"]):
                status = "pending"
            else:
                status = "failed"
            connection.execute(
                """
                UPDATE jobs SET status=?,pid=NULL,gpu_id=NULL,finished_at=?,
                    lease_expires_at=NULL,error=? WHERE job_id=?
                """,
                (
                    status,
                    now,
                    None if exit_code == 0 else f"exit_code={exit_code}",
                    job_id,
                ),
            )
            connection.execute(
                "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                (job_id, status, now, f"exit_code={exit_code}"),
            )

    def recover_orphans(self) -> None:
        for job in self.jobs(status="running"):
            pid = job.pid
            alive = False
            if pid is not None:
                try:
                    os.kill(pid, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
            state = _process_state(pid) if alive and pid is not None else None
            external_zombie = state is not None and state[0] == "Z" and state[1] != os.getpid()
            # Popen.poll() owns reaping a local child.  Between polling loops,
            # that child may briefly be a zombie but must remain running in the
            # queue until its owning scheduler records the exit status.
            if alive and not external_zombie:
                continue
            with self.connect() as connection:
                connection.execute(
                    """
                    UPDATE jobs SET status='pending',pid=NULL,gpu_id=NULL,
                        lease_expires_at=NULL,error='scheduler_orphan_recovery'
                    WHERE job_id=? AND status='running'
                    """,
                    (job.job_id,),
                )

    def block_failed_dependents(self) -> int:
        jobs = self.jobs()
        statuses = {job.job_id: job.status for job in jobs}
        blocked = [
            job.job_id
            for job in jobs
            if job.status == "pending"
            and any(
                statuses.get(dependency) in {"failed", "blocked"} for dependency in job.dependencies
            )
        ]
        if not blocked:
            return 0
        now = time.time()
        with self.connect() as connection:
            for job_id in blocked:
                connection.execute(
                    "UPDATE jobs SET status='blocked',finished_at=? WHERE job_id=?",
                    (now, job_id),
                )
                connection.execute(
                    "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                    (job_id, "blocked", now, "failed dependency"),
                )
        return len(blocked)

    def retry_failed(self, job_ids: tuple[str, ...]) -> Mapping[str, int]:
        """Retry explicit failed roots and reopen descendants blocked by them."""

        if not job_ids or len(set(job_ids)) != len(job_ids):
            raise ValueError("retry job ids must be non-empty and unique")
        jobs = self.jobs()
        by_id = {job.job_id: job for job in jobs}
        unknown = set(job_ids) - set(by_id)
        if unknown:
            raise KeyError(f"Unknown retry jobs: {sorted(unknown)}")
        invalid = {
            job_id: by_id[job_id].status for job_id in job_ids if by_id[job_id].status != "failed"
        }
        if invalid:
            raise ValueError(f"Only failed jobs can be retried: {invalid}")

        statuses = {job.job_id: job.status for job in jobs}
        for job_id in job_ids:
            statuses[job_id] = "pending"
        unblocked = []
        changed = True
        while changed:
            changed = False
            for job in jobs:
                if statuses[job.job_id] != "blocked":
                    continue
                if all(
                    statuses.get(dependency) not in {"failed", "blocked"}
                    for dependency in job.dependencies
                ):
                    statuses[job.job_id] = "pending"
                    unblocked.append(job.job_id)
                    changed = True

        now = time.time()
        with self.connect() as connection:
            for job_id in job_ids:
                connection.execute(
                    """
                    UPDATE jobs SET status='pending',attempts=0,pid=NULL,gpu_id=NULL,
                        started_at=NULL,finished_at=NULL,lease_expires_at=NULL,error=NULL
                    WHERE job_id=? AND status='failed'
                    """,
                    (job_id,),
                )
                connection.execute(
                    "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                    (job_id, "manual_retry", now, "attempts reset after code repair"),
                )
            for job_id in unblocked:
                connection.execute(
                    """
                    UPDATE jobs SET status='pending',finished_at=NULL,error=NULL
                    WHERE job_id=? AND status='blocked'
                    """,
                    (job_id,),
                )
                connection.execute(
                    "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                    (job_id, "unblocked", now, "failed dependency retried"),
                )
        return {"retried": len(job_ids), "unblocked": len(unblocked)}

    def requeue_jobs(self, job_ids: tuple[str, ...], *, reason: str) -> Mapping[str, int]:
        """Requeue explicit succeeded jobs so a contract change can re-run them."""

        if not job_ids or len(set(job_ids)) != len(job_ids):
            raise ValueError("requeue job ids must be non-empty and unique")
        if not reason.strip():
            raise ValueError("requeue reason must be non-empty")
        jobs = self.jobs()
        by_id = {job.job_id: job for job in jobs}
        unknown = set(job_ids) - set(by_id)
        if unknown:
            raise KeyError(f"Unknown requeue jobs: {sorted(unknown)}")
        invalid = {
            job_id: by_id[job_id].status
            for job_id in job_ids
            if by_id[job_id].status != "succeeded"
        }
        if invalid:
            raise ValueError(f"Only succeeded jobs can be requeued: {invalid}")

        now = time.time()
        with self.connect() as connection:
            for job_id in job_ids:
                connection.execute(
                    """
                    UPDATE jobs SET status='pending',attempts=0,pid=NULL,gpu_id=NULL,
                        started_at=NULL,finished_at=NULL,lease_expires_at=NULL,error=NULL
                    WHERE job_id=? AND status='succeeded'
                    """,
                    (job_id,),
                )
                connection.execute(
                    "INSERT INTO events(job_id,event,created_at,detail) VALUES(?,?,?,?)",
                    (job_id, "requeued", now, reason),
                )
        return {"requeued": len(job_ids)}


def _profile_job_id(profile_id: str) -> str:
    return f"profile:{profile_id}"


def _phase2_profile_job_id(profile_id: str) -> str:
    return f"phase2_profile:{profile_id}"


def _adversarial_job_id(task_id: str) -> str:
    return f"adversarial:{task_id}"


def _phase1_job_id(task_id: str) -> str:
    return f"phase1:{task_id}"


def _phase2_job_id(task_id: str) -> str:
    return f"phase2:{task_id}"


def _planned_job_ids(
    experiment: SimpleExperiment,
    *,
    include_phase2: bool,
) -> frozenset[str]:
    result = {
        *(_profile_job_id(profile.profile_id) for profile in experiment.profiles()),
        *(_adversarial_job_id(task.task_id) for task in experiment.adversarial_tasks()),
        *(_phase1_job_id(task.task_id) for task in experiment.phase1_tasks()),
    }
    if include_phase2:
        result.update(
            _phase2_profile_job_id(profile.profile_id) for profile in experiment.phase2_profiles()
        )
        result.update(_phase2_job_id(task.task_id) for task in experiment.phase2_tasks())
    return frozenset(result)


def submit_plan(
    experiment: SimpleExperiment,
    store: SimpleJobStore,
    *,
    include_phase2: bool,
) -> Mapping[str, int]:
    """Idempotently submit profiles, attacked datasets, Phase 1, and Phase 2."""

    config = str(experiment.source_path)
    counts = {"profile": 0, "phase2_profile": 0, "phase1": 0, "phase2": 0}
    for profile in experiment.profiles():
        existing = load_profile(experiment, profile)
        status = "succeeded" if existing is not None else "pending"
        job = QueueJob(
            job_id=_profile_job_id(profile.profile_id),
            kind="profile",
            command=(
                sys.executable,
                "-m",
                "xai_ensemble.cli",
                "simple",
                "profile",
                "--config",
                config,
                "--profile-id",
                profile.profile_id,
            ),
            dependencies=(),
            resource_ids=(),
            reservation_bytes=None,
            status=status,
            attempts=0,
            max_retries=experiment.runtime.max_retries,
            pid=None,
            gpu_id=None,
            log_path=str(experiment.runtime.log_directory / f"profile--{profile.profile_id}.log"),
        )
        store.submit(job)
        counts["profile"] += 1
    if include_phase2:
        for profile in experiment.phase2_profiles():
            existing = load_phase2_profile(experiment, profile)
            status = "succeeded" if existing is not None else "pending"
            job = QueueJob(
                job_id=_phase2_profile_job_id(profile.profile_id),
                kind="phase2_profile",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "simple",
                    "phase2-profile",
                    "--config",
                    config,
                    "--profile-id",
                    profile.profile_id,
                ),
                dependencies=(),
                resource_ids=(),
                reservation_bytes=None,
                status=status,
                attempts=0,
                max_retries=experiment.runtime.max_retries,
                pid=None,
                gpu_id=None,
                log_path=str(
                    experiment.runtime.log_directory / f"phase2-profile--{profile.profile_id}.log"
                ),
            )
            store.submit(job)
            counts["phase2_profile"] += 1
    adversarial_tasks = experiment.adversarial_tasks()
    if adversarial_tasks:
        from .adversarial import completed_adversarial_manifest

        counts["adversarial"] = 0
        for task in adversarial_tasks:
            existing = completed_adversarial_manifest(experiment, task)
            job = QueueJob(
                job_id=_adversarial_job_id(task.task_id),
                kind="adversarial",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "simple",
                    "adversarial",
                    "--config",
                    config,
                    "--task-id",
                    task.task_id,
                ),
                dependencies=(),
                resource_ids=(),
                reservation_bytes=ADVERSARIAL_RESERVATION_BYTES[task.model.architecture],
                status="succeeded" if existing is not None else "pending",
                attempts=0,
                max_retries=experiment.runtime.max_retries,
                pid=None,
                gpu_id=None,
                log_path=str(experiment.runtime.log_directory / f"adversarial--{task.task_id}.log"),
            )
            store.submit(job)
            counts["adversarial"] += 1
    for task in experiment.phase1_tasks():
        dependencies_list = [_profile_job_id(item) for item in task.profile_ids]
        if task.condition.kind == "adversarial":
            attack_task = experiment.adversarial_task_for(
                dataset_id=task.dataset.dataset_id,
                model_id=task.model.model_id,
                split=task.split,
                condition_id=task.condition.condition_id,
            )
            dependencies_list.append(_adversarial_job_id(attack_task.task_id))
        dependencies = tuple(dependencies_list)
        job = QueueJob(
            job_id=_phase1_job_id(task.task_id),
            kind="phase1",
            command=(
                sys.executable,
                "-m",
                "xai_ensemble.cli",
                "simple",
                "phase1",
                "--config",
                config,
                "--task-id",
                task.task_id,
            ),
            dependencies=dependencies,
            resource_ids=task.profile_ids,
            reservation_bytes=None,
            status="pending",
            attempts=0,
            max_retries=experiment.runtime.max_retries,
            pid=None,
            gpu_id=None,
            log_path=str(experiment.runtime.log_directory / f"phase1--{task.task_id}.log"),
        )
        store.submit(job)
        counts["phase1"] += 1
    if include_phase2:
        phase1_tasks = experiment.phase1_tasks()
        phase2_tasks = experiment.phase2_tasks()
        for task in phase2_tasks:
            inference_profile = experiment.phase2_profile_for_model(task.model)
            dependencies_list = [
                _phase1_job_id(item.task_id)
                for item in phase1_tasks
                if item.dataset.dataset_id == task.dataset.dataset_id
                and item.model.model_id == task.model.model_id
                and item.split == task.split
                and item.condition.condition_id == task.condition.condition_id
            ]
            dependencies_list.append(_phase2_profile_job_id(inference_profile.profile_id))
            if task.condition.kind != "clean":
                clean_matches = [
                    item
                    for item in phase2_tasks
                    if item.condition.kind == "clean"
                    and item.dataset.dataset_id == task.dataset.dataset_id
                    and item.model.model_id == task.model.model_id
                    and item.split == task.split
                    and item.ensemble.ensemble_id == task.ensemble.ensemble_id
                    and item.patch_size == task.patch_size
                ]
                if len(clean_matches) != 1:
                    raise RuntimeError("Cannot resolve one clean Phase 2 dependency")
                dependencies_list.append(_phase2_job_id(clean_matches[0].task_id))
            dependencies = tuple(dependencies_list)
            job = QueueJob(
                job_id=_phase2_job_id(task.task_id),
                kind="phase2",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "simple",
                    "phase2",
                    "--config",
                    config,
                    "--task-id",
                    task.task_id,
                ),
                dependencies=dependencies,
                resource_ids=(inference_profile.profile_id,),
                reservation_bytes=None,
                status="pending",
                attempts=0,
                max_retries=experiment.runtime.max_retries,
                pid=None,
                gpu_id=None,
                log_path=str(experiment.runtime.log_directory / f"phase2--{task.task_id}.log"),
            )
            store.submit(job)
            counts["phase2"] += 1
    return counts


def _profiles_by_id(experiment: SimpleExperiment) -> Mapping[str, tuple[str, Any]]:
    result: dict[str, tuple[str, Any]] = {
        profile.profile_id: ("phase1", profile) for profile in experiment.profiles()
    }
    result.update(
        {profile.profile_id: ("phase2", profile) for profile in experiment.phase2_profiles()}
    )
    return result


def _resolve_reservation(
    experiment: SimpleExperiment,
    job: QueueJob,
    *,
    device_total_bytes: int,
) -> int | None:
    if job.kind in PROFILE_JOB_KINDS:
        # The short probes need an otherwise idle device so their peaks are
        # attributable. Formal Phase 1 and Phase 2 jobs use the persisted peaks.
        return int(device_total_bytes * (1.0 - experiment.runtime.headroom_fraction))
    if job.kind in COMPUTE_EXCLUSIVE_JOB_KINDS:
        return job.reservation_bytes
    profiles = _profiles_by_id(experiment)
    values = []
    for profile_id in job.resource_ids:
        profile_kind, profile = profiles[profile_id]
        result = (
            load_profile(experiment, profile)
            if profile_kind == "phase1"
            else load_phase2_profile(experiment, profile)
        )
        if result is None:
            return None
        values.append(result.reservation_bytes)
    return max(values) if values else None


def _ready_jobs(
    store: SimpleJobStore,
    *,
    runnable_kinds: frozenset[str] | None = None,
    runnable_job_ids: frozenset[str] | None = None,
) -> tuple[QueueJob, ...]:
    jobs = store.jobs()
    statuses = {job.job_id: job.status for job in jobs}
    return tuple(
        job
        for job in jobs
        if job.status == "pending"
        and (runnable_kinds is None or job.kind in runnable_kinds)
        and (runnable_job_ids is None or job.job_id in runnable_job_ids)
        and all(statuses.get(dependency) == "succeeded" for dependency in job.dependencies)
    )


def _reservation_fits(
    *,
    job_kind: str,
    reservation_bytes: int,
    live_free_bytes: int,
    outstanding_bytes: int,
    headroom_bytes: int,
) -> bool:
    if job_kind in PROFILE_JOB_KINDS:
        # Exclusive profiles already reserve (1 - headroom) of the whole device.
        # Subtracting headroom again would require a completely unused GPU,
        # including memory that the driver always occupies.
        return reservation_bytes <= live_free_bytes
    return reservation_bytes <= live_free_bytes - outstanding_bytes - headroom_bytes


def _launch(
    store: SimpleJobStore,
    job: QueueJob,
    *,
    gpu_id: int,
    reservation_bytes: int,
    signal_directory: Path,
    environment: Mapping[str, str] | None = None,
) -> RunningProcess:
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    child_environment = dict(os.environ)
    if environment is not None:
        child_environment.update(environment)
    child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    child_environment["PYTHONUNBUFFERED"] = "1"
    signal_directory.mkdir(parents=True, exist_ok=True)
    release_token = uuid.uuid4().hex
    release_marker = signal_directory / f"{release_token}.json"
    child_environment[GPU_RELEASE_PATH_ENV] = str(release_marker)
    child_environment[GPU_RELEASE_TOKEN_ENV] = release_token
    child_environment[GPU_RELEASE_JOB_ENV] = job.job_id
    try:
        process = subprocess.Popen(
            job.command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=child_environment,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        release_marker.unlink(missing_ok=True)
        raise
    store.start(job.job_id, pid=process.pid, gpu_id=gpu_id)
    refreshed = next(item for item in store.jobs(status="running") if item.job_id == job.job_id)
    return RunningProcess(
        refreshed,
        process,
        log_handle,
        reservation_bytes,
        release_marker,
        release_token,
    )


def _outstanding_reservation(
    *,
    reservation_bytes: int,
    observed_process_bytes: int,
    gpu_admission_released: bool,
) -> int:
    if gpu_admission_released:
        return 0
    return max(0, reservation_bytes - observed_process_bytes)


def _gpu_release_blocks_admission(active: Sequence[RunningProcess]) -> bool:
    """Return whether an unqualified publication tail owns this GPU."""

    return any(item.gpu_release_pending_admission for item in active)


def _observe_gpu_release_admission(
    active: RunningProcess,
    *,
    observed_process_bytes: int,
) -> str | None:
    """Update one publication-tail worker's NVML-gated admission state.

    A valid worker marker only permits checking for release; it never grants
    immediate admission.  Two consecutive observations at the small CUDA
    context baseline are required, and a later increase revokes admission.
    """

    if not active.gpu_released or active.job.gpu_id is None or active.job.gpu_id < 0:
        return None
    if observed_process_bytes > GPU_RELEASE_ADMISSION_FLOOR_BYTES:
        active.gpu_release_safe_observations = 0
        if active.gpu_admission_released:
            active.gpu_admission_released = False
            return "reblocked"
        return None
    if active.gpu_admission_released:
        return None
    active.gpu_release_safe_observations += 1
    if active.gpu_release_safe_observations < GPU_RELEASE_ADMISSION_CONFIRMATIONS:
        return None
    active.gpu_admission_released = True
    return "released"


def _refresh_gpu_release_admission(
    store: SimpleJobStore,
    running: Mapping[str, RunningProcess],
    probe: NvidiaSmiProbe,
) -> None:
    """Record NVML-qualified publication tails before another job is admitted."""

    candidates = tuple(
        active
        for active in running.values()
        if active.gpu_released and active.job.gpu_id is not None and active.job.gpu_id >= 0
    )
    if not candidates:
        return
    probe.snapshot(force=True)
    for active in candidates:
        assert active.job.gpu_id is not None
        observed = probe.process_memory_bytes(active.process.pid, (active.job.gpu_id,)).get(
            active.job.gpu_id, 0
        )
        transition = _observe_gpu_release_admission(
            active,
            observed_process_bytes=observed,
        )
        if transition is None:
            continue
        detail = json.dumps(
            {
                "baseline_bytes": GPU_RELEASE_ADMISSION_FLOOR_BYTES,
                "gpu": active.job.gpu_id,
                "observed_bytes": observed,
                "pid": active.process.pid,
                "safe_observations": active.gpu_release_safe_observations,
            },
            sort_keys=True,
        )
        event = "gpu_admission_released" if transition == "released" else "gpu_admission_reblocked"
        store.record_event(active.job.job_id, event, detail)
        print(
            f"GPU_ADMISSION_{transition.upper()} job={active.job.job_id} "
            f"gpu={active.job.gpu_id} pid={active.process.pid} "
            f"observed_mib={observed / 2**20:.1f} "
            f"baseline_mib={GPU_RELEASE_ADMISSION_FLOOR_BYTES / 2**20:.1f}",
            flush=True,
        )


def run_scheduler(
    experiment: SimpleExperiment,
    *,
    include_phase2: bool = False,
    poll_seconds: float = 2.0,
) -> Mapping[str, int]:
    """Run profiles first, then pack formal jobs by live and reserved memory."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    from .runtime_dependencies import require_relprop_runtime

    runtime = require_relprop_runtime(
        (task.family, task.model.architecture) for task in experiment.phase1_tasks()
    )
    if runtime["required"]:
        print(
            "RELPROP_RUNTIME_READY "
            f"revision={runtime['revision']} source_digest={runtime['source_digest']} "
            f"repository={runtime['repository']}",
            flush=True,
        )
    store = SimpleJobStore(
        experiment.runtime.database_path, experiment_digest=experiment.scheduler_digest
    )
    store.recover_orphans()
    submit_plan(experiment, store, include_phase2=include_phase2)
    runnable_job_ids = _planned_job_ids(experiment, include_phase2=include_phase2)
    runnable_kinds = PHASE1_RUN_JOB_KINDS
    if include_phase2:
        runnable_kinds |= PHASE2_RUN_JOB_KINDS
    print(f"RUN_SCOPE kinds={','.join(sorted(runnable_kinds))}", flush=True)
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
                        f"pid={active.process.pid},gpu={active.job.gpu_id}",
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
        # A scheduler restart cannot recover a Popen exit code.  Once such a
        # child disappears, put its idempotent task back in the queue; an
        # already-published artifact makes the rerun return immediately.
        store.recover_orphans()
        now = time.monotonic()
        if now - last_heartbeat >= 30.0:
            for job_id in running:
                store.heartbeat(job_id)
            last_heartbeat = now
        store.block_failed_dependents()
        if gpu_probe_error is None:
            _refresh_gpu_release_admission(store, running, probe)
        jobs = tuple(
            job
            for job in store.jobs()
            if job.kind in runnable_kinds and job.job_id in runnable_job_ids
        )
        if not any(job.status in {"pending", "running"} for job in jobs):
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
        profile_pending = any(
            job.kind in PROFILE_JOB_KINDS and job.status in {"pending", "running"} for job in jobs
        )
        ready = [
            job
            for job in _ready_jobs(
                store,
                runnable_kinds=runnable_kinds,
                runnable_job_ids=runnable_job_ids,
            )
            if not profile_pending or job.kind in PROFILE_JOB_KINDS
        ]
        # Large commitments first; best-fit placement below then packs smaller jobs
        # into genuine remaining memory without changing their selected batch size.
        resolved: list[tuple[QueueJob, int]] = []
        for job in ready:
            if gpu_probe_error is not None:
                resolved.append((job, job.reservation_bytes))
                continue
            totals = [device.total_bytes for device in configured.values()]
            reservation = _resolve_reservation(experiment, job, device_total_bytes=min(totals))
            if reservation is None:
                continue
            if job.reservation_bytes != reservation:
                store.update_reservation(job.job_id, reservation)
            resolved.append((job, reservation))
        resolved.sort(
            key=lambda item: (
                0 if _requires_exclusive_gpu(item[0]) else 1,
                -item[1],
                item[0].job_id,
            )
        )

        launched = False
        for job, reservation in resolved:
            if job.job_id in running:
                continue
            if gpu_probe_error is not None:
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
                    "reservation_gib=0.00 mode=cpu-serial-fallback"
                )
                launched = True
                continue
            snapshot = probe.snapshot(force=True)
            candidates = []
            for gpu_id, device in configured.items():
                active_on_gpu = [item for item in running.values() if item.job.gpu_id == gpu_id]
                if _gpu_release_blocks_admission(active_on_gpu):
                    continue
                gpu_holders = [item for item in active_on_gpu if not item.gpu_admission_released]
                job_is_exclusive = _requires_exclusive_gpu(job)
                if job_is_exclusive and gpu_holders:
                    continue
                if any(_requires_exclusive_gpu(item.job) for item in gpu_holders):
                    continue
                observed = 0
                outstanding = 0
                for item in active_on_gpu:
                    memory = probe.process_memory_bytes(item.process.pid, (gpu_id,)).get(gpu_id, 0)
                    observed += memory
                    outstanding += _outstanding_reservation(
                        reservation_bytes=item.reservation_bytes,
                        observed_process_bytes=memory,
                        gpu_admission_released=item.gpu_admission_released,
                    )
                headroom = int(device.total_bytes * experiment.runtime.headroom_fraction)
                live_free = snapshot.device(gpu_id).free_bytes
                available = live_free - outstanding - headroom
                if job_is_exclusive and not gpu_holders:
                    if _reservation_fits(
                        job_kind=job.kind,
                        reservation_bytes=reservation,
                        live_free_bytes=live_free,
                        outstanding_bytes=outstanding,
                        headroom_bytes=headroom,
                    ):
                        candidates.append((0, gpu_id, observed, outstanding))
                elif _reservation_fits(
                    job_kind=job.kind,
                    reservation_bytes=reservation,
                    live_free_bytes=live_free,
                    outstanding_bytes=outstanding,
                    headroom_bytes=headroom,
                ):
                    candidates.append((available - reservation, gpu_id, observed, outstanding))
            if not candidates:
                continue
            _, gpu_id, _, _ = min(candidates)
            running[job.job_id] = _launch(
                store,
                job,
                gpu_id=gpu_id,
                reservation_bytes=reservation,
                signal_directory=experiment.storage.spool_root / ".signals",
            )
            print(
                f"SCHEDULED job={job.job_id} gpu={gpu_id} reservation_gib={reservation / 2**30:.2f}"
            )
            launched = True
        if not launched:
            time.sleep(poll_seconds)
    status_counts: dict[str, int] = {status: 0 for status in STATUSES}
    for job in store.jobs():
        if job.kind not in runnable_kinds or job.job_id not in runnable_job_ids:
            continue
        status_counts[job.status] += 1
    return status_counts


def scheduler_status(experiment: SimpleExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {"database": str(experiment.runtime.database_path), "exists": False, "counts": {}}
    store = SimpleJobStore(
        experiment.runtime.database_path, experiment_digest=experiment.scheduler_digest
    )
    planned_job_ids = _planned_job_ids(experiment, include_phase2=True)
    all_jobs = store.jobs()
    jobs = tuple(job for job in all_jobs if job.job_id in planned_job_ids)
    stale = tuple(job for job in all_jobs if job.job_id not in planned_job_ids)
    counts = {status: sum(job.status == status for job in jobs) for status in STATUSES}
    running = [
        {
            "job_id": job.job_id,
            "kind": job.kind,
            "pid": job.pid,
            "gpu_id": job.gpu_id,
            "log_path": job.log_path,
        }
        for job in jobs
        if job.status == "running"
    ]
    failed = [job.job_id for job in jobs if job.status in {"failed", "blocked"}]
    return {
        "database": str(experiment.runtime.database_path),
        "exists": True,
        "planned_job_count": len(planned_job_ids),
        "counts": counts,
        "running": running,
        "failed_or_blocked": failed,
        "stale_job_count": len(stale),
        "stale_by_kind": {
            kind: sum(job.kind == kind for job in stale)
            for kind in sorted({job.kind for job in stale})
        },
    }


__all__ = [
    "QueueJob",
    "SimpleJobStore",
    "run_scheduler",
    "scheduler_status",
    "submit_plan",
]
