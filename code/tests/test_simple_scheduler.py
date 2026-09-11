from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.simple.config import load_experiment
from xai_ensemble.simple.profiler import (
    BatchProfile,
    CandidateMeasurement,
    phase2_profile_identity,
    profile_path,
)
from xai_ensemble.simple.scheduler import (
    EXCLUSIVE_PROFILE_KINDS,
    GPU_RELEASE_ADMISSION_FLOOR_BYTES,
    PHASE1_RUN_JOB_KINDS,
    PHASE2_RUN_JOB_KINDS,
    QueueJob,
    RunningProcess,
    SimpleJobStore,
    _gpu_release_blocks_admission,
    _launch,
    _observe_gpu_release_admission,
    _outstanding_reservation,
    _planned_job_ids,
    _process_is_zombie,
    _ready_jobs,
    _refresh_gpu_release_admission,
    _requires_exclusive_gpu,
    _reservation_fits,
    _resolve_reservation,
    run_scheduler,
    scheduler_status,
    submit_plan,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


def _experiment(tmp_path: Path):
    source = CODE_ROOT / "configs/simple/example.yaml"
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    value["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    value["storage"]["remote_root"] = str(tmp_path / "artifacts")
    value["storage"]["scratch_root"] = str(tmp_path / "scratch")
    value["runtime"]["profile_directory"] = str(tmp_path / "profiles")
    value["runtime"]["database_path"] = str(tmp_path / "jobs.sqlite3")
    value["runtime"]["log_directory"] = str(tmp_path / "logs")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return load_experiment(path)


def test_process_is_zombie_reads_linux_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xai_ensemble.simple.scheduler.Path.read_text",
        lambda *_args, **_kwargs: "123 (scheduler worker) Z 1 2 3",
    )

    assert _process_is_zombie(123) is True


def test_orphan_recovery_requeues_zombie_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    job = QueueJob(
        job_id="zombie-worker",
        kind="profile",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=1,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="zombie-worker.log",
    )
    store.submit(job)
    store.start(job.job_id, pid=1234, gpu_id=0)
    monkeypatch.setattr("xai_ensemble.simple.scheduler.os.kill", lambda *_args: None)
    monkeypatch.setattr(
        "xai_ensemble.simple.scheduler._process_state",
        lambda _pid: ("Z", 999999),
    )

    store.recover_orphans()

    recovered = next(item for item in store.jobs() if item.job_id == job.job_id)
    assert recovered.status == "pending"
    assert recovered.pid is None
    assert recovered.gpu_id is None
    assert recovered.attempts == 1


def test_orphan_recovery_leaves_own_zombie_for_popen_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    job = QueueJob(
        job_id="local-zombie-worker",
        kind="profile",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=1,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="local-zombie-worker.log",
    )
    store.submit(job)
    store.start(job.job_id, pid=1234, gpu_id=0)
    monkeypatch.setattr("xai_ensemble.simple.scheduler.os.kill", lambda *_args: None)
    monkeypatch.setattr(
        "xai_ensemble.simple.scheduler._process_state",
        lambda _pid: ("Z", os.getpid()),
    )

    store.recover_orphans()

    retained = next(item for item in store.jobs() if item.job_id == job.job_id)
    assert retained.status == "running"
    assert retained.pid == 1234
    assert retained.gpu_id == 0


def test_phase2_jobs_depend_on_one_reusable_inference_profile(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )

    counts = submit_plan(experiment, store, include_phase2=True)

    assert counts == {"profile": 30, "phase2_profile": 2, "phase1": 22, "phase2": 2}
    jobs = store.jobs()
    profile_jobs = [job for job in jobs if job.kind == "phase2_profile"]
    phase2_jobs = [job for job in jobs if job.kind == "phase2"]
    assert len(profile_jobs) == 2
    assert len(phase2_jobs) == 2
    assert "phase2" not in EXCLUSIVE_PROFILE_KINDS
    for job in phase2_jobs:
        assert len(job.resource_ids) == 1
        assert f"phase2_profile:{job.resource_ids[0]}" in job.dependencies
    vit_job = next(job for job in phase2_jobs if "--imagenet100-vit-b16--" in job.job_id)
    cnn_job = next(job for job in phase2_jobs if "--imagenet100-resnet18--" in job.job_id)
    assert sum(dependency.startswith("phase1:") for dependency in vit_job.dependencies) == 11
    assert sum(dependency.startswith("phase1:") for dependency in cnn_job.dependencies) == 11


def test_formal_phase2_reservation_uses_measured_peak_not_device_fraction(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    task = experiment.phase2_tasks()[0]
    profile = experiment.phase2_profile_for_model(task.model)
    measured = 5 * 2**30
    result = BatchProfile(
        schema_version=1,
        profile_id=profile.profile_id,
        identity_digest=object_sha256(phase2_profile_identity(profile)),
        model_key=profile.model_key,
        architecture=profile.architecture,
        method="Phase2MaskGame",
        variant="removed_retained",
        params={"forward_batch_size": profile.inference_batch_size},
        precision="fp32",
        input_shape=(3, profile.input_size, profile.input_size),
        selected_batch_size=profile.inference_batch_size,
        peak_allocated_bytes=4 * 2**30,
        peak_reserved_bytes=measured,
        device_total_bytes=48 * 2**30,
        headroom_fraction=experiment.runtime.headroom_fraction,
        probe_kind="phase2_removed_retained_inference",
        measurements=(
            CandidateMeasurement(
                batch_size=profile.inference_batch_size,
                passed=True,
                peak_allocated_bytes=4 * 2**30,
                peak_reserved_bytes=measured,
                elapsed_seconds=0.1,
                reason=None,
            ),
        ),
        created_utc="2026-07-25T00:00:00+00:00",
    )
    atomic_write_json(profile_path(experiment, profile.profile_id), result.to_dict())
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=True)
    job = next(item for item in store.jobs() if item.kind == "phase2")

    reservation = _resolve_reservation(
        experiment,
        job,
        device_total_bytes=48 * 2**30,
    )

    assert reservation == measured


def test_exclusive_profile_does_not_subtract_gpu_headroom_twice(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=True)
    job = next(item for item in store.jobs() if item.kind == "profile")
    total = 48 * 2**30
    driver_baseline = 14 * 2**20
    headroom = int(total * experiment.runtime.headroom_fraction)
    reservation = _resolve_reservation(
        experiment,
        job,
        device_total_bytes=total,
    )

    assert reservation == int(total * (1.0 - experiment.runtime.headroom_fraction))
    assert _reservation_fits(
        job_kind=job.kind,
        reservation_bytes=reservation,
        live_free_bytes=total - driver_baseline,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )
    assert not _reservation_fits(
        job_kind=job.kind,
        reservation_bytes=reservation,
        live_free_bytes=reservation - 1,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )


def test_formal_job_still_reserves_configured_headroom() -> None:
    total = 48 * 2**30
    headroom = int(total * 0.10)

    assert _reservation_fits(
        job_kind="phase1",
        reservation_bytes=total - headroom - 14 * 2**20,
        live_free_bytes=total - 14 * 2**20,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )
    assert not _reservation_fits(
        job_kind="phase1",
        reservation_bytes=total - headroom,
        live_free_bytes=total - 14 * 2**20,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )


def test_gpu_release_marker_requires_nvml_admission_confirmation(tmp_path: Path) -> None:
    assert GPU_RELEASE_ADMISSION_FLOOR_BYTES == 1 * 2**30
    reservation = 36 * 2**30
    observed_context = 300 * 2**20
    job = QueueJob(
        job_id="publication-tail",
        kind="phase1",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=reservation,
        status="running",
        attempts=1,
        max_retries=2,
        pid=4321,
        gpu_id=0,
        log_path=str(tmp_path / "publication-tail.log"),
    )
    active = RunningProcess(
        job,
        SimpleNamespace(pid=4321),
        None,
        reservation,
        tmp_path / "release.json",
        "release-token",
        gpu_released=True,
    )

    assert (
        _outstanding_reservation(
            reservation_bytes=reservation,
            observed_process_bytes=observed_context,
            gpu_admission_released=active.gpu_admission_released,
        )
        == reservation - observed_context
    )
    assert (
        _observe_gpu_release_admission(
            active,
            observed_process_bytes=GPU_RELEASE_ADMISSION_FLOOR_BYTES + 1,
        )
        is None
    )
    assert active.gpu_release_safe_observations == 0
    assert active.gpu_admission_released is False
    assert active.gpu_release_pending_admission is True
    assert _gpu_release_blocks_admission((active,)) is True
    assert (
        _observe_gpu_release_admission(
            active,
            observed_process_bytes=GPU_RELEASE_ADMISSION_FLOOR_BYTES,
        )
        is None
    )
    assert active.gpu_release_safe_observations == 1
    assert active.gpu_admission_released is False
    assert (
        _observe_gpu_release_admission(
            active,
            observed_process_bytes=GPU_RELEASE_ADMISSION_FLOOR_BYTES,
        )
        == "released"
    )
    assert active.gpu_admission_released is True
    assert active.gpu_release_pending_admission is False
    assert _gpu_release_blocks_admission((active,)) is False
    assert (
        _outstanding_reservation(
            reservation_bytes=reservation,
            observed_process_bytes=observed_context,
            gpu_admission_released=active.gpu_admission_released,
        )
        == 0
    )
    assert (
        _observe_gpu_release_admission(
            active,
            observed_process_bytes=GPU_RELEASE_ADMISSION_FLOOR_BYTES + 1,
        )
        == "reblocked"
    )
    assert active.gpu_admission_released is False
    assert active.gpu_release_pending_admission is True
    assert _gpu_release_blocks_admission((active,)) is True


def test_gpu_release_admission_records_only_nvml_qualified_transition(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    job = QueueJob(
        job_id="publication-tail",
        kind="phase1",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=1,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path=str(tmp_path / "publication-tail.log"),
    )
    store.submit(job)
    active = RunningProcess(
        replace(job, status="running", pid=4321, gpu_id=0),
        SimpleNamespace(pid=4321),
        None,
        1,
        tmp_path / "release.json",
        "release-token",
        gpu_released=True,
    )

    class Probe:
        def __init__(self) -> None:
            self.observed = iter(
                (
                    GPU_RELEASE_ADMISSION_FLOOR_BYTES + 1,
                    GPU_RELEASE_ADMISSION_FLOOR_BYTES,
                    GPU_RELEASE_ADMISSION_FLOOR_BYTES,
                )
            )

        def snapshot(self, *, force: bool = False) -> None:
            assert force
            return None

        def process_memory_bytes(self, _pid: int, _gpu_ids: tuple[int, ...]) -> dict[int, int]:
            return {0: next(self.observed)}

    probe = Probe()
    _refresh_gpu_release_admission(store, {job.job_id: active}, probe)
    _refresh_gpu_release_admission(store, {job.job_id: active}, probe)
    assert active.gpu_admission_released is False
    _refresh_gpu_release_admission(store, {job.job_id: active}, probe)
    assert active.gpu_admission_released is True
    with store.connect() as connection:
        events = connection.execute(
            "SELECT event FROM events WHERE job_id=? ORDER BY sequence", (job.job_id,)
        ).fetchall()
    assert [row["event"] for row in events] == ["submitted", "gpu_admission_released"]


def test_launch_merges_worker_environment_and_preserves_release_signal(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    output = tmp_path / "environment.json"
    job = QueueJob(
        job_id="test:environment",
        kind="phase1",
        command=(
            sys.executable,
            "-c",
            "import json, os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text("
            "json.dumps({key: os.environ.get(key) for key in sys.argv[2:]}))",
            str(output),
            "CUDA_VISIBLE_DEVICES",
            "XAI_CLOUD_STORAGE_LOCK_ROOT",
            "XAI_SIMPLE_GPU_RELEASE_PATH",
            "XAI_SIMPLE_GPU_RELEASE_TOKEN",
            "XAI_SIMPLE_GPU_RELEASE_JOB",
        ),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=1,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path=str(tmp_path / "environment.log"),
    )
    store.submit(job)

    running = _launch(
        store,
        job,
        gpu_id=7,
        reservation_bytes=1,
        signal_directory=tmp_path / "signals",
        environment={"XAI_CLOUD_STORAGE_LOCK_ROOT": "/tmp/locked"},
    )
    assert running.process.wait(timeout=10) == 0
    running.log_handle.close()
    store.finish(job.job_id, exit_code=0)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "CUDA_VISIBLE_DEVICES": "7",
        "XAI_CLOUD_STORAGE_LOCK_ROOT": "/tmp/locked",
        "XAI_SIMPLE_GPU_RELEASE_PATH": str(running.release_marker),
        "XAI_SIMPLE_GPU_RELEASE_TOKEN": running.release_token,
        "XAI_SIMPLE_GPU_RELEASE_JOB": job.job_id,
    }


def test_patch_mask_phase1_jobs_are_gpu_exclusive(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=False)
    phase1 = [job for job in store.jobs() if job.kind == "phase1"]
    feature_ablation = next(job for job in phase1 if "--FeatureAblation--" in job.job_id)
    occlusion = next(job for job in phase1 if "--Occlusion--" in job.job_id)
    saliency = next(job for job in phase1 if "--Saliency--" in job.job_id)

    assert _requires_exclusive_gpu(feature_ablation)
    assert _requires_exclusive_gpu(occlusion)
    assert not _requires_exclusive_gpu(saliency)


def test_retry_failed_reopens_blocked_descendants(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=True)
    phase2 = next(job for job in store.jobs() if job.kind == "phase2")
    failed_root = next(
        job for job in store.jobs() if job.kind == "phase1" and job.job_id in phase2.dependencies
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE jobs SET status='failed',attempts=2,error='old bug' WHERE job_id=?",
            (failed_root.job_id,),
        )
        connection.execute(
            "UPDATE jobs SET status='blocked' WHERE job_id=?",
            (phase2.job_id,),
        )

    assert store.retry_failed((failed_root.job_id,)) == {"retried": 1, "unblocked": 1}
    by_id = {job.job_id: job for job in store.jobs()}
    assert by_id[failed_root.job_id].status == "pending"
    assert by_id[failed_root.job_id].attempts == 0
    assert by_id[phase2.job_id].status == "pending"


def _requeue_job(job_id: str, dependencies: tuple[str, ...] = ()) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        kind="phase1",
        command=("true",),
        dependencies=dependencies,
        resource_ids=(),
        reservation_bytes=1,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path=f"{job_id.replace(':', '-')}.log",
    )


def test_requeue_jobs_resets_succeeded_row_and_holds_dependents(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    root = _requeue_job("requeue:root")
    child = _requeue_job("requeue:child", dependencies=(root.job_id,))
    store.submit(root)
    store.submit(child)
    with store.connect() as connection:
        connection.execute(
            """
            UPDATE jobs SET status='succeeded',attempts=2,pid=4321,gpu_id=1,
                started_at=1.0,finished_at=2.0,lease_expires_at=3.0,error='stale'
            WHERE job_id=?
            """,
            (root.job_id,),
        )
    assert [job.job_id for job in _ready_jobs(store)] == [child.job_id]

    assert store.requeue_jobs((root.job_id,), reason="places365 train cap") == {"requeued": 1}

    by_id = {job.job_id: job for job in store.jobs()}
    requeued = by_id[root.job_id]
    assert requeued.status == "pending"
    assert requeued.attempts == 0
    assert requeued.pid is None
    assert requeued.gpu_id is None
    assert by_id[child.job_id].status == "pending"
    with store.connect() as connection:
        row = connection.execute(
            "SELECT started_at,finished_at,lease_expires_at,error FROM jobs WHERE job_id=?",
            (root.job_id,),
        ).fetchone()
        events = connection.execute(
            "SELECT event,detail FROM events WHERE job_id=? ORDER BY sequence",
            (root.job_id,),
        ).fetchall()
    assert (row["started_at"], row["finished_at"], row["lease_expires_at"], row["error"]) == (
        None,
        None,
        None,
        None,
    )
    assert (events[-1]["event"], events[-1]["detail"]) == ("requeued", "places365 train cap")
    assert [job.job_id for job in _ready_jobs(store)] == [root.job_id]


def test_requeue_jobs_rejects_invalid_requests(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    for status in ("pending", "running", "succeeded", "failed"):
        store.submit(_requeue_job(f"requeue:{status}"))
    with store.connect() as connection:
        for status in ("running", "succeeded", "failed"):
            connection.execute(
                "UPDATE jobs SET status=? WHERE job_id=?",
                (status, f"requeue:{status}"),
            )

    with pytest.raises(ValueError, match="non-empty and unique"):
        store.requeue_jobs((), reason="contract change")
    with pytest.raises(ValueError, match="non-empty and unique"):
        store.requeue_jobs(("requeue:succeeded", "requeue:succeeded"), reason="contract change")
    with pytest.raises(ValueError, match="reason must be non-empty"):
        store.requeue_jobs(("requeue:succeeded",), reason="  ")
    with pytest.raises(KeyError, match="Unknown requeue jobs"):
        store.requeue_jobs(("requeue:missing",), reason="contract change")
    for status in ("pending", "running", "failed"):
        with pytest.raises(ValueError, match=f"Only succeeded jobs can be requeued.*{status}"):
            store.requeue_jobs((f"requeue:{status}",), reason="contract change")

    by_id = {job.job_id: job for job in store.jobs()}
    assert by_id["requeue:pending"].status == "pending"
    assert by_id["requeue:running"].status == "running"
    assert by_id["requeue:succeeded"].status == "succeeded"
    assert by_id["requeue:failed"].status == "failed"


def test_phase1_scope_ignores_phase2_jobs_already_in_database(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=True)
    with store.connect() as connection:
        placeholders = ",".join("?" for _ in PHASE1_RUN_JOB_KINDS)
        connection.execute(
            f"UPDATE jobs SET status='succeeded' WHERE kind IN ({placeholders})",
            tuple(PHASE1_RUN_JOB_KINDS),
        )

    counts = run_scheduler(experiment, include_phase2=False, poll_seconds=0.01)

    assert counts == {
        "pending": 0,
        "running": 0,
        "succeeded": 52,
        "failed": 0,
        "blocked": 0,
    }
    by_kind = {(job.kind, job.status) for job in store.jobs()}
    assert ("phase2_profile", "pending") in by_kind
    assert ("phase2", "pending") in by_kind


def test_phase1_ready_queue_ignores_pending_phase2_profile(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=True)
    with store.connect() as connection:
        connection.execute("UPDATE jobs SET status='succeeded' WHERE kind='profile'")

    ready = _ready_jobs(store, runnable_kinds=PHASE1_RUN_JOB_KINDS)

    assert ready
    assert {job.kind for job in ready} == {"phase1"}


def test_ready_queue_ignores_superseded_phase2_task_ids(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    submit_plan(experiment, store, include_phase2=True)
    current = next(
        job for job in store.jobs() if job.kind == "phase2" and "--clean--" in job.job_id
    )
    stale = replace(current, job_id=f"{current.job_id}--superseded")
    store.submit(stale)
    with store.connect() as connection:
        connection.execute("UPDATE jobs SET status='succeeded'")
        connection.execute(
            "UPDATE jobs SET status='pending' WHERE job_id IN (?,?)",
            (current.job_id, stale.job_id),
        )

    unfiltered = _ready_jobs(store, runnable_kinds=PHASE2_RUN_JOB_KINDS)
    filtered = _ready_jobs(
        store,
        runnable_kinds=PHASE2_RUN_JOB_KINDS,
        runnable_job_ids=_planned_job_ids(experiment, include_phase2=True),
    )

    assert {current.job_id, stale.job_id} <= {job.job_id for job in unfiltered}
    assert current.job_id in {job.job_id for job in filtered}
    assert stale.job_id not in {job.job_id for job in filtered}
    status = scheduler_status(experiment)
    assert status["stale_job_count"] == 1
    assert status["stale_by_kind"] == {"phase2": 1}
    assert status["counts"]["pending"] == 1
