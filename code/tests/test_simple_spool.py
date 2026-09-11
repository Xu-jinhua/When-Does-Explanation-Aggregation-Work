from __future__ import annotations

import json
import os
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xai_ensemble.core.hashing import file_sha256
from xai_ensemble.simple import spool as spool_module
from xai_ensemble.simple.artifacts import (
    ArtifactError,
    ArtifactStore,
    load_safetensors,
    write_phase2_shard,
)
from xai_ensemble.simple.config import load_experiment
from xai_ensemble.simple.phase1 import _Phase1Publisher
from xai_ensemble.simple.rank_ready import (
    compact_rank_input_fp32_equivalence,
    rank_field,
    simpleavg_score_field,
)
from xai_ensemble.simple.runtime import (
    GPU_RELEASE_JOB_ENV,
    GPU_RELEASE_PATH_ENV,
    GPU_RELEASE_TOKEN_ENV,
    emit_gpu_release_signal,
    gpu_release_signal_matches,
)
from xai_ensemble.simple.spool import SpoolCapacityError, SpoolQuota, UploadGate

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_process_identity_keeps_live_record_when_procfs_is_temporarily_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = spool_module._process_start_time(os.getpid())
    assert expected is not None
    monkeypatch.setattr(spool_module, "_process_start_time", lambda _: None)

    assert not spool_module._process_is_stale(os.getpid(), expected)
    assert spool_module._process_is_stale(2**31 - 1, expected)


def test_sqlite_transaction_retries_database_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    class FakeConnection:
        def execute(self, statement: str) -> None:
            nonlocal attempts
            if statement == "BEGIN IMMEDIATE":
                attempts += 1
                if attempts == 1:
                    raise sqlite3.OperationalError("database is locked")

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(spool_module.time, "sleep", sleeps.append)
    result = spool_module._run_sqlite_transaction(
        FakeConnection,
        lambda connection: connection.execute("SELECT 1"),
    )

    assert result is None
    assert attempts == 2
    assert sleeps == [0.05]


def test_sqlite_transaction_waits_out_a_held_write_lock(tmp_path: Path) -> None:
    # Regression for the 2026-09-08 tissuemnist IG death: a concurrent writer
    # holding the lock longer than the old 2 s budget must not fail the call.
    database = tmp_path / "spool.sqlite3"

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(
            database,
            timeout=spool_module._SQLITE_BUSY_TIMEOUT_SECONDS,
        )
        connection.execute(
            f"PRAGMA busy_timeout={int(spool_module._SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}"
        )
        return connection

    with connect() as connection:
        connection.execute("CREATE TABLE quota (value INTEGER)")

    hold_seconds = 2.5
    assert hold_seconds > 2.0  # must exceed the pre-fix budget

    def hold_write_lock() -> None:
        connection = sqlite3.connect(database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO quota VALUES (1)")
            time.sleep(hold_seconds)
            connection.commit()
        finally:
            connection.close()

    holder = threading.Thread(target=hold_write_lock)
    holder.start()
    time.sleep(0.2)  # let the holder take the lock first
    started = time.monotonic()
    spool_module._run_sqlite_transaction(
        connect,
        lambda connection: connection.execute("INSERT INTO quota VALUES (2)"),
    )
    waited = time.monotonic() - started
    holder.join()

    assert waited >= hold_seconds - 0.2
    with connect() as connection:
        rows = connection.execute("SELECT COUNT(*) FROM quota").fetchone()
    assert rows == (2,)


def test_spool_quota_is_shared_across_instances(tmp_path: Path) -> None:
    first = SpoolQuota(tmp_path, max_bytes=100, min_free_bytes=0, poll_seconds=0.01)
    second = SpoolQuota(tmp_path, max_bytes=100, min_free_bytes=0, poll_seconds=0.01)
    first_work = tmp_path / "workers" / "first"
    second_work = tmp_path / "workers" / "second"
    first_work.mkdir(parents=True)
    second_work.mkdir(parents=True)

    reservation = first.acquire(80, work_directory=first_work)
    with pytest.raises(SpoolCapacityError, match="Timed out"):
        second.acquire(30, work_directory=second_work, timeout_seconds=0.02)

    first.release(reservation)
    replacement = second.acquire(30, work_directory=second_work, timeout_seconds=0.1)
    assert second.reserved_bytes() == 30
    second.release(replacement)
    assert first.reserved_bytes() == 0


def test_spool_quota_skips_a_higher_priority_waiter_that_cannot_fit(tmp_path: Path) -> None:
    quota = SpoolQuota(tmp_path, max_bytes=100, min_free_bytes=0, poll_seconds=0.005)
    held_work = tmp_path / "workers" / "held"
    large_work = tmp_path / "workers" / "large"
    small_work = tmp_path / "workers" / "small"
    for directory in (held_work, large_work, small_work):
        directory.mkdir(parents=True)
    held = quota.acquire(80, work_directory=held_work)
    large_result: list[object] = []

    def acquire_large() -> None:
        try:
            large_result.append(
                quota.acquire(
                    30,
                    work_directory=large_work,
                    timeout_seconds=1.0,
                    priority=-100,
                )
            )
        except BaseException as error:
            large_result.append(error)

    thread = threading.Thread(target=acquire_large)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with sqlite3.connect(quota.database_path) as connection:
            if int(connection.execute("SELECT COUNT(*) FROM waiters").fetchone()[0]) == 1:
                break
        time.sleep(0.005)
    else:
        pytest.fail("large quota waiter was not registered")

    small = quota.acquire(
        20,
        work_directory=small_work,
        timeout_seconds=0.2,
        priority=10,
    )
    quota.release(small)
    quota.release(held)
    thread.join(timeout=1.0)
    assert len(large_result) == 1
    assert not isinstance(large_result[0], BaseException)
    quota.release(large_result[0])
    assert quota.reserved_bytes() == 0


def test_upload_gate_bounds_concurrent_remote_operations(tmp_path: Path) -> None:
    gate = UploadGate(tmp_path, max_slots=2, poll_seconds=0.005)
    first = gate.acquire()
    second = gate.acquire()
    acquired = threading.Event()
    result: list[object] = []

    def wait_for_slot() -> None:
        try:
            result.append(gate.acquire(timeout_seconds=1.0))
            acquired.set()
        except BaseException as error:
            result.append(error)

    thread = threading.Thread(target=wait_for_slot)
    thread.start()
    time.sleep(0.03)
    assert not acquired.is_set()
    assert gate.active_slots() == 2
    gate.release(first)
    assert acquired.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert len(result) == 1
    assert not isinstance(result[0], BaseException)
    gate.release(second)
    gate.release(result[0])
    assert gate.active_slots() == 0


def test_gpu_release_signal_is_bound_to_job_pid_and_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "release.json"
    monkeypatch.setenv(GPU_RELEASE_PATH_ENV, str(marker))
    monkeypatch.setenv(GPU_RELEASE_TOKEN_ENV, "attempt-token")
    monkeypatch.setenv(GPU_RELEASE_JOB_ENV, "phase1:task")

    assert emit_gpu_release_signal()
    assert gpu_release_signal_matches(
        marker,
        job_id="phase1:task",
        pid=os.getpid(),
        token="attempt-token",
    )
    assert not gpu_release_signal_matches(
        marker,
        job_id="phase1:other",
        pid=os.getpid(),
        token="attempt-token",
    )


def test_remote_sha256_uses_backend_metadata_without_cat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "Size": 123,
                    "Hashes": {"sha256": "a" * 64},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert store._remote_sha256("shards/one.safetensors") == ("a" * 64, 123)
    assert len(commands) == 1
    assert commands[0][1:] == (
        "lsjson",
        "Remote:experiment/shards/one.safetensors",
        "--stat",
        "--hash",
    )


def test_remote_copy_ensures_parent_chain_once_under_host_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
            spool_root=tmp_path / "spool",
        )
    )
    store = ArtifactStore(experiment)
    commands: list[tuple[str, ...]] = []

    def fake_run(*arguments: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(arguments))
        return subprocess.CompletedProcess(("rclone", *arguments), 0, stdout="", stderr="")

    monkeypatch.setattr(store, "_run", fake_run)

    store._copy(Path("/tmp/payload"), "chain/a/shards/one.npz")
    store._copy(Path("/tmp/payload"), "chain/a/shards/two.npz")
    store._copy(Path("/tmp/payload"), "chain/b/shards/three.npz")

    assert commands == [
        ("mkdir", "Remote:experiment/chain/a/shards"),
        ("copyto", "/tmp/payload", "Remote:experiment/chain/a/shards/one.npz", "--immutable"),
        ("copyto", "/tmp/payload", "Remote:experiment/chain/a/shards/two.npz", "--immutable"),
        ("mkdir", "Remote:experiment/chain/b/shards"),
        ("copyto", "/tmp/payload", "Remote:experiment/chain/b/shards/three.npz", "--immutable"),
    ]
    assert (tmp_path / "spool" / ".remote-mkdir.lock").is_file()


def test_publish_retries_post_upload_verification_lag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    payload = tmp_path / "shard.npz"
    payload.write_bytes(b"payload-bytes")
    digest = file_sha256(payload)

    monkeypatch.setattr(store, "exists", lambda _path: False)
    monkeypatch.setattr(store, "_copy", lambda *_args: None)
    monkeypatch.setattr(store, "_ensure_remote_parents", lambda *_args: None)
    monkeypatch.setattr("xai_ensemble.simple.artifacts.time.sleep", lambda _: None)
    attempts: list[str] = []

    def flaky_sha256(relative_path: str) -> tuple[str, int]:
        attempts.append(relative_path)
        if len(attempts) < 3:
            raise ArtifactError("rclone lsjson failed: directory not found")
        return digest, payload.stat().st_size

    monkeypatch.setattr(store, "_remote_sha256", flaky_sha256)

    published = store.publish(payload, "x/shard.npz", write_receipt=False)
    assert published.sha256 == digest
    assert len(attempts) == 3

    monkeypatch.setattr(
        store,
        "_remote_sha256",
        lambda _path: (_ for _ in ()).throw(ArtifactError("checksum mismatch")),
    )
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        store.publish(payload, "x/shard.npz", write_receipt=False)


def test_remote_transport_retries_only_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    attempts = 0

    def transient_then_success(*_: str, **__: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ArtifactError("rclone copyto failed: input/output error")
        return subprocess.CompletedProcess(("rclone",), 0, stdout="", stderr="")

    monkeypatch.setattr(store, "_run", transient_then_success)
    monkeypatch.setattr("xai_ensemble.simple.artifacts.time.sleep", lambda _: None)
    store._run_transport_with_retry("copyto", "source", "target", "--immutable")
    assert attempts == 3


def test_remote_transport_does_not_retry_immutable_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    attempts = 0

    def immutable_failure(*_: str, **__: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        raise ArtifactError("rclone copyto failed: immutable destination exists")

    monkeypatch.setattr(store, "_run", immutable_failure)
    with pytest.raises(ArtifactError, match="immutable"):
        store._run_transport_with_retry("copyto", "source", "target", "--immutable")
    assert attempts == 1


def test_phase2_safetensors_metadata_has_deterministic_file_hash(tmp_path: Path) -> None:
    import torch

    tensors = {
        "indices": torch.arange(4),
        "values": torch.arange(12, dtype=torch.float32).reshape(4, 3),
    }
    hashes = set()
    for index in range(12):
        path = tmp_path / f"deterministic-{index}.safetensors"
        write_phase2_shard(
            path,
            tensors=tensors,
            metadata={"schema": "fixture", "zeta": "last", "alpha": "first"},
        )
        hashes.add(file_sha256(path))
    assert len(hashes) == 1


def test_remote_exists_does_not_treat_transport_failure_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
        )
    )
    store = ArtifactStore(experiment)

    def failed_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 5, stdout="", stderr="rate limit exceeded")

    monkeypatch.setattr(subprocess, "run", failed_run)
    monkeypatch.setattr("xai_ensemble.simple.artifacts.time.sleep", lambda _: None)
    with pytest.raises(ArtifactError, match="rate limit exceeded"):
        store.exists("shards/one.safetensors")


def test_remote_exists_recognizes_rclone_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root="Remote:experiment",
            rclone_binary=Path("/opt/rclone"),
        )
    )
    store = ArtifactStore(experiment)

    def missing_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="directory not found")

    monkeypatch.setattr(subprocess, "run", missing_run)
    assert store.exists("shards/missing.safetensors") is False


def test_publish_adopts_unreceipted_safetensors_with_metadata_order_only_difference(
    tmp_path: Path,
) -> None:
    import torch

    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "remote"),
            rclone_binary=Path("/unused/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    relative = "shards/legacy.safetensors"
    remote = Path(store.locator(relative))
    remote.parent.mkdir(parents=True)
    tensors = {
        "indices": torch.arange(4),
        "values": torch.arange(12, dtype=torch.float32).reshape(4, 3),
    }
    local = tmp_path / "canonical.safetensors"
    write_phase2_shard(
        local,
        tensors=tensors,
        metadata={"zeta": "last", "alpha": "first"},
    )
    shutil.copyfile(local, remote)
    with remote.open("r+b") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header_bytes = handle.read(header_length)
        header = json.loads(header_bytes.rstrip(b" "))
        header["__metadata__"] = {
            key: header["__metadata__"][key] for key in reversed(tuple(header["__metadata__"]))
        }
        reordered = json.dumps(header, separators=(",", ":")).encode("utf-8")
        assert len(reordered) <= header_length
        handle.seek(8)
        handle.write(reordered + b" " * (header_length - len(reordered)))
    legacy_sha = file_sha256(remote)
    assert legacy_sha != file_sha256(local)

    published = store.publish(local, relative)

    assert published.sha256 == legacy_sha
    assert file_sha256(remote) == legacy_sha
    receipt = store.read_json(f"{relative}.receipt.json")
    assert receipt["sha256"] == legacy_sha
    republished = store.publish(local, relative)
    assert republished.sha256 == legacy_sha


def test_publish_rejects_unreceipted_safetensors_with_different_tensor_data(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "remote"),
            rclone_binary=Path("/unused/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    relative = "shards/contradiction.safetensors"
    remote = Path(store.locator(relative))
    remote.parent.mkdir(parents=True)
    save_file(
        {"indices": torch.arange(2), "values": torch.tensor([[1.0], [2.0]])},
        str(remote),
        metadata={"schema": "fixture"},
    )
    local = tmp_path / "different.safetensors"
    write_phase2_shard(
        local,
        tensors={"indices": torch.arange(2), "values": torch.tensor([[1.0], [3.0]])},
        metadata={"schema": "fixture"},
    )

    with pytest.raises(ArtifactError, match="verification failed"):
        store.publish(local, relative)


def test_publish_adopts_rank_exact_compact_shard_with_fp32_score_noise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import torch

    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "remote"),
            rclone_binary=Path("/unused/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    metadata = {
        "schema": "simple-assumptions-source-rank-input-v1",
        "schema_version": "1",
        "patch_size": "16",
        "method": "GradientShap",
    }
    rank = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32)
    common = {
        "indices": torch.arange(2, dtype=torch.int64),
        "labels": torch.tensor([1, 0], dtype=torch.int64),
        "predictions": torch.tensor([1, 0], dtype=torch.int64),
        "logits": torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32),
        "targets": torch.tensor([1, 0], dtype=torch.int64),
        rank_field(16): rank,
    }
    remote_source = tmp_path / "remote-source.safetensors"
    remote_scores = torch.zeros((2, 4), dtype=torch.float32)
    write_phase2_shard(
        remote_source,
        tensors={**common, simpleavg_score_field(16): remote_scores},
        metadata=metadata,
    )
    relative = "source-rank-inputs/fixture/shard-00000.safetensors"
    original = store.publish(remote_source, relative)

    equivalent = tmp_path / "equivalent.safetensors"
    equivalent_scores = remote_scores.clone()
    equivalent_scores[0, 0] = 4 * torch.finfo(torch.float32).eps
    write_phase2_shard(
        equivalent,
        tensors={**common, simpleavg_score_field(16): equivalent_scores},
        metadata=metadata,
    )
    adopted = store.publish(
        equivalent,
        relative,
        existing_payload_equivalence=compact_rank_input_fp32_equivalence,
    )
    assert adopted.sha256 == original.sha256
    assert "mode=compact_rank_input_rank_exact_fp32_score_equivalence" in capsys.readouterr().out

    changed_rank = tmp_path / "changed-rank.safetensors"
    different_rank = rank.clone()
    different_rank[0, :2] = different_rank[0, :2].flip(0)
    write_phase2_shard(
        changed_rank,
        tensors={
            **common,
            rank_field(16): different_rank,
            simpleavg_score_field(16): equivalent_scores,
        },
        metadata=metadata,
    )
    assert compact_rank_input_fp32_equivalence(changed_rank, remote_source) is None

    changed_score = tmp_path / "changed-score.safetensors"
    different_scores = remote_scores.clone()
    different_scores[0, 0] = 16 * torch.finfo(torch.float32).eps
    write_phase2_shard(
        changed_score,
        tensors={**common, simpleavg_score_field(16): different_scores},
        metadata=metadata,
    )
    assert compact_rank_input_fp32_equivalence(changed_score, remote_source) is None


def test_phase1_publisher_returns_while_local_publication_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    source = CODE_ROOT / "configs/simple/example.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    config["storage"].update(
        {
            "remote_root": str(tmp_path / "artifacts"),
            "scratch_root": str(tmp_path / "scratch"),
            "spool_root": str(tmp_path / "spool"),
            "spool_max_gib": 1,
            "spool_min_free_gib": 0,
        }
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    experiment = load_experiment(path)
    task = experiment.phase1_tasks()[0]
    variant = task.variants[0]
    store = ArtifactStore(experiment)
    publisher = _Phase1Publisher(experiment, task, store)
    publication_started = threading.Event()
    permit_publication = threading.Event()
    original_publish = store.publish

    def delayed_publish(*args: object, **kwargs: object):
        publication_started.set()
        assert permit_publication.wait(timeout=2.0)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(store, "publish", delayed_publish)
    predictions = torch.tensor([1, 0])
    try:
        future = publisher.submit_shard(
            variant,
            root="phase1/test",
            shard_index=0,
            start=0,
            stop=2,
            attributions=torch.ones((2, 3, 4, 4)),
            labels=torch.tensor([1, 0]),
            indices=torch.tensor([10, 11]),
            predictions=predictions,
            logits=torch.tensor([[0.1, 0.9], [2.0, -1.0]]),
            targets=predictions,
            profile_id="profile",
            batch_size=2,
            source_identity_digest="source",
        )
        assert publication_started.wait(timeout=1.0)
        assert not future.done()
        permit_publication.set()
        record = future.result(timeout=2.0)
    finally:
        permit_publication.set()
        publisher.shutdown()

    payload = tmp_path / "artifacts" / record["payload"]["relative_path"]
    fields = load_safetensors(payload)
    torch.testing.assert_close(fields["targets"], fields["predictions"])
    assert publisher.quota.reserved_bytes() == 0


def test_phase1_publisher_emits_one_combined_rank_ready_sidecar(tmp_path: Path) -> None:
    import torch

    source = CODE_ROOT / "configs/simple/example.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    config["storage"].update(
        {
            "remote_root": str(tmp_path / "artifacts"),
            "scratch_root": str(tmp_path / "scratch"),
            "spool_root": str(tmp_path / "spool"),
            "spool_max_gib": 1,
            "spool_min_free_gib": 0,
        }
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    experiment = load_experiment(path)
    task = experiment.phase1_tasks()[0]
    variant = task.variants[0]
    store = ArtifactStore(experiment)
    publisher = _Phase1Publisher(experiment, task, store)
    assert publisher.input_quota.root == experiment.storage.spool_root / "input-prefetch"
    assert publisher.input_quota.max_bytes == 64 * 2**30
    assert experiment.runtime.phase1_stage_workers == 2
    logits = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    try:
        record = publisher.submit_shard(
            variant,
            root="phase1/test",
            shard_index=0,
            start=0,
            stop=1,
            attributions=torch.ones((1, 3, 224, 224)),
            labels=torch.tensor([1]),
            indices=torch.tensor([10]),
            predictions=torch.tensor([1]),
            logits=logits,
            targets=torch.tensor([1]),
            profile_id="profile",
            batch_size=1,
            source_identity_digest="source",
        ).result(timeout=5.0)
    finally:
        publisher.shutdown()

    compact = record["rank_ready"]
    local = tmp_path / "rank-ready.safetensors"
    store.materialize(
        compact["relative_path"],
        local,
        expected_sha256=compact["sha256"],
    )
    fields = load_safetensors(local)
    for patch_size in (8, 14, 16):
        assert rank_field(patch_size) in fields
        assert simpleavg_score_field(patch_size) in fields


def test_phase1_publisher_starts_payload_and_rank_ready_uploads_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    source = CODE_ROOT / "configs/simple/example.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    config["storage"].update(
        {
            "remote_root": str(tmp_path / "artifacts"),
            "scratch_root": str(tmp_path / "scratch"),
            "spool_root": str(tmp_path / "spool"),
            "spool_max_gib": 1,
            "spool_min_free_gib": 0,
        }
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    experiment = load_experiment(path)
    task = experiment.phase1_tasks()[0]
    variant = task.variants[0]
    store = ArtifactStore(experiment)
    publisher = _Phase1Publisher(experiment, task, store)
    payload_started = threading.Event()
    sidecar_started = threading.Event()
    both_started = threading.Event()
    original_publish = store.publish

    def observe_publish(local_path: Path, relative_path: str, **kwargs: object):
        if local_path.name.endswith(".rank-ready.safetensors"):
            sidecar_started.set()
        elif local_path.suffix == ".safetensors":
            payload_started.set()
        if payload_started.is_set() and sidecar_started.is_set():
            both_started.set()
        if local_path.suffix == ".safetensors":
            assert both_started.wait(timeout=2.0)
        return original_publish(local_path, relative_path, **kwargs)

    monkeypatch.setattr(store, "publish", observe_publish)
    try:
        future = publisher.submit_shard(
            variant,
            root="phase1/test",
            shard_index=0,
            start=0,
            stop=1,
            attributions=torch.ones((1, 3, 224, 224)),
            labels=torch.tensor([1]),
            indices=torch.tensor([10]),
            predictions=torch.tensor([1]),
            logits=torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            targets=torch.tensor([1]),
            profile_id="profile",
            batch_size=1,
            source_identity_digest="source",
        )
        record = future.result(timeout=5.0)
    finally:
        publisher.shutdown()

    assert both_started.is_set()
    assert "rank_ready" in record
