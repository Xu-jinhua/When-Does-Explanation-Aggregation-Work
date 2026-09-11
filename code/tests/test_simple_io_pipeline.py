from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from xai_ensemble.core.cache import ShardCache
from xai_ensemble.simple.config import ConditionConfig
from xai_ensemble.simple.data import (
    BatchedTensor,
    DatasetBundle,
    LoadedModel,
    apply_condition,
    materialize_model_inputs,
    materialize_shared_conditioned_raw_images,
)
from xai_ensemble.simple.io_pipeline import (
    AsyncSpoolWriter,
    ByteBoundedPrefetcher,
    PrefetchItem,
)
from xai_ensemble.simple.shared_cache import materialize_shared_tensor
from xai_ensemble.simple.spool import SpoolQuota
from xai_ensemble.simple.telemetry import GpuUtilizationSampler, StageTimings


def test_streaming_batch_tensor_and_raw_cache_are_bounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XAI_SIMPLE_STREAMING_INPUTS", "1")
    images = [{"image": torch.full((3, 4, 4), float(index) / 10.0)} for index in range(6)]
    bundle = DatasetBundle(
        dataset=images,
        raw_images=None,
        labels=torch.arange(6, dtype=torch.int64),
        indices=torch.arange(6, dtype=torch.int64),
        sample_ids=tuple(f"sample-{index}" for index in range(6)),
        row_to_position={index: index for index in range(6)},
        raw_source_identity_digest="streaming-fixture",
        raw_image_shape=(3, 4, 4),
        raw_batch_cache=ShardCache(tmp_path / "hot", max_bytes=1024),
        raw_cache_shard_size=2,
    )
    loaded = LoadedModel(
        model=torch.nn.Identity(),
        preprocessing={"input_size": 4, "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        normalize=lambda values: values,
    )
    condition = ConditionConfig("clean", "clean", None, {})

    values = materialize_model_inputs(
        bundle,
        loaded,
        condition,
        device="cpu",
        batch_size=2,
        seed=123,
        shared_cache_root=tmp_path / "shared",
    )
    assert isinstance(values, BatchedTensor)
    first = values[0:2]
    torch.testing.assert_close(first, torch.stack([images[0]["image"], images[1]["image"]]))
    images[0]["image"].fill_(1.0)
    second = materialize_model_inputs(
        bundle,
        loaded,
        condition,
        device="cpu",
        batch_size=1,
        seed=123,
        shared_cache_root=tmp_path / "shared",
    )
    torch.testing.assert_close(second[0:2], first)
    torch.testing.assert_close(values[4:6], torch.stack([images[4]["image"], images[5]["image"]]))
    assert bundle.raw_batch_cache.stats().bytes <= 1024


def test_prefetch_looks_beyond_n_plus_one_and_obeys_byte_quota(tmp_path: Path) -> None:
    quota = SpoolQuota(tmp_path / "spool", max_bytes=45, min_free_bytes=0, poll_seconds=0.005)
    loaded: list[int] = []
    condition = threading.Condition()

    def loader(index: int):
        def load(_directory: Path) -> int:
            with condition:
                loaded.append(index)
                condition.notify_all()
            return index

        return load

    items = tuple(PrefetchItem(key=index, byte_count=20, load=loader(index)) for index in range(4))
    with ByteBoundedPrefetcher(quota, items, workers=4, namespace="test") as prefetcher:
        deadline = time.monotonic() + 2.0
        with condition:
            while len(loaded) < 2 and time.monotonic() < deadline:
                condition.wait(timeout=0.02)
        assert loaded[:2] == [0, 1]
        # Two completed lookahead units hold 40/45 bytes; N+2 cannot enter yet.
        time.sleep(0.03)
        assert loaded == [0, 1]

        first = prefetcher.get(0)
        assert first.value == 0
        prefetcher.release(0)
        deadline = time.monotonic() + 2.0
        with condition:
            while len(loaded) < 3 and time.monotonic() < deadline:
                condition.wait(timeout=0.02)
        assert loaded[:3] == [0, 1, 2]

        for key in (1, 2, 3):
            value = prefetcher.get(key)
            assert value.value == key
            prefetcher.release(key)
    assert quota.reserved_bytes() == 0


def test_stage_timings_is_safe_for_staging_and_upload_threads() -> None:
    timings = StageTimings()

    def record() -> None:
        for _ in range(100):
            timings.add("upload", 0.001)

    threads = [threading.Thread(target=record) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert timings.summary()["upload"]["count"] == 400


def test_shared_mmap_tensor_is_populated_once_and_reopened(tmp_path: Path) -> None:
    calls = 0

    def populate(values: np.memmap) -> None:
        nonlocal calls
        calls += 1
        values[:] = np.arange(12, dtype=np.float32).reshape(3, 4)

    first = materialize_shared_tensor(
        tmp_path,
        namespace="datasets",
        identity={"dataset": "fixture", "split": "test"},
        shape=(3, 4),
        dtype=np.float32,
        populate=populate,
    )
    second = materialize_shared_tensor(
        tmp_path,
        namespace="datasets",
        identity={"dataset": "fixture", "split": "test"},
        shape=(3, 4),
        dtype=np.float32,
        populate=populate,
    )

    assert calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.path == second.path
    np.testing.assert_array_equal(first.tensor.numpy(), second.tensor.numpy())


def test_async_spool_writer_overlaps_upload_with_caller(tmp_path: Path) -> None:
    quota = SpoolQuota(tmp_path / "spool", max_bytes=1024, min_free_bytes=0)
    writer = AsyncSpoolWriter(quota, namespace="test")
    upload_started = threading.Event()
    permit_upload = threading.Event()

    def stage(path: Path) -> int:
        path.write_bytes(b"payload")
        return 7

    def upload(path: Path, staged: int) -> int:
        upload_started.set()
        assert permit_upload.wait(timeout=2.0)
        assert path.read_bytes() == b"payload"
        return staged + 1

    try:
        future = writer.submit(
            byte_count=128,
            basename="payload.bin",
            stage=stage,
            upload=upload,
        )
        assert upload_started.wait(timeout=1.0)
        assert future.done() is False
        permit_upload.set()
        assert future.result(timeout=2.0) == 8
    finally:
        permit_upload.set()
        writer.shutdown()
    assert quota.reserved_bytes() == 0


def test_async_spool_writer_does_not_block_submit_when_quota_is_busy(tmp_path: Path) -> None:
    quota = SpoolQuota(tmp_path / "spool", max_bytes=128, min_free_bytes=0, poll_seconds=0.005)
    held_directory = quota.root / "held"
    held_directory.mkdir()
    held = quota.acquire(128, work_directory=held_directory)
    writer = AsyncSpoolWriter(quota, namespace="deferred-reservation")
    staged = threading.Event()

    def stage(path: Path) -> None:
        path.write_bytes(b"payload")
        staged.set()

    future = writer.submit(
        byte_count=64,
        basename="payload.bin",
        stage=stage,
        upload=lambda path, _: path.read_bytes(),
    )
    assert future.done() is False
    assert staged.wait(timeout=0.05) is False

    quota.release(held)
    try:
        assert future.result(timeout=1.0) == b"payload"
    finally:
        writer.shutdown()
    assert quota.reserved_bytes() == 0


def test_natural_condition_mmap_is_exact_and_reused(tmp_path: Path) -> None:
    raw = torch.linspace(0.0, 1.0, 3 * 3 * 8 * 8).reshape(3, 3, 8, 8)
    labels = torch.tensor([0, 1, 0])
    indices = torch.tensor([10, 11, 12])
    bundle = DatasetBundle(
        dataset=None,
        raw_images=raw,
        labels=labels,
        indices=indices,
        sample_ids=("a", "b", "c"),
        row_to_position={10: 0, 11: 1, 12: 2},
        raw_source_identity_digest="fixture-source-a",
    )
    loaded = LoadedModel(
        model=torch.nn.Identity(),
        preprocessing={"input_size": 8, "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        normalize=lambda values: values,
    )
    condition = ConditionConfig(
        "gaussian-0.15",
        "factory",
        "xai_ensemble.simple.conditions:natural_corruption",
        {"kind": "gaussian", "severity": 0.15},
    )
    expected = apply_condition(
        condition,
        raw,
        labels=labels,
        indices=indices,
        model=loaded.model,
        normalize=loaded.normalize,
        seed=123,
    )

    first = materialize_shared_conditioned_raw_images(
        bundle,
        loaded,
        condition,
        device="cpu",
        batch_size=2,
        seed=123,
        shared_cache_root=tmp_path,
    )
    second = materialize_shared_conditioned_raw_images(
        bundle,
        loaded,
        condition,
        device="cpu",
        batch_size=1,
        seed=123,
        shared_cache_root=tmp_path,
    )

    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert len(tuple((tmp_path / "conditions").rglob("*.npy"))) == 1

    other_bundle = DatasetBundle(
        dataset=None,
        raw_images=1.0 - raw,
        labels=labels,
        indices=indices,
        sample_ids=("a", "b", "c"),
        row_to_position={10: 0, 11: 1, 12: 2},
        raw_source_identity_digest="fixture-source-b",
    )
    other = materialize_shared_conditioned_raw_images(
        other_bundle,
        loaded,
        condition,
        device="cpu",
        batch_size=2,
        seed=123,
        shared_cache_root=tmp_path,
    )
    assert not torch.equal(other, first)
    assert len(tuple((tmp_path / "conditions").rglob("*.npy"))) == 2


def test_gpu_telemetry_maps_logical_to_physical_visible_device(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")

    assert GpuUtilizationSampler(requested_device="cuda:0").device_id == "3"
    assert GpuUtilizationSampler(requested_device="cuda:1").device_id == "7"
