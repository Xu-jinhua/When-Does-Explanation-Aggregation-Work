from __future__ import annotations

import fcntl
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xai_ensemble.simple.artifacts import ArtifactStore, write_phase1_shard
from xai_ensemble.simple.rank_ready import (
    RANK_READY_PATCH_SIZES,
    RankReadyPublisher,
    attribution_to_patch_scores,
    build_rank_ready_tensors,
    ensure_rank_ready_sidecar,
    normalize_spatial,
    rank_field,
    scores_to_ranks,
    simpleavg_score_field,
    simpleavg_spatial_field,
)


def _fields() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    logits = torch.tensor([[0.1, 0.9, -1.0], [2.0, 0.5, -0.2]], dtype=torch.float32)
    return {
        "indices": torch.tensor([10, 11]),
        "labels": torch.tensor([1, 0]),
        "predictions": logits.argmax(dim=1),
        "logits": logits,
        "targets": torch.tensor([1, 0]),
        "attributions": torch.randn((2, 3, 224, 224), generator=generator),
    }


def test_rank_ready_p8_p14_p16_matches_direct_paper_reductions() -> None:
    fields = _fields()
    compact = build_rank_ready_tensors(
        fields,
        simpleavg_normalization="minmax",
    )
    attribution = fields["attributions"].numpy()
    magnitude = np.mean(np.abs(attribution), axis=1, dtype=np.float32)
    normalized = normalize_spatial(magnitude, "minmax")
    np.testing.assert_array_equal(
        compact[simpleavg_spatial_field()].numpy(),
        normalized,
    )

    for patch_size in RANK_READY_PATCH_SIZES:
        direct_rank = scores_to_ranks(attribution_to_patch_scores(attribution, patch_size)).astype(
            np.int32
        )
        grid_h, grid_w = 224 // patch_size, 224 // patch_size
        direct_simple = normalized.reshape(
            2,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        ).mean(axis=(2, 4), dtype=np.float32)
        np.testing.assert_array_equal(
            compact[rank_field(patch_size)].numpy(),
            direct_rank,
        )
        np.testing.assert_array_equal(
            compact[simpleavg_score_field(patch_size)].numpy(),
            direct_simple.reshape(2, -1),
        )


def test_rank_ready_preserves_established_simpleavg_fp32_operation_order() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)
    old_sum = None
    compact_sum = None
    for _ in range(5):
        fields = _fields()
        fields["attributions"] = torch.randn((2, 3, 224, 224), generator=generator)
        attribution = fields["attributions"].numpy()
        normalized = normalize_spatial(
            np.mean(np.abs(attribution), axis=1, dtype=np.float32),
            "minmax",
        )
        compact = build_rank_ready_tensors(
            fields,
            simpleavg_normalization="minmax",
        )
        compact_spatial = compact[simpleavg_spatial_field()].numpy()
        old_sum = normalized.copy() if old_sum is None else old_sum + normalized
        compact_sum = (
            compact_spatial.copy() if compact_sum is None else compact_sum + compact_spatial
        )

    old_average = old_sum / 5.0
    compact_average = compact_sum / 5.0
    np.testing.assert_array_equal(compact_average, old_average)
    for patch_size in RANK_READY_PATCH_SIZES:
        grid = 224 // patch_size
        old_scores = old_average.reshape(2, grid, patch_size, grid, patch_size).mean(
            axis=(2, 4), dtype=np.float32
        )
        compact_scores = compact_average.reshape(2, grid, patch_size, grid, patch_size).mean(
            axis=(2, 4), dtype=np.float32
        )
        np.testing.assert_array_equal(compact_scores, old_scores)
        np.testing.assert_array_equal(
            scores_to_ranks(compact_scores),
            scores_to_ranks(old_scores),
        )


def test_legacy_phase1_sidecar_is_published_once_then_reused(
    tmp_path: Path,
) -> None:
    fields = _fields()
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "remote"),
            rclone_binary=Path("/unused/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    local_source = tmp_path / "source.safetensors"
    write_phase1_shard(
        local_source,
        indices=fields["indices"],
        labels=fields["labels"],
        predictions=fields["predictions"],
        logits=fields["logits"],
        targets=fields["targets"],
        attributions=fields["attributions"],
        metadata={"schema_version": "2"},
    )
    source = store.publish(local_source, "phase1/source/shard-00000.safetensors")
    payload = {
        "relative_path": source.relative_path,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }

    first = ensure_rank_ready_sidecar(
        store,
        source_payload=payload,
        work_directory=tmp_path / "first",
        lock_root=tmp_path / "locks",
        simpleavg_normalization="minmax",
        count=2,
    )
    assert first.generated is True
    assert store.exists(first.descriptor.relative_path)
    assert store.exists(f"{first.descriptor.relative_path}.receipt.json")

    materialized: list[str] = []
    original = store.materialize

    def recording_materialize(relative_path, destination, *, expected_sha256):
        materialized.append(str(relative_path))
        return original(relative_path, destination, expected_sha256=expected_sha256)

    store.materialize = recording_materialize  # type: ignore[method-assign]
    second = ensure_rank_ready_sidecar(
        store,
        source_payload=payload,
        work_directory=tmp_path / "second",
        lock_root=tmp_path / "locks",
        simpleavg_normalization="minmax",
        count=2,
    )

    assert second.generated is False
    assert materialized == [first.descriptor.relative_path]
    torch.testing.assert_close(
        second.fields[simpleavg_score_field(16)],
        first.fields[simpleavg_score_field(16)],
        rtol=0,
        atol=0,
    )


def test_lazy_sidecar_upload_can_finish_after_prefetch_returns(tmp_path: Path) -> None:
    fields = _fields()
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "remote"),
            rclone_binary=Path("/unused/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    local_source = tmp_path / "source.safetensors"
    write_phase1_shard(
        local_source,
        indices=fields["indices"],
        labels=fields["labels"],
        predictions=fields["predictions"],
        logits=fields["logits"],
        targets=fields["targets"],
        attributions=fields["attributions"],
        metadata={"schema_version": "2"},
    )
    source = store.publish(local_source, "phase1/source/shard.safetensors")
    payload = {
        "relative_path": source.relative_path,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }
    upload_started = threading.Event()
    permit_upload = threading.Event()
    original_publish = store.publish

    def delayed_publish(local_path, relative_path, **kwargs):
        if str(relative_path).startswith("derived/rank-ready"):
            upload_started.set()
            assert permit_upload.wait(timeout=2.0)
        return original_publish(local_path, relative_path, **kwargs)

    store.publish = delayed_publish  # type: ignore[method-assign]
    publisher = RankReadyPublisher(
        spool_root=tmp_path / "spool",
        spool_max_bytes=2**30,
        spool_min_free_bytes=0,
        namespace="fixture",
    )
    try:
        compact = ensure_rank_ready_sidecar(
            store,
            source_payload=payload,
            work_directory=tmp_path / "work",
            lock_root=tmp_path / "locks",
            simpleavg_normalization="none",
            count=2,
            publisher=publisher,
        )
        assert compact.generated is True
        assert compact.publication_future is not None
        assert upload_started.wait(timeout=1.0)
        assert compact.publication_future.done() is False
        lock_path = tmp_path / "locks" / f"{compact.descriptor.identity_digest}.lock"
        with lock_path.open("a+b") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        permit_upload.set()
    finally:
        permit_upload.set()
        publisher.shutdown()

    assert store.exists(compact.descriptor.relative_path)
    assert store.exists(f"{compact.descriptor.relative_path}.receipt.json")
    with lock_path.open("a+b") as contender:
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
    publisher.shutdown()


def test_failed_lazy_sidecar_upload_releases_lock_and_quota(tmp_path: Path) -> None:
    fields = _fields()
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "remote"),
            rclone_binary=Path("/unused/rclone"),
        )
    )
    store = ArtifactStore(experiment)
    local_source = tmp_path / "source.safetensors"
    write_phase1_shard(
        local_source,
        indices=fields["indices"],
        labels=fields["labels"],
        predictions=fields["predictions"],
        logits=fields["logits"],
        targets=fields["targets"],
        attributions=fields["attributions"],
        metadata={"schema_version": "2"},
    )
    source = store.publish(local_source, "phase1/source/shard.safetensors")
    payload = {
        "relative_path": source.relative_path,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }
    original_publish = store.publish

    def failed_publish(local_path, relative_path, **kwargs):
        if str(relative_path).startswith("derived/rank-ready"):
            raise RuntimeError("injected sidecar upload failure")
        return original_publish(local_path, relative_path, **kwargs)

    store.publish = failed_publish  # type: ignore[method-assign]
    publisher = RankReadyPublisher(
        spool_root=tmp_path / "spool",
        spool_max_bytes=2**30,
        spool_min_free_bytes=0,
        namespace="failure-fixture",
    )
    compact = ensure_rank_ready_sidecar(
        store,
        source_payload=payload,
        work_directory=tmp_path / "work",
        lock_root=tmp_path / "locks",
        simpleavg_normalization="none",
        count=2,
        publisher=publisher,
    )
    with pytest.raises(RuntimeError, match="Background rank-ready publication failed"):
        publisher.shutdown()

    assert publisher.quota.reserved_bytes() == 0
    lock_path = tmp_path / "locks" / f"{compact.descriptor.identity_digest}.lock"
    with lock_path.open("a+b") as contender:
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
