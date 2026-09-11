from __future__ import annotations

import json
from dataclasses import replace

import pytest

from xai_ensemble.core.hashing import file_sha256
from xai_ensemble.data.manifest import (
    DatasetManifest,
    ManifestMetadata,
    ManifestRecord,
    write_manifest,
)
from xai_ensemble.simple.manifest_identity import (
    MANIFEST_IDENTITY_SCHEMA,
    dataset_manifest_identity_sha256,
    manifest_identity_lock_path,
)


def _manifest(created_utc: str) -> DatasetManifest:
    return DatasetManifest(
        metadata=ManifestMetadata(
            dataset_key="fixture",
            dataset_id="fixture/dataset",
            revision="fixture-revision",
            dataset_spec_fingerprint="b" * 64,
            hash_mode="decoded_rgb",
            created_utc=created_utc,
        ),
        records=(
            ManifestRecord(
                split="test",
                row_index=0,
                sample_id="c" * 64,
                label=0,
                label_name="fixture",
                content_sha256="d" * 64,
            ),
        ),
    )


def _write_lock(path, manifest: DatasetManifest, *, identity_sha256: str = "a" * 64) -> None:
    value = {
        "schema": MANIFEST_IDENTITY_SCHEMA,
        "identity_sha256": identity_sha256,
        "materialized_sha256": file_sha256(path),
        "dataset_manifest_fingerprint": manifest.fingerprint,
        "dataset_id": manifest.metadata.dataset_id,
        "dataset_revision": manifest.metadata.revision,
        "record_count": len(manifest.records),
        "split_sizes": {"test": 1},
    }
    manifest_identity_lock_path(path).write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )


def test_dataset_manifest_identity_defaults_to_materialized_sha256(tmp_path) -> None:
    path = write_manifest(_manifest("2026-08-01T00:00:00+00:00"), tmp_path / "manifest.json")

    assert dataset_manifest_identity_sha256(path) == file_sha256(path)


def test_dataset_manifest_identity_lock_preserves_published_identity(tmp_path) -> None:
    manifest = _manifest("2026-08-01T01:00:00+00:00")
    regenerated = replace(
        manifest,
        metadata=replace(manifest.metadata, created_utc="2026-08-01T02:00:00+00:00"),
    )
    assert regenerated.fingerprint == manifest.fingerprint
    path = write_manifest(regenerated, tmp_path / "manifest.json")
    _write_lock(path, regenerated)

    assert dataset_manifest_identity_sha256(path) == "a" * 64


def test_dataset_manifest_identity_lock_rejects_new_materialized_bytes(tmp_path) -> None:
    manifest = _manifest("2026-08-01T01:00:00+00:00")
    path = write_manifest(manifest, tmp_path / "manifest.json")
    _write_lock(path, manifest)
    # ``write_manifest`` is immutable and therefore accepts the timestamp-only
    # retry as a no-op. Simulate an external replacement to exercise the lock.
    path.write_bytes(
        path.read_bytes().replace(
            b"2026-08-01T01:00:00+00:00",
            b"2026-08-01T03:00:00+00:00",
        )
    )

    with pytest.raises(ValueError, match="materialized manifest SHA-256 changed"):
        dataset_manifest_identity_sha256(path)
