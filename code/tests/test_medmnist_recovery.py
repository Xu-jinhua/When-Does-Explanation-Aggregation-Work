from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from xai_ensemble.data import medmnist_recovery as recovery
from xai_ensemble.data.specs import HFDatasetSpec


def _spec(md5: str) -> HFDatasetSpec:
    return HFDatasetSpec(
        key="tiny-octmnist",
        dataset_id="medmnist/octmnist",
        revision="medmnist-test-revision",
        num_classes=2,
        image_column="image",
        label_column="label",
        text_column=None,
        splits=("train", "validation", "test"),
        expected_split_sizes={"train": 2, "validation": 2, "test": 2},
        expected_examples_per_class={"train": 1, "validation": 1, "test": 1},
        source_url="https://medmnist.com/",
        upstream_source="synthetic",
        license="test",
        provider="medmnist",
        provider_options={
            "data_flag": "octmnist",
            "version": "3.0.2",
            "size": 224,
            "as_rgb": True,
            "zenodo_record": "10519652",
            "filename": "octmnist_224.npz",
            "source_url": "https://example.invalid/octmnist_224.npz",
            "md5": md5,
        },
    )


def _archive(path: Path) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for split in ("train", "val", "test"):
            for suffix in ("images", "labels"):
                archive.writestr(f"{split}_{suffix}.npy", b"payload")


def test_verified_source_marker_is_content_derived_and_idempotent(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "octmnist_224.npz"
    _archive(archive)
    observed_size = archive.stat().st_size
    observed_md5 = recovery._md5(archive)
    monkeypatch.setitem(recovery.MEDMNIST_224_FILE_SIZES, archive.name, observed_size)
    # The fixed contract is patched only for this small fixture; the production
    # values remain the immutable Zenodo sizes above.
    spec = _spec(observed_md5)

    first = recovery.validate_medmnist_source(spec, tmp_path)
    marker = recovery.source_marker_path(tmp_path, spec)
    before = marker.read_bytes()
    second = recovery.validate_medmnist_source(spec, tmp_path)

    assert first == second
    assert marker.read_bytes() == before
    assert first["identity_digest"] == recovery.object_sha256(first["identity"])
    assert recovery.ensure_verified_medmnist_source(spec, tmp_path, download=False) == first


def test_verified_source_marker_rejects_tampering(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "octmnist_224.npz"
    _archive(archive)
    monkeypatch.setitem(recovery.MEDMNIST_224_FILE_SIZES, archive.name, archive.stat().st_size)
    spec = _spec(recovery._md5(archive))
    recovery.validate_medmnist_source(spec, tmp_path)
    marker = recovery.source_marker_path(tmp_path, spec)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["identity"]["observed_md5"] = "1" * 32
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="verified source identity is invalid"):
        recovery.require_verified_medmnist_source(spec, tmp_path)
