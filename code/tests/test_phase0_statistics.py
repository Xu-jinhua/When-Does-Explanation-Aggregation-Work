from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xai_ensemble.core.manifest import load_manifest, validate_manifest_files
from xai_ensemble.phase0.statistics import (
    ArtifactValidationError,
    ImageMeanAccumulator,
    write_image_mean_artifact,
)


def test_streaming_image_and_class_means() -> None:
    accumulator = ImageMeanAccumulator(2)
    accumulator.update(
        np.asarray(
            [
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 1]),
    )
    accumulator.update(
        np.asarray([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=np.float32),
        np.asarray([0]),
    )
    values = accumulator.finalize()
    np.testing.assert_allclose(
        values.dataset_mean, np.asarray([[[1 / 3, 2 / 3], [1 / 3, 2 / 3]]])
    )
    np.testing.assert_allclose(
        values.class_means[0], np.asarray([[[0.0, 0.5], [0.0, 0.5]]])
    )
    np.testing.assert_allclose(values.class_means[1], np.ones((1, 2, 2)))
    assert values.class_counts.tolist() == [2, 1]


def test_mean_rejects_out_of_range() -> None:
    accumulator = ImageMeanAccumulator(2)
    with pytest.raises(ValueError, match=r"raw \[0, 1\]"):
        accumulator.update(np.full((1, 1, 2, 2), 2.0), np.asarray([0]))


def test_mean_artifact_is_train_only_valid_and_immutable(tmp_path: Path) -> None:
    accumulator = ImageMeanAccumulator(2)
    accumulator.update(np.zeros((1, 1, 2, 2)), np.asarray([0]))
    accumulator.update(np.ones((1, 1, 2, 2)), np.asarray([1]))
    values = accumulator.finalize()
    with pytest.raises(ValueError, match="train split"):
        write_image_mean_artifact(
            tmp_path,
            values,
            dataset_id="tiny",
            dataset_revision="fixed",
            dataset_manifest_fingerprint="abc",
            model_id="model",
            preprocessing={},
            source_split="test",
            protocol_digest="protocol",
        )
    manifest = write_image_mean_artifact(
        tmp_path,
        values,
        dataset_id="tiny",
        dataset_revision="fixed",
        dataset_manifest_fingerprint="abc",
        model_id="model",
        preprocessing={"input_size": 2},
        source_split="train",
        protocol_digest="protocol",
    )
    assert load_manifest(tmp_path / "manifest.json").artifact_id == manifest.artifact_id
    validate_manifest_files(tmp_path, manifest)
    repeated = write_image_mean_artifact(
        tmp_path,
        values,
        dataset_id="tiny",
        dataset_revision="fixed",
        dataset_manifest_fingerprint="abc",
        model_id="model",
        preprocessing={"input_size": 2},
        source_split="train",
        protocol_digest="protocol",
    )
    assert repeated.to_bytes() == manifest.to_bytes()


def test_mean_artifact_recovers_matching_payload_without_manifest(tmp_path: Path) -> None:
    accumulator = ImageMeanAccumulator(2)
    accumulator.update(np.zeros((1, 1, 2, 2)), np.asarray([0]))
    accumulator.update(np.ones((1, 1, 2, 2)), np.asarray([1]))
    values = accumulator.finalize()
    kwargs = {
        "dataset_id": "tiny",
        "dataset_revision": "fixed",
        "dataset_manifest_fingerprint": "abc",
        "model_id": "model",
        "preprocessing": {"input_size": 2},
        "source_split": "train",
        "protocol_digest": "protocol",
    }
    write_image_mean_artifact(tmp_path, values, **kwargs)
    (tmp_path / "manifest.json").unlink()

    recovered = write_image_mean_artifact(tmp_path, values, **kwargs)

    assert load_manifest(tmp_path / "manifest.json").to_bytes() == recovered.to_bytes()
    validate_manifest_files(tmp_path, recovered)


def test_mean_artifact_rejects_mismatched_payload_without_manifest(tmp_path: Path) -> None:
    accumulator = ImageMeanAccumulator(2)
    accumulator.update(np.zeros((1, 1, 2, 2)), np.asarray([0]))
    accumulator.update(np.ones((1, 1, 2, 2)), np.asarray([1]))
    values = accumulator.finalize()
    kwargs = {
        "dataset_id": "tiny",
        "dataset_revision": "fixed",
        "dataset_manifest_fingerprint": "abc",
        "model_id": "model",
        "preprocessing": {"input_size": 2},
        "source_split": "train",
        "protocol_digest": "protocol",
    }
    write_image_mean_artifact(tmp_path, values, **kwargs)
    (tmp_path / "manifest.json").unlink()
    np.savez_compressed(
        tmp_path / "means.npz",
        dataset_mean=np.zeros_like(values.dataset_mean),
        class_means=values.class_means,
        class_counts=values.class_counts,
        sample_count=np.asarray([values.sample_count], dtype=np.int64),
    )

    with pytest.raises(ArtifactValidationError, match="existing mean payload array differs"):
        write_image_mean_artifact(tmp_path, values, **kwargs)
