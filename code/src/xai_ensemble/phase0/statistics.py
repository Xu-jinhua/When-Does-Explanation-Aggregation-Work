"""Streaming train-split image statistics used as masking/baseline artifacts."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xai_ensemble.core.manifest import (
    ArtifactManifest,
    ArtifactValidationError,
    load_manifest,
    make_entry,
    validate_manifest_files,
    write_manifest,
)


@dataclass(frozen=True)
class ImageMeanValues:
    dataset_mean: NDArray[np.float32]
    class_means: NDArray[np.float32]
    class_counts: NDArray[np.int64]
    sample_count: int


class ImageMeanAccumulator:
    """Numerically stable bounded-memory mean of equal-size raw images."""

    def __init__(self, num_classes: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self._sum: NDArray[np.float64] | None = None
        self._class_sums: NDArray[np.float64] | None = None
        self._class_counts = np.zeros(self.num_classes, dtype=np.int64)
        self._count = 0

    def update(self, images: ArrayLike, labels: ArrayLike) -> None:
        values = images.detach().cpu().numpy() if hasattr(images, "detach") else images
        batch = np.asarray(values)
        target_values = labels.detach().cpu().numpy() if hasattr(labels, "detach") else labels
        targets = np.asarray(target_values).reshape(-1)
        if batch.ndim != 4 or batch.shape[0] == 0:
            raise ValueError("images must have non-empty BCHW shape")
        if targets.shape != (batch.shape[0],):
            raise ValueError("labels must contain one value per image")
        if not np.issubdtype(batch.dtype, np.floating):
            raise TypeError("raw images must use a floating dtype")
        if (
            not np.all(np.isfinite(batch))
            or float(batch.min()) < 0
            or float(batch.max()) > 1
        ):
            raise ValueError("images must be finite values in raw [0, 1] space")
        if not np.issubdtype(targets.dtype, np.integer):
            if not np.all(np.equal(targets, np.floor(targets))):
                raise TypeError("labels must be integers")
            targets = targets.astype(np.int64)
        targets = targets.astype(np.int64, copy=False)
        if np.any(targets < 0) or np.any(targets >= self.num_classes):
            raise ValueError("label is outside the registered class range")

        batch64 = batch.astype(np.float64, copy=False)
        if self._sum is None:
            self._sum = np.zeros(batch.shape[1:], dtype=np.float64)
            self._class_sums = np.zeros(
                (self.num_classes, *batch.shape[1:]), dtype=np.float64
            )
        elif batch.shape[1:] != self._sum.shape:
            raise ValueError(
                f"image shape changed from {self._sum.shape} to {batch.shape[1:]}"
            )
        assert self._class_sums is not None
        self._sum += np.sum(batch64, axis=0, dtype=np.float64)
        for label in np.unique(targets):
            selected = batch64[targets == label]
            self._class_sums[label] += np.sum(selected, axis=0, dtype=np.float64)
            self._class_counts[label] += selected.shape[0]
        self._count += batch.shape[0]

    def finalize(self, *, require_every_class: bool = True) -> ImageMeanValues:
        if self._count == 0 or self._sum is None or self._class_sums is None:
            raise ValueError("cannot finalize an empty image mean")
        if require_every_class and np.any(self._class_counts == 0):
            missing = np.flatnonzero(self._class_counts == 0).tolist()
            raise ValueError(f"training split has no samples for classes {missing}")
        class_means = np.zeros_like(self._class_sums, dtype=np.float64)
        counts = self._class_counts.reshape(-1, 1, 1, 1)
        np.divide(self._class_sums, counts, out=class_means, where=counts > 0)
        return ImageMeanValues(
            dataset_mean=(self._sum / self._count).astype(np.float32),
            class_means=class_means.astype(np.float32),
            class_counts=self._class_counts.copy(),
            sample_count=self._count,
        )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def write_image_mean_artifact(
    output_directory: str | Path,
    values: ImageMeanValues,
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_manifest_fingerprint: str,
    model_id: str,
    preprocessing: dict[str, Any],
    source_split: str,
    protocol_digest: str,
) -> ArtifactManifest:
    if not source_split.lower().startswith("train"):
        raise ValueError("image means used for evaluation must come from a train split")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "means.npz"
    metadata = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint,
        "model_id": model_id,
        "preprocessing": preprocessing,
        "source_split": source_split,
        "protocol_digest": protocol_digest,
        "sample_count": values.sample_count,
        "definition": "pixelwise arithmetic mean after deterministic model resize/crop",
        "domain": "raw_0_1",
    }
    if data_path.exists():
        manifest_path = root / "manifest.json"
        expected = {
            "dataset_mean": values.dataset_mean,
            "class_means": values.class_means,
            "class_counts": values.class_counts,
            "sample_count": np.asarray([values.sample_count], dtype=np.int64),
        }
        with np.load(data_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ArtifactValidationError(
                    "existing mean payload arrays differ from the requested artifact"
                )
            for name, expected_array in expected.items():
                observed = np.asarray(archive[name])
                if (
                    observed.dtype != expected_array.dtype
                    or observed.shape != expected_array.shape
                    or not np.array_equal(observed, expected_array)
                ):
                    raise ArtifactValidationError(
                        f"existing mean payload array differs: {name}"
                    )
        if not manifest_path.is_file():
            # An interrupted FUSE publication can leave a payload before the
            # manifest commit marker.  Recompute-and-compare above proves it
            # is the exact requested mean before sealing it; no payload bytes
            # are replaced or silently adopted.
            entry = make_entry(
                root,
                "means.npz",
                metadata={"array_key": "dataset_mean"},
                media_type="application/x-npz",
            )
            manifest = ArtifactManifest.create(
                kind="train_image_means",
                producer="xai_ensemble.phase0.statistics",
                files=[entry],
                metadata=metadata,
            )
            write_manifest(manifest_path, manifest)
            return manifest
        manifest = load_manifest(manifest_path)
        if (
            manifest.kind != "train_image_means"
            or manifest.producer != "xai_ensemble.phase0.statistics"
            or dict(manifest.metadata) != metadata
        ):
            raise ArtifactValidationError(
                f"existing mean manifest is incompatible with {data_path}"
            )
        validate_manifest_files(root, manifest)
        try:
            entry = manifest.entry("means.npz")
        except KeyError as error:
            raise ArtifactValidationError(
                "existing mean manifest does not declare means.npz"
            ) from error
        if entry.metadata.get("array_key") != "dataset_mean":
            raise ArtifactValidationError(
                "existing mean manifest does not identify dataset_mean"
            )
        return manifest
    _atomic_npz(
        data_path,
        dataset_mean=values.dataset_mean,
        class_means=values.class_means,
        class_counts=values.class_counts,
        sample_count=np.asarray([values.sample_count], dtype=np.int64),
    )
    entry = make_entry(
        root,
        "means.npz",
        metadata={"array_key": "dataset_mean"},
        media_type="application/x-npz",
    )
    manifest = ArtifactManifest.create(
        kind="train_image_means",
        producer="xai_ensemble.phase0.statistics",
        files=[entry],
        metadata=metadata,
    )
    write_manifest(root / "manifest.json", manifest)
    return manifest
