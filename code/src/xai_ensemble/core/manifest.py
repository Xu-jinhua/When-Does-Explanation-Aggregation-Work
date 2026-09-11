"""Immutable, content-addressed artifact manifests and validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .atomic import atomic_write_bytes, canonical_json_bytes, sha256_file

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ManifestError(ValueError):
    """A manifest is malformed or contradicts a local artifact."""


class ImmutableManifestError(ManifestError):
    """An existing manifest would be replaced by different content."""


class ArtifactValidationError(ManifestError):
    """One or more artifact files failed integrity or dimensional validation."""


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ManifestError(f"artifact path must be a safe POSIX relative path: {value!r}")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise ManifestError("artifact path cannot be empty")
    return normalized


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ManifestError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Expected identity and optional tabular/tensor dimensions of one shard."""

    path: str
    sha256: str
    size_bytes: int
    rows: int | None = None
    shape: tuple[int, ...] | None = None
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_relative_path(self.path))
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ManifestError(f"invalid SHA-256 for {self.path!r}")
        if self.size_bytes < 0:
            raise ManifestError("size_bytes cannot be negative")
        if self.rows is not None and self.rows < 0:
            raise ManifestError("rows cannot be negative")
        if self.shape is not None:
            shape = tuple(int(dimension) for dimension in self.shape)
            if any(dimension < 0 for dimension in shape):
                raise ManifestError("shape dimensions cannot be negative")
            object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.rows is not None:
            value["rows"] = self.rows
        if self.shape is not None:
            value["shape"] = list(self.shape)
        if self.media_type is not None:
            value["media_type"] = self.media_type
        if self.metadata:
            value["metadata"] = _thaw_json(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactEntry:
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            rows=None if value.get("rows") is None else int(value["rows"]),
            shape=None
            if value.get("shape") is None
            else tuple(int(item) for item in value["shape"]),
            media_type=value.get("media_type"),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """A frozen manifest published only after all referenced shards exist."""

    artifact_id: str
    kind: str
    created_at: str
    producer: str
    files: tuple[ArtifactEntry, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not _ARTIFACT_ID_RE.fullmatch(self.artifact_id):
            raise ManifestError(f"invalid artifact_id: {self.artifact_id!r}")
        if not self.kind or not self.producer:
            raise ManifestError("kind and producer must be non-empty")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ManifestError("created_at must be an ISO-8601 timestamp") from error
        files = tuple(self.files)
        if not files:
            raise ManifestError("a manifest must contain at least one file")
        paths = [entry.path for entry in files]
        if len(paths) != len(set(paths)):
            raise ManifestError("manifest file paths must be unique")
        if paths != sorted(paths):
            files = tuple(sorted(files, key=lambda entry: entry.path))
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        if self.schema_version != 1:
            raise ManifestError(f"unsupported manifest schema version {self.schema_version}")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        producer: str,
        files: Iterable[ArtifactEntry],
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        created_at: str | None = None,
    ) -> ArtifactManifest:
        entries = tuple(sorted(files, key=lambda entry: entry.path))
        frozen_metadata = metadata or {}
        if artifact_id is None:
            identity = {
                "kind": kind,
                "producer": producer,
                "files": [entry.to_dict() for entry in entries],
                "metadata": _thaw_json(_freeze_json(frozen_metadata)),
            }
            artifact_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:32]
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            created_at=created_at
            or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            producer=producer,
            files=entries,
            metadata=frozen_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "created_at": self.created_at,
            "producer": self.producer,
            "files": [entry.to_dict() for entry in self.files],
            "metadata": _thaw_json(self.metadata),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            artifact_id=str(value["artifact_id"]),
            kind=str(value["kind"]),
            created_at=str(value["created_at"]),
            producer=str(value["producer"]),
            files=tuple(ArtifactEntry.from_dict(item) for item in value["files"]),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def from_bytes(cls, content: bytes) -> ArtifactManifest:
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ManifestError("manifest is not valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ManifestError("manifest root must be an object")
        return cls.from_dict(value)

    def entry(self, path: str) -> ArtifactEntry:
        normalized = _validate_relative_path(path)
        for entry in self.files:
            if entry.path == normalized:
                return entry
        raise KeyError(normalized)


def write_manifest(
    path: str | os.PathLike[str], manifest: ArtifactManifest
) -> None:
    """Create a manifest once; accepting an identical repeated publication."""

    destination = Path(path)
    content = manifest.to_bytes()
    if destination.exists():
        existing = ArtifactManifest.from_bytes(destination.read_bytes())
        if existing.to_bytes() != content:
            raise ImmutableManifestError(f"refusing to replace immutable manifest {destination}")
        return
    try:
        atomic_write_bytes(destination, content, overwrite=False)
    except FileExistsError as error:
        # Another process won the publication race. It is safe only if identical.
        existing = ArtifactManifest.from_bytes(destination.read_bytes())
        if existing.to_bytes() != content:
            raise ImmutableManifestError(f"manifest publication race at {destination}") from error
    else:
        # The FUSE fallback uses a cooperative local lock around atomic rename.
        # Read back the commit marker before reporting publication success.
        if destination.read_bytes() != content:
            raise ImmutableManifestError(f"manifest publication changed unexpectedly: {destination}")


def load_manifest(path: str | os.PathLike[str]) -> ArtifactManifest:
    return ArtifactManifest.from_bytes(Path(path).read_bytes())


def probe_rows_shape(
    path: str | os.PathLike[str], *, array_key: str | None = None
) -> tuple[int | None, tuple[int, ...] | None]:
    """Read cheap format metadata, importing optional packages only on demand."""

    artifact = Path(path)
    suffix = artifact.suffix.lower()
    if suffix == ".npy":
        try:
            import numpy as np
        except ImportError as error:
            raise ArtifactValidationError("NumPy is required to inspect .npy shape") from error
        array = np.load(artifact, mmap_mode="r", allow_pickle=False)
        shape = tuple(int(item) for item in array.shape)
        return (shape[0] if shape else 1), shape
    if suffix == ".npz":
        try:
            import numpy as np
        except ImportError as error:
            raise ArtifactValidationError("NumPy is required to inspect .npz shape") from error
        with np.load(artifact, allow_pickle=False) as archive:
            if array_key is None:
                preferred = ("ranks", "patch_scores", "attributions", "sample_ids")
                array_key = next((key for key in preferred if key in archive.files), None)
            if array_key is None and len(archive.files) == 1:
                array_key = archive.files[0]
            if array_key is None:
                raise ArtifactValidationError(
                    "shape validation for multi-array .npz requires metadata.array_key"
                )
            if array_key not in archive.files:
                raise ArtifactValidationError(
                    f"array key {array_key!r} is absent from {artifact.name}"
                )
            shape = tuple(int(item) for item in archive[array_key].shape)
        return (shape[0] if shape else 1), shape
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ArtifactValidationError("PyArrow is required to inspect Parquet") from error
        metadata = parquet.ParquetFile(artifact).metadata
        return int(metadata.num_rows), (int(metadata.num_rows), int(metadata.num_columns))
    if suffix in {".jsonl", ".ndjson"}:
        with artifact.open("rb") as handle:
            rows = sum(1 for line in handle if line.strip())
        return rows, (rows,)
    if suffix == ".csv":
        with artifact.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return 0, (0, 0)
            rows = sum(1 for _ in reader)
        return rows, (rows, len(header))
    return None, None


def make_entry(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    rows: int | None = None,
    shape: Sequence[int] | None = None,
    infer_dimensions: bool = True,
    media_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactEntry:
    normalized = _validate_relative_path(relative_path)
    source = Path(root) / normalized
    if not source.is_file():
        raise ArtifactValidationError(f"artifact file does not exist: {source}")
    if infer_dimensions and rows is None and shape is None:
        rows, inferred_shape = probe_rows_shape(
            source, array_key=None if metadata is None else metadata.get("array_key")
        )
        shape = inferred_shape
    return ArtifactEntry(
        path=normalized,
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        rows=rows,
        shape=None if shape is None else tuple(int(item) for item in shape),
        media_type=media_type,
        metadata=metadata or {},
    )


def validate_entry(
    root: str | os.PathLike[str],
    entry: ArtifactEntry,
    *,
    verify_hash: bool = True,
    verify_dimensions: bool = True,
) -> None:
    source = Path(root) / entry.path
    if not source.is_file():
        raise ArtifactValidationError(f"missing artifact file: {entry.path}")
    actual_size = source.stat().st_size
    if actual_size != entry.size_bytes:
        raise ArtifactValidationError(
            f"size mismatch for {entry.path}: expected {entry.size_bytes}, got {actual_size}"
        )
    if verify_hash:
        actual_hash = sha256_file(source)
        if actual_hash != entry.sha256:
            raise ArtifactValidationError(
                f"SHA-256 mismatch for {entry.path}: expected {entry.sha256}, got {actual_hash}"
            )
    if verify_dimensions and (entry.rows is not None or entry.shape is not None):
        actual_rows, actual_shape = probe_rows_shape(
            source, array_key=entry.metadata.get("array_key")
        )
        if actual_rows is None and actual_shape is None:
            raise ArtifactValidationError(
                f"cannot validate rows/shape for unsupported format: {entry.path}"
            )
        if entry.rows is not None and actual_rows != entry.rows:
            raise ArtifactValidationError(
                f"row mismatch for {entry.path}: expected {entry.rows}, got {actual_rows}"
            )
        if entry.shape is not None and actual_shape != entry.shape:
            raise ArtifactValidationError(
                f"shape mismatch for {entry.path}: expected {entry.shape}, got {actual_shape}"
            )


def validate_manifest_files(
    root: str | os.PathLike[str],
    manifest: ArtifactManifest,
    *,
    verify_hash: bool = True,
    verify_dimensions: bool = True,
) -> None:
    for entry in manifest.files:
        validate_entry(
            root,
            entry,
            verify_hash=verify_hash,
            verify_dimensions=verify_dimensions,
        )
