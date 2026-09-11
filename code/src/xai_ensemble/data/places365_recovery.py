"""Fail-closed direct-Parquet recovery for the pinned Places365 sources.

``datasets.load_dataset`` materializes the large Places365 Parquet source as
an Arrow cache.  That is not safe for the constrained server cache used by the
full matrix.  This module reads only the immutable, content-addressed Parquet
snapshot after an explicit validation step has written a small marker outside
the provider cache.  It deliberately has no fallback to a datasets builder.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import numbers
import os
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, read_json

RECOVERY_SCHEMA = "places365-direct-parquet-v1"
TRAIN_DATASET_ID = "Andron00e/Places365-custom"
TRAIN_REVISION = "7895e75528d78c16e0c31182ce02e541f44ccaaf"
VALIDATION_DATASET_ID = "dpdl-benchmark/Places365-Validation"
VALIDATION_REVISION = "f11b9b3c7ddd678ba92fd6862296b0d42c8723bb"

# These values were reviewed from the immutable Hugging Face LFS trees at the
# pinned revisions.  The train manifest digest covers ordered
# ``{name, sha256, size}`` records for all 322 Parquet files, rather than a
# locally generated marker that could otherwise be self-consistent.
PLACES365_TRAIN_LFS_MANIFEST_SHA256 = (
    "9bfd4342cf0319245433f40cf92795eba30bccc8081498f3e391640e211895a5"
)
PLACES365_VALIDATION_LFS_MANIFEST_SHA256 = (
    "4d4ae32fa7e13c05000cafdc90a1786413cab76b80b159f689f94c6e13bad6a0"
)
PLACES365_TRAIN_LABEL_NAMES_SHA256 = (
    "683a44ce72b0d675177facc27d068bd5b1043ae4be9f603604935507f8b12b7c"
)
PLACES365_VALIDATION_LABEL_NAMES_SHA256 = (
    "f5af5c46aacc1cc518a9d630c1818e84fd95e2c09c842cecb86c8b829b1240c8"
)
PLACES365_TRAIN_ARROW_SCHEMA_SHA256 = (
    "ba3230096d922bb919d59ba5cd3abb5dbc1e2690dc61f6932b7f34c293537ebc"
)
PLACES365_VALIDATION_ARROW_SCHEMA_SHA256 = (
    "5336654d5749c02a1c66b1cc8e184628a29ffda6ec8d4ee035a8e34dc486915c"
)
PLACES365_STANDARD_LABEL_MAPPING_SCHEMA = "places365-standard-label-index-v1"
PLACES365_STANDARD_LABEL_MAPPING_SHA256 = (
    "15b9a668579948a910da502ffea3fc01dce1c289e424abd8a75f705275b65ee7"
)

# These bounds are per DataLoader process.  Tables are only raw Arrow values;
# decoded PIL images and tensors are deliberately never retained here.
MAX_OPEN_PARQUET_FILES = 12
ROW_GROUP_CACHE_MAX_BYTES = 64 * 2**20


@dataclass(frozen=True, slots=True)
class Places365RecoverySource:
    """The immutable raw-Parquet contract for one Places365 source split."""

    key: str
    dataset_id: str
    revision: str
    source_split: str
    label_column: str
    file_count: int
    total_rows: int
    num_classes: int
    root_environment: str
    marker_environment: str
    file_sha256: tuple[str, ...] | None = None
    cache_path_prefix: tuple[str, ...] = ("hub",)
    expected_lfs_manifest_sha256: str | None = None
    expected_total_file_bytes: int | None = None
    expected_label_names_sha256: str | None = None
    expected_arrow_schema_sha256: str | None = None
    standard_label_mapping_schema: str | None = None
    expected_standard_label_mapping_sha256: str | None = None
    validation_examples_per_class: int | None = None
    allowed_duplicate_files: tuple[tuple[str, str], ...] = ()

    def file_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.source_split}-{index:05d}-of-{self.file_count:05d}.parquet"
            for index in range(self.file_count)
        )


PLACES365_TRAIN_SOURCE = Places365RecoverySource(
    key="places365-custom-train",
    dataset_id=TRAIN_DATASET_ID,
    revision=TRAIN_REVISION,
    source_split="train",
    label_column="labels",
    file_count=322,
    total_rows=1_839_960,
    num_classes=365,
    root_environment="XAI_PLACES365_PARQUET_ROOT",
    marker_environment="XAI_PLACES365_RECOVERY_MARKER",
    expected_lfs_manifest_sha256=PLACES365_TRAIN_LFS_MANIFEST_SHA256,
    expected_total_file_bytes=113_509_227_101,
    expected_label_names_sha256=PLACES365_TRAIN_LABEL_NAMES_SHA256,
    expected_arrow_schema_sha256=PLACES365_TRAIN_ARROW_SCHEMA_SHA256,
)
PLACES365_VALIDATION_SOURCE = Places365RecoverySource(
    key="places365-validation-train",
    dataset_id=VALIDATION_DATASET_ID,
    revision=VALIDATION_REVISION,
    source_split="train",
    label_column="label",
    file_count=5,
    total_rows=36_500,
    num_classes=365,
    root_environment="XAI_PLACES365_VALIDATION_PARQUET_ROOT",
    marker_environment="XAI_PLACES365_VALIDATION_RECOVERY_MARKER",
    file_sha256=(
        "d42e989798fa52d6b8de1b1e658d9815494553d68aefcda754bb404fec263ee1",
        "e8e2926ffd9f475b5ff87249d80ba5fe3c097c35b905f801fd6d2c007bf29004",
        "b84457f203f1e95bbeec89c14e6b8e9f2f36d704a7edca34e147cebc28041dd7",
        "d15b1392ed5833477ec8d1c4806ff1d78db994969237fb3191dfef1b691d4979",
        "baec146abb4b309eba080ca999f7d5b83b94d3eb89d14db0f67855a2831cc263",
    ),
    # This snapshot was sealed by huggingface_hub.snapshot_download directly.
    # Its immutable cache is deliberately outside HF_HOME/hub and must not be
    # copied or moved merely to fit datasets' usual cache layout.
    cache_path_prefix=(),
    expected_lfs_manifest_sha256=PLACES365_VALIDATION_LFS_MANIFEST_SHA256,
    expected_total_file_bytes=2_241_697_128,
    expected_label_names_sha256=PLACES365_VALIDATION_LABEL_NAMES_SHA256,
    expected_arrow_schema_sha256=PLACES365_VALIDATION_ARROW_SCHEMA_SHA256,
    standard_label_mapping_schema=PLACES365_STANDARD_LABEL_MAPPING_SCHEMA,
    expected_standard_label_mapping_sha256=PLACES365_STANDARD_LABEL_MAPPING_SHA256,
    validation_examples_per_class=100,
    allowed_duplicate_files=tuple(
        (
            f"val-{index:05d}-of-00005.parquet",
            f"train-{index:05d}-of-00005.parquet",
        )
        for index in range(5)
    ),
)
RECOVERY_SOURCES = (PLACES365_TRAIN_SOURCE, PLACES365_VALIDATION_SOURCE)


def recovery_source_by_key(key: str) -> Places365RecoverySource:
    """Return one known Places365 recovery source by its stable local key."""

    for source in RECOVERY_SOURCES:
        if source.key == key:
            return source
    known = ", ".join(source.key for source in RECOVERY_SOURCES)
    raise ValueError(f"Unknown Places365 recovery source {key!r}; expected one of {known}")


def recovery_source_for_config(
    source_config: Mapping[str, Any],
) -> Places365RecoverySource | None:
    """Match only the exact pinned Places365 raw source contracts."""

    dataset_id = str(source_config.get("dataset_id", ""))
    revision = str(source_config.get("revision", ""))
    source_split = str(source_config.get("split", ""))
    label_column = str(source_config.get("label_column", "label"))
    for source in RECOVERY_SOURCES:
        if (
            dataset_id == source.dataset_id
            and revision == source.revision
            and source_split == source.source_split
            and label_column == source.label_column
        ):
            return source
    return None


def places365_recovery_environment(
    *,
    hf_cache_root: str | Path,
    marker_root: str | Path,
) -> dict[str, str]:
    """Return the explicit worker environment for both locked raw sources."""

    cache_root = Path(hf_cache_root)
    markers = Path(marker_root)
    result: dict[str, str] = {}
    for source in RECOVERY_SOURCES:
        repository = f"datasets--{source.dataset_id.replace('/', '--')}"
        raw_root = (
            cache_root
            .joinpath(*source.cache_path_prefix)
            / repository
            / "snapshots"
            / source.revision
            / "data"
        )
        result[source.root_environment] = str(raw_root)
        result[source.marker_environment] = str(markers / f"{source.key}.json")
    return result


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - dependency is required on the server
        raise RuntimeError("Places365 direct-Parquet recovery requires pyarrow") from error
    return pa, pq


def _schema_sha256(schema: Any) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _label_names(schema: Any, *, label_column: str, num_classes: int) -> tuple[str, ...]:
    raw_metadata = schema.metadata or {}
    payload = raw_metadata.get(b"huggingface")
    if payload is None:
        raise RuntimeError("Places365 Parquet schema has no Hugging Face feature metadata")
    try:
        value = json.loads(payload.decode("utf-8"))
        features = value["info"]["features"]
        feature = features[label_column]
        names_value = feature["names"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Places365 Parquet label metadata is invalid") from error
    if not isinstance(names_value, Sequence) or isinstance(names_value, (str, bytes)):
        raise RuntimeError("Places365 Parquet label names must be a sequence")
    names = tuple(str(name) for name in names_value)
    if len(names) != num_classes or len(set(names)) != num_classes:
        raise RuntimeError(
            f"Places365 Parquet labels must declare {num_classes} distinct classes, got {len(names)}"
        )
    return names


def _validate_schema(
    schema: Any,
    *,
    source: Places365RecoverySource,
    pa: Any,
) -> tuple[str, ...]:
    if source.label_column not in schema.names or "image" not in schema.names:
        raise RuntimeError(
            f"Places365 Parquet schema must contain image and {source.label_column!r} columns"
        )
    label_field = schema.field(source.label_column)
    if not pa.types.is_int64(label_field.type):
        raise RuntimeError(
            f"Places365 label column {source.label_column!r} must be int64, got {label_field.type}"
        )
    image_field = schema.field("image")
    if not pa.types.is_struct(image_field.type):
        raise RuntimeError(f"Places365 image column must be a struct, got {image_field.type}")
    image_fields = {field.name: field.type for field in image_field.type}
    if not pa.types.is_binary(image_fields.get("bytes")) or not pa.types.is_string(
        image_fields.get("path")
    ):
        raise RuntimeError("Places365 image struct must contain binary bytes and string path fields")
    return _label_names(schema, label_column=source.label_column, num_classes=source.num_classes)


def _marker_without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "marker_digest"}


def _marker_error(message: str) -> RuntimeError:
    return RuntimeError(f"Places365 direct-Parquet recovery marker is invalid: {message}")


def _is_sha256_name(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _close_parquet_file(value: Any) -> None:
    """Close a PyArrow Parquet handle when its LRU entry is evicted."""

    close = getattr(value, "close", None)
    if callable(close):
        close()


def _duplicate_file_map(source: Places365RecoverySource) -> dict[str, str]:
    """Return the explicit optional duplicate-file allowance for one source."""

    result: dict[str, str] = {}
    canonical = set(source.file_names())
    for duplicate, original in source.allowed_duplicate_files:
        if duplicate in result or duplicate in canonical or original not in canonical:
            raise RuntimeError(f"Places365 duplicate-file contract is invalid for {source.key}")
        result[duplicate] = original
    return result


def _parquet_inventory(
    root: Path,
    source: Places365RecoverySource,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Fail closed on every unexpected top-level Parquet file."""

    expected_names = source.file_names()
    expected = set(expected_names)
    allowed_duplicates = _duplicate_file_map(source)
    present = {
        path.name: path
        for path in root.iterdir()
        if path.name.endswith(".parquet")
    }
    missing = tuple(
        name for name in expected_names if name not in present or not present[name].is_file()
    )
    unexpected = tuple(sorted(set(present) - expected - set(allowed_duplicates)))
    if missing or unexpected:
        details = [f"expected={len(expected_names)}"]
        if missing:
            details.append(f"missing={len(missing)}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected[:3])}")
        raise RuntimeError(
            f"Places365 raw files do not match {source.key}: {' '.join(details)}"
        )
    duplicates = {
        name: path
        for name, path in present.items()
        if name in allowed_duplicates
    }
    if any(not path.is_file() for path in duplicates.values()):
        raise RuntimeError(f"Places365 allowed duplicate is not a regular file for {source.key}")
    return ({name: present[name] for name in expected_names}, duplicates)


def _lfs_manifest_records(files: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact fixed upstream manifest payload for marker hashing."""

    records: list[dict[str, Any]] = []
    for item in files:
        try:
            name = str(item["name"])
            digest = str(item["sha256"])
            size = int(item["size"])
        except (KeyError, TypeError, ValueError) as error:
            raise _marker_error("file entry cannot form an LFS manifest record") from error
        if not _is_sha256_name(digest) or size < 0:
            raise _marker_error("file entry has an invalid LFS manifest record")
        records.append({"name": name, "sha256": digest, "size": size})
    return records


def _standard_label_mapping_digest(source: Places365RecoverySource) -> str | None:
    """Validate and return the fixed identity map for numeric validation labels."""

    schema = source.standard_label_mapping_schema
    expected = source.expected_standard_label_mapping_sha256
    if schema is None and expected is None:
        return None
    if not schema or not expected:
        raise RuntimeError(f"Places365 standard label-map contract is incomplete for {source.key}")
    actual = object_sha256({"schema": schema, "indices": list(range(source.num_classes))})
    if actual != expected:
        raise RuntimeError(f"Places365 standard label-map contract digest is invalid for {source.key}")
    return actual


def _source_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"Places365 raw Parquet root is missing: {path}")
    return path.resolve()


def _metadata_for_file(
    path: Path,
    *,
    source: Places365RecoverySource,
    pa: Any,
    pq: Any,
    require_hash: bool,
) -> tuple[dict[str, Any], str, tuple[str, ...], Counter[int]]:
    parquet_file = pq.ParquetFile(path)
    try:
        schema = parquet_file.schema_arrow
        label_names = _validate_schema(schema, source=source, pa=pa)
        table = pq.read_table(path, columns=[source.label_column])
        labels = table.column(source.label_column).to_pylist()
        if len(labels) != parquet_file.metadata.num_rows:
            raise RuntimeError(f"Places365 label read count differs from Parquet metadata: {path.name}")
        if any(isinstance(label, bool) or not isinstance(label, numbers.Integral) for label in labels):
            raise RuntimeError(f"Places365 labels must be numeric integers in {path.name}")
        label_counts = Counter(int(label) for label in labels)
        if any(label < 0 or label >= source.num_classes for label in label_counts):
            raise RuntimeError(f"Places365 label is out of range in {path.name}")
        resolved = path.resolve()
        stat = path.stat()
        digest = file_sha256(path) if require_hash else ""
        if require_hash and _is_sha256_name(resolved.name) and resolved.name != digest:
            raise RuntimeError(f"Places365 blob name/content digest mismatch for {path.name}")
        return (
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "resolved_path": str(resolved),
                "blob_name": resolved.name,
                "sha256": digest,
                "row_count": parquet_file.metadata.num_rows,
            },
            _schema_sha256(schema),
            label_names,
            label_counts,
        )
    finally:
        _close_parquet_file(parquet_file)


def validate_places365_parquet(
    source: Places365RecoverySource,
    *,
    parquet_root: str | Path,
    marker_path: str | Path,
) -> dict[str, Any]:
    """Validate immutable raw files and atomically publish their completion marker.

    This is intentionally explicit and may hash a large source once.  Runtime
    readers only accept the marker and perform lightweight identity checks.
    """

    pa, pq = _require_pyarrow()
    root = _source_root(parquet_root)
    expected_names = source.file_names()
    expected_paths, duplicate_paths = _parquet_inventory(root, source)

    files: list[dict[str, Any]] = []
    schema_digest: str | None = None
    label_names: tuple[str, ...] | None = None
    observed_label_counts: Counter[int] = Counter()
    for index, name in enumerate(expected_names):
        item, current_schema_digest, current_label_names, label_counts = _metadata_for_file(
            expected_paths[name],
            source=source,
            pa=pa,
            pq=pq,
            require_hash=True,
        )
        if source.file_sha256 is not None and item["sha256"] != source.file_sha256[index]:
            raise RuntimeError(f"Places365 pinned digest mismatch for {name}")
        if schema_digest is None:
            schema_digest = current_schema_digest
        elif schema_digest != current_schema_digest:
            raise RuntimeError(f"Places365 Parquet schemas differ at {name}")
        if label_names is None:
            label_names = current_label_names
        elif label_names != current_label_names:
            raise RuntimeError(f"Places365 label names differ at {name}")
        files.append(item)
        observed_label_counts.update(label_counts)

    total_rows = sum(int(item["row_count"]) for item in files)
    if total_rows != source.total_rows:
        raise RuntimeError(
            f"Places365 row count mismatch for {source.key}: "
            f"expected={source.total_rows} actual={total_rows}"
        )
    expected_labels = set(range(source.num_classes))
    if set(observed_label_counts) != expected_labels:
        raise RuntimeError(
            f"Places365 labels do not cover exactly 0..{source.num_classes - 1}: "
            f"observed={len(observed_label_counts)}"
        )
    if schema_digest is None or label_names is None:  # pragma: no cover - fixed source counts are positive
        raise RuntimeError("Places365 recovery source has no Parquet files")

    lfs_manifest_records = _lfs_manifest_records(files)
    lfs_manifest_sha256 = object_sha256(lfs_manifest_records)
    total_file_bytes = sum(record["size"] for record in lfs_manifest_records)
    label_names_sha256 = object_sha256(list(label_names))
    if source.expected_lfs_manifest_sha256 is not None and (
        lfs_manifest_sha256 != source.expected_lfs_manifest_sha256
    ):
        raise RuntimeError(f"Places365 fixed LFS manifest digest mismatch for {source.key}")
    if source.expected_total_file_bytes is not None and total_file_bytes != source.expected_total_file_bytes:
        raise RuntimeError(f"Places365 fixed file-byte total mismatch for {source.key}")
    if source.expected_label_names_sha256 is not None and (
        label_names_sha256 != source.expected_label_names_sha256
    ):
        raise RuntimeError(f"Places365 fixed label-name digest mismatch for {source.key}")
    if source.expected_arrow_schema_sha256 is not None and (
        schema_digest != source.expected_arrow_schema_sha256
    ):
        raise RuntimeError(f"Places365 fixed Arrow schema digest mismatch for {source.key}")
    validation_counts: list[int] | None = None
    mapping_digest = _standard_label_mapping_digest(source)
    if source.validation_examples_per_class is not None:
        validation_counts = [observed_label_counts[index] for index in range(source.num_classes)]
        if any(count != source.validation_examples_per_class for count in validation_counts):
            raise RuntimeError(
                f"Places365 validation must contain exactly {source.validation_examples_per_class} "
                "examples per class"
            )
        if label_names != tuple(str(index) for index in range(source.num_classes)):
            raise RuntimeError("Places365 validation labels do not use the fixed identity mapping")
    duplicate_records: list[dict[str, Any]] = []
    for duplicate_name, original_name in _duplicate_file_map(source).items():
        duplicate_path = duplicate_paths.get(duplicate_name)
        if duplicate_path is None:
            continue
        duplicate_digest = file_sha256(duplicate_path)
        original = files[expected_names.index(original_name)]
        if duplicate_path.stat().st_size != original["size"] or duplicate_digest != original["sha256"]:
            raise RuntimeError(
                f"Places365 allowed duplicate {duplicate_name} does not match {original_name}"
            )
        duplicate_records.append(
            {
                "name": duplicate_name,
                "matches": original_name,
                "size": original["size"],
                "sha256": original["sha256"],
            }
        )

    marker: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA,
        "source_key": source.key,
        "dataset_id": source.dataset_id,
        "revision": source.revision,
        "source_split": source.source_split,
        "label_column": source.label_column,
        "num_classes": source.num_classes,
        "raw_root": str(root),
        "file_count": source.file_count,
        "total_rows": total_rows,
        "arrow_schema_sha256": schema_digest,
        "label_names": list(label_names),
        "label_names_sha256": label_names_sha256,
        "lfs_manifest_sha256": lfs_manifest_sha256,
        "total_file_bytes": total_file_bytes,
        "files": files,
    }
    marker["expected_lfs_manifest_sha256"] = source.expected_lfs_manifest_sha256
    marker["expected_total_file_bytes"] = source.expected_total_file_bytes
    marker["expected_label_names_sha256"] = source.expected_label_names_sha256
    marker["expected_arrow_schema_sha256"] = source.expected_arrow_schema_sha256
    if mapping_digest is not None:
        marker["standard_label_mapping_schema"] = source.standard_label_mapping_schema
        marker["standard_label_mapping_sha256"] = mapping_digest
    if validation_counts is not None:
        marker["validation_examples_per_class"] = source.validation_examples_per_class
        marker["validation_class_counts"] = validation_counts
    if duplicate_records:
        marker["allowed_duplicate_files"] = duplicate_records
    marker["marker_digest"] = object_sha256(marker)
    atomic_write_json(marker_path, marker)
    return marker


def _load_marker(
    source: Places365RecoverySource,
    *,
    parquet_root: str | Path,
    marker_path: str | Path,
) -> tuple[Path, Mapping[str, Any]]:
    root = _source_root(parquet_root)
    path = Path(marker_path).expanduser()
    if not path.is_file():
        raise _marker_error(f"missing {path}")
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise _marker_error(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise _marker_error("root is not a mapping")
    expected_identity = {
        "schema": RECOVERY_SCHEMA,
        "source_key": source.key,
        "dataset_id": source.dataset_id,
        "revision": source.revision,
        "source_split": source.source_split,
        "label_column": source.label_column,
        "num_classes": source.num_classes,
        "raw_root": str(root),
        "file_count": source.file_count,
        "total_rows": source.total_rows,
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise _marker_error(f"{key} differs from the locked source contract")
    if value.get("marker_digest") != object_sha256(_marker_without_digest(value)):
        raise _marker_error("digest mismatch")
    names = value.get("label_names")
    if not isinstance(names, list) or len(names) != source.num_classes:
        raise _marker_error("label names are missing or have the wrong length")
    if value.get("label_names_sha256") != object_sha256(names):
        raise _marker_error("label-name digest mismatch")
    if (
        source.expected_label_names_sha256 is not None
        and value.get("label_names_sha256") != source.expected_label_names_sha256
    ):
        raise _marker_error("label-name digest differs from the fixed source contract")
    if (
        source.expected_arrow_schema_sha256 is not None
        and value.get("arrow_schema_sha256") != source.expected_arrow_schema_sha256
    ):
        raise _marker_error("Arrow schema digest differs from the fixed source contract")
    files = value.get("files")
    if not isinstance(files, list) or [item.get("name") for item in files if isinstance(item, Mapping)] != list(
        source.file_names()
    ):
        raise _marker_error("file names do not match the locked source contract")
    try:
        lfs_records = _lfs_manifest_records(files)
    except RuntimeError as error:
        raise _marker_error(str(error)) from error
    lfs_manifest_sha256 = object_sha256(lfs_records)
    total_file_bytes = sum(record["size"] for record in lfs_records)
    if value.get("lfs_manifest_sha256") != lfs_manifest_sha256:
        raise _marker_error("LFS manifest digest mismatch")
    if value.get("total_file_bytes") != total_file_bytes:
        raise _marker_error("LFS file-byte total mismatch")
    if (
        source.expected_lfs_manifest_sha256 is not None
        and lfs_manifest_sha256 != source.expected_lfs_manifest_sha256
    ):
        raise _marker_error("LFS manifest digest differs from the fixed source contract")
    if (
        source.expected_total_file_bytes is not None
        and total_file_bytes != source.expected_total_file_bytes
    ):
        raise _marker_error("LFS file-byte total differs from the fixed source contract")
    if value.get("expected_lfs_manifest_sha256") != source.expected_lfs_manifest_sha256:
        raise _marker_error("stored fixed LFS manifest contract differs")
    if value.get("expected_total_file_bytes") != source.expected_total_file_bytes:
        raise _marker_error("stored fixed LFS byte contract differs")
    if value.get("expected_label_names_sha256") != source.expected_label_names_sha256:
        raise _marker_error("stored fixed label-name contract differs")
    if value.get("expected_arrow_schema_sha256") != source.expected_arrow_schema_sha256:
        raise _marker_error("stored fixed Arrow schema contract differs")
    mapping_digest = _standard_label_mapping_digest(source)
    if mapping_digest is not None:
        if value.get("standard_label_mapping_schema") != source.standard_label_mapping_schema:
            raise _marker_error("standard label-map schema differs from the fixed source contract")
        if value.get("standard_label_mapping_sha256") != mapping_digest:
            raise _marker_error("standard label-map digest differs from the fixed source contract")
        counts = value.get("validation_class_counts")
        expected_count = source.validation_examples_per_class
        if (
            value.get("validation_examples_per_class") != expected_count
            or not isinstance(counts, list)
            or len(counts) != source.num_classes
            or any(count != expected_count for count in counts)
        ):
            raise _marker_error("validation class-count contract differs from the fixed source")
        if names != list(map(str, range(source.num_classes))):
            raise _marker_error("validation labels do not use the fixed identity mapping")
    duplicate_contract = _duplicate_file_map(source)
    marker_duplicates = value.get("allowed_duplicate_files", [])
    if not isinstance(marker_duplicates, list):
        raise _marker_error("allowed duplicate-file records are invalid")
    if {
        str(item.get("name")): str(item.get("matches"))
        for item in marker_duplicates
        if isinstance(item, Mapping)
    } != {
        name: original for name, original in duplicate_contract.items() if (root / name).is_file()
    }:
        raise _marker_error("allowed duplicate-file records differ from the raw root")
    return root, value


def _fast_validate_marker_files(
    source: Places365RecoverySource,
    *,
    root: Path,
    marker: Mapping[str, Any],
) -> tuple[tuple[Path, ...], tuple[int, ...], str, tuple[str, ...]]:
    pa, pq = _require_pyarrow()
    files_value = marker["files"]
    if not isinstance(files_value, list):  # guarded in _load_marker
        raise _marker_error("files are not a list")
    expected_schema = str(marker.get("arrow_schema_sha256", ""))
    if len(expected_schema) != 64:
        raise _marker_error("schema digest has an invalid length")
    paths: list[Path] = []
    row_counts: list[int] = []
    marker_names = tuple(str(name) for name in marker["label_names"])
    expected_paths, duplicate_paths = _parquet_inventory(root, source)
    for item in files_value:
        if not isinstance(item, Mapping):
            raise _marker_error("file entry is not a mapping")
        name = str(item.get("name", ""))
        path = expected_paths.get(name)
        if path is None:
            raise _marker_error(f"raw file is not part of the fixed inventory: {name}")
        if not path.is_file():
            raise _marker_error(f"raw file is missing: {name}")
        stat = path.stat()
        for key, actual in (
            ("size", stat.st_size),
            ("mtime_ns", stat.st_mtime_ns),
            ("device", stat.st_dev),
            ("inode", stat.st_ino),
            ("resolved_path", str(path.resolve())),
            ("blob_name", path.resolve().name),
        ):
            if item.get(key) != actual:
                raise _marker_error(f"raw file identity changed for {name}: {key}")
        sha256 = str(item.get("sha256", ""))
        if len(sha256) != 64:
            raise _marker_error(f"stored hash is invalid for {name}")
        if _is_sha256_name(path.resolve().name) and path.resolve().name != sha256:
            raise _marker_error(f"content-addressed blob name differs for {name}")
        parquet_file = pq.ParquetFile(path)
        try:
            schema = parquet_file.schema_arrow
            names = _validate_schema(schema, source=source, pa=pa)
            if _schema_sha256(schema) != expected_schema:
                raise _marker_error(f"Parquet schema changed for {name}")
            if names != marker_names:
                raise _marker_error(f"label names changed for {name}")
            row_count = parquet_file.metadata.num_rows
            if item.get("row_count") != row_count:
                raise _marker_error(f"row count changed for {name}")
        finally:
            _close_parquet_file(parquet_file)
        paths.append(path)
        row_counts.append(row_count)
    if sum(row_counts) != source.total_rows:
        raise _marker_error("total row count changed")
    for record in marker.get("allowed_duplicate_files", []):
        if not isinstance(record, Mapping):
            raise _marker_error("allowed duplicate-file record is invalid")
        duplicate = str(record.get("name", ""))
        original = str(record.get("matches", ""))
        duplicate_path = duplicate_paths.get(duplicate)
        if duplicate_path is None:
            raise _marker_error(f"allowed duplicate disappeared: {duplicate}")
        original_item = next((item for item in files_value if item.get("name") == original), None)
        if not isinstance(original_item, Mapping):
            raise _marker_error(f"allowed duplicate source is missing: {original}")
        if duplicate_path.stat().st_size != original_item.get("size"):
            raise _marker_error(f"allowed duplicate size changed: {duplicate}")
        if _is_sha256_name(duplicate_path.resolve().name) and (
            duplicate_path.resolve().name != original_item.get("sha256")
        ):
            raise _marker_error(f"allowed duplicate content-addressed blob changed: {duplicate}")
    return tuple(paths), tuple(row_counts), expected_schema, marker_names


class ParquetMapDataset:
    """A read-only map-style image dataset backed by validated Parquet shards."""

    def __init__(
        self,
        source: Places365RecoverySource,
        *,
        root: Path,
        marker: Mapping[str, Any],
    ) -> None:
        _, pq = _require_pyarrow()
        paths, row_counts, _schema_digest, label_names = _fast_validate_marker_files(
            source,
            root=root,
            marker=marker,
        )
        self._source = source
        self._paths = paths
        self._row_counts = row_counts
        offsets = [0]
        for count in row_counts:
            offsets.append(offsets[-1] + count)
        self._shard_offsets = tuple(offsets)
        row_group_offsets: list[tuple[int, ...]] = []
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            try:
                row_group_offsets.append(self._row_group_boundaries(parquet_file))
            finally:
                _close_parquet_file(parquet_file)
        self._row_group_offsets = tuple(row_group_offsets)
        self._label_names = label_names
        self._file_pid = os.getpid()
        self._files: OrderedDict[int, Any] = OrderedDict()
        self._row_groups: OrderedDict[tuple[int, int], tuple[Any, int]] = OrderedDict()
        self._row_group_cache_bytes = 0

    @staticmethod
    def _row_group_boundaries(parquet_file: Any) -> tuple[int, ...]:
        offsets = [0]
        for index in range(parquet_file.num_row_groups):
            offsets.append(offsets[-1] + parquet_file.metadata.row_group(index).num_rows)
        if offsets[-1] != parquet_file.metadata.num_rows:
            raise _marker_error("Parquet row-group metadata is inconsistent")
        return tuple(offsets)

    def __len__(self) -> int:
        return self._shard_offsets[-1]

    @property
    def features(self) -> Mapping[str, Any]:
        class LabelFeature:
            def __init__(self, names: tuple[str, ...]) -> None:
                self.names = names

        return {"image": object(), self._source.label_column: LabelFeature(self._label_names)}

    def _reset_after_fork(self) -> None:
        if self._file_pid != os.getpid():
            self._file_pid = os.getpid()
            # Forked DataLoader workers must not retain Arrow tables or file
            # handles inherited from the parent process.
            for parquet_file in self._files.values():
                _close_parquet_file(parquet_file)
            self._files = OrderedDict()
            self._row_groups = OrderedDict()
            self._row_group_cache_bytes = 0

    def close(self) -> None:
        """Release only local Parquet file descriptors held by this process."""

        for parquet_file in self._files.values():
            _close_parquet_file(parquet_file)
        self._files.clear()
        self._row_groups.clear()
        self._row_group_cache_bytes = 0

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is non-deterministic
        try:
            self.close()
        except Exception:
            return

    @property
    def open_parquet_file_count(self) -> int:
        """Expose the bounded per-process handle count for diagnostics/tests."""

        self._reset_after_fork()
        return len(self._files)

    @property
    def row_group_cache_bytes(self) -> int:
        """Expose raw Arrow-table cache usage, never decoded image memory."""

        self._reset_after_fork()
        return self._row_group_cache_bytes

    def _parquet_file(self, shard_index: int) -> Any:
        self._reset_after_fork()
        try:
            value = self._files.pop(shard_index)
        except KeyError:
            _, pq = _require_pyarrow()
            value = pq.ParquetFile(self._paths[shard_index])
        self._files[shard_index] = value
        while len(self._files) > MAX_OPEN_PARQUET_FILES:
            _evicted_index, evicted = self._files.popitem(last=False)
            _close_parquet_file(evicted)
        return value

    def _row_group(self, shard_index: int, row_group_index: int) -> Any:
        key = (shard_index, row_group_index)
        try:
            value, byte_count = self._row_groups.pop(key)
            self._row_group_cache_bytes -= byte_count
        except KeyError:
            value = self._parquet_file(shard_index).read_row_group(
                row_group_index,
                columns=["image", self._source.label_column],
            )
            byte_count = int(getattr(value, "nbytes", 0))
            if byte_count < 0:
                raise RuntimeError(
                    "Places365 row-group table reports a negative byte size"
                ) from None
        if byte_count > ROW_GROUP_CACHE_MAX_BYTES:
            return value
        self._row_groups[key] = (value, byte_count)
        self._row_group_cache_bytes += byte_count
        while self._row_group_cache_bytes > ROW_GROUP_CACHE_MAX_BYTES:
            _evicted_key, (_evicted, evicted_bytes) = self._row_groups.popitem(last=False)
            self._row_group_cache_bytes -= evicted_bytes
        return value

    def _normalized_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Places365 direct-Parquet rows require an integer index")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Places365 direct-Parquet row index is out of range")
        return index

    def _row_location(self, index: int) -> tuple[int, int, int]:
        normalized = self._normalized_index(index)
        shard_index = bisect.bisect_right(self._shard_offsets, normalized) - 1
        local_index = normalized - self._shard_offsets[shard_index]
        boundaries = self._row_group_offsets[shard_index]
        row_group_index = bisect.bisect_right(boundaries, local_index) - 1
        row_index = local_index - boundaries[row_group_index]
        return shard_index, row_group_index, row_index

    def row_group_key(self, index: int) -> tuple[int, int]:
        """Return an immutable row-group key without reading or decoding a row."""

        shard_index, row_group_index, _row_index = self._row_location(index)
        return shard_index, row_group_index

    @staticmethod
    def _decode_image(value: Any) -> Any:
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - Pillow is required by Phase 0
            raise RuntimeError("Places365 direct-Parquet recovery requires Pillow") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("Places365 image value is not a mapping")
        payload = value.get("bytes")
        if payload is None:
            image_path = value.get("path")
            if not image_path:
                raise RuntimeError("Places365 image has neither bytes nor a path")
            payload = Path(str(image_path)).read_bytes()
        with Image.open(BytesIO(bytes(payload))) as opened:
            image = opened.convert("RGB")
            image.load()
        return image

    def __getitem__(self, index: int) -> Any:
        shard_index, row_group_index, row_index = self._row_location(index)
        row_group = self._row_group(shard_index, row_group_index)
        image = row_group.column("image")[row_index].as_py()
        label = row_group.column(self._source.label_column)[row_index].as_py()
        return {"image": self._decode_image(image), self._source.label_column: int(label)}

    def column_values(self, column: str) -> list[int]:
        """Read the label column without decoding any image values."""

        if column != self._source.label_column:
            raise KeyError(f"Places365 direct-Parquet does not expose column {column!r}")
        _, pq = _require_pyarrow()
        values: list[int] = []
        for path in self._paths:
            table = pq.read_table(path, columns=[column])
            values.extend(int(value) for value in table.column(column).to_pylist())
        if len(values) != len(self):
            raise RuntimeError("Places365 label column length changed while reading")
        return values


def load_validated_places365_parquet(
    source: Places365RecoverySource,
    *,
    parquet_root: str | Path,
    marker_path: str | Path,
) -> ParquetMapDataset:
    """Load a marked source or fail closed before any HF builder is imported."""

    root, marker = _load_marker(source, parquet_root=parquet_root, marker_path=marker_path)
    return ParquetMapDataset(source, root=root, marker=marker)


def load_places365_recovery_source(source_config: Mapping[str, Any]) -> ParquetMapDataset:
    """Load the exact recovery source selected by a composite split contract."""

    source = recovery_source_for_config(source_config)
    if source is None:
        raise ValueError("Source is not an exact Places365 direct-Parquet recovery contract")
    parquet_root = os.environ.get(source.root_environment)
    marker_path = os.environ.get(source.marker_environment)
    if not parquet_root or not marker_path:
        raise RuntimeError(
            f"Places365 direct-Parquet recovery requires {source.root_environment} and "
            f"{source.marker_environment}"
        )
    return load_validated_places365_parquet(
        source,
        parquet_root=parquet_root,
        marker_path=marker_path,
    )


def preflight_places365_recovery_environment(environment: Mapping[str, str] | None = None) -> None:
    """Prove both locked recovery markers before a scheduler opens its SQLite DB.

    This deliberately performs only marker and Parquet-metadata reads.  It never
    writes a marker, invokes a Hugging Face builder, or creates queue state.
    """

    values = os.environ if environment is None else environment
    for source in RECOVERY_SOURCES:
        parquet_root = values.get(source.root_environment)
        marker_path = values.get(source.marker_environment)
        if not parquet_root or not marker_path:
            raise RuntimeError(
                f"Places365 scheduler preflight requires {source.root_environment} and "
                f"{source.marker_environment}"
            )
        dataset = load_validated_places365_parquet(
            source,
            parquet_root=parquet_root,
            marker_path=marker_path,
        )
        dataset.close()


__all__ = [
    "PLACES365_TRAIN_SOURCE",
    "PLACES365_VALIDATION_SOURCE",
    "ParquetMapDataset",
    "Places365RecoverySource",
    "load_places365_recovery_source",
    "load_validated_places365_parquet",
    "places365_recovery_environment",
    "preflight_places365_recovery_environment",
    "recovery_source_by_key",
    "recovery_source_for_config",
    "validate_places365_parquet",
]
