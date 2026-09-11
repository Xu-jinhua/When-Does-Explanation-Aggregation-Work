"""Fail-closed local Parquet recovery for pinned Hugging Face snapshots.

The full-matrix server keeps a verified Hugging Face snapshot outside the
normal ``datasets`` Arrow cache.  This module reads that snapshot directly
only when the dataset, revision, shard inventory, schema, and content
manifest all match a fixed source contract.  A matching contract never falls
back to ``datasets.load_dataset``; a missing contract leaves the normal
provider path available for datasets that do not use this recovery source.
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

RECOVERY_SCHEMA = "hf-direct-parquet-v1"
RECOVERY_HUB_ENV = "XAI_HF_PARQUET_RECOVERY_HUB"
RECOVERY_MARKER_ROOT_ENV = "XAI_HF_PARQUET_RECOVERY_MARKER_ROOT"

MAX_OPEN_PARQUET_FILES = 12
ROW_GROUP_CACHE_MAX_BYTES = 64 * 2**20


@dataclass(frozen=True, slots=True)
class HfParquetRecoverySource:
    """Immutable raw-Parquet contract for one pinned provider split."""

    key: str
    dataset_id: str
    revision: str
    source_split: str
    image_column: str
    label_column: str
    text_column: str | None
    file_count: int
    total_rows: int
    num_classes: int
    expected_file_manifest_sha256: str
    expected_lfs_manifest_sha256: str
    expected_total_file_bytes: int
    expected_arrow_schema_sha256: str
    expected_label_names_sha256: str | None = None
    require_label_names: bool = False
    expected_label_counts: tuple[int, ...] | None = None
    marker_environment: str = ""

    def file_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.source_split}-{index:05d}-of-{self.file_count:05d}.parquet"
            for index in range(self.file_count)
        )


# These are derived from the preserved pinned snapshot and bind the recovery
# to the exact upstream files.  Marker generation recomputes all values; the
# constants merely prevent a self-consistent replacement snapshot from being
# accepted at runtime.
FOOD101_TRAIN_SOURCE = HfParquetRecoverySource(
    key="food101-hf-train",
    dataset_id="ethz/food101",
    revision="83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9",
    source_split="train",
    image_column="image",
    label_column="label",
    text_column=None,
    file_count=8,
    total_rows=75_750,
    num_classes=101,
    expected_file_manifest_sha256="afaaceb8177a32370be399a14b9fc609b4442ab8f4914c0757849766db1b6d88",
    expected_lfs_manifest_sha256="39871afd63675fac6fc312ca12935a2fa86231f1213cad84cb200b4a17e24ac9",
    expected_total_file_bytes=3_798_520_995,
    expected_arrow_schema_sha256="aa7513ca35a26afa91ac3c4eeb387d0153f3df3cebb5562d53342ce23cf902d1",
    expected_label_names_sha256="ab9db3a50d3f60ae967fecf60b375753e4412b53b9c675343c64fb0653e51430",
    require_label_names=True,
    expected_label_counts=(750,) * 101,
    marker_environment="XAI_FOOD101_TRAIN_RECOVERY_MARKER",
)
FOOD101_VALIDATION_SOURCE = HfParquetRecoverySource(
    key="food101-hf-validation",
    dataset_id="ethz/food101",
    revision="83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9",
    source_split="validation",
    image_column="image",
    label_column="label",
    text_column=None,
    file_count=3,
    total_rows=25_250,
    num_classes=101,
    expected_file_manifest_sha256="38130b0c99d3e5161edb7a42808db0c1184d44fa5f219f5eef9f35ae47d80067",
    expected_lfs_manifest_sha256="8b142b53bc0d233bd79402874d9d704ff1dab21fe2d036132d4729608bbcf03d",
    expected_total_file_bytes=1_261_451_313,
    expected_arrow_schema_sha256="aa7513ca35a26afa91ac3c4eeb387d0153f3df3cebb5562d53342ce23cf902d1",
    expected_label_names_sha256="ab9db3a50d3f60ae967fecf60b375753e4412b53b9c675343c64fb0653e51430",
    require_label_names=True,
    expected_label_counts=(250,) * 101,
    marker_environment="XAI_FOOD101_VALIDATION_RECOVERY_MARKER",
)
IMAGENET100_TRAIN_SOURCE = HfParquetRecoverySource(
    key="imagenet100-hf-train",
    dataset_id="ilee0022/ImageNet100",
    revision="c55b2f2967c034db17be30f7d430e41c80fd4281",
    source_split="train",
    image_column="image",
    label_column="label",
    text_column="text",
    file_count=30,
    total_rows=117_000,
    num_classes=100,
    expected_file_manifest_sha256="c5618279a5c8c58427897d8eb499c9b05ceca0012b7f64e9c51eaff59a0287dd",
    expected_lfs_manifest_sha256="613b63cd4dc9d23ad9bc90e936716bdeb174afdc98577e3a6471bcc9ac4967c8",
    expected_total_file_bytes=14_981_586_173,
    expected_arrow_schema_sha256="357cb0ef2c2f16759232799172de6b39ecc7549a3582fe998c17cf59b66269b2",
    expected_label_counts=(
        1175,
        1170,
        1178,
        1180,
        1179,
        1170,
        1166,
        1187,
        1181,
        1166,
        1183,
        1170,
        1180,
        1179,
        1158,
        1171,
        1171,
        1159,
        1156,
        1177,
        1177,
        1175,
        1167,
        1173,
        1166,
        1146,
        1152,
        1165,
        1183,
        1184,
        1174,
        1159,
        1162,
        1179,
        1164,
        1176,
        1156,
        1159,
        1164,
        1190,
        1176,
        1165,
        1169,
        1199,
        1167,
        1162,
        1164,
        1174,
        1162,
        1173,
        1171,
        1158,
        1173,
        1179,
        1158,
        1171,
        1176,
        1158,
        1186,
        1170,
        1168,
        1173,
        1184,
        1172,
        1174,
        1157,
        1160,
        1159,
        1176,
        1172,
        1170,
        1180,
        1171,
        1165,
        1177,
        1143,
        1179,
        1180,
        1174,
        1186,
        1171,
        1172,
        1181,
        1145,
        1169,
        1193,
        1162,
        1163,
        1182,
        1161,
        1176,
        1160,
        1153,
        1163,
        1171,
        1187,
        1146,
        1153,
        1173,
        1161,
    ),
    marker_environment="XAI_IMAGENET100_TRAIN_RECOVERY_MARKER",
)
IMAGENET100_VALIDATION_SOURCE = HfParquetRecoverySource(
    key="imagenet100-hf-validation",
    dataset_id="ilee0022/ImageNet100",
    revision="c55b2f2967c034db17be30f7d430e41c80fd4281",
    source_split="validation",
    image_column="image",
    label_column="label",
    text_column="text",
    file_count=4,
    total_rows=13_000,
    num_classes=100,
    expected_file_manifest_sha256="9c4b031166e34dfe745d4e01fd4209ba79507900195f2136b1e472063bce9145",
    expected_lfs_manifest_sha256="08c9ba336cd96a16ef924ed678d2a604729301e395add1d2ce2081af29ec4ac4",
    expected_total_file_bytes=1_651_313_722,
    expected_arrow_schema_sha256="357cb0ef2c2f16759232799172de6b39ecc7549a3582fe998c17cf59b66269b2",
    expected_label_counts=(
        125,
        130,
        122,
        120,
        121,
        130,
        134,
        113,
        119,
        134,
        117,
        130,
        120,
        121,
        142,
        129,
        129,
        141,
        144,
        123,
        123,
        125,
        133,
        127,
        134,
        154,
        148,
        135,
        117,
        116,
        126,
        141,
        138,
        121,
        136,
        124,
        144,
        141,
        136,
        110,
        124,
        135,
        131,
        101,
        133,
        138,
        136,
        126,
        138,
        127,
        129,
        142,
        127,
        121,
        142,
        129,
        124,
        142,
        114,
        130,
        132,
        127,
        116,
        128,
        126,
        143,
        140,
        141,
        124,
        128,
        130,
        120,
        129,
        135,
        123,
        157,
        121,
        120,
        126,
        114,
        129,
        128,
        119,
        155,
        131,
        107,
        138,
        137,
        118,
        139,
        124,
        140,
        147,
        137,
        129,
        113,
        154,
        147,
        127,
        139,
    ),
    marker_environment="XAI_IMAGENET100_VALIDATION_RECOVERY_MARKER",
)
IMAGENET100_TEST_SOURCE = HfParquetRecoverySource(
    key="imagenet100-hf-test",
    dataset_id="ilee0022/ImageNet100",
    revision="c55b2f2967c034db17be30f7d430e41c80fd4281",
    source_split="test",
    image_column="image",
    label_column="label",
    text_column="text",
    file_count=2,
    total_rows=5_000,
    num_classes=100,
    expected_file_manifest_sha256="01975547fd8b547717ed68b92058f58f4de38104a16ee0e301ec8712a3015a8d",
    expected_lfs_manifest_sha256="64eb06e3f532013496cfc93bf68c076ac01c269f692da906c5296bd576fcf2d2",
    expected_total_file_bytes=734_545_584,
    expected_arrow_schema_sha256="357cb0ef2c2f16759232799172de6b39ecc7549a3582fe998c17cf59b66269b2",
    expected_label_counts=(50,) * 100,
    marker_environment="XAI_IMAGENET100_TEST_RECOVERY_MARKER",
)
RECOVERY_SOURCES = (
    FOOD101_TRAIN_SOURCE,
    FOOD101_VALIDATION_SOURCE,
    IMAGENET100_TRAIN_SOURCE,
    IMAGENET100_VALIDATION_SOURCE,
    IMAGENET100_TEST_SOURCE,
)


def recovery_source_by_key(key: str) -> HfParquetRecoverySource:
    for source in RECOVERY_SOURCES:
        if source.key == key:
            return source
    known = ", ".join(source.key for source in RECOVERY_SOURCES)
    raise ValueError(f"Unknown HF Parquet recovery source {key!r}; expected one of {known}")


def recovery_source_for_config(
    source_config: Mapping[str, Any],
) -> HfParquetRecoverySource | None:
    """Match only an exact dataset/revision/split/column contract."""

    dataset_id = str(source_config.get("dataset_id", ""))
    revision = str(source_config.get("revision", ""))
    source_split = str(source_config.get("split", ""))
    image_column = str(source_config.get("image_column", "image"))
    label_column = str(source_config.get("label_column", "label"))
    text_column_value = source_config.get("text_column")
    text_column = None if text_column_value is None else str(text_column_value)
    for source in RECOVERY_SOURCES:
        if (
            dataset_id == source.dataset_id
            and revision == source.revision
            and source_split == source.source_split
            and image_column == source.image_column
            and label_column == source.label_column
            and text_column == source.text_column
        ):
            return source
    return None


def _hub_source_root(hub_root: str | Path, source: HfParquetRecoverySource) -> Path:
    repository = f"datasets--{source.dataset_id.replace('/', '--')}"
    return (
        Path(hub_root).expanduser().resolve() / repository / "snapshots" / source.revision / "data"
    )


def recovery_environment(*, hub_root: str | Path, marker_root: str | Path) -> dict[str, str]:
    """Return runtime bindings for all fixed sources."""

    hub = str(Path(hub_root).expanduser().resolve())
    markers = Path(marker_root).expanduser().resolve()
    result = {RECOVERY_HUB_ENV: hub}
    result.update(
        {
            source.marker_environment: str(markers / f"{source.key}.json")
            for source in RECOVERY_SOURCES
        }
    )
    return result


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - required server dependency
        raise RuntimeError("HF direct-Parquet recovery requires pyarrow") from error
    return pa, pq


def _schema_sha256(schema: Any) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _is_sha256_name(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _label_names(schema: Any, source: HfParquetRecoverySource) -> tuple[str, ...] | None:
    payload = (schema.metadata or {}).get(b"huggingface")
    if payload is None:
        if source.require_label_names:
            raise RuntimeError("HF Parquet schema has no Hugging Face feature metadata")
        return None
    try:
        features = json.loads(payload.decode("utf-8"))["info"]["features"]
        feature = features[source.label_column]
        names = feature.get("names")
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("HF Parquet label metadata is invalid") from error
    if names is None:
        if source.require_label_names:
            raise RuntimeError("HF Parquet source is missing fixed label names")
        return None
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise RuntimeError("HF Parquet label names must be a sequence")
    result = tuple(str(name) for name in names)
    if len(result) != source.num_classes or len(set(result)) != source.num_classes:
        raise RuntimeError("HF Parquet label names have the wrong cardinality")
    return result


def _validate_schema(
    schema: Any, source: HfParquetRecoverySource, pa: Any
) -> tuple[str, ...] | None:
    required = {source.image_column, source.label_column}
    if source.text_column is not None:
        required.add(source.text_column)
    if not required.issubset(schema.names):
        raise RuntimeError(
            f"HF Parquet schema must contain {sorted(required)}; got {list(schema.names)}"
        )
    label_field = schema.field(source.label_column)
    if not pa.types.is_int64(label_field.type):
        raise RuntimeError(f"HF label column must be int64, got {label_field.type}")
    image_field = schema.field(source.image_column)
    if not pa.types.is_struct(image_field.type):
        raise RuntimeError(f"HF image column must be a struct, got {image_field.type}")
    image_fields = {field.name: field.type for field in image_field.type}
    if not pa.types.is_binary(image_fields.get("bytes")) or not pa.types.is_string(
        image_fields.get("path")
    ):
        raise RuntimeError("HF image struct must contain binary bytes and string path fields")
    return _label_names(schema, source)


def _source_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"HF direct-Parquet root is missing: {path}")
    return path


def _inventory(root: Path, source: HfParquetRecoverySource) -> dict[str, Path]:
    expected = source.file_names()
    present = {path.name: path for path in root.iterdir() if path.name.endswith(".parquet")}
    # Hugging Face snapshots keep every split in one ``data/`` directory.  A
    # split validator must therefore tolerate the other registered split
    # inventories from the same pinned repository while rejecting unknown
    # Parquet files.
    known = set(expected)
    known.update(
        name
        for candidate in RECOVERY_SOURCES
        if candidate.dataset_id == source.dataset_id and candidate.revision == source.revision
        for name in candidate.file_names()
    )
    missing = [name for name in expected if name not in present or not present[name].is_file()]
    unexpected = sorted(set(present) - known)
    if missing or unexpected:
        details = [f"expected={len(expected)}"]
        if missing:
            details.append(f"missing={','.join(missing[:3])}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected[:3])}")
        raise RuntimeError(f"HF raw files do not match {source.key}: {' '.join(details)}")
    return {name: present[name] for name in expected}


def _metadata_for_file(
    path: Path,
    *,
    source: HfParquetRecoverySource,
    pa: Any,
    pq: Any,
) -> tuple[dict[str, Any], str, tuple[str, ...] | None, Counter[int]]:
    parquet_file = pq.ParquetFile(path)
    try:
        schema = parquet_file.schema_arrow
        label_names = _validate_schema(schema, source, pa)
        table = pq.read_table(path, columns=[source.label_column])
        labels = table.column(source.label_column).to_pylist()
        if len(labels) != parquet_file.metadata.num_rows:
            raise RuntimeError(f"HF label read count differs from Parquet metadata: {path.name}")
        if any(
            isinstance(label, bool) or not isinstance(label, numbers.Integral) for label in labels
        ):
            raise RuntimeError(f"HF labels must be integer values in {path.name}")
        counts = Counter(int(label) for label in labels)
        if any(label < 0 or label >= source.num_classes for label in counts):
            raise RuntimeError(f"HF label is out of range in {path.name}")
        resolved = path.resolve()
        digest = file_sha256(path)
        if not _is_sha256_name(resolved.name) or resolved.name != digest:
            raise RuntimeError(f"HF source file is not a content-addressed blob: {path.name}")
        stat = path.stat()
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
            counts,
        )
    finally:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()


def _lfs_records(files: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": str(item["name"]), "sha256": str(item["sha256"]), "size": int(item["size"])}
        for item in files
    ]


def _file_records(files: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item["name"]),
            "sha256": str(item["sha256"]),
            "size": int(item["size"]),
            "row_count": int(item["row_count"]),
        }
        for item in files
    ]


def validate_hf_parquet(
    source: HfParquetRecoverySource,
    *,
    parquet_root: str | Path,
    marker_path: str | Path,
) -> dict[str, Any]:
    """Hash and validate one source, then atomically seal its marker."""

    pa, pq = _require_pyarrow()
    root = _source_root(parquet_root)
    paths = _inventory(root, source)
    files: list[dict[str, Any]] = []
    schema_digest: str | None = None
    label_names: tuple[str, ...] | None = None
    counts: Counter[int] = Counter()
    for name in source.file_names():
        item, current_schema, current_names, current_counts = _metadata_for_file(
            paths[name], source=source, pa=pa, pq=pq
        )
        if schema_digest is None:
            schema_digest = current_schema
        elif schema_digest != current_schema:
            raise RuntimeError(f"HF Parquet schemas differ at {name}")
        if label_names is None:
            label_names = current_names
        elif label_names != current_names:
            raise RuntimeError(f"HF label names differ at {name}")
        files.append(item)
        counts.update(current_counts)

    if sum(int(item["row_count"]) for item in files) != source.total_rows:
        raise RuntimeError(f"HF row count mismatch for {source.key}")
    if set(counts) != set(range(source.num_classes)):
        raise RuntimeError(f"HF labels do not cover 0..{source.num_classes - 1}")
    if (
        source.expected_label_counts is not None
        and tuple(counts[index] for index in range(source.num_classes))
        != source.expected_label_counts
    ):
        raise RuntimeError(f"HF class counts differ from the fixed contract for {source.key}")
    if schema_digest != source.expected_arrow_schema_sha256:
        raise RuntimeError(
            f"HF Arrow schema digest differs from the fixed contract for {source.key}"
        )
    if source.expected_label_names_sha256 is not None:
        if (
            label_names is None
            or object_sha256(list(label_names)) != source.expected_label_names_sha256
        ):
            raise RuntimeError(
                f"HF label-name digest differs from the fixed contract for {source.key}"
            )

    lfs_digest = object_sha256(_lfs_records(files))
    file_digest = object_sha256(_file_records(files))
    total_bytes = sum(int(item["size"]) for item in files)
    if lfs_digest != source.expected_lfs_manifest_sha256:
        raise RuntimeError(
            f"HF LFS manifest digest differs from the fixed contract for {source.key}"
        )
    if file_digest != source.expected_file_manifest_sha256:
        raise RuntimeError(
            f"HF file manifest digest differs from the fixed contract for {source.key}"
        )
    if total_bytes != source.expected_total_file_bytes:
        raise RuntimeError(f"HF file-byte total differs from the fixed contract for {source.key}")

    marker: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA,
        "source_key": source.key,
        "dataset_id": source.dataset_id,
        "revision": source.revision,
        "source_split": source.source_split,
        "image_column": source.image_column,
        "label_column": source.label_column,
        "text_column": source.text_column,
        "num_classes": source.num_classes,
        "raw_root": str(root),
        "file_count": source.file_count,
        "total_rows": source.total_rows,
        "arrow_schema_sha256": schema_digest,
        "label_names": None if label_names is None else list(label_names),
        "label_names_sha256": None if label_names is None else object_sha256(list(label_names)),
        "lfs_manifest_sha256": lfs_digest,
        "file_manifest_sha256": file_digest,
        "total_file_bytes": total_bytes,
        "class_counts": [counts[index] for index in range(source.num_classes)],
        "files": files,
        "expected_lfs_manifest_sha256": source.expected_lfs_manifest_sha256,
        "expected_file_manifest_sha256": source.expected_file_manifest_sha256,
        "expected_total_file_bytes": source.expected_total_file_bytes,
        "expected_arrow_schema_sha256": source.expected_arrow_schema_sha256,
        "expected_label_names_sha256": source.expected_label_names_sha256,
    }
    marker["marker_digest"] = object_sha256(marker)
    atomic_write_json(marker_path, marker)
    return marker


def _marker_without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "marker_digest"}


def _marker_error(message: str) -> RuntimeError:
    return RuntimeError(f"HF direct-Parquet recovery marker is invalid: {message}")


def _load_marker(
    source: HfParquetRecoverySource,
    *,
    parquet_root: str | Path,
    marker_path: str | Path,
) -> tuple[Path, Mapping[str, Any]]:
    root = _source_root(parquet_root)
    path = Path(marker_path).expanduser().resolve()
    if not path.is_file():
        raise _marker_error(f"missing {path}")
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise _marker_error(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise _marker_error("root is not a mapping")
    expected = {
        "schema": RECOVERY_SCHEMA,
        "source_key": source.key,
        "dataset_id": source.dataset_id,
        "revision": source.revision,
        "source_split": source.source_split,
        "image_column": source.image_column,
        "label_column": source.label_column,
        "text_column": source.text_column,
        "num_classes": source.num_classes,
        "raw_root": str(root),
        "file_count": source.file_count,
        "total_rows": source.total_rows,
        "expected_lfs_manifest_sha256": source.expected_lfs_manifest_sha256,
        "expected_file_manifest_sha256": source.expected_file_manifest_sha256,
        "expected_total_file_bytes": source.expected_total_file_bytes,
        "expected_arrow_schema_sha256": source.expected_arrow_schema_sha256,
        "expected_label_names_sha256": source.expected_label_names_sha256,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise _marker_error(f"{key} differs from the fixed source contract")
    if value.get("marker_digest") != object_sha256(_marker_without_digest(value)):
        raise _marker_error("digest mismatch")
    names = value.get("label_names")
    if names is not None:
        if not isinstance(names, list) or len(names) != source.num_classes:
            raise _marker_error("label names have the wrong cardinality")
        if value.get("label_names_sha256") != object_sha256(names):
            raise _marker_error("label-name digest mismatch")
    elif source.require_label_names:
        raise _marker_error("fixed label names are missing")
    if names is None and value.get("label_names_sha256") is not None:
        raise _marker_error("label-name digest is present without label names")
    if source.expected_label_names_sha256 is not None and (
        names is None or object_sha256(names) != source.expected_label_names_sha256
    ):
        raise _marker_error("label-name digest differs from the fixed source contract")
    if value.get("arrow_schema_sha256") != source.expected_arrow_schema_sha256:
        raise _marker_error("Arrow schema digest differs from the fixed source contract")
    if value.get("lfs_manifest_sha256") != source.expected_lfs_manifest_sha256:
        raise _marker_error("LFS manifest digest differs from the fixed source contract")
    if value.get("file_manifest_sha256") != source.expected_file_manifest_sha256:
        raise _marker_error("file manifest digest differs from the fixed source contract")
    files = value.get("files")
    if not isinstance(files, list) or [
        item.get("name") for item in files if isinstance(item, Mapping)
    ] != list(source.file_names()):
        raise _marker_error("file inventory differs from the fixed source")
    if len(files) != source.file_count or any(not isinstance(item, Mapping) for item in files):
        raise _marker_error("file entries are malformed")
    try:
        file_manifest_digest = object_sha256(_file_records(files))
        lfs_manifest_digest = object_sha256(_lfs_records(files))
    except (KeyError, TypeError, ValueError) as error:
        raise _marker_error("file entries are malformed") from error
    if file_manifest_digest != source.expected_file_manifest_sha256:
        raise _marker_error("file manifest digest differs from the fixed source")
    if lfs_manifest_digest != source.expected_lfs_manifest_sha256:
        raise _marker_error("LFS manifest digest differs from the fixed source")
    if value.get("total_file_bytes") != source.expected_total_file_bytes:
        raise _marker_error("file-byte total differs from the fixed source")
    counts = value.get("class_counts")
    if not isinstance(counts, list) or len(counts) != source.num_classes:
        raise _marker_error("class counts are invalid")
    if source.expected_label_counts is not None and tuple(counts) != source.expected_label_counts:
        raise _marker_error("class counts differ from the fixed source")
    return root, value


def _fast_validate_marker_files(
    source: HfParquetRecoverySource,
    *,
    root: Path,
    marker: Mapping[str, Any],
) -> tuple[tuple[Path, ...], tuple[int, ...], str, tuple[str, ...] | None]:
    """Check the marked source without rehashing large Parquet payloads.

    The marker is only an optimization for content-addressed files: every
    filesystem identity and Parquet metadata field that contributed to the
    marker is checked again.  A changed blob therefore fails before a worker
    decodes a single image.
    """

    pa, pq = _require_pyarrow()
    files_value = marker.get("files")
    if not isinstance(files_value, list):  # guarded by _load_marker
        raise _marker_error("files are not a list")
    expected_schema = str(marker.get("arrow_schema_sha256", ""))
    if not _is_sha256_name(expected_schema):
        raise _marker_error("schema digest is invalid")
    marker_names_value = marker.get("label_names")
    if marker_names_value is None:
        marker_names: tuple[str, ...] | None = None
    elif isinstance(marker_names_value, list):
        marker_names = tuple(str(name) for name in marker_names_value)
    else:
        raise _marker_error("label names are malformed")

    expected_paths = _inventory(root, source)
    paths: list[Path] = []
    row_counts: list[int] = []
    actual_records: list[dict[str, Any]] = []
    for item in files_value:
        if not isinstance(item, Mapping):
            raise _marker_error("file entry is not a mapping")
        name = str(item.get("name", ""))
        path = expected_paths.get(name)
        if path is None:
            raise _marker_error(f"raw file is not part of the fixed inventory: {name}")
        resolved = path.resolve()
        if not _is_sha256_name(resolved.name):
            raise _marker_error(f"raw file is not a content-addressed blob: {name}")
        stat = path.stat()
        for key, actual in (
            ("size", stat.st_size),
            ("mtime_ns", stat.st_mtime_ns),
            ("device", stat.st_dev),
            ("inode", stat.st_ino),
            ("resolved_path", str(resolved)),
            ("blob_name", resolved.name),
        ):
            if item.get(key) != actual:
                raise _marker_error(f"raw file identity changed for {name}: {key}")
        stored_hash = str(item.get("sha256", ""))
        if not _is_sha256_name(stored_hash) or stored_hash != resolved.name:
            raise _marker_error(f"stored blob hash is invalid for {name}")
        parquet_file = pq.ParquetFile(path)
        try:
            schema = parquet_file.schema_arrow
            names = _validate_schema(schema, source, pa)
            if _schema_sha256(schema) != expected_schema:
                raise _marker_error(f"Parquet schema changed for {name}")
            if names != marker_names:
                raise _marker_error(f"label names changed for {name}")
            row_count = int(parquet_file.metadata.num_rows)
            if item.get("row_count") != row_count:
                raise _marker_error(f"row count changed for {name}")
        finally:
            _close(parquet_file)
        actual_records.append(
            {
                "name": name,
                "sha256": stored_hash,
                "size": stat.st_size,
                "row_count": row_count,
            }
        )
        paths.append(path)
        row_counts.append(row_count)

    if object_sha256(actual_records) != source.expected_file_manifest_sha256:
        raise _marker_error("actual file manifest differs from the fixed source")
    if (
        object_sha256(
            [
                {"name": row["name"], "sha256": row["sha256"], "size": row["size"]}
                for row in actual_records
            ]
        )
        != source.expected_lfs_manifest_sha256
    ):
        raise _marker_error("actual LFS manifest differs from the fixed source")
    if sum(row_counts) != source.total_rows:
        raise _marker_error("total row count changed")
    return tuple(paths), tuple(row_counts), expected_schema, marker_names


def _close(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


class ParquetMapDataset:
    """Read-only row-addressable dataset backed by validated Parquet shards."""

    def __init__(
        self, source: HfParquetRecoverySource, *, root: Path, marker: Mapping[str, Any]
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
        for count in self._row_counts:
            offsets.append(offsets[-1] + count)
        self._offsets = tuple(offsets)
        groups: list[tuple[int, ...]] = []
        for path in self._paths:
            parquet_file = pq.ParquetFile(path)
            try:
                boundary = [0]
                for index in range(parquet_file.num_row_groups):
                    boundary.append(boundary[-1] + parquet_file.metadata.row_group(index).num_rows)
                if boundary[-1] != parquet_file.metadata.num_rows:
                    raise _marker_error("Parquet row-group metadata is inconsistent")
                groups.append(tuple(boundary))
            finally:
                _close(parquet_file)
        self._group_offsets = tuple(groups)
        self._label_names = label_names
        self._pid = os.getpid()
        self._files: OrderedDict[int, Any] = OrderedDict()
        self._row_groups: OrderedDict[tuple[int, int], tuple[Any, int]] = OrderedDict()
        self._row_group_bytes = 0

    def __len__(self) -> int:
        return self._offsets[-1]

    @property
    def features(self) -> Mapping[str, Any]:
        class LabelFeature:
            def __init__(self, names: tuple[str, ...] | None) -> None:
                self.names = names

        return {
            self._source.image_column: object(),
            self._source.label_column: LabelFeature(self._label_names),
        }

    def _reset_after_fork(self) -> None:
        if self._pid != os.getpid():
            self._pid = os.getpid()
            for value in self._files.values():
                _close(value)
            self._files.clear()
            self._row_groups.clear()
            self._row_group_bytes = 0

    def close(self) -> None:
        for value in self._files.values():
            _close(value)
        self._files.clear()
        self._row_groups.clear()
        self._row_group_bytes = 0

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is non-deterministic
        try:
            self.close()
        except Exception:
            pass

    @property
    def open_parquet_file_count(self) -> int:
        """Expose the bounded per-process handle count for diagnostics/tests."""

        self._reset_after_fork()
        return len(self._files)

    @property
    def row_group_cache_bytes(self) -> int:
        """Expose raw Arrow-table cache usage, never decoded image memory."""

        self._reset_after_fork()
        return self._row_group_bytes

    def _file(self, shard: int) -> Any:
        self._reset_after_fork()
        try:
            value = self._files.pop(shard)
        except KeyError:
            _, pq = _require_pyarrow()
            value = pq.ParquetFile(self._paths[shard])
        self._files[shard] = value
        while len(self._files) > MAX_OPEN_PARQUET_FILES:
            _, evicted = self._files.popitem(last=False)
            _close(evicted)
        return value

    def _row_group(self, shard: int, group: int) -> Any:
        key = (shard, group)
        try:
            value, size = self._row_groups.pop(key)
            self._row_group_bytes -= size
        except KeyError:
            columns = [self._source.image_column, self._source.label_column]
            if self._source.text_column is not None:
                columns.append(self._source.text_column)
            value = self._file(shard).read_row_group(group, columns=columns)
            size = int(getattr(value, "nbytes", 0))
        if size <= ROW_GROUP_CACHE_MAX_BYTES:
            self._row_groups[key] = (value, size)
            self._row_group_bytes += size
            while self._row_group_bytes > ROW_GROUP_CACHE_MAX_BYTES:
                _, (_, evicted_size) = self._row_groups.popitem(last=False)
                self._row_group_bytes -= evicted_size
        return value

    def _location(self, index: int) -> tuple[int, int, int]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("HF direct-Parquet rows require an integer index")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("HF direct-Parquet row index is out of range")
        shard = bisect.bisect_right(self._offsets, index) - 1
        local = index - self._offsets[shard]
        boundaries = self._group_offsets[shard]
        group = bisect.bisect_right(boundaries, local) - 1
        return shard, group, local - boundaries[group]

    def row_group_key(self, index: int) -> tuple[int, int]:
        shard, group, _ = self._location(index)
        return shard, group

    @staticmethod
    def _decode(value: Any) -> Any:
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - Pillow is required by Phase 0
            raise RuntimeError("HF direct-Parquet recovery requires Pillow") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("HF image value is not a mapping")
        payload = value.get("bytes")
        if payload is None:
            image_path = value.get("path")
            if not image_path:
                raise RuntimeError("HF image has neither bytes nor a path")
            payload = Path(str(image_path)).read_bytes()
        with Image.open(BytesIO(bytes(payload))) as opened:
            image = opened.convert("RGB")
            image.load()
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, group, row_index = self._location(index)
        row_group = self._row_group(shard, group)
        row: dict[str, Any] = {
            self._source.image_column: self._decode(
                row_group.column(self._source.image_column)[row_index].as_py()
            ),
            self._source.label_column: int(
                row_group.column(self._source.label_column)[row_index].as_py()
            ),
        }
        if self._source.text_column is not None:
            row[self._source.text_column] = row_group.column(self._source.text_column)[
                row_index
            ].as_py()
        return row

    def column_values(self, column: str) -> list[int]:
        if column != self._source.label_column:
            raise KeyError(f"HF direct-Parquet does not expose column {column!r}")
        _, pq = _require_pyarrow()
        values: list[int] = []
        for path in self._paths:
            table = pq.read_table(path, columns=[column])
            values.extend(int(value) for value in table.column(column).to_pylist())
        return values


def load_validated_hf_parquet(
    source: HfParquetRecoverySource,
    *,
    parquet_root: str | Path,
    marker_path: str | Path,
) -> ParquetMapDataset:
    root, marker = _load_marker(source, parquet_root=parquet_root, marker_path=marker_path)
    return ParquetMapDataset(source, root=root, marker=marker)


def load_hf_parquet_recovery_source(source_config: Mapping[str, Any]) -> ParquetMapDataset:
    source = recovery_source_for_config(source_config)
    if source is None:
        raise ValueError("Source is not an exact HF direct-Parquet recovery contract")
    hub_root = os.environ.get(RECOVERY_HUB_ENV)
    marker_path = os.environ.get(source.marker_environment)
    if not hub_root or not marker_path:
        raise RuntimeError(
            f"HF direct-Parquet recovery requires {RECOVERY_HUB_ENV} and {source.marker_environment}"
        )
    return load_validated_hf_parquet(
        source,
        parquet_root=_hub_source_root(hub_root, source),
        marker_path=marker_path,
    )


def preflight_hf_parquet_recovery_environment(environment: Mapping[str, str] | None = None) -> None:
    """Validate configured markers without writing or invoking a HF builder."""

    values = os.environ if environment is None else environment
    hub_root = values.get(RECOVERY_HUB_ENV)
    if not hub_root:
        return
    for source in RECOVERY_SOURCES:
        marker_path = values.get(source.marker_environment)
        if not marker_path:
            raise RuntimeError(f"HF direct-Parquet recovery requires {source.marker_environment}")
        dataset = load_validated_hf_parquet(
            source,
            parquet_root=_hub_source_root(hub_root, source),
            marker_path=marker_path,
        )
        dataset.close()


__all__ = [
    "FOOD101_TRAIN_SOURCE",
    "FOOD101_VALIDATION_SOURCE",
    "HfParquetRecoverySource",
    "IMAGENET100_TEST_SOURCE",
    "IMAGENET100_TRAIN_SOURCE",
    "IMAGENET100_VALIDATION_SOURCE",
    "ParquetMapDataset",
    "RECOVERY_HUB_ENV",
    "RECOVERY_MARKER_ROOT_ENV",
    "RECOVERY_SCHEMA",
    "RECOVERY_SOURCES",
    "load_hf_parquet_recovery_source",
    "load_validated_hf_parquet",
    "preflight_hf_parquet_recovery_environment",
    "recovery_environment",
    "recovery_source_by_key",
    "recovery_source_for_config",
    "validate_hf_parquet",
]
