from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, read_json
from xai_ensemble.data.hf_parquet_recovery import (
    FOOD101_TRAIN_SOURCE,
    RECOVERY_HUB_ENV,
    HfParquetRecoverySource,
    load_validated_hf_parquet,
    validate_hf_parquet,
)


def _fixture_source(tmp_path: Path) -> tuple[HfParquetRecoverySource, Path, Path]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image

    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            pa.field("image", image_type),
            pa.field("label", pa.int64()),
            pa.field("text", pa.string()),
        ]
    )
    label_names = ("zero", "one")
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "image": {"_type": "Image"},
                        "label": {"names": list(label_names), "_type": "ClassLabel"},
                        "text": {"dtype": "string", "_type": "Value"},
                    }
                }
            }
        ).encode()
    }
    schema = schema.with_metadata(metadata)
    schema_digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
    file_records: list[dict[str, object]] = []
    for shard, labels in enumerate(((0, 1), (1, 0))):
        rows = []
        for row, label in enumerate(labels):
            image_path = tmp_path / f"image-{shard}-{row}.png"
            image = Image.new("RGB", (2, 2), (label * 100, 20, 30))
            image.save(image_path, format="PNG")
            rows.append(
                {
                    "bytes": image_path.read_bytes(),
                    "path": None,
                    "text": f"sample-{shard}-{row}",
                    "label": label,
                }
            )
        table = pa.Table.from_pydict(
            {
                "image": pa.array(
                    [{"bytes": row["bytes"], "path": row["path"]} for row in rows],
                    type=image_type,
                ),
                "label": pa.array([row["label"] for row in rows], type=pa.int64()),
                "text": pa.array([row["text"] for row in rows], type=pa.string()),
            },
            schema=schema,
        )
        temporary = source_root / f"temporary-{shard}.parquet"
        pq.write_table(table, temporary, row_group_size=1)
        digest = file_sha256(temporary)
        blob = source_root / digest
        temporary.rename(blob)
        link = source_root / f"train-{shard:05d}-of-00002.parquet"
        link.symlink_to(blob.name)
        stat = blob.stat()
        file_records.append(
            {
                "name": link.name,
                "sha256": digest,
                "size": stat.st_size,
                "row_count": 2,
            }
        )

    source = replace(
        FOOD101_TRAIN_SOURCE,
        key="fixture-hf-train",
        text_column="text",
        file_count=2,
        total_rows=4,
        num_classes=2,
        expected_file_manifest_sha256=object_sha256(file_records),
        expected_lfs_manifest_sha256=object_sha256(
            [{key: row[key] for key in ("name", "sha256", "size")} for row in file_records]
        ),
        expected_total_file_bytes=sum(int(row["size"]) for row in file_records),
        expected_arrow_schema_sha256=schema_digest,
        expected_label_names_sha256=object_sha256(list(label_names)),
        expected_label_counts=(2, 2),
        marker_environment="XAI_HF_FIXTURE_MARKER",
    )
    marker = tmp_path / "marker.json"
    return source, source_root, marker


def _rewrite_marker(path: Path, **updates: object) -> None:
    value = dict(read_json(path))
    value.update(updates)
    value["marker_digest"] = object_sha256(
        {key: item for key, item in value.items() if key != "marker_digest"}
    )
    atomic_write_json(path, value)


def test_validated_hf_parquet_reads_rows_and_text(tmp_path: Path) -> None:
    source, root, marker = _fixture_source(tmp_path)
    value = validate_hf_parquet(source, parquet_root=root, marker_path=marker)

    dataset = load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)
    try:
        assert value["file_count"] == 2
        assert len(dataset) == 4
        assert [dataset[index]["label"] for index in range(4)] == [0, 1, 1, 0]
        assert dataset[0]["text"] == "sample-0-0"
        assert dataset[-1]["image"].size == (2, 2)
        assert dataset.features["label"].names == ("zero", "one")
        assert dataset.row_group_key(2) == (1, 0)
    finally:
        dataset.close()


def test_hf_marker_rejects_missing_or_extra_shards(tmp_path: Path) -> None:
    source, root, marker = _fixture_source(tmp_path)
    validate_hf_parquet(source, parquet_root=root, marker_path=marker)

    (root / "train-00001-of-00002.parquet").unlink()
    with pytest.raises(RuntimeError, match="raw files do not match"):
        load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)

    source, root, marker = _fixture_source(tmp_path / "extra")
    validate_hf_parquet(source, parquet_root=root, marker_path=marker)
    (root / "unexpected.parquet").symlink_to(next(root.glob("[0-9a-f]*")).name)
    with pytest.raises(RuntimeError, match="raw files do not match"):
        load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("arrow_schema_sha256", "0" * 64, "Arrow schema digest"),
        ("class_counts", [1, 3], "class counts"),
        ("label_names", ["wrong", "labels"], "label-name digest"),
    ),
)
def test_hf_marker_rejects_fixed_schema_and_label_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source, root, marker = _fixture_source(tmp_path)
    validate_hf_parquet(source, parquet_root=root, marker_path=marker)
    _rewrite_marker(marker, **{field: value})

    with pytest.raises(RuntimeError, match=message):
        load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)


def test_hf_marker_rejects_digest_and_row_count_changes(tmp_path: Path) -> None:
    source, root, marker = _fixture_source(tmp_path)
    validate_hf_parquet(source, parquet_root=root, marker_path=marker)
    value = dict(read_json(marker))
    value["marker_digest"] = "f" * 64
    atomic_write_json(marker, value)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)

    source, root, marker = _fixture_source(tmp_path / "row-count")
    validate_hf_parquet(source, parquet_root=root, marker_path=marker)
    files = list(read_json(marker)["files"])
    files[0]["row_count"] = 1
    _rewrite_marker(marker, files=files)
    with pytest.raises(RuntimeError, match="file manifest digest"):
        load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)


def test_hf_marker_rejects_content_addressed_blob_replacement(tmp_path: Path) -> None:
    source, root, marker = _fixture_source(tmp_path)
    validate_hf_parquet(source, parquet_root=root, marker_path=marker)
    first = root / "train-00000-of-00002.parquet"
    replacement = root / ("a" * 64)
    replacement.write_bytes(first.resolve().read_bytes() + b"changed")
    first.unlink()
    first.symlink_to(replacement.name)

    with pytest.raises(RuntimeError, match="raw file identity changed|stored blob hash is invalid"):
        load_validated_hf_parquet(source, parquet_root=root, marker_path=marker)


def test_matching_source_requires_recovery_environment_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xai_ensemble.data.hf_parquet_recovery import recovery_source_for_config
    from xai_ensemble.data.specs import IMAGENET100
    from xai_ensemble.phase0.dataset import load_hf_split

    config = {
        "dataset_id": FOOD101_TRAIN_SOURCE.dataset_id,
        "revision": FOOD101_TRAIN_SOURCE.revision,
        "split": FOOD101_TRAIN_SOURCE.source_split,
        "image_column": FOOD101_TRAIN_SOURCE.image_column,
        "label_column": FOOD101_TRAIN_SOURCE.label_column,
        "text_column": FOOD101_TRAIN_SOURCE.text_column,
    }
    assert recovery_source_for_config(config) is FOOD101_TRAIN_SOURCE
    monkeypatch.delenv(RECOVERY_HUB_ENV, raising=False)
    monkeypatch.delenv(FOOD101_TRAIN_SOURCE.marker_environment, raising=False)
    spec = replace(
        IMAGENET100,
        splits=("train",),
        expected_split_sizes={"train": 117_000},
        expected_examples_per_class={"train": None},
    )
    with pytest.raises(RuntimeError, match="direct-Parquet recovery requires"):
        load_hf_split(spec, "train")
