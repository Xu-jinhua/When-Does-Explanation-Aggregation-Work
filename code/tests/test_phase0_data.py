from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from xai_ensemble.data.composite import _selected_indices, load_composite_split
from xai_ensemble.data.manifest import (
    DatasetManifest,
    ManifestMetadata,
    ManifestRecord,
    build_hf_manifest,
    build_medmnist_manifest,
    hash_image_content,
    read_manifest,
    stable_sample_id,
    write_manifest,
)
from xai_ensemble.data.partitions import (
    make_ind_partitions,
    make_overlap_partitions,
    make_reference_partition,
    read_partition_plan,
    write_partition_plan,
)
from xai_ensemble.data.places365_recovery import (
    Places365RecoverySource,
    load_validated_places365_parquet,
    validate_places365_parquet,
)
from xai_ensemble.data.specs import (
    DERMAMNIST,
    FOOD101,
    IMAGENET100,
    PATHMNIST,
    PLACES365,
    HFDatasetSpec,
    get_dataset_spec,
)
from xai_ensemble.phase0.cli import _class_balance_parameters, _source_training_seed, main
from xai_ensemble.phase0.config import LoaderConfig
from xai_ensemble.phase0.dataset import (
    ManifestIndexedDataset,
    RowGroupLocalDistributedSampler,
    build_dataloader,
    load_hf_split,
)


def _recovery_source(*, file_count: int = 2, total_rows: int = 4) -> Places365RecoverySource:
    return Places365RecoverySource(
        key="test-places365",
        dataset_id="test/places365",
        revision="test-revision",
        source_split="train",
        label_column="labels",
        file_count=file_count,
        total_rows=total_rows,
        num_classes=2,
        root_environment="TEST_PLACES365_ROOT",
        marker_environment="TEST_PLACES365_MARKER",
    )


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (2, 2), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _write_recovery_parquet(
    root: Path,
    source: Places365RecoverySource,
    rows_by_file: list[list[tuple[int, tuple[int, int, int]]]],
    *,
    row_group_size: int = 1,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    names = ["zero", "one"]
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "image": {"_type": "Image"},
                        "labels": {"names": names, "_type": "ClassLabel"},
                    }
                }
            }
        ).encode("utf-8")
    }
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    schema = pa.schema([pa.field("image", image_type), pa.field("labels", pa.int64())])
    schema = schema.with_metadata(metadata)
    for name, rows in zip(source.file_names(), rows_by_file, strict=True):
        images = [{"bytes": _image_bytes(color), "path": None} for _label, color in rows]
        labels = [label for label, _color in rows]
        table = pa.Table.from_arrays(
            [pa.array(images, type=image_type), pa.array(labels, type=pa.int64())], schema=schema
        )
        pq.write_table(table, root / name, row_group_size=row_group_size)


def test_places365_direct_parquet_adapter_validates_and_reads_cross_shard_rows(
    tmp_path: Path,
) -> None:
    source = _recovery_source()
    root = tmp_path / "raw"
    marker = tmp_path / "markers" / "source.json"
    _write_recovery_parquet(
        root,
        source,
        [
            [(0, (255, 0, 0)), (1, (0, 255, 0))],
            [(0, (0, 0, 255)), (1, (255, 255, 0))],
        ],
    )

    validated = validate_places365_parquet(source, parquet_root=root, marker_path=marker)
    loaded = load_validated_places365_parquet(source, parquet_root=root, marker_path=marker)

    assert validated["total_rows"] == 4
    assert marker.is_file()
    assert len(loaded) == 4
    assert loaded.column_values("labels") == [0, 1, 0, 1]
    assert loaded[0]["labels"] == 0
    assert loaded[2]["labels"] == 0
    assert loaded[-1]["labels"] == 1
    assert loaded[2]["image"].mode == "RGB"
    assert loaded[2]["image"].getpixel((0, 0)) == (0, 0, 255)
    assert loaded.features["labels"].names == ("zero", "one")
    assert loaded.row_group_key(0) == (0, 0)
    assert loaded.row_group_key(2) == (1, 0)
    with pytest.raises(TypeError, match="integer index"):
        loaded["image"]


def test_places365_row_group_local_sampler_preserves_membership_and_locality(
    tmp_path: Path,
) -> None:
    source = _recovery_source()
    root = tmp_path / "raw"
    marker = tmp_path / "marker.json"
    _write_recovery_parquet(
        root,
        source,
        [
            [(0, (1, 2, 3)), (1, (4, 5, 6))],
            [(0, (7, 8, 9)), (1, (10, 11, 12))],
        ],
        row_group_size=2,
    )
    validate_places365_parquet(source, parquet_root=root, marker_path=marker)
    loaded = load_validated_places365_parquet(source, parquet_root=root, marker_path=marker)
    records = tuple(
        ManifestRecord(
            sample_id=hashlib.sha256(f"sample-{index}".encode()).hexdigest(),
            content_sha256=hashlib.sha256(f"content-{index}".encode()).hexdigest(),
            split="train",
            row_index=row_index,
            label=loaded[row_index]["labels"],
            label_name=str(loaded[row_index]["labels"]),
        )
        for index, row_index in enumerate((0, 2, 1, 3))
    )
    indexed = ManifestIndexedDataset(
        loaded,
        records,
        image_column="image",
        label_column="labels",
        transform=None,
    )
    loader = build_dataloader(
        indexed,
        LoaderConfig(batch_size=2, num_workers=0, pin_memory=False),
        training=True,
        seed=17,
    )

    assert isinstance(loader.sampler, RowGroupLocalDistributedSampler)
    loader.sampler.set_epoch(3)
    first = list(loader.sampler)
    loader.sampler.set_epoch(3)
    second = list(loader.sampler)
    assert first == second
    assert sorted(first) == list(range(len(indexed)))
    keys = [indexed.row_group_key(index) for index in first]
    assert len({left for left, right in zip(keys, keys[1:], strict=False) if left != right}) <= 1

    rank_zero = RowGroupLocalDistributedSampler(
        indexed,
        num_replicas=2,
        rank=0,
        seed=17,
        drop_last=False,
    )
    rank_one = RowGroupLocalDistributedSampler(
        indexed,
        num_replicas=2,
        rank=1,
        seed=17,
        drop_last=False,
    )
    rank_zero.set_epoch(3)
    rank_one.set_epoch(3)
    assert sorted([*rank_zero, *rank_one]) == list(range(len(indexed)))


def test_places365_direct_parquet_marker_fails_closed_when_a_file_changes(tmp_path: Path) -> None:
    source = _recovery_source()
    root = tmp_path / "raw"
    marker = tmp_path / "marker.json"
    _write_recovery_parquet(
        root,
        source,
        [[(0, (1, 2, 3)), (1, (4, 5, 6))], [(0, (7, 8, 9)), (1, (10, 11, 12))]],
    )
    validate_places365_parquet(source, parquet_root=root, marker_path=marker)
    changed = root / source.file_names()[1]
    changed.touch()

    with pytest.raises(RuntimeError, match="identity changed"):
        load_validated_places365_parquet(source, parquet_root=root, marker_path=marker)


def test_places365_direct_parquet_holdout_uses_label_column_without_image_decoding(
    tmp_path: Path,
) -> None:
    source = _recovery_source()
    root = tmp_path / "raw"
    marker = tmp_path / "marker.json"
    _write_recovery_parquet(
        root,
        source,
        [[(0, (1, 2, 3)), (1, (4, 5, 6))], [(0, (7, 8, 9)), (1, (10, 11, 12))]],
    )
    validate_places365_parquet(source, parquet_root=root, marker_path=marker)
    loaded = load_validated_places365_parquet(source, parquet_root=root, marker_path=marker)

    assert _selected_indices(
        loaded,
        source_revision=source.revision,
        source_split=source.source_split,
        source_label_column="labels",
        selection={
            "strategy": "stratified_hash_holdout_v1",
            "role": "holdout",
            "count_per_class": 1,
            "seed": "test",
        },
    ) == _selected_indices(
        [
            {"image": b"x", "labels": 0},
            {"image": b"x", "labels": 1},
            {"image": b"x", "labels": 0},
            {"image": b"x", "labels": 1},
        ],
        source_revision=source.revision,
        source_split=source.source_split,
        source_label_column="labels",
        selection={
            "strategy": "stratified_hash_holdout_v1",
            "role": "holdout",
            "count_per_class": 1,
            "seed": "test",
        },
    )


def _cap_fixture_source(per_class: int = 10, num_classes: int = 3) -> list[dict[str, object]]:
    return [
        {"image": b"x", "labels": label} for label in range(num_classes) for _ in range(per_class)
    ]


def _cap_fixture_selection(
    source: list[dict[str, object]], selection: dict[str, object]
) -> tuple[int, ...]:
    return _selected_indices(
        source,
        source_revision="rev",
        source_split="train",
        source_label_column="labels",
        selection=selection,
    )


def _cap_fixture_hash_key(seed: str, index: int, *, per_class: int = 10) -> str:
    label = index // per_class
    return hashlib.sha256(f"{seed}\0rev\0train\0{label}\0{index}".encode()).hexdigest()


def test_composite_complement_cap_per_class_selects_deterministic_hash_prefix() -> None:
    source = _cap_fixture_source()
    base: dict[str, object] = {
        "strategy": "stratified_hash_holdout_v1",
        "count_per_class": 2,
        "seed": "cap-test",
    }
    holdout = _cap_fixture_selection(source, {**base, "role": "holdout"})
    complement = _cap_fixture_selection(source, {**base, "role": "complement"})
    capped = _cap_fixture_selection(source, {**base, "role": "complement", "cap_per_class": 5})

    assert len(holdout) == 6
    assert len(complement) == 24
    assert len(capped) == 15
    assert Counter(index // 10 for index in capped) == {0: 5, 1: 5, 2: 5}
    assert not set(capped) & set(holdout)
    assert set(capped) <= set(complement)
    assert sorted(set(holdout) | set(complement)) == list(range(30))
    assert capped == _cap_fixture_selection(
        source, {**base, "role": "complement", "cap_per_class": 5}
    )
    for label in range(3):
        complement_class = sorted(
            (index for index in complement if index // 10 == label),
            key=lambda index: _cap_fixture_hash_key("cap-test", index),
        )
        capped_class = sorted(
            (index for index in capped if index // 10 == label),
            key=lambda index: _cap_fixture_hash_key("cap-test", index),
        )
        assert capped_class == complement_class[:5]


def test_composite_holdout_cap_per_class_truncates_the_holdout_prefix() -> None:
    source = _cap_fixture_source()
    base: dict[str, object] = {
        "strategy": "stratified_hash_holdout_v1",
        "count_per_class": 4,
        "seed": "cap-test",
    }
    holdout = _cap_fixture_selection(source, {**base, "role": "holdout"})
    capped = _cap_fixture_selection(source, {**base, "role": "holdout", "cap_per_class": 2})

    assert len(holdout) == 12
    assert len(capped) == 6
    assert Counter(index // 10 for index in capped) == {0: 2, 1: 2, 2: 2}
    assert set(capped) <= set(holdout)
    for label in range(3):
        holdout_class = sorted(
            (index for index in holdout if index // 10 == label),
            key=lambda index: _cap_fixture_hash_key("cap-test", index),
        )
        capped_class = sorted(
            (index for index in capped if index // 10 == label),
            key=lambda index: _cap_fixture_hash_key("cap-test", index),
        )
        assert capped_class == holdout_class[:2]


def test_composite_selection_rejects_invalid_cap_and_unknown_fields() -> None:
    source = _cap_fixture_source()
    base: dict[str, object] = {
        "strategy": "stratified_hash_holdout_v1",
        "role": "complement",
        "count_per_class": 2,
        "seed": "cap-test",
    }

    with pytest.raises(ValueError, match="cap_per_class must be positive"):
        _cap_fixture_selection(source, {**base, "cap_per_class": 0})
    with pytest.raises(ValueError, match="cap_per_class must be positive"):
        _cap_fixture_selection(source, {**base, "cap_per_class": -3})
    with pytest.raises(ValueError, match="Unsupported composite selection fields"):
        _cap_fixture_selection(source, {**base, "unknown_option": 1})
    # Without a cap the selection keeps the pre-cap behavior exactly.
    assert len(_cap_fixture_selection(source, base)) == 24


def test_places365_train_is_capped_at_200_per_class() -> None:
    assert PLACES365.expected_split_sizes == {
        "train": 73_000,
        "validation": 18_250,
        "test": 36_500,
    }
    assert "train-cap-200-per-class" in PLACES365.revision
    sources = PLACES365.provider_options["split_sources"]
    assert sources["train"]["selection"] == {
        "strategy": "stratified_hash_holdout_v1",
        "role": "complement",
        "count_per_class": 50,
        "seed": "places365-validation-v1",
        "cap_per_class": 200,
    }
    assert "cap_per_class" not in sources["validation"]["selection"]


def test_composite_places365_uses_direct_recovery_without_hf_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xai_ensemble.data import places365_recovery

    class LabelFeature:
        names = tuple(f"class-{index}" for index in range(365))

    class PlacesSource:
        # 251 rows per class: after the 50-per-class validation holdout the
        # complement still exceeds the 200-per-class train cap.
        _per_class = 251

        features = {"labels": LabelFeature()}

        def __len__(self) -> int:
            return 365 * self._per_class

        def __getitem__(self, index: int) -> dict[str, object]:
            return {"image": f"image-{index}", "labels": index // self._per_class}

        def column_values(self, column: str) -> list[int]:
            assert column == "labels"
            return [index // self._per_class for index in range(len(self))]

    direct_calls: list[dict[str, object]] = []

    def load_direct(source_config: dict[str, object]) -> PlacesSource:
        direct_calls.append(source_config)
        return PlacesSource()

    datasets = ModuleType("datasets")

    def unexpected_hf_builder(**_kwargs: object) -> None:
        raise AssertionError("Places365 recovery must not invoke datasets.load_dataset")

    datasets.load_dataset = unexpected_hf_builder
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(places365_recovery, "load_places365_recovery_source", load_direct)

    train = load_composite_split(PLACES365, "train")
    validation = load_composite_split(PLACES365, "validation")

    assert len(train) == 73_000
    assert len(validation) == 18_250
    assert train[0]["label"] in range(365)
    assert validation[0]["label"] in range(365)
    assert [call["label_column"] for call in direct_calls] == ["labels", "labels"]


def test_composite_places365_marker_failure_never_falls_back_to_hf_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xai_ensemble.data import places365_recovery

    builder_calls: list[dict[str, object]] = []
    datasets = ModuleType("datasets")

    def unexpected_hf_builder(**kwargs: object) -> None:
        builder_calls.append(dict(kwargs))
        raise AssertionError("Places365 marker failure must not invoke datasets.load_dataset")

    def missing_marker(_source_config: object) -> None:
        raise RuntimeError("Places365 direct-Parquet recovery marker is invalid: missing marker")

    datasets.load_dataset = unexpected_hf_builder
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(places365_recovery, "load_places365_recovery_source", missing_marker)

    with pytest.raises(RuntimeError, match="marker is invalid"):
        load_composite_split(PLACES365, "test")
    assert builder_calls == []


def test_other_composite_sources_continue_to_use_hf_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xai_ensemble.data import hf_parquet_recovery

    # This fixture deliberately exercises the generic composite path.  The
    # pinned Food101 source itself is covered by the direct-recovery tests.
    monkeypatch.setattr(hf_parquet_recovery, "recovery_source_for_config", lambda _config: None)
    calls: list[dict[str, object]] = []
    datasets = ModuleType("datasets")
    rows = [{"image": "food", "label": 3}]

    def fake_load_dataset(**kwargs: object) -> list[dict[str, object]]:
        calls.append(dict(kwargs))
        return rows

    datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    loaded = load_composite_split(FOOD101, "test", cache_dir="food-cache")

    assert len(loaded) == 1
    assert loaded[0] == {"image": "food", "label": 3}
    assert calls == [
        {
            "path": "ethz/food101",
            "revision": "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9",
            "split": "validation",
            "keep_in_memory": False,
            "cache_dir": str(Path("food-cache").resolve()),
        }
    ]


def _tiny_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        key="tiny",
        dataset_id="test/tiny",
        revision="0123456789abcdef",
        num_classes=2,
        image_column="image",
        label_column="label",
        text_column=None,
        splits=("train", "validation", "test"),
        expected_split_sizes={"train": 12, "validation": 6, "test": 4},
        expected_examples_per_class={"train": 6, "validation": 3, "test": 2},
        source_url="https://example.invalid/tiny",
        upstream_source="synthetic",
        license="test",
    )


def _tiny_manifest() -> tuple[HFDatasetSpec, DatasetManifest]:
    spec = _tiny_spec()
    records = []
    split_counts = {"train": 6, "validation": 3, "test": 2}
    for split, per_class in split_counts.items():
        row_index = 0
        for label in range(spec.num_classes):
            for offset in range(per_class):
                content = hashlib.sha256(f"{split}-{label}-{offset}".encode("ascii")).hexdigest()
                records.append(
                    ManifestRecord(
                        sample_id=stable_sample_id(
                            dataset_id=spec.dataset_id,
                            revision=spec.revision,
                            split=split,
                            row_index=row_index,
                            label=label,
                            content_sha256=content,
                        ),
                        content_sha256=content,
                        split=split,
                        row_index=row_index,
                        label=label,
                        label_name=str(label),
                    )
                )
                row_index += 1
    manifest = DatasetManifest(
        metadata=ManifestMetadata(
            dataset_key=spec.key,
            dataset_id=spec.dataset_id,
            revision=spec.revision,
            dataset_spec_fingerprint=spec.fingerprint,
            hash_mode="encoded_bytes",
        ),
        records=tuple(records),
    )
    return spec, manifest


def test_imagenet100_source_is_pinned_with_observed_class_counts() -> None:
    assert IMAGENET100.dataset_id == "ilee0022/ImageNet100"
    assert IMAGENET100.revision == "c55b2f2967c034db17be30f7d430e41c80fd4281"
    assert IMAGENET100.expected_split_sizes == {
        "train": 117_000,
        "validation": 13_000,
        "test": 5_000,
    }
    assert IMAGENET100.expected_examples_per_class == {
        "train": None,
        "validation": None,
        "test": None,
    }
    observed = IMAGENET100.provider_options["expected_class_counts"]
    assert set(observed) == {"train", "validation", "test"}
    assert all(len(counts) == 100 for counts in observed.values())
    assert sum(observed["train"].values()) == 117_000
    assert sum(observed["validation"].values()) == 13_000
    assert sum(observed["test"].values()) == 5_000
    assert min(observed["train"].values()) == 1_143
    assert max(observed["train"].values()) == 1_199
    assert min(observed["validation"].values()) == 101
    assert max(observed["validation"].values()) == 157
    assert set(observed["test"].values()) == {50}
    assert len(IMAGENET100.fingerprint) == 64
    assert get_dataset_spec("imagenet100") is IMAGENET100


def test_dermamnist_224_release_and_imbalanced_counts_are_pinned() -> None:
    assert get_dataset_spec("dermamnist") is DERMAMNIST
    assert DERMAMNIST.provider == "medmnist"
    assert DERMAMNIST.revision == (
        "medmnist-3.0.2-zenodo-10519652-dermamnist-224-8974907d8e169bef5f5b96bc506ae45d"
    )
    assert DERMAMNIST.provider_options["version"] == "3.0.2"
    assert DERMAMNIST.provider_options["size"] == 224
    assert DERMAMNIST.provider_options["zenodo_record"] == "10519652"
    assert DERMAMNIST.provider_options["filename"] == "dermamnist_224.npz"
    assert DERMAMNIST.provider_options["md5"] == "8974907d8e169bef5f5b96bc506ae45d"
    assert DERMAMNIST.provider_options["expected_shape"]["test"] == [2005, 224, 224, 3]
    assert DERMAMNIST.expected_split_sizes["test"] == 2_005
    test_counts = DERMAMNIST.provider_options["expected_class_counts"]["test"]
    assert sum(test_counts.values()) == 2_005


def test_pathmnist_224_release_is_pinned() -> None:
    assert get_dataset_spec("pathmnist") is PATHMNIST
    assert PATHMNIST.provider == "medmnist"
    assert PATHMNIST.num_classes == 9
    assert PATHMNIST.revision == (
        "medmnist-3.0.2-zenodo-10519652-pathmnist-224-2c51a510bcdc9cf8ddb2af93af1eadec"
    )
    assert PATHMNIST.expected_split_sizes == {
        "train": 89_996,
        "validation": 10_004,
        "test": 7_180,
    }
    assert PATHMNIST.provider_options["size"] == 224
    assert PATHMNIST.provider_options["filename"] == "pathmnist_224.npz"
    assert PATHMNIST.provider_options["md5"] == "2c51a510bcdc9cf8ddb2af93af1eadec"
    assert PATHMNIST.provider_options["expected_shape"]["test"] == [7_180, 224, 224, 3]


def test_dataset_loaders_resolve_relative_cache_directories(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _tiny_spec()
    calls = []
    rows = [
        {spec.image_column: f"image-{index}".encode(), spec.label_column: index % 2}
        for index in range(spec.expected_split_sizes["train"])
    ]
    datasets = ModuleType("datasets")

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return rows

    datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.chdir(tmp_path)

    loaded = load_hf_split(spec, "train", cache_dir="cache/datasets/tiny")
    manifest = build_hf_manifest(
        spec,
        splits=("train",),
        cache_dir="cache/datasets/tiny",
        hash_mode="encoded_bytes",
        require_expected_counts=False,
    )

    expected = str((tmp_path / "cache/datasets/tiny").resolve())
    assert loaded == rows
    assert len(manifest.records) == len(rows)
    assert [call["cache_dir"] for call in calls] == [expected, expected]


def test_medmnist_loaders_create_an_explicit_cache_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = HFDatasetSpec(
        key="tiny-medmnist",
        dataset_id="medmnist/tiny",
        revision="medmnist-test",
        num_classes=2,
        image_column="image",
        label_column="label",
        text_column=None,
        splits=("train",),
        expected_split_sizes={"train": 2},
        expected_examples_per_class={"train": 1},
        source_url="https://example.invalid/tiny-medmnist",
        upstream_source="synthetic",
        license="test",
        provider="medmnist",
        provider_options={
            "version": "3.0.2",
            "data_flag": "tiny",
            "as_rgb": True,
            "size": 224,
        },
    )
    roots = []

    class TinyMNIST:
        def __init__(self, *, root, split, download, as_rgb, size):
            assert Path(root).is_dir()
            assert split == "train"
            assert download is True
            assert as_rgb is True
            assert size == 224
            roots.append(root)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            assert index in {0, 1}
            return Image.new("RGB", (1, 1)), index

    medmnist = ModuleType("medmnist")
    medmnist.INFO = {"tiny": {"python_class": "TinyMNIST", "label": {"0": "zero", "1": "one"}}}
    medmnist.TinyMNIST = TinyMNIST
    monkeypatch.setitem(sys.modules, "medmnist", medmnist)
    monkeypatch.setattr("importlib.metadata.version", lambda _package: "3.0.2")

    manifest_root = tmp_path / "manifest-cache"
    runtime_root = tmp_path / "runtime-cache"
    assert not manifest_root.exists()
    assert not runtime_root.exists()

    manifest = build_medmnist_manifest(spec, root=manifest_root)
    loaded = load_hf_split(spec, "train", cache_dir=runtime_root)

    assert len(manifest.records) == 2
    assert len(loaded) == 2
    assert roots == [str(manifest_root.resolve()), str(runtime_root.resolve())]


def test_manifest_identity_validation_and_round_trip(tmp_path) -> None:
    spec, manifest = _tiny_manifest()
    audit = manifest.validate(spec, reject_cross_split_duplicates=True)
    assert audit.split_sizes["train"] == 12
    assert audit.class_counts["validation"] == {0: 3, 1: 3}
    assert audit.duplicate_sample_ids == ()

    path = write_manifest(manifest, tmp_path / "manifest.jsonl")
    restored = read_manifest(path)
    assert restored == manifest
    assert restored.fingerprint == manifest.fingerprint


def test_manifest_detects_content_leakage_across_splits() -> None:
    spec, manifest = _tiny_manifest()
    train = manifest.records_for_split("train")[0]
    test = manifest.records_for_split("test")[0]
    records = tuple(
        replace(
            record,
            content_sha256=train.content_sha256,
            sample_id=stable_sample_id(
                dataset_id=manifest.metadata.dataset_id,
                revision=manifest.metadata.revision,
                split=record.split,
                row_index=record.row_index,
                label=record.label,
                content_sha256=train.content_sha256,
            ),
        )
        if record.sample_id == test.sample_id
        else record
        for record in manifest.records
    )
    leaked = replace(manifest, records=records)
    assert leaked.audit().has_cross_split_content_duplicates
    with pytest.raises(ValueError, match="multiple splits"):
        leaked.validate(spec, reject_cross_split_duplicates=True)


def test_encoded_content_hash_is_exact() -> None:
    payload = b"not-an-image-but-valid-for-encoded-mode"
    digest, length = hash_image_content(payload, mode="encoded_bytes")
    assert digest == hashlib.sha256(payload).hexdigest()
    assert length == len(payload)


def test_ind_overlap_and_reference_partition_contracts(tmp_path) -> None:
    _, manifest = _tiny_manifest()
    ind = make_ind_partitions(manifest, split="train", num_sources=3, seed=17)
    assert ind.strategy == "disjoint_stratified_full"
    assert [source.class_counts for source in ind.sources] == [{0: 2, 1: 2}] * 3
    assert len(set().union(*(set(source.sample_ids) for source in ind.sources))) == 12
    for row_index, row in enumerate(ind.overlap_matrix()):
        for column_index, overlap in enumerate(row):
            assert overlap == (4 if row_index == column_index else 0)

    shared = make_overlap_partitions(
        manifest,
        split="train",
        num_sources=3,
        seed=17,
        mode="shared",
    )
    assert shared.strategy == "shared_stratified"
    assert len({source.sample_ids for source in shared.sources}) == 1
    assert shared.overlap_matrix() == ((4, 4, 4), (4, 4, 4), (4, 4, 4))

    reference = make_reference_partition(manifest, split="train")
    assert reference.sources[0].size == 12
    assert reference.strategy == "full_split"

    path = write_partition_plan(ind, tmp_path / "ind.json")
    restored = read_partition_plan(path)
    restored.validate(manifest)
    assert restored == ind
    assert restored.digest == ind.digest


def test_ind_distributes_nondivisible_classes_without_dropping_samples() -> None:
    _, manifest = _tiny_manifest()
    plan = make_ind_partitions(manifest, split="train", num_sources=4, seed=1)
    assert sum(source.size for source in plan.sources) == 12
    assert (
        max(source.size for source in plan.sources) - min(source.size for source in plan.sources)
        <= 1
    )
    assert len(set().union(*(set(source.sample_ids) for source in plan.sources))) == 12


def test_overlap_can_use_exact_drop_remainder_ind_quotas() -> None:
    _, manifest = _tiny_manifest()
    ind = make_ind_partitions(
        manifest,
        split="train",
        num_sources=4,
        seed=9,
        require_full_coverage=False,
    )
    quotas = dict(ind.sources[0].class_counts)
    assert all(dict(source.class_counts) == quotas for source in ind.sources)
    overlap = make_overlap_partitions(
        manifest,
        split="train",
        num_sources=4,
        seed=9,
        class_quotas=quotas,
        mode="shared",
        matched_ind_digest=ind.digest,
    )
    assert overlap.matched_ind_digest == ind.digest
    assert all(dict(source.class_counts) == quotas for source in overlap.sources)
    assert [source.size for source in overlap.sources] == [source.size for source in ind.sources]


def test_phase0_cli_matches_drop_remainder_ind_with_zero_class_quota(tmp_path) -> None:
    records = []
    row_index = 0
    for label, count in ((0, 4), (1, 4), (2, 1)):
        for offset in range(count):
            content = hashlib.sha256(f"sparse-{label}-{offset}".encode("ascii")).hexdigest()
            records.append(
                ManifestRecord(
                    sample_id=stable_sample_id(
                        dataset_id="test/sparse",
                        revision="revision-1",
                        split="train",
                        row_index=row_index,
                        label=label,
                        content_sha256=content,
                    ),
                    content_sha256=content,
                    split="train",
                    row_index=row_index,
                    label=label,
                    label_name=str(label),
                )
            )
            row_index += 1
    manifest = DatasetManifest(
        metadata=ManifestMetadata(
            dataset_key="sparse",
            dataset_id="test/sparse",
            revision="revision-1",
            dataset_spec_fingerprint="s" * 64,
            hash_mode="encoded_bytes",
        ),
        records=tuple(records),
    )
    manifest_path = write_manifest(manifest, tmp_path / "manifest.jsonl")
    ind_path = tmp_path / "ind.json"
    overlap_path = tmp_path / "overlap.json"

    assert (
        main(
            [
                "build-partitions",
                "--manifest",
                str(manifest_path),
                "--output",
                str(ind_path),
                "--kind",
                "ind",
                "--split",
                "train",
                "--num-sources",
                "2",
                "--seed",
                "13",
                "--allow-remainder",
            ]
        )
        == 0
    )
    ind = read_partition_plan(ind_path)
    assert all(dict(source.class_counts) == {0: 2, 1: 2} for source in ind.sources)

    assert (
        main(
            [
                "build-partitions",
                "--manifest",
                str(manifest_path),
                "--output",
                str(overlap_path),
                "--kind",
                "overlap",
                "--split",
                "train",
                "--num-sources",
                "2",
                "--seed",
                "13",
                "--overlap-mode",
                "shared",
                "--matched-ind-partition",
                str(ind_path),
            ]
        )
        == 0
    )
    overlap = read_partition_plan(overlap_path)
    assert overlap.matched_ind_digest == ind.digest
    assert all(dict(source.class_counts) == {0: 2, 1: 2} for source in overlap.sources)
    assert [source.size for source in overlap.sources] == [source.size for source in ind.sources]


def test_source_training_seed_is_matched_by_source_not_partition_kind() -> None:
    source_0 = _source_training_seed(
        7, dataset="imagenet100", model="resnet18", source_id="source-00"
    )
    repeated = _source_training_seed(
        7, dataset="imagenet100", model="resnet18", source_id="source-00"
    )
    source_1 = _source_training_seed(
        7, dataset="imagenet100", model="resnet18", source_id="source-01"
    )
    assert source_0 == repeated
    assert source_0 != source_1


def test_class_balance_parameters_are_train_only_inverse_frequency_weights() -> None:
    records = [
        replace(_tiny_manifest()[1].records_for_split("train")[0], label=0),
        replace(_tiny_manifest()[1].records_for_split("train")[1], label=0),
        replace(_tiny_manifest()[1].records_for_split("train")[2], label=1),
        replace(_tiny_manifest()[1].records_for_split("train")[3], label=2),
        replace(_tiny_manifest()[1].records_for_split("train")[4], label=2),
        replace(_tiny_manifest()[1].records_for_split("train")[5], label=2),
    ]
    counts, weights = _class_balance_parameters(records, num_classes=3)
    assert counts == [2, 1, 3]
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert weights[1] > weights[0] > weights[2]
    with pytest.raises(ValueError, match="every training class"):
        _class_balance_parameters(records[:2], num_classes=3)
