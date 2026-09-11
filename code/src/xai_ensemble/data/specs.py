"""Pinned dataset specifications used by the experiment pipeline.

The revision is part of the dataset identity.  Callers must not silently fall
back to the moving default branch of a Hugging Face dataset repository.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class HFDatasetSpec:
    """An immutable Hugging Face dataset contract.

    ``expected_examples_per_class`` is intentionally explicit.  It lets the
    initial Phase 0 scan fail before expensive training when a remote dataset
    revision or label mapping is not the one used by the experiment protocol.
    """

    key: str
    dataset_id: str
    revision: str
    num_classes: int
    image_column: str
    label_column: str
    text_column: str | None
    splits: tuple[str, ...]
    expected_split_sizes: Mapping[str, int]
    expected_examples_per_class: Mapping[str, int | None]
    source_url: str
    upstream_source: str
    license: str
    provider: str = "huggingface"
    provider_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.dataset_id:
            raise ValueError("Dataset key and dataset_id must be non-empty")
        if not self.revision or (self.provider == "huggingface" and len(self.revision) < 12):
            raise ValueError("Hugging Face datasets require a pinned commit revision")
        if self.provider not in {"huggingface", "composite_huggingface", "medmnist", "imagenet"}:
            raise ValueError(f"Unsupported dataset provider: {self.provider}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if not self.splits:
            raise ValueError("At least one split is required")
        if set(self.splits) != set(self.expected_split_sizes):
            raise ValueError("Every split must have an expected size")
        if set(self.splits) != set(self.expected_examples_per_class):
            raise ValueError("Every split must have an expected per-class size")
        for split in self.splits:
            per_class = self.expected_examples_per_class[split]
            expected = None if per_class is None else self.num_classes * per_class
            if expected is not None and self.expected_split_sizes[split] != expected:
                raise ValueError(
                    f"Invalid expected size for {split}: "
                    f"{self.expected_split_sizes[split]} != {expected}"
                )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic digest of the complete source contract."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_split_size(self, split: str, size: int) -> None:
        if split not in self.expected_split_sizes:
            raise KeyError(f"Unknown split {split!r}; expected one of {self.splits}")
        expected = self.expected_split_sizes[split]
        if size != expected:
            raise ValueError(
                f"Dataset {self.dataset_id}@{self.revision} split {split!r} "
                f"contains {size} rows; expected {expected}"
            )


IMAGENET1K = HFDatasetSpec(
    key="imagenet1k_ilsvrc2012",
    dataset_id="ILSVRC2012/ImageNet-1K",
    revision="ilsvrc2012-train-val-hash-holdout-50-per-class-v1",
    num_classes=1000,
    image_column="image",
    label_column="label",
    text_column="filename",
    splits=("train", "validation", "test"),
    expected_split_sizes={"train": 1_231_167, "validation": 50_000, "test": 50_000},
    expected_examples_per_class={"train": None, "validation": 50, "test": 50},
    source_url="https://image-net.org/challenges/LSVRC/2012/",
    upstream_source="Official ILSVRC2012 training and validation archives",
    license="ImageNet terms of access",
    provider="imagenet",
    provider_options={
        "root_environment": "XAI_IMAGENET1K_ROOT",
        "official_train_size": 1_281_167,
        "official_validation_size": 50_000,
        "holdout_per_class": 50,
        "holdout_seed": "imagenet1k-validation-v1",
        "label_order": "sorted_official_wnids",
        "logical_split_policy": "official_validation_as_test_deterministic_train_holdout",
    },
)


# Historical dataset identity; never alias it to the 1000-class experiment.
IMAGENET100 = HFDatasetSpec(
    key="imagenet100_ilee0022",
    dataset_id="ilee0022/ImageNet100",
    revision="c55b2f2967c034db17be30f7d430e41c80fd4281",
    num_classes=100,
    image_column="image",
    label_column="label",
    text_column="text",
    splits=("train", "validation", "test"),
    expected_split_sizes={"train": 117_000, "validation": 13_000, "test": 5_000},
    # The pinned HF revision keeps all 100 classes but is not exactly balanced
    # in train/validation.  Bind the observed map so a source change fails
    # closed without imposing a false equal-quota contract.
    expected_examples_per_class={"train": None, "validation": None, "test": None},
    source_url="https://huggingface.co/datasets/ilee0022/ImageNet100",
    upstream_source="https://www.kaggle.com/datasets/ambityga/imagenet100",
    license="unknown",
    provider_options={
        "expected_class_counts": {
            "train": {
                0: 1175, 1: 1170, 2: 1178, 3: 1180, 4: 1179, 5: 1170, 6: 1166,
                7: 1187, 8: 1181, 9: 1166, 10: 1183, 11: 1170, 12: 1180, 13: 1179,
                14: 1158, 15: 1171, 16: 1171, 17: 1159, 18: 1156, 19: 1177,
                20: 1177, 21: 1175, 22: 1167, 23: 1173, 24: 1166, 25: 1146,
                26: 1152, 27: 1165, 28: 1183, 29: 1184, 30: 1174, 31: 1159,
                32: 1162, 33: 1179, 34: 1164, 35: 1176, 36: 1156, 37: 1159,
                38: 1164, 39: 1190, 40: 1176, 41: 1165, 42: 1169, 43: 1199,
                44: 1167, 45: 1162, 46: 1164, 47: 1174, 48: 1162, 49: 1173,
                50: 1171, 51: 1158, 52: 1173, 53: 1179, 54: 1158, 55: 1171,
                56: 1176, 57: 1158, 58: 1186, 59: 1170, 60: 1168, 61: 1173,
                62: 1184, 63: 1172, 64: 1174, 65: 1157, 66: 1160, 67: 1159,
                68: 1176, 69: 1172, 70: 1170, 71: 1180, 72: 1171, 73: 1165,
                74: 1177, 75: 1143, 76: 1179, 77: 1180, 78: 1174, 79: 1186,
                80: 1171, 81: 1172, 82: 1181, 83: 1145, 84: 1169, 85: 1193,
                86: 1162, 87: 1163, 88: 1182, 89: 1161, 90: 1176, 91: 1160,
                92: 1153, 93: 1163, 94: 1171, 95: 1187, 96: 1146, 97: 1153,
                98: 1173, 99: 1161,
            },
            "validation": {
                0: 125, 1: 130, 2: 122, 3: 120, 4: 121, 5: 130, 6: 134, 7: 113,
                8: 119, 9: 134, 10: 117, 11: 130, 12: 120, 13: 121, 14: 142,
                15: 129, 16: 129, 17: 141, 18: 144, 19: 123, 20: 123, 21: 125,
                22: 133, 23: 127, 24: 134, 25: 154, 26: 148, 27: 135, 28: 117,
                29: 116, 30: 126, 31: 141, 32: 138, 33: 121, 34: 136, 35: 124,
                36: 144, 37: 141, 38: 136, 39: 110, 40: 124, 41: 135, 42: 131,
                43: 101, 44: 133, 45: 138, 46: 136, 47: 126, 48: 138, 49: 127,
                50: 129, 51: 142, 52: 127, 53: 121, 54: 142, 55: 129, 56: 124,
                57: 142, 58: 114, 59: 130, 60: 132, 61: 127, 62: 116, 63: 128,
                64: 126, 65: 143, 66: 140, 67: 141, 68: 124, 69: 128, 70: 130,
                71: 120, 72: 129, 73: 135, 74: 123, 75: 157, 76: 121, 77: 120,
                78: 126, 79: 114, 80: 129, 81: 128, 82: 119, 83: 155, 84: 131,
                85: 107, 86: 138, 87: 137, 88: 118, 89: 139, 90: 124, 91: 140,
                92: 147, 93: 137, 94: 129, 95: 113, 96: 154, 97: 147, 98: 127,
                99: 139,
            },
            "test": {label: 50 for label in range(100)},
        },
    },
)


DERMAMNIST = HFDatasetSpec(
    key="dermamnist_medmnist_3_0_2_224",
    dataset_id="medmnist/dermamnist",
    revision="medmnist-3.0.2-zenodo-10519652-dermamnist-224-8974907d8e169bef5f5b96bc506ae45d",
    num_classes=7,
    image_column="image",
    label_column="label",
    text_column=None,
    splits=("train", "validation", "test"),
    expected_split_sizes={"train": 7_007, "validation": 1_003, "test": 2_005},
    # DermaMNIST is intentionally imbalanced; exact counts are pinned below.
    expected_examples_per_class={"train": None, "validation": None, "test": None},
    source_url="https://medmnist.com/",
    upstream_source="HAM10000",
    license="CC BY-NC 4.0",
    provider="medmnist",
    provider_options={
        "data_flag": "dermamnist",
        "version": "3.0.2",
        "size": 224,
        "as_rgb": True,
        "zenodo_record": "10519652",
        "filename": "dermamnist_224.npz",
        "source_url": "https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1",
        "md5": "8974907d8e169bef5f5b96bc506ae45d",
        "expected_shape": {
            "train": [7007, 224, 224, 3],
            "validation": [1003, 224, 224, 3],
            "test": [2005, 224, 224, 3],
        },
        "expected_class_counts": {
            "train": {0: 228, 1: 359, 2: 769, 3: 80, 4: 779, 5: 4_693, 6: 99},
            "validation": {0: 33, 1: 52, 2: 110, 3: 12, 4: 111, 5: 671, 6: 14},
            "test": {0: 66, 1: 103, 2: 220, 3: 23, 4: 223, 5: 1_341, 6: 29},
        },
    },
)


PATHMNIST = HFDatasetSpec(
    key="pathmnist_medmnist_3_0_2_224",
    dataset_id="medmnist/pathmnist",
    revision="medmnist-3.0.2-zenodo-10519652-pathmnist-224-2c51a510bcdc9cf8ddb2af93af1eadec",
    num_classes=9,
    image_column="image",
    label_column="label",
    text_column=None,
    splits=("train", "validation", "test"),
    expected_split_sizes={"train": 89_996, "validation": 10_004, "test": 7_180},
    expected_examples_per_class={"train": None, "validation": None, "test": None},
    source_url="https://medmnist.com/",
    upstream_source="NCT-CRC-HE-100K and CRC-VAL-HE-7K",
    license="CC BY 4.0",
    provider="medmnist",
    provider_options={
        "data_flag": "pathmnist",
        "version": "3.0.2",
        "size": 224,
        "as_rgb": True,
        "zenodo_record": "10519652",
        "filename": "pathmnist_224.npz",
        "source_url": "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1",
        "md5": "2c51a510bcdc9cf8ddb2af93af1eadec",
        "expected_shape": {
            "train": [89_996, 224, 224, 3],
            "validation": [10_004, 224, 224, 3],
            "test": [7_180, 224, 224, 3],
        },
    },
)


def _medmnist_224(
    *,
    key: str,
    flag: str,
    num_classes: int,
    split_sizes: Mapping[str, int],
    md5: str,
    upstream_source: str,
    license: str = "CC BY 4.0",
) -> HFDatasetSpec:
    """Build one pinned MedMNIST+ 224 contract.

    The official 224 files are used directly.  In particular, they are not
    generated by resizing the already-downsampled 28-pixel files.  The
    MedMNIST adapter still exposes grayscale sources as RGB when ``as_rgb``
    is enabled, which is the model-input contract used by this experiment.
    """

    return HFDatasetSpec(
        key=key,
        dataset_id=f"medmnist/{flag}",
        revision=f"medmnist-3.0.2-zenodo-10519652-{flag}-224-{md5}",
        num_classes=num_classes,
        image_column="image",
        label_column="label",
        text_column=None,
        splits=("train", "validation", "test"),
        expected_split_sizes=dict(split_sizes),
        expected_examples_per_class={split: None for split in split_sizes},
        source_url="https://medmnist.com/",
        upstream_source=upstream_source,
        license=license,
        provider="medmnist",
        provider_options={
            "data_flag": flag,
            "version": "3.0.2",
            "size": 224,
            "as_rgb": True,
            "zenodo_record": "10519652",
            "filename": f"{flag}_224.npz",
            "source_url": (
                f"https://zenodo.org/records/10519652/files/{flag}_224.npz?download=1"
            ),
            "md5": md5,
            "official_resolution": "224x224",
            "grayscale_channel_policy": "replicate_to_rgb",
        },
    )


OCTMNIST = _medmnist_224(
    key="octmnist_medmnist_3_0_2_224",
    flag="octmnist",
    num_classes=4,
    split_sizes={"train": 97_477, "validation": 10_832, "test": 1_000},
    md5="abc493b6d529d5de7569faaef2773ba3",
    upstream_source="OCT retinal disease images",
)
PNEUMONIAMNIST = _medmnist_224(
    key="pneumoniamnist_medmnist_3_0_2_224",
    flag="pneumoniamnist",
    num_classes=2,
    split_sizes={"train": 4_708, "validation": 524, "test": 624},
    md5="d6a3c71de1b945ea11211b03746c1fe1",
    upstream_source="pediatric chest X-ray images",
)
RETINAMNIST = _medmnist_224(
    key="retinamnist_medmnist_3_0_2_224",
    flag="retinamnist",
    num_classes=5,
    split_sizes={"train": 1_080, "validation": 120, "test": 400},
    md5="eae7e3b6f3fcbda4ae613ebdcbe35348",
    upstream_source="DeepDRiD retinal fundus images",
)
BREASTMNIST = _medmnist_224(
    key="breastmnist_medmnist_3_0_2_224",
    flag="breastmnist",
    num_classes=2,
    split_sizes={"train": 546, "validation": 78, "test": 156},
    md5="b56378a6eefa9fed602bb16d192d4c8b",
    upstream_source="breast ultrasound images",
)
BLOODMNIST = _medmnist_224(
    key="bloodmnist_medmnist_3_0_2_224",
    flag="bloodmnist",
    num_classes=8,
    split_sizes={"train": 11_959, "validation": 1_712, "test": 3_421},
    md5="b718ff6835fcbdb22ba9eacccd7b2601",
    upstream_source="single-cell blood images",
)
TISSUEMNIST = _medmnist_224(
    key="tissuemnist_medmnist_3_0_2_224",
    flag="tissuemnist",
    num_classes=8,
    split_sizes={"train": 165_466, "validation": 23_640, "test": 47_280},
    md5="b077128c4a949f0a4eb01517f9037b9c",
    upstream_source="BBBC051 kidney cortex cells",
)
ORGANAMNIST = _medmnist_224(
    key="organamnist_medmnist_3_0_2_224",
    flag="organamnist",
    num_classes=11,
    split_sizes={"train": 34_561, "validation": 6_491, "test": 17_778},
    md5="50747347e05c87dd3aaf92c49f9f3170",
    upstream_source="LiTS abdominal CT, axial view",
)
ORGANCMNIST = _medmnist_224(
    key="organcmnist_medmnist_3_0_2_224",
    flag="organcmnist",
    num_classes=11,
    split_sizes={"train": 12_975, "validation": 2_392, "test": 8_216},
    md5="050f5e875dc056f6768abf94ec9995d1",
    upstream_source="LiTS abdominal CT, coronal view",
)
ORGANSMNIST = _medmnist_224(
    key="organsmnist_medmnist_3_0_2_224",
    flag="organsmnist",
    num_classes=11,
    split_sizes={"train": 13_932, "validation": 2_452, "test": 8_827},
    md5="b354719e553fbbb2513d5533f52a4cb1",
    upstream_source="LiTS abdominal CT, sagittal view",
)


FOOD101 = HFDatasetSpec(
    key="food101_ethz_composite_v1",
    dataset_id="ethz/food101",
    revision=(
        "composite-hf-v1-food101-83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9-"
        "deterministic-train-holdout-75-per-class"
    ),
    num_classes=101,
    image_column="image",
    label_column="label",
    text_column=None,
    splits=("train", "validation", "test"),
    expected_split_sizes={"train": 68_175, "validation": 7_575, "test": 25_250},
    expected_examples_per_class={"train": None, "validation": None, "test": None},
    source_url="https://huggingface.co/datasets/ethz/food101",
    upstream_source="Food-101",
    license="unknown",
    provider="composite_huggingface",
    provider_options={
        "split_sources": {
            "train": {
                "dataset_id": "ethz/food101",
                "revision": "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9",
                "split": "train",
                "image_column": "image",
                "label_column": "label",
                "selection": {
                    "strategy": "stratified_hash_holdout_v1",
                    "role": "complement",
                    "count_per_class": 75,
                    "seed": "food101-validation-v1",
                },
            },
            "validation": {
                "dataset_id": "ethz/food101",
                "revision": "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9",
                "split": "train",
                "image_column": "image",
                "label_column": "label",
                "selection": {
                    "strategy": "stratified_hash_holdout_v1",
                    "role": "holdout",
                    "count_per_class": 75,
                    "seed": "food101-validation-v1",
                },
            },
            "test": {
                "dataset_id": "ethz/food101",
                "revision": "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9",
                "split": "validation",
                "image_column": "image",
                "label_column": "label",
            },
        },
        "logical_split_policy": "official_test_as_test_deterministic_train_holdout",
    },
)


# Use the complete pinned Places365 training mirror, excluding only the
# deterministic 50-per-class validation holdout. No per-class train cap.
PLACES365 = HFDatasetSpec(
    key="places365_composite_v1",
    dataset_id="places365/standard",
    revision=(
        "composite-hf-v1-places-train-7895e75528d78c16e0c31182ce02e541f44ccaaf-"
        "val-f11b9b3c7ddd678ba92fd6862296b0d42c8723bb-holdout-50-per-class"
    ),
    num_classes=365,
    image_column="image",
    label_column="label",
    text_column=None,
    splits=("train", "validation", "test"),
    expected_split_sizes={"train": 1_821_710, "validation": 18_250, "test": 36_500},
    expected_examples_per_class={"train": None, "validation": None, "test": None},
    source_url=(
        "https://huggingface.co/datasets/Andron00e/Places365-custom; "
        "https://huggingface.co/datasets/dpdl-benchmark/Places365-Validation"
    ),
    upstream_source="Places365 standard",
    license="unknown",
    provider="composite_huggingface",
    provider_options={
        "split_sources": {
            "train": {
                "dataset_id": "Andron00e/Places365-custom",
                "revision": "7895e75528d78c16e0c31182ce02e541f44ccaaf",
                "split": "train",
                "image_column": "image",
                "label_column": "labels",
                "selection": {
                    "strategy": "stratified_hash_holdout_v1",
                    "role": "complement",
                    "count_per_class": 50,
                    "seed": "places365-validation-v1",
                },
            },
            "validation": {
                "dataset_id": "Andron00e/Places365-custom",
                "revision": "7895e75528d78c16e0c31182ce02e541f44ccaaf",
                "split": "train",
                "image_column": "image",
                "label_column": "labels",
                "selection": {
                    "strategy": "stratified_hash_holdout_v1",
                    "role": "holdout",
                    "count_per_class": 50,
                    "seed": "places365-validation-v1",
                },
            },
            "test": {
                "dataset_id": "dpdl-benchmark/Places365-Validation",
                "revision": "f11b9b3c7ddd678ba92fd6862296b0d42c8723bb",
                "split": "train",
                "image_column": "image",
                "label_column": "label",
            },
        },
        "logical_split_policy": "official_validation_as_test_deterministic_train_holdout",
    },
)


DATASET_REGISTRY: Mapping[str, HFDatasetSpec] = {
    IMAGENET1K.key: IMAGENET1K,
    IMAGENET100.key: IMAGENET100,
    DERMAMNIST.key: DERMAMNIST,
    PATHMNIST.key: PATHMNIST,
    OCTMNIST.key: OCTMNIST,
    PNEUMONIAMNIST.key: PNEUMONIAMNIST,
    RETINAMNIST.key: RETINAMNIST,
    BREASTMNIST.key: BREASTMNIST,
    BLOODMNIST.key: BLOODMNIST,
    TISSUEMNIST.key: TISSUEMNIST,
    ORGANAMNIST.key: ORGANAMNIST,
    ORGANCMNIST.key: ORGANCMNIST,
    ORGANSMNIST.key: ORGANSMNIST,
    FOOD101.key: FOOD101,
    PLACES365.key: PLACES365,
}
DATASET_ALIASES: Mapping[str, str] = {
    "imagenet1k": IMAGENET1K.key,
    "imagenet": IMAGENET1K.key,
    "imagenet100": IMAGENET100.key,
    "dermamnist": DERMAMNIST.key,
    "pathmnist": PATHMNIST.key,
    "octmnist": OCTMNIST.key,
    "pneumoniamnist": PNEUMONIAMNIST.key,
    "retinamnist": RETINAMNIST.key,
    "breastmnist": BREASTMNIST.key,
    "bloodmnist": BLOODMNIST.key,
    "tissuemnist": TISSUEMNIST.key,
    "organamnist": ORGANAMNIST.key,
    "organcmnist": ORGANCMNIST.key,
    "organsmnist": ORGANSMNIST.key,
    "food101": FOOD101.key,
    "places365": PLACES365.key,
}


def get_dataset_spec(key: str) -> HFDatasetSpec:
    """Resolve a registered dataset without importing ``datasets``."""

    try:
        return DATASET_REGISTRY[DATASET_ALIASES.get(key, key)]
    except KeyError as exc:
        available = ", ".join(sorted(set(DATASET_REGISTRY) | set(DATASET_ALIASES)))
        raise KeyError(f"Unknown dataset {key!r}; available: {available}") from exc
