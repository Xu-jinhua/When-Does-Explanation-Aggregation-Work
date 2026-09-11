"""Stable, auditable manifests for image datasets.

The manifest deliberately stores both the row index and a content digest.
Pinned row indices make access efficient; the digest prevents a row index from
being mistaken for identity when data are copied, reordered, or corrupted.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .specs import HFDatasetSpec

MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_HASH_MODES = frozenset({"decoded_rgb", "encoded_bytes"})


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    sample_id: str
    content_sha256: str
    split: str
    row_index: int
    label: int
    label_name: str
    encoded_num_bytes: int | None = None
    source_text: str | None = None

    def __post_init__(self) -> None:
        if len(self.sample_id) != 64 or len(self.content_sha256) != 64:
            raise ValueError("sample_id and content_sha256 must be SHA-256 hex digests")
        if self.row_index < 0 or self.label < 0:
            raise ValueError("row_index and label must be non-negative")


@dataclass(frozen=True, slots=True)
class ManifestMetadata:
    dataset_key: str
    dataset_id: str
    revision: str
    dataset_spec_fingerprint: str
    hash_mode: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    created_utc: str = ""

    def __post_init__(self) -> None:
        if self.hash_mode not in SUPPORTED_HASH_MODES:
            raise ValueError(f"Unsupported image hash mode: {self.hash_mode}")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema {self.schema_version}")


@dataclass(frozen=True, slots=True)
class ManifestAudit:
    split_sizes: Mapping[str, int]
    class_counts: Mapping[str, Mapping[int, int]]
    duplicate_sample_ids: tuple[str, ...]
    duplicate_contents_within_split: Mapping[str, tuple[str, ...]]
    duplicate_contents_across_splits: Mapping[str, tuple[str, ...]]

    @property
    def is_identity_clean(self) -> bool:
        return not self.duplicate_sample_ids

    @property
    def has_cross_split_content_duplicates(self) -> bool:
        return bool(self.duplicate_contents_across_splits)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    metadata: ManifestMetadata
    records: tuple[ManifestRecord, ...]

    @property
    def fingerprint(self) -> str:
        """Digest the identity-bearing fields, independent of scan time/order."""

        digest = hashlib.sha256()
        digest.update(str(self.metadata.schema_version).encode("ascii"))
        digest.update(b"\0")
        digest.update(self.metadata.dataset_spec_fingerprint.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.metadata.hash_mode.encode("ascii"))
        for record in sorted(self.records, key=lambda item: (item.split, item.row_index)):
            digest.update(b"\0")
            digest.update(record.sample_id.encode("ascii"))
            digest.update(b"\0")
            digest.update(record.content_sha256.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(record.label).encode("ascii"))
        return digest.hexdigest()

    def records_for_split(self, split: str) -> tuple[ManifestRecord, ...]:
        return tuple(record for record in self.records if record.split == split)

    def by_sample_id(self) -> dict[str, ManifestRecord]:
        result = {record.sample_id: record for record in self.records}
        if len(result) != len(self.records):
            raise ValueError("Manifest contains duplicate sample IDs")
        return result

    def audit(self) -> ManifestAudit:
        split_sizes: Counter[str] = Counter()
        class_counts: dict[str, Counter[int]] = defaultdict(Counter)
        sample_ids: Counter[str] = Counter()
        content_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for record in self.records:
            split_sizes[record.split] += 1
            class_counts[record.split][record.label] += 1
            sample_ids[record.sample_id] += 1
            content_locations[record.content_sha256].append((record.split, record.sample_id))

        duplicate_within: dict[str, list[str]] = defaultdict(list)
        duplicate_across: dict[str, tuple[str, ...]] = {}
        for content_hash, locations in content_locations.items():
            location_splits = [split for split, _ in locations]
            unique_splits = sorted(set(location_splits))
            if len(unique_splits) > 1:
                duplicate_across[content_hash] = tuple(unique_splits)
            for split, count in Counter(location_splits).items():
                if count > 1:
                    duplicate_within[split].append(content_hash)

        return ManifestAudit(
            split_sizes=dict(split_sizes),
            class_counts={split: dict(counts) for split, counts in class_counts.items()},
            duplicate_sample_ids=tuple(
                sorted(key for key, count in sample_ids.items() if count > 1)
            ),
            duplicate_contents_within_split={
                split: tuple(sorted(hashes)) for split, hashes in duplicate_within.items()
            },
            duplicate_contents_across_splits=dict(sorted(duplicate_across.items())),
        )

    def validate(
        self,
        spec: HFDatasetSpec | None = None,
        *,
        require_expected_counts: bool = True,
        reject_cross_split_duplicates: bool = False,
    ) -> ManifestAudit:
        audit = self.audit()
        if audit.duplicate_sample_ids:
            raise ValueError(f"Duplicate stable sample IDs: {audit.duplicate_sample_ids[:3]}")
        locations: set[tuple[str, int]] = set()
        for record in self.records:
            location = (record.split, record.row_index)
            if location in locations:
                raise ValueError(f"Duplicate manifest row location: {location}")
            locations.add(location)
            expected_sample_id = stable_sample_id(
                dataset_id=self.metadata.dataset_id,
                revision=self.metadata.revision,
                split=record.split,
                row_index=record.row_index,
                label=record.label,
                content_sha256=record.content_sha256,
            )
            if record.sample_id != expected_sample_id:
                raise ValueError(
                    f"Stable sample ID does not match identity fields at "
                    f"{record.split}[{record.row_index}]"
                )

        if spec is not None:
            if self.metadata.dataset_spec_fingerprint != spec.fingerprint:
                raise ValueError("Manifest was produced from a different dataset specification")
            unknown_splits = set(audit.split_sizes) - set(spec.splits)
            if unknown_splits:
                raise ValueError(f"Manifest contains unknown splits: {sorted(unknown_splits)}")
            if require_expected_counts:
                for split in spec.splits:
                    spec.validate_split_size(split, audit.split_sizes.get(split, 0))
                    expected_per_class = spec.expected_examples_per_class[split]
                    counts = audit.class_counts.get(split, {})
                    expected_labels = set(range(spec.num_classes))
                    if set(counts) != expected_labels:
                        missing = sorted(expected_labels - set(counts))
                        extra = sorted(set(counts) - expected_labels)
                        raise ValueError(
                            f"Unexpected labels in {split}: missing={missing}, extra={extra}"
                        )
                    exact_counts = dict(
                        spec.provider_options.get("expected_class_counts", {}).get(split, {})
                    )
                    if expected_per_class is None and not exact_counts:
                        bad = {}
                    elif exact_counts:
                        bad = {
                            label: count
                            for label, count in counts.items()
                            if count
                            != int(exact_counts.get(label, exact_counts.get(str(label), -1)))
                        }
                    else:
                        bad = {
                            label: count
                            for label, count in counts.items()
                            if count != expected_per_class
                        }
                    if bad:
                        raise ValueError(
                            f"Unexpected class counts in {split}; expected "
                            f"{exact_counts or expected_per_class}, "
                            f"examples={dict(list(sorted(bad.items()))[:5])}"
                        )

        if reject_cross_split_duplicates and audit.has_cross_split_content_duplicates:
            examples = list(audit.duplicate_contents_across_splits.items())[:3]
            raise ValueError(f"Image content occurs in multiple splits: {examples}")
        return audit


def stable_sample_id(
    *,
    dataset_id: str,
    revision: str,
    split: str,
    row_index: int,
    label: int,
    content_sha256: str,
) -> str:
    """Combine pinned location and content into a stable sample identifier."""

    if row_index < 0 or label < 0:
        raise ValueError("row_index and label must be non-negative")
    canonical = "\0".join((dataset_id, revision, split, str(row_index), str(label), content_sha256))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _encoded_bytes(image: Any) -> bytes:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, Mapping):
        payload = image.get("bytes")
        if payload is not None:
            return bytes(payload)
        path = image.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(image, (str, os.PathLike)):
        return Path(image).read_bytes()
    raise TypeError("Image does not expose encoded bytes or a filesystem path")


def hash_image_content(image: Any, *, mode: str = "decoded_rgb") -> tuple[str, int | None]:
    """Hash encoded bytes or canonical decoded RGB pixels.

    The default canonical pixel hash detects the same image even if JPEG/PNG
    container metadata differs.  Pillow is imported only when this mode is
    actually used.
    """

    if mode not in SUPPORTED_HASH_MODES:
        raise ValueError(f"Unsupported image hash mode: {mode}")

    encoded: bytes | None = None
    try:
        encoded = _encoded_bytes(image)
    except TypeError:
        pass

    if mode == "encoded_bytes":
        if encoded is None:
            raise TypeError("encoded_bytes hashing requires bytes or a path")
        return hashlib.sha256(encoded).hexdigest(), len(encoded)

    try:
        from PIL import Image as PILImage
    except ImportError as exc:  # pragma: no cover - exercised in minimal deployments
        raise RuntimeError("decoded_rgb hashing requires Pillow") from exc

    if encoded is not None:
        with PILImage.open(io.BytesIO(encoded)) as opened:
            canonical = opened.convert("RGB")
            width, height = canonical.size
            pixels = canonical.tobytes()
    elif isinstance(image, PILImage.Image):
        canonical = image.convert("RGB")
        width, height = canonical.size
        pixels = canonical.tobytes()
    else:
        raise TypeError(f"Unsupported image value for hashing: {type(image).__name__}")

    digest = hashlib.sha256()
    digest.update(b"decoded-rgb-v1\0")
    digest.update(str(width).encode("ascii"))
    digest.update(b"x")
    digest.update(str(height).encode("ascii"))
    digest.update(b"\0")
    digest.update(pixels)
    return digest.hexdigest(), None if encoded is None else len(encoded)


def _label_names(dataset: Any, label_column: str) -> Sequence[str] | None:
    features = getattr(dataset, "features", None)
    if features is None:
        return None
    try:
        names = features[label_column].names
    except (AttributeError, KeyError, TypeError):
        return None
    if names is None:
        return None
    return tuple(str(name) for name in names)


def records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    spec: HFDatasetSpec,
    split: str,
    label_names: Sequence[str] | None = None,
    hash_mode: str = "decoded_rgb",
) -> Iterator[ManifestRecord]:
    """Convert dataset rows into records without depending on HF datasets."""

    for row_index, row in enumerate(rows):
        label = int(row[spec.label_column])
        if not 0 <= label < spec.num_classes:
            raise ValueError(f"Label {label} at {split}[{row_index}] is out of range")
        content_hash, encoded_num_bytes = hash_image_content(row[spec.image_column], mode=hash_mode)
        sample_id = stable_sample_id(
            dataset_id=spec.dataset_id,
            revision=spec.revision,
            split=split,
            row_index=row_index,
            label=label,
            content_sha256=content_hash,
        )
        label_name = str(label) if label_names is None else str(label_names[label])
        source_text = None
        if spec.text_column is not None and row.get(spec.text_column) is not None:
            source_text = str(row[spec.text_column])
        yield ManifestRecord(
            sample_id=sample_id,
            content_sha256=content_hash,
            split=split,
            row_index=row_index,
            label=label,
            label_name=label_name,
            encoded_num_bytes=encoded_num_bytes,
            source_text=source_text,
        )


def build_hf_manifest(
    spec: HFDatasetSpec,
    *,
    splits: Sequence[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    streaming: bool = False,
    token: str | bool | None = None,
    hash_mode: str = "decoded_rgb",
    require_expected_counts: bool = True,
) -> DatasetManifest:
    """Scan a pinned HF revision and construct its complete manifest.

    ``datasets`` is an optional runtime dependency and therefore imported only
    here.  Streaming avoids a second prepared Arrow cache when local disk is
    scarce; non-streaming is faster when the Parquet shards are already cached.
    """

    if spec.provider not in {"huggingface", "composite_huggingface"}:
        raise ValueError(f"Dataset {spec.key} is provided by {spec.provider}, not Hugging Face")
    selected_splits = tuple(spec.splits if splits is None else splits)
    unknown = set(selected_splits) - set(spec.splits)
    if unknown:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown)}")

    records: list[ManifestRecord] = []
    for split in selected_splits:
        if spec.provider == "composite_huggingface":
            if streaming:
                raise ValueError("Composite Hugging Face datasets do not support streaming")
            from .composite import load_composite_split

            dataset = load_composite_split(
                spec,
                split,
                cache_dir=cache_dir,
                keep_in_memory=False,
                token=token,
            )
        else:
            from xai_ensemble.data.hf_parquet_recovery import (
                load_hf_parquet_recovery_source,
                recovery_source_for_config,
            )

            recovery_config = {
                "dataset_id": spec.dataset_id,
                "revision": spec.revision,
                "split": split,
                "image_column": spec.image_column,
                "label_column": spec.label_column,
                "text_column": spec.text_column,
            }
            if recovery_source_for_config(recovery_config) is not None:
                if streaming:
                    raise ValueError("Direct Parquet recovery does not support streaming mode")
                dataset = load_hf_parquet_recovery_source(recovery_config)
            else:
                try:
                    from datasets import load_dataset
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError("build_hf_manifest requires the 'datasets' package") from exc
                load_kwargs: dict[str, Any] = {
                    "path": spec.dataset_id,
                    "revision": spec.revision,
                    "split": split,
                    "streaming": streaming,
                }
                if cache_dir is not None:
                    load_kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
                if token is not None:
                    load_kwargs["token"] = token
                dataset = load_dataset(**load_kwargs)
        names = _label_names(dataset, spec.label_column)
        records.extend(
            records_from_rows(
                dataset,
                spec=spec,
                split=split,
                label_names=names,
                hash_mode=hash_mode,
            )
        )

    metadata = ManifestMetadata(
        dataset_key=spec.key,
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        dataset_spec_fingerprint=spec.fingerprint,
        hash_mode=hash_mode,
        created_utc=datetime.now(UTC).isoformat(),
    )
    manifest = DatasetManifest(metadata=metadata, records=tuple(records))
    manifest.validate(
        spec,
        require_expected_counts=require_expected_counts
        and set(selected_splits) == set(spec.splits),
    )
    return manifest


def build_medmnist_manifest(
    spec: HFDatasetSpec,
    *,
    splits: Sequence[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    download: bool = True,
    hash_mode: str = "decoded_rgb",
    require_expected_counts: bool = True,
) -> DatasetManifest:
    """Scan a pinned MedMNIST release using its official split definitions."""

    if spec.provider != "medmnist":
        raise ValueError(f"Dataset {spec.key} is provided by {spec.provider}, not MedMNIST")
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("build_medmnist_manifest requires the 'medmnist' package") from exc

    required_version = str(spec.provider_options["version"])
    installed_version = importlib.metadata.version("medmnist")
    if installed_version != required_version:
        raise RuntimeError(
            f"MedMNIST version mismatch: installed={installed_version}, required={required_version}"
        )
    data_flag = str(spec.provider_options["data_flag"])
    info = INFO[data_flag]
    data_class = getattr(medmnist, info["python_class"])
    label_values = info["label"]
    label_names = tuple(
        str(label_values[str(index)] if str(index) in label_values else label_values[index])
        for index in range(spec.num_classes)
    )
    selected_splits = tuple(spec.splits if splits is None else splits)
    unknown = set(selected_splits) - set(spec.splits)
    if unknown:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown)}")

    root_path: Path | None = None
    if root is not None:
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)

    verified_local_source = None
    if root_path is not None and {
        "filename",
        "source_url",
        "md5",
        "zenodo_record",
    }.issubset(spec.provider_options):
        from xai_ensemble.data.medmnist_recovery import (
            ensure_verified_medmnist_source,
            has_verified_source_contract,
        )

        if has_verified_source_contract(spec):
            verified_local_source = ensure_verified_medmnist_source(
                spec, root_path, download=download
            )

    records: list[ManifestRecord] = []
    for split in selected_splits:
        provider_split = "val" if split == "validation" else split
        dataset_kwargs: dict[str, Any] = {
            "split": provider_split,
            "download": download and verified_local_source is None,
            "as_rgb": bool(spec.provider_options.get("as_rgb", True)),
        }
        if "size" in inspect.signature(data_class.__init__).parameters:
            dataset_kwargs["size"] = int(spec.provider_options.get("size", 224))
        if root_path is not None:
            dataset_kwargs["root"] = str(root_path)
        dataset = data_class(**dataset_kwargs)

        def rows(current_dataset: Any = dataset) -> Iterator[Mapping[str, Any]]:
            for index in range(len(current_dataset)):
                image, label = current_dataset[index][:2]
                if hasattr(label, "item"):
                    label = label.item()
                yield {spec.image_column: image, spec.label_column: int(label)}

        records.extend(
            records_from_rows(
                rows(),
                spec=spec,
                split=split,
                label_names=label_names,
                hash_mode=hash_mode,
            )
        )

    metadata = ManifestMetadata(
        dataset_key=spec.key,
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        dataset_spec_fingerprint=spec.fingerprint,
        hash_mode=hash_mode,
        created_utc=datetime.now(UTC).isoformat(),
    )
    manifest = DatasetManifest(metadata=metadata, records=tuple(records))
    manifest.validate(
        spec,
        require_expected_counts=require_expected_counts
        and set(selected_splits) == set(spec.splits),
    )
    return manifest


def build_dataset_manifest(
    spec: HFDatasetSpec,
    *,
    splits: Sequence[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    streaming: bool = False,
    token: str | bool | None = None,
    hash_mode: str = "decoded_rgb",
    require_expected_counts: bool = True,
) -> DatasetManifest:
    """Dispatch manifest construction to the spec's pinned provider."""

    if spec.provider == "imagenet":
        from .imagenet import load_imagenet_split

        if streaming:
            raise ValueError("Local ImageNet does not use the HF streaming interface")
        selected_splits = tuple(spec.splits if splits is None else splits)
        records = []
        for split in selected_splits:
            dataset = load_imagenet_split(spec, split, root=cache_dir)
            records.extend(records_from_rows(
                dataset, spec=spec, split=split,
                label_names=_label_names(dataset, spec.label_column), hash_mode=hash_mode,
            ))
        manifest = DatasetManifest(
            metadata=ManifestMetadata(
                dataset_key=spec.key, dataset_id=spec.dataset_id, revision=spec.revision,
                dataset_spec_fingerprint=spec.fingerprint, hash_mode=hash_mode,
                created_utc=datetime.now(UTC).isoformat(),
            ),
            records=tuple(records),
        )
        manifest.validate(
            spec, require_expected_counts=require_expected_counts
            and set(selected_splits) == set(spec.splits),
        )
        return manifest
    if spec.provider in {"huggingface", "composite_huggingface"}:
        return build_hf_manifest(
            spec,
            splits=splits,
            cache_dir=cache_dir,
            streaming=streaming,
            token=token,
            hash_mode=hash_mode,
            require_expected_counts=require_expected_counts,
        )
    if streaming:
        raise ValueError("MedMNIST does not support streaming mode")
    if token is not None:
        raise ValueError("MedMNIST does not use a Hugging Face token")
    return build_medmnist_manifest(
        spec,
        splits=splits,
        root=cache_dir,
        hash_mode=hash_mode,
        require_expected_counts=require_expected_counts,
    )


def write_manifest(manifest: DatasetManifest, path: str | os.PathLike[str]) -> Path:
    """Atomically seal a manifest and accept identical retries.

    A regenerated scan may have a different ``created_utc`` value, but it must
    describe the exact same dataset identity and records.  Existing content is
    never replaced silently.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = read_manifest(destination)
        existing_identity = (
            existing.metadata.dataset_key,
            existing.metadata.dataset_id,
            existing.metadata.revision,
            existing.metadata.dataset_spec_fingerprint,
            existing.metadata.hash_mode,
            existing.metadata.schema_version,
        )
        requested_identity = (
            manifest.metadata.dataset_key,
            manifest.metadata.dataset_id,
            manifest.metadata.revision,
            manifest.metadata.dataset_spec_fingerprint,
            manifest.metadata.hash_mode,
            manifest.metadata.schema_version,
        )
        if existing_identity != requested_identity or existing.fingerprint != manifest.fingerprint:
            raise ValueError(f"refusing to replace immutable dataset manifest {destination}")
        return destination
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            header = {"_type": "metadata", **asdict(manifest.metadata)}
            handle.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
            for record in manifest.records:
                row = {"_type": "record", **asdict(record)}
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def read_manifest(path: str | os.PathLike[str]) -> DatasetManifest:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        try:
            header = json.loads(next(handle))
        except StopIteration as exc:
            raise ValueError(f"Empty manifest: {source}") from exc
        if header.pop("_type", None) != "metadata":
            raise ValueError("Manifest first line must contain metadata")
        metadata = ManifestMetadata(**header)
        records: list[ManifestRecord] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.pop("_type", None) != "record":
                raise ValueError(f"Invalid manifest row type at line {line_number}")
            records.append(ManifestRecord(**row))
    return DatasetManifest(metadata=metadata, records=tuple(records))
