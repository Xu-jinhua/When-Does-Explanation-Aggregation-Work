"""Deterministic class-stratified partitions for IND and OVERLAP studies."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .manifest import DatasetManifest, ManifestRecord

PARTITION_SCHEMA_VERSION = 1
PARTITION_KINDS = frozenset({"ind", "overlap", "reference"})


@dataclass(frozen=True, slots=True)
class SourcePartition:
    source_id: str
    sample_ids: tuple[str, ...]
    class_counts: Mapping[int, int]

    @property
    def size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    kind: str
    strategy: str
    split: str
    seed: int
    dataset_manifest_fingerprint: str
    sources: tuple[SourcePartition, ...]
    samples_per_class_per_source: int | None
    matched_ind_digest: str | None = None
    schema_version: int = PARTITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in PARTITION_KINDS:
            raise ValueError(f"Unknown partition kind: {self.kind}")
        if not self.strategy:
            raise ValueError("Partition strategy must be explicit")
        if not self.sources:
            raise ValueError("Partition plan must contain at least one source")
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("source_id values must be unique")

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "strategy": self.strategy,
            "split": self.split,
            "seed": self.seed,
            "dataset_manifest_fingerprint": self.dataset_manifest_fingerprint,
            "samples_per_class_per_source": self.samples_per_class_per_source,
            "matched_ind_digest": self.matched_ind_digest,
            "sources": [
                {
                    "source_id": source.source_id,
                    "sample_ids": list(source.sample_ids),
                    "class_counts": {str(k): v for k, v in sorted(source.class_counts.items())},
                }
                for source in self.sources
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def source(self, source_id: str) -> SourcePartition:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"Unknown source_id {source_id!r}")

    def overlap_matrix(self) -> tuple[tuple[int, ...], ...]:
        sample_sets = [set(source.sample_ids) for source in self.sources]
        return tuple(tuple(len(left & right) for right in sample_sets) for left in sample_sets)

    def validate(self, manifest: DatasetManifest) -> None:
        if self.dataset_manifest_fingerprint != manifest.fingerprint:
            raise ValueError("Partition plan belongs to a different dataset manifest")
        available_records = manifest.records_for_split(self.split)
        available = {record.sample_id: record for record in available_records}
        if not available:
            raise ValueError(f"Manifest does not contain split {self.split!r}")

        for source in self.sources:
            if len(source.sample_ids) != len(set(source.sample_ids)):
                raise ValueError(f"Source {source.source_id} contains duplicate samples")
            missing = set(source.sample_ids) - set(available)
            if missing:
                raise ValueError(
                    f"Source {source.source_id} references missing samples: {sorted(missing)[:3]}"
                )
            actual = Counter(available[sample_id].label for sample_id in source.sample_ids)
            if dict(actual) != dict(source.class_counts):
                raise ValueError(f"Class-count metadata mismatch in {source.source_id}")
            if self.samples_per_class_per_source is not None:
                if set(actual.values()) != {self.samples_per_class_per_source}:
                    raise ValueError(f"Source {source.source_id} is not class balanced")

        if self.kind == "ind":
            seen: set[str] = set()
            for source in self.sources:
                overlap = seen.intersection(source.sample_ids)
                if overlap:
                    raise ValueError(
                        f"IND partitions are not disjoint; first duplicates: {sorted(overlap)[:3]}"
                    )
                seen.update(source.sample_ids)
            if self.strategy == "disjoint_stratified_full" and seen != set(available):
                raise ValueError("Full-coverage IND plan does not use every split sample")
        elif self.kind == "reference":
            if len(self.sources) != 1:
                raise ValueError("Reference plan must contain exactly one full-data source")
            if set(self.sources[0].sample_ids) != set(available):
                raise ValueError("Reference source must use the complete split")
        elif self.matched_ind_digest is not None:
            if len(self.matched_ind_digest) != 64:
                raise ValueError("matched OVERLAP must record a valid IND plan digest")


def _group_by_class(records: Iterable[ManifestRecord]) -> dict[int, list[ManifestRecord]]:
    grouped: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)
    if not grouped:
        raise ValueError("Cannot partition an empty record collection")
    return dict(grouped)


def _class_seed(seed: int, label: int, source_id: str = "") -> int:
    payload = f"partition-v1\0{seed}\0{label}\0{source_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_ind_partitions(
    manifest: DatasetManifest,
    *,
    split: str = "train",
    num_sources: int,
    seed: int,
    require_full_coverage: bool = True,
) -> PartitionPlan:
    """Partition every class into mutually disjoint, equal-size sources."""

    if num_sources < 2:
        raise ValueError("IND requires at least two source models")
    grouped = _group_by_class(manifest.records_for_split(split))
    class_sizes = {label: len(records) for label, records in grouped.items()}
    if sum(class_sizes.values()) < num_sources:
        raise ValueError("Not enough samples for the requested source count")

    assigned: list[list[ManifestRecord]] = [[] for _ in range(num_sources)]
    assigned_sizes = [0] * num_sources
    for label, records in sorted(grouped.items()):
        shuffled = sorted(records, key=lambda record: record.sample_id)
        random.Random(_class_seed(seed, label)).shuffle(shuffled)
        quota, remainder = divmod(len(shuffled), num_sources)
        for source_index in range(num_sources):
            start = source_index * quota
            assigned[source_index].extend(shuffled[start : start + quota])
            assigned_sizes[source_index] += quota
        if require_full_coverage and remainder:
            extras = shuffled[quota * num_sources :]
            tie_order = list(range(num_sources))
            random.Random(_class_seed(seed, label, "remainder")).shuffle(tie_order)
            tie_position = {source_index: index for index, source_index in enumerate(tie_order)}
            recipients = sorted(
                range(num_sources),
                key=lambda source_index: (assigned_sizes[source_index], tie_position[source_index]),
            )[:remainder]
            for source_index, record in zip(recipients, extras, strict=True):
                assigned[source_index].append(record)
                assigned_sizes[source_index] += 1

    sources = tuple(
        _source_partition(f"source-{index:02d}", records) for index, records in enumerate(assigned)
    )
    plan = PartitionPlan(
        kind="ind",
        strategy=(
            "disjoint_stratified_full"
            if require_full_coverage
            else "disjoint_stratified_drop_remainder"
        ),
        split=split,
        seed=seed,
        dataset_manifest_fingerprint=manifest.fingerprint,
        sources=sources,
        samples_per_class_per_source=_uniform_class_quota(sources),
    )
    plan.validate(manifest)
    return plan


def make_overlap_partitions(
    manifest: DatasetManifest,
    *,
    split: str = "train",
    num_sources: int,
    seed: int,
    samples_per_class: int | None = None,
    class_quotas: Mapping[int, int] | None = None,
    mode: str = "shared",
    matched_ind_digest: str | None = None,
) -> PartitionPlan:
    """Create matched-size sources with overlapping class-balanced data.

    ``shared`` is the primary matched-OVERLAP control: every source sees the
    exact same stratified subset, maximizing training-data dependence while
    matching each IND source's sample count. ``independent`` samples each source
    separately and is provided as a secondary partial-overlap diagnostic.
    """

    if num_sources < 2:
        raise ValueError("OVERLAP requires at least two source models")
    if mode not in {"shared", "independent"}:
        raise ValueError("OVERLAP mode must be shared or independent")
    grouped = _group_by_class(manifest.records_for_split(split))
    if samples_per_class is not None and class_quotas is not None:
        raise ValueError("Specify either samples_per_class or class_quotas, not both")
    if class_quotas is not None:
        quotas = {int(label): int(value) for label, value in class_quotas.items()}
        if set(quotas) != set(grouped):
            raise ValueError("class_quotas must define every observed class")
    elif samples_per_class is not None:
        quotas = {label: samples_per_class for label in grouped}
    else:
        quotas = _proportional_source_quotas(grouped, num_sources)
    for label, quota in quotas.items():
        if not 0 <= quota <= len(grouped[label]):
            raise ValueError(
                f"Invalid quota {quota} for class {label} with {len(grouped[label])} samples"
            )

    assigned: list[list[ManifestRecord]] = [[] for _ in range(num_sources)]
    for label, records in sorted(grouped.items()):
        quota = quotas[label]
        canonical = sorted(records, key=lambda record: record.sample_id)
        if mode == "shared":
            shuffled = list(canonical)
            random.Random(_class_seed(seed, label, "shared")).shuffle(shuffled)
            selected = shuffled[:quota]
            for source_records in assigned:
                source_records.extend(selected)
            continue
        for source_index in range(num_sources):
            source_id = f"source-{source_index:02d}"
            shuffled = list(canonical)
            random.Random(_class_seed(seed, label, source_id)).shuffle(shuffled)
            assigned[source_index].extend(shuffled[:quota])

    sources = tuple(
        _source_partition(f"source-{index:02d}", records) for index, records in enumerate(assigned)
    )
    plan = PartitionPlan(
        kind="overlap",
        strategy=f"{mode}_stratified",
        split=split,
        seed=seed,
        dataset_manifest_fingerprint=manifest.fingerprint,
        sources=sources,
        samples_per_class_per_source=_uniform_class_quota(sources),
        matched_ind_digest=matched_ind_digest,
    )
    plan.validate(manifest)
    return plan


def make_reference_partition(
    manifest: DatasetManifest,
    *,
    split: str = "train",
    seed: int = 0,
) -> PartitionPlan:
    """Register the common evaluator/reference model's full-data training set."""

    records = manifest.records_for_split(split)
    source = _source_partition("reference-full", records)
    plan = PartitionPlan(
        kind="reference",
        strategy="full_split",
        split=split,
        seed=seed,
        dataset_manifest_fingerprint=manifest.fingerprint,
        sources=(source,),
        samples_per_class_per_source=None,
    )
    plan.validate(manifest)
    return plan


def _source_partition(source_id: str, records: Iterable[ManifestRecord]) -> SourcePartition:
    ordered = sorted(records, key=lambda record: (record.label, record.sample_id))
    return SourcePartition(
        source_id=source_id,
        sample_ids=tuple(record.sample_id for record in ordered),
        class_counts=dict(Counter(record.label for record in ordered)),
    )


def _uniform_class_quota(sources: Iterable[SourcePartition]) -> int | None:
    values = {count for source in sources for count in source.class_counts.values()}
    return next(iter(values)) if len(values) == 1 else None


def _proportional_source_quotas(
    grouped: Mapping[int, list[ManifestRecord]],
    num_sources: int,
) -> dict[int, int]:
    """Match one source's size while preserving an imbalanced class prior."""

    sizes = {label: len(records) for label, records in grouped.items()}
    quotas = {label: size // num_sources for label, size in sizes.items()}
    target_size = round(sum(sizes.values()) / num_sources)
    remaining = target_size - sum(quotas.values())
    priority = sorted(
        sizes,
        key=lambda label: (-(sizes[label] % num_sources), label),
    )
    for label in priority[:remaining]:
        quotas[label] += 1
    return quotas


def write_partition_plan(plan: PartitionPlan, path: str | os.PathLike[str]) -> Path:
    """Atomically seal a partition and accept an identical retry only."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = read_partition_plan(destination)
        if existing.digest != plan.digest:
            raise ValueError(f"refusing to replace immutable partition plan {destination}")
        return destination
    payload = {
        "schema_version": plan.schema_version,
        "kind": plan.kind,
        "strategy": plan.strategy,
        "split": plan.split,
        "seed": plan.seed,
        "dataset_manifest_fingerprint": plan.dataset_manifest_fingerprint,
        "samples_per_class_per_source": plan.samples_per_class_per_source,
        "matched_ind_digest": plan.matched_ind_digest,
        "digest": plan.digest,
        "sources": [
            {
                "source_id": source.source_id,
                "sample_ids": list(source.sample_ids),
                "class_counts": {str(key): value for key, value in source.class_counts.items()},
            }
            for source in plan.sources
        ],
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
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


def read_partition_plan(path: str | os.PathLike[str]) -> PartitionPlan:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_digest = payload.pop("digest", None)
    sources = tuple(
        SourcePartition(
            source_id=row["source_id"],
            sample_ids=tuple(row["sample_ids"]),
            class_counts={int(key): int(value) for key, value in row["class_counts"].items()},
        )
        for row in payload.pop("sources")
    )
    plan = PartitionPlan(sources=sources, **payload)
    if expected_digest is not None and plan.digest != expected_digest:
        raise ValueError("Partition plan digest does not match its contents")
    return plan
