"""Pinned composite Hugging Face dataset views.

Some of the public classification datasets expose only a train/test pair,
while the experiment protocol needs train/validation/test.  This module keeps
that conversion explicit: a logical split is a deterministic view over one
or more pinned source revisions.  The view is deliberately small and
map-style so it can be consumed by the existing manifest and DataLoader
implementations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .specs import HFDatasetSpec


def _source_row(source: Any, index: int, *, image_column: str, label_column: str) -> dict[str, Any]:
    row = source[index]
    if not isinstance(row, Mapping):
        raise TypeError("Composite Hugging Face sources must return mapping rows")
    label = row[label_column]
    if hasattr(label, "item"):
        label = label.item()
    if isinstance(label, Sequence) and not isinstance(label, (str, bytes)):
        if len(label) != 1:
            raise ValueError("Composite classification sources must have one label per row")
        label = label[0]
    return {"image": row[image_column], "label": int(label)}


@dataclass(frozen=True, slots=True)
class CompositeDatasetView:
    """A logical split with normalized ``image`` and ``label`` columns."""

    source: Any
    indices: tuple[int, ...]
    image_column: str
    label_column: str
    label_names: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        return _source_row(
            self.source,
            source_index,
            image_column=self.image_column,
            label_column=self.label_column,
        )

    def row_group_key(self, index: int) -> Any:
        """Forward Parquet locality only when the pinned source provides it."""

        provider = getattr(self.source, "row_group_key", None)
        if not callable(provider):
            raise AttributeError("Composite source has no Parquet row-group locality")
        return provider(self.indices[index])

    @property
    def features(self) -> Mapping[str, Any]:
        # ``manifest._label_names`` only needs a ``.names`` attribute.  Keep
        # this adapter independent of the datasets package's feature classes.
        class LabelFeature:
            def __init__(self, names: tuple[str, ...] | None) -> None:
                self.names = names

        return {"label": LabelFeature(self.label_names)}


def _source_config(spec: HFDatasetSpec, split: str) -> Mapping[str, Any]:
    sources = spec.provider_options.get("split_sources")
    if not isinstance(sources, Mapping) or split not in sources:
        raise ValueError(f"Composite dataset has no source mapping for split {split!r}")
    value = sources[split]
    if not isinstance(value, Mapping):
        raise TypeError(f"Composite split source {split!r} must be a mapping")
    return value


def _selected_indices(
    source: Any,
    *,
    source_revision: str,
    source_split: str,
    source_label_column: str,
    selection: Mapping[str, Any] | None,
) -> tuple[int, ...]:
    if selection is None:
        return tuple(range(len(source)))
    strategy = str(selection.get("strategy", ""))
    if strategy != "stratified_hash_holdout_v1":
        raise ValueError(f"Unsupported composite split selection strategy: {strategy!r}")
    unknown_fields = set(selection) - {
        "strategy",
        "role",
        "count_per_class",
        "cap_per_class",
        "seed",
        "image_column",
    }
    if unknown_fields:
        raise ValueError(f"Unsupported composite selection fields: {sorted(unknown_fields)}")
    role = str(selection.get("role", ""))
    if role not in {"holdout", "complement"}:
        raise ValueError("Composite selection role must be holdout or complement")
    count = int(selection.get("count_per_class", 0))
    seed = str(selection.get("seed", "20260811"))
    if count <= 0:
        raise ValueError("Composite holdout count must be positive")
    cap: int | None = None
    if selection.get("cap_per_class") is not None:
        cap = int(selection["cap_per_class"])
        if cap <= 0:
            raise ValueError("Composite cap_per_class must be positive")

    # The direct Places365 adapter exposes this hook so holdout selection can
    # read labels without decoding every image.  The fallback keeps ordinary
    # datasets.Dataset and small map-style test fixtures compatible.
    try:
        labels_value = source.column_values(source_label_column)
    except (AttributeError, KeyError, TypeError, IndexError):
        try:
            labels_value = source[source_label_column]
        except (KeyError, TypeError, IndexError):
            labels_value = None
    if labels_value is not None:
        labels = [int(value.item() if hasattr(value, "item") else value) for value in labels_value]
    else:
        labels = [
            int(
                _source_row(
                    source,
                    index,
                    image_column=str(selection.get("image_column", "image")),
                    label_column=source_label_column,
                )["label"]
            )
            for index in range(len(source))
        ]

    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    selected: set[int] = set()
    for label, indices in sorted(by_label.items()):
        if len(indices) <= count:
            raise ValueError(
                f"Composite holdout count={count} exhausts class {label} ({len(indices)} rows)"
            )
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}\0{source_revision}\0{source_split}\0{label}\0{index}".encode()
            ).hexdigest(),
        )
        # The cap keeps the leading rows of the same per-class hash order used
        # to carve out the holdout, so a capped complement stays disjoint from
        # the holdout and both roles remain deterministic.
        pool = ordered[:count] if role == "holdout" else ordered[count:]
        if cap is not None:
            pool = pool[:cap]
        selected.update(pool)
    return tuple(index for index in range(len(source)) if index in selected)


def load_composite_split(
    spec: HFDatasetSpec,
    split: str,
    *,
    cache_dir: str | Path | None = None,
    keep_in_memory: bool = False,
    token: str | bool | None = None,
) -> CompositeDatasetView:
    """Load one logical split from its pinned source contract."""

    if spec.provider != "composite_huggingface":
        raise ValueError(f"{spec.key} is not a composite Hugging Face specification")
    source_config = _source_config(spec, split)
    source_id = str(source_config["dataset_id"])
    source_revision = str(source_config["revision"])
    source_split = str(source_config.get("split", split))
    image_column = str(source_config.get("image_column", "image"))
    label_column = str(source_config.get("label_column", "label"))
    # Importing the direct adapters themselves does not require PyArrow.  Do
    # not treat an import failure as permission to fall back to the HF builder
    # for a matching recovery contract: a missing direct-reader dependency
    # must fail before a large Arrow cache can be materialized.
    from .hf_parquet_recovery import (
        load_hf_parquet_recovery_source,
    )
    from .hf_parquet_recovery import (
        recovery_source_for_config as hf_recovery_source_for_config,
    )
    from .places365_recovery import (
        load_places365_recovery_source,
        recovery_source_for_config,
    )

    places365_source = recovery_source_for_config(source_config)
    hf_recovery_source = hf_recovery_source_for_config(source_config)
    if places365_source is not None and hf_recovery_source is not None:
        raise RuntimeError("A composite source matches multiple direct recovery contracts")
    if places365_source is not None:
        source = load_places365_recovery_source(source_config)
    elif hf_recovery_source is not None:
        source = load_hf_parquet_recovery_source(source_config)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Composite datasets require the 'datasets' package") from exc
    kwargs: dict[str, Any] = {
        "path": source_id,
        "revision": source_revision,
        "split": source_split,
        "keep_in_memory": keep_in_memory,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    if token is not None:
        kwargs["token"] = token
    if places365_source is None and hf_recovery_source is None:
        source = load_dataset(**kwargs)
    selection = source_config.get("selection")
    if selection is not None and not isinstance(selection, Mapping):
        raise TypeError(f"Composite selection for {split!r} must be a mapping")
    indices = _selected_indices(
        source,
        source_revision=source_revision,
        source_split=source_split,
        source_label_column=label_column,
        selection=selection,
    )
    label_names: tuple[str, ...] | None = None
    try:
        names = source.features[label_column].names
        if names is not None:
            label_names = tuple(str(name) for name in names)
    except (AttributeError, KeyError, TypeError):
        pass
    return CompositeDatasetView(source, indices, image_column, label_column, label_names)


__all__ = ["CompositeDatasetView", "load_composite_split"]
