"""Paired fixed-q bootstrap for NOISE Effective Robustness comparisons.

The calculation consumes existing Phase 2 prediction shards only.  It compares
the already selected NOISE-S/NOISE-K prefixes with the exact q=11 NAIVE rule on
the same test images, and refits the original-single-explainer ER curve inside
every class-stratified bootstrap replicate.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from xai_ensemble.core.hashing import file_sha256, object_sha256, stable_seed
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.comparison import holm_adjust
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from .artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    phase2_artifact_root,
)
from .config import SimpleExperiment
from .noise_prefix.artifacts import completed_evaluation_manifest, output_store
from .noise_prefix.config import RULE_IDS, NoisePrefixExperiment

ER_NOISE_BOOTSTRAP_SCHEMA = "simple-effective-robustness-noise-bootstrap-v1"
ER_NOISE_BOOTSTRAP_POLICY = "fixed_selected_q_class_stratified_paired_bootstrap_refit_er_curve"
NOISE_KEYS = ("g", "p", "s", "a")
CONDITION_KEYS = ("clean", *NOISE_KEYS)
GEOMETRY_ORDER = ("spearman", "kendall")
GEOMETRY_LABELS = {"spearman": "NOISE-S", "kendall": "NOISE-K"}
CONDITION_LABELS = {
    "g": "gaussian",
    "p": "salt_pepper",
    "s": "speckle",
    "a": "adversarial",
}
_METRIC_COUNT = len(QUALITY_METRICS)
_RULE_COUNT = len(RULE_IDS)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactError(f"{context} must be a sequence")
    return value


def _condition_key(kind: str, *, kwargs: Mapping[str, Any] | None = None) -> str:
    if kind == "factory":
        kind = str((kwargs or {}).get("kind", ""))
    mapping = {
        "clean": "clean",
        "gaussian": "g",
        "salt_pepper": "p",
        "speckle": "s",
        "adversarial": "a",
    }
    try:
        return mapping[kind]
    except KeyError as error:
        raise ArtifactError(f"Unsupported NOISE bootstrap condition kind {kind!r}") from error


def _metric_contributions(
    *,
    labels: NDArray[np.int64],
    unmasked_predictions: NDArray[np.int64],
    removed_predictions: NDArray[np.int64],
    retained_predictions: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return one F/Fbar/C/Cbar contribution vector for every image."""

    correct = unmasked_predictions == labels
    values = np.stack(
        (
            correct.astype(np.float64) - (removed_predictions == labels).astype(np.float64),
            correct.astype(np.float64) - (retained_predictions == labels).astype(np.float64),
            (removed_predictions != unmasked_predictions).astype(np.float64),
            (retained_predictions != unmasked_predictions).astype(np.float64),
        ),
        axis=1,
    )
    if values.shape != (labels.size, _METRIC_COUNT):
        raise AssertionError("unexpected per-sample metric shape")
    return values


@dataclass(frozen=True, slots=True)
class _ConditionValues:
    indices: NDArray[np.int64]
    labels: NDArray[np.int64]
    targets: NDArray[np.int64]
    unmasked_predictions: NDArray[np.int64]
    values: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class CellMetricBank:
    """Aligned per-image contributions for one dataset/model NOISE comparison."""

    cell: str
    dataset: str
    model: str
    labels: NDArray[np.int64]
    reference: NDArray[np.float64]
    naive: NDArray[np.float64]
    noise: NDArray[np.float64]
    q_by_geometry: Mapping[str, int]
    condition_transition_audit: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        sample_count = self.labels.size
        if self.reference.shape != (len(CONDITION_KEYS), sample_count, 11, _METRIC_COUNT):
            raise ValueError(
                f"invalid ER reference bank shape for {self.cell}: {self.reference.shape}"
            )
        if self.naive.shape != (len(CONDITION_KEYS), sample_count, _RULE_COUNT, _METRIC_COUNT):
            raise ValueError(f"invalid q=11 NAIVE bank shape for {self.cell}: {self.naive.shape}")
        if self.noise.shape != (
            len(GEOMETRY_ORDER),
            len(CONDITION_KEYS),
            sample_count,
            _RULE_COUNT,
            _METRIC_COUNT,
        ):
            raise ValueError(f"invalid NOISE bank shape for {self.cell}: {self.noise.shape}")
        if set(self.q_by_geometry) != set(GEOMETRY_ORDER):
            raise ValueError(f"NOISE q geometry coverage is invalid for {self.cell}")
        if set(self.condition_transition_audit) != set(CONDITION_KEYS):
            raise ValueError(f"NOISE condition audit coverage is invalid for {self.cell}")


@dataclass(frozen=True, slots=True)
class BootstrapGains:
    naive_er: NDArray[np.float64]
    noise_er: NDArray[np.float64]
    gain: NDArray[np.float64]
    replicates: NDArray[np.float64]


def _oriented(values: NDArray[np.float64], metric_index: int) -> NDArray[np.float64]:
    metric = QUALITY_METRICS[metric_index]
    return values if DEFAULT_METRIC_DIRECTIONS[metric] == "max" else -values


def _fit_expected(
    reference_clean: NDArray[np.float64],
    reference_perturbed: NDArray[np.float64],
    candidate_clean: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized nonnegative-slope OLS prediction for B bootstrap replicates."""

    if reference_clean.ndim != 2 or reference_perturbed.shape != reference_clean.shape:
        raise ValueError("ER reference arrays must have shape (bootstrap, explainers)")
    if candidate_clean.ndim != 2 or candidate_clean.shape[0] != reference_clean.shape[0]:
        raise ValueError("candidate clean values must align with ER reference replicates")
    mean_x = reference_clean.mean(axis=1, keepdims=True)
    mean_y = reference_perturbed.mean(axis=1, keepdims=True)
    centered_x = reference_clean - mean_x
    centered_y = reference_perturbed - mean_y
    sum_xx = np.sum(centered_x * centered_x, axis=1, keepdims=True)
    sum_xy = np.sum(centered_x * centered_y, axis=1, keepdims=True)
    unconstrained = np.divide(
        sum_xy,
        sum_xx,
        out=np.zeros_like(sum_xy),
        where=sum_xx > 1e-15,
    )
    slope = np.maximum(0.0, unconstrained)
    intercept = mean_y - slope * mean_x
    return intercept + slope * candidate_clean


def er_values_from_means(
    reference: NDArray[np.float64],
    naive: NDArray[np.float64],
    noise: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return q=11 ER, NOISE ER, and NOISE-minus-NAIVE ER from sample means.

    Shapes are ``reference=(B, 5, 11, 4)``, ``naive=(B, 5, 5, 4)``, and
    ``noise=(B, 2, 5, 5, 4)``.  Conditions are clean, g, p, s, a.
    """

    if reference.ndim != 4 or reference.shape[1:] != (len(CONDITION_KEYS), 11, _METRIC_COUNT):
        raise ValueError("reference means have an invalid shape")
    batch = reference.shape[0]
    if naive.shape != (batch, len(CONDITION_KEYS), _RULE_COUNT, _METRIC_COUNT):
        raise ValueError("q=11 NAIVE means have an invalid shape")
    if noise.shape != (
        batch,
        len(GEOMETRY_ORDER),
        len(CONDITION_KEYS),
        _RULE_COUNT,
        _METRIC_COUNT,
    ):
        raise ValueError("NOISE means have an invalid shape")

    naive_er = np.empty((batch, len(NOISE_KEYS), _RULE_COUNT, _METRIC_COUNT), dtype=np.float64)
    noise_er = np.empty(
        (batch, len(GEOMETRY_ORDER), len(NOISE_KEYS), _RULE_COUNT, _METRIC_COUNT),
        dtype=np.float64,
    )
    for condition_index, _condition in enumerate(NOISE_KEYS, start=1):
        for metric_index in range(_METRIC_COUNT):
            reference_clean = _oriented(reference[:, 0, :, metric_index], metric_index)
            reference_perturbed = _oriented(
                reference[:, condition_index, :, metric_index], metric_index
            )
            naive_clean = _oriented(naive[:, 0, :, metric_index], metric_index)
            naive_perturbed = _oriented(naive[:, condition_index, :, metric_index], metric_index)
            naive_er[:, condition_index - 1, :, metric_index] = naive_perturbed - _fit_expected(
                reference_clean,
                reference_perturbed,
                naive_clean,
            )
            for geometry_index in range(len(GEOMETRY_ORDER)):
                noise_clean = _oriented(noise[:, geometry_index, 0, :, metric_index], metric_index)
                noise_perturbed = _oriented(
                    noise[:, geometry_index, condition_index, :, metric_index], metric_index
                )
                noise_er[:, geometry_index, condition_index - 1, :, metric_index] = (
                    noise_perturbed
                    - _fit_expected(
                        reference_clean,
                        reference_perturbed,
                        noise_clean,
                    )
                )
    return naive_er, noise_er, noise_er - naive_er[:, None, :, :, :]


def _flat_bank(bank: CellMetricBank) -> NDArray[np.float64]:
    sample_count = bank.labels.size
    reference = bank.reference.transpose(1, 0, 2, 3).reshape(sample_count, -1)
    naive = bank.naive.transpose(1, 0, 2, 3).reshape(sample_count, -1)
    noise = bank.noise.transpose(2, 0, 1, 3, 4).reshape(sample_count, -1)
    return np.concatenate((reference, naive, noise), axis=1)


def _unpack_means(
    values: NDArray[np.float64],
    *,
    reference_method_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if values.ndim != 2:
        raise ValueError("bootstrap means must be a matrix")
    batch = values.shape[0]
    reference_size = len(CONDITION_KEYS) * reference_method_count * _METRIC_COUNT
    naive_size = len(CONDITION_KEYS) * _RULE_COUNT * _METRIC_COUNT
    noise_size = len(GEOMETRY_ORDER) * naive_size
    if values.shape[1] != reference_size + naive_size + noise_size:
        raise ValueError("bootstrap mean width does not match ER bank layout")
    position = 0
    reference = values[:, position : position + reference_size].reshape(
        batch, len(CONDITION_KEYS), reference_method_count, _METRIC_COUNT
    )
    position += reference_size
    naive = values[:, position : position + naive_size].reshape(
        batch, len(CONDITION_KEYS), _RULE_COUNT, _METRIC_COUNT
    )
    position += naive_size
    noise = values[:, position:].reshape(
        batch,
        len(GEOMETRY_ORDER),
        len(CONDITION_KEYS),
        _RULE_COUNT,
        _METRIC_COUNT,
    )
    return reference, naive, noise


def _stratified_multiplicities(
    labels: NDArray[np.int64],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> NDArray[np.uint16]:
    """Draw class-stratified bootstrap multiplicities for one batch."""

    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("bootstrap labels must be a non-empty vector")
    counts = np.zeros((batch_size, labels.size), dtype=np.uint16)
    rows = np.arange(batch_size, dtype=np.intp)[:, None]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label).astype(np.intp, copy=False)
        sampled = rng.choice(members, size=(batch_size, members.size), replace=True)
        np.add.at(counts, (np.broadcast_to(rows, sampled.shape), sampled), 1)
    if not np.all(counts.sum(axis=1) == labels.size):
        raise AssertionError("stratified bootstrap changed the sample count")
    return counts


def bootstrap_er_gains(
    bank: CellMetricBank,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 32,
) -> BootstrapGains:
    """Bootstrap fixed-q NOISE-minus-q=11 ER while refitting every curve."""

    if replicates <= 0 or batch_size <= 0:
        raise ValueError("bootstrap replicate and batch counts must be positive")
    values = _flat_bank(bank)
    full_means = values.mean(axis=0, keepdims=True)
    point_reference, point_naive, point_noise = _unpack_means(
        full_means,
        reference_method_count=bank.reference.shape[2],
    )
    point_naive_er, point_noise_er, point_gain = er_values_from_means(
        point_reference,
        point_naive,
        point_noise,
    )
    result = np.empty(
        (
            replicates,
            len(GEOMETRY_ORDER),
            len(NOISE_KEYS),
            _RULE_COUNT,
            _METRIC_COUNT,
        ),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        current = min(batch_size, replicates - start)
        counts = _stratified_multiplicities(bank.labels, batch_size=current, rng=rng)
        means = counts.astype(np.float64) @ values / float(bank.labels.size)
        current_reference, current_naive, current_noise = _unpack_means(
            means,
            reference_method_count=bank.reference.shape[2],
        )
        _, _, gains = er_values_from_means(current_reference, current_naive, current_noise)
        result[start : start + current] = gains
    return BootstrapGains(
        naive_er=point_naive_er[0],
        noise_er=point_noise_er[0],
        gain=point_gain[0],
        replicates=result,
    )


def _field_labels(record: Mapping[str, Any], *, context: str) -> Mapping[str, str]:
    values = _mapping(record.get("rule_labels"), context=f"{context}/rule_labels")
    inverse = {str(label): str(field) for field, label in values.items()}
    if len(inverse) != len(values):
        raise ArtifactError(f"{context} has duplicate rule labels")
    return inverse


def _load_condition_values(
    *,
    store: ArtifactStore,
    manifest: Mapping[str, Any],
    required_labels: Sequence[str],
    work_root: Path,
    context: str,
) -> _ConditionValues:
    """Materialize one artifact shard at a time and derive per-image metrics."""

    from safetensors.torch import load_file

    shards = _sequence(manifest.get("shards"), context=f"{context}/shards")
    if not shards:
        raise ArtifactError(f"{context} has no shards")
    pieces: dict[str, list[NDArray[np.float64]]] = {label: [] for label in required_labels}
    index_parts: list[NDArray[np.int64]] = []
    label_parts: list[NDArray[np.int64]] = []
    target_parts: list[NDArray[np.int64]] = []
    prediction_parts: list[NDArray[np.int64]] = []
    expected_label_fields: Mapping[str, str] | None = None
    for shard_value in sorted(
        (_mapping(value, context=f"{context}/shard") for value in shards),
        key=lambda value: int(value["shard_index"]),
    ):
        field_by_label = _field_labels(shard_value, context=context)
        missing = sorted(set(required_labels).difference(field_by_label))
        if missing:
            raise ArtifactError(f"{context} shard lacks required rules: {missing}")
        current = {label: field_by_label[label] for label in required_labels}
        if expected_label_fields is None:
            expected_label_fields = current
        elif current != expected_label_fields:
            raise ArtifactError(f"{context} rule field layout changed across shards")
        payload = _mapping(shard_value.get("payload"), context=f"{context}/payload")
        relative_path = str(payload.get("relative_path", ""))
        sha256 = str(payload.get("sha256", ""))
        if not relative_path or len(sha256) != 64:
            raise ArtifactError(f"{context} has an invalid shard payload identity")
        path = store.materialize(
            relative_path,
            work_root
            / f"{context.replace('/', '-')}-{int(shard_value['shard_index']):05d}.safetensors",
            expected_sha256=sha256,
        )
        try:
            fields = load_file(str(path), device="cpu")
            required_fixed = ("indices", "labels", "targets", "unmasked_predictions")
            if any(name not in fields for name in required_fixed):
                raise ArtifactError(f"{context} shard lacks a fixed prediction field")
            indices = fields["indices"].numpy().astype(np.int64, copy=False)
            labels = fields["labels"].numpy().astype(np.int64, copy=False)
            targets = fields["targets"].numpy().astype(np.int64, copy=False)
            unmasked = fields["unmasked_predictions"].numpy().astype(np.int64, copy=False)
            count = int(shard_value["count"])
            if any(value.shape != (count,) for value in (indices, labels, targets, unmasked)):
                raise ArtifactError(f"{context} fixed prediction shapes contradict shard count")
            index_parts.append(indices.copy())
            label_parts.append(labels.copy())
            target_parts.append(targets.copy())
            prediction_parts.append(unmasked.copy())
            for label, field_id in current.items():
                removed_name = f"removed_predictions__{field_id}"
                retained_name = f"retained_predictions__{field_id}"
                if removed_name not in fields or retained_name not in fields:
                    raise ArtifactError(f"{context} shard lacks predictions for {label}")
                removed = fields[removed_name].numpy().astype(np.int64, copy=False)
                retained = fields[retained_name].numpy().astype(np.int64, copy=False)
                if removed.shape != (count,) or retained.shape != (count,):
                    raise ArtifactError(f"{context} metric prediction shape is invalid for {label}")
                pieces[label].append(
                    _metric_contributions(
                        labels=labels,
                        unmasked_predictions=unmasked,
                        removed_predictions=removed,
                        retained_predictions=retained,
                    )
                )
        finally:
            path.unlink(missing_ok=True)
    indices = np.concatenate(index_parts)
    labels = np.concatenate(label_parts)
    targets = np.concatenate(target_parts)
    predictions = np.concatenate(prediction_parts)
    if len(np.unique(indices)) != indices.size:
        raise ArtifactError(f"{context} has duplicate sample indices")
    order = np.argsort(indices, kind="stable")
    return _ConditionValues(
        indices=indices[order],
        labels=labels[order],
        targets=targets[order],
        unmasked_predictions=predictions[order],
        values={label: np.concatenate(parts, axis=0)[order] for label, parts in pieces.items()},
    )


def _assert_aligned(
    first: _ConditionValues,
    second: _ConditionValues,
    *,
    context: str,
) -> None:
    for name in ("indices", "labels", "targets", "unmasked_predictions"):
        if not np.array_equal(getattr(first, name), getattr(second, name)):
            raise ArtifactError(f"{context} differs in {name}")


def _assert_condition_sample_identity(
    reference: _ConditionValues,
    current: _ConditionValues,
    *,
    context: str,
) -> None:
    # ER pairs the same images across conditions, so provider-row identity and
    # true labels must remain invariant here.  Prediction/target transitions
    # are audited separately because they are condition-specific fields; the
    # complete same-condition NAIVE/NOISE alignment above still checks both
    # byte-for-byte.
    for name in ("indices", "labels"):
        if not np.array_equal(getattr(reference, name), getattr(current, name)):
            raise ArtifactError(f"{context} differs in {name}")


def _condition_transition_audit(
    clean: _ConditionValues,
    current: _ConditionValues,
) -> dict[str, int]:
    """Record mutable prediction fields without treating them as sample identity."""

    return {
        "target_changes_from_clean": int(np.count_nonzero(clean.targets != current.targets)),
        "unmasked_prediction_changes_from_clean": int(
            np.count_nonzero(clean.unmasked_predictions != current.unmasked_predictions)
        ),
    }


def _completed_base_manifest(
    *,
    experiment: SimpleExperiment,
    task: Any,
    store: ArtifactStore,
) -> Mapping[str, Any]:
    root = phase2_artifact_root(task)
    manifest = completed_manifest(
        store,
        root,
        expected_task_digest=task.digest,
        expected_schema_version=PHASE2_SCHEMA_VERSION,
    )
    if manifest is None:
        raise FileNotFoundError(f"Missing completed q=11 NAIVE Phase 2 artifact: {root}")
    return manifest


def _noise_rows(
    summary: Mapping[str, Any],
) -> Mapping[tuple[str, str, str], Mapping[str, Any]]:
    if summary.get("schema") != "simple-effective-robustness-v1":
        raise ArtifactError("NOISE ER input has an unsupported schema")
    rows = {}
    for row_value in _sequence(summary.get("cells"), context="NOISE ER rows"):
        row = _mapping(row_value, context="NOISE ER row")
        geometry = row.get("geometry")
        method = str(row.get("method", ""))
        if geometry not in GEOMETRY_ORDER or method not in RULE_IDS:
            continue
        key = (str(row.get("cell", "")), str(geometry), method)
        if not key[0] or key in rows:
            raise ArtifactError(f"NOISE ER rows have a duplicate or invalid key: {key}")
        if row.get("q") is None or int(row["q"]) < 2:
            raise ArtifactError(f"NOISE ER row has no selected q: {key}")
        rows[key] = row
    if not rows:
        raise ArtifactError("NOISE ER summary contains no NOISE-S/NOISE-K rule rows")
    return rows


def _naive_rows(summary: Mapping[str, Any]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    if summary.get("schema") != "simple-effective-robustness-v1":
        raise ArtifactError("NAIVE ER input has an unsupported schema")
    rows = {}
    for row_value in _sequence(summary.get("cells"), context="NAIVE ER rows"):
        row = _mapping(row_value, context="NAIVE ER row")
        method = str(row.get("method", ""))
        if method not in RULE_IDS:
            continue
        key = (str(row.get("cell", "")), method)
        if not key[0] or key in rows:
            raise ArtifactError(f"NAIVE ER rows have a duplicate or invalid key: {key}")
        rows[key] = row
    if not rows:
        raise ArtifactError("NAIVE ER summary contains no q=11 aggregate rule rows")
    return rows


def _raw_value(row: Mapping[str, Any], *, condition: str, metric: str) -> float:
    if condition == "clean":
        source = _mapping(row.get("quality"), context="ER candidate clean quality")
    else:
        source = _mapping(row.get("perturbed_quality"), context="ER candidate perturbed quality")
        source = _mapping(source.get(condition), context=f"ER candidate {condition} quality")
    value = float(source[metric])
    if not math.isfinite(value):
        raise ArtifactError("ER candidate has a non-finite quality value")
    return value


def _recorded_er(row: Mapping[str, Any], *, condition: str, metric: str) -> float:
    endpoints = _mapping(row.get("effective_robustness"), context="ER candidate endpoints")
    values = _mapping(endpoints.get(condition), context=f"ER candidate {condition} endpoints")
    value = float(_mapping(values.get(metric), context=f"ER candidate {metric} endpoint")["value"])
    if not math.isfinite(value):
        raise ArtifactError("ER candidate has a non-finite ER value")
    return value


def _verify_point_values(
    bank: CellMetricBank,
    *,
    naive_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    noise_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    values = _flat_bank(bank).mean(axis=0, keepdims=True)
    reference, naive, noise = _unpack_means(values, reference_method_count=bank.reference.shape[2])
    naive_er, noise_er, _gain = er_values_from_means(reference, naive, noise)
    for rule_index, rule in enumerate(RULE_IDS):
        naive_row = naive_rows[(bank.cell, rule)]
        for metric_index, metric in enumerate(QUALITY_METRICS):
            observed = float(naive[0, 0, rule_index, metric_index])
            expected = _raw_value(naive_row, condition="clean", metric=metric)
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ArtifactError(
                    f"q=11 NAIVE clean quality changed for {bank.cell}/{rule}/{metric}"
                )
        for condition_index, condition in enumerate(NOISE_KEYS, start=1):
            for metric_index, metric in enumerate(QUALITY_METRICS):
                observed = float(naive[0, condition_index, rule_index, metric_index])
                expected = _raw_value(naive_row, condition=condition, metric=metric)
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ArtifactError(
                        f"q=11 NAIVE perturbed quality changed for {bank.cell}/{rule}/{condition}/{metric}"
                    )
                expected_er = _recorded_er(naive_row, condition=condition, metric=metric)
                if not math.isclose(
                    float(naive_er[0, condition_index - 1, rule_index, metric_index]),
                    expected_er,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ArtifactError(
                        f"q=11 NAIVE ER changed for {bank.cell}/{rule}/{condition}/{metric}"
                    )
    for geometry_index, geometry in enumerate(GEOMETRY_ORDER):
        for rule_index, rule in enumerate(RULE_IDS):
            noise_row = noise_rows[(bank.cell, geometry, rule)]
            for metric_index, metric in enumerate(QUALITY_METRICS):
                observed = float(noise[0, geometry_index, 0, rule_index, metric_index])
                expected = _raw_value(noise_row, condition="clean", metric=metric)
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ArtifactError(
                        f"NOISE clean quality changed for {bank.cell}/{geometry}/{rule}/{metric}"
                    )
            for condition_index, condition in enumerate(NOISE_KEYS, start=1):
                for metric_index, metric in enumerate(QUALITY_METRICS):
                    observed = float(
                        noise[0, geometry_index, condition_index, rule_index, metric_index]
                    )
                    expected = _raw_value(noise_row, condition=condition, metric=metric)
                    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                        raise ArtifactError(
                            f"NOISE perturbed quality changed for {bank.cell}/{geometry}/{rule}/{condition}/{metric}"
                        )
                    expected_er = _recorded_er(noise_row, condition=condition, metric=metric)
                    if not math.isclose(
                        float(
                            noise_er[
                                0, geometry_index, condition_index - 1, rule_index, metric_index
                            ]
                        ),
                        expected_er,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ArtifactError(
                            f"NOISE ER changed for {bank.cell}/{geometry}/{rule}/{condition}/{metric}"
                        )


def _build_cell_bank(
    *,
    base_experiment: SimpleExperiment,
    prefix_experiment: NoisePrefixExperiment,
    cell: Any,
    q_by_geometry: Mapping[str, int],
    work_root: Path,
) -> CellMetricBank:
    base_store = ArtifactStore(base_experiment)
    prefix_store = output_store(prefix_experiment)
    prefix_tasks = {
        _condition_key(str(task.condition.kind), kwargs=task.condition.kwargs): task
        for task in prefix_experiment.evaluation_tasks()
        if task.cell.cell_id == cell.cell_id
    }
    if set(prefix_tasks) != set(CONDITION_KEYS):
        raise ArtifactError(f"NOISE prefix tasks are incomplete for {cell.cell_id}")
    reference_labels = tuple(f"single__{method}" for method in cell.methods)
    if len(reference_labels) != 11 or len(set(reference_labels)) != 11:
        raise ArtifactError(f"NOISE ER expects exactly eleven source methods for {cell.cell_id}")
    base_labels = (*RULE_IDS, *reference_labels)
    base_by_condition: dict[str, _ConditionValues] = {}
    prefix_by_condition: dict[str, _ConditionValues] = {}
    selected_prefix_labels = {
        geometry: tuple(f"q{q_by_geometry[geometry]:02d}__{rule}" for rule in RULE_IDS)
        for geometry in GEOMETRY_ORDER
    }
    prefix_labels = tuple(
        sorted({label for labels in selected_prefix_labels.values() for label in labels})
    )
    for condition in CONDITION_KEYS:
        prefix_task = prefix_tasks[condition]
        base_task = prefix_experiment.base_phase2_task(cell, prefix_task.condition.condition_id)
        base_manifest = _completed_base_manifest(
            experiment=base_experiment,
            task=base_task,
            store=base_store,
        )
        prefix_manifest = completed_evaluation_manifest(
            prefix_experiment,
            prefix_task,
            store=prefix_store,
        )
        if prefix_manifest is None:
            raise FileNotFoundError(f"NOISE prefix artifact is incomplete: {prefix_task.task_id}")
        base_values = _load_condition_values(
            store=base_store,
            manifest=base_manifest,
            required_labels=base_labels,
            work_root=work_root,
            context=f"{cell.cell_id}-{condition}-naive",
        )
        prefix_values = _load_condition_values(
            store=prefix_store,
            manifest=prefix_manifest,
            required_labels=prefix_labels,
            work_root=work_root,
            context=f"{cell.cell_id}-{condition}-prefix",
        )
        _assert_aligned(
            base_values, prefix_values, context=f"NOISE/q=11 alignment {cell.cell_id}/{condition}"
        )
        base_by_condition[condition] = base_values
        prefix_by_condition[condition] = prefix_values
    clean = base_by_condition["clean"]
    condition_transition_audit = {}
    for condition in CONDITION_KEYS:
        _assert_condition_sample_identity(
            clean,
            base_by_condition[condition],
            context=f"NAIVE identity {cell.cell_id}/{condition}",
        )
        _assert_condition_sample_identity(
            clean,
            prefix_by_condition[condition],
            context=f"NOISE identity {cell.cell_id}/{condition}",
        )
        condition_transition_audit[condition] = _condition_transition_audit(
            clean,
            base_by_condition[condition],
        )
    reference = np.stack(
        [
            np.stack(
                [base_by_condition[condition].values[label] for label in reference_labels], axis=1
            )
            for condition in CONDITION_KEYS
        ],
        axis=0,
    )
    naive = np.stack(
        [
            np.stack([base_by_condition[condition].values[rule] for rule in RULE_IDS], axis=1)
            for condition in CONDITION_KEYS
        ],
        axis=0,
    )
    noise = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            prefix_by_condition[condition].values[label]
                            for label in selected_prefix_labels[geometry]
                        ],
                        axis=1,
                    )
                    for condition in CONDITION_KEYS
                ],
                axis=0,
            )
            for geometry in GEOMETRY_ORDER
        ],
        axis=0,
    )
    return CellMetricBank(
        cell=cell.cell_id,
        dataset=cell.dataset.dataset_id,
        model=cell.reference_model.model_id,
        labels=clean.labels,
        reference=reference,
        naive=naive,
        noise=noise,
        q_by_geometry={geometry: int(q_by_geometry[geometry]) for geometry in GEOMETRY_ORDER},
        condition_transition_audit=condition_transition_audit,
    )


def _one_sided_bootstrap_p(estimate: float, replicates: NDArray[np.float64]) -> tuple[float, float]:
    """Return centered-null positive-gain p value and raw nonpositive mass."""

    if replicates.ndim != 1 or replicates.size == 0:
        raise ValueError("bootstrap replicates must be a non-empty vector")
    centered_null = replicates - estimate
    positive_p = (1 + int(np.count_nonzero(centered_null >= estimate))) / (replicates.size + 1)
    nonpositive = (1 + int(np.count_nonzero(replicates <= 0.0))) / (replicates.size + 1)
    return float(positive_p), float(nonpositive)


def _holm_resolution(
    *,
    endpoint_count: int,
    alpha: float,
    replicates: int,
) -> dict[str, float | int | bool]:
    """Describe whether bootstrap p-value granularity permits Holm rejection."""

    if endpoint_count <= 0 or not 0.0 < alpha < 1.0 or replicates <= 0:
        raise ValueError("Holm resolution inputs must be positive and finite")
    first_holm_threshold = alpha / float(endpoint_count)
    minimum_replicates = max(1, math.ceil(float(endpoint_count) / alpha) - 1)
    minimum_p_value = 1.0 / float(replicates + 1)
    return {
        "endpoint_count": endpoint_count,
        "first_holm_threshold": first_holm_threshold,
        "minimum_attainable_one_sided_p_value": minimum_p_value,
        "minimum_replicates_for_any_holm_rejection": minimum_replicates,
        "sufficient_for_any_holm_rejection": minimum_p_value <= first_holm_threshold,
    }


def _estimate_row(
    *,
    bank: CellMetricBank,
    gains: BootstrapGains,
    geometry_index: int,
    condition_index: int,
    rule_index: int,
    metric_index: int,
    confidence: float,
) -> dict[str, Any]:
    samples = gains.replicates[:, geometry_index, condition_index, rule_index, metric_index]
    gain = float(gains.gain[geometry_index, condition_index, rule_index, metric_index])
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(samples, (alpha / 2.0, 1.0 - alpha / 2.0))
    raw_p, nonpositive_probability = _one_sided_bootstrap_p(gain, samples)
    geometry = GEOMETRY_ORDER[geometry_index]
    condition = NOISE_KEYS[condition_index]
    metric = QUALITY_METRICS[metric_index]
    return {
        "comparison_id": object_sha256(
            {
                "schema": ER_NOISE_BOOTSTRAP_SCHEMA,
                "cell": bank.cell,
                "geometry": geometry,
                "q": bank.q_by_geometry[geometry],
                "rule": RULE_IDS[rule_index],
                "condition": condition,
                "metric": metric,
            }
        ),
        "cell": bank.cell,
        "dataset": bank.dataset,
        "model": bank.model,
        "sample_count": int(bank.labels.size),
        "geometry": geometry,
        "geometry_label": GEOMETRY_LABELS[geometry],
        "q": int(bank.q_by_geometry[geometry]),
        "rule": RULE_IDS[rule_index],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "metric": metric,
        "naive_er": float(gains.naive_er[condition_index, rule_index, metric_index]),
        "noise_er": float(
            gains.noise_er[geometry_index, condition_index, rule_index, metric_index]
        ),
        "gain": gain,
        "standard_error": float(np.std(samples, ddof=1)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": confidence,
        "one_sided_positive_gain_p_value": raw_p,
        "bootstrap_nonpositive_probability": nonpositive_probability,
        "point_outcome": "better" if gain > 1e-12 else "worse" if gain < -1e-12 else "equal",
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "comparison_id",
        "cell",
        "dataset",
        "model",
        "sample_count",
        "geometry",
        "geometry_label",
        "q",
        "rule",
        "condition",
        "condition_label",
        "metric",
        "naive_er",
        "noise_er",
        "gain",
        "standard_error",
        "ci_low",
        "ci_high",
        "confidence",
        "one_sided_positive_gain_p_value",
        "holm_adjusted_p_value",
        "holm_reject_positive_gain_null",
        "bootstrap_nonpositive_probability",
        "point_outcome",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()


def _readme(summary: Mapping[str, Any]) -> str:
    analysis = _mapping(summary["analysis"], context="ER NOISE bootstrap analysis")
    bootstrap = _mapping(summary["bootstrap"], context="ER NOISE bootstrap settings")
    resolution = _mapping(bootstrap["holm_resolution"], context="Holm resolution")
    return "\n".join(
        (
            "# NOISE Effective Robustness vs q=11 NAIVE",
            "",
            "Each row compares a fixed selected NOISE-S or NOISE-K prefix with the exact q=11 NAIVE result using the same dataset/model, aggregation rule, perturbation, and metric.",
            "Every class-stratified paired image bootstrap replicate refits the original-NAIVE-single ER curve before calculating the NOISE-minus-NAIVE difference.",
            "",
            "The q values are held fixed because they were selected on the complete test set.  Therefore this report quantifies conditional post-selection image-sampling uncertainty; it is not a held-out estimate of q-selection generalization.",
            "",
            "`noise_vs_naive_er.csv` contains one-sided positive-gain bootstrap p-values, 95% percentile confidence intervals, and Holm-adjusted decisions across all reported endpoints.",
            "Bootstrap p-value resolution: "
            f"{float(resolution['minimum_attainable_one_sided_p_value']):.8g}; "
            "minimum repetitions for any Holm rejection: "
            f"{int(resolution['minimum_replicates_for_any_holm_rejection'])}.",
            f"Holm-positive wins: {analysis['holm_positive_gain_rejections']}/{analysis['endpoints']}.",
            "",
            f"Result digest: `{summary['result_digest']}`.",
            "",
        )
    )


def write_noise_vs_naive_bootstrap(
    *,
    base_experiment: SimpleExperiment,
    prefix_experiment: NoisePrefixExperiment,
    naive_er_summary_path: str | Path,
    noise_er_summary_path: str | Path,
    output_directory: str | Path,
    bootstrap_replicates: int = 19_999,
    confidence: float = 0.95,
    seed: int = 0,
    bootstrap_batch_size: int = 32,
) -> Mapping[str, Any]:
    """Write a fixed-q, selection-aware-in-label-only NOISE ER comparison report."""

    if bootstrap_replicates <= 0 or bootstrap_batch_size <= 0:
        raise ValueError("bootstrap replicate and batch counts must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    naive_path = Path(naive_er_summary_path).expanduser().resolve()
    noise_path = Path(noise_er_summary_path).expanduser().resolve()
    naive_summary = read_json(naive_path)
    noise_summary = read_json(noise_path)
    if not isinstance(naive_summary, Mapping) or not isinstance(noise_summary, Mapping):
        raise ArtifactError("ER summary inputs must be JSON mappings")
    naive_rows = _naive_rows(naive_summary)
    noise_rows = _noise_rows(noise_summary)
    cells = {cell.cell_id: cell for cell in prefix_experiment.cells()}
    expected = {
        (cell_id, geometry, rule)
        for cell_id in cells
        for geometry in GEOMETRY_ORDER
        for rule in RULE_IDS
    }
    if set(noise_rows) != expected:
        missing = sorted(expected.difference(noise_rows))
        unexpected = sorted(set(noise_rows).difference(expected))
        raise ArtifactError(
            f"NOISE ER row roster differs from prefix experiment: missing={missing}, unexpected={unexpected}"
        )
    if set(naive_rows) != {(cell_id, rule) for cell_id in cells for rule in RULE_IDS}:
        raise ArtifactError("NAIVE ER row roster differs from the prefix experiment")
    endpoint_count = (
        len(cells) * len(GEOMETRY_ORDER) * len(RULE_IDS) * len(NOISE_KEYS) * len(QUALITY_METRICS)
    )
    alpha = 1.0 - confidence
    holm_resolution = _holm_resolution(
        endpoint_count=endpoint_count,
        alpha=alpha,
        replicates=bootstrap_replicates,
    )

    q_by_cell = {}
    for cell_id in cells:
        q_by_geometry = {}
        for geometry in GEOMETRY_ORDER:
            qs = {int(noise_rows[(cell_id, geometry, rule)]["q"]) for rule in RULE_IDS}
            if len(qs) != 1:
                raise ArtifactError(
                    f"NOISE geometry has inconsistent q across rules: {cell_id}/{geometry}"
                )
            q_by_geometry[geometry] = qs.pop()
        q_by_cell[cell_id] = q_by_geometry

    rows = []
    cell_audits = []
    with TemporaryDirectory(prefix="xai-er-noise-bootstrap-", dir="/dev/shm") as directory:
        work_root = Path(directory)
        for cell_id in sorted(cells):
            bank = _build_cell_bank(
                base_experiment=base_experiment,
                prefix_experiment=prefix_experiment,
                cell=cells[cell_id],
                q_by_geometry=q_by_cell[cell_id],
                work_root=work_root / cell_id,
            )
            _verify_point_values(bank, naive_rows=naive_rows, noise_rows=noise_rows)
            cell_seed = stable_seed(seed, ER_NOISE_BOOTSTRAP_SCHEMA, cell_id)
            gains = bootstrap_er_gains(
                bank,
                replicates=bootstrap_replicates,
                seed=cell_seed,
                batch_size=bootstrap_batch_size,
            )
            for geometry_index in range(len(GEOMETRY_ORDER)):
                for condition_index in range(len(NOISE_KEYS)):
                    for rule_index in range(_RULE_COUNT):
                        for metric_index in range(_METRIC_COUNT):
                            rows.append(
                                _estimate_row(
                                    bank=bank,
                                    gains=gains,
                                    geometry_index=geometry_index,
                                    condition_index=condition_index,
                                    rule_index=rule_index,
                                    metric_index=metric_index,
                                    confidence=confidence,
                                )
                            )
            cell_audits.append(
                {
                    "cell": cell_id,
                    "dataset": bank.dataset,
                    "model": bank.model,
                    "sample_count": int(bank.labels.size),
                    "class_count": int(np.unique(bank.labels).size),
                    "q_by_geometry": dict(bank.q_by_geometry),
                    "condition_transition_audit": {
                        condition: dict(values)
                        for condition, values in bank.condition_transition_audit.items()
                    },
                    "bootstrap_seed": cell_seed,
                }
            )

    raw_p_values = {
        str(row["comparison_id"]): float(row["one_sided_positive_gain_p_value"]) for row in rows
    }
    holm = holm_adjust(raw_p_values, alpha=1.0 - confidence)
    for row in rows:
        adjusted = holm[str(row["comparison_id"])]
        row["holm_adjusted_p_value"] = float(adjusted["holm_adjusted_p_value"])
        row["holm_reject_positive_gain_null"] = bool(adjusted["reject_familywise_null"])
    rows.sort(
        key=lambda row: (
            str(row["cell"]),
            GEOMETRY_ORDER.index(str(row["geometry"])),
            RULE_IDS.index(str(row["rule"])),
            NOISE_KEYS.index(str(row["condition"])),
            QUALITY_METRICS.index(str(row["metric"])),
        )
    )
    by_geometry: dict[str, dict[str, Any]] = {}
    for geometry in GEOMETRY_ORDER:
        selected = [row for row in rows if row["geometry"] == geometry]
        by_geometry[geometry] = {
            "endpoints": len(selected),
            "point_better": sum(row["point_outcome"] == "better" for row in selected),
            "point_equal": sum(row["point_outcome"] == "equal" for row in selected),
            "point_worse": sum(row["point_outcome"] == "worse" for row in selected),
            "mean_gain": float(np.mean([float(row["gain"]) for row in selected])),
            "raw_positive_gain_p_le_alpha": sum(
                float(row["one_sided_positive_gain_p_value"]) <= 1.0 - confidence
                for row in selected
            ),
            "holm_positive_gain_rejections": sum(
                bool(row["holm_reject_positive_gain_null"]) for row in selected
            ),
        }
    summary: dict[str, Any] = {
        "schema": ER_NOISE_BOOTSTRAP_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "science": {
            "comparison": "fixed_selected_NOISE_geometry_q_vs_exact_q11_NAIVE_same_rule",
            "er_reference": "original_naive_single_explainers_only_refit_within_each_bootstrap",
            "bootstrap_unit": "image_stratified_by_true_class",
            "bootstrap_pairing": "same_resampled_image_indices_for_NOISE_NAIVE_and_reference_pool",
            "q_selection_policy": "fixed_after_complete_test_set_selection",
            "inference_scope": "conditional_post_selection_image_sampling_not_held_out_q_selection_generalization",
            "multiple_comparison_policy": "Holm_one_sided_positive_ER_gain_across_all_endpoints",
            "additional_model_forwards": 0,
            "attribution_shards_read": 0,
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "confidence": confidence,
            "alpha": alpha,
            "base_seed": seed,
            "batch_size": bootstrap_batch_size,
            "holm_resolution": holm_resolution,
        },
        "inputs": {
            "base_config": str(base_experiment.source_path),
            "base_experiment_digest": base_experiment.digest,
            "noise_prefix_config": str(prefix_experiment.source_path),
            "noise_prefix_digest": prefix_experiment.digest,
            "naive_er_summary": {
                "path": str(naive_path),
                "sha256": file_sha256(naive_path),
                "result_digest": naive_summary.get("result_digest"),
            },
            "noise_er_summary": {
                "path": str(noise_path),
                "sha256": file_sha256(noise_path),
                "result_digest": noise_summary.get("result_digest"),
            },
        },
        "cell_audits": cell_audits,
        "rows": rows,
        "analysis": {
            "endpoints": len(rows),
            "point_better": sum(row["point_outcome"] == "better" for row in rows),
            "point_equal": sum(row["point_outcome"] == "equal" for row in rows),
            "point_worse": sum(row["point_outcome"] == "worse" for row in rows),
            "mean_gain": float(np.mean([float(row["gain"]) for row in rows])),
            "raw_positive_gain_p_le_alpha": sum(
                float(row["one_sided_positive_gain_p_value"]) <= 1.0 - confidence for row in rows
            ),
            "holm_positive_gain_rejections": sum(
                bool(row["holm_reject_positive_gain_null"]) for row in rows
            ),
            "by_geometry": by_geometry,
        },
    }
    summary["result_digest"] = object_sha256(summary)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = atomic_write_text(destination / "noise_vs_naive_er.csv", _csv_text(rows))
    summary_path = atomic_write_json(destination / "summary.json", summary)
    readme_path = atomic_write_text(destination / "README.md", _readme(summary))
    return {
        "status": "complete",
        "result_digest": summary["result_digest"],
        "endpoints": len(rows),
        "summary_json": str(summary_path),
        "comparison_csv": str(csv_path),
        "readme": str(readme_path),
    }


__all__ = [
    "BootstrapGains",
    "CellMetricBank",
    "ER_NOISE_BOOTSTRAP_POLICY",
    "ER_NOISE_BOOTSTRAP_SCHEMA",
    "bootstrap_er_gains",
    "er_values_from_means",
    "write_noise_vs_naive_bootstrap",
]
