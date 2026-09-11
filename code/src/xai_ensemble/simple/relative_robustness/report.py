"""Null-anchored relative robustness from completed Phase-2 prediction shards."""

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

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from ..artifacts import ArtifactError, load_safetensors
from ..effective_robustness_noise import (
    CONDITION_KEYS,
    CONDITION_LABELS,
    GEOMETRY_LABELS,
    GEOMETRY_ORDER,
    NOISE_KEYS,
    _build_cell_bank,
    _condition_key,
)
from ..noise_prefix.config import RULE_IDS
from .artifacts import completed_control_manifest, output_store
from .config import CONTROL_SCHEMA, RandomControlTask, RelativeRobustnessExperiment

RELATIVE_ROBUSTNESS_SCHEMA = "simple-relative-robustness-report-v1"
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


def _metric_contributions(
    *,
    labels: NDArray[np.int64],
    unmasked_predictions: NDArray[np.int64],
    removed_predictions: NDArray[np.int64],
    retained_predictions: NDArray[np.int64],
) -> NDArray[np.float64]:
    correct = unmasked_predictions == labels
    return np.stack(
        (
            correct.astype(np.float64) - (removed_predictions == labels).astype(np.float64),
            correct.astype(np.float64) - (retained_predictions == labels).astype(np.float64),
            (removed_predictions != unmasked_predictions).astype(np.float64),
            (retained_predictions != unmasked_predictions).astype(np.float64),
        ),
        axis=1,
    )


@dataclass(frozen=True, slots=True)
class _ControlConditionValues:
    indices: NDArray[np.int64]
    labels: NDArray[np.int64]
    targets: NDArray[np.int64]
    unmasked_predictions: NDArray[np.int64]
    selected_patch_indices: NDArray[np.int64]
    values: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RandomControlBank:
    cell: str
    dataset: str
    model: str
    indices: NDArray[np.int64]
    labels: NDArray[np.int64]
    random_seed_bank: tuple[int, ...]
    values: NDArray[np.float64]
    condition_transition_audit: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        samples = self.labels.size
        expected = (len(CONDITION_KEYS), samples, len(self.random_seed_bank), _METRIC_COUNT)
        if self.values.shape != expected:
            raise ValueError(
                f"invalid random-control bank shape for {self.cell}: {self.values.shape}"
            )
        if self.indices.shape != (samples,) or samples == 0:
            raise ValueError("random-control sample identity is invalid")
        if np.unique(self.indices).size != samples:
            raise ValueError("random-control indices must be unique")
        if len(set(self.random_seed_bank)) != len(self.random_seed_bank):
            raise ValueError("random-control seed bank must be unique")
        if set(self.condition_transition_audit) != set(CONDITION_KEYS):
            raise ValueError("random-control condition audit coverage is incomplete")


@dataclass(frozen=True, slots=True)
class RelativeValues:
    naive_clean_excess: NDArray[np.float64]
    naive_perturbed_excess: NDArray[np.float64]
    naive_rrel: NDArray[np.float64]
    noise_clean_excess: NDArray[np.float64]
    noise_perturbed_excess: NDArray[np.float64]
    noise_rrel: NDArray[np.float64]
    rrel_gain: NDArray[np.float64]
    perturbed_excess_gain: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RelativeBootstrap:
    point: RelativeValues
    rrel_gain_replicates: NDArray[np.float64]
    perturbed_excess_gain_replicates: NDArray[np.float64]


def _orient(values: NDArray[np.float64]) -> NDArray[np.float64]:
    signs = np.asarray(
        [1.0 if DEFAULT_METRIC_DIRECTIONS[metric] == "max" else -1.0 for metric in QUALITY_METRICS],
        dtype=np.float64,
    )
    return values * signs


def _relative(perturbed: NDArray[np.float64], clean: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return 1 - E_perturbed/E_clean, undefined where E_clean is nonpositive."""

    denominator = np.broadcast_to(clean, perturbed.shape)
    ratio = np.divide(
        perturbed,
        denominator,
        out=np.full(perturbed.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return 1.0 - ratio


def relative_values_from_means(
    random: NDArray[np.float64],
    naive: NDArray[np.float64],
    noise: NDArray[np.float64],
) -> RelativeValues:
    """Calculate R_rel from dataset means with no epsilon fallback.

    Inputs have shapes ``random=(B,5,seeds,4)``, ``naive=(B,5,5,4)``, and
    ``noise=(B,2,5,5,4)``.  The condition order is clean, Gaussian,
    salt-and-pepper, speckle, adversarial.  All excess quantities are
    direction-oriented, so larger means farther above the random baseline.
    """

    if (
        random.ndim != 4
        or random.shape[1] != len(CONDITION_KEYS)
        or random.shape[3] != _METRIC_COUNT
    ):
        raise ValueError("random control means have an invalid shape")
    batch = random.shape[0]
    if random.shape[2] < 2:
        raise ValueError("relative robustness requires at least two random seeds")
    if naive.shape != (batch, len(CONDITION_KEYS), _RULE_COUNT, _METRIC_COUNT):
        raise ValueError("NAIVE means have an invalid shape")
    if noise.shape != (
        batch,
        len(GEOMETRY_ORDER),
        len(CONDITION_KEYS),
        _RULE_COUNT,
        _METRIC_COUNT,
    ):
        raise ValueError("NOISE means have an invalid shape")

    random_oriented = _orient(random).mean(axis=2)
    naive_oriented = _orient(naive)
    noise_oriented = _orient(noise)
    naive_clean_excess = naive_oriented[:, 0] - random_oriented[:, 0, None, :]
    naive_perturbed_excess = naive_oriented[:, 1:] - random_oriented[:, 1:, None, :]
    noise_clean_excess = noise_oriented[:, :, 0] - random_oriented[:, None, 0, None, :]
    noise_perturbed_excess = noise_oriented[:, :, 1:] - random_oriented[:, None, 1:, None, :]
    naive_rrel = _relative(
        naive_perturbed_excess,
        naive_clean_excess[:, None, :, :],
    )
    noise_rrel = _relative(
        noise_perturbed_excess,
        noise_clean_excess[:, :, None, :, :],
    )
    return RelativeValues(
        naive_clean_excess=naive_clean_excess,
        naive_perturbed_excess=naive_perturbed_excess,
        naive_rrel=naive_rrel,
        noise_clean_excess=noise_clean_excess,
        noise_perturbed_excess=noise_perturbed_excess,
        noise_rrel=noise_rrel,
        rrel_gain=naive_rrel[:, None] - noise_rrel,
        perturbed_excess_gain=noise_perturbed_excess - naive_perturbed_excess[:, None],
    )


def _control_manifest_identity(
    experiment: RelativeRobustnessExperiment,
    task: RandomControlTask,
    manifest: Mapping[str, Any],
) -> None:
    expected = {
        "schema": CONTROL_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell_id,
        "dataset": task.prefix_task.cell.dataset.dataset_id,
        "model": task.prefix_task.cell.reference_model.model_id,
        "split": experiment.split,
        "condition": task.prefix_task.condition.condition_id,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "random_seed_bank": list(experiment.random_seed_bank),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Relative random-control manifest identity mismatch: {mismatches}")


def _load_control_condition(
    experiment: RelativeRobustnessExperiment,
    task: RandomControlTask,
    *,
    work_root: Path,
) -> _ControlConditionValues:
    store = output_store(experiment)
    manifest = completed_control_manifest(experiment, task, store=store)
    if manifest is None:
        raise FileNotFoundError(f"Random control is incomplete: {task.task_id}")
    _control_manifest_identity(experiment, task, manifest)
    seed_labels = tuple(
        str(value)
        for value in _sequence(manifest.get("random_seed_labels"), context="random seed labels")
    )
    expected_labels = tuple(
        f"random_seed_{index:02d}" for index in range(len(experiment.random_seed_bank))
    )
    if seed_labels != expected_labels:
        raise ArtifactError("Random-control seed labels differ from the declared seed bank")
    records = _sequence(manifest.get("shards"), context="random-control shards")
    index_parts = []
    label_parts = []
    target_parts = []
    prediction_parts = []
    selected_parts = []
    value_parts: list[list[NDArray[np.float64]]] = [[] for _ in seed_labels]
    patch_count = (224 // experiment.patch_size) ** 2
    for record_value in records:
        record = _mapping(record_value, context="random-control shard record")
        payload = _mapping(record.get("payload"), context="random-control shard payload")
        local = work_root / str(payload["sha256"])
        store.materialize(
            str(payload["relative_path"]), local, expected_sha256=str(payload["sha256"])
        )
        try:
            fields = load_safetensors(local)
            indices = fields["indices"].numpy().astype(np.int64, copy=False)
            labels = fields["labels"].numpy().astype(np.int64, copy=False)
            targets = fields["targets"].numpy().astype(np.int64, copy=False)
            predictions = fields["unmasked_predictions"].numpy().astype(np.int64, copy=False)
            count = indices.size
            if count <= 0 or any(
                array.shape != (count,) for array in (labels, targets, predictions)
            ):
                raise ArtifactError("Random-control fixed tensor shapes are invalid")
            rule_labels = _mapping(record.get("rule_labels"), context="random-control rules")
            if (
                tuple(str(rule_labels.get(f"r{seed:03d}", "")) for seed in range(len(seed_labels)))
                != seed_labels
            ):
                raise ArtifactError("Random-control shard rule labels differ from the seed bank")
            selected_rows = []
            for seed_position, _seed_label in enumerate(seed_labels):
                field_id = f"r{seed_position:03d}"
                selected = (
                    fields[f"top_patch_indices__{field_id}"].numpy().astype(np.int64, copy=False)
                )
                removed = (
                    fields[f"removed_predictions__{field_id}"].numpy().astype(np.int64, copy=False)
                )
                retained = (
                    fields[f"retained_predictions__{field_id}"].numpy().astype(np.int64, copy=False)
                )
                if (
                    selected.shape != (count, experiment.k)
                    or removed.shape != (count,)
                    or retained.shape != (count,)
                ):
                    raise ArtifactError(
                        "Random-control prediction or selected-patch shape is invalid"
                    )
                if np.any(selected < 0) or np.any(selected >= patch_count):
                    raise ArtifactError("Random-control selected patches are outside the p=16 grid")
                if np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
                    raise ArtifactError("Random-control top-k patches are not unique")
                selected_rows.append(selected)
                value_parts[seed_position].append(
                    _metric_contributions(
                        labels=labels,
                        unmasked_predictions=predictions,
                        removed_predictions=removed,
                        retained_predictions=retained,
                    )
                )
            index_parts.append(indices)
            label_parts.append(labels)
            target_parts.append(targets)
            prediction_parts.append(predictions)
            selected_parts.append(np.stack(selected_rows, axis=1))
        finally:
            local.unlink(missing_ok=True)
    indices = np.concatenate(index_parts)
    labels = np.concatenate(label_parts)
    targets = np.concatenate(target_parts)
    predictions = np.concatenate(prediction_parts)
    selected = np.concatenate(selected_parts, axis=0)
    values = np.stack([np.concatenate(parts, axis=0) for parts in value_parts], axis=1)
    if np.unique(indices).size != indices.size:
        raise ArtifactError("Random-control artifact has duplicate sample indices")
    order = np.argsort(indices, kind="stable")
    values = values[order]
    selected = selected[order]
    result = _ControlConditionValues(
        indices=indices[order],
        labels=labels[order],
        targets=targets[order],
        unmasked_predictions=predictions[order],
        selected_patch_indices=selected,
        values=values,
    )
    metrics = _mapping(manifest.get("metrics"), context="random-control summary metrics")
    for position, label in enumerate(seed_labels):
        summary = _mapping(metrics.get(label), context=f"random-control metrics/{label}")
        for metric_index, metric in enumerate(QUALITY_METRICS):
            observed = float(result.values[:, position, metric_index].mean())
            if not math.isclose(observed, float(summary[metric]), rel_tol=0.0, abs_tol=1e-12):
                raise ArtifactError(
                    f"Random-control metrics disagree with shards: {label}/{metric}"
                )
    return result


def _load_random_control_bank(
    experiment: RelativeRobustnessExperiment,
    *,
    cell_id: str,
    work_root: Path,
) -> RandomControlBank:
    tasks = {
        _condition_key(
            str(task.prefix_task.condition.kind), kwargs=task.prefix_task.condition.kwargs
        ): task
        for task in experiment.control_tasks()
        if task.cell_id == cell_id
    }
    if set(tasks) != set(CONDITION_KEYS):
        raise ArtifactError(f"Random controls do not cover every condition for {cell_id}")
    by_condition = {
        condition: _load_control_condition(
            experiment,
            tasks[condition],
            work_root=work_root / condition,
        )
        for condition in CONDITION_KEYS
    }
    clean = by_condition["clean"]
    transition_audit = {}
    for condition, current in by_condition.items():
        for name in ("indices", "labels"):
            if not np.array_equal(getattr(clean, name), getattr(current, name)):
                raise ArtifactError(f"Random-control {cell_id}/{condition} differs in {name}")
        if not np.array_equal(clean.selected_patch_indices, current.selected_patch_indices):
            raise ArtifactError(
                f"Random-control masks are not shared across clean and {condition} for {cell_id}"
            )
        transition_audit[condition] = {
            "target_changes_from_clean": int(np.count_nonzero(clean.targets != current.targets)),
            "unmasked_prediction_changes_from_clean": int(
                np.count_nonzero(clean.unmasked_predictions != current.unmasked_predictions)
            ),
        }
    cell = tasks["clean"].prefix_task.cell
    return RandomControlBank(
        cell=cell_id,
        dataset=cell.dataset.dataset_id,
        model=cell.reference_model.model_id,
        indices=clean.indices,
        labels=clean.labels,
        random_seed_bank=experiment.random_seed_bank,
        values=np.stack([by_condition[key].values for key in CONDITION_KEYS], axis=0),
        condition_transition_audit=transition_audit,
    )


def _flat_bank(
    random: RandomControlBank,
    naive: NDArray[np.float64],
    noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    samples = random.labels.size
    random_flat = random.values.transpose(1, 0, 2, 3).reshape(samples, -1)
    naive_flat = naive.transpose(1, 0, 2, 3).reshape(samples, -1)
    noise_flat = noise.transpose(2, 0, 1, 3, 4).reshape(samples, -1)
    return np.concatenate((random_flat, naive_flat, noise_flat), axis=1)


def _unpack_means(
    values: NDArray[np.float64],
    *,
    random_seed_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if values.ndim != 2:
        raise ValueError("bootstrap means must be a matrix")
    batch = values.shape[0]
    random_size = len(CONDITION_KEYS) * random_seed_count * _METRIC_COUNT
    naive_size = len(CONDITION_KEYS) * _RULE_COUNT * _METRIC_COUNT
    noise_size = len(GEOMETRY_ORDER) * naive_size
    if values.shape[1] != random_size + naive_size + noise_size:
        raise ValueError("relative robustness bootstrap width is invalid")
    position = 0
    random = values[:, position : position + random_size].reshape(
        batch, len(CONDITION_KEYS), random_seed_count, _METRIC_COUNT
    )
    position += random_size
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
    return random, naive, noise


def _stratified_multiplicities(
    labels: NDArray[np.int64],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> NDArray[np.uint16]:
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


def bootstrap_relative_values(
    *,
    random: RandomControlBank,
    naive: NDArray[np.float64],
    noise: NDArray[np.float64],
    replicates: int,
    seed: int,
    batch_size: int,
) -> RelativeBootstrap:
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("bootstrap counts must be positive")
    flat = _flat_bank(random, naive, noise)
    point_random, point_naive, point_noise = _unpack_means(
        flat.mean(axis=0, keepdims=True),
        random_seed_count=len(random.random_seed_bank),
    )
    point = relative_values_from_means(point_random, point_naive, point_noise)
    rrel_gain = np.empty(
        (replicates, len(GEOMETRY_ORDER), len(NOISE_KEYS), _RULE_COUNT, _METRIC_COUNT),
        dtype=np.float64,
    )
    excess_gain = np.empty_like(rrel_gain)
    generator = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        multiplicities = _stratified_multiplicities(
            random.labels,
            batch_size=stop - start,
            rng=generator,
        )
        means = multiplicities.astype(np.float64) @ flat / float(random.labels.size)
        current_random, current_naive, current_noise = _unpack_means(
            means,
            random_seed_count=len(random.random_seed_bank),
        )
        values = relative_values_from_means(current_random, current_naive, current_noise)
        rrel_gain[start:stop] = values.rrel_gain
        excess_gain[start:stop] = values.perturbed_excess_gain
    return RelativeBootstrap(
        point=point,
        rrel_gain_replicates=rrel_gain,
        perturbed_excess_gain_replicates=excess_gain,
    )


def _selected_q(
    experiment: RelativeRobustnessExperiment,
) -> Mapping[str, Mapping[str, int]]:
    summary = read_json(experiment.independent_noise_summary)
    summary = _mapping(summary, context="independent NOISE summary")
    if (
        summary.get("schema") != "simple-independent-geometry-noise-report-v1"
        or summary.get("status") != "complete"
        or summary.get("sweep_id") != experiment.prefix.sweep_id
        or summary.get("sweep_digest") != experiment.prefix.digest
    ):
        raise ArtifactError("Independent NOISE summary does not match the prefix sweep")
    selected = _mapping(summary.get("selected_q"), context="independent NOISE selected q")
    report_rows = {
        str(_mapping(value, context="independent NOISE cell").get("cell", "")): _mapping(
            value,
            context="independent NOISE cell",
        )
        for value in _sequence(summary.get("cells"), context="independent NOISE cells")
    }
    cells = {}
    for cell_id, selection in selected.items():
        cell = str(cell_id)
        values = _mapping(selection, context=f"independent NOISE selected q/{cell}")
        q_values = {}
        for geometry in GEOMETRY_ORDER:
            q = int(values.get(geometry, 0))
            if q not in experiment.prefix.q_values:
                raise ArtifactError(f"Independent NOISE q is outside the completed sweep: {cell}")
            q_values[geometry] = q
        row = report_rows.get(cell)
        geometries = (
            ()
            if row is None
            else _sequence(row.get("geometries"), context=f"independent NOISE geometries/{cell}")
        )
        reported = {
            str(
                _mapping(item, context=f"independent NOISE geometry/{cell}").get("geometry", "")
            ): int(_mapping(item, context=f"independent NOISE geometry/{cell}").get("q", 0))
            for item in geometries
        }
        if reported != q_values:
            raise ArtifactError(f"Independent NOISE selected q disagrees with report rows: {cell}")
        cells[cell] = q_values
    expected = {cell.cell_id for cell in experiment.prefix.cells()}
    if set(cells) != expected:
        raise ArtifactError("Independent NOISE summary does not cover the random-control cells")
    return cells


def _finite_summary(values: NDArray[np.float64], *, confidence: float) -> Mapping[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "valid_replicates": 0,
            "valid_fraction": 0.0,
            "standard_error": None,
            "ci_low": None,
            "ci_high": None,
        }
    alpha = 1.0 - confidence
    return {
        "valid_replicates": int(finite.size),
        "valid_fraction": float(finite.size / values.size),
        "standard_error": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "ci_low": float(np.quantile(finite, alpha / 2.0)),
        "ci_high": float(np.quantile(finite, 1.0 - alpha / 2.0)),
    }


def _outcome(rrel_gain: float, excess_gain: float) -> str:
    if not math.isfinite(rrel_gain) or not math.isfinite(excess_gain):
        return "undefined_clean_excess"
    relative = rrel_gain > 1e-12
    quality = excess_gain > 1e-12
    if relative and quality:
        return "advantage"
    if relative:
        return "relative_only"
    if quality:
        return "quality_only"
    if abs(rrel_gain) <= 1e-12 and abs(excess_gain) <= 1e-12:
        return "equal"
    return "no_advantage"


def _point_row(
    *,
    cell: Any,
    random: RandomControlBank,
    q_by_geometry: Mapping[str, int],
    bootstrap: RelativeBootstrap,
    geometry_index: int,
    condition_index: int,
    rule_index: int,
    metric_index: int,
    confidence: float,
) -> dict[str, Any]:
    point = bootstrap.point
    geometry = GEOMETRY_ORDER[geometry_index]
    condition = NOISE_KEYS[condition_index]
    rule = RULE_IDS[rule_index]
    metric = QUALITY_METRICS[metric_index]
    rrel_gain = float(point.rrel_gain[0, geometry_index, condition_index, rule_index, metric_index])
    excess_gain = float(
        point.perturbed_excess_gain[0, geometry_index, condition_index, rule_index, metric_index]
    )
    rrel_samples = bootstrap.rrel_gain_replicates[
        :, geometry_index, condition_index, rule_index, metric_index
    ]
    excess_samples = bootstrap.perturbed_excess_gain_replicates[
        :, geometry_index, condition_index, rule_index, metric_index
    ]
    rrel_summary = _finite_summary(rrel_samples, confidence=confidence)
    excess_summary = _finite_summary(excess_samples, confidence=confidence)
    joint = np.isfinite(rrel_samples) & np.isfinite(excess_samples)
    joint_probability = (
        None
        if not np.any(joint)
        else float(np.mean((rrel_samples[joint] > 0.0) & (excess_samples[joint] > 0.0)))
    )
    support = bool(
        rrel_summary["ci_low"] is not None
        and excess_summary["ci_low"] is not None
        and float(rrel_summary["ci_low"]) > 0.0
        and float(excess_summary["ci_low"]) > 0.0
    )
    metric_direction = DEFAULT_METRIC_DIRECTIONS[metric]
    random_mean = random.values.mean(axis=(1, 2))
    return {
        "comparison_id": object_sha256(
            {
                "schema": RELATIVE_ROBUSTNESS_SCHEMA,
                "cell": cell.cell_id,
                "geometry": geometry,
                "q": int(q_by_geometry[geometry]),
                "rule": rule,
                "condition": condition,
                "metric": metric,
            }
        ),
        "cell": cell.cell_id,
        "dataset": cell.dataset.dataset_id,
        "model": cell.reference_model.model_id,
        "sample_count": int(random.labels.size),
        "geometry": geometry,
        "geometry_label": GEOMETRY_LABELS[geometry],
        "q": int(q_by_geometry[geometry]),
        "rule": rule,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "metric": metric,
        "metric_direction": metric_direction,
        "random_clean_quality": float(random_mean[0, metric_index]),
        "random_perturbed_quality": float(random_mean[condition_index + 1, metric_index]),
        "naive_clean_excess": float(point.naive_clean_excess[0, rule_index, metric_index]),
        "naive_perturbed_excess": float(
            point.naive_perturbed_excess[0, condition_index, rule_index, metric_index]
        ),
        "naive_rrel": float(point.naive_rrel[0, condition_index, rule_index, metric_index]),
        "noise_clean_excess": float(
            point.noise_clean_excess[0, geometry_index, rule_index, metric_index]
        ),
        "noise_perturbed_excess": float(
            point.noise_perturbed_excess[
                0, geometry_index, condition_index, rule_index, metric_index
            ]
        ),
        "noise_rrel": float(
            point.noise_rrel[0, geometry_index, condition_index, rule_index, metric_index]
        ),
        "rrel_gain_naive_minus_noise": rrel_gain,
        "perturbed_excess_gain_noise_minus_naive": excess_gain,
        "rrel_gain_standard_error": rrel_summary["standard_error"],
        "rrel_gain_ci_low": rrel_summary["ci_low"],
        "rrel_gain_ci_high": rrel_summary["ci_high"],
        "rrel_gain_valid_bootstrap_fraction": rrel_summary["valid_fraction"],
        "perturbed_excess_gain_standard_error": excess_summary["standard_error"],
        "perturbed_excess_gain_ci_low": excess_summary["ci_low"],
        "perturbed_excess_gain_ci_high": excess_summary["ci_high"],
        "perturbed_excess_gain_valid_bootstrap_fraction": excess_summary["valid_fraction"],
        "joint_advantage_bootstrap_fraction": joint_probability,
        "both_marginal_ci_lower_positive": support,
        "point_outcome": _outcome(rrel_gain, excess_gain),
    }


def _seed_stability_rows(random: RandomControlBank) -> list[dict[str, Any]]:
    result = []
    per_seed = random.values.mean(axis=1)
    for condition_index, condition in enumerate(CONDITION_KEYS):
        for metric_index, metric in enumerate(QUALITY_METRICS):
            values = per_seed[condition_index, :, metric_index]
            result.append(
                {
                    "cell": random.cell,
                    "dataset": random.dataset,
                    "model": random.model,
                    "condition": condition,
                    "metric": metric,
                    "seed_count": len(random.random_seed_bank),
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)),
                    "standard_error": float(values.std(ddof=1) / math.sqrt(values.size)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "range": float(values.max() - values.min()),
                    "first_half_mean": float(values[: values.size // 2].mean()),
                    "second_half_mean": float(values[values.size // 2 :].mean()),
                    "half_bank_absolute_difference": float(
                        abs(values[: values.size // 2].mean() - values[values.size // 2 :].mean())
                    ),
                }
            )
    return result


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()


_RESULT_COLUMNS = (
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
    "metric_direction",
    "random_clean_quality",
    "random_perturbed_quality",
    "naive_clean_excess",
    "naive_perturbed_excess",
    "naive_rrel",
    "noise_clean_excess",
    "noise_perturbed_excess",
    "noise_rrel",
    "rrel_gain_naive_minus_noise",
    "perturbed_excess_gain_noise_minus_naive",
    "rrel_gain_standard_error",
    "rrel_gain_ci_low",
    "rrel_gain_ci_high",
    "rrel_gain_valid_bootstrap_fraction",
    "perturbed_excess_gain_standard_error",
    "perturbed_excess_gain_ci_low",
    "perturbed_excess_gain_ci_high",
    "perturbed_excess_gain_valid_bootstrap_fraction",
    "joint_advantage_bootstrap_fraction",
    "both_marginal_ci_lower_positive",
    "point_outcome",
)

_STABILITY_COLUMNS = (
    "cell",
    "dataset",
    "model",
    "condition",
    "metric",
    "seed_count",
    "mean",
    "standard_deviation",
    "standard_error",
    "minimum",
    "maximum",
    "range",
    "first_half_mean",
    "second_half_mean",
    "half_bank_absolute_difference",
)


def _readme(summary: Mapping[str, Any]) -> str:
    analysis = _mapping(summary["analysis"], context="relative robustness analysis")
    return "\n".join(
        (
            "# Relative Robustness vs q=11 NAIVE",
            "",
            "For each metric, quality is first oriented so larger is better: F/C are unchanged and Fbar/Cbar are negated.",
            "The random control is a fixed bank of uniform p=16 top-k patch masks shared by the clean image and every perturbation of that image.",
            "",
            "`E = oriented_quality - oriented_random_quality` is the above-random excess quality.",
            "`R_rel = 1 - E_perturbed / E_clean`; lower is better, zero retains the clean above-random quality, and a negative value improves it.",
            "Endpoints with nonpositive clean excess are undefined; no epsilon is introduced.",
            "",
            "A NOISE advantage requires both a positive NAIVE-minus-NOISE R_rel gain and positive NOISE-minus-NAIVE perturbed excess gain.",
            "Bootstrap rows resample images within true-label strata and recompute the dataset-level F/Fbar/C/Cbar means. They do not treat those metrics as per-image outcomes.",
            "",
            f"Point advantages: {analysis['point_advantages']}/{analysis['defined_endpoints']} defined endpoints.",
            f"Both marginal {float(summary['bootstrap']['confidence']) * 100:.0f}% CI lower bounds positive: {analysis['ci_supported_advantages']}/{analysis['defined_endpoints']}.",
            f"Result digest: `{summary['result_digest']}`.",
            "",
        )
    )


def write_relative_robustness_report(
    experiment: RelativeRobustnessExperiment,
    *,
    output_directory: str | Path,
    bootstrap_replicates: int | None = None,
    bootstrap_batch_size: int | None = None,
    confidence: float | None = None,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Calculate and write R_rel after all fixed random controls complete."""

    replicates = (
        experiment.bootstrap_replicates if bootstrap_replicates is None else bootstrap_replicates
    )
    batch_size = (
        experiment.bootstrap_batch_size if bootstrap_batch_size is None else bootstrap_batch_size
    )
    level = experiment.confidence if confidence is None else confidence
    if replicates <= 0 or batch_size <= 0 or not 0.0 < level < 1.0:
        raise ValueError("relative robustness bootstrap settings are invalid")
    q_by_cell = _selected_q(experiment)
    rows = []
    stability_rows = []
    cell_audits = []
    with TemporaryDirectory(prefix="xai-relative-robustness-", dir="/dev/shm") as directory:
        root = Path(directory)
        for cell in sorted(experiment.prefix.cells(), key=lambda value: value.cell_id):
            random = _load_random_control_bank(
                experiment,
                cell_id=cell.cell_id,
                work_root=root / f"controls-{cell.cell_id}",
            )
            candidate = _build_cell_bank(
                base_experiment=experiment.base,
                prefix_experiment=experiment.prefix,
                cell=cell,
                q_by_geometry=q_by_cell[cell.cell_id],
                work_root=root / f"candidates-{cell.cell_id}",
            )
            if random.labels.size != candidate.labels.size or not np.array_equal(
                random.labels, candidate.labels
            ):
                raise ArtifactError(
                    f"Random controls and NOISE candidates are not sample-aligned: {cell.cell_id}"
                )
            bootstrap = bootstrap_relative_values(
                random=random,
                naive=candidate.naive,
                noise=candidate.noise,
                replicates=replicates,
                seed=stable_seed(seed, RELATIVE_ROBUSTNESS_SCHEMA, cell.cell_id),
                batch_size=batch_size,
            )
            for geometry_index in range(len(GEOMETRY_ORDER)):
                for condition_index in range(len(NOISE_KEYS)):
                    for rule_index in range(_RULE_COUNT):
                        for metric_index in range(_METRIC_COUNT):
                            rows.append(
                                _point_row(
                                    cell=cell,
                                    random=random,
                                    q_by_geometry=q_by_cell[cell.cell_id],
                                    bootstrap=bootstrap,
                                    geometry_index=geometry_index,
                                    condition_index=condition_index,
                                    rule_index=rule_index,
                                    metric_index=metric_index,
                                    confidence=level,
                                )
                            )
            stability_rows.extend(_seed_stability_rows(random))
            cell_audits.append(
                {
                    "cell": cell.cell_id,
                    "dataset": random.dataset,
                    "model": random.model,
                    "sample_count": int(random.labels.size),
                    "class_count": int(np.unique(random.labels).size),
                    "random_seed_bank": list(random.random_seed_bank),
                    "q_by_geometry": dict(q_by_cell[cell.cell_id]),
                    "random_control_condition_transition_audit": {
                        key: dict(value) for key, value in random.condition_transition_audit.items()
                    },
                    "candidate_condition_transition_audit": {
                        key: dict(value)
                        for key, value in candidate.condition_transition_audit.items()
                    },
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["cell"]),
            GEOMETRY_ORDER.index(str(row["geometry"])),
            RULE_IDS.index(str(row["rule"])),
            NOISE_KEYS.index(str(row["condition"])),
            QUALITY_METRICS.index(str(row["metric"])),
        )
    )
    stability_rows.sort(
        key=lambda row: (
            str(row["cell"]),
            CONDITION_KEYS.index(str(row["condition"])),
            QUALITY_METRICS.index(str(row["metric"])),
        )
    )
    defined = [row for row in rows if row["point_outcome"] != "undefined_clean_excess"]
    analysis = {
        "endpoints": len(rows),
        "defined_endpoints": len(defined),
        "undefined_clean_excess": len(rows) - len(defined),
        "point_advantages": sum(row["point_outcome"] == "advantage" for row in rows),
        "relative_only": sum(row["point_outcome"] == "relative_only" for row in rows),
        "quality_only": sum(row["point_outcome"] == "quality_only" for row in rows),
        "ci_supported_advantages": sum(
            bool(row["both_marginal_ci_lower_positive"]) for row in rows
        ),
        "by_geometry": {
            geometry: {
                "endpoints": sum(row["geometry"] == geometry for row in rows),
                "defined": sum(
                    row["geometry"] == geometry and row["point_outcome"] != "undefined_clean_excess"
                    for row in rows
                ),
                "point_advantages": sum(
                    row["geometry"] == geometry and row["point_outcome"] == "advantage"
                    for row in rows
                ),
                "ci_supported_advantages": sum(
                    row["geometry"] == geometry and bool(row["both_marginal_ci_lower_positive"])
                    for row in rows
                ),
            }
            for geometry in GEOMETRY_ORDER
        },
    }
    value: dict[str, Any] = {
        "schema": RELATIVE_ROBUSTNESS_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "prefix_sweep_id": experiment.prefix.sweep_id,
        "prefix_sweep_digest": experiment.prefix.digest,
        "independent_noise_summary": str(experiment.independent_noise_summary),
        "science": {
            "comparison": "fixed_selected_NOISE_S_or_K_q_vs_exact_q11_NAIVE_same_rule",
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "random_control": "uniform_patch_permutation_shared_across_conditions_per_image_seed",
            "random_seed_count": len(experiment.random_seed_bank),
            "oriented_quality": {
                "F": "identity",
                "Fbar": "negated",
                "C": "identity",
                "Cbar": "negated",
            },
            "relative_robustness": "1 - perturbed_excess_over_random / clean_excess_over_random",
            "rrel_direction": "min",
            "undefined_policy": "clean_excess_at_or_below_zero_is_undefined_no_epsilon",
            "advantage_policy": "rrel_gain_naive_minus_noise_positive_and_perturbed_excess_gain_noise_minus_naive_positive",
            "bootstrap_unit": "image_stratified_by_true_class_recomputing_dataset_metric_means",
            "bootstrap_interpretation": "conditional_post_selection_image_sampling_not_held_out_q_selection_generalization",
        },
        "bootstrap": {
            "replicates": replicates,
            "batch_size": batch_size,
            "confidence": level,
            "seed": seed,
            "ci_policy": "marginal_percentile_intervals_no_multiple_comparison_claim",
        },
        "cells": cell_audits,
        "analysis": analysis,
        "rows": rows,
        "random_control_seed_stability": stability_rows,
    }
    value["result_digest"] = object_sha256(value)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", value)
    atomic_write_text(output / "relative_robustness.csv", _csv_text(rows, _RESULT_COLUMNS))
    atomic_write_text(
        output / "random_control_seed_stability.csv", _csv_text(stability_rows, _STABILITY_COLUMNS)
    )
    atomic_write_text(output / "README.md", _readme(value))
    return {
        "status": "complete",
        "summary": str(output / "summary.json"),
        "relative_robustness_csv": str(output / "relative_robustness.csv"),
        "seed_stability_csv": str(output / "random_control_seed_stability.csv"),
        "result_digest": value["result_digest"],
        "analysis": analysis,
    }


__all__ = [
    "RELATIVE_ROBUSTNESS_SCHEMA",
    "RandomControlBank",
    "RelativeBootstrap",
    "RelativeValues",
    "bootstrap_relative_values",
    "relative_values_from_means",
    "write_relative_robustness_report",
]
