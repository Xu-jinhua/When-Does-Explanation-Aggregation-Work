"""Current-manuscript quality metrics from per-sample sufficient statistics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricDirection = Literal["max", "min"]
QUALITY_METRICS = ("F", "Fbar", "C", "Cbar")

# Under the formulas and explanatory text in the current manuscript, removing
# top-k evidence should make F/C large, while retaining top-k evidence should
# make complementary Fbar/Cbar small.  Robustness is always minimized.
DEFAULT_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "F": "max",
    "Fbar": "min",
    "C": "max",
    "Cbar": "min",
    "R_F": "min",
    "R_Fbar": "min",
    "R_C": "min",
    "R_Cbar": "min",
}


def _bool_vector(values: ArrayLike, *, name: str) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain booleans")
    return array.astype(np.bool_, copy=False)


def _prediction_vector(values: ArrayLike, *, name: str) -> NDArray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True)
class MetricValues:
    """Dataset means for the four manuscript metrics."""

    F: float
    Fbar: float
    C: float
    Cbar: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in QUALITY_METRICS}

    def __getitem__(self, metric: str) -> float:
        if metric not in QUALITY_METRICS:
            raise KeyError(metric)
        return float(getattr(self, metric))


@dataclass(frozen=True)
class MetricSufficientStats:
    """Boolean per-sample events sufficient for F/Fbar/C/Cbar.

    ``removed`` denotes manuscript mask :math:`M` (top-k patches removed), and
    ``retained`` denotes :math:`\bar M` (only top-k patches retained).  Keeping
    the events rather than only final means permits paired and stratified
    bootstrap analysis without another model forward pass.
    """

    clean_correct: NDArray[np.bool_]
    removed_correct: NDArray[np.bool_]
    retained_correct: NDArray[np.bool_]
    removed_changed: NDArray[np.bool_]
    retained_changed: NDArray[np.bool_]
    sample_ids: NDArray | None = None
    class_labels: NDArray | None = None

    def __post_init__(self) -> None:
        names = (
            "clean_correct",
            "removed_correct",
            "retained_correct",
            "removed_changed",
            "retained_changed",
        )
        vectors = [_bool_vector(getattr(self, name), name=name) for name in names]
        size = vectors[0].size
        if any(vector.size != size for vector in vectors[1:]):
            raise ValueError("all sufficient-statistic arrays must have equal length")
        for name, vector in zip(names, vectors, strict=True):
            object.__setattr__(self, name, vector)
        for name in ("sample_ids", "class_labels"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value)
                if array.ndim != 1 or array.size != size:
                    raise ValueError(f"{name} must be one-dimensional with length {size}")
                object.__setattr__(self, name, array)
        if self.sample_ids is not None and np.unique(self.sample_ids).size != size:
            raise ValueError("sample_ids must be unique")

    @classmethod
    def from_predictions(
        cls,
        *,
        true_labels: ArrayLike,
        clean_predictions: ArrayLike,
        removed_predictions: ArrayLike,
        retained_predictions: ArrayLike,
        sample_ids: ArrayLike | None = None,
        class_labels: ArrayLike | None = None,
    ) -> MetricSufficientStats:
        """Construct sufficient events from clean and two masked predictions."""

        true = _prediction_vector(true_labels, name="true_labels")
        clean = _prediction_vector(clean_predictions, name="clean_predictions")
        removed = _prediction_vector(removed_predictions, name="removed_predictions")
        retained = _prediction_vector(retained_predictions, name="retained_predictions")
        if not (true.shape == clean.shape == removed.shape == retained.shape):
            raise ValueError("all label and prediction arrays must have equal shape")
        if class_labels is None:
            class_labels = true
        return cls(
            clean_correct=clean == true,
            removed_correct=removed == true,
            retained_correct=retained == true,
            removed_changed=removed != clean,
            retained_changed=retained != clean,
            sample_ids=None if sample_ids is None else np.asarray(sample_ids),
            class_labels=np.asarray(class_labels),
        )

    @property
    def n_samples(self) -> int:
        return int(self.clean_correct.size)

    def contributions(self) -> dict[str, NDArray[np.float64]]:
        """Return per-sample addends whose means are the four metrics."""

        clean = self.clean_correct.astype(np.float64)
        return {
            "F": clean - self.removed_correct.astype(np.float64),
            "Fbar": clean - self.retained_correct.astype(np.float64),
            "C": self.removed_changed.astype(np.float64),
            "Cbar": self.retained_changed.astype(np.float64),
        }

    def values(self) -> MetricValues:
        contributions = self.contributions()
        return MetricValues(
            **{
                metric: float(np.mean(contributions[metric]))
                for metric in QUALITY_METRICS
            }
        )

    def take(self, indices: ArrayLike) -> MetricSufficientStats:
        index = np.asarray(indices)
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise TypeError("indices must be a one-dimensional integer array")
        kwargs = {
            name: getattr(self, name)[index]
            for name in (
                "clean_correct",
                "removed_correct",
                "retained_correct",
                "removed_changed",
                "retained_changed",
            )
        }
        # Bootstrap resampling creates repeated ids; omit ids in a resampled
        # stats object rather than falsely claiming uniqueness.
        kwargs["sample_ids"] = None
        kwargs["class_labels"] = (
            None if self.class_labels is None else self.class_labels[index]
        )
        return MetricSufficientStats(**kwargs)


def compute_metrics(stats: MetricSufficientStats) -> MetricValues:
    """Compute F/Fbar/C/Cbar exactly as defined in the current manuscript."""

    return stats.values()


@dataclass(frozen=True)
class RobustnessValues:
    """Absolute manuscript robustness and direction-aware signed degradation."""

    absolute: Mapping[str, float]
    signed_degradation: Mapping[str, float]
    clean: MetricValues
    perturbed: MetricValues

    def __getitem__(self, metric: str) -> float:
        return float(self.absolute[metric])


def _assert_aligned(
    clean: MetricSufficientStats, perturbed: MetricSufficientStats
) -> None:
    if clean.n_samples != perturbed.n_samples:
        raise ValueError("clean and perturbed stats must contain the same samples")
    if clean.sample_ids is not None and perturbed.sample_ids is not None:
        if not np.array_equal(clean.sample_ids, perturbed.sample_ids):
            raise ValueError("clean and perturbed sample_ids are not aligned")


def robustness_contributions(
    clean: MetricSufficientStats, perturbed: MetricSufficientStats
) -> dict[str, NDArray[np.float64]]:
    """Direction-aware paired degradation addends before the final absolute value."""

    _assert_aligned(clean, perturbed)
    clean_contributions = clean.contributions()
    perturbed_contributions = perturbed.contributions()
    result: dict[str, NDArray[np.float64]] = {}
    for metric in QUALITY_METRICS:
        if DEFAULT_METRIC_DIRECTIONS[metric] == "max":
            result[metric] = clean_contributions[metric] - perturbed_contributions[metric]
        else:
            result[metric] = perturbed_contributions[metric] - clean_contributions[metric]
    return result


def compute_robustness(
    clean: MetricSufficientStats, perturbed: MetricSufficientStats
) -> RobustnessValues:
    """Compute ``R_* = |metric(clean) - metric(perturbed)|`` for all metrics."""

    _assert_aligned(clean, perturbed)
    clean_values = clean.values()
    perturbed_values = perturbed.values()
    absolute: dict[str, float] = {}
    signed: dict[str, float] = {}
    for metric in QUALITY_METRICS:
        clean_value = clean_values[metric]
        perturbed_value = perturbed_values[metric]
        robustness_name = f"R_{metric}"
        absolute[robustness_name] = abs(clean_value - perturbed_value)
        if DEFAULT_METRIC_DIRECTIONS[metric] == "max":
            signed[robustness_name] = clean_value - perturbed_value
        else:
            signed[robustness_name] = perturbed_value - clean_value
    return RobustnessValues(
        absolute=absolute,
        signed_degradation=signed,
        clean=clean_values,
        perturbed=perturbed_values,
    )


@dataclass(frozen=True)
class OracleChoice:
    metric: str
    method: str
    value: float
    direction: MetricDirection


def oracle_best_single(
    values_by_method: Mapping[str, Mapping[str, float] | MetricValues],
    *,
    directions: Mapping[str, MetricDirection] | None = None,
) -> dict[str, OracleChoice]:
    """Select a potentially different single explainer for every metric.

    Ties are resolved by lexicographic method id.  The output is therefore the
    paper's *Oracle Best Single* row, not one deployable explainer selected once
    for all metrics.
    """

    if not values_by_method:
        raise ValueError("values_by_method must not be empty")
    resolved_directions = dict(DEFAULT_METRIC_DIRECTIONS)
    if directions is not None:
        resolved_directions.update(directions)
    normalized: dict[str, dict[str, float]] = {}
    for method, values in values_by_method.items():
        if isinstance(values, MetricValues):
            normalized[str(method)] = values.as_dict()
        else:
            normalized[str(method)] = {
                str(metric): float(value) for metric, value in values.items()
            }
    metrics = sorted(set.intersection(*(set(row) for row in normalized.values())))
    if not metrics:
        raise ValueError("methods do not share any metrics")
    choices: dict[str, OracleChoice] = {}
    for metric in metrics:
        if metric not in resolved_directions:
            raise ValueError(f"no optimization direction declared for metric {metric!r}")
        direction = resolved_directions[metric]
        if direction not in ("max", "min"):
            raise ValueError(f"invalid direction {direction!r} for {metric!r}")
        candidates: list[tuple[float, str, float]] = []
        for method, row in normalized.items():
            value = row[metric]
            if not np.isfinite(value):
                raise ValueError(f"non-finite {metric!r} value for method {method!r}")
            key = -value if direction == "max" else value
            candidates.append((key, method, value))
        _, method, value = min(candidates, key=lambda item: (item[0], item[1]))
        choices[metric] = OracleChoice(metric, method, value, direction)
    return choices


def oracle_best_single_from_stats(
    stats_by_method: Mapping[str, MetricSufficientStats],
) -> dict[str, OracleChoice]:
    return oracle_best_single(
        {method: stats.values() for method, stats in stats_by_method.items()}
    )


__all__ = [
    "DEFAULT_METRIC_DIRECTIONS",
    "MetricDirection",
    "MetricSufficientStats",
    "MetricValues",
    "OracleChoice",
    "QUALITY_METRICS",
    "RobustnessValues",
    "compute_metrics",
    "compute_robustness",
    "oracle_best_single",
    "oracle_best_single_from_stats",
    "robustness_contributions",
]
