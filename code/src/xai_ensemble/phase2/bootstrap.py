"""Paired, class-stratified, and IND-hierarchical uncertainty estimates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Statistic = Callable[[NDArray[np.float64]], float]


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _validate_bootstrap(B: int, confidence: float) -> None:
    if not isinstance(B, (int, np.integer)) or B <= 0:
        raise ValueError("B must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")


def _strata(class_labels: ArrayLike | None, size: int) -> list[NDArray[np.int64]]:
    if class_labels is None:
        return [np.arange(size, dtype=np.int64)]
    labels = np.asarray(class_labels)
    if labels.ndim != 1 or labels.size != size:
        raise ValueError(f"class_labels must be one-dimensional with length {size}")
    # np.unique supplies a stable, deterministic ordering for numeric/string
    # class labels; resampling itself remains random through the supplied seed.
    _, inverse = np.unique(labels, return_inverse=True)
    return [
        np.flatnonzero(inverse == class_index).astype(np.int64, copy=False)
        for class_index in range(int(np.max(inverse)) + 1)
    ]


def _resample_indices(
    strata: list[NDArray[np.int64]], rng: np.random.Generator
) -> NDArray[np.int64]:
    return np.concatenate(
        [rng.choice(group, size=group.size, replace=True) for group in strata]
    )


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    replicates: NDArray[np.float64]


def _summarize_replicates(
    estimate: float,
    replicates: NDArray[np.float64],
    confidence: float,
) -> BootstrapEstimate:
    alpha = 1.0 - confidence
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    standard_error = (
        float(np.std(replicates, ddof=1)) if replicates.size > 1 else 0.0
    )
    return BootstrapEstimate(
        estimate=float(estimate),
        standard_error=standard_error,
        ci_low=float(low),
        ci_high=float(high),
        confidence=float(confidence),
        replicates=replicates,
    )


def paired_class_stratified_bootstrap(
    first: ArrayLike,
    second: ArrayLike | None = None,
    *,
    class_labels: ArrayLike | None = None,
    statistic: Statistic | None = None,
    B: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapEstimate:
    """Bootstrap a paired difference while preserving each class count.

    If ``second`` is supplied, the resampled unit is ``first - second``; the
    same sampled index is always applied to both methods.  ``statistic`` then
    receives that paired vector and defaults to its mean.  Passing
    ``statistic=lambda x: abs(mean(x))`` is useful for robustness metrics,
    whose absolute value must be applied *after* the dataset mean.
    """

    _validate_bootstrap(B, confidence)
    first_array = _finite_vector(first, name="first")
    if second is None:
        paired = first_array
    else:
        second_array = _finite_vector(second, name="second")
        if second_array.shape != first_array.shape:
            raise ValueError("paired arrays must have equal shape")
        paired = first_array - second_array
    if statistic is None:
        def statistic(values: NDArray[np.float64]) -> float:
            return float(np.mean(values))
    estimate = float(statistic(paired))
    if not np.isfinite(estimate):
        raise ValueError("statistic returned a non-finite estimate")
    groups = _strata(class_labels, paired.size)
    rng = np.random.default_rng(seed)
    replicates = np.empty(B, dtype=np.float64)
    for iteration in range(B):
        indices = _resample_indices(groups, rng)
        replicates[iteration] = float(statistic(paired[indices]))
    if not np.all(np.isfinite(replicates)):
        raise ValueError("statistic returned non-finite bootstrap replicates")
    return _summarize_replicates(estimate, replicates, confidence)


@dataclass(frozen=True)
class OracleBootstrapResult:
    difference: BootstrapEstimate
    selection_frequency: Mapping[str, float]
    selected_on_full_data: str


def bootstrap_oracle_vs_comparator(
    method_contributions: Mapping[str, ArrayLike],
    comparator_contributions: ArrayLike,
    *,
    direction: Literal["max", "min"],
    class_labels: ArrayLike | None = None,
    B: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> OracleBootstrapResult:
    """Reselect the Oracle Best Single inside every bootstrap replicate."""

    _validate_bootstrap(B, confidence)
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'")
    if not method_contributions:
        raise ValueError("method_contributions must not be empty")
    comparator = _finite_vector(
        comparator_contributions, name="comparator_contributions"
    )
    methods = sorted(str(method) for method in method_contributions)
    arrays: dict[str, NDArray[np.float64]] = {}
    for original_method, values in method_contributions.items():
        method = str(original_method)
        array = _finite_vector(values, name=f"method_contributions[{method!r}]")
        if array.shape != comparator.shape:
            raise ValueError("all contribution arrays must have equal shape")
        arrays[method] = array

    sign = 1.0 if direction == "max" else -1.0

    def choose(indices: NDArray[np.int64]) -> tuple[str, float]:
        candidates = [
            (-(sign * float(np.mean(arrays[method][indices]))), method)
            for method in methods
        ]
        _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        difference = float(
            np.mean(arrays[selected][indices]) - np.mean(comparator[indices])
        )
        return selected, difference

    full_indices = np.arange(comparator.size, dtype=np.int64)
    selected_full, estimate = choose(full_indices)
    groups = _strata(class_labels, comparator.size)
    rng = np.random.default_rng(seed)
    replicates = np.empty(B, dtype=np.float64)
    selections = {method: 0 for method in methods}
    for iteration in range(B):
        indices = _resample_indices(groups, rng)
        selected, difference = choose(indices)
        selections[selected] += 1
        replicates[iteration] = difference
    return OracleBootstrapResult(
        difference=_summarize_replicates(estimate, replicates, confidence),
        selection_frequency={
            method: count / B for method, count in selections.items()
        },
        selected_on_full_data=selected_full,
    )


@dataclass(frozen=True)
class INDHierarchicalSummary:
    """IND minus a matched comparator with family/source/sample uncertainty."""

    raw_difference: BootstrapEstimate
    direction_adjusted_gain: BootstrapEstimate
    family_differences: NDArray[np.float64]
    n_families: int
    n_sources: int
    n_samples: int


def _hierarchical_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3 or any(dimension == 0 for dimension in array.shape):
        raise ValueError(
            f"{name} must have shape (families, samples) or "
            "(families, sources, samples)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def ind_hierarchical_summary(
    ind_values: ArrayLike,
    matched_values: ArrayLike,
    *,
    class_labels: ArrayLike | None = None,
    direction: Literal["max", "min"] = "max",
    B: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> INDHierarchicalSummary:
    """Hierarchically bootstrap paired IND and matched-comparator observations.

    Arrays may be ``(training_families, test_samples)`` or
    ``(training_families, source_models, test_samples)``.  Each replicate
    resamples training families, source models within a selected family, and
    common-test samples within class.  IND and its matched comparator always
    share the same sampled indices at every level.
    """

    _validate_bootstrap(B, confidence)
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'")
    ind = _hierarchical_array(ind_values, name="ind_values")
    matched = _hierarchical_array(matched_values, name="matched_values")
    if ind.shape != matched.shape:
        raise ValueError("IND and matched arrays must have equal shape")
    differences = ind - matched
    n_families, n_sources, n_samples = differences.shape
    groups = _strata(class_labels, n_samples)
    family_differences = np.mean(differences, axis=(1, 2))
    raw_estimate = float(np.mean(family_differences))

    rng = np.random.default_rng(seed)
    raw_replicates = np.empty(B, dtype=np.float64)
    for iteration in range(B):
        sampled_families = rng.choice(n_families, size=n_families, replace=True)
        selected_values: list[float] = []
        for family in sampled_families:
            sampled_sources = rng.choice(n_sources, size=n_sources, replace=True)
            for source in sampled_sources:
                sample_indices = _resample_indices(groups, rng)
                selected_values.append(
                    float(np.mean(differences[family, source, sample_indices]))
                )
        raw_replicates[iteration] = float(np.mean(selected_values))

    raw_summary = _summarize_replicates(
        raw_estimate, raw_replicates, confidence
    )
    sign = 1.0 if direction == "max" else -1.0
    gain_replicates = sign * raw_replicates
    gain_summary = _summarize_replicates(
        sign * raw_estimate, gain_replicates, confidence
    )
    return INDHierarchicalSummary(
        raw_difference=raw_summary,
        direction_adjusted_gain=gain_summary,
        family_differences=family_differences,
        n_families=n_families,
        n_sources=n_sources,
        n_samples=n_samples,
    )


__all__ = [
    "BootstrapEstimate",
    "INDHierarchicalSummary",
    "OracleBootstrapResult",
    "bootstrap_oracle_vs_comparator",
    "ind_hierarchical_summary",
    "paired_class_stratified_bootstrap",
]
