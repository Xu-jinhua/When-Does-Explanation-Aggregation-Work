"""Fidelity-anchored Mallows diagnostics on the metric-relevant top-k sets.

The mask game observes only which patches enter the top-k set.  This module
therefore works on the fixed-size subset space rather than treating the
irrelevant ordering of the remaining patches as signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class TopKSubsetMallowsFit:
    n_items: int
    k: int
    theta: float
    mean_distance: float
    expected_distance: float
    log_normalizer: float
    boundary: bool


@dataclass(frozen=True, slots=True)
class TopKSubsetMallowsGof:
    fit: TopKSubsetMallowsFit
    ks_statistic: float
    total_variation: float
    log_likelihood: float
    empirical_probabilities: tuple[float, ...]
    fitted_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FidelityAnchoredPrefix:
    q: int
    score: float
    theta: float
    mean_distance: float
    per_sample_score: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class JointFidelityAnchoredPrefix:
    q: int
    score: float
    component_scores: tuple[float, ...]
    per_sample_score: NDArray[np.float64]


def _subset_support(n_items: int, k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if n_items <= 1 or not 0 < k < n_items:
        raise ValueError("top-k subset dimensions require 0 < k < n_items")
    maximum = min(k, n_items - k)
    support = np.arange(maximum + 1, dtype=np.float64)
    log_multiplicity = np.asarray(
        [
            math.lgamma(k + 1)
            - math.lgamma(distance + 1)
            - math.lgamma(k - distance + 1)
            + math.lgamma(n_items - k + 1)
            - math.lgamma(distance + 1)
            - math.lgamma(n_items - k - distance + 1)
            for distance in range(maximum + 1)
        ],
        dtype=np.float64,
    )
    return support, log_multiplicity


def _moments(
    theta: float,
    support: NDArray[np.float64],
    log_multiplicity: NDArray[np.float64],
) -> tuple[float, float]:
    if theta < 0.0 or math.isnan(theta):
        raise ValueError("theta must be non-negative")
    if math.isinf(theta):
        return 0.0, 0.0
    log_mass = log_multiplicity - theta * support
    maximum = float(np.max(log_mass))
    mass = np.exp(log_mass - maximum)
    total = float(np.sum(mass))
    probabilities = mass / total
    return maximum + math.log(total), float(probabilities @ support)


def _probabilities(
    theta: float,
    support: NDArray[np.float64],
    log_multiplicity: NDArray[np.float64],
) -> NDArray[np.float64]:
    if theta < 0.0 or math.isnan(theta):
        raise ValueError("theta must be non-negative")
    if math.isinf(theta):
        result = np.zeros_like(support, dtype=np.float64)
        result[0] = 1.0
        return result
    log_mass = log_multiplicity - theta * support
    maximum = float(np.max(log_mass))
    mass = np.exp(log_mass - maximum)
    return mass / float(np.sum(mass))


def fit_topk_subset_mallows(
    distances: ArrayLike,
    *,
    n_items: int,
    k: int,
) -> TopKSubsetMallowsFit:
    """Fit ``P(D=d) propto C(k,d) C(n-k,d) exp(-theta d)`` exactly."""

    values = np.asarray(distances)
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("distances must be a non-empty numeric array")
    numeric = values.astype(np.float64, copy=False)
    maximum_distance = min(k, n_items - k)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or np.any(numeric > maximum_distance)
        or not np.all(numeric == np.floor(numeric))
    ):
        raise ValueError("distances lie outside the fixed-size subset support")

    support, log_multiplicity = _subset_support(n_items, k)
    target = float(np.mean(numeric))
    log_z_zero, uniform_mean = _moments(0.0, support, log_multiplicity)
    if target >= uniform_mean:
        return TopKSubsetMallowsFit(
            n_items=n_items,
            k=k,
            theta=0.0,
            mean_distance=target,
            expected_distance=uniform_mean,
            log_normalizer=log_z_zero,
            boundary=target > uniform_mean + 1e-12,
        )
    if target == 0.0:
        return TopKSubsetMallowsFit(
            n_items=n_items,
            k=k,
            theta=math.inf,
            mean_distance=0.0,
            expected_distance=0.0,
            log_normalizer=0.0,
            boundary=True,
        )

    low = 0.0
    high = 1.0
    while _moments(high, support, log_multiplicity)[1] > target:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _moments(middle, support, log_multiplicity)[1] > target:
            low = middle
        else:
            high = middle
    theta = (low + high) / 2.0
    log_normalizer, expected = _moments(theta, support, log_multiplicity)
    return TopKSubsetMallowsFit(
        n_items=n_items,
        k=k,
        theta=theta,
        mean_distance=target,
        expected_distance=expected,
        log_normalizer=log_normalizer,
        boundary=False,
    )


def fit_topk_subset_mallows_gof(
    distances: ArrayLike,
    *,
    n_items: int,
    k: int,
) -> TopKSubsetMallowsGof:
    """Fit the exact subset-Mallows family and compare analytic CDFs.

    The statistic never uses a simulated model CDF.  This is important for
    the random-subset control: every candidate has the same finite support and
    sample size, so its empirical-CDF KS error is directly comparable with
    the frozen Fidelity-prefix reference at the same ``q``.
    """

    values = np.asarray(distances)
    fit = fit_topk_subset_mallows(values, n_items=n_items, k=k)
    numeric = values.astype(np.int64, copy=False).reshape(-1)
    maximum_distance = min(k, n_items - k)
    counts = np.bincount(numeric, minlength=maximum_distance + 1).astype(np.float64)
    empirical = counts / float(numeric.size)
    support, log_multiplicity = _subset_support(n_items, k)
    fitted = _probabilities(fit.theta, support, log_multiplicity)
    ks = float(np.max(np.abs(np.cumsum(empirical) - np.cumsum(fitted))))
    total_variation = float(0.5 * np.sum(np.abs(empirical - fitted)))
    if math.isinf(fit.theta):
        log_likelihood = 0.0
    else:
        log_probabilities = log_multiplicity - fit.theta * support - fit.log_normalizer
        log_likelihood = float(np.sum(counts * log_probabilities))
    return TopKSubsetMallowsGof(
        fit=fit,
        ks_statistic=ks,
        total_variation=total_variation,
        log_likelihood=log_likelihood,
        empirical_probabilities=tuple(float(value) for value in empirical),
        fitted_probabilities=tuple(float(value) for value in fitted),
    )


def topk_set_distances(
    ballot_ranks: ArrayLike,
    consensus_top_indices: ArrayLike,
    *,
    k: int,
) -> NDArray[np.int64]:
    """Return Johnson-graph distances from each ballot to each image center."""

    ranks = np.asarray(ballot_ranks)
    centers = np.asarray(consensus_top_indices)
    if ranks.ndim != 3:
        raise ValueError("ballot_ranks must have [images, methods, items] shape")
    if centers.shape != (ranks.shape[0], k):
        raise ValueError("consensus_top_indices must have [images, k] shape")
    if not np.issubdtype(ranks.dtype, np.integer) or not np.issubdtype(centers.dtype, np.integer):
        raise TypeError("ranks and top-k indices must contain integers")
    n_items = int(ranks.shape[2])
    if np.any(centers < 0) or np.any(centers >= n_items):
        raise ValueError("consensus top-k index lies outside the item range")
    if np.any(np.sort(centers, axis=1)[:, 1:] == np.sort(centers, axis=1)[:, :-1]):
        raise ValueError("consensus top-k indices must be unique per image")

    center_mask = np.zeros((ranks.shape[0], n_items), dtype=np.bool_)
    np.put_along_axis(center_mask, centers, True, axis=1)
    ballot_mask = ranks < k
    intersections = np.sum(ballot_mask & center_mask[:, None, :], axis=2, dtype=np.int64)
    return k - intersections


def _normalized_mallows_weights(distances: NDArray[np.int64], theta: float) -> NDArray[np.float64]:
    shifted = distances - np.min(distances, axis=1, keepdims=True)
    if math.isinf(theta):
        weights = shifted == 0
        return weights / np.sum(weights, axis=1, keepdims=True)
    weights = np.exp(-theta * shifted, dtype=np.float64)
    return weights / np.sum(weights, axis=1, keepdims=True)


def fidelity_anchored_prefix(
    distances: ArrayLike,
    fidelity_contributions: ArrayLike,
    *,
    q: int,
    n_items: int,
    k: int,
) -> FidelityAnchoredPrefix:
    """Estimate a consensus's Fidelity from Mallows-local individual marks.

    ``theta`` is fitted only to the retained prefix, while all available
    individual explanations remain labeled anchors for the local utility
    estimate.  No aggregate masked-model forward is used.
    """

    return fidelity_anchored_subset(
        distances,
        fidelity_contributions,
        selected_positions=tuple(range(q)),
        n_items=n_items,
        k=k,
    )


def fidelity_anchored_subset(
    distances: ArrayLike,
    fidelity_contributions: ArrayLike,
    *,
    selected_positions: ArrayLike,
    n_items: int,
    k: int,
) -> FidelityAnchoredPrefix:
    """Estimate one arbitrary subset center from the common utility anchors."""

    distance_array = np.asarray(distances)
    utilities = np.asarray(fidelity_contributions, dtype=np.float64)
    if distance_array.ndim != 2 or utilities.shape != distance_array.shape:
        raise ValueError("distances and contributions must share [images, methods] shape")
    positions = np.asarray(selected_positions)
    if (
        positions.ndim != 1
        or positions.size == 0
        or not np.issubdtype(positions.dtype, np.integer)
        or np.any(positions < 0)
        or np.any(positions >= distance_array.shape[1])
        or np.unique(positions).size != positions.size
    ):
        raise ValueError("selected_positions must identify one non-empty unique method subset")
    if not np.all(np.isfinite(utilities)):
        raise ValueError("fidelity contributions must be finite")
    fit = fit_topk_subset_mallows(distance_array[:, positions], n_items=n_items, k=k)
    weights = _normalized_mallows_weights(distance_array.astype(np.int64, copy=False), fit.theta)
    per_sample = np.sum(weights * utilities, axis=1, dtype=np.float64)
    return FidelityAnchoredPrefix(
        q=int(positions.size),
        score=float(np.mean(per_sample)),
        theta=fit.theta,
        mean_distance=fit.mean_distance,
        per_sample_score=per_sample,
    )


def select_fidelity_anchored_prefix(
    candidates: tuple[FidelityAnchoredPrefix, ...],
) -> FidelityAnchoredPrefix:
    if not candidates:
        raise ValueError("at least one prefix candidate is required")
    ordered = sorted(candidates, key=lambda item: item.q)
    if len({item.q for item in ordered}) != len(ordered):
        raise ValueError("prefix candidates must have unique q values")
    return max(ordered, key=lambda item: item.score)


def combine_fidelity_anchored_prefixes(
    components: tuple[FidelityAnchoredPrefix, ...],
) -> JointFidelityAnchoredPrefix:
    """Average same-q Fidelity estimates from complementary rank geometries."""

    if not components:
        raise ValueError("at least one geometry component is required")
    q = components[0].q
    shape = components[0].per_sample_score.shape
    if any(component.q != q for component in components):
        raise ValueError("geometry components must describe the same q")
    if any(component.per_sample_score.shape != shape for component in components):
        raise ValueError("geometry components must cover the same samples")
    scores = tuple(float(component.score) for component in components)
    if not np.all(np.isfinite(scores)):
        raise ValueError("geometry component scores must be finite")
    per_sample = np.mean(
        np.stack([component.per_sample_score for component in components], axis=0),
        axis=0,
        dtype=np.float64,
    )
    return JointFidelityAnchoredPrefix(
        q=q,
        score=float(np.mean(scores)),
        component_scores=scores,
        per_sample_score=per_sample,
    )


def select_joint_fidelity_anchored_prefix(
    candidates: tuple[JointFidelityAnchoredPrefix, ...],
) -> JointFidelityAnchoredPrefix:
    if not candidates:
        raise ValueError("at least one joint prefix candidate is required")
    ordered = sorted(candidates, key=lambda item: item.q)
    if len({item.q for item in ordered}) != len(ordered):
        raise ValueError("joint prefix candidates must have unique q values")
    return max(ordered, key=lambda item: item.score)


def as_serializable(candidate: FidelityAnchoredPrefix) -> dict[str, Any]:
    return {
        "q": candidate.q,
        "score": candidate.score,
        "theta": candidate.theta,
        "mean_distance": candidate.mean_distance,
    }


__all__ = [
    "FidelityAnchoredPrefix",
    "JointFidelityAnchoredPrefix",
    "TopKSubsetMallowsFit",
    "TopKSubsetMallowsGof",
    "as_serializable",
    "combine_fidelity_anchored_prefixes",
    "fidelity_anchored_prefix",
    "fidelity_anchored_subset",
    "fit_topk_subset_mallows",
    "fit_topk_subset_mallows_gof",
    "select_fidelity_anchored_prefix",
    "select_joint_fidelity_anchored_prefix",
    "topk_set_distances",
]
