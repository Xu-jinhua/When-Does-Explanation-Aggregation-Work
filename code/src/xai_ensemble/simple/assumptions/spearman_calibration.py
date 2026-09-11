"""Calibrate an auditable Spearman--Mallows distance base measure.

The exact multiplicity of a squared Spearman distance is a permanent-counting
problem and is not tractable for the 256 spatial patches used by the formal
protocol.  This pilot estimates that base measure with deterministic-mixture
importance sampling.  Its proposal bank consists of *exactly normalized*
Kendall--Mallows laws, including both an identity-focused proposal and the
uniform permutation law.  Every importance weight is therefore known; no
uniform-on-distance-support surrogate is used.

Independent streams provide split-R-hat and repeated-family diagnostics.  A
piecewise-constant density-of-states representation makes refitted bootstrap
GOF practical while retaining all even Spearman distances inside each bin.
Small problems are enumerated exactly and serve as an implementation oracle.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import math
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from xai_ensemble.core.atomic import atomic_write_bytes
from xai_ensemble.core.hashing import file_sha256, object_sha256, stable_seed
from xai_ensemble.phase2.selection import (
    BinnedDiscreteExponentialFamily,
    DiagnosticDiscreteExponentialFamily,
    discrete_parametric_bootstrap_gof,
)

OFFICIAL_ADAPTER = "xai_ensemble.phase2.selection:locked_spearman_family_factory"
ARTIFACT_SCHEMA_VERSION = 3


class SpearmanCalibrationError(RuntimeError):
    """Raised before publication when a calibration resource contract is invalid."""


@dataclass(frozen=True)
class SpearmanCalibrationOutcome:
    artifact_path: Path
    measurement: dict[str, Any]
    converged: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class _BaseEstimate:
    support: NDArray[np.int64]
    upper: NDArray[np.int64]
    log_mass: NDArray[np.float64]
    empty_bin_run: int
    validation_bin_ess: float
    maximum_distance: int
    maximum_unsampled_gap_fraction: float
    maximum_symmetry_residual: float
    support_completion: str
    full_support: bool
    original_maximum_distance: int


def _theoretical_maximum_distance(n_items: int) -> int:
    return n_items * (n_items * n_items - 1) // 3


def _section(config: Mapping[str, Any]) -> dict[str, Any]:
    nested = config.get("spearman_mallows_approximation")
    value = dict(nested) if isinstance(nested, Mapping) else dict(config)
    if str(value.get("algorithm", "deterministic_mixture_importance_sampling")) != (
        "deterministic_mixture_importance_sampling"
    ):
        raise SpearmanCalibrationError(
            "Spearman pilot requires deterministic_mixture_importance_sampling"
        )
    return value


def _integer_validation_distances(distances: ArrayLike, *, n_items: int) -> NDArray[np.int64]:
    raw = np.asarray(distances)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError("validation distances must be a one-dimensional array of size >= 2")
    numeric = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("validation distances must contain finite integers")
    values = numeric.astype(np.int64)
    maximum = _theoretical_maximum_distance(n_items)
    if np.any(values < 0) or np.any(values > maximum) or np.any(values % 2):
        raise ValueError("validation distances are outside the even Spearman support")
    return values


def _proposal_log_normalizer(n_items: int, q: float) -> float:
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return math.lgamma(n_items + 1.0)
    log_q = math.log(q)
    widths = np.arange(1, n_items + 1, dtype=np.float64)
    return float(np.sum(np.log(-np.expm1(widths * log_q)) - math.log(-math.expm1(log_q))))


def _sample_inversions(n_items: int, q: float, rng: np.random.Generator) -> NDArray[np.int64]:
    widths = np.arange(1, n_items + 1, dtype=np.int64)
    if q == 0.0:
        return np.zeros(n_items, dtype=np.int64)
    uniforms = rng.random(n_items)
    if q == 1.0:
        return np.floor(uniforms * widths).astype(np.int64)
    log_q = math.log(q)
    remainder = 1.0 - uniforms * (1.0 - np.exp(widths * log_q))
    values = np.ceil(np.log(remainder) / log_q - 1e-12).astype(np.int64) - 1
    return np.minimum(np.maximum(values, 0), widths - 1)


def _inversions_to_distance(inversions: NDArray[np.int64]) -> int:
    order: list[int] = []
    for item, inversions_for_item in enumerate(inversions.tolist()):
        order.insert(len(order) - int(inversions_for_item), item)
    permutation = np.asarray(order, dtype=np.int64)
    delta = permutation - np.arange(permutation.size, dtype=np.int64)
    return int(np.dot(delta, delta))


def _sample_stream(
    *,
    n_items: int,
    q_values: NDArray[np.float64],
    draws: int,
    seed: int,
    deadline: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    proposal_count = q_values.size
    counts = np.full(proposal_count, draws // proposal_count, dtype=np.int64)
    counts[: draws % proposal_count] += 1
    if np.any(counts == 0):
        raise SpearmanCalibrationError(
            "draws_per_chain must allocate at least one draw to every proposal"
        )
    mixture = counts / draws
    normalizers = np.asarray(
        [_proposal_log_normalizer(n_items, float(q)) for q in q_values],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    distances = np.empty(draws, dtype=np.int64)
    kendall = np.empty(draws, dtype=np.int64)
    components = np.empty(draws, dtype=np.int64)
    cursor = 0
    for proposal_index, (q, count) in enumerate(zip(q_values, counts, strict=True)):
        for _ in range(int(count)):
            inversions = _sample_inversions(n_items, float(q), rng)
            kendall[cursor] = int(np.sum(inversions))
            distances[cursor] = _inversions_to_distance(inversions)
            components[cursor] = proposal_index
            cursor += 1
        if time.monotonic() > deadline:
            raise SpearmanCalibrationError("Spearman calibration exceeded maximum_wall_seconds")
    log_probability = np.full(draws, -np.inf, dtype=np.float64)
    for index, q in enumerate(q_values):
        if q == 0.0:
            component_log_probability = np.where(kendall == 0, 0.0, -np.inf)
        else:
            component_log_probability = kendall * math.log(float(q)) - normalizers[index]
        log_probability = np.logaddexp(
            log_probability,
            math.log(float(mixture[index])) + component_log_probability,
        )
    log_weights = -log_probability
    ordering = rng.permutation(draws)
    return (
        distances[ordering],
        kendall[ordering],
        log_weights[ordering],
        components[ordering],
    )


def synthetic_validation_distances(
    *,
    n_items: int,
    count: int,
    q_values: ArrayLike,
    seed: int,
    maximum_count: int = 200_000,
    maximum_wall_seconds: float = 21_600.0,
) -> NDArray[np.int64]:
    """Generate a deterministic pre-formal cost/stability distance profile.

    These distances do not decide a scientific result and are not treated as
    observed test data.  They only exercise fitting, refitted bootstrap cost,
    and split-family p-value sensitivity before the formal Phase 1 DAG exists.
    """

    if n_items < 2 or count < 2:
        raise ValueError("n_items and synthetic validation count must be >= 2")
    if count > maximum_count:
        raise SpearmanCalibrationError(
            "synthetic validation count exceeds maximum_validation_distances"
        )
    proposals = np.asarray(q_values, dtype=np.float64)
    if (
        proposals.ndim != 1
        or proposals.size == 0
        or np.any(~np.isfinite(proposals))
        or np.any(proposals < 0)
        or np.any(proposals > 1)
    ):
        raise ValueError("synthetic validation q values must lie in [0,1]")
    counts = np.full(proposals.size, count // proposals.size, dtype=np.int64)
    counts[: count % proposals.size] += 1
    rng = np.random.default_rng(stable_seed("spearman-synthetic-validation", seed))
    result = np.empty(count, dtype=np.int64)
    cursor = 0
    deadline = time.monotonic() + maximum_wall_seconds
    for q, proposal_count in zip(proposals, counts, strict=True):
        for _ in range(int(proposal_count)):
            result[cursor] = _inversions_to_distance(_sample_inversions(n_items, float(q), rng))
            cursor += 1
        if time.monotonic() > deadline:
            raise SpearmanCalibrationError(
                "synthetic validation generation exceeded maximum_wall_seconds"
            )
    return result[rng.permutation(result.size)]


def _maximum_empty_run(nonempty: NDArray[np.bool_]) -> int:
    maximum = 0
    current = 0
    for occupied in nonempty.tolist():
        if occupied:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _grouped_log_sums(
    bins: NDArray[np.int64], log_weights: NDArray[np.float64], size: int
) -> NDArray[np.float64]:
    result = np.full(size, -np.inf, dtype=np.float64)
    ordering = np.argsort(bins, kind="stable")
    sorted_bins = bins[ordering]
    sorted_weights = log_weights[ordering]
    unique, starts = np.unique(sorted_bins, return_index=True)
    result[unique] = np.logaddexp.reduceat(sorted_weights, starts)
    return result


def _estimate_base(
    distances: NDArray[np.int64],
    log_weights: NDArray[np.float64],
    validation: NDArray[np.int64],
    *,
    n_items: int,
    requested_bins: int,
) -> _BaseEstimate:
    maximum_theoretical = _theoretical_maximum_distance(n_items)
    sample_units = distances // 2
    validation_units = validation // 2
    ordered_units = np.sort(sample_units)
    anchor_count = min(requested_bins + 1, ordered_units.size)
    anchor_indices = np.linspace(0, ordered_units.size - 1, anchor_count, dtype=np.int64)
    anchors = np.unique(ordered_units[anchor_indices])
    if anchors[0] != 0:
        raise SpearmanCalibrationError("identity-focused proposal did not anchor distance zero")
    if anchors[-1] != ordered_units[-1]:
        anchors = np.append(anchors, ordered_units[-1])
    cuts = (anchors[:-1] + anchors[1:]) // 2 + 1
    edges = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.unique(cuts).astype(np.int64, copy=False),
            np.asarray([ordered_units[-1] + 1], dtype=np.int64),
        )
    )
    edges = np.unique(edges)
    sample_bins = np.searchsorted(edges, sample_units, side="right") - 1
    active_bins = edges.size - 1
    validation_bins = np.searchsorted(edges, validation_units, side="right") - 1
    outside = (validation_bins < 0) | (validation_bins >= active_bins)
    if np.any(outside):
        # The squared Spearman multiplicities have an exact reverse-value
        # symmetry g(d) = g(D_max-d).  The symmetrization step below uses the
        # reflected sample range, so validation points in that range can use
        # the corresponding sampled bin for their local ESS diagnostic.
        reflected_units = maximum_theoretical // 2 - validation_units[outside]
        reflected_bins = np.searchsorted(edges, reflected_units, side="right") - 1
        reflected_valid = (reflected_bins >= 0) & (reflected_bins < active_bins)
        if not np.all(reflected_valid):
            raise SpearmanCalibrationError(
                "validation distances exceed the symmetrizable calibration range"
            )
        validation_bins = validation_bins.copy()
        validation_bins[outside] = reflected_bins
    log_sums = _grouped_log_sums(sample_bins, log_weights, active_bins)
    nonempty = np.isfinite(log_sums)
    if not nonempty[0] or not nonempty[-1]:
        raise SpearmanCalibrationError("calibration range has an unanchored endpoint")
    if not np.all(nonempty):
        raise SpearmanCalibrationError(
            "adaptive calibration unexpectedly produced an empty distance bin"
        )
    log_mass = log_sums - math.log(distances.size)
    log_mass -= float(np.max(log_mass))
    support = (2 * edges[:-1]).astype(np.int64, copy=False)
    upper = (2 * (edges[1:] - 1)).astype(np.int64, copy=False)

    log_square_sums = _grouped_log_sums(sample_bins, 2.0 * log_weights, active_bins)
    local_ess: list[float] = []
    for bin_index in np.unique(validation_bins):
        if not np.isfinite(log_square_sums[bin_index]):
            local_ess.append(0.0)
        else:
            local_ess.append(float(np.exp(2.0 * log_sums[bin_index] - log_square_sums[bin_index])))
    unique_units = np.unique(ordered_units)
    reflected_units = maximum_theoretical // 2 - unique_units
    covered_units = np.unique(np.concatenate((unique_units, reflected_units)))
    maximum_gap = 0 if covered_units.size < 2 else max(0, int(np.max(np.diff(covered_units))) - 1)
    return _BaseEstimate(
        support=support,
        upper=upper,
        log_mass=log_mass,
        empty_bin_run=_maximum_empty_run(nonempty),
        validation_bin_ess=min(local_ess),
        maximum_distance=int(upper[-1]),
        maximum_unsampled_gap_fraction=(2.0 * maximum_gap / maximum_theoretical),
        maximum_symmetry_residual=0.0,
        support_completion="sampled_range",
        full_support=False,
        original_maximum_distance=int(upper[-1]),
    )


def _symmetrize_estimate(
    estimate: _BaseEstimate,
    *,
    n_items: int,
) -> _BaseEstimate:
    """Complete a sampled density with the exact reverse-value symmetry.

    The raw importance estimate is a piecewise-constant *mass* over bins.
    Symmetry applies to the density of states, so masses are first divided by
    their bin widths.  Original and reflected bins are split on a common,
    symmetric boundary grid; in overlaps their density estimates are
    averaged, and outside the sampled range the reflected estimate supplies
    the missing values.
    """

    maximum_theoretical = _theoretical_maximum_distance(n_items)
    maximum_units = maximum_theoretical // 2
    lower = estimate.support // 2
    upper = estimate.upper // 2
    if (
        lower.size == 0
        or lower[0] != 0
        or np.any(lower[1:] != upper[:-1] + 1)
        or np.any(upper < lower)
        or upper[-1] > maximum_units
    ):
        raise SpearmanCalibrationError("sampled Spearman bins cannot be symmetrized")
    if estimate.maximum_distance < maximum_theoretical // 2:
        raise SpearmanCalibrationError(
            "sampled Spearman range does not reach the symmetry midpoint"
        )

    widths = upper - lower + 1
    log_density = estimate.log_mass - np.log(widths.astype(np.float64))
    boundaries = np.unique(
        np.concatenate(
            (
                np.asarray([0, maximum_units + 1], dtype=np.int64),
                lower,
                upper + 1,
                maximum_units - upper,
                maximum_units - lower + 1,
            )
        )
    )
    if boundaries[0] != 0 or boundaries[-1] != maximum_units + 1:
        raise SpearmanCalibrationError("symmetrized Spearman bins do not cover the full range")
    interval_lower = boundaries[:-1]
    interval_upper = boundaries[1:] - 1
    midpoints = (interval_lower + interval_upper) // 2

    def lookup(points: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        indices = np.searchsorted(upper, points, side="left")
        valid = (indices >= 0) & (indices < upper.size)
        valid &= points >= lower[np.minimum(indices, upper.size - 1)]
        values = np.full(points.shape, -np.inf, dtype=np.float64)
        values[valid] = log_density[indices[valid]]
        return values, valid

    direct, direct_valid = lookup(midpoints)
    reflected, reflected_valid = lookup(maximum_units - midpoints)
    if not np.all(direct_valid | reflected_valid):
        raise SpearmanCalibrationError("symmetry completion left an uncovered Spearman interval")

    both = direct_valid & reflected_valid
    symmetrized_density = np.full(midpoints.shape, -np.inf, dtype=np.float64)
    symmetrized_density[direct_valid & ~reflected_valid] = direct[direct_valid & ~reflected_valid]
    symmetrized_density[reflected_valid & ~direct_valid] = reflected[
        reflected_valid & ~direct_valid
    ]
    symmetrized_density[both] = np.logaddexp(direct[both], reflected[both]) - math.log(2.0)

    if np.any(both):
        log_difference = np.abs(direct[both] - reflected[both])
        # This is a scale-free residual in [0, 1] and remains stable for
        # densities represented in log space.
        residual = float(np.max(-np.expm1(-np.minimum(log_difference, 745.0))))
    else:
        residual = 0.0
    log_mass = symmetrized_density + np.log(
        (interval_upper - interval_lower + 1).astype(np.float64)
    )
    log_mass -= float(np.max(log_mass))
    support = (2 * interval_lower).astype(np.int64, copy=False)
    completed_upper = (2 * interval_upper).astype(np.int64, copy=False)
    return _BaseEstimate(
        support=support,
        upper=completed_upper,
        log_mass=log_mass,
        empty_bin_run=0,
        validation_bin_ess=estimate.validation_bin_ess,
        maximum_distance=maximum_theoretical,
        maximum_unsampled_gap_fraction=estimate.maximum_unsampled_gap_fraction,
        maximum_symmetry_residual=residual,
        support_completion="exact_reverse_symmetry",
        full_support=True,
        original_maximum_distance=estimate.original_maximum_distance,
    )


def _split_rhat(values: NDArray[np.float64]) -> float:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        return float("inf")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within_variances = np.var(split, axis=1, ddof=1)
    within = float(np.mean(within_variances))
    if within == 0.0:
        return 1.0 if np.all(split == split[0, 0]) else float("inf")
    means = np.mean(split, axis=1)
    between = half * float(np.var(means, ddof=1))
    variance = (half - 1.0) / half * within + between / half
    return float(np.sqrt(max(variance / within, 0.0)))


def _importance_ess(log_weights: NDArray[np.float64]) -> float:
    return float(np.exp(2.0 * logsumexp(log_weights) - logsumexp(2.0 * log_weights)))


def _exact_base(n_items: int) -> _BaseEstimate:
    counts: dict[int, int] = {}
    identity = np.arange(n_items, dtype=np.int64)
    for permutation in itertools.permutations(range(n_items)):
        delta = np.asarray(permutation, dtype=np.int64) - identity
        distance = int(np.dot(delta, delta))
        counts[distance] = counts.get(distance, 0) + 1
    support = np.asarray(sorted(counts), dtype=np.int64)
    weights = np.log(np.asarray([counts[int(item)] for item in support], dtype=np.float64))
    weights -= float(np.max(weights))
    return _BaseEstimate(
        support=support,
        upper=support.copy(),
        log_mass=weights,
        empty_bin_run=0,
        validation_bin_ess=float(math.factorial(n_items)),
        maximum_distance=int(support[-1]),
        maximum_unsampled_gap_fraction=0.0,
        maximum_symmetry_residual=0.0,
        support_completion="exact_enumeration",
        full_support=True,
        original_maximum_distance=int(support[-1]),
    )


def _family(
    estimate: _BaseEstimate,
    *,
    exact: bool,
) -> DiagnosticDiscreteExponentialFamily | BinnedDiscreteExponentialFamily:
    if exact:
        return DiagnosticDiscreteExponentialFamily(
            estimate.support,
            base_log_weights=estimate.log_mass,
            model_name="spearman_mallows_exact_distance",
            diagnostics={"exact": True, "converged": True, "ess": 1.0},
        )
    return BinnedDiscreteExponentialFamily(
        estimate.support,
        estimate.upper,
        base_log_weights=estimate.log_mass,
        model_name="spearman_mallows_multi_proposal_importance_binned_symmetric_approximation",
    )


def _array_digest(value: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _deterministic_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(
        payload, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for key in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(npy, np.asarray(arrays[key]), allow_pickle=False)
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    atomic_write_bytes(path, payload.getvalue())


def calibrate_spearman_mallows(
    validation_distances: ArrayLike,
    *,
    n_items: int,
    config: Mapping[str, Any],
    artifact_path: str | Path,
    seed: int = 0,
    bootstrap_replicates: int | None = None,
    projected_prefix_count: int | None = None,
) -> SpearmanCalibrationOutcome:
    """Build, diagnose, and atomically publish one locked family artifact."""

    if not isinstance(n_items, int) or n_items < 2:
        raise ValueError("n_items must be an integer >= 2")
    section = _section(config)
    pass_config = dict(section.get("pass", {}))
    limits = dict(section.get("resource_limits", {}))
    maximum_items = int(limits.get("maximum_items", 512))
    if n_items > maximum_items:
        raise SpearmanCalibrationError(f"n_items={n_items} exceeds resource limit {maximum_items}")
    validation = _integer_validation_distances(validation_distances, n_items=n_items)
    if validation.size > int(limits.get("maximum_validation_distances", 200_000)):
        raise SpearmanCalibrationError("validation distances exceed maximum_validation_distances")
    B = int(
        bootstrap_replicates
        if bootstrap_replicates is not None
        else section.get("bootstrap_replicates", 499)
    )
    prefix_count = int(
        projected_prefix_count
        if projected_prefix_count is not None
        else section.get("projected_prefix_count", 14)
    )
    if B <= 0 or prefix_count <= 0:
        raise ValueError("bootstrap_replicates and projected_prefix_count must be positive")
    if B > int(limits.get("maximum_bootstrap_replicates", 2_000)):
        raise SpearmanCalibrationError("bootstrap_replicates exceeds maximum_bootstrap_replicates")
    if prefix_count > int(limits.get("maximum_projected_prefix_count", 64)):
        raise SpearmanCalibrationError(
            "projected_prefix_count exceeds maximum_projected_prefix_count"
        )
    exact_max = int(section.get("exact_enumeration_max_items", 8))
    exact = n_items <= exact_max
    maximum_wall = float(limits.get("maximum_wall_seconds", 21_600))
    if maximum_wall <= 0:
        raise SpearmanCalibrationError("maximum_wall_seconds must be positive")
    deadline = time.monotonic() + maximum_wall
    started = time.perf_counter()

    failures: list[str] = []
    if exact:
        if math.factorial(n_items) > int(limits.get("maximum_exact_permutations", 1_000_000)):
            raise SpearmanCalibrationError("exact enumeration exceeds maximum_exact_permutations")
        estimate = _exact_base(n_items)
        first_estimate = second_estimate = estimate
        r_hat = 1.0
        ess = float(math.factorial(n_items))
        upper_tail_bound = 0.0
        approximation = "exact_enumeration"
        chain_count = 0
        initial_draws_per_chain = 0
        draws_per_chain = 0
        range_expansions = 0
        q_values = np.asarray([], dtype=np.float64)
    else:
        q_values = np.asarray(
            section.get(
                "proposal_q",
                [
                    0.0,
                    0.001,
                    0.005,
                    0.01,
                    0.03,
                    0.05,
                    0.1,
                    0.15,
                    0.25,
                    0.35,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.87,
                    0.92,
                    0.95,
                    0.97,
                    0.98,
                    0.99,
                    0.995,
                    0.998,
                    0.9995,
                    1.0,
                ],
            ),
            dtype=np.float64,
        )
        if (
            q_values.ndim != 1
            or q_values.size < 3
            or np.any(~np.isfinite(q_values))
            or np.any(q_values < 0)
            or np.any(q_values > 1)
            or np.unique(q_values).size != q_values.size
            or not np.any(q_values == 0.0)
            or not np.any(q_values == 1.0)
        ):
            raise SpearmanCalibrationError(
                "proposal_q must be unique values in [0,1] including 0 and 1"
            )
        q_values.sort()
        chain_count = int(section.get("chains", 4))
        initial_draws_per_chain = int(section.get("draws_per_chain", 7_200))
        draws_per_chain = initial_draws_per_chain
        range_expansions = 0
        if chain_count < 4 or chain_count % 2:
            raise SpearmanCalibrationError("importance calibration requires an even chains >= 4")
        if draws_per_chain < 2 * q_values.size:
            raise SpearmanCalibrationError("draws_per_chain is too small for proposal bank")
        maximum_total_draws = int(limits.get("maximum_total_draws", 2_000_000))
        maximum_working_bytes = int(limits.get("maximum_working_bytes", 2 * 1024**3))
        symmetry_midpoint = _theoretical_maximum_distance(n_items) // 2
        while True:
            total_draws = chain_count * draws_per_chain
            if total_draws > maximum_total_draws:
                raise SpearmanCalibrationError(
                    "validation range coverage exceeds maximum_total_draws"
                )
            estimated_bytes = total_draws * 40 + draws_per_chain * q_values.size * 8
            if estimated_bytes > maximum_working_bytes:
                raise SpearmanCalibrationError(
                    "validation range coverage exceeds maximum_working_bytes"
                )
            streams = [
                _sample_stream(
                    n_items=n_items,
                    q_values=q_values,
                    draws=draws_per_chain,
                    seed=stable_seed("spearman-mallows", seed, chain),
                    deadline=deadline,
                )
                for chain in range(chain_count)
            ]
            distance_matrix = np.stack([item[0] for item in streams])
            midpoint = chain_count // 2
            split_maxima = (
                int(np.max(distance_matrix[:midpoint])),
                int(np.max(distance_matrix[midpoint:])),
            )
            if min(split_maxima) >= symmetry_midpoint:
                break
            draws_per_chain *= 2
            range_expansions += 1
        log_weight_matrix = np.stack([item[2] for item in streams])
        component_matrix = np.stack([item[3] for item in streams])
        all_distances = distance_matrix.reshape(-1)
        all_log_weights = log_weight_matrix.reshape(-1)
        histogram_bins = int(section.get("histogram_bins", 512))
        if histogram_bins < 8 or histogram_bins > 65_536:
            raise SpearmanCalibrationError("histogram_bins must lie in [8, 65536]")
        estimate = _symmetrize_estimate(
            _estimate_base(
                all_distances,
                all_log_weights,
                validation,
                n_items=n_items,
                requested_bins=histogram_bins,
            ),
            n_items=n_items,
        )
        first_estimate = _symmetrize_estimate(
            _estimate_base(
                distance_matrix[:midpoint].reshape(-1),
                log_weight_matrix[:midpoint].reshape(-1),
                validation,
                n_items=n_items,
                requested_bins=histogram_bins,
            ),
            n_items=n_items,
        )
        second_estimate = _symmetrize_estimate(
            _estimate_base(
                distance_matrix[midpoint:].reshape(-1),
                log_weight_matrix[midpoint:].reshape(-1),
                validation,
                n_items=n_items,
                requested_bins=histogram_bins,
            ),
            n_items=n_items,
        )
        maximum_distance = _theoretical_maximum_distance(n_items)
        r_hat = max(
            _split_rhat(distance_matrix.astype(np.float64) / maximum_distance),
            _split_rhat(log_weight_matrix),
        )
        ess = _importance_ess(all_log_weights)
        uniform_index = int(np.flatnonzero(q_values == 1.0)[0])
        uniform_draws = int(np.count_nonzero(component_matrix == uniform_index))
        upper_tail_bound = 1.0 - float(pass_config.get("upper_tail_confidence_alpha", 0.05)) ** (
            1.0 / uniform_draws
        )
        approximation = "multi_proposal_importance_binned_symmetric"

    first_family = _family(first_estimate, exact=exact)
    second_family = _family(second_estimate, exact=exact)
    repeated_seed = stable_seed("spearman-repeat-gof", seed)
    first_gof = discrete_parametric_bootstrap_gof(
        validation, first_family, B=B, seed=repeated_seed, refit=True
    )
    second_gof = discrete_parametric_bootstrap_gof(
        validation, second_family, B=B, seed=repeated_seed, refit=True
    )
    repeated_difference = abs(first_gof.p_value - second_gof.p_value)

    combined_family = _family(estimate, exact=exact)
    benchmark_start = time.perf_counter()
    combined_gof = discrete_parametric_bootstrap_gof(
        validation,
        combined_family,
        B=B,
        seed=stable_seed("spearman-combined-gof", seed),
        refit=True,
    )
    benchmark_seconds = time.perf_counter() - benchmark_start
    projected_seconds = benchmark_seconds * prefix_count
    elapsed = time.perf_counter() - started

    r_hat_max = float(pass_config.get("r_hat_max", 1.05))
    ess_min = float(pass_config.get("effective_sample_size_min", 1_000))
    repeated_max = float(pass_config.get("repeated_p_value_absolute_difference_max", 0.02))
    local_ess_min = float(pass_config.get("validation_bin_effective_sample_size_min", 1))
    maximum_empty = int(pass_config.get("maximum_interpolated_empty_bin_run", 4))
    maximum_gap_fraction = float(pass_config.get("maximum_unsampled_distance_gap_fraction", 0.05))
    tail_max = float(pass_config.get("upper_tail_probability_bound_max", 0.01))
    checks = {
        "r_hat": r_hat <= r_hat_max,
        "effective_sample_size": ess >= ess_min,
        "repeated_p_value": repeated_difference <= repeated_max,
        "validation_bin_effective_sample_size": (
            exact or estimate.validation_bin_ess >= local_ess_min
        ),
        "interpolated_empty_bin_run": exact or estimate.empty_bin_run <= maximum_empty,
        "unsampled_distance_gap": (
            exact or estimate.maximum_unsampled_gap_fraction <= maximum_gap_fraction
        ),
        "upper_tail_probability_bound": exact or upper_tail_bound <= tail_max,
        "validation_range": int(np.max(validation)) <= estimate.maximum_distance,
        "wall_time": elapsed <= maximum_wall,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    converged = not failures
    config_digest = object_sha256(dict(config))
    validation_digest = _array_digest(validation)
    identity = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "algorithm": approximation,
        "n_items": n_items,
        "theoretical_maximum_distance": _theoretical_maximum_distance(n_items),
        "support_completion": estimate.support_completion,
        "full_support": estimate.full_support,
        "seed": seed,
        "config_digest": config_digest,
        "validation_distances_digest": validation_digest,
        "support_digest": _array_digest(estimate.support),
        "upper_digest": _array_digest(estimate.upper),
        "base_log_weights_digest": _array_digest(estimate.log_mass),
        "maximum_symmetry_residual": estimate.maximum_symmetry_residual,
        "r_hat": r_hat,
        "ess": ess,
        "repeated_p_value_absolute_difference": repeated_difference,
        "converged": converged,
        "initial_draws_per_chain": initial_draws_per_chain,
        "draws_per_chain": draws_per_chain,
        "range_expansions": range_expansions,
    }
    pilot_digest = object_sha256(identity)
    arrays: dict[str, NDArray[Any]] = {
        "artifact_schema_version": np.asarray([ARTIFACT_SCHEMA_VERSION], dtype=np.int64),
        "support": estimate.support,
        "bin_upper": estimate.upper,
        "base_log_weights": estimate.log_mass,
        "n_items": np.asarray([n_items], dtype=np.int64),
        "theoretical_maximum_distance": np.asarray(
            [_theoretical_maximum_distance(n_items)], dtype=np.int64
        ),
        "maximum_calibrated_distance": np.asarray([estimate.maximum_distance], dtype=np.int64),
        "original_maximum_distance": np.asarray(
            [estimate.original_maximum_distance], dtype=np.int64
        ),
        "full_support": np.asarray([estimate.full_support], dtype=np.bool_),
        "support_completion": np.asarray([estimate.support_completion]),
        "maximum_symmetry_residual": np.asarray(
            [estimate.maximum_symmetry_residual], dtype=np.float64
        ),
        "pilot_digest": np.asarray([pilot_digest]),
        "converged": np.asarray([converged], dtype=np.bool_),
        "ess": np.asarray([ess], dtype=np.float64),
        "r_hat": np.asarray([r_hat], dtype=np.float64),
        "repeated_p_value_absolute_difference": np.asarray([repeated_difference], dtype=np.float64),
        "approximation": np.asarray([approximation]),
        "seed": np.asarray([seed], dtype=np.int64),
        "exact": np.asarray([exact], dtype=np.bool_),
        "proposal_q": q_values,
        "chains": np.asarray([chain_count], dtype=np.int64),
        "draws_per_chain": np.asarray([draws_per_chain], dtype=np.int64),
        "initial_draws_per_chain": np.asarray([initial_draws_per_chain], dtype=np.int64),
        "range_expansions": np.asarray([range_expansions], dtype=np.int64),
        "validation_bin_ess": np.asarray([estimate.validation_bin_ess], dtype=np.float64),
        "maximum_interpolated_empty_bin_run": np.asarray([estimate.empty_bin_run], dtype=np.int64),
        "upper_tail_probability_bound": np.asarray([upper_tail_bound], dtype=np.float64),
        "maximum_unsampled_distance_gap_fraction": np.asarray(
            [estimate.maximum_unsampled_gap_fraction], dtype=np.float64
        ),
        "config_digest": np.asarray([config_digest]),
        "validation_distances_digest": np.asarray([validation_digest]),
    }
    uncompressed_bytes = sum(int(value.nbytes) for value in arrays.values())
    if uncompressed_bytes > int(limits.get("maximum_artifact_bytes", 64 * 1024**2)):
        raise SpearmanCalibrationError(
            "calibrated family exceeds maximum_artifact_bytes before compression"
        )
    destination = Path(artifact_path).expanduser().resolve()
    _deterministic_npz(destination, arrays)
    artifact_sha = file_sha256(destination)
    measurement: dict[str, Any] = {
        "B": B,
        "measured_seconds": benchmark_seconds,
        "projected_seconds": projected_seconds,
        "projected_prefix_count": prefix_count,
        "deterministic_seed": True,
        "spearman_family_adapter": OFFICIAL_ADAPTER,
        "spearman_converged": converged,
        "spearman_r_hat": r_hat,
        "spearman_ess": ess,
        "spearman_repeated_p_value_absolute_difference": repeated_difference,
        "spearman_pilot_digest": pilot_digest,
        "spearman_approximation": approximation,
        "spearman_support_completion": estimate.support_completion,
        "spearman_full_support": estimate.full_support,
        "spearman_theoretical_maximum_distance": _theoretical_maximum_distance(n_items),
        "spearman_maximum_symmetry_residual": estimate.maximum_symmetry_residual,
        "spearman_config": {
            "path": str(destination),
            "sha256": artifact_sha,
            "r_hat_max": r_hat_max,
            "repeated_p_value_absolute_difference_max": repeated_max,
            "validation_bin_effective_sample_size_min": local_ess_min,
            "maximum_unsampled_distance_gap_fraction": maximum_gap_fraction,
            "upper_tail_probability_bound_max": tail_max,
        },
        "diagnostics": {
            "exact": exact,
            "combined_validation_p_value": combined_gof.p_value,
            "first_split_validation_p_value": first_gof.p_value,
            "second_split_validation_p_value": second_gof.p_value,
            "validation_bin_effective_sample_size": estimate.validation_bin_ess,
            "maximum_interpolated_empty_bin_run": estimate.empty_bin_run,
            "maximum_unsampled_distance_gap_fraction": (estimate.maximum_unsampled_gap_fraction),
            "upper_tail_probability_bound": upper_tail_bound,
            "maximum_calibrated_distance": estimate.maximum_distance,
            "theoretical_maximum_distance": _theoretical_maximum_distance(n_items),
            "original_maximum_distance": estimate.original_maximum_distance,
            "support_completion": estimate.support_completion,
            "full_support": estimate.full_support,
            "maximum_symmetry_residual": estimate.maximum_symmetry_residual,
            "artifact_uncompressed_bytes": uncompressed_bytes,
            "validation_distance_count": int(validation.size),
            "validation_distances_digest": validation_digest,
            "initial_draws_per_chain": initial_draws_per_chain,
            "draws_per_chain": draws_per_chain,
            "range_expansions": range_expansions,
            "wall_seconds": elapsed,
            "checks": checks,
            "failures": failures,
        },
    }
    return SpearmanCalibrationOutcome(
        artifact_path=destination,
        measurement=measurement,
        converged=converged,
        failures=tuple(failures),
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "OFFICIAL_ADAPTER",
    "SpearmanCalibrationError",
    "SpearmanCalibrationOutcome",
    "calibrate_spearman_mallows",
    "synthetic_validation_distances",
]
