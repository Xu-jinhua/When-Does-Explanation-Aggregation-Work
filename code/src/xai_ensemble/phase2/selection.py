"""Global in-sample NOISE prefix selection and discrete bootstrap diagnostics.

The user's locked protocol deliberately selects on the complete test pool.  The
result objects therefore carry ``scope='global_in_sample'`` so downstream
tables cannot accidentally describe the estimates as held-out performance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xai_ensemble.core.hashing import file_sha256

from .aggregation import borda, kemeny_young
from .rankings import (
    canonicalize_rankings,
    kendall_distance,
    spearman_distance,
)


def _integer_distances(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("distances must be a non-empty one-dimensional array")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise TypeError("distances must contain non-negative integers")
    numeric = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("distances must contain finite integers")
    result = numeric.astype(np.int64)
    if np.any(result < 0):
        raise ValueError("distances must be non-negative")
    return result


class FittedDiscreteModel(Protocol):
    model_name: str
    parameters: Mapping[str, float | str | bool]

    def cdf(self, points: ArrayLike) -> NDArray[np.float64]: ...

    def sample(self, size: int, rng: np.random.Generator) -> NDArray[np.int64]: ...


class DiscreteModelFamily(Protocol):
    def fit(self, distances: ArrayLike) -> FittedDiscreteModel: ...


@dataclass(frozen=True)
class FittedFiniteDiscreteModel:
    support: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    model_name: str
    parameters: Mapping[str, float | str | bool]

    def cdf(self, points: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(points)
        cumulative = np.cumsum(self.probabilities)
        positions = np.searchsorted(self.support, values, side="right") - 1
        result = np.zeros(values.shape, dtype=np.float64)
        valid = positions >= 0
        result[valid] = cumulative[
            np.minimum(positions[valid], cumulative.size - 1)
        ]
        result[positions >= cumulative.size] = 1.0
        return result

    def sample(self, size: int, rng: np.random.Generator) -> NDArray[np.int64]:
        if size <= 0:
            raise ValueError("sample size must be positive")
        return rng.choice(self.support, size=size, p=self.probabilities).astype(
            np.int64, copy=False
        )


class FiniteDiscreteExponentialFamily:
    """Finite discrete exponential family with a caller-declared base measure.

    ``P(D=d)`` is proportional to ``base_weight[d] * exp(-theta*d)``.
    Supplying correct permutation-distance multiplicities yields the intended
    distance model.  Uniform base weights are a *distance-support surrogate*,
    not an exact Mallows model; that distinction is exposed in ``model_name``.
    """

    def __init__(
        self,
        support: ArrayLike,
        *,
        base_log_weights: ArrayLike | None = None,
        model_name: str = "finite_discrete_exponential",
    ) -> None:
        support_array = _integer_distances(support)
        if np.unique(support_array).size != support_array.size:
            raise ValueError("support values must be unique")
        order = np.argsort(support_array)
        self.support = support_array[order]
        if base_log_weights is None:
            self.base_log_weights = np.zeros(self.support.size, dtype=np.float64)
        else:
            weights = np.asarray(base_log_weights, dtype=np.float64)
            if weights.shape != self.support.shape:
                raise ValueError("base_log_weights and support must have equal shape")
            weights = weights[order]
            if np.any(np.isnan(weights)) or np.all(np.isneginf(weights)):
                raise ValueError("base_log_weights must contain valid positive mass")
            self.base_log_weights = weights
        self.model_name = model_name

    def _probabilities(self, theta: float) -> NDArray[np.float64]:
        log_mass = self.base_log_weights - theta * self.support
        finite = np.isfinite(log_mass)
        maximum = float(np.max(log_mass[finite]))
        mass = np.zeros(log_mass.shape, dtype=np.float64)
        mass[finite] = np.exp(log_mass[finite] - maximum)
        return mass / np.sum(mass)

    def fit(self, distances: ArrayLike) -> FittedFiniteDiscreteModel:
        observed = _integer_distances(distances)
        if np.any(~np.isin(observed, self.support)):
            raise ValueError("observed distance lies outside the declared support")
        target_mean = float(np.mean(observed))
        probabilities_zero = self._probabilities(0.0)
        maximum_mean = float(np.dot(probabilities_zero, self.support))
        boundary = False
        if target_mean >= maximum_mean:
            theta = 0.0
            boundary = target_mean > maximum_mean + 1e-12
        else:
            low = 0.0
            high = 1.0
            while float(np.dot(self._probabilities(high), self.support)) > target_mean:
                high *= 2.0
                if high >= 1_048_576.0:
                    break
            for _ in range(80):
                middle = (low + high) / 2.0
                mean = float(np.dot(self._probabilities(middle), self.support))
                if mean > target_mean:
                    low = middle
                else:
                    high = middle
            theta = (low + high) / 2.0
        probabilities = self._probabilities(theta)
        return FittedFiniteDiscreteModel(
            support=self.support,
            probabilities=probabilities,
            model_name=self.model_name,
            parameters={"theta": theta, "boundary": boundary},
        )


class DiagnosticDiscreteExponentialFamily(FiniteDiscreteExponentialFamily):
    """Finite approximation carrying mandatory pilot diagnostics."""

    def __init__(self, *args: Any, diagnostics: Mapping[str, float | str | bool], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.diagnostics = dict(diagnostics)

    def fit(self, distances: ArrayLike) -> FittedFiniteDiscreteModel:
        fitted = super().fit(distances)
        return FittedFiniteDiscreteModel(
            support=fitted.support,
            probabilities=fitted.probabilities,
            model_name=fitted.model_name,
            parameters={**fitted.parameters, **self.diagnostics},
        )


@dataclass(frozen=True)
class FittedBinnedDiscreteModel:
    """Exponentially tilted even-integer law with a piecewise-constant base.

    ``bin_probabilities`` are probabilities of inclusive even-distance bins.
    Within a bin the *base measure* is constant; the fitted exponential tilt
    is nevertheless applied to every individual distance.  This keeps the
    approximation explicit while avoiding a multi-million-element support for
    the 256-patch Spearman distance.
    """

    bin_lower: NDArray[np.int64]
    bin_upper: NDArray[np.int64]
    bin_probabilities: NDArray[np.float64]
    theta: float
    model_name: str
    parameters: Mapping[str, float | str | bool]

    def cdf(self, points: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        flat = values.reshape(-1)
        result = np.zeros(flat.shape, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(self.bin_probabilities)))
        upper_units = self.bin_upper // 2
        lower_units = self.bin_lower // 2
        point_units = np.floor(flat / 2.0).astype(np.int64)
        bins = np.searchsorted(upper_units, point_units, side="left")
        above = bins >= self.bin_lower.size
        result[above] = 1.0
        valid = (~above) & (point_units >= lower_units[0])
        if np.any(valid):
            selected = bins[valid]
            offsets = point_units[valid] - lower_units[selected]
            widths = upper_units[selected] - lower_units[selected] + 1
            result[valid] = cumulative[selected] + self.bin_probabilities[
                selected
            ] * _truncated_geometric_cdf(offsets, widths, 2.0 * self.theta)
        return result.reshape(values.shape)

    def sample(self, size: int, rng: np.random.Generator) -> NDArray[np.int64]:
        if size <= 0:
            raise ValueError("sample size must be positive")
        bins = rng.choice(
            self.bin_lower.size, size=size, p=self.bin_probabilities
        ).astype(np.int64, copy=False)
        lower_units = self.bin_lower[bins] // 2
        widths = (self.bin_upper[bins] - self.bin_lower[bins]) // 2 + 1
        offsets = _sample_truncated_geometric(
            widths, 2.0 * self.theta, rng
        )
        return (2 * (lower_units + offsets)).astype(np.int64, copy=False)


def _truncated_geometric_cdf(
    offsets: NDArray[np.int64],
    widths: NDArray[np.int64],
    rate: float,
) -> NDArray[np.float64]:
    if rate <= 1e-12:
        return (offsets + 1) / widths
    numerator = -np.expm1(-rate * (offsets + 1))
    denominator = -np.expm1(-rate * widths)
    return np.minimum(1.0, numerator / denominator)


def _sample_truncated_geometric(
    widths: NDArray[np.int64],
    rate: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    uniforms = rng.random(widths.size)
    if rate <= 1e-12:
        return np.floor(uniforms * widths).astype(np.int64)
    # Invert (1-exp(-rate*(v+1))) / (1-exp(-rate*width)).
    logarithm = np.log1p(uniforms * np.expm1(-rate * widths))
    offsets = np.ceil(-logarithm / rate).astype(np.int64) - 1
    return np.minimum(np.maximum(offsets, 0), widths - 1)


class BinnedDiscreteExponentialFamily:
    """Piecewise-constant base measure on contiguous even-distance bins."""

    def __init__(
        self,
        bin_lower: ArrayLike,
        bin_upper: ArrayLike,
        *,
        base_log_weights: ArrayLike,
        model_name: str,
        diagnostics: Mapping[str, float | str | bool] | None = None,
    ) -> None:
        lower = _integer_distances(bin_lower)
        upper = _integer_distances(bin_upper)
        weights = np.asarray(base_log_weights, dtype=np.float64)
        if lower.shape != upper.shape or weights.shape != lower.shape:
            raise ValueError("bin bounds and base_log_weights must have equal shape")
        if np.any(lower % 2) or np.any(upper % 2):
            raise ValueError("Spearman distance bins must have even endpoints")
        if np.any(upper < lower):
            raise ValueError("distance-bin upper endpoints precede lower endpoints")
        if lower[0] != 0 or np.any(lower[1:] != upper[:-1] + 2):
            raise ValueError("distance bins must be contiguous and begin at zero")
        if np.any(np.isnan(weights)) or np.all(np.isneginf(weights)):
            raise ValueError("base_log_weights must contain valid positive mass")
        self.bin_lower = lower
        self.bin_upper = upper
        self.base_log_weights = weights
        self.model_name = model_name
        self.diagnostics = dict(diagnostics or {})

    def _log_bin_mass(self, theta: float) -> NDArray[np.float64]:
        widths = (self.bin_upper - self.bin_lower) // 2 + 1
        lower_units = self.bin_lower // 2
        if theta <= 1e-12:
            return self.base_log_weights.copy()
        rate = 2.0 * theta
        log_geometric_sum = (
            -rate * lower_units
            + np.log(-np.expm1(-rate * widths))
            - np.log(-np.expm1(-rate))
        )
        return self.base_log_weights - np.log(widths) + log_geometric_sum

    def _probabilities_and_mean(
        self, theta: float
    ) -> tuple[NDArray[np.float64], float]:
        log_mass = self._log_bin_mass(theta)
        finite = np.isfinite(log_mass)
        maximum = float(np.max(log_mass[finite]))
        mass = np.zeros(log_mass.shape, dtype=np.float64)
        mass[finite] = np.exp(log_mass[finite] - maximum)
        probabilities = mass / np.sum(mass)
        widths = (self.bin_upper - self.bin_lower) // 2 + 1
        rate = 2.0 * theta
        if rate <= 1e-7:
            offsets = (
                (widths - 1.0) / 2.0
                - (widths.astype(np.float64) ** 2 - 1.0) * rate / 12.0
            )
        else:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                offsets = 1.0 / np.expm1(rate) - widths / np.expm1(
                    widths * rate
                )
            offsets = np.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0)
        bin_means = self.bin_lower + 2.0 * offsets
        return probabilities, float(np.dot(probabilities, bin_means))

    def fit(self, distances: ArrayLike) -> FittedBinnedDiscreteModel:
        observed = _integer_distances(distances)
        if np.any(observed % 2):
            raise ValueError("squared Spearman distances must be even")
        if np.max(observed) > self.bin_upper[-1]:
            raise ValueError(
                "observed distance exceeds the pilot-calibrated Spearman range"
            )
        target_mean = float(np.mean(observed))
        _, maximum_mean = self._probabilities_and_mean(0.0)
        boundary = target_mean > maximum_mean + 1e-12
        if target_mean >= maximum_mean:
            theta = 0.0
        else:
            low = 0.0
            high = 1.0
            for _ in range(64):
                _, mean = self._probabilities_and_mean(high)
                if mean <= target_mean or high >= 1_048_576.0:
                    break
                high *= 2.0
            for _ in range(80):
                middle = (low + high) / 2.0
                _, mean = self._probabilities_and_mean(middle)
                if mean > target_mean:
                    low = middle
                else:
                    high = middle
            theta = (low + high) / 2.0
        probabilities, _ = self._probabilities_and_mean(theta)
        return FittedBinnedDiscreteModel(
            bin_lower=self.bin_lower,
            bin_upper=self.bin_upper,
            bin_probabilities=probabilities,
            theta=theta,
            model_name=self.model_name,
            parameters={
                "theta": theta,
                "boundary": boundary,
                "bin_count": int(self.bin_lower.size),
                "maximum_calibrated_distance": int(self.bin_upper[-1]),
                **self.diagnostics,
            },
        )


def locked_spearman_family_factory(
    *,
    distance: str,
    n_items: int,
    seed: int,
    pilot_digest: str | None,
    config: Mapping[str, Any],
) -> DiscreteModelFamily:
    """Load a reproducible pilot-built Spearman base-measure approximation.

    The NPZ must contain ``support`` and ``base_log_weights`` plus scalar
    ``n_items``, ``pilot_digest``, ``converged``, ``ess``, ``approximation``,
    and ``seed``.  The formal producer estimates the Spearman permutation
    distance base measure with independent, exactly normalized proposal laws
    and balance-heuristic importance weights.  This loader refuses missing
    convergence diagnostics and never upgrades an approximation to exact.
    """

    if distance != "spearman":
        raise ValueError("locked Spearman family cannot model another distance")
    path = Path(str(config.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError("locked Spearman pilot NPZ is missing")
    expected_sha = str(config.get("sha256", ""))
    if len(expected_sha) != 64 or file_sha256(path) != expected_sha:
        raise ValueError("locked Spearman pilot checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "artifact_schema_version",
            "support",
            "bin_upper",
            "base_log_weights",
            "n_items",
            "theoretical_maximum_distance",
            "maximum_calibrated_distance",
            "original_maximum_distance",
            "full_support",
            "support_completion",
            "maximum_symmetry_residual",
            "pilot_digest",
            "converged",
            "ess",
            "r_hat",
            "repeated_p_value_absolute_difference",
            "validation_bin_ess",
            "maximum_unsampled_distance_gap_fraction",
            "upper_tail_probability_bound",
            "config_digest",
            "validation_distances_digest",
            "approximation",
            "seed",
            "exact",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"locked Spearman pilot is missing diagnostics: {missing}"
            )
        schema = int(np.asarray(archive["artifact_schema_version"]).reshape(-1)[0])
        support = np.asarray(archive["support"])
        bin_upper = np.asarray(archive["bin_upper"])
        weights = np.asarray(archive["base_log_weights"], dtype=np.float64)
        stored_items = int(np.asarray(archive["n_items"]).reshape(-1)[0])
        theoretical_maximum = int(
            np.asarray(archive["theoretical_maximum_distance"]).reshape(-1)[0]
        )
        maximum_calibrated = int(
            np.asarray(archive["maximum_calibrated_distance"]).reshape(-1)[0]
        )
        original_maximum = int(
            np.asarray(archive["original_maximum_distance"]).reshape(-1)[0]
        )
        full_support = bool(np.asarray(archive["full_support"]).reshape(-1)[0])
        support_completion = str(
            np.asarray(archive["support_completion"]).reshape(-1)[0]
        )
        symmetry_residual = float(
            np.asarray(archive["maximum_symmetry_residual"]).reshape(-1)[0]
        )
        stored_digest = str(np.asarray(archive["pilot_digest"]).reshape(-1)[0])
        converged = bool(np.asarray(archive["converged"]).reshape(-1)[0])
        ess = float(np.asarray(archive["ess"]).reshape(-1)[0])
        r_hat = float(np.asarray(archive["r_hat"]).reshape(-1)[0])
        repeated_difference = float(
            np.asarray(archive["repeated_p_value_absolute_difference"]).reshape(-1)[0]
        )
        validation_bin_ess = float(
            np.asarray(archive["validation_bin_ess"]).reshape(-1)[0]
        )
        maximum_gap_fraction = float(
            np.asarray(archive["maximum_unsampled_distance_gap_fraction"])
            .reshape(-1)[0]
        )
        upper_tail_bound = float(
            np.asarray(archive["upper_tail_probability_bound"]).reshape(-1)[0]
        )
        config_digest = str(np.asarray(archive["config_digest"]).reshape(-1)[0])
        validation_digest = str(
            np.asarray(archive["validation_distances_digest"]).reshape(-1)[0]
        )
        approximation = str(np.asarray(archive["approximation"]).reshape(-1)[0])
        pilot_seed = int(np.asarray(archive["seed"]).reshape(-1)[0])
        exact = bool(np.asarray(archive["exact"]).reshape(-1)[0])
    if schema != 3:
        raise ValueError(f"unsupported locked Spearman artifact schema {schema}")
    if stored_items != n_items or stored_digest != pilot_digest:
        raise ValueError("locked Spearman pilot identity mismatch")
    expected_maximum = n_items * (n_items * n_items - 1) // 3
    if (
        theoretical_maximum != expected_maximum
        or maximum_calibrated != expected_maximum
        or not full_support
        or not np.isfinite(symmetry_residual)
        or symmetry_residual < 0
        or symmetry_residual > 1
        or original_maximum < 0
        or original_maximum > expected_maximum
        or support.ndim != 1
        or not np.issubdtype(support.dtype, np.integer)
        or not np.issubdtype(bin_upper.dtype, np.integer)
        or bin_upper.shape != support.shape
        or weights.shape != support.shape
        or support.size == 0
        or np.any(np.isnan(weights))
        or not np.any(np.isfinite(weights))
        or np.any(support[1:] <= support[:-1])
        or np.any(support % 2)
        or np.any(bin_upper % 2)
        or np.any(bin_upper < support)
        or support[0] != 0
        or bin_upper[-1] != expected_maximum
    ):
        raise ValueError("locked Spearman artifact does not declare the full theoretical support")
    if exact:
        if (
            approximation != "exact_enumeration"
            or support_completion != "exact_enumeration"
            or original_maximum != expected_maximum
            or not np.array_equal(support, bin_upper)
            or not np.array_equal(support, expected_maximum - support[::-1])
            or not np.allclose(weights, weights[::-1], rtol=0.0, atol=1e-12)
        ):
            raise ValueError("exact Spearman artifact has an invalid support completion")
    else:
        if (
            approximation != "multi_proposal_importance_binned_symmetric"
            or support_completion != "exact_reverse_symmetry"
            or original_maximum < expected_maximum // 2
        ):
            raise ValueError("approximate Spearman artifact lacks exact reverse symmetry completion")
        if np.any(support[1:] != bin_upper[:-1] + 2):
            raise ValueError("approximate Spearman artifact bins do not cover every even distance")
        lower_units = support // 2
        upper_units = bin_upper // 2
        maximum_units = expected_maximum // 2
        log_density = weights - np.log((upper_units - lower_units + 1).astype(np.float64))
        if (
            not np.array_equal(lower_units, maximum_units - upper_units[::-1])
            or not np.array_equal(upper_units, maximum_units - lower_units[::-1])
            or not np.allclose(log_density, log_density[::-1], rtol=0.0, atol=1e-12)
        ):
            raise ValueError("approximate Spearman artifact density is not exactly symmetric")
    r_hat_max = float(config.get("r_hat_max", 1.05))
    repeated_max = float(
        config.get("repeated_p_value_absolute_difference_max", 0.02)
    )
    validation_ess_min = float(
        config.get("validation_bin_effective_sample_size_min", 1.0)
    )
    maximum_gap = float(
        config.get("maximum_unsampled_distance_gap_fraction", 0.05)
    )
    maximum_tail = float(config.get("upper_tail_probability_bound_max", 0.01))
    if (
        not converged
        or not np.isfinite(ess)
        or ess <= 0
        or not np.isfinite(r_hat)
        or r_hat <= 0
        or r_hat > r_hat_max
        or not np.isfinite(repeated_difference)
        or repeated_difference < 0
        or repeated_difference > repeated_max
        or not np.isfinite(validation_bin_ess)
        or validation_bin_ess < validation_ess_min
        or not np.isfinite(maximum_gap_fraction)
        or maximum_gap_fraction < 0
        or maximum_gap_fraction > maximum_gap
        or not np.isfinite(upper_tail_bound)
        or upper_tail_bound < 0
        or upper_tail_bound > maximum_tail
    ):
        raise ValueError("locked Spearman approximation did not converge")
    allowed_approximations = {
        "exact_enumeration",
        "multi_proposal_importance_binned_symmetric",
    }
    if (
        len(stored_digest) != 64
        or len(config_digest) != 64
        or len(validation_digest) != 64
        or approximation not in allowed_approximations
    ):
        raise ValueError("unrecognized locked Spearman estimator identity")
    diagnostics: dict[str, float | str | bool] = {
        "exact": exact,
        "converged": converged,
        "ess": ess,
        "r_hat": r_hat,
        "repeated_p_value_absolute_difference": repeated_difference,
        "validation_bin_ess": validation_bin_ess,
        "maximum_unsampled_distance_gap_fraction": maximum_gap_fraction,
        "upper_tail_probability_bound": upper_tail_bound,
        "calibration_config_digest": config_digest,
        "validation_distances_digest": validation_digest,
        "pilot_digest": stored_digest,
        "pilot_seed": pilot_seed,
        "runtime_seed": seed,
        "approximation": approximation,
        "support_completion": support_completion,
        "full_support": full_support,
        "theoretical_maximum_distance": theoretical_maximum,
        "maximum_calibrated_distance": maximum_calibrated,
        "original_maximum_distance": original_maximum,
        "maximum_symmetry_residual": symmetry_residual,
    }
    model_name = (
        "spearman_mallows_exact_distance"
        if exact
        else f"spearman_mallows_{approximation}_approximation"
    )
    if np.array_equal(support, bin_upper):
        return DiagnosticDiscreteExponentialFamily(
            support,
            base_log_weights=weights,
            model_name=model_name,
            diagnostics=diagnostics,
        )
    return BinnedDiscreteExponentialFamily(
        support,
        bin_upper,
        base_log_weights=weights,
        model_name=model_name,
        diagnostics=diagnostics,
    )


def _expected_kendall_distance(n_items: int, q: float) -> float:
    if q <= 0:
        return 0.0
    if q >= 1:
        return n_items * (n_items - 1) / 4.0
    # For inversion component V_i in {0, ..., i-1},
    # E[V_i] = 1/expm1(theta) - i/expm1(i*theta), theta=-log(q).
    # The vectorized form matters because bootstrap GOF refits q B times.
    theta = -log(q)
    widths = np.arange(2, n_items + 1, dtype=np.float64)
    if theta < 1e-7:
        # First-order expansion around the uniform q=1 boundary avoids
        # catastrophic cancellation between the two reciprocal terms.
        terms = (widths - 1.0) / 2.0 - (widths * widths - 1.0) * theta / 12.0
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            terms = 1.0 / np.expm1(theta) - widths / np.expm1(widths * theta)
    return float(np.sum(terms))


def _kendall_pmf(n_items: int, q: float) -> NDArray[np.float64]:
    """Exact Kendall-Mallows distance PMF via independent inversions."""

    distribution = np.ones(1, dtype=np.float64)
    if q <= 0:
        result = np.zeros(n_items * (n_items - 1) // 2 + 1, dtype=np.float64)
        result[0] = 1.0
        return result
    for width in range(2, n_items + 1):
        new_size = distribution.size + width - 1
        if q >= 1.0 - 1e-14:
            prefix = np.concatenate(([0.0], np.cumsum(distribution)))
            positions = np.arange(new_size)
            starts = np.maximum(0, positions - width + 1)
            ends = np.minimum(distribution.size, positions + 1)
            updated = (prefix[ends] - prefix[starts]) / width
        else:
            updated = np.empty(new_size, dtype=np.float64)
            normalizer = (1.0 - q**width) / (1.0 - q)
            q_width = q**width
            running = 0.0
            for distance in range(new_size):
                incoming = (
                    float(distribution[distance])
                    if distance < distribution.size
                    else 0.0
                )
                outgoing = (
                    q_width * float(distribution[distance - width])
                    if distance >= width
                    else 0.0
                )
                running = incoming + q * running - outgoing
                updated[distance] = max(0.0, running / normalizer)
        total = float(np.sum(updated))
        distribution = updated / total
    return distribution


class KendallMallowsFamily:
    """Exact one-parameter Kendall-Mallows distance family.

    The distance law factorizes into independent truncated-geometric inversion
    counts.  The first fit computes that law once.  Bootstrap refits are exact
    exponential tilts of the cached law, avoiding an expensive dynamic program
    for every bootstrap replicate.
    """

    def __init__(self, n_items: int) -> None:
        if not isinstance(n_items, (int, np.integer)) or n_items <= 0:
            raise ValueError("n_items must be a positive integer")
        self.n_items = int(n_items)
        self.support = np.arange(
            self.n_items * (self.n_items - 1) // 2 + 1, dtype=np.int64
        )
        self._anchor_log_base: NDArray[np.float64] | None = None

    def _fit_q(self, observed: NDArray[np.int64]) -> tuple[float, bool]:
        target = float(np.mean(observed))
        uniform_mean = self.n_items * (self.n_items - 1) / 4.0
        if target <= 0:
            return 0.0, target < 0
        if target >= uniform_mean:
            return 1.0, target > uniform_mean + 1e-12
        low = 0.0
        high = 1.0
        for _ in range(64):
            middle = (low + high) / 2.0
            if _expected_kendall_distance(self.n_items, middle) < target:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0, False

    def _probabilities(self, q: float) -> NDArray[np.float64]:
        if q <= 0:
            probabilities = np.zeros(self.support.size, dtype=np.float64)
            probabilities[0] = 1.0
            return probabilities
        if self._anchor_log_base is None:
            probabilities = _kendall_pmf(self.n_items, q)
            with np.errstate(divide="ignore"):
                self._anchor_log_base = (
                    np.log(probabilities) - self.support * log(q)
                )
            return probabilities
        log_mass = self._anchor_log_base + self.support * log(q)
        finite = np.isfinite(log_mass)
        maximum = float(np.max(log_mass[finite]))
        mass = np.zeros(log_mass.shape, dtype=np.float64)
        mass[finite] = np.exp(log_mass[finite] - maximum)
        return mass / np.sum(mass)

    def fit(self, distances: ArrayLike) -> FittedFiniteDiscreteModel:
        observed = _integer_distances(distances)
        if np.max(observed) > self.support[-1]:
            raise ValueError("Kendall distance exceeds the maximum for n_items")
        q, boundary = self._fit_q(observed)
        probabilities = self._probabilities(q)
        theta = float("inf") if q == 0 else -log(q)
        return FittedFiniteDiscreteModel(
            support=self.support,
            probabilities=probabilities,
            model_name="kendall_mallows_exact_distance",
            parameters={"q": q, "theta": theta, "boundary": boundary},
        )


def _discrete_ks_statistic(
    observed: NDArray[np.int64], fitted: FittedDiscreteModel
) -> float:
    values = np.unique(observed)
    sorted_observed = np.sort(observed)
    empirical_right = np.searchsorted(
        sorted_observed, values, side="right"
    ) / observed.size
    empirical_left = np.searchsorted(
        sorted_observed, values, side="left"
    ) / observed.size
    model_right = fitted.cdf(values)
    model_left = fitted.cdf(np.nextafter(values.astype(np.float64), -np.inf))
    return float(
        max(
            np.max(np.abs(empirical_right - model_right)),
            np.max(np.abs(empirical_left - model_left)),
        )
    )


@dataclass(frozen=True)
class DiscreteGOFResult:
    statistic: float
    p_value: float
    B: int
    model_name: str
    parameters: Mapping[str, float | str | bool]
    refit: bool
    bootstrap_statistics: NDArray[np.float64]


def discrete_parametric_bootstrap_gof(
    distances: ArrayLike,
    family: DiscreteModelFamily,
    *,
    B: int = 999,
    seed: int = 0,
    refit: bool = True,
) -> DiscreteGOFResult:
    """Discrete KS-type GOF with parameter estimation in each bootstrap.

    The plus-one p-value ``(1 + exceedances)/(B + 1)`` is never spuriously
    zero.  Re-fitting is enabled by default and is the statistically preferred
    parametric-bootstrap protocol; it can be disabled only for cost pilots.
    """

    if not isinstance(B, (int, np.integer)) or B <= 0:
        raise ValueError("B must be a positive integer")
    observed = _integer_distances(distances)
    fitted = family.fit(observed)
    statistic = _discrete_ks_statistic(observed, fitted)
    rng = np.random.default_rng(seed)
    bootstrap_statistics = np.empty(B, dtype=np.float64)
    for iteration in range(B):
        simulated = fitted.sample(observed.size, rng)
        simulated_fitted = family.fit(simulated) if refit else fitted
        bootstrap_statistics[iteration] = _discrete_ks_statistic(
            simulated, simulated_fitted
        )
    exceedances = int(np.count_nonzero(bootstrap_statistics >= statistic - 1e-15))
    p_value = (1.0 + exceedances) / (B + 1.0)
    return DiscreteGOFResult(
        statistic=statistic,
        p_value=float(p_value),
        B=int(B),
        model_name=fitted.model_name,
        parameters=dict(fitted.parameters),
        refit=refit,
        bootstrap_statistics=bootstrap_statistics,
    )


@dataclass(frozen=True)
class NoisePrefixEvaluation:
    size: int
    methods: tuple[str, ...]
    distance: Literal["kendall", "spearman"]
    n_distances: int
    mean_distance: float
    gof: DiscreteGOFResult


@dataclass(frozen=True)
class NoiseSelectionResult:
    selected_size: int
    selected_methods: tuple[str, ...]
    ordered_methods: tuple[str, ...]
    evaluations: tuple[NoisePrefixEvaluation, ...]
    selection_rule: Literal["max_pvalue", "largest_not_rejected"]
    alpha: float
    forced_fallback: bool
    scope: str = "global_in_sample"


def choose_noise_prefix(
    evaluations: tuple[NoisePrefixEvaluation, ...],
    *,
    selection_rule: Literal["max_pvalue", "largest_not_rejected"] = "max_pvalue",
    alpha: float = 0.05,
) -> tuple[NoisePrefixEvaluation, bool]:
    """Choose a prefix and guarantee a highest-p fallback if all are rejected."""

    if not evaluations:
        raise ValueError("evaluations must not be empty")
    if selection_rule not in ("max_pvalue", "largest_not_rejected"):
        raise ValueError(f"unknown selection_rule {selection_rule!r}")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    accepted = [evaluation for evaluation in evaluations if evaluation.gof.p_value >= alpha]
    forced_fallback = not accepted
    if selection_rule == "largest_not_rejected" and accepted:
        selected = max(accepted, key=lambda evaluation: evaluation.size)
    else:
        # The user's preferred rule is maximum fitted GOF p-value.  A larger
        # prefix wins exact ties to retain more available explanation methods.
        selected = max(
            evaluations,
            key=lambda evaluation: (evaluation.gof.p_value, evaluation.size),
        )
    return selected, forced_fallback


def _rank_tensor(rankings: ArrayLike, *, index_base: int) -> NDArray[np.int64]:
    raw = np.asarray(rankings)
    if raw.ndim != 3 or any(dimension == 0 for dimension in raw.shape):
        raise ValueError(
            "rankings must have shape (samples, methods, items), "
            f"got {raw.shape}"
        )
    return np.stack(
        [canonicalize_rankings(sample, index_base=index_base) for sample in raw],
        axis=0,
    )


def _method_order(
    method_names: tuple[str, ...], fidelity: Mapping[str, float] | ArrayLike
) -> NDArray[np.int64]:
    if len(set(method_names)) != len(method_names):
        raise ValueError("method_names must be unique")
    if isinstance(fidelity, Mapping):
        if set(fidelity) != set(method_names):
            raise ValueError("fidelity mapping keys must exactly match method_names")
        values = np.asarray([fidelity[name] for name in method_names], dtype=np.float64)
    else:
        values = np.asarray(fidelity, dtype=np.float64)
        if values.shape != (len(method_names),):
            raise ValueError("fidelity must contain one value per method")
    if not np.all(np.isfinite(values)):
        raise ValueError("fidelity contains NaN or infinite values")
    # Fidelity descending; method id is the deterministic tie break.
    return np.asarray(
        sorted(range(len(method_names)), key=lambda i: (-values[i], method_names[i])),
        dtype=np.int64,
    )


def _prefix_distances(
    rankings: NDArray[np.int64],
    method_indices: NDArray[np.int64],
    *,
    aggregation: Literal["borda", "kemeny"],
    kemeny_n_starts: int,
    kemeny_max_passes: int,
    kemeny_neighborhood: Literal["adjacent", "insertion"],
    seed: int,
) -> tuple[NDArray[np.int64], Literal["kendall", "spearman"]]:
    all_distances: list[int] = []
    distance_name: Literal["kendall", "spearman"]
    for sample_index, sample in enumerate(rankings):
        ballots = sample[method_indices]
        if aggregation == "borda":
            consensus = borda(ballots)
            distance_name = "spearman"
            all_distances.extend(
                spearman_distance(ballot, consensus) for ballot in ballots
            )
        elif aggregation == "kemeny":
            consensus = kemeny_young(
                ballots,
                n_starts=kemeny_n_starts,
                max_passes=kemeny_max_passes,
                neighborhood=kemeny_neighborhood,
                seed=seed + sample_index,
            ).ranking
            distance_name = "kendall"
            all_distances.extend(
                kendall_distance(ballot, consensus) for ballot in ballots
            )
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
    return np.asarray(all_distances, dtype=np.int64), distance_name


GOFFamilyFactory = Callable[[Literal["kendall", "spearman"], int], DiscreteModelFamily]


def order_methods_by_fidelity(
    method_names: list[str] | tuple[str, ...],
    fidelity: Mapping[str, float] | ArrayLike,
) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    """Return deterministic descending-fidelity indices and method ids."""

    names = tuple(str(name) for name in method_names)
    ordering = _method_order(names, fidelity)
    return ordering, tuple(names[index] for index in ordering)


def compute_prefix_distances(
    rankings: ArrayLike,
    method_indices: ArrayLike,
    *,
    aggregation: Literal["borda", "kemeny"],
    index_base: int = 0,
    kemeny_n_starts: int = 16,
    kemeny_max_passes: int = 10_000,
    kemeny_neighborhood: Literal["adjacent", "insertion"] = "insertion",
    seed: int = 0,
) -> tuple[NDArray[np.int64], Literal["kendall", "spearman"]]:
    """Compute consensus distances for one streaming batch and one prefix."""

    canonical = _rank_tensor(rankings, index_base=index_base)
    indices = np.asarray(method_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("method_indices must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= canonical.shape[1]):
        raise ValueError("method_indices are empty or outside the method axis")
    if np.unique(indices).size != indices.size:
        raise ValueError("method_indices must be unique")
    return _prefix_distances(
        canonical,
        indices,
        aggregation=aggregation,
        kemeny_n_starts=kemeny_n_starts,
        kemeny_max_passes=kemeny_max_passes,
        kemeny_neighborhood=kemeny_neighborhood,
        seed=seed,
    )


def select_noise_from_distance_samples(
    distances_by_size: Mapping[int, ArrayLike],
    *,
    ordered_methods: tuple[str, ...],
    distance: Literal["kendall", "spearman"],
    n_items: int,
    B: int = 999,
    alpha: float = 0.05,
    selection_rule: Literal["max_pvalue", "largest_not_rejected"] = "max_pvalue",
    seed: int = 0,
    gof_family_factory: GOFFamilyFactory | None = None,
    allow_spearman_surrogate: bool = False,
    bootstrap_refit: bool = True,
) -> NoiseSelectionResult:
    """Finish global NOISE selection from distances accumulated shard-wise."""

    evaluations: list[NoisePrefixEvaluation] = []
    for evaluation_index, size in enumerate(sorted(distances_by_size)):
        if not 1 <= size <= len(ordered_methods):
            raise ValueError("distance prefix size is outside ordered_methods")
        distances = _integer_distances(distances_by_size[size])
        family = (
            _default_family(
                distance,
                n_items,
                allow_spearman_surrogate=allow_spearman_surrogate,
            )
            if gof_family_factory is None
            else gof_family_factory(distance, n_items)
        )
        gof = discrete_parametric_bootstrap_gof(
            distances,
            family,
            B=B,
            seed=seed + 1_000_003 * (evaluation_index + 1),
            refit=bootstrap_refit,
        )
        evaluations.append(
            NoisePrefixEvaluation(
                size=size,
                methods=ordered_methods[:size],
                distance=distance,
                n_distances=int(distances.size),
                mean_distance=float(np.mean(distances)),
                gof=gof,
            )
        )
    frozen = tuple(evaluations)
    selected, forced = choose_noise_prefix(
        frozen, selection_rule=selection_rule, alpha=alpha
    )
    return NoiseSelectionResult(
        selected_size=selected.size,
        selected_methods=selected.methods,
        ordered_methods=ordered_methods,
        evaluations=frozen,
        selection_rule=selection_rule,
        alpha=alpha,
        forced_fallback=forced,
    )


def _default_family(
    distance: Literal["kendall", "spearman"],
    n_items: int,
    *,
    allow_spearman_surrogate: bool,
) -> DiscreteModelFamily:
    if distance == "kendall":
        return KendallMallowsFamily(n_items)
    if not allow_spearman_surrogate:
        raise ValueError(
            "exact Spearman-Mallows multiplicities are not tractable at this "
            "rank size; provide gof_family_factory, or explicitly set "
            "allow_spearman_surrogate=True for the labeled cost-pilot surrogate"
        )
    maximum = n_items * (n_items * n_items - 1) // 3
    # Squared Spearman distances between permutations are even.
    return FiniteDiscreteExponentialFamily(
        np.arange(0, maximum + 1, 2, dtype=np.int64),
        model_name="spearman_uniform_distance_support_surrogate",
    )


def select_noise_prefix(
    rankings: ArrayLike,
    method_names: list[str] | tuple[str, ...],
    fidelity: Mapping[str, float] | ArrayLike,
    *,
    aggregation: Literal["borda", "kemeny"],
    index_base: int = 0,
    prefix_sizes: ArrayLike | None = None,
    min_prefix: int = 2,
    B: int = 999,
    alpha: float = 0.05,
    selection_rule: Literal["max_pvalue", "largest_not_rejected"] = "max_pvalue",
    seed: int = 0,
    gof_family_factory: GOFFamilyFactory | None = None,
    allow_spearman_surrogate: bool = False,
    bootstrap_refit: bool = True,
    kemeny_n_starts: int = 16,
    kemeny_max_passes: int = 10_000,
    kemeny_neighborhood: Literal["adjacent", "insertion"] = "insertion",
) -> NoiseSelectionResult:
    """Select one fidelity-ordered prefix globally on the complete test pool."""

    canonical = _rank_tensor(rankings, index_base=index_base)
    names = tuple(str(name) for name in method_names)
    if len(names) != canonical.shape[1]:
        raise ValueError("method_names length does not match rankings method axis")
    ordering = _method_order(names, fidelity)
    ordered_names = tuple(names[index] for index in ordering)
    n_methods = len(names)
    if prefix_sizes is None:
        if min_prefix <= 0 or min_prefix > n_methods:
            raise ValueError("min_prefix must lie between 1 and n_methods")
        sizes = np.arange(min_prefix, n_methods + 1, dtype=np.int64)
    else:
        sizes = _integer_distances(prefix_sizes)
        if np.any(sizes < 1) or np.any(sizes > n_methods):
            raise ValueError("every prefix size must lie between 1 and n_methods")
        sizes = np.unique(sizes)

    evaluations: list[NoisePrefixEvaluation] = []
    for evaluation_index, size in enumerate(sizes):
        method_indices = ordering[: int(size)]
        distances, distance_name = _prefix_distances(
            canonical,
            method_indices,
            aggregation=aggregation,
            kemeny_n_starts=kemeny_n_starts,
            kemeny_max_passes=kemeny_max_passes,
            kemeny_neighborhood=kemeny_neighborhood,
            seed=seed + evaluation_index * canonical.shape[0],
        )
        if gof_family_factory is None:
            family = _default_family(
                distance_name,
                canonical.shape[2],
                allow_spearman_surrogate=allow_spearman_surrogate,
            )
        else:
            family = gof_family_factory(distance_name, canonical.shape[2])
        gof = discrete_parametric_bootstrap_gof(
            distances,
            family,
            B=B,
            seed=seed + 1_000_003 * (evaluation_index + 1),
            refit=bootstrap_refit,
        )
        evaluations.append(
            NoisePrefixEvaluation(
                size=int(size),
                methods=ordered_names[: int(size)],
                distance=distance_name,
                n_distances=int(distances.size),
                mean_distance=float(np.mean(distances)),
                gof=gof,
            )
        )
    frozen_evaluations = tuple(evaluations)
    selected, forced_fallback = choose_noise_prefix(
        frozen_evaluations, selection_rule=selection_rule, alpha=alpha
    )
    return NoiseSelectionResult(
        selected_size=selected.size,
        selected_methods=selected.methods,
        ordered_methods=ordered_names,
        evaluations=frozen_evaluations,
        selection_rule=selection_rule,
        alpha=alpha,
        forced_fallback=forced_fallback,
    )


@dataclass(frozen=True)
class NoiseCostPilot:
    pilot_samples: int
    pilot_B: int
    target_samples: int
    target_B: int
    measured_seconds: float
    projected_seconds: float
    projection_rule: str
    selection: NoiseSelectionResult


def noise_selection_cost_pilot(
    rankings: ArrayLike,
    method_names: list[str] | tuple[str, ...],
    fidelity: Mapping[str, float] | ArrayLike,
    *,
    aggregation: Literal["borda", "kemeny"],
    pilot_samples: int = 16,
    pilot_B: int = 19,
    target_samples: int | None = None,
    target_B: int = 999,
    **selection_kwargs: object,
) -> NoiseCostPilot:
    """Run a small end-to-end pilot and conservatively project full cost."""

    raw = np.asarray(rankings)
    if raw.ndim != 3:
        raise ValueError("rankings must have shape (samples, methods, items)")
    if pilot_samples <= 0 or pilot_B <= 0 or target_B <= 0:
        raise ValueError("pilot_samples, pilot_B, and target_B must be positive")
    actual_pilot_samples = min(int(pilot_samples), raw.shape[0])
    if target_samples is None:
        target_samples = raw.shape[0]
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    start = perf_counter()
    selection = select_noise_prefix(
        raw[:actual_pilot_samples],
        method_names,
        fidelity,
        aggregation=aggregation,
        B=pilot_B,
        **selection_kwargs,
    )
    elapsed = perf_counter() - start
    # Both consensus construction and each bootstrap sample scale with sample
    # count; multiplying both factors is deliberately conservative because the
    # fixed fitting overhead is small at production scale.
    scale = (target_samples / actual_pilot_samples) * (
        (target_B + 1) / (pilot_B + 1)
    )
    return NoiseCostPilot(
        pilot_samples=actual_pilot_samples,
        pilot_B=pilot_B,
        target_samples=int(target_samples),
        target_B=int(target_B),
        measured_seconds=float(elapsed),
        projected_seconds=float(elapsed * scale),
        projection_rule="linear_in_samples_times_(B+1); conservative",
        selection=selection,
    )


__all__ = [
    "BinnedDiscreteExponentialFamily",
    "DiscreteGOFResult",
    "DiscreteModelFamily",
    "DiagnosticDiscreteExponentialFamily",
    "FiniteDiscreteExponentialFamily",
    "FittedDiscreteModel",
    "FittedBinnedDiscreteModel",
    "KendallMallowsFamily",
    "NoiseCostPilot",
    "NoisePrefixEvaluation",
    "NoiseSelectionResult",
    "choose_noise_prefix",
    "discrete_parametric_bootstrap_gof",
    "noise_selection_cost_pilot",
    "locked_spearman_family_factory",
    "compute_prefix_distances",
    "order_methods_by_fidelity",
    "select_noise_prefix",
    "select_noise_from_distance_samples",
]
