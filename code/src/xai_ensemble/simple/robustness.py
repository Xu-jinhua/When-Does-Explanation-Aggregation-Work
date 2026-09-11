"""Paper-facing signed robustness derived from aligned raw quality metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from .artifacts import ArtifactError

SIGNED_ROBUSTNESS_POLICY = "signed_quality_change_fixed_clean_metric_anchor_v1"
SIGNED_ROBUSTNESS_DIRECTION = "max"
SIGNED_ROBUSTNESS_SOURCE = "aligned_clean_and_perturbed_raw_metrics"


def signed_robustness(metric: str, *, clean: float, perturbed: float) -> float:
    """Return signed quality change, oriented so that larger is always better."""

    if metric not in QUALITY_METRICS:
        raise ValueError(f"Unknown quality metric {metric!r}")
    clean_value = float(clean)
    perturbed_value = float(perturbed)
    if not math.isfinite(clean_value) or not math.isfinite(perturbed_value):
        raise ValueError(f"Non-finite {metric} quality value")
    if DEFAULT_METRIC_DIRECTIONS[metric] == "max":
        return perturbed_value - clean_value
    return clean_value - perturbed_value


def signed_robustness_values(
    clean: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    *,
    legacy_absolute: Mapping[str, Any] | None = None,
    context: str,
) -> dict[str, float]:
    """Compute all signed R values and optionally audit the legacy absolute field."""

    expected = set(QUALITY_METRICS)
    if set(clean) != expected or set(perturbed) != expected:
        raise ArtifactError(f"{context} raw quality fields are invalid")
    if legacy_absolute is not None:
        if not isinstance(legacy_absolute, Mapping) or set(legacy_absolute) != expected:
            raise ArtifactError(f"{context} legacy absolute robustness fields are invalid")
        for metric in QUALITY_METRICS:
            recorded = float(legacy_absolute[metric])
            expected_absolute = abs(float(perturbed[metric]) - float(clean[metric]))
            if not math.isfinite(recorded) or not math.isclose(
                recorded,
                expected_absolute,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ArtifactError(
                    f"{context}/{metric} legacy absolute robustness contradicts raw metrics"
                )
    return {
        metric: signed_robustness(
            metric,
            clean=float(clean[metric]),
            perturbed=float(perturbed[metric]),
        )
        for metric in QUALITY_METRICS
    }


def perturbed_quality_values(values: Mapping[str, Any], *, context: str) -> dict[str, float]:
    """Validate and normalize one perturbed raw-quality record."""

    if set(values) != set(QUALITY_METRICS):
        raise ArtifactError(f"{context} perturbed quality fields are invalid")
    result = {metric: float(values[metric]) for metric in QUALITY_METRICS}
    if any(not math.isfinite(value) for value in result.values()):
        raise ArtifactError(f"{context} perturbed quality contains a non-finite value")
    return result


__all__ = [
    "SIGNED_ROBUSTNESS_DIRECTION",
    "SIGNED_ROBUSTNESS_POLICY",
    "SIGNED_ROBUSTNESS_SOURCE",
    "perturbed_quality_values",
    "signed_robustness",
    "signed_robustness_values",
]
