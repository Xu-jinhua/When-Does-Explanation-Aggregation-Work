"""NOISE goodness-of-fit cost: Spearman family calibration and budget adjudication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256

from .base import PilotResult


def noise_gof_cost_pilot(
    reports: Sequence[Mapping[str, Any]], *, config: dict[str, Any]
) -> PilotResult:
    maximum_minutes = float(
        config.get("pass", {}).get(
            "maximum_minutes_per_dataset_model_aggregation", 30
        )
    )
    maximum_se = float(
        config.get("pass", {}).get("pvalue_monte_carlo_se_at_0_05_max", 0.01)
    )
    spearman_config = config.get("spearman_mallows_approximation")
    spearman_required = isinstance(spearman_config, Mapping)
    spearman_pass = (
        spearman_config.get("pass", {}) if spearman_required else {}
    )
    rows = []
    eligible = []
    for index, report in enumerate(reports):
        B = int(report.get("B", report.get("target_B", 0)))
        seconds = float(
            report.get("projected_seconds", report.get("measured_seconds", np.nan))
        )
        if B <= 0 or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"invalid NOISE cost report at index {index}")
        monte_carlo_se = float(np.sqrt(0.05 * 0.95 / (B + 1)))
        deterministic = bool(report.get("deterministic_seed", True))
        adapter = report.get("spearman_family_adapter")
        converged = bool(report.get("spearman_converged", False))
        r_hat = float(report.get("spearman_r_hat", np.inf))
        ess = float(report.get("spearman_ess", 0.0))
        repeated_difference = float(
            report.get("spearman_repeated_p_value_absolute_difference", np.inf)
        )
        artifact_config = report.get("spearman_config", {})
        artifact_ok = False
        stored_digest = ""
        stored_approximation = ""
        stored_validation_digest = ""
        stored_support_completion = ""
        stored_full_support = False
        stored_theoretical_maximum = 0
        stored_maximum_calibrated = 0
        stored_symmetry_residual = np.inf
        if spearman_required and isinstance(artifact_config, Mapping):
            path = Path(str(artifact_config.get("path", "")))
            expected_sha = str(artifact_config.get("sha256", ""))
            try:
                if not path.is_file() or len(expected_sha) != 64:
                    raise ValueError("missing Spearman family artifact")
                if file_sha256(path) != expected_sha:
                    raise ValueError("Spearman family artifact checksum mismatch")
                with np.load(path, allow_pickle=False) as archive:
                    stored_digest = str(
                        np.asarray(archive["pilot_digest"]).reshape(-1)[0]
                    )
                    stored_approximation = str(
                        np.asarray(archive["approximation"]).reshape(-1)[0]
                    )
                    stored_validation_digest = str(
                        np.asarray(archive["validation_distances_digest"])
                        .reshape(-1)[0]
                    )
                    stored_support_completion = str(
                        np.asarray(archive["support_completion"]).reshape(-1)[0]
                    )
                    stored_full_support = bool(
                        np.asarray(archive["full_support"]).reshape(-1)[0]
                    )
                    stored_theoretical_maximum = int(
                        np.asarray(archive["theoretical_maximum_distance"])
                        .reshape(-1)[0]
                    )
                    stored_maximum_calibrated = int(
                        np.asarray(archive["maximum_calibrated_distance"])
                        .reshape(-1)[0]
                    )
                    stored_symmetry_residual = float(
                        np.asarray(archive["maximum_symmetry_residual"])
                        .reshape(-1)[0]
                    )
                    stored_items = int(np.asarray(archive["n_items"]).reshape(-1)[0])
                    support = np.asarray(archive["support"])
                    bin_upper = np.asarray(archive["bin_upper"])
                    weights = np.asarray(archive["base_log_weights"], dtype=np.float64)
                    expected_maximum = stored_items * (stored_items**2 - 1) // 3
                    stored_exact = bool(np.asarray(archive["exact"]).reshape(-1)[0])
                    support_complete = (
                        support.ndim == 1
                        and bin_upper.shape == support.shape
                        and weights.shape == support.shape
                        and support.size > 0
                        and np.issubdtype(support.dtype, np.integer)
                        and np.issubdtype(bin_upper.dtype, np.integer)
                        and not np.any(np.isnan(weights))
                        and np.any(np.isfinite(weights))
                        and support[0] == 0
                        and bin_upper[-1] == expected_maximum
                        and np.all(support[1:] > support[:-1])
                        and np.all(support % 2 == 0)
                        and np.all(bin_upper % 2 == 0)
                        and np.all(bin_upper >= support)
                        and (
                            (
                                stored_exact
                                and stored_approximation == "exact_enumeration"
                                and stored_support_completion == "exact_enumeration"
                                and np.array_equal(support, bin_upper)
                                and np.array_equal(
                                    support, expected_maximum - support[::-1]
                                )
                                and np.allclose(
                                    weights, weights[::-1], rtol=0.0, atol=1e-12
                                )
                            )
                            or (
                                not stored_exact
                                and stored_approximation
                                == "multi_proposal_importance_binned_symmetric"
                                and stored_support_completion
                                == "exact_reverse_symmetry"
                                and np.all(support[1:] == bin_upper[:-1] + 2)
                                and np.array_equal(
                                    support // 2,
                                    expected_maximum // 2 - (bin_upper // 2)[::-1],
                                )
                                and np.array_equal(
                                    bin_upper // 2,
                                    expected_maximum // 2 - (support // 2)[::-1],
                                )
                                and np.allclose(
                                    weights
                                    - np.log((bin_upper - support) // 2 + 1),
                                    (
                                        weights
                                        - np.log((bin_upper - support) // 2 + 1)
                                    )[::-1],
                                    rtol=0.0,
                                    atol=1e-12,
                                )
                            )
                        )
                    )
                    artifact_ok = (
                        int(
                            np.asarray(archive["artifact_schema_version"])
                            .reshape(-1)[0]
                        )
                        == 3
                        and bool(
                            np.asarray(archive["converged"]).reshape(-1)[0]
                        )
                        == converged
                        and np.isclose(
                            float(np.asarray(archive["r_hat"]).reshape(-1)[0]),
                            r_hat,
                        )
                        and np.isclose(
                            float(np.asarray(archive["ess"]).reshape(-1)[0]),
                            ess,
                        )
                        and np.isclose(
                            float(
                                np.asarray(
                                    archive[
                                        "repeated_p_value_absolute_difference"
                                    ]
                                ).reshape(-1)[0]
                            ),
                            repeated_difference,
                        )
                        and stored_digest
                        == str(report.get("spearman_pilot_digest", ""))
                        and len(stored_digest) == 64
                        and len(stored_validation_digest) == 64
                        and stored_full_support
                        and stored_theoretical_maximum == expected_maximum
                        and stored_maximum_calibrated == expected_maximum
                        and np.isfinite(stored_symmetry_residual)
                        and stored_symmetry_residual >= 0
                        and stored_symmetry_residual <= 1
                        and support_complete
                        and stored_approximation
                        in {
                            "exact_enumeration",
                            "multi_proposal_importance_binned_symmetric",
                        }
                    )
            except (OSError, KeyError, TypeError, ValueError):
                artifact_ok = False
        required_adapter = str(
            spearman_config.get(
                "adapter",
                "xai_ensemble.phase2.selection:locked_spearman_family_factory",
            )
        ) if spearman_required else ""
        spearman_ok = (
            not spearman_required
            or (
                isinstance(adapter, str)
                and adapter == required_adapter
                and converged
                and r_hat <= float(spearman_pass.get("r_hat_max", 1.05))
                and ess >= float(spearman_pass.get("effective_sample_size_min", 1000))
                and repeated_difference
                <= float(
                    spearman_pass.get(
                        "repeated_p_value_absolute_difference_max", 0.02
                    )
                )
                and artifact_ok
            )
        )
        passed = (
            seconds <= maximum_minutes * 60
            and monte_carlo_se <= maximum_se
            and deterministic
            and spearman_ok
        )
        row = {
            "B": B,
            "projected_seconds": seconds,
            "pvalue_monte_carlo_se_at_0_05": monte_carlo_se,
            "deterministic_seed": deterministic,
            "spearman_family_adapter": adapter,
            "spearman_converged": converged,
            "spearman_r_hat": r_hat,
            "spearman_ess": ess,
            "spearman_repeated_p_value_absolute_difference": repeated_difference,
            "spearman_config": dict(artifact_config),
            "spearman_artifact_valid": artifact_ok,
            "spearman_pilot_digest": stored_digest,
            "spearman_approximation": stored_approximation,
            "spearman_validation_distances_digest": stored_validation_digest,
            "spearman_support_completion": stored_support_completion,
            "spearman_full_support": stored_full_support,
            "spearman_theoretical_maximum_distance": stored_theoretical_maximum,
            "spearman_maximum_calibrated_distance": stored_maximum_calibrated,
            "spearman_maximum_symmetry_residual": stored_symmetry_residual,
            "spearman_diagnostics": dict(report.get("diagnostics", {})),
            "passed": passed,
        }
        rows.append(row)
        if passed:
            eligible.append(row)
    failures = [] if eligible else ["no bootstrap budget met time and p-value precision gates"]
    winner = min(eligible, key=lambda row: (row["B"], row["projected_seconds"])) if eligible else None
    selected = {}
    if winner is not None:
        selected = {"bootstrap_replicates": winner["B"]}
        if spearman_required:
            selected.update(
                {
                    "spearman_family_adapter": winner["spearman_family_adapter"],
                    "spearman_config": winner["spearman_config"],
                    "spearman_pilot_digest": winner["spearman_pilot_digest"],
                    "spearman_approximation": winner["spearman_approximation"],
                    "spearman_validation_distances_digest": winner[
                        "spearman_validation_distances_digest"
                    ],
                    "spearman_min_ess": float(
                        spearman_pass.get("effective_sample_size_min", 1000)
                    ),
                }
            )
    return PilotResult(
        name="noise_gof_cost",
        status="passed" if winner else "failed",
        selected=selected,
        measurements={"candidates": rows},
        failures=failures,
        config_digest=object_sha256(config),
    )


def run_noise_gof_cost_pilot(
    *,
    config: dict[str, Any],
    seed: int,
    artifact_path: str | Path,
    n_items: int,
    synthetic_validation_count: int | None = None,
    synthetic_proposal_q: Sequence[float] | None = None,
    bootstrap_replicates: int | None = None,
    projected_prefix_count: int | None = None,
) -> PilotResult:
    """Calibrate the Spearman distance family on synthetic validation distances."""

    from xai_ensemble.simple.assumptions.spearman_calibration import (
        calibrate_spearman_mallows,
        synthetic_validation_distances,
    )

    section = config.get("spearman_mallows_approximation")
    if not isinstance(section, Mapping):
        raise ValueError("config lacks a spearman_mallows_approximation section")
    limits = section.get("resource_limits", {})
    if not isinstance(limits, Mapping):
        raise TypeError("spearman resource_limits must be a mapping")
    count = int(
        synthetic_validation_count
        if synthetic_validation_count is not None
        else section.get("synthetic_validation_count", 80_000)
    )
    q_values = (
        list(synthetic_proposal_q)
        if synthetic_proposal_q is not None
        else list(section.get("synthetic_validation_q", [0.6, 0.8, 0.92, 0.97, 0.99]))
    )
    validation = synthetic_validation_distances(
        n_items=n_items,
        count=count,
        q_values=q_values,
        seed=seed,
        maximum_count=int(limits.get("maximum_validation_distances", 200_000)),
        maximum_wall_seconds=float(limits.get("maximum_wall_seconds", 21_600)),
    )
    outcome = calibrate_spearman_mallows(
        validation,
        n_items=n_items,
        config=config,
        artifact_path=artifact_path,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        projected_prefix_count=projected_prefix_count,
    )
    result = noise_gof_cost_pilot([outcome.measurement], config=config)
    result.measurements.update(
        {
            "seed": seed,
            "n_items": n_items,
            "synthetic_validation_count": int(validation.size),
            "producer": "synthetic_validation_distances",
        }
    )
    return result


__all__ = ["noise_gof_cost_pilot", "run_noise_gof_cost_pilot"]
