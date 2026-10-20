from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from xai_ensemble.phase2.aggregation import exact_kemeny_young
from xai_ensemble.pilots.base import PilotResult, save_pilot_result
from xai_ensemble.pilots.kemeny_solver import (
    _exact_kemeny_objective_dp,
    _synthetic_ballots,
    kemeny_benchmark,
    kemeny_solver_pilot,
)
from xai_ensemble.pilots.noise_gof_cost import noise_gof_cost_pilot

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "pilots"


def _result(name: str = "example") -> PilotResult:
    return PilotResult(
        name=name,
        status="passed",
        selected={"value": 3},
        measurements={"candidates": [{"id": "a", "passed": True}]},
        config_digest="abc",
    )


def test_pilot_result_digest_roundtrip(tmp_path: Path) -> None:
    import json

    result = _result()
    path = save_pilot_result(tmp_path / "result.json", result)

    loaded = PilotResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.digest == result.digest
    assert loaded.selected == result.selected
    assert loaded.passed

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["status"] = "failed"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        PilotResult.from_dict(json.loads(tampered_path.read_text(encoding="utf-8")))


def test_pilot_result_digest_ignores_created_at() -> None:
    first = _result()
    second = _result()
    second.created_at = "2000-01-01T00:00:00+00:00"
    assert first.digest == second.digest


def test_pilot_result_require_passed_raises_on_failure() -> None:
    result = _result()
    result.status = "failed"
    result.failures.append("nothing passed")
    with pytest.raises(RuntimeError, match="has not passed"):
        result.require_passed()


def test_synthetic_ballots_are_strict_and_deterministic() -> None:
    first = _synthetic_ballots(
        np.random.default_rng(7), ballot_count=5, item_count=16
    )
    second = _synthetic_ballots(
        np.random.default_rng(7), ballot_count=5, item_count=16
    )
    assert first.shape == (5, 16)
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    for row in first:
        assert np.array_equal(np.sort(row), np.arange(16))


def test_exact_kemeny_dp_matches_brute_force() -> None:
    rng = np.random.default_rng(1234)
    for item_count in (3, 5, 6):
        ballots = _synthetic_ballots(
            rng, ballot_count=4, item_count=item_count
        )
        expected = exact_kemeny_young(ballots).objective
        assert _exact_kemeny_objective_dp(ballots) == expected


def _kemeny_report(
    objectives: list[float],
    borda: list[float],
    exact: list[float],
    **overrides: float,
) -> dict[str, object]:
    report: dict[str, object] = {
        "config": {"n_starts": 1, "neighborhood": "insertion"},
        "objectives": objectives,
        "borda_objectives": borda,
        "exact_objectives": exact,
        "wall_seconds": 1.0,
        "deterministic_seed": True,
        "within_time_limit": True,
    }
    report.update(overrides)
    return report


def test_kemeny_solver_pilot_selects_fastest_passing_config() -> None:
    config = {"pass": {"exact_optimum_rate_n16_min": 0.95}}
    reports = [
        _kemeny_report(
            [10, 12], [10, 13], [10, 12], wall_seconds=2.0
        ),
        _kemeny_report(
            [10, 12], [10, 13], [10, 12], wall_seconds=1.0
        ),
    ]
    reports[0]["config"] = {"n_starts": 4, "neighborhood": "insertion"}
    reports[1]["config"] = {"n_starts": 1, "neighborhood": "insertion"}
    result = kemeny_solver_pilot(reports, config=config)
    assert result.passed
    assert result.selected == {"n_starts": 1, "neighborhood": "insertion"}
    assert len(result.measurements["candidates"]) == 2

    failing = [
        _kemeny_report([11, 12], [10, 13], [10, 12]),
        _kemeny_report([10, 12], [10, 13], [10, 12], deterministic_seed=False),
        _kemeny_report([10, 13], [10, 13], [10, 12]),
    ]
    for report in failing:
        outcome = kemeny_solver_pilot([report], config=config)
        assert not outcome.passed
        assert outcome.failures


def test_kemeny_solver_pilot_rejects_invalid_reports() -> None:
    config = {"pass": {"exact_optimum_rate_n16_min": 0.95}}
    with pytest.raises(ValueError, match="invalid objective arrays"):
        kemeny_solver_pilot(
            [_kemeny_report([1.0], [1.0, 2.0], [1.0])], config=config
        )
    with pytest.raises(ValueError, match="invalid measurements"):
        kemeny_solver_pilot(
            [_kemeny_report([1.0], [1.0], [1.0], wall_seconds=0.0)],
            config=config,
        )


def test_kemeny_benchmark_tiny_grid_passes_against_exact_oracle() -> None:
    config = {
        "patch_counts": [16],
        "ballot_counts": [4],
        "starts": [1, 4],
        "neighborhoods": ["insertion"],
        "time_limits_seconds": [30],
        "synthetic_instances_per_cell": 2,
        "pass": {"exact_optimum_rate_n16_min": 0.95},
    }
    result = kemeny_benchmark(config=config, seed=20260714)
    assert result.passed
    candidates = result.measurements["candidates"]
    assert candidates
    winner = next(
        row for row in candidates if row["config"] == result.selected
    )
    assert winner["exact_optimum_rate"] == 1.0
    assert winner["never_worse_than_borda"]
    assert result.measurements["exact_instance_count"] == 2


def test_noise_gof_cost_pilot_pass_and_select_smallest_budget() -> None:
    config = {
        "pass": {
            "maximum_minutes_per_dataset_model_aggregation": 30,
            "pvalue_monte_carlo_se_at_0_05_max": 0.01,
        }
    }
    reports = [
        {"B": 999, "projected_seconds": 25.0 * 60, "deterministic_seed": True},
        {"B": 499, "projected_seconds": 12.0, "deterministic_seed": True},
    ]
    result = noise_gof_cost_pilot(reports, config=config)
    assert result.passed
    assert result.selected == {"bootstrap_replicates": 499}

    se_only = noise_gof_cost_pilot(
        [{"B": 99, "projected_seconds": 5.0, "deterministic_seed": True}],
        config=config,
    )
    assert not se_only.passed

    time_only = noise_gof_cost_pilot(
        [{"B": 999, "projected_seconds": 31.0 * 60, "deterministic_seed": True}],
        config=config,
    )
    assert not time_only.passed

    nondeterministic = noise_gof_cost_pilot(
        [{"B": 999, "projected_seconds": 5.0, "deterministic_seed": False}],
        config=config,
    )
    assert not nondeterministic.passed


def test_noise_gof_cost_pilot_requires_spearman_artifact_when_configured() -> None:
    config = {
        "pass": {
            "maximum_minutes_per_dataset_model_aggregation": 30,
            "pvalue_monte_carlo_se_at_0_05_max": 0.01,
        },
        "spearman_mallows_approximation": {
            "adapter": (
                "xai_ensemble.phase2.selection:locked_spearman_family_factory"
            ),
            "pass": {},
        },
    }
    report = {
        "B": 499,
        "projected_seconds": 5.0,
        "deterministic_seed": True,
        "spearman_family_adapter": (
            "xai_ensemble.phase2.selection:locked_spearman_family_factory"
        ),
        "spearman_converged": True,
        "spearman_r_hat": 1.0,
        "spearman_ess": 5000.0,
        "spearman_repeated_p_value_absolute_difference": 0.001,
    }
    result = noise_gof_cost_pilot([dict(report)], config=config)
    assert not result.passed
    assert not result.measurements["candidates"][0]["spearman_artifact_valid"]


def test_noise_gof_cost_pilot_rejects_invalid_reports() -> None:
    config = {"pass": {}}
    with pytest.raises(ValueError, match="invalid NOISE cost report"):
        noise_gof_cost_pilot(
            [{"B": 0, "projected_seconds": 5.0}], config=config
        )


def test_pilot_configs_are_loadable() -> None:
    expected = {
        "ig_step_budget.yaml": "ig_step_budget",
        "baseline_selection.yaml": "baseline_selection",
        "kemeny_solver.yaml": "kemeny_solver",
        "noise_gof_cost.yaml": "noise_gof_cost",
    }
    for filename, name in expected.items():
        value = yaml.safe_load((CONFIGS / filename).read_text(encoding="utf-8"))
        assert isinstance(value, dict)
        assert value["name"] == name
        assert isinstance(value.get("pass"), dict)
    ig_steps = yaml.safe_load(
        (CONFIGS / "ig_step_budget.yaml").read_text(encoding="utf-8")
    )["step_candidates"]
    assert ig_steps == [50, 75, 100, 125, 150, 175, 200]
    baselines = yaml.safe_load(
        (CONFIGS / "baseline_selection.yaml").read_text(encoding="utf-8")
    )
    assert baselines["reference_candidates"] == [
        "zero",
        "dataset_mean",
        "class_mean",
        "gaussian_noise",
        "uniform_noise",
    ]


def test_pilot_modules_import_without_torch() -> None:
    import xai_ensemble.pilots.baseline_selection  # noqa: F401
    import xai_ensemble.pilots.cli  # noqa: F401
    import xai_ensemble.pilots.ig_step_budget  # noqa: F401
