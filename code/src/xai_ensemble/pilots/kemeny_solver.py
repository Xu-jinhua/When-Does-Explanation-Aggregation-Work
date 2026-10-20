"""Kemeny-Young solver benchmark with a small exact subset-DP oracle."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.phase2.aggregation import borda, kemeny_objective, kemeny_young
from xai_ensemble.phase2.rankings import pairwise_preference_counts

from .base import PilotResult


def _exact_kemeny_objective_dp(ballots: np.ndarray) -> int:
    """Exact subset-DP Kemeny oracle, practical for the configured n=16 gate."""

    values = np.asarray(ballots, dtype=np.int64)
    if values.ndim != 2 or not 2 <= values.shape[1] <= 20:
        raise ValueError("exact pilot DP supports 2..20 ranked items")
    counts = pairwise_preference_counts(values)
    n_items = values.shape[1]
    size = 1 << n_items
    dp = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    dp[0] = 0
    for mask in range(1, size):
        best = np.iinfo(np.int64).max
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            item = bit.bit_length() - 1
            previous = mask ^ bit
            cost = int(dp[previous])
            other = previous
            while other:
                other_bit = other & -other
                earlier = other_bit.bit_length() - 1
                cost += int(counts[item, earlier])
                other ^= other_bit
            best = min(best, cost)
            remaining ^= bit
        dp[mask] = best
    return int(dp[-1])


def _synthetic_ballots(
    rng: np.random.Generator, *, ballot_count: int, item_count: int
) -> np.ndarray:
    """Generate heterogeneous, correlated strict rank ballots."""

    center = rng.permutation(item_count)
    ballots: list[np.ndarray] = []
    for _ in range(ballot_count):
        order = center.copy()
        swaps = int(rng.integers(1, max(2, item_count // 3 + 1)))
        for _ in range(swaps):
            first, second = rng.choice(item_count, size=2, replace=False)
            order[first], order[second] = order[second], order[first]
        ranking = np.empty(item_count, dtype=np.int64)
        ranking[order] = np.arange(item_count)
        ballots.append(ranking)
    return np.stack(ballots)


def kemeny_solver_pilot(
    reports: Sequence[Mapping[str, Any]], *, config: dict[str, Any]
) -> PilotResult:
    exact_min = float(config.get("pass", {}).get("exact_optimum_rate_n16_min", 0.95))
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        objectives = np.asarray(report.get("objectives", ()), dtype=np.float64)
        borda_objectives = np.asarray(report.get("borda_objectives", ()), dtype=np.float64)
        exact = np.asarray(report.get("exact_objectives", ()), dtype=np.float64)
        seconds = float(report.get("wall_seconds", np.nan))
        if objectives.ndim != 1 or objectives.size == 0 or objectives.shape != borda_objectives.shape:
            raise ValueError(f"Kemeny report {index} has invalid objective arrays")
        if not np.all(np.isfinite(objectives)) or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Kemeny report {index} has invalid measurements")
        never_worse = bool(np.all(objectives <= borda_objectives))
        exact_rate = (
            None
            if exact.size == 0
            else float(np.mean(objectives[: exact.size] == exact))
        )
        deterministic = bool(report.get("deterministic_seed", True))
        within_time_limit = bool(report.get("within_time_limit", True))
        passed = (
            never_worse
            and deterministic
            and within_time_limit
            and exact_rate is not None
            and exact_rate >= exact_min
        )
        row = {
            "config": dict(report.get("config", {})),
            "wall_seconds": seconds,
            "instance_count": int(objectives.size),
            "never_worse_than_borda": never_worse,
            "exact_optimum_rate": exact_rate,
            "deterministic_seed": deterministic,
            "within_time_limit": within_time_limit,
            "runtime_cells": list(report.get("runtime_cells", ())),
            "passed": passed,
        }
        rows.append(row)
        if passed:
            eligible.append(row)
    winner = min(eligible, key=lambda row: row["wall_seconds"]) if eligible else None
    failures = [] if winner else ["no Kemeny budget met exact, Borda, and determinism gates"]
    return PilotResult(
        name="kemeny_solver",
        status="passed" if winner else "failed",
        selected={} if winner is None else winner["config"],
        measurements={"candidates": rows},
        failures=failures,
        config_digest=object_sha256(config),
    )


def kemeny_benchmark(
    *, config: dict[str, Any], seed: int, max_passes: int = 10_000
) -> PilotResult:
    """Run the complete synthetic Kemeny quality/runtime grid."""

    patch_counts = tuple(int(item) for item in config.get("patch_counts", (16,)))
    ballot_counts = tuple(int(item) for item in config.get("ballot_counts", (4,)))
    starts = tuple(int(item) for item in config.get("starts", (16,)))
    neighborhoods = tuple(str(item) for item in config.get("neighborhoods", ("insertion",)))
    instances = int(config.get("synthetic_instances_per_cell", 20))
    time_limits = tuple(
        sorted(set(float(item) for item in config.get("time_limits_seconds", (30,))))
    )
    if (
        not patch_counts
        or not ballot_counts
        or not starts
        or not neighborhoods
        or not time_limits
        or time_limits[0] <= 0
        or instances <= 0
        or 16 not in patch_counts
    ):
        raise ValueError("invalid Kemeny benchmark grid; patch_counts must include 16")
    quality_profiles: list[np.ndarray] = []
    exact_objectives: list[int] = []
    for ballot_count in ballot_counts:
        for instance in range(instances):
            rng = np.random.default_rng(
                stable_seed(seed, "kemeny-exact", ballot_count, instance)
            )
            ballots = _synthetic_ballots(
                rng, ballot_count=ballot_count, item_count=16
            )
            quality_profiles.append(ballots)
            exact_objectives.append(_exact_kemeny_objective_dp(ballots))
    reports: list[dict[str, Any]] = []
    for neighborhood in neighborhoods:
        for n_starts in starts:
            objectives: list[int] = []
            borda_objectives: list[int] = []
            repeated_ok = True
            start_time = time.perf_counter()
            for index, ballots in enumerate(quality_profiles):
                solver_seed = stable_seed(seed, "kemeny-solver", index)
                result = kemeny_young(
                    ballots,
                    n_starts=n_starts,
                    max_passes=max_passes,
                    neighborhood=neighborhood,  # type: ignore[arg-type]
                    seed=solver_seed,
                )
                repeated = kemeny_young(
                    ballots,
                    n_starts=n_starts,
                    max_passes=max_passes,
                    neighborhood=neighborhood,  # type: ignore[arg-type]
                    seed=solver_seed,
                )
                objectives.append(result.objective)
                borda_objectives.append(kemeny_objective(borda(ballots), ballots))
                repeated_ok = repeated_ok and result.objective == repeated.objective
                repeated_ok = repeated_ok and np.array_equal(
                    result.ranking, repeated.ranking
                )
            runtime_cells: list[dict[str, Any]] = []
            for item_count in patch_counts:
                for ballot_count in ballot_counts:
                    cell_start = time.perf_counter()
                    for instance in range(instances):
                        rng = np.random.default_rng(
                            stable_seed(
                                seed,
                                "kemeny-runtime",
                                item_count,
                                ballot_count,
                                instance,
                            )
                        )
                        ballots = _synthetic_ballots(
                            rng,
                            ballot_count=ballot_count,
                            item_count=item_count,
                        )
                        kemeny_young(
                            ballots,
                            n_starts=n_starts,
                            max_passes=max_passes,
                            neighborhood=neighborhood,  # type: ignore[arg-type]
                            seed=stable_seed(seed, "runtime-solver", instance),
                        )
                    elapsed = time.perf_counter() - cell_start
                    runtime_cells.append(
                        {
                            "item_count": item_count,
                            "ballot_count": ballot_count,
                            "instance_count": instances,
                            "wall_seconds": elapsed,
                            "seconds_per_instance": elapsed / instances,
                        }
                    )
            wall = time.perf_counter() - start_time
            worst_seconds = max(
                float(cell["seconds_per_instance"]) for cell in runtime_cells
            )
            eligible_limits = [limit for limit in time_limits if worst_seconds <= limit]
            selected_time_limit = eligible_limits[0] if eligible_limits else time_limits[-1]
            reports.append(
                {
                    "config": {
                        "n_starts": n_starts,
                        "neighborhood": neighborhood,
                        "max_passes": max_passes,
                        "time_limit_seconds": selected_time_limit,
                    },
                    "objectives": objectives,
                    "borda_objectives": borda_objectives,
                    "exact_objectives": exact_objectives,
                    "wall_seconds": wall,
                    "deterministic_seed": repeated_ok,
                    "within_time_limit": bool(eligible_limits),
                    "runtime_cells": runtime_cells,
                    "exact_solver": "subset_dp",
                    "exact_item_count": 16,
                }
            )
    result = kemeny_solver_pilot(reports, config=config)
    result.measurements.update(
        {
            "seed": seed,
            "max_passes": max_passes,
            "exact_instance_count": len(quality_profiles),
            "producer": "synthetic_profiles_with_exact_subset_dp",
        }
    )
    return result


__all__ = [
    "_exact_kemeny_objective_dp",
    "_synthetic_ballots",
    "kemeny_benchmark",
    "kemeny_solver_pilot",
]
