from __future__ import annotations

import os

import numpy as np
import pytest

from xai_ensemble.phase2.aggregation import (
    borda,
    kemeny_objective,
    kemeny_young,
    rrf,
    schulze,
)
from xai_ensemble.phase2.torch_aggregation import (
    _initial_orders,
    aggregate_rankings_torch,
)


def _profile(seed: int, *, samples: int, methods: int, patches: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return np.stack(
        [np.stack([generator.permutation(patches) for _ in range(methods)]) for _ in range(samples)]
    ).astype(np.int64)


def _score_ranks(scores: np.ndarray) -> np.ndarray:
    flat = scores.reshape(scores.shape[0], -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(flat.shape[1]), order.shape),
        axis=1,
    )
    return ranks


def _expected(
    ballots: np.ndarray,
    simple_scores: np.ndarray,
    seeds: tuple[int, ...],
    *,
    starts: int,
    passes: int,
) -> dict[str, np.ndarray]:
    samples, _, patches = ballots.shape
    return {
        "simpleavg": _score_ranks(simple_scores),
        "borda": np.stack([borda(ballots[index]) for index in range(samples)]),
        "rrf": np.stack([rrf(ballots[index]) for index in range(samples)]),
        "kemeny": np.stack(
            [
                kemeny_young(
                    ballots[index],
                    n_starts=starts,
                    max_passes=passes,
                    seed=seeds[index],
                ).ranking
                for index in range(samples)
            ]
        ),
        "schulze": np.stack([schulze(ballots[index]) for index in range(samples)]),
    }


def test_torch_cpu_backend_matches_every_formal_rule() -> None:
    ballots = _profile(17, samples=4, methods=5, patches=7)
    simple_scores = np.random.default_rng(23).normal(size=(4, 7)).astype(np.float32)
    simple_scores[0, 1:3] = 2.0
    seeds = (11, 19, 31, 43)
    starts = 8
    passes = 100

    observed = aggregate_rankings_torch(
        ballots,
        simple_scores,
        requested=("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"),
        rrf_c=60.0,
        kemeny_starts=starts,
        kemeny_max_passes=passes,
        seeds=seeds,
        device="cpu",
        workspace_bytes=64 * 2**20,
    )

    expected = _expected(ballots, simple_scores, seeds, starts=starts, passes=passes)
    assert tuple(observed) == ("simpleavg", "borda", "rrf", "kemeny", "schulze")
    for rule in expected:
        np.testing.assert_array_equal(observed[rule], expected[rule], err_msg=rule)
        np.testing.assert_array_equal(
            np.sort(observed[rule], axis=1),
            np.broadcast_to(np.arange(ballots.shape[2]), observed[rule].shape),
        )


def test_torch_backend_preserves_schulze_and_kemeny_tie_breaks() -> None:
    ballots = np.asarray(
        [
            [[0, 1, 2], [2, 1, 0], [0, 1, 2], [2, 1, 0]],
            [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]],
        ],
        dtype=np.int64,
    )
    seeds = (5, 7)

    observed = aggregate_rankings_torch(
        ballots,
        None,
        requested=("Kemeny", "Schulze"),
        rrf_c=60.0,
        kemeny_starts=8,
        kemeny_max_passes=100,
        seeds=seeds,
        device="cpu",
        workspace_bytes=1,
    )

    for sample in range(len(ballots)):
        np.testing.assert_array_equal(observed["schulze"][sample], schulze(ballots[sample]))
        np.testing.assert_array_equal(
            observed["kemeny"][sample],
            kemeny_young(ballots[sample], n_starts=8, seed=seeds[sample]).ranking,
        )


def test_workspace_chunking_does_not_change_consensus() -> None:
    ballots = _profile(47, samples=5, methods=4, patches=8)
    kwargs = {
        "simple_scores": None,
        "requested": ("Kemeny", "Schulze"),
        "rrf_c": 60.0,
        "kemeny_starts": 8,
        "kemeny_max_passes": 100,
        "seeds": (53, 59, 61, 67, 71),
        "device": "cpu",
    }

    one_instance = aggregate_rankings_torch(ballots, workspace_bytes=1, **kwargs)
    one_chunk = aggregate_rankings_torch(ballots, workspace_bytes=64 * 2**20, **kwargs)

    for rule in one_instance:
        np.testing.assert_array_equal(one_instance[rule], one_chunk[rule], err_msg=rule)
    assert one_instance.statistics == one_chunk.statistics


def test_single_kemeny_start_is_borda_and_never_worsens_its_objective() -> None:
    ballots = _profile(73, samples=4, methods=5, patches=9)
    borda_rankings = np.stack([borda(profile) for profile in ballots])
    fake_schulze = np.flip(borda_rankings, axis=1).copy()
    starts = _initial_orders(
        ballots[0],
        borda_rankings[0],
        fake_schulze[0],
        n_starts=1,
        seed=79,
    )
    np.testing.assert_array_equal(starts, np.argsort(borda_rankings[0])[None, :])

    observed = aggregate_rankings_torch(
        ballots,
        None,
        requested=("Borda", "Kemeny"),
        rrf_c=60.0,
        kemeny_starts=1,
        kemeny_max_passes=1_024,
        seeds=(83, 89, 97, 101),
        device="cpu",
        workspace_bytes=1,
    )
    borda_objectives = np.asarray(
        [
            kemeny_objective(ranking, profile)
            for profile, ranking in zip(ballots, borda_rankings, strict=True)
        ]
    )
    final_objectives = np.asarray(
        [
            kemeny_objective(ranking, profile)
            for profile, ranking in zip(ballots, observed["kemeny"], strict=True)
        ]
    )
    assert bool(np.all(final_objectives <= borda_objectives))

    statistics = observed.statistics["kemeny"]
    assert statistics["sample_count"] == len(ballots)
    assert statistics["search_instance_count"] == len(ballots)
    assert statistics["starts_requested"] == 1
    assert statistics["max_passes"] == 1_024
    assert statistics["cap_hit_count"] == 0
    assert statistics["converged_fraction"] == 1.0
    assert statistics["borda_objective_sum"] == int(borda_objectives.sum())
    assert statistics["final_objective_sum"] == int(final_objectives.sum())
    assert statistics["objective_improvement_sum"] == int(
        np.sum(borda_objectives - final_objectives)
    )


def test_kemeny_statistics_report_a_real_move_cap() -> None:
    ballots = _profile(103, samples=6, methods=7, patches=20)
    observed = aggregate_rankings_torch(
        ballots,
        None,
        requested=("Kemeny",),
        rrf_c=60.0,
        kemeny_starts=1,
        kemeny_max_passes=1,
        seeds=(107, 109, 113, 127, 131, 137),
        device="cpu",
        workspace_bytes=1,
    )

    statistics = observed.statistics["kemeny"]
    assert statistics["max_moves"] == 1
    assert 0 < statistics["cap_hit_count"] <= len(ballots)
    assert statistics["converged_count"] + statistics["cap_hit_count"] == len(ballots)
    assert statistics["converged_fraction"] < 1.0


@pytest.mark.skipif(
    os.environ.get("XAI_RUN_CUDA_TESTS") != "1",
    reason="set XAI_RUN_CUDA_TESTS=1 on an idle experiment GPU",
)
def test_cuda_backend_matches_cpu_backend() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    ballots = _profile(29, samples=3, methods=4, patches=6)
    simple_scores = np.random.default_rng(31).normal(size=(3, 6)).astype(np.float32)
    kwargs = {
        "requested": ("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"),
        "rrf_c": 60.0,
        "kemeny_starts": 8,
        "kemeny_max_passes": 100,
        "seeds": (37, 41, 43),
        "workspace_bytes": 64 * 2**20,
    }

    expected = aggregate_rankings_torch(ballots, simple_scores, device="cpu", **kwargs)
    observed = aggregate_rankings_torch(ballots, simple_scores, device="cuda:0", **kwargs)

    for rule in expected:
        np.testing.assert_array_equal(observed[rule], expected[rule], err_msg=rule)
    assert observed.statistics == expected.statistics
