from __future__ import annotations

import math

import numpy as np
import pytest

from xai_ensemble.simple.noise_prefix.anchored import (
    FidelityAnchoredPrefix,
    combine_fidelity_anchored_prefixes,
    fidelity_anchored_prefix,
    fidelity_anchored_subset,
    fit_topk_subset_mallows,
    fit_topk_subset_mallows_gof,
    select_fidelity_anchored_prefix,
    select_joint_fidelity_anchored_prefix,
    topk_set_distances,
)


def test_topk_subset_mallows_has_exact_combinatorial_normalizer() -> None:
    fit = fit_topk_subset_mallows([2, 2], n_items=6, k=2)

    assert fit.theta == 0.0
    assert fit.boundary is True
    assert fit.expected_distance == pytest.approx(4.0 / 3.0)
    assert fit.log_normalizer == pytest.approx(math.log(math.comb(6, 2)))


def test_topk_subset_mallows_matches_the_observed_sufficient_statistic() -> None:
    fit = fit_topk_subset_mallows([0, 1, 1, 2], n_items=6, k=2)

    assert fit.theta > 0.0
    assert fit.expected_distance == pytest.approx(1.0, abs=1e-12)
    assert fit.mean_distance == 1.0


def test_topk_subset_mallows_gof_uses_the_exact_analytic_cdf() -> None:
    gof = fit_topk_subset_mallows_gof([0, 1, 1, 1, 1, 2], n_items=4, k=2)

    assert gof.fit.theta == 0.0
    assert gof.ks_statistic == pytest.approx(0.0, abs=1e-15)
    assert gof.total_variation == pytest.approx(0.0, abs=1e-15)
    assert gof.empirical_probabilities == pytest.approx((1 / 6, 4 / 6, 1 / 6))
    assert gof.fitted_probabilities == pytest.approx((1 / 6, 4 / 6, 1 / 6))


def test_topk_set_distance_ignores_irrelevant_tail_order() -> None:
    ballots = np.asarray(
        [
            [
                [0, 1, 2, 3],
                [2, 3, 0, 1],
            ]
        ],
        dtype=np.int64,
    )

    distances = topk_set_distances(ballots, [[0, 2]], k=2)

    np.testing.assert_array_equal(distances, [[1, 1]])


def test_fidelity_anchor_prefers_a_center_near_successful_explanations() -> None:
    contributions = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    near_first = fidelity_anchored_prefix(
        [[0, 1, 2], [0, 1, 2]],
        contributions,
        q=2,
        n_items=6,
        k=2,
    )
    far_from_first = fidelity_anchored_prefix(
        [[2, 1, 0], [2, 1, 0]],
        contributions,
        q=3,
        n_items=6,
        k=2,
    )

    assert near_first.score > far_from_first.score
    assert select_fidelity_anchored_prefix((far_from_first, near_first)).q == 2


def test_fidelity_anchor_rejects_misaligned_marks() -> None:
    with pytest.raises(ValueError, match="share"):
        fidelity_anchored_prefix(
            [[0, 1]],
            [[1.0]],
            q=1,
            n_items=6,
            k=2,
        )


def test_fidelity_anchor_supports_an_arbitrary_method_subset() -> None:
    distances = np.asarray([[0, 2, 1], [0, 2, 1]])
    contributions = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    arbitrary = fidelity_anchored_subset(
        distances,
        contributions,
        selected_positions=[0, 2],
        n_items=6,
        k=2,
    )
    reordered = fidelity_anchored_prefix(
        distances[:, [0, 2, 1]],
        contributions[:, [0, 2, 1]],
        q=2,
        n_items=6,
        k=2,
    )

    assert arbitrary.q == 2
    assert arbitrary.score == pytest.approx(reordered.score)
    assert arbitrary.theta == pytest.approx(reordered.theta)


@pytest.mark.parametrize("positions", [[], [0, 0], [-1], [3]])
def test_fidelity_anchor_rejects_invalid_subset_positions(positions: list[int]) -> None:
    with pytest.raises(ValueError, match="selected_positions"):
        fidelity_anchored_subset(
            [[0, 1, 2]],
            [[1.0, 0.0, 0.0]],
            selected_positions=positions,
            n_items=6,
            k=2,
        )


def test_joint_fidelity_anchor_averages_complementary_geometries() -> None:
    first = FidelityAnchoredPrefix(3, 0.2, 1.0, 1.0, np.asarray([0.1, 0.3]))
    second = FidelityAnchoredPrefix(3, 0.4, 2.0, 2.0, np.asarray([0.2, 0.6]))

    joint = combine_fidelity_anchored_prefixes((first, second))

    assert joint.q == 3
    assert joint.score == pytest.approx(0.3)
    assert joint.component_scores == (0.2, 0.4)
    np.testing.assert_allclose(joint.per_sample_score, [0.15, 0.45])


def test_joint_fidelity_anchor_selects_smallest_q_on_an_exact_tie() -> None:
    first_q2 = FidelityAnchoredPrefix(2, 0.4, 1.0, 1.0, np.asarray([0.4]))
    second_q2 = FidelityAnchoredPrefix(2, 0.2, 1.0, 1.0, np.asarray([0.2]))
    first_q3 = FidelityAnchoredPrefix(3, 0.3, 1.0, 1.0, np.asarray([0.3]))
    second_q3 = FidelityAnchoredPrefix(3, 0.3, 1.0, 1.0, np.asarray([0.3]))

    q2 = combine_fidelity_anchored_prefixes((first_q2, second_q2))
    q3 = combine_fidelity_anchored_prefixes((first_q3, second_q3))

    assert select_joint_fidelity_anchored_prefix((q3, q2)).q == 2


def test_joint_fidelity_anchor_rejects_different_q_values() -> None:
    first = FidelityAnchoredPrefix(2, 0.2, 1.0, 1.0, np.asarray([0.2]))
    second = FidelityAnchoredPrefix(3, 0.3, 1.0, 1.0, np.asarray([0.3]))

    with pytest.raises(ValueError, match="same q"):
        combine_fidelity_anchored_prefixes((first, second))
