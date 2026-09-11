from __future__ import annotations

import unittest

import numpy as np

from xai_ensemble.phase2.aggregation import (
    borda,
    borda_result,
    exact_kemeny_young,
    kemeny_objective,
    kemeny_young,
    rrf_scores,
    schulze_result,
    simple_average,
    simple_average_attributions,
)
from xai_ensemble.phase2.rankings import (
    canonicalize_ranking,
    ranking_to_order,
    sample_hash_tie_break,
    scores_to_ranking_result,
)


class RankingValidationTests(unittest.TestCase):
    def test_strict_base_and_intentional_ties_are_distinct(self) -> None:
        np.testing.assert_array_equal(
            canonicalize_ranking([1, 3, 2], index_base=1), [0, 2, 1]
        )
        with self.assertRaises(ValueError):
            canonicalize_ranking([1, 1, 2], index_base=1)
        np.testing.assert_array_equal(
            canonicalize_ranking(
                [1, 1, 2], index_base=1, tie_policy="item_id"
            ),
            [0, 1, 2],
        )

    def test_sample_hash_tie_break_is_deterministic_and_audited(self) -> None:
        first = sample_hash_tie_break(8, "sample-content-sha256")
        second = sample_hash_tie_break(8, "sample-content-sha256")
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.arange(8))
        result = scores_to_ranking_result(np.zeros(8), tie_break=first)
        np.testing.assert_array_equal(ranking_to_order(result.ranking), np.argsort(first))
        self.assertEqual(result.tie_rate, 1.0)
        self.assertEqual(result.tie_break_policy, "deterministic_key")


class AggregationTests(unittest.TestCase):
    def test_borda_default_and_hash_ties(self) -> None:
        profile = np.asarray([[0, 1], [1, 0]])
        default = borda_result(profile)
        np.testing.assert_array_equal(default.ranking, [0, 1])
        self.assertEqual(default.tie_rate, 1.0)
        keys = np.asarray([1, 0])
        keyed = borda_result(profile, tie_break=keys)
        np.testing.assert_array_equal(keyed.ranking, [1, 0])

    def test_rrf_uses_paper_one_based_positions_and_c60(self) -> None:
        profile = np.asarray([[0, 1, 2], [1, 0, 2]])
        expected = np.asarray(
            [
                (1 / 61 + 1 / 62) / 2,
                (1 / 62 + 1 / 61) / 2,
                1 / 63,
            ]
        )
        np.testing.assert_allclose(rrf_scores(profile), expected)

    def test_simple_average_means_independently_normalized_maps(self) -> None:
        maps = np.asarray([[[-2.0, 0.0], [2.0, 0.0]], [[0.0, 4.0], [0.0, 2.0]]])
        result = simple_average(maps, normalization="max", absolute=True)
        np.testing.assert_allclose(result, [[0.5, 0.5], [0.5, 0.25]])

    def test_formal_simpleavg_zscores_attributions_before_patch_conversion(self) -> None:
        attributions = np.asarray(
            [
                [[[0.0, 1.0], [2.0, 3.0]]],
                [[[0.0, 0.0], [0.0, 10.0]]],
            ]
        )
        transformed = attributions.copy()
        transformed[0] = transformed[0] * 100.0 + 5.0
        transformed[1] = transformed[1] * 0.2 - 3.0
        first = simple_average_attributions(attributions, patch_size=1)
        second = simple_average_attributions(transformed, patch_size=1)
        np.testing.assert_allclose(first.mean_attribution, second.mean_attribution)
        np.testing.assert_allclose(first.patch_scores, second.patch_scores)
        np.testing.assert_array_equal(first.ranking, second.ranking)

    def test_schulze_tie_uses_borda_then_deterministic_key(self) -> None:
        profile = np.asarray([[0, 1, 2], [2, 1, 0]])
        default = schulze_result(profile)
        self.assertGreater(default.tie_rate, 0.0)
        np.testing.assert_array_equal(default.ranking, [0, 1, 2])
        keyed = schulze_result(profile, tie_break=[2, 1, 0])
        np.testing.assert_array_equal(keyed.ranking, [2, 1, 0])

    def test_kemeny_optimizes_kendall_not_borda_surrogate(self) -> None:
        # In this profile the Borda order is not a Kemeny optimum.
        profile = np.asarray(
            [[0, 1, 2, 3], [1, 2, 3, 0], [3, 2, 0, 1]], dtype=np.int64
        )
        borda_ranking = borda(profile)
        exact = exact_kemeny_young(profile)
        heuristic = kemeny_young(profile, n_starts=8, seed=9)
        self.assertEqual(exact.objective, 7)
        self.assertEqual(kemeny_objective(borda_ranking, profile), 8)
        self.assertEqual(heuristic.objective, exact.objective)
        self.assertFalse(np.array_equal(heuristic.ranking, borda_ranking))
        for trace in heuristic.trace:
            self.assertTrue(
                all(
                    later < earlier
                    for earlier, later in zip(
                        trace.objective_path, trace.objective_path[1:], strict=False
                    )
                )
            )

    def test_local_search_matches_bruteforce_oracle_on_small_profile(self) -> None:
        profile = np.asarray(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 2, 4, 3],
                [0, 2, 1, 3, 4],
                [2, 0, 1, 4, 3],
                [0, 1, 3, 2, 4],
            ]
        )
        exact = exact_kemeny_young(profile)
        heuristic = kemeny_young(
            profile, n_starts=16, neighborhood="insertion", seed=3
        )
        self.assertEqual(heuristic.objective, exact.objective)


if __name__ == "__main__":
    unittest.main()
