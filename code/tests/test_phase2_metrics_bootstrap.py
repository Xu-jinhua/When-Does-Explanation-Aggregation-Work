from __future__ import annotations

import unittest

import numpy as np

from xai_ensemble.phase2.bootstrap import (
    bootstrap_oracle_vs_comparator,
    ind_hierarchical_summary,
    paired_class_stratified_bootstrap,
)
from xai_ensemble.phase2.metrics import (
    MetricSufficientStats,
    compute_metrics,
    compute_robustness,
    oracle_best_single,
)


class CurrentPaperMetricTests(unittest.TestCase):
    def _stats(self) -> MetricSufficientStats:
        return MetricSufficientStats.from_predictions(
            true_labels=[0, 1, 2, 3],
            clean_predictions=[0, 0, 2, 3],
            removed_predictions=[1, 1, 0, 3],
            retained_predictions=[0, 0, 0, 3],
            sample_ids=["a", "b", "c", "d"],
        )

    def test_metric_formulas_and_consistency_direction(self) -> None:
        values = compute_metrics(self._stats())
        self.assertAlmostEqual(values.F, 0.25)
        self.assertAlmostEqual(values.Fbar, 0.25)
        # Current C is prediction-change rate.  The legacy prediction-preserve
        # direction would be 0.25 here and must not be inherited.
        self.assertAlmostEqual(values.C, 0.75)
        self.assertAlmostEqual(values.Cbar, 0.25)

    def test_robustness_is_absolute_global_metric_difference(self) -> None:
        clean = self._stats()
        perturbed = MetricSufficientStats.from_predictions(
            true_labels=[0, 1, 2, 3],
            clean_predictions=[0, 0, 2, 3],
            removed_predictions=[0, 0, 2, 3],
            retained_predictions=[0, 0, 2, 3],
            sample_ids=["a", "b", "c", "d"],
        )
        result = compute_robustness(clean, perturbed)
        self.assertAlmostEqual(result.absolute["R_C"], 0.75)
        self.assertAlmostEqual(result.absolute["R_Cbar"], 0.25)
        self.assertAlmostEqual(result.signed_degradation["R_C"], 0.75)
        # Cbar is minimized, so a decrease under perturbation is negative
        # signed degradation even though absolute robustness stays positive.
        self.assertAlmostEqual(result.signed_degradation["R_Cbar"], -0.25)

    def test_oracle_best_single_is_selected_independently_per_metric(self) -> None:
        choices = oracle_best_single(
            {
                "alpha": {"F": 0.8, "Fbar": 0.3, "C": 0.4, "Cbar": 0.2},
                "beta": {"F": 0.7, "Fbar": 0.1, "C": 0.9, "Cbar": 0.3},
            }
        )
        self.assertEqual(choices["F"].method, "alpha")
        self.assertEqual(choices["Fbar"].method, "beta")
        self.assertEqual(choices["C"].method, "beta")
        self.assertEqual(choices["Cbar"].method, "alpha")


class BootstrapTests(unittest.TestCase):
    def test_class_stratification_preserves_class_composition(self) -> None:
        result = paired_class_stratified_bootstrap(
            [0.0, 0.0, 1.0, 1.0],
            class_labels=[0, 0, 1, 1],
            B=100,
            seed=4,
        )
        self.assertEqual(result.estimate, 0.5)
        np.testing.assert_allclose(result.replicates, 0.5)

    def test_oracle_is_reselected_inside_each_bootstrap(self) -> None:
        result = bootstrap_oracle_vs_comparator(
            {
                "a": [1.0, 1.0, 0.0, 0.0],
                "b": [0.0, 0.0, 1.0, 1.0],
            },
            [0.0, 0.0, 0.0, 0.0],
            direction="max",
            B=100,
            seed=5,
        )
        self.assertAlmostEqual(sum(result.selection_frequency.values()), 1.0)
        self.assertIn(result.selected_on_full_data, {"a", "b"})
        self.assertTrue(np.all(result.difference.replicates >= 0.0))

    def test_ind_hierarchy_resamples_family_source_and_class(self) -> None:
        matched = np.zeros((2, 2, 4), dtype=np.float64)
        ind = np.asarray(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[3, 3, 3, 3], [3, 3, 3, 3]],
            ],
            dtype=np.float64,
        )
        summary = ind_hierarchical_summary(
            ind,
            matched,
            class_labels=[0, 0, 1, 1],
            B=200,
            seed=7,
        )
        self.assertEqual(summary.raw_difference.estimate, 2.0)
        np.testing.assert_allclose(summary.family_differences, [1.0, 3.0])
        self.assertEqual(summary.n_families, 2)
        self.assertEqual(summary.n_sources, 2)


if __name__ == "__main__":
    unittest.main()
