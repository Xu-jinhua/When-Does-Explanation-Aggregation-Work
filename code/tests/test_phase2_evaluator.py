from __future__ import annotations

import unittest

import numpy as np

from xai_ensemble.phase2.evaluator import (
    FillReference,
    build_masked_inputs,
    evaluate_reference_model,
    evaluate_reference_model_bank,
)


class NumpyMaskingTests(unittest.TestCase):
    def test_p14_topk_removed_and_retained_masks_are_complements(self) -> None:
        image = np.linspace(0.0, 1.0, 28 * 28, dtype=np.float32).reshape(1, 1, 28, 28)
        fill = FillReference(
            values=np.asarray([0.5], dtype=np.float32),
            source_split="train",
            artifact_id="dataset-mean-sha256",
        )
        result = build_masked_inputs(image, [[0, 1, 2, 3]], fill, patch_size=14, k=1)
        self.assertEqual(result.patch_mask.shape, (1, 2, 2))
        self.assertEqual(int(np.sum(result.patch_mask)), 1)
        np.testing.assert_allclose(result.removed[0, 0, :14, :14], 0.5)
        np.testing.assert_allclose(result.removed[0, 0, 14:, :], image[0, 0, 14:, :])
        np.testing.assert_allclose(result.retained[0, 0, :14, :14], image[0, 0, :14, :14])
        np.testing.assert_allclose(result.retained[0, 0, 14:, :], 0.5)
        np.testing.assert_array_equal(result.selected_patch_indices, [[0]])

    def test_shape_rank_range_and_train_fill_invariants_are_strict(self) -> None:
        image = np.zeros((1, 3, 28, 28), dtype=np.float32)
        fill = FillReference(0.0, "train", "mu_d")
        with self.assertRaises(ValueError):
            build_masked_inputs(image[:, :, :27], [[0, 1]], fill, patch_size=14, k=1)
        with self.assertRaises(ValueError):
            build_masked_inputs(image, [[0, 1, 1, 3]], fill, patch_size=14, k=1)
        with self.assertRaises(ValueError):
            FillReference(0.0, "test", "leaky")


try:
    import torch
except ImportError:  # pragma: no cover - depends on optional environment
    torch = None


@unittest.skipIf(torch is None, "torch optional dependency is not installed")
class TorchReferenceEvaluatorTests(unittest.TestCase):
    def test_rule_bank_matches_independent_rules_and_bounds_forward_batch(self) -> None:
        class RecordingModel(torch.nn.Module):  # type: ignore[union-attr]
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(self, inputs):
                self.batch_sizes.append(int(inputs.shape[0]))
                means = inputs.mean(dim=(1, 2, 3))
                return torch.stack((0.5 - means, means - 0.5), dim=1)

        images = np.linspace(0.0, 1.0, 5 * 28 * 28, dtype=np.float32).reshape(5, 1, 28, 28)
        ranks = {
            "forward": np.broadcast_to(np.arange(4), (5, 4)),
            "reverse": np.broadcast_to(np.arange(3, -1, -1), (5, 4)),
            "mixed": np.broadcast_to(np.asarray([1, 3, 0, 2]), (5, 4)),
        }
        common = {
            "true_labels": np.zeros(5, dtype=np.int64),
            "target_labels": np.zeros(5, dtype=np.int64),
            "fill_reference": FillReference(0.25, "train", "mu_d"),
            "reference_model_id": "f_ref",
            "patch_size": 14,
            "k": 1,
            "batch_size": 7,
            "autocast": False,
            "require_target_matches_clean": False,
            "clean_predictions": np.zeros(5, dtype=np.int64),
        }
        model = RecordingModel()
        bank = evaluate_reference_model_bank(model, images, ranks, **common)

        self.assertEqual(tuple(bank), tuple(ranks))
        self.assertLessEqual(max(model.batch_sizes), 7)
        self.assertEqual(sum(model.batch_sizes), 2 * len(images) * len(ranks))
        for name, values in ranks.items():
            independent = evaluate_reference_model(RecordingModel(), images, values, **common)
            np.testing.assert_array_equal(
                bank[name].removed_predictions, independent.removed_predictions
            )
            np.testing.assert_array_equal(
                bank[name].retained_predictions, independent.retained_predictions
            )
            np.testing.assert_array_equal(
                bank[name].selected_patch_indices,
                independent.selected_patch_indices,
            )
            for metric, contributions in independent.stats.contributions().items():
                np.testing.assert_array_equal(
                    bank[name].stats.contributions()[metric], contributions
                )

    def test_batched_common_reference_model_returns_trace_and_stats(self) -> None:
        class MeanThresholdModel(torch.nn.Module):  # type: ignore[union-attr]
            def forward(self, inputs):
                means = inputs.mean(dim=(1, 2, 3))
                return torch.stack((0.5 - means, means - 0.5), dim=1)

        images = np.stack(
            (
                np.zeros((1, 28, 28), dtype=np.float32),
                np.ones((1, 28, 28), dtype=np.float32),
            )
        )
        ranks = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
        trace = evaluate_reference_model(
            MeanThresholdModel(),
            images,
            ranks,
            true_labels=[0, 1],
            target_labels=[0, 1],
            fill_reference=FillReference(0.5, "train", "mu_d"),
            reference_model_id="full-data-reference",
            sample_ids=["zero", "one"],
            patch_size=14,
            k=1,
            batch_size=1,
            autocast=True,
        )
        np.testing.assert_array_equal(trace.clean_predictions, [0, 1])
        self.assertEqual(trace.stats.n_samples, 2)
        self.assertEqual(trace.selected_patch_indices.shape, (2, 1))
        self.assertFalse(trace.autocast_used)
        records = list(trace.records())
        self.assertEqual(records[0]["sample_id"], "zero")
        self.assertEqual(records[1]["target_label"], 1)

    def test_reference_target_mismatch_is_rejected(self) -> None:
        class ConstantModel(torch.nn.Module):  # type: ignore[union-attr]
            def forward(self, inputs):
                return torch.stack(
                    (torch.ones(inputs.shape[0]), torch.zeros(inputs.shape[0])), dim=1
                )

        with self.assertRaises(ValueError):
            evaluate_reference_model(
                ConstantModel(),
                np.zeros((1, 1, 28, 28), dtype=np.float32),
                [[0, 1, 2, 3]],
                true_labels=[0],
                target_labels=[1],
                fill_reference=FillReference(0.0, "train", "mu_d"),
                reference_model_id="f_ref",
                k=1,
            )

    def test_supplied_clean_predictions_skip_clean_forward_and_bound_model_batch(self) -> None:
        class RecordingModel(torch.nn.Module):  # type: ignore[union-attr]
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(self, inputs):
                self.batch_sizes.append(int(inputs.shape[0]))
                if bool(torch.all(inputs == 0.25, dim=(1, 2, 3)).any()):
                    raise AssertionError("unmasked images must not be forwarded")
                means = inputs.mean(dim=(1, 2, 3))
                return torch.stack((0.5 - means, means - 0.5), dim=1)

        model = RecordingModel()
        trace = evaluate_reference_model(
            model,
            np.full((5, 1, 28, 28), 0.25, dtype=np.float32),
            np.broadcast_to(np.arange(4), (5, 4)),
            true_labels=np.zeros(5, dtype=np.int64),
            target_labels=np.ones(5, dtype=np.int64),
            fill_reference=FillReference(0.75, "train", "mu_d"),
            reference_model_id="f_ref",
            patch_size=14,
            k=1,
            batch_size=3,
            require_target_matches_clean=False,
            clean_predictions=np.ones(5, dtype=np.int64),
        )

        self.assertEqual(sum(model.batch_sizes), 10)
        self.assertLessEqual(max(model.batch_sizes), 3)
        np.testing.assert_array_equal(trace.clean_predictions, np.ones(5))

    def test_forward_batch_limit_also_applies_when_clean_is_inferred(self) -> None:
        class RecordingModel(torch.nn.Module):  # type: ignore[union-attr]
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(self, inputs):
                self.batch_sizes.append(int(inputs.shape[0]))
                means = inputs.mean(dim=(1, 2, 3))
                return torch.stack((1.0 - means, means), dim=1)

        model = RecordingModel()
        evaluate_reference_model(
            model,
            np.zeros((3, 1, 28, 28), dtype=np.float32),
            np.broadcast_to(np.arange(4), (3, 4)),
            true_labels=np.zeros(3, dtype=np.int64),
            target_labels=np.zeros(3, dtype=np.int64),
            fill_reference=FillReference(0.5, "train", "mu_d"),
            reference_model_id="f_ref",
            patch_size=14,
            k=1,
            batch_size=2,
        )

        self.assertEqual(sum(model.batch_sizes), 9)
        self.assertLessEqual(max(model.batch_sizes), 2)

    def test_reused_clean_predictions_preserve_metrics(self) -> None:
        class MeanThresholdModel(torch.nn.Module):  # type: ignore[union-attr]
            def forward(self, inputs):
                means = inputs.mean(dim=(1, 2, 3))
                return torch.stack((0.5 - means, means - 0.5), dim=1)

        kwargs = {
            "reference_model": MeanThresholdModel(),
            "images": np.stack(
                (
                    np.zeros((1, 28, 28), dtype=np.float32),
                    np.ones((1, 28, 28), dtype=np.float32),
                )
            ),
            "ranks": np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]]),
            "true_labels": [0, 1],
            "target_labels": [0, 1],
            "fill_reference": FillReference(0.5, "train", "mu_d"),
            "reference_model_id": "f_ref",
            "patch_size": 14,
            "k": 1,
            "batch_size": 2,
            "autocast": False,
        }
        recomputed = evaluate_reference_model(**kwargs)
        reused = evaluate_reference_model(
            **kwargs,
            clean_predictions=recomputed.clean_predictions,
        )

        np.testing.assert_array_equal(reused.removed_predictions, recomputed.removed_predictions)
        np.testing.assert_array_equal(reused.retained_predictions, recomputed.retained_predictions)
        for metric, values in recomputed.stats.contributions().items():
            np.testing.assert_array_equal(reused.stats.contributions()[metric], values)

    def test_perturbed_metrics_use_unmasked_prediction_not_fixed_target(self) -> None:
        class ConstantZeroModel(torch.nn.Module):  # type: ignore[union-attr]
            def forward(self, inputs):
                return torch.stack(
                    (torch.ones(inputs.shape[0]), torch.zeros(inputs.shape[0])), dim=1
                )

        trace = evaluate_reference_model(
            ConstantZeroModel(),
            np.zeros((1, 1, 28, 28), dtype=np.float32),
            [[0, 1, 2, 3]],
            true_labels=[0],
            target_labels=[0],
            fill_reference=FillReference(0.0, "train", "mu_d"),
            reference_model_id="f_ref",
            patch_size=14,
            k=1,
            require_target_matches_clean=False,
            clean_predictions=[1],
        )

        self.assertTrue(bool(trace.stats.removed_changed[0]))
        self.assertTrue(bool(trace.stats.retained_changed[0]))
        self.assertEqual(trace.stats.values().C, 1.0)
        self.assertEqual(trace.stats.values().Cbar, 1.0)


if __name__ == "__main__":
    unittest.main()
