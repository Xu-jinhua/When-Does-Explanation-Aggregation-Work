from __future__ import annotations

import numpy as np
import pytest

from xai_ensemble.simple.adversarial_pilot import (
    SaraAttackConfig,
    rank_change_statistics,
    sara_attack_batch,
    sara_postprocess_attribution,
)


def test_sara_postprocessing_sums_positive_channels_and_normalizes() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor(
        [
            [
                [[-2.0, 1.0], [2.0, 0.0]],
                [[1.0, 2.0], [2.0, -1.0]],
            ]
        ]
    )

    observed = sara_postprocess_attribution(values)

    expected = torch.tensor([[[[0.0, 0.75], [1.0, 0.0]]]])
    torch.testing.assert_close(observed, expected)


def test_rank_change_statistics_detects_patch_and_topk_reordering() -> None:
    clean = np.zeros((1, 1, 4, 4), dtype=np.float32)
    adversarial = np.zeros_like(clean)
    clean[:, :, :2, :2] = 4.0
    clean[:, :, :2, 2:] = 3.0
    clean[:, :, 2:, :2] = 2.0
    clean[:, :, 2:, 2:] = 1.0
    adversarial[:] = clean
    adversarial[:, :, :2, :2] = 1.0
    adversarial[:, :, 2:, 2:] = 4.0

    result = rank_change_statistics(clean, adversarial, patch_sizes=(2,), top_k=1)["p2"]

    assert result["rank_changed_fraction"] == 1.0
    assert result["top_k_changed_fraction"] == 1.0
    assert result["top_k_replacement_mean"] == 1.0
    assert 0.0 < result["normalized_kendall_mean"] <= 1.0
    assert 0.0 < result["normalized_footrule_mean"] <= 1.0


def test_sara_attack_keeps_each_candidate_bounded_and_prediction_preserving() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    ).eval()
    torch.manual_seed(7)
    images = torch.rand(2, 3, 4, 4) * 0.8 + 0.1
    config = SaraAttackConfig(epsilon=0.05, steps=2, learning_rate=0.02)

    result = sara_attack_batch(
        model,
        images,
        source_method="DeepLift",
        config=config,
        sample_seeds=(11, 12),
    )

    assert float(result.deltas.abs().max()) <= config.epsilon + 1e-6
    assert float(result.adversarial_images.min()) >= 0.0
    assert float(result.adversarial_images.max()) <= 1.0
    torch.testing.assert_close(result.adversarial_images, images + result.deltas)
    torch.testing.assert_close(result.clean_logits.argmax(1), result.targets)
    torch.testing.assert_close(result.adversarial_logits.argmax(1), result.targets)
    assert bool((result.adversarial_objective <= result.clean_objective).all())
    assert result.random_images.shape == images.shape
    assert result.random_attributions.shape == result.clean_attributions.shape
    assert result.random_objective.shape == result.clean_objective.shape
