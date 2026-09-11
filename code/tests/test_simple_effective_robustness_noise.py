from __future__ import annotations

import numpy as np
import pytest

from xai_ensemble.simple.artifacts import ArtifactError
from xai_ensemble.simple.effective_robustness_noise import (
    CellMetricBank,
    _assert_condition_sample_identity,
    _condition_key,
    _ConditionValues,
    _holm_resolution,
    bootstrap_er_gains,
    er_values_from_means,
)


def _raw(oriented: np.ndarray, metric_index: int) -> np.ndarray:
    return oriented if metric_index in (0, 2) else -oriented


def _constant_metric_bank() -> CellMetricBank:
    sample_count = 8
    method_count = 11
    rule_count = 5
    conditions = 5
    metrics = 4
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    reference = np.empty((conditions, sample_count, method_count, metrics), dtype=np.float64)
    naive = np.empty((conditions, sample_count, rule_count, metrics), dtype=np.float64)
    noise = np.empty((2, conditions, sample_count, rule_count, metrics), dtype=np.float64)

    for metric_index in range(metrics):
        methods = np.linspace(0.1, 1.1, method_count, dtype=np.float64)
        naive_clean = np.linspace(0.25, 0.65, rule_count, dtype=np.float64)
        for condition_index in range(conditions):
            scale = 1.0 if condition_index == 0 else 0.5
            reference[condition_index, :, :, metric_index] = _raw(
                np.broadcast_to(scale * methods, (sample_count, method_count)), metric_index
            )
            naive[condition_index, :, :, metric_index] = _raw(
                np.broadcast_to(scale * naive_clean, (sample_count, rule_count)), metric_index
            )
            # The selected NOISE construction has identical clean quality but an
            # oriented perturbed-quality residual of +0.1 for every rule.
            noise_oriented = scale * naive_clean
            if condition_index != 0:
                noise_oriented = noise_oriented + 0.1
            noise[:, condition_index, :, :, metric_index] = _raw(
                np.broadcast_to(noise_oriented, (2, sample_count, rule_count)), metric_index
            )
    return CellMetricBank(
        cell="demo--model",
        dataset="demo",
        model="model",
        labels=labels,
        reference=reference,
        naive=naive,
        noise=noise,
        q_by_geometry={"spearman": 2, "kendall": 3},
        condition_transition_audit={
            condition: {
                "target_changes_from_clean": 0,
                "unmasked_prediction_changes_from_clean": 0,
            }
            for condition in ("clean", "g", "p", "s", "a")
        },
    )


def test_fixed_q_bootstrap_refits_er_curve_and_preserves_known_gain() -> None:
    bank = _constant_metric_bank()
    result = bootstrap_er_gains(bank, replicates=17, seed=7, batch_size=5)

    assert result.naive_er.shape == (4, 5, 4)
    assert result.noise_er.shape == (2, 4, 5, 4)
    assert result.gain.shape == (2, 4, 5, 4)
    assert result.replicates.shape == (17, 2, 4, 5, 4)
    assert result.naive_er == pytest.approx(0.0)
    assert result.noise_er == pytest.approx(0.1)
    assert result.gain == pytest.approx(0.1)
    assert result.replicates == pytest.approx(0.1)


def test_er_value_shape_contract_rejects_misaligned_noise_geometry() -> None:
    bank = _constant_metric_bank()
    reference = bank.reference.mean(axis=1, keepdims=False)[None, ...]
    naive = bank.naive.mean(axis=1, keepdims=False)[None, ...]
    malformed_noise = bank.noise.mean(axis=2, keepdims=False)[None, :1, ...]

    with pytest.raises(ValueError, match="NOISE means"):
        er_values_from_means(reference, naive, malformed_noise)


def test_factory_condition_uses_the_declared_natural_noise_kind() -> None:
    assert _condition_key("factory", kwargs={"kind": "salt_pepper"}) == "p"
    assert _condition_key("adversarial") == "a"


def test_holm_resolution_requires_enough_bootstrap_repetitions() -> None:
    coarse = _holm_resolution(endpoint_count=640, alpha=0.05, replicates=1_999)
    resolved = _holm_resolution(endpoint_count=640, alpha=0.05, replicates=19_999)

    assert coarse["minimum_replicates_for_any_holm_rejection"] == 12_799
    assert coarse["sufficient_for_any_holm_rejection"] is False
    assert resolved["sufficient_for_any_holm_rejection"] is True


def test_cross_condition_identity_allows_changed_predictions_and_targets() -> None:
    clean = _ConditionValues(
        indices=np.asarray([3, 7], dtype=np.int64),
        labels=np.asarray([1, 2], dtype=np.int64),
        targets=np.asarray([1, 2], dtype=np.int64),
        unmasked_predictions=np.asarray([1, 2], dtype=np.int64),
        values={},
    )
    perturbed = _ConditionValues(
        indices=np.asarray([3, 7], dtype=np.int64),
        labels=np.asarray([1, 2], dtype=np.int64),
        targets=np.asarray([1, 0], dtype=np.int64),
        unmasked_predictions=np.asarray([1, 0], dtype=np.int64),
        values={},
    )

    _assert_condition_sample_identity(clean, perturbed, context="test")

    with pytest.raises(ArtifactError, match="labels"):
        _assert_condition_sample_identity(
            clean,
            _ConditionValues(
                indices=np.asarray([3, 7], dtype=np.int64),
                labels=np.asarray([1, 3], dtype=np.int64),
                targets=perturbed.targets,
                unmasked_predictions=perturbed.unmasked_predictions,
                values={},
            ),
            context="test",
        )
