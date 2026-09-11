from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from xai_ensemble.simple.relative_robustness import cli as relative_cli
from xai_ensemble.simple.relative_robustness.config import load_experiment
from xai_ensemble.simple.relative_robustness.control import random_rank_bank
from xai_ensemble.simple.relative_robustness.report import (
    RandomControlBank,
    bootstrap_relative_values,
    relative_values_from_means,
)
from xai_ensemble.simple.relative_robustness.scheduler import _external_process_released_gpu

ROOT = Path(__file__).resolve().parents[1]


def _raw_from_oriented(values: np.ndarray) -> np.ndarray:
    signs = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    return values * signs


def _means() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random_oriented = np.broadcast_to(
        np.asarray([0.1, -0.9, 0.1, -0.9], dtype=np.float64),
        (1, 5, 2, 4),
    ).copy()
    naive_oriented = np.broadcast_to(
        np.asarray([0.6, -0.3, 0.6, -0.3], dtype=np.float64),
        (1, 5, 5, 4),
    ).copy()
    noise_oriented = np.broadcast_to(
        np.asarray([0.7, -0.2, 0.7, -0.2], dtype=np.float64),
        (1, 2, 5, 5, 4),
    ).copy()
    naive_oriented[:, 1:] = np.asarray([0.4, -0.4, 0.4, -0.4], dtype=np.float64)
    noise_oriented[:, :, 1:] = np.asarray([0.55, -0.3, 0.55, -0.3], dtype=np.float64)
    return (
        _raw_from_oriented(random_oriented),
        _raw_from_oriented(naive_oriented),
        _raw_from_oriented(noise_oriented),
    )


def test_random_rank_bank_is_strict_and_condition_independent() -> None:
    indices = np.asarray([3, 17, 41], dtype=np.int64)
    first = random_rank_bank(
        indices=indices,
        cell_id="dataset--model",
        patch_count=196,
        seed_bank=(7, 11),
    )
    repeated = random_rank_bank(
        indices=indices,
        cell_id="dataset--model",
        patch_count=196,
        seed_bank=(7, 11),
    )

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(
        np.sort(first, axis=2),
        np.broadcast_to(np.arange(196, dtype=np.int32), first.shape),
    )


def test_relative_robustness_requires_positive_clean_excess_and_orients_fbar() -> None:
    random, naive, noise = _means()
    values = relative_values_from_means(random, naive, noise)

    assert values.naive_rrel[0, 0, 0, 0] == pytest.approx(0.4)
    assert values.noise_rrel[0, 0, 0, 0, 0] == pytest.approx(0.25)
    assert values.rrel_gain[0, 0, 0, 0, 0] == pytest.approx(0.15)
    assert values.perturbed_excess_gain[0, 0, 0, 0, 0] == pytest.approx(0.15)
    # Fbar is minimized in raw form, but the same calculation operates after
    # orientation and therefore retains a positive above-random clean excess.
    assert values.naive_clean_excess[0, 0, 1] == pytest.approx(0.6)
    assert values.noise_clean_excess[0, 0, 0, 1] == pytest.approx(0.7)

    invalid = naive.copy()
    invalid[:, 0, :, 0] = random[:, 0, 0, 0]
    invalid_values = relative_values_from_means(random, invalid, noise)
    assert np.isnan(invalid_values.naive_rrel[:, :, :, 0]).all()


def test_cluster_bootstrap_recomputes_dataset_means_with_known_constant_gain() -> None:
    random, naive, noise = _means()
    samples = 4
    random_bank = RandomControlBank(
        cell="dataset--model",
        dataset="dataset",
        model="model",
        indices=np.asarray([3, 7, 11, 13], dtype=np.int64),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        random_seed_bank=(7, 11),
        values=np.broadcast_to(random[0, :, None], (5, samples, 2, 4)).copy(),
        condition_transition_audit={
            condition: {
                "target_changes_from_clean": 0,
                "unmasked_prediction_changes_from_clean": 0,
            }
            for condition in ("clean", "g", "p", "s", "a")
        },
    )
    result = bootstrap_relative_values(
        random=random_bank,
        naive=np.broadcast_to(naive[0, :, None], (5, samples, 5, 4)).copy(),
        noise=np.broadcast_to(noise[0, :, :, None], (2, 5, samples, 5, 4)).copy(),
        replicates=9,
        seed=5,
        batch_size=4,
    )

    assert result.rrel_gain_replicates.shape == (9, 2, 4, 5, 4)
    assert result.perturbed_excess_gain_replicates.shape == (9, 2, 4, 5, 4)
    assert result.rrel_gain_replicates[:, 0, 0, 0, 0] == pytest.approx(0.15)
    assert result.perturbed_excess_gain_replicates[:, 0, 0, 0, 0] == pytest.approx(0.15)


def test_relative_robustness_config_expands_twenty_phase2_only_controls() -> None:
    experiment = load_experiment(ROOT / "configs/simple/paper-relative-robustness.yaml")

    assert len(experiment.control_tasks()) == 20
    assert len(experiment.random_seed_bank) == 8
    assert all(task.patch_size == 16 and task.k == 20 for task in experiment.control_tasks())


def test_external_release_marker_requires_matching_worker_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "release.json"
    marker.write_text(
        '{"schema_version":1,"job_id":"phase1:example","pid":4321,"token":"secret"}',
        encoding="utf-8",
    )
    environment = (
        b"XAI_SIMPLE_GPU_RELEASE_PATH="
        + str(marker).encode("utf-8")
        + b"\0XAI_SIMPLE_GPU_RELEASE_TOKEN=secret"
        + b"\0XAI_SIMPLE_GPU_RELEASE_JOB=phase1:example\0"
    )
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if str(path) == "/proc/4321/environ":
            return environment
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert _external_process_released_gpu(4321)
    assert not _external_process_released_gpu(4322)


def test_run_option_writes_report_after_controls_complete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment = object()
    monkeypatch.setattr(relative_cli, "_load", lambda _args: experiment)
    from xai_ensemble.simple.relative_robustness import report, scheduler

    monkeypatch.setattr(scheduler, "run_scheduler", lambda *_args, **_kwargs: {"succeeded": 20})
    monkeypatch.setattr(
        report,
        "write_relative_robustness_report",
        lambda *_args, **_kwargs: {"status": "complete", "result_digest": "digest"},
    )

    assert (
        relative_cli._run(
            argparse.Namespace(
                config="unused",
                poll_seconds=1.0,
                headroom_fraction=None,
                report_output_directory="results/relative",
            )
        )
        == 0
    )
    assert '"succeeded": 20' in capsys.readouterr().out
