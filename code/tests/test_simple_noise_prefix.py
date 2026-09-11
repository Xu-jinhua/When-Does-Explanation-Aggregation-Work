from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xai_ensemble.simple.noise_prefix import evaluator
from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
from xai_ensemble.simple.noise_prefix.evaluator import aggregate_prefix_bank
from xai_ensemble.simple.noise_prefix.inputs import _fidelity_order
from xai_ensemble.simple.noise_prefix.scheduler import planned_job_ids, submit_plan
from xai_ensemble.simple.noise_prefix.summary import _compact_per_q, _csv_text, _optimum_rows
from xai_ensemble.simple.rank_ready import existing_rank_ready_payload
from xai_ensemble.simple.scheduler import SimpleJobStore

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/simple/paper-noise-prefix-sweep.yaml"


def test_prefix_config_expands_formal_q_rule_and_task_space() -> None:
    experiment = load_noise_prefix_experiment(CONFIG)

    assert experiment.q_values == tuple(range(2, 12))
    assert experiment.rules == ("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze")
    assert len(experiment.cells()) == 4
    assert len(experiment.evaluation_tasks()) == 20
    assert all(len(cell.methods) == 11 for cell in experiment.cells())


def test_fidelity_order_is_descending_with_method_id_tie_break() -> None:
    experiment = load_noise_prefix_experiment(CONFIG)
    cell = experiment.cells()[0]
    values = {method: 0.1 for method in cell.methods}
    values[cell.methods[0]] = 0.9
    values[cell.methods[1]] = 0.8
    manifest = {"metrics": {f"single__{method}": {"F": value} for method, value in values.items()}}

    fidelity, ordered = _fidelity_order(cell, manifest)

    assert fidelity == values
    assert ordered[:2] == cell.methods[:2]
    assert ordered[2:] == tuple(sorted(cell.methods[2:]))


def test_prefix_aggregation_uses_exact_first_q_ballots_and_excludes_q11() -> None:
    experiment = load_noise_prefix_experiment(CONFIG)
    task = experiment.evaluation_tasks()[0]
    generator = np.random.default_rng(19)
    ballots = np.stack(
        [np.stack([generator.permutation(6) for _ in range(11)], axis=0) for _ in range(3)],
        axis=0,
    )
    scores = {q: generator.normal(size=(3, 6)).astype(np.float32) for q in experiment.q_values[:-1]}

    bank, statistics = aggregate_prefix_bank(
        experiment,
        task,
        ballots=ballots,
        simple_scores_by_q=scores,
        indices=np.asarray([7, 11, 13]),
        device="cpu",
    )

    assert len(bank) == 45
    assert set(statistics) == {f"q{q:02d}__kemeny" for q in range(2, 11)}
    assert not any(name.startswith("q11__") for name in bank)
    expected_q2_borda = np.argsort(
        np.argsort(-((5 - ballots[:, :2]).sum(axis=1)), axis=1, kind="stable"),
        axis=1,
        kind="stable",
    )
    np.testing.assert_array_equal(bank["q02__borda"], expected_q2_borda)


def test_scheduler_has_clean_dependencies_and_disjoint_database(tmp_path: Path) -> None:
    original = load_noise_prefix_experiment(CONFIG)
    experiment = replace(
        original,
        storage=replace(
            original.storage,
            remote_root=str(tmp_path / "remote"),
            scratch_root=tmp_path / "scratch",
            spool_root=tmp_path / "spool",
            spool_min_free_bytes=0,
        ),
        runtime=replace(
            original.runtime,
            database_path=tmp_path / "jobs.sqlite3",
            log_directory=tmp_path / "logs",
            input_catalog_path=tmp_path / "catalog.json",
            shared_cache_root=tmp_path / "shared-cache",
        ),
    )
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )

    result = submit_plan(experiment, store, scan_existing=False)
    jobs = store.jobs()
    clean = {job.job_id for job in jobs if "--clean--" in job.job_id}

    assert result["planned"] == 20
    assert planned_job_ids(experiment) == {job.job_id for job in jobs}
    assert len(clean) == 4
    assert all(not job.dependencies for job in jobs if job.job_id in clean)
    assert all(
        len(job.dependencies) == 1 and job.dependencies[0] in clean
        for job in jobs
        if job.job_id not in clean
    )


def test_missing_input_catalog_is_prepared_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = type("Runtime", (), {"input_catalog_path": tmp_path / "catalog.json"})()
    experiment = type("Experiment", (), {"runtime": runtime})()
    calls = []
    expected = {"catalog_digest": "c" * 64}

    monkeypatch.setattr(
        evaluator,
        "prepare_input_catalog",
        lambda value: calls.append(value) or expected,
    )
    monkeypatch.setattr(evaluator, "load_input_catalog", lambda _value: expected)

    assert evaluator._load_or_prepare_input_catalog(experiment) == expected
    assert calls == [experiment]

    runtime.input_catalog_path.write_text("{}", encoding="utf-8")
    assert evaluator._load_or_prepare_input_catalog(experiment) == expected
    assert calls == [experiment]


def test_missing_rank_ready_sidecar_fails_without_attribution_fallback() -> None:
    calls = []

    class Store:
        def exists(self, relative_path: str) -> bool:
            calls.append(relative_path)
            return False

    with pytest.raises(FileNotFoundError, match="never falls back to full attribution"):
        existing_rank_ready_payload(
            Store(),  # type: ignore[arg-type]
            source_payload={"sha256": "a" * 64},
            simpleavg_normalization="minmax",
        )

    assert len(calls) == 1
    assert calls[0].endswith(".receipt.json")


def test_optimum_direction_and_tie_policy() -> None:
    common = {
        "cell": "cell",
        "dataset": "dataset",
        "model": "model",
        "rule": "borda",
        "condition": "clean",
        "value_kind": "quality",
        "method_prefix": "[]",
    }
    measurements = [
        {**common, "q": 2, "metric": "F", "value": 0.5},
        {**common, "q": 3, "metric": "F", "value": 0.7},
        {**common, "q": 4, "metric": "F", "value": 0.7},
        {**common, "q": 2, "metric": "Fbar", "value": 0.3},
        {**common, "q": 3, "metric": "Fbar", "value": 0.2},
        {**common, "q": 4, "metric": "Fbar", "value": 0.25},
        {
            **common,
            "q": 2,
            "condition": "gaussian",
            "value_kind": "robustness",
            "metric": "F",
            "value": -0.1,
        },
        {
            **common,
            "q": 3,
            "condition": "gaussian",
            "value_kind": "robustness",
            "metric": "F",
            "value": 0.2,
        },
    ]

    rows = _optimum_rows(measurements)
    by_metric = {row["metric"]: row for row in rows if row["value_kind"] == "quality"}

    assert by_metric["F"]["direction"] == "max"
    assert by_metric["F"]["best_q"] == 3
    assert by_metric["F"]["tied_q"] == "[3,4]"
    assert by_metric["Fbar"]["direction"] == "min"
    assert by_metric["Fbar"]["best_q"] == 3
    robustness = next(row for row in rows if row["value_kind"] == "robustness")
    assert robustness["direction"] == "max"
    assert robustness["best_q"] == 3


def test_compact_per_q_is_lossless_and_one_row_per_cell_q_rule() -> None:
    measurements = []
    for condition, value_kind, offset in (
        ("clean", "quality", 0.0),
        ("gaussian-0.15", "conditioned_quality", 0.1),
        ("gaussian-0.15", "robustness", -0.1),
    ):
        for metric_index, metric in enumerate(("F", "Fbar", "C", "Cbar")):
            measurements.append(
                {
                    "cell": "dataset--model",
                    "dataset": "dataset",
                    "model": "model",
                    "q": 2,
                    "rule": "borda",
                    "condition": condition,
                    "value_kind": value_kind,
                    "metric": metric,
                    "value": metric_index + offset,
                    "method_prefix": '["A","B"]',
                    "q11_reference": False,
                }
            )
    summary = {
        "sweep_id": "sweep",
        "sweep_digest": "a" * 64,
        "summary_digest": "b" * 64,
        "input_catalog_digest": "c" * 64,
        "science": {"robustness_direction": "max"},
        "manifest_content_digests": {"task": "d" * 64},
        "conditions": [
            {"condition_id": "clean", "kind": "clean"},
            {"condition_id": "gaussian-0.15", "kind": "gaussian"},
        ],
        "measurements": measurements,
    }

    compact = _compact_per_q(summary)

    assert compact["schema"] == "simple-noise-prefix-per-q-v1"
    assert compact["counts"] == {
        "rows": 1,
        "cells": 1,
        "q_values": 1,
        "rules": 1,
        "perturbations": 1,
    }
    row = compact["rows"][0]
    assert row["method_prefix"] == ["A", "B"]
    assert row["quality"]["F"] == 0.0
    assert row["perturbed_quality"]["gaussian-0.15"]["F"] == 0.1
    assert row["robustness"]["gaussian-0.15"]["F"] == -0.1


def test_prefix_summary_csv_uses_repository_safe_lf_line_endings() -> None:
    payload = _csv_text(({"column": "value"},), ("column",)).encode()

    assert payload == b"column\nvalue\n"
    assert b"\r" not in payload
