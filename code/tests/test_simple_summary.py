from __future__ import annotations

from pathlib import Path

import pytest

from xai_ensemble.phase2.metrics import QUALITY_METRICS
from xai_ensemble.simple.artifacts import PHASE2_SCHEMA_VERSION, ArtifactError
from xai_ensemble.simple.config import load_experiment
from xai_ensemble.simple.summary import (
    PAPER_RULES,
    _analysis,
    _check_metrics_manifest,
    _columns,
    _metrics_csv_text,
    _metrics_member_order,
    _metrics_tex_text,
    _table_rows,
    _validate_manifest,
    build_metrics_summary,
)


def _metrics(offset: float) -> dict[str, float]:
    return {
        "F": 0.5 + offset,
        "Fbar": 0.5 - offset,
        "C": 0.4 + offset,
        "Cbar": 0.4 - offset,
    }


def test_table1_summary_selects_oracle_single_per_metric_cell() -> None:
    rules = {
        "single__Alpha": _metrics(0.1),
        "single__Beta": _metrics(-0.1),
        "simpleavg": _metrics(0.05),
        "borda": _metrics(0.04),
        "kemeny": _metrics(0.03),
        "rrf": _metrics(0.02),
        "schulze": _metrics(0.01),
    }
    clean = {
        "sample_count": 8,
        "methods": ["Alpha", "Beta"],
        "metrics": rules,
    }
    manifests = {"clean": clean}
    for index, noise in enumerate(("g", "p", "s", "a"), start=1):
        perturbed = {
            rule: {
                metric: value + (index * 0.01 if rule.endswith("Alpha") else index * 0.02)
                for metric, value in values.items()
            }
            for rule, values in rules.items()
        }
        manifests[noise] = {
            "sample_count": 8,
            "methods": ["Alpha", "Beta"],
            "metrics": perturbed,
            "robustness": {
                rule: {
                    "absolute": {
                        metric: abs(values[metric] - perturbed[rule][metric]) for metric in values
                    }
                }
                for rule, values in rules.items()
            },
        }

    rows = _table_rows(manifests)
    best = rows[0]

    assert best["selected_sources"]["quality"] == {
        "F": "Alpha",
        "Fbar": "Alpha",
        "C": "Alpha",
        "Cbar": "Alpha",
    }
    assert all(
        method == "Alpha"
        for noise in ("g", "p", "s", "a")
        for method in best["selected_sources"]["robustness"][noise].values()
    )
    assert best["perturbed_quality"]["g"]["F"] == pytest.approx(0.61)
    assert best["robustness"]["g"] == pytest.approx(
        {"F": 0.01, "Fbar": -0.01, "C": 0.01, "Cbar": -0.01}
    )
    assert all(direction == "max" for column, direction in _columns() if column.startswith("R_"))
    assert len(rows) == 6

    analysis = _analysis(
        (
            {
                "dataset": "example",
                "model": "example-model",
                "sample_count": 8,
                "rows": rows,
            },
        )
    )
    assert analysis["comparison_cell_count"] == 20
    assert all(sum(counts.values()) == 20 for counts in analysis["versus_best_individual"].values())
    assert analysis["groups"]["quality"]["comparison_cell_count"] == 4
    assert analysis["groups"]["robustness_all"]["comparison_cell_count"] == 16
    assert all(
        analysis["groups"][noise]["comparison_cell_count"] == 4
        for noise in ("gaussian", "salt_pepper", "speckle", "adversarial")
    )
    assert len(analysis["by_dataset_model"]) == 1
    assert analysis["oracle_source_counts"] == {
        "selection_count": 20,
        "overall": {"Alpha": 20},
        "quality": {"Alpha": 4},
        "robustness_all": {"Alpha": 16},
        "by_noise": {
            "gaussian": {"Alpha": 4},
            "salt_pepper": {"Alpha": 4},
            "speckle": {"Alpha": 4},
            "adversarial": {"Alpha": 4},
        },
    }


def test_table1_summary_reuses_task_identical_prior_experiment_manifest() -> None:
    experiment = load_experiment(
        Path(__file__).parents[1] / "configs" / "simple" / "paper-main.yaml"
    )
    task = next(
        task
        for task in experiment.phase2_tasks()
        if task.model.model_id == "imagenet100-resnet18"
        and task.condition.kind == "clean"
        and task.patch_size == 16
    )
    methods = tuple(
        method.family for method in experiment.methods.for_architecture(task.model.architecture)
    )
    rules = (*PAPER_RULES, *(f"single__{method}" for method in methods))
    manifest = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": experiment.experiment_id,
        "experiment_digest": "0" * 64,
        "phase1_experiment_digest": experiment.phase1_digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "ensemble": task.ensemble.ensemble_id,
        "patch_size": task.patch_size,
        "k": experiment.phase2.k,
        "sample_count": 1,
        "methods": list(methods),
        "metrics": {rule: {metric: 0.5 for metric in QUALITY_METRICS} for rule in rules},
    }

    assert _validate_manifest(experiment, task, manifest) == {
        "artifact_experiment_digest": "0" * 64,
        "reused_from_prior_experiment": True,
    }

    manifest["methods"] = list(methods[:-1])
    with pytest.raises(ArtifactError, match="method roster"):
        _validate_manifest(experiment, task, manifest)


def _quickstart_task():
    experiment = load_experiment(
        Path(__file__).parents[1] / "configs" / "simple" / "quickstart.yaml"
    )
    tasks = list(experiment.phase2_tasks())
    assert len(tasks) == 1
    return experiment, tasks[0]


def _identity_manifest(experiment, task, metrics) -> dict:
    return {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "complete",
        "task_id": task.task_id,
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "ensemble": task.ensemble.ensemble_id,
        "patch_size": task.patch_size,
        "k": experiment.phase2.k,
        "sample_count": 8,
        "metrics": metrics,
    }


def test_metrics_member_order_groups_paper_rules_first() -> None:
    metrics = {
        "single__Beta": _metrics(0.0),
        "borda": _metrics(0.0),
        "schulze": _metrics(0.0),
        "single__Alpha": _metrics(0.0),
        "custom_rule": _metrics(0.0),
        "simpleavg": _metrics(0.0),
    }

    assert _metrics_member_order(metrics) == (
        "simpleavg",
        "borda",
        "schulze",
        "custom_rule",
        "single__Alpha",
        "single__Beta",
    )


def test_check_metrics_manifest_accepts_valid_manifest() -> None:
    experiment, task = _quickstart_task()
    manifest = _identity_manifest(experiment, task, {"borda": _metrics(0.1)})

    assert _check_metrics_manifest(experiment, task, manifest) is None


def test_check_metrics_manifest_rejects_identity_mismatch() -> None:
    experiment, task = _quickstart_task()
    manifest = _identity_manifest(experiment, task, {"borda": _metrics(0.1)})
    manifest["condition"] = "gaussian"

    with pytest.raises(ArtifactError, match="identity mismatch"):
        _check_metrics_manifest(experiment, task, manifest)


def test_check_metrics_manifest_rejects_non_finite_metric() -> None:
    experiment, task = _quickstart_task()
    manifest = _identity_manifest(
        experiment, task, {"borda": {**_metrics(0.1), "F": float("nan")}}
    )

    with pytest.raises(ArtifactError, match="Non-finite"):
        _check_metrics_manifest(experiment, task, manifest)


def test_build_metrics_summary_covers_every_task_and_member(monkeypatch) -> None:
    experiment, task = _quickstart_task()
    metrics = {
        "simpleavg": _metrics(0.1),
        "borda": _metrics(0.2),
        "single__Saliency": _metrics(0.3),
    }
    manifest = _identity_manifest(experiment, task, metrics)
    origin = {"source": "local", "path": "/unused"}

    monkeypatch.setattr(
        "xai_ensemble.simple.summary._load_metrics_manifest",
        lambda _experiment, _task, *, source: (manifest, origin),
    )
    summary = build_metrics_summary(experiment, manifest_source="local")

    assert summary["table"] == "metrics"
    assert summary["row_count"] == 3
    assert [row["member"] for row in summary["rows"]] == [
        "simpleavg",
        "borda",
        "single__Saliency",
    ]
    assert [row["member_kind"] for row in summary["rows"]] == [
        "aggregation",
        "aggregation",
        "single",
    ]
    row = summary["rows"][0]
    assert row["dataset"] == task.dataset.dataset_id
    assert row["condition"] == task.condition.condition_id
    assert row["metrics"]["F"] == pytest.approx(0.6)
    assert summary["source_tasks"][0]["task_digest"] == task.digest

    csv_text = _metrics_csv_text(summary)
    assert csv_text.splitlines()[0].startswith("dataset,model,architecture")
    assert "single__Saliency" in csv_text
    tex_text = _metrics_tex_text(summary)
    assert "single\\_\\_Saliency" in tex_text
    assert tex_text.count("\\\\") == 3
