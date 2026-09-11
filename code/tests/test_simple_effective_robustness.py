from __future__ import annotations

import csv
from pathlib import Path

import pytest

from xai_ensemble.simple.effective_robustness import (
    build_effective_robustness_summary,
    build_reference_pool,
    fit_reference_curve,
    oriented_quality,
    write_effective_robustness_outputs,
)


def _quality(value: float) -> dict[str, float]:
    return {"F": value, "Fbar": -value, "C": value, "Cbar": -value}


def _perturbed(value: float) -> dict[str, dict[str, float]]:
    return {noise: _quality(value) for noise in ("g", "p", "s", "a")}


def _reference_pool() -> dict[str, object]:
    singles = []
    for method, value in (("Alpha", 0.0), ("Beta", 1.0), ("Gamma", 2.0)):
        singles.append(
            {
                "method": method,
                "quality": _quality(value),
                "perturbed_quality": _perturbed(value),
            }
        )
    return {
        "demo--model": {
            "cell": "demo--model",
            "dataset": "demo",
            "model": "model",
            "split": "test",
            "sample_count": 3,
            "single_explainers": singles,
        }
    }


def _candidate_summary() -> dict[str, object]:
    return {
        "table": "Table_1",
        "cells": [
            {
                "dataset": "demo",
                "model": "model",
                "split": "test",
                "sample_count": 3,
                "rows": [
                    {
                        "method": "best_individual",
                        "setting": "best-individual",
                        "selected_sources": {
                            "quality": {metric: "Beta" for metric in ("F", "Fbar", "C", "Cbar")}
                        },
                        "quality": _quality(1.0),
                        "perturbed_quality": _perturbed(1.0),
                    },
                    {
                        "method": "simpleavg",
                        "setting": "naive",
                        "quality": _quality(1.5),
                        "perturbed_quality": _perturbed(1.8),
                    },
                    {
                        "method": "borda",
                        "setting": "oracle-noise",
                        "quality": _quality(4.0),
                        "perturbed_quality": _perturbed(4.0),
                    },
                ],
            }
        ],
    }


def _row(summary: dict[str, object], *, setting: str) -> dict[str, object]:
    return next(row for row in summary["cells"] if row["setting"] == setting)  # type: ignore[index,return-value]


def test_effective_robustness_orients_min_metrics_and_uses_leave_one_out() -> None:
    summary = build_effective_robustness_summary(
        _candidate_summary(),
        reference_pool=_reference_pool(),
    )
    best = _row(summary, setting="best-individual")
    simpleavg = _row(summary, setting="naive")
    noise = _row(summary, setting="oracle-noise")

    best_f = best["effective_robustness"]["g"]["F"]  # type: ignore[index]
    best_fbar = best["effective_robustness"]["g"]["Fbar"]  # type: ignore[index]
    assert best_f["value"] == pytest.approx(0.0)
    assert best_fbar["value"] == pytest.approx(0.0)
    assert best_f["excluded_reference_method"] == "Beta"
    assert best_f["reference_pool_policy"] == "leave_one_out_naive_single"

    simple_f = simpleavg["effective_robustness"]["g"]["F"]  # type: ignore[index]
    simple_fbar = simpleavg["effective_robustness"]["g"]["Fbar"]  # type: ignore[index]
    assert simple_f["value"] == pytest.approx(0.3)
    assert simple_fbar["value"] == pytest.approx(0.3)
    assert simple_f["excluded_reference_method"] is None
    assert simple_f["reference_pool_policy"] == "full_naive_single_pool"
    assert simple_f["clean_quality_extrapolated"] is False

    noise_f = noise["effective_robustness"]["g"]["F"]  # type: ignore[index]
    assert noise_f["excluded_reference_method"] is None
    assert noise_f["reference_pool_policy"] == "full_naive_single_pool"
    assert noise_f["clean_quality_extrapolated"] is True
    assert noise_f["value"] == pytest.approx(0.0)
    assert oriented_quality("Fbar", -1.5) == pytest.approx(1.5)


def test_reference_curve_clamps_an_inverted_relationship_to_a_constant() -> None:
    curve = fit_reference_curve(
        cell="demo--model",
        metric="F",
        condition="g",
        points=(("Alpha", 0.0, 3.0), ("Beta", 1.0, 2.0), ("Gamma", 2.0, 1.0)),
        excluded_method=None,
    )

    assert curve.unconstrained_slope == pytest.approx(-1.0)
    assert curve.slope == pytest.approx(0.0)
    assert curve.intercept == pytest.approx(2.0)
    assert curve.slope_was_constrained is True


def test_reference_pool_contains_only_original_naive_single_explainers() -> None:
    source_summary = {
        "table": "Table_1",
        "cells": [
            {
                "dataset": "demo",
                "model": "model",
                "split": "test",
                "sample_count": 3,
                "source_tasks": {
                    condition: {"task_id": f"task-{condition}"}
                    for condition in ("clean", "g", "p", "s", "a")
                },
            }
        ],
    }

    def loader(
        _cell: dict[str, object], condition: str, source: dict[str, object]
    ) -> dict[str, object]:
        values = _quality(0.0 if condition == "clean" else 1.0)
        return {
            "dataset": "demo",
            "model": "model",
            "sample_count": 3,
            "methods": ["Alpha", "Beta", "Gamma"],
            "task_id": source["task_id"],
            "metrics": {
                "single__Alpha": values,
                "single__Beta": values,
                "single__Gamma": values,
                "oracle-noise": _quality(100.0),
            },
        }

    pool = build_reference_pool(source_summary, manifest_loader=loader)

    assert [item["method"] for item in pool["demo--model"]["single_explainers"]] == [
        "Alpha",
        "Beta",
        "Gamma",
    ]
    assert all(
        item["method"] != "oracle-noise" for item in pool["demo--model"]["single_explainers"]
    )


def test_effective_robustness_outputs_are_table_and_audit_ready(tmp_path: Path) -> None:
    summary = build_effective_robustness_summary(
        _candidate_summary(),
        reference_pool=_reference_pool(),
    )
    result = write_effective_robustness_outputs(summary, output_directory=tmp_path)

    assert result["candidate_rows"] == 3
    assert result["endpoints"] == 48
    for key in (
        "summary_json",
        "er_table_csv",
        "er_detail_csv",
        "reference_curves_csv",
        "er_table_rows_tex",
        "readme",
    ):
        assert Path(result[key]).is_file()
    with Path(result["er_table_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert set(("ER_F_g", "ER_Fbar_g", "ER_C_a", "ER_Cbar_a")).issubset(rows[0])


def test_independent_geometry_rows_inherit_verified_reference_sample_count() -> None:
    simpleavg = _candidate_summary()["cells"][0]["rows"][1]
    summary = build_effective_robustness_summary(
        {
            "schema": "simple-independent-geometry-noise-report-v1",
            "cells": [
                {
                    "dataset": "demo",
                    "model": "model",
                    "geometries": [
                        {
                            "geometry": "spearman",
                            "geometry_label": "NOISE-S",
                            "q": 2,
                            "rows": [simpleavg],
                        }
                    ],
                }
            ],
        },
        reference_pool=_reference_pool(),
    )

    row = summary["cells"][0]
    assert row["sample_count"] == 3
    assert row["geometry"] == "spearman"
    assert row["q"] == 2
