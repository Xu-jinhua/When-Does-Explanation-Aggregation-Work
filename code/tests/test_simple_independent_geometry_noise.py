from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.simple.noise_prefix.anchored import FidelityAnchoredPrefix
from xai_ensemble.simple.noise_prefix.independent_geometry import (
    REPORT_SCHEMA,
    SELECTOR_SCHEMA,
    build_independent_geometry_report,
    build_independent_geometry_selector,
    load_independent_geometry_selector,
    write_independent_geometry_report,
)
from xai_ensemble.simple.robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
    signed_robustness_values,
)
from xai_ensemble.simple.summary import NOISE_ORDER, PAPER_RULES

METHODS = tuple(f"method-{position:02d}" for position in range(11))
CONDITIONS = {
    "g": "gaussian-0.15",
    "p": "salt-pepper-0.05",
    "s": "speckle-0.15",
    "a": "adversarial-sara-2-255",
}


def _digest(value: dict, key: str) -> dict:
    value[key] = object_sha256(value)
    return value


def _experiment() -> SimpleNamespace:
    cell = SimpleNamespace(
        cell_id="dataset--model",
        dataset=SimpleNamespace(dataset_id="dataset"),
        reference_model=SimpleNamespace(model_id="model"),
        methods=METHODS,
    )
    base_task = SimpleNamespace(digest="b" * 64)
    prefix_task = SimpleNamespace(
        digest="p" * 64,
        cell=cell,
        condition=SimpleNamespace(kind="clean"),
    )
    return SimpleNamespace(
        sweep_id="sweep",
        digest="a" * 64,
        q_values=tuple(range(2, 12)),
        patch_size=16,
        k=20,
        cells=lambda: (cell,),
        base_phase2_task=lambda selected_cell, condition: base_task,
        evaluation_tasks=lambda: (prefix_task,),
    )


def _candidate(q: int, score: float) -> FidelityAnchoredPrefix:
    return FidelityAnchoredPrefix(
        q=q,
        score=score,
        theta=1.0,
        mean_distance=2.0,
        per_sample_score=np.asarray([score], dtype=np.float64),
    )


def _selector(experiment: SimpleNamespace) -> dict:
    geometry_rows = {}
    for geometry, selected_q, center_rule in (
        ("spearman", 2, "borda"),
        ("kendall", 3, "kemeny"),
    ):
        candidates = [
            {
                "q": q,
                "score": 1.0 if q == selected_q else 0.0,
                "theta": 1.0,
                "mean_distance": 2.0,
            }
            for q in experiment.q_values
        ]
        geometry_rows[geometry] = {
            "label": "NOISE-S" if geometry == "spearman" else "NOISE-K",
            "center_rule": center_rule,
            "q": selected_q,
            "score": 1.0,
            "theta": 1.0,
            "mean_distance": 2.0,
            "method_prefix": list(METHODS[:selected_q]),
            "candidates": candidates,
        }
    return _digest(
        {
            "schema": SELECTOR_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "experiment_id": "independent-geometry-noise-v1",
            "scope": "post_hoc_complete_test_set_independent_geometry_boundary_experiment",
            "sweep_id": experiment.sweep_id,
            "sweep_digest": experiment.digest,
            "selection_input_contract": {
                "q_sweep_summary_accepted": False,
                "q_level_aggregate_metrics_read": False,
                "q_level_masked_predictions_read": False,
            },
            "science": {
                "q_values": list(experiment.q_values),
                "geometry_to_center_rule": {"spearman": "borda", "kendall": "kemeny"},
                "shared_q_across_geometries": False,
            },
            "selection_source_digest": "s" * 64,
            "cells": [
                {
                    "cell": "dataset--model",
                    "dataset": "dataset",
                    "model": "model",
                    "q_S": 2,
                    "q_K": 3,
                    "ordered_methods": list(METHODS),
                    "geometries": geometry_rows,
                    "source": {},
                }
            ],
        },
        "selector_digest",
    )


def _nested(value: float) -> dict:
    quality = {
        "F": value,
        "Fbar": 0.8 - value / 10.0,
        "C": value + 0.1,
        "Cbar": 0.7 - value / 10.0,
    }
    perturbed = {}
    robustness = {}
    for position, condition in enumerate(CONDITIONS.values(), start=1):
        delta = position / 1000.0
        current = {
            "F": quality["F"] + delta,
            "Fbar": quality["Fbar"] + delta,
            "C": quality["C"] - delta,
            "Cbar": quality["Cbar"] - delta,
        }
        perturbed[condition] = current
        robustness[condition] = signed_robustness_values(
            quality, current, context=f"fixture/{condition}"
        )
    return {"quality": quality, "perturbed_quality": perturbed, "robustness": robustness}


def _compact(experiment: SimpleNamespace) -> dict:
    rows = []
    for q in experiment.q_values:
        for rule in PAPER_RULES:
            target = 2 if rule == "borda" else 3 if rule == "kemeny" else 4
            value = 0.5 - abs(q - target) / 100.0
            rows.append(
                {
                    "cell": "dataset--model",
                    "dataset": "dataset",
                    "model": "model",
                    "q": q,
                    "rule": rule,
                    "method_prefix": list(METHODS[:q]),
                    "q11_reference": q == 11,
                    **_nested(value),
                }
            )
    return _digest(
        {
            "schema": "simple-noise-prefix-per-q-v1",
            "schema_version": 1,
            "status": "complete",
            "sweep_id": experiment.sweep_id,
            "sweep_digest": experiment.digest,
            "source_summary_digest": "q" * 64,
            "input_catalog_digest": "i" * 64,
            "science": {
                "patch_size": 16,
                "k": 20,
                "q_values": list(experiment.q_values),
                "rules": ["simpleavg", "borda", "rrf", "kemeny", "schulze"],
                "optimum_tie_policy": "smallest_q_within_absolute_1e-12",
                "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
                "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
                "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
                "legacy_absolute_R_used": False,
                "perturbed_quality_reported": True,
            },
            "manifest_content_digests": {"task": "m" * 64},
            "conditions": list(CONDITIONS.values()),
            "counts": {
                "rows": 50,
                "cells": 1,
                "q_values": 10,
                "rules": 5,
                "perturbations": 4,
            },
            "rows": rows,
        },
        "compact_digest",
    )


def _short_noise_row(row: dict) -> dict:
    return {
        "quality": row["quality"],
        "perturbed_quality": {
            noise: row["perturbed_quality"][condition] for noise, condition in CONDITIONS.items()
        },
        "robustness": {
            noise: row["robustness"][condition] for noise, condition in CONDITIONS.items()
        },
    }


def _table1(compact: dict) -> dict:
    q11 = {(row["q"], row["rule"]): row for row in compact["rows"] if row["q"] == 11}
    rows = [
        {"method": "best_individual", **_short_noise_row(_nested(0.45))},
        *[{"method": rule, **_short_noise_row(q11[(11, rule)])} for rule in PAPER_RULES],
    ]
    return _digest(
        {
            "status": "complete",
            "settings": {
                "noise": {
                    key: {"key": key, "condition_id": condition}
                    for key, condition in CONDITIONS.items()
                }
            },
            "cells": [{"dataset": "dataset", "model": "model", "rows": rows}],
        },
        "summary_digest",
    )


def _oracle(compact: dict) -> dict:
    q11 = {(row["q"], row["rule"]): row for row in compact["rows"] if row["q"] == 11}
    cells = []
    for geometry in ("spearman", "kendall"):
        cells.append(
            {
                "cell": "dataset--model",
                "distance_model": geometry,
                "rows": [
                    {
                        "method": rule,
                        "setting": "oracle-noise",
                        **_short_noise_row(q11[(11, rule)]),
                    }
                    for rule in PAPER_RULES
                ],
            }
        )
    return _digest({"status": "complete", "cells": cells}, "summary_digest")


def _dual(experiment: SimpleNamespace, compact: dict) -> dict:
    selected = {(row["q"], row["rule"]): row for row in compact["rows"] if row["q"] == 2}
    return _digest(
        {
            "schema": "simple-dual-geometry-noise-report-v3",
            "status": "complete",
            "sweep_id": experiment.sweep_id,
            "sweep_digest": experiment.digest,
            "cells": [
                {
                    "cell": "dataset--model",
                    "q": 2,
                    "rows": [
                        {
                            "method": rule,
                            "q": 2,
                            **_short_noise_row(selected[(2, rule)]),
                        }
                        for rule in PAPER_RULES
                    ],
                }
            ],
        },
        "result_digest",
    )


def test_selector_builder_cannot_accept_q_level_summary() -> None:
    parameters = inspect.signature(build_independent_geometry_selector).parameters

    assert tuple(parameters) == ("experiment", "base_clean_cache", "prefix_clean_cache")
    assert not any("summary" in name or "measurement" in name for name in parameters)


def test_selector_selects_each_geometry_independently(monkeypatch, tmp_path) -> None:
    experiment = _experiment()
    base = {
        "indices": np.asarray([0]),
        "ordered_methods": METHODS,
        "fidelity": {method: 1.0 - position / 100.0 for position, method in enumerate(METHODS)},
        "task_digest": "b" * 64,
        "manifest_content_digest": "c" * 64,
        "ballots": np.zeros((1, 11, 20), dtype=np.int16),
        "contributions": np.zeros((1, 11), dtype=np.float64),
        "q11_centers": {
            "borda": np.zeros((1, 20), dtype=np.int64),
            "kemeny": np.zeros((1, 20), dtype=np.int64),
        },
    }
    prefix = {
        "indices": np.asarray([0]),
        "manifest_content_digest": "d" * 64,
        "manifest": {
            "ordered_methods": list(METHODS),
            "fidelity": base["fidelity"],
            "task_digest": "p" * 64,
            "q11_reference": {
                "task_digest": "b" * 64,
                "manifest_content_digest": "c" * 64,
            },
        },
        "centers": {
            rule: {q: np.zeros((1, 20), dtype=np.int64) for q in range(2, 11)}
            for rule in ("borda", "kemeny")
        },
    }
    monkeypatch.setattr(
        "xai_ensemble.simple.noise_prefix.independent_geometry._base_selector_inputs",
        lambda *args, **kwargs: base,
    )
    monkeypatch.setattr(
        "xai_ensemble.simple.noise_prefix.independent_geometry._prefix_selector_inputs",
        lambda *args, **kwargs: prefix,
    )
    monkeypatch.setattr(
        "xai_ensemble.simple.noise_prefix.independent_geometry._selector_candidates",
        lambda **kwargs: {
            "spearman": tuple(_candidate(q, 1.0 if q == 2 else 0.0) for q in range(2, 12)),
            "kendall": tuple(_candidate(q, 1.0 if q == 3 else 0.0) for q in range(2, 12)),
        },
    )

    selector = build_independent_geometry_selector(
        experiment,
        base_clean_cache=tmp_path,
        prefix_clean_cache=tmp_path,
    )

    assert selector["cells"][0]["q_S"] == 2
    assert selector["cells"][0]["q_K"] == 3
    assert selector["science"]["shared_q_across_geometries"] is False
    assert selector["selection_input_contract"]["q_level_aggregate_metrics_read"] is False


def test_selector_loader_enforces_rank_only_contract(tmp_path) -> None:
    selector = _selector(_experiment())
    path = tmp_path / "selector.json"
    path.write_text(json.dumps(selector), encoding="utf-8")

    loaded = load_independent_geometry_selector(path)
    assert loaded["cells"][0]["q_S"] == 2
    assert loaded["cells"][0]["q_K"] == 3

    selector["selection_input_contract"]["q_level_aggregate_metrics_read"] = True
    selector["selector_digest"] = object_sha256(
        {key: value for key, value in selector.items() if key != "selector_digest"}
    )
    path.write_text(json.dumps(selector), encoding="utf-8")
    try:
        load_independent_geometry_selector(path)
    except ValueError as error:
        assert "rank-only" in str(error)
    else:
        raise AssertionError("A selector that reads q-level metrics was accepted")


def test_report_applies_each_q_to_all_rules_and_recomputes_signed_r(tmp_path) -> None:
    experiment = _experiment()
    compact = _compact(experiment)
    summary, rows, comparisons = build_independent_geometry_report(
        experiment,
        selector=_selector(experiment),
        compact_q_summary=compact,
        table1_summary=_table1(compact),
        oracle_noise_summary=_oracle(compact),
        dual_geometry_summary=_dual(experiment, compact),
    )

    assert summary["schema"] == REPORT_SCHEMA
    assert summary["selected_q"] == {"dataset--model": {"spearman": 2, "kendall": 3}}
    assert len(rows) == 10
    assert {(row["geometry"], row["q"]) for row in rows} == {
        ("spearman", 2),
        ("kendall", 3),
    }
    assert {row["method"] for row in rows if row["geometry"] == "spearman"} == set(PAPER_RULES)
    assert summary["counts"] == {
        "cells": 1,
        "geometries": 2,
        "rules_per_geometry": 5,
        "rows": 10,
        "endpoints": 200,
        "comparison_rows": 800,
        "selection_validation_rows": 2,
    }
    assert len(comparisons) == 800
    assert summary["selection_validation"]["aggregate"]["exact"] == 2
    first = rows[0]
    assert first["R_F_g"] == first["F_perturbed_g"] - first["F"]

    paths = write_independent_geometry_report(
        summary,
        rows,
        comparisons,
        output_directory=tmp_path,
    )
    assert (tmp_path / "summary.json").is_file()
    assert len((tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()) == 11
    assert paths["selection_validation_csv"] == str(tmp_path / "selection_validation.csv")


def test_report_rejects_signed_robustness_that_contradicts_raw_quality() -> None:
    experiment = _experiment()
    compact = _compact(experiment)
    compact["rows"][0]["robustness"][CONDITIONS["g"]]["F"] += 0.1
    compact["compact_digest"] = object_sha256(
        {key: value for key, value in compact.items() if key != "compact_digest"}
    )

    try:
        build_independent_geometry_report(
            experiment,
            selector=_selector(experiment),
            compact_q_summary=compact,
            table1_summary=_table1(_compact(experiment)),
            oracle_noise_summary=_oracle(_compact(experiment)),
            dual_geometry_summary=_dual(experiment, _compact(experiment)),
        )
    except ValueError as error:
        assert "signed robustness mismatch" in str(error)
    else:
        raise AssertionError("A contradictory signed robustness value was accepted")


def test_noise_keys_remain_in_paper_order() -> None:
    assert tuple(CONDITIONS) == NOISE_ORDER
