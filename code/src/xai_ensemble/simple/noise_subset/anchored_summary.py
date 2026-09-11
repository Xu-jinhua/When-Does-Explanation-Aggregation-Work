"""Post-processing for the random-order anchored NOISE mechanism control."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from ..artifacts import ArtifactError
from .artifacts import completed_evaluation_manifest, completed_selection_manifest
from .config import ANCHORED_CONTROL_MODE, GEOMETRY_RULES, NoiseSubsetExperiment

_TOLERANCE = 1e-12


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _candidate_key(position: int, rule: str) -> str:
    return f"candidate_{position:03d}__{rule.lower()}"


def _oriented_delta(metric: str, candidate: float, reference: float) -> float:
    return (
        candidate - reference
        if DEFAULT_METRIC_DIRECTIONS[metric] == "max"
        else reference - candidate
    )


def _outcome(delta: float) -> str:
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=_TOLERANCE):
        return "tie"
    return "win" if delta > 0.0 else "loss"


def _finite_vector(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"correlation field is not a finite vector: {field}")
    return values


def _correlation(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry: str,
    predictor: str,
    outcome: str,
    method: str,
) -> Mapping[str, Any]:
    x = _finite_vector(rows, predictor)
    y = _finite_vector(rows, outcome)
    if x.size < 3 or np.ptp(x) <= _TOLERANCE or np.ptp(y) <= _TOLERANCE:
        coefficient = None
        p_value = None
    else:
        result = stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
        coefficient = float(result.statistic)
        p_value = float(result.pvalue)
    return {
        "geometry": geometry,
        "analysis": method,
        "predictor": predictor,
        "outcome": outcome,
        "controls": "",
        "n": int(x.size),
        "coefficient": coefficient,
        "p_value": p_value,
        "base_r2": None,
        "full_r2": None,
        "delta_r2": None,
    }


def _residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(values.size, dtype=np.float64), controls])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _r_squared(values: np.ndarray, design: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    total = values - float(np.mean(values))
    denominator = float(total @ total)
    return 0.0 if denominator <= _TOLERANCE else float(1.0 - (residual @ residual) / denominator)


def _controlled_relationship(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry: str,
    outcome: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    anchored = _finite_vector(rows, "anchored_score")
    values = _finite_vector(rows, outcome)
    controls = np.column_stack(
        [
            _finite_vector(rows, "mean_individual_clean_F"),
            _finite_vector(rows, "q"),
        ]
    )
    x_residual = _residual(anchored, controls)
    y_residual = _residual(values, controls)
    if np.ptp(x_residual) <= _TOLERANCE or np.ptp(y_residual) <= _TOLERANCE:
        coefficient = None
        p_value = None
    else:
        result = stats.pearsonr(x_residual, y_residual)
        coefficient = float(result.statistic)
        p_value = float(result.pvalue)
    partial = {
        "geometry": geometry,
        "analysis": "partial_pearson",
        "predictor": "anchored_score",
        "outcome": outcome,
        "controls": "mean_individual_clean_F,q",
        "n": int(values.size),
        "coefficient": coefficient,
        "p_value": p_value,
        "base_r2": None,
        "full_r2": None,
        "delta_r2": None,
    }
    base_design = np.column_stack([np.ones(values.size), controls])
    full_design = np.column_stack([base_design, anchored])
    base_r2 = _r_squared(values, base_design)
    full_r2 = _r_squared(values, full_design)
    incremental = {
        "geometry": geometry,
        "analysis": "incremental_ols",
        "predictor": "anchored_score",
        "outcome": outcome,
        "controls": "mean_individual_clean_F,q",
        "n": int(values.size),
        "coefficient": None,
        "p_value": None,
        "base_r2": base_r2,
        "full_r2": full_r2,
        "delta_r2": full_r2 - base_r2,
    }
    return partial, incremental


def _correlation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry: str,
) -> list[Mapping[str, Any]]:
    result = []
    for outcome in ("actual_F", "gain_vs_q11"):
        for predictor in ("anchored_score", "mean_individual_clean_F"):
            for method in ("pearson", "spearman"):
                result.append(
                    _correlation(
                        rows,
                        geometry=geometry,
                        predictor=predictor,
                        outcome=outcome,
                        method=method,
                    )
                )
        result.extend(_controlled_relationship(rows, geometry=geometry, outcome=outcome))
    return result


def build_anchored_summary(experiment: NoiseSubsetExperiment) -> Mapping[str, Any]:
    if experiment.control_mode != ANCHORED_CONTROL_MODE:
        raise ValueError("anchored summary requires the random-order anchored control")
    selection_by_cell = {}
    selection_digests = {}
    for task in experiment.selection_tasks():
        manifest = completed_selection_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(
                f"anchored random-order selection is incomplete: {task.task_id}"
            )
        selection_by_cell[task.cell.cell_id] = manifest
        selection_digests[task.task_id] = object_sha256(manifest)

    candidate_rows = []
    order_rows = []
    comparison_rows = []
    correlation_rows = []
    evaluation_digests = {}
    for task in experiment.evaluation_tasks():
        manifest = completed_evaluation_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(
                f"anchored random-order evaluation is incomplete: {task.task_id}"
            )
        evaluation_digests[task.task_id] = object_sha256(manifest)
        if task.condition.condition_id != "clean":
            raise ArtifactError("anchored random-order mechanism audit must use clean inputs")
        selection = selection_by_cell[task.cell.cell_id]
        geometry = selection["geometries"][task.geometry]
        candidates = manifest.get("candidate_bank")
        metrics = manifest.get("metrics")
        references = manifest.get("reference_prefix_metrics_by_q")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not isinstance(metrics, Mapping)
            or not isinstance(references, Mapping)
        ):
            raise ArtifactError("anchored evaluation summary fields are malformed")
        center_rule = GEOMETRY_RULES[task.geometry]
        q11_F = float(references["11"][center_rule]["F"])
        rows_by_digest = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ArtifactError("anchored candidate row is malformed")
            position = int(candidate["candidate_position"])
            q = int(candidate["q"])
            key = _candidate_key(position, center_rule)
            actual = metrics.get(key)
            if not isinstance(actual, Mapping):
                raise ArtifactError(f"anchored candidate center metric is missing: {key}")
            actual_F = float(actual["F"])
            same_q_F = float(references[str(q)][center_rule]["F"])
            row = {
                "cell": task.cell.cell_id,
                "dataset": task.cell.dataset.dataset_id,
                "model": task.cell.reference_model.model_id,
                "geometry": task.geometry,
                "center_rule": center_rule,
                "candidate_position": position,
                "candidate_digest": str(candidate["candidate_digest"]),
                "q": q,
                "methods": list(candidate["methods"]),
                "mean_individual_clean_F": float(candidate["mean_individual_clean_F"]),
                "anchored_score": float(candidate["anchored_score"]),
                "theta": candidate["theta"],
                "mean_distance": float(candidate["mean_distance"]),
                "actual_F": actual_F,
                "q11_F": q11_F,
                "gain_vs_q11": actual_F - q11_F,
                "fidelity_prefix_same_q_F": same_q_F,
                "difference_vs_fidelity_prefix_same_q": actual_F - same_q_F,
                "selected_order_count": sum(
                    str(selected["candidate_digest"]) == str(candidate["candidate_digest"])
                    for selected in geometry["selected"]
                ),
            }
            candidate_rows.append(row)
            rows_by_digest[row["candidate_digest"]] = row

        formal = geometry["formal_fidelity_order_reference"]
        formal_q = int(formal["q"])
        formal_F = float(references[str(formal_q)][center_rule]["F"])
        for order in geometry["orders"]:
            selected = rows_by_digest[str(order["selected_candidate_digest"])]
            prefix_rows = [
                rows_by_digest[str(row["candidate_digest"])] for row in order["prefixes"]
            ]
            oracle = max(prefix_rows, key=lambda row: float(row["actual_F"]))
            order_rows.append(
                {
                    "cell": task.cell.cell_id,
                    "dataset": task.cell.dataset.dataset_id,
                    "model": task.cell.reference_model.model_id,
                    "geometry": task.geometry,
                    "center_rule": center_rule,
                    "random_order_index": int(order["random_order_index"]),
                    "order_methods": list(order["order_methods"]),
                    "selected_candidate_digest": selected["candidate_digest"],
                    "selected_q": int(selected["q"]),
                    "selected_anchored_score": float(selected["anchored_score"]),
                    "selected_actual_F": float(selected["actual_F"]),
                    "oracle_q": int(oracle["q"]),
                    "oracle_actual_F": float(oracle["actual_F"]),
                    "selection_regret": float(oracle["actual_F"] - selected["actual_F"]),
                    "selected_q_matches_oracle": int(selected["q"]) == int(oracle["q"]),
                    "formal_NOISE_q": formal_q,
                    "formal_NOISE_actual_F": formal_F,
                    "selected_minus_formal_NOISE_F": float(selected["actual_F"] - formal_F),
                    "q11_F": q11_F,
                    "selected_gain_vs_q11": float(selected["actual_F"] - q11_F),
                }
            )
            candidate_position = int(selected["candidate_position"])
            for rule in experiment.rules:
                candidate_metrics = metrics.get(_candidate_key(candidate_position, rule))
                formal_metrics = references[str(formal_q)].get(rule.lower())
                q11_metrics = references["11"].get(rule.lower())
                if not all(
                    isinstance(value, Mapping)
                    for value in (candidate_metrics, formal_metrics, q11_metrics)
                ):
                    raise ArtifactError("selected anchored candidate rule coverage is incomplete")
                for metric in QUALITY_METRICS:
                    candidate_value = float(candidate_metrics[metric])
                    formal_value = float(formal_metrics[metric])
                    q11_value = float(q11_metrics[metric])
                    formal_delta = _oriented_delta(metric, candidate_value, formal_value)
                    q11_delta = _oriented_delta(metric, candidate_value, q11_value)
                    comparison_rows.append(
                        {
                            "cell": task.cell.cell_id,
                            "geometry": task.geometry,
                            "random_order_index": int(order["random_order_index"]),
                            "selected_q": int(selected["q"]),
                            "rule": rule.lower(),
                            "metric": metric,
                            "candidate_value": candidate_value,
                            "formal_NOISE_value": formal_value,
                            "oriented_delta_vs_formal_NOISE": formal_delta,
                            "outcome_vs_formal_NOISE": _outcome(formal_delta),
                            "q11_NAIVE_value": q11_value,
                            "oriented_delta_vs_q11_NAIVE": q11_delta,
                            "outcome_vs_q11_NAIVE": _outcome(q11_delta),
                        }
                    )
        geometry_candidates = [row for row in candidate_rows if row["geometry"] == task.geometry]
        correlation_rows.extend(_correlation_rows(geometry_candidates, geometry=task.geometry))

    selected_outcomes = {
        reference: {
            outcome: sum(row[f"outcome_vs_{reference}"] == outcome for row in comparison_rows)
            for outcome in ("win", "tie", "loss")
        }
        for reference in ("formal_NOISE", "q11_NAIVE")
    }
    value: dict[str, Any] = {
        "schema": "simple-noise-random-order-anchored-summary-v1",
        "schema_version": 1,
        "status": "complete",
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "science": {
            **dict(experiment.raw_science),
            "formal_control_difference": "method_order_only",
            "candidate_search_budget": "ten_nested_q_prefixes_per_order",
            "candidate_metric": "geometry_center_clean_F",
            "gain_reference": "same_geometry_q11_NAIVE",
            "controlled_relationship": "partial_Pearson_and_incremental_OLS_given_mean_F_and_q",
            "test_set_in_sample": True,
        },
        "selector_digest": experiment.selector_digest,
        "selection_manifest_content_digests": selection_digests,
        "evaluation_manifest_content_digests": evaluation_digests,
        "counts": {
            "candidate_rows": len(candidate_rows),
            "random_order_rows": len(order_rows),
            "selected_comparison_endpoints": len(comparison_rows),
            "correlation_rows": len(correlation_rows),
        },
        "selected_outcomes": selected_outcomes,
        "correlations": correlation_rows,
        "candidate_rows": candidate_rows,
        "order_rows": order_rows,
        "selected_comparisons": comparison_rows,
    }
    value["result_digest"] = object_sha256(value)
    return value


def write_anchored_summary(
    experiment: NoiseSubsetExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    value = build_anchored_summary(experiment)
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else experiment.storage.scratch_root / "summaries"
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", value)

    candidate_columns = (
        "cell",
        "dataset",
        "model",
        "geometry",
        "center_rule",
        "candidate_position",
        "candidate_digest",
        "q",
        "methods",
        "mean_individual_clean_F",
        "anchored_score",
        "theta",
        "mean_distance",
        "actual_F",
        "q11_F",
        "gain_vs_q11",
        "fidelity_prefix_same_q_F",
        "difference_vs_fidelity_prefix_same_q",
        "selected_order_count",
    )
    candidates = [
        {**row, "methods": json.dumps(row["methods"], separators=(",", ":"))}
        for row in value["candidate_rows"]
    ]
    atomic_write_text(output / "candidate_analysis.csv", _csv_text(candidates, candidate_columns))
    order_columns = (
        "cell",
        "dataset",
        "model",
        "geometry",
        "center_rule",
        "random_order_index",
        "order_methods",
        "selected_candidate_digest",
        "selected_q",
        "selected_anchored_score",
        "selected_actual_F",
        "oracle_q",
        "oracle_actual_F",
        "selection_regret",
        "selected_q_matches_oracle",
        "formal_NOISE_q",
        "formal_NOISE_actual_F",
        "selected_minus_formal_NOISE_F",
        "q11_F",
        "selected_gain_vs_q11",
    )
    orders = [
        {**row, "order_methods": json.dumps(row["order_methods"], separators=(",", ":"))}
        for row in value["order_rows"]
    ]
    atomic_write_text(output / "order_analysis.csv", _csv_text(orders, order_columns))
    correlation_columns = (
        "geometry",
        "analysis",
        "predictor",
        "outcome",
        "controls",
        "n",
        "coefficient",
        "p_value",
        "base_r2",
        "full_r2",
        "delta_r2",
    )
    atomic_write_text(
        output / "correlations.csv",
        _csv_text(value["correlations"], correlation_columns),
    )
    comparison_columns = (
        "cell",
        "geometry",
        "random_order_index",
        "selected_q",
        "rule",
        "metric",
        "candidate_value",
        "formal_NOISE_value",
        "oriented_delta_vs_formal_NOISE",
        "outcome_vs_formal_NOISE",
        "q11_NAIVE_value",
        "oriented_delta_vs_q11_NAIVE",
        "outcome_vs_q11_NAIVE",
    )
    atomic_write_text(
        output / "selected_comparisons.csv",
        _csv_text(value["selected_comparisons"], comparison_columns),
    )
    lines = [
        "# Random-Order Anchored Control",
        "",
        "Each fixed random method order receives the same q=2..11 anchored selector as",
        "formal NOISE. Candidate-level clean F is evaluated for the matching Borda or Kemeny",
        "center; selected prefixes additionally receive all five paper aggregation rules.",
        "",
    ]
    for geometry in experiment.geometries:
        row = next(
            item
            for item in value["correlations"]
            if item["geometry"] == geometry
            and item["analysis"] == "partial_pearson"
            and item["outcome"] == "actual_F"
        )
        lines.append(
            f"- {geometry}: partial anchored-score/actual-F correlation "
            f"(controlling mean individual F and q) = {row['coefficient']}."
        )
    lines.extend(
        [
            "",
            f"Selected outcomes: `{json.dumps(value['selected_outcomes'], sort_keys=True)}`.",
            f"Result digest: `{value['result_digest']}`.",
            "",
        ]
    )
    atomic_write_text(output / "README.md", "\n".join(lines))
    return {**value, "output_directory": str(output)}


__all__ = ["build_anchored_summary", "write_anchored_summary"]
