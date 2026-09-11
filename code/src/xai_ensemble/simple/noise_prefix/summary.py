"""Deterministic summaries relating NOISE fit to the best Fidelity prefix."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from ..artifacts import ArtifactError
from ..robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
    signed_robustness_values,
)
from .artifacts import completed_evaluation_manifest
from .config import RULE_IDS, NoisePrefixExperiment
from .inputs import load_input_catalog


def _rule_key(q: int, rule: str) -> str:
    return f"q{q:02d}__{rule}"


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return output.getvalue()


def _validate_manifest(
    experiment: NoisePrefixExperiment,
    task: Any,
    manifest: Mapping[str, Any],
    *,
    catalog_digest: str,
) -> None:
    expected = {
        "status": "complete",
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell.cell_id,
        "condition": task.condition.condition_id,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "input_catalog_digest": catalog_digest,
        "q11_policy": "exact_immutable_naive_p16_reference",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Prefix evaluation identity mismatch: {mismatches}")
    expected_rules = {_rule_key(q, rule) for q in experiment.q_values for rule in RULE_IDS}
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != expected_rules:
        raise ArtifactError(f"Prefix metric coverage is incomplete: {task.task_id}")
    robustness = manifest.get("robustness")
    if task.condition.kind == "clean":
        if robustness is not None:
            raise ArtifactError("Clean prefix manifest unexpectedly contains robustness")
    elif not isinstance(robustness, Mapping) or set(robustness) != expected_rules:
        raise ArtifactError(f"Prefix robustness coverage is incomplete: {task.task_id}")


def _measurement_rows(
    experiment: NoisePrefixExperiment,
    manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    clean_by_cell = {
        str(manifest["cell"]): manifest
        for manifest in manifests
        if str(manifest["condition"]) == "clean"
    }
    expected_cells = {str(manifest["cell"]) for manifest in manifests}
    if set(clean_by_cell) != expected_cells:
        raise ArtifactError("Prefix measurements require exactly one clean manifest per cell")
    for manifest in manifests:
        condition = str(manifest["condition"])
        clean = clean_by_cell[str(manifest["cell"])]
        if int(manifest["sample_count"]) != int(clean["sample_count"]) or set(
            manifest["metrics"]
        ) != set(clean["metrics"]):
            raise ArtifactError(f"Prefix {manifest['cell']}/{condition} is not aligned with clean")
        for q in experiment.q_values:
            methods = manifest["method_prefixes"][str(q)]
            for rule in RULE_IDS:
                key = _rule_key(q, rule)
                signed = None
                if condition != "clean":
                    robustness = manifest.get("robustness")
                    if not isinstance(robustness, Mapping):
                        raise ArtifactError(
                            f"Prefix {manifest['cell']}/{condition} lacks robustness"
                        )
                    record = robustness[key]
                    if not isinstance(record, Mapping):
                        raise ArtifactError(
                            f"Prefix {manifest['cell']}/{condition}/{key} robustness is invalid"
                        )
                    signed = signed_robustness_values(
                        clean["metrics"][key],
                        manifest["metrics"][key],
                        legacy_absolute=record.get("absolute"),
                        context=f"Prefix {manifest['cell']}/{condition}/{key}",
                    )
                for metric in QUALITY_METRICS:
                    rows.append(
                        {
                            "cell": manifest["cell"],
                            "dataset": manifest["dataset"],
                            "model": manifest["model"],
                            "q": q,
                            "rule": rule,
                            "condition": condition,
                            "value_kind": (
                                "quality" if condition == "clean" else "conditioned_quality"
                            ),
                            "metric": metric,
                            "value": float(manifest["metrics"][key][metric]),
                            "method_prefix": json.dumps(methods, separators=(",", ":")),
                            "q11_reference": q == 11,
                        }
                    )
                    if condition != "clean":
                        rows.append(
                            {
                                "cell": manifest["cell"],
                                "dataset": manifest["dataset"],
                                "model": manifest["model"],
                                "q": q,
                                "rule": rule,
                                "condition": condition,
                                "value_kind": "robustness",
                                "metric": metric,
                                "value": signed[metric],
                                "method_prefix": json.dumps(methods, separators=(",", ":")),
                                "q11_reference": q == 11,
                            }
                        )
    return sorted(
        rows,
        key=lambda row: (
            row["cell"],
            row["value_kind"],
            row["condition"],
            row["metric"],
            row["rule"],
            row["q"],
        ),
    )


def _noise_fit_rows(
    experiment: NoisePrefixExperiment,
    catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if experiment.selection_input_mode != "legacy_assumptions":
        return []
    rows = []
    for cell in experiment.cells():
        selections = catalog["noise_selections"][cell.cell_id]
        for distance_model in ("spearman", "kendall"):
            record = selections[distance_model]
            selection = record["selection"]
            selected_q = int(selection["selected_size"])
            forced_fallback = bool(selection.get("forced_fallback", False))
            for evaluation in selection["evaluations"]:
                q = int(evaluation["size"])
                gof = evaluation.get("gof")
                if not isinstance(gof, Mapping):
                    raise ArtifactError(f"NOISE q={q} has no goodness-of-fit record")
                parameters = gof.get("parameters")
                parameters = parameters if isinstance(parameters, Mapping) else {}
                statistic = gof.get("statistic", gof.get("ks_statistic"))
                p_value = gof.get("p_value")
                rows.append(
                    {
                        "cell": cell.cell_id,
                        "dataset": cell.dataset.dataset_id,
                        "model": cell.reference_model.model_id,
                        "distance_model": distance_model,
                        "aggregation": record["aggregation"],
                        "q": q,
                        "mean_distance": float(evaluation["mean_distance"]),
                        "n_distances": int(evaluation["n_distances"]),
                        "fit_model": gof.get("model_name"),
                        "fit_statistic": (None if statistic is None else float(statistic)),
                        "fit_p_value": None if p_value is None else float(p_value),
                        "fit_theta": (
                            None if parameters.get("theta") is None else float(parameters["theta"])
                        ),
                        "selected_q": selected_q,
                        "selected_by_noise": q == selected_q,
                        "forced_fallback": forced_fallback,
                        "methods": json.dumps(evaluation["methods"], separators=(",", ":")),
                        "gof_json": json.dumps(gof, sort_keys=True, separators=(",", ":")),
                        "selection_task_id": record["task_id"],
                        "selection_task_digest": record["task_digest"],
                    }
                )
    return sorted(rows, key=lambda row: (row["cell"], row["distance_model"], row["q"]))


def _objective_direction(row: Mapping[str, Any]) -> str:
    if row["value_kind"] == "robustness":
        return SIGNED_ROBUSTNESS_DIRECTION
    return str(DEFAULT_METRIC_DIRECTIONS[str(row["metric"])])


def _optimum_rows(measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in measurements:
        if row["value_kind"] not in {"quality", "robustness"}:
            continue
        key = tuple(
            row[name]
            for name in (
                "cell",
                "dataset",
                "model",
                "rule",
                "value_kind",
                "condition",
                "metric",
            )
        )
        grouped[key].append(row)
    result = []
    for key, candidates in sorted(grouped.items(), key=lambda item: str(item[0])):
        direction = _objective_direction(candidates[0])
        best_value = (
            max(float(row["value"]) for row in candidates)
            if direction == "max"
            else min(float(row["value"]) for row in candidates)
        )
        tied = sorted(
            int(row["q"])
            for row in candidates
            if math.isclose(float(row["value"]), best_value, rel_tol=0.0, abs_tol=1e-12)
        )
        chosen_q = tied[0]
        selected = next(row for row in candidates if int(row["q"]) == chosen_q)
        result.append(
            {
                "cell": key[0],
                "dataset": key[1],
                "model": key[2],
                "rule": key[3],
                "value_kind": key[4],
                "condition": key[5],
                "metric": key[6],
                "direction": direction,
                "best_q": chosen_q,
                "best_value": best_value,
                "tied_q": json.dumps(tied, separators=(",", ":")),
                "tie_policy": "smallest_q_within_absolute_1e-12",
                "method_prefix": selected["method_prefix"],
            }
        )
    return result


def _relationship_rows(
    measurements: Sequence[Mapping[str, Any]],
    fits: Sequence[Mapping[str, Any]],
    optima: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fit_index = {(row["cell"], int(row["q"]), row["distance_model"]): row for row in fits}
    optimum_index = {
        (
            row["cell"],
            row["rule"],
            row["value_kind"],
            row["condition"],
            row["metric"],
        ): int(row["best_q"])
        for row in optima
    }
    rows = []
    for measurement in measurements:
        if measurement["value_kind"] not in {"quality", "robustness"}:
            continue
        direction = _objective_direction(measurement)
        value = float(measurement["value"])
        optimum_key = (
            measurement["cell"],
            measurement["rule"],
            measurement["value_kind"],
            measurement["condition"],
            measurement["metric"],
        )
        for distance_model in ("spearman", "kendall"):
            fit = fit_index[(measurement["cell"], int(measurement["q"]), distance_model)]
            statistic = fit["fit_statistic"]
            rows.append(
                {
                    **dict(measurement),
                    "distance_model": distance_model,
                    "fit_model": fit["fit_model"],
                    "fit_statistic": statistic,
                    "fit_score": None if statistic is None else -float(statistic),
                    "fit_p_value": fit["fit_p_value"],
                    "mean_distance": fit["mean_distance"],
                    "noise_selected_q": fit["selected_q"],
                    "selected_by_noise": fit["selected_by_noise"],
                    "noise_forced_fallback": fit["forced_fallback"],
                    "objective_direction": direction,
                    "objective_value": value if direction == "max" else -value,
                    "best_q": optimum_index[optimum_key],
                    "is_optimal_q": int(measurement["q"]) == optimum_index[optimum_key],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["cell"],
            row["distance_model"],
            row["value_kind"],
            row["condition"],
            row["metric"],
            row["rule"],
            row["q"],
        ),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _relationship_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["fit_score"] is None:
            continue
        key = tuple(
            row[name]
            for name in (
                "cell",
                "distance_model",
                "rule",
                "value_kind",
                "condition",
                "metric",
            )
        )
        grouped[key].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = sorted(values, key=lambda row: int(row["q"]))
        fit = [float(row["fit_score"]) for row in values]
        objective = [float(row["objective_value"]) for row in values]
        fit_ranks = _rankdata(np.asarray(fit))
        objective_ranks = _rankdata(np.asarray(objective))
        result.append(
            {
                "cell": key[0],
                "distance_model": key[1],
                "rule": key[2],
                "value_kind": key[3],
                "condition": key[4],
                "metric": key[5],
                "n_q": len(values),
                "pearson_fit_objective": _correlation(fit, objective),
                "spearman_fit_objective": _correlation(fit_ranks, objective_ranks),
            }
        )
    return result


def _compact_per_q(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pivot the lossless long measurements into one row per cell/q/rule."""

    measurements = summary["measurements"]
    if not isinstance(measurements, Sequence):
        raise ArtifactError("Prefix summary measurements are invalid")
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for measurement in measurements:
        key = (
            str(measurement["cell"]),
            int(measurement["q"]),
            str(measurement["rule"]),
        )
        methods = json.loads(str(measurement["method_prefix"]))
        metadata = {
            "cell": key[0],
            "dataset": str(measurement["dataset"]),
            "model": str(measurement["model"]),
            "q": key[1],
            "rule": key[2],
            "method_prefix": methods,
            "q11_reference": bool(measurement["q11_reference"]),
        }
        row = grouped.setdefault(
            key,
            {
                **metadata,
                "quality": {},
                "perturbed_quality": {},
                "robustness": {},
            },
        )
        if any(row[name] != value for name, value in metadata.items()):
            raise ArtifactError(f"Prefix compact metadata changed within {key}")
        metric = str(measurement["metric"])
        if metric not in QUALITY_METRICS:
            raise ArtifactError(f"Prefix compact row has unknown metric {metric!r}")
        condition = str(measurement["condition"])
        value_kind = str(measurement["value_kind"])
        if value_kind == "quality" and condition == "clean":
            target = row["quality"]
        elif value_kind == "conditioned_quality" and condition != "clean":
            target = row["perturbed_quality"].setdefault(condition, {})
        elif value_kind == "robustness" and condition != "clean":
            target = row["robustness"].setdefault(condition, {})
        else:
            raise ArtifactError(
                f"Prefix compact row has invalid kind/condition {value_kind}/{condition}"
            )
        if metric in target:
            raise ArtifactError(f"Prefix compact row duplicates {key}/{condition}/{metric}")
        target[metric] = float(measurement["value"])

    noisy_conditions = tuple(
        str(condition["condition_id"])
        for condition in summary["conditions"]
        if condition["kind"] != "clean"
    )
    expected_metrics = set(QUALITY_METRICS)
    rows = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], RULE_IDS.index(item[2]))):
        row = grouped[key]
        if set(row["quality"]) != expected_metrics:
            raise ArtifactError(f"Prefix compact clean metric coverage is incomplete: {key}")
        for field in ("perturbed_quality", "robustness"):
            if set(row[field]) != set(noisy_conditions) or any(
                set(row[field][condition]) != expected_metrics for condition in noisy_conditions
            ):
                raise ArtifactError(f"Prefix compact {field} coverage is incomplete: {key}")
        rows.append(row)

    value: dict[str, Any] = {
        "schema": "simple-noise-prefix-per-q-v1",
        "schema_version": 1,
        "status": "complete",
        "sweep_id": summary["sweep_id"],
        "sweep_digest": summary["sweep_digest"],
        "source_summary_digest": summary["summary_digest"],
        "input_catalog_digest": summary["input_catalog_digest"],
        "science": summary["science"],
        "manifest_content_digests": summary["manifest_content_digests"],
        "conditions": list(noisy_conditions),
        "counts": {
            "rows": len(rows),
            "cells": len({row["cell"] for row in rows}),
            "q_values": len({row["q"] for row in rows}),
            "rules": len({row["rule"] for row in rows}),
            "perturbations": len(noisy_conditions),
        },
        "rows": rows,
    }
    value["compact_digest"] = object_sha256(value)
    return value


def _compact_per_q_csv(compact: Mapping[str, Any]) -> str:
    conditions = tuple(str(value) for value in compact["conditions"])
    columns = [
        "cell",
        "dataset",
        "model",
        "q",
        "rule",
        "q11_reference",
        "method_prefix",
        *(f"{metric}_clean" for metric in QUALITY_METRICS),
        *(
            f"{metric}_perturbed_{condition}"
            for condition in conditions
            for metric in QUALITY_METRICS
        ),
        *(f"R_{metric}_{condition}" for condition in conditions for metric in QUALITY_METRICS),
    ]
    rows = []
    for source in compact["rows"]:
        row = {name: source[name] for name in ("cell", "dataset", "model", "q", "rule")}
        row["q11_reference"] = source["q11_reference"]
        row["method_prefix"] = json.dumps(source["method_prefix"], separators=(",", ":"))
        row.update({f"{metric}_clean": source["quality"][metric] for metric in QUALITY_METRICS})
        row.update(
            {
                f"{metric}_perturbed_{condition}": source["perturbed_quality"][condition][metric]
                for condition in conditions
                for metric in QUALITY_METRICS
            }
        )
        row.update(
            {
                f"R_{metric}_{condition}": source["robustness"][condition][metric]
                for condition in conditions
                for metric in QUALITY_METRICS
            }
        )
        rows.append(row)
    return _csv_text(rows, columns)


def build_summary(experiment: NoisePrefixExperiment) -> Mapping[str, Any]:
    catalog = load_input_catalog(experiment)
    manifests = []
    manifest_digests = {}
    for task in experiment.evaluation_tasks():
        manifest = completed_evaluation_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(f"Prefix evaluation is incomplete: {task.task_id}")
        _validate_manifest(
            experiment,
            task,
            manifest,
            catalog_digest=str(catalog["catalog_digest"]),
        )
        manifests.append(manifest)
        manifest_digests[task.task_id] = object_sha256(manifest)
    measurements = _measurement_rows(experiment, manifests)
    fits = _noise_fit_rows(experiment, catalog)
    optima = _optimum_rows(measurements)
    relationship = (
        _relationship_rows(measurements, fits, optima)
        if experiment.selection_input_mode == "legacy_assumptions"
        else []
    )
    analysis = _relationship_analysis(relationship)
    value: dict[str, Any] = {
        "schema": "simple-noise-prefix-summary-v2",
        "schema_version": 2,
        "status": "complete",
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "input_catalog_digest": catalog["catalog_digest"],
        "conditions": [
            {"condition_id": condition.condition_id, "kind": condition.kind}
            for condition in experiment.base.conditions
        ],
        "science": {
            "question": "relation_between_noise_fit_and_best_fidelity_prefix_size",
            "scope": "complete_test_set_in_sample_oracle_diagnostic",
            "fidelity_order": "descending_F_then_method_id",
            "q_values": list(experiment.q_values),
            "rules": list(RULE_IDS),
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "fill": "dataset_mean",
            "q11": "exact_immutable_naive_p16_reference",
            "selection_input_mode": experiment.selection_input_mode,
            "optimum_tie_policy": "smallest_q_within_absolute_1e-12",
            "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
            "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
            "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
            "legacy_absolute_R_used": False,
            "perturbed_quality_reported": True,
        },
        "manifest_content_digests": manifest_digests,
        "counts": {
            "manifests": len(manifests),
            "measurements": len(measurements),
            "noise_fits": len(fits),
            "relationship_rows": len(relationship),
            "optima": len(optima),
        },
        "optimal_q_frequency": {
            str(q): sum(int(row["best_q"]) == q for row in optima) for q in experiment.q_values
        },
        "measurements": measurements,
        "noise_fit": fits,
        "relationship_analysis": analysis,
        "optima": optima,
    }
    value["summary_digest"] = object_sha256(value)
    return value


def write_summary(
    experiment: NoisePrefixExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    summary = build_summary(experiment)
    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else experiment.storage.scratch_root / "summaries"
    )
    measurements = summary["measurements"]
    fits = summary["noise_fit"]
    optima = summary["optima"]
    relationship = (
        _relationship_rows(measurements, fits, optima)
        if experiment.selection_input_mode == "legacy_assumptions"
        else []
    )
    compact = _compact_per_q(summary)
    paths = {
        "summary_json": atomic_write_json(destination / "summary.json", summary),
        "per_q_json": atomic_write_json(destination / "per_q.json", compact),
        "per_q_csv": atomic_write_text(
            destination / "per_q.csv",
            _compact_per_q_csv(compact),
        ),
        "measurements_csv": atomic_write_text(
            destination / "measurements.csv",
            _csv_text(
                measurements,
                (
                    "cell",
                    "dataset",
                    "model",
                    "q",
                    "rule",
                    "condition",
                    "value_kind",
                    "metric",
                    "value",
                    "q11_reference",
                    "method_prefix",
                ),
            ),
        ),
        "noise_fit_csv": atomic_write_text(
            destination / "noise_fit.csv",
            _csv_text(
                fits,
                (
                    "cell",
                    "dataset",
                    "model",
                    "distance_model",
                    "aggregation",
                    "q",
                    "mean_distance",
                    "n_distances",
                    "fit_model",
                    "fit_statistic",
                    "fit_p_value",
                    "fit_theta",
                    "selected_q",
                    "selected_by_noise",
                    "forced_fallback",
                    "methods",
                    "selection_task_id",
                    "selection_task_digest",
                    "gof_json",
                ),
            ),
        ),
        "relationship_csv": atomic_write_text(
            destination / "relationship.csv",
            _csv_text(
                relationship,
                (
                    "cell",
                    "dataset",
                    "model",
                    "distance_model",
                    "q",
                    "rule",
                    "condition",
                    "value_kind",
                    "metric",
                    "value",
                    "objective_direction",
                    "objective_value",
                    "fit_model",
                    "fit_statistic",
                    "fit_score",
                    "fit_p_value",
                    "mean_distance",
                    "noise_selected_q",
                    "selected_by_noise",
                    "noise_forced_fallback",
                    "best_q",
                    "is_optimal_q",
                    "q11_reference",
                    "method_prefix",
                ),
            ),
        ),
        "optima_csv": atomic_write_text(
            destination / "optima.csv",
            _csv_text(
                optima,
                (
                    "cell",
                    "dataset",
                    "model",
                    "rule",
                    "value_kind",
                    "condition",
                    "metric",
                    "direction",
                    "best_q",
                    "best_value",
                    "tied_q",
                    "tie_policy",
                    "method_prefix",
                ),
            ),
        ),
    }
    return {
        "status": "complete",
        "sweep_id": experiment.sweep_id,
        "summary_digest": summary["summary_digest"],
        **{key: str(path) for key, path in paths.items()},
    }


__all__ = ["build_summary", "write_summary"]
