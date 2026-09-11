"""Independent Spearman/Borda and Kendall/Kemeny NOISE prefix selection.

Selection is deliberately separated from evaluation. The selector reads only
individual clean-F marks and rank-derived candidate centers. The report opens
the completed q-sweep only after a digest-bound selector has been frozen.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from ..robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
    signed_robustness_values,
)
from ..summary import NOISE_NAMES, NOISE_ORDER, PAPER_RULES
from .anchored import (
    as_serializable,
    fidelity_anchored_prefix,
    select_fidelity_anchored_prefix,
    topk_set_distances,
)
from .config import NoisePrefixExperiment
from .dual_geometry import (
    PERTURBED_REPORT_COLUMNS,
    REPORT_COLUMNS,
    TRANSFER_FIDELITY_REGRET_TOLERANCE,
    _base_selector_inputs,
    _mapping,
    _prefix_selector_inputs,
    _verified_digest,
)

EXPERIMENT_ID = "independent-geometry-noise-v1"
SCOPE = "post_hoc_complete_test_set_independent_geometry_boundary_experiment"
SELECTOR_SCHEMA = "simple-independent-geometry-noise-selector-v1"
REPORT_SCHEMA = "simple-independent-geometry-noise-report-v1"
GEOMETRY_ORDER = ("spearman", "kendall")
GEOMETRY_RULES = {"spearman": "borda", "kendall": "kemeny"}
GEOMETRY_LABELS = {"spearman": "NOISE-S", "kendall": "NOISE-K"}
COLUMN_DIRECTIONS = {
    **{metric: DEFAULT_METRIC_DIRECTIONS[metric] for metric in QUALITY_METRICS},
    **{column: SIGNED_ROBUSTNESS_DIRECTION for column in REPORT_COLUMNS if column.startswith("R_")},
}


def _finite_metric_values(value: Any, *, context: str) -> dict[str, float]:
    mapping = _mapping(value, context=context)
    if set(mapping) != set(QUALITY_METRICS):
        raise ValueError(f"{context} has invalid metric coverage")
    result = {metric: float(mapping[metric]) for metric in QUALITY_METRICS}
    if any(not math.isfinite(item) for item in result.values()):
        raise ValueError(f"{context} contains a non-finite value")
    return result


def _selector_candidates(
    *,
    ballots: np.ndarray,
    contributions: np.ndarray,
    centers: Mapping[str, Mapping[int, np.ndarray]],
    q_values: Sequence[int],
    k: int,
) -> Mapping[str, tuple[Any, ...]]:
    return {
        geometry: tuple(
            fidelity_anchored_prefix(
                topk_set_distances(ballots, centers[rule][q], k=k),
                contributions,
                q=q,
                n_items=ballots.shape[2],
                k=k,
            )
            for q in q_values
        )
        for geometry, rule in GEOMETRY_RULES.items()
    }


def build_independent_geometry_selector(
    experiment: NoisePrefixExperiment,
    *,
    base_clean_cache: str | Path,
    prefix_clean_cache: str | Path,
) -> Mapping[str, Any]:
    """Select q_S and q_K independently without accepting q-level metrics."""

    base_cache = Path(base_clean_cache).expanduser().resolve()
    prefix_cache = Path(prefix_clean_cache).expanduser().resolve()
    cells = []
    source_rows = []
    for cell in sorted(experiment.cells(), key=lambda value: value.cell_id):
        expected_base_task = experiment.base_phase2_task(cell, "clean")
        expected_prefix_tasks = tuple(
            task
            for task in experiment.evaluation_tasks()
            if task.cell.cell_id == cell.cell_id and task.condition.kind == "clean"
        )
        if len(expected_prefix_tasks) != 1:
            raise ValueError(f"Expected one configured clean prefix task for {cell.cell_id}")
        base = _base_selector_inputs(
            base_cache,
            cell=cell,
            expected_task_digest=expected_base_task.digest,
            patch_size=experiment.patch_size,
            k=experiment.k,
        )
        prefix = _prefix_selector_inputs(
            prefix_cache,
            cell=cell,
            q_values=experiment.q_values,
            expected_task_digest=expected_prefix_tasks[0].digest,
            sweep_id=experiment.sweep_id,
            sweep_digest=experiment.digest,
            patch_size=experiment.patch_size,
            k=experiment.k,
        )
        manifest = _mapping(prefix["manifest"], context="prefix manifest")
        q11 = _mapping(manifest["q11_reference"], context="q11 reference")
        if (
            not np.array_equal(base["indices"], prefix["indices"])
            or manifest.get("ordered_methods") != list(base["ordered_methods"])
            or manifest.get("fidelity") != base["fidelity"]
            or q11.get("task_digest") != base["task_digest"]
            or (
                base["manifest_content_digest"] is not None
                and q11.get("manifest_content_digest") != base["manifest_content_digest"]
            )
        ):
            raise ValueError(f"Selector inputs disagree for {cell.cell_id}")

        maximum_q = max(experiment.q_values)
        centers = {
            rule: {
                **prefix["centers"][rule],
                maximum_q: base["q11_centers"][rule],
            }
            for rule in GEOMETRY_RULES.values()
        }
        candidates = _selector_candidates(
            ballots=base["ballots"],
            contributions=base["contributions"],
            centers=centers,
            q_values=experiment.q_values,
            k=experiment.k,
        )
        selected = {
            geometry: select_fidelity_anchored_prefix(values)
            for geometry, values in candidates.items()
        }
        source = {
            "base_phase2_task_digest": base["task_digest"],
            "base_phase2_manifest_content_digest": (
                base["manifest_content_digest"] or q11["manifest_content_digest"]
            ),
            "prefix_task_digest": manifest["task_digest"],
            "prefix_manifest_content_digest": prefix["manifest_content_digest"],
            "individual_clean_f_digest": object_sha256(base["fidelity"]),
        }
        source["source_digest"] = object_sha256(source)
        source_rows.append({"cell": cell.cell_id, **source})
        geometry_rows = {}
        for geometry in GEOMETRY_ORDER:
            choice = selected[geometry]
            geometry_rows[geometry] = {
                "label": GEOMETRY_LABELS[geometry],
                "center_rule": GEOMETRY_RULES[geometry],
                "q": choice.q,
                "score": choice.score,
                "theta": choice.theta,
                "mean_distance": choice.mean_distance,
                "method_prefix": list(base["ordered_methods"][: choice.q]),
                "candidates": [as_serializable(candidate) for candidate in candidates[geometry]],
            }
        cells.append(
            {
                "cell": cell.cell_id,
                "dataset": cell.dataset.dataset_id,
                "model": cell.reference_model.model_id,
                "q_S": int(selected["spearman"].q),
                "q_K": int(selected["kendall"].q),
                "ordered_methods": list(base["ordered_methods"]),
                "geometries": geometry_rows,
                "source": source,
            }
        )

    payload: dict[str, Any] = {
        "schema": SELECTOR_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "scope": SCOPE,
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "selection_input_contract": {
            "q_sweep_summary_accepted": False,
            "q_level_aggregate_metrics_read": False,
            "q_level_masked_predictions_read": False,
            "base_tensor_roles_read": [
                "indices",
                "labels",
                "unmasked_predictions",
                "individual_ranks",
                "individual_removed_predictions_for_clean_F_marks",
                "q11_borda_and_kemeny_ranks",
            ],
            "prefix_tensor_roles_read": ["indices", "borda_and_kemeny_top_patch_indices"],
        },
        "science": {
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "q_values": list(experiment.q_values),
            "method_order": "descending_individual_clean_F_then_method_id",
            "distance": "johnson_graph_half_symmetric_difference",
            "theta_fit": "exact_fixed_size_subset_mallows_mle",
            "utility_mark": "per_sample_individual_clean_F_contribution",
            "selection": "per_geometry_maximum_anchored_score_smallest_q_tie",
            "geometry_to_center_rule": dict(GEOMETRY_RULES),
            "shared_q_across_geometries": False,
            "shared_q_within_geometry_across_reported_aggregation_rules": True,
        },
        "selection_source_digest": object_sha256(source_rows),
        "cells": cells,
    }
    payload["selector_digest"] = object_sha256(payload)
    return payload


def load_independent_geometry_selector(path: str | Path) -> Mapping[str, Any]:
    value = _mapping(read_json(Path(path).expanduser().resolve()), context="selector")
    if (
        value.get("schema") != SELECTOR_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "complete"
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("scope") != SCOPE
    ):
        raise ValueError("Independent-geometry selector identity is invalid")
    _verified_digest(value, key="selector_digest", context="selector")
    contract = _mapping(value.get("selection_input_contract"), context="selection contract")
    forbidden = (
        contract.get("q_sweep_summary_accepted"),
        contract.get("q_level_aggregate_metrics_read"),
        contract.get("q_level_masked_predictions_read"),
    )
    if any(item is not False for item in forbidden):
        raise ValueError("Selector does not enforce rank-only q selection")
    science = _mapping(value.get("science"), context="selector science")
    q_values = tuple(int(q) for q in science.get("q_values", ()))
    if (
        not q_values
        or len(set(q_values)) != len(q_values)
        or tuple(sorted(q_values)) != q_values
        or science.get("shared_q_across_geometries") is not False
        or science.get("geometry_to_center_rule") != GEOMETRY_RULES
    ):
        raise ValueError("Independent-geometry selector science is invalid")
    cells_value = value.get("cells")
    if not isinstance(cells_value, Sequence) or isinstance(cells_value, (str, bytes)):
        raise TypeError("Selector cells must be a sequence")
    seen_cells = set()
    for cell_value in cells_value:
        cell = _mapping(cell_value, context="selector cell")
        cell_id = str(cell.get("cell", ""))
        if not cell_id or cell_id in seen_cells:
            raise ValueError("Selector cell identities must be non-empty and unique")
        seen_cells.add(cell_id)
        methods = cell.get("ordered_methods")
        if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
            raise TypeError(f"Selector methods must be a sequence for {cell_id}")
        geometries = _mapping(cell.get("geometries"), context=f"selector geometries/{cell_id}")
        if set(geometries) != set(GEOMETRY_ORDER):
            raise ValueError(f"Selector geometry coverage is invalid for {cell_id}")
        for geometry in GEOMETRY_ORDER:
            row = _mapping(geometries[geometry], context=f"selector/{cell_id}/{geometry}")
            q = int(row.get("q", -1))
            candidates = row.get("candidates")
            if (
                q not in q_values
                or row.get("center_rule") != GEOMETRY_RULES[geometry]
                or row.get("label") != GEOMETRY_LABELS[geometry]
                or row.get("method_prefix") != list(methods[:q])
                or not isinstance(candidates, Sequence)
                or isinstance(candidates, (str, bytes))
            ):
                raise ValueError(f"Selector geometry record is invalid for {cell_id}/{geometry}")
            candidate_rows = [_mapping(item, context="selector candidate") for item in candidates]
            if [int(item["q"]) for item in candidate_rows] != list(q_values):
                raise ValueError(
                    f"Selector candidate q coverage is invalid for {cell_id}/{geometry}"
                )
            scores = [float(item["score"]) for item in candidate_rows]
            maximum = max(scores)
            expected_q = min(
                int(item["q"]) for item in candidate_rows if float(item["score"]) == maximum
            )
            if q != expected_q or not math.isclose(float(row["score"]), maximum, abs_tol=0.0):
                raise ValueError(f"Selector maximum is inconsistent for {cell_id}/{geometry}")
        if int(cell.get("q_S", -1)) != int(geometries["spearman"]["q"]) or int(
            cell.get("q_K", -1)
        ) != int(geometries["kendall"]["q"]):
            raise ValueError(f"Selector q aliases are inconsistent for {cell_id}")
    return value


def _condition_map(table1_summary: Mapping[str, Any]) -> Mapping[str, str]:
    settings = _mapping(table1_summary.get("settings"), context="Table 1 settings")
    noise = _mapping(settings.get("noise"), context="Table 1 noise settings")
    if set(noise) != set(NOISE_ORDER):
        raise ValueError("Table 1 noise settings do not cover the paper perturbations")
    result = {}
    for key in NOISE_ORDER:
        row = _mapping(noise[key], context=f"Table 1 noise/{key}")
        if row.get("key") != key:
            raise ValueError(f"Table 1 noise key changed for {key}")
        condition = str(row.get("condition_id", ""))
        if not condition or condition in result.values():
            raise ValueError("Table 1 condition ids must be non-empty and unique")
        result[key] = condition
    return result


def _normalize_nested_row(
    row: Mapping[str, Any],
    *,
    noise_keys: Sequence[str],
    context: str,
) -> Mapping[str, Any]:
    quality = _finite_metric_values(row.get("quality"), context=f"{context}/quality")
    perturbed_value = _mapping(row.get("perturbed_quality"), context=f"{context}/perturbed")
    robustness_value = _mapping(row.get("robustness"), context=f"{context}/robustness")
    if set(perturbed_value) != set(noise_keys) or set(robustness_value) != set(noise_keys):
        raise ValueError(f"{context} has invalid perturbation coverage")
    perturbed = {}
    robustness = {}
    for noise in noise_keys:
        perturbed[noise] = _finite_metric_values(
            perturbed_value[noise], context=f"{context}/perturbed/{noise}"
        )
        recorded = _finite_metric_values(
            robustness_value[noise], context=f"{context}/robustness/{noise}"
        )
        expected = signed_robustness_values(
            quality,
            perturbed[noise],
            context=f"{context}/signed-R/{noise}",
        )
        for metric in QUALITY_METRICS:
            if not math.isclose(recorded[metric], expected[metric], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{context} signed robustness mismatch for {noise}/{metric}")
        robustness[noise] = expected
    return {"quality": quality, "perturbed_quality": perturbed, "robustness": robustness}


def _flat_values(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        **{metric: float(row["quality"][metric]) for metric in QUALITY_METRICS},
        **{
            f"R_{metric}_{noise}": float(row["robustness"][noise][metric])
            for noise in NOISE_ORDER
            for metric in QUALITY_METRICS
        },
        **{
            f"{metric}_perturbed_{noise}": float(row["perturbed_quality"][noise][metric])
            for noise in NOISE_ORDER
            for metric in QUALITY_METRICS
        },
    }


def _compact_index(
    experiment: NoisePrefixExperiment,
    compact: Mapping[str, Any],
    *,
    condition_by_noise: Mapping[str, str],
) -> Mapping[tuple[str, int, str], Mapping[str, Any]]:
    if (
        compact.get("schema") != "simple-noise-prefix-per-q-v1"
        or compact.get("schema_version") != 1
        or compact.get("status") != "complete"
        or compact.get("sweep_id") != experiment.sweep_id
        or compact.get("sweep_digest") != experiment.digest
    ):
        raise ValueError("Compact q-sweep identity is invalid")
    _verified_digest(compact, key="compact_digest", context="compact q-sweep")
    science = _mapping(compact.get("science"), context="compact q-sweep science")
    expected_science = {
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "q_values": list(experiment.q_values),
        "optimum_tie_policy": "smallest_q_within_absolute_1e-12",
        "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
        "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
        "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
        "legacy_absolute_R_used": False,
        "perturbed_quality_reported": True,
    }
    contradictions = {
        key: (science.get(key), expected)
        for key, expected in expected_science.items()
        if science.get(key) != expected
    }
    if contradictions:
        raise ValueError(f"Compact q-sweep science is incompatible: {contradictions}")
    rules = science.get("rules")
    if (
        not isinstance(rules, Sequence)
        or isinstance(rules, (str, bytes))
        or len(rules) != len(PAPER_RULES)
        or set(rules) != set(PAPER_RULES)
    ):
        raise ValueError("Compact q-sweep rule coverage is incompatible")
    expected_conditions = [condition_by_noise[key] for key in NOISE_ORDER]
    if compact.get("conditions") != expected_conditions:
        raise ValueError("Compact q-sweep perturbation order changed")
    rows_value = compact.get("rows")
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        raise TypeError("Compact q-sweep rows must be a sequence")
    expected_cells = {cell.cell_id: cell for cell in experiment.cells()}
    index = {}
    for value in rows_value:
        row = _mapping(value, context="compact q-sweep row")
        key = (str(row.get("cell", "")), int(row.get("q", -1)), str(row.get("rule", "")))
        if key in index:
            raise ValueError(f"Duplicate compact q-sweep row: {key}")
        cell = expected_cells.get(key[0])
        if (
            cell is None
            or key[1] not in experiment.q_values
            or key[2] not in PAPER_RULES
            or row.get("dataset") != cell.dataset.dataset_id
            or row.get("model") != cell.reference_model.model_id
            or row.get("q11_reference") is not (key[1] == max(experiment.q_values))
        ):
            raise ValueError(f"Compact q-sweep row metadata is invalid: {key}")
        methods = row.get("method_prefix")
        if (
            not isinstance(methods, Sequence)
            or isinstance(methods, (str, bytes))
            or len(methods) != key[1]
        ):
            raise ValueError(f"Compact q-sweep method prefix is invalid: {key}")
        normalized = _normalize_nested_row(
            row,
            noise_keys=expected_conditions,
            context=f"compact/{key[0]}/q{key[1]}/{key[2]}",
        )
        index[key] = {**row, **normalized}
    expected_keys = {
        (cell_id, q, rule)
        for cell_id in expected_cells
        for q in experiment.q_values
        for rule in PAPER_RULES
    }
    if set(index) != expected_keys:
        raise ValueError("Compact q-sweep coverage is incomplete")
    counts = _mapping(compact.get("counts"), context="compact q-sweep counts")
    expected_counts = {
        "rows": len(expected_keys),
        "cells": len(expected_cells),
        "q_values": len(experiment.q_values),
        "rules": len(PAPER_RULES),
        "perturbations": len(NOISE_ORDER),
    }
    if counts != expected_counts:
        raise ValueError(f"Compact q-sweep counts are invalid: {counts}")
    return index


def _reference_indexes(
    experiment: NoisePrefixExperiment,
    *,
    table1_summary: Mapping[str, Any],
    oracle_noise_summary: Mapping[str, Any],
    dual_geometry_summary: Mapping[str, Any],
) -> tuple[
    Mapping[tuple[str, str], Mapping[str, Any]],
    Mapping[tuple[str, str, str], Mapping[str, Any]],
    Mapping[tuple[str, str], Mapping[str, Any]],
]:
    _verified_digest(table1_summary, key="summary_digest", context="Table 1 summary")
    _verified_digest(oracle_noise_summary, key="summary_digest", context="Oracle NOISE summary")
    _verified_digest(dual_geometry_summary, key="result_digest", context="Dual-geometry summary")
    expected_cells = {cell.cell_id: cell for cell in experiment.cells()}

    table1 = {}
    for cell_value in table1_summary.get("cells", ()):
        cell = _mapping(cell_value, context="Table 1 cell")
        cell_id = f"{cell.get('dataset')}--{cell.get('model')}"
        if cell_id not in expected_cells:
            continue
        for row_value in cell.get("rows", ()):
            row = _mapping(row_value, context="Table 1 row")
            method = str(row.get("method", ""))
            if method not in {"best_individual", *PAPER_RULES}:
                continue
            key = (cell_id, method)
            if key in table1:
                raise ValueError(f"Duplicate Table 1 row: {key}")
            table1[key] = {
                **row,
                **_normalize_nested_row(row, noise_keys=NOISE_ORDER, context=f"Table1/{key}"),
            }
    expected_table1 = {
        (cell_id, method)
        for cell_id in expected_cells
        for method in ("best_individual", *PAPER_RULES)
    }
    if set(table1) != expected_table1:
        raise ValueError("Table 1 reference coverage is incomplete")

    oracle = {}
    for cell_value in oracle_noise_summary.get("cells", ()):
        cell = _mapping(cell_value, context="Oracle NOISE cell")
        cell_id = str(cell.get("cell", ""))
        geometry = str(cell.get("distance_model", ""))
        if cell_id not in expected_cells or geometry not in GEOMETRY_ORDER:
            continue
        for row_value in cell.get("rows", ()):
            row = _mapping(row_value, context="Oracle NOISE row")
            method = str(row.get("method", ""))
            if row.get("setting") != "oracle-noise" or method not in PAPER_RULES:
                continue
            key = (cell_id, geometry, method)
            if key in oracle:
                raise ValueError(f"Duplicate Oracle NOISE row: {key}")
            oracle[key] = {
                **row,
                **_normalize_nested_row(row, noise_keys=NOISE_ORDER, context=f"oracle/{key}"),
            }
    expected_oracle = {
        (cell_id, geometry, rule)
        for cell_id in expected_cells
        for geometry in GEOMETRY_ORDER
        for rule in PAPER_RULES
    }
    if set(oracle) != expected_oracle:
        raise ValueError("Oracle NOISE reference coverage is incomplete")

    if (
        dual_geometry_summary.get("schema") != "simple-dual-geometry-noise-report-v3"
        or dual_geometry_summary.get("status") != "complete"
        or dual_geometry_summary.get("sweep_id") != experiment.sweep_id
        or dual_geometry_summary.get("sweep_digest") != experiment.digest
    ):
        raise ValueError("Dual-geometry reference identity is invalid")
    dual = {}
    for cell_value in dual_geometry_summary.get("cells", ()):
        cell = _mapping(cell_value, context="Dual-geometry cell")
        cell_id = str(cell.get("cell", ""))
        if cell_id not in expected_cells:
            continue
        for row_value in cell.get("rows", ()):
            row = _mapping(row_value, context="Dual-geometry row")
            method = str(row.get("method", ""))
            if method not in PAPER_RULES:
                continue
            key = (cell_id, method)
            if key in dual:
                raise ValueError(f"Duplicate Dual-geometry row: {key}")
            dual[key] = {
                **row,
                **_normalize_nested_row(row, noise_keys=NOISE_ORDER, context=f"dual/{key}"),
            }
    expected_dual = {(cell_id, rule) for cell_id in expected_cells for rule in PAPER_RULES}
    if set(dual) != expected_dual:
        raise ValueError("Dual-geometry reference coverage is incomplete")
    return table1, oracle, dual


def _compact_as_report_row(
    row: Mapping[str, Any], *, condition_by_noise: Mapping[str, str]
) -> Mapping[str, Any]:
    return {
        "quality": dict(row["quality"]),
        "perturbed_quality": {
            noise: dict(row["perturbed_quality"][condition_by_noise[noise]])
            for noise in NOISE_ORDER
        },
        "robustness": {
            noise: dict(row["robustness"][condition_by_noise[noise]]) for noise in NOISE_ORDER
        },
    }


def _outcome(value: float, reference: float, direction: str) -> str:
    benefit = value - reference if direction == "max" else reference - value
    if benefit > 1e-12:
        return "better"
    if benefit < -1e-12:
        return "worse"
    return "equal"


def _comparison_groups(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, int]]:
    groups = {
        "all": REPORT_COLUMNS,
        "quality": QUALITY_METRICS,
        "robustness": tuple(column for column in REPORT_COLUMNS if column.startswith("R_")),
        **{
            NOISE_NAMES[noise]: tuple(f"R_{metric}_{noise}" for metric in QUALITY_METRICS)
            for noise in NOISE_ORDER
        },
    }
    result = {}
    for name, columns in groups.items():
        allowed = set(columns)
        counts = {"better": 0, "equal": 0, "worse": 0}
        for row in rows:
            if row["column"] in allowed:
                counts[str(row["outcome"])] += 1
        result[name] = counts
    return result


def _selection_validation(
    experiment: NoisePrefixExperiment,
    *,
    selector_cells: Mapping[str, Mapping[str, Any]],
    compact_index: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    endpoints = []
    for cell in sorted(experiment.cells(), key=lambda value: value.cell_id):
        selected_cell = selector_cells[cell.cell_id]
        geometries = _mapping(selected_cell["geometries"], context="selector geometries")
        for geometry in GEOMETRY_ORDER:
            center_rule = GEOMETRY_RULES[geometry]
            selected = _mapping(geometries[geometry], context=f"selector/{geometry}")
            selected_q = int(selected["q"])
            curve = {
                q: float(compact_index[(cell.cell_id, q, center_rule)]["quality"]["F"])
                for q in experiment.q_values
            }
            best_value = max(curve.values())
            tied_q = [q for q in experiment.q_values if abs(curve[q] - best_value) <= 1e-12]
            oracle_q = min(tied_q)
            selected_row = compact_index[(cell.cell_id, selected_q, center_rule)]
            if selected_row["method_prefix"] != selected["method_prefix"]:
                raise ValueError(f"Selected method prefix changed for {cell.cell_id}/{geometry}")
            regret = max(0.0, best_value - curve[selected_q])
            endpoints.append(
                {
                    "cell": cell.cell_id,
                    "dataset": cell.dataset.dataset_id,
                    "model": cell.reference_model.model_id,
                    "geometry": geometry,
                    "center_rule": center_rule,
                    "selected_q": selected_q,
                    "oracle_q": oracle_q,
                    "q_error": selected_q - oracle_q,
                    "absolute_q_error": abs(selected_q - oracle_q),
                    "exact": selected_q == oracle_q,
                    "within_one": abs(selected_q - oracle_q) <= 1,
                    "selected_is_oracle_tied": selected_q in tied_q,
                    "selected_fidelity": curve[selected_q],
                    "oracle_fidelity": best_value,
                    "fidelity_regret": regret,
                    "oracle_tied_q": json.dumps(tied_q, separators=(",", ":")),
                    "tie_policy": "smallest_q_within_absolute_1e-12",
                }
            )
    by_cell = {}
    for cell in sorted({str(row["cell"]) for row in endpoints}):
        rows = [row for row in endpoints if row["cell"] == cell]
        exact = sum(bool(row["exact"]) for row in rows)
        within_one = sum(bool(row["within_one"]) for row in rows)
        max_regret = max(float(row["fidelity_regret"]) for row in rows)
        verdict = (
            "strong_support"
            if exact == len(GEOMETRY_ORDER)
            else (
                "support"
                if within_one == len(GEOMETRY_ORDER)
                and max_regret <= TRANSFER_FIDELITY_REGRET_TOLERANCE
                else "not_supported"
            )
        )
        by_cell[cell] = verdict
    regrets = [float(row["fidelity_regret"]) for row in endpoints]
    overall = (
        "strong_support"
        if all(value == "strong_support" for value in by_cell.values())
        else (
            "support"
            if all(value in {"strong_support", "support"} for value in by_cell.values())
            else "not_supported"
        )
    )
    return {
        "scope": "geometry_specific_center_rule_clean_F",
        "tie_policy": "smallest_q_within_absolute_1e-12",
        "transfer_criterion": {
            "strong_support": "both_geometry_oracle_q_exact",
            "support": "both_geometry_q_within_one_and_max_absolute_clean_F_regret_at_most_0.005",
            "not_supported": "otherwise",
            "fidelity_regret_tolerance": TRANSFER_FIDELITY_REGRET_TOLERANCE,
            "status": "inherited_from_predeclared_dual_geometry_transfer_test",
        },
        "endpoints": endpoints,
        "transfer_verdict": {"overall": overall, "by_cell": by_cell},
        "aggregate": {
            "endpoints": len(endpoints),
            "exact": sum(bool(row["exact"]) for row in endpoints),
            "within_one": sum(bool(row["within_one"]) for row in endpoints),
            "selected_is_oracle_tied": sum(
                bool(row["selected_is_oracle_tied"]) for row in endpoints
            ),
            "mean_fidelity_regret": float(np.mean(regrets)),
            "max_fidelity_regret": max(regrets),
        },
    }


def build_independent_geometry_report(
    experiment: NoisePrefixExperiment,
    *,
    selector: Mapping[str, Any],
    compact_q_summary: Mapping[str, Any],
    table1_summary: Mapping[str, Any],
    oracle_noise_summary: Mapping[str, Any],
    dual_geometry_summary: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
    """Evaluate both independently frozen q values from completed q-sweep rows."""

    _verified_digest(selector, key="selector_digest", context="selector")
    if (
        selector.get("schema") != SELECTOR_SCHEMA
        or selector.get("experiment_id") != EXPERIMENT_ID
        or selector.get("scope") != SCOPE
        or selector.get("sweep_id") != experiment.sweep_id
        or selector.get("sweep_digest") != experiment.digest
    ):
        raise ValueError("Selector and experiment identities disagree")
    selector_cells = {
        str(row["cell"]): _mapping(row, context="selector cell") for row in selector["cells"]
    }
    expected_cells = {cell.cell_id: cell for cell in experiment.cells()}
    if set(selector_cells) != set(expected_cells):
        raise ValueError("Selector cell coverage does not match the experiment")
    condition_by_noise = _condition_map(table1_summary)
    compact = _compact_index(
        experiment,
        compact_q_summary,
        condition_by_noise=condition_by_noise,
    )
    table1, oracle, dual = _reference_indexes(
        experiment,
        table1_summary=table1_summary,
        oracle_noise_summary=oracle_noise_summary,
        dual_geometry_summary=dual_geometry_summary,
    )
    validation = _selection_validation(
        experiment,
        selector_cells=selector_cells,
        compact_index=compact,
    )

    rows = []
    flat_rows = []
    comparison_rows = []
    for cell_id in sorted(expected_cells):
        cell = expected_cells[cell_id]
        selected_cell = selector_cells[cell_id]
        geometries = _mapping(selected_cell["geometries"], context="selector geometries")
        for geometry in GEOMETRY_ORDER:
            selected = _mapping(geometries[geometry], context=f"selector/{cell_id}/{geometry}")
            q = int(selected["q"])
            methods = list(selected["method_prefix"])
            for rule in PAPER_RULES:
                compact_row = compact[(cell_id, q, rule)]
                if compact_row["method_prefix"] != methods:
                    raise ValueError(f"Selected prefix changed for {cell_id}/{geometry}/{rule}")
                nested = _compact_as_report_row(compact_row, condition_by_noise=condition_by_noise)
                row = {
                    "cell": cell_id,
                    "dataset": cell.dataset.dataset_id,
                    "model": cell.reference_model.model_id,
                    "geometry": geometry,
                    "geometry_label": GEOMETRY_LABELS[geometry],
                    "center_rule": GEOMETRY_RULES[geometry],
                    "method": rule,
                    "setting": GEOMETRY_LABELS[geometry],
                    "q": q,
                    "methods": methods,
                    **nested,
                }
                rows.append(row)
                values = _flat_values(row)
                flat_rows.append(
                    {
                        "cell": cell_id,
                        "dataset": cell.dataset.dataset_id,
                        "model": cell.reference_model.model_id,
                        "geometry": geometry,
                        "geometry_label": GEOMETRY_LABELS[geometry],
                        "center_rule": GEOMETRY_RULES[geometry],
                        "method": rule,
                        "setting": GEOMETRY_LABELS[geometry],
                        "q": q,
                        "methods": json.dumps(methods, separators=(",", ":")),
                        **values,
                    }
                )

                q11 = _compact_as_report_row(
                    compact[(cell_id, max(experiment.q_values), rule)],
                    condition_by_noise=condition_by_noise,
                )
                q11_values = _flat_values(q11)
                table1_naive_values = _flat_values(table1[(cell_id, rule)])
                for column in (*REPORT_COLUMNS, *PERTURBED_REPORT_COLUMNS):
                    if not math.isclose(
                        q11_values[column],
                        table1_naive_values[column],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"q=11 NAIVE reference changed for {cell_id}/{rule}/{column}"
                        )
                references = {
                    "clean_selected_best_individual": table1[(cell_id, "best_individual")],
                    "naive_q11": table1[(cell_id, rule)],
                    "oracle_noise_same_geometry": oracle[(cell_id, geometry, rule)],
                    "dual_geometry_shared_q": dual[(cell_id, rule)],
                }
                for reference_name, reference_row in references.items():
                    reference_values = _flat_values(reference_row)
                    for column in REPORT_COLUMNS:
                        if column in QUALITY_METRICS:
                            metric = column
                            condition = "clean"
                            endpoint_group = "quality"
                        else:
                            metric, noise = column.removeprefix("R_").rsplit("_", 1)
                            condition = noise
                            endpoint_group = "robustness"
                        direction = COLUMN_DIRECTIONS[column]
                        comparison_rows.append(
                            {
                                "cell": cell_id,
                                "dataset": cell.dataset.dataset_id,
                                "model": cell.reference_model.model_id,
                                "geometry": geometry,
                                "geometry_label": GEOMETRY_LABELS[geometry],
                                "center_rule": GEOMETRY_RULES[geometry],
                                "rule": rule,
                                "q": q,
                                "column": column,
                                "endpoint_group": endpoint_group,
                                "condition": condition,
                                "metric": metric,
                                "value": values[column],
                                "reference": reference_name,
                                "reference_value": reference_values[column],
                                "direction": direction,
                                "outcome": _outcome(
                                    values[column], reference_values[column], direction
                                ),
                            }
                        )

    by_reference = {}
    by_geometry = {}
    for reference in (
        "clean_selected_best_individual",
        "naive_q11",
        "oracle_noise_same_geometry",
        "dual_geometry_shared_q",
    ):
        items = [row for row in comparison_rows if row["reference"] == reference]
        by_reference[reference] = _comparison_groups(items)
        for geometry in GEOMETRY_ORDER:
            by_geometry.setdefault(geometry, {})[reference] = _comparison_groups(
                [row for row in items if row["geometry"] == geometry]
            )

    summary: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "scope": SCOPE,
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "science": {
            "selection": "fidelity_anchored_topk_mallows_independent_geometry_q",
            "selection_scope": "complete_test_set_post_hoc",
            "q_shared_across_geometries": False,
            "q_shared_within_geometry_across_rules": list(PAPER_RULES),
            "geometry_to_center_rule": dict(GEOMETRY_RULES),
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "additional_model_forwards": 0,
            "q_level_metrics_used_for_selection": False,
            "q_level_metrics_used_for_evaluation_after_selection": True,
            "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
            "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
            "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
            "legacy_absolute_R_used": False,
            "perturbed_quality_reported": True,
        },
        "columns": list(REPORT_COLUMNS),
        "column_directions": dict(COLUMN_DIRECTIONS),
        "perturbed_quality_columns": list(PERTURBED_REPORT_COLUMNS),
        "counts": {
            "cells": len(expected_cells),
            "geometries": len(GEOMETRY_ORDER),
            "rules_per_geometry": len(PAPER_RULES),
            "rows": len(rows),
            "endpoints": len(rows) * len(REPORT_COLUMNS),
            "comparison_rows": len(comparison_rows),
            "selection_validation_rows": len(validation["endpoints"]),
        },
        "selected_q": {
            cell_id: {
                geometry: int(selector_cells[cell_id]["geometries"][geometry]["q"])
                for geometry in GEOMETRY_ORDER
            }
            for cell_id in sorted(expected_cells)
        },
        "selection_validation": validation,
        "cells": [
            {
                "cell": cell_id,
                "dataset": expected_cells[cell_id].dataset.dataset_id,
                "model": expected_cells[cell_id].reference_model.model_id,
                "q_S": int(selector_cells[cell_id]["geometries"]["spearman"]["q"]),
                "q_K": int(selector_cells[cell_id]["geometries"]["kendall"]["q"]),
                "geometries": [
                    {
                        "geometry": geometry,
                        "geometry_label": GEOMETRY_LABELS[geometry],
                        "center_rule": GEOMETRY_RULES[geometry],
                        "q": int(selector_cells[cell_id]["geometries"][geometry]["q"]),
                        "method_prefix": selector_cells[cell_id]["geometries"][geometry][
                            "method_prefix"
                        ],
                        "rows": [
                            row
                            for row in rows
                            if row["cell"] == cell_id and row["geometry"] == geometry
                        ],
                    }
                    for geometry in GEOMETRY_ORDER
                ],
            }
            for cell_id in sorted(expected_cells)
        ],
        "comparisons": by_reference,
        "comparisons_by_geometry": by_geometry,
        "sources": {
            "selector_digest": selector["selector_digest"],
            "selection_source_digest": selector["selection_source_digest"],
            "compact_q_sweep_digest": compact_q_summary["compact_digest"],
            "source_q_sweep_summary_digest": compact_q_summary["source_summary_digest"],
            "prefix_manifest_content_digests": compact_q_summary["manifest_content_digests"],
            "table1_summary_digest": table1_summary["summary_digest"],
            "oracle_noise_summary_digest": oracle_noise_summary["summary_digest"],
            "dual_geometry_result_digest": dual_geometry_summary["result_digest"],
        },
    }
    summary["result_digest"] = object_sha256(summary)
    return summary, flat_rows, comparison_rows


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _readme(summary: Mapping[str, Any]) -> str:
    q_rows = "\n".join(
        f"| {cell} | {values['spearman']} | {values['kendall']} |"
        for cell, values in sorted(summary["selected_q"].items())
    )
    validation = summary["selection_validation"]["aggregate"]
    lines = [
        "# Independent-Geometry NOISE Boundary Experiment",
        "",
        "This complete-test-set post-hoc result keeps the Spearman/Borda and",
        "Kendall/Kemeny NOISE assumptions separate. It does not overwrite the",
        "shared-q Dual-Geometry result or the immutable Oracle NOISE result.",
        "",
        "The selector was frozen before opening q-sweep metrics. NOISE-S maximizes",
        "the Borda-center anchored score and NOISE-K independently maximizes the",
        "Kemeny-center anchored score. Each selected q is then applied to all five",
        "aggregation rules. Existing q-sweep outputs supply the metrics, so this",
        "report requires no additional attribution generation or model forward.",
        "",
        "| Cell | q_S | q_K |",
        "|---|---:|---:|",
        q_rows,
        "",
        "## q Selection Validation",
        "",
        (
            f"Across {validation['endpoints']} geometry-specific clean-F endpoints, "
            f"the selected q is exact for {validation['exact']} and within one for "
            f"{validation['within_one']}."
        ),
        (
            f"Mean Fidelity regret is {validation['mean_fidelity_regret']:.12g}; maximum "
            f"regret is {validation['max_fidelity_regret']:.12g}."
        ),
        "",
        "## Comparison Counts",
        "",
    ]
    for geometry in GEOMETRY_ORDER:
        lines.append(f"### {GEOMETRY_LABELS[geometry]}")
        lines.append("")
        for name, values in summary["comparisons_by_geometry"][geometry].items():
            counts = values["all"]
            lines.append(
                f"- {name}: {counts['better']} better, {counts['equal']} equal, "
                f"{counts['worse']} worse across {sum(counts.values())} endpoints."
            )
        lines.append("")
    lines.extend(
        [
            "## Identity",
            "",
            f"- Selector digest: `{summary['sources']['selector_digest']}`",
            f"- Result digest: `{summary['result_digest']}`",
            f"- Compact q-sweep digest: `{summary['sources']['compact_q_sweep_digest']}`",
            "",
            "`summary.csv` contains both geometry branches and all five rules.",
            "`selection_validation.csv` records selected q versus the corresponding",
            "Borda/Kemeny clean-F oracle. `comparisons.csv` retains every unrounded",
            "endpoint comparison. JSON and CSV retain raw perturbed quality beside",
            "direction-aware signed robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_independent_geometry_selector(
    selector: Mapping[str, Any], *, output_path: str | Path
) -> Path:
    return atomic_write_json(Path(output_path).expanduser().resolve(), selector)


def write_independent_geometry_report(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    output_directory: str | Path,
) -> Mapping[str, str]:
    destination = Path(output_directory).expanduser().resolve()
    summary_columns = (
        "cell",
        "dataset",
        "model",
        "geometry",
        "geometry_label",
        "center_rule",
        "method",
        "setting",
        "q",
        "methods",
        *REPORT_COLUMNS,
        *PERTURBED_REPORT_COLUMNS,
    )
    comparison_columns = (
        "cell",
        "dataset",
        "model",
        "geometry",
        "geometry_label",
        "center_rule",
        "rule",
        "q",
        "column",
        "endpoint_group",
        "condition",
        "metric",
        "value",
        "reference",
        "reference_value",
        "direction",
        "outcome",
    )
    selection_columns = (
        "cell",
        "dataset",
        "model",
        "geometry",
        "center_rule",
        "selected_q",
        "oracle_q",
        "q_error",
        "absolute_q_error",
        "exact",
        "within_one",
        "selected_is_oracle_tied",
        "selected_fidelity",
        "oracle_fidelity",
        "fidelity_regret",
        "oracle_tied_q",
        "tie_policy",
    )
    paths = {
        "summary_json": atomic_write_json(destination / "summary.json", summary),
        "summary_csv": atomic_write_text(
            destination / "summary.csv", _csv_text(rows, summary_columns)
        ),
        "comparisons_csv": atomic_write_text(
            destination / "comparisons.csv", _csv_text(comparison_rows, comparison_columns)
        ),
        "selection_validation_csv": atomic_write_text(
            destination / "selection_validation.csv",
            _csv_text(summary["selection_validation"]["endpoints"], selection_columns),
        ),
        "readme": atomic_write_text(destination / "README.md", _readme(summary)),
    }
    return {name: str(path) for name, path in paths.items()}


__all__ = [
    "EXPERIMENT_ID",
    "GEOMETRY_LABELS",
    "GEOMETRY_ORDER",
    "GEOMETRY_RULES",
    "REPORT_SCHEMA",
    "SCOPE",
    "SELECTOR_SCHEMA",
    "build_independent_geometry_report",
    "build_independent_geometry_selector",
    "load_independent_geometry_selector",
    "write_independent_geometry_report",
    "write_independent_geometry_selector",
]
