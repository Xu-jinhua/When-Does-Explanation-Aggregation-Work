"""Post-processing for selected NOISE rows and complete-matrix coverage."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json


def _load_summary(path: str | Path) -> Mapping[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        target = target / "summary.json"
    value = read_json(target)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a JSON mapping: {target}")
    return value


def write_selected_noise_summary(
    *,
    prefix_config: str | Path,
    selector_path: str | Path,
    q_summary_path: str | Path,
    output_directory: str | Path,
) -> Mapping[str, Any]:
    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
    from xai_ensemble.simple.noise_prefix.independent_geometry import (
        EXPERIMENT_ID,
        GEOMETRY_LABELS,
        GEOMETRY_ORDER,
        GEOMETRY_RULES,
        SCOPE,
        SELECTOR_SCHEMA,
        load_independent_geometry_selector,
    )

    experiment = load_noise_prefix_experiment(prefix_config)
    selector = load_independent_geometry_selector(selector_path)
    if (
        selector.get("schema") != SELECTOR_SCHEMA
        or selector.get("experiment_id") != EXPERIMENT_ID
        or selector.get("scope") != SCOPE
        or selector.get("sweep_id") != experiment.sweep_id
        or selector.get("sweep_digest") != experiment.digest
    ):
        raise ValueError("Selected NOISE selector does not match the prefix sweep")
    q_summary = _load_summary(q_summary_path)
    if q_summary.get("schema") != "simple-noise-prefix-summary-v2":
        raise ValueError("Unexpected q-sweep summary schema")
    compact = q_summary.get("per_q")
    if not isinstance(compact, Mapping):
        # The standard writer stores the compact bank as a sibling file.  A
        # direct summary path is accepted when the caller has embedded it.
        q_summary_target = Path(q_summary_path).expanduser().resolve()
        compact = _load_summary(
            (q_summary_target if q_summary_target.is_dir() else q_summary_target.parent)
            / "per_q.json"
        )
    rows = compact.get("rows")
    if not isinstance(rows, Sequence):
        raise ValueError("q-sweep compact summary has no rows")
    by_key = {
        (str(row["cell"]), int(row["q"]), str(row["rule"])): row
        for row in rows
        if isinstance(row, Mapping)
    }
    selector_cells = {
        str(row["cell"]): row for row in selector.get("cells", ()) if isinstance(row, Mapping)
    }
    expected_cells = {cell.cell_id for cell in experiment.cells()}
    if set(selector_cells) != expected_cells:
        raise ValueError("Selector does not cover every active cell")

    selected_rows = []
    for cell in experiment.cells():
        geometry_rows = selector_cells[cell.cell_id]["geometries"]
        for geometry in GEOMETRY_ORDER:
            selected = geometry_rows[geometry]
            q = int(selected["q"])
            for rule in experiment.rules:
                key = (cell.cell_id, q, rule.lower())
                source = by_key.get(key)
                if source is None:
                    raise ValueError(f"Missing q-sweep row: {key}")
                selected_rows.append(
                    {
                        "cell": cell.cell_id,
                        "dataset": cell.dataset.dataset_id,
                        "model": cell.reference_model.model_id,
                        "geometry": geometry,
                        "geometry_label": GEOMETRY_LABELS[geometry],
                        "center_rule": GEOMETRY_RULES[geometry],
                        "q": q,
                        "method_prefix": list(source["method_prefix"]),
                        "rule": rule,
                        "quality": dict(source["quality"]),
                        "perturbed_quality": dict(source["perturbed_quality"]),
                        "robustness": dict(source["robustness"]),
                    }
                )

    payload: dict[str, Any] = {
        "schema": "simple-full-matrix-selected-noise-v1",
        "schema_version": 1,
        "status": "complete",
        "selection_contract": {
            "selector": "fidelity_anchored_topk_mallows_independent_geometry_q",
            "scope": "complete_test_set_post_hoc",
            "q_level_aggregate_metrics_used_for_selection": False,
            "q_level_metrics_read_after_selector_freeze": True,
            "geometry_to_center_rule": {
                geometry: GEOMETRY_RULES[geometry] for geometry in GEOMETRY_ORDER
            },
        },
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "selector_digest": selector["selector_digest"],
        "q_summary_digest": q_summary.get("summary_digest"),
        "selected_q": {
            cell_id: {
                geometry: int(selector_cells[cell_id]["geometries"][geometry]["q"])
                for geometry in GEOMETRY_ORDER
            }
            for cell_id in sorted(selector_cells)
        },
        "counts": {
            "cells": len(expected_cells),
            "geometries": len(GEOMETRY_ORDER),
            "rules": len(experiment.rules),
            "rows": len(selected_rows),
            "endpoints": len(selected_rows) * 4 * (1 + len(experiment.base.conditions) - 1),
        },
        "rows": selected_rows,
    }
    payload["result_digest"] = object_sha256(payload)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    columns = [
        "cell",
        "dataset",
        "model",
        "geometry",
        "geometry_label",
        "center_rule",
        "q",
        "rule",
        "method_prefix",
        "quality",
        "perturbed_quality",
        "robustness",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in selected_rows:
        writer.writerow(
            {
                **{key: row[key] for key in columns[:8]},
                "method_prefix": json.dumps(row["method_prefix"], separators=(",", ":")),
                "quality": json.dumps(row["quality"], sort_keys=True, separators=(",", ":")),
                "perturbed_quality": json.dumps(
                    row["perturbed_quality"], sort_keys=True, separators=(",", ":")
                ),
                "robustness": json.dumps(row["robustness"], sort_keys=True, separators=(",", ":")),
            }
        )
    paths = {
        "summary": str(atomic_write_json(destination / "summary.json", payload)),
        "csv": str(atomic_write_text(destination / "selected_noise.csv", stream.getvalue())),
    }
    return {"result_digest": payload["result_digest"], "paths": paths, "counts": payload["counts"]}


def write_matrix_coverage(
    *,
    experiment_id: str,
    matrix_digest: str,
    active_manifest_path: str | Path,
    component_summaries: Mapping[str, str | Path],
    output_directory: str | Path,
    execution_scope: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    active = read_json(Path(active_manifest_path))
    payload: dict[str, Any] = {
        "schema": "simple-full-matrix-coverage-v1",
        "schema_version": 1,
        "status": "complete",
        "experiment_id": experiment_id,
        "matrix_digest": matrix_digest,
        "active_manifest_digest": active.get("active_manifest_digest"),
        "active_cells": active.get("counts", {}),
        "component_summaries": {},
    }
    for name, path in sorted(component_summaries.items()):
        value = _load_summary(path)
        payload["component_summaries"][name] = {
            "path": str(Path(path).expanduser().resolve()),
            "digest": value.get("summary_digest", value.get("result_digest", value.get("digest"))),
            "status": value.get("status"),
            "counts": value.get("counts", {}),
        }
    if execution_scope is not None:
        payload["execution_scope"] = dict(execution_scope)
    payload["coverage_digest"] = object_sha256(payload)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return {
        "coverage_digest": payload["coverage_digest"],
        "path": str(atomic_write_json(destination / "coverage.json", payload)),
    }


__all__ = ["write_matrix_coverage", "write_selected_noise_summary"]
