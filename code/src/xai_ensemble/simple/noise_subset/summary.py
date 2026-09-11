"""Deterministic comparison of random NOISE subsets with Fidelity prefixes."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from ..artifacts import ArtifactError
from .artifacts import completed_evaluation_manifest, completed_selection_manifest
from .config import ANCHORED_CONTROL_MODE, NoiseSubsetExperiment

_TIE_TOLERANCE = 1e-12


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _oriented_delta(metric: str, candidate: float, reference: float) -> float:
    if DEFAULT_METRIC_DIRECTIONS[metric] == "max":
        return candidate - reference
    return reference - candidate


def _outcome(delta: float) -> str:
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=_TIE_TOLERANCE):
        return "tie"
    return "win" if delta > 0.0 else "loss"


def build_summary(experiment: NoiseSubsetExperiment) -> Mapping[str, Any]:
    selection_digests = {}
    selections = {}
    for task in experiment.selection_tasks():
        manifest = completed_selection_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(f"random NOISE selection is incomplete: {task.task_id}")
        selections[task.cell.cell_id] = manifest
        selection_digests[task.task_id] = object_sha256(manifest)

    rows = []
    manifest_digests = {}
    for task in experiment.evaluation_tasks():
        manifest = completed_evaluation_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(f"random NOISE evaluation is incomplete: {task.task_id}")
        manifest_digests[task.task_id] = object_sha256(manifest)
        selected = manifest.get("selected_candidates")
        metrics = manifest.get("metrics")
        reference_metrics = manifest.get("reference_prefix_metrics")
        if (
            not isinstance(selected, Sequence)
            or isinstance(selected, (str, bytes))
            or not isinstance(metrics, Mapping)
            or not isinstance(reference_metrics, Mapping)
        ):
            raise ArtifactError("random NOISE evaluation summary fields are malformed")
        selection = selections[task.cell.cell_id]
        geometry_selection = selection["geometries"][task.geometry]
        reference = geometry_selection["reference"]
        for candidate in selected:
            if not isinstance(candidate, Mapping):
                raise ArtifactError("random NOISE selected candidate is malformed")
            position = int(candidate["selection_position"])
            for rule in experiment.rules:
                key = f"random_{position:02d}__{rule.lower()}"
                values = metrics.get(key)
                baseline = reference_metrics.get(rule.lower())
                if not isinstance(values, Mapping) or not isinstance(baseline, Mapping):
                    raise ArtifactError(f"random NOISE metric row is missing: {key}")
                for metric in QUALITY_METRICS:
                    candidate_value = float(values[metric])
                    reference_value = float(baseline[metric])
                    delta = _oriented_delta(metric, candidate_value, reference_value)
                    rows.append(
                        {
                            "cell": task.cell.cell_id,
                            "dataset": task.cell.dataset.dataset_id,
                            "model": task.cell.reference_model.model_id,
                            "condition": task.condition.condition_id,
                            "geometry": task.geometry,
                            "q": int(manifest["q"]),
                            "candidate_position": position,
                            "candidate_digest": str(candidate["candidate_digest"]),
                            "methods": list(candidate["methods"]),
                            "mean_individual_clean_F": float(candidate["mean_individual_clean_F"]),
                            "reference_methods": list(reference["methods"]),
                            "reference_mean_individual_clean_F": float(
                                reference["mean_individual_clean_F"]
                            ),
                            "candidate_ks": float(candidate["gof"]["ks_statistic"]),
                            "reference_ks": float(reference["gof"]["ks_statistic"]),
                            "rule": rule.lower(),
                            "metric": metric,
                            "direction": DEFAULT_METRIC_DIRECTIONS[metric],
                            "candidate_value": candidate_value,
                            "reference_value": reference_value,
                            "oriented_delta": delta,
                            "outcome": _outcome(delta),
                        }
                    )

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[("overall", "all")].append(row)
        grouped[("geometry", str(row["geometry"]))].append(row)
        grouped[("metric", str(row["metric"]))].append(row)
        grouped[("rule", str(row["rule"]))].append(row)
    aggregates = []
    for (dimension, value), group in sorted(grouped.items()):
        deltas = [float(row["oriented_delta"]) for row in group]
        outcomes = [str(row["outcome"]) for row in group]
        aggregates.append(
            {
                "group_dimension": dimension,
                "group_value": value,
                "endpoints": len(group),
                "wins": outcomes.count("win"),
                "ties": outcomes.count("tie"),
                "losses": outcomes.count("loss"),
                "mean_oriented_delta": float(sum(deltas) / len(deltas)),
                "median_oriented_delta": float(statistics.median(deltas)),
            }
        )
    value: dict[str, Any] = {
        "schema": "simple-noise-random-subset-summary-v1",
        "schema_version": 1,
        "status": "complete",
        "study_id": experiment.study_id,
        "study_digest": experiment.digest,
        "science": {
            **dict(experiment.raw_science),
            "comparison": "uniform_noise_consistent_random_subset_minus_fidelity_prefix",
            "metric_orientation": dict(DEFAULT_METRIC_DIRECTIONS),
            "selection_never_reads_final_quality": True,
        },
        "selector_digest": experiment.selector_digest,
        "selection_manifest_content_digests": selection_digests,
        "evaluation_manifest_content_digests": manifest_digests,
        "counts": {
            "rows": len(rows),
            "evaluations": len(experiment.evaluation_tasks()),
            "cells": len(experiment.cells()),
            "conditions": len(experiment.conditions()),
        },
        "aggregates": aggregates,
        "rows": rows,
    }
    value["result_digest"] = object_sha256(value)
    return value


def write_summary(
    experiment: NoiseSubsetExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    if experiment.control_mode == ANCHORED_CONTROL_MODE:
        from .anchored_summary import write_anchored_summary

        return write_anchored_summary(experiment, output_directory=output_directory)
    value = build_summary(experiment)
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else experiment.storage.scratch_root / "summaries"
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", value)
    row_columns = (
        "cell",
        "dataset",
        "model",
        "condition",
        "geometry",
        "q",
        "candidate_position",
        "candidate_digest",
        "methods",
        "mean_individual_clean_F",
        "reference_methods",
        "reference_mean_individual_clean_F",
        "candidate_ks",
        "reference_ks",
        "rule",
        "metric",
        "direction",
        "candidate_value",
        "reference_value",
        "oriented_delta",
        "outcome",
    )
    csv_rows = [
        {
            **row,
            "methods": json.dumps(row["methods"], separators=(",", ":")),
            "reference_methods": json.dumps(row["reference_methods"], separators=(",", ":")),
        }
        for row in value["rows"]
    ]
    atomic_write_text(output / "comparisons.csv", _csv_text(csv_rows, row_columns))
    aggregate_columns = (
        "group_dimension",
        "group_value",
        "endpoints",
        "wins",
        "ties",
        "losses",
        "mean_oriented_delta",
        "median_oriented_delta",
    )
    atomic_write_text(
        output / "aggregate_summary.csv",
        _csv_text(value["aggregates"], aggregate_columns),
    )
    overall = next(
        row
        for row in value["aggregates"]
        if row["group_dimension"] == "overall" and row["group_value"] == "all"
    )
    readme = "\n".join(
        (
            "# Noise-consistent Random Subsets",
            "",
            "Random subsets are sampled uniformly from the fixed-q candidate pool whose exact",
            "analytic subset-Mallows empirical-CDF KS error is no worse than the frozen",
            "Fidelity-prefix reference. Candidate selection never reads aggregate F/Fbar/C/Cbar.",
            "",
            f"Overall wins/ties/losses: {overall['wins']}/{overall['ties']}/{overall['losses']}.",
            f"Mean oriented difference: {overall['mean_oriented_delta']}.",
            f"Result digest: `{value['result_digest']}`.",
            "",
        )
    )
    atomic_write_text(output / "README.md", readme)
    return {**value, "output_directory": str(output)}


__all__ = ["build_summary", "write_summary"]
