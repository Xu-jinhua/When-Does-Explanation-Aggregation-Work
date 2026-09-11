"""Deterministic table-ready JSON/CSV summaries for IND and Oracle NOISE."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from ..artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    phase2_artifact_root,
)
from ..robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
    perturbed_quality_values,
    signed_robustness_values,
)
from ..summary import NOISE_ORDER, PAPER_RULES, build_table1_summary
from .artifacts import completed_evaluation_manifest, completed_selection_manifest
from .config import AssumptionExperiment

DISTANCE_MODELS = ("spearman", "kendall")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty collection")
    return float(sum(values) / len(values))


def _rows_from_manifest(
    manifest: Mapping[str, Any],
    *,
    clean_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    condition = str(manifest["condition"])
    source = {
        "cell": manifest["cell"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "setting": manifest["setting"],
        "source_id": manifest.get("source_id"),
        "family_id": manifest.get("family_id"),
        "distance_model": manifest.get("distance_model"),
        "condition": condition,
    }
    if condition == "clean":
        for rule, values in manifest["metrics"].items():
            for metric in QUALITY_METRICS:
                rows.append(
                    {
                        **source,
                        "value_kind": "quality",
                        "rule": rule,
                        "metric": metric,
                        "value": float(values[metric]),
                    }
                )
    else:
        if set(manifest["metrics"]) != set(clean_manifest["metrics"]):
            raise ArtifactError(f"Assumption {condition} rule roster differs from clean")
        robustness = manifest.get("robustness")
        if not isinstance(robustness, Mapping) or set(robustness) != set(manifest["metrics"]):
            raise ArtifactError(f"Assumption {condition} robustness roster differs from metrics")
        for rule, values in manifest["metrics"].items():
            record = robustness[rule]
            if not isinstance(record, Mapping):
                raise ArtifactError(f"Assumption {condition}/{rule} robustness is invalid")
            perturbed = perturbed_quality_values(
                values,
                context=f"Assumption {condition}/{rule}",
            )
            signed = signed_robustness_values(
                clean_manifest["metrics"][rule],
                values,
                legacy_absolute=record.get("absolute"),
                context=f"Assumption {condition}/{rule}",
            )
            for metric in QUALITY_METRICS:
                rows.extend(
                    (
                        {
                            **source,
                            "value_kind": "conditioned_quality",
                            "rule": rule,
                            "metric": metric,
                            "value": perturbed[metric],
                        },
                        {
                            **source,
                            "value_kind": "robustness",
                            "rule": rule,
                            "metric": metric,
                            "value": signed[metric],
                        },
                    )
                )
    return rows


def _rows_from_manifests(manifests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    collection_fields = ("cell", "setting", "source_id", "family_id", "distance_model")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        groups[tuple(manifest.get(name) for name in collection_fields)].append(manifest)
    rows = []
    for key, group in groups.items():
        clean = [manifest for manifest in group if manifest.get("condition") == "clean"]
        if len(clean) != 1:
            raise ArtifactError(f"Assumption collection {key} has {len(clean)} clean manifests")
        sample_count = int(clean[0]["sample_count"])
        for manifest in group:
            if int(manifest["sample_count"]) != sample_count:
                raise ArtifactError(f"Assumption collection {key} sample counts are not aligned")
            rows.extend(_rows_from_manifest(manifest, clean_manifest=clean[0]))
    return rows


def _average_matched(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    template = {}
    for row in rows:
        if row["setting"] != "matched-naive":
            continue
        key = tuple(
            row[name]
            for name in (
                "cell",
                "dataset",
                "model",
                "distance_model",
                "condition",
                "value_kind",
                "rule",
                "metric",
            )
        )
        grouped[key].append(float(row["value"]))
        template[key] = dict(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = template[key]
        row["source_id"] = "mean-over-sources"
        row["value"] = _mean(values)
        row["source_count"] = len(values)
        result.append(row)
    return result


def _naive_rows(experiment: AssumptionExperiment) -> list[dict[str, Any]]:
    rows = []
    store = ArtifactStore(experiment.base)
    for cell in experiment.cells():
        manifests = {}
        for condition in experiment.base.conditions:
            task = experiment.base_phase2_task(cell, condition.condition_id)
            root = phase2_artifact_root(task)
            manifest = completed_manifest(
                store,
                root,
                expected_task_digest=task.digest,
                expected_schema_version=PHASE2_SCHEMA_VERSION,
            )
            if manifest is None:
                raise FileNotFoundError(f"NAIVE p=16 Phase 2 artifact is incomplete: {root}")
            manifests[condition.condition_id] = manifest
        clean_condition = next(
            condition.condition_id
            for condition in experiment.base.conditions
            if condition.kind == "clean"
        )
        clean = manifests[clean_condition]
        for condition in experiment.base.conditions:
            manifest = manifests[condition.condition_id]
            source = {
                "cell": cell.cell_id,
                "dataset": cell.dataset.dataset_id,
                "model": cell.reference_model.model_id,
                "setting": "naive",
                "source_id": "reference-full",
                "distance_model": None,
                "condition": condition.condition_id,
            }
            if condition.kind == "clean":
                for rule, values in manifest["metrics"].items():
                    for metric in QUALITY_METRICS:
                        rows.append(
                            {
                                **source,
                                "value_kind": "quality",
                                "rule": rule,
                                "metric": metric,
                                "value": float(values[metric]),
                            }
                        )
            else:
                if set(manifest["metrics"]) != set(clean["metrics"]):
                    raise ArtifactError(
                        f"NAIVE {cell.cell_id}/{condition.condition_id} rule roster changed"
                    )
                robustness = manifest.get("robustness")
                if not isinstance(robustness, Mapping) or set(robustness) != set(
                    manifest["metrics"]
                ):
                    raise ArtifactError(
                        f"NAIVE {cell.cell_id}/{condition.condition_id} robustness is invalid"
                    )
                for rule, values in manifest["metrics"].items():
                    record = robustness[rule]
                    if not isinstance(record, Mapping):
                        raise ArtifactError(
                            f"NAIVE {cell.cell_id}/{condition.condition_id}/{rule} is invalid"
                        )
                    signed = signed_robustness_values(
                        clean["metrics"][rule],
                        values,
                        legacy_absolute=record.get("absolute"),
                        context=f"NAIVE {cell.cell_id}/{condition.condition_id}/{rule}",
                    )
                    for metric in QUALITY_METRICS:
                        rows.extend(
                            (
                                {
                                    **source,
                                    "value_kind": "conditioned_quality",
                                    "rule": rule,
                                    "metric": metric,
                                    "value": float(values[metric]),
                                },
                                {
                                    **source,
                                    "value_kind": "robustness",
                                    "rule": rule,
                                    "metric": metric,
                                    "value": signed[metric],
                                },
                            )
                        )
    return rows


def _best_individual_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    collection_fields = ("cell", "setting", "source_id", "distance_model")
    clean_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not str(row["rule"]).startswith("single__") or row["value_kind"] != "quality":
            continue
        key = (
            *tuple(row[name] for name in collection_fields),
            row["metric"],
        )
        clean_groups[key].append(row)

    selected: dict[tuple[Any, ...], str] = {}
    for key, candidates in clean_groups.items():
        metric = str(candidates[0]["metric"])
        maximize = DEFAULT_METRIC_DIRECTIONS[metric] == "max"
        chosen = min(
            candidates,
            key=lambda row: (
                -float(row["value"]) if maximize else float(row["value"]),
                str(row["rule"]),
            ),
        )
        selected[key] = str(chosen["rule"])

    result = []
    for row in rows:
        if not str(row["rule"]).startswith("single__"):
            continue
        key = (
            *tuple(row[name] for name in collection_fields),
            row["metric"],
        )
        selected_rule = selected.get(key)
        if selected_rule is None or row["rule"] != selected_rule:
            continue
        value = dict(row)
        value["rule"] = "best-individual"
        value["selected_individual"] = selected_rule[len("single__") :]
        value["best_individual_collection"] = str(row["setting"])
        value["selection_policy"] = "clean_metric_anchored"
        result.append(value)
    return sorted(
        result,
        key=lambda row: (
            str(row["cell"]),
            str(row["setting"]),
            str(row.get("source_id")),
            str(row.get("distance_model")),
            str(row["condition"]),
            str(row["metric"]),
        ),
    )


def _oracle_noise_best_individual_rows(
    best_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind NOISE comparisons to the original complete NAIVE collection."""

    result = []
    for row in best_rows:
        if row["setting"] != "naive":
            continue
        for distance_model in ("spearman", "kendall"):
            value = dict(row)
            value["setting"] = "oracle-noise"
            value["distance_model"] = distance_model
            value["best_individual_collection"] = "original-complete-naive"
            result.append(value)
    return result


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "cell",
        "dataset",
        "model",
        "setting",
        "source_id",
        "source_count",
        "distance_model",
        "condition",
        "value_kind",
        "rule",
        "metric",
        "value",
        "selected_individual",
        "best_individual_collection",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return output.getvalue()


def _table_columns() -> tuple[tuple[str, str], ...]:
    quality = tuple((metric, DEFAULT_METRIC_DIRECTIONS[metric]) for metric in QUALITY_METRICS)
    robustness = tuple(
        (f"R_{metric}_{noise}", SIGNED_ROBUSTNESS_DIRECTION)
        for noise in NOISE_ORDER
        for metric in QUALITY_METRICS
    )
    return (*quality, *robustness)


def _flat_table_row(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        **{metric: float(row["quality"][metric]) for metric in QUALITY_METRICS},
        **{
            f"R_{metric}_{noise}": float(row["robustness"][noise][metric])
            for noise in NOISE_ORDER
            for metric in QUALITY_METRICS
        },
    }


def _flat_perturbed_quality(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"{metric}_perturbed_{noise}": float(row["perturbed_quality"][noise][metric])
        for noise in NOISE_ORDER
        for metric in QUALITY_METRICS
    }


def _manifest_assumption_identity(manifest: Mapping[str, Any], *, context: str) -> tuple[str, str]:
    assumption_id = manifest.get("assumption_id")
    assumption_digest = manifest.get("assumption_digest")
    if not isinstance(assumption_id, str) or not assumption_id:
        raise ArtifactError(f"{context} has no valid assumption_id")
    if (
        not isinstance(assumption_digest, str)
        or len(assumption_digest) != 64
        or any(character not in "0123456789abcdef" for character in assumption_digest)
    ):
        raise ArtifactError(f"{context} has no valid assumption_digest")
    return assumption_id, assumption_digest


def _validate_noise_manifest(
    experiment: AssumptionExperiment,
    task: Any,
    manifest: Mapping[str, Any],
    *,
    allowed_assumption_identities: set[tuple[str, str]],
) -> tuple[str, str]:
    assumption_identity = _manifest_assumption_identity(
        manifest, context=f"Oracle NOISE evaluation {task.task_id}"
    )
    if assumption_identity not in allowed_assumption_identities:
        raise ArtifactError(
            "Oracle NOISE evaluation has an unrelated assumption identity: "
            f"actual={assumption_identity}, allowed={sorted(allowed_assumption_identities)}"
        )
    expected = {
        "status": "complete",
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell.cell_id,
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "setting": "oracle-noise",
        "source_id": None,
        "distance_model": task.distance_model,
        "condition": task.condition.condition_id,
        "split": experiment.split,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "fill": "dataset_mean",
        "target_policy": "full_reference_clean_fp32_prediction",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Oracle NOISE evaluation identity mismatch: {mismatches}")
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ArtifactError(f"Oracle NOISE evaluation has invalid sample_count: {task.task_id}")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping) or not set(PAPER_RULES).issubset(metrics):
        raise ArtifactError(f"Oracle NOISE evaluation lacks paper rules: {task.task_id}")
    for rule, values in metrics.items():
        if not isinstance(values, Mapping) or set(values) != set(QUALITY_METRICS):
            raise ArtifactError(f"Oracle NOISE quality fields are invalid: {task.task_id}/{rule}")
        if any(not math.isfinite(float(values[metric])) for metric in QUALITY_METRICS):
            raise ArtifactError(f"Oracle NOISE quality value is non-finite: {task.task_id}/{rule}")
    return assumption_identity


def _noise_input_groups(
    experiment: AssumptionExperiment,
    *,
    condition_keys: Mapping[str, str],
    selection_identities: Mapping[tuple[str, str], tuple[str, str]],
) -> tuple[
    Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
]:
    tasks = tuple(task for task in experiment.evaluation_tasks() if task.setting == "oracle-noise")
    expected_count = len(experiment.cells()) * len(DISTANCE_MODELS) * len(condition_keys)
    if len(tasks) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} Oracle NOISE evaluations; found {len(tasks)}"
        )
    store = ArtifactStore(experiment)  # type: ignore[arg-type]
    manifests: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    sources: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for task in tasks:
        if task.distance_model not in DISTANCE_MODELS:
            raise ArtifactError(f"Oracle NOISE task has invalid distance model: {task.task_id}")
        condition_key = condition_keys.get(task.condition.condition_id)
        if condition_key is None:
            raise ArtifactError(f"Oracle NOISE task has unknown condition: {task.task_id}")
        manifest = completed_evaluation_manifest(experiment, task, store=store)
        if manifest is None:
            raise FileNotFoundError(f"Oracle NOISE evaluation is incomplete: {task.task_id}")
        group = (task.cell.cell_id, task.distance_model)
        selection_identity = selection_identities.get(group)
        if selection_identity is None:
            raise ArtifactError(f"Oracle NOISE evaluation has no selection identity: {group}")
        manifest_identity = _validate_noise_manifest(
            experiment,
            task,
            manifest,
            allowed_assumption_identities={
                (experiment.assumption_id, experiment.digest),
                selection_identity,
            },
        )
        if condition_key in manifests[group]:
            raise ArtifactError(f"Duplicate Oracle NOISE evaluation for {group}/{condition_key}")
        manifests[group][condition_key] = manifest
        sources[group][condition_key] = {
            "task_id": task.task_id,
            "task_digest": task.digest,
            "rank_task_id": task.rank_task_id,
            "artifact_root": task.artifact_root,
            "manifest_locator": store.locator(f"{task.artifact_root}/manifest.json"),
            "manifest_content_digest": object_sha256(manifest),
            "manifest_assumption_id": manifest_identity[0],
            "manifest_assumption_digest": manifest_identity[1],
        }
    required = set(condition_keys.values())
    for group, values in manifests.items():
        if set(values) != required:
            raise ArtifactError(
                f"Oracle NOISE condition coverage mismatch for {group}: {sorted(values)}"
            )
    return manifests, sources


def _selection_inputs(
    experiment: AssumptionExperiment,
) -> tuple[Mapping[tuple[str, str], Mapping[str, Any]], list[Mapping[str, Any]]]:
    store = ArtifactStore(experiment)  # type: ignore[arg-type]
    by_group = {}
    sources = []
    for task in experiment.selection_tasks():
        manifest = completed_selection_manifest(experiment, task, store=store)
        if manifest is None:
            raise FileNotFoundError(f"Oracle NOISE selection is incomplete: {task.task_id}")
        expected = {
            "status": "complete",
            "task_id": task.task_id,
            "task_digest": task.digest,
            "cell": task.cell.cell_id,
            "distance_model": task.distance_model,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ArtifactError(f"Oracle NOISE selection identity mismatch: {mismatches}")
        manifest_identity = _manifest_assumption_identity(
            manifest, context=f"Oracle NOISE selection {task.task_id}"
        )
        selection = manifest.get("selection")
        if not isinstance(selection, Mapping):
            raise ArtifactError(f"Oracle NOISE selection is malformed: {task.task_id}")
        group = (task.cell.cell_id, task.distance_model)
        if group in by_group:
            raise ArtifactError(f"Duplicate Oracle NOISE selection for {group}")
        source = {
            "cell": task.cell.cell_id,
            "dataset": task.cell.dataset.dataset_id,
            "model": task.cell.reference_model.model_id,
            "distance_model": task.distance_model,
            "task_id": task.task_id,
            "task_digest": task.digest,
            "artifact_root": task.artifact_root,
            "manifest_locator": store.locator(f"{task.artifact_root}/manifest.json"),
            "manifest_content_digest": object_sha256(manifest),
            "manifest_assumption_id": manifest_identity[0],
            "manifest_assumption_digest": manifest_identity[1],
            **dict(selection),
        }
        by_group[group] = source
        sources.append(source)
    expected_count = len(experiment.cells()) * len(DISTANCE_MODELS)
    if len(by_group) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} Oracle NOISE selections; found {len(by_group)}"
        )
    return by_group, sources


def _noise_table_row(
    rule: str,
    manifests: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    clean = manifests["clean"]
    perturbed_quality = {
        noise: perturbed_quality_values(
            manifests[noise]["metrics"][rule],
            context=f"Oracle NOISE {noise}/{rule}",
        )
        for noise in NOISE_ORDER
    }
    return {
        "method": rule,
        "setting": "oracle-noise",
        "quality": {metric: float(clean["metrics"][rule][metric]) for metric in QUALITY_METRICS},
        "perturbed_quality": perturbed_quality,
        "robustness": {
            noise: signed_robustness_values(
                clean["metrics"][rule],
                manifests[noise]["metrics"][rule],
                context=f"Oracle NOISE {noise}/{rule}",
            )
            for noise in NOISE_ORDER
        },
    }


def _validate_noise_group(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    clean = manifests["clean"]
    sample_count = int(clean["sample_count"])
    rules = set(clean["metrics"])
    for noise in NOISE_ORDER:
        manifest = manifests[noise]
        if int(manifest["sample_count"]) != sample_count or set(manifest["metrics"]) != rules:
            raise ArtifactError(f"Oracle NOISE {noise} evaluation is not aligned with clean")
        robustness = manifest.get("robustness")
        if not isinstance(robustness, Mapping) or set(robustness) != rules:
            raise ArtifactError(f"Oracle NOISE {noise} robustness roster is not aligned")
        for rule in rules:
            record = robustness[rule]
            if not isinstance(record, Mapping):
                raise ArtifactError(f"Oracle NOISE {noise}/{rule} robustness fields are invalid")
            signed_robustness_values(
                clean["metrics"][rule],
                manifest["metrics"][rule],
                legacy_absolute=record.get("absolute"),
                context=f"Oracle NOISE {noise}/{rule}",
            )


def _outcome(value: float, reference: float, direction: str) -> str:
    benefit = value - reference if direction == "max" else reference - value
    if benefit > 1e-12:
        return "better"
    if benefit < -1e-12:
        return "worse"
    return "equal"


def _noise_analysis(cells: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def summarize(selected: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        versus_naive = {rule: {"better": 0, "equal": 0, "worse": 0} for rule in PAPER_RULES}
        versus_best = {rule: {"better": 0, "equal": 0, "worse": 0} for rule in PAPER_RULES}
        for cell in selected:
            by_key = {(row["setting"], row["method"]): row for row in cell["rows"]}
            best = _flat_table_row(by_key[("best-individual", "best_individual")])
            for rule in PAPER_RULES:
                naive = _flat_table_row(by_key[("naive", rule)])
                noise = _flat_table_row(by_key[("oracle-noise", rule)])
                for column, direction in _table_columns():
                    versus_naive[rule][_outcome(noise[column], naive[column], direction)] += 1
                    versus_best[rule][_outcome(noise[column], best[column], direction)] += 1
        return {
            "comparison_cells": len(selected) * len(_table_columns()),
            "oracle_noise_versus_same_rule_naive": versus_naive,
            "oracle_noise_versus_best_individual": versus_best,
        }

    return {
        "overall": summarize(cells),
        "by_distance_model": {
            distance: summarize([cell for cell in cells if cell["distance_model"] == distance])
            for distance in DISTANCE_MODELS
        },
    }


def build_noise_summary(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    """Build the complete Oracle NOISE result without waiting for IND jobs."""

    if experiment.patch_size != 16 or experiment.k != 20:
        raise ValueError("The paper Oracle NOISE summary is fixed to p=16 and k=20")
    naive = build_table1_summary(experiment.base, manifest_source="remote")
    noise_settings = naive["settings"]["noise"]
    clean_condition = next(
        condition.condition_id
        for condition in experiment.base.conditions
        if condition.kind == "clean"
    )
    condition_keys = {
        clean_condition: "clean",
        **{str(noise_settings[key]["condition_id"]): key for key in NOISE_ORDER},
    }
    selections, selection_sources = _selection_inputs(experiment)
    selection_identities = {
        group: (
            str(selection["manifest_assumption_id"]),
            str(selection["manifest_assumption_digest"]),
        )
        for group, selection in selections.items()
    }
    manifests, evaluation_sources = _noise_input_groups(
        experiment,
        condition_keys=condition_keys,
        selection_identities=selection_identities,
    )
    naive_cells = {f"{cell['dataset']}--{cell['model']}": cell for cell in naive["cells"]}
    cells = []
    for cell in experiment.cells():
        naive_cell = naive_cells.get(cell.cell_id)
        if naive_cell is None:
            raise ArtifactError(f"Missing original NAIVE result for {cell.cell_id}")
        naive_rows = {row["method"]: row for row in naive_cell["rows"]}
        for distance in DISTANCE_MODELS:
            group = (cell.cell_id, distance)
            group_manifests = manifests[group]
            _validate_noise_group(group_manifests)
            if int(group_manifests["clean"]["sample_count"]) != int(naive_cell["sample_count"]):
                raise ArtifactError(f"NOISE and NAIVE sample counts differ for {group}")
            rows = [
                {
                    **dict(naive_rows["best_individual"]),
                    "setting": "best-individual",
                    "best_individual_collection": "original-complete-naive",
                }
            ]
            for rule in PAPER_RULES:
                rows.append({**dict(naive_rows[rule]), "setting": "naive"})
                rows.append(dict(_noise_table_row(rule, group_manifests)))
            selection = selections[group]
            cells.append(
                {
                    "cell": cell.cell_id,
                    "dataset": cell.dataset.dataset_id,
                    "model": cell.reference_model.model_id,
                    "model_key": cell.reference_model.model_key,
                    "architecture": cell.reference_model.architecture,
                    "split": experiment.split,
                    "sample_count": int(group_manifests["clean"]["sample_count"]),
                    "distance_model": distance,
                    "selected_methods": list(selection["selected_methods"]),
                    "selected_size": int(selection["selected_size"]),
                    "forced_fallback": bool(selection["forced_fallback"]),
                    "source_tasks": dict(evaluation_sources[group]),
                    "rows": rows,
                }
            )
    identity = {
        "schema": "simple-oracle-noise-summary-v2",
        "assumption_digest": experiment.digest,
        "base_experiment_digest": experiment.base.digest,
        "base_summary_digest": object_sha256(naive),
        "artifact_assumption_identity_policy": ("current_or_task_identical_selection_predecessor"),
        "artifact_assumption_identities": [
            {"assumption_id": assumption_id, "assumption_digest": assumption_digest}
            for assumption_id, assumption_digest in sorted(
                {
                    (
                        str(source["manifest_assumption_id"]),
                        str(source["manifest_assumption_digest"]),
                    )
                    for group in evaluation_sources.values()
                    for source in group.values()
                }
                | {
                    (
                        str(source["manifest_assumption_id"]),
                        str(source["manifest_assumption_digest"]),
                    )
                    for source in selection_sources
                }
            )
        ],
        "evaluation_tasks": [
            {
                "task_id": source[condition]["task_id"],
                "task_digest": source[condition]["task_digest"],
                "manifest_content_digest": source[condition]["manifest_content_digest"],
            }
            for source in evaluation_sources.values()
            for condition in ("clean", *NOISE_ORDER)
        ],
        "selection_tasks": [
            {
                "task_id": selection["task_id"],
                "task_digest": selection["task_digest"],
                "manifest_content_digest": selection["manifest_content_digest"],
            }
            for selection in selection_sources
        ],
    }
    summary = {
        "schema_version": 2,
        "status": "complete",
        "result": "Oracle_NOISE",
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "base_summary_digest": object_sha256(naive),
        "science": {
            "split": experiment.split,
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "fill": "dataset_mean",
            "target_policy": "full_reference_clean_fp32_prediction",
            "selection_scope": "complete_test_set_in_sample",
            "selection_alpha": experiment.selection.alpha,
            "best_individual_policy": "clean_metric_anchored_from_original_complete_naive_collection",
            "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
            "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
            "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
            "legacy_absolute_R_used": False,
            "perturbed_quality_reported": True,
            "noise": noise_settings,
        },
        "columns": [name for name, _ in _table_columns()],
        "column_directions": dict(_table_columns()),
        "perturbed_quality_columns": [
            f"{metric}_perturbed_{noise}" for noise in NOISE_ORDER for metric in QUALITY_METRICS
        ],
        "source_identity": identity,
        "selections": selection_sources,
        "cells": cells,
        "analysis": _noise_analysis(cells),
    }
    return {**summary, "summary_digest": object_sha256(summary)}


def _noise_csv_text(summary: Mapping[str, Any]) -> str:
    fields = (
        "cell",
        "dataset",
        "model",
        "architecture",
        "split",
        "sample_count",
        "distance_model",
        "method",
        "setting",
        *summary["columns"],
        *summary["perturbed_quality_columns"],
        "selected_sources",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cell in summary["cells"]:
        for row in cell["rows"]:
            selected = row.get("selected_sources")
            writer.writerow(
                {
                    **{field: cell.get(field, "") for field in fields},
                    "method": row["method"],
                    "setting": row["setting"],
                    **_flat_table_row(row),
                    **_flat_perturbed_quality(row),
                    "selected_sources": "" if selected is None else object_sha256(selected),
                }
            )
    return output.getvalue()


def _selection_csv_text(summary: Mapping[str, Any]) -> str:
    fields = (
        "cell",
        "dataset",
        "model",
        "distance_model",
        "selected_size",
        "candidate_count",
        "forced_fallback",
        "selection_rule",
        "scope",
        "alpha",
        "selected_methods",
        "ordered_methods",
        "prefix_p_values",
        "task_id",
        "task_digest",
        "artifact_root",
        "manifest_content_digest",
        "manifest_assumption_id",
        "manifest_assumption_digest",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for selection in summary["selections"]:
        evaluations = selection["evaluations"]
        writer.writerow(
            {
                **{field: selection.get(field, "") for field in fields},
                "candidate_count": len(selection["ordered_methods"]),
                "selected_methods": json.dumps(
                    selection["selected_methods"], separators=(",", ":")
                ),
                "ordered_methods": json.dumps(selection["ordered_methods"], separators=(",", ":")),
                "prefix_p_values": json.dumps(
                    [
                        {"size": int(value["size"]), "p_value": float(value["gof"]["p_value"])}
                        for value in evaluations
                    ],
                    separators=(",", ":"),
                ),
            }
        )
    return output.getvalue()


def write_noise_summary(
    experiment: AssumptionExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    summary = build_noise_summary(experiment)
    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else experiment.storage.scratch_root / "summaries" / "noise"
    )
    json_path = atomic_write_json(destination / "summary.json", summary)
    csv_path = atomic_write_text(destination / "summary.csv", _noise_csv_text(summary))
    selections_path = atomic_write_text(
        destination / "selections.csv", _selection_csv_text(summary)
    )
    return {
        "status": "complete",
        "result": "Oracle_NOISE",
        "summary_digest": summary["summary_digest"],
        "cells": len(summary["cells"]),
        "rows": sum(len(cell["rows"]) for cell in summary["cells"]),
        "selections": len(summary["selections"]),
        "json": str(json_path),
        "csv": str(csv_path),
        "selections_csv": str(selections_path),
    }


def write_summary(
    experiment: AssumptionExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    manifest_sources = []
    evaluation_manifests = []
    for task in experiment.evaluation_tasks():
        manifest = completed_evaluation_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(f"Assumption evaluation is incomplete: {task.task_id}")
        manifest_sources.append({"task_id": task.task_id, "task_digest": task.digest})
        evaluation_manifests.append(manifest)
    rows = _rows_from_manifests(evaluation_manifests)
    averaged_matched = _average_matched(rows)
    non_matched = [row for row in rows if row["setting"] != "matched-naive"]
    naive = _naive_rows(experiment)
    final_rows = [*non_matched, *averaged_matched, *naive]
    # Selected NOISE singles are retained as diagnostics, but they must never
    # redefine the table's Oracle Best Individual comparator.
    best = _best_individual_rows([row for row in final_rows if row["setting"] != "oracle-noise"])
    final_rows.extend([*best, *_oracle_noise_best_individual_rows(best)])
    final_rows.sort(
        key=lambda row: (
            str(row["cell"]),
            str(row["setting"]),
            str(row.get("distance_model")),
            str(row["condition"]),
            str(row["rule"]),
            str(row["metric"]),
        )
    )
    selections = []
    for task in experiment.selection_tasks():
        manifest = completed_selection_manifest(experiment, task)
        if manifest is None:
            raise FileNotFoundError(f"Oracle NOISE selection is incomplete: {task.task_id}")
        selections.append(
            {
                "cell": task.cell.cell_id,
                "distance_model": task.distance_model,
                "task_id": task.task_id,
                "task_digest": task.digest,
                **dict(manifest["selection"]),
            }
        )
    identity = {
        "schema": "simple-assumptions-summary-v4",
        "assumption_digest": experiment.digest,
        "oracle_noise_best_individual": "original_complete_naive_collection",
        "best_individual_policy": "clean_metric_anchored",
        "evaluation_tasks": manifest_sources,
        "selection_tasks": [
            {"task_id": task.task_id, "task_digest": task.digest}
            for task in experiment.selection_tasks()
        ],
    }
    summary = {
        "schema_version": 4,
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "summary_digest": object_sha256(identity),
        "science": {
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "fill": "dataset_mean",
            "target_policy": "full_reference_clean_fp32_prediction",
            "matched_naive_reduction": "metric_mean_across_source_models",
            "best_individual_policy": "clean_metric_anchored",
            "oracle_noise_scope": "complete_test_set_in_sample",
            "oracle_noise_best_individual": "original_complete_naive_collection",
            "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
            "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
            "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
            "legacy_absolute_R_used": False,
            "perturbed_quality_reported": True,
        },
        "method_assignments": {
            cell.cell_id: [list(pair) for pair in experiment.method_assignment(cell)]
            for cell in experiment.cells()
        },
        "selections": selections,
        "rows": final_rows,
    }
    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else experiment.storage.scratch_root / "summaries"
    )
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "ind-noise-summary.json"
    csv_path = destination / "ind-noise-summary.csv"
    atomic_write_json(json_path, summary)
    atomic_write_text(csv_path, _csv_text(final_rows))
    return {
        "summary_digest": summary["summary_digest"],
        "rows": len(final_rows),
        "json": str(json_path),
        "csv": str(csv_path),
    }


__all__ = ["build_noise_summary", "write_noise_summary", "write_summary"]
