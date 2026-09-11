"""Table-ready q=11 IND and full-source matched-NAIVE summary."""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
from ..robustness import (
    signed_robustness as _signed_robustness,
)
from ..summary import NOISE_ORDER, _noise_settings
from .artifacts import completed_evaluation_manifest
from .config import RULES, AssumptionExperiment
from .table_priority import (
    ind_evaluation_tasks,
    matched_evaluation_tasks,
    matched_source_ids,
)

TABLE_COLUMN_ORDER = (
    "F",
    "Fbar",
    "C",
    "Cbar",
    *(f"R_{metric}_{noise}" for metric in ("F", "C") for noise in NOISE_ORDER),
    *(f"R_{metric}_{noise}" for metric in ("Fbar", "Cbar") for noise in NOISE_ORDER),
)
PAPER_TABLE_COLUMN_ORDER = (
    "F",
    "Fbar",
    "C",
    "Cbar",
    *(f"R_{metric}_{noise}" for noise in NOISE_ORDER for metric in ("F", "C")),
    *(f"R_{metric}_{noise}" for noise in NOISE_ORDER for metric in ("Fbar", "Cbar")),
)
PERTURBED_COLUMN_ORDER = tuple(
    f"{metric}_perturbed_{noise}" for noise in NOISE_ORDER for metric in QUALITY_METRICS
)
AGGREGATE_RULE_KEYS = tuple(rule.lower() for rule in RULES)
PAPER_RULE_ORDER = ("SimpleAvg", "Borda", "Kemeny", "RRF", "Schulze")
PAPER_RULE_LABELS = {
    "SimpleAvg": "Simple Averaging",
    "Borda": "Borda",
    "Kemeny": "Kemeny--Young",
    "RRF": "RRF",
    "Schulze": "Schulze",
}
PAPER_DATASET_LABELS = {
    "imagenet100": "ImageNet100",
    "imagenet1k": "ImageNet-1K",
    "dermamnist": "DermaMNIST",
    "pathmnist": "PathMNIST",
    "octmnist": "OCTMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "retinamnist": "RetinaMNIST",
    "breastmnist": "BreastMNIST",
    "bloodmnist": "BloodMNIST",
    "tissuemnist": "TissueMNIST",
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
}
PAPER_MODEL_LABELS = {
    "imagenet1k-resnet18": "ResNet-18",
    "imagenet1k-vit-b16": "ViT-B/16",
    "imagenet100-resnet18": "ResNet-18",
    "imagenet100-vit-b16": "ViT-B/16",
    "dermamnist-resnet18": "ResNet-18",
    "dermamnist-vit-b16": "ViT-B/16",
    "pathmnist-vit-b16": "ViT-B/16",
    "octmnist-vit-b16": "ViT-B/16",
    "pneumoniamnist-vit-b16": "ViT-B/16",
    "retinamnist-vit-b16": "ViT-B/16",
    "breastmnist-vit-b16": "ViT-B/16",
    "bloodmnist-vit-b16": "ViT-B/16",
    "tissuemnist-vit-b16": "ViT-B/16",
    "organamnist-vit-b16": "ViT-B/16",
    "organcmnist-vit-b16": "ViT-B/16",
    "organsmnist-vit-b16": "ViT-B/16",
}
SUMMARY_SCAN_WORKERS = 8


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty model collection")
    return float(statistics.fmean(values)), float(statistics.stdev(values)) if len(
        values
    ) > 1 else 0.0


def _validated_manifest(
    experiment: AssumptionExperiment,
    task: Any,
) -> Mapping[str, Any]:
    manifest = completed_evaluation_manifest(experiment, task)
    if manifest is None:
        raise FileNotFoundError(f"IND table evaluation is incomplete: {task.task_id}")
    expected = {
        "task_id": task.task_id,
        "task_digest": task.digest,
        "setting": task.setting,
        "cell": task.cell.cell_id,
        "condition": task.condition.condition_id,
        "source_id": task.source_id,
        "family_id": task.family_id,
        "target_policy": "full_reference_clean_fp32_prediction",
        "patch_size": 16,
        "k": 20,
        "fill": "dataset_mean",
    }
    mismatches = {
        key: {"artifact": manifest.get(key), "current": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"IND table evaluation identity changed: {mismatches}")
    return manifest


def _input_manifests(
    experiment: AssumptionExperiment,
) -> tuple[Mapping[str, Mapping[str, Mapping[str, Any]]], list[Mapping[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    provenance = []
    tasks = (*matched_evaluation_tasks(experiment), *ind_evaluation_tasks(experiment))

    def inspect(task: Any) -> tuple[Any, Mapping[str, Any]]:
        return task, _validated_manifest(experiment, task)

    with ThreadPoolExecutor(
        max_workers=min(SUMMARY_SCAN_WORKERS, len(tasks)),
        thread_name_prefix="ind-table-summary-scan",
    ) as executor:
        inspected = tuple(executor.map(inspect, tasks))

    for task, manifest in inspected:
        source = str(task.source_id)
        key = f"{task.setting}:{source}"
        cell_group = grouped.setdefault(task.cell.cell_id, {})
        condition_group = cell_group.setdefault(key, {})
        if task.condition.condition_id in condition_group:
            raise ArtifactError(f"Duplicate IND table evaluation input: {task.task_id}")
        condition_group[task.condition.condition_id] = manifest
        provenance.append(
            {
                "task_id": task.task_id,
                "task_digest": task.digest,
                "artifact_root": task.artifact_root,
                "setting": task.setting,
                "cell": task.cell.cell_id,
                "source_id": task.source_id,
                "family_id": task.family_id,
                "condition": task.condition.condition_id,
                "manifest_content_digest": object_sha256(manifest),
            }
        )
    return grouped, provenance


def _condition_ids(experiment: AssumptionExperiment) -> tuple[str, Mapping[str, str]]:
    clean = next(
        condition.condition_id
        for condition in experiment.base.conditions
        if condition.kind == "clean"
    )
    settings = _noise_settings(experiment.base)
    return clean, {key: value.condition_id for key, value in settings.items()}


def _require_aligned_group(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    clean_condition: str,
    noise_conditions: Mapping[str, str],
    expected_rules: set[str],
) -> None:
    required_conditions = {clean_condition, *noise_conditions.values()}
    if set(manifests) != required_conditions:
        raise ArtifactError("IND table condition group is incomplete")
    clean = manifests[clean_condition]
    sample_count = int(clean["sample_count"])
    if set(clean["metrics"]) != expected_rules:
        raise ArtifactError("IND table clean rule roster changed")
    for condition_id in noise_conditions.values():
        current = manifests[condition_id]
        if int(current["sample_count"]) != sample_count:
            raise ArtifactError("IND table evaluation sample counts are not aligned")
        if (
            set(current["metrics"]) != expected_rules
            or set(current["robustness"]) != expected_rules
        ):
            raise ArtifactError("IND table perturbed rule roster changed")
        for rule in expected_rules:
            signed_robustness_values(
                clean["metrics"][rule],
                current["metrics"][rule],
                legacy_absolute=current["robustness"][rule].get("absolute"),
                context=f"IND table {condition_id}/{rule}",
            )


def _single_method_statistics(
    source_groups: Sequence[Mapping[str, Mapping[str, Any]]],
    methods: Sequence[str],
    *,
    clean_condition: str,
    noise_conditions: Mapping[str, str],
) -> Mapping[str, Any]:
    quality: dict[str, dict[str, tuple[float, float]]] = {}
    perturbed_quality: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    robustness: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for method in methods:
        rule = f"single__{method}"
        quality[method] = {}
        for metric in QUALITY_METRICS:
            quality[method][metric] = _mean_sd(
                [float(group[clean_condition]["metrics"][rule][metric]) for group in source_groups]
            )
        perturbed_quality[method] = {}
        robustness[method] = {}
        for noise, condition_id in noise_conditions.items():
            perturbed_quality[method][noise] = {}
            robustness[method][noise] = {}
            for metric in QUALITY_METRICS:
                perturbed_quality[method][noise][metric] = _mean_sd(
                    [float(group[condition_id]["metrics"][rule][metric]) for group in source_groups]
                )
                robustness[method][noise][metric] = _mean_sd(
                    [
                        _signed_robustness(
                            metric,
                            clean=float(group[clean_condition]["metrics"][rule][metric]),
                            perturbed=float(group[condition_id]["metrics"][rule][metric]),
                        )
                        for group in source_groups
                    ]
                )
    return {
        "quality": quality,
        "perturbed_quality": perturbed_quality,
        "robustness": robustness,
    }


def _best_individual_row(
    statistics_by_method: Mapping[str, Any],
    methods: Sequence[str],
) -> Mapping[str, Any]:
    quality_statistics = statistics_by_method["quality"]
    selected = {}
    for metric in QUALITY_METRICS:
        maximize = DEFAULT_METRIC_DIRECTIONS[metric] == "max"
        selected[metric] = min(
            methods,
            key=lambda method: (
                -quality_statistics[method][metric][0]
                if maximize
                else quality_statistics[method][metric][0],
                method,
            ),
        )
    return {
        "method": "Best Individual",
        "setting": "best-individual",
        "quality": {
            metric: quality_statistics[selected[metric]][metric][0] for metric in QUALITY_METRICS
        },
        "perturbed_quality": {
            noise: {
                metric: statistics_by_method["perturbed_quality"][selected[metric]][noise][metric][
                    0
                ]
                for metric in QUALITY_METRICS
            }
            for noise in NOISE_ORDER
        },
        "robustness": {
            noise: {
                metric: statistics_by_method["robustness"][selected[metric]][noise][metric][0]
                for metric in QUALITY_METRICS
            }
            for noise in NOISE_ORDER
        },
        "standard_deviation": {
            "quality": {
                metric: quality_statistics[selected[metric]][metric][1]
                for metric in QUALITY_METRICS
            },
            "perturbed_quality": {
                noise: {
                    metric: statistics_by_method["perturbed_quality"][selected[metric]][noise][
                        metric
                    ][1]
                    for metric in QUALITY_METRICS
                }
                for noise in NOISE_ORDER
            },
            "robustness": {
                noise: {
                    metric: statistics_by_method["robustness"][selected[metric]][noise][metric][1]
                    for metric in QUALITY_METRICS
                }
                for noise in NOISE_ORDER
            },
        },
        "selected_individual": selected,
        "selection_policy": "clean_metric_anchored_after_mean_across_matched_models",
        "q": 1,
    }


def _matched_rule_row(
    rule: str,
    source_groups: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    display_name: str,
    clean_condition: str,
    noise_conditions: Mapping[str, str],
) -> Mapping[str, Any]:
    quality = {}
    quality_sd = {}
    for metric in QUALITY_METRICS:
        quality[metric], quality_sd[metric] = _mean_sd(
            [float(group[clean_condition]["metrics"][rule][metric]) for group in source_groups]
        )
    robustness = {}
    robustness_sd = {}
    perturbed_quality = {}
    perturbed_quality_sd = {}
    for noise, condition_id in noise_conditions.items():
        robustness[noise] = {}
        robustness_sd[noise] = {}
        perturbed_quality[noise] = {}
        perturbed_quality_sd[noise] = {}
        for metric in QUALITY_METRICS:
            perturbed_values = [
                float(group[condition_id]["metrics"][rule][metric]) for group in source_groups
            ]
            (
                perturbed_quality[noise][metric],
                perturbed_quality_sd[noise][metric],
            ) = _mean_sd(perturbed_values)
            values = [
                _signed_robustness(
                    metric,
                    clean=float(group[clean_condition]["metrics"][rule][metric]),
                    perturbed=float(group[condition_id]["metrics"][rule][metric]),
                )
                for group in source_groups
            ]
            robustness[noise][metric], robustness_sd[noise][metric] = _mean_sd(values)
    return {
        "method": display_name,
        "setting": "matched-naive",
        "quality": quality,
        "perturbed_quality": perturbed_quality,
        "robustness": robustness,
        "standard_deviation": {
            "quality": quality_sd,
            "perturbed_quality": perturbed_quality_sd,
            "robustness": robustness_sd,
        },
        "source_count": len(source_groups),
        "q": 11,
    }


def _ind_rule_row(
    rule: str,
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    display_name: str,
    clean_condition: str,
    noise_conditions: Mapping[str, str],
) -> Mapping[str, Any]:
    clean_metrics = manifests[clean_condition]["metrics"][rule]
    return {
        "method": display_name,
        "setting": "ind",
        "quality": {metric: float(clean_metrics[metric]) for metric in QUALITY_METRICS},
        "perturbed_quality": {
            noise: {
                metric: float(manifests[condition_id]["metrics"][rule][metric])
                for metric in QUALITY_METRICS
            }
            for noise, condition_id in noise_conditions.items()
        },
        "robustness": {
            noise: {
                metric: _signed_robustness(
                    metric,
                    clean=float(clean_metrics[metric]),
                    perturbed=float(manifests[condition_id]["metrics"][rule][metric]),
                )
                for metric in QUALITY_METRICS
            }
            for noise, condition_id in noise_conditions.items()
        },
        "source_count": 11,
        "q": 11,
    }


def _wide_values(row: Mapping[str, Any], *, standard_deviation: bool = False) -> Mapping[str, Any]:
    source = row.get("standard_deviation") if standard_deviation else row
    if not isinstance(source, Mapping):
        return {column: None for column in TABLE_COLUMN_ORDER}
    quality = source["quality"]
    robustness = source["robustness"]
    return {
        **{metric: float(quality[metric]) for metric in QUALITY_METRICS},
        **{
            f"R_{metric}_{noise}": float(robustness[noise][metric])
            for noise in NOISE_ORDER
            for metric in QUALITY_METRICS
        },
    }


def _wide_perturbed(
    row: Mapping[str, Any], *, standard_deviation: bool = False
) -> Mapping[str, Any]:
    source = row.get("standard_deviation") if standard_deviation else row
    if not isinstance(source, Mapping):
        return {column: None for column in PERTURBED_COLUMN_ORDER}
    values = source["perturbed_quality"]
    return {
        f"{metric}_perturbed_{noise}": float(values[noise][metric])
        for noise in NOISE_ORDER
        for metric in QUALITY_METRICS
    }


def build_ind_table_priority_summary(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    if experiment.patch_size != 16 or experiment.k != 20:
        raise ValueError("The IND paper table is fixed to p=16 and k=20")
    grouped, provenance = _input_manifests(experiment)
    clean_condition, noise_conditions = _condition_ids(experiment)
    cells = []
    selected_sources = matched_source_ids(experiment)
    for cell in experiment.cells():
        source_ids = selected_sources[cell.cell_id]
        source_groups = [grouped[cell.cell_id][f"matched-naive:{source}"] for source in source_ids]
        expected_matched_rules = {
            *AGGREGATE_RULE_KEYS, *(f"single__{method}" for method in cell.methods),
        }
        for manifests in source_groups:
            _require_aligned_group(
                manifests, clean_condition=clean_condition, noise_conditions=noise_conditions,
                expected_rules=expected_matched_rules,
            )
        ind_groups = []
        families = []
        for family_id in range(experiment.partition_families):
            family_sources = experiment.source_ids(cell, family_id)
            family_groups = [grouped[cell.cell_id][f"matched-naive:{source}"] for source in family_sources]
            ind_group = grouped[cell.cell_id][f"ind:family-{family_id:02d}"]
            assignment = experiment.method_assignment(cell, family_id)
            _require_aligned_group(
                ind_group, clean_condition=clean_condition, noise_conditions=noise_conditions,
                expected_rules={
                    *AGGREGATE_RULE_KEYS,
                    *(f"single__{source}__{method}" for source, method in assignment),
                },
            )
            ind_groups.append(ind_group)
            family_rows = []
            for display_name, rule in zip(RULES, AGGREGATE_RULE_KEYS, strict=True):
                family_rows.extend((
                    _matched_rule_row(
                        rule, family_groups, display_name=display_name,
                        clean_condition=clean_condition, noise_conditions=noise_conditions,
                    ),
                    _ind_rule_row(
                        rule, ind_group, display_name=display_name,
                        clean_condition=clean_condition, noise_conditions=noise_conditions,
                    ),
                ))
            families.append({
                "family_id": family_id,
                "partition_seed": experiment.partition_seed(cell, family_id),
                "ind_assignment": [list(pair) for pair in assignment],
                "matched_source_ids": list(family_sources),
                "rows": family_rows,
            })
        sample_counts = {
            int(group[clean_condition]["sample_count"]) for group in (*source_groups, *ind_groups)
        }
        if len(sample_counts) != 1:
            raise ArtifactError(f"IND and matched sample counts differ for {cell.cell_id}")
        singles = _single_method_statistics(
            source_groups, cell.methods,
            clean_condition=clean_condition, noise_conditions=noise_conditions,
        )
        rows = [{**_best_individual_row(singles, cell.methods), "source_count": len(source_ids)}]
        for display_name, rule in zip(RULES, AGGREGATE_RULE_KEYS, strict=True):
            rows.append(_matched_rule_row(
                rule, source_groups, display_name=display_name,
                clean_condition=clean_condition, noise_conditions=noise_conditions,
            ))
            rows.append({
                **_matched_rule_row(
                    rule, ind_groups, display_name=display_name,
                    clean_condition=clean_condition, noise_conditions=noise_conditions,
                ),
                "setting": "ind",
                "source_count": len(source_ids),
                "family_count": experiment.partition_families,
                "standard_deviation_unit": "partition_family",
            })
        cells.append({
            "cell": cell.cell_id,
            "dataset": cell.dataset.dataset_id,
            "model": cell.reference_model.model_id,
            "model_key": cell.reference_model.model_key,
            "architecture": cell.reference_model.architecture,
            "split": experiment.split,
            "sample_count": sample_counts.pop(),
            "families": families,
            "matched_source_ids": list(source_ids),
            "rows": rows,
        })
    identity = {
        "schema": "simple-assumptions-full-ind-summary-v3",
        "assumption_digest": experiment.digest,
        "base_experiment_digest": experiment.base.digest,
        "ind_q": 11,
        "partition_families": experiment.partition_families,
        "matched_source_count": experiment.partition_families * 11,
        "matched_source_ids": {
            key: list(value) for key, value in selected_sources.items()
        },
        "matched_selection_policy": experiment.matched_source_selection,
        "evaluation_inputs": provenance,
    }
    value = {
        "schema_version": 3,
        "status": "complete",
        "result": "IND_q11_matched_NAIVE_full_grid_three_families",
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "base_experiment_id": experiment.base.experiment_id,
        "base_experiment_digest": experiment.base.digest,
        "science": {
            "split": experiment.split,
            "patch_size": 16,
            "k": 20,
            "fill": "dataset_mean",
            "target_policy": "full_reference_clean_fp32_prediction",
            "ind_q": 11,
            "ind_source_models": experiment.partition_families * 11,
            "source_models_per_family": 11,
            "matched_methods_per_model": 11,
            "partition_families": experiment.partition_families,
            "matched_source_models": experiment.partition_families * 11,
            "matched_reduction": "mean_of_source_level_metrics",
            "matched_standard_deviation": "sample_sd_across_models_ddof_1",
            "best_individual_policy": "clean_anchored_per_metric_after_matched_model_mean",
            "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
            "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
            "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
            "legacy_absolute_R_used": False,
            "perturbed_quality_reported": True,
            "full_11_model_matched_grid_pending": False,
            "noise": {
                key: {
                    "condition_id": setting.condition_id,
                    "parameter": setting.parameter,
                    "value": setting.value,
                    "display_value": setting.display_value,
                }
                for key, setting in _noise_settings(experiment.base).items()
            },
        },
        "columns": list(TABLE_COLUMN_ORDER),
        "perturbed_quality_columns": list(PERTURBED_COLUMN_ORDER),
        "column_directions": {
            **{metric: DEFAULT_METRIC_DIRECTIONS[metric] for metric in QUALITY_METRICS},
            **{column: "max" for column in TABLE_COLUMN_ORDER if column.startswith("R_")},
        },
        "source_identity": identity,
        "cells": cells,
    }
    return {**value, "summary_digest": object_sha256(value)}


def _csv_text(summary: Mapping[str, Any]) -> str:
    fields = (
        "cell",
        "dataset",
        "model",
        "architecture",
        "split",
        "sample_count",
        "method",
        "setting",
        "q",
        "source_count",
        *TABLE_COLUMN_ORDER,
        *PERTURBED_COLUMN_ORDER,
        *(f"sd_{column}" for column in TABLE_COLUMN_ORDER),
        *(f"sd_{column}" for column in PERTURBED_COLUMN_ORDER),
        "selected_individual",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cell in summary["cells"]:
        for row in cell["rows"]:
            writer.writerow(
                {
                    **{name: cell.get(name, "") for name in fields},
                    "method": row["method"],
                    "setting": row["setting"],
                    "q": row["q"],
                    "source_count": row["source_count"],
                    **_wide_values(row),
                    **_wide_perturbed(row),
                    **{
                        f"sd_{key}": value
                        for key, value in _wide_values(row, standard_deviation=True).items()
                    },
                    **{
                        f"sd_{key}": value
                        for key, value in _wide_perturbed(row, standard_deviation=True).items()
                    },
                    "selected_individual": json.dumps(
                        row.get("selected_individual"), separators=(",", ":"), sort_keys=True
                    )
                    if row.get("selected_individual")
                    else "",
                }
            )
    return output.getvalue()


def _tex_value(value: float) -> str:
    rounded = round(float(value), 3)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.3f}"


def _tex_values(row: Mapping[str, Any]) -> str:
    values = _wide_values(row)
    return " & ".join(_tex_value(values[column]) for column in PAPER_TABLE_COLUMN_ORDER)


def _tex_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "% Generated IND table rows. The NAIVE rows are the full eleven-model matched baseline in each of three families.",
        "% Values are ordered exactly as the columns in paper/Table_IND.tex.",
        "% Raw perturbed quality values accompany R in summary.json and summary.csv.",
    ]
    for cell in summary["cells"]:
        lines.append(f"% {cell['dataset']} / {cell['model']}")
        for row in cell["rows"]:
            if row["setting"] == "best-individual":
                label = "\\multicolumn{2}{|l||}{Best Individual}"
            else:
                setting = "matched NAIVE" if row["setting"] == "matched-naive" else "IND"
                label = f"{row['method']} & {setting}"
            lines.append(f"{label} & {_tex_values(row)} \\\\")
    return "\n".join(lines) + "\n"


def _paper_metric_header(column: str) -> str:
    quality = {
        "F": r"\mathtt{F}",
        "Fbar": r"\bar{\mathtt{F}}",
        "C": r"\mathtt{C}",
        "Cbar": r"\bar{\mathtt{C}}",
    }
    if column in quality:
        direction = r"\uparrow" if DEFAULT_METRIC_DIRECTIONS[column] == "max" else r"\downarrow"
        return "$" + quality[column] + direction + "$"
    _, metric, noise = column.split("_", maxsplit=2)
    return rf"$\mathtt{{R}}_{{{quality[metric]}}}^{{\mathtt{{{noise}}}}}\uparrow$"


def _paper_cell_labels(cell: Mapping[str, Any]) -> tuple[str, str]:
    try:
        return (
            PAPER_DATASET_LABELS[str(cell["dataset"])],
            PAPER_MODEL_LABELS[str(cell["model"])],
        )
    except KeyError as error:
        raise ValueError(f"Unsupported IND paper-table cell label: {error.args[0]}") from error


def _paper_cell_label(cell: Mapping[str, Any]) -> str:
    dataset, model = _paper_cell_labels(cell)
    return f"\\texttt{{{dataset}}}, \\texttt{{{model}}}"


def _paper_label_list(labels: Sequence[str]) -> str:
    items = [rf"\texttt{{{label}}}" for label in labels]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _paper_rows(cell: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    best = [row for row in cell["rows"] if row["setting"] == "best-individual"]
    if len(best) != 1:
        raise ValueError(f"IND paper-table cell must have one Best Individual row: {cell['cell']}")
    indexed = {
        (str(row["method"]), str(row["setting"])): row
        for row in cell["rows"]
        if row["setting"] != "best-individual"
    }
    ordered = [best[0]]
    for method in PAPER_RULE_ORDER:
        for setting in ("matched-naive", "ind"):
            try:
                ordered.append(indexed[(method, setting)])
            except KeyError as error:
                raise ValueError(
                    f"IND paper-table row is missing for {cell['cell']}: {method}/{setting}"
                ) from error
    return tuple(ordered)


def _full_tex_text(summary: Mapping[str, Any]) -> str:
    header = " & ".join(_paper_metric_header(column) for column in PAPER_TABLE_COLUMN_ORDER)
    dataset_text = _paper_label_list(
        list(dict.fromkeys(_paper_cell_labels(cell)[0] for cell in summary["cells"]))
    )
    model_text = _paper_label_list(
        list(dict.fromkeys(_paper_cell_labels(cell)[1] for cell in summary["cells"]))
    )
    lines = [
        "% Generated from the three-family q=11 IND and full-source matched-NAIVE summary.",
        "% This is a paper-ready result artifact; paper/Table_IND.tex remains server read-only.",
        r"\begin{landscape}",
        r"\begin{table}[!ht]",
        r"\tiny",
        r"\setlength{\tabcolsep}{0.05cm}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\centering",
        (
            "\\caption{IND setting ("
            + dataset_text
            + " with "
            + model_text
            + r"): performance of the Best "
            r"Individual Explanation, Simple Averaging, Borda, Kemeny--Young, RRF, and "
            r"Schulze under matched NAIVE and IND."
        ),
        (
            r"IND uses $q=11$ disjoint-subset source models, each paired with one assigned "
            r"explanation method; matched NAIVE averages the corresponding 11-method "
            r"ensembles over all eleven source models in each of three partition families."
        ),
        (
            r"All metrics use $p=16$, $k=20$, dataset-mean filling, "
            r"$\sigma_{\mathtt{g}}=0.15$, $\pi_{\mathtt{p}}=0.05$, "
            r"$\sigma_{\mathtt{s}}=0.15$, and $\epsilon=2/255$."
        ),
        (
            r"Robustness is signed direction-aware quality change; positive values denote "
            r"improvement under perturbation and all $\mathtt{R}$ columns are maximized.}"
        ),
        r"\label{tab:ind_single_case}",
        r"\begin{tabular}{|l l||c|c||c|c||c|c|c|c||c|c|c|c||c|c|c|c||c|c|c|c||}",
        r"\hline",
        r"\hline",
        f"Method & Setting & {header} \\\\",
        r"\hline",
        r"\hline",
    ]
    for cell in summary["cells"]:
        rows = _paper_rows(cell)
        lines.extend(
            (
                f"\\multicolumn{{22}}{{c}}{{{_paper_cell_label(cell)}}} \\\\",
                r"\hline",
                f"\\multicolumn{{2}}{{|l||}}{{Best Individual}} & {_tex_values(rows[0])} \\\\",
                r"\hline",
            )
        )
        for index, method in enumerate(PAPER_RULE_ORDER):
            matched = rows[1 + index * 2]
            ind = rows[2 + index * 2]
            label = PAPER_RULE_LABELS[method]
            lines.extend(
                (
                    f"\\multirow{{2}}{{*}}{{{label}}} & matched NAIVE & "
                    f"{_tex_values(matched)} \\\\",
                    f"& IND & {_tex_values(ind)} \\\\",
                    r"\hline",
                )
            )
        lines.append(r"\hline")
    lines.extend((r"\end{tabular}", r"\end{table}", r"\end{landscape}"))
    return "\n".join(lines) + "\n"


def write_ind_table_priority_summary(
    experiment: AssumptionExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    summary = build_ind_table_priority_summary(experiment)
    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else experiment.storage.scratch_root / "summaries" / "ind-table-priority"
    )
    json_path = atomic_write_json(destination / "summary.json", summary)
    csv_path = atomic_write_text(destination / "summary.csv", _csv_text(summary))
    tex_rows_path = atomic_write_text(destination / "table-ind-rows.tex", _tex_text(summary))
    tex_path = atomic_write_text(destination / "table-ind.tex", _full_tex_text(summary))
    return {
        "status": "complete",
        "result": summary["result"],
        "summary_digest": summary["summary_digest"],
        "cells": len(summary["cells"]),
        "rows": sum(len(cell["rows"]) for cell in summary["cells"]),
        "json": str(json_path),
        "csv": str(csv_path),
        "tex_rows": str(tex_rows_path),
        "tex_table": str(tex_path),
    }


__all__ = [
    "TABLE_COLUMN_ORDER",
    "PAPER_TABLE_COLUMN_ORDER",
    "build_ind_table_priority_summary",
    "write_ind_table_priority_summary",
]
