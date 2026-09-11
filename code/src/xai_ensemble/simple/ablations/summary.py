"""Deterministic JSON/CSV summaries for the four NAIVE ablation tables."""

from __future__ import annotations

import csv
import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.phase2.metrics import QUALITY_METRICS

from ..artifacts import ArtifactError, phase2_artifact_root
from ..robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
    perturbed_quality_values,
    signed_robustness_values,
)
from .artifacts import completed_evaluation_manifest, output_store
from .config import NOISE_ORDER, AblationExperiment, RankSourceSpec
from .rank_source import load_rank_manifest

PAPER_RULES = ("simpleavg", "borda", "kemeny", "rrf", "schulze")
QUALITY_DIRECTIONS = {"F": "max", "Fbar": "min", "C": "max", "Cbar": "min"}
NOISE_LABELS = {
    "gaussian": "g",
    "salt_pepper": "p",
    "speckle": "s",
    "adversarial": "a",
}
PATCH_SIZES = (8, 14, 16)
CSV_FORMAT = {
    "schema_version": 2,
    "delimiter": ",",
    "line_terminator": "LF",
}


def _signed_values(
    clean: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    *,
    rule: str,
    context: str,
) -> Mapping[str, float]:
    robustness = perturbed.get("robustness")
    if not isinstance(robustness, Mapping) or rule not in robustness:
        raise ArtifactError(f"{context} is missing robustness for {rule}")
    record = robustness[rule]
    if not isinstance(record, Mapping):
        raise ArtifactError(f"{context} has invalid robustness for {rule}")
    return signed_robustness_values(
        clean["metrics"][rule],
        perturbed["metrics"][rule],
        legacy_absolute=record.get("absolute"),
        context=f"{context}/{rule}",
    )


def _result_manifest(
    experiment: AblationExperiment,
    *,
    condition_id: str,
    k: int,
    fill: str,
) -> Mapping[str, Any]:
    matches = [
        task
        for task in experiment.evaluation_tasks()
        if task.condition.condition_id == condition_id and task.k == k and task.fill == fill
    ]
    if len(matches) > 1:
        table_priority = {"table2-k": 0, "table4-fill": 1, "table5-noise": 2}
        matches.sort(key=lambda task: table_priority[task.table_id])
    if matches:
        manifest = completed_evaluation_manifest(experiment, matches[0])
        if manifest is None:
            raise FileNotFoundError(f"Evaluation is incomplete: {matches[0].task_id}")
        return manifest
    if k != 20 or fill != "dataset_mean":
        raise FileNotFoundError(
            f"No evaluation source for condition={condition_id}, k={k}, fill={fill}"
        )
    condition = experiment.base.condition(condition_id)
    manifest = load_rank_manifest(experiment, experiment.rank_source(condition))
    if not isinstance(manifest.get("metrics"), Mapping):
        raise ArtifactError(f"Immutable Phase 2 source has no metrics: {condition_id}")
    return manifest


def _best_single(
    manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    clean = manifests["clean"]
    clean_metrics = clean["metrics"]
    singles = tuple(name for name in clean_metrics if name.startswith("single__"))
    if not singles:
        raise ArtifactError("No individual methods exist for Best Individual selection")
    quality = {}
    sources: dict[str, Any] = {
        "quality": {},
        "perturbed_quality": {},
        "robustness": {},
    }
    selected = {}
    for metric in QUALITY_METRICS:
        key = (
            max(singles, key=lambda name: float(clean_metrics[name][metric]))
            if QUALITY_DIRECTIONS[metric] == "max"
            else min(singles, key=lambda name: float(clean_metrics[name][metric]))
        )
        quality[metric] = float(clean_metrics[key][metric])
        sources["quality"][metric] = key.removeprefix("single__")
        selected[metric] = key
    perturbed_quality = {}
    robustness = {}
    for noise_type, manifest in manifests.items():
        if noise_type == "clean":
            continue
        perturbed_quality[noise_type] = {}
        robustness[noise_type] = {}
        sources["perturbed_quality"][noise_type] = {}
        sources["robustness"][noise_type] = {}
        for metric in QUALITY_METRICS:
            key = selected[metric]
            perturbed_quality[noise_type][metric] = float(manifest["metrics"][key][metric])
            robustness[noise_type][metric] = _signed_values(
                clean,
                manifest,
                rule=key,
                context=f"Ablation {noise_type}",
            )[metric]
            method = key.removeprefix("single__")
            sources["perturbed_quality"][noise_type][metric] = method
            sources["robustness"][noise_type][metric] = method
    return {
        "quality": quality,
        "perturbed_quality": perturbed_quality,
        "robustness": robustness,
    }, sources


def _paper_rows(manifests: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    clean_metrics = manifests["clean"]["metrics"]
    rows = []
    best, sources = _best_single(manifests)
    rows.append({"method": "best_individual", **best, "selected_sources": sources})
    for rule in PAPER_RULES:
        if rule not in clean_metrics:
            raise ArtifactError(f"Clean result is missing paper rule {rule}")
        perturbed_quality = {}
        robustness = {}
        for noise_type, manifest in manifests.items():
            if noise_type == "clean":
                continue
            perturbed_quality[noise_type] = perturbed_quality_values(
                manifest["metrics"][rule],
                context=f"Ablation {noise_type}/{rule}",
            )
            robustness[noise_type] = _signed_values(
                manifests["clean"],
                manifest,
                rule=rule,
                context=f"Ablation {noise_type}",
            )
        rows.append(
            {
                "method": rule,
                "quality": {
                    metric: float(clean_metrics[rule][metric]) for metric in QUALITY_METRICS
                },
                "perturbed_quality": perturbed_quality,
                "robustness": robustness,
            }
        )
    return rows


def _setting_manifests(
    experiment: AblationExperiment,
    *,
    k: int,
    fill: str,
) -> Mapping[str, Mapping[str, Any]]:
    conditions = experiment.base_conditions()
    result: dict[str, Mapping[str, Any]] = {
        "clean": _result_manifest(
            experiment,
            condition_id=conditions[0].condition_id,
            k=k,
            fill=fill,
        )
    }
    for noise_type, condition in zip(NOISE_ORDER, conditions[1:], strict=True):
        result[noise_type] = _result_manifest(
            experiment,
            condition_id=condition.condition_id,
            k=k,
            fill=fill,
        )
    return result


def _table2(experiment: AblationExperiment) -> Mapping[str, Any]:
    settings = []
    for k in experiment.top_k_values:
        manifests = _setting_manifests(experiment, k=k, fill="dataset_mean")
        settings.append(
            {
                "k": k,
                "source_task_digests": {
                    key: value["task_digest"] for key, value in manifests.items()
                },
                "rows": _paper_rows(manifests),
            }
        )
    return {
        "schema_version": 2,
        "status": "complete",
        "table": "Table_2",
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "patch_size": experiment.patch_size,
        "fill": "dataset_mean",
        "science": _robustness_science(),
        "settings": settings,
    }


def _base_result_manifest(
    experiment: AblationExperiment,
    *,
    condition_id: str,
    patch_size: int,
) -> Mapping[str, Any]:
    condition = experiment.base.condition(condition_id)
    matches = [
        task
        for task in experiment.base.phase2_tasks()
        if task.dataset.dataset_id == experiment.dataset_id
        and task.model.model_id == experiment.model_id
        and task.split == experiment.split
        and task.condition.condition_id == condition_id
        and task.ensemble.ensemble_id == experiment.ensemble_id
        and task.patch_size == patch_size
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one immutable p={patch_size} main result for {condition_id}; "
            f"found {len(matches)}"
        )
    task = matches[0]
    source = RankSourceSpec(
        source_id=task.task_id,
        kind="existing_phase2",
        store="base",
        root=phase2_artifact_root(task),
        digest=task.digest,
        condition=condition,
        patch_size=patch_size,
    )
    manifest = load_rank_manifest(experiment, source)
    if not isinstance(manifest.get("metrics"), Mapping):
        raise ArtifactError(f"Immutable p={patch_size} result has no metrics: {condition_id}")
    return manifest


def _table3(experiment: AblationExperiment) -> Mapping[str, Any]:
    if tuple(experiment.base.phase2.patch_sizes) != PATCH_SIZES:
        raise ValueError(f"Table 3 requires patch sizes {PATCH_SIZES}")
    conditions = experiment.base_conditions()
    settings = []
    for patch_size in PATCH_SIZES:
        manifests = {
            "clean": _base_result_manifest(
                experiment,
                condition_id=conditions[0].condition_id,
                patch_size=patch_size,
            )
        }
        for noise_type, condition in zip(NOISE_ORDER, conditions[1:], strict=True):
            manifests[noise_type] = _base_result_manifest(
                experiment,
                condition_id=condition.condition_id,
                patch_size=patch_size,
            )
        settings.append(
            {
                "p": patch_size,
                "source_task_digests": {
                    key: value["task_digest"] for key, value in manifests.items()
                },
                "rows": _paper_rows(manifests),
            }
        )
    return {
        "schema_version": 2,
        "status": "complete",
        "table": "Table_3",
        "experiment_id": experiment.base.experiment_id,
        "experiment_digest": experiment.base.digest,
        "phase1_experiment_digest": experiment.base.phase1_digest,
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "k": 20,
        "fill": "dataset_mean",
        "science": _robustness_science(),
        "settings": settings,
    }


def _table4(experiment: AblationExperiment) -> Mapping[str, Any]:
    settings = []
    for fill in experiment.fill_values:
        manifests = _setting_manifests(experiment, k=20, fill=fill)
        settings.append(
            {
                "fill": fill,
                "source_task_digests": {
                    key: value["task_digest"] for key, value in manifests.items()
                },
                "rows": _paper_rows(manifests),
            }
        )
    return {
        "schema_version": 2,
        "status": "complete",
        "table": "Table_4",
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "patch_size": experiment.patch_size,
        "k": 20,
        "science": _robustness_science(),
        "settings": settings,
    }


def _table5(experiment: AblationExperiment) -> Mapping[str, Any]:
    clean = experiment.base_conditions()[0]
    clean_manifest = _result_manifest(
        experiment,
        condition_id=clean.condition_id,
        k=20,
        fill="dataset_mean",
    )
    clean_metrics = clean_manifest["metrics"]
    singles = tuple(name for name in clean_metrics if name.startswith("single__"))
    selected = {
        metric: (
            max(singles, key=lambda name: float(clean_metrics[name][metric]))
            if QUALITY_DIRECTIONS[metric] == "max"
            else min(singles, key=lambda name: float(clean_metrics[name][metric]))
        )
        for metric in QUALITY_METRICS
    }
    noise_sections = []
    for noise_type in NOISE_ORDER:
        levels = []
        for level in experiment.noise_levels[noise_type]:
            manifest = _result_manifest(
                experiment,
                condition_id=level.condition_id,
                k=20,
                fill="dataset_mean",
            )
            rows = []
            best_quality = {}
            best_perturbed = {}
            best_values = {}
            best_sources = {}
            for metric in QUALITY_METRICS:
                source = selected[metric]
                best_quality[metric] = float(clean_metrics[source][metric])
                best_perturbed[metric] = float(manifest["metrics"][source][metric])
                best_values[metric] = _signed_values(
                    clean_manifest,
                    manifest,
                    rule=source,
                    context=f"Table 5 {level.condition_id}",
                )[metric]
                best_sources[metric] = source.removeprefix("single__")
            rows.append(
                {
                    "method": "best_individual",
                    "quality": best_quality,
                    "perturbed_quality": best_perturbed,
                    "robustness": best_values,
                    "selected_sources": best_sources,
                }
            )
            for rule in PAPER_RULES:
                rows.append(
                    {
                        "method": rule,
                        "quality": {
                            metric: float(clean_metrics[rule][metric]) for metric in QUALITY_METRICS
                        },
                        "perturbed_quality": perturbed_quality_values(
                            manifest["metrics"][rule],
                            context=f"Table 5 {level.condition_id}/{rule}",
                        ),
                        "robustness": _signed_values(
                            clean_manifest,
                            manifest,
                            rule=rule,
                            context=f"Table 5 {level.condition_id}",
                        ),
                    }
                )
            levels.append(
                {
                    "severity": level.severity,
                    "condition": level.condition_id,
                    "center_reused": level.center,
                    "source_task_digest": manifest["task_digest"],
                    "rows": rows,
                }
            )
        noise_sections.append({"noise": noise_type, "levels": levels})
    return {
        "schema_version": 2,
        "status": "complete",
        "table": "Table_5",
        "ablation_id": experiment.ablation_id,
        "ablation_digest": experiment.digest,
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "patch_size": experiment.patch_size,
        "k": 20,
        "fill": "dataset_mean",
        "science": _robustness_science(),
        "best_individual_sources": {
            metric: source.removeprefix("single__") for metric, source in selected.items()
        },
        "noise": noise_sections,
    }


def _robustness_science() -> Mapping[str, Any]:
    return {
        "best_individual_policy": "clean_metric_anchored",
        "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
        "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
        "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
        "legacy_absolute_R_used": False,
        "perturbed_quality_reported": True,
    }


def _wide_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    output = []
    setting_names = {"Table_2": "k", "Table_3": "p", "Table_4": "fill"}
    if summary["table"] in setting_names:
        setting_name = setting_names[summary["table"]]
        for setting in summary["settings"]:
            for row in setting["rows"]:
                value = {setting_name: setting[setting_name], "method": row["method"]}
                value.update(row["quality"])
                for noise_type, metrics in row["perturbed_quality"].items():
                    label = NOISE_LABELS[noise_type]
                    value.update(
                        {
                            f"{metric}_perturbed_{label}": number
                            for metric, number in metrics.items()
                        }
                    )
                for noise_type, metrics in row["robustness"].items():
                    label = NOISE_LABELS[noise_type]
                    value.update(
                        {f"R_{label}_{metric}": number for metric, number in metrics.items()}
                    )
                output.append(value)
        return tuple(output)
    for section in summary["noise"]:
        for level in section["levels"]:
            for row in level["rows"]:
                output.append(
                    {
                        "noise": section["noise"],
                        "severity": level["severity"],
                        "method": row["method"],
                        **{f"{metric}_clean": value for metric, value in row["quality"].items()},
                        **{
                            f"{metric}_perturbed": value
                            for metric, value in row["perturbed_quality"].items()
                        },
                        **{f"R_{metric}": value for metric, value in row["robustness"].items()},
                    }
                )
    return tuple(output)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty summary CSV")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(experiment: AblationExperiment) -> Mapping[str, Mapping[str, str]]:
    store = output_store(experiment)
    summaries = (
        _table2(experiment),
        _table3(experiment),
        _table4(experiment),
        _table5(experiment),
    )
    result = {}
    for raw_summary in summaries:
        summary = {**raw_summary, "csv_format": CSV_FORMAT}
        table = str(summary["table"])
        digest = object_sha256(summary)
        root = posixpath.join("summaries", "naive", table, digest)
        scratch = experiment.output_storage.scratch_root / "summaries" / table / digest
        json_path = scratch / "summary.json"
        csv_path = scratch / "summary.csv"
        atomic_write_json(json_path, summary)
        _write_csv(csv_path, _wide_rows(summary))
        json_file = store.publish(json_path, posixpath.join(root, "summary.json"))
        csv_file = store.publish(csv_path, posixpath.join(root, "summary.csv"))
        result[table] = {
            "digest": digest,
            "json": store.locator(json_file.relative_path),
            "csv": store.locator(csv_file.relative_path),
            "local_json": str(json_path),
            "local_csv": str(csv_path),
        }
    return result


__all__ = ["write_summaries"]
