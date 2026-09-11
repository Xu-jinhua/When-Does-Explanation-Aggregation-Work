"""Deterministic, table-ready summaries for completed simple experiments."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.metrics import (
    DEFAULT_METRIC_DIRECTIONS,
    QUALITY_METRICS,
    oracle_best_single,
)

from .artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    phase2_artifact_root,
)
from .config import ConditionConfig, Phase2Task, SimpleExperiment
from .robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
    perturbed_quality_values,
    signed_robustness_values,
)

ManifestSource = Literal["auto", "local", "remote"]

PAPER_RULES = ("simpleavg", "borda", "kemeny", "rrf", "schulze")
PAPER_METHODS = ("best_individual", *PAPER_RULES)
METHOD_LABELS = {
    "best_individual": "Best Individual",
    "simpleavg": "Simple Averaging",
    "borda": "Borda",
    "kemeny": "Kemeny",
    "rrf": "RRF",
    "schulze": "Schulze",
}
DATASET_LABELS = {
    "imagenet100": "ImageNet100",
    "imagenet1k": "ImageNet-1K",
    "dermamnist": "DermaMNIST",
    "pathmnist": "PathMNIST",
}
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vit_base_patch16_224": "ViT-B/16",
    "densenet121": "DenseNet-121",
}
NOISE_ORDER = ("g", "p", "s", "a")
NOISE_NAMES = {
    "g": "gaussian",
    "p": "salt_pepper",
    "s": "speckle",
    "a": "adversarial",
}
PERTURBED_QUALITY_COLUMNS = tuple(
    f"{metric}_perturbed_{noise}" for noise in NOISE_ORDER for metric in QUALITY_METRICS
)


@dataclass(frozen=True, slots=True)
class NoiseSetting:
    key: str
    name: str
    condition_id: str
    parameter: str
    value: float
    display_value: str


def _noise_setting(condition: ConditionConfig) -> NoiseSetting | None:
    if condition.kind == "clean":
        return None
    if condition.kind == "adversarial":
        epsilon = float(condition.kwargs["epsilon"])
        numerator = round(epsilon * 255.0)
        display = (
            f"{numerator}/255"
            if math.isclose(epsilon, numerator / 255.0, abs_tol=1e-12)
            else f"{epsilon:.12g}"
        )
        return NoiseSetting(
            key="a",
            name="adversarial",
            condition_id=condition.condition_id,
            parameter="epsilon",
            value=epsilon,
            display_value=display,
        )
    kind = str(condition.kwargs.get("kind"))
    definitions = {
        "gaussian": ("g", "sigma_g"),
        "salt_pepper": ("p", "pi_p"),
        "speckle": ("s", "sigma_s"),
    }
    if kind not in definitions:
        raise ValueError(f"Unsupported Table 1 perturbation kind {kind!r}")
    key, parameter = definitions[kind]
    severity = float(condition.kwargs["severity"])
    return NoiseSetting(
        key=key,
        name=kind,
        condition_id=condition.condition_id,
        parameter=parameter,
        value=severity,
        display_value=f"{severity:.12g}",
    )


def _noise_settings(experiment: SimpleExperiment) -> Mapping[str, NoiseSetting]:
    settings = tuple(
        setting
        for condition in experiment.conditions
        if (setting := _noise_setting(condition)) is not None
    )
    by_key = {setting.key: setting for setting in settings}
    if tuple(by_key) != NOISE_ORDER or len(settings) != len(NOISE_ORDER):
        raise ValueError(
            "Table 1 requires exactly one Gaussian, salt-and-pepper, speckle, "
            "and adversarial condition in that order"
        )
    return by_key


def _table1_task_groups(
    experiment: SimpleExperiment,
) -> tuple[tuple[Any, str, Mapping[str, Phase2Task]], ...]:
    if experiment.phase2.primary_patch_size != 16 or experiment.phase2.k != 20:
        raise ValueError("Table 1 is fixed to p=16 and k=20")
    ensembles = tuple(item for item in experiment.phase2.ensembles if item.include_singles)
    if len(ensembles) != 1:
        raise ValueError("Table 1 requires exactly one ensemble with individual methods")
    ensemble = ensembles[0]
    clean = next(condition for condition in experiment.conditions if condition.kind == "clean")
    settings = _noise_settings(experiment)
    condition_by_key = {
        "clean": clean.condition_id,
        **{key: value.condition_id for key, value in settings.items()},
    }
    tasks = tuple(
        task
        for task in experiment.phase2_tasks()
        if task.patch_size == experiment.phase2.primary_patch_size
        and task.ensemble.ensemble_id == ensemble.ensemble_id
    )
    result = []
    for model in experiment.models:
        dataset = experiment.dataset(model.dataset_id)
        for split in dataset.splits:
            group = {}
            for key, condition_id in condition_by_key.items():
                matches = tuple(
                    task
                    for task in tasks
                    if task.dataset.dataset_id == dataset.dataset_id
                    and task.model.model_id == model.model_id
                    and task.split == split
                    and task.condition.condition_id == condition_id
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one Table 1 task for {model.model_id}/{split}/{condition_id}; "
                        f"found {len(matches)}"
                    )
                group[key] = matches[0]
            result.append((model, split, group))
    return tuple(result)


def _local_manifest_path(experiment: SimpleExperiment, task: Phase2Task) -> Path:
    return experiment.storage.scratch_root / "phase2" / task.task_id / "manifest.json"


def _validate_manifest(
    experiment: SimpleExperiment,
    task: Phase2Task,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": experiment.experiment_id,
        "phase1_experiment_digest": experiment.phase1_digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "ensemble": task.ensemble.ensemble_id,
        "patch_size": task.patch_size,
        "k": experiment.phase2.k,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Table 1 manifest identity mismatch: {mismatches}")
    artifact_experiment_digest = manifest.get("experiment_digest")
    if (
        not isinstance(artifact_experiment_digest, str)
        or len(artifact_experiment_digest) != 64
        or any(character not in "0123456789abcdef" for character in artifact_experiment_digest)
    ):
        raise ArtifactError("Table 1 manifest has an invalid experiment_digest")
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ArtifactError("Table 1 manifest has an invalid sample_count")
    methods = manifest.get("methods")
    metrics = manifest.get("metrics")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
        raise ArtifactError("Table 1 manifest has no method roster")
    if not isinstance(metrics, Mapping):
        raise ArtifactError("Table 1 manifest has no metrics")
    from .phase2 import _method_names

    expected_methods = _method_names(experiment, task)
    if tuple(str(method) for method in methods) != expected_methods:
        raise ArtifactError("Table 1 manifest method roster differs from the current Phase 2 task")
    expected_rules = {*PAPER_RULES, *(f"single__{method}" for method in methods)}
    if set(metrics) != expected_rules:
        raise ArtifactError("Table 1 manifest rule roster is incomplete or contradictory")
    for rule, values in metrics.items():
        if not isinstance(values, Mapping) or set(values) != set(QUALITY_METRICS):
            raise ArtifactError(f"Invalid quality metric fields for {rule}")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ArtifactError(f"Non-finite quality metric for {rule}")
    return {
        "artifact_experiment_digest": artifact_experiment_digest,
        "reused_from_prior_experiment": artifact_experiment_digest != experiment.digest,
    }


def _load_manifest(
    experiment: SimpleExperiment,
    task: Phase2Task,
    *,
    source: ManifestSource,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    local_path = _local_manifest_path(experiment, task)
    if source in {"auto", "local"} and local_path.is_file():
        value = read_json(local_path)
        if not isinstance(value, Mapping):
            raise ArtifactError(f"Local Phase 2 manifest is not a mapping: {local_path}")
        provenance = _validate_manifest(experiment, task, value)
        return value, {
            "source": "local",
            "path": str(local_path),
            "sha256": file_sha256(local_path),
            **provenance,
        }
    if source == "local":
        raise FileNotFoundError(f"Missing local Phase 2 manifest: {local_path}")
    store = ArtifactStore(experiment)
    root = phase2_artifact_root(task)
    value = completed_manifest(
        store,
        root,
        expected_task_digest=task.digest,
        expected_schema_version=PHASE2_SCHEMA_VERSION,
    )
    if value is None:
        raise FileNotFoundError(f"Missing remote Phase 2 manifest: {root}")
    provenance = _validate_manifest(experiment, task, value)
    return value, {
        "source": "remote",
        "root": root,
        "locator": store.locator(f"{root}/manifest.json"),
        "content_digest": object_sha256(value),
        **provenance,
    }


def _metric_values(values: Mapping[str, Any]) -> Mapping[str, float]:
    return {metric: float(values[metric]) for metric in QUALITY_METRICS}


def _table_rows(manifests: Mapping[str, Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    clean = manifests["clean"]
    clean_metrics = clean["metrics"]
    methods = tuple(str(value) for value in clean["methods"])
    expected_rules = set(clean_metrics)
    sample_count = int(clean["sample_count"])
    for noise in NOISE_ORDER:
        manifest = manifests[noise]
        if (
            int(manifest["sample_count"]) != sample_count
            or tuple(manifest["methods"]) != methods
            or set(manifest["metrics"]) != expected_rules
        ):
            raise ArtifactError(f"Table 1 {noise} manifest is not aligned with clean")
        robustness = manifest.get("robustness")
        if not isinstance(robustness, Mapping) or set(robustness) != expected_rules:
            raise ArtifactError(f"Table 1 {noise} manifest lacks aligned robustness")
        for rule in expected_rules:
            record = robustness[rule]
            if not isinstance(record, Mapping):
                raise ArtifactError(f"Table 1 {noise}/{rule} robustness is invalid")
            signed_robustness_values(
                clean_metrics[rule],
                manifest["metrics"][rule],
                legacy_absolute=record.get("absolute"),
                context=f"Table 1 {noise}/{rule}",
            )

    quality_choices = oracle_best_single(
        {method: _metric_values(clean_metrics[f"single__{method}"]) for method in methods}
    )
    selected_methods = {metric: quality_choices[metric].method for metric in QUALITY_METRICS}

    best = {
        "method": "best_individual",
        "quality": {metric: float(quality_choices[metric].value) for metric in QUALITY_METRICS},
        "perturbed_quality": {
            noise: {
                metric: float(
                    manifests[noise]["metrics"][f"single__{selected_methods[metric]}"][metric]
                )
                for metric in QUALITY_METRICS
            }
            for noise in NOISE_ORDER
        },
        "robustness": {
            noise: {
                metric: signed_robustness_values(
                    clean_metrics[f"single__{selected_methods[metric]}"],
                    manifests[noise]["metrics"][f"single__{selected_methods[metric]}"],
                    context=f"Table 1 best individual {noise}/{selected_methods[metric]}",
                )[metric]
                for metric in QUALITY_METRICS
            }
            for noise in NOISE_ORDER
        },
        "selected_sources": {
            "quality": dict(selected_methods),
            "perturbed_quality": {noise: dict(selected_methods) for noise in NOISE_ORDER},
            "robustness": {noise: dict(selected_methods) for noise in NOISE_ORDER},
        },
    }
    rows = [best]
    for rule in PAPER_RULES:
        perturbed_quality = {
            noise: perturbed_quality_values(
                manifests[noise]["metrics"][rule],
                context=f"Table 1 {noise}/{rule}",
            )
            for noise in NOISE_ORDER
        }
        rows.append(
            {
                "method": rule,
                "quality": _metric_values(clean_metrics[rule]),
                "perturbed_quality": perturbed_quality,
                "robustness": {
                    noise: signed_robustness_values(
                        clean_metrics[rule],
                        manifests[noise]["metrics"][rule],
                        context=f"Table 1 {noise}/{rule}",
                    )
                    for noise in NOISE_ORDER
                },
            }
        )
    return tuple(rows)


def _columns() -> tuple[tuple[str, str], ...]:
    quality = tuple((metric, DEFAULT_METRIC_DIRECTIONS[metric]) for metric in QUALITY_METRICS)
    robustness = tuple(
        (f"R_{metric}_{noise}", SIGNED_ROBUSTNESS_DIRECTION)
        for noise in NOISE_ORDER
        for metric in QUALITY_METRICS
    )
    return (*quality, *robustness)


def _flat_row(row: Mapping[str, Any]) -> Mapping[str, float]:
    return {
        **{metric: float(row["quality"][metric]) for metric in QUALITY_METRICS},
        **{
            f"R_{metric}_{noise}": float(row["robustness"][noise][metric])
            for noise in NOISE_ORDER
            for metric in QUALITY_METRICS
        },
    }


def _flat_perturbed_quality(row: Mapping[str, Any]) -> Mapping[str, float]:
    return {
        f"{metric}_perturbed_{noise}": float(row["perturbed_quality"][noise][metric])
        for noise in NOISE_ORDER
        for metric in QUALITY_METRICS
    }


def _outcome(value: float, reference: float, direction: str) -> str:
    benefit = value - reference if direction == "max" else reference - value
    if benefit > 1e-12:
        return "better"
    if benefit < -1e-12:
        return "worse"
    return "equal"


def _column_groups() -> Mapping[str, tuple[tuple[str, str], ...]]:
    columns = _columns()
    quality_count = len(QUALITY_METRICS)
    robustness = columns[quality_count:]
    result: dict[str, tuple[tuple[str, str], ...]] = {
        "quality": columns[:quality_count],
        "robustness_all": robustness,
    }
    for index, noise in enumerate(NOISE_ORDER):
        start = index * quality_count
        result[NOISE_NAMES[noise]] = robustness[start : start + quality_count]
    return result


def _comparison_analysis(
    cells: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    include_column_winners: bool = False,
) -> Mapping[str, Any]:
    winner_counts = {method: 0 for method in PAPER_METHODS}
    column_winners = []
    versus_best = {method: {"better": 0, "equal": 0, "worse": 0} for method in PAPER_RULES}
    versus_simpleavg = {
        method: {"better": 0, "equal": 0, "worse": 0}
        for method in ("borda", "kemeny", "rrf", "schulze")
    }
    for cell in cells:
        by_method = {row["method"]: _flat_row(row) for row in cell["rows"]}
        for column, direction in columns:
            values = {method: row[column] for method, row in by_method.items()}
            best_value = max(values.values()) if direction == "max" else min(values.values())
            winners = tuple(
                method
                for method in PAPER_METHODS
                if math.isclose(values[method], best_value, rel_tol=0.0, abs_tol=1e-12)
            )
            for method in winners:
                winner_counts[method] += 1
            if include_column_winners:
                column_winners.append(
                    {
                        "dataset": cell["dataset"],
                        "model": cell["model"],
                        "column": column,
                        "direction": direction,
                        "value": best_value,
                        "methods": list(winners),
                    }
                )
            for method in PAPER_RULES:
                versus_best[method][
                    _outcome(values[method], values["best_individual"], direction)
                ] += 1
            for method in versus_simpleavg:
                versus_simpleavg[method][
                    _outcome(values[method], values["simpleavg"], direction)
                ] += 1
    result: dict[str, Any] = {
        "comparison_cell_count": len(cells) * len(columns),
        "winner_counts": winner_counts,
        "versus_best_individual": versus_best,
        "rank_rules_versus_simpleavg": versus_simpleavg,
    }
    if include_column_winners:
        result["column_winners"] = column_winners
    return result


def _increment_count(counts: dict[str, int], method: str) -> None:
    counts[method] = counts.get(method, 0) + 1


def _oracle_source_analysis(cells: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    overall: dict[str, int] = {}
    quality: dict[str, int] = {}
    robustness: dict[str, int] = {}
    by_noise: dict[str, dict[str, int]] = {NOISE_NAMES[key]: {} for key in NOISE_ORDER}
    for cell in cells:
        best = next(row for row in cell["rows"] if row["method"] == "best_individual")
        selected = best["selected_sources"]
        for method in selected["quality"].values():
            _increment_count(overall, method)
            _increment_count(quality, method)
        for noise in NOISE_ORDER:
            noise_counts = by_noise[NOISE_NAMES[noise]]
            for method in selected["robustness"][noise].values():
                _increment_count(overall, method)
                _increment_count(robustness, method)
                _increment_count(noise_counts, method)
    return {
        "selection_count": sum(overall.values()),
        "overall": dict(sorted(overall.items())),
        "quality": dict(sorted(quality.items())),
        "robustness_all": dict(sorted(robustness.items())),
        "by_noise": {noise: dict(sorted(counts.items())) for noise, counts in by_noise.items()},
    }


def _analysis(cells: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    groups = _column_groups()
    overall = _comparison_analysis(cells, _columns(), include_column_winners=True)
    return {
        **overall,
        "groups": {
            group: _comparison_analysis(cells, columns) for group, columns in groups.items()
        },
        "by_dataset_model": [
            {
                "dataset": cell["dataset"],
                "model": cell["model"],
                "sample_count": cell["sample_count"],
                "all": _comparison_analysis((cell,), _columns()),
                "groups": {
                    group: _comparison_analysis((cell,), columns)
                    for group, columns in groups.items()
                },
            }
            for cell in cells
        ],
        "oracle_source_counts": _oracle_source_analysis(cells),
    }


def build_table1_summary(
    experiment: SimpleExperiment,
    *,
    manifest_source: ManifestSource = "auto",
) -> Mapping[str, Any]:
    if manifest_source not in {"auto", "local", "remote"}:
        raise ValueError(f"Unsupported manifest source {manifest_source!r}")
    settings = _noise_settings(experiment)
    cells = []
    for model, split, tasks in _table1_task_groups(experiment):
        manifests = {}
        sources = {}
        for condition_key, task in tasks.items():
            manifests[condition_key], sources[condition_key] = _load_manifest(
                experiment,
                task,
                source=manifest_source,
            )
        dataset_id = model.dataset_id
        cells.append(
            {
                "dataset": dataset_id,
                "dataset_label": DATASET_LABELS.get(dataset_id, dataset_id),
                "model": model.model_id,
                "model_key": model.model_key,
                "model_label": MODEL_LABELS.get(model.model_key, model.model_id),
                "architecture": model.architecture,
                "split": split,
                "sample_count": int(manifests["clean"]["sample_count"]),
                "methods": list(manifests["clean"]["methods"]),
                "source_tasks": {
                    key: {
                        "task_id": tasks[key].task_id,
                        "task_digest": tasks[key].digest,
                        "artifact_root": phase2_artifact_root(tasks[key]),
                        "manifest": sources[key],
                    }
                    for key in ("clean", *NOISE_ORDER)
                },
                "rows": list(_table_rows(manifests)),
            }
        )
    summary: Mapping[str, Any] = {
        "schema_version": 2,
        "status": "complete",
        "table": "Table_1",
        "experiment_id": experiment.experiment_id,
        "experiment_digest": experiment.digest,
        "phase1_experiment_digest": experiment.phase1_digest,
        "settings": {
            "patch_size": experiment.phase2.primary_patch_size,
            "k": experiment.phase2.k,
            "fill": "dataset_mean",
            "noise": {key: asdict(settings[key]) for key in NOISE_ORDER},
            "best_individual_policy": "clean_metric_anchored",
            "robustness_policy": SIGNED_ROBUSTNESS_POLICY,
            "robustness_direction": SIGNED_ROBUSTNESS_DIRECTION,
            "robustness_source": SIGNED_ROBUSTNESS_SOURCE,
            "legacy_absolute_R_used": False,
            "perturbed_quality_reported": True,
        },
        "columns": [name for name, _ in _columns()],
        "column_directions": dict(_columns()),
        "perturbed_quality_columns": list(PERTURBED_QUALITY_COLUMNS),
        "cells": cells,
        "analysis": _analysis(cells),
    }
    return summary


def _csv_text(summary: Mapping[str, Any]) -> str:
    fields = (
        "dataset",
        "model",
        "architecture",
        "split",
        "sample_count",
        "method",
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
                    "dataset": cell["dataset"],
                    "model": cell["model"],
                    "architecture": cell["architecture"],
                    "split": cell["split"],
                    "sample_count": cell["sample_count"],
                    "method": row["method"],
                    **_flat_row(row),
                    **_flat_perturbed_quality(row),
                    "selected_sources": "" if selected is None else object_sha256(selected),
                }
            )
    return output.getvalue()


def _analysis_csv_text(summary: Mapping[str, Any]) -> str:
    fields = (
        "scope",
        "dataset",
        "model",
        "group",
        "comparison_cells",
        "method",
        "wins",
        "vs_best_better",
        "vs_best_equal",
        "vs_best_worse",
        "vs_simpleavg_better",
        "vs_simpleavg_equal",
        "vs_simpleavg_worse",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def write_scope(
        analysis: Mapping[str, Any],
        *,
        scope: str,
        dataset: str = "",
        model: str = "",
    ) -> None:
        scopes = (("all", analysis), *analysis.get("groups", {}).items())
        for group, values in scopes:
            for method in PAPER_METHODS:
                versus_best = values["versus_best_individual"].get(method, {})
                versus_simpleavg = values["rank_rules_versus_simpleavg"].get(method, {})
                writer.writerow(
                    {
                        "scope": scope,
                        "dataset": dataset,
                        "model": model,
                        "group": group,
                        "comparison_cells": values["comparison_cell_count"],
                        "method": method,
                        "wins": values["winner_counts"][method],
                        "vs_best_better": versus_best.get("better", ""),
                        "vs_best_equal": versus_best.get("equal", ""),
                        "vs_best_worse": versus_best.get("worse", ""),
                        "vs_simpleavg_better": versus_simpleavg.get("better", ""),
                        "vs_simpleavg_equal": versus_simpleavg.get("equal", ""),
                        "vs_simpleavg_worse": versus_simpleavg.get("worse", ""),
                    }
                )

    analysis = summary["analysis"]
    write_scope(analysis, scope="overall")
    for cell in analysis["by_dataset_model"]:
        write_scope(
            {**cell["all"], "groups": cell["groups"]},
            scope="dataset_model",
            dataset=cell["dataset"],
            model=cell["model"],
        )
    return output.getvalue()


def _tex_value(value: float, *, bold: bool) -> str:
    formatted = f"{value:.3f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def _tex_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "% Generated from immutable simple Phase 2 manifests.",
        "% Adversarial robustness uses the configured 2/255 condition.",
    ]
    columns = _columns()
    for cell in summary["cells"]:
        by_method = {row["method"]: _flat_row(row) for row in cell["rows"]}
        displayed_best = {}
        for column, direction in columns:
            values = {method: float(f"{row[column]:.3f}") for method, row in by_method.items()}
            displayed_best[column] = (
                max(values.values()) if direction == "max" else min(values.values())
            )
        lines.extend(
            (
                "\\multicolumn{21}{c}{"
                f"\\texttt{{{cell['dataset_label']}}}, "
                f"\\texttt{{{cell['model_label']}}}"
                "}\\\\",
                "\\hline",
            )
        )
        for method in PAPER_METHODS:
            values = by_method[method]
            rendered = [
                _tex_value(
                    values[column],
                    bold=math.isclose(
                        float(f"{values[column]:.3f}"),
                        displayed_best[column],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                )
                for column, _ in columns
            ]
            lines.append(f"{METHOD_LABELS[method]} & " + " & ".join(rendered) + " \\\\")
            if method in {"best_individual", "simpleavg"}:
                lines.append("\\hline")
        lines.extend(("\\hline", "\\hline"))
    return "\n".join(lines) + "\n"


def write_table1_summary(
    experiment: SimpleExperiment,
    *,
    manifest_source: ManifestSource = "auto",
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    summary = build_table1_summary(experiment, manifest_source=manifest_source)
    digest = object_sha256(summary)
    root = (
        experiment.storage.scratch_root / "summaries" / "table1"
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    destination = root / digest
    value = {**summary, "summary_digest": digest}
    json_path = atomic_write_json(destination / "summary.json", value)
    csv_path = atomic_write_text(destination / "table1.csv", _csv_text(summary))
    analysis_csv_path = atomic_write_text(
        destination / "table1_analysis.csv", _analysis_csv_text(summary)
    )
    tex_path = atomic_write_text(destination / "table1_rows.tex", _tex_text(summary))
    analysis = summary["analysis"]
    return {
        "status": "complete",
        "table": "Table_1",
        "summary_digest": digest,
        "cells": len(summary["cells"]),
        "rows": sum(len(cell["rows"]) for cell in summary["cells"]),
        "comparison_cells": summary["analysis"]["comparison_cell_count"],
        "output_directory": str(destination),
        "summary_json": str(json_path),
        "table_csv": str(csv_path),
        "analysis_csv": str(analysis_csv_path),
        "latex_rows": str(tex_path),
        "analysis": {
            "winner_counts": analysis["winner_counts"],
            "versus_best_individual": analysis["versus_best_individual"],
            "rank_rules_versus_simpleavg": analysis["rank_rules_versus_simpleavg"],
            "group_winner_counts": {
                group: values["winner_counts"] for group, values in analysis["groups"].items()
            },
        },
    }


def _check_metrics_manifest(
    experiment: SimpleExperiment,
    task: Phase2Task,
    manifest: Mapping[str, Any],
) -> None:
    """Generic Phase 2 manifest identity/metric contract (not Table 1 specific)."""

    expected = {
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "ensemble": task.ensemble.ensemble_id,
        "patch_size": task.patch_size,
        "k": experiment.phase2.k,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Phase 2 manifest identity mismatch: {mismatches}")
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ArtifactError("Phase 2 manifest has an invalid sample_count")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ArtifactError("Phase 2 manifest has no metrics")
    for rule, values in metrics.items():
        if not isinstance(values, Mapping) or not set(values) >= set(QUALITY_METRICS):
            raise ArtifactError(f"Invalid quality metric fields for {rule}")
        if any(not math.isfinite(float(values[metric])) for metric in QUALITY_METRICS):
            raise ArtifactError(f"Non-finite quality metric for {rule}")


def _load_metrics_manifest(
    experiment: SimpleExperiment,
    task: Phase2Task,
    *,
    source: ManifestSource,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    local_path = _local_manifest_path(experiment, task)
    value: Mapping[str, Any] | None = None
    origin: Mapping[str, Any] | None = None
    if source in {"auto", "local"} and local_path.is_file():
        raw = read_json(local_path)
        if not isinstance(raw, Mapping):
            raise ArtifactError(f"Local Phase 2 manifest is not a mapping: {local_path}")
        value = raw
        origin = {
            "source": "local",
            "path": str(local_path),
            "sha256": file_sha256(local_path),
        }
    elif source == "local":
        raise FileNotFoundError(f"Missing local Phase 2 manifest: {local_path}")
    if value is None:
        store = ArtifactStore(experiment)
        root = phase2_artifact_root(task)
        value = completed_manifest(
            store,
            root,
            expected_task_digest=task.digest,
            expected_schema_version=PHASE2_SCHEMA_VERSION,
        )
        if value is None:
            raise FileNotFoundError(f"Missing remote Phase 2 manifest: {root}")
        origin = {
            "source": "remote",
            "root": root,
            "locator": store.locator(f"{root}/manifest.json"),
            "content_digest": object_sha256(value),
        }
    _check_metrics_manifest(experiment, task, value)
    return value, origin


def _metrics_member_order(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    rules = [rule for rule in PAPER_RULES if rule in metrics]
    others = sorted(
        key for key in metrics if key not in PAPER_RULES and not str(key).startswith("single__")
    )
    singles = sorted(key for key in metrics if str(key).startswith("single__"))
    return (*rules, *others, *singles)


def build_metrics_summary(
    experiment: SimpleExperiment,
    *,
    manifest_source: ManifestSource = "auto",
) -> Mapping[str, Any]:
    """Condition-agnostic metric table over every completed Phase 2 task.

    Unlike Table 1 this mode accepts any condition set, so it also works for
    small demonstration configurations such as ``quickstart.yaml``.
    """

    if manifest_source not in {"auto", "local", "remote"}:
        raise ValueError(f"Unsupported manifest source {manifest_source!r}")
    rows: list[Mapping[str, Any]] = []
    sources: list[Mapping[str, Any]] = []
    for task in experiment.phase2_tasks():
        manifest, origin = _load_metrics_manifest(experiment, task, source=manifest_source)
        metrics = manifest["metrics"]
        sample_count = int(manifest["sample_count"])
        for member in _metrics_member_order(metrics):
            values = metrics[member]
            rows.append(
                {
                    "dataset": task.dataset.dataset_id,
                    "dataset_label": DATASET_LABELS.get(
                        task.dataset.dataset_id, task.dataset.dataset_id
                    ),
                    "model": task.model.model_id,
                    "model_key": task.model.model_key,
                    "model_label": MODEL_LABELS.get(task.model.model_key, task.model.model_id),
                    "architecture": task.model.architecture,
                    "split": task.split,
                    "condition": task.condition.condition_id,
                    "ensemble": task.ensemble.ensemble_id,
                    "patch_size": task.patch_size,
                    "k": experiment.phase2.k,
                    "member": member,
                    "member_kind": "single" if member.startswith("single__") else "aggregation",
                    "sample_count": sample_count,
                    "metrics": {metric: float(values[metric]) for metric in QUALITY_METRICS},
                    "task_id": task.task_id,
                    "task_digest": task.digest,
                }
            )
        sources.append(
            {
                "task_id": task.task_id,
                "task_digest": task.digest,
                "artifact_root": phase2_artifact_root(task),
                "manifest": origin,
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "table": "metrics",
        "experiment_id": experiment.experiment_id,
        "experiment_digest": experiment.digest,
        "phase1_experiment_digest": experiment.phase1_digest,
        "metrics": list(QUALITY_METRICS),
        "row_count": len(rows),
        "rows": rows,
        "source_tasks": sources,
    }


def _metrics_csv_text(summary: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    metrics = summary["metrics"]
    writer.writerow(
        (
            "dataset",
            "model",
            "architecture",
            "split",
            "condition",
            "ensemble",
            "patch_size",
            "k",
            "member",
            "member_kind",
            "sample_count",
            *metrics,
            "task_digest",
        )
    )
    for row in summary["rows"]:
        writer.writerow(
            (
                row["dataset"],
                row["model"],
                row["architecture"],
                row["split"],
                row["condition"],
                row["ensemble"],
                row["patch_size"],
                row["k"],
                row["member"],
                row["member_kind"],
                row["sample_count"],
                *(f"{row['metrics'][metric]:.12g}" for metric in metrics),
                row["task_digest"],
            )
        )
    return buffer.getvalue()


def _metrics_tex_text(summary: Mapping[str, Any]) -> str:
    def esc(value: Any) -> str:
        return str(value).replace("_", r"\_")

    lines = []
    for row in summary["rows"]:
        cells = " & ".join(f"{row['metrics'][metric]:.4f}" for metric in summary["metrics"])
        lines.append(
            f"{esc(row['dataset_label'])} & {esc(row['model_label'])} & "
            f"{esc(row['condition'])} & {esc(row['member'])} & {cells} \\\\"
        )
    return "\n".join(lines) + "\n"


def write_metrics_summary(
    experiment: SimpleExperiment,
    *,
    manifest_source: ManifestSource = "auto",
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    summary = build_metrics_summary(experiment, manifest_source=manifest_source)
    digest = object_sha256(summary)
    root = (
        experiment.storage.scratch_root / "summaries" / "metrics"
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    destination = root / digest
    value = {**summary, "summary_digest": digest}
    json_path = atomic_write_json(destination / "summary.json", value)
    csv_path = atomic_write_text(destination / "metrics.csv", _metrics_csv_text(summary))
    tex_path = atomic_write_text(destination / "metrics_rows.tex", _metrics_tex_text(summary))
    return {
        "status": "complete",
        "table": "metrics",
        "summary_digest": digest,
        "rows": summary["row_count"],
        "output_directory": str(destination),
        "summary_json": str(json_path),
        "metrics_csv": str(csv_path),
        "latex_rows": str(tex_path),
    }


__all__ = [
    "build_metrics_summary",
    "build_table1_summary",
    "write_metrics_summary",
    "write_table1_summary",
]
