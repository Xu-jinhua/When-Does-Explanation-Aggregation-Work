"""Offline effective-robustness post-processing for simple experiment results.

Effective robustness compares a method's perturbed quality with the quality
expected from an original NAIVE single-explainer reference at the same clean
quality.  It deliberately consumes only completed Phase 2 manifests and
published summary files; it never reruns an explanation or a model forward.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS

from .artifacts import ArtifactError, ArtifactStore
from .config import SimpleExperiment
from .robustness import signed_robustness_values

NOISE_ORDER = ("g", "p", "s", "a")
EFFECTIVE_ROBUSTNESS_SCHEMA = "simple-effective-robustness-v1"
EFFECTIVE_ROBUSTNESS_POLICY = "naive_single_nonnegative_ols_residual_v1"
EFFECTIVE_ROBUSTNESS_DIRECTION = "max"
REFERENCE_POOL_POLICY = "original_naive_single_explainers_only"
REFERENCE_FIT_MODEL = "nonnegative_slope_ordinary_least_squares"

ManifestSource = Literal["auto", "local", "remote"]
ManifestLoader = Callable[[Mapping[str, Any], str, Mapping[str, Any]], Mapping[str, Any]]


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactError(f"{context} must be a sequence")
    return value


def _finite_metric_values(value: Any, *, context: str) -> dict[str, float]:
    values = _mapping(value, context=context)
    if set(values) != set(QUALITY_METRICS):
        raise ArtifactError(f"{context} does not cover {QUALITY_METRICS}")
    result = {metric: float(values[metric]) for metric in QUALITY_METRICS}
    if any(not math.isfinite(item) for item in result.values()):
        raise ArtifactError(f"{context} contains a non-finite metric")
    return result


def _finite_perturbed_values(value: Any, *, context: str) -> dict[str, dict[str, float]]:
    values = _mapping(value, context=context)
    if set(values) != set(NOISE_ORDER):
        raise ArtifactError(f"{context} does not cover {NOISE_ORDER}")
    return {
        noise: _finite_metric_values(values[noise], context=f"{context}/{noise}")
        for noise in NOISE_ORDER
    }


def oriented_quality(metric: str, value: float) -> float:
    """Put every quality metric on a larger-is-better scale without clipping."""

    if metric not in QUALITY_METRICS:
        raise ValueError(f"Unknown quality metric {metric!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite {metric} quality")
    return numeric if DEFAULT_METRIC_DIRECTIONS[metric] == "max" else -numeric


@dataclass(frozen=True, slots=True)
class ReferenceCurve:
    """One condition- and metric-specific calibrated NAIVE reference curve."""

    curve_id: str
    cell: str
    metric: str
    condition: str
    excluded_method: str | None
    reference_methods: tuple[str, ...]
    slope: float
    unconstrained_slope: float
    intercept: float
    residual_sum_squares: float
    residual_standard_deviation: float
    r_squared: float | None
    clean_min: float
    clean_max: float
    perturbed_min: float
    perturbed_max: float

    @property
    def count(self) -> int:
        return len(self.reference_methods)

    @property
    def slope_was_constrained(self) -> bool:
        return self.unconstrained_slope < 0.0

    def predict(self, clean_quality: float) -> float:
        return self.intercept + self.slope * float(clean_quality)

    def as_dict(self) -> dict[str, Any]:
        return {
            "curve_id": self.curve_id,
            "cell": self.cell,
            "metric": self.metric,
            "condition": self.condition,
            "excluded_method": self.excluded_method,
            "reference_methods": list(self.reference_methods),
            "reference_count": self.count,
            "fit_model": REFERENCE_FIT_MODEL,
            "slope": self.slope,
            "unconstrained_slope": self.unconstrained_slope,
            "slope_was_constrained": self.slope_was_constrained,
            "intercept": self.intercept,
            "residual_sum_squares": self.residual_sum_squares,
            "residual_standard_deviation": self.residual_standard_deviation,
            "r_squared": self.r_squared,
            "clean_min": self.clean_min,
            "clean_max": self.clean_max,
            "perturbed_min": self.perturbed_min,
            "perturbed_max": self.perturbed_max,
        }


def fit_reference_curve(
    *,
    cell: str,
    metric: str,
    condition: str,
    points: Sequence[tuple[str, float, float]],
    excluded_method: str | None,
) -> ReferenceCurve:
    """Fit the monotone clean-to-perturbed reference curve in oriented space."""

    if metric not in QUALITY_METRICS:
        raise ValueError(f"Unknown quality metric {metric!r}")
    if condition not in NOISE_ORDER:
        raise ValueError(f"Unknown perturbation condition {condition!r}")
    if len(points) < 2:
        raise ArtifactError(
            f"ER reference curve needs at least two single explainers: {cell}/{metric}/{condition}"
        )
    ordered = tuple(sorted(points, key=lambda point: point[0]))
    methods = tuple(point[0] for point in ordered)
    if len(set(methods)) != len(methods):
        raise ArtifactError(f"ER reference methods are not unique: {cell}/{metric}/{condition}")
    x = tuple(float(point[1]) for point in ordered)
    y = tuple(float(point[2]) for point in ordered)
    if any(not math.isfinite(value) for value in (*x, *y)):
        raise ArtifactError(
            f"ER reference curve contains non-finite values: {cell}/{metric}/{condition}"
        )

    count = len(x)
    mean_x = sum(x) / count
    mean_y = sum(y) / count
    sum_xx = sum((value - mean_x) ** 2 for value in x)
    sum_xy = sum(
        (x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(x, y, strict=True)
    )
    unconstrained_slope = 0.0 if math.isclose(sum_xx, 0.0, abs_tol=1e-15) else sum_xy / sum_xx
    slope = max(0.0, unconstrained_slope)
    intercept = mean_y - slope * mean_x
    residuals = tuple(
        y_value - (intercept + slope * x_value) for x_value, y_value in zip(x, y, strict=True)
    )
    residual_sum_squares = sum(value * value for value in residuals)
    residual_standard_deviation = (
        math.sqrt(residual_sum_squares / (count - 2)) if count > 2 else 0.0
    )
    total_sum_squares = sum((value - mean_y) ** 2 for value in y)
    r_squared = (
        None
        if math.isclose(total_sum_squares, 0.0, abs_tol=1e-15)
        else 1.0 - residual_sum_squares / total_sum_squares
    )
    identity = {
        "schema": EFFECTIVE_ROBUSTNESS_SCHEMA,
        "cell": cell,
        "metric": metric,
        "condition": condition,
        "excluded_method": excluded_method,
        "reference_methods": methods,
        "slope": slope,
        "intercept": intercept,
    }
    return ReferenceCurve(
        curve_id=object_sha256(identity),
        cell=cell,
        metric=metric,
        condition=condition,
        excluded_method=excluded_method,
        reference_methods=methods,
        slope=slope,
        unconstrained_slope=unconstrained_slope,
        intercept=intercept,
        residual_sum_squares=residual_sum_squares,
        residual_standard_deviation=residual_standard_deviation,
        r_squared=r_squared,
        clean_min=min(x),
        clean_max=max(x),
        perturbed_min=min(y),
        perturbed_max=max(y),
    )


def _validate_reference_manifests(
    *,
    cell: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    clean = _mapping(manifests["clean"], context="ER clean Phase 2 manifest")
    cell_id = str(cell.get("cell") or f"{cell.get('dataset')}--{cell.get('model')}")
    methods = tuple(
        str(value) for value in _sequence(clean.get("methods"), context=f"{cell_id} methods")
    )
    if not methods or len(set(methods)) != len(methods):
        raise ArtifactError(f"ER reference method roster is invalid for {cell_id}")
    expected_sample_count = int(cell.get("sample_count", clean.get("sample_count", 0)))
    if expected_sample_count <= 0 or int(clean.get("sample_count", -1)) != expected_sample_count:
        raise ArtifactError(f"ER reference sample count is invalid for {cell_id}/clean")
    expected_single_rules = {f"single__{method}" for method in methods}
    for condition in ("clean", *NOISE_ORDER):
        manifest = _mapping(manifests[condition], context=f"ER {cell_id}/{condition} manifest")
        if manifest.get("dataset") != cell.get("dataset") or manifest.get("model") != cell.get(
            "model"
        ):
            raise ArtifactError(f"ER reference identity mismatch for {cell_id}/{condition}")
        if int(manifest.get("sample_count", -1)) != expected_sample_count:
            raise ArtifactError(f"ER reference sample counts differ for {cell_id}/{condition}")
        current_methods = tuple(
            str(value)
            for value in _sequence(
                manifest.get("methods"), context=f"{cell_id}/{condition} methods"
            )
        )
        if current_methods != methods:
            raise ArtifactError(f"ER reference method roster differs for {cell_id}/{condition}")
        metrics = _mapping(manifest.get("metrics"), context=f"{cell_id}/{condition} metrics")
        if not expected_single_rules.issubset(metrics):
            missing = sorted(expected_single_rules.difference(metrics))
            raise ArtifactError(
                f"ER reference is missing NAIVE single explainers for {cell_id}: {missing}"
            )
        for rule in expected_single_rules:
            _finite_metric_values(metrics[rule], context=f"{cell_id}/{condition}/{rule}")
    return {
        "cell": cell_id,
        "dataset": str(cell["dataset"]),
        "model": str(cell["model"]),
        "split": str(cell.get("split", "test")),
        "sample_count": expected_sample_count,
        "methods": methods,
    }


def build_reference_pool(
    reference_summary: Mapping[str, Any],
    *,
    manifest_loader: ManifestLoader,
) -> Mapping[str, Mapping[str, Any]]:
    """Extract the immutable original-NAIVE individual reference pool per cell."""

    if reference_summary.get("table") != "Table_1":
        raise ArtifactError("ER reference summary must be an original NAIVE Table 1 summary")
    cells = _sequence(reference_summary.get("cells"), context="ER reference summary cells")
    result: dict[str, Mapping[str, Any]] = {}
    for cell_value in cells:
        cell = _mapping(cell_value, context="ER reference cell")
        cell_id = str(cell.get("cell") or f"{cell.get('dataset')}--{cell.get('model')}")
        if cell_id in result:
            raise ArtifactError(f"Duplicate ER reference cell: {cell_id}")
        source_tasks = _mapping(cell.get("source_tasks"), context=f"ER source tasks for {cell_id}")
        if set(source_tasks) != {"clean", *NOISE_ORDER}:
            raise ArtifactError(f"ER reference conditions are incomplete for {cell_id}")
        manifests = {
            condition: manifest_loader(
                cell, condition, _mapping(source_tasks[condition], context="ER source task")
            )
            for condition in ("clean", *NOISE_ORDER)
        }
        metadata = _validate_reference_manifests(cell=cell, manifests=manifests)
        methods = tuple(metadata["methods"])
        singles = []
        for method in methods:
            rule = f"single__{method}"
            singles.append(
                {
                    "method": method,
                    "quality": _finite_metric_values(
                        manifests["clean"]["metrics"][rule],
                        context=f"ER reference {cell_id}/clean/{rule}",
                    ),
                    "perturbed_quality": {
                        noise: _finite_metric_values(
                            manifests[noise]["metrics"][rule],
                            context=f"ER reference {cell_id}/{noise}/{rule}",
                        )
                        for noise in NOISE_ORDER
                    },
                }
            )
        result[cell_id] = {
            **metadata,
            "reference_pool_policy": REFERENCE_POOL_POLICY,
            "single_explainers": singles,
            "source_manifests": {
                condition: {
                    "task_id": manifests[condition].get("task_id"),
                    "task_digest": manifests[condition].get("task_digest"),
                    "content_digest": object_sha256(manifests[condition]),
                }
                for condition in ("clean", *NOISE_ORDER)
            },
        }
    if not result:
        raise ArtifactError("ER reference summary contains no cells")
    return result


def _load_manifest_from_source(
    experiment: SimpleExperiment,
    source: Mapping[str, Any],
    *,
    manifest_source: ManifestSource,
) -> Mapping[str, Any]:
    if manifest_source not in {"auto", "local", "remote"}:
        raise ValueError(f"Unsupported manifest source {manifest_source!r}")
    task_id = str(source.get("task_id", ""))
    local_paths: list[Path] = []
    source_manifest = source.get("manifest")
    if isinstance(source_manifest, Mapping) and isinstance(source_manifest.get("path"), str):
        local_paths.append(Path(str(source_manifest["path"])))
    if task_id:
        local_paths.append(experiment.storage.scratch_root / "phase2" / task_id / "manifest.json")
    if manifest_source in {"auto", "local"}:
        for path in local_paths:
            if path.is_file():
                value = read_json(path)
                if not isinstance(value, Mapping):
                    raise ArtifactError(f"Local ER reference manifest is not a mapping: {path}")
                if task_id and value.get("task_id") != task_id:
                    raise ArtifactError(f"Local ER reference manifest task id differs: {path}")
                return value
    if manifest_source == "local":
        raise FileNotFoundError(f"No local ER reference manifest for task {task_id}")
    artifact_root = str(source.get("artifact_root", ""))
    if not artifact_root:
        raise ArtifactError(f"ER reference source has no artifact root: {task_id}")
    value = ArtifactStore(experiment).read_json(f"{artifact_root}/manifest.json")
    if task_id and value.get("task_id") != task_id:
        raise ArtifactError(f"Remote ER reference manifest task id differs: {artifact_root}")
    return value


def load_reference_pool(
    reference_summary_path: str | Path,
    *,
    experiment: SimpleExperiment,
    manifest_source: ManifestSource = "auto",
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Load an ER pool from the exact original NAIVE manifests named by Table 1."""

    path = Path(reference_summary_path).expanduser().resolve()
    summary = read_json(path)
    if not isinstance(summary, Mapping):
        raise ArtifactError(f"ER reference summary is not a mapping: {path}")
    pool = build_reference_pool(
        summary,
        manifest_loader=lambda _cell, _condition, source: _load_manifest_from_source(
            experiment,
            source,
            manifest_source=manifest_source,
        ),
    )
    return pool, {
        "path": str(path),
        "sha256": file_sha256(path),
        "summary_digest": summary.get("summary_digest"),
        "manifest_source": manifest_source,
        "cell_count": len(pool),
    }


def _candidate_metadata(
    *,
    cell: Mapping[str, Any],
    row: Mapping[str, Any],
    extra: Mapping[str, Any],
    context: str,
) -> Mapping[str, Any]:
    dataset = str(row.get("dataset", cell.get("dataset", "")))
    model = str(row.get("model", cell.get("model", "")))
    if not dataset or not model:
        raise ArtifactError(f"{context} has no dataset/model identity")
    cell_id = str(row.get("cell", cell.get("cell", f"{dataset}--{model}")))
    method = str(row.get("method", ""))
    if not method:
        raise ArtifactError(f"{context} has no method")
    quality = _finite_metric_values(row.get("quality"), context=f"{context}/quality")
    perturbed_quality = _finite_perturbed_values(
        row.get("perturbed_quality"), context=f"{context}/perturbed_quality"
    )
    robustness = row.get("robustness")
    if robustness is not None:
        robustness_by_noise = _mapping(robustness, context=f"{context}/robustness")
        if set(robustness_by_noise) != set(NOISE_ORDER):
            raise ArtifactError(f"{context} robustness condition coverage is invalid")
        for noise in NOISE_ORDER:
            recorded = _finite_metric_values(
                robustness_by_noise[noise], context=f"{context}/robustness/{noise}"
            )
            expected = signed_robustness_values(
                quality,
                perturbed_quality[noise],
                context=f"{context}/signed robustness/{noise}",
            )
            if any(
                not math.isclose(recorded[metric], expected[metric], rel_tol=0.0, abs_tol=1e-12)
                for metric in QUALITY_METRICS
            ):
                raise ArtifactError(f"{context} signed robustness contradicts raw quality")
    setting = str(row.get("setting", extra.get("setting", "naive")))
    sample_count_value = row.get("sample_count", cell.get("sample_count"))
    sample_count = None if sample_count_value is None else int(sample_count_value)
    metadata = {
        "cell": cell_id,
        "dataset": dataset,
        "model": model,
        "split": str(row.get("split", cell.get("split", "test"))),
        "sample_count": sample_count,
        "setting": setting,
        "method": method,
        "distance_model": row.get(
            "distance_model", extra.get("distance_model", cell.get("distance_model"))
        ),
        "geometry": row.get("geometry", extra.get("geometry")),
        "geometry_label": row.get("geometry_label", extra.get("geometry_label")),
        "q": row.get("q", extra.get("q")),
        "source_id": row.get("source_id", extra.get("source_id")),
        "methods": row.get("methods", extra.get("methods")),
        "selected_sources": row.get("selected_sources"),
        "selected_individual": row.get("selected_individual"),
        "quality": quality,
        "perturbed_quality": perturbed_quality,
    }
    if metadata["sample_count"] is not None and metadata["sample_count"] <= 0:
        raise ArtifactError(f"{context} has an invalid sample count")
    identity = {
        key: metadata[key]
        for key in (
            "cell",
            "setting",
            "method",
            "distance_model",
            "geometry",
            "q",
            "source_id",
        )
    }
    return {**metadata, "candidate_id": object_sha256(identity)}


def extract_candidate_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Normalize completed table-level NAIVE, IND, and NOISE summaries."""

    cells = _sequence(summary.get("cells"), context="ER candidate summary cells")
    schema = summary.get("schema")
    rows: list[Mapping[str, Any]] = []
    if schema == "simple-independent-geometry-noise-report-v1":
        for cell_value in cells:
            cell = _mapping(cell_value, context="independent-geometry ER cell")
            geometries = _sequence(cell.get("geometries"), context="independent-geometry rows")
            for geometry_value in geometries:
                geometry = _mapping(geometry_value, context="independent-geometry record")
                for row_value in _sequence(
                    geometry.get("rows"), context="independent-geometry rows"
                ):
                    rows.append(
                        _candidate_metadata(
                            cell=cell,
                            row=_mapping(row_value, context="independent-geometry candidate"),
                            extra=geometry,
                            context="independent-geometry candidate",
                        )
                    )
    else:
        for cell_value in cells:
            cell = _mapping(cell_value, context="ER candidate cell")
            direct_rows = _sequence(cell.get("rows"), context="ER candidate rows")
            for row_value in direct_rows:
                rows.append(
                    _candidate_metadata(
                        cell=cell,
                        row=_mapping(row_value, context="ER candidate row"),
                        extra={},
                        context="ER candidate row",
                    )
                )
    if not rows:
        raise ArtifactError("ER candidate summary contains no completed rows")
    identities = [str(row["candidate_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ArtifactError("ER candidate summary has duplicate row identities")
    return tuple(rows)


def _best_individual_source(candidate: Mapping[str, Any], metric: str) -> str | None:
    selected_sources = candidate.get("selected_sources")
    if isinstance(selected_sources, Mapping):
        quality_sources = selected_sources.get("quality")
        if isinstance(quality_sources, Mapping) and metric in quality_sources:
            return str(quality_sources[metric]).removeprefix("single__")
    selected_individual = candidate.get("selected_individual")
    if isinstance(selected_individual, Mapping) and metric in selected_individual:
        return str(selected_individual[metric]).removeprefix("single__")
    return None


def _excluded_reference_method(candidate: Mapping[str, Any], metric: str) -> str | None:
    method = str(candidate["method"])
    normalized_method = method.lower().replace(" ", "_").replace("-", "_")
    setting = str(candidate["setting"]).lower().replace("_", "-")
    if method.startswith("single__"):
        return method.removeprefix("single__")
    if setting in {"single", "individual", "naive-single"}:
        return method.removeprefix("single__")
    if normalized_method in {"best_individual", "bestindividual"} or setting == "best-individual":
        source = _best_individual_source(candidate, metric)
        if source is None:
            raise ArtifactError(
                f"ER Best Individual row has no clean selected source for {candidate['cell']}/{metric}"
            )
        return source
    return None


def _reference_points(
    reference_cell: Mapping[str, Any],
    *,
    metric: str,
    condition: str,
    excluded_method: str | None,
) -> tuple[tuple[str, float, float], ...]:
    points = []
    methods = set()
    for source in _sequence(
        reference_cell.get("single_explainers"), context="ER single reference pool"
    ):
        source_value = _mapping(source, context="ER single explainer")
        method = str(source_value.get("method", ""))
        if not method:
            raise ArtifactError("ER reference single explainer has no method")
        methods.add(method)
        if method == excluded_method:
            continue
        quality = _finite_metric_values(
            source_value.get("quality"), context=f"ER reference {method}"
        )
        perturbed = _finite_perturbed_values(
            source_value.get("perturbed_quality"), context=f"ER reference {method}/perturbed"
        )
        points.append(
            (
                method,
                oriented_quality(metric, quality[metric]),
                oriented_quality(metric, perturbed[condition][metric]),
            )
        )
    if excluded_method is not None and excluded_method not in methods:
        raise ArtifactError(
            f"ER row asks to exclude unknown NAIVE single {excluded_method!r} "
            f"for {reference_cell['cell']}"
        )
    return tuple(points)


def _endpoint_columns() -> tuple[str, ...]:
    return tuple(f"ER_{metric}_{noise}" for noise in NOISE_ORDER for metric in QUALITY_METRICS)


def build_effective_robustness_summary(
    candidate_summary: Mapping[str, Any],
    *,
    reference_pool: Mapping[str, Mapping[str, Any]],
    input_provenance: Mapping[str, Any] | None = None,
    reference_provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Calculate ER for every completed candidate row without Phase 2 execution."""

    candidates = extract_candidate_rows(candidate_summary)
    curves: dict[tuple[str, str, str, str | None], ReferenceCurve] = {}
    output_rows = []
    for candidate in candidates:
        cell_id = str(candidate["cell"])
        reference_cell = reference_pool.get(cell_id)
        if reference_cell is None:
            raise ArtifactError(f"ER has no original NAIVE reference pool for {cell_id}")
        candidate_sample_count = candidate["sample_count"]
        if candidate_sample_count is not None and int(candidate_sample_count) != int(
            reference_cell["sample_count"]
        ):
            raise ArtifactError(f"ER sample count differs from reference for {cell_id}")
        endpoints: dict[str, dict[str, Mapping[str, Any]]] = {}
        for noise in NOISE_ORDER:
            endpoints[noise] = {}
            for metric in QUALITY_METRICS:
                excluded_method = _excluded_reference_method(candidate, metric)
                curve_key = (cell_id, metric, noise, excluded_method)
                curve = curves.get(curve_key)
                if curve is None:
                    curve = fit_reference_curve(
                        cell=cell_id,
                        metric=metric,
                        condition=noise,
                        points=_reference_points(
                            reference_cell,
                            metric=metric,
                            condition=noise,
                            excluded_method=excluded_method,
                        ),
                        excluded_method=excluded_method,
                    )
                    curves[curve_key] = curve
                clean_raw = float(candidate["quality"][metric])
                perturbed_raw = float(candidate["perturbed_quality"][noise][metric])
                clean_oriented = oriented_quality(metric, clean_raw)
                perturbed_oriented = oriented_quality(metric, perturbed_raw)
                expected_oriented = curve.predict(clean_oriented)
                endpoints[noise][metric] = {
                    "value": perturbed_oriented - expected_oriented,
                    "clean_quality_raw": clean_raw,
                    "perturbed_quality_raw": perturbed_raw,
                    "clean_quality_oriented": clean_oriented,
                    "perturbed_quality_oriented": perturbed_oriented,
                    "expected_perturbed_quality_oriented": expected_oriented,
                    "reference_curve_id": curve.curve_id,
                    "reference_pool_policy": (
                        "leave_one_out_naive_single"
                        if excluded_method is not None
                        else "full_naive_single_pool"
                    ),
                    "excluded_reference_method": excluded_method,
                    "clean_quality_extrapolated": (
                        clean_oriented < curve.clean_min or clean_oriented > curve.clean_max
                    ),
                }
        output_rows.append(
            {
                **{
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "cell",
                        "dataset",
                        "model",
                        "split",
                        "setting",
                        "method",
                        "distance_model",
                        "geometry",
                        "geometry_label",
                        "q",
                        "source_id",
                        "methods",
                        "selected_sources",
                        "selected_individual",
                        "quality",
                        "perturbed_quality",
                    )
                },
                "sample_count": int(reference_cell["sample_count"]),
                "effective_robustness": endpoints,
            }
        )

    by_setting: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "zero": 0, "negative": 0, "endpoints": 0}
    )
    extrapolated = 0
    for row in output_rows:
        counts = by_setting[str(row["setting"])]
        for noise in NOISE_ORDER:
            for metric in QUALITY_METRICS:
                endpoint = row["effective_robustness"][noise][metric]
                value = float(endpoint["value"])
                counts["endpoints"] += 1
                if value > 1e-12:
                    counts["positive"] += 1
                elif value < -1e-12:
                    counts["negative"] += 1
                else:
                    counts["zero"] += 1
                extrapolated += int(bool(endpoint["clean_quality_extrapolated"]))
    value: dict[str, Any] = {
        "schema": EFFECTIVE_ROBUSTNESS_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "definition": {
            "name": "Effective Robustness",
            "formula": "ER_m^o = S_m(Q^o) - beta_hat_m^o(S_m(Q))",
            "direction": EFFECTIVE_ROBUSTNESS_DIRECTION,
            "quality_orientation": {
                metric: ("identity" if DEFAULT_METRIC_DIRECTIONS[metric] == "max" else "negate")
                for metric in QUALITY_METRICS
            },
            "reference_pool": REFERENCE_POOL_POLICY,
            "reference_fit": REFERENCE_FIT_MODEL,
            "candidate_pool_excluded_from_reference": True,
            "single_and_best_individual_reference_policy": "leave_one_out",
            "aggregate_reference_policy": "full_naive_single_pool",
        },
        "input": dict(input_provenance or {}),
        "reference": {
            **dict(reference_provenance or {}),
            "cell_count": len(reference_pool),
            "single_method_count_by_cell": {
                cell: len(_sequence(value["single_explainers"], context="ER reference pool"))
                for cell, value in sorted(reference_pool.items())
            },
        },
        "columns": list(_endpoint_columns()),
        "column_directions": {
            column: EFFECTIVE_ROBUSTNESS_DIRECTION for column in _endpoint_columns()
        },
        "cells": output_rows,
        "reference_curves": [
            curve.as_dict() for curve in sorted(curves.values(), key=lambda value: value.curve_id)
        ],
        "analysis": {
            "candidate_rows": len(output_rows),
            "endpoints": len(output_rows) * len(NOISE_ORDER) * len(QUALITY_METRICS),
            "clean_quality_extrapolated_endpoints": extrapolated,
            "by_setting": {key: dict(value) for key, value in sorted(by_setting.items())},
        },
    }
    value["result_digest"] = object_sha256(value)
    return value


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()


def _flat_table_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["cells"]:
        value = {
            key: row.get(key)
            for key in (
                "candidate_id",
                "cell",
                "dataset",
                "model",
                "split",
                "sample_count",
                "setting",
                "method",
                "distance_model",
                "geometry",
                "geometry_label",
                "q",
                "source_id",
            )
        }
        value["methods"] = "" if row.get("methods") is None else ";".join(row["methods"])
        value["selected_individual"] = (
            "" if row.get("selected_individual") is None else str(row["selected_individual"])
        )
        for noise in NOISE_ORDER:
            for metric in QUALITY_METRICS:
                value[f"ER_{metric}_{noise}"] = row["effective_robustness"][noise][metric]["value"]
        rows.append(value)
    return rows


def _detail_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["cells"]:
        identity = {
            key: row.get(key)
            for key in (
                "candidate_id",
                "cell",
                "dataset",
                "model",
                "split",
                "sample_count",
                "setting",
                "method",
                "distance_model",
                "geometry",
                "geometry_label",
                "q",
                "source_id",
            )
        }
        for noise in NOISE_ORDER:
            for metric in QUALITY_METRICS:
                endpoint = row["effective_robustness"][noise][metric]
                rows.append(
                    {
                        **identity,
                        "condition": noise,
                        "metric": metric,
                        "raw_metric_direction": DEFAULT_METRIC_DIRECTIONS[metric],
                        "ER": endpoint["value"],
                        "clean_quality_raw": endpoint["clean_quality_raw"],
                        "perturbed_quality_raw": endpoint["perturbed_quality_raw"],
                        "clean_quality_oriented": endpoint["clean_quality_oriented"],
                        "perturbed_quality_oriented": endpoint["perturbed_quality_oriented"],
                        "expected_perturbed_quality_oriented": endpoint[
                            "expected_perturbed_quality_oriented"
                        ],
                        "reference_curve_id": endpoint["reference_curve_id"],
                        "reference_pool_policy": endpoint["reference_pool_policy"],
                        "excluded_reference_method": endpoint["excluded_reference_method"],
                        "clean_quality_extrapolated": endpoint["clean_quality_extrapolated"],
                    }
                )
    return rows


def _curve_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for curve in summary["reference_curves"]:
        rows.append(
            {
                **{
                    key: curve.get(key)
                    for key in (
                        "curve_id",
                        "cell",
                        "metric",
                        "condition",
                        "excluded_method",
                        "reference_count",
                        "fit_model",
                        "slope",
                        "unconstrained_slope",
                        "slope_was_constrained",
                        "intercept",
                        "residual_sum_squares",
                        "residual_standard_deviation",
                        "r_squared",
                        "clean_min",
                        "clean_max",
                        "perturbed_min",
                        "perturbed_max",
                    )
                },
                "reference_methods": ";".join(curve["reference_methods"]),
            }
        )
    return rows


def _tex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _tex_rows(summary: Mapping[str, Any]) -> str:
    lines = [
        "% Generated by simple effective-robustness post-processing.",
        "% Every ER column is larger-is-better; raw full-precision values are in er_table.csv.",
        "% Columns: dataset, model, setting, geometry, method, then ER_F/Fbar/C/Cbar for g,p,s,a.",
    ]
    for row in _flat_table_rows(summary):
        leading = [
            _tex_escape(row["dataset"]),
            _tex_escape(row["model"]),
            _tex_escape(row["setting"]),
            _tex_escape(row["geometry_label"] or row["distance_model"] or "-"),
            _tex_escape(row["method"]),
        ]
        values = [f"{float(row[column]):.6f}" for column in _endpoint_columns()]
        lines.append(" & ".join([*leading, *values]) + r" \\")
    return "\n".join(lines) + "\n"


def _readme(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# Effective Robustness",
            "",
            "This directory is an offline post-processing result. No Phase 1 attribution, Phase 2 inference, or GPU job was rerun.",
            "",
            "For each metric and perturbation, ER is the oriented perturbed quality minus the value expected from the original NAIVE single-explainer reference curve at the same oriented clean quality.",
            "`F` and `C` retain their sign; `Fbar` and `Cbar` are negated only to make every ER value larger-is-better.",
            "",
            "The reference pool contains only original NAIVE individual explainers for the matching dataset-model cell. It never contains NOISE, IND, matched-NAIVE, or aggregate candidates.",
            "A direct individual row and a Best Individual row exclude their actual selected source method before fitting; aggregate rows use the full single-explainer pool.",
            "",
            "`er_table.csv` and `er_table_rows.tex` contain the 16 paper-facing ER entries per result row.",
            "`er_detail.csv` preserves raw and oriented clean/perturbed values, expected values, extrapolation flags, and the curve used by every endpoint.",
            "`reference_curves.csv` records the fitted coefficients and diagnostics for every full or leave-one-out reference curve.",
            "",
            f"Result digest: `{summary['result_digest']}`.",
            "",
        )
    )


def write_effective_robustness_outputs(
    summary: Mapping[str, Any],
    *,
    output_directory: str | Path,
) -> Mapping[str, Any]:
    """Write the compact table plus complete ER audit records."""

    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    table_columns = (
        "candidate_id",
        "cell",
        "dataset",
        "model",
        "split",
        "sample_count",
        "setting",
        "method",
        "distance_model",
        "geometry",
        "geometry_label",
        "q",
        "source_id",
        "methods",
        "selected_individual",
        *_endpoint_columns(),
    )
    detail_columns = (
        "candidate_id",
        "cell",
        "dataset",
        "model",
        "split",
        "sample_count",
        "setting",
        "method",
        "distance_model",
        "geometry",
        "geometry_label",
        "q",
        "source_id",
        "condition",
        "metric",
        "raw_metric_direction",
        "ER",
        "clean_quality_raw",
        "perturbed_quality_raw",
        "clean_quality_oriented",
        "perturbed_quality_oriented",
        "expected_perturbed_quality_oriented",
        "reference_curve_id",
        "reference_pool_policy",
        "excluded_reference_method",
        "clean_quality_extrapolated",
    )
    curve_columns = (
        "curve_id",
        "cell",
        "metric",
        "condition",
        "excluded_method",
        "reference_count",
        "fit_model",
        "slope",
        "unconstrained_slope",
        "slope_was_constrained",
        "intercept",
        "residual_sum_squares",
        "residual_standard_deviation",
        "r_squared",
        "clean_min",
        "clean_max",
        "perturbed_min",
        "perturbed_max",
        "reference_methods",
    )
    paths = {
        "summary_json": atomic_write_json(destination / "summary.json", summary),
        "er_table_csv": atomic_write_text(
            destination / "er_table.csv", _csv_text(_flat_table_rows(summary), table_columns)
        ),
        "er_detail_csv": atomic_write_text(
            destination / "er_detail.csv", _csv_text(_detail_rows(summary), detail_columns)
        ),
        "reference_curves_csv": atomic_write_text(
            destination / "reference_curves.csv", _csv_text(_curve_rows(summary), curve_columns)
        ),
        "er_table_rows_tex": atomic_write_text(
            destination / "er_table_rows.tex", _tex_rows(summary)
        ),
        "readme": atomic_write_text(destination / "README.md", _readme(summary)),
    }
    return {
        "status": "complete",
        "result_digest": summary["result_digest"],
        "candidate_rows": len(summary["cells"]),
        "endpoints": summary["analysis"]["endpoints"],
        "reference_curves": len(summary["reference_curves"]),
        **{key: str(path) for key, path in paths.items()},
    }


def write_effective_robustness(
    *,
    input_summary_path: str | Path,
    reference_summary_path: str | Path,
    experiment: SimpleExperiment,
    manifest_source: ManifestSource = "auto",
    output_directory: str | Path,
) -> Mapping[str, Any]:
    """Load existing results, compute ER, and publish only Git-sized outputs."""

    input_path = Path(input_summary_path).expanduser().resolve()
    candidate_summary = read_json(input_path)
    if not isinstance(candidate_summary, Mapping):
        raise ArtifactError(f"ER input summary is not a mapping: {input_path}")
    reference_pool, reference_provenance = load_reference_pool(
        reference_summary_path,
        experiment=experiment,
        manifest_source=manifest_source,
    )
    summary = build_effective_robustness_summary(
        candidate_summary,
        reference_pool=reference_pool,
        input_provenance={
            "path": str(input_path),
            "sha256": file_sha256(input_path),
            "summary_digest": candidate_summary.get(
                "summary_digest", candidate_summary.get("result_digest")
            ),
        },
        reference_provenance=reference_provenance,
    )
    return write_effective_robustness_outputs(summary, output_directory=output_directory)


__all__ = [
    "EFFECTIVE_ROBUSTNESS_DIRECTION",
    "EFFECTIVE_ROBUSTNESS_POLICY",
    "EFFECTIVE_ROBUSTNESS_SCHEMA",
    "REFERENCE_FIT_MODEL",
    "REFERENCE_POOL_POLICY",
    "build_effective_robustness_summary",
    "build_reference_pool",
    "extract_candidate_rows",
    "fit_reference_curve",
    "load_reference_pool",
    "oriented_quality",
    "write_effective_robustness",
    "write_effective_robustness_outputs",
]
