"""Absolute quality, clean retention, and their geometric mean.

This is an offline post-processing stage. It combines completed NOISE reports
with the exact q=11 NAIVE Phase 2 manifests and recovers each condition's
unmasked reference accuracy from immutable Phase 2 shards. It never reruns a
model forward pass.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Literal

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.metrics import QUALITY_METRICS

from .artifacts import ArtifactError, ArtifactStore
from .config import SimpleExperiment
from .summary import NOISE_NAMES, NOISE_ORDER, PAPER_RULES

QUALITY_RETENTION_SCHEMA = "simple-quality-retention-v1"
QUALITY_RETENTION_POLICY = "condition_normalized_absolute_times_clean_retention_v1"
COMPARISON_TOLERANCE = 1e-12
ManifestSource = Literal["auto", "local", "remote"]
CONDITION_IDS = {
    "clean": "clean",
    "g": "gaussian-0.15",
    "p": "salt-pepper-0.05",
    "s": "speckle-0.15",
    "a": "adversarial-sara-2-255",
}


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactError(f"{context} must be a sequence")
    return value


def _finite_metrics(value: Any, *, context: str) -> dict[str, float]:
    source = _mapping(value, context=context)
    if set(source) != set(QUALITY_METRICS):
        raise ArtifactError(f"{context} must cover exactly {QUALITY_METRICS}")
    result = {metric: float(source[metric]) for metric in QUALITY_METRICS}
    if any(not math.isfinite(item) for item in result.values()):
        raise ArtifactError(f"{context} contains a non-finite value")
    return result


def normalize_quality(metric: str, value: float, *, unmasked_accuracy: float) -> Mapping[str, Any]:
    """Map a raw quality metric to [0, 1], with one meaning best quality."""

    if metric not in QUALITY_METRICS:
        raise ValueError(f"Unknown quality metric {metric!r}")
    numeric = float(value)
    accuracy = float(unmasked_accuracy)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite {metric} value")
    if not math.isfinite(accuracy) or not 0.0 < accuracy <= 1.0:
        raise ValueError(f"unmasked_accuracy must be in (0, 1], observed {accuracy!r}")
    if metric == "F":
        raw = numeric / accuracy
    elif metric == "Fbar":
        raw = 1.0 - numeric / accuracy
    elif metric == "C":
        raw = numeric
    else:
        raw = 1.0 - numeric
    clipped = min(1.0, max(0.0, raw))
    return {
        "raw_normalized": raw,
        "value": clipped,
        "clipped": clipped != raw,
        "clipped_low": raw < 0.0,
        "clipped_high": raw > 1.0,
    }


def quality_retention_endpoint(
    *,
    metric: str,
    clean_quality: float,
    perturbed_quality: float,
    clean_unmasked_accuracy: float,
    perturbed_unmasked_accuracy: float,
) -> Mapping[str, Any]:
    """Calculate A, Q, and G for one metric and one perturbation condition."""

    clean = normalize_quality(
        metric,
        clean_quality,
        unmasked_accuracy=clean_unmasked_accuracy,
    )
    perturbed = normalize_quality(
        metric,
        perturbed_quality,
        unmasked_accuracy=perturbed_unmasked_accuracy,
    )
    clean_value = float(clean["value"])
    absolute_quality = float(perturbed["value"])
    if clean_value == 0.0:
        retention = None
        raw_retention = None
        geometric_mean = None
        retention_capped = False
        zero_clean_boundary = True
    else:
        raw_retention = absolute_quality / clean_value
        retention = min(1.0, raw_retention)
        geometric_mean = math.sqrt(absolute_quality * retention)
        retention_capped = raw_retention > 1.0
        zero_clean_boundary = False
    return {
        "clean_quality_raw": float(clean_quality),
        "perturbed_quality_raw": float(perturbed_quality),
        "A0_clean": float(clean_unmasked_accuracy),
        "A0_perturbed": float(perturbed_unmasked_accuracy),
        "normalized_clean_raw": clean["raw_normalized"],
        "normalized_perturbed_raw": perturbed["raw_normalized"],
        "Qnorm_clean": clean_value,
        "Qnorm_perturbed": absolute_quality,
        "A": absolute_quality,
        "Q": retention,
        "Q_raw": raw_retention,
        "G": geometric_mean,
        "clean_clipped": clean["clipped"],
        "clean_clipped_low": clean["clipped_low"],
        "clean_clipped_high": clean["clipped_high"],
        "perturbed_clipped": perturbed["clipped"],
        "perturbed_clipped_low": perturbed["clipped_low"],
        "perturbed_clipped_high": perturbed["clipped_high"],
        "retention_capped": retention_capped,
        "zero_clean_boundary": zero_clean_boundary,
    }


def _read_manifest(
    *,
    cell: Mapping[str, Any],
    condition: str,
    source: Mapping[str, Any],
    experiment: SimpleExperiment,
    store: ArtifactStore,
    manifest_source: ManifestSource,
    expected_sample_count: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if manifest_source not in {"auto", "local", "remote"}:
        raise ValueError(f"Unsupported manifest source {manifest_source!r}")
    task_id = str(source.get("task_id", ""))
    task_digest = str(source.get("task_digest", ""))
    artifact_root = str(source.get("artifact_root", ""))
    if not task_id or not task_digest or not artifact_root:
        raise ArtifactError(f"Incomplete source task for {cell.get('dataset')}/{condition}")
    source_manifest = source.get("manifest")
    expected_sha256 = None
    local_paths: list[Path] = []
    if isinstance(source_manifest, Mapping):
        if isinstance(source_manifest.get("sha256"), str):
            expected_sha256 = str(source_manifest["sha256"])
        if isinstance(source_manifest.get("path"), str):
            local_paths.append(Path(str(source_manifest["path"])))
    local_paths.append(experiment.storage.scratch_root / "phase2" / task_id / "manifest.json")

    raw: bytes | None = None
    locator = ""
    source_kind = ""
    if manifest_source in {"auto", "local"}:
        for path in local_paths:
            if path.is_file():
                raw = path.read_bytes()
                locator = str(path.resolve())
                source_kind = "local"
                break
    if raw is None and manifest_source == "local":
        raise FileNotFoundError(f"No local Phase 2 manifest for {task_id}")
    if raw is None:
        relative_path = f"{artifact_root}/manifest.json"
        raw = store.read_bytes(relative_path)
        locator = store.locator(relative_path)
        source_kind = "remote"
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ArtifactError(
            f"Phase 2 manifest SHA-256 mismatch for {task_id}: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Invalid Phase 2 manifest JSON for {task_id}") from error
    manifest = _mapping(manifest, context=f"Phase 2 manifest {task_id}")
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version") != 2
        or manifest.get("task_id") != task_id
        or manifest.get("task_digest") != task_digest
        or manifest.get("dataset") != cell.get("dataset")
        or manifest.get("model") != cell.get("model")
        or manifest.get("condition") != CONDITION_IDS[condition]
        or manifest.get("ensemble") != "all-paper-methods"
        or int(manifest.get("patch_size", -1)) != 16
        or int(manifest.get("sample_count", -1)) != expected_sample_count
    ):
        raise ArtifactError(f"Phase 2 manifest identity mismatch for {task_id}")
    manifest_with_origin = dict(manifest)
    manifest_with_origin["_artifact_root"] = artifact_root
    return manifest_with_origin, {
        "condition": condition,
        "task_id": task_id,
        "task_digest": task_digest,
        "artifact_root": artifact_root,
        "manifest_source": source_kind,
        "manifest_locator": locator,
        "manifest_sha256": observed_sha256,
    }


def _validate_shards(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    task_id = str(manifest["task_id"])
    task_digest = str(manifest["task_digest"])
    sample_count = int(manifest.get("sample_count", 0))
    if sample_count <= 0:
        raise ArtifactError(f"Invalid sample count for {task_id}")
    rows = sorted(
        (
            _mapping(row, context=f"{task_id} shard")
            for row in _sequence(manifest.get("shards"), context=f"{task_id} shards")
        ),
        key=lambda row: int(row.get("shard_index", -1)),
    )
    if not rows:
        raise ArtifactError(f"No Phase 2 shards for {task_id}")
    expected_start = 0
    for expected_index, row in enumerate(rows):
        start = int(row.get("start", -1))
        stop = int(row.get("stop", -1))
        count = int(row.get("count", -1))
        payload = _mapping(row.get("payload"), context=f"{task_id}/{expected_index} payload")
        relative_path = str(payload.get("relative_path", ""))
        expected_root = str(
            manifest.get("_artifact_root", manifest.get("artifact_root", ""))
        ).rstrip("/")
        if expected_root and not relative_path.startswith(f"{expected_root}/shards/"):
            raise ArtifactError(f"Phase 2 shard path is outside its manifest root: {relative_path}")
        if (
            int(row.get("shard_index", -1)) != expected_index
            or row.get("task_digest") != task_digest
            or start != expected_start
            or stop <= start
            or count != stop - start
            or not isinstance(payload.get("relative_path"), str)
            or not isinstance(payload.get("sha256"), str)
            or len(str(payload.get("sha256"))) != 64
            or int(payload.get("size_bytes", 0)) <= 0
        ):
            raise ArtifactError(f"Invalid Phase 2 shard record {task_id}/{expected_index}")
        expected_start = stop
    if expected_start != sample_count:
        raise ArtifactError(
            f"Phase 2 shard coverage differs for {task_id}: {expected_start} != {sample_count}"
        )
    return tuple(rows)


def _read_accuracy_shard(
    *,
    store: ArtifactStore,
    row: Mapping[str, Any],
    destination: Path,
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open

    payload = _mapping(row["payload"], context="Phase 2 shard payload")
    expected_sha256 = str(payload["sha256"])
    path = store.materialize(
        str(payload["relative_path"]),
        destination,
        expected_sha256=expected_sha256,
    )
    try:
        if path.stat().st_size != int(payload["size_bytes"]):
            raise ArtifactError(f"Phase 2 shard size mismatch: {payload['relative_path']}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"indices", "labels", "unmasked_predictions"}
            if not required.issubset(keys):
                raise ArtifactError(
                    f"Phase 2 shard lacks {sorted(required.difference(keys))}: "
                    f"{payload['relative_path']}"
                )
            indices = handle.get_tensor("indices")
            labels = handle.get_tensor("labels")
            predictions = handle.get_tensor("unmasked_predictions")
        count = int(row["count"])
        if any(
            value.ndim != 1 or value.numel() != count for value in (indices, labels, predictions)
        ):
            raise ArtifactError(f"Phase 2 shard tensor shape mismatch: {payload['relative_path']}")
        if any(torch.is_floating_point(value) for value in (indices, labels, predictions)):
            raise ArtifactError(
                f"Phase 2 shard identity tensors must be integral: {payload['relative_path']}"
            )
        correct_count = int((predictions == labels).sum().item())
        evidence = {
            "indices": indices.tolist(),
            "labels": labels.tolist(),
            "unmasked_predictions": predictions.tolist(),
        }
        return {
            "shard_index": int(row["shard_index"]),
            "count": count,
            "correct_count": correct_count,
            "indices": tuple(int(value) for value in indices.tolist()),
            "payload_sha256": expected_sha256,
            "evidence_digest": object_sha256(evidence),
        }
    finally:
        path.unlink(missing_ok=True)


def recover_unmasked_accuracy(
    manifest: Mapping[str, Any],
    *,
    store: ArtifactStore,
    temporary_directory: str | Path,
    download_workers: int,
) -> Mapping[str, Any]:
    """Recover A0 from verified labels and unmasked predictions in every shard."""

    if download_workers <= 0:
        raise ValueError("download_workers must be positive")
    rows = _validate_shards(manifest)
    temporary_root = Path(temporary_directory).expanduser().resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="xai-quality-retention-", dir=temporary_root
    ) as directory:
        root = Path(directory)

        def read(row: Mapping[str, Any]) -> Mapping[str, Any]:
            return _read_accuracy_shard(
                store=store,
                row=row,
                destination=root / f"shard-{int(row['shard_index']):05d}.safetensors",
            )

        if download_workers == 1:
            evidence = tuple(read(row) for row in rows)
        else:
            with ThreadPoolExecutor(max_workers=download_workers) as executor:
                evidence = tuple(executor.map(read, rows))
    evidence = tuple(sorted(evidence, key=lambda row: int(row["shard_index"])))
    all_indices = tuple(index for row in evidence for index in row["indices"])
    sample_count = int(manifest["sample_count"])
    if len(all_indices) != sample_count or len(set(all_indices)) != sample_count:
        raise ArtifactError(f"Phase 2 sample identities are incomplete for {manifest['task_id']}")
    correct_count = sum(int(row["correct_count"]) for row in evidence)
    return {
        "unmasked_accuracy": correct_count / sample_count,
        "correct_count": correct_count,
        "sample_count": sample_count,
        "sample_identity_digest": object_sha256(all_indices),
        "shard_evidence_digest": object_sha256(
            [
                {
                    key: row[key]
                    for key in (
                        "shard_index",
                        "count",
                        "correct_count",
                        "payload_sha256",
                        "evidence_digest",
                    )
                }
                for row in evidence
            ]
        ),
        "shard_count": len(evidence),
        "payload_sha256": [str(row["payload_sha256"]) for row in evidence],
    }


def load_reference_manifests_and_accuracies(
    reference_summary: Mapping[str, Any],
    *,
    experiment: SimpleExperiment,
    manifest_source: ManifestSource,
    temporary_directory: str | Path,
    download_workers: int,
) -> tuple[
    Mapping[str, Mapping[str, Mapping[str, Any]]],
    Mapping[str, Mapping[str, float]],
    Mapping[str, Any],
]:
    """Load exact q=11 manifests and recover condition-specific A0 values."""

    if reference_summary.get("table") != "Table_1":
        raise ArtifactError("Reference summary must be the original NAIVE Table 1")
    store = ArtifactStore(experiment)
    manifests: dict[str, dict[str, Mapping[str, Any]]] = {}
    accuracies: dict[str, dict[str, float]] = {}
    provenance_cells = []
    for cell_value in _sequence(reference_summary.get("cells"), context="Table 1 cells"):
        cell = _mapping(cell_value, context="Table 1 cell")
        cell_id = f"{cell.get('dataset')}--{cell.get('model')}"
        if cell_id in manifests:
            raise ArtifactError(f"Duplicate Table 1 cell {cell_id}")
        source_tasks = _mapping(cell.get("source_tasks"), context=f"{cell_id} source tasks")
        if set(source_tasks) != {"clean", *NOISE_ORDER}:
            raise ArtifactError(f"Incomplete Table 1 source-task set for {cell_id}")
        cell_manifests: dict[str, Mapping[str, Any]] = {}
        cell_accuracies: dict[str, float] = {}
        cell_provenance = []
        for condition in ("clean", *NOISE_ORDER):
            manifest, provenance = _read_manifest(
                cell=cell,
                condition=condition,
                source=_mapping(source_tasks[condition], context=f"{cell_id}/{condition} source"),
                experiment=experiment,
                store=store,
                manifest_source=manifest_source,
                expected_sample_count=int(cell.get("sample_count", -1)),
            )
            expected_methods = tuple(str(value) for value in cell.get("methods", ()))
            observed_methods = tuple(
                str(value)
                for value in _sequence(
                    manifest.get("methods"), context=f"{cell_id}/{condition} methods"
                )
            )
            if expected_methods and observed_methods != expected_methods:
                raise ArtifactError(f"Phase 2 method roster differs for {cell_id}/{condition}")
            metrics = _mapping(manifest.get("metrics"), context=f"{cell_id}/{condition} metrics")
            missing_rules = set(PAPER_RULES).difference(metrics)
            if missing_rules:
                raise ArtifactError(
                    f"Missing q=11 rules for {cell_id}/{condition}: {sorted(missing_rules)}"
                )
            for rule in PAPER_RULES:
                _finite_metrics(metrics[rule], context=f"{cell_id}/{condition}/{rule}")
            accuracy = recover_unmasked_accuracy(
                manifest,
                store=store,
                temporary_directory=temporary_directory,
                download_workers=download_workers,
            )
            cell_manifests[condition] = manifest
            cell_accuracies[condition] = float(accuracy["unmasked_accuracy"])
            cell_provenance.append({**provenance, "accuracy_evidence": accuracy})
        manifests[cell_id] = cell_manifests
        accuracies[cell_id] = cell_accuracies
        provenance_cells.append(
            {
                "cell": cell_id,
                "dataset": cell["dataset"],
                "model": cell["model"],
                "conditions": cell_provenance,
            }
        )
    return (
        manifests,
        accuracies,
        {
            "manifest_source": manifest_source,
            "temporary_directory": str(Path(temporary_directory).expanduser().resolve()),
            "download_workers": download_workers,
            "cells": provenance_cells,
            "content_digest": object_sha256(provenance_cells),
        },
    )


def _extract_noise_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if summary.get("schema") != "simple-independent-geometry-noise-report-v1":
        raise ArtifactError("Candidate summary is not an independent-geometry NOISE report")
    rows = []
    seen = set()
    seen_cells = set()
    for cell_value in _sequence(summary.get("cells"), context="NOISE cells"):
        cell = _mapping(cell_value, context="NOISE cell")
        cell_id = str(cell.get("cell", ""))
        dataset = str(cell.get("dataset", ""))
        model = str(cell.get("model", ""))
        if cell_id != f"{dataset}--{model}":
            raise ArtifactError(f"Invalid NOISE cell identity {cell_id!r}")
        if cell_id in seen_cells:
            raise ArtifactError(f"Duplicate NOISE cell {cell_id!r}")
        seen_cells.add(cell_id)
        cell_geometries = set()
        for geometry_value in _sequence(cell.get("geometries"), context=f"{cell_id} geometries"):
            geometry = _mapping(geometry_value, context=f"{cell_id} geometry")
            geometry_id = str(geometry.get("geometry", ""))
            geometry_label = str(geometry.get("geometry_label", ""))
            q = int(geometry.get("q", 0))
            if geometry_id not in {"spearman", "kendall"} or q < 2 or q > 11:
                raise ArtifactError(f"Invalid NOISE geometry for {cell_id}")
            if geometry_id in cell_geometries:
                raise ArtifactError(f"Duplicate NOISE geometry {cell_id}/{geometry_id}")
            cell_geometries.add(geometry_id)
            geometry_rules = set()
            for row_value in _sequence(
                geometry.get("rows"), context=f"{cell_id}/{geometry_id} rows"
            ):
                row = _mapping(row_value, context=f"{cell_id}/{geometry_id} row")
                rule = str(row.get("method", ""))
                key = (cell_id, geometry_id, rule)
                if rule not in PAPER_RULES or key in seen:
                    raise ArtifactError(f"Invalid or duplicate NOISE row {key}")
                seen.add(key)
                geometry_rules.add(rule)
                perturbed_source = _mapping(
                    row.get("perturbed_quality"), context=f"{key} perturbed quality"
                )
                if set(perturbed_source) != set(NOISE_ORDER):
                    raise ArtifactError(f"Incomplete NOISE conditions for {key}")
                rows.append(
                    {
                        "cell": cell_id,
                        "dataset": dataset,
                        "model": model,
                        "setting": str(row.get("setting", geometry_label)),
                        "geometry": geometry_id,
                        "geometry_label": geometry_label,
                        "q": q,
                        "rule": rule,
                        "methods": tuple(str(value) for value in row.get("methods", ())),
                        "quality": _finite_metrics(row.get("quality"), context=f"{key} quality"),
                        "perturbed_quality": {
                            noise: _finite_metrics(
                                perturbed_source[noise], context=f"{key}/{noise} quality"
                            )
                            for noise in NOISE_ORDER
                        },
                    }
                )
            if geometry_rules != set(PAPER_RULES):
                raise ArtifactError(f"Incomplete NOISE rule bank for {cell_id}/{geometry_id}")
        if cell_geometries != {"spearman", "kendall"}:
            raise ArtifactError(f"Incomplete NOISE geometry bank for {cell_id}")
    expected_rows = len(seen_cells) * 2 * len(PAPER_RULES)
    expected_endpoints = expected_rows * len(NOISE_ORDER) * len(QUALITY_METRICS)
    observed_endpoints = len(seen) * len(NOISE_ORDER) * len(QUALITY_METRICS)
    if not seen_cells or len(rows) != expected_rows or observed_endpoints != expected_endpoints:
        raise ArtifactError(
            "Incomplete NOISE coverage: "
            f"rows={len(rows)}/{expected_rows}, endpoints={observed_endpoints}/{expected_endpoints}"
        )
    return tuple(rows)


def _endpoint_record(
    *,
    source_kind: str,
    cell: str,
    dataset: str,
    model: str,
    setting: str,
    geometry: str | None,
    geometry_label: str | None,
    q: int,
    rule: str,
    methods: Sequence[str],
    noise: str,
    metric: str,
    clean_quality: float,
    perturbed_quality: float,
    accuracies: Mapping[str, float],
) -> Mapping[str, Any]:
    values = quality_retention_endpoint(
        metric=metric,
        clean_quality=clean_quality,
        perturbed_quality=perturbed_quality,
        clean_unmasked_accuracy=accuracies["clean"],
        perturbed_unmasked_accuracy=accuracies[noise],
    )
    identity = {
        "source_kind": source_kind,
        "cell": cell,
        "geometry": geometry,
        "q": q,
        "rule": rule,
        "noise": noise,
        "metric": metric,
    }
    return {
        "endpoint_id": object_sha256(identity),
        **identity,
        "dataset": dataset,
        "model": model,
        "setting": setting,
        "geometry_label": geometry_label,
        "methods": list(methods),
        "condition": NOISE_NAMES[noise],
        **values,
    }


def _outcome(candidate: float | None, baseline: float | None) -> str | None:
    if candidate is None or baseline is None:
        return None
    difference = candidate - baseline
    if difference > COMPARISON_TOLERANCE:
        return "win"
    if difference < -COMPARISON_TOLERANCE:
        return "loss"
    return "tie"


def _comparison_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    result = {}
    for measure in ("A", "Q", "G"):
        usable = [
            row
            for row in rows
            if row[f"NOISE_{measure}"] is not None and row[f"NAIVE_{measure}"] is not None
        ]
        differences = [float(row[f"delta_{measure}"]) for row in usable]
        outcomes = [str(row[f"outcome_{measure}"]) for row in usable]
        result[measure] = {
            "endpoints": len(usable),
            "excluded_zero_clean_boundaries": len(rows) - len(usable),
            "wins": outcomes.count("win"),
            "ties": outcomes.count("tie"),
            "losses": outcomes.count("loss"),
            "win_rate": outcomes.count("win") / len(usable) if usable else None,
            "mean_NOISE": sum(float(row[f"NOISE_{measure}"]) for row in usable) / len(usable)
            if usable
            else None,
            "mean_NAIVE": sum(float(row[f"NAIVE_{measure}"]) for row in usable) / len(usable)
            if usable
            else None,
            "mean_difference": sum(differences) / len(differences) if differences else None,
            "median_difference": median(differences) if differences else None,
        }
    return result


def _grouped_analysis(comparisons: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    dimensions = {
        "geometry": "geometry_label",
        "cell": "cell",
        "rule": "rule",
        "metric": "metric",
        "noise": "noise",
    }
    analysis: dict[str, Any] = {"overall": _comparison_summary(comparisons)}
    for output_name, field in dimensions.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in comparisons:
            groups[str(row[field])].append(row)
        analysis[f"by_{output_name}"] = {
            value: _comparison_summary(rows) for value, rows in sorted(groups.items())
        }
    return analysis


def build_quality_retention_summary(
    candidate_summary: Mapping[str, Any],
    *,
    reference_summary: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Mapping[str, Any]]],
    accuracies: Mapping[str, Mapping[str, float]],
    input_provenance: Mapping[str, Any] | None = None,
    reference_provenance: Mapping[str, Any] | None = None,
    accuracy_provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build complete NOISE and q=11 NAIVE A/Q/G endpoint comparisons."""

    if reference_summary.get("table") != "Table_1":
        raise ArtifactError("Reference summary must be the original NAIVE Table 1")
    candidate_rows = _extract_noise_rows(candidate_summary)
    reference_cells = {}
    for cell_value in _sequence(reference_summary.get("cells"), context="Table 1 cells"):
        cell = _mapping(cell_value, context="Table 1 cell")
        cell_id = f"{cell.get('dataset')}--{cell.get('model')}"
        if cell_id in reference_cells:
            raise ArtifactError(f"Duplicate Table 1 cell {cell_id}")
        reference_cells[cell_id] = cell
    candidate_cells = {str(row["cell"]) for row in candidate_rows}
    if (
        candidate_cells != set(reference_cells)
        or candidate_cells != set(manifests)
        or candidate_cells != set(accuracies)
    ):
        raise ArtifactError("NOISE, Table 1, manifest, and A0 cell sets differ")

    naive_endpoints: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for cell_id in sorted(candidate_cells):
        cell = reference_cells[cell_id]
        cell_manifests = manifests[cell_id]
        cell_accuracies = accuracies[cell_id]
        if set(cell_manifests) != {"clean", *NOISE_ORDER} or set(cell_accuracies) != {
            "clean",
            *NOISE_ORDER,
        }:
            raise ArtifactError(f"Incomplete manifest or A0 conditions for {cell_id}")
        clean_metrics = _mapping(
            cell_manifests["clean"].get("metrics"), context=f"{cell_id}/clean metrics"
        )
        if not set(PAPER_RULES).issubset(clean_metrics):
            raise ArtifactError(f"Incomplete q=11 clean rule bank for {cell_id}")
        for rule in PAPER_RULES:
            clean = _finite_metrics(clean_metrics[rule], context=f"{cell_id}/clean/{rule}")
            for noise in NOISE_ORDER:
                perturbed_metrics = _mapping(
                    cell_manifests[noise].get("metrics"), context=f"{cell_id}/{noise} metrics"
                )
                if rule not in perturbed_metrics:
                    raise ArtifactError(f"Missing q=11 rule {cell_id}/{noise}/{rule}")
                perturbed = _finite_metrics(
                    perturbed_metrics[rule], context=f"{cell_id}/{noise}/{rule}"
                )
                for metric in QUALITY_METRICS:
                    key = (cell_id, rule, noise, metric)
                    naive_endpoints[key] = _endpoint_record(
                        source_kind="NAIVE",
                        cell=cell_id,
                        dataset=str(cell["dataset"]),
                        model=str(cell["model"]),
                        setting="q11-naive",
                        geometry=None,
                        geometry_label=None,
                        q=11,
                        rule=rule,
                        methods=tuple(str(value) for value in cell.get("methods", ())),
                        noise=noise,
                        metric=metric,
                        clean_quality=clean[metric],
                        perturbed_quality=perturbed[metric],
                        accuracies=cell_accuracies,
                    )

    noise_endpoints = []
    comparisons = []
    for row in candidate_rows:
        cell_id = str(row["cell"])
        for noise in NOISE_ORDER:
            for metric in QUALITY_METRICS:
                candidate = _endpoint_record(
                    source_kind="NOISE",
                    cell=cell_id,
                    dataset=str(row["dataset"]),
                    model=str(row["model"]),
                    setting=str(row["setting"]),
                    geometry=str(row["geometry"]),
                    geometry_label=str(row["geometry_label"]),
                    q=int(row["q"]),
                    rule=str(row["rule"]),
                    methods=tuple(str(value) for value in row["methods"]),
                    noise=noise,
                    metric=metric,
                    clean_quality=float(row["quality"][metric]),
                    perturbed_quality=float(row["perturbed_quality"][noise][metric]),
                    accuracies=accuracies[cell_id],
                )
                baseline = naive_endpoints[(cell_id, str(row["rule"]), noise, metric)]
                noise_endpoints.append(candidate)
                comparison: dict[str, Any] = {
                    "comparison_id": object_sha256(
                        {
                            "candidate_endpoint_id": candidate["endpoint_id"],
                            "baseline_endpoint_id": baseline["endpoint_id"],
                        }
                    ),
                    "candidate_endpoint_id": candidate["endpoint_id"],
                    "baseline_endpoint_id": baseline["endpoint_id"],
                    "cell": cell_id,
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "setting": row["setting"],
                    "geometry": row["geometry"],
                    "geometry_label": row["geometry_label"],
                    "q": row["q"],
                    "rule": row["rule"],
                    "noise": noise,
                    "condition": NOISE_NAMES[noise],
                    "metric": metric,
                }
                for measure in ("A", "Q", "G"):
                    candidate_value = candidate[measure]
                    baseline_value = baseline[measure]
                    comparison[f"NOISE_{measure}"] = candidate_value
                    comparison[f"NAIVE_{measure}"] = baseline_value
                    comparison[f"delta_{measure}"] = (
                        None
                        if candidate_value is None or baseline_value is None
                        else float(candidate_value) - float(baseline_value)
                    )
                    comparison[f"outcome_{measure}"] = _outcome(candidate_value, baseline_value)
                comparisons.append(comparison)

    all_endpoints = [*noise_endpoints, *naive_endpoints.values()]
    expected_noise_endpoints = len(candidate_rows) * len(NOISE_ORDER) * len(QUALITY_METRICS)
    expected_naive_endpoints = (
        len(candidate_cells) * len(PAPER_RULES) * len(NOISE_ORDER) * len(QUALITY_METRICS)
    )
    if (
        len(noise_endpoints) != expected_noise_endpoints
        or len(naive_endpoints) != expected_naive_endpoints
        or len(comparisons) != expected_noise_endpoints
    ):
        raise ArtifactError("Unexpected quality-retention endpoint coverage")
    analysis = _grouped_analysis(comparisons)
    boundary_counts = {
        "normalized_values_clipped": sum(
            int(row["clean_clipped"]) + int(row["perturbed_clipped"]) for row in all_endpoints
        ),
        "endpoints_with_any_clipping": sum(
            bool(row["clean_clipped"] or row["perturbed_clipped"]) for row in all_endpoints
        ),
        "retention_caps": sum(bool(row["retention_capped"]) for row in all_endpoints),
        "zero_clean_boundaries": sum(bool(row["zero_clean_boundary"]) for row in all_endpoints),
    }
    value: dict[str, Any] = {
        "schema": QUALITY_RETENTION_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "policy": QUALITY_RETENTION_POLICY,
        "direction": "max",
        "science": {
            "normalization": {
                "F": "clip(F / A0, 0, 1)",
                "Fbar": "clip(1 - Fbar / A0, 0, 1)",
                "C": "clip(C, 0, 1)",
                "Cbar": "clip(1 - Cbar, 0, 1)",
                "A0": "condition-specific mean(unmasked_predictions == labels)",
            },
            "A": "normalized perturbed quality",
            "Q": "min(1, normalized perturbed quality / normalized clean quality)",
            "G": "sqrt(A * Q)",
            "improvement_policy": "retention capped at one; improvements remain represented by A",
            "zero_clean_policy": "Q and G are null; no epsilon is introduced",
            "comparison": "paired NOISE versus same-cell same-rule q=11 NAIVE",
            "comparison_tolerance": COMPARISON_TOLERANCE,
            "inference_rerun": False,
        },
        "counts": {
            "cells": len(candidate_cells),
            "noise_candidate_rows": len(candidate_rows),
            "noise_endpoints": len(noise_endpoints),
            "unique_naive_endpoints": len(naive_endpoints),
            "paired_comparisons": len(comparisons),
        },
        "boundaries": boundary_counts,
        "sources": {
            "candidate": dict(input_provenance or {}),
            "reference": dict(reference_provenance or {}),
            "accuracy": dict(accuracy_provenance or {}),
        },
        "accuracies": {cell: dict(values) for cell, values in sorted(accuracies.items())},
        "quality_retention": all_endpoints,
        "comparisons": comparisons,
        "analysis": analysis,
    }
    value["result_digest"] = object_sha256(value)
    return value


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        value = dict(row)
        if "methods" in value:
            value["methods"] = "|".join(str(item) for item in value["methods"])
        writer.writerow(value)
    return buffer.getvalue()


def _analysis_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = []
    analysis = _mapping(summary["analysis"], context="analysis")
    groups = [("overall", "all", analysis["overall"])]
    for dimension in ("geometry", "cell", "rule", "metric", "noise"):
        for value, metrics in _mapping(analysis[f"by_{dimension}"], context=dimension).items():
            groups.append((dimension, str(value), metrics))
    for dimension, value, metrics_value in groups:
        metrics = _mapping(metrics_value, context=f"{dimension}/{value}")
        for measure in ("A", "Q", "G"):
            rows.append(
                {
                    "group_dimension": dimension,
                    "group_value": value,
                    "measure": measure,
                    **_mapping(metrics[measure], context=f"{dimension}/{value}/{measure}"),
                }
            )
    return tuple(rows)


def _row_summary_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Aggregate each selected NOISE row over its 16 metric/condition endpoints."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _sequence(summary["comparisons"], context="comparisons"):
        groups[(str(row["cell"]), str(row["geometry_label"]), str(row["rule"]))].append(row)
    output = []
    for (cell, geometry_label, rule), rows in sorted(groups.items()):
        first = rows[0]
        record: dict[str, Any] = {
            "cell": cell,
            "dataset": first["dataset"],
            "model": first["model"],
            "setting": first["setting"],
            "geometry": first["geometry"],
            "geometry_label": geometry_label,
            "q": first["q"],
            "rule": rule,
            "endpoint_count": len(rows),
        }
        for measure in ("A", "Q", "G"):
            usable = [row for row in rows if row[f"NOISE_{measure}"] is not None]
            candidate_values = [float(row[f"NOISE_{measure}"]) for row in usable]
            naive_values = [float(row[f"NAIVE_{measure}"]) for row in usable]
            differences = [float(row[f"delta_{measure}"]) for row in usable]
            outcomes = [str(row[f"outcome_{measure}"]) for row in usable]
            record.update(
                {
                    f"NOISE_{measure}_mean": sum(candidate_values) / len(candidate_values)
                    if candidate_values
                    else None,
                    f"NAIVE_{measure}_mean": sum(naive_values) / len(naive_values)
                    if naive_values
                    else None,
                    f"delta_{measure}_mean": sum(differences) / len(differences)
                    if differences
                    else None,
                    f"{measure}_wins": outcomes.count("win"),
                    f"{measure}_ties": outcomes.count("tie"),
                    f"{measure}_losses": outcomes.count("loss"),
                }
            )
        output.append(record)
    return tuple(output)


def _readme(summary: Mapping[str, Any]) -> str:
    overall = summary["analysis"]["overall"]
    geometric = overall["G"]
    return "\n".join(
        (
            "# Absolute Quality and Clean Retention",
            "",
            "This is deterministic offline post-processing of completed Phase 2 artifacts. No attribution or model inference was rerun.",
            "",
            "Each raw metric is first converted to a condition-specific normalized quality in [0, 1], where one is best. F and Fbar use the unmasked reference accuracy A0 recovered from that condition's verified Phase 2 shards; C and Cbar are already rates.",
            "",
            "- `A` is normalized quality under perturbation.",
            "- `Q` is `min(1, A / normalized_clean_quality)`, the retained fraction of clean quality.",
            "- `G` is `sqrt(A * Q)`, the equal-weight geometric mean of absolute perturbed quality and clean-quality retention.",
            "",
            "A perturbation-induced improvement caps Q at one so it is rewarded through A only. If normalized clean quality is zero, Q and G are null instead of using an arbitrary epsilon.",
            "",
            "Every NOISE endpoint is compared with q=11 NAIVE for the same dataset, model, aggregation rule, perturbation, and metric. Results are descriptive complete-test-set comparisons; they are not a statistical significance test.",
            "",
            f"Overall G comparison: NOISE wins/ties/losses = {geometric['wins']}/{geometric['ties']}/{geometric['losses']}, mean difference = {geometric['mean_difference']:.12g}.",
            "",
            "Files:",
            "",
            "- `quality_retention.csv`: all 640 NOISE endpoints and 320 unique q=11 NAIVE endpoints.",
            "- `comparisons.csv`: all 640 strictly matched endpoint comparisons.",
            "- `row_summary.csv`: each selected NOISE row averaged over its 16 endpoints.",
            "- `comparison_summary.csv`: overall and grouped A/Q/G summaries.",
            "- `summary.json`: formulas, provenance, A0 evidence digests, boundaries, endpoints, and analysis.",
            "",
            f"Result digest: `{summary['result_digest']}`.",
            "",
        )
    )


def write_quality_retention_outputs(
    summary: Mapping[str, Any], *, output_directory: str | Path
) -> Mapping[str, Any]:
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    endpoint_columns = (
        "endpoint_id",
        "source_kind",
        "cell",
        "dataset",
        "model",
        "setting",
        "geometry",
        "geometry_label",
        "q",
        "rule",
        "methods",
        "noise",
        "condition",
        "metric",
        "clean_quality_raw",
        "perturbed_quality_raw",
        "A0_clean",
        "A0_perturbed",
        "normalized_clean_raw",
        "normalized_perturbed_raw",
        "Qnorm_clean",
        "Qnorm_perturbed",
        "A",
        "Q",
        "Q_raw",
        "G",
        "clean_clipped",
        "perturbed_clipped",
        "retention_capped",
        "zero_clean_boundary",
    )
    comparison_columns = (
        "comparison_id",
        "candidate_endpoint_id",
        "baseline_endpoint_id",
        "cell",
        "dataset",
        "model",
        "setting",
        "geometry",
        "geometry_label",
        "q",
        "rule",
        "noise",
        "condition",
        "metric",
        "NOISE_A",
        "NAIVE_A",
        "delta_A",
        "outcome_A",
        "NOISE_Q",
        "NAIVE_Q",
        "delta_Q",
        "outcome_Q",
        "NOISE_G",
        "NAIVE_G",
        "delta_G",
        "outcome_G",
    )
    analysis_columns = (
        "group_dimension",
        "group_value",
        "measure",
        "endpoints",
        "excluded_zero_clean_boundaries",
        "wins",
        "ties",
        "losses",
        "win_rate",
        "mean_NOISE",
        "mean_NAIVE",
        "mean_difference",
        "median_difference",
    )
    row_columns = (
        "cell",
        "dataset",
        "model",
        "setting",
        "geometry",
        "geometry_label",
        "q",
        "rule",
        "endpoint_count",
        "NOISE_A_mean",
        "NAIVE_A_mean",
        "delta_A_mean",
        "A_wins",
        "A_ties",
        "A_losses",
        "NOISE_Q_mean",
        "NAIVE_Q_mean",
        "delta_Q_mean",
        "Q_wins",
        "Q_ties",
        "Q_losses",
        "NOISE_G_mean",
        "NAIVE_G_mean",
        "delta_G_mean",
        "G_wins",
        "G_ties",
        "G_losses",
    )
    paths = {
        "summary_json": atomic_write_json(destination / "summary.json", summary),
        "quality_retention_csv": atomic_write_text(
            destination / "quality_retention.csv",
            _csv_text(summary["quality_retention"], endpoint_columns),
        ),
        "comparisons_csv": atomic_write_text(
            destination / "comparisons.csv", _csv_text(summary["comparisons"], comparison_columns)
        ),
        "comparison_summary_csv": atomic_write_text(
            destination / "comparison_summary.csv",
            _csv_text(_analysis_rows(summary), analysis_columns),
        ),
        "row_summary_csv": atomic_write_text(
            destination / "row_summary.csv",
            _csv_text(_row_summary_rows(summary), row_columns),
        ),
        "readme": atomic_write_text(destination / "README.md", _readme(summary)),
    }
    return {
        "status": "complete",
        "result_digest": summary["result_digest"],
        **summary["counts"],
        **{key: str(path) for key, path in paths.items()},
    }


def write_quality_retention(
    *,
    candidate_summary_path: str | Path,
    reference_summary_path: str | Path,
    experiment: SimpleExperiment,
    manifest_source: ManifestSource,
    temporary_directory: str | Path,
    download_workers: int,
    output_directory: str | Path,
) -> Mapping[str, Any]:
    """Load completed artifacts, calculate A/Q/G, and write Git-sized outputs."""

    candidate_path = Path(candidate_summary_path).expanduser().resolve()
    reference_path = Path(reference_summary_path).expanduser().resolve()
    candidate_summary = read_json(candidate_path)
    reference_summary = read_json(reference_path)
    if not isinstance(candidate_summary, Mapping) or not isinstance(reference_summary, Mapping):
        raise ArtifactError("Quality-retention inputs must be JSON mappings")
    manifests, accuracies, accuracy_provenance = load_reference_manifests_and_accuracies(
        reference_summary,
        experiment=experiment,
        manifest_source=manifest_source,
        temporary_directory=temporary_directory,
        download_workers=download_workers,
    )
    summary = build_quality_retention_summary(
        candidate_summary,
        reference_summary=reference_summary,
        manifests=manifests,
        accuracies=accuracies,
        input_provenance={
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
            "result_digest": candidate_summary.get("result_digest"),
        },
        reference_provenance={
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "summary_digest": reference_summary.get("summary_digest"),
        },
        accuracy_provenance=accuracy_provenance,
    )
    return write_quality_retention_outputs(summary, output_directory=output_directory)


__all__ = [
    "QUALITY_RETENTION_POLICY",
    "QUALITY_RETENTION_SCHEMA",
    "build_quality_retention_summary",
    "load_reference_manifests_and_accuracies",
    "normalize_quality",
    "quality_retention_endpoint",
    "recover_unmasked_accuracy",
    "write_quality_retention",
    "write_quality_retention_outputs",
]
