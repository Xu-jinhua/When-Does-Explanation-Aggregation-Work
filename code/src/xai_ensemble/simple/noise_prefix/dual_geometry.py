"""Dual-geometry NOISE selector and post-hoc report.

Selection and evaluation deliberately have separate entry points.  The
selector reads individual clean-F marks and rank-only candidate centers.  It
does not accept the q-sweep summary or read q-level masked predictions.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, atomic_write_text, read_json
from xai_ensemble.phase2.metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS
from xai_ensemble.phase2.selection import order_methods_by_fidelity

from ..artifacts import PHASE2_SCHEMA_VERSION
from ..robustness import (
    SIGNED_ROBUSTNESS_DIRECTION,
    SIGNED_ROBUSTNESS_POLICY,
    SIGNED_ROBUSTNESS_SOURCE,
)
from ..summary import NOISE_NAMES, NOISE_ORDER, PAPER_RULES
from .anchored import (
    combine_fidelity_anchored_prefixes,
    fidelity_anchored_prefix,
    select_joint_fidelity_anchored_prefix,
    topk_set_distances,
)
from .config import NoisePrefixExperiment

EXPERIMENT_ID = "dual-geometry-noise-v1"
SCOPE = "post_hoc_complete_test_set_boundary_experiment"
PRIMARY_GEOMETRIES = ("borda", "kemeny")
BASE_RULE_FIELDS = {"borda": "r001", "kemeny": "r003"}
TRANSFER_FIDELITY_REGRET_TOLERANCE = 0.005
REPORT_COLUMNS = (
    *QUALITY_METRICS,
    *(f"R_{metric}_{noise}" for noise in NOISE_ORDER for metric in QUALITY_METRICS),
)
PERTURBED_REPORT_COLUMNS = tuple(
    f"{metric}_perturbed_{noise}" for noise in NOISE_ORDER for metric in QUALITY_METRICS
)
COLUMN_DIRECTIONS = {
    **{metric: DEFAULT_METRIC_DIRECTIONS[metric] for metric in QUALITY_METRICS},
    **{column: SIGNED_ROBUSTNESS_DIRECTION for column in REPORT_COLUMNS if column.startswith("R_")},
}


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _stack(values: Sequence[np.ndarray], *, context: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No arrays collected for {context}")
    return np.concatenate(values, axis=0)


def _verified_digest(value: Mapping[str, Any], *, key: str, context: str) -> str:
    recorded = value.get(key)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(f"{context} has no valid {key}")
    payload = {name: item for name, item in value.items() if name != key}
    actual = object_sha256(payload)
    if recorded != actual:
        raise ValueError(f"{context} digest mismatch: recorded={recorded}, actual={actual}")
    return recorded


def _prefix_root(cache: Path, cell_id: str) -> Path:
    parent = cache / cell_id / "clean" / "p16" / "k20"
    roots = tuple(path for path in parent.iterdir() if path.is_dir())
    if len(roots) != 1:
        raise ValueError(f"Expected one clean prefix root for {cell_id}; found {len(roots)}")
    return roots[0]


def _base_source_records(
    cache: Path,
    *,
    cell: Any,
    expected_task_digest: str,
    patch_size: int,
    k: int,
) -> tuple[tuple[tuple[Path, Mapping[str, str]], ...], str | None]:
    """Resolve a verified nested Phase 2 manifest or the legacy flat cache."""

    methods = tuple(cell.methods)
    matches: list[tuple[Path, Mapping[str, Any]]] = []
    for path in cache.rglob("manifest.json"):
        value = read_json(path)
        if isinstance(value, Mapping) and value.get("task_digest") == expected_task_digest:
            matches.append((path.parent, value))
    if len(matches) > 1:
        raise ValueError(f"Multiple base manifests match {cell.cell_id}")

    if matches:
        root, manifest = matches[0]
        expected = {
            "schema_version": PHASE2_SCHEMA_VERSION,
            "status": "complete",
            "task_digest": expected_task_digest,
            "dataset": cell.dataset.dataset_id,
            "model": cell.reference_model.model_id,
            "split": "test",
            "condition": "clean",
            "methods": list(methods),
            "patch_size": patch_size,
            "k": k,
            "rank_base": 0,
        }
        contradictions = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if contradictions:
            raise ValueError(
                f"Base manifest has incompatible semantics for {cell.cell_id}: {contradictions}"
            )
        shards_value = manifest.get("shards")
        if not isinstance(shards_value, Sequence) or isinstance(shards_value, (str, bytes)):
            raise TypeError(f"Base manifest shards must be a sequence for {cell.cell_id}")
        ordered = sorted(
            (_mapping(value, context="base shard record") for value in shards_value),
            key=lambda value: int(value["shard_index"]),
        )
        if not ordered:
            raise ValueError(f"Base manifest has no shards for {cell.cell_id}")
        records = []
        expected_start = 0
        seen_indices = set()
        required_labels = {
            *(f"single__{method}" for method in methods),
            *PRIMARY_GEOMETRIES,
        }
        for record in ordered:
            shard_index = int(record["shard_index"])
            start = int(record["start"])
            stop = int(record["stop"])
            if (
                shard_index in seen_indices
                or shard_index != len(seen_indices)
                or start != expected_start
                or stop <= start
                or int(record.get("count", -1)) != stop - start
                or record.get("task_digest") != expected_task_digest
            ):
                raise ValueError(f"Base shard layout is invalid for {cell.cell_id}")
            seen_indices.add(shard_index)
            expected_start = stop
            labels = _mapping(record.get("rule_labels"), context="base rule labels")
            by_name = {str(name): str(field) for field, name in labels.items()}
            if len(by_name) != len(labels) or not required_labels.issubset(by_name):
                missing = sorted(required_labels.difference(by_name))
                raise ValueError(f"Base shard is missing rule labels for {cell.cell_id}: {missing}")
            payload = _mapping(record.get("payload"), context="base shard payload")
            relative_path = str(payload.get("relative_path", ""))
            path = root / "shards" / Path(relative_path).name
            if not path.is_file():
                raise FileNotFoundError(f"Base shard is absent from the local cache: {path}")
            expected_size = int(payload.get("size_bytes", -1))
            if expected_size != path.stat().st_size or payload.get("sha256") != file_sha256(path):
                raise ValueError(f"Base shard payload does not match its manifest: {path}")
            records.append((path, by_name))
        if expected_start != int(manifest.get("sample_count", -1)):
            raise ValueError(f"Base shard coverage is incomplete for {cell.cell_id}")
        return tuple(records), object_sha256(manifest)

    shard_paths = tuple(sorted(cache.glob(f"{cell.cell_id}--shard*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No base clean shards for {cell.cell_id} in {cache}")
    fallback_labels = {
        **BASE_RULE_FIELDS,
        **{
            f"single__{method}": f"r{position:03d}"
            for position, method in enumerate(methods, start=5)
        },
    }
    return tuple((path, fallback_labels) for path in shard_paths), None


def _base_selector_inputs(
    cache: Path,
    *,
    cell: Any,
    expected_task_digest: str,
    patch_size: int,
    k: int,
) -> Mapping[str, Any]:
    """Read only q=11 ranks and method-level clean-F sufficient statistics."""

    methods = tuple(cell.methods)
    source_records, manifest_content_digest = _base_source_records(
        cache,
        cell=cell,
        expected_task_digest=expected_task_digest,
        patch_size=patch_size,
        k=k,
    )
    labels = []
    predictions = []
    indices = []
    ballots = []
    contributions = []
    centers = {rule: [] for rule in PRIMARY_GEOMETRIES}
    task_digest = None
    for path, by_name in source_records:
        with safe_open(path, framework="np") as shard:
            metadata = shard.metadata() or {}
            if metadata.get("patch_size") != str(patch_size) or metadata.get("rank_base") != "0":
                raise ValueError(f"Base shard has incompatible rank semantics: {path}")
            current_digest = str(metadata.get("task_digest", ""))
            if current_digest != expected_task_digest:
                raise ValueError(f"Base shard has the wrong task digest: {path}")
            task_digest = current_digest if task_digest is None else task_digest
            if current_digest != task_digest:
                raise ValueError(f"Base task digest changed across {cell.cell_id}")
            current_labels = shard.get_tensor("labels").astype(np.int64, copy=False)
            current_predictions = shard.get_tensor("unmasked_predictions").astype(
                np.int64, copy=False
            )
            clean_correct = (current_predictions == current_labels).astype(np.int8)
            method_ballots = []
            method_contributions = []
            for method in methods:
                field = by_name[f"single__{method}"]
                method_ballots.append(shard.get_tensor(f"rank__{field}"))
                removed = shard.get_tensor(f"removed_predictions__{field}")
                method_contributions.append(
                    clean_correct - (removed == current_labels).astype(np.int8)
                )
            indices.append(shard.get_tensor("indices").astype(np.int64, copy=False))
            labels.append(current_labels)
            predictions.append(current_predictions)
            ballots.append(np.stack(method_ballots, axis=1).astype(np.int16, copy=False))
            contributions.append(np.stack(method_contributions, axis=1))
            for rule in PRIMARY_GEOMETRIES:
                field = by_name[rule]
                rank = shard.get_tensor(f"rank__{field}")
                centers[rule].append(np.argsort(rank, axis=1, kind="stable")[:, :k])

    row_ids = _stack(indices, context="base indices")
    order = np.argsort(row_ids, kind="stable")
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError(f"Base sample ids are not unique for {cell.cell_id}")
    values: dict[str, Any] = {
        "indices": row_ids[order],
        "labels": _stack(labels, context="base labels")[order],
        "predictions": _stack(predictions, context="base predictions")[order],
        "ballots": _stack(ballots, context="base ballots")[order],
        "contributions": _stack(contributions, context="base contributions")[order],
        "q11_centers": {
            rule: _stack(parts, context=f"{rule} q11 centers")[order]
            for rule, parts in centers.items()
        },
        "task_digest": task_digest,
        "manifest_content_digest": manifest_content_digest,
    }
    fidelity = {
        method: float(values["contributions"][:, position].mean())
        for position, method in enumerate(methods)
    }
    method_order, ordered_methods = order_methods_by_fidelity(methods, fidelity)
    values["fidelity"] = fidelity
    values["ordered_methods"] = tuple(ordered_methods)
    values["ballots"] = values["ballots"][:, method_order]
    values["contributions"] = values["contributions"][:, method_order]
    return values


def _prefix_selector_inputs(
    cache: Path,
    *,
    cell: Any,
    q_values: Sequence[int],
    expected_task_digest: str,
    sweep_id: str,
    sweep_digest: str,
    patch_size: int,
    k: int,
) -> Mapping[str, Any]:
    """Read rank-only candidate centers; never read q-level model outputs."""

    cell_id = str(cell.cell_id)
    root = _prefix_root(cache, cell_id)
    manifest = _mapping(read_json(root / "manifest.json"), context="prefix manifest")
    expected = {
        "schema": "simple-noise-prefix-evaluation-v1",
        "schema_version": 1,
        "status": "complete",
        "sweep_id": sweep_id,
        "sweep_digest": sweep_digest,
        "task_digest": expected_task_digest,
        "cell": cell_id,
        "dataset": cell.dataset.dataset_id,
        "model": cell.reference_model.model_id,
        "split": "test",
        "condition": "clean",
        "patch_size": patch_size,
        "k": k,
        "q_values": list(q_values),
        "computed_q_values": list(q_values[:-1]),
        "rank_payload": "top_k_patch_indices_only_no_full_consensus_rank",
    }
    contradictions = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if contradictions:
        raise ValueError(f"Prefix manifest has incompatible semantics: {contradictions}")
    computed_q = tuple(q for q in q_values if q < max(q_values))
    indices = []
    centers = {rule: {q: [] for q in computed_q} for rule in PRIMARY_GEOMETRIES}
    shards_value = manifest.get("shards")
    if not isinstance(shards_value, Sequence) or isinstance(shards_value, (str, bytes)):
        raise TypeError(f"Prefix manifest shards must be a sequence for {cell_id}")
    ordered = sorted(
        (_mapping(value, context="prefix shard record") for value in shards_value),
        key=lambda value: int(value["shard_index"]),
    )
    expected_start = 0
    for expected_index, record_value in enumerate(ordered):
        record = _mapping(record_value, context="prefix shard record")
        start = int(record["start"])
        stop = int(record["stop"])
        if (
            int(record["shard_index"]) != expected_index
            or record.get("task_digest") != expected_task_digest
            or start != expected_start
            or stop <= start
            or int(record.get("count", -1)) != stop - start
        ):
            raise ValueError(f"Prefix shard layout is invalid for {cell_id}")
        expected_start = stop
        payload = _mapping(record["payload"], context="prefix payload")
        path = root / "shards" / Path(str(payload["relative_path"])).name
        if (
            not path.is_file()
            or int(payload.get("size_bytes", -1)) != path.stat().st_size
            or file_sha256(path) != payload.get("sha256")
        ):
            raise ValueError(f"Prefix shard digest mismatch: {path}")
        labels = _mapping(record["rule_labels"], context="prefix rule labels")
        by_name = {str(name): str(field) for field, name in labels.items()}
        required = {f"q{q:02d}__{rule}" for rule in PRIMARY_GEOMETRIES for q in computed_q}
        if len(by_name) != len(labels) or not required.issubset(by_name):
            raise ValueError(f"Prefix shard is missing required rank labels for {cell_id}")
        with safe_open(path, framework="np") as shard:
            metadata = shard.metadata() or {}
            if (
                metadata.get("task_digest") != expected_task_digest
                or metadata.get("patch_size") != str(patch_size)
                or metadata.get("k") != str(k)
                or metadata.get("rank_payload") != "top_k_only"
            ):
                raise ValueError(f"Prefix shard has incompatible metadata: {path}")
            indices.append(shard.get_tensor("indices").astype(np.int64, copy=False))
            for rule in PRIMARY_GEOMETRIES:
                for q in computed_q:
                    field = by_name[f"q{q:02d}__{rule}"]
                    centers[rule][q].append(
                        shard.get_tensor(f"top_patch_indices__{field}").astype(np.int64, copy=False)
                    )
    if expected_start != int(manifest.get("sample_count", -1)):
        raise ValueError(f"Prefix shard coverage is incomplete for {cell_id}")
    row_ids = _stack(indices, context="prefix indices")
    order = np.argsort(row_ids, kind="stable")
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError(f"Prefix sample ids are not unique for {cell_id}")
    return {
        "manifest": manifest,
        "manifest_content_digest": object_sha256(manifest),
        "indices": row_ids[order],
        "centers": {
            rule: {
                q: _stack(parts, context=f"{rule} q{q} centers")[order] for q, parts in by_q.items()
            }
            for rule, by_q in centers.items()
        },
    }


def build_dual_geometry_selector(
    experiment: NoisePrefixExperiment,
    *,
    base_clean_cache: str | Path,
    prefix_clean_cache: str | Path,
) -> Mapping[str, Any]:
    """Select one q per cell without accepting or reading q-level metrics."""

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
        manifest = prefix["manifest"]
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
        candidates_by_rule = {}
        for rule in PRIMARY_GEOMETRIES:
            centers = {
                **prefix["centers"][rule],
                max(experiment.q_values): base["q11_centers"][rule],
            }
            candidates_by_rule[rule] = tuple(
                fidelity_anchored_prefix(
                    topk_set_distances(base["ballots"], centers[q], k=experiment.k),
                    base["contributions"],
                    q=q,
                    n_items=base["ballots"].shape[2],
                    k=experiment.k,
                )
                for q in experiment.q_values
            )
        joint = tuple(
            combine_fidelity_anchored_prefixes(
                tuple(candidates_by_rule[rule][position] for rule in PRIMARY_GEOMETRIES)
            )
            for position in range(len(experiment.q_values))
        )
        selected = select_joint_fidelity_anchored_prefix(joint)
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
        q = selected.q
        cells.append(
            {
                "cell": cell.cell_id,
                "dataset": cell.dataset.dataset_id,
                "model": cell.reference_model.model_id,
                "q": q,
                "ordered_methods": list(base["ordered_methods"]),
                "method_prefix": list(base["ordered_methods"][:q]),
                "source": source,
            }
        )

    payload: dict[str, Any] = {
        "schema": "simple-dual-geometry-noise-selector-v1",
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
            "selection": "maximum_unweighted_mean_borda_kemeny_anchored_score_smallest_q_tie",
            "shared_q_across_geometries": True,
            "shared_q_across_reported_aggregation_rules": True,
        },
        "selection_source_digest": object_sha256(source_rows),
        "cells": cells,
    }
    payload["selector_digest"] = object_sha256(payload)
    return payload


def load_dual_geometry_selector(path: str | Path) -> Mapping[str, Any]:
    value = _mapping(read_json(Path(path).expanduser().resolve()), context="selector")
    if (
        value.get("schema") != "simple-dual-geometry-noise-selector-v1"
        or value.get("status") != "complete"
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("scope") != SCOPE
    ):
        raise ValueError("Dual-geometry selector identity is invalid")
    _verified_digest(value, key="selector_digest", context="selector")
    contract = _mapping(value.get("selection_input_contract"), context="selection contract")
    forbidden = (
        contract.get("q_sweep_summary_accepted"),
        contract.get("q_level_aggregate_metrics_read"),
        contract.get("q_level_masked_predictions_read"),
    )
    if any(item is not False for item in forbidden):
        raise ValueError("Selector does not enforce rank-only q selection")
    return value


def _measurement_index(summary: Mapping[str, Any]) -> Mapping[tuple[Any, ...], Mapping[str, Any]]:
    index = {}
    for row_value in summary["measurements"]:
        row = _mapping(row_value, context="measurement")
        if row.get("value_kind") not in {"quality", "conditioned_quality", "robustness"}:
            continue
        key = (
            row["cell"],
            int(row["q"]),
            row["rule"],
            row["value_kind"],
            row["condition"],
            row["metric"],
        )
        if key in index:
            raise ValueError(f"Duplicate q-sweep measurement: {key}")
        index[key] = row
    return index


def _flat_values(
    measurements: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    cell: str,
    q: int,
    rule: str,
    condition_by_noise: Mapping[str, str],
    expected_methods: Sequence[str],
) -> dict[str, float]:
    values = {}
    for column in (*REPORT_COLUMNS, *PERTURBED_REPORT_COLUMNS):
        if column in QUALITY_METRICS:
            key = (cell, q, rule, "quality", "clean", column)
        elif "_perturbed_" in column:
            metric, noise = column.split("_perturbed_", maxsplit=1)
            key = (
                cell,
                q,
                rule,
                "conditioned_quality",
                condition_by_noise[noise],
                metric,
            )
        else:
            metric, noise = column.removeprefix("R_").rsplit("_", 1)
            key = (cell, q, rule, "robustness", condition_by_noise[noise], metric)
        row = measurements.get(key)
        if row is None:
            raise ValueError(f"Missing q-sweep endpoint: {key}")
        methods = json.loads(str(row["method_prefix"]))
        if methods != list(expected_methods):
            raise ValueError(f"Method prefix mismatch for {key}")
        values[column] = float(row["value"])
    return values


def _flat_nested_row(row: Mapping[str, Any]) -> dict[str, float]:
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


def _nested_values(values: Mapping[str, float]) -> Mapping[str, Any]:
    return {
        "quality": {metric: values[metric] for metric in QUALITY_METRICS},
        "perturbed_quality": {
            noise: {metric: values[f"{metric}_perturbed_{noise}"] for metric in QUALITY_METRICS}
            for noise in NOISE_ORDER
        },
        "robustness": {
            noise: {metric: values[f"R_{metric}_{noise}"] for metric in QUALITY_METRICS}
            for noise in NOISE_ORDER
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
        counts = {"better": 0, "equal": 0, "worse": 0}
        allowed = set(columns)
        for row in rows:
            if row["column"] in allowed:
                counts[str(row["outcome"])] += 1
        result[name] = counts
    return result


def _selection_validation(
    experiment: NoisePrefixExperiment,
    *,
    prefix_summary: Mapping[str, Any],
    measurements: Mapping[tuple[Any, ...], Mapping[str, Any]],
    selector_cells: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Compare the frozen shared q with clean-F oracle q for both geometries."""

    science = _mapping(prefix_summary.get("science"), context="q-sweep science")
    tie_policy = str(science.get("optimum_tie_policy", ""))
    if tie_policy != "smallest_q_within_absolute_1e-12":
        raise ValueError(f"Unsupported q-sweep optimum tie policy: {tie_policy!r}")
    optima_value = prefix_summary.get("optima")
    if not isinstance(optima_value, Sequence) or isinstance(optima_value, (str, bytes)):
        raise TypeError("q-sweep optima must be a sequence")
    optima = {}
    for value in optima_value:
        row = _mapping(value, context="q-sweep optimum")
        if (
            row.get("rule") not in PRIMARY_GEOMETRIES
            or row.get("value_kind") != "quality"
            or row.get("condition") != "clean"
            or row.get("metric") != "F"
        ):
            continue
        key = (str(row["cell"]), str(row["rule"]))
        if key in optima:
            raise ValueError(f"Duplicate clean-F oracle optimum: {key}")
        optima[key] = row
    expected_keys = {
        (cell.cell_id, rule) for cell in experiment.cells() for rule in PRIMARY_GEOMETRIES
    }
    if set(optima) != expected_keys:
        raise ValueError("clean-F oracle q coverage does not match both geometries and all cells")

    endpoints = []
    cells = []
    for cell in sorted(experiment.cells(), key=lambda value: value.cell_id):
        selected = selector_cells[cell.cell_id]
        selected_q = int(selected["q"])
        selected_methods = list(selected["method_prefix"])
        cell_rows = []
        for rule in PRIMARY_GEOMETRIES:
            optimum = optima[(cell.cell_id, rule)]
            oracle_q = int(optimum["best_q"])
            if (
                optimum.get("direction") != "max"
                or optimum.get("tie_policy") != tie_policy
                or oracle_q not in experiment.q_values
            ):
                raise ValueError(f"Invalid clean-F oracle optimum for {cell.cell_id}/{rule}")
            selected_key = (cell.cell_id, selected_q, rule, "quality", "clean", "F")
            oracle_key = (cell.cell_id, oracle_q, rule, "quality", "clean", "F")
            selected_measurement = measurements.get(selected_key)
            oracle_measurement = measurements.get(oracle_key)
            if selected_measurement is None or oracle_measurement is None:
                raise ValueError(f"Missing clean-F q-sweep measurement for {cell.cell_id}/{rule}")
            if json.loads(str(selected_measurement["method_prefix"])) != selected_methods:
                raise ValueError(f"Selected clean-F prefix changed for {cell.cell_id}/{rule}")
            oracle_value = float(oracle_measurement["value"])
            recorded_best = float(optimum["best_value"])
            if abs(oracle_value - recorded_best) > 1e-12:
                raise ValueError(f"Oracle clean-F value changed for {cell.cell_id}/{rule}")
            tied_q = json.loads(str(optimum["tied_q"]))
            if (
                not isinstance(tied_q, list)
                or not tied_q
                or any(int(q) not in experiment.q_values for q in tied_q)
                or [int(q) for q in tied_q] != sorted({int(q) for q in tied_q})
                or oracle_q != min(int(q) for q in tied_q)
            ):
                raise ValueError(f"Invalid tied-q set for {cell.cell_id}/{rule}")
            if json.loads(str(oracle_measurement["method_prefix"])) != json.loads(
                str(optimum["method_prefix"])
            ):
                raise ValueError(f"Oracle clean-F prefix changed for {cell.cell_id}/{rule}")
            selected_value = float(selected_measurement["value"])
            regret = oracle_value - selected_value
            if regret < -1e-12:
                raise ValueError(f"Negative clean-F regret for {cell.cell_id}/{rule}")
            row = {
                "cell": cell.cell_id,
                "dataset": cell.dataset.dataset_id,
                "model": cell.reference_model.model_id,
                "geometry": rule,
                "selected_q": selected_q,
                "oracle_q": oracle_q,
                "q_error": selected_q - oracle_q,
                "absolute_q_error": abs(selected_q - oracle_q),
                "exact": selected_q == oracle_q,
                "within_one": abs(selected_q - oracle_q) <= 1,
                "selected_is_oracle_tied": selected_q in {int(q) for q in tied_q},
                "selected_fidelity": selected_value,
                "oracle_fidelity": oracle_value,
                "fidelity_regret": max(0.0, regret),
                "oracle_tied_q": json.dumps(tied_q, separators=(",", ":")),
                "tie_policy": tie_policy,
            }
            endpoints.append(row)
            cell_rows.append(row)
        regrets = [float(row["fidelity_regret"]) for row in cell_rows]
        exact_count = sum(bool(row["exact"]) for row in cell_rows)
        within_one_count = sum(bool(row["within_one"]) for row in cell_rows)
        max_regret = max(regrets)
        verdict = (
            "strong_support"
            if exact_count == len(PRIMARY_GEOMETRIES)
            else (
                "support"
                if within_one_count == len(PRIMARY_GEOMETRIES)
                and max_regret <= TRANSFER_FIDELITY_REGRET_TOLERANCE
                else "not_supported"
            )
        )
        cells.append(
            {
                "cell": cell.cell_id,
                "dataset": cell.dataset.dataset_id,
                "model": cell.reference_model.model_id,
                "selected_q": selected_q,
                "exact": exact_count,
                "within_one": within_one_count,
                "mean_fidelity_regret": float(np.mean(regrets)),
                "max_fidelity_regret": max_regret,
                "transfer_verdict": verdict,
                "geometries": {
                    str(row["geometry"]): {
                        key: row[key]
                        for key in (
                            "oracle_q",
                            "exact",
                            "within_one",
                            "selected_is_oracle_tied",
                            "selected_fidelity",
                            "oracle_fidelity",
                            "fidelity_regret",
                            "oracle_tied_q",
                        )
                    }
                    for row in cell_rows
                },
            }
        )
    regrets = [float(row["fidelity_regret"]) for row in endpoints]
    cell_verdicts = {str(row["cell"]): str(row["transfer_verdict"]) for row in cells}
    overall_verdict = (
        "strong_support"
        if all(value == "strong_support" for value in cell_verdicts.values())
        else (
            "support"
            if all(value in {"strong_support", "support"} for value in cell_verdicts.values())
            else "not_supported"
        )
    )
    return {
        "scope": "borda_and_kemeny_clean_F",
        "tie_policy": tie_policy,
        "transfer_criterion": {
            "strong_support": "both_geometry_oracle_q_exact",
            "support": ("both_geometry_q_within_one_and_max_absolute_clean_F_regret_at_most_0.005"),
            "not_supported": "otherwise",
            "fidelity_regret_tolerance": TRANSFER_FIDELITY_REGRET_TOLERANCE,
            "status": "fixed_before_external_generalization_results",
        },
        "endpoints": endpoints,
        "cells": cells,
        "transfer_verdict": {
            "overall": overall_verdict,
            "by_cell": cell_verdicts,
        },
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


def build_dual_geometry_report(
    experiment: NoisePrefixExperiment,
    *,
    selector: Mapping[str, Any],
    prefix_summary: Mapping[str, Any],
    table1_summary: Mapping[str, Any],
    oracle_noise_summary: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
    """Read selected q metrics only after a frozen selector is supplied."""

    if selector.get("selector_digest") is None:
        raise ValueError("A frozen selector digest is required before report construction")
    _verified_digest(selector, key="selector_digest", context="selector")
    if (
        selector.get("sweep_id") != experiment.sweep_id
        or selector.get("sweep_digest") != experiment.digest
    ):
        raise ValueError("Selector does not match the configured q sweep")
    if (
        prefix_summary.get("schema") != "simple-noise-prefix-summary-v2"
        or prefix_summary.get("status") != "complete"
        or prefix_summary.get("sweep_id") != experiment.sweep_id
        or prefix_summary.get("sweep_digest") != experiment.digest
    ):
        raise ValueError("q-sweep summary identity mismatch")
    prefix_digest = _verified_digest(
        prefix_summary, key="summary_digest", context="q-sweep summary"
    )
    table1_digest = _verified_digest(table1_summary, key="summary_digest", context="Table 1")
    oracle_digest = _verified_digest(
        oracle_noise_summary, key="summary_digest", context="Oracle NOISE summary"
    )
    condition_by_noise = {
        noise: str(table1_summary["settings"]["noise"][noise]["condition_id"])
        for noise in NOISE_ORDER
    }
    measurements = _measurement_index(prefix_summary)
    selector_cells = {str(row["cell"]): row for row in selector["cells"]}
    expected_cells = {cell.cell_id for cell in experiment.cells()}
    if set(selector_cells) != expected_cells:
        raise ValueError("Selector cell coverage does not match the experiment")
    selection_validation = _selection_validation(
        experiment,
        prefix_summary=prefix_summary,
        measurements=measurements,
        selector_cells=selector_cells,
    )

    best_individual = {}
    for cell in table1_summary["cells"]:
        cell_id = f"{cell['dataset']}--{cell['model']}"
        row = next(item for item in cell["rows"] if item["method"] == "best_individual")
        best_individual[cell_id] = _flat_nested_row(row)
    oracle = {}
    for cell in oracle_noise_summary["cells"]:
        for row in cell["rows"]:
            if row.get("setting") == "oracle-noise":
                oracle[(cell["distance_model"], cell["cell"], row["method"])] = _flat_nested_row(
                    row
                )

    rows = []
    flat_rows = []
    references: dict[str, dict[tuple[str, str], Mapping[str, float]]] = {
        "clean_selected_best_individual": {},
        "naive_q11": {},
        "oracle_noise_spearman": {},
        "oracle_noise_kendall": {},
    }
    for cell in sorted(experiment.cells(), key=lambda value: value.cell_id):
        selected = selector_cells[cell.cell_id]
        ordered = tuple(str(method) for method in selected["ordered_methods"])
        q = int(selected["q"])
        method_prefix = tuple(str(method) for method in selected["method_prefix"])
        if (
            len(ordered) != 11
            or len(set(ordered)) != 11
            or method_prefix != ordered[:q]
            or q not in experiment.q_values
        ):
            raise ValueError(f"Invalid selector prefix for {cell.cell_id}")
        for rule in PAPER_RULES:
            values = _flat_values(
                measurements,
                cell=cell.cell_id,
                q=q,
                rule=rule,
                condition_by_noise=condition_by_noise,
                expected_methods=method_prefix,
            )
            q11_values = _flat_values(
                measurements,
                cell=cell.cell_id,
                q=max(experiment.q_values),
                rule=rule,
                condition_by_noise=condition_by_noise,
                expected_methods=ordered,
            )
            key = (cell.cell_id, rule)
            references["clean_selected_best_individual"][key] = best_individual[cell.cell_id]
            references["naive_q11"][key] = q11_values
            for distance in ("spearman", "kendall"):
                oracle_key = (distance, cell.cell_id, rule)
                if oracle_key not in oracle:
                    raise ValueError(f"Missing Oracle NOISE reference {oracle_key}")
                references[f"oracle_noise_{distance}"][key] = oracle[oracle_key]
            rows.append(
                {
                    "cell": cell.cell_id,
                    "dataset": cell.dataset.dataset_id,
                    "model": cell.reference_model.model_id,
                    "method": rule,
                    "setting": "dual_geometry_noise",
                    "q": q,
                    "methods": list(method_prefix),
                    **_nested_values(values),
                }
            )
            flat_rows.append(
                {
                    "cell": cell.cell_id,
                    "dataset": cell.dataset.dataset_id,
                    "model": cell.reference_model.model_id,
                    "method": rule,
                    "setting": "dual_geometry_noise",
                    "q": q,
                    "methods": json.dumps(method_prefix, separators=(",", ":")),
                    **values,
                }
            )

    comparison_rows = []
    by_reference: dict[str, list[Mapping[str, Any]]] = {name: [] for name in references}
    for row in flat_rows:
        key = (str(row["cell"]), str(row["method"]))
        for reference_name, reference_index in references.items():
            reference = reference_index[key]
            for column in REPORT_COLUMNS:
                if column in QUALITY_METRICS:
                    group = "quality"
                    condition = "clean"
                    metric = column
                else:
                    metric, noise = column.removeprefix("R_").rsplit("_", 1)
                    group = "robustness"
                    condition = NOISE_NAMES[noise]
                item = {
                    "cell": row["cell"],
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "rule": row["method"],
                    "q": row["q"],
                    "column": column,
                    "endpoint_group": group,
                    "condition": condition,
                    "metric": metric,
                    "value": float(row[column]),
                    "reference": reference_name,
                    "reference_value": float(reference[column]),
                    "direction": COLUMN_DIRECTIONS[column],
                    "outcome": _outcome(
                        float(row[column]),
                        float(reference[column]),
                        COLUMN_DIRECTIONS[column],
                    ),
                }
                comparison_rows.append(item)
                by_reference[reference_name].append(item)

    summary: dict[str, Any] = {
        "schema": "simple-dual-geometry-noise-report-v3",
        "schema_version": 3,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "scope": SCOPE,
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "science": {
            "selection": "fidelity_anchored_topk_mallows_dual_geometry_shared_q",
            "selection_scope": "complete_test_set_post_hoc",
            "q_shared_across_rules": list(PAPER_RULES),
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
            "rules": len(PAPER_RULES),
            "rows": len(rows),
            "endpoints": len(rows) * len(REPORT_COLUMNS),
            "comparison_rows": len(comparison_rows),
            "selection_validation_rows": len(selection_validation["endpoints"]),
        },
        "selected_q": {cell: int(row["q"]) for cell, row in selector_cells.items()},
        "selection_validation": selection_validation,
        "cells": [
            {
                "cell": cell_id,
                "q": int(selector_cells[cell_id]["q"]),
                "method_prefix": selector_cells[cell_id]["method_prefix"],
                "rows": [row for row in rows if row["cell"] == cell_id],
            }
            for cell_id in sorted(expected_cells)
        ],
        "comparisons": {name: _comparison_groups(items) for name, items in by_reference.items()},
        "sources": {
            "selector_digest": selector["selector_digest"],
            "selection_source_digest": selector["selection_source_digest"],
            "prefix_summary_digest": prefix_digest,
            "prefix_manifest_content_digests": prefix_summary["manifest_content_digests"],
            "table1_summary_digest": table1_digest,
            "oracle_noise_summary_digest": oracle_digest,
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
    q_rows = "\n".join(f"| {cell} | {q} |" for cell, q in sorted(summary["selected_q"].items()))
    comparisons = summary["comparisons"]
    validation = summary["selection_validation"]["aggregate"]
    transfer = summary["selection_validation"]["transfer_verdict"]
    lines = [
        "# Dual-Geometry NOISE Boundary Experiment",
        "",
        "This is a complete-test-set post-hoc boundary experiment. It does not",
        "replace the immutable Oracle NOISE result and is not a held-out estimate.",
        "",
        "The selector is frozen before the q-sweep metrics are opened. It uses",
        "individual clean-F marks, exact top-k subset Mallows fits, and one shared",
        "q from the unweighted Borda/Kemeny anchored scores. The selected q is then",
        "applied to all five aggregation rules. No additional model forward is run.",
        "",
        "| Cell | Shared q |",
        "|---|---:|",
        q_rows,
        "",
        "## q Selection Validation",
        "",
        (
            f"Across {validation['endpoints']} Borda/Kemeny clean-F endpoints, the shared q is "
            f"exact for {validation['exact']} and within one for {validation['within_one']}."
        ),
        (
            f"Mean Fidelity regret is {validation['mean_fidelity_regret']:.12g}; maximum regret "
            f"is {validation['max_fidelity_regret']:.12g}."
        ),
        f"Predeclared transfer verdict: **{transfer['overall']}**.",
        "",
        "## Comparison Counts",
        "",
    ]
    for name, values in comparisons.items():
        all_values = values["all"]
        lines.append(
            f"- {name}: {all_values['better']} better, {all_values['equal']} equal, "
            f"{all_values['worse']} worse across {sum(all_values.values())} endpoints."
        )
    lines.extend(
        [
            "",
            "## Identity",
            "",
            f"- Selector digest: `{summary['sources']['selector_digest']}`",
            f"- Result digest: `{summary['result_digest']}`",
            f"- q-sweep digest: `{summary['sweep_digest']}`",
            "",
            "`summary.csv` contains the selected rule rows. `selection_validation.csv`",
            "contains selected q versus clean-F oracle q for Borda and Kemeny.",
            "`comparisons.csv` contains every endpoint/reference comparison without rounding.",
            "The JSON/CSV also retain raw perturbed quality beside signed robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dual_geometry_selector(selector: Mapping[str, Any], *, output_path: str | Path) -> Path:
    return atomic_write_json(Path(output_path).expanduser().resolve(), selector)


def write_dual_geometry_report(
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
    selection_validation_columns = (
        "cell",
        "dataset",
        "model",
        "geometry",
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
            destination / "comparisons.csv",
            _csv_text(comparison_rows, comparison_columns),
        ),
        "selection_validation_csv": atomic_write_text(
            destination / "selection_validation.csv",
            _csv_text(
                summary["selection_validation"]["endpoints"],
                selection_validation_columns,
            ),
        ),
        "readme": atomic_write_text(destination / "README.md", _readme(summary)),
    }
    return {name: str(path) for name, path in paths.items()}


__all__ = [
    "COLUMN_DIRECTIONS",
    "EXPERIMENT_ID",
    "PERTURBED_REPORT_COLUMNS",
    "REPORT_COLUMNS",
    "SCOPE",
    "build_dual_geometry_report",
    "build_dual_geometry_selector",
    "load_dual_geometry_selector",
    "write_dual_geometry_report",
    "write_dual_geometry_selector",
]
