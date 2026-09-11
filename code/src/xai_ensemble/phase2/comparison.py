"""Formal clean-condition comparisons, hierarchical uncertainty, and Holm control."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from xai_ensemble.core.atomic import atomic_open, atomic_write_json
from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.core.manifest import (
    ArtifactManifest,
    load_manifest,
    make_entry,
    validate_manifest_files,
    write_manifest,
)
from xai_ensemble.core.provenance import collect_provenance

from .bootstrap import bootstrap_oracle_vs_comparator, ind_hierarchical_summary
from .metrics import DEFAULT_METRIC_DIRECTIONS, QUALITY_METRICS


@dataclass(frozen=True)
class ComparatorInput:
    comparison_id: str
    manifest_path: Path


@dataclass(frozen=True)
class INDFamilyPairInput:
    aggregation: str
    family_id: int
    ind_manifest_path: Path
    matched_overlap_manifest_path: Path


@dataclass(frozen=True)
class ComparisonReportResult:
    status: Literal["completed", "skipped"]
    manifest_path: Path
    artifact_id: str


@dataclass(frozen=True)
class _FormalRun:
    manifest_path: Path
    manifest: ArtifactManifest
    summary: dict[str, Any]
    records: dict[str, dict[str, dict[str, Any]]]


def holm_adjust(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> dict[str, dict[str, float | bool]]:
    """Return deterministic Holm step-down adjusted p-values and decisions."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not p_values:
        raise ValueError("Holm correction requires at least one hypothesis")
    checked: dict[str, float] = {}
    for hypothesis, raw in p_values.items():
        value = float(raw)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid p-value for {hypothesis!r}: {raw!r}")
        checked[str(hypothesis)] = value
    ordered = sorted(checked, key=lambda key: (checked[key], key))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, hypothesis in enumerate(ordered):
        candidate = min(1.0, (total - index) * checked[hypothesis])
        running = max(running, candidate)
        adjusted[hypothesis] = running
    return {
        hypothesis: {
            "raw_p_value": checked[hypothesis],
            "holm_adjusted_p_value": adjusted[hypothesis],
            "reject_familywise_null": adjusted[hypothesis] <= alpha,
        }
        for hypothesis in sorted(checked)
    }


def _load_formal_run(path: str | Path) -> _FormalRun:
    manifest_path = Path(path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest.kind != "phase2-evaluation":
        raise ValueError(f"not a formal Phase 2 evaluation artifact: {manifest_path}")
    root = manifest_path.parent
    validate_manifest_files(root, manifest)
    summary = json.loads(
        (root / manifest.entry("summary.json").path).read_text(encoding="utf-8")
    )
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in sorted(item for item in manifest.files if item.path.endswith(".jsonl")):
        with (root / entry.path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"non-object trace record in {entry.path}")
                if (
                    record.get("condition_id") != summary.get("condition_id")
                    or record.get("condition_digest")
                    != summary.get("condition_digest")
                ):
                    raise ValueError("trace condition identity contradicts its summary")
                construction = str(record["construction_id"])
                sample_id = str(record["sample_id"])
                bucket = records.setdefault(construction, {})
                if sample_id in bucket:
                    raise ValueError(
                        f"duplicate ({construction}, {sample_id}) in Phase 2 trace"
                    )
                bucket[sample_id] = record
    if not records:
        raise ValueError("formal Phase 2 artifact has no trace records")
    if summary.get("condition_id") != "clean" or summary.get("condition_type") != "clean":
        raise ValueError("formal comparison inputs must all be clean-condition runs")
    return _FormalRun(manifest_path, manifest, summary, records)


def _checkpoint_sha(run: _FormalRun) -> str:
    value = str(
        run.summary.get("evaluation_adapter", {}).get(
            "reference_checkpoint_sha256", ""
        )
    )
    if len(value) != 64:
        raise ValueError("comparison input lacks a reference checkpoint SHA-256")
    return value


def _validate_common_identity(reference: _FormalRun, candidate: _FormalRun) -> None:
    fields = (
        "dataset_id",
        "dataset_revision",
        "dataset_manifest_fingerprint",
        "protocol_digest",
        "split",
        "condition_id",
        "condition_digest",
        "reference_model_id",
        "fill_artifact_id",
    )
    mismatches = {
        field: {
            "reference": reference.summary.get(field),
            "candidate": candidate.summary.get(field),
        }
        for field in fields
        if reference.summary.get(field) != candidate.summary.get(field)
    }
    if mismatches:
        raise ValueError(f"Phase 2 comparison identity mismatch: {mismatches}")
    if _checkpoint_sha(reference) != _checkpoint_sha(candidate):
        raise ValueError("Phase 2 comparison inputs use different reference checkpoints")


def _sample_order(run: _FormalRun) -> tuple[list[str], NDArray[np.int64]]:
    first = run.records[sorted(run.records)[0]]
    sample_ids = sorted(first)
    labels = np.asarray([int(first[sample_id]["label"]) for sample_id in sample_ids])
    targets = [first[sample_id]["target_label"] for sample_id in sample_ids]
    for construction, rows in run.records.items():
        if set(rows) != set(sample_ids):
            raise ValueError(f"construction {construction!r} has a different sample set")
        for index, sample_id in enumerate(sample_ids):
            if int(rows[sample_id]["label"]) != int(labels[index]):
                raise ValueError("one artifact assigns different labels across constructions")
            if rows[sample_id]["target_label"] != targets[index]:
                raise ValueError("one artifact explains different targets across constructions")
    return sample_ids, labels


def _validate_sample_alignment(
    reference: _FormalRun,
    candidate: _FormalRun,
    sample_ids: Sequence[str],
    labels: NDArray[np.int64],
) -> None:
    reference_rows = reference.records[sorted(reference.records)[0]]
    for construction, rows in candidate.records.items():
        if set(rows) != set(sample_ids):
            raise ValueError(
                f"comparison construction {construction!r} has a different sample set"
            )
        for index, sample_id in enumerate(sample_ids):
            row = rows[sample_id]
            reference_row = reference_rows[sample_id]
            if int(row["label"]) != int(labels[index]):
                raise ValueError("comparison inputs disagree on class labels")
            if row["target_label"] != reference_row["target_label"]:
                raise ValueError("comparison inputs do not explain the same fixed target")


def _construction_matrix(
    run: _FormalRun, metric: str, sample_ids: Sequence[str]
) -> NDArray[np.float64]:
    return np.asarray(
        [
            [float(run.records[name][sample_id][metric]) for sample_id in sample_ids]
            for name in sorted(run.records)
        ],
        dtype=np.float64,
    )


def _estimate_dict(value: Any) -> dict[str, Any]:
    result = asdict(value)
    result.pop("replicates", None)
    return result


def _direction_adjusted_estimate(value: Any, sign: float) -> dict[str, Any]:
    if sign > 0:
        low, high = value.ci_low, value.ci_high
    else:
        low, high = -value.ci_high, -value.ci_low
    return {
        "estimate": sign * value.estimate,
        "standard_error": value.standard_error,
        "ci_low": low,
        "ci_high": high,
        "confidence": value.confidence,
    }


def _bootstrap_p_values(
    gain_replicates: NDArray[np.float64], estimate: float
) -> tuple[float, float, float]:
    """Centered-null bootstrap p-values plus the raw nonpositive tail mass."""

    count = int(gain_replicates.size)
    centered_null = gain_replicates - float(estimate)
    one_sided = (
        1 + int(np.count_nonzero(centered_null >= float(estimate)))
    ) / (count + 1)
    two_sided = (
        1
        + int(
            np.count_nonzero(
                np.abs(centered_null) >= abs(float(estimate))
            )
        )
    ) / (count + 1)
    nonpositive_probability = (
        1 + int(np.count_nonzero(gain_replicates <= 0.0))
    ) / (count + 1)
    return float(one_sided), float(two_sided), float(nonpositive_probability)


def build_comparison_report(
    oracle_best_single_manifest: str | Path,
    comparator_inputs: Sequence[ComparatorInput],
    ind_family_pairs: Sequence[INDFamilyPairInput],
    *,
    expected_partition_families: int,
    output_dir: str | Path,
    B: int = 2_000,
    confidence: float = 0.95,
    alpha: float = 0.05,
    seed: int = 0,
    project_root: str | Path = ".",
) -> ComparisonReportResult:
    """Run the predeclared clean comparison family and seal an artifact.

    Oracle selection is repeated inside every paired class-stratified
    bootstrap replicate.  IND minus matched OVERLAP is bootstrapped over
    partition family, cyclic assignment/source, and class-stratified sample.
    Every directional hypothesis from both analyses shares one Holm family.
    """

    if expected_partition_families <= 0:
        raise ValueError("expected_partition_families must be positive")
    if not comparator_inputs:
        raise ValueError("at least one Oracle comparator is required")
    if not ind_family_pairs:
        raise ValueError("IND comparison requires partition-family pairs")
    comparison_ids = [item.comparison_id for item in comparator_inputs]
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("Oracle comparator ids must be unique")

    oracle = _load_formal_run(oracle_best_single_manifest)
    if oracle.summary.get("regime") != "oracle_best_single":
        raise ValueError("oracle_best_single_manifest has the wrong regime")
    sample_ids, labels = _sample_order(oracle)
    oracle_methods = {
        name.removeprefix("single/"): rows
        for name, rows in oracle.records.items()
        if name.startswith("single/")
    }
    if len(oracle_methods) != len(oracle.records) or not oracle_methods:
        raise ValueError("Oracle artifact contains non-single or empty constructions")

    comparators = [(item, _load_formal_run(item.manifest_path)) for item in comparator_inputs]
    for _item, run in comparators:
        _validate_common_identity(oracle, run)
        _validate_sample_alignment(oracle, run, sample_ids, labels)

    pairs_by_aggregation: dict[str, list[tuple[INDFamilyPairInput, _FormalRun, _FormalRun]]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    for pair in ind_family_pairs:
        key = (pair.aggregation, pair.family_id)
        if key in seen_pairs:
            raise ValueError(f"duplicate IND family pair {key}")
        seen_pairs.add(key)
        ind = _load_formal_run(pair.ind_manifest_path)
        overlap = _load_formal_run(pair.matched_overlap_manifest_path)
        if ind.summary.get("regime") != "ind":
            raise ValueError("IND family input has the wrong regime")
        if overlap.summary.get("regime") != "matched_overlap":
            raise ValueError("matched OVERLAP family input has the wrong regime")
        for run in (ind, overlap):
            _validate_common_identity(oracle, run)
            _validate_sample_alignment(oracle, run, sample_ids, labels)
        pairs_by_aggregation.setdefault(pair.aggregation, []).append((pair, ind, overlap))

    for aggregation, pairs in pairs_by_aggregation.items():
        family_ids = {item.family_id for item, _ind, _overlap in pairs}
        if len(pairs) != expected_partition_families:
            raise ValueError(
                f"{aggregation} has {len(pairs)} partition families; "
                f"expected {expected_partition_families}"
            )
        if family_ids != set(range(expected_partition_families)):
            raise ValueError(
                f"{aggregation} partition family ids must be contiguous from zero"
            )

    input_identity = {
        "oracle_artifact_id": oracle.manifest.artifact_id,
        "comparators": [
            {
                "comparison_id": item.comparison_id,
                "artifact_id": run.manifest.artifact_id,
            }
            for item, run in comparators
        ],
        "ind_pairs": [
            {
                "aggregation": pair.aggregation,
                "family_id": pair.family_id,
                "ind_artifact_id": ind.manifest.artifact_id,
                "matched_overlap_artifact_id": overlap.manifest.artifact_id,
            }
            for aggregation in sorted(pairs_by_aggregation)
            for pair, ind, overlap in sorted(
                pairs_by_aggregation[aggregation], key=lambda item: item[0].family_id
            )
        ],
        "B": B,
        "confidence": confidence,
        "alpha": alpha,
        "seed": seed,
        "expected_partition_families": expected_partition_families,
    }
    analysis_digest = object_sha256(input_identity)
    output = Path(output_dir)
    output_manifest = output / "manifest.json"
    if output_manifest.exists():
        manifest = load_manifest(output_manifest)
        if manifest.metadata.get("analysis_digest") != analysis_digest:
            raise ValueError("output directory contains a different comparison analysis")
        validate_manifest_files(output, manifest)
        return ComparisonReportResult("skipped", output_manifest, manifest.artifact_id)

    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_digest": analysis_digest,
        "dataset_id": oracle.summary["dataset_id"],
        "dataset_revision": oracle.summary["dataset_revision"],
        "dataset_manifest_fingerprint": oracle.summary[
            "dataset_manifest_fingerprint"
        ],
        "protocol_digest": oracle.summary["protocol_digest"],
        "reference_model_id": oracle.summary["reference_model_id"],
        "reference_checkpoint_sha256": _checkpoint_sha(oracle),
        "fill_artifact_id": oracle.summary["fill_artifact_id"],
        "sample_count": len(sample_ids),
        "B": B,
        "confidence": confidence,
        "familywise_alpha": alpha,
        "multiplicity_family": (
            "all predeclared one-sided clean Oracle-vs-comparator and "
            "IND-vs-matched-OVERLAP hypotheses"
        ),
        "p_value_method": (
            "plus-one centered-null nonparametric bootstrap; Oracle method is "
            "reselected inside every class-stratified paired replicate"
        ),
        "oracle_vs_comparator": {},
        "ind_vs_matched_overlap": {},
    }
    replicates: dict[str, NDArray[np.float64]] = {}
    raw_p_values: dict[str, float] = {}
    hypothesis_locations: dict[str, dict[str, Any]] = {}
    replicate_index = 0

    for comparator, run in sorted(comparators, key=lambda item: item[0].comparison_id):
        comparator_result: dict[str, Any] = {
            "artifact_id": run.manifest.artifact_id,
            "regime": run.summary["regime"],
            "metrics": {},
        }
        for metric in QUALITY_METRICS:
            direction = DEFAULT_METRIC_DIRECTIONS[metric]
            methods = {
                method: np.asarray(
                    [float(rows[sample_id][metric]) for sample_id in sample_ids],
                    dtype=np.float64,
                )
                for method, rows in oracle_methods.items()
            }
            comparator_values = np.mean(
                _construction_matrix(run, metric, sample_ids), axis=0
            )
            result = bootstrap_oracle_vs_comparator(
                methods,
                comparator_values,
                direction=direction,
                class_labels=labels,
                B=B,
                confidence=confidence,
                seed=int(stable_seed(seed, "oracle", comparator.comparison_id, metric)),
            )
            sign = 1.0 if direction == "max" else -1.0
            gain_replicates = sign * result.difference.replicates
            gain_estimate = sign * result.difference.estimate
            one_sided, two_sided, nonpositive_probability = _bootstrap_p_values(
                gain_replicates, gain_estimate
            )
            hypothesis_id = f"oracle::{comparator.comparison_id}::{metric}"
            replicate_key = f"bootstrap_{replicate_index:05d}"
            replicate_index += 1
            replicates[replicate_key] = gain_replicates
            metric_payload = {
                "direction": direction,
                "raw_oracle_minus_comparator": _estimate_dict(result.difference),
                "direction_adjusted_gain": _direction_adjusted_estimate(
                    result.difference, sign
                ),
                "selected_on_full_data": result.selected_on_full_data,
                "selection_frequency": dict(result.selection_frequency),
                "p_value_one_sided_gain": one_sided,
                "p_value_two_sided": two_sided,
                "bootstrap_probability_nonpositive_gain": (
                    nonpositive_probability
                ),
                "replicate_key": replicate_key,
            }
            comparator_result["metrics"][metric] = metric_payload
            raw_p_values[hypothesis_id] = one_sided
            hypothesis_locations[hypothesis_id] = metric_payload
        report["oracle_vs_comparator"][comparator.comparison_id] = comparator_result

    for aggregation in sorted(pairs_by_aggregation):
        ordered_pairs = sorted(
            pairs_by_aggregation[aggregation], key=lambda item: item[0].family_id
        )
        aggregation_result: dict[str, Any] = {
            "partition_family_ids": [pair.family_id for pair, _ind, _overlap in ordered_pairs],
            "ind_artifact_ids": [ind.manifest.artifact_id for _pair, ind, _overlap in ordered_pairs],
            "matched_overlap_artifact_ids": [
                overlap.manifest.artifact_id for _pair, _ind, overlap in ordered_pairs
            ],
            "metrics": {},
        }
        for metric in QUALITY_METRICS:
            ind_values = np.stack(
                [_construction_matrix(ind, metric, sample_ids) for _pair, ind, _overlap in ordered_pairs]
            )
            overlap_values = np.stack(
                [
                    _construction_matrix(overlap, metric, sample_ids)
                    for _pair, _ind, overlap in ordered_pairs
                ]
            )
            direction = DEFAULT_METRIC_DIRECTIONS[metric]
            result = ind_hierarchical_summary(
                ind_values,
                overlap_values,
                class_labels=labels,
                direction=direction,
                B=B,
                confidence=confidence,
                seed=int(stable_seed(seed, "ind-overlap", aggregation, metric)),
            )
            gain_replicates = result.direction_adjusted_gain.replicates
            one_sided, two_sided, nonpositive_probability = _bootstrap_p_values(
                gain_replicates,
                result.direction_adjusted_gain.estimate,
            )
            hypothesis_id = f"ind-vs-overlap::{aggregation}::{metric}"
            replicate_key = f"bootstrap_{replicate_index:05d}"
            replicate_index += 1
            replicates[replicate_key] = gain_replicates
            metric_payload = {
                "direction": direction,
                "raw_ind_minus_matched_overlap": _estimate_dict(
                    result.raw_difference
                ),
                "direction_adjusted_gain": _estimate_dict(
                    result.direction_adjusted_gain
                ),
                "family_differences": result.family_differences.tolist(),
                "n_families": result.n_families,
                "n_assignments_or_sources": result.n_sources,
                "n_samples": result.n_samples,
                "p_value_one_sided_gain": one_sided,
                "p_value_two_sided": two_sided,
                "bootstrap_probability_nonpositive_gain": (
                    nonpositive_probability
                ),
                "replicate_key": replicate_key,
            }
            aggregation_result["metrics"][metric] = metric_payload
            raw_p_values[hypothesis_id] = one_sided
            hypothesis_locations[hypothesis_id] = metric_payload
        report["ind_vs_matched_overlap"][aggregation] = aggregation_result

    corrections = holm_adjust(raw_p_values, alpha=alpha)
    report["holm"] = corrections
    for hypothesis, correction in corrections.items():
        hypothesis_locations[hypothesis]["holm_adjusted_p_value"] = correction[
            "holm_adjusted_p_value"
        ]
        hypothesis_locations[hypothesis]["reject_familywise_null"] = correction[
            "reject_familywise_null"
        ]

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "comparison.json", report)
    with atomic_open(output / "bootstrap-replicates.npz", "wb") as handle:
        np.savez_compressed(handle, **replicates)
    provenance = collect_provenance(project_root)
    provenance["analysis"] = input_identity
    provenance["input_manifests"] = [
        str(oracle.manifest_path),
        *(str(run.manifest_path) for _item, run in comparators),
        *(
            str(run.manifest_path)
            for pairs in pairs_by_aggregation.values()
            for _pair, ind, overlap in pairs
            for run in (ind, overlap)
        ),
    ]
    atomic_write_json(output / "provenance.json", provenance)
    entries = [
        make_entry(
            output,
            "bootstrap-replicates.npz",
            infer_dimensions=False,
            media_type="application/x-npz",
            metadata={"arrays": len(replicates), "replicates_per_array": B},
        ),
        make_entry(output, "comparison.json"),
        make_entry(output, "provenance.json"),
    ]
    manifest = ArtifactManifest.create(
        kind="phase2-comparison",
        producer="xai_ensemble.phase2.comparison",
        files=entries,
        metadata={
            "analysis_digest": analysis_digest,
            "dataset_id": oracle.summary["dataset_id"],
            "protocol_digest": oracle.summary["protocol_digest"],
            "oracle_artifact_id": oracle.manifest.artifact_id,
            "input_artifact_ids": sorted(
                {
                    oracle.manifest.artifact_id,
                    *(run.manifest.artifact_id for _item, run in comparators),
                    *(
                        run.manifest.artifact_id
                        for pairs in pairs_by_aggregation.values()
                        for _pair, ind, overlap in pairs
                        for run in (ind, overlap)
                    ),
                }
            ),
            "B": B,
            "confidence": confidence,
            "familywise_alpha": alpha,
            "expected_partition_families": expected_partition_families,
        },
    )
    write_manifest(output_manifest, manifest)
    validate_manifest_files(output, manifest)
    return ComparisonReportResult("completed", output_manifest, manifest.artifact_id)


__all__ = [
    "ComparatorInput",
    "ComparisonReportResult",
    "INDFamilyPairInput",
    "build_comparison_report",
    "holm_adjust",
]
