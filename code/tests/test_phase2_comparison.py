from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xai_ensemble.core.manifest import (
    ArtifactManifest,
    load_manifest,
    make_entry,
    validate_manifest_files,
    write_manifest,
)
from xai_ensemble.phase2.comparison import (
    ComparatorInput,
    INDFamilyPairInput,
    build_comparison_report,
    holm_adjust,
)
from xai_ensemble.phase2.metrics import QUALITY_METRICS


def _write_run(
    root: Path,
    *,
    regime: str,
    constructions: dict[str, dict[str, list[float]]],
) -> Path:
    root.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "regime": regime,
        "dataset_id": "tiny",
        "dataset_revision": "fixed",
        "dataset_manifest_fingerprint": "dataset-fingerprint",
        "protocol_digest": "protocol-fixed",
        "split": "test",
        "condition_id": "clean",
        "condition_digest": "clean-digest",
        "condition_group_id": "clean",
        "condition_type": "clean",
        "realization_id": "clean",
        "reference_model_id": "reference-full",
        "fill_artifact_id": "train-mean",
        "evaluation_adapter": {"reference_checkpoint_sha256": "a" * 64},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    labels = [0, 0, 1, 1]
    with (root / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for construction, metrics in sorted(constructions.items()):
            for index, label in enumerate(labels):
                handle.write(
                    json.dumps(
                        {
                            "condition_id": "clean",
                            "condition_digest": "clean-digest",
                            "construction_id": construction,
                            "sample_id": f"sample-{index}",
                            "label": label,
                            "target_label": label,
                            **{
                                metric: float(metrics[metric][index])
                                for metric in QUALITY_METRICS
                            },
                        }
                    )
                    + "\n"
                )
    manifest = ArtifactManifest.create(
        kind="phase2-evaluation",
        producer="unit-test",
        files=[make_entry(root, "summary.json"), make_entry(root, "trace.jsonl")],
        metadata={"regime": regime},
    )
    write_manifest(root / "manifest.json", manifest)
    return root / "manifest.json"


def _metrics(maximum: list[float], minimum: list[float]) -> dict[str, list[float]]:
    return {"F": maximum, "Fbar": minimum, "C": maximum, "Cbar": minimum}


def test_holm_adjustment_is_monotone_and_deterministic() -> None:
    result = holm_adjust({"b": 0.03, "a": 0.01, "c": 0.04}, alpha=0.05)
    assert result["a"]["holm_adjusted_p_value"] == 0.03
    assert result["b"]["holm_adjusted_p_value"] == 0.06
    assert result["c"]["holm_adjusted_p_value"] == 0.06
    assert result["a"]["reject_familywise_null"] is True
    assert result["b"]["reject_familywise_null"] is False


def test_formal_comparison_reselects_oracle_bootstraps_three_ind_families_and_holm(
    tmp_path: Path,
) -> None:
    oracle = _write_run(
        tmp_path / "oracle",
        regime="oracle_best_single",
        constructions={
            "single/method-a": _metrics([1, 1, 1, 1], [0, 0, 0, 0]),
            "single/method-b": _metrics([1, 0, 1, 0], [0, 1, 0, 1]),
        },
    )
    comparator = _write_run(
        tmp_path / "comparator",
        regime="original_naive",
        constructions={
            "original_naive/borda": _metrics([0, 0, 0, 0], [1, 1, 1, 1])
        },
    )
    pairs = []
    for family in range(3):
        ind = _write_run(
            tmp_path / f"ind-{family}",
            regime="ind",
            constructions={
                f"ind/borda/assignment-{assignment:02d}": _metrics(
                    [1, 1, 1, 1], [0, 0, 0, 0]
                )
                for assignment in range(2)
            },
        )
        overlap = _write_run(
            tmp_path / f"overlap-{family}",
            regime="matched_overlap",
            constructions={
                f"matched_overlap/borda/assignment-{assignment:02d}": _metrics(
                    [0, 0, 0, 0], [1, 1, 1, 1]
                )
                for assignment in range(2)
            },
        )
        pairs.append(INDFamilyPairInput("borda", family, ind, overlap))

    output = tmp_path / "comparison-output"
    result = build_comparison_report(
        oracle,
        [ComparatorInput("original_naive/borda/reference", comparator)],
        pairs,
        expected_partition_families=3,
        output_dir=output,
        B=19,
        confidence=0.9,
        alpha=0.05,
        seed=7,
        project_root=tmp_path,
    )
    assert result.status == "completed"
    validate_manifest_files(output, load_manifest(result.manifest_path))
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    oracle_f = report["oracle_vs_comparator"][
        "original_naive/borda/reference"
    ]["metrics"]["F"]
    assert oracle_f["selected_on_full_data"] == "method-a"
    assert np.isclose(sum(oracle_f["selection_frequency"].values()), 1.0)
    hierarchical = report["ind_vs_matched_overlap"]["borda"]["metrics"]["F"]
    assert hierarchical["n_families"] == 3
    assert hierarchical["n_assignments_or_sources"] == 2
    assert hierarchical["direction_adjusted_gain"]["estimate"] == 1.0
    assert len(report["holm"]) == 8
    with np.load(output / "bootstrap-replicates.npz") as archive:
        assert len(archive.files) == 8
        assert all(archive[key].shape == (19,) for key in archive.files)

    repeated = build_comparison_report(
        oracle,
        [ComparatorInput("original_naive/borda/reference", comparator)],
        pairs,
        expected_partition_families=3,
        output_dir=output,
        B=19,
        confidence=0.9,
        alpha=0.05,
        seed=7,
        project_root=tmp_path,
    )
    assert repeated.status == "skipped"
    assert repeated.artifact_id == result.artifact_id
