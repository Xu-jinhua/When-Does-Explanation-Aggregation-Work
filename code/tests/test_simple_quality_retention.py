from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from xai_ensemble.core.hashing import file_sha256
from xai_ensemble.simple.artifacts import ArtifactError, ArtifactStore
from xai_ensemble.simple.quality_retention import (
    build_quality_retention_summary,
    normalize_quality,
    quality_retention_endpoint,
    recover_unmasked_accuracy,
    write_quality_retention_outputs,
)
from xai_ensemble.simple.summary import NOISE_ORDER, PAPER_RULES


def test_normalization_orients_all_metrics_and_records_clipping() -> None:
    assert normalize_quality("F", 0.4, unmasked_accuracy=0.8)["value"] == pytest.approx(0.5)
    assert normalize_quality("Fbar", 0.2, unmasked_accuracy=0.8)["value"] == pytest.approx(0.75)
    assert normalize_quality("C", 0.6, unmasked_accuracy=0.8)["value"] == pytest.approx(0.6)
    assert normalize_quality("Cbar", 0.3, unmasked_accuracy=0.8)["value"] == pytest.approx(0.7)

    clipped_low = normalize_quality("F", -0.1, unmasked_accuracy=0.8)
    clipped_high = normalize_quality("Fbar", -0.1, unmasked_accuracy=0.8)
    assert clipped_low["value"] == 0.0
    assert clipped_low["clipped_low"] is True
    assert clipped_high["value"] == 1.0
    assert clipped_high["clipped_high"] is True


def test_endpoint_uses_condition_specific_accuracy_and_caps_improvement() -> None:
    endpoint = quality_retention_endpoint(
        metric="F",
        clean_quality=0.4,
        perturbed_quality=0.3,
        clean_unmasked_accuracy=0.8,
        perturbed_unmasked_accuracy=0.5,
    )

    assert endpoint["Qnorm_clean"] == pytest.approx(0.5)
    assert endpoint["A"] == pytest.approx(0.6)
    assert endpoint["Q_raw"] == pytest.approx(1.2)
    assert endpoint["Q"] == pytest.approx(1.0)
    assert endpoint["G"] == pytest.approx(0.6**0.5)
    assert endpoint["retention_capped"] is True


def test_zero_clean_boundary_is_explicit_without_epsilon() -> None:
    endpoint = quality_retention_endpoint(
        metric="C",
        clean_quality=0.0,
        perturbed_quality=0.2,
        clean_unmasked_accuracy=0.8,
        perturbed_unmasked_accuracy=0.7,
    )

    assert endpoint["A"] == pytest.approx(0.2)
    assert endpoint["Q"] is None
    assert endpoint["G"] is None
    assert endpoint["zero_clean_boundary"] is True


def _store(tmp_path: Path) -> ArtifactStore:
    experiment = SimpleNamespace(
        storage=SimpleNamespace(
            remote_root=str(tmp_path / "artifact-store"),
            rclone_binary=tmp_path / "unused-rclone",
        )
    )
    return ArtifactStore(experiment)


def _accuracy_manifest(tmp_path: Path) -> tuple[dict[str, object], ArtifactStore]:
    store = _store(tmp_path)
    task_digest = "d" * 64
    records = []
    rows = (
        (
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 1], dtype=torch.int64),
            torch.tensor([0, 0, 1], dtype=torch.int64),
        ),
        (
            torch.tensor([3, 4], dtype=torch.int64),
            torch.tensor([1, 0], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ),
    )
    start = 0
    for shard_index, (indices, labels, predictions) in enumerate(rows):
        relative_path = f"phase2/task/shards/shard-{shard_index:05d}.safetensors"
        path = Path(store.locator(relative_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "indices": indices,
                "labels": labels,
                "unmasked_predictions": predictions,
            },
            path,
        )
        stop = start + indices.numel()
        records.append(
            {
                "task_digest": task_digest,
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "payload": {
                    "relative_path": relative_path,
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                },
            }
        )
        start = stop
    return {
        "task_id": "task",
        "task_digest": task_digest,
        "sample_count": 5,
        "shards": records,
    }, store


def test_accuracy_is_recovered_from_all_verified_shards(tmp_path: Path) -> None:
    manifest, store = _accuracy_manifest(tmp_path)

    result = recover_unmasked_accuracy(
        manifest,
        store=store,
        temporary_directory=tmp_path / "temporary",
        download_workers=2,
    )

    assert result["correct_count"] == 3
    assert result["sample_count"] == 5
    assert result["unmasked_accuracy"] == pytest.approx(0.6)
    assert result["shard_count"] == 2


def test_accuracy_recovery_rejects_payload_sha_mismatch(tmp_path: Path) -> None:
    manifest, store = _accuracy_manifest(tmp_path)
    manifest["shards"][0]["payload"]["sha256"] = "0" * 64  # type: ignore[index]

    with pytest.raises(ArtifactError, match="digest mismatch"):
        recover_unmasked_accuracy(
            manifest,
            store=store,
            temporary_directory=tmp_path / "temporary",
            download_workers=1,
        )


def _metrics(value: float) -> dict[str, float]:
    return {
        "F": value,
        "Fbar": value / 2.0,
        "C": value + 0.1,
        "Cbar": value / 2.0,
    }


def _candidate_summary() -> dict[str, object]:
    geometries = []
    for geometry, label, q in (("spearman", "NOISE-S", 2), ("kendall", "NOISE-K", 3)):
        rows = []
        for rule_index, rule in enumerate(PAPER_RULES):
            clean = 0.4 + rule_index / 100.0
            rows.append(
                {
                    "method": rule,
                    "setting": label,
                    "methods": ["Alpha", "Beta"] if q == 2 else ["Alpha", "Beta", "Gamma"],
                    "quality": _metrics(clean),
                    "perturbed_quality": {
                        noise: _metrics(clean - 0.01 * (noise_index + 1))
                        for noise_index, noise in enumerate(NOISE_ORDER)
                    },
                }
            )
        geometries.append(
            {
                "geometry": geometry,
                "geometry_label": label,
                "q": q,
                "rows": rows,
            }
        )
    return {
        "schema": "simple-independent-geometry-noise-report-v1",
        "cells": [
            {
                "cell": "demo--model",
                "dataset": "demo",
                "model": "model",
                "geometries": geometries,
            }
        ],
    }


def _reference_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    reference = {
        "table": "Table_1",
        "cells": [
            {
                "dataset": "demo",
                "model": "model",
                "methods": ["Alpha", "Beta", "Gamma"],
            }
        ],
    }
    manifests = {
        "demo--model": {
            condition: {
                "metrics": {
                    rule: _metrics(0.35 + rule_index / 100.0 - condition_index / 100.0)
                    for rule_index, rule in enumerate(PAPER_RULES)
                }
            }
            for condition_index, condition in enumerate(("clean", *NOISE_ORDER))
        }
    }
    accuracies = {
        "demo--model": {
            condition: 0.8 - condition_index / 20.0
            for condition_index, condition in enumerate(("clean", *NOISE_ORDER))
        }
    }
    return reference, manifests, accuracies


def test_summary_strictly_matches_same_rule_q11_naive_and_writes_outputs(
    tmp_path: Path,
) -> None:
    reference, manifests, accuracies = _reference_inputs()

    summary = build_quality_retention_summary(
        _candidate_summary(),
        reference_summary=reference,
        manifests=manifests,
        accuracies=accuracies,
    )

    assert summary["counts"] == {
        "cells": 1,
        "noise_candidate_rows": 10,
        "noise_endpoints": 160,
        "unique_naive_endpoints": 80,
        "paired_comparisons": 160,
    }
    first = summary["comparisons"][0]
    assert first["rule"] == "simpleavg"
    assert first["NOISE_A"] != first["NAIVE_A"]
    assert summary["analysis"]["overall"]["G"]["endpoints"] == 160

    result = write_quality_retention_outputs(summary, output_directory=tmp_path / "result")
    for name in (
        "summary_json",
        "quality_retention_csv",
        "comparisons_csv",
        "comparison_summary_csv",
        "row_summary_csv",
        "readme",
    ):
        assert Path(result[name]).is_file()


def test_summary_rejects_a_missing_matching_rule() -> None:
    reference, manifests, accuracies = _reference_inputs()
    del manifests["demo--model"]["g"]["metrics"]["simpleavg"]  # type: ignore[index]

    with pytest.raises(ArtifactError, match="Missing q=11 rule"):
        build_quality_retention_summary(
            _candidate_summary(),
            reference_summary=reference,
            manifests=manifests,
            accuracies=accuracies,
        )
