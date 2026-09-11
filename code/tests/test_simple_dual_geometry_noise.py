from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.simple.noise_prefix.dual_geometry import (
    PERTURBED_REPORT_COLUMNS,
    REPORT_COLUMNS,
    _base_selector_inputs,
    _prefix_selector_inputs,
    build_dual_geometry_report,
    build_dual_geometry_selector,
    load_dual_geometry_selector,
    write_dual_geometry_report,
)
from xai_ensemble.simple.summary import NOISE_ORDER, PAPER_RULES


def _digest(value: dict, key: str) -> dict:
    value[key] = object_sha256(value)
    return value


def _flat(
    value: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    quality = {metric: value for metric in ("F", "Fbar", "C", "Cbar")}
    robustness = {noise: {metric: value for metric in quality} for noise in NOISE_ORDER}
    perturbed = {noise: {metric: value + 0.05 for metric in quality} for noise in NOISE_ORDER}
    return quality, robustness, perturbed


def _experiment() -> SimpleNamespace:
    methods = tuple(f"method-{position:02d}" for position in range(11))
    cell = SimpleNamespace(
        cell_id="dataset--model",
        dataset=SimpleNamespace(dataset_id="dataset"),
        reference_model=SimpleNamespace(model_id="model"),
        methods=methods,
    )
    task = SimpleNamespace(digest="c" * 64)
    return SimpleNamespace(
        sweep_id="sweep",
        digest="a" * 64,
        q_values=tuple(range(2, 12)),
        patch_size=16,
        k=20,
        cells=lambda: (cell,),
        base_phase2_task=lambda selected_cell, condition: task,
    )


def _selector(experiment: SimpleNamespace) -> dict:
    methods = list(experiment.cells()[0].methods)
    return _digest(
        {
            "schema": "simple-dual-geometry-noise-selector-v1",
            "schema_version": 1,
            "status": "complete",
            "experiment_id": "dual-geometry-noise-v1",
            "scope": "post_hoc_complete_test_set_boundary_experiment",
            "sweep_id": experiment.sweep_id,
            "sweep_digest": experiment.digest,
            "selection_input_contract": {
                "q_sweep_summary_accepted": False,
                "q_level_aggregate_metrics_read": False,
                "q_level_masked_predictions_read": False,
            },
            "science": {},
            "selection_source_digest": "b" * 64,
            "cells": [
                {
                    "cell": "dataset--model",
                    "dataset": "dataset",
                    "model": "model",
                    "q": 3,
                    "ordered_methods": methods,
                    "method_prefix": methods[:3],
                    "source": {},
                }
            ],
        },
        "selector_digest",
    )


def _prefix_summary(experiment: SimpleNamespace) -> dict:
    methods = list(experiment.cells()[0].methods)
    conditions = {
        "g": "gaussian",
        "p": "salt-pepper",
        "s": "speckle",
        "a": "adversarial",
    }
    measurements = []
    for rule in PAPER_RULES:
        for q, value in ((3, 0.7), (11, 0.4)):
            prefix = json.dumps(methods[:q], separators=(",", ":"))
            for metric in ("F", "Fbar", "C", "Cbar"):
                measurements.append(
                    {
                        "cell": "dataset--model",
                        "condition": "clean",
                        "dataset": "dataset",
                        "method_prefix": prefix,
                        "metric": metric,
                        "model": "model",
                        "q": q,
                        "rule": rule,
                        "value": value,
                        "value_kind": "quality",
                    }
                )
                for condition in conditions.values():
                    measurements.append(
                        {
                            "cell": "dataset--model",
                            "condition": condition,
                            "dataset": "dataset",
                            "method_prefix": prefix,
                            "metric": metric,
                            "model": "model",
                            "q": q,
                            "rule": rule,
                            "value": value + 0.05,
                            "value_kind": "conditioned_quality",
                        }
                    )
                    measurements.append(
                        {
                            "cell": "dataset--model",
                            "condition": condition,
                            "dataset": "dataset",
                            "method_prefix": prefix,
                            "metric": metric,
                            "model": "model",
                            "q": q,
                            "rule": rule,
                            "value": value,
                            "value_kind": "robustness",
                        }
                    )
    return _digest(
        {
            "schema": "simple-noise-prefix-summary-v2",
            "schema_version": 2,
            "status": "complete",
            "sweep_id": experiment.sweep_id,
            "sweep_digest": experiment.digest,
            "science": {
                "optimum_tie_policy": "smallest_q_within_absolute_1e-12",
            },
            "measurements": measurements,
            "manifest_content_digests": {},
            # The report may audit these optima but must not use them to change
            # the already-frozen selector.
            "optima": [
                {
                    "cell": "dataset--model",
                    "dataset": "dataset",
                    "model": "model",
                    "rule": rule,
                    "value_kind": "quality",
                    "condition": "clean",
                    "metric": "F",
                    "direction": "max",
                    "best_q": 3,
                    "best_value": 0.7,
                    "tied_q": "[3]",
                    "tie_policy": "smallest_q_within_absolute_1e-12",
                    "method_prefix": json.dumps(methods[:3], separators=(",", ":")),
                }
                for rule in ("borda", "kemeny")
            ],
        },
        "summary_digest",
    )


def _table1_summary() -> dict:
    quality, robustness, perturbed = _flat(0.5)
    return _digest(
        {
            "status": "complete",
            "settings": {
                "noise": {
                    "g": {"condition_id": "gaussian"},
                    "p": {"condition_id": "salt-pepper"},
                    "s": {"condition_id": "speckle"},
                    "a": {"condition_id": "adversarial"},
                }
            },
            "cells": [
                {
                    "dataset": "dataset",
                    "model": "model",
                    "rows": [
                        {
                            "method": "best_individual",
                            "quality": quality,
                            "perturbed_quality": perturbed,
                            "robustness": robustness,
                        }
                    ],
                }
            ],
        },
        "summary_digest",
    )


def _oracle_summary() -> dict:
    cells = []
    for distance in ("spearman", "kendall"):
        rows = []
        for rule in PAPER_RULES:
            quality, robustness, perturbed = _flat(0.4)
            rows.append(
                {
                    "method": rule,
                    "setting": "oracle-noise",
                    "quality": quality,
                    "perturbed_quality": perturbed,
                    "robustness": robustness,
                }
            )
        cells.append(
            {
                "cell": "dataset--model",
                "distance_model": distance,
                "rows": rows,
            }
        )
    return _digest(
        {"status": "complete", "cells": cells},
        "summary_digest",
    )


def test_selector_builder_cannot_accept_q_level_summary() -> None:
    parameters = inspect.signature(build_dual_geometry_selector).parameters

    assert tuple(parameters) == (
        "experiment",
        "base_clean_cache",
        "prefix_clean_cache",
    )
    assert not any("summary" in name or "measurement" in name for name in parameters)


def test_base_selector_reads_verified_nested_phase2_manifest(tmp_path) -> None:
    experiment = _experiment()
    cell = experiment.cells()[0]
    task_digest = experiment.base_phase2_task(cell, "clean").digest
    root = tmp_path / "dataset" / "model" / "test" / "clean" / "ensemble" / "p16" / task_digest
    shards = root / "shards"
    shards.mkdir(parents=True)
    shard_path = shards / "shard-00000.safetensors"
    rule_labels = {"borda-field": "borda", "kemeny-field": "kemeny"}
    tensors = {
        "indices": np.asarray([9, 4], dtype=np.int64),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "unmasked_predictions": np.asarray([0, 1], dtype=np.int64),
        "rank__borda-field": np.tile(np.arange(20, dtype=np.int16), (2, 1)),
        "rank__kemeny-field": np.tile(np.arange(19, -1, -1, dtype=np.int16), (2, 1)),
    }
    for position, method in enumerate(cell.methods):
        field = f"method-field-{position:02d}"
        rule_labels[field] = f"single__{method}"
        tensors[f"rank__{field}"] = np.tile(np.roll(np.arange(20), position), (2, 1)).astype(
            np.int16
        )
        tensors[f"removed_predictions__{field}"] = np.asarray([1, 1], dtype=np.int64)
    save_file(
        tensors,
        str(shard_path),
        metadata={
            "patch_size": "16",
            "rank_base": "0",
            "task_digest": task_digest,
        },
    )
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "task_digest": task_digest,
        "dataset": "dataset",
        "model": "model",
        "split": "test",
        "condition": "clean",
        "methods": list(cell.methods),
        "patch_size": 16,
        "k": 20,
        "rank_base": 0,
        "sample_count": 2,
        "shards": [
            {
                "task_digest": task_digest,
                "shard_index": 0,
                "start": 0,
                "stop": 2,
                "count": 2,
                "rule_labels": rule_labels,
                "payload": {
                    "relative_path": f"phase2/irrelevant/prefix/{shard_path.name}",
                    "sha256": file_sha256(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                },
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    values = _base_selector_inputs(
        tmp_path,
        cell=cell,
        expected_task_digest=task_digest,
        patch_size=16,
        k=20,
    )

    assert values["indices"].tolist() == [4, 9]
    assert values["task_digest"] == task_digest
    assert values["manifest_content_digest"] == object_sha256(manifest)
    assert values["ballots"].shape == (2, 11, 20)
    assert values["q11_centers"]["borda"].shape == (2, 20)


def test_prefix_selector_binds_nested_manifest_to_configured_sweep(tmp_path) -> None:
    experiment = _experiment()
    cell = experiment.cells()[0]
    task_digest = "d" * 64
    root = tmp_path / cell.cell_id / "clean" / "p16" / "k20" / task_digest
    shards = root / "shards"
    shards.mkdir(parents=True)
    shard_path = shards / "shard-00000.safetensors"
    tensors = {"indices": np.asarray([1, 0], dtype=np.int64)}
    rule_labels = {}
    for rule in ("borda", "kemeny"):
        for q in range(2, 11):
            field = f"{rule}-{q}"
            rule_labels[field] = f"q{q:02d}__{rule}"
            tensors[f"top_patch_indices__{field}"] = np.tile(np.arange(20, dtype=np.int64), (2, 1))
    save_file(
        tensors,
        str(shard_path),
        metadata={
            "task_digest": task_digest,
            "patch_size": "16",
            "k": "20",
            "rank_payload": "top_k_only",
        },
    )
    manifest = {
        "schema": "simple-noise-prefix-evaluation-v1",
        "schema_version": 1,
        "status": "complete",
        "sweep_id": experiment.sweep_id,
        "sweep_digest": experiment.digest,
        "task_digest": task_digest,
        "cell": cell.cell_id,
        "dataset": cell.dataset.dataset_id,
        "model": cell.reference_model.model_id,
        "split": "test",
        "condition": "clean",
        "patch_size": 16,
        "k": 20,
        "q_values": list(experiment.q_values),
        "computed_q_values": list(experiment.q_values[:-1]),
        "rank_payload": "top_k_patch_indices_only_no_full_consensus_rank",
        "sample_count": 2,
        "shards": [
            {
                "task_digest": task_digest,
                "shard_index": 0,
                "start": 0,
                "stop": 2,
                "count": 2,
                "rule_labels": rule_labels,
                "payload": {
                    "relative_path": f"evaluations/irrelevant/{shard_path.name}",
                    "sha256": file_sha256(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                },
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    values = _prefix_selector_inputs(
        tmp_path,
        cell=cell,
        q_values=experiment.q_values,
        expected_task_digest=task_digest,
        sweep_id=experiment.sweep_id,
        sweep_digest=experiment.digest,
        patch_size=16,
        k=20,
    )

    assert values["indices"].tolist() == [0, 1]
    assert values["centers"]["borda"][2].shape == (2, 20)

    manifest["sweep_digest"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        _prefix_selector_inputs(
            tmp_path,
            cell=cell,
            q_values=experiment.q_values,
            expected_task_digest=task_digest,
            sweep_id=experiment.sweep_id,
            sweep_digest=experiment.digest,
            patch_size=16,
            k=20,
        )
    except ValueError as error:
        assert "sweep_digest" in str(error)
    else:
        raise AssertionError("A prefix manifest from another sweep was accepted")


def test_selector_loader_enforces_frozen_rank_only_contract(tmp_path) -> None:
    selector = _selector(_experiment())
    path = tmp_path / "selector.json"
    path.write_text(json.dumps(selector), encoding="utf-8")

    loaded = load_dual_geometry_selector(path)
    assert loaded["cells"][0]["q"] == 3

    selector["selection_input_contract"]["q_level_aggregate_metrics_read"] = True
    selector["selector_digest"] = object_sha256(
        {key: value for key, value in selector.items() if key != "selector_digest"}
    )
    path.write_text(json.dumps(selector), encoding="utf-8")

    try:
        load_dual_geometry_selector(path)
    except ValueError as error:
        assert "rank-only" in str(error)
    else:
        raise AssertionError("A selector that reads q-level metrics was accepted")


def test_report_applies_frozen_q_to_all_rules_and_covers_every_endpoint() -> None:
    experiment = _experiment()
    summary, rows, comparisons = build_dual_geometry_report(
        experiment,
        selector=_selector(experiment),
        prefix_summary=_prefix_summary(experiment),
        table1_summary=_table1_summary(),
        oracle_noise_summary=_oracle_summary(),
    )

    assert len(rows) == len(PAPER_RULES)
    assert {row["method"] for row in rows} == set(PAPER_RULES)
    assert {row["q"] for row in rows} == {3}
    assert len(REPORT_COLUMNS) == 20
    assert len(PERTURBED_REPORT_COLUMNS) == 16
    assert summary["counts"] == {
        "cells": 1,
        "rules": 5,
        "rows": 5,
        "endpoints": 100,
        "comparison_rows": 400,
        "selection_validation_rows": 2,
    }
    assert len(comparisons) == 400
    assert summary["comparisons"]["naive_q11"]["all"] == {
        "better": 90,
        "equal": 0,
        "worse": 10,
    }
    assert rows[0]["F_perturbed_g"] == 0.75
    assert summary["selected_q"] == {"dataset--model": 3}
    assert summary["selection_validation"]["aggregate"] == {
        "endpoints": 2,
        "exact": 2,
        "within_one": 2,
        "selected_is_oracle_tied": 2,
        "mean_fidelity_regret": 0.0,
        "max_fidelity_regret": 0.0,
    }
    assert summary["selection_validation"]["transfer_verdict"] == {
        "overall": "strong_support",
        "by_cell": {"dataset--model": "strong_support"},
    }
    assert (
        summary["selection_validation"]["transfer_criterion"]["fidelity_regret_tolerance"] == 0.005
    )


def test_report_writes_q_selection_validation_csv(tmp_path) -> None:
    experiment = _experiment()
    summary, rows, comparisons = build_dual_geometry_report(
        experiment,
        selector=_selector(experiment),
        prefix_summary=_prefix_summary(experiment),
        table1_summary=_table1_summary(),
        oracle_noise_summary=_oracle_summary(),
    )

    paths = write_dual_geometry_report(
        summary,
        rows,
        comparisons,
        output_directory=tmp_path,
    )

    validation = (tmp_path / "selection_validation.csv").read_text(encoding="utf-8")
    assert "geometry,selected_q,oracle_q" in validation
    assert len(validation.splitlines()) == 3
    assert paths["selection_validation_csv"] == str(tmp_path / "selection_validation.csv")
