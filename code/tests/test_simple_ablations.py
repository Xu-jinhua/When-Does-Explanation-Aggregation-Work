from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from xai_ensemble.phase2.evaluator import ClassMeanFillReference, build_masked_inputs
from xai_ensemble.simple.ablations import summary as ablation_summary
from xai_ensemble.simple.ablations.config import load_ablation_experiment
from xai_ensemble.simple.ablations.scheduler import planned_job_ids
from xai_ensemble.simple.config import ConditionConfig
from xai_ensemble.simple.data import apply_condition

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs/simple/paper-naive-ablations.yaml"


def test_naive_ablation_matrix_reuses_centers_and_plans_only_missing_work() -> None:
    experiment = load_ablation_experiment(CONFIG)

    assert [level.condition_id for level in experiment.center_levels()] == [
        "gaussian-0.15",
        "salt-pepper-0.05",
        "speckle-0.15",
        "adversarial-sara-2-255",
    ]
    assert [condition.condition_id for condition in experiment.additional_conditions] == [
        "gaussian-0.10",
        "gaussian-0.20",
        "salt-pepper-0.03",
        "salt-pepper-0.08",
        "speckle-0.10",
        "speckle-0.20",
        "adversarial-sara-1-255",
        "adversarial-sara-4-255",
    ]
    assert len(experiment.adversarial_tasks()) == 2
    assert len(experiment.phase1_tasks()) == 88
    assert len(experiment.rank_tasks()) == 8
    assert len(experiment.evaluation_tasks()) == 28
    assert len(planned_job_ids(experiment)) == 126
    assert sum(task.table_id == "table2-k" for task in experiment.evaluation_tasks()) == 15
    assert sum(task.table_id == "table4-fill" for task in experiment.evaluation_tasks()) == 5
    assert sum(task.table_id == "table5-noise" for task in experiment.evaluation_tasks()) == 8
    assert experiment.generation_experiment().phase1_digest == experiment.base.phase1_digest


def test_execution_batch_change_does_not_rekey_scientific_tasks(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["base_config"] = str(CODE_ROOT / "configs/simple/paper-main.yaml")
    value["storage"]["remote_root"] = str(tmp_path / "artifacts")
    value["storage"]["scratch_root"] = str(tmp_path / "scratch")
    value["storage"]["spool_root"] = str(tmp_path / "spool")
    value["runtime"]["database_path"] = str(tmp_path / "jobs.sqlite3")
    value["runtime"]["log_directory"] = str(tmp_path / "logs")
    first_path = tmp_path / "first.yaml"
    first_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    first = load_ablation_experiment(first_path)

    value["runtime"]["inference_batch_size"] = 256
    value["runtime"]["evaluation_reservation_gib"] = 11
    value["runtime"]["phase1_batch_caps"]["GradientShap"] = 16
    second_path = tmp_path / "second.yaml"
    second_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    second = load_ablation_experiment(second_path)

    assert first.digest == second.digest
    assert first.scheduler_digest == second.scheduler_digest
    assert [task.task_id for task in first.phase1_tasks()] == [
        task.task_id for task in second.phase1_tasks()
    ]
    assert [task.task_id for task in first.rank_tasks()] == [
        task.task_id for task in second.rank_tasks()
    ]
    assert [task.task_id for task in first.evaluation_tasks()] == [
        task.task_id for task in second.evaluation_tasks()
    ]


def test_seed_group_shares_one_random_field_across_severities() -> None:
    image = torch.full((1, 3, 16, 16), 0.5)
    common = {
        "kind": "gaussian",
        "seed_group": "gaussian-center",
    }
    low = ConditionConfig(
        condition_id="gaussian-low",
        kind="factory",
        factory="xai_ensemble.simple.conditions:natural_corruption",
        kwargs={**common, "severity": 0.001},
    )
    high = ConditionConfig(
        condition_id="gaussian-high",
        kind="factory",
        factory="xai_ensemble.simple.conditions:natural_corruption",
        kwargs={**common, "severity": 0.002},
    )
    kwargs = {
        "labels": torch.tensor([0]),
        "indices": torch.tensor([17]),
        "model": None,
        "normalize": lambda value: value,
        "seed": 1234,
    }

    low_result = apply_condition(low, image, **kwargs)
    high_result = apply_condition(high, image, **kwargs)

    torch.testing.assert_close(
        (low_result - image) / 0.001,
        (high_result - image) / 0.002,
        rtol=2e-4,
        atol=2e-4,
    )


def test_class_mean_fill_selects_bank_by_fixed_target() -> None:
    images = np.full((2, 1, 4, 4), 0.5, dtype=np.float32)
    ranks = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    fill = ClassMeanFillReference(
        values=np.asarray(
            [
                np.zeros((1, 4, 4), dtype=np.float32),
                np.ones((1, 4, 4), dtype=np.float32),
            ]
        ),
        class_counts=np.asarray([3, 5], dtype=np.int64),
        source_split="train",
        artifact_id="class-means",
    )

    masked = build_masked_inputs(
        images,
        ranks,
        fill,
        patch_size=2,
        k=1,
        fill_labels=np.asarray([0, 1], dtype=np.int64),
    )

    np.testing.assert_array_equal(masked.removed[0, :, :2, :2], 0.0)
    np.testing.assert_array_equal(masked.removed[1, :, :2, :2], 1.0)
    np.testing.assert_array_equal(masked.retained[0, :, 2:, 2:], 0.0)
    np.testing.assert_array_equal(masked.retained[1, :, 2:, 2:], 1.0)


def test_table3_summary_covers_all_patch_sizes_and_center_conditions(monkeypatch) -> None:
    experiment = load_ablation_experiment(CONFIG)
    calls = []

    def fake_manifest(_, *, condition_id: str, patch_size: int):
        calls.append((patch_size, condition_id))
        offset = 0.0 if condition_id == "clean" else 0.01
        rules = {
            "single__Alpha": {
                "F": 0.8 + offset,
                "Fbar": 0.2 - offset,
                "C": 0.7 + offset,
                "Cbar": 0.3 - offset,
            },
            "single__Beta": {
                "F": 0.7 + offset,
                "Fbar": 0.3 - offset,
                "C": 0.6 + offset,
                "Cbar": 0.4 - offset,
            },
            "simpleavg": {
                "F": 0.6 + offset,
                "Fbar": 0.4 - offset,
                "C": 0.5 + offset,
                "Cbar": 0.5 - offset,
            },
            "borda": {
                "F": 0.5 + offset,
                "Fbar": 0.5 - offset,
                "C": 0.4 + offset,
                "Cbar": 0.6 - offset,
            },
            "kemeny": {
                "F": 0.4 + offset,
                "Fbar": 0.6 - offset,
                "C": 0.3 + offset,
                "Cbar": 0.7 - offset,
            },
            "rrf": {
                "F": 0.3 + offset,
                "Fbar": 0.7 - offset,
                "C": 0.2 + offset,
                "Cbar": 0.8 - offset,
            },
            "schulze": {
                "F": 0.2 + offset,
                "Fbar": 0.8 - offset,
                "C": 0.1 + offset,
                "Cbar": 0.9 - offset,
            },
        }
        robustness = {
            rule: {"absolute": {metric: 0.01 for metric in ("F", "Fbar", "C", "Cbar")}}
            for rule in rules
        }
        return {
            "task_digest": f"{condition_id}-p{patch_size}",
            "metrics": rules,
            "robustness": None if condition_id == "clean" else robustness,
        }

    monkeypatch.setattr(ablation_summary, "_base_result_manifest", fake_manifest)

    summary = ablation_summary._table3(experiment)
    rows = ablation_summary._wide_rows(summary)

    assert [setting["p"] for setting in summary["settings"]] == [8, 14, 16]
    assert len(calls) == 15
    assert len(rows) == 18
    assert {row["p"] for row in rows} == {8, 14, 16}
    assert all(len(row) == 38 for row in rows)
    assert all(
        row["R_g_F"] == pytest.approx(0.01) and row["R_g_Fbar"] == pytest.approx(0.01)
        for row in rows
    )


def test_summary_csv_uses_repository_safe_lf_line_endings(tmp_path: Path) -> None:
    output = tmp_path / "summary.csv"

    ablation_summary._write_csv(output, ({"k": 20, "method": "borda", "F": 0.5},))

    payload = output.read_bytes()
    assert payload == b"k,method,F\n20,borda,0.5\n"
    assert b"\r" not in payload
