from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from xai_ensemble.simple.noise_subset.config import load_noise_subset_experiment
from xai_ensemble.simple.noise_subset.evaluator import (
    aggregate_anchored_candidate_bank,
    aggregate_subset_bank,
)
from xai_ensemble.simple.noise_subset.scheduler import planned_job_ids, submit_plan
from xai_ensemble.simple.noise_subset.selection import (
    candidate_position_sets,
    draw_candidate_position_sets,
    draw_random_method_orders,
    select_uniform_accepted_candidates,
)
from xai_ensemble.simple.scheduler import SimpleJobStore

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/simple/paper-noise-random-subset.yaml"
ANCHORED_CONFIG = ROOT / "configs/simple/paper-noise-random-order-anchored.yaml"


def test_noise_subset_config_freezes_small_mechanism_audit() -> None:
    experiment = load_noise_subset_experiment(CONFIG)

    assert experiment.cell_ids == ("dermamnist--dermamnist-resnet18",)
    assert experiment.condition_ids == ("clean",)
    assert experiment.candidate_draw_count == 48
    assert experiment.selected_candidate_count == 10
    assert len(experiment.selection_tasks()) == 1
    assert len(experiment.evaluation_tasks()) == 2
    selector = experiment.selector_cell(experiment.cell_ids[0])
    assert selector["q_S"] == 2
    assert selector["q_K"] == 3


def test_random_order_config_uses_the_complete_anchored_q_bank() -> None:
    experiment = load_noise_subset_experiment(ANCHORED_CONFIG)

    assert experiment.control_mode == "random_order_anchored"
    assert experiment.q_values == tuple(range(2, 12))
    assert experiment.random_order_count == 10
    assert experiment.artifact_schema_version == 2
    assert len(experiment.selection_tasks()) == 1
    assert len(experiment.evaluation_tasks()) == 2
    assert all("random-order" in task.task_id for task in experiment.evaluation_tasks())


def test_random_method_orders_are_reproducible_unique_and_nonreference() -> None:
    first = draw_random_method_orders(5, 10, seed=41)
    second = draw_random_method_orders(5, 10, seed=41)

    assert first == second
    assert len(first) == len(set(first)) == 10
    assert tuple(range(5)) not in first
    assert all(sorted(order) == list(range(5)) for order in first)


def test_candidate_pool_excludes_reference_and_draws_without_replacement() -> None:
    all_candidates = candidate_position_sets(5, 2, reference=(0, 1))
    first = draw_candidate_position_sets(
        5,
        2,
        reference=(0, 1),
        draw_count=6,
        seed=17,
    )
    second = draw_candidate_position_sets(
        5,
        2,
        reference=(0, 1),
        draw_count=6,
        seed=17,
    )

    assert len(all_candidates) == 9
    assert (0, 1) not in all_candidates
    assert first == second
    assert len(first) == len(set(first)) == 6
    assert set(first) <= set(all_candidates)


def test_accepted_candidate_selection_never_reads_quality_values() -> None:
    candidates = [
        {"candidate_digest": str(index), "accepted": index % 2 == 0, "quality": 100 - index}
        for index in range(10)
    ]

    selected = select_uniform_accepted_candidates(
        candidates,
        selected_count=3,
        minimum_count=3,
        seed=29,
    )

    assert len(selected) == 3
    assert all(row["accepted"] for row in selected)
    assert {row["candidate_digest"] for row in selected} <= {"0", "2", "4", "6", "8"}


def test_subset_aggregation_uses_exact_frozen_method_positions() -> None:
    experiment = load_noise_subset_experiment(CONFIG)
    task = experiment.evaluation_tasks()[0]
    generator = np.random.default_rng(3)
    ballots = np.stack([np.stack([generator.permutation(6) for _ in range(3)]) for _ in range(2)])
    scores = generator.normal(size=(2, 3, 6)).astype(np.float32)
    candidate = {
        "selection_position": 0,
        "candidate_digest": "a" * 64,
        "methods": ["A", "C"],
        "positions": [0, 2],
        "q": 2,
    }

    bank, statistics = aggregate_subset_bank(
        experiment,
        task,
        ballots=ballots,
        method_patch_scores=scores,
        ordered_methods=("A", "B", "C"),
        selected_candidates=(candidate,),
        indices=np.asarray([7, 11]),
        device="cpu",
    )

    assert set(bank) == {
        "random_00__simpleavg",
        "random_00__borda",
        "random_00__rrf",
        "random_00__kemeny",
        "random_00__schulze",
    }
    assert set(statistics) == {"random_00__kemeny"}
    expected_borda = np.argsort(
        np.argsort(-((5 - ballots[:, (0, 2)]).sum(axis=1)), axis=1, kind="stable"),
        axis=1,
        kind="stable",
    )
    np.testing.assert_array_equal(bank["random_00__borda"], expected_borda)


def test_anchored_candidate_bank_evaluates_all_centers_and_full_selected_rules() -> None:
    experiment = load_noise_subset_experiment(ANCHORED_CONFIG)
    task = experiment.evaluation_tasks()[0]
    generator = np.random.default_rng(7)
    ballots = np.stack([np.stack([generator.permutation(6) for _ in range(3)]) for _ in range(2)])
    scores = generator.normal(size=(2, 3, 6)).astype(np.float32)
    first = {
        "candidate_position": 0,
        "candidate_digest": "a" * 64,
        "methods": ["A", "B"],
        "positions": [0, 1],
        "q": 2,
    }
    second = {
        "candidate_position": 1,
        "candidate_digest": "b" * 64,
        "methods": ["A", "C"],
        "positions": [0, 2],
        "q": 2,
    }

    bank, _ = aggregate_anchored_candidate_bank(
        experiment,
        task,
        ballots=ballots,
        method_patch_scores=scores,
        ordered_methods=("A", "B", "C"),
        geometry_selection={"candidates": [first, second], "selected": [first]},
        indices=np.asarray([3, 5]),
        device="cpu",
    )

    assert set(bank) == {
        "candidate_000__simpleavg",
        "candidate_000__borda",
        "candidate_000__rrf",
        "candidate_000__kemeny",
        "candidate_000__schulze",
        "candidate_001__borda",
    }


def test_noise_subset_scheduler_has_one_selector_and_two_dependents(tmp_path: Path) -> None:
    original = load_noise_subset_experiment(CONFIG)
    experiment = replace(
        original,
        storage=replace(
            original.storage,
            remote_root=str(tmp_path / "remote"),
            scratch_root=tmp_path / "scratch",
            spool_root=tmp_path / "spool",
            spool_min_free_bytes=0,
        ),
        runtime=replace(
            original.runtime,
            database_path=tmp_path / "jobs.sqlite3",
            log_directory=tmp_path / "logs",
            shared_cache_root=tmp_path / "shared-cache",
        ),
    )
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )

    result = submit_plan(experiment, store, scan_existing=False)
    jobs = store.jobs()
    selectors = [job for job in jobs if job.kind.endswith("selection")]
    evaluations = [job for job in jobs if job.kind.endswith("evaluation")]

    assert result["planned"] == 3
    assert planned_job_ids(experiment) == {job.job_id for job in jobs}
    assert len(selectors) == 1 and not selectors[0].dependencies
    assert len(evaluations) == 2
    assert all(job.dependencies == (selectors[0].job_id,) for job in evaluations)
