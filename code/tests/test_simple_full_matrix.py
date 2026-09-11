from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xai_ensemble.core.protocol import load_protocol
from xai_ensemble.simple import config as simple_config
from xai_ensemble.simple import scheduler as simple_scheduler
from xai_ensemble.simple.full_matrix import assets as full_matrix_assets
from xai_ensemble.simple.full_matrix import cli as full_matrix_cli
from xai_ensemble.simple.full_matrix import planner as full_matrix_planner
from xai_ensemble.simple.full_matrix import scheduler as full_matrix_scheduler
from xai_ensemble.simple.full_matrix.assets import (
    compatibility_candidate_digest,
    compatibility_candidates,
    compatibility_complete,
    planned_asset_jobs,
    static_compatibility_failures,
    write_static_compatibility_gate,
)
from xai_ensemble.simple.full_matrix.catalog import (
    DATASET_IDS,
    MODEL_KEYS,
    MatrixCell,
    matrix_cells,
    validate_method_rosters,
)
from xai_ensemble.simple.full_matrix.config import FullMatrixExperiment, load_full_matrix_experiment
from xai_ensemble.simple.full_matrix.planner import (
    _base_config,
    build_execution_scope,
    execution_paths,
    materialize_component_configs,
)
from xai_ensemble.simple.full_matrix.scheduler import (
    _HIGH_RESERVATION_STRICT_THRESHOLD_BYTES,
    BASE_SUMMARY_JOB_ID,
    DEFERRED_DATASETS_SCHEMA,
    MATERIALIZE_JOB_ID,
    _capacity_action,
    _capacity_floor_failure,
    _capacity_signal_detail,
    _effective_matrix_reservation,
    _ensure_deferred_ready_scope,
    _extend_plan,
    _launch_cpu,
    _live_external_gpu_ids,
    _prefix_jobs,
    _priority,
    _process_group_exists,
    _requires_exclusive_matrix_gpu,
    _requires_strict_matrix_gpu,
    _scope_compatibility_prerequisites,
    _scope_stage_id,
    _signal_directory,
    _stop_running_for_capacity,
    _submit_initial_plan,
    _worker_environment,
    clear_deferred_datasets,
    deferred_datasets_path,
    load_deferred_datasets,
    scheduler_status,
    write_deferred_datasets,
)
from xai_ensemble.simple.full_matrix.selector import _subset_experiment
from xai_ensemble.simple.methods import PATCH_METHODS
from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
from xai_ensemble.simple.scheduler import (
    QueueJob,
    RunningProcess,
    SimpleJobStore,
    _reservation_fits,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/simple/full-matrix.yaml"


def test_barrier_cli_accepts_current_and_legacy_stage_forms() -> None:
    from xai_ensemble.cli import build_parser

    parser = build_parser()
    common = [
        "simple",
        "full-matrix",
        "barrier",
        "--config",
        str(CONFIG),
    ]
    positional = parser.parse_args([*common, "prefix-complete"])
    option = parser.parse_args([*common, "--stage", "prefix-complete"])
    assert positional.stage_positional == "prefix-complete"
    assert positional.stage_option is None
    assert option.stage_positional is None
    assert option.stage_option == "prefix-complete"


def _experiment(tmp_path: Path):
    original = load_full_matrix_experiment(CONFIG)
    storage = replace(
        original.storage,
        asset_root=tmp_path / "assets",
        run_root=tmp_path / "run",
        cache_root=tmp_path / "cache",
        selector_cache_root=tmp_path / "selector-cache",
    )
    runtime = replace(
        original.runtime,
        database_path=tmp_path / "jobs.sqlite3",
        log_directory=tmp_path / "logs",
    )
    return replace(original, storage=storage, runtime=runtime)


def _mark_scope_cell_ready(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    cell: MatrixCell,
) -> QueueJob:
    gate = next(job for job in store.jobs() if job.job_id == f"matrix-compatibility:{cell.cell_id}")
    with store.connect() as connection:
        connection.executemany(
            "UPDATE jobs SET status='succeeded' WHERE job_id=?",
            [(dependency,) for dependency in (*gate.dependencies, gate.job_id)],
        )
    gate_path = experiment.compatibility_directory(cell) / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate_digest": "test-gate",
                "blocked_methods": [],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    return gate


def test_catalog_is_the_frozen_14_by_8_matrix() -> None:
    experiment = load_full_matrix_experiment(CONFIG)

    assert experiment.dataset_ids == DATASET_IDS
    assert experiment.model_keys == MODEL_KEYS
    assert len(matrix_cells()) == 112
    assert len(experiment.cells()) == 112
    assert experiment.runtime.headroom_fraction == 0.05


def test_selector_cell_scopes_base_without_mutating_formal_identity() -> None:
    prefix = load_noise_prefix_experiment(ROOT / "configs/simple/paper-noise-prefix-sweep.yaml")
    cell_id = prefix.cells()[0].cell_id

    subset = _subset_experiment(prefix, cell_id)

    assert subset.digest == prefix.digest
    assert tuple(item.dataset_id for item in subset.base.datasets) == (
        prefix.base.dataset(cell_id.split("--")[0]).dataset_id,
    )
    assert tuple(item.model_id for item in subset.base.models) == (cell_id.split("--", 1)[1],)


def test_full_matrix_identity_ignores_ambient_core_run_cache_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = load_full_matrix_experiment(CONFIG)
    monkeypatch.setenv("XAI_RUN_ROOT", "/tmp/unrelated-run-root")
    monkeypatch.setenv("XAI_CACHE_ROOT", "/tmp/unrelated-cache-root")

    reloaded = load_full_matrix_experiment(CONFIG)

    assert reloaded.protocol_digest == baseline.protocol_digest
    assert reloaded.digest == baseline.digest
    assert reloaded.scheduler_digest == baseline.scheduler_digest


def test_method_rosters_are_strictly_eleven_per_architecture() -> None:
    experiment = load_full_matrix_experiment(CONFIG)
    rosters = validate_method_rosters(experiment.methods)

    assert set(rosters) == {"cnn", "vit"}
    assert {name: len(methods) for name, methods in rosters.items()} == {"cnn": 11, "vit": 11}
    assert rosters["cnn"][-1] == "LRP"
    assert rosters["vit"][-1] == "AttentionGradCAM"


def test_compatibility_gate_uses_final_p16_method_parameters() -> None:
    experiment = load_full_matrix_experiment(CONFIG)
    candidates = compatibility_candidates(experiment, experiment.cells()[0])
    params = {candidate.family: candidate.params for candidate in candidates}

    assert len(candidates) == 11
    assert params["IntegratedGradients"]["n_steps"] == 50
    assert params["GradientShap"]["n_samples"] == 20
    assert params["FeatureAblation"] == {
        "baseline": "zero",
        "baseline_space": "model_input",
        "patch_size": 16,
    }
    assert params["Occlusion"]["patch_size"] == 16
    assert compatibility_candidate_digest(experiment, experiment.cells()[0])


def test_static_provider_preflight_blocks_only_known_unsupported_transformers(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    by_model = {cell.model_key: cell for cell in experiment.cells()}

    assert not static_compatibility_failures(experiment, by_model["vit_base_patch16_224"])
    deit = static_compatibility_failures(experiment, by_model["deit_base_patch16_224"])
    assert [row["method"] for row in deit] == ["PartialLRP", "FullLRP"]
    swin = static_compatibility_failures(experiment, by_model["swin_base_patch4_window7_224"])
    assert [row["method"] for row in swin] == [
        "CheferTransformerAttribution",
        "PartialLRP",
        "FullLRP",
    ]

    gate = write_static_compatibility_gate(
        experiment,
        cell_id=by_model["swin_base_patch4_window7_224"].cell_id,
    )
    assert gate["status"] == "blocked"
    assert gate["gate_type"] == "static_provider_preflight"
    assert compatibility_complete(experiment, by_model["swin_base_patch4_window7_224"])


def test_scheduler_gate_poll_does_not_rehash_remote_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    cell = next(
        item for item in experiment.cells() if not static_compatibility_failures(experiment, item)
    )
    checkpoint = experiment.checkpoint_path(cell)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"sealed-checkpoint")
    gate = {
        "schema": "simple-full-matrix-compatibility-gate-v1",
        "schema_version": 1,
        "matrix_digest": experiment.digest,
        "cell": cell.cell_id,
        "gate_type": "real_checkpoint",
        "status": "passed",
        "checkpoint_sha256": "a" * 64,
        "method_catalog_digest": experiment.methods.source_digest,
        "candidate_digest": compatibility_candidate_digest(experiment, cell),
        "rank_patch_size": 16,
    }
    gate["gate_digest"] = full_matrix_assets.object_sha256(gate)
    gate_path = experiment.compatibility_directory(cell) / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(
        full_matrix_assets,
        "file_sha256",
        lambda _path: pytest.fail("scheduler marker poll rehashed the checkpoint"),
    )

    assert compatibility_complete(experiment, cell, verify_checkpoint=False)


def test_full_matrix_phase1_filter_only_emits_p16(tmp_path: Path) -> None:
    matrix = _experiment(tmp_path)
    raw = _base_config(
        matrix,
        (matrix.cells()[0],),
        {"active_manifest_digest": "a" * 64},
    )
    config = tmp_path / "p16-only.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    experiment = simple_config.load_experiment(config)
    assert experiment.phase1_patch_sizes == (16,)
    assert any(
        len(definition.instances("cnn")) == 3
        for definition in experiment.methods.for_architecture("cnn")
        if definition.family in PATCH_METHODS
    )
    for task in experiment.phase1_tasks():
        if task.family in PATCH_METHODS:
            assert [variant.variant for variant in task.variants] == ["p16"]


def test_projected_plan_covers_assets_and_compact_ind(tmp_path: Path) -> None:
    # Projection is a pure planning unit test.  Keep completion probes off the
    # live CloudStorage FUSE mount; real mounted-storage behavior is checked
    # separately by the server preflight.
    experiment = _experiment(tmp_path)
    projected = full_matrix_cli._projected_counts(experiment)

    assert projected["asset_jobs"] == 336
    assert projected["asset_by_kind"] == {
        "matrix-compatibility": 84,
        "matrix-manifest": 14,
        "matrix-mean": 84,
        "matrix-partition": 28,
        "matrix-reference-training": 84,
        "matrix-samples": 14,
        "matrix-static-compatibility": 28,
    }
    assert projected["cells"] == {
        "declared": 112,
        "static_provider_blocked": 28,
        "eligible_for_real_checkpoint_gate": 84,
    }
    assert projected["base"]["phase1_method_tasks"] == 4_620
    assert projected["noise_prefix"]["evaluation_tasks"] == 420
    assert projected["compact_ind"] == {
        "source_models": 924,
        "source_method_pairs_per_condition": 41,
        "partition_jobs": 84,
        "source_method_jobs": 17_220,
        "rank_jobs": 1_680,
        "evaluation_jobs": 1_680,
        "matched_naive_full_sources_per_cell": 3,
    }


def test_projected_plan_does_not_probe_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_full_matrix_experiment(CONFIG)

    def unexpected_probe(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("read-only plan must not probe an existing artifact")

    for name in (
        "manifest_complete",
        "partition_complete",
        "samples_complete",
        "mean_complete",
        "checkpoint_complete",
        "compatibility_complete",
    ):
        monkeypatch.setattr(full_matrix_assets, name, unexpected_probe)

    assert full_matrix_cli._projected_counts(experiment)["asset_jobs"] == 336


def test_asset_plan_has_exact_phase_zero_job_counts(tmp_path: Path) -> None:
    jobs = planned_asset_jobs(_experiment(tmp_path))

    assert len(jobs) == 336
    assert Counter(job.kind for job in jobs) == {
        "matrix-manifest": 14,
        "matrix-partition": 28,
        "matrix-samples": 14,
        "matrix-mean": 84,
        "matrix-reference-training": 84,
        "matrix-compatibility": 84,
        "matrix-static-compatibility": 28,
    }
    compatibility = next(job for job in jobs if job.kind == "matrix-compatibility")
    assert len(compatibility.dependencies) == 3
    assert compatibility.reservation_bytes is not None


def test_initial_queue_cannot_extend_before_materialization(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)

    jobs = {job.job_id: job for job in store.jobs()}
    assert len(jobs) == 337
    assert set(jobs[MATERIALIZE_JOB_ID].dependencies) == {
        job.job_id for job in planned_asset_jobs(experiment)
    }
    _extend_plan(experiment, store)
    assert not any(job.job_id.startswith("base:") for job in store.jobs())
    assert scheduler_status(experiment)["counts"]["pending"] == 337


def test_scheduler_signal_directory_uses_storage_run_root(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)

    assert _signal_directory(experiment) == experiment.storage.run_root / "signals"


def test_full_matrix_workers_bind_the_protocol_cloudstorage_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    monkeypatch.setenv("XAI_CLOUD_STORAGE_LOCK_ROOT", "/tmp/wrong-lock-root")
    cache_root = tmp_path / "huggingface"
    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root / "hub"))

    assert _worker_environment(experiment) == {
        "XAI_CLOUD_STORAGE_LOCK_ROOT": str(experiment.storage.cloudstorage_lock_root),
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "hub"),
        "XAI_SIMPLE_STREAMING_INPUTS": "1",
        "XAI_SIMPLE_HOT_CACHE_GIB": "12",
        "XAI_SIMPLE_HOT_CACHE_ROOT": str(
            experiment.storage.selector_cache_root.parent / f"{experiment.experiment_id}-hot-cache"
        ),
        "XAI_SIMPLE_SPOOL_MAX_GIB": "48",
        "XAI_SIMPLE_SPOOL_MIN_FREE_GIB": "40",
        "XAI_SIMPLE_PHASE1_UPLOAD_GLOBAL_LIMIT": "8",
        "XAI_SIMPLE_PHASE1_STAGE_WORKERS": "2",
        "XAI_SIMPLE_PHASE1_PREFETCH_MAX_GIB": "64",
        "XAI_SIMPLE_PHASE1_PREFETCH_MIN_FREE_GIB": "40",
        "XAI_HF_PARQUET_RECOVERY_HUB": str(cache_root / "hub"),
        "XAI_FOOD101_TRAIN_RECOVERY_MARKER": str(
            experiment.storage.run_root / "hf-parquet-recovery" / "food101-hf-train.json"
        ),
        "XAI_FOOD101_VALIDATION_RECOVERY_MARKER": str(
            experiment.storage.run_root / "hf-parquet-recovery" / "food101-hf-validation.json"
        ),
        "XAI_IMAGENET100_TRAIN_RECOVERY_MARKER": str(
            experiment.storage.run_root / "hf-parquet-recovery" / "imagenet100-hf-train.json"
        ),
        "XAI_IMAGENET100_VALIDATION_RECOVERY_MARKER": str(
            experiment.storage.run_root / "hf-parquet-recovery" / "imagenet100-hf-validation.json"
        ),
        "XAI_IMAGENET100_TEST_RECOVERY_MARKER": str(
            experiment.storage.run_root / "hf-parquet-recovery" / "imagenet100-hf-test.json"
        ),
        "XAI_PLACES365_PARQUET_ROOT": str(
            cache_root
            / "hub"
            / "datasets--Andron00e--Places365-custom"
            / "snapshots"
            / "7895e75528d78c16e0c31182ce02e541f44ccaaf"
            / "data"
        ),
        "XAI_PLACES365_RECOVERY_MARKER": str(
            experiment.storage.run_root / "places365-recovery" / "places365-custom-train.json"
        ),
        "XAI_PLACES365_VALIDATION_PARQUET_ROOT": str(
            cache_root
            / "datasets--dpdl-benchmark--Places365-Validation"
            / "snapshots"
            / "f11b9b3c7ddd678ba92fd6862296b0d42c8723bb"
            / "data"
        ),
        "XAI_PLACES365_VALIDATION_RECOVERY_MARKER": str(
            experiment.storage.run_root / "places365-recovery" / "places365-validation-train.json"
        ),
    }
    protocol = load_protocol(ROOT / "configs" / "protocols" / "full-matrix.yaml")
    assert experiment.storage.cloudstorage_lock_root == Path(
        protocol.data["storage"]["lock_root"]
    )


def test_full_matrix_workers_reject_incoherent_hf_cache_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "huggingface"
    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "different-hub"))

    with pytest.raises(RuntimeError, match="HF_HUB_CACHE must equal HF_HOME/hub"):
        _worker_environment(_experiment(tmp_path))


def test_dataset_deferral_is_runtime_control_and_does_not_touch_queue(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    payload = write_deferred_datasets(
        experiment,
        datasets=["places365"],
        reason="defer the remaining Places365 matrix until the reference models finish",
    )

    assert payload["schema"] == DEFERRED_DATASETS_SCHEMA
    assert load_deferred_datasets(experiment) == frozenset({"places365"})
    assert deferred_datasets_path(experiment).is_file()
    assert experiment.scheduler_digest == _experiment(tmp_path).scheduler_digest

    cleared = clear_deferred_datasets(experiment)
    assert cleared["cleared_datasets"] == ["places365"]
    assert load_deferred_datasets(experiment) == frozenset()
    assert not deferred_datasets_path(experiment).exists()


def test_dataset_deferral_rejects_foreign_or_malformed_control(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    path = deferred_datasets_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": DEFERRED_DATASETS_SCHEMA,
                "experiment_id": "other-experiment",
                "scheduler_digest": experiment.scheduler_digest,
                "datasets": ["places365"],
                "reason": "test",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity is contradictory"):
        load_deferred_datasets(experiment)


def test_dataset_deferral_filters_only_dataset_bound_jobs(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    jobs = (
        QueueJob(
            job_id="matrix-reference-training:places365--places365-resnet18",
            kind="matrix-reference-training",
            command=("python", "--cell", "places365--places365-resnet18"),
            dependencies=(),
            resource_ids=(),
            reservation_bytes=None,
            status="pending",
            attempts=0,
            max_retries=1,
            pid=None,
            gpu_id=None,
            log_path="places.log",
        ),
        QueueJob(
            job_id="base:phase1:places365--model--clean--Saliency--digest",
            kind="phase1",
            command=("python", "--task-id", "places365--model--clean--Saliency--digest"),
            dependencies=(),
            resource_ids=(),
            reservation_bytes=None,
            status="pending",
            attempts=0,
            max_retries=1,
            pid=None,
            gpu_id=None,
            log_path="phase1.log",
        ),
        QueueJob(
            job_id="base:profile:resnet18--Saliency",
            kind="profile",
            command=("python", "--profile-id", "resnet18--Saliency"),
            dependencies=(),
            resource_ids=(),
            reservation_bytes=None,
            status="pending",
            attempts=0,
            max_retries=1,
            pid=None,
            gpu_id=None,
            log_path="profile.log",
        ),
    )
    store.submit_many(jobs)
    write_deferred_datasets(experiment, datasets=["places365"], reason="test")
    status = scheduler_status(experiment)
    assert status["deferred_datasets"] == ["places365"]
    assert status["deferred_pending_count"] == 2


def test_deferred_scope_freezes_only_ready_non_places365_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    cell = next(
        item for item in experiment.cells() if item.cell_id == "bloodmnist--bloodmnist-resnet18"
    )
    monkeypatch.setattr(
        full_matrix_scheduler,
        "compatibility_complete",
        lambda *_args, **_kwargs: True,
    )
    gate = _mark_scope_cell_ready(experiment, store, cell)

    write_deferred_datasets(experiment, datasets=["places365"], reason="test partial scope")
    scopes = _ensure_deferred_ready_scope(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
    )

    assert len(scopes) == 1
    scope = next(iter(scopes.values()))
    assert scope.deferred_datasets == ("places365",)
    assert scope.cell_ids == (cell.cell_id,)
    scope_job = next(
        job for job in store.jobs() if job.job_id == _scope_stage_id(scope, "materialize")
    )
    assert scope_job.dependencies == (gate.job_id,)
    scope_cells = tuple(
        scope_job.command[index + 1]
        for index, value in enumerate(scope_job.command[:-1])
        if value == "--scope-cell"
    )
    assert all(not item.startswith("places365--") for item in scope_cells)
    global_materialize = next(job for job in store.jobs() if job.job_id == MATERIALIZE_JOB_ID)
    assert len(global_materialize.dependencies) == 336
    assert gate.job_id in global_materialize.dependencies

    status = scheduler_status(experiment)
    assert status["execution_scopes"] == [
        {
            "scope_id": scope.scope_id,
            "scope_digest": scope.scope_digest,
            "deferred_datasets": ["places365"],
            "planned_cells": 1,
            "materialize_job_id": scope_job.job_id,
            "materialize_status": "pending",
        }
    ]


def test_deferred_scope_appends_newly_ready_cells_without_reusing_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    cells = [
        cell
        for cell in experiment.cells()
        if cell.dataset_id != "places365"
        and any(job.job_id == f"matrix-reference-training:{cell.cell_id}" for job in store.jobs())
    ]
    first, second = cells[:2]
    monkeypatch.setattr(
        full_matrix_scheduler,
        "compatibility_complete",
        lambda *_args, **_kwargs: True,
    )
    _mark_scope_cell_ready(experiment, store, first)

    first_scopes = _ensure_deferred_ready_scope(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
    )
    assert len(first_scopes) == 1
    assert next(iter(first_scopes.values())).cell_ids == (first.cell_id,)

    _mark_scope_cell_ready(experiment, store, second)
    second_scopes = _ensure_deferred_ready_scope(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
    )

    assert len(second_scopes) == 2
    assert {cell_id for scope in second_scopes.values() for cell_id in scope.cell_ids} == {
        first.cell_id,
        second.cell_id,
    }
    assert all(
        not set(left.cell_ids).intersection(right.cell_ids)
        for left in second_scopes.values()
        for right in second_scopes.values()
        if left.scope_id != right.scope_id
    )


def test_deferred_scope_waits_for_the_ready_compatibility_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    cells = [
        cell
        for cell in experiment.cells()
        if cell.dataset_id != "places365"
        and any(job.job_id == f"matrix-reference-training:{cell.cell_id}" for job in store.jobs())
    ]
    first, second = cells[:2]
    monkeypatch.setattr(
        full_matrix_scheduler,
        "compatibility_complete",
        lambda *_args, **_kwargs: True,
    )
    _mark_scope_cell_ready(experiment, store, first)
    second_gate = next(
        job for job in store.jobs() if job.job_id == f"matrix-compatibility:{second.cell_id}"
    )
    with store.connect() as connection:
        connection.executemany(
            "UPDATE jobs SET status='succeeded' WHERE job_id=?",
            [(dependency,) for dependency in second_gate.dependencies],
        )

    assert (
        _ensure_deferred_ready_scope(
            experiment,
            store,
            deferred_datasets=frozenset({"places365"}),
        )
        == {}
    )

    _mark_scope_cell_ready(experiment, store, second)
    scopes = _ensure_deferred_ready_scope(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
    )

    assert len(scopes) == 1
    assert next(iter(scopes.values())).cell_ids == tuple(sorted((first.cell_id, second.cell_id)))


def test_deferred_scope_excludes_blocked_compatibility_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    cell = next(
        item for item in experiment.cells() if item.cell_id == "bloodmnist--bloodmnist-resnet18"
    )
    monkeypatch.setattr(
        full_matrix_scheduler,
        "compatibility_complete",
        lambda *_args, **_kwargs: True,
    )
    gate = _mark_scope_cell_ready(experiment, store, cell)
    gate_path = experiment.compatibility_directory(cell) / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "gate_digest": "test-gate",
                "blocked_methods": ["LRP"],
                "failures": ["compatibility failed"],
            }
        ),
        encoding="utf-8",
    )

    scopes = _ensure_deferred_ready_scope(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
    )

    assert next(job for job in store.jobs() if job.job_id == gate.job_id).status == "succeeded"
    assert scopes == {}


def test_deferred_scope_prioritizes_ready_compatibility_before_new_training(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    cell = next(
        item for item in experiment.cells() if item.cell_id == "bloodmnist--bloodmnist-resnet18"
    )
    gate = next(job for job in store.jobs() if job.job_id == f"matrix-compatibility:{cell.cell_id}")
    with store.connect() as connection:
        connection.executemany(
            "UPDATE jobs SET status='succeeded' WHERE job_id=?",
            [(dependency,) for dependency in gate.dependencies],
        )
    unrelated = next(
        job
        for job in store.jobs()
        if job.kind == "matrix-reference-training" and "pneumoniamnist" in job.job_id
    )
    prerequisites = _scope_compatibility_prerequisites(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
        scopes={},
    )

    assert prerequisites == frozenset({gate.job_id})
    assert _priority(gate, scope_prerequisites=prerequisites, scopes={}) < _priority(
        unrelated,
        scope_prerequisites=prerequisites,
        scopes={},
    )
    assert (
        _scope_compatibility_prerequisites(
            experiment,
            store,
            deferred_datasets=frozenset(),
            scopes={},
        )
        == frozenset()
    )


def test_scoped_materialization_excludes_places365_from_every_component_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    cell = next(
        item for item in experiment.cells() if item.cell_id == "bloodmnist--bloodmnist-resnet18"
    )
    gate_path = experiment.compatibility_directory(cell) / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate_digest": "gate-digest",
                "blocked_methods": [],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(full_matrix_planner, "compatibility_complete", lambda *_args: True)
    scope = build_execution_scope(
        experiment,
        deferred_datasets=("places365",),
        cell_ids=(cell.cell_id,),
    )

    result = materialize_component_configs(experiment, scope=scope)
    paths = execution_paths(experiment, scope)
    base = yaml.safe_load(paths.base_config_path.read_text(encoding="utf-8"))
    assumptions = yaml.safe_load(paths.assumptions_config_path.read_text(encoding="utf-8"))
    prefix = yaml.safe_load(paths.prefix_config_path.read_text(encoding="utf-8"))

    assert result["execution_scope"]["scope_digest"] == scope.scope_digest
    assert result["scope_manifest"] == str(paths.scope_manifest_path)
    assert [row["id"] for row in base["datasets"]] == ["bloodmnist"]
    assert [row["dataset"] for row in base["models"]] == ["bloodmnist"]
    assert base["full_matrix"]["execution_scope"]["deferred_datasets"] == ["places365"]
    assert assumptions["full_matrix"]["execution_scope"]["scope_id"] == scope.scope_id
    assert prefix["full_matrix"]["execution_scope"]["planned_cells"] == [cell.cell_id]
    assert scope.scope_id in base["storage"]["remote_root"]
    assert "places365--" not in paths.base_config_path.read_text(encoding="utf-8")
    assert "places365--" not in paths.assumptions_config_path.read_text(encoding="utf-8")
    assert "places365--" not in paths.prefix_config_path.read_text(encoding="utf-8")


def test_completed_scoped_materialization_submits_scoped_base_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    cell = next(
        item for item in experiment.cells() if item.cell_id == "bloodmnist--bloodmnist-resnet18"
    )
    monkeypatch.setattr(
        full_matrix_scheduler,
        "compatibility_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(full_matrix_planner, "compatibility_complete", lambda *_args: True)
    _mark_scope_cell_ready(experiment, store, cell)
    scope = build_execution_scope(
        experiment,
        deferred_datasets=("places365",),
        cell_ids=(cell.cell_id,),
    )
    materialize_component_configs(experiment, scope=scope)

    scopes = _ensure_deferred_ready_scope(
        experiment,
        store,
        deferred_datasets=frozenset({"places365"}),
    )
    assert scopes == {scope.scope_id: scope}
    materialize_id = _scope_stage_id(scope, "materialize")
    assert next(job for job in store.jobs() if job.job_id == materialize_id).status == "succeeded"

    _extend_plan(experiment, store, scopes=scopes)

    base_jobs = tuple(
        job for job in store.jobs() if job.job_id.startswith(f"scope:{scope.scope_id}:base:")
    )
    assert base_jobs
    assert all(materialize_id in job.dependencies for job in base_jobs)


def test_deferred_places365_skips_its_recovery_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    monkeypatch.setattr(
        full_matrix_scheduler, "_worker_environment", lambda _experiment: {"ok": "1"}
    )
    monkeypatch.setattr(
        full_matrix_scheduler,
        "_preflight_places365_recovery",
        lambda _experiment: pytest.fail("deferred Places365 must not run its recovery preflight"),
    )

    assert full_matrix_scheduler._scheduler_worker_environment(
        experiment,
        deferred_datasets=frozenset({"places365"}),
    ) == {"ok": "1"}


def test_live_external_gpu_lease_is_reserved_without_signalling_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    job = QueueJob(
        job_id="matrix-reference-training:places365--places365-resnet18",
        kind="matrix-reference-training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )
    store.submit(job)
    store.start(job.job_id, pid=1234, gpu_id=1)
    monkeypatch.setattr("xai_ensemble.simple.full_matrix.scheduler.os.kill", lambda *_args: None)
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler._process_is_zombie",
        lambda _pid: False,
    )

    assert _live_external_gpu_ids(store, {}) == frozenset({1})


def test_zombie_external_gpu_lease_does_not_reserve_its_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    job = QueueJob(
        job_id="matrix-reference-training:places365--places365-resnet18",
        kind="matrix-reference-training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )
    store.submit(job)
    store.start(job.job_id, pid=1234, gpu_id=1)
    monkeypatch.setattr("xai_ensemble.simple.full_matrix.scheduler.os.kill", lambda *_args: None)
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler._process_is_zombie",
        lambda _pid: True,
    )

    assert _live_external_gpu_ids(store, {}) == frozenset()


def test_capacity_floors_accept_available_local_and_tmpfs_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=100 * 2**30),
    )

    assert _capacity_floor_failure(experiment) is None


def test_capacity_floor_stops_before_local_disk_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    calls = iter((SimpleNamespace(free=29 * 2**30),))
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler.shutil.disk_usage",
        lambda _path: next(calls),
    )

    assert _capacity_floor_failure(experiment) == (
        f"local_disk_below_floor free_bytes={29 * 2**30} floor_bytes={30 * 2**30}"
    )


def test_capacity_floor_stops_at_local_disk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    calls = iter((SimpleNamespace(free=30 * 2**30),))
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler.shutil.disk_usage",
        lambda _path: next(calls),
    )

    assert _capacity_floor_failure(experiment) == (
        f"local_disk_below_floor free_bytes={30 * 2**30} floor_bytes={30 * 2**30}"
    )


def test_capacity_floor_stops_before_tmpfs_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    calls = iter((SimpleNamespace(free=31 * 2**30), SimpleNamespace(free=39 * 2**30)))
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler.shutil.disk_usage",
        lambda _path: next(calls),
    )

    assert _capacity_floor_failure(experiment) == (
        f"shm_below_floor free_bytes={39 * 2**30} floor_bytes={40 * 2**30}"
    )


def test_capacity_floor_stops_at_tmpfs_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    calls = iter((SimpleNamespace(free=31 * 2**30), SimpleNamespace(free=40 * 2**30)))
    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler.shutil.disk_usage",
        lambda _path: next(calls),
    )

    assert _capacity_floor_failure(experiment) == (
        f"shm_below_floor free_bytes={40 * 2**30} floor_bytes={40 * 2**30}"
    )


@pytest.mark.parametrize(
    ("failure", "running", "expected"),
    (
        (None, False, "proceed"),
        ("shm_below_floor", True, "pause"),
        ("shm_below_floor", False, "stop"),
    ),
)
def test_capacity_action_waits_for_active_workers(
    failure: str | None, running: bool, expected: str
) -> None:
    assert _capacity_action(failure, running=running) == expected


def test_capacity_floor_uses_existing_tmpfs_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    selector_cache_root = tmp_path / "missing" / "selector-cache"
    experiment = replace(
        experiment,
        storage=replace(experiment.storage, selector_cache_root=selector_cache_root),
    )
    observed: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        observed.append(path)
        return SimpleNamespace(free=100 * 2**30)

    monkeypatch.setattr(
        "xai_ensemble.simple.full_matrix.scheduler.shutil.disk_usage",
        disk_usage,
    )

    assert _capacity_floor_failure(experiment) is None
    assert observed == [experiment.storage.run_root, tmp_path]


def test_static_matrix_gpu_jobs_preserve_headroom_on_smaller_l40s() -> None:
    job = QueueJob(
        job_id="matrix-reference-training:cell",
        kind="matrix-reference-training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )
    headroom = int(45 * 2**30 * 0.05)
    device_total = 45 * 2**30

    assert _requires_exclusive_matrix_gpu(job)
    assert _requires_strict_matrix_gpu(job)
    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=44 * 2**30,
            device_total_bytes=device_total,
            headroom_bytes=headroom,
        )
        == device_total - headroom
    )


@pytest.mark.parametrize("kind", ["matrix-reference-training", "matrix-compatibility", "training"])
def test_static_matrix_gpu_jobs_are_both_strictly_exclusive(kind: str) -> None:
    job = QueueJob(
        job_id=f"{kind}:cell",
        kind=kind,
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )

    assert _requires_exclusive_matrix_gpu(job)
    assert _requires_strict_matrix_gpu(job)


@pytest.mark.parametrize(
    "reservation_bytes",
    [_HIGH_RESERVATION_STRICT_THRESHOLD_BYTES, 7 * 2**30],
)
def test_measured_reference_training_packs_below_the_strict_threshold(
    reservation_bytes: int,
) -> None:
    job = QueueJob(
        job_id="matrix-reference-training:cell",
        kind="matrix-reference-training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=reservation_bytes,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )

    assert not _requires_exclusive_matrix_gpu(job)
    assert not _requires_strict_matrix_gpu(job)


@pytest.mark.parametrize("reservation_bytes", [None, 0, 44 * 2**30])
def test_unmeasured_reference_training_stays_strictly_exclusive(
    reservation_bytes: int | None,
) -> None:
    job = QueueJob(
        job_id="matrix-reference-training:cell",
        kind="matrix-reference-training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=reservation_bytes,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )

    assert _requires_exclusive_matrix_gpu(job)
    assert _requires_strict_matrix_gpu(job)


def test_ind_training_reservation_is_clamped_to_the_device_ceiling() -> None:
    job = QueueJob(
        job_id="scope:ready:ind:training:cell",
        kind="training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )
    headroom = int(45 * 2**30 * 0.05)
    device_total = 45 * 2**30

    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=job.reservation_bytes,
            device_total_bytes=device_total,
            headroom_bytes=headroom,
        )
        == device_total - headroom
    )


def test_clamped_reservation_fits_an_idle_l40s_without_double_headroom() -> None:
    # A 44 GiB IND training declaration clamps to the static ceiling
    # (device total - headroom).  An idle L40S still reports a few hundred
    # MiB of driver occupancy below the total, so the fits check must not
    # subtract the headroom a second time or the job can never start.
    job = QueueJob(
        job_id="scope:ready:ind:training:cell",
        kind="training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )
    device_total = 46068 * 2**20
    live_free = 45454 * 2**20
    headroom = int(device_total * 0.05)

    effective = _effective_matrix_reservation(
        job,
        reservation_bytes=job.reservation_bytes,
        device_total_bytes=device_total,
        headroom_bytes=headroom,
    )
    assert effective is not None and effective < job.reservation_bytes
    assert simple_scheduler._reservation_fits(
        job_kind=job.kind,
        reservation_bytes=effective,
        live_free_bytes=live_free,
        outstanding_bytes=0,
        headroom_bytes=0,
    )
    assert not simple_scheduler._reservation_fits(
        job_kind=job.kind,
        reservation_bytes=effective,
        live_free_bytes=live_free,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )


def test_large_source_method_reservation_is_clamped_to_the_device_ceiling() -> None:
    job = QueueJob(
        job_id="scope:ready:ind:source-method:cell--GradientShap",
        kind="source-method",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=int(43.433 * 2**30),
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="source-method.log",
    )
    headroom = int(45 * 2**30 * 0.05)
    device_total = 45 * 2**30

    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=job.reservation_bytes,
            device_total_bytes=device_total,
            headroom_bytes=headroom,
        )
        == device_total - headroom
    )


def test_large_source_method_reservation_is_strictly_exclusive() -> None:
    job = QueueJob(
        job_id="scope:ready:ind:source-method:cell--GradientShap",
        kind="source-method",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=_HIGH_RESERVATION_STRICT_THRESHOLD_BYTES + 1,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="source-method.log",
    )

    assert _requires_exclusive_matrix_gpu(job)
    assert _requires_strict_matrix_gpu(job)


def test_static_matrix_reservation_fits_without_a_clamp() -> None:
    job = QueueJob(
        job_id="matrix-reference-training:cell",
        kind="matrix-reference-training",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )
    requested = 44 * 2**30

    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=requested,
            device_total_bytes=48 * 2**30,
            headroom_bytes=2 * 2**30,
        )
        == requested
    )


def test_static_matrix_reservation_has_no_capacity() -> None:
    job = QueueJob(
        job_id="matrix-compatibility:cell",
        kind="matrix-compatibility",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )

    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=44 * 2**30,
            device_total_bytes=2 * 2**30,
            headroom_bytes=2 * 2**30,
        )
        is None
    )


def test_non_static_matrix_reservation_is_not_clamped() -> None:
    job = QueueJob(
        job_id="phase1:cell",
        kind="phase1",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=12 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="training.log",
    )

    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=12 * 2**30,
            device_total_bytes=2 * 2**30,
            headroom_bytes=1 * 2**30,
        )
        == 12 * 2**30
    )


def test_source_method_reservation_never_shrinks_to_fit_a_busy_device() -> None:
    # Regression for the OOM wave in which a 36 GiB source-method estimate was
    # clamped to the crumbs left by a busy device (93 MiB in the incident) and
    # became invisible to outstanding-capacity accounting, so further jobs
    # kept being admitted until the device ran out of memory. The effective
    # reservation must equal the declared estimate regardless of occupancy,
    # and the fits check must reject the job so it waits for capacity.
    job = QueueJob(
        job_id="scope:ready:ind:source-method:cell--CheferTransformerAttribution",
        kind="source-method",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=36 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="source-method.log",
    )
    requested = 36 * 2**30
    headroom = int(45 * 2**30 * 0.05)

    effective = _effective_matrix_reservation(
        job,
        reservation_bytes=requested,
        device_total_bytes=45 * 2**30,
        headroom_bytes=headroom,
    )
    assert effective == requested
    assert not _reservation_fits(
        job_kind=job.kind,
        reservation_bytes=effective,
        live_free_bytes=40 * 2**30,
        outstanding_bytes=10 * 2**30,
        headroom_bytes=headroom,
    )


def test_ceiling_clamped_reservation_still_blocks_co_admission() -> None:
    # An over-ceiling request clamps to the static device ceiling. The clamped
    # value fits only a genuinely empty device: even a 1 GiB publication tail
    # leaves less free memory than the clamped reservation once headroom is
    # subtracted, so no co-admission is possible while any worker remains.
    job = QueueJob(
        job_id="scope:ready:ind:source-method:cell--GradientShap",
        kind="source-method",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=int(43.4 * 2**30),
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="source-method.log",
    )
    headroom = int(45 * 2**30 * 0.05)
    device_total = 45 * 2**30

    effective = _effective_matrix_reservation(
        job,
        reservation_bytes=job.reservation_bytes,
        device_total_bytes=device_total,
        headroom_bytes=headroom,
    )
    assert effective == device_total - headroom
    assert _reservation_fits(
        job_kind=job.kind,
        reservation_bytes=effective,
        live_free_bytes=device_total,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )
    assert not _reservation_fits(
        job_kind=job.kind,
        reservation_bytes=effective,
        live_free_bytes=device_total - 1 * 2**30,
        outstanding_bytes=0,
        headroom_bytes=headroom,
    )


def test_static_matrix_gpu_reservation_is_not_rewritten_in_the_queue() -> None:
    job = QueueJob(
        job_id="matrix-compatibility:cell",
        kind="matrix-compatibility",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=44 * 2**30,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="compatibility.log",
    )

    assert job.reservation_bytes == 44 * 2**30
    assert (
        _effective_matrix_reservation(
            job,
            reservation_bytes=job.reservation_bytes,
            device_total_bytes=43 * 2**30,
            headroom_bytes=1 * 2**30,
        )
        == 42 * 2**30
    )
    assert job.reservation_bytes == 44 * 2**30


def test_cpu_launch_merges_worker_environment_and_preserves_release_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    cache_root = tmp_path / "huggingface"
    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root / "hub"))
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    output = tmp_path / "environment.json"
    job = QueueJob(
        job_id="matrix-manifest:environment",
        kind="matrix-manifest",
        command=(
            sys.executable,
            "-c",
            "import json, os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text("
            "json.dumps({key: os.environ.get(key) for key in sys.argv[2:]}))",
            str(output),
            "CUDA_VISIBLE_DEVICES",
            "XAI_CLOUD_STORAGE_LOCK_ROOT",
            "XAI_SIMPLE_GPU_RELEASE_PATH",
            "XAI_SIMPLE_GPU_RELEASE_TOKEN",
            "XAI_SIMPLE_GPU_RELEASE_JOB",
            "HF_HOME",
            "HF_HUB_CACHE",
            "XAI_PLACES365_PARQUET_ROOT",
            "XAI_PLACES365_RECOVERY_MARKER",
            "XAI_PLACES365_VALIDATION_PARQUET_ROOT",
            "XAI_PLACES365_VALIDATION_RECOVERY_MARKER",
        ),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=0,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path=str(tmp_path / "environment.log"),
    )
    store.submit(job)

    running = _launch_cpu(
        store,
        job,
        signal_directory=tmp_path / "signals",
        environment=_worker_environment(experiment),
    )
    assert running.process.wait(timeout=10) == 0
    running.log_handle.close()
    store.finish(job.job_id, exit_code=0)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "CUDA_VISIBLE_DEVICES": "",
        "XAI_CLOUD_STORAGE_LOCK_ROOT": str(experiment.storage.cloudstorage_lock_root),
        "XAI_SIMPLE_GPU_RELEASE_PATH": str(running.release_marker),
        "XAI_SIMPLE_GPU_RELEASE_TOKEN": running.release_token,
        "XAI_SIMPLE_GPU_RELEASE_JOB": job.job_id,
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "hub"),
        "XAI_PLACES365_PARQUET_ROOT": str(
            cache_root
            / "hub"
            / "datasets--Andron00e--Places365-custom"
            / "snapshots"
            / "7895e75528d78c16e0c31182ce02e541f44ccaaf"
            / "data"
        ),
        "XAI_PLACES365_RECOVERY_MARKER": str(
            experiment.storage.run_root / "places365-recovery" / "places365-custom-train.json"
        ),
        "XAI_PLACES365_VALIDATION_PARQUET_ROOT": str(
            cache_root
            / "datasets--dpdl-benchmark--Places365-Validation"
            / "snapshots"
            / "f11b9b3c7ddd678ba92fd6862296b0d42c8723bb"
            / "data"
        ),
        "XAI_PLACES365_VALIDATION_RECOVERY_MARKER": str(
            experiment.storage.run_root / "places365-recovery" / "places365-validation-train.json"
        ),
    }


def test_capacity_stop_terminates_owned_process_group_and_defers_queue_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    job = QueueJob(
        job_id="matrix-manifest:capacity-stop",
        kind="matrix-manifest",
        command=(
            sys.executable,
            "-c",
            "import pathlib, subprocess, sys, time; "
            "child = subprocess.Popen((sys.executable, '-c', sys.argv[3], sys.argv[4])); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8'); "
            "pathlib.Path(sys.argv[2]).touch(); time.sleep(60)",
            str(tmp_path / "child.pid"),
            str(tmp_path / "child.ready"),
            "import pathlib, signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)",
            str(tmp_path / "child.signal-ready"),
        ),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=0,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path=str(tmp_path / "capacity-stop.log"),
    )
    store.submit(job)
    log_handle = Path(job.log_path).open("ab", buffering=0)
    process = subprocess.Popen(
        job.command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        store.start(job.job_id, pid=process.pid, gpu_id=-1)
        running_job = next(
            item for item in store.jobs(status="running") if item.job_id == job.job_id
        )
        active = RunningProcess(
            running_job,
            process,
            log_handle,
            0,
            tmp_path / "missing-release-marker.json",
            "capacity-stop-token",
        )

        ready = tmp_path / "child.ready"
        deadline = time.monotonic() + 5.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file()
        signal_ready = tmp_path / "child.signal-ready"
        deadline = time.monotonic() + 5.0
        while not signal_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert signal_ready.is_file()
        child_pid = int((tmp_path / "child.pid").read_text(encoding="utf-8"))

        original_record_event = store.record_event
        event_attempts = 0

        def fail_first_capacity_event(job_id: str, event: str, detail: str | None = None) -> None:
            nonlocal event_attempts
            event_attempts += 1
            if event_attempts == 1:
                raise OSError("simulated telemetry write failure")
            original_record_event(job_id, event, detail)

        monkeypatch.setattr(store, "record_event", fail_first_capacity_event)
        assert (
            _stop_running_for_capacity(
                store,
                {job.job_id: active},
                reason="shm_below_floor free_bytes=0 floor_bytes=1",
                grace_seconds=0.1,
            )
            == ()
        )

        assert process.wait(timeout=5) == -signal.SIGTERM
        deadline = time.monotonic() + 5.0
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_group_exists(process.pid)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert next(item for item in store.jobs() if item.job_id == job.job_id).status == "running"
        with store.connect() as connection:
            events = [
                row["event"]
                for row in connection.execute(
                    "SELECT event FROM events WHERE job_id=? ORDER BY sequence", (job.job_id,)
                )
            ]
        assert "scheduler_capacity_stop_sigterm" in events
        assert "scheduler_capacity_stop_sigkill" in events

        store.recover_orphans()
        recovered = next(item for item in store.jobs() if item.job_id == job.job_id)
        assert recovered.status == "pending"
        assert recovered.job_id == job.job_id
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_handle.close()


@pytest.mark.parametrize("pgid", (0, -1))
def test_process_group_exists_rejects_nonpositive_group_ids(pgid: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _process_group_exists(pgid)


@pytest.mark.parametrize(
    ("outcome", "error", "expected_suffix", "expected_detail"),
    (
        ("sent", None, "", "pid=17"),
        ("already_exited", None, "", "pid=17,already_exited"),
        ("failed", "permission denied", "_failed", "pid=17,error=permission denied"),
    ),
)
def test_capacity_signal_detail_reports_the_actual_signal_outcome(
    outcome: str,
    error: str | None,
    expected_suffix: str,
    expected_detail: str,
) -> None:
    assert _capacity_signal_detail(pid=17, outcome=outcome, error=error) == (
        expected_suffix,
        expected_detail,
    )


def test_component_namespace_preserves_internal_dependencies() -> None:
    first = QueueJob(
        job_id="profile:one",
        kind="profile",
        command=("python", "worker.py"),
        dependencies=(),
        resource_ids=("one",),
        reservation_bytes=None,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="/tmp/one.log",
    )
    second = replace(first, job_id="phase1:two", kind="phase1", dependencies=(first.job_id,))

    namespaced = _prefix_jobs(
        (first, second),
        "base",
        dependency=MATERIALIZE_JOB_ID,
        log_directory=Path("/logs"),
    )

    assert [job.job_id for job in namespaced] == ["base:profile:one", "base:phase1:two"]
    assert namespaced[0].dependencies == (MATERIALIZE_JOB_ID,)
    assert namespaced[1].dependencies == ("base:profile:one", MATERIALIZE_JOB_ID)


def test_dynamic_queue_expands_base_only_after_materialization_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    _submit_initial_plan(experiment, store)
    with store.connect() as connection:
        connection.execute(
            "UPDATE jobs SET status='succeeded' WHERE job_id=?",
            (MATERIALIZE_JOB_ID,),
        )

    first = QueueJob(
        job_id="profile:one",
        kind="profile",
        command=("true",),
        dependencies=(),
        resource_ids=(),
        reservation_bytes=None,
        status="pending",
        attempts=0,
        max_retries=1,
        pid=None,
        gpu_id=None,
        log_path="profile.log",
    )
    second = replace(first, job_id="phase1:one", kind="phase1", dependencies=(first.job_id,))

    def submit_base(_experiment, collector, *, include_phase2):
        assert include_phase2 is True
        collector.submit(first)
        collector.submit(second)
        return {"profile": 1, "phase1": 1}

    monkeypatch.setattr(simple_config, "load_experiment", lambda _path: object())
    monkeypatch.setattr(simple_scheduler, "submit_plan", submit_base)
    _extend_plan(experiment, store)

    base_jobs = [job for job in store.jobs() if job.job_id.startswith("base:")]
    assert [job.job_id for job in base_jobs] == ["base:phase1:one", "base:profile:one"]
    assert all(MATERIALIZE_JOB_ID in job.dependencies for job in base_jobs)
    summary = next(job for job in store.jobs() if job.job_id == BASE_SUMMARY_JOB_ID)
    assert set(summary.dependencies) == {job.job_id for job in base_jobs}


def test_run_requires_explicit_full_matrix_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm-full-matrix"):
        full_matrix_cli._run(Namespace(confirm_full_matrix=False))


def test_prepare_prefix_inputs_builds_the_evaluator_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "results"
    experiment = SimpleNamespace(digest="d" * 64)
    prefix = SimpleNamespace(digest="p" * 64)
    readiness = {
        "catalog_path": str(tmp_path / "noise-prefix" / "input-catalog.json"),
        "catalog_digest": "c" * 64,
        "tasks": 20,
    }

    monkeypatch.setattr(full_matrix_cli, "_load", lambda _args: experiment)
    monkeypatch.setattr(full_matrix_cli, "_loaded_scope", lambda _args, _exp: None)
    monkeypatch.setattr(
        full_matrix_cli,
        "execution_paths",
        lambda _exp, _scope: SimpleNamespace(
            prefix_config_path=tmp_path / "prefix.yaml", result_root=result_root
        ),
    )
    monkeypatch.setattr(
        "xai_ensemble.simple.noise_prefix.config.load_noise_prefix_experiment",
        lambda _path: prefix,
    )
    monkeypatch.setattr(
        "xai_ensemble.simple.noise_prefix.inputs.materialize_missing_rank_ready",
        lambda _prefix: {"status": "complete", "generated": 0},
    )
    monkeypatch.setattr(
        "xai_ensemble.simple.noise_prefix.inputs.readiness_report",
        lambda _prefix: readiness,
    )

    assert full_matrix_cli._prepare_prefix_inputs(Namespace()) == 0
    payload = json.loads((result_root / "control" / "prefix-ready.json").read_text())
    assert payload["input_catalog"] == {
        "path": readiness["catalog_path"],
        "catalog_digest": readiness["catalog_digest"],
        "tasks": readiness["tasks"],
    }


def test_plan_is_read_only_and_status_uses_the_single_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)

    # Direct status is intentionally empty before submission. The public plan
    # path only materializes an in-memory projection and does not create SQLite.
    assert scheduler_status(experiment)["exists"] is False
    monkeypatch.setattr(full_matrix_cli, "_load", lambda _args: experiment)
    full_matrix_cli._plan(Namespace(config=CONFIG, output=None))
    capsys.readouterr()
    assert not experiment.runtime.database_path.exists()
