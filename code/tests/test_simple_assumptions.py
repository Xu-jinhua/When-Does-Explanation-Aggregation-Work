from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.simple.artifacts import ArtifactError, ArtifactStore, load_safetensors
from xai_ensemble.simple.assumptions import phase1 as assumption_phase1
from xai_ensemble.simple.assumptions import prepare as assumption_prepare
from xai_ensemble.simple.assumptions import scheduler as assumption_scheduler
from xai_ensemble.simple.assumptions import table_priority as assumption_table_priority
from xai_ensemble.simple.assumptions.artifacts import (
    EVALUATION_SCHEMA_VERSION,
    RANK_SCHEMA_VERSION,
    completed_evaluation_manifest,
    completed_rank_manifest,
    existing_shard_records,
    output_store,
    publish_manifest,
    publish_shard,
    task_spool_path,
)
from xai_ensemble.simple.assumptions.config import load_assumption_experiment
from xai_ensemble.simple.assumptions.ind_table_summary import (
    AGGREGATE_RULE_KEYS,
    PAPER_TABLE_COLUMN_ORDER,
    _best_individual_row,
    _full_tex_text,
    _require_aligned_group,
    _signed_robustness,
    _single_method_statistics,
    _tex_values,
)
from xai_ensemble.simple.assumptions.phase1 import (
    SOURCE_RANK_INPUT_REPRESENTATION,
    SOURCE_RANK_INPUT_SCHEMA_VERSION,
    _generation_experiment,
    _reference_cache_identity,
)
from xai_ensemble.simple.assumptions.prepare import (
    completed_spearman_family,
    prepare_spearman_family,
    spearman_identity,
    spearman_local_artifact_path,
)
from xai_ensemble.simple.assumptions.ranks import AttributionSource, _load_attribution_shard
from xai_ensemble.simple.assumptions.readiness import (
    _artifact_group,
    _profile_group,
    _relprop_requirements,
)
from xai_ensemble.simple.assumptions.scheduler import (
    _effective_headroom_fraction,
    _execution_priority,
    _failed_noise_jobs,
    _noise_barrier_active,
    planned_job_ids,
    spearman_job_id,
    submit_plan,
)
from xai_ensemble.simple.assumptions.selection import _distances
from xai_ensemble.simple.assumptions.summary import (
    NOISE_ORDER,
    _best_individual_rows,
    _noise_analysis,
    _oracle_noise_best_individual_rows,
    _validate_noise_manifest,
)
from xai_ensemble.simple.phase1 import (
    CLEAN_MODEL_TARGET_POLICY,
    FULL_REFERENCE_CLEAN_TARGET_POLICY,
    FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
    TASK_MODEL_OUTPUT_SOURCE,
    _manifest_value,
    _Phase1Publisher,
    _validate_completed_source,
)
from xai_ensemble.simple.rank_ready import (
    normalize_spatial,
    rank_field,
    scores_to_ranks,
    simpleavg_score_field,
)
from xai_ensemble.simple.scheduler import QueueJob, SimpleJobStore

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/simple/paper-assumptions.yaml"
GENERALIZATION_CONFIG = (
    ROOT / "configs/simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml"
)
GENERALIZATION_PREFIX_CONFIG = (
    ROOT / "configs/simple/paper-noise-generalization-pathmnist-densenet121-prefix.yaml"
)


def _fixture_base_config(tmp_path: Path) -> Path:
    """Materialize tiny stand-in assets for the paper-main base config.

    The published configs carry ``/path/to/...`` asset placeholders, but DAG
    expansion hashes manifests, checkpoints, and mean artifacts. Tests
    therefore write byte-level fixtures; their content is irrelevant beyond
    existing on disk.
    """
    value = yaml.safe_load((ROOT / "configs/simple/paper-main.yaml").read_text(encoding="utf-8"))
    assets = tmp_path / "assets"
    assets.mkdir()
    for dataset in value["datasets"]:
        manifest = assets / f"{dataset['id']}.manifest.json"
        manifest.write_text(json.dumps({"fixture": dataset["id"]}), encoding="utf-8")
        dataset["manifest_path"] = str(manifest)
    for model in value["models"]:
        checkpoint = assets / f"{model['id']}.pt"
        checkpoint.write_bytes(f"fixture-checkpoint-{model['id']}".encode())
        model["checkpoint_path"] = str(checkpoint)
        mean = assets / f"{model['id']}-mean"
        mean.mkdir()
        (mean / "manifest.json").write_text(json.dumps({"fixture": model["id"]}), encoding="utf-8")
        model["mean_path"] = str(mean)
    value["methods_file"] = str(ROOT / "configs/simple/methods.yaml")
    path = tmp_path / "paper-main.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _fixture_experiment(tmp_path: Path):
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["base_config"] = str(_fixture_base_config(tmp_path))
    value["selection"]["spearman_calibration_config"] = str(
        ROOT / "configs" / "pilots" / "noise_gof_cost.yaml"
    )
    path = tmp_path / "paper-assumptions.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    experiment = load_assumption_experiment(path)
    _write_fixture_profiles(experiment)
    return experiment


# Batch sizes mirror the measured main-run profiles: ViT values match
# table_priority.MEASURED_METHOD_PEAK_BYTES, CNN values stay within the
# configured phase1_batch_caps.
_FIXTURE_BATCHES = {
    ("cnn", "Saliency"): 256,
    ("cnn", "InputXGradient"): 256,
    ("cnn", "IntegratedGradients"): 32,
    ("cnn", "GuidedBackprop"): 256,
    ("cnn", "Deconvolution"): 256,
    ("cnn", "FeatureAblation"): 1024,
    ("cnn", "Occlusion"): 1024,
    ("cnn", "DeepLift"): 256,
    ("cnn", "GradientShap"): 64,
    ("cnn", "DeepLiftShap"): 64,
    ("cnn", "LRP"): 256,
    ("vit", "Saliency"): 256,
    ("vit", "InputXGradient"): 256,
    ("vit", "IntegratedGradients"): 4,
    ("vit", "FeatureAblation"): 1024,
    ("vit", "Occlusion"): 1024,
    ("vit", "GradientShap"): 16,
    ("vit", "CheferTransformerAttribution"): 128,
    ("vit", "PartialLRP"): 128,
    ("vit", "FullLRP"): 128,
    ("vit", "GradientAttentionRollout"): 128,
    ("vit", "AttentionGradCAM"): 128,
}


def _write_fixture_profiles(experiment) -> None:
    from xai_ensemble.simple.profiler import profile_identity, profile_path

    base = experiment.base
    for profile in base.profiles():
        batch = _FIXTURE_BATCHES[(profile.architecture, profile.method.family)]
        peak = batch * 2**20
        payload = {
            "schema_version": 1,
            "profile_id": profile.profile_id,
            "identity_digest": object_sha256(profile_identity(profile)),
            "model_key": profile.model_key,
            "architecture": profile.architecture,
            "method": profile.method.family,
            "variant": profile.method.variant,
            "params": dict(profile.method.params),
            "precision": "fp32",
            "input_shape": [3, profile.input_size, profile.input_size],
            "selected_batch_size": batch,
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": peak,
            "device_total_bytes": 48 * 2**30,
            "headroom_fraction": 0.10,
            "probe_kind": "fixture",
            "measurements": [
                {
                    "batch_size": batch,
                    "passed": True,
                    "peak_allocated_bytes": peak,
                    "peak_reserved_bytes": peak,
                    "elapsed_seconds": 0.1,
                    "reason": None,
                }
            ],
            "created_utc": "fixture",
        }
        destination = profile_path(base, profile.profile_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload), encoding="utf-8")


def test_assumption_scheduler_headroom_runtime_override() -> None:
    assert _effective_headroom_fraction(0.10, None) == 0.10
    assert _effective_headroom_fraction(0.10, 0.05) == 0.05
    with pytest.raises(ValueError, match="headroom_fraction"):
        _effective_headroom_fraction(0.10, -0.01)
    with pytest.raises(ValueError, match="headroom_fraction"):
        _effective_headroom_fraction(0.10, 0.50)


def test_assumption_scheduler_prioritizes_noise_before_ind() -> None:
    def job(job_id: str, kind: str) -> QueueJob:
        return QueueJob(
            job_id=job_id,
            kind=kind,
            command=("true",),
            dependencies=(),
            resource_ids=(),
            reservation_bytes=1,
            status="pending",
            attempts=0,
            max_retries=1,
            pid=None,
            gpu_id=None,
            log_path="job.log",
        )

    prerequisites = job("selection:select--cell--spearman--digest", "selection")
    noise = job("rank:rank--oracle-noise--cell--spearman--clean--digest", "rank")
    source = job("source-phase1:source-phase1--cell--source-00--clean--digest", "source-phase1")
    matched = job("rank:rank--matched-naive--cell--source-00--clean--digest", "rank")

    assert _execution_priority(prerequisites) < _execution_priority(noise)
    assert _execution_priority(noise) < _execution_priority(source)
    assert _execution_priority(source) < _execution_priority(matched)

    assert _noise_barrier_active((noise, source, matched)) is True
    completed_noise = replace(noise, status="succeeded")
    assert _noise_barrier_active((completed_noise, source, matched)) is False

    failed_noise = replace(noise, status="failed")
    blocked_noise = replace(noise, status="blocked")
    assert _noise_barrier_active((failed_noise, source, matched)) is False
    assert _failed_noise_jobs((failed_noise, source, matched)) == (failed_noise.job_id,)
    assert _failed_noise_jobs((blocked_noise, source, matched)) == (blocked_noise.job_id,)


def test_assumption_scheduler_checks_runtime_before_opening_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xai_ensemble.simple.assumptions import readiness as assumption_readiness

    experiment = _local_experiment(tmp_path)

    def reject_runtime(_experiment):
        raise RuntimeError("missing runtime provider")

    monkeypatch.setattr(
        assumption_readiness,
        "require_runtime_dependencies",
        reject_runtime,
    )

    with pytest.raises(RuntimeError, match="missing runtime provider"):
        assumption_scheduler.run_scheduler(experiment, poll_seconds=0.01)
    assert not experiment.runtime.database_path.exists()


def test_assumption_dag_has_the_paper_scope(tmp_path: Path) -> None:
    experiment = load_assumption_experiment(CONFIG)

    assert experiment.settings == ("ind", "matched-naive", "oracle-noise")
    assert experiment.digest == "ccc9c72ecf3f3bb4ae56ea5469b6254e1df49722dc11bb992054d91484035c08"
    assert (
        experiment.scheduler_digest
        == "9e37e594baf0e7c77d0cb46e2a848dbf8635bd426d9bff28aa0630c6fdfb9562"
    )
    assert len(experiment.cells()) == 4

    # Task expansion hashes dataset manifests, checkpoints, and mean
    # artifacts, so it runs against byte-level fixtures.
    experiment = _fixture_experiment(tmp_path)
    assert len(experiment.partition_tasks()) == 4
    assert len(experiment.training_tasks()) == 44
    assert len(experiment.source_phase1_tasks()) == 220
    assert len(experiment.selection_tasks()) == 8
    assert len(experiment.rank_tasks()) == 280
    assert len(experiment.evaluation_tasks()) == 280
    assert len(planned_job_ids(experiment)) == 837

    settings = [task.setting for task in experiment.rank_tasks()]
    assert settings.count("ind") == 20
    assert settings.count("matched-naive") == 220
    assert settings.count("oracle-noise") == 40
    assert all(
        task.condition.condition_id in {item.condition_id for item in experiment.base.conditions}
        for task in experiment.rank_tasks()
    )


def test_oracle_noise_only_setting_registers_no_source_bank(tmp_path: Path) -> None:
    original = _fixture_experiment(tmp_path)
    base = replace(
        original.base,
        datasets=(original.base.datasets[0],),
        models=(original.base.models[0],),
    )
    experiment = replace(original, base=base, settings=("oracle-noise",))

    assert experiment.partition_tasks() == ()
    assert experiment.training_tasks() == ()
    assert experiment.source_phase1_tasks() == ()
    assert _profile_group(experiment)["skipped"] == "no_source_bank_setting"
    assert _relprop_requirements(experiment) == ()
    assert len(experiment.selection_tasks()) == 2
    assert len(experiment.rank_tasks()) == 10
    assert len(experiment.evaluation_tasks()) == 10
    assert len(planned_job_ids(experiment)) == 23


def test_pathmnist_generalization_configs_bind_one_cnn_cell() -> None:
    experiment = load_assumption_experiment(GENERALIZATION_CONFIG)
    assert experiment.assumption_id == "noise-generalization-pathmnist-densenet121-oracle-noise-v1"
    assert experiment.settings == ("oracle-noise",)
    assert [cell.cell_id for cell in experiment.cells()] == ["pathmnist--pathmnist-densenet121"]
    assert experiment.cells()[0].methods == (
        "Saliency",
        "InputXGradient",
        "IntegratedGradients",
        "GuidedBackprop",
        "Deconvolution",
        "FeatureAblation",
        "Occlusion",
        "DeepLift",
        "GradientShap",
        "DeepLiftShap",
        "LRP",
    )

    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment

    prefix = load_noise_prefix_experiment(GENERALIZATION_PREFIX_CONFIG)
    assert prefix.sweep_id == "noise-generalization-pathmnist-densenet121-prefix-v1"
    assert prefix.q_values == tuple(range(2, 12))
    assert prefix.rules == ("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze")


def test_oracle_noise_only_scheduler_has_only_selection_rank_and_evaluation_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _fixture_experiment(tmp_path)
    base = replace(
        original.base,
        datasets=(original.base.datasets[0],),
        models=(original.base.models[0],),
    )
    runtime = replace(
        original.runtime,
        database_path=tmp_path / "jobs.sqlite3",
        log_directory=tmp_path / "logs",
    )
    experiment = replace(
        original,
        base=base,
        runtime=runtime,
        settings=("oracle-noise",),
    )
    store = SimpleJobStore(runtime.database_path, experiment_digest=experiment.scheduler_digest)
    monkeypatch.setattr(
        assumption_scheduler,
        "_complete",
        lambda *_args, **_kwargs: pytest.fail("new queue submission scanned remote artifacts"),
    )
    monkeypatch.setattr(
        assumption_scheduler,
        "_spearman_complete",
        lambda *_args, **_kwargs: pytest.fail("new queue submission scanned Spearman artifacts"),
    )

    counts = submit_plan(experiment, store, scan_existing=False)

    assert counts == {
        "partition": 0,
        "spearman": 1,
        "training": 0,
        "source-phase1": 0,
        "selection": 2,
        "rank": 10,
        "evaluation": 10,
    }
    jobs = tuple(store.jobs())
    assert {job.kind for job in jobs} == {"spearman", "selection", "rank", "evaluation"}
    assert all(
        not job.dependencies
        or all(
            value.startswith(("spearman:", "selection:", "rank:", "evaluation:"))
            for value in job.dependencies
        )
        for job in jobs
    )


def test_ind_assignment_is_seeded_bijective_and_recorded_in_tasks(tmp_path: Path) -> None:
    experiment = _fixture_experiment(tmp_path)
    for cell in experiment.cells():
        first = experiment.method_assignment(cell)
        second = experiment.method_assignment(cell)
        assert first == second
        assert tuple(source for source, _ in first) == experiment.source_ids(cell)
        assert set(method for _, method in first) == set(cell.methods)
        ind = next(
            task
            for task in experiment.rank_tasks()
            if task.cell.cell_id == cell.cell_id
            and task.setting == "ind"
            and task.condition.kind == "clean"
        )
        assert ind.source_method_pairs == first


def test_table_priority_dag_runs_only_assigned_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _local_experiment(tmp_path)
    paths = assumption_table_priority.table_priority_paths(
        experiment, root=tmp_path / "table-priority"
    )
    store = SimpleJobStore(
        paths.database,
        experiment_digest=assumption_table_priority.table_priority_digest(experiment, paths),
    )
    monkeypatch.setattr(assumption_table_priority, "_complete", lambda *_args: False)
    monkeypatch.setattr(
        assumption_table_priority,
        "_method_reservation_bytes",
        lambda *_args: 4 * 2**30,
    )

    counts = assumption_table_priority.submit_plan(experiment, store, paths=paths)

    assert counts == {
        "partition": 4,
        "training": 44,
        "source-method": 220,
        "rank": 20,
        "evaluation": 20,
    }
    assert len(assumption_table_priority.planned_job_ids(experiment)) == 308
    assert len(assumption_table_priority.matched_evaluation_tasks(experiment)) == 60
    jobs = {job.job_id: job for job in store.jobs()}
    for task in assumption_table_priority.ind_rank_tasks(experiment):
        job = jobs[f"rank:{task.task_id}"]
        assert len(job.dependencies) == 11
        assert all(dependency.startswith("source-method:") for dependency in job.dependencies)

    monkeypatch.setattr(
        assumption_table_priority,
        "_complete",
        lambda *_args: pytest.fail("existing priority jobs triggered a remote rescan"),
    )
    assert assumption_table_priority.submit_plan(experiment, store, paths=paths) == counts


def test_compact_table_priority_generates_three_full_sources_and_eleven_assigned_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _local_experiment(tmp_path)
    experiment = replace(original, matched_source_selection="sha256_v1")
    paths = assumption_table_priority.table_priority_paths(
        experiment, root=tmp_path / "compact-table-priority"
    )
    store = SimpleJobStore(
        paths.database,
        experiment_digest=assumption_table_priority.table_priority_digest(experiment, paths),
    )
    monkeypatch.setattr(assumption_table_priority, "_complete", lambda *_args: False)
    monkeypatch.setattr(
        assumption_table_priority,
        "_method_reservation_bytes",
        lambda *_args: 4 * 2**30,
    )

    counts = assumption_table_priority.submit_plan(experiment, store, paths=paths)

    # Four cells x five conditions x (3*11 selected matched sources plus
    # eight assigned-only sources) = 820 source-method jobs.
    assert counts == {
        "partition": 4,
        "training": 44,
        "source-method": 820,
        "rank": 80,
        "evaluation": 80,
    }
    assert len(assumption_table_priority.matched_source_ids(experiment)) == 4
    assert all(
        len(source_ids) == 3
        for source_ids in assumption_table_priority.matched_source_ids(experiment).values()
    )
    assert len(assumption_table_priority.planned_job_ids(experiment)) == 1028
    assert len(assumption_table_priority.planned_rank_tasks(experiment)) == 80
    assert len(assumption_table_priority.planned_evaluation_tasks(experiment)) == 80

    selected = assumption_table_priority.matched_source_ids(experiment)
    for cell in experiment.cells():
        selected_ids = set(selected[cell.cell_id])
        requirements = [
            (scope.source_id, family)
            for scope, family in assumption_table_priority.assigned_method_requirements(experiment)
            if scope.cell.cell_id == cell.cell_id and scope.condition.kind == "clean"
        ]
        assert len(requirements) == 41
        assert {
            source_id
            for source_id, _family in requirements
            if source_id in selected_ids
        } == selected_ids
        assert all(
            sum(source_id == selected_id for source_id, _family in requirements) == 11
            for selected_id in selected_ids
        )


def _vit_method_scope(experiment, family: str):
    for scope in experiment.source_phase1_tasks():
        if scope.cell.reference_model.architecture != "vit":
            continue
        training = experiment.find_training_task(scope.training_task_id)
        checkpoint, _ = assumption_table_priority.checkpoint_paths(experiment, training)
        method_task = next(
            (
                task
                for task in experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
                if task.family == family
            ),
            None,
        )
        if method_task is not None:
            return scope, method_task
    raise AssertionError(f"No vit scope provides method family {family}")


def test_method_reservation_prefers_measured_peaks(tmp_path: Path) -> None:
    experiment = _local_experiment(tmp_path)
    scope, method_task = _vit_method_scope(experiment, "FullLRP")
    batches = {
        assumption_table_priority._base_profile(experiment, scope, variant)[1]
        for variant in method_task.variants
    }
    keys = {("vit", "FullLRP", batch) for batch in batches}
    table = assumption_table_priority.MEASURED_METHOD_PEAK_BYTES
    assert keys <= set(table), f"measured peaks missing for batches {sorted(batches)}"
    peak = max(table[key] for key in keys)
    expected = int(
        (peak + assumption_table_priority.METHOD_RUNTIME_OVERHEAD_BYTES)
        * assumption_table_priority.METHOD_RESERVATION_SAFETY_FACTOR
    )
    expected = min(
        experiment.runtime.phase1_reservation_bytes["vit"],
        max(assumption_table_priority.MINIMUM_METHOD_RESERVATION_BYTES, expected),
    )
    value = assumption_table_priority._method_reservation_bytes(experiment, scope, "FullLRP")
    assert value == expected
    # The stored profiles for relprop families were inflated several-fold; the
    # measured table must land far below the historical ~35 GiB reservation.
    assert value < 12 * 2**30


def test_method_reservation_falls_back_to_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _local_experiment(tmp_path)
    scope, method_task = _vit_method_scope(experiment, "FullLRP")
    monkeypatch.setattr(assumption_table_priority, "MEASURED_METHOD_PEAK_BYTES", {})
    batches = {
        assumption_table_priority._base_profile(experiment, scope, variant)[1]
        for variant in method_task.variants
    }
    profile_peak = 20 * 2**30

    def fake_profile(_base, profile_class):
        return SimpleNamespace(
            profile_id=profile_class.profile_id,
            measurements=[
                SimpleNamespace(
                    batch_size=batch,
                    passed=True,
                    peak_allocated_bytes=profile_peak,
                    peak_reserved_bytes=profile_peak,
                )
                for batch in batches
            ],
        )

    monkeypatch.setattr(assumption_table_priority, "load_profile", fake_profile)
    value = assumption_table_priority._method_reservation_bytes(experiment, scope, "FullLRP")
    expected = int(
        (profile_peak + assumption_table_priority.METHOD_RUNTIME_OVERHEAD_BYTES)
        * assumption_table_priority.METHOD_RESERVATION_SAFETY_FACTOR
    )
    expected = min(
        experiment.runtime.phase1_reservation_bytes["vit"],
        max(assumption_table_priority.MINIMUM_METHOD_RESERVATION_BYTES, expected),
    )
    assert value == expected

    monkeypatch.setattr(assumption_table_priority, "load_profile", lambda *_args: None)
    with pytest.raises(FileNotFoundError, match="Missing Phase 1 profile"):
        assumption_table_priority._method_reservation_bytes(experiment, scope, "FullLRP")


def test_source_phase1_method_task_selects_only_requested_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _local_experiment(tmp_path)
    scope = experiment.source_phase1_tasks()[0]
    checkpoint = tmp_path / "inference.pt"
    observed = {}
    monkeypatch.setattr(
        assumption_phase1,
        "ensure_checkpoint",
        lambda *_args: (checkpoint, {"status": "complete"}),
    )

    def capture(_experiment, _scope, method_tasks, *, device):
        observed["families"] = tuple(task.family for task in method_tasks)
        observed["device"] = device
        return ({"status": "complete"},)

    monkeypatch.setattr(assumption_phase1, "_run_source_phase1_methods", capture)

    value = assumption_phase1.run_source_phase1_method_task(
        experiment, scope, "Saliency", device="cuda:1"
    )

    assert value == ({"status": "complete"},)
    assert observed == {"families": ("Saliency",), "device": "cuda:1"}


def test_incomplete_source_method_set_does_not_publish_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _local_experiment(tmp_path)
    scope = experiment.source_phase1_tasks()[0]
    checkpoint = tmp_path / "inference.pt"
    monkeypatch.setattr(
        assumption_phase1,
        "_source_identity",
        lambda *_args, **_kwargs: {"digest": "a" * 64},
    )
    monkeypatch.setattr(
        assumption_phase1,
        "_completed_method_manifests",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        assumption_phase1,
        "publish_manifest",
        lambda *_args, **_kwargs: pytest.fail("incomplete scope was published"),
    )

    value = assumption_phase1._publish_scope_if_complete(
        experiment,
        scope,
        checkpoint=checkpoint,
        training_manifest={},
        reference_manifest={},
        reference_task=SimpleNamespace(),
        store=ArtifactStore(
            assumption_phase1._generation_experiment(experiment, scope, checkpoint)
        ),
    )

    assert value is None


def test_scheduler_dependencies_keep_the_three_constructions_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _fixture_experiment(tmp_path)
    runtime = replace(
        original.runtime,
        database_path=tmp_path / "jobs.sqlite3",
        log_directory=tmp_path / "logs",
    )
    experiment = replace(original, runtime=runtime)
    store = SimpleJobStore(runtime.database_path, experiment_digest=experiment.scheduler_digest)
    monkeypatch.setattr(
        assumption_scheduler,
        "_complete",
        lambda *_args, **_kwargs: pytest.fail("new queue submission scanned remote artifacts"),
    )
    monkeypatch.setattr(
        assumption_scheduler,
        "_spearman_complete",
        lambda *_args, **_kwargs: pytest.fail("new queue submission scanned Spearman artifacts"),
    )

    counts = submit_plan(experiment, store, scan_existing=False)

    assert sum(counts.values()) == 837
    jobs = {job.job_id: job for job in store.jobs()}
    family_job_id = spearman_job_id(experiment)
    assert family_job_id in jobs
    assert family_job_id.startswith("spearman:p196--")
    assert "spearman:p196" not in jobs
    for task in experiment.selection_tasks():
        expected = (family_job_id,) if task.distance_model == "spearman" else ()
        assert jobs[f"selection:{task.task_id}"].dependencies == expected
    clean_ind = next(
        task
        for task in experiment.rank_tasks()
        if task.setting == "ind" and task.condition.kind == "clean"
    )
    clean_matched = next(
        task
        for task in experiment.rank_tasks()
        if task.setting == "matched-naive" and task.condition.kind == "clean"
    )
    clean_noise = next(
        task
        for task in experiment.rank_tasks()
        if task.setting == "oracle-noise" and task.condition.kind == "clean"
    )
    assert len(jobs[f"rank:{clean_ind.task_id}"].dependencies) == 11
    assert len(jobs[f"rank:{clean_matched.task_id}"].dependencies) == 1
    assert jobs[f"rank:{clean_noise.task_id}"].dependencies == (
        f"selection:{clean_noise.selection_task_id}",
    )

    perturbed = next(
        task
        for task in experiment.evaluation_tasks()
        if task.setting == "ind" and task.condition.kind != "clean"
    )
    dependencies = jobs[f"evaluation:{perturbed.task_id}"].dependencies
    assert len(dependencies) == 2
    assert any(value.startswith("evaluation:") for value in dependencies)
    assert any(value.startswith("rank:") for value in dependencies)

    existing = next(iter(jobs.values()))
    added = replace(existing, job_id="test:new-batch-job", status="pending")
    contradictory = replace(existing, command=("contradictory",))
    with pytest.raises(ValueError, match="contradicts plan"):
        store.submit_many((added, contradictory))
    assert "test:new-batch-job" not in {job.job_id for job in store.jobs()}


def test_scheduler_validates_compact_source_artifacts_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _local_experiment(tmp_path)
    task = experiment.source_phase1_tasks()[0]
    calls = []
    monkeypatch.setattr(
        assumption_phase1,
        "source_scope_complete",
        lambda observed_experiment, observed_task: (
            calls.append((observed_experiment, observed_task)) or True
        ),
    )

    assert assumption_scheduler._complete(experiment, "source-phase1", task)
    assert calls == [(experiment, task)]


def test_gpu_distance_kernels_match_known_permutations() -> None:
    ballots = np.asarray(
        [
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [0, 2, 1, 3],
            ]
        ],
        dtype=np.int64,
    )
    consensus = np.asarray([[0, 1, 2, 3]], dtype=np.int64)

    spearman = _distances(ballots, consensus, distance="spearman", device=torch.device("cpu"))
    kendall = _distances(ballots, consensus, distance="kendall", device=torch.device("cpu"))

    assert spearman.tolist() == [0, 20, 2]
    assert kendall.tolist() == [0, 6, 1]


def test_reference_cache_identity_accepts_legacy_and_current_manifests() -> None:
    task = SimpleNamespace(task_id="reference-task", digest="t" * 64)
    variant = SimpleNamespace(digest="v" * 64)
    source_digest = "a" * 64
    shard = {
        "shard_index": 0,
        "start": 0,
        "stop": 1,
        "source_identity_digest": source_digest,
        "payload": {"sha256": "b" * 64},
    }
    legacy = {
        "sample_count": 1,
        "source_identity_digest": source_digest,
        "shards": [shard],
    }
    current = {**legacy, "source_identity": {"digest": source_digest}}

    assert (
        _reference_cache_identity(task, variant, legacy)["source_identity_digest"] == source_digest
    )
    assert (
        _reference_cache_identity(task, variant, current)["source_identity_digest"] == source_digest
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest.update(source_identity={"digest": "c" * 64}),
        lambda manifest: manifest["shards"][0].update(source_identity_digest="c" * 64),
    ],
)
def test_reference_cache_identity_rejects_contradictory_manifest(
    change,
) -> None:
    task = SimpleNamespace(task_id="reference-task", digest="t" * 64)
    variant = SimpleNamespace(digest="v" * 64)
    manifest = {
        "sample_count": 1,
        "source_identity_digest": "a" * 64,
        "shards": [
            {
                "shard_index": 0,
                "start": 0,
                "stop": 1,
                "source_identity_digest": "a" * 64,
                "payload": {"sha256": "b" * 64},
            }
        ],
    }
    change(manifest)

    with pytest.raises(ArtifactError, match="source identity"):
        _reference_cache_identity(task, variant, manifest)


def test_readiness_combines_queue_success_with_manifest_verification() -> None:
    tasks = tuple(SimpleNamespace(task_id=f"task-{index}") for index in range(4))

    def probe(task: SimpleNamespace) -> bool:
        if task.task_id == "task-3":
            raise ValueError("contradictory manifest")
        return task.task_id == "task-1"

    result = _artifact_group(
        tasks,
        kind="phase1",
        statuses={"phase1:task-0": "succeeded", "phase1:task-2": "pending"},
        queue_authoritative=False,
        probe=probe,
    )

    assert result["ready"] is False
    assert result["required"] == 4
    assert result["complete"] == 2
    assert result["missing"] == ["task-2"]
    assert result["invalid"] == [{"task_id": "task-3", "error": "contradictory manifest"}]
    assert result["evidence"] == {"queue_succeeded": 1, "verified_manifest": 1}

    authoritative = _artifact_group(
        (tasks[1],),
        kind="phase1",
        statuses={},
        queue_authoritative=True,
        probe=lambda _task: pytest.fail("authoritative queue should prevent a remote scan"),
    )
    assert authoritative["missing"] == ["task-1"]


def test_oracle_noise_best_individual_comes_from_complete_naive_collection() -> None:
    common = {
        "cell": "dataset--model",
        "dataset": "dataset",
        "model": "model",
        "source_id": "reference-full",
        "distance_model": None,
        "condition": "clean",
        "value_kind": "quality",
        "metric": "F",
    }
    naive = [
        {**common, "setting": "naive", "rule": "single__A", "value": 0.7},
        {**common, "setting": "naive", "rule": "single__B", "value": 0.8},
    ]
    selected_noise = [
        {
            **common,
            "setting": "oracle-noise",
            "distance_model": "spearman",
            "rule": "single__A",
            "value": 0.99,
        }
    ]

    best = _best_individual_rows(naive)
    noise_best = _oracle_noise_best_individual_rows(best)

    assert _best_individual_rows(selected_noise)[0]["selected_individual"] == "A"
    assert {row["distance_model"] for row in noise_best} == {"spearman", "kendall"}
    assert {row["selected_individual"] for row in noise_best} == {"B"}
    assert {row["value"] for row in noise_best} == {0.8}
    assert {row["best_individual_collection"] for row in noise_best} == {"original-complete-naive"}


def test_best_individual_is_fixed_by_clean_metric_for_robustness() -> None:
    common = {
        "cell": "dataset--model",
        "dataset": "dataset",
        "model": "model",
        "setting": "matched-naive",
        "source_id": "mean-over-sources",
        "distance_model": None,
        "metric": "F",
    }
    rows = [
        {
            **common,
            "condition": "clean",
            "value_kind": "quality",
            "rule": "single__A",
            "value": 0.9,
        },
        {
            **common,
            "condition": "clean",
            "value_kind": "quality",
            "rule": "single__B",
            "value": 0.8,
        },
        {
            **common,
            "condition": "gaussian-0.15",
            "value_kind": "robustness",
            "rule": "single__A",
            "value": 0.3,
        },
        {
            **common,
            "condition": "gaussian-0.15",
            "value_kind": "robustness",
            "rule": "single__B",
            "value": 0.01,
        },
    ]

    best = _best_individual_rows(rows)

    assert len(best) == 2
    assert {row["selected_individual"] for row in best} == {"A"}
    assert {(row["condition"], row["value_kind"], row["value"]) for row in best} == {
        ("clean", "quality", 0.9),
        ("gaussian-0.15", "robustness", 0.3),
    }


def test_priority_summary_uses_lowercase_artifact_rule_keys() -> None:
    expected = {*AGGREGATE_RULE_KEYS, "single__A"}
    metrics = {rule: {metric: 0.0 for metric in ("F", "Fbar", "C", "Cbar")} for rule in expected}
    manifests = {
        "clean": {"sample_count": 2, "metrics": metrics},
        "condition-g": {
            "sample_count": 2,
            "metrics": metrics,
            "robustness": {rule: {} for rule in expected},
        },
    }

    _require_aligned_group(
        manifests,
        clean_condition="clean",
        noise_conditions={"g": "condition-g"},
        expected_rules=expected,
    )


def test_signed_robustness_follows_each_quality_direction() -> None:
    assert _signed_robustness("F", clean=0.4, perturbed=0.5) == pytest.approx(0.1)
    assert _signed_robustness("C", clean=0.6, perturbed=0.4) == pytest.approx(-0.2)
    assert _signed_robustness("Fbar", clean=0.4, perturbed=0.3) == pytest.approx(0.1)
    assert _signed_robustness("Cbar", clean=0.2, perturbed=0.3) == pytest.approx(-0.1)


def test_priority_paper_table_matches_header_order_and_signed_semantics() -> None:
    quality = {"F": 1.0, "Fbar": 2.0, "C": 3.0, "Cbar": 4.0}
    robustness = {
        noise: {
            metric: float(5 + noise_index * 4 + metric_index)
            for metric_index, metric in enumerate(("F", "C", "Fbar", "Cbar"))
        }
        for noise_index, noise in enumerate(NOISE_ORDER)
    }

    def row(method: str, setting: str) -> dict[str, object]:
        return {
            "method": method,
            "setting": setting,
            "quality": quality,
            "robustness": robustness,
            "q": 11,
            "source_count": 3 if setting != "ind" else 11,
        }

    best = row("Best Individual", "best-individual")
    rows = [best]
    for method in ("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"):
        rows.extend((row(method, "matched-naive"), row(method, "ind")))
    summary = {
        "cells": [
            {
                "cell": "imagenet100--imagenet100-resnet18",
                "dataset": "imagenet100",
                "model": "imagenet100-resnet18",
                "rows": rows,
            }
        ]
    }

    assert PAPER_TABLE_COLUMN_ORDER == (
        "F",
        "Fbar",
        "C",
        "Cbar",
        "R_F_g",
        "R_C_g",
        "R_F_p",
        "R_C_p",
        "R_F_s",
        "R_C_s",
        "R_F_a",
        "R_C_a",
        "R_Fbar_g",
        "R_Cbar_g",
        "R_Fbar_p",
        "R_Cbar_p",
        "R_Fbar_s",
        "R_Cbar_s",
        "R_Fbar_a",
        "R_Cbar_a",
    )
    expected_values = (1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 17, 18, 7, 8, 11, 12, 15, 16, 19, 20)
    assert _tex_values(best) == " & ".join(f"{value:.3f}" for value in expected_values)

    rendered = _full_tex_text(summary)
    assert r"$\epsilon=2/255$" in rendered
    assert r"\mathtt{R}_{\mathtt{F}}^{\mathtt{g}}\uparrow" in rendered
    assert r"\mathtt{R}_{\mathtt{F}}^{\mathtt{g}}\downarrow" not in rendered
    assert rendered.index("Kemeny--Young") < rendered.index(r"\multirow{2}{*}{RRF}")
    assert "matched NAIVE" in rendered
    assert "& IND &" in rendered


def test_priority_summary_averages_signed_robustness_across_matched_sources() -> None:
    methods = ("A", "B")
    clean_condition = "clean"
    noise_conditions = {noise: f"condition-{noise}" for noise in NOISE_ORDER}
    source_groups = []
    for source_index, robustness_value in enumerate((0.01, 0.02, 0.03)):
        clean_metrics = {}
        for method in methods:
            clean_metrics[f"single__{method}"] = {
                "F": 0.9 + source_index * 0.01 if method == "A" else 0.8,
                "Fbar": 0.1 + source_index * 0.01 if method == "A" else 0.2,
                "C": 0.85 + source_index * 0.01 if method == "A" else 0.75,
                "Cbar": 0.15 + source_index * 0.01 if method == "A" else 0.25,
            }
        group = {clean_condition: {"metrics": clean_metrics}}
        for condition_id in noise_conditions.values():
            perturbed_metrics = {}
            for method in methods:
                clean = clean_metrics[f"single__{method}"]
                delta = robustness_value if method == "A" else -0.001
                perturbed_metrics[f"single__{method}"] = {
                    "F": clean["F"] + delta,
                    "Fbar": clean["Fbar"] - delta,
                    "C": clean["C"] + delta,
                    "Cbar": clean["Cbar"] - delta,
                }
            group[condition_id] = {
                "metrics": perturbed_metrics,
                "robustness": {
                    "single__A": {
                        "absolute": {metric: 0.999 for metric in ("F", "Fbar", "C", "Cbar")}
                    },
                    "single__B": {
                        "absolute": {metric: 0.999 for metric in ("F", "Fbar", "C", "Cbar")}
                    },
                },
            }
        source_groups.append(group)

    statistics = _single_method_statistics(
        source_groups,
        methods,
        clean_condition=clean_condition,
        noise_conditions=noise_conditions,
    )
    row = _best_individual_row(statistics, methods)

    assert set(row["selected_individual"].values()) == {"A"}
    for noise in NOISE_ORDER:
        for metric in ("F", "Fbar", "C", "Cbar"):
            assert row["robustness"][noise][metric] == pytest.approx(0.02)
            assert row["standard_deviation"]["robustness"][noise][metric] == pytest.approx(0.01)
            assert row["perturbed_quality"][noise][metric] == pytest.approx(
                sum(
                    group[noise_conditions[noise]]["metrics"]["single__A"][metric]
                    for group in source_groups
                )
                / 3.0
            )


def test_noise_summary_analysis_compares_paired_naive_rows() -> None:
    quality = {"F": 0.8, "Fbar": 0.2, "C": 0.7, "Cbar": 0.3}
    robustness = {noise: {metric: 0.2 for metric in quality} for noise in ("g", "p", "s", "a")}
    rows = [
        {
            "method": "best_individual",
            "setting": "best-individual",
            "quality": quality,
            "robustness": robustness,
        }
    ]
    for rule in ("simpleavg", "borda", "kemeny", "rrf", "schulze"):
        rows.extend(
            (
                {
                    "method": rule,
                    "setting": "naive",
                    "quality": quality,
                    "robustness": robustness,
                },
                {
                    "method": rule,
                    "setting": "oracle-noise",
                    "quality": {**quality, "F": quality["F"] + 0.1},
                    "robustness": {
                        noise: {metric: 0.1 for metric in quality} for noise in ("g", "p", "s", "a")
                    },
                },
            )
        )
    cells = [{"distance_model": distance, "rows": rows} for distance in ("spearman", "kendall")]

    analysis = _noise_analysis(cells)

    assert analysis["overall"]["comparison_cells"] == 40
    for rule in ("simpleavg", "borda", "kemeny", "rrf", "schulze"):
        comparison = analysis["overall"]["oracle_noise_versus_same_rule_naive"][rule]
        assert comparison == {"better": 2, "equal": 6, "worse": 32}
    assert analysis["by_distance_model"]["spearman"]["comparison_cells"] == 20


def test_noise_summary_accepts_only_task_identical_predecessor_identity() -> None:
    current_digest = "c" * 64
    predecessor_digest = "d" * 64
    experiment = SimpleNamespace(
        assumption_id="paper-assumptions-compact-v1",
        digest=current_digest,
        split="test",
        patch_size=16,
        k=20,
    )
    task = SimpleNamespace(
        task_id="evaluation-task",
        digest="e" * 64,
        cell=SimpleNamespace(
            cell_id="dataset--model",
            dataset=SimpleNamespace(dataset_id="dataset"),
            reference_model=SimpleNamespace(model_id="model"),
        ),
        distance_model="kendall",
        condition=SimpleNamespace(condition_id="clean"),
    )
    manifest = {
        "status": "complete",
        "assumption_id": "paper-assumptions-v1",
        "assumption_digest": predecessor_digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell.cell_id,
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "setting": "oracle-noise",
        "source_id": None,
        "distance_model": task.distance_model,
        "condition": task.condition.condition_id,
        "split": experiment.split,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "fill": "dataset_mean",
        "target_policy": "full_reference_clean_fp32_prediction",
        "sample_count": 10,
        "metrics": {
            rule: {metric: 0.0 for metric in ("F", "Fbar", "C", "Cbar")}
            for rule in ("simpleavg", "borda", "kemeny", "rrf", "schulze")
        },
    }
    allowed = {
        (experiment.assumption_id, experiment.digest),
        (manifest["assumption_id"], manifest["assumption_digest"]),
    }

    assert _validate_noise_manifest(
        experiment, task, manifest, allowed_assumption_identities=allowed
    ) == ("paper-assumptions-v1", predecessor_digest)

    unrelated = {**manifest, "assumption_digest": "f" * 64}
    with pytest.raises(ArtifactError, match="unrelated assumption identity"):
        _validate_noise_manifest(experiment, task, unrelated, allowed_assumption_identities=allowed)


def test_source_phase1_generates_only_the_formal_patch_variant(tmp_path: Path) -> None:
    experiment = _fixture_experiment(tmp_path)
    scope = experiment.source_phase1_tasks()[0]
    tasks = experiment.method_phase1_tasks(scope, checkpoint_path=tmp_path / "inference.pt")
    variants = {task.family: tuple(item.artifact_name for item in task.variants) for task in tasks}

    assert variants["FeatureAblation"] == ("FeatureAblation__p16",)
    assert variants["Occlusion"] == ("Occlusion__p16",)
    assert set(variants) == set(scope.cell.methods)


def test_compact_source_publisher_discards_full_attribution(tmp_path: Path) -> None:
    experiment = _local_experiment(tmp_path)
    scope = experiment.source_phase1_tasks()[0]
    generated = _generation_experiment(experiment, scope, tmp_path / "inference.pt")
    task = next(
        item
        for item in experiment.method_phase1_tasks(scope, checkpoint_path=tmp_path / "inference.pt")
        if item.family == "Saliency"
    )
    variant = task.variants[0]
    store = ArtifactStore(generated)
    publisher = _Phase1Publisher(
        generated,
        task,
        store,
        compact_patch_size=16,
        artifact_schema_version=SOURCE_RANK_INPUT_SCHEMA_VERSION,
    )
    generator = torch.Generator(device="cpu").manual_seed(71)
    logits = torch.randn((2, task.model.num_classes), generator=generator)
    attribution = torch.randn((2, 3, 224, 224), generator=generator)
    try:
        record = publisher.submit_shard(
            variant,
            root="source-rank-inputs/fixture",
            shard_index=0,
            start=0,
            stop=2,
            attributions=attribution,
            labels=torch.tensor([0, 1]),
            indices=torch.tensor([10, 11]),
            predictions=logits.argmax(dim=1),
            logits=logits,
            targets=torch.tensor([0, 1]),
            profile_id="fixture-profile",
            batch_size=2,
            source_identity_digest="a" * 64,
        ).result(timeout=10)
    finally:
        publisher.shutdown()

    local = tmp_path / "compact.safetensors"
    payload = record["payload"]
    store.materialize(
        str(payload["relative_path"]),
        local,
        expected_sha256=str(payload["sha256"]),
    )
    fields = load_safetensors(local)
    assert set(fields) == {
        "indices",
        "labels",
        "predictions",
        "logits",
        "targets",
        rank_field(16),
        simpleavg_score_field(16),
    }
    assert "attributions" not in fields
    assert "simpleavg_spatial" not in fields
    assert record["artifact_representation"] == SOURCE_RANK_INPUT_REPRESENTATION
    assert local.stat().st_size < attribution.numel() * attribution.element_size() // 50


def test_compact_simpleavg_patch_scores_preserve_formal_rank() -> None:
    generator = np.random.default_rng(83)
    spatial_sum = None
    patch_sum = None
    for _ in range(11):
        attribution = generator.standard_normal((3, 3, 224, 224), dtype=np.float32)
        normalized = normalize_spatial(
            np.mean(np.abs(attribution), axis=1, dtype=np.float32),
            "minmax",
        )
        patch = normalized.reshape(3, 14, 16, 14, 16).mean(axis=(2, 4), dtype=np.float32)
        spatial_sum = normalized.copy() if spatial_sum is None else spatial_sum + normalized
        patch_sum = patch.copy() if patch_sum is None else patch_sum + patch

    old_scores = (spatial_sum / 11.0).reshape(3, 14, 16, 14, 16).mean(axis=(2, 4), dtype=np.float32)
    compact_scores = patch_sum / 11.0
    np.testing.assert_allclose(compact_scores, old_scores, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(
        scores_to_ranks(compact_scores),
        scores_to_ranks(old_scores),
    )


def test_source_phase1_manifest_binds_full_reference_targets_and_outputs(
    tmp_path: Path,
) -> None:
    experiment = _fixture_experiment(tmp_path)
    scope = experiment.source_phase1_tasks()[0]
    task = next(
        item
        for item in experiment.method_phase1_tasks(scope, checkpoint_path=tmp_path / "inference.pt")
        if item.family == "Saliency"
    )
    variant = task.variants[0]
    source_identity = {
        "dataset_manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "mean_artifact_digest": "c" * 64,
        "target_policy": FULL_REFERENCE_CLEAN_TARGET_POLICY,
        "model_output_source": FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
        "digest": "d" * 64,
    }

    manifest = _manifest_value(
        experiment.base,
        task,
        variant,
        records=[{"start": 0, "stop": 2}],
        input_shape=(3, 224, 224),
        checkpoint_sha256=source_identity["checkpoint_sha256"],
        mean_digest=source_identity["mean_artifact_digest"],
        source_identity=source_identity,
        dataset_manifest_sha256=source_identity["dataset_manifest_sha256"],
        profile_id="profile-test",
        batch_size=2,
        attribution_provider={},
        target_policy=FULL_REFERENCE_CLEAN_TARGET_POLICY,
        model_output_source=FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
    )

    assert manifest["target_policy"] == FULL_REFERENCE_CLEAN_TARGET_POLICY
    assert manifest["model_output_source"] == FULL_REFERENCE_MODEL_OUTPUT_SOURCE
    assert "full-reference model" in manifest["safetensors_fields"]["predictions"]
    assert "full-reference model" in manifest["safetensors_fields"]["logits"]
    validation_kwargs = {
        "variant": variant,
        "source_identity": source_identity,
        "target_policy": FULL_REFERENCE_CLEAN_TARGET_POLICY,
        "model_output_source": FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
    }
    _validate_completed_source(manifest, **validation_kwargs)

    wrong_target = dict(manifest)
    wrong_target["target_policy"] = CLEAN_MODEL_TARGET_POLICY
    with pytest.raises(ArtifactError, match="target_policy"):
        _validate_completed_source(wrong_target, **validation_kwargs)

    wrong_outputs = dict(manifest)
    wrong_outputs["model_output_source"] = TASK_MODEL_OUTPUT_SOURCE
    with pytest.raises(ArtifactError, match="model_output_source"):
        _validate_completed_source(wrong_outputs, **validation_kwargs)

    wrong_identity = dict(manifest)
    wrong_identity["source_identity"] = {
        **source_identity,
        "target_policy": CLEAN_MODEL_TARGET_POLICY,
    }
    with pytest.raises(ArtifactError, match="source_identity.target_policy"):
        _validate_completed_source(wrong_identity, **validation_kwargs)


def test_assumption_attribution_materialization_uses_tmpfs_spool(tmp_path: Path) -> None:
    experiment = _local_experiment(tmp_path)
    destinations = []

    class Store:
        def materialize(
            self, _relative_path: str, destination: Path, *, expected_sha256: str
        ) -> None:
            from safetensors.torch import save_file

            assert expected_sha256 == "payload-digest"
            destinations.append(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_file({"attributions": torch.ones(1, 1, 2, 2)}, str(destination))

    source = AttributionSource(
        label="Saliency",
        family="Saliency",
        experiment=experiment.base,
        store=Store(),  # type: ignore[arg-type]
        root="phase1/test",
        manifest={
            "shards": [
                {
                    "shard_index": 0,
                    "payload": {
                        "relative_path": "phase1/test/shard.safetensors",
                        "sha256": "payload-digest",
                    },
                }
            ]
        },
        source_id="reference-full",
    )

    fields = _load_attribution_shard(experiment, source, shard_index=0)

    assert torch.equal(fields["attributions"], torch.ones(1, 1, 2, 2))
    assert len(destinations) == 1
    assert destinations[0].is_relative_to(experiment.storage.spool_root)
    assert not destinations[0].exists()


def test_task_spool_paths_are_process_and_task_isolated(tmp_path: Path) -> None:
    experiment = _local_experiment(tmp_path)
    relative_path = "ranks/cell/clean/shards/shard-00000.safetensors"

    first = task_spool_path(
        experiment,
        namespace="rank-cache",
        task_digest="selection-spearman",
        relative_path=relative_path,
    )
    second = task_spool_path(
        experiment,
        namespace="rank-cache",
        task_digest="selection-kendall",
        relative_path=relative_path,
    )

    assert first != second
    assert first.name == second.name == "shard-00000.safetensors"
    assert first.parent == (
        experiment.storage.spool_root / "rank-cache" / str(os.getpid()) / "selection-spearman"
    )
    assert second.parent.name == "selection-kendall"


def _local_experiment(tmp_path: Path):
    original = _fixture_experiment(tmp_path)
    storage = replace(
        original.storage,
        remote_root=str(tmp_path / "remote"),
        scratch_root=tmp_path / "scratch",
        spool_root=tmp_path / "spool",
    )
    runtime = replace(
        original.runtime,
        database_path=tmp_path / "jobs.sqlite3",
        log_directory=tmp_path / "logs",
    )
    selection = replace(
        original.selection,
        spearman_artifact=tmp_path / "statistics" / "spearman-p196.npz",
    )
    return replace(original, storage=storage, runtime=runtime, selection=selection)


def test_spearman_artifact_is_published_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _local_experiment(tmp_path)
    path = spearman_local_artifact_path(experiment)
    metadata_path = path.with_suffix(path.suffix + ".json")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"deterministic-spearman-family")
    metadata = {
        "schema_version": 1,
        "identity": spearman_identity(experiment),
        "identity_digest": "test-identity",
        "pilot_digest": "test-pilot",
        "sha256": file_sha256(path),
        "outcome": {},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        assumption_prepare,
        "locked_spearman_family_factory",
        lambda **_kwargs: object(),
    )

    assert prepare_spearman_family(experiment)["pilot_digest"] == "test-pilot"
    assert completed_spearman_family(experiment, restore=False) is not None

    path.unlink()
    metadata_path.unlink()
    assert completed_spearman_family(experiment, restore=True) is not None
    assert path.read_bytes() == b"deterministic-spearman-family"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["pilot_digest"] == "test-pilot"


def test_rank_and_evaluation_artifacts_round_trip(tmp_path: Path) -> None:
    experiment = _local_experiment(tmp_path)
    rank_task = experiment.rank_tasks()[0]
    evaluation_task = next(
        task for task in experiment.evaluation_tasks() if task.rank_task_id == rank_task.task_id
    )
    store = output_store(experiment)
    tensors = {
        "indices": torch.tensor([3, 7]),
        "labels": torch.tensor([1, 0]),
        "targets": torch.tensor([1, 1]),
        "unmasked_predictions": torch.tensor([1, 0]),
        "rank__r000": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32),
    }
    rank_record = publish_shard(
        experiment,
        store,
        root=rank_task.artifact_root,
        task_id=rank_task.task_id,
        task_digest=rank_task.digest,
        schema_version=RANK_SCHEMA_VERSION,
        shard_index=0,
        start=0,
        stop=2,
        tensors=tensors,
        metadata={"rank_base": "0"},
        record_fields={"rule_labels": {"r000": "Borda"}, "aggregation_statistics": {}},
    )
    rank_manifest = {
        "schema_version": RANK_SCHEMA_VERSION,
        "status": "complete",
        "task_id": rank_task.task_id,
        "task_digest": rank_task.digest,
        "sample_count": 2,
        "shards": [rank_record],
    }
    publish_manifest(
        experiment,
        store,
        root=rank_task.artifact_root,
        task_id=rank_task.task_id,
        manifest=rank_manifest,
    )
    assert completed_rank_manifest(experiment, rank_task, store=store) == rank_manifest
    assert set(
        existing_shard_records(
            store,
            rank_task.artifact_root,
            task_digest=rank_task.digest,
            schema_version=RANK_SCHEMA_VERSION,
            source_layout=((0, 0, 2),),
        )
    ) == {0}
    local_rank = tmp_path / "loaded-rank.safetensors"
    store.materialize(
        str(rank_record["payload"]["relative_path"]),
        local_rank,
        expected_sha256=str(rank_record["payload"]["sha256"]),
    )
    assert torch.equal(load_safetensors(local_rank)["rank__r000"], tensors["rank__r000"])

    evaluation_record = publish_shard(
        experiment,
        store,
        root=evaluation_task.artifact_root,
        task_id=evaluation_task.task_id,
        task_digest=evaluation_task.digest,
        schema_version=EVALUATION_SCHEMA_VERSION,
        shard_index=0,
        start=0,
        stop=2,
        tensors={
            "indices": tensors["indices"],
            "removed_predictions__r000": torch.tensor([0, 0]),
            "retained_predictions__r000": torch.tensor([1, 1]),
        },
        metadata={"patch_size": "16", "k": "20"},
        record_fields={
            "rule_labels": {"r000": "Borda"},
            "metric_sums": {"r000": {"F": 1.0, "Fbar": 0.0, "C": 1.0, "Cbar": 0.0}},
        },
    )
    evaluation_manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "task_id": evaluation_task.task_id,
        "task_digest": evaluation_task.digest,
        "sample_count": 2,
        "shards": [evaluation_record],
    }
    publish_manifest(
        experiment,
        store,
        root=evaluation_task.artifact_root,
        task_id=evaluation_task.task_id,
        manifest=evaluation_manifest,
    )
    assert (
        completed_evaluation_manifest(experiment, evaluation_task, store=store)
        == evaluation_manifest
    )
