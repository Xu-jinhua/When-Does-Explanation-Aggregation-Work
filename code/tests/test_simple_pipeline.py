from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import xai_ensemble.phase1.relprop as relprop
from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.phase1.relprop import relprop_attribution_provider
from xai_ensemble.simple.artifacts import (
    ArtifactError,
    load_safetensors,
    phase2_artifact_root,
    write_phase1_shard,
)
from xai_ensemble.simple.conditions import natural_corruption
from xai_ensemble.simple.config import load_experiment
from xai_ensemble.simple.methods import PAPER_CNN_METHODS, PAPER_VIT_METHODS
from xai_ensemble.simple.phase1 import (
    CLEAN_MODEL_TARGET_POLICY,
    TASK_MODEL_OUTPUT_SOURCE,
    _validate_completed_source,
)
from xai_ensemble.simple.phase2 import (
    _scores_to_ranks,
    _summarize_aggregation_statistics,
    attribution_to_patch_scores,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_example_expands_only_the_paper_rosters() -> None:
    experiment = load_experiment(CODE_ROOT / "configs/simple/example.yaml")

    assert experiment.precision == "fp32"
    assert experiment.storage.spool_root == Path("/dev/shm/xai-simple/paper-main-v1")
    assert experiment.storage.spool_max_bytes == 64 * 2**30
    assert experiment.storage.spool_min_free_bytes == 32 * 2**30
    assert len(experiment.profiles()) == 30
    assert len(experiment.phase2_profiles()) == 2
    assert len(experiment.phase1_tasks()) == 22
    assert len(experiment.phase2_tasks()) == 2

    cnn = [task for task in experiment.phase1_tasks() if task.model.architecture == "cnn"]
    vit = [task for task in experiment.phase1_tasks() if task.model.architecture == "vit"]
    assert tuple(task.family for task in cnn) == PAPER_CNN_METHODS
    assert tuple(task.family for task in vit) == PAPER_VIT_METHODS
    assert len(PAPER_CNN_METHODS) == 11
    assert len(PAPER_VIT_METHODS) == 11
    assert {
        "FeatureAblation",
        "CheferTransformerAttribution",
        "PartialLRP",
        "FullLRP",
        "GradientAttentionRollout",
        "AttentionGradCAM",
    } <= {task.family for task in vit}
    assert {
        "GuidedBackprop",
        "DeepLift",
        "DeepLiftShap",
        "Rollout",
        "AttnLast",
    }.isdisjoint(task.family for task in vit)

    patch_tasks = {
        task.family: tuple(variant.params["patch_size"] for variant in task.variants)
        for task in cnn
        if task.family in {"FeatureAblation", "Occlusion"}
    }
    assert patch_tasks == {
        "FeatureAblation": (8, 14, 16),
        "Occlusion": (8, 14, 16),
    }
    vit_patch_tasks = {
        task.family: tuple(variant.params["patch_size"] for variant in task.variants)
        for task in vit
        if task.family in {"FeatureAblation", "Occlusion"}
    }
    assert vit_patch_tasks == patch_tasks
    relprop_profiles = [
        profile
        for profile in experiment.profiles()
        if profile.architecture == "vit"
        and profile.method.family in {"CheferTransformerAttribution", "PartialLRP", "FullLRP"}
    ]
    assert len(relprop_profiles) == 3
    assert {profile.model_id for profile in relprop_profiles} == {"imagenet100-vit-b16"}
    assert experiment.phase2.primary_patch_size == 16
    assert experiment.phase2.inference_batch_size == 512
    assert experiment.phase2.kemeny_starts == 1
    assert experiment.phase2.kemeny_max_passes == 1_024
    assert {task.patch_size for task in experiment.phase2_tasks()} == {16}
    assert len({model.mean_path for model in experiment.models}) == 2


def test_spool_execution_overrides_do_not_change_experiment_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CODE_ROOT / "configs/simple/example.yaml"
    original = load_experiment(source)
    monkeypatch.setenv("XAI_SIMPLE_SPOOL_MAX_GIB", "24")
    monkeypatch.setenv("XAI_SIMPLE_SPOOL_MIN_FREE_GIB", "40")
    overridden = load_experiment(source)

    assert overridden.storage.spool_max_bytes == 24 * 2**30
    assert overridden.storage.spool_min_free_bytes == 40 * 2**30
    assert overridden.digest == original.digest
    assert overridden.scheduler_digest == original.scheduler_digest


def test_phase1_execution_controls_do_not_change_scheduler_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CODE_ROOT / "configs/simple/example.yaml"
    original = load_experiment(source)
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_UPLOAD_WORKERS", "3")
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_UPLOAD_GLOBAL_LIMIT", "6")
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_STAGE_WORKERS", "2")
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_PREFETCH_WORKERS", "4")
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_PREFETCH_MAX_GIB", "72")
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_PREFETCH_MIN_FREE_GIB", "44")
    monkeypatch.setenv("XAI_SIMPLE_PHASE1_TELEMETRY_INTERVAL_SECONDS", "0.25")
    overridden = load_experiment(source)

    assert overridden.runtime.phase1_upload_workers == 3
    assert overridden.runtime.phase1_upload_global_limit == 6
    assert overridden.runtime.phase1_stage_workers == 2
    assert overridden.runtime.phase1_prefetch_workers == 4
    assert overridden.runtime.phase1_prefetch_max_gib == pytest.approx(72)
    assert overridden.runtime.phase1_prefetch_min_free_gib == pytest.approx(44)
    assert overridden.runtime.phase1_telemetry_interval_seconds == pytest.approx(0.25)
    assert overridden.digest == original.digest
    assert overridden.scheduler_digest == original.scheduler_digest


def test_relprop_provider_rekeys_only_relprop_tasks_and_vit_phase2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CODE_ROOT / "configs/simple/example.yaml"
    original = load_experiment(source)
    original_phase1 = {
        (task.model.model_id, task.condition.condition_id, task.family): task.task_id
        for task in original.phase1_tasks()
    }
    original_phase2 = {
        (task.model.model_id, task.condition.condition_id, task.patch_size): task.task_id
        for task in original.phase2_tasks()
    }
    original_profiles = {
        (
            profile.model_key,
            profile.model_id,
            profile.method.family,
            profile.method.variant,
        ): profile.profile_id
        for profile in original.profiles()
    }

    monkeypatch.setattr(relprop, "RELPROP_REVISION", "0" * 40)
    monkeypatch.setattr(relprop, "RELPROP_SOURCE_DIGEST", "1" * 64)
    changed = load_experiment(source)

    for task in changed.phase1_tasks():
        key = (task.model.model_id, task.condition.condition_id, task.family)
        if task.model.architecture == "vit" and task.family in {
            "CheferTransformerAttribution",
            "PartialLRP",
            "FullLRP",
        }:
            assert task.task_id != original_phase1[key]
        else:
            assert task.task_id == original_phase1[key]
    for task in changed.phase2_tasks():
        key = (task.model.model_id, task.condition.condition_id, task.patch_size)
        if task.model.architecture == "vit":
            assert task.task_id != original_phase2[key]
        else:
            assert task.task_id == original_phase2[key]
    for profile in changed.profiles():
        key = (
            profile.model_key,
            profile.model_id,
            profile.method.family,
            profile.method.variant,
        )
        if profile.model_id is not None:
            assert profile.profile_id != original_profiles[key]
        else:
            assert profile.profile_id == original_profiles[key]


def test_phase1_manifest_requires_current_relprop_provider_but_allows_legacy_timm() -> None:
    experiment = load_experiment(CODE_ROOT / "configs/simple/example.yaml")
    relprop_task = next(
        task
        for task in experiment.phase1_tasks()
        if task.model.architecture == "vit" and task.family == "CheferTransformerAttribution"
    )
    variant = relprop_task.variants[0]
    source_identity = {
        "dataset_manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "mean_artifact_digest": "c" * 64,
        **{
            f"attribution_provider_{key}": value
            for key, value in relprop_attribution_provider(
                "CheferTransformerAttribution", "vit"
            ).items()
        },
    }
    source_identity["digest"] = object_sha256(source_identity)
    manifest = {
        "method": {"variant_digest": variant.digest},
        "dataset": {"manifest_sha256": source_identity["dataset_manifest_sha256"]},
        "model": {
            "checkpoint_sha256": source_identity["checkpoint_sha256"],
            "attribution_provider": relprop_attribution_provider(
                "CheferTransformerAttribution", "vit"
            ),
        },
        "baseline": {"mean_artifact_digest": source_identity["mean_artifact_digest"]},
        "source_identity_digest": source_identity["digest"],
        "target_policy": CLEAN_MODEL_TARGET_POLICY,
    }
    validation_kwargs = {
        "variant": variant,
        "source_identity": source_identity,
        "target_policy": CLEAN_MODEL_TARGET_POLICY,
        "model_output_source": TASK_MODEL_OUTPUT_SOURCE,
        "allow_legacy_model_output_source": True,
    }
    _validate_completed_source(manifest, **validation_kwargs)

    manifest["target_policy"] = "incorrect_target"
    with pytest.raises(ArtifactError, match="target_policy"):
        _validate_completed_source(manifest, **validation_kwargs)
    manifest["target_policy"] = CLEAN_MODEL_TARGET_POLICY

    manifest["model"]["attribution_provider"]["factory_revision"] = "0" * 40
    with pytest.raises(ArtifactError, match="attribution_provider.factory_revision"):
        _validate_completed_source(manifest, **validation_kwargs)

    saliency_task = next(
        task
        for task in experiment.phase1_tasks()
        if task.model.architecture == "vit" and task.family == "Saliency"
    )
    legacy_source = {
        "dataset_manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "mean_artifact_digest": "c" * 64,
    }
    legacy_source["digest"] = object_sha256(legacy_source)
    legacy_manifest = {
        "method": {"variant_digest": saliency_task.variants[0].digest},
        "dataset": {"manifest_sha256": legacy_source["dataset_manifest_sha256"]},
        "model": {"checkpoint_sha256": legacy_source["checkpoint_sha256"]},
        "baseline": {"mean_artifact_digest": legacy_source["mean_artifact_digest"]},
        "source_identity_digest": legacy_source["digest"],
        "target_policy": CLEAN_MODEL_TARGET_POLICY,
    }
    _validate_completed_source(
        legacy_manifest,
        variant=saliency_task.variants[0],
        source_identity=legacy_source,
        target_policy=CLEAN_MODEL_TARGET_POLICY,
        model_output_source=TASK_MODEL_OUTPUT_SOURCE,
        allow_legacy_model_output_source=True,
    )


def test_patch_scores_follow_paper_absolute_mean_semantics() -> None:
    attributions = np.zeros((1, 2, 4, 4), dtype=np.float32)
    attributions[:, 0, :2, :2] = -2.0
    attributions[:, 1, :2, :2] = 4.0
    attributions[:, :, :2, 2:] = 1.0

    scores = attribution_to_patch_scores(attributions, patch_size=2)

    np.testing.assert_allclose(scores, np.asarray([[[3.0, 1.0], [0.0, 0.0]]]))


def test_score_ranks_are_strict_and_row_major_on_ties() -> None:
    scores = np.asarray([[1.0, 3.0, 3.0, -1.0]], dtype=np.float32)
    ranks = _scores_to_ranks(scores)

    np.testing.assert_array_equal(ranks, np.asarray([[2, 0, 1, 3]]))
    np.testing.assert_array_equal(np.sort(ranks, axis=1), np.arange(4)[None, :])


def test_adding_phase2_patch_size_reuses_phase1_and_existing_phase2(tmp_path: Path) -> None:
    source = CODE_ROOT / "configs/simple/example.yaml"
    original = load_experiment(source)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    config["phase2"]["patch_sizes"] = [8, 16]
    expanded_path = tmp_path / "expanded.yaml"
    expanded_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    expanded = load_experiment(expanded_path)

    assert expanded.digest != original.digest
    assert expanded.phase1_digest == original.phase1_digest
    assert expanded.scheduler_digest == original.scheduler_digest
    assert [task.task_id for task in expanded.phase1_tasks()] == [
        task.task_id for task in original.phase1_tasks()
    ]
    original_p16 = {task.task_id for task in original.phase2_tasks()}
    expanded_p16 = {task.task_id for task in expanded.phase2_tasks() if task.patch_size == 16}
    assert expanded_p16 == original_p16


def test_phase2_kemeny_defaults_match_the_formal_policy(tmp_path: Path) -> None:
    source = CODE_ROOT / "configs/simple/example.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    config["phase2"].pop("kemeny_starts")
    config["phase2"].pop("kemeny_max_passes")
    path = tmp_path / "default-kemeny.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = load_experiment(path)

    assert experiment.phase2.kemeny_starts == 1
    assert experiment.phase2.kemeny_max_passes == 1_024


def test_kemeny_policy_change_rekeys_only_phase2_tasks(tmp_path: Path) -> None:
    source = CODE_ROOT / "configs/simple/example.yaml"
    current = load_experiment(source)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    config["phase2"]["kemeny_starts"] = 16
    config["phase2"]["kemeny_max_passes"] = 10_000
    path = tmp_path / "old-kemeny.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    old_policy = load_experiment(path)

    assert old_policy.phase1_digest == current.phase1_digest
    assert old_policy.scheduler_digest == current.scheduler_digest
    assert [task.task_id for task in old_policy.phase1_tasks()] == [
        task.task_id for task in current.phase1_tasks()
    ]
    assert {task.task_id for task in old_policy.phase2_tasks()}.isdisjoint(
        task.task_id for task in current.phase2_tasks()
    )
    assert {phase2_artifact_root(task) for task in old_policy.phase2_tasks()}.isdisjoint(
        phase2_artifact_root(task) for task in current.phase2_tasks()
    )
    assert all(task.digest in phase2_artifact_root(task) for task in current.phase2_tasks())


def test_phase2_kemeny_statistics_merge_across_resumed_shards() -> None:
    first = {
        "count": 3,
        "aggregation_statistics": {
            "kemeny": {
                "sample_count": 3,
                "search_instance_count": 3,
                "starts_requested": 1,
                "max_passes": 1_024,
                "total_moves": 21,
                "max_moves": 9,
                "cap_hit_count": 0,
                "converged_count": 3,
                "converged_fraction": 1.0,
                "borda_objective_sum": 100,
                "final_objective_sum": 90,
                "objective_improvement_sum": 10,
            }
        },
    }
    second = {
        "count": 2,
        "aggregation_statistics": {
            "kemeny": {
                "sample_count": 2,
                "search_instance_count": 2,
                "starts_requested": 1,
                "max_passes": 1_024,
                "total_moves": 13,
                "max_moves": 11,
                "cap_hit_count": 1,
                "converged_count": 1,
                "converged_fraction": 0.5,
                "borda_objective_sum": 80,
                "final_objective_sum": 75,
                "objective_improvement_sum": 5,
            }
        },
    }

    observed = _summarize_aggregation_statistics(
        (first, second),
        require_kemeny=True,
    )["kemeny"]

    assert observed == {
        "sample_count": 5,
        "search_instance_count": 5,
        "total_moves": 34,
        "cap_hit_count": 1,
        "converged_count": 4,
        "borda_objective_sum": 180,
        "final_objective_sum": 165,
        "objective_improvement_sum": 15,
        "starts_requested": 1,
        "max_passes": 1_024,
        "max_moves": 11,
        "converged_fraction": 0.8,
    }


def test_natural_corruption_is_seeded_and_batch_independent() -> None:
    import torch

    image = torch.full((1, 3, 4, 4), 0.5)
    kwargs = {
        "raw_images": image,
        "labels": torch.tensor([0]),
        "indices": torch.tensor([7]),
        "model": None,
        "normalize": None,
        "seed": 123,
        "kind": "gaussian",
        "severity": 0.15,
    }
    first = natural_corruption(**kwargs)
    second = natural_corruption(**kwargs)

    torch.testing.assert_close(first, second)
    assert bool(((first >= 0.0) & (first <= 1.0)).all())


def test_phase1_shard_stores_fp32_logits_and_matching_predictions(
    tmp_path: Path,
) -> None:
    import torch

    path = tmp_path / "phase1.safetensors"
    predictions = torch.tensor([1, 0])
    write_phase1_shard(
        path,
        indices=torch.tensor([10, 11]),
        labels=torch.tensor([1, 0]),
        predictions=predictions,
        logits=torch.tensor([[0.1, 0.9], [2.0, -1.0]], dtype=torch.float64),
        # Clean explanations intentionally use the same values. The writer
        # must still give the two safetensors fields independent storage.
        targets=predictions,
        attributions=torch.ones((2, 3, 4, 4), dtype=torch.float64),
        metadata={"schema_version": "2"},
    )

    fields = load_safetensors(path)
    assert fields["logits"].dtype == torch.float32
    assert fields["logits"].shape == (2, 2)
    assert fields["attributions"].dtype == torch.float32
    torch.testing.assert_close(fields["predictions"], fields["logits"].argmax(dim=1))
    torch.testing.assert_close(fields["targets"], fields["predictions"])


def test_phase1_shard_rejects_prediction_logit_disagreement(tmp_path: Path) -> None:
    import torch

    with pytest.raises(ValueError, match=r"predictions must equal argmax\(logits\)"):
        write_phase1_shard(
            tmp_path / "invalid.safetensors",
            indices=torch.tensor([0]),
            labels=torch.tensor([0]),
            predictions=torch.tensor([1]),
            logits=torch.tensor([[1.0, 0.0]]),
            targets=torch.tensor([0]),
            attributions=torch.zeros((1, 3, 4, 4)),
            metadata={"schema_version": "2"},
        )
