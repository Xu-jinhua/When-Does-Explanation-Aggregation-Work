from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_phase0_data import _tiny_manifest

from xai_ensemble.data.partitions import make_ind_partitions, make_reference_partition
from xai_ensemble.phase0.checkpoint import (
    CheckpointManager,
    make_checkpoint_metadata,
    validate_resume_metadata,
)
from xai_ensemble.phase0.config import default_training_config
from xai_ensemble.phase0.dataset import DistributedEvalSampler
from xai_ensemble.phase0.models import (
    MODEL_REGISTRY,
    ModelBuildRequest,
    copy_imagenet1k_classifier_subset,
    create_model,
    disable_inplace_activations,
    initialization_for_recipe,
    make_model_training_tasks,
    validate_imagenet1k_class_map,
)
from xai_ensemble.phase0.trainer import cosine_lr_multiplier


def test_model_registry_is_available_without_gpu_dependencies() -> None:
    assert MODEL_REGISTRY["resnet18"].provider == "timm"
    assert MODEL_REGISTRY["vit_base_patch16_224"].provider == "timm"
    assert MODEL_REGISTRY["vit_base_patch16_224"].family == "vit"
    request = ModelBuildRequest("resnet18", num_classes=100, init_mode="random")
    assert request.definition.family == "cnn"

    with pytest.raises(ValueError, match="requires checkpoint_path"):
        ModelBuildRequest("resnet18", num_classes=100, init_mode="checkpoint")
    with pytest.raises(ValueError, match="only valid"):
        ModelBuildRequest(
            "resnet18",
            num_classes=100,
            init_mode="random",
            checkpoint_path="unexpected.pt",
        )
    with pytest.raises(ValueError, match="requires class_index_map"):
        ModelBuildRequest(
            "resnet18", num_classes=100, init_mode="imagenet1k_subset"
        )


def test_initialization_recipe_and_class_map_contract_without_torch() -> None:
    indices = tuple(range(100))
    digest = validate_imagenet1k_class_map(indices, expected_classes=100)
    request = ModelBuildRequest(
        "resnet18",
        num_classes=100,
        init_mode="imagenet1k_subset",
        class_index_map=indices,
        class_map_sha256=digest,
    )
    assert request.class_index_map == indices
    assert initialization_for_recipe("timm_subset_logits") == "imagenet1k_subset"
    assert initialization_for_recipe("timm_finetune") == "imagenet1k"
    assert initialization_for_recipe("random_scratch") == "random"
    with pytest.raises(ValueError, match="digest mismatch"):
        ModelBuildRequest(
            "resnet18",
            num_classes=100,
            init_mode="imagenet1k_subset",
            class_index_map=indices,
            class_map_sha256="0" * 64,
        )


def test_subset_head_copies_exact_selected_logits_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 1000)

        def forward(self, values):
            return self.fc(values)

    torch.manual_seed(7)
    model = TinyClassifier().eval()
    inputs = torch.randn(5, 3)
    indices = (7, 123, 999)
    expected = model(inputs)[:, list(indices)]
    copy_imagenet1k_classifier_subset(model, indices)
    assert model.fc.out_features == len(indices)
    torch.testing.assert_close(model(inputs), expected)


def test_timm_residual_patch_preserves_eval_logits() -> None:
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    torch.manual_seed(19)
    model = timm.create_model("resnet18", pretrained=False, num_classes=3).eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = model(inputs)
    disable_inplace_activations(model)
    with torch.no_grad():
        observed = model(inputs)
    torch.testing.assert_close(observed, expected)


def test_finetune_build_requests_pretrained_weights_and_100_class_head(
    monkeypatch,
) -> None:
    calls = []

    class FakeModel:
        def modules(self):
            return ()

    def create_timm_model(name, *, pretrained, num_classes):
        calls.append((name, pretrained, num_classes))
        return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "timm",
        SimpleNamespace(create_model=create_timm_model),
    )
    model = create_model(
        ModelBuildRequest(
            "resnet18",
            num_classes=100,
            init_mode="imagenet1k",
        )
    )
    assert calls == [("resnet18", True, 100)]
    assert model.initialization == "imagenet1k"


def test_training_tasks_include_one_common_full_data_reference() -> None:
    _, manifest = _tiny_manifest()
    source_plan = make_ind_partitions(manifest, num_sources=3, seed=5)
    reference_plan = make_reference_partition(manifest)
    tasks = make_model_training_tasks(
        model_key="resnet18",
        num_classes=2,
        source_plan=source_plan,
        reference_plan=reference_plan,
        init_mode="random",
    )
    assert len(tasks) == 4
    assert [task.role for task in tasks].count("source") == 3
    reference = [task for task in tasks if task.role == "reference"]
    assert len(reference) == 1
    assert reference[0].source_id == "reference-full"
    assert len({task.model.seed for task in tasks}) == len(tasks)


def test_distributed_evaluation_sampler_never_pads_duplicates() -> None:
    dataset = list(range(7))
    shards = [list(DistributedEvalSampler(dataset, rank=rank, world_size=3)) for rank in range(3)]
    assert shards == [[0, 3, 6], [1, 4], [2, 5]]
    assert sorted(item for shard in shards for item in shard) == list(range(7))


def test_default_recipes_and_cosine_schedule() -> None:
    cnn = default_training_config("cnn", epochs=10)
    vit = default_training_config("vit", epochs=10)
    assert cnn.optimizer.name == "sgd"
    assert cnn.train_loader.batch_size == 64
    assert vit.optimizer.name == "adamw"
    assert vit.warmup_epochs == 5
    assert cosine_lr_multiplier(
        0, total_steps=100, warmup_steps=10, minimum_ratio=0.0
    ) == pytest.approx(0.1)
    assert cosine_lr_multiplier(
        10, total_steps=100, warmup_steps=10, minimum_ratio=0.0
    ) == pytest.approx(1.0)
    assert cosine_lr_multiplier(
        100, total_steps=100, warmup_steps=10, minimum_ratio=0.1
    ) == pytest.approx(0.1)


def _metadata():
    return make_checkpoint_metadata(
        run_id="run",
        task_id="task",
        role="source",
        source_id="source-00",
        dataset_id="test/tiny",
        dataset_revision="0123456789abcdef",
        dataset_spec_fingerprint="a" * 64,
        dataset_manifest_fingerprint="b" * 64,
        partition_kind="ind",
        partition_digest="c" * 64,
        model_key="resnet18",
        model_provider="torchvision",
        initialization="random",
        num_classes=2,
        training_config={"epochs": 10},
        protocol_digest="d" * 64,
        seed=7,
    )


def test_checkpoint_metadata_rejects_cross_partition_resume(tmp_path) -> None:
    metadata = _metadata()
    validate_resume_metadata(metadata, metadata.with_progress(
        completed_epochs=3,
        global_step=100,
        best_validation_top1=42.0,
        metrics={"validation_top1": 42.0},
    ))
    incompatible = replace(metadata, partition_digest="e" * 64)
    with pytest.raises(ValueError, match="partition_digest"):
        validate_resume_metadata(metadata, incompatible)

    manager = CheckpointManager(tmp_path / "checkpoints")
    assert manager.resolve_resume("auto") is None
    manager.latest_path.touch()
    assert manager.resolve_resume("auto") == manager.latest_path
