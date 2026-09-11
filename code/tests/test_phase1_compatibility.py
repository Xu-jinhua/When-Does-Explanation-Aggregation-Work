from __future__ import annotations

from pathlib import Path

import pytest

from xai_ensemble.core.protocol import load_protocol
from xai_ensemble.phase1.compatibility import _candidate_methods, _seed_attribution
from xai_ensemble.phase1.explainers import (
    _clone_with_explicit_cnn_lrp_batch_norms,
    build_explainer,
    compute_attribution,
)


def test_vit_compatibility_roster_contains_generic_and_dedicated_methods() -> None:
    protocol = load_protocol(Path(__file__).parents[1] / "configs" / "protocols" / "core.yaml")
    methods = _candidate_methods(protocol.resolved_dict(), "vit_base_patch16_224", "vit")
    families = {method.family for method in methods}
    assert {
        "Saliency",
        "IntegratedGradients",
        "CheferTransformerAttribution",
        "PartialLRP",
        "FullLRP",
        "GradientAttentionRollout",
        "AttentionGradCAM",
    }.issubset(families)
    assert {
        "GuidedBackprop",
        "DeepLift",
        "DeepLiftShap",
        "LRP",
        "Rollout",
        "AttnLast",
    }.isdisjoint(families)
    assert len(families) == 11


def test_cnn_roster_keeps_genuine_lrp_but_excludes_chefer_vit_variants() -> None:
    protocol = load_protocol(Path(__file__).parents[1] / "configs" / "protocols" / "core.yaml")
    methods = _candidate_methods(protocol.resolved_dict(), "resnet18", "cnn")
    families = {method.family for method in methods}
    assert "LRP" in families
    assert "PartialLRP" not in families
    assert "FullLRP" not in families


def test_gradient_shap_repeat_binds_numpy_and_torch_rngs() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from captum.attr import GradientShap

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 2),
    ).eval()
    images = torch.rand(4, 3, 4, 4)
    targets = torch.tensor([0, 1, 0, 1])
    baselines = torch.rand(3, 3, 4, 4)
    values = []
    for _ in range(2):
        _seed_attribution(20260714, torch.device("cpu"))
        values.append(
            GradientShap(model).attribute(
                images,
                baselines=baselines,
                target=targets,
                n_samples=40,
                stdevs=0.0,
            )
        )
    assert torch.equal(values[0], values[1])


def test_attribution_strips_semantic_baseline_space_fields() -> None:
    torch = pytest.importorskip("torch")

    class RecordingExplainer:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def attribute(self, inputs, **kwargs):
            self.kwargs = kwargs
            return inputs

    explainer = RecordingExplainer()
    images = torch.rand(1, 3, 4, 4)
    compute_attribution(
        explainer,
        "IntegratedGradients",
        images,
        torch.tensor([0]),
        params={
            "n_steps": 50,
            "baseline": "zero",
            "baseline_space": "model_input",
            "gaussian_space": "model_input",
        },
        baseline=torch.zeros_like(images),
    )

    assert explainer.kwargs is not None
    assert explainer.kwargs["n_steps"] == 50
    assert "baseline_space" not in explainer.kwargs
    assert "gaussian_space" not in explainer.kwargs


def test_cnn_lrp_attaches_rules_to_timm_bookkeeping_layers() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 2, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Identity(),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * 4 * 4, 2),
    ).eval()
    explainer = build_explainer(model, "LRP", architecture="cnn")
    for batch_size in (1, 2):
        values = compute_attribution(
            explainer,
            "LRP",
            torch.rand(batch_size, 3, 4, 4),
            torch.zeros(batch_size, dtype=torch.long),
        )
        assert values.shape == (batch_size, 3, 4, 4)
        assert bool(torch.isfinite(values).all())


def _batch_norm_act_classifier() -> object:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from timm.layers.norm_act import BatchNormAct2d

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=1, bias=False),
        BatchNormAct2d(4, eps=1e-5, momentum=0.03, inplace=False),
        torch.nn.Conv2d(4, 4, kernel_size=1, bias=False),
        torch.nn.Sequential(
            BatchNormAct2d(4, eps=1e-5, momentum=0.07, inplace=False),
        ),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 4, 3),
    ).eval()
    generator = torch.Generator(device="cpu").manual_seed(20260801)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
        for module in model.modules():
            if isinstance(module, BatchNormAct2d):
                module.running_mean.copy_(
                    torch.randn(module.running_mean.shape, generator=generator) * 0.1
                )
                module.running_var.copy_(
                    torch.rand(module.running_var.shape, generator=generator) + 0.01
                )
    return model


def test_cnn_lrp_expands_composite_batch_norm_without_mutating_reference() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from timm.layers.norm_act import BatchNormAct2d

    model = _batch_norm_act_classifier()
    original_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    clone, count = _clone_with_explicit_cnn_lrp_batch_norms(model)

    assert count == 2
    assert sum(isinstance(module, BatchNormAct2d) for module in model.modules()) == 2
    assert not any(isinstance(module, BatchNormAct2d) for module in clone.modules())
    assert list(clone[1]._modules) == ["batch_norm", "drop", "activation"]
    assert model[1] is not clone[1]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original_state.items())

    images = torch.randn(4, 3, 4, 4, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        reference_logits = model(images)
        clone_logits = clone(images)
    assert torch.equal(reference_logits, clone_logits)
    assert torch.equal(reference_logits.argmax(dim=1), clone_logits.argmax(dim=1))


def test_cnn_lrp_explicit_batch_norm_is_finite_with_extreme_running_stats() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from timm.layers.norm_act import BatchNormAct2d

    model = _batch_norm_act_classifier()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BatchNormAct2d):
                module.running_var.fill_(torch.finfo(torch.float32).tiny)
                module.weight.fill_(2.25)
                module.bias.zero_()
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                module.weight.mul_(0.01)
                if module.bias is not None:
                    module.bias.zero_()

    images = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(11)) * 0.01
    with torch.no_grad():
        targets = model(images).argmax(dim=1)
    explainer = build_explainer(model, "LRP", architecture="cnn")
    values = compute_attribution(explainer, "LRP", images, targets)

    assert values.shape == images.shape
    assert bool(torch.isfinite(values).all())
    assert explainer.compatibility_checked is True
    assert explainer.expanded_batch_norm_count == 2


def test_cnn_lrp_compatibility_gate_rejects_non_equivalent_clone() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")

    model = _batch_norm_act_classifier()
    explainer = build_explainer(model, "LRP", architecture="cnn")
    with torch.no_grad():
        explainer.model[-1].bias.add_(1.0)
    images = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(19))
    targets = torch.zeros(2, dtype=torch.long)

    with pytest.raises(RuntimeError, match="not forward-equivalent"):
        compute_attribution(explainer, "LRP", images, targets)
    assert explainer.compatibility_checked is False
