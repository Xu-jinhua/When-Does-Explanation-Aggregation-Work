from __future__ import annotations

import copy
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

Architecture = Literal["cnn", "vit"]


@dataclass(frozen=True)
class ExplainerSpec:
    name: str
    candidate_architectures: tuple[Architecture, ...]
    differentiable_input: bool
    baseline_kind: str | None = None
    dedicated_vit: bool = False
    requires_relprop: bool = False


EXPLAINER_SPECS: dict[str, ExplainerSpec] = {
    "Saliency": ExplainerSpec("Saliency", ("cnn", "vit"), True),
    "InputXGradient": ExplainerSpec("InputXGradient", ("cnn", "vit"), True),
    "IntegratedGradients": ExplainerSpec("IntegratedGradients", ("cnn", "vit"), True, "tensor"),
    "GuidedBackprop": ExplainerSpec("GuidedBackprop", ("cnn",), True),
    "Deconvolution": ExplainerSpec("Deconvolution", ("cnn",), True),
    "FeatureAblation": ExplainerSpec("FeatureAblation", ("cnn", "vit"), False, "tensor"),
    "Occlusion": ExplainerSpec("Occlusion", ("cnn", "vit"), False, "tensor"),
    "DeepLift": ExplainerSpec("DeepLift", ("cnn",), True, "tensor"),
    "GradientShap": ExplainerSpec("GradientShap", ("cnn", "vit"), True, "distribution"),
    "DeepLiftShap": ExplainerSpec("DeepLiftShap", ("cnn",), True, "distribution"),
    "LRP": ExplainerSpec("LRP", ("cnn",), True),
    "CheferTransformerAttribution": ExplainerSpec(
        "CheferTransformerAttribution", ("vit",), True, requires_relprop=True
    ),
    # Chefer Partial/Full LRP are transformer algorithms.  They must never be
    # advertised as CNN methods unless a separate genuine CNN implementation
    # is added and compatibility-gated.
    "PartialLRP": ExplainerSpec("PartialLRP", ("vit",), True, requires_relprop=True),
    "FullLRP": ExplainerSpec("FullLRP", ("vit",), True, requires_relprop=True),
    "GradientAttentionRollout": ExplainerSpec(
        "GradientAttentionRollout", ("vit",), True, dedicated_vit=True
    ),
    "AttentionGradCAM": ExplainerSpec("AttentionGradCAM", ("vit",), True, dedicated_vit=True),
}


def candidate_roster(architecture: Architecture) -> list[str]:
    return [
        name
        for name, spec in EXPLAINER_SPECS.items()
        if architecture in spec.candidate_architectures
    ]


def build_explainer(model: Any, method: str, *, architecture: Architecture) -> Any:
    if method not in EXPLAINER_SPECS:
        raise KeyError(f"Unknown explainer method: {method}")
    spec = EXPLAINER_SPECS[method]
    if architecture not in spec.candidate_architectures:
        raise ValueError(f"{method} is not a candidate for architecture={architecture}")
    if spec.dedicated_vit:
        from .transformer import TransformerAttentionExplainer

        return TransformerAttentionExplainer(model, method)
    from .relprop import relprop_required

    if relprop_required(method, architecture):
        from .transformer import RelPropExplainer

        return RelPropExplainer(model, method)

    try:
        from captum import attr as captum_attr
    except ImportError as error:
        raise RuntimeError("Captum is required for generic attribution methods") from error

    if method == "LRP" and architecture == "cnn":
        return _CNNLRPAdapter(model)

    classes = {
        "Saliency": captum_attr.Saliency,
        "InputXGradient": captum_attr.InputXGradient,
        "IntegratedGradients": captum_attr.IntegratedGradients,
        "GuidedBackprop": captum_attr.GuidedBackprop,
        "Deconvolution": captum_attr.Deconvolution,
        "FeatureAblation": captum_attr.FeatureAblation,
        "Occlusion": captum_attr.Occlusion,
        "DeepLift": captum_attr.DeepLift,
        "GradientShap": captum_attr.GradientShap,
        "DeepLiftShap": captum_attr.DeepLiftShap,
        "LRP": captum_attr.LRP,
    }
    return classes[method](model)


def _attach_cnn_lrp_rules(model: Any) -> None:
    """Attach explicit Captum rules for timm CNN bookkeeping layers.

    Captum has rules for the mathematical layers in a ResNet, but its default
    registry intentionally excludes ``Identity`` and ``Flatten`` leaves used
    by timm's residual/downsample and global-pool wrappers.  Both are
    relevance-preserving transformations, so the numerically stable epsilon
    propagation rule is the appropriate explicit adapter.  This keeps CNN
    ``LRP`` genuine Captum LRP; it does not substitute a gradient method.
    """

    try:
        import torch.nn as nn
        from captum.attr._utils.lrp_rules import EpsilonRule
    except ImportError as error:  # pragma: no cover - optional GPU dependency
        raise RuntimeError("Captum is required for CNN LRP") from error

    for module in model.modules():
        if isinstance(module, (nn.Identity, nn.Flatten)):
            module.rule = EpsilonRule()


def _explicit_batch_norm_act(module: Any) -> Any:
    """Expand timm's fused BatchNormAct2d into Captum-visible leaves."""

    import torch.nn as nn

    reference_tensor = next(
        (
            tensor
            for tensor in (
                module.weight,
                module.bias,
                module.running_mean,
                module.running_var,
            )
            if tensor is not None
        ),
        None,
    )
    factory_kwargs = (
        {}
        if reference_tensor is None
        else {"device": reference_tensor.device, "dtype": reference_tensor.dtype}
    )
    batch_norm = nn.BatchNorm2d(
        module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        **factory_kwargs,
    )
    source_state = module.state_dict()
    batch_norm.load_state_dict(
        {key: source_state[key] for key in batch_norm.state_dict()},
        strict=True,
    )
    if module.affine:
        batch_norm.weight.requires_grad_(module.weight.requires_grad)
        batch_norm.bias.requires_grad_(module.bias.requires_grad)
    batch_norm.training = module.training

    expanded = nn.Sequential(
        OrderedDict(
            (
                ("batch_norm", batch_norm),
                ("drop", module.drop),
                ("activation", module.act),
            )
        )
    )
    expanded.training = module.training
    return expanded


def _clone_with_explicit_cnn_lrp_batch_norms(model: Any) -> tuple[Any, int]:
    """Clone a CNN and expose every fused timm BN operation to Captum LRP.

    ``BatchNormAct2d`` inherits from ``BatchNorm2d`` while also registering its
    drop and activation children. Captum traverses only leaf modules, so it
    sees those children but misses the batch-normalization operation performed
    by the non-leaf parent. The explicit ``BatchNorm2d -> drop -> activation``
    sequence is forward-equivalent and gives every operation its own LRP rule.
    """

    try:
        from timm.layers.norm_act import BatchNormAct2d
    except ImportError:  # pragma: no cover - timm is optional for generic tests
        return copy.deepcopy(model), 0

    cloned = copy.deepcopy(model)

    def expand(module: Any) -> tuple[Any, int]:
        if isinstance(module, BatchNormAct2d):
            return _explicit_batch_norm_act(module), 1
        count = 0
        for name, child in tuple(module.named_children()):
            replacement, child_count = expand(child)
            count += child_count
            if replacement is not child:
                module._modules[name] = replacement
        return module, count

    return expand(cloned)


def _cnn_lrp_forward_equivalence(
    reference_model: Any,
    attribution_model: Any,
    inputs: Any,
    *,
    sample_count: int = 4,
) -> tuple[int, float]:
    """Require exact FP32 forward equivalence on a small caller-provided batch."""

    import torch

    if not torch.is_tensor(inputs) or inputs.ndim < 1 or int(inputs.shape[0]) < 1:
        raise RuntimeError(
            "Captum CNN LRP compatibility gate requires a non-empty tensor input batch"
        )
    if reference_model.training or attribution_model.training:
        raise RuntimeError("Captum CNN LRP compatibility gate requires eval-mode models")
    count = min(int(inputs.shape[0]), int(sample_count))
    probe = inputs[:count].detach()
    with torch.inference_mode():
        reference_logits = reference_model(probe)
        attribution_logits = attribution_model(probe)
    if not torch.is_tensor(reference_logits) or not torch.is_tensor(attribution_logits):
        raise RuntimeError("Captum CNN LRP compatibility gate requires tensor logits")
    if reference_logits.shape != attribution_logits.shape:
        raise RuntimeError(
            "Captum CNN LRP compatibility gate failed: explicit-BN model changed "
            f"the logit shape from {tuple(reference_logits.shape)} to "
            f"{tuple(attribution_logits.shape)}"
        )
    if not bool(torch.isfinite(reference_logits).all()) or not bool(
        torch.isfinite(attribution_logits).all()
    ):
        raise RuntimeError("Captum CNN LRP compatibility gate found non-finite logits")
    difference = (reference_logits - attribution_logits).abs()
    maximum = 0.0 if difference.numel() == 0 else float(difference.max().item())
    predictions_equal = torch.equal(
        reference_logits.argmax(dim=1),
        attribution_logits.argmax(dim=1),
    )
    if not torch.equal(reference_logits, attribution_logits) or not predictions_equal:
        raise RuntimeError(
            "Captum CNN LRP compatibility gate failed: explicit-BN model is not "
            f"forward-equivalent (samples={count}, max_abs_logit_diff={maximum:.9g}, "
            f"predictions_equal={predictions_equal})"
        )
    return count, maximum


def _finite_attribution_output(value: Any) -> bool:
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if torch.is_tensor(item)]
        return bool(tensors) and all(bool(torch.isfinite(item).all()) for item in tensors)
    return False


class _CNNLRPAdapter:
    """Run Captum LRP on an isolated, forward-equivalent explicit-BN CNN.

    Captum removes custom ``rule`` attributes after every call.  Compatibility
    probes intentionally exercise several batch sizes and repeat the selected
    batch, so a single long-lived ``LRP`` instance would pass its first call
    and then fail on timm's Identity/Flatten leaves.  Rebuilding the small
    Captum wrapper also clears any transient hooks after a failed candidate.

    The first caller batch is the formal Phase-1 batch in production. A small
    prefix therefore checks the real checkpoint-backed reference model against
    the attribution-only clone before any artifact is written. The same first
    call must also produce entirely finite relevance values before the gate is
    recorded as passed.
    """

    def __init__(self, model: Any) -> None:
        self.reference_model = model
        self.model, self.expanded_batch_norm_count = _clone_with_explicit_cnn_lrp_batch_norms(model)
        self.compatibility_checked = False

    def attribute(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        from captum import attr as captum_attr

        was_deterministic = torch.are_deterministic_algorithms_enabled()
        if not was_deterministic:
            torch.use_deterministic_algorithms(True)
        try:
            compatibility: tuple[int, float] | None = None
            if not self.compatibility_checked:
                inputs = args[0] if args else kwargs.get("inputs")
                compatibility = _cnn_lrp_forward_equivalence(
                    self.reference_model,
                    self.model,
                    inputs,
                )
            _attach_cnn_lrp_rules(self.model)
            values = captum_attr.LRP(self.model).attribute(*args, **kwargs)
            if compatibility is not None:
                if not _finite_attribution_output(values):
                    raise RuntimeError(
                        "Captum CNN LRP compatibility gate failed: attribution "
                        "contains NaN or infinite values"
                    )
                self.compatibility_checked = True
                count, maximum = compatibility
                print(
                    "CNN_LRP_COMPATIBILITY_PASSED "
                    f"samples={count} expanded_batch_norm_act="
                    f"{self.expanded_batch_norm_count} "
                    f"max_abs_logit_diff={maximum:.9g} attribution_finite=true",
                    flush=True,
                )
            return values
        finally:
            if not was_deterministic:
                torch.use_deterministic_algorithms(False)


def _feature_mask(images: Any, patch_size: int) -> Any:
    import torch

    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Feature-ablation patch size must divide the input shape")
    grid_h, grid_w = height // patch_size, width // patch_size
    indices = torch.arange(grid_h * grid_w, device=images.device).reshape(grid_h, grid_w)
    mask = indices.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    return mask.reshape(1, 1, height, width).expand(batch, channels, -1, -1)


def compute_attribution(
    explainer: Any,
    method: str,
    images: Any,
    targets: Any,
    *,
    params: dict[str, Any] | None = None,
    baseline: Any | None = None,
    baseline_distribution: Any | None = None,
) -> Any:
    options = dict(params or {})
    # Baseline names and spaces are part of the scientific method lock, but
    # Captum expects resolved tensors through ``baselines``. Keeping these
    # semantic keys in ``options`` would pass unsupported arguments to Captum.
    options.pop("baseline", None)
    options.pop("baseline_distribution", None)
    options.pop("baseline_space", None)
    options.pop("gaussian_space", None)
    kwargs: dict[str, Any] = {"target": targets}
    if method in {
        "GradientAttentionRollout",
        "AttentionGradCAM",
        "CheferTransformerAttribution",
        "PartialLRP",
        "FullLRP",
    }:
        return explainer.attribute(images, target=targets, **options)
    if method == "Occlusion":
        patch = int(options.pop("patch_size", 14))
        kwargs.update(
            sliding_window_shapes=(images.shape[1], patch, patch),
            strides=(images.shape[1], patch, patch),
        )
        if baseline is not None:
            kwargs["baselines"] = baseline
    elif method == "FeatureAblation":
        patch = int(options.pop("patch_size", 14))
        kwargs["feature_mask"] = _feature_mask(images, patch)
        if baseline is not None:
            kwargs["baselines"] = baseline
    elif method in {"IntegratedGradients", "DeepLift"}:
        if baseline is None:
            raise ValueError(f"{method} requires a locked baseline")
        kwargs["baselines"] = baseline
    elif method in {"GradientShap", "DeepLiftShap"}:
        if baseline_distribution is None:
            raise ValueError(f"{method} requires a locked baseline distribution")
        kwargs["baselines"] = baseline_distribution
    kwargs.update(options)

    # Captum's FeatureAblation materializes masked model outputs and, in the
    # CUDA path, eventually converts them through a NumPy boundary that does
    # not support torch.bfloat16.  Keep the job's outer autocast for the
    # reference/current target evaluation, but run this attribution itself in
    # fp32.  The target tensor is already fixed before this call, so this does
    # not introduce a method-dependent target decision.
    if method == "FeatureAblation":
        try:
            import torch

            device = getattr(images, "device", None)
            context = (
                torch.autocast(device_type="cuda", enabled=False)
                if device is not None and device.type == "cuda"
                else nullcontext()
            )
        except ImportError:  # pragma: no cover - Captum itself is optional
            context = nullcontext()
        with context:
            return explainer.attribute(images, **kwargs)
    return explainer.attribute(images, **kwargs)
