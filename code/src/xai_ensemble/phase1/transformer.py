from __future__ import annotations

import math
from contextlib import AbstractContextManager
from typing import Any


class AttentionCapture(AbstractContextManager):
    """Capture timm-style ViT attention tensors and optional gradients.

    Fused attention is disabled because its softmax matrix is otherwise not
    observable. Compatibility pilots must verify that at least one attention
    layer was captured for every model identifier.
    """

    def __init__(self, model: Any, *, retain_grad: bool) -> None:
        self.model = model
        self.retain_grad = retain_grad
        self.handles: list[Any] = []
        self.attentions: list[Any] = []

    def __enter__(self) -> AttentionCapture:
        for module in self.model.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False
        for name, module in self.model.named_modules():
            if name.endswith("attn_drop"):
                self.handles.append(module.register_forward_hook(self._hook))
        return self

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        if getattr(output, "ndim", 0) != 4:
            return
        if self.retain_grad and getattr(output, "requires_grad", False):
            output.retain_grad()
        self.attentions.append(output)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()


def _normalize_transition(attention: Any) -> Any:
    import torch

    batch, tokens, _ = attention.shape
    identity = torch.eye(tokens, device=attention.device, dtype=attention.dtype)
    augmented = attention + identity.unsqueeze(0)
    return augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _rollout(matrices: list[Any]) -> Any:
    if not matrices:
        raise RuntimeError("No transformer attention matrices were captured")
    joint = _normalize_transition(matrices[0])
    for matrix in matrices[1:]:
        joint = _normalize_transition(matrix).bmm(joint)
    return joint


def _cls_patch_map(values: Any) -> Any:
    token_count = values.shape[-1] - 1
    grid = math.isqrt(token_count)
    if grid * grid != token_count:
        raise RuntimeError(f"Cannot reshape {token_count} patch tokens to a square grid")
    return values[:, 0, 1:].reshape(values.shape[0], 1, grid, grid)


class TransformerAttentionExplainer:
    DEDICATED_METHODS = {"GradientAttentionRollout", "AttentionGradCAM"}

    def __init__(self, model: Any, method: str) -> None:
        if method not in self.DEDICATED_METHODS:
            raise ValueError(f"Unknown transformer attention method: {method}")
        self.model = model
        self.method = method

    def attribute(self, inputs: Any, target: Any | None = None, **_: Any) -> Any:

        with AttentionCapture(self.model, retain_grad=True) as capture:
            logits = self.model(inputs)
            if target is None:
                target = logits.argmax(dim=1)
            selected = logits.gather(1, target.reshape(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            selected.backward(retain_graph=True)
        attentions = capture.attentions
        if not attentions:
            raise RuntimeError("Model did not expose timm-style attention dropout modules")

        weighted = []
        for attention in attentions:
            if attention.grad is None:
                raise RuntimeError("Attention gradients were not retained")
            weighted.append((attention * attention.grad).clamp_min(0).mean(dim=1))
        if self.method == "AttentionGradCAM":
            attention = attentions[-1][:, :, 0, 1:]
            gradient = attentions[-1].grad[:, :, 0, 1:]
            head_weights = gradient.mean(dim=-1, keepdim=True)
            values = (attention * head_weights).mean(dim=1).clamp_min(0)
            token_count = int(values.shape[-1])
            grid = math.isqrt(token_count)
            if grid * grid != token_count:
                raise RuntimeError(
                    f"Cannot reshape {token_count} GradCAM patch tokens to a square grid"
                )
            return values.reshape(values.shape[0], 1, grid, grid)
        return _cls_patch_map(_rollout(weighted))


class RelPropExplainer:
    """Adapter for the pinned Chefer models that expose genuine ``relprop``.

    Upstream ``last_layer`` and ``transformer_attribution`` index the first
    batch member internally.  Running one sample at a time is therefore a
    correctness requirement, not merely a memory optimization.
    """

    _METHOD_MAP = {
        "CheferTransformerAttribution": "transformer_attribution",
        "PartialLRP": "last_layer",
        "FullLRP": "full",
    }

    def __init__(self, model: Any, method: str) -> None:
        if not hasattr(model, "relprop"):
            raise TypeError("LRP variants require a model exposing relprop or a compatible plugin")
        self.model = model
        self.method = method

    def attribute(self, inputs: Any, target: Any | None = None, **_: Any) -> Any:
        import torch

        if self.method not in self._METHOD_MAP:
            raise ValueError(f"Unknown RelProp method: {self.method}")
        if target is not None and int(target.reshape(-1).shape[0]) != int(inputs.shape[0]):
            raise ValueError("RelProp targets must have one value per input")
        maps = []
        for index in range(int(inputs.shape[0])):
            sample = inputs[index : index + 1]
            logits = self.model(sample)
            sample_target = (
                logits.argmax(dim=1)
                if target is None
                else target[index : index + 1].to(logits.device)
            )
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, sample_target.reshape(-1, 1), 1)
            self.model.zero_grad(set_to_none=True)
            (one_hot * logits).sum().backward(retain_graph=True)
            result = self.model.relprop(
                one_hot,
                method=self._METHOD_MAP[self.method],
                is_ablation=False,
                start_layer=1,
                alpha=1,
            )
            maps.append(self._format_one(result))
        return torch.cat(maps, dim=0)

    @staticmethod
    def _format_one(result: Any) -> Any:
        """Canonicalize one upstream result to ``[1,1,H,W]``."""

        if result.ndim == 1:
            values = result.reshape(1, -1)
        elif result.ndim == 2:
            # FullLRP returns [H,W]; patch-level variants may return [1,T].
            if result.shape[0] == result.shape[1] and int(result.shape[0]) > 1:
                return result.unsqueeze(0).unsqueeze(0)
            values = result
        elif result.ndim == 3:
            if int(result.shape[0]) != 1:
                raise RuntimeError(
                    f"One-sample RelProp returned unexpected shape {tuple(result.shape)}"
                )
            return result.unsqueeze(1)
        elif result.ndim == 4:
            if int(result.shape[0]) != 1 or int(result.shape[1]) != 1:
                raise RuntimeError(
                    f"One-sample RelProp returned unexpected shape {tuple(result.shape)}"
                )
            return result
        else:
            raise RuntimeError(f"Unsupported RelProp shape: {tuple(result.shape)}")

        tokens = int(values.shape[-1])
        grid = math.isqrt(tokens)
        if grid * grid != tokens:
            if tokens <= 1:
                raise RuntimeError(f"Cannot reshape {tokens} RelProp values to a patch grid")
            values = values[:, 1:]
            tokens = int(values.shape[-1])
            grid = math.isqrt(tokens)
        if grid * grid != tokens:
            raise RuntimeError(f"Cannot reshape {tokens} RelProp values to a square patch grid")
        return values.reshape(1, 1, grid, grid)
