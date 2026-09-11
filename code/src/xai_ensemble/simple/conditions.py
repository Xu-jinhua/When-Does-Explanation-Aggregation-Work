"""Deterministic raw-image conditions for the simple experiment."""

from __future__ import annotations

from typing import Any


def natural_corruption(
    *,
    raw_images: Any,
    labels: Any,
    indices: Any,
    model: Any,
    normalize: Any,
    seed: int,
    kind: str,
    severity: float,
) -> Any:
    """Apply one paper-defined corruption to one raw [0,1] image."""

    import torch

    del labels, indices, model, normalize
    if raw_images.ndim != 4 or int(raw_images.shape[0]) != 1:
        raise ValueError("natural_corruption expects one BCHW image")
    if not 0.0 <= float(severity) <= 1.0:
        raise ValueError("Corruption severity must lie in [0,1]")
    generator = torch.Generator(device=raw_images.device)
    generator.manual_seed(int(seed))
    if kind in {"gaussian", "speckle"}:
        noise = torch.randn(
            raw_images.shape,
            generator=generator,
            device=raw_images.device,
            dtype=raw_images.dtype,
        )
        result = (
            raw_images + float(severity) * noise
            if kind == "gaussian"
            else raw_images + raw_images * float(severity) * noise
        )
    elif kind == "salt_pepper":
        selector = torch.rand(
            raw_images.shape,
            generator=generator,
            device=raw_images.device,
            dtype=raw_images.dtype,
        )
        result = raw_images.clone()
        half = float(severity) / 2.0
        result[selector < half] = 0.0
        result[(selector >= half) & (selector < float(severity))] = 1.0
    else:
        raise ValueError(f"Unsupported natural corruption: {kind}")
    return result.clamp(0.0, 1.0)


__all__ = ["natural_corruption"]
