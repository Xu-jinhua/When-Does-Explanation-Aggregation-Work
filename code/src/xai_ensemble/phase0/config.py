"""Validated Phase 0 runtime configuration.

These dataclasses are deliberately independent of PyTorch so that planning,
job generation, and unit tests work on a login node without GPU packages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    batch_size: int
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LoaderConfig:
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)

    def __post_init__(self) -> None:
        if self.name not in {"sgd", "adamw"}:
            raise ValueError("optimizer name must be 'sgd' or 'adamw'")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if len(self.betas) != 2 or not all(0 <= value < 1 for value in self.betas):
            raise ValueError("AdamW betas must contain two values in [0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OptimizerConfig:
        normalized = dict(values)
        if "betas" in normalized:
            normalized["betas"] = tuple(float(value) for value in normalized["betas"])
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    enabled: bool = True
    backend: str = "auto"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    timeout_seconds: int = 1_800

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "nccl", "gloo"}:
            raise ValueError("distributed backend must be auto, nccl, or gloo")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    optimizer: OptimizerConfig
    train_loader: LoaderConfig
    validation_loader: LoaderConfig
    warmup_epochs: int = 0
    scheduler: str = "cosine"
    min_learning_rate_ratio: float = 0.0
    label_smoothing: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None
    amp: str = "auto"
    channels_last: bool = True
    seed: int = 20260714
    deterministic: bool = False
    validate_every: int = 1
    checkpoint_every: int = 1
    keep_epoch_checkpoints: bool = False
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if self.scheduler not in {"cosine", "constant"}:
            raise ValueError("scheduler must be cosine or constant")
        if not 0 <= self.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when configured")
        if self.amp not in {"auto", "off", "fp16", "bf16"}:
            raise ValueError("amp must be auto, off, fp16, or bf16")
        if self.validate_every <= 0 or self.checkpoint_every <= 0:
            raise ValueError("validation/checkpoint intervals must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        normalized = dict(values)
        normalized["optimizer"] = OptimizerConfig.from_mapping(normalized["optimizer"])
        normalized["train_loader"] = LoaderConfig.from_mapping(normalized["train_loader"])
        normalized["validation_loader"] = LoaderConfig.from_mapping(
            normalized["validation_loader"]
        )
        if "distributed" in normalized:
            normalized["distributed"] = DistributedConfig(**normalized["distributed"])
        return cls(**normalized)


def default_training_config(model_family: str, *, epochs: int = 100) -> TrainingConfig:
    """Return the manuscript's starting recipe, to be checked by Phase 0 pilots."""

    family = model_family.lower()
    if family == "cnn":
        return TrainingConfig(
            epochs=epochs,
            optimizer=OptimizerConfig(
                name="sgd", learning_rate=1e-3, weight_decay=1e-4, momentum=0.9
            ),
            train_loader=LoaderConfig(batch_size=64),
            validation_loader=LoaderConfig(batch_size=128, drop_last=False),
            warmup_epochs=0,
            scheduler="cosine",
            amp="auto",
        )
    if family == "vit":
        return TrainingConfig(
            epochs=epochs,
            optimizer=OptimizerConfig(name="adamw", learning_rate=5e-5, weight_decay=0.05),
            train_loader=LoaderConfig(batch_size=32),
            validation_loader=LoaderConfig(batch_size=64, drop_last=False),
            warmup_epochs=min(5, max(0, epochs - 1)),
            scheduler="cosine",
            gradient_clip_norm=1.0,
            amp="auto",
        )
    raise ValueError(f"Unknown model family: {model_family!r}")
