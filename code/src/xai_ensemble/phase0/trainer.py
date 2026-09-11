"""Resumable mixed-precision classifier training for one GPU or torchrun DDP."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

from .checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    export_inference_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .config import DistributedConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    owns_process_group: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_top1: float
    train_top5: float
    validation_loss: float | None
    validation_top1: float | None
    validation_top5: float | None
    learning_rate: float


@dataclass(frozen=True, slots=True)
class FitResult:
    completed_epochs: int
    global_step: int
    best_validation_top1: float | None
    history: tuple[EpochMetrics, ...]
    latest_checkpoint: str | None


def initialize_distributed(
    config: DistributedConfig,
    *,
    requested_device: str | None = None,
) -> DistributedContext:
    """Initialize from torchrun environment variables when WORLD_SIZE > 1."""

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Training requires PyTorch") from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = config.enabled and world_size > 1

    if requested_device is not None:
        device = requested_device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}" if use_ddp else "cuda"
    else:
        device = "cpu"

    owns = False
    if use_ddp:
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        backend = config.backend
        if backend == "auto":
            backend = "nccl" if device.startswith("cuda") else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=config.timeout_seconds),
            )
            owns = True
    return DistributedContext(
        enabled=use_ddp,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        owns_process_group=owns,
    )


def close_distributed(context: DistributedContext) -> None:
    if not context.owns_process_group:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def seed_training(seed: int, rank: int, deterministic: bool) -> None:
    derived = seed + rank
    random.seed(derived)
    try:
        import numpy as np

        np.random.seed(derived % (2**32))
    except ImportError:  # pragma: no cover - NumPy is a base dependency
        pass
    import torch

    torch.manual_seed(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def cosine_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    """Warm-up followed by a cosine decay; exposed for deterministic tests."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("minimum_ratio must be in [0, 1]")
    bounded_step = min(max(step, 0), total_steps)
    if warmup_steps and bounded_step < warmup_steps:
        return float(bounded_step + 1) / float(warmup_steps)
    progress = (bounded_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _build_optimizer(model: Any, config: TrainingConfig) -> Any:
    import torch

    values = config.optimizer
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if values.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=values.learning_rate,
            momentum=values.momentum,
            weight_decay=values.weight_decay,
        )
    return torch.optim.AdamW(
        parameters,
        lr=values.learning_rate,
        betas=values.betas,
        weight_decay=values.weight_decay,
    )


def _build_scheduler(optimizer: Any, config: TrainingConfig, steps_per_epoch: int) -> Any:
    import torch

    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    if config.scheduler == "constant":

        def multiplier(_step: int) -> float:
            return 1.0

    else:

        def multiplier(step: int) -> float:
            return cosine_lr_multiplier(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                minimum_ratio=config.min_learning_rate_ratio,
            )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _resolve_amp(torch: Any, device: str, requested: str) -> tuple[bool, Any | None]:
    if requested == "off" or not device.startswith("cuda"):
        return False, None
    if requested == "fp16":
        return True, torch.float16
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this CUDA device does not support it")
        return True, torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def _extract_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        if "image" in batch:
            images = batch["image"]
        elif "pixel_values" in batch:
            images = batch["pixel_values"]
        else:
            raise KeyError("Batch mapping must contain image or pixel_values")
        if "label" in batch:
            labels = batch["label"]
        elif "labels" in batch:
            labels = batch["labels"]
        else:
            raise KeyError("Batch mapping must contain label or labels")
        return images, labels
    if isinstance(batch, Sequence) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Batch must be a mapping or an (images, labels) sequence")


def _batch_statistics(logits: Any, labels: Any, loss: Any) -> tuple[float, float, float, int]:
    batch_size = int(labels.shape[0])
    max_k = min(5, int(logits.shape[1]))
    predictions = logits.topk(max_k, dim=1).indices
    correct = predictions.eq(labels.reshape(-1, 1))
    top1 = float(correct[:, :1].sum().detach().item())
    top5 = float(correct[:, :max_k].sum().detach().item())
    loss_sum = float(loss.detach().item()) * batch_size
    return loss_sum, top1, top5, batch_size


def _reduce_statistics(values: tuple[float, float, float, int], context: DistributedContext) -> tuple[float, float, float]:
    import torch

    totals = torch.tensor(values, dtype=torch.float64, device=context.device)
    if context.enabled:
        import torch.distributed as dist

        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(float(totals[3].item()), 1.0)
    return (
        float(totals[0].item()) / count,
        100.0 * float(totals[1].item()) / count,
        100.0 * float(totals[2].item()) / count,
    )


def _optimizer_to_device(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        *,
        device: str | None = None,
        context: DistributedContext | None = None,
    ) -> None:
        self.config = config
        self.context = context or initialize_distributed(
            config.distributed, requested_device=device
        )

    def fit(
        self,
        model: Any,
        train_loader: Any,
        validation_loader: Any,
        *,
        checkpoint_manager: CheckpointManager,
        checkpoint_metadata: CheckpointMetadata,
        resume_from: str | os.PathLike[str] | None = "auto",
        class_weights: Sequence[float] | None = None,
    ) -> FitResult:
        import torch

        seed_training(self.config.seed, self.context.rank, self.config.deterministic)
        device = torch.device(self.context.device)
        model = model.to(device)
        if self.config.channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        if self.context.enabled:
            from torch.nn.parallel import DistributedDataParallel

            model = DistributedDataParallel(
                model,
                device_ids=[self.context.local_rank] if device.type == "cuda" else None,
                find_unused_parameters=self.config.distributed.find_unused_parameters,
                broadcast_buffers=self.config.distributed.broadcast_buffers,
            )

        steps_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        optimizer = _build_optimizer(model, self.config)
        scheduler = _build_scheduler(optimizer, self.config, steps_per_epoch)
        amp_enabled, amp_dtype = _resolve_amp(torch, self.context.device, self.config.amp)
        scaler_enabled = amp_enabled and amp_dtype == torch.float16
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # PyTorch compatibility
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        state: dict[str, Any] = {
            "completed_epochs": 0,
            "global_step": 0,
            "best_validation_top1": None,
            "history": [],
        }
        resume_path = checkpoint_manager.resolve_resume(resume_from)
        if resume_path is not None:
            _, resumed = load_training_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_metadata=checkpoint_metadata,
            )
            state.update(resumed)
            _optimizer_to_device(optimizer, self.context.device)

        weight_tensor = None
        if class_weights is not None:
            values = tuple(float(value) for value in class_weights)
            if not values or not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("class_weights must contain finite positive values")
            weight_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=self.config.label_smoothing,
        )
        history = [EpochMetrics(**row) for row in state.get("history", [])]
        start_epoch = int(state["completed_epochs"])
        global_step = int(state["global_step"])
        best = state.get("best_validation_top1")

        for epoch in range(start_epoch, self.config.epochs):
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_values, global_step = self._train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                amp_enabled,
                amp_dtype,
                global_step,
            )
            validation_values: tuple[float, float, float] | None = None
            if (epoch + 1) % self.config.validate_every == 0 or epoch + 1 == self.config.epochs:
                validation_values = self.evaluate(
                    model,
                    validation_loader,
                    criterion=criterion,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            validation_loss, validation_top1, validation_top5 = (
                (None, None, None) if validation_values is None else validation_values
            )
            metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_values[0],
                train_top1=train_values[1],
                train_top5=train_values[2],
                validation_loss=validation_loss,
                validation_top1=validation_top1,
                validation_top5=validation_top5,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            history.append(metrics)
            improved = validation_top1 is not None and (best is None or validation_top1 > best)
            if improved:
                best = validation_top1
            state = {
                "completed_epochs": epoch + 1,
                "global_step": global_step,
                "best_validation_top1": best,
                "history": [asdict(item) for item in history],
            }
            metadata = checkpoint_metadata.with_progress(
                completed_epochs=epoch + 1,
                global_step=global_step,
                best_validation_top1=best,
                metrics={
                    key: value
                    for key, value in asdict(metrics).items()
                    if key != "epoch" and value is not None
                },
            )
            should_checkpoint = (
                (epoch + 1) % self.config.checkpoint_every == 0
                or epoch + 1 == self.config.epochs
            )
            if self.context.is_primary and should_checkpoint:
                save_training_checkpoint(
                    checkpoint_manager.latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    trainer_state=state,
                    metadata=metadata,
                )
                if self.config.keep_epoch_checkpoints:
                    save_training_checkpoint(
                        checkpoint_manager.epoch_path(epoch + 1),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        trainer_state=state,
                        metadata=metadata,
                    )
            if self.context.is_primary and improved:
                save_training_checkpoint(
                    checkpoint_manager.best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    trainer_state=state,
                    metadata=metadata,
                )
            if self.context.enabled:
                import torch.distributed as dist

                dist.barrier()

        latest = checkpoint_manager.latest_path
        if self.context.is_primary:
            if not checkpoint_manager.best_path.is_file():
                raise RuntimeError("training completed without a best validation checkpoint")
            export_inference_checkpoint(
                checkpoint_manager.best_path,
                checkpoint_manager.inference_path,
            )
        if self.context.enabled:
            import torch.distributed as dist

            dist.barrier()
        return FitResult(
            completed_epochs=int(state["completed_epochs"]),
            global_step=int(state["global_step"]),
            best_validation_top1=state.get("best_validation_top1"),
            history=tuple(history),
            latest_checkpoint=str(latest) if latest.is_file() else None,
        )

    def _train_epoch(
        self,
        model: Any,
        loader: Any,
        criterion: Any,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        amp_enabled: bool,
        amp_dtype: Any,
        global_step: int,
    ) -> tuple[tuple[float, float, float], int]:
        import torch

        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = [0.0, 0.0, 0.0, 0]
        accumulation = self.config.gradient_accumulation_steps
        final_group_size = len(loader) % accumulation or accumulation
        for batch_index, batch in enumerate(loader):
            images, labels = _extract_batch(batch)
            images = images.to(self.context.device, non_blocking=True)
            labels = labels.to(self.context.device, non_blocking=True).long().reshape(-1)
            if self.config.channels_last and images.ndim == 4 and images.is_cuda:
                images = images.contiguous(memory_format=torch.channels_last)
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            sync_context = nullcontext()
            if self.context.enabled and not should_step and hasattr(model, "no_sync"):
                sync_context = model.no_sync()
            divisor = (
                final_group_size
                if batch_index >= len(loader) - final_group_size
                else accumulation
            )
            with sync_context:
                with torch.autocast(
                    device_type=torch.device(self.context.device).type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    scaled_loss = loss / divisor
                scaler.scale(scaled_loss).backward()
            batch_values = _batch_statistics(logits, labels, loss)
            totals = [left + right for left, right in zip(totals, batch_values, strict=True)]
            if should_step:
                if self.config.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        return _reduce_statistics(tuple(totals), self.context), global_step

    def evaluate(
        self,
        model: Any,
        loader: Any,
        *,
        criterion: Any | None = None,
        amp_enabled: bool | None = None,
        amp_dtype: Any | None = None,
    ) -> tuple[float, float, float]:
        import torch

        model.eval()
        criterion = criterion or torch.nn.CrossEntropyLoss()
        if amp_enabled is None:
            amp_enabled, amp_dtype = _resolve_amp(
                torch, self.context.device, self.config.amp
            )
        totals = [0.0, 0.0, 0.0, 0]
        with torch.inference_mode():
            for batch in loader:
                images, labels = _extract_batch(batch)
                images = images.to(self.context.device, non_blocking=True)
                labels = labels.to(self.context.device, non_blocking=True).long().reshape(-1)
                with torch.autocast(
                    device_type=torch.device(self.context.device).type,
                    dtype=amp_dtype,
                    enabled=bool(amp_enabled),
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)
                values = _batch_statistics(logits, labels, loss)
                totals = [left + right for left, right in zip(totals, values, strict=True)]
        return _reduce_statistics(tuple(totals), self.context)
