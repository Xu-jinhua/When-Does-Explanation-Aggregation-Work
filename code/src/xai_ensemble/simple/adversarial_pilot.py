"""GPU pilot for Sara-style, explanation-targeted adversarial noise."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256, stable_seed
from xai_ensemble.core.io import atomic_write_json

from .config import ModelConfig, SimpleExperiment
from .data import load_model, load_split
from .manifest_identity import dataset_manifest_identity_sha256
from .phase2 import _scores_to_ranks, attribution_to_patch_scores

SourceMethod = Literal[
    "DeepLift",
    "GradientAttentionRollout",
    "TransformerAttribution",
]
LEGACY_GRADIENT_ATTENTION_ROLLOUT_ID = "TransformerAttribution"


def attack_source_semantic_name(source_method: str) -> str:
    """Map the immutable legacy artifact id to its scientific method name."""

    if source_method == LEGACY_GRADIENT_ATTENTION_ROLLOUT_ID:
        return "GradientAttentionRollout"
    return source_method


@dataclass(frozen=True, slots=True)
class SaraAttackConfig:
    """Scientific and optimizer parameters for one per-sample attack."""

    epsilon: float = 2.0 / 255.0
    steps: int = 100
    learning_rate: float = 0.1
    classification_weight: float = 1e-4
    top_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in (0,1]")
        if self.steps <= 0 or self.learning_rate <= 0.0:
            raise ValueError("steps and learning_rate must be positive")
        if self.classification_weight < 0.0:
            raise ValueError("classification_weight cannot be negative")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0,1]")


@dataclass(slots=True)
class AttackBatchResult:
    clean_images: Any
    adversarial_images: Any
    deltas: Any
    targets: Any
    clean_logits: Any
    adversarial_logits: Any
    random_logits: Any
    clean_attributions: Any
    adversarial_attributions: Any
    random_attributions: Any
    clean_objective: Any
    adversarial_objective: Any
    random_objective: Any
    random_images: Any
    best_steps: Any


def sara_postprocess_attribution(attributions: Any) -> Any:
    """Sara's channel sum, positive relevance, and per-sample min-max map."""

    import torch

    if attributions.ndim != 4:
        raise ValueError("Source attribution must have [N,C,H,W] shape")
    values = torch.relu(attributions.sum(dim=1, keepdim=True))
    flat = values.flatten(start_dim=1)
    minima = flat.min(dim=1).values.reshape(-1, 1, 1, 1)
    maxima = flat.max(dim=1).values.reshape(-1, 1, 1, 1)
    return (values - minima) / (maxima - minima + 1e-8)


def _raw_input_model(classifier: Any, preprocessing: Mapping[str, Any]) -> Any:
    import torch

    class RawInputModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = classifier
            self.register_buffer(
                "input_mean",
                torch.as_tensor(preprocessing["mean"], dtype=torch.float32).reshape(1, -1, 1, 1),
            )
            self.register_buffer(
                "input_std",
                torch.as_tensor(preprocessing["std"], dtype=torch.float32).reshape(1, -1, 1, 1),
            )

        def forward(self, raw_images: Any) -> Any:
            return self.classifier((raw_images - self.input_mean) / self.input_std)

    return RawInputModel().eval()


def _gradient_attention_rollout(
    model: Any,
    raw_images: Any,
    targets: Any,
    *,
    create_graph: bool,
) -> Any:
    """Raw-attention gradient rollout with a true second-order attack graph."""

    import torch

    from xai_ensemble.phase1.transformer import AttentionCapture, _cls_patch_map, _rollout

    with AttentionCapture(model, retain_grad=False) as capture:
        logits = model(raw_images)
    if not capture.attentions:
        raise RuntimeError("Model did not expose timm-style attention matrices")
    selected = logits.gather(1, targets.reshape(-1, 1)).sum()
    gradients = torch.autograd.grad(
        selected,
        tuple(capture.attentions),
        create_graph=create_graph,
        retain_graph=True,
    )
    weighted = [
        (attention * gradient).clamp_min(0).mean(dim=1)
        for attention, gradient in zip(capture.attentions, gradients, strict=True)
    ]
    return _cls_patch_map(_rollout(weighted))


def _source_attribution(
    model: Any,
    raw_images: Any,
    targets: Any,
    *,
    source_method: SourceMethod,
    create_graph: bool,
) -> Any:
    if source_method == "DeepLift":
        from captum.attr import DeepLift

        values = DeepLift(model).attribute(
            raw_images,
            baselines=raw_images.new_zeros(raw_images.shape),
            target=targets,
        )
    elif source_method in {
        "GradientAttentionRollout",
        LEGACY_GRADIENT_ATTENTION_ROLLOUT_ID,
    }:
        values = _gradient_attention_rollout(
            model,
            raw_images,
            targets,
            create_graph=create_graph,
        )
    else:  # pragma: no cover - guarded by the public entry points
        raise ValueError(f"Unsupported attack source: {source_method}")
    return sara_postprocess_attribution(values)


def _top_objective(values: Any, indices: Any) -> Any:
    return values.flatten(start_dim=1).gather(1, indices).mean(dim=1)


def _initial_delta(images: Any, epsilon: float, sample_seeds: Sequence[int]) -> Any:
    import torch

    if len(sample_seeds) != int(images.shape[0]):
        raise ValueError("sample_seeds must contain one seed per image")
    delta = torch.empty_like(images)
    for position, seed in enumerate(sample_seeds):
        generator = torch.Generator(device=images.device)
        generator.manual_seed(int(seed))
        delta[position].uniform_(-epsilon, epsilon, generator=generator)
    lower = torch.maximum(-images, torch.full_like(images, -epsilon))
    upper = torch.minimum(1.0 - images, torch.full_like(images, epsilon))
    return delta.clamp_(lower, upper), lower, upper


def sara_attack_batch(
    model: Any,
    raw_images: Any,
    *,
    source_method: SourceMethod,
    config: SaraAttackConfig,
    sample_seeds: Sequence[int],
) -> AttackBatchResult:
    """Attack independent images together while tracking valid candidates per row."""

    import torch
    import torch.nn.functional as functional

    if raw_images.ndim != 4 or int(raw_images.shape[1]) != 3:
        raise ValueError("raw_images must have [N,3,H,W] shape")
    if not bool(torch.isfinite(raw_images).all()):
        raise ValueError("raw_images contain NaN or infinite values")
    if float(raw_images.min()) < 0.0 or float(raw_images.max()) > 1.0:
        raise ValueError("raw_images must lie in raw pixel space [0,1]")

    clean_images = raw_images.detach().clone()
    clean_inputs = clean_images.clone().requires_grad_(True)
    clean_logits = model(clean_inputs)
    targets = clean_logits.argmax(dim=1).detach()
    clean_maps = _source_attribution(
        model,
        clean_inputs,
        targets,
        source_method=source_method,
        create_graph=False,
    ).detach()
    feature_count = int(clean_maps[0].numel())
    source_top_k = max(1, int(feature_count * config.top_fraction))
    top_indices = torch.argsort(
        clean_maps.flatten(start_dim=1),
        dim=1,
        descending=True,
        stable=True,
    )[:, :source_top_k].detach()
    clean_explanation_loss = _top_objective(clean_maps, top_indices)
    clean_classification_loss = functional.cross_entropy(
        clean_logits,
        targets,
        reduction="none",
    ).detach()
    clean_objective = (
        clean_explanation_loss + config.classification_weight * clean_classification_loss
    ).detach()

    initial, lower, upper = _initial_delta(clean_images, config.epsilon, sample_seeds)
    delta = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([delta], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.learning_rate / 10.0,
    )

    best_images = clean_images.clone()
    best_logits = clean_logits.detach().clone()
    best_maps = clean_maps.clone()
    best_objective = clean_objective.clone()
    best_steps = torch.full(
        (int(clean_images.shape[0]),),
        -1,
        device=clean_images.device,
        dtype=torch.int64,
    )
    random_images = None
    random_logits = None
    random_maps = None
    random_objective = None

    def record_candidates(
        candidate_images: Any,
        candidate_logits: Any,
        candidate_maps: Any,
        candidate_objective: Any,
        step: int,
    ) -> None:
        nonlocal best_images, best_logits, best_maps, best_objective, best_steps

        preserved = candidate_logits.argmax(dim=1) == targets
        improved = preserved & (candidate_objective < best_objective)
        if not bool(improved.any()):
            return
        image_mask = improved.reshape(-1, 1, 1, 1)
        logit_mask = improved.reshape(-1, 1)
        best_images = torch.where(image_mask, candidate_images.detach(), best_images)
        best_logits = torch.where(logit_mask, candidate_logits.detach(), best_logits)
        best_maps = torch.where(image_mask, candidate_maps.detach(), best_maps)
        best_objective = torch.where(improved, candidate_objective.detach(), best_objective)
        best_steps = torch.where(
            improved,
            torch.full_like(best_steps, step),
            best_steps,
        )

    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        candidate_images = clean_images + delta
        candidate_logits = model(candidate_images)
        candidate_maps = _source_attribution(
            model,
            candidate_images,
            targets,
            source_method=source_method,
            create_graph=True,
        )
        explanation_loss = _top_objective(candidate_maps, top_indices)
        classification_loss = functional.cross_entropy(
            candidate_logits,
            targets,
            reduction="none",
        )
        objective = explanation_loss + config.classification_weight * classification_loss
        if step == 0:
            random_images = candidate_images.detach().clone()
            random_logits = candidate_logits.detach().clone()
            random_maps = candidate_maps.detach().clone()
            random_objective = objective.detach().clone()
        record_candidates(candidate_images, candidate_logits, candidate_maps, objective, step)

        # Every row has disjoint delta parameters. Summing keeps one image's
        # optimizer update independent of the runtime batch size.
        objective.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            delta.clamp_(lower, upper)

    final_images = (clean_images + delta.detach()).requires_grad_(True)
    final_logits = model(final_images)
    final_maps = _source_attribution(
        model,
        final_images,
        targets,
        source_method=source_method,
        create_graph=False,
    )
    final_objective = _top_objective(final_maps, top_indices) + (
        config.classification_weight
        * functional.cross_entropy(final_logits, targets, reduction="none")
    )
    record_candidates(final_images, final_logits, final_maps, final_objective, config.steps)

    adversarial_images = best_images.detach()
    deltas = adversarial_images - clean_images
    if float(deltas.abs().max()) > config.epsilon + 1e-6:
        raise RuntimeError("Attack escaped its L-infinity constraint")
    if float(adversarial_images.min()) < 0.0 or float(adversarial_images.max()) > 1.0:
        raise RuntimeError("Attack escaped the raw image bounds")
    if not bool(best_logits.argmax(dim=1).eq(targets).all()):
        raise RuntimeError("Per-sample candidate selection changed a clean prediction")
    if any(
        value is None for value in (random_images, random_logits, random_maps, random_objective)
    ):
        raise RuntimeError("Attack did not record its random-start control")

    return AttackBatchResult(
        clean_images=clean_images.detach(),
        adversarial_images=adversarial_images,
        deltas=deltas.detach(),
        targets=targets.detach(),
        clean_logits=clean_logits.detach(),
        adversarial_logits=best_logits.detach(),
        random_logits=random_logits,
        clean_attributions=clean_maps.detach(),
        adversarial_attributions=best_maps.detach(),
        random_attributions=random_maps,
        clean_objective=clean_objective.detach(),
        adversarial_objective=best_objective.detach(),
        random_objective=random_objective,
        random_images=random_images,
        best_steps=best_steps.detach(),
    )


def _inversion_count(permutation: np.ndarray) -> int:
    count = 0
    size = int(permutation.size)
    tree = np.zeros(size + 1, dtype=np.int64)
    for seen, raw_value in enumerate(permutation):
        value = int(raw_value) + 1
        prefix = 0
        cursor = value
        while cursor > 0:
            prefix += int(tree[cursor])
            cursor -= cursor & -cursor
        count += seen - prefix
        cursor = value
        while cursor <= size:
            tree[cursor] += 1
            cursor += cursor & -cursor
    return count


def rank_change_statistics(
    clean_attributions: Any,
    adversarial_attributions: Any,
    *,
    patch_sizes: Sequence[int] = (8, 14, 16),
    top_k: int = 20,
) -> Mapping[str, Mapping[str, float | int]]:
    """Measure whether paper-defined patch ranks actually changed."""

    clean = np.asarray(clean_attributions, dtype=np.float32)
    adversarial = np.asarray(adversarial_attributions, dtype=np.float32)
    if clean.shape != adversarial.shape or clean.ndim != 4:
        raise ValueError("Clean and adversarial maps must have matching [N,C,H,W] shapes")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    output: dict[str, Mapping[str, float | int]] = {}
    for patch_size in patch_sizes:
        clean_scores = attribution_to_patch_scores(clean, int(patch_size))
        adversarial_scores = attribution_to_patch_scores(adversarial, int(patch_size))
        clean_ranks = _scores_to_ranks(clean_scores)
        adversarial_ranks = _scores_to_ranks(adversarial_scores)
        patches = int(clean_ranks.shape[1])
        selected = min(top_k, patches)
        rank_delta = np.abs(clean_ranks - adversarial_ranks)
        maximum_footrule = max(1, (patches * patches) // 2)
        footrule = rank_delta.sum(axis=1, dtype=np.float64) / maximum_footrule
        squared = np.square(
            clean_ranks.astype(np.float64) - adversarial_ranks.astype(np.float64)
        ).sum(axis=1)
        denominator = patches * (patches * patches - 1)
        spearman = np.ones(clean_ranks.shape[0], dtype=np.float64)
        if denominator:
            spearman -= 6.0 * squared / denominator
        kendall = []
        for clean_row, adversarial_row in zip(clean_ranks, adversarial_ranks, strict=True):
            clean_order = np.argsort(clean_row, kind="stable")
            sequence = adversarial_row[clean_order]
            pairs = max(1, patches * (patches - 1) // 2)
            kendall.append(_inversion_count(sequence) / pairs)
        clean_top = clean_ranks < selected
        adversarial_top = adversarial_ranks < selected
        overlap = np.logical_and(clean_top, adversarial_top).sum(axis=1) / selected
        output[f"p{int(patch_size)}"] = {
            "patch_count": patches,
            "sample_count": int(clean_ranks.shape[0]),
            "rank_changed_fraction": float(np.any(rank_delta != 0, axis=1).mean()),
            "changed_patch_fraction": float((rank_delta != 0).mean()),
            "normalized_footrule_mean": float(footrule.mean()),
            "normalized_kendall_mean": float(np.mean(kendall)),
            "spearman_mean": float(spearman.mean()),
            "top_k": selected,
            "top_k_overlap_mean": float(overlap.mean()),
            "top_k_replacement_mean": float((1.0 - overlap).mean()),
            "top_k_changed_fraction": float((overlap < 1.0).mean()),
        }
    return output


def _to_cpu(result: AttackBatchResult) -> AttackBatchResult:
    import torch

    values = {
        field.name: (
            getattr(result, field.name).detach().to(device="cpu")
            if torch.is_tensor(getattr(result, field.name))
            else getattr(result, field.name)
        )
        for field in fields(result)
    }
    return AttackBatchResult(**values)


def _concatenate(results: Sequence[AttackBatchResult]) -> AttackBatchResult:
    import torch

    if not results:
        raise ValueError("At least one attack batch is required")
    return AttackBatchResult(
        **{
            field.name: torch.cat(
                [getattr(item, field.name) for item in results],
                dim=0,
            )
            for field in fields(results[0])
        }
    )


def _resize_maps(values: Any, height: int, width: int) -> Any:
    import torch.nn.functional as functional

    if tuple(values.shape[-2:]) == (height, width):
        return values
    return functional.interpolate(
        values, size=(height, width), mode="bilinear", align_corners=False
    )


def _attack_summary(result: AttackBatchResult, epsilon: float) -> Mapping[str, Any]:
    import torch

    delta_flat = result.deltas.flatten(start_dim=1)
    linf = delta_flat.abs().max(dim=1).values
    l2 = torch.linalg.vector_norm(delta_flat, ord=2, dim=1)
    improved = result.adversarial_objective < result.clean_objective
    preserved = result.adversarial_logits.argmax(dim=1).eq(result.targets)
    return {
        "sample_count": int(result.targets.shape[0]),
        "prediction_preserved_fraction": float(preserved.float().mean()),
        "objective_improved_fraction": float(improved.float().mean()),
        "clean_objective_mean": float(result.clean_objective.mean()),
        "adversarial_objective_mean": float(result.adversarial_objective.mean()),
        "linf_max": float(linf.max()),
        "linf_mean": float(linf.mean()),
        "l2_mean": float(l2.mean()),
        "epsilon": float(epsilon),
        "nonzero_delta_fraction": float((linf > 0).float().mean()),
    }


def _is_cuda_oom(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return "out of memory" in str(error).lower()


def _model_source(model: ModelConfig, requested: str) -> SourceMethod:
    selected = (
        "DeepLift"
        if requested == "auto" and model.architecture == "cnn"
        else "GradientAttentionRollout"
        if requested == "auto"
        else requested
    )
    expected = "DeepLift" if model.architecture == "cnn" else "GradientAttentionRollout"
    if attack_source_semantic_name(selected) != expected:
        raise ValueError(
            f"architecture={model.architecture} requires attack source {expected}, found {selected}"
        )
    return selected  # type: ignore[return-value]


def _selected_positions(count: int, required: int) -> np.ndarray:
    if required <= 0 or required > count:
        raise ValueError(f"Requested {required} pilot samples from a split of size {count}")
    if required == 1:
        return np.asarray([0], dtype=np.int64)
    return np.linspace(0, count - 1, num=required, dtype=np.int64)


def _save_pilot_artifact(
    path: Path,
    *,
    result: AttackBatchResult,
    indices: Any,
    labels: Any,
    clean_maps: Any,
    adversarial_maps: Any,
    metadata: Mapping[str, str],
) -> None:
    import torch
    from safetensors.torch import save_file

    tensors = {
        "indices": indices.to(dtype=torch.int64).contiguous(),
        "labels": labels.to(dtype=torch.int64).contiguous(),
        "targets": result.targets.to(dtype=torch.int64).contiguous(),
        "best_steps": result.best_steps.to(dtype=torch.int64).contiguous(),
        "clean_logits": result.clean_logits.to(dtype=torch.float32).contiguous(),
        "adversarial_logits": result.adversarial_logits.to(dtype=torch.float32).contiguous(),
        "deltas": result.deltas.to(dtype=torch.float32).contiguous(),
        "clean_source_attributions": clean_maps.to(dtype=torch.float32).contiguous(),
        "adversarial_source_attributions": adversarial_maps.to(dtype=torch.float32).contiguous(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary), metadata=dict(metadata))
    os.replace(temporary, path)


def run_adversarial_pilot(
    experiment: SimpleExperiment,
    *,
    model_id: str,
    split: str = "test",
    source_method: str = "auto",
    batch_sizes: Sequence[int] = (1, 2, 4, 8),
    profile_steps: int = 3,
    analysis_samples: int = 8,
    attack_config: SaraAttackConfig | None = None,
    patch_sizes: Sequence[int] = (8, 14, 16),
    top_k: int = 20,
    device: str = "cuda:0",
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    """Profile attack batches, run a real attack slice, and persist its deltas."""

    import torch

    model_config = experiment.model(model_id)
    attack_config = attack_config or SaraAttackConfig()
    dataset_config = experiment.dataset(model_config.dataset_id)
    if split not in dataset_config.splits:
        raise ValueError(f"Split {split!r} is not configured for {dataset_config.dataset_id}")
    candidates = tuple(int(value) for value in batch_sizes)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("batch_sizes must be non-empty positive integers")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("batch_sizes must be unique and increasing")
    if profile_steps <= 0 or analysis_samples <= 0:
        raise ValueError("profile_steps and analysis_samples must be positive")
    if not patch_sizes or any(int(value) <= 0 for value in patch_sizes):
        raise ValueError("patch_sizes must be non-empty positive integers")

    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The adversarial performance pilot requires CUDA")
    selected_source = _model_source(model_config, source_method)
    loaded = load_model(model_config, device=target_device, include_checkpoint=True)
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    raw_model = _raw_input_model(loaded.model, loaded.preprocessing).to(target_device).eval()
    bundle = load_split(
        dataset_config,
        loaded,
        split=split,
        workers=experiment.runtime.dataloader_workers,
    )
    profile_positions = _selected_positions(len(bundle), max(candidates))
    profile_indices = bundle.indices[profile_positions]
    profile_images, _ = bundle.rows(profile_indices)
    profile_seeds = tuple(
        int(
            stable_seed(
                experiment.runtime.seed,
                "sara-adversarial",
                model_config.model_id,
                int(index),
            )
        )
        for index in profile_indices.tolist()
    )
    analysis_positions = _selected_positions(len(bundle), analysis_samples)
    analysis_indices = bundle.indices[analysis_positions]
    analysis_labels = bundle.labels[analysis_positions]
    analysis_images, _ = bundle.rows(analysis_indices)
    analysis_seeds = tuple(
        int(
            stable_seed(
                experiment.runtime.seed,
                "sara-adversarial",
                model_config.model_id,
                int(index),
            )
        )
        for index in analysis_indices.tolist()
    )

    profile_config = replace(attack_config, steps=profile_steps)
    measurements = []
    total_memory = int(torch.cuda.get_device_properties(target_device).total_memory)
    memory_limit = int(total_memory * (1.0 - experiment.runtime.headroom_fraction))
    for candidate in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
        started = time.perf_counter()
        try:
            batch_result = sara_attack_batch(
                raw_model,
                profile_images[:candidate].to(target_device, dtype=torch.float32),
                source_method=selected_source,
                config=profile_config,
                sample_seeds=profile_seeds[:candidate],
            )
            torch.cuda.synchronize(target_device)
            elapsed = time.perf_counter() - started
            peak_allocated = int(torch.cuda.max_memory_allocated(target_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(target_device))
            within_headroom = peak_reserved <= memory_limit
            summary = _attack_summary(batch_result, attack_config.epsilon)
            measurements.append(
                {
                    "batch_size": candidate,
                    "passed": True,
                    "within_headroom": within_headroom,
                    "profile_steps": profile_steps,
                    "elapsed_seconds": elapsed,
                    "samples_per_second": candidate / elapsed,
                    "sample_steps_per_second": candidate * profile_steps / elapsed,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_allocated_gib": peak_allocated / 2**30,
                    "peak_reserved_gib": peak_reserved / 2**30,
                    "attack": summary,
                    "reason": None,
                }
            )
            del batch_result
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            torch.cuda.synchronize(target_device)
            elapsed = time.perf_counter() - started
            measurements.append(
                {
                    "batch_size": candidate,
                    "passed": False,
                    "within_headroom": False,
                    "profile_steps": profile_steps,
                    "elapsed_seconds": elapsed,
                    "samples_per_second": 0.0,
                    "sample_steps_per_second": 0.0,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(target_device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(target_device) / 2**30,
                    "attack": None,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    eligible = [item for item in measurements if item["passed"] and item["within_headroom"]]
    if not eligible:
        raise RuntimeError("No attack batch size passed with configured GPU headroom")
    selected_measurement = max(
        eligible,
        key=lambda item: float(item["sample_steps_per_second"]),
    )
    selected_batch_size = int(selected_measurement["batch_size"])

    analysis_results = []
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(target_device)
    analysis_started = time.perf_counter()
    for start in range(0, analysis_samples, selected_batch_size):
        stop = min(analysis_samples, start + selected_batch_size)
        result = sara_attack_batch(
            raw_model,
            analysis_images[start:stop].to(target_device, dtype=torch.float32),
            source_method=selected_source,
            config=attack_config,
            sample_seeds=analysis_seeds[start:stop],
        )
        analysis_results.append(_to_cpu(result))
        del result
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize(target_device)
    analysis_elapsed = time.perf_counter() - analysis_started
    analysis_peak_allocated = int(torch.cuda.max_memory_allocated(target_device))
    analysis_peak_reserved = int(torch.cuda.max_memory_reserved(target_device))
    combined = _concatenate(analysis_results)
    height, width = (int(combined.clean_images.shape[-2]), int(combined.clean_images.shape[-1]))
    clean_maps = _resize_maps(combined.clean_attributions, height, width)
    adversarial_maps = _resize_maps(combined.adversarial_attributions, height, width)
    random_maps = _resize_maps(combined.random_attributions, height, width)
    ranks = rank_change_statistics(
        clean_maps.numpy(),
        adversarial_maps.numpy(),
        patch_sizes=patch_sizes,
        top_k=top_k,
    )
    random_ranks = rank_change_statistics(
        clean_maps.numpy(),
        random_maps.numpy(),
        patch_sizes=patch_sizes,
        top_k=top_k,
    )
    summary = _attack_summary(combined, attack_config.epsilon)
    random_prediction_preserved = float(
        combined.random_logits.argmax(dim=1).eq(combined.targets).float().mean()
    )
    primary_rank = ranks.get("p16")
    rank_effective = bool(
        primary_rank is not None and float(primary_rank["top_k_changed_fraction"]) > 0.0
    )

    identity = {
        "schema": "simple-sara-adversarial-pilot-v2",
        "experiment_id": experiment.experiment_id,
        "dataset_id": dataset_config.dataset_id,
        "dataset_manifest_sha256": dataset_manifest_identity_sha256(dataset_config.manifest_path),
        "model_id": model_config.model_id,
        "model_key": model_config.model_key,
        "checkpoint_sha256": (
            None
            if model_config.checkpoint_path is None
            else file_sha256(model_config.checkpoint_path)
        ),
        "source_method": selected_source,
        "source_method_semantics": attack_source_semantic_name(selected_source),
        "split": split,
        "attack": asdict(attack_config),
        "patch_sizes": [int(value) for value in patch_sizes],
        "top_k": top_k,
        "sample_indices": [int(value) for value in analysis_indices],
    }
    identity_digest = object_sha256(identity)
    destination = (
        Path(output_directory)
        if output_directory is not None
        else experiment.runtime.profile_directory.parent
        / "pilots"
        / "adversarial"
        / model_config.model_id
        / identity_digest[:16]
    )
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "attack-deltas.safetensors"
    report_path = destination / "report.json"
    if artifact_path.exists() or report_path.exists():
        raise FileExistsError(f"Pilot output already exists below {destination}")
    _save_pilot_artifact(
        artifact_path,
        result=combined,
        indices=analysis_indices,
        labels=analysis_labels,
        clean_maps=clean_maps,
        adversarial_maps=adversarial_maps,
        metadata={
            "schema_version": "2",
            "identity_digest": identity_digest,
            "source_method": selected_source,
            "source_method_semantics": attack_source_semantic_name(selected_source),
            "delta_space": "raw_pixels",
            "reconstruction": "clamp(clean_image + deltas, 0, 1)",
        },
    )
    report = {
        "schema_version": 2,
        "status": "completed",
        "feasible": bool(
            summary["prediction_preserved_fraction"] == 1.0
            and summary["linf_max"] <= attack_config.epsilon + 1e-6
            and rank_effective
        ),
        "identity": identity,
        "identity_digest": identity_digest,
        "created_utc": datetime.now(UTC).isoformat(),
        "device": {
            "requested": device,
            "name": torch.cuda.get_device_name(target_device),
            "total_bytes": total_memory,
            "headroom_fraction": experiment.runtime.headroom_fraction,
        },
        "batch_profile": {
            "selected_batch_size": selected_batch_size,
            "selection_policy": (
                "maximum_profile_sample_steps_per_second_with_configured_headroom"
            ),
            "profile_steps": profile_steps,
            "measurements": measurements,
        },
        "analysis": {
            **summary,
            "steps": attack_config.steps,
            "elapsed_seconds": analysis_elapsed,
            "samples_per_second": analysis_samples / analysis_elapsed,
            "sample_steps_per_second": analysis_samples * attack_config.steps / analysis_elapsed,
            "peak_allocated_bytes": analysis_peak_allocated,
            "peak_reserved_bytes": analysis_peak_reserved,
            "peak_allocated_gib": analysis_peak_allocated / 2**30,
            "peak_reserved_gib": analysis_peak_reserved / 2**30,
            "native_source_map_shape": list(combined.clean_attributions.shape[1:]),
            "rank_effective_at_p16": rank_effective,
            "rank_change": ranks,
            "random_start_control": {
                "prediction_preserved_fraction": random_prediction_preserved,
                "objective_mean": float(combined.random_objective.mean()),
                "rank_change": random_ranks,
            },
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": file_sha256(artifact_path),
            "delta_dtype": "float32",
            "delta_shape": list(combined.deltas.shape),
        },
    }
    atomic_write_json(report_path, report)
    print(
        f"ADVERSARIAL_PILOT_COMPLETE model={model_config.model_id} "
        f"source={selected_source} batch={selected_batch_size} "
        f"rank_effective_p16={rank_effective} report={report_path}",
        flush=True,
    )

    del combined, raw_model, loaded, bundle
    gc.collect()
    torch.cuda.empty_cache()
    return report


__all__ = [
    "AttackBatchResult",
    "LEGACY_GRADIENT_ATTENTION_ROLLOUT_ID",
    "SaraAttackConfig",
    "attack_source_semantic_name",
    "rank_change_statistics",
    "run_adversarial_pilot",
    "sara_attack_batch",
    "sara_postprocess_attribution",
]
