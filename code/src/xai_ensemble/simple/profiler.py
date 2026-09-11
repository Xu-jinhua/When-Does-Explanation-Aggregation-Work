"""One-time FP32 batch-size search per model architecture and final method."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase1.explainers import (
    EXPLAINER_SPECS,
    build_explainer,
    compute_attribution,
)
from xai_ensemble.phase1.relprop import (
    relprop_attribution_provider,
    relprop_required,
)
from xai_ensemble.phase2.evaluator import (
    FillReference,
    evaluate_reference_model_bank,
)

from .config import ModelConfig, Phase2ProfileClass, ProfileClass, SimpleExperiment
from .data import load_model, load_relprop_model
from .relprop_equivalence import ensure_relprop_equivalence_certificate

SEMANTIC_PARAM_KEYS = frozenset(
    {"baseline", "baseline_distribution", "baseline_space", "gaussian_space"}
)


def captum_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key not in SEMANTIC_PARAM_KEYS}


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    batch_size: int
    passed: bool
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    elapsed_seconds: float
    reason: str | None


@dataclass(frozen=True, slots=True)
class BatchProfile:
    schema_version: int
    profile_id: str
    identity_digest: str
    model_key: str
    architecture: str
    method: str
    variant: str
    params: Mapping[str, Any]
    precision: str
    input_shape: tuple[int, int, int]
    selected_batch_size: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    device_total_bytes: int
    headroom_fraction: float
    probe_kind: str
    measurements: tuple[CandidateMeasurement, ...]
    created_utc: str

    @property
    def reservation_bytes(self) -> int:
        return max(self.peak_allocated_bytes, self.peak_reserved_bytes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_path(experiment: SimpleExperiment, profile_id: str) -> Path:
    return experiment.runtime.profile_directory / f"{profile_id}.json"


def profile_identity(profile: ProfileClass) -> Mapping[str, Any]:
    value = {
        "schema": "simple-profile-v1",
        "model_architecture": profile.model_key,
        "architecture_family": profile.architecture,
        "method": profile.method.family,
        "variant": profile.method.variant,
        "params": dict(profile.method.params),
        "precision": "fp32",
        "input_shape": [3, profile.input_size, profile.input_size],
    }
    if profile.model_id is not None:
        value["model_id"] = profile.model_id
        value["attribution_provider"] = relprop_attribution_provider(
            profile.method.family,
            profile.architecture,
        )
    return value


def phase2_profile_identity(profile: Phase2ProfileClass) -> Mapping[str, Any]:
    return {
        "schema": "simple-phase2-inference-profile-v1",
        "model_key": profile.model_key,
        "num_classes": profile.num_classes,
        "architecture": profile.architecture,
        "precision": "fp32",
        "input_shape": [3, profile.input_size, profile.input_size],
        "forward_batch_size": profile.inference_batch_size,
        "masked_variants": ["removed", "retained"],
    }


def _load_batch_profile(
    path: Path,
    *,
    profile_id: str,
    identity_digest: str,
) -> BatchProfile | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("profile_id") != profile_id or value.get("identity_digest") != identity_digest:
        raise ValueError(f"Stale or contradictory batch profile: {path}")
    measurements = tuple(CandidateMeasurement(**item) for item in value["measurements"])
    return BatchProfile(
        schema_version=int(value["schema_version"]),
        profile_id=str(value["profile_id"]),
        identity_digest=str(value["identity_digest"]),
        model_key=str(value["model_key"]),
        architecture=str(value["architecture"]),
        method=str(value["method"]),
        variant=str(value["variant"]),
        params=dict(value["params"]),
        precision=str(value["precision"]),
        input_shape=tuple(int(item) for item in value["input_shape"]),  # type: ignore[arg-type]
        selected_batch_size=int(value["selected_batch_size"]),
        peak_allocated_bytes=int(value["peak_allocated_bytes"]),
        peak_reserved_bytes=int(value["peak_reserved_bytes"]),
        device_total_bytes=int(value["device_total_bytes"]),
        headroom_fraction=float(value["headroom_fraction"]),
        probe_kind=str(value["probe_kind"]),
        measurements=measurements,
        created_utc=str(value["created_utc"]),
    )


def load_profile(experiment: SimpleExperiment, profile: ProfileClass) -> BatchProfile | None:
    return _load_batch_profile(
        profile_path(experiment, profile.profile_id),
        profile_id=profile.profile_id,
        identity_digest=object_sha256(profile_identity(profile)),
    )


def load_phase2_profile(
    experiment: SimpleExperiment, profile: Phase2ProfileClass
) -> BatchProfile | None:
    return _load_batch_profile(
        profile_path(experiment, profile.profile_id),
        profile_id=profile.profile_id,
        identity_digest=object_sha256(phase2_profile_identity(profile)),
    )


def _resize(values: Any, height: int, width: int) -> Any:
    if values.ndim != 4:
        raise ValueError(f"Explainer returned non-BCHW attribution shape {tuple(values.shape)}")
    if tuple(values.shape[-2:]) == (height, width):
        return values
    import torch.nn.functional as functional

    return functional.interpolate(values, (height, width), mode="bilinear", align_corners=False)


def _perturbation_probe(
    model: Any,
    images: Any,
    targets: Any,
    *,
    patch_size: int,
    perturbations: int,
    baseline: Any,
) -> Any:
    """Exercise a few full-batch perturbations with the real output footprint.

    FeatureAblation and Occlusion differ mainly in the number of sequential
    perturbations when Captum's perturbations_per_eval remains one.  Running
    all 196--784 windows during every grid candidate would measure time, not a
    different memory peak.  This probe keeps the real BxCxHxW tensors and full
    model forward while limiting only that sequential loop count.
    """

    import torch

    result = torch.zeros_like(images)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    base = baseline.expand(1, -1, height, width)
    with torch.inference_mode():
        reference = model(images).gather(1, targets[:, None]).squeeze(1)
        grid_width = width // patch_size
        total = (height // patch_size) * grid_width
        for feature in range(min(perturbations, total)):
            row, column = divmod(feature, grid_width)
            top, left = row * patch_size, column * patch_size
            perturbed = images.clone()
            perturbed[:, :, top : top + patch_size, left : left + patch_size] = base[
                :, :, top : top + patch_size, left : left + patch_size
            ]
            score = reference - model(perturbed).gather(1, targets[:, None]).squeeze(1)
            result[:, :, top : top + patch_size, left : left + patch_size] = score[
                :, None, None, None
            ]
    return result


def _candidate_order(grid: tuple[int, ...], start: int) -> tuple[int, ...]:
    index = grid.index(start)
    return (start, *grid[index + 1 :], *reversed(grid[:index]))


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error: memory allocation" in text


def run_profile(
    experiment: SimpleExperiment,
    profile: ProfileClass,
    *,
    device: str = "cuda:0",
) -> BatchProfile | None:
    """Search the locked grid once and persist the selected real batch size.

    Returns ``None`` when no CUDA device is available; profiling is a pure
    resource measurement, so callers then use the configured fixed batch size.
    """

    import torch

    existing = load_profile(experiment, profile)
    if existing is not None:
        return existing
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        # Profiling is a resource measurement only and never enters the task
        # digest; without CUDA there is nothing to measure.  Callers fall
        # back to the configured fixed batch size.
        print(
            f"WARNING batch profile {profile.profile_id} skipped: no CUDA device available",
            flush=True,
        )
        return None
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    uses_relprop = relprop_required(profile.method.family, profile.architecture)
    if uses_relprop:
        if profile.model_id is None:
            raise RuntimeError("RelProp profile is missing its checkpoint-specific model id")
        model_config = experiment.model(profile.model_id)
        reference = load_model(model_config, device=target_device, include_checkpoint=True)
        loaded = load_relprop_model(
            model_config,
            method=profile.method.family,
            device=target_device,
        )
        if model_config.checkpoint_path is None:
            raise RuntimeError("RelProp profile requires a checkpoint-backed model")
        ensure_relprop_equivalence_certificate(
            experiment,
            model_config,
            method=profile.method.family,
            reference_model=reference.model,
            relprop_model=loaded.model,
            preprocessing=reference.preprocessing,
            checkpoint_sha256=file_sha256(
                resolve_full_matrix_runtime_path(model_config.checkpoint_path)
            ),
            device=target_device,
        )
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model_config = ModelConfig(
            model_id=f"profile-{profile.model_key}",
            dataset_id="profile",
            model_key=profile.model_key,
            num_classes=profile.num_classes,
            init_mode="random",
            checkpoint_path=None,
            strict_checkpoint=True,
            class_index_map=None,
            mean_path=Path("/profile/unused-dataset-mean"),
            mean_key="dataset_mean",
            architecture=profile.architecture,
        )
        reference = None
        loaded = load_model(model_config, device=target_device, include_checkpoint=False)
    model = loaded.model
    # Production workers compute target predictions on the unwrapped reference
    # model.  RelProp forward hooks stash activations even under no_grad, so
    # probing targets through the wrapped model inflates the measured peak
    # several-fold (vit FullLRP at batch 128: 5.6 -> 28.6 GiB observed).
    target_model = reference.model if reference is not None else model
    total_bytes = int(torch.cuda.get_device_properties(target_device).total_memory)
    maximum = max(experiment.runtime.search_grid)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(experiment.runtime.seed)
    # The agreed probe bank is allocated once in CPU RAM and sliced for each candidate.
    random_bank = torch.randn(
        (maximum, 3, profile.input_size, profile.input_size),
        generator=generator,
        dtype=torch.float32,
    )
    raw_mean = torch.full((1, 3, profile.input_size, profile.input_size), 0.5, dtype=torch.float32)
    mean = loaded.normalize(raw_mean.to(target_device))
    zero = torch.zeros_like(mean)
    gaussian_generator = torch.Generator(device="cpu")
    gaussian_generator.manual_seed(experiment.runtime.seed)
    gaussian = torch.randn(mean.shape, generator=gaussian_generator).to(target_device)
    distribution = torch.cat((zero, gaussian, mean), dim=0)
    start = profile.method.profile_start or experiment.runtime.default_profile_start
    if start not in experiment.runtime.search_grid:
        raise ValueError(f"Profile start {start} is absent from the locked search grid")
    measurements: list[CandidateMeasurement] = []
    successful: dict[int, CandidateMeasurement] = {}
    initial_failed = False
    probe_kind = (
        "limited_sequential_perturbations"
        if profile.method.family in {"FeatureAblation", "Occlusion"}
        else "full_attribution"
    )

    for batch_size in _candidate_order(experiment.runtime.search_grid, start):
        # Once upward search fails, only the lower side of the start remains relevant.
        if (
            successful
            and batch_size > start
            and any(item.batch_size > batch_size and not item.passed for item in measurements)
        ):
            continue
        if initial_failed and batch_size > start:
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
        free_before, _ = torch.cuda.mem_get_info(target_device)
        allocated_before = int(torch.cuda.memory_allocated(target_device))
        nonprocess_bytes = max(0, total_bytes - int(free_before) - allocated_before)
        started = time.monotonic()
        output = images = targets = explainer = None
        reason = None
        passed = False
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        try:
            images = random_bank[:batch_size].to(target_device, non_blocking=True)
            images.requires_grad_(EXPLAINER_SPECS[profile.method.family].differentiable_input)
            with torch.no_grad():
                targets = target_model(images).argmax(dim=1)
            if profile.method.family in {"FeatureAblation", "Occlusion"}:
                output = _perturbation_probe(
                    model,
                    images,
                    targets,
                    patch_size=int(profile.method.params["patch_size"]),
                    perturbations=profile.method.probe_perturbations,
                    baseline=zero if profile.method.family == "FeatureAblation" else mean,
                )
            else:
                explainer = build_explainer(
                    model,
                    profile.method.family,
                    architecture=profile.architecture,
                )
                output = compute_attribution(
                    explainer,
                    profile.method.family,
                    images,
                    targets,
                    params=captum_params(profile.method.params),
                    baseline=(
                        zero
                        if profile.method.params.get("baseline") == "zero"
                        else mean
                        if profile.method.params.get("baseline") == "dataset_mean"
                        else None
                    ),
                    baseline_distribution=(
                        distribution if profile.method.params.get("baseline_distribution") else None
                    ),
                )
                output = _resize(output, profile.input_size, profile.input_size)
            if tuple(output.shape[0:1]) != (batch_size,) or not bool(torch.isfinite(output).all()):
                raise RuntimeError("Profiler attribution output is invalid")
            torch.cuda.synchronize(target_device)
            peak_allocated = int(torch.cuda.max_memory_allocated(target_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(target_device))
            limit = int(total_bytes * (1.0 - experiment.runtime.headroom_fraction))
            estimated_global_peak = nonprocess_bytes + max(peak_allocated, peak_reserved)
            if estimated_global_peak > limit:
                reason = "headroom_limit"
            else:
                passed = True
        except BaseException as error:
            if not _is_oom(error):
                raise
            reason = "cuda_oom"
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        measurement = CandidateMeasurement(
            batch_size=batch_size,
            passed=passed,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            elapsed_seconds=elapsed,
            reason=reason,
        )
        measurements.append(measurement)
        stop_search = False
        if passed:
            successful[batch_size] = measurement
            if batch_size == max(experiment.runtime.search_grid):
                stop_search = True
            elif initial_failed and batch_size < start:
                # Descending search found the largest candidate below the failed start.
                stop_search = True
        elif batch_size == start:
            initial_failed = True
        elif batch_size > start and successful:
            # The grid is monotonic for these fixed-shape calls; stop upward search.
            stop_search = True
        del output, images, targets, explainer
        gc.collect()
        torch.cuda.empty_cache()
        if stop_search:
            break

    if not successful:
        raise RuntimeError(f"No batch size fits profile class {profile.profile_id}")
    selected_size = max(successful)
    selected = successful[selected_size]
    assert selected.peak_allocated_bytes is not None
    assert selected.peak_reserved_bytes is not None
    result = BatchProfile(
        schema_version=1,
        profile_id=profile.profile_id,
        identity_digest=object_sha256(profile_identity(profile)),
        model_key=profile.model_key,
        architecture=profile.architecture,
        method=profile.method.family,
        variant=profile.method.variant,
        params=dict(profile.method.params),
        precision="fp32",
        input_shape=(3, profile.input_size, profile.input_size),
        selected_batch_size=selected_size,
        peak_allocated_bytes=selected.peak_allocated_bytes,
        peak_reserved_bytes=selected.peak_reserved_bytes,
        device_total_bytes=total_bytes,
        headroom_fraction=experiment.runtime.headroom_fraction,
        probe_kind=probe_kind,
        measurements=tuple(measurements),
        created_utc=datetime.now(UTC).isoformat(),
    )
    path = profile_path(experiment, profile.profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result.to_dict())
    del random_bank, distribution, gaussian, zero, mean, model, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase2_profile(
    experiment: SimpleExperiment,
    profile: Phase2ProfileClass,
    *,
    device: str = "cuda:0",
) -> BatchProfile | None:
    """Measure the exact FP32 removed/retained model-forward memory once.

    Returns ``None`` when no CUDA device is available; profiling is a pure
    resource measurement, so callers then use the configured fixed batch size.
    """

    import torch

    existing = load_phase2_profile(experiment, profile)
    if existing is not None:
        return existing
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        print(
            f"WARNING phase2 profile {profile.profile_id} skipped: no CUDA device available",
            flush=True,
        )
        return None
    model_config = ModelConfig(
        model_id=f"phase2-profile-{profile.model_key}",
        dataset_id="profile",
        model_key=profile.model_key,
        num_classes=profile.num_classes,
        init_mode="random",
        checkpoint_path=None,
        strict_checkpoint=True,
        class_index_map=None,
        mean_path=Path("/profile/unused-dataset-mean"),
        mean_key="dataset_mean",
        architecture=profile.architecture,
    )
    loaded = load_model(model_config, device=target_device, include_checkpoint=False)
    total_bytes = int(torch.cuda.get_device_properties(target_device).total_memory)
    rule_count = min(16, max(1, profile.inference_batch_size // 2))
    source_count = max(1, profile.inference_batch_size // (2 * rule_count))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(experiment.runtime.seed)
    images = torch.rand(
        (source_count, 3, profile.input_size, profile.input_size),
        generator=generator,
        dtype=torch.float32,
    )
    grid_size = profile.input_size // experiment.phase2.primary_patch_size
    patch_count = grid_size * grid_size
    ranks = np.broadcast_to(
        np.arange(patch_count, dtype=np.int64), (source_count, patch_count)
    ).copy()
    labels = np.zeros(source_count, dtype=np.int64)
    fill = FillReference(
        values=np.full((3, profile.input_size, profile.input_size), 0.5, dtype=np.float32),
        source_split="train",
        artifact_id="phase2-profile-fill",
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(target_device)
    free_before, _ = torch.cuda.mem_get_info(target_device)
    allocated_before = int(torch.cuda.memory_allocated(target_device))
    nonprocess_bytes = max(0, total_bytes - int(free_before) - allocated_before)
    started = time.monotonic()
    try:
        traces = evaluate_reference_model_bank(
            loaded.model,
            images,
            {f"rule-{index:02d}": ranks for index in range(rule_count)},
            true_labels=labels,
            target_labels=labels,
            fill_reference=fill,
            reference_model_id=f"profile-{profile.model_key}",
            patch_size=experiment.phase2.primary_patch_size,
            k=experiment.phase2.k,
            batch_size=profile.inference_batch_size,
            device=str(target_device),
            autocast=False,
            normalize=loaded.normalize,
            clean_predictions=labels,
        )
        if len(traces) != rule_count or any(
            trace.stats.n_samples != source_count for trace in traces.values()
        ):
            raise RuntimeError("Phase 2 profiler returned an incomplete trace")
        torch.cuda.synchronize(target_device)
        peak_allocated = int(torch.cuda.max_memory_allocated(target_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(target_device))
    except BaseException as error:
        if _is_oom(error):
            raise RuntimeError(
                f"Phase 2 forward batch {profile.inference_batch_size} does not fit "
                f"profile {profile.profile_id}"
            ) from error
        raise
    elapsed = time.monotonic() - started
    limit = int(total_bytes * (1.0 - experiment.runtime.headroom_fraction))
    if nonprocess_bytes + max(peak_allocated, peak_reserved) > limit:
        raise RuntimeError(f"Phase 2 profile {profile.profile_id} violates configured GPU headroom")
    measurement = CandidateMeasurement(
        batch_size=profile.inference_batch_size,
        passed=True,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        elapsed_seconds=elapsed,
        reason=None,
    )
    result = BatchProfile(
        schema_version=1,
        profile_id=profile.profile_id,
        identity_digest=object_sha256(phase2_profile_identity(profile)),
        model_key=profile.model_key,
        architecture=profile.architecture,
        method="Phase2MaskGame",
        variant="removed_retained",
        params={
            "forward_batch_size": profile.inference_batch_size,
            "patch_size": experiment.phase2.primary_patch_size,
            "k": experiment.phase2.k,
            "rule_bank_size": rule_count,
        },
        precision="fp32",
        input_shape=(3, profile.input_size, profile.input_size),
        selected_batch_size=profile.inference_batch_size,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        device_total_bytes=total_bytes,
        headroom_fraction=experiment.runtime.headroom_fraction,
        probe_kind="phase2_rule_bank_removed_retained_inference",
        measurements=(measurement,),
        created_utc=datetime.now(UTC).isoformat(),
    )
    path = profile_path(experiment, profile.profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result.to_dict())
    del traces, ranks, labels, images, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def missing_profiles(experiment: SimpleExperiment) -> tuple[ProfileClass, ...]:
    return tuple(
        profile for profile in experiment.profiles() if load_profile(experiment, profile) is None
    )


def missing_phase2_profiles(
    experiment: SimpleExperiment,
) -> tuple[Phase2ProfileClass, ...]:
    return tuple(
        profile
        for profile in experiment.phase2_profiles()
        if load_phase2_profile(experiment, profile) is None
    )


__all__ = [
    "BatchProfile",
    "CandidateMeasurement",
    "captum_params",
    "load_phase2_profile",
    "load_profile",
    "missing_phase2_profiles",
    "missing_profiles",
    "phase2_profile_identity",
    "profile_identity",
    "profile_path",
    "run_phase2_profile",
    "run_profile",
]
