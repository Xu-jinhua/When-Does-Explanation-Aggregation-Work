"""Executable explainer-compatibility and per-method batch-size pilot."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import read_json
from xai_ensemble.core.manifest import load_manifest, validate_manifest_files
from xai_ensemble.core.protocol import load_protocol
from xai_ensemble.core.provenance import collect_provenance
from xai_ensemble.data.manifest import read_manifest
from xai_ensemble.data.specs import get_dataset_spec

from ._compat_support import (
    ATTACK_BATCH_SIZE,
    ModelArtifactSpec,
    PilotResult,
    _build_model,
    _load_distribution,
    _normalizer,
    _raw_eval_transform,
    _resize_to_input,
    _special_attack_attribution,
    attribution_to_patch_scores,
    save_pilot_result,
    scores_to_ranks,
)
from .explainers import EXPLAINER_SPECS, build_explainer, compute_attribution
from .method_lock import LockedMethod, MethodLock, save_method_lock
from .relprop import (
    RELPROP_REVISION,
    RELPROP_SOURCE_DIGEST,
    create_relprop_model_for_method,
    relprop_implementation,
    relprop_required,
    verify_relprop_equivalence,
)

# Captum's CNN-LRP backward pass uses cuBLAS reductions.  Set this before the
# first optional Torch import so the per-method deterministic gate can enable
# deterministic algorithms on CUDA without a late cuBLAS configuration error.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MINIMUM_CUDA_HEADROOM_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class CompatibilityMeasurement:
    method: str
    passed: bool
    selected_batch_size: int | None
    batch_candidates: tuple[dict[str, Any], ...]
    deterministic_repeat: bool | None
    target_sensitive: bool | None
    target_sensitivity_required: bool
    attack_batch_size: int | None
    attack_seconds_per_sample: float | None
    attack_peak_cuda_bytes: int | None
    output_shape: tuple[int, ...] | None
    finite_fraction: float | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CompatibilityPilotOutput:
    result_path: Path
    batch_result_path: Path
    method_lock_path: Path
    result: PilotResult
    batch_result: PilotResult
    method_lock: MethodLock


def explanation_batch_size_pilot_result(
    compatibility_result: PilotResult,
    *,
    model_id: str,
    dataloader_workers: int,
) -> PilotResult:
    """Bridge compatibility measurements to the canonical batch-size lock.

    Explanation memory is method-specific, while the compatibility pilot also
    owns the real explainer instances and their fixed parameters.  This bridge
    keeps that single measurement source but emits an independent
    ``batch_size`` result that can be frozen at ``stage=explanation`` without
    hand-authoring a lock.
    """

    if dataloader_workers < 0:
        raise ValueError("dataloader_workers cannot be negative")
    if compatibility_result.name != "explainer_compatibility":
        raise ValueError("explanation batch bridge requires explainer_compatibility")
    by_model = compatibility_result.selected.get("method_batch_size_by_model")
    raw_sizes = by_model.get(model_id) if isinstance(by_model, Mapping) else None
    method_sizes = (
        {str(method): int(size) for method, size in raw_sizes.items()}
        if isinstance(raw_sizes, Mapping)
        else {}
    )
    if compatibility_result.passed and (
        not method_sizes or any(size <= 0 for size in method_sizes.values())
    ):
        raise ValueError(
            f"passed compatibility result has no positive method batches for {model_id}"
        )
    attack_by_model = compatibility_result.selected.get("attack_batch_size_by_model")
    raw_attack_sizes = (
        attack_by_model.get(model_id) if isinstance(attack_by_model, Mapping) else None
    )
    measured_attack_sizes = (
        {str(method): int(size) for method, size in raw_attack_sizes.items()}
        if isinstance(raw_attack_sizes, Mapping)
        else {}
    )
    if any(size != ATTACK_BATCH_SIZE for size in measured_attack_sizes.values()):
        raise ValueError("rank-PGD compatibility measurements must use batch_size=1")
    status = "passed" if compatibility_result.passed else "failed"
    source_failures = list(compatibility_result.failures)
    if not compatibility_result.passed and not source_failures:
        source_failures.append("Explainer compatibility pilot did not pass")
    return PilotResult(
        name="batch_size",
        status=status,
        selected=(
            {
                "model": model_id,
                "stage": "explanation",
                "batch_size": min(method_sizes.values()),
                "method_batch_size_by_model": {model_id: method_sizes},
                "dataloader_workers": dataloader_workers,
                "attack_batch_size_algorithm_constraint": ATTACK_BATCH_SIZE,
            }
            if compatibility_result.passed
            else {}
        ),
        measurements={
            "source_pilot": compatibility_result.name,
            "source_pilot_digest": compatibility_result.digest,
            "measured_attack_batch_size_by_method": measured_attack_sizes,
            "attack_batching": "per_sample_independent_restart_optimization",
        },
        failures=source_failures,
        config_digest=object_sha256(
            {
                "source_pilot_digest": compatibility_result.digest,
                "source_config_digest": compatibility_result.config_digest,
                "model": model_id,
                "stage": "explanation",
                "dataloader_workers": dataloader_workers,
                "attack_batch_size_algorithm_constraint": ATTACK_BATCH_SIZE,
            }
        ),
    )


def _sample_ids(path: Path) -> tuple[str, ...]:
    value = read_json(path)
    raw = value.get("sample_ids") if isinstance(value, Mapping) else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Pilot sample file must contain sample_ids")
    identifiers = tuple(str(item) for item in raw)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Pilot sample IDs must be non-empty and unique")
    return identifiers


def _sync(device: Any) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _seed_attribution(seed: int, device: Any) -> None:
    """Bind every RNG used by Captum and the historical explainers.

    Captum's GradientShap samples interpolation coefficients with NumPy while
    its noise tunnel uses Torch, so resetting Torch alone does not make a
    repeated pilot attribution reproducible.  The old implementation also
    used Python/NumPy-generated baseline draws; bind all three streams to the
    recorded pilot seed.
    """

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _cuda_headroom_passed(
    peak_cuda_bytes: int | None,
    total_device_bytes: int | None,
    *,
    minimum_fraction: float = MINIMUM_CUDA_HEADROOM_FRACTION,
) -> bool:
    if peak_cuda_bytes is None or total_device_bytes is None:
        return True
    if total_device_bytes <= 0 or not 0 <= minimum_fraction < 1:
        raise ValueError("Invalid CUDA memory/headroom values")
    return peak_cuda_bytes <= int(total_device_bytes * (1.0 - minimum_fraction))


def _strict_ranks(ranks: Any) -> bool:
    values = ranks.detach().cpu().numpy() if hasattr(ranks, "detach") else np.asarray(ranks)
    flat = values.reshape(values.shape[0], -1)
    expected = np.arange(1, flat.shape[1] + 1)
    return all(np.array_equal(np.sort(row), expected) for row in flat)


def _attribution(
    *,
    explainer: Any,
    method: LockedMethod,
    images: Any,
    targets: Any,
    baseline: Any,
    distribution: Any,
    patch_size: int,
) -> tuple[Any, Any, Any]:
    attributions = compute_attribution(
        explainer,
        method.family,
        images,
        targets,
        params=method.params,
        baseline=baseline,
        baseline_distribution=distribution,
    )
    attributions = _resize_to_input(attributions, images.shape[-2], images.shape[-1])
    scores = attribution_to_patch_scores(attributions, patch_size)
    ranks = scores_to_ranks(scores, base=1)
    return attributions, scores, ranks


def _method_baseline(method: LockedMethod, zero: Any, mean: Any) -> Any | None:
    name = method.params.get("baseline")
    if name == "zero":
        return zero
    if name == "dataset_mean":
        return mean
    return None


def _candidate_methods(
    protocol: Mapping[str, Any],
    model: str,
    architecture: str,
    *,
    explicit: Sequence[LockedMethod] | None = None,
) -> tuple[LockedMethod, ...]:
    if explicit is not None:
        methods = tuple(explicit)
        if not methods:
            raise ValueError("Explicit compatibility candidates cannot be empty")
        if any(method.architecture != architecture for method in methods):
            raise ValueError("Explicit compatibility candidates have the wrong architecture")
        families = [method.family for method in methods]
        if len(families) != len(set(families)):
            raise ValueError("Explicit compatibility candidates must have unique families")
        return methods
    candidates = protocol["explainers"][f"{architecture}_candidates"]
    defaults = protocol["explainers"].get("locked_defaults", {})
    methods = []
    for family in candidates:
        params = dict(defaults.get(family, {}))
        methods.append(
            LockedMethod(
                family=str(family),
                instance_id=f"{family}-{object_sha256(params)[:12]}",
                params=params,
                architecture=architecture,
            )
        )
    return tuple(methods)


def _compatibility_gate_failures(
    protocol: Mapping[str, Any],
    *,
    architecture: str,
    measurements: Sequence[CompatibilityMeasurement],
) -> list[str]:
    """Evaluate the mandatory subset while allowing candidate filtering."""

    configuration = protocol["explainers"].get("compatibility_gate", {})
    required_by_architecture = configuration.get("required_by_architecture", {})
    required = {str(value) for value in required_by_architecture.get(architecture, ())}
    minimum_by_architecture = configuration.get("minimum_generic_compatible_by_architecture", {})
    minimum_generic = int(minimum_by_architecture.get(architecture, 1))
    if minimum_generic < 0:
        return ["minimum_generic_compatible must be non-negative"]
    by_method = {item.method: item for item in measurements}
    failures = []
    absent = sorted(required - set(by_method))
    if absent:
        failures.append(f"required methods are absent from the candidate roster: {absent}")
    failed_required = sorted(
        method for method in required if method in by_method and not by_method[method].passed
    )
    if failed_required:
        failures.append(f"required methods failed compatibility: {failed_required}")
    compatible_generic = [
        item.method
        for item in measurements
        if item.passed
        and not EXPLAINER_SPECS[item.method].dedicated_vit
        and not relprop_required(item.method, architecture)
    ]
    if len(compatible_generic) < minimum_generic:
        failures.append(
            "insufficient compatible generic methods: "
            f"observed={len(compatible_generic)}, required={minimum_generic}"
        )
    if not any(item.passed for item in measurements):
        failures.append("no explainer candidate passed compatibility")
    return failures


def _mean_and_baselines(
    mean_root: Path,
    *,
    protocol_digest: str,
    dataset_id: str,
    dataset_revision: str,
    manifest_fingerprint: str,
    model: str,
    preprocessing: Mapping[str, Any],
    normalize: Any,
    device: Any,
    seed: int,
) -> tuple[Any, Any, str]:
    manifest_path = mean_root / "manifest.json" if mean_root.is_dir() else mean_root
    manifest = load_manifest(manifest_path)
    validate_manifest_files(manifest_path.parent, manifest)
    checks = {
        "dataset_id": (manifest.metadata.get("dataset_id"), dataset_id),
        "dataset_revision": (
            manifest.metadata.get("dataset_revision"),
            dataset_revision,
        ),
        "dataset_manifest_fingerprint": (
            manifest.metadata.get("dataset_manifest_fingerprint"),
            manifest_fingerprint,
        ),
        "model_id": (manifest.metadata.get("model_id"), model),
        "protocol_digest": (
            manifest.metadata.get("protocol_digest"),
            protocol_digest,
        ),
    }
    mismatches = [name for name, (observed, expected) in checks.items() if observed != expected]
    if mismatches:
        raise ValueError(f"Mean artifact identity mismatch: {mismatches}")
    artifact_preprocessing = {
        key: value for key, value in manifest.metadata["preprocessing"].items()
    }
    if object_sha256(artifact_preprocessing) != object_sha256(dict(preprocessing)):
        raise ValueError("Mean artifact preprocessing differs from pilot model")
    entry = manifest.files[0]
    mean = _load_distribution(manifest_path.parent / entry.path, "dataset_mean")
    size = int(preprocessing["input_size"])
    if mean.shape != (3, size, size):
        raise ValueError("Pilot mean artifact must retain full [3,H,W] spatial shape")
    if not np.isfinite(mean).all() or bool(((mean < 0) | (mean > 1)).any()):
        raise ValueError("Pilot mean artifact is outside raw [0,1]")

    import torch

    mean_raw = torch.as_tensor(mean, dtype=torch.float32, device=device).unsqueeze(0)
    zero_raw = torch.zeros_like(mean_raw)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise_raw = torch.rand(
        mean_raw.shape,
        generator=generator,
        device=device,
        dtype=mean_raw.dtype,
    )
    return (
        normalize(mean_raw),
        normalize(torch.cat((zero_raw, noise_raw, mean_raw), dim=0)),
        manifest.artifact_id,
    )


def run_compatibility_pilot(
    *,
    protocol_path: str | os.PathLike[str],
    dataset_key: str,
    dataset_manifest_path: str | os.PathLike[str],
    sample_ids_path: str | os.PathLike[str],
    split: str,
    model_key: str,
    checkpoint_path: str | os.PathLike[str],
    means_artifact: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    batch_candidates: Sequence[int],
    source_id: str = "reference-full",
    model_factory: str | None = None,
    device_name: str = "auto",
    cache_dir: str | os.PathLike[str] | None = None,
    token_env: str | None = None,
    seed: int = 20260714,
    dataloader_workers: int = 8,
    project_root: str | os.PathLike[str] = ".",
    candidate_methods: Sequence[LockedMethod] | None = None,
    patch_size: int | None = None,
) -> CompatibilityPilotOutput:
    import torch

    from xai_ensemble.phase0.dataset import ManifestIndexedDataset, load_hf_split
    from xai_ensemble.phase0.models import get_model_definition, resolved_preprocessing

    if dataloader_workers < 0:
        raise ValueError("dataloader_workers cannot be negative")
    protocol = load_protocol(protocol_path)
    dataset_spec = get_dataset_spec(dataset_key)
    manifest = read_manifest(dataset_manifest_path)
    manifest.validate(dataset_spec, require_expected_counts=False)
    identifiers = _sample_ids(Path(sample_ids_path))
    if len(identifiers) < 2:
        raise ValueError("Compatibility pilot requires at least two samples")
    by_id = manifest.by_sample_id()
    records = tuple(by_id[sample_id] for sample_id in identifiers)
    if any(record.split != split for record in records):
        raise ValueError("Pilot sample IDs must all belong to --split")

    checkpoint = Path(checkpoint_path).resolve()
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    metadata = read_json(sidecar)["metadata"]
    checks = {
        "dataset_id": dataset_spec.dataset_id,
        "dataset_revision": dataset_spec.revision,
        "dataset_manifest_fingerprint": manifest.fingerprint,
        "model_key": model_key,
        "source_id": source_id,
        "protocol_digest": protocol.digest,
    }
    wrong = [name for name, expected in checks.items() if metadata.get(name) != expected]
    if wrong:
        raise ValueError(f"Compatibility checkpoint identity mismatch: {wrong}")
    model_spec = ModelArtifactSpec(
        model_key=model_key,
        num_classes=dataset_spec.num_classes,
        init_mode="checkpoint",
        source_model_id=source_id,
        checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_path=checkpoint,
        factory=model_factory,
    )
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = _build_model(model_spec).eval().to(device)
    definition = get_model_definition(model_key)
    preprocessing = resolved_preprocessing(model, definition)
    normalize = _normalizer(preprocessing["mean"], preprocessing["std"])
    token = None if token_env is None else os.environ.get(token_env)
    if token_env is not None and token is None:
        raise RuntimeError(f"Dataset token variable is not set: {token_env}")
    provider = load_hf_split(
        dataset_spec,
        split,
        cache_dir=cache_dir,
        keep_in_memory=False,
        token=token,
    )
    dataset = ManifestIndexedDataset(
        provider,
        records,
        image_column=dataset_spec.image_column,
        label_column=dataset_spec.label_column,
        transform=_raw_eval_transform(preprocessing),
    )
    raw_images = torch.stack([dataset[index]["image"] for index in range(len(dataset))]).to(device)
    images = normalize(raw_images)
    with torch.no_grad():
        logits = model(images)
    targets = logits.argmax(dim=1)
    alternate_targets = logits.argsort(dim=1, descending=True)[:, 1]
    mean_baseline, distribution, mean_artifact_id = _mean_and_baselines(
        Path(means_artifact).resolve(),
        protocol_digest=protocol.digest,
        dataset_id=dataset_spec.dataset_id,
        dataset_revision=dataset_spec.revision,
        manifest_fingerprint=manifest.fingerprint,
        model=model_key,
        preprocessing=preprocessing,
        normalize=normalize,
        device=device,
        seed=seed,
    )
    zero_baseline = normalize(torch.zeros_like(raw_images[:1]))
    candidates = tuple(sorted(set(int(value) for value in batch_candidates)))
    if not candidates or candidates[0] <= 0:
        raise ValueError("batch_candidates must contain positive integers")
    total_device_bytes = (
        None
        if device.type != "cuda"
        else int(torch.cuda.get_device_properties(device).total_memory)
    )
    methods = _candidate_methods(
        protocol.data,
        model_key,
        definition.family,
        explicit=candidate_methods,
    )
    rank_patch_size = (
        int(protocol.data["evaluation"]["patch_size"]) if patch_size is None else int(patch_size)
    )
    if rank_patch_size <= 0:
        raise ValueError("Compatibility patch_size must be positive")

    measurements: list[CompatibilityMeasurement] = []
    successful_methods: list[LockedMethod] = []
    class_agnostic: set[str] = set()
    relprop_models: dict[str, Any] = {}
    relprop_equivalences: dict[str, dict[str, Any]] = {}
    relprop_equivalence: dict[str, Any] | None = None
    for method in methods:
        candidate_reports: list[dict[str, Any]] = []
        explainer: Any | None = None
        chosen_batch: int | None = None
        output_shape: tuple[int, ...] | None = None
        finite_fraction: float | None = None
        deterministic_repeat: bool | None = None
        target_sensitive: bool | None = None
        attack_seconds: float | None = None
        attack_peak: int | None = None
        attack_batch: int | None = None
        error_text: str | None = None
        try:
            method_model = model
            if relprop_required(method.family, definition.family):
                implementation = relprop_implementation(
                    method.family,
                    definition.family,
                )
                assert implementation is not None
                if implementation not in relprop_models:
                    candidate_model = (
                        create_relprop_model_for_method(
                            method.family,
                            model_key=model_key,
                            num_classes=dataset_spec.num_classes,
                            init_mode="checkpoint",
                            checkpoint_path=str(checkpoint),
                            strict_checkpoint=True,
                        )
                        .eval()
                        .to(device)
                    )
                    candidate_preprocessing = resolved_preprocessing(candidate_model, definition)
                    if object_sha256(candidate_preprocessing) != object_sha256(preprocessing):
                        raise RuntimeError(
                            "RelProp preprocessing differs from the Phase-0 timm model"
                        )
                    relprop_equivalences[implementation] = verify_relprop_equivalence(
                        model,
                        candidate_model,
                        images[: min(2, len(identifiers))],
                    )
                    relprop_models[implementation] = candidate_model
                method_model = relprop_models[implementation]
                relprop_equivalence = relprop_equivalences[implementation]
            explainer = build_explainer(
                method_model,
                method.family,
                architecture=definition.family,  # type: ignore[arg-type]
            )
            for batch_size in candidates:
                if batch_size > len(identifiers):
                    continue
                try:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    _seed_attribution(seed, device)
                    method_model.zero_grad(set_to_none=True)
                    started = time.perf_counter()
                    attribution, _scores, ranks = _attribution(
                        explainer=explainer,
                        method=method,
                        images=images[:batch_size],
                        targets=targets[:batch_size],
                        baseline=_method_baseline(method, zero_baseline, mean_baseline),
                        distribution=distribution,
                        patch_size=rank_patch_size,
                    )
                    _sync(device)
                    elapsed = time.perf_counter() - started
                    finite = float(torch.isfinite(attribution).float().mean().item())
                    peak_allocated_bytes = (
                        None
                        if device.type != "cuda"
                        else int(torch.cuda.max_memory_allocated(device))
                    )
                    peak_reserved_bytes = (
                        None
                        if device.type != "cuda"
                        else int(torch.cuda.max_memory_reserved(device))
                    )
                    # Reserved memory is the conservative capacity measure;
                    # the caching allocator cannot promise that slack to a
                    # larger formal batch.
                    peak_cuda_bytes = peak_reserved_bytes
                    headroom_passed = _cuda_headroom_passed(
                        peak_cuda_bytes,
                        total_device_bytes,
                    )
                    valid = (
                        attribution.ndim == 4
                        and attribution.shape[0] == batch_size
                        and finite == 1.0
                        and _strict_ranks(ranks)
                        and headroom_passed
                    )
                    candidate_reports.append(
                        {
                            "batch_size": batch_size,
                            "passed": valid,
                            "seconds": elapsed,
                            "images_per_second": batch_size / elapsed,
                            "peak_cuda_bytes": peak_cuda_bytes,
                            "peak_cuda_allocated_bytes": peak_allocated_bytes,
                            "peak_cuda_reserved_bytes": peak_reserved_bytes,
                            "total_device_bytes": total_device_bytes,
                            "minimum_headroom_fraction": MINIMUM_CUDA_HEADROOM_FRACTION,
                            "headroom_passed": headroom_passed,
                            "output_shape": list(attribution.shape),
                        }
                    )
                except Exception as error:
                    peak_allocated_bytes = (
                        None
                        if device.type != "cuda"
                        else int(torch.cuda.max_memory_allocated(device))
                    )
                    peak_reserved_bytes = (
                        None
                        if device.type != "cuda"
                        else int(torch.cuda.max_memory_reserved(device))
                    )
                    peak_cuda_bytes = peak_reserved_bytes
                    candidate_reports.append(
                        {
                            "batch_size": batch_size,
                            "passed": False,
                            "peak_cuda_bytes": peak_cuda_bytes,
                            "peak_cuda_allocated_bytes": peak_allocated_bytes,
                            "peak_cuda_reserved_bytes": peak_reserved_bytes,
                            "total_device_bytes": total_device_bytes,
                            "minimum_headroom_fraction": MINIMUM_CUDA_HEADROOM_FRACTION,
                            "headroom_passed": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            passed_candidates = [item for item in candidate_reports if item["passed"]]
            if not passed_candidates:
                raise RuntimeError("No explanation batch candidate passed")
            selected_candidate = max(
                passed_candidates,
                key=lambda item: (item["images_per_second"], item["batch_size"]),
            )
            chosen_batch = int(selected_candidate["batch_size"])
            output_shape = tuple(int(item) for item in selected_candidate["output_shape"])

            repeat_size = min(chosen_batch, len(identifiers))
            repeated = []
            repeated_ranks = []
            for _repeat in range(2):
                _seed_attribution(seed, device)
                method_model.zero_grad(set_to_none=True)
                attribution, _scores, ranks = _attribution(
                    explainer=explainer,
                    method=method,
                    images=images[:repeat_size],
                    targets=targets[:repeat_size],
                    baseline=_method_baseline(method, zero_baseline, mean_baseline),
                    distribution=distribution,
                    patch_size=rank_patch_size,
                )
                repeated.append(attribution.detach())
                repeated_ranks.append(ranks.detach())
            deterministic_repeat = bool(
                torch.allclose(repeated[0], repeated[1], rtol=1e-5, atol=1e-7)
                and torch.equal(repeated_ranks[0], repeated_ranks[1])
            )
            finite_fraction = float(torch.isfinite(repeated[0]).float().mean().item())
            if not deterministic_repeat:
                raise RuntimeError("Deterministic repeat check failed")

            required_sensitivity = method.family not in class_agnostic
            if required_sensitivity:
                _seed_attribution(seed, device)
                alternate, _scores, alternate_ranks = _attribution(
                    explainer=explainer,
                    method=method,
                    images=images[:1],
                    targets=alternate_targets[:1],
                    baseline=_method_baseline(method, zero_baseline, mean_baseline),
                    distribution=distribution,
                    patch_size=int(protocol.data["evaluation"]["patch_size"]),
                )
                target_sensitive = bool(
                    not torch.allclose(repeated[0][:1], alternate, rtol=1e-5, atol=1e-7)
                    or not torch.equal(repeated_ranks[0][:1], alternate_ranks)
                )
                if not target_sensitive:
                    raise RuntimeError("Class-specific target sensitivity check failed")
            else:
                target_sensitive = None

            if EXPLAINER_SPECS[method.family].differentiable_input:
                try:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    raw = raw_images[:1].detach().clone().requires_grad_(True)
                    attack_inputs = normalize(raw)
                    started = time.perf_counter()
                    attack_attr = _special_attack_attribution(
                        method_model,
                        method.family,
                        attack_inputs,
                        targets[:1],
                        create_graph=True,
                    )
                    if attack_attr is None:
                        attack_attr = compute_attribution(
                            explainer,
                            method.family,
                            attack_inputs,
                            targets[:1],
                            params=method.params,
                            baseline=_method_baseline(method, zero_baseline, mean_baseline),
                            baseline_distribution=distribution,
                        )
                    attack_scores = attribution_to_patch_scores(
                        _resize_to_input(attack_attr, raw.shape[-2], raw.shape[-1]),
                        rank_patch_size,
                    )
                    gradient = torch.autograd.grad(attack_scores.sum(), raw, allow_unused=True)[0]
                    _sync(device)
                    if gradient is None or not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError("Attack source has no finite input gradient")
                    attack_seconds = time.perf_counter() - started
                    attack_peak = (
                        None
                        if device.type != "cuda"
                        else int(torch.cuda.max_memory_allocated(device))
                    )
                    attack_batch = ATTACK_BATCH_SIZE
                except Exception:
                    # Explanation compatibility and attack-source compatibility
                    # are reported separately; the attack pilot will only select
                    # methods with attack_batch_size=1.
                    attack_batch = None
            successful_methods.append(method)
            passed = True
        except Exception as error:
            passed = False
            error_text = f"{type(error).__name__}: {error}"
        measurements.append(
            CompatibilityMeasurement(
                method=method.family,
                passed=passed,
                selected_batch_size=chosen_batch,
                batch_candidates=tuple(candidate_reports),
                deterministic_repeat=deterministic_repeat,
                target_sensitive=target_sensitive,
                target_sensitivity_required=method.family not in class_agnostic,
                attack_batch_size=attack_batch,
                attack_seconds_per_sample=attack_seconds,
                attack_peak_cuda_bytes=attack_peak,
                output_shape=output_shape,
                finite_fraction=finite_fraction,
                error=error_text,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    excluded_candidates = [
        {"method": item.method, "error": item.error} for item in measurements if not item.passed
    ]
    failures = _compatibility_gate_failures(
        protocol.data,
        architecture=definition.family,
        measurements=measurements,
    )
    status = "passed" if not failures else "failed"
    selected_methods = [
        {
            "family": method.family,
            "instance_id": method.instance_id,
            "params": method.params,
            "architecture": method.architecture,
        }
        for method in successful_methods
    ]
    result = PilotResult(
        name="explainer_compatibility",
        status=status,
        selected={
            "methods_by_model": {model_key: selected_methods},
            "method_batch_size_by_model": {
                model_key: {
                    item.method: item.selected_batch_size for item in measurements if item.passed
                }
            },
            "attack_batch_size_by_model": {
                model_key: {
                    item.method: item.attack_batch_size
                    for item in measurements
                    if item.attack_batch_size is not None
                }
            },
            "attack_batch_size_algorithm_constraint": ATTACK_BATCH_SIZE,
            "excluded_methods_by_model": {
                model_key: [item["method"] for item in excluded_candidates]
            },
        },
        measurements={
            "model": model_key,
            "architecture": definition.family,
            "checkpoint_sha256": file_sha256(checkpoint),
            "mean_artifact_id": mean_artifact_id,
            "sample_ids_sha256": object_sha256(identifiers),
            "minimum_cuda_headroom_fraction": MINIMUM_CUDA_HEADROOM_FRACTION,
            "total_device_bytes": total_device_bytes,
            "relprop_provider": {
                "required": any(
                    relprop_required(item.family, definition.family) for item in methods
                ),
                "revision": RELPROP_REVISION,
                "source_digest": RELPROP_SOURCE_DIGEST,
                "equivalence": relprop_equivalence,
            },
            "excluded_candidates": excluded_candidates,
            "methods": [asdict(item) for item in measurements],
            "provenance": collect_provenance(project_root),
        },
        failures=failures,
        config_digest=object_sha256(
            {
                "protocol_digest": protocol.digest,
                "dataset_manifest_fingerprint": manifest.fingerprint,
                "checkpoint_sha256": file_sha256(checkpoint),
                "mean_artifact_id": mean_artifact_id,
                "sample_ids": identifiers,
                "batch_candidates": candidates,
                "candidate_methods": [
                    {
                        "family": method.family,
                        "instance_id": method.instance_id,
                        "params": dict(method.params),
                        "architecture": method.architecture,
                    }
                    for method in methods
                ],
                "patch_size": rank_patch_size,
                "minimum_cuda_headroom_fraction": MINIMUM_CUDA_HEADROOM_FRACTION,
                "relprop_revision": RELPROP_REVISION,
                "relprop_source_digest": RELPROP_SOURCE_DIGEST,
                "seed": seed,
            }
        ),
    )
    method_lock = MethodLock(
        dataset_id=dataset_spec.dataset_id,
        model_id=model_key,
        methods=tuple(successful_methods),
        pilot_digest=result.digest,
    )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = save_pilot_result(output / "result.json", result)
    batch_result = explanation_batch_size_pilot_result(
        result,
        model_id=model_key,
        dataloader_workers=dataloader_workers,
    )
    batch_result_path = save_pilot_result(output / "batch-size-result.json", batch_result)
    lock_path = save_method_lock(output / "method-lock.json", method_lock)
    return CompatibilityPilotOutput(
        result_path=result_path,
        batch_result_path=batch_result_path,
        method_lock_path=lock_path,
        result=result,
        batch_result=batch_result,
        method_lock=method_lock,
    )


__all__ = [
    "CompatibilityMeasurement",
    "CompatibilityPilotOutput",
    "explanation_batch_size_pilot_result",
    "run_compatibility_pilot",
]
