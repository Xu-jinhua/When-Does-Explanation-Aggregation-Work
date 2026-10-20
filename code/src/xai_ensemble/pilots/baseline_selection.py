"""Select each baseline-dependent Captum method's reference by validation Fidelity."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.phase2.evaluator import FillReference, evaluate_reference_model_bank

from .base import PilotResult

if TYPE_CHECKING:
    from xai_ensemble.simple.config import SimpleExperiment

SHAP_METHODS = frozenset({"GradientShap", "DeepLiftShap"})
DEFAULT_REFERENCE_CANDIDATES = (
    "zero",
    "dataset_mean",
    "class_mean",
    "gaussian_noise",
    "uniform_noise",
)


def _seed_everything(seed: int, *, device: Any) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _evenly_spaced_positions(count: int, required: int) -> np.ndarray:
    if required <= 0 or required > count:
        raise ValueError(f"Requested {required} pilot samples from a split of size {count}")
    if required == 1:
        return np.asarray([0], dtype=np.int64)
    return np.linspace(0, count - 1, num=required, dtype=np.int64)


def _reference_tensor(
    candidate: str,
    *,
    zero: Any,
    mean: Any,
    class_means: Any,
    labels: Any,
    generator: Any,
) -> Any:
    """Resolve one reference candidate to a tensor in model-input space."""

    import torch

    if candidate == "zero":
        return zero
    if candidate == "dataset_mean":
        return mean
    if candidate == "class_mean":
        return class_means[labels]
    if candidate == "gaussian_noise":
        return torch.randn(zero.shape, generator=generator, dtype=torch.float32).to(zero.device)
    if candidate == "uniform_noise":
        return torch.rand(zero.shape, generator=generator, dtype=torch.float32).to(zero.device)
    raise ValueError(f"unknown baseline reference candidate {candidate!r}")


def _reference_distribution(
    candidate: str,
    *,
    member_count: int,
    zero: Any,
    mean: Any,
    class_means: Any,
    generator: Any,
) -> Any:
    """Resolve one reference candidate to a SHAP baseline distribution."""

    import torch

    if candidate == "zero":
        return zero.repeat(member_count, 1, 1, 1)
    if candidate == "dataset_mean":
        return mean.repeat(member_count, 1, 1, 1)
    if candidate == "class_mean":
        choices = torch.randint(
            int(class_means.shape[0]), (member_count,), generator=generator, dtype=torch.int64
        )
        return class_means[choices]
    if candidate == "gaussian_noise":
        return torch.randn(
            (member_count, *zero.shape[1:]), generator=generator, dtype=torch.float32
        ).to(zero.device)
    if candidate == "uniform_noise":
        return torch.rand(
            (member_count, *zero.shape[1:]), generator=generator, dtype=torch.float32
        ).to(zero.device)
    raise ValueError(f"unknown baseline reference candidate {candidate!r}")


def run_baseline_selection_pilot(
    experiment: SimpleExperiment,
    *,
    model_id: str,
    config: Mapping[str, Any],
    split: str | None = None,
    sample_count: int | None = None,
    device: str = "cuda:0",
    seed: int | None = None,
) -> PilotResult:
    """Evaluate every reference candidate for every baseline-dependent method."""

    import torch

    from xai_ensemble.phase1.explainers import (
        EXPLAINER_SPECS,
        build_explainer,
        compute_attribution,
    )
    from xai_ensemble.simple.data import (
        fixed_baselines,
        load_model,
        load_raw_class_means,
        load_raw_dataset_mean,
        load_split,
    )
    from xai_ensemble.simple.rank_ready import attribution_to_patch_scores, scores_to_ranks

    patch_size = int(config.get("patch_size", 16))
    top_k = int(config.get("top_k", 20))
    if str(config.get("fill", "dataset_mean")) != "dataset_mean":
        raise ValueError("baseline_selection is defined for the dataset_mean fill")
    model_config = experiment.model(model_id)
    dataset_config = experiment.dataset(model_config.dataset_id)
    available = {
        definition.family
        for definition in experiment.methods.for_architecture(model_config.architecture)
        if EXPLAINER_SPECS[definition.family].baseline_kind is not None
    }
    methods = tuple(str(item) for item in config.get("methods", ()))
    if not methods:
        methods = tuple(
            definition.family
            for definition in experiment.methods.for_architecture(model_config.architecture)
            if EXPLAINER_SPECS[definition.family].baseline_kind is not None
        )
    unknown = sorted(set(methods) - available)
    if unknown:
        raise ValueError(
            f"Methods unavailable for {model_config.architecture} or not baseline-dependent: "
            f"{unknown}"
        )
    candidates = tuple(str(item) for item in config.get("reference_candidates", ()))
    if not candidates:
        candidates = DEFAULT_REFERENCE_CANDIDATES
    chosen_split = str(split or config.get("split", "validation"))
    if chosen_split not in dataset_config.splits:
        raise ValueError(
            f"Split {chosen_split!r} is not configured for {dataset_config.dataset_id}"
        )
    count = int(sample_count if sample_count is not None else config.get("sample_count", 128))
    attribution_batch_size = int(config.get("attribution_batch_size", 16))
    evaluation_batch_size = int(config.get("evaluation_batch_size", 64))
    shap_distribution_members = int(config.get("shap_distribution_members", 20))
    shap_n_samples = int(config.get("shap_n_samples", 20))
    if count <= 0 or attribution_batch_size <= 0 or evaluation_batch_size <= 0:
        raise ValueError("sample_count and batch sizes must be positive")
    if shap_distribution_members <= 0 or shap_n_samples <= 0:
        raise ValueError("SHAP distribution members and n_samples must be positive")
    pilot_seed = int(seed if seed is not None else config.get("seed", experiment.runtime.seed))

    target_device = torch.device(device)
    loaded = load_model(model_config, device=target_device, include_checkpoint=True)
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    input_size = int(loaded.preprocessing["input_size"])
    bundle = load_split(
        dataset_config,
        loaded,
        split=chosen_split,
        workers=experiment.runtime.dataloader_workers,
    )
    positions = _evenly_spaced_positions(len(bundle), count)
    indices = bundle.indices[positions]
    raw_images, label_rows = bundle.rows(indices)
    labels = label_rows.to(dtype=torch.int64).numpy()
    indices_np = indices.to(dtype=torch.int64).numpy()
    raw_mean = load_raw_dataset_mean(model_config, input_size=input_size)
    fill = FillReference(
        values=raw_mean.squeeze(0).numpy(),
        source_split="train",
        artifact_id=object_sha256(
            {"path": str(model_config.mean_path), "key": model_config.mean_key}
        ),
    )
    zero, _ = fixed_baselines(raw_mean, loaded, device=target_device, seed=pilot_seed)
    mean = loaded.normalize(raw_mean.to(target_device, dtype=torch.float32))
    raw_class_means, _ = load_raw_class_means(model_config, input_size=input_size)
    class_means = loaded.normalize(raw_class_means.to(target_device, dtype=torch.float32))

    inputs = loaded.normalize(raw_images.to(target_device, dtype=torch.float32))
    with torch.no_grad():
        logits = loaded.model(inputs)
    predictions = torch.argmax(logits, dim=1)
    targets = predictions.detach().clone()

    method_rows: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        spec = EXPLAINER_SPECS[method]
        explainer = build_explainer(
            loaded.model,
            method,
            architecture=model_config.architecture,
        )
        params = {"patch_size": patch_size} if method in {"FeatureAblation", "Occlusion"} else {}
        if method == "GradientShap":
            params["n_samples"] = shap_n_samples
        rank_bank: dict[str, np.ndarray] = {}
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                stable_seed(pilot_seed, "baseline-selection", model_id, method, candidate)
            )
            if method in SHAP_METHODS:
                reference: Any = _reference_distribution(
                    candidate,
                    member_count=shap_distribution_members,
                    zero=zero,
                    mean=mean,
                    class_means=class_means,
                    generator=generator,
                )
            else:
                reference = _reference_tensor(
                    candidate,
                    zero=zero,
                    mean=mean,
                    class_means=class_means,
                    labels=targets,
                    generator=generator,
                )
            _seed_everything(
                stable_seed(pilot_seed, "baseline-attribution", model_id, method, candidate),
                device=target_device,
            )
            started = time.perf_counter()
            rank_chunks: list[np.ndarray] = []
            for start in range(0, int(inputs.shape[0]), attribution_batch_size):
                stop = min(int(inputs.shape[0]), start + attribution_batch_size)
                batch = inputs[start:stop].detach().clone()
                if spec.differentiable_input:
                    batch.requires_grad_(True)
                if method in SHAP_METHODS:
                    attribution = compute_attribution(
                        explainer,
                        method,
                        batch,
                        targets[start:stop],
                        params=params,
                        baseline_distribution=reference,
                    )
                else:
                    attribution = compute_attribution(
                        explainer,
                        method,
                        batch,
                        targets[start:stop],
                        params=params,
                        baseline=reference,
                    )
                if int(attribution.shape[0]) != stop - start:
                    raise RuntimeError("Explainer changed the attribution batch dimension")
                if not bool(torch.isfinite(attribution).all()):
                    raise ValueError(
                        f"{method} with reference={candidate} produced NaN or infinite values"
                    )
                scores = attribution_to_patch_scores(
                    attribution.detach().to("cpu", dtype=torch.float32).numpy(), patch_size
                )
                rank_chunks.append(scores_to_ranks(scores))
            rank_bank[f"{method}::{candidate}"] = np.concatenate(rank_chunks, axis=0)
            rows.append(
                {
                    "reference": candidate,
                    "wall_seconds": time.perf_counter() - started,
                    "attribution_finite": True,
                    "passed": True,
                }
            )
        traces = evaluate_reference_model_bank(
            loaded.model,
            raw_images.numpy(),
            rank_bank,
            true_labels=labels,
            target_labels=targets.detach().cpu().numpy(),
            fill_reference=fill,
            reference_model_id=model_config.model_id,
            sample_ids=indices_np,
            patch_size=patch_size,
            k=top_k,
            index_base=0,
            batch_size=evaluation_batch_size,
            device=str(target_device),
            autocast=False,
            normalize=loaded.normalize,
            require_target_matches_clean=True,
            clean_predictions=predictions.detach().cpu().numpy(),
        )
        for row in rows:
            values = traces[f"{method}::{row['reference']}"].stats.values()
            row.update(
                {
                    "fidelity": values.F,
                    "fidelity_bar": values.Fbar,
                    "consistency": values.C,
                    "consistency_bar": values.Cbar,
                }
            )
        method_rows[method] = rows

    pass_config = config.get("pass", {})
    minimum_sample_count = int(pass_config.get("minimum_sample_count", 1))
    failures: list[str] = []
    if count < minimum_sample_count:
        failures.append(
            f"sample_count {count} below pass.minimum_sample_count {minimum_sample_count}"
        )
    selected: dict[str, str] = {}
    for method, rows in method_rows.items():
        if not rows:
            failures.append(f"method {method} has no evaluated reference candidate")
            continue
        order = {candidate: index for index, candidate in enumerate(candidates)}
        winner = max(rows, key=lambda row: (row["fidelity"], -order[row["reference"]]))
        selected[method] = winner["reference"]
    result = PilotResult(
        name="baseline_selection",
        status="passed" if not failures else "failed",
        selected=selected,
        measurements={
            "dataset_id": dataset_config.dataset_id,
            "model_id": model_config.model_id,
            "split": chosen_split,
            "sample_count": int(inputs.shape[0]),
            "patch_size": patch_size,
            "top_k": top_k,
            "fill": "dataset_mean",
            "shap_distribution_members": shap_distribution_members,
            "shap_n_samples": shap_n_samples,
            "selection": "highest_validation_fidelity_configured_order_on_ties",
            "methods": {
                method: {"candidates": rows} for method, rows in method_rows.items()
            },
        },
        failures=failures,
        config_digest=object_sha256(dict(config)),
    )
    print(
        f"BASELINE_SELECTION {result.status} model={model_config.model_id} "
        f"split={chosen_split} samples={int(inputs.shape[0])} selected={selected}",
        flush=True,
    )
    return result


__all__ = ["run_baseline_selection_pilot"]
