"""Sweep the Integrated Gradients approximation depth on validation Fidelity."""

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


def _resolve_ig_baseline(definition_params: Mapping[str, Any], zero: Any, mean: Any) -> Any:
    kind = str(definition_params.get("baseline", "zero"))
    if kind == "zero":
        return zero
    if kind == "dataset_mean":
        return mean
    raise ValueError(f"Unsupported IntegratedGradients baseline {kind!r}")


def run_ig_step_budget_pilot(
    experiment: SimpleExperiment,
    *,
    model_id: str,
    config: Mapping[str, Any],
    split: str | None = None,
    sample_count: int | None = None,
    device: str = "cuda:0",
    seed: int | None = None,
) -> PilotResult:
    """Evaluate every candidate ``n_steps`` on one dataset-model pair."""

    import torch

    from xai_ensemble.phase1.explainers import build_explainer, compute_attribution
    from xai_ensemble.simple.data import (
        fixed_baselines,
        load_model,
        load_raw_dataset_mean,
        load_split,
    )
    from xai_ensemble.simple.rank_ready import attribution_to_patch_scores, scores_to_ranks

    method = str(config.get("method", "IntegratedGradients"))
    if method != "IntegratedGradients":
        raise ValueError(f"ig_step_budget only supports IntegratedGradients, got {method!r}")
    step_candidates = tuple(int(item) for item in config.get("step_candidates", ()))
    if not step_candidates or any(item <= 0 for item in step_candidates):
        raise ValueError("step_candidates must be non-empty positive integers")
    patch_size = int(config.get("patch_size", 16))
    top_k = int(config.get("top_k", 20))
    if str(config.get("fill", "dataset_mean")) != "dataset_mean":
        raise ValueError("ig_step_budget is defined for the dataset_mean fill")
    model_config = experiment.model(model_id)
    dataset_config = experiment.dataset(model_config.dataset_id)
    chosen_split = str(split or config.get("split", "validation"))
    if chosen_split not in dataset_config.splits:
        raise ValueError(
            f"Split {chosen_split!r} is not configured for {dataset_config.dataset_id}"
        )
    count = int(sample_count if sample_count is not None else config.get("sample_count", 128))
    attribution_batch_size = int(config.get("attribution_batch_size", 16))
    evaluation_batch_size = int(config.get("evaluation_batch_size", 64))
    if count <= 0 or attribution_batch_size <= 0 or evaluation_batch_size <= 0:
        raise ValueError("sample_count and batch sizes must be positive")
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
    zero, distribution = fixed_baselines(
        raw_mean, loaded, device=target_device, seed=pilot_seed
    )
    ig_params = dict(experiment.methods.family(method).params)
    baseline = _resolve_ig_baseline(ig_params, zero, distribution[2:3])

    inputs = loaded.normalize(raw_images.to(target_device, dtype=torch.float32))
    with torch.no_grad():
        logits = loaded.model(inputs)
    predictions = torch.argmax(logits, dim=1)
    targets = predictions.detach().clone()

    explainer = build_explainer(
        loaded.model,
        method,
        architecture=model_config.architecture,
    )
    rank_bank: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for n_steps in step_candidates:
        _seed_everything(
            stable_seed(pilot_seed, "ig-step-budget", model_id, n_steps),
            device=target_device,
        )
        started = time.perf_counter()
        rank_chunks: list[np.ndarray] = []
        for start in range(0, int(inputs.shape[0]), attribution_batch_size):
            stop = min(int(inputs.shape[0]), start + attribution_batch_size)
            batch = inputs[start:stop].detach().clone().requires_grad_(True)
            attribution = compute_attribution(
                explainer,
                method,
                batch,
                targets[start:stop],
                params={"n_steps": n_steps},
                baseline=baseline,
            )
            if int(attribution.shape[0]) != stop - start:
                raise RuntimeError("Explainer changed the attribution batch dimension")
            if not bool(torch.isfinite(attribution).all()):
                raise ValueError(
                    f"IntegratedGradients n_steps={n_steps} produced NaN or infinite values"
                )
            scores = attribution_to_patch_scores(
                attribution.detach().to("cpu", dtype=torch.float32).numpy(), patch_size
            )
            rank_chunks.append(scores_to_ranks(scores))
        rank_bank[f"n_steps_{n_steps}"] = np.concatenate(rank_chunks, axis=0)
        rows.append(
            {
                "n_steps": n_steps,
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
        values = traces[f"n_steps_{row['n_steps']}"].stats.values()
        row.update(
            {
                "fidelity": values.F,
                "fidelity_bar": values.Fbar,
                "consistency": values.C,
                "consistency_bar": values.Cbar,
            }
        )

    pass_config = config.get("pass", {})
    minimum_sample_count = int(pass_config.get("minimum_sample_count", 1))
    minimum_candidates = int(pass_config.get("minimum_step_candidates", 1))
    winner = None
    if rows:
        winner = max(rows, key=lambda row: (row["fidelity"], -row["n_steps"]))
    failures = []
    if count < minimum_sample_count:
        failures.append(
            f"sample_count {count} below pass.minimum_sample_count {minimum_sample_count}"
        )
    if len(rows) < minimum_candidates:
        failures.append("not enough evaluated step candidates")
    if winner is None:
        failures.append("no step candidate produced a validation Fidelity")
    status = "passed" if not failures else "failed"
    selected = {} if winner is None else {"n_steps": winner["n_steps"]}
    result = PilotResult(
        name="ig_step_budget",
        status=status,
        selected=selected,
        measurements={
            "dataset_id": dataset_config.dataset_id,
            "model_id": model_config.model_id,
            "split": chosen_split,
            "sample_count": int(inputs.shape[0]),
            "patch_size": patch_size,
            "top_k": top_k,
            "fill": "dataset_mean",
            "baseline": dict(ig_params).get("baseline", "zero"),
            "selection": "highest_validation_fidelity_smallest_steps_on_ties",
            "candidates": rows,
        },
        failures=failures,
        config_digest=object_sha256(dict(config)),
    )
    print(
        f"IG_STEP_BUDGET {result.status} model={model_config.model_id} "
        f"split={chosen_split} samples={int(inputs.shape[0])} selected={selected}",
        flush=True,
    )
    return result


__all__ = ["run_ig_step_budget_pilot"]
