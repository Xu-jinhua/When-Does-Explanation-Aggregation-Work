"""Reference-model masking evaluator in the raw ``[0, 1]`` image domain."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import MetricSufficientStats
from .rankings import canonicalize_rankings


@dataclass(frozen=True)
class FillReference:
    """A fill tensor computed from a registered training split artifact."""

    values: Any
    source_split: str
    artifact_id: str

    def __post_init__(self) -> None:
        if not self.source_split.lower().startswith("train"):
            raise ValueError(
                "evaluation fill must be derived from a training split, "
                f"got source_split={self.source_split!r}"
            )
        if not self.artifact_id or not self.artifact_id.strip():
            raise ValueError("fill artifact_id must be non-empty")


@dataclass(frozen=True)
class ClassMeanFillReference:
    """A train-derived class-mean bank selected by the fixed clean target."""

    values: Any
    class_counts: Any
    source_split: str
    artifact_id: str

    def __post_init__(self) -> None:
        if not self.source_split.lower().startswith("train"):
            raise ValueError(
                "class-mean fill must be derived from a training split, "
                f"got source_split={self.source_split!r}"
            )
        if not self.artifact_id or not self.artifact_id.strip():
            raise ValueError("fill artifact_id must be non-empty")


@dataclass(frozen=True)
class LoadedFillReference:
    reference: FillReference
    manifest_path: Path
    metadata: Mapping[str, Any]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def load_fill_reference_artifact(
    manifest_path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
) -> LoadedFillReference:
    """Load and verify the immutable Phase 0 train-image-mean artifact."""

    from xai_ensemble.core.manifest import load_manifest, validate_manifest_files

    path = Path(manifest_path)
    manifest = load_manifest(path)
    if manifest.kind != "train_image_means":
        raise ValueError(f"formal fill requires kind='train_image_means', got {manifest.kind!r}")
    validate_manifest_files(path.parent, manifest)
    metadata = dict(manifest.metadata)
    if not str(metadata.get("source_split", "")).lower().startswith("train"):
        raise ValueError("formal fill artifact source_split must be train")
    if metadata.get("domain") != "raw_0_1":
        raise ValueError("formal fill artifact must be computed in raw_0_1 domain")
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_identity.items()
        if _plain_json(metadata.get(key)) != _plain_json(expected)
    }
    if mismatches:
        raise ValueError(f"fill artifact identity mismatch: {mismatches}")
    try:
        entry = manifest.entry("means.npz")
    except KeyError as error:
        raise ValueError("fill artifact manifest does not contain means.npz") from error
    with np.load(path.parent / entry.path, allow_pickle=False) as archive:
        if "dataset_mean" not in archive.files:
            raise ValueError("means.npz does not contain dataset_mean")
        dataset_mean = np.asarray(archive["dataset_mean"], dtype=np.float32)
    if dataset_mean.ndim != 3 or not np.all(np.isfinite(dataset_mean)):
        raise ValueError("dataset_mean must be a finite CxHxW tensor")
    if float(np.min(dataset_mean)) < 0.0 or float(np.max(dataset_mean)) > 1.0:
        raise ValueError("dataset_mean must be in raw [0, 1] space")
    reference = FillReference(
        values=dataset_mean,
        source_split=str(metadata["source_split"]),
        artifact_id=manifest.artifact_id,
    )
    return LoadedFillReference(reference, path.resolve(), metadata)


@dataclass(frozen=True)
class MaskedInputs:
    removed: NDArray[np.floating]
    retained: NDArray[np.floating]
    patch_mask: NDArray[np.bool_]
    selected_patch_indices: NDArray[np.int64]


def _raw_numpy_images(images: ArrayLike) -> NDArray[np.floating]:
    raw = np.asarray(images)
    if raw.ndim != 4 or any(size == 0 for size in raw.shape):
        raise ValueError("images must have non-empty BCHW shape")
    if not np.issubdtype(raw.dtype, np.floating):
        raise TypeError("raw [0, 1] images must have a floating dtype")
    if not np.all(np.isfinite(raw)):
        raise ValueError("images contain NaN or infinite values")
    if float(np.min(raw)) < 0.0 or float(np.max(raw)) > 1.0:
        raise ValueError("images must be in the raw [0, 1] domain")
    return raw


def _flat_canonical_ranks(
    ranks: ArrayLike,
    *,
    n_samples: int,
    grid_height: int,
    grid_width: int,
    index_base: int,
) -> NDArray[np.int64]:
    raw = np.asarray(ranks)
    n_patches = grid_height * grid_width
    if raw.shape == (n_samples, grid_height, grid_width):
        raw = raw.reshape(n_samples, n_patches)
    if raw.shape != (n_samples, n_patches):
        raise ValueError(
            "ranks must have shape "
            f"({n_samples}, {n_patches}) or "
            f"({n_samples}, {grid_height}, {grid_width}), got {raw.shape}"
        )
    return canonicalize_rankings(raw, index_base=index_base)


def _numpy_fill(
    fill_reference: FillReference | ClassMeanFillReference,
    *,
    channels: int,
    height: int,
    width: int,
    dtype: np.dtype,
    class_labels: ArrayLike | None = None,
) -> NDArray[np.floating]:
    source_values = fill_reference.values
    if hasattr(source_values, "detach") and hasattr(source_values, "cpu"):
        source_values = source_values.detach().cpu().numpy()
    values = np.asarray(source_values)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise TypeError("fill reference must contain real numeric values")
    values = values.astype(dtype, copy=False)
    if isinstance(fill_reference, ClassMeanFillReference):
        counts = fill_reference.class_counts
        if hasattr(counts, "detach") and hasattr(counts, "cpu"):
            counts = counts.detach().cpu().numpy()
        counts = np.asarray(counts)
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("class_counts must be an integer vector")
        if (
            values.ndim != 4
            or values.shape[0] != counts.shape[0]
            or values.shape[1:]
            not in {
                (channels, 1, 1),
                (channels, height, width),
            }
        ):
            raise ValueError(
                "class-mean fill must have KxCx1x1 or KxCxHxW shape aligned with class_counts"
            )
        if class_labels is None:
            raise ValueError("class-mean fill requires one fixed class label per image")
        labels = np.asarray(class_labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("class-mean fill labels must be an integer vector")
        labels = labels.astype(np.int64, copy=False)
        if np.any(labels < 0) or np.any(labels >= len(counts)):
            raise ValueError("class-mean fill labels are outside the class bank")
        if np.any(counts[labels] <= 0):
            raise ValueError("class-mean fill selected a class absent from the training split")
        values = values[labels]
        try:
            broadcast = np.broadcast_to(values, (len(labels), channels, height, width))
        except ValueError as error:
            raise ValueError("class-mean fill is not broadcastable to BCHW images") from error
        if not np.all(np.isfinite(broadcast)):
            raise ValueError("class-mean fill contains NaN or infinite values")
        if float(np.min(broadcast)) < 0.0 or float(np.max(broadcast)) > 1.0:
            raise ValueError("class-mean fill must be in the raw [0, 1] domain")
        return np.array(broadcast, copy=True, order="C")

    if values.ndim == 0:
        values = values.reshape(1, 1, 1, 1)
    elif values.shape == (channels,):
        values = values.reshape(1, channels, 1, 1)
    elif values.shape in ((channels, 1, 1), (channels, height, width)):
        values = values[None, ...]
    elif values.shape not in ((1, channels, 1, 1), (1, channels, height, width)):
        raise ValueError(
            "dataset fill must be scalar, C, Cx1x1, CxHxW, or have one "
            "leading singleton batch dimension; per-test-sample fills are forbidden"
        )
    try:
        broadcast = np.broadcast_to(values, (1, channels, height, width))
    except ValueError as error:
        raise ValueError("fill reference is not broadcastable to BCHW images") from error
    if not np.all(np.isfinite(broadcast)):
        raise ValueError("fill reference contains NaN or infinite values")
    if float(np.min(broadcast)) < 0.0 or float(np.max(broadcast)) > 1.0:
        raise ValueError("fill reference must be in the raw [0, 1] domain")
    return np.array(broadcast, copy=True, order="C")


def build_masked_inputs(
    images: ArrayLike,
    ranks: ArrayLike,
    fill_reference: FillReference | ClassMeanFillReference,
    *,
    patch_size: int = 14,
    k: int = 20,
    index_base: int = 0,
    fill_labels: ArrayLike | None = None,
) -> MaskedInputs:
    """Build top-k removed/retained images without model normalization.

    The input and fill are validated in raw image space.  A mask rank is
    expanded over its exact ``patch_size x patch_size`` block and broadcast
    across channels; no interpolation or overlapping patch convention is used.
    """

    raw = _raw_numpy_images(images)
    n_samples, channels, height, width = raw.shape
    if not isinstance(patch_size, (int, np.integer)) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image shape {(height, width)} is not divisible by patch_size={patch_size}"
        )
    grid_height = height // patch_size
    grid_width = width // patch_size
    n_patches = grid_height * grid_width
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= n_patches:
        raise ValueError(f"k must lie between 1 and {n_patches}, got {k!r}")
    canonical = _flat_canonical_ranks(
        ranks,
        n_samples=n_samples,
        grid_height=grid_height,
        grid_width=grid_width,
        index_base=index_base,
    )
    patch_mask = (canonical < k).reshape(n_samples, grid_height, grid_width)
    if not np.all(np.sum(patch_mask, axis=(1, 2)) == k):
        raise RuntimeError("strict rank invariant failed to produce exactly k patches")
    pixel_mask = np.repeat(np.repeat(patch_mask, patch_size, axis=1), patch_size, axis=2)[
        :, None, :, :
    ]
    fill = _numpy_fill(
        fill_reference,
        channels=channels,
        height=height,
        width=width,
        dtype=raw.dtype,
        class_labels=fill_labels,
    )
    removed = np.where(pixel_mask, fill, raw)
    retained = np.where(pixel_mask, raw, fill)
    selected = np.argsort(canonical, axis=1, kind="stable")[:, :k]
    return MaskedInputs(
        removed=removed,
        retained=retained,
        patch_mask=patch_mask,
        selected_patch_indices=selected.astype(np.int64, copy=False),
    )


@dataclass(frozen=True)
class ReferenceEvaluationTrace:
    """Per-sample predictions and metric events produced by one common f_ref."""

    stats: MetricSufficientStats
    target_labels: NDArray
    clean_predictions: NDArray
    removed_predictions: NDArray
    retained_predictions: NDArray
    selected_patch_indices: NDArray[np.int64]
    reference_model_id: str
    fill_artifact_id: str
    patch_size: int
    k: int
    batch_size: int
    device: str
    autocast_used: bool

    def records(self) -> Iterator[dict[str, Any]]:
        """Yield compact per-sample records suitable for Parquet/JSONL traces."""

        contributions = self.stats.contributions()
        ids = (
            np.arange(self.stats.n_samples)
            if self.stats.sample_ids is None
            else self.stats.sample_ids
        )
        for index in range(self.stats.n_samples):
            yield {
                "sample_id": ids[index].item()
                if isinstance(ids[index], np.generic)
                else ids[index],
                "target_label": _scalar(self.target_labels[index]),
                "clean_prediction": _scalar(self.clean_predictions[index]),
                "removed_prediction": _scalar(self.removed_predictions[index]),
                "retained_prediction": _scalar(self.retained_predictions[index]),
                "selected_patch_indices": self.selected_patch_indices[index].tolist(),
                **{metric: float(values[index]) for metric, values in contributions.items()},
            }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _model_device(model: Any, requested: str | None, torch: Any) -> Any:
    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _extract_logits(output: Any, torch: Any) -> Any:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError("reference model must return logits tensor, .logits, or {'logits': ...}")


def evaluate_reference_model_bank(
    reference_model: Any,
    images: Any,
    rank_bank: Mapping[str, ArrayLike] | ArrayLike,
    *,
    true_labels: ArrayLike,
    target_labels: ArrayLike,
    fill_reference: FillReference | ClassMeanFillReference,
    reference_model_id: str,
    sample_ids: ArrayLike | None = None,
    patch_size: int = 14,
    k: int = 20,
    index_base: int = 0,
    batch_size: int = 64,
    device: str | None = None,
    autocast: bool = True,
    autocast_dtype: Any | None = None,
    normalize: Callable[[Any], Any] | None = None,
    require_target_matches_clean: bool = True,
    clean_predictions: ArrayLike | None = None,
) -> Mapping[str, ReferenceEvaluationTrace]:
    """Evaluate a complete rule bank while transferring each image batch once.

    Masking always occurs on raw ``[0,1]`` tensors.  ``normalize`` is applied
    immediately before each common ``reference_model`` forward pass.  All
    rules for one source mini-batch share the same H2D image transfer and fill
    tensor.  Rules are chunked only when needed to keep ``batch_size`` as the
    hard upper bound on every actual model forward.

    For clean evaluation, ``require_target_matches_clean=True`` enforces that
    every ballot explains the same registered reference decision.  Set it
    false only when evaluating perturbed inputs against targets fixed from
    their clean counterparts.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on CPU-only install
        raise RuntimeError("evaluate_reference_model requires the optional torch extra") from error

    if not reference_model_id or not reference_model_id.strip():
        raise ValueError("reference_model_id must be non-empty")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(rank_bank, Mapping):
        rank_items = tuple((str(name), values) for name, values in rank_bank.items())
    else:
        raw_bank = np.asarray(rank_bank)
        if raw_bank.ndim != 3:
            raise ValueError("rank_bank must be a mapping or have [N,R,P] shape")
        rank_items = tuple(
            (f"r{index:03d}", raw_bank[:, index]) for index in range(raw_bank.shape[1])
        )
    if not rank_items or len({name for name, _ in rank_items}) != len(rank_items):
        raise ValueError("rank_bank labels must be non-empty and unique")
    image_tensor = torch.as_tensor(images)
    if image_tensor.ndim != 4 or any(size == 0 for size in image_tensor.shape):
        raise ValueError("images must have non-empty BCHW shape")
    if not image_tensor.dtype.is_floating_point:
        raise TypeError("raw [0, 1] images must have a floating dtype")
    if not bool(torch.isfinite(image_tensor).all()):
        raise ValueError("images contain NaN or infinite values")
    if float(image_tensor.min()) < 0.0 or float(image_tensor.max()) > 1.0:
        raise ValueError("images must be in the raw [0, 1] domain")
    n_samples, channels, height, width = image_tensor.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image shape {(height, width)} is not divisible by patch_size={patch_size}"
        )
    grid_height, grid_width = height // patch_size, width // patch_size
    n_patches = grid_height * grid_width
    if not 1 <= k <= n_patches:
        raise ValueError(f"k must lie between 1 and {n_patches}")
    canonical = np.stack(
        [
            _flat_canonical_ranks(
                values,
                n_samples=n_samples,
                grid_height=grid_height,
                grid_width=grid_width,
                index_base=index_base,
            )
            for _, values in rank_items
        ],
        axis=1,
    )
    true = np.asarray(true_labels)
    targets = np.asarray(target_labels)
    if true.shape != (n_samples,) or targets.shape != (n_samples,):
        raise ValueError("true_labels and target_labels must have one entry per image")
    supplied_clean = None if clean_predictions is None else np.asarray(clean_predictions)
    if supplied_clean is not None:
        if supplied_clean.shape != (n_samples,):
            raise ValueError("clean_predictions must have one entry per image")
        if np.issubdtype(supplied_clean.dtype, np.floating) and not np.all(
            np.isfinite(supplied_clean)
        ):
            raise ValueError("clean_predictions contain NaN or infinite values")
    ids = np.arange(n_samples) if sample_ids is None else np.asarray(sample_ids)
    if ids.shape != (n_samples,):
        raise ValueError("sample_ids must have one entry per image")

    fill_numpy = _numpy_fill(
        fill_reference,
        channels=channels,
        height=height,
        width=width,
        dtype=np.dtype("float32"),
        class_labels=targets,
    )
    selected = np.argsort(canonical, axis=2, kind="stable")[:, :, :k].astype(np.int64, copy=False)
    model_device = _model_device(reference_model, device, torch)
    if device is not None and hasattr(reference_model, "to"):
        reference_model.to(model_device)
    use_autocast = bool(autocast and model_device.type == "cuda")
    if autocast_dtype is None and use_autocast:
        autocast_dtype = torch.float16
    inferred_clean_predictions: list[Any] = []
    rule_count = len(rank_items)
    removed_predictions: list[list[Any]] = [[] for _ in range(rule_count)]
    retained_predictions: list[list[Any]] = [[] for _ in range(rule_count)]
    rule_chunk_size = min(rule_count, max(1, int(batch_size) // 2))
    source_batch_size = max(1, int(batch_size) // (2 * rule_chunk_size))
    was_training = bool(getattr(reference_model, "training", False))
    reference_model.eval()

    def to_device(value: Any, *, dtype: Any) -> Any:
        tensor = torch.as_tensor(value)
        if model_device.type == "cuda" and tensor.device.type == "cpu" and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(device=model_device, dtype=dtype, non_blocking=True)

    def predict(combined: Any) -> np.ndarray:
        prediction_chunks = []
        for forward_start in range(0, int(combined.shape[0]), int(batch_size)):
            forward_stop = min(int(combined.shape[0]), forward_start + int(batch_size))
            model_batch = combined[forward_start:forward_stop]
            if normalize is not None:
                model_batch = normalize(model_batch)
            autocast_context = (
                torch.autocast(
                    device_type=model_device.type,
                    dtype=autocast_dtype,
                    enabled=True,
                )
                if use_autocast
                else nullcontext()
            )
            with autocast_context:
                logits = _extract_logits(reference_model(model_batch), torch)
            if logits.ndim != 2 or int(logits.shape[0]) != forward_stop - forward_start:
                raise ValueError(
                    "reference model logits must have shape (forward_batch, n_classes)"
                )
            prediction_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(prediction_chunks)

    try:
        with torch.inference_mode():
            if supplied_clean is None:
                for start in range(0, n_samples, int(batch_size)):
                    stop = min(n_samples, start + int(batch_size))
                    inferred_clean_predictions.append(
                        predict(to_device(image_tensor[start:stop], dtype=torch.float32))
                    )
            for start in range(0, n_samples, source_batch_size):
                stop = min(n_samples, start + source_batch_size)
                raw_batch = to_device(image_tensor[start:stop], dtype=torch.float32)
                fill_batch = fill_numpy if fill_numpy.shape[0] == 1 else fill_numpy[start:stop]
                fill = to_device(fill_batch, dtype=raw_batch.dtype)
                for rule_start in range(0, rule_count, rule_chunk_size):
                    rule_stop = min(rule_count, rule_start + rule_chunk_size)
                    current_rules = rule_stop - rule_start
                    patch_mask = torch.as_tensor(
                        canonical[start:stop, rule_start:rule_stop] < k,
                        device=model_device,
                        dtype=torch.bool,
                    ).reshape(stop - start, current_rules, grid_height, grid_width)
                    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(
                        patch_size, dim=3
                    )[:, :, None]
                    raw_rules = raw_batch[:, None]
                    fill_rules = fill[:, None] if int(fill.shape[0]) != 1 else fill[None]
                    removed = torch.where(pixel_mask, fill_rules, raw_rules)
                    retained = torch.where(pixel_mask, raw_rules, fill_rules)
                    predictions = predict(
                        torch.cat(
                            (
                                removed.flatten(0, 1),
                                retained.flatten(0, 1),
                            ),
                            dim=0,
                        )
                    )
                    split = (stop - start) * current_rules
                    removed_chunk = predictions[:split].reshape(stop - start, current_rules)
                    retained_chunk = predictions[split:].reshape(stop - start, current_rules)
                    for offset, rule_index in enumerate(range(rule_start, rule_stop)):
                        removed_predictions[rule_index].append(removed_chunk[:, offset])
                        retained_predictions[rule_index].append(retained_chunk[:, offset])
    finally:
        if was_training:
            reference_model.train()

    clean_array = (
        np.concatenate(inferred_clean_predictions)
        if supplied_clean is None
        else supplied_clean.copy()
    )
    if require_target_matches_clean and not np.array_equal(targets, clean_array):
        mismatch = np.flatnonzero(targets != clean_array)
        raise ValueError(
            "reference targets do not match f_ref clean predictions for "
            f"{mismatch.size} samples; first indices={mismatch[:5].tolist()}"
        )
    result = {}
    for rule_index, (rule_name, _) in enumerate(rank_items):
        removed_array = np.concatenate(removed_predictions[rule_index])
        retained_array = np.concatenate(retained_predictions[rule_index])
        stats = MetricSufficientStats.from_predictions(
            true_labels=true,
            clean_predictions=clean_array,
            removed_predictions=removed_array,
            retained_predictions=retained_array,
            sample_ids=ids,
            class_labels=true,
        )
        result[rule_name] = ReferenceEvaluationTrace(
            stats=stats,
            target_labels=targets,
            clean_predictions=clean_array.copy(),
            removed_predictions=removed_array,
            retained_predictions=retained_array,
            selected_patch_indices=selected[:, rule_index],
            reference_model_id=reference_model_id,
            fill_artifact_id=fill_reference.artifact_id,
            patch_size=int(patch_size),
            k=int(k),
            batch_size=int(batch_size),
            device=str(model_device),
            autocast_used=use_autocast,
        )
    return result


def evaluate_reference_model(
    reference_model: Any,
    images: Any,
    ranks: ArrayLike,
    **kwargs: Any,
) -> ReferenceEvaluationTrace:
    """Backward-compatible single-rule wrapper around the rule-bank evaluator."""

    return evaluate_reference_model_bank(
        reference_model,
        images,
        {"rule": ranks},
        **kwargs,
    )["rule"]


__all__ = [
    "ClassMeanFillReference",
    "FillReference",
    "LoadedFillReference",
    "MaskedInputs",
    "ReferenceEvaluationTrace",
    "build_masked_inputs",
    "evaluate_reference_model",
    "evaluate_reference_model_bank",
    "load_fill_reference_artifact",
]
