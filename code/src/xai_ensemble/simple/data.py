"""Dataset/model loading shared by the two simple experiment phases."""

from __future__ import annotations

import importlib
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.cache import ShardCache
from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.data import get_dataset_spec, read_manifest
from xai_ensemble.phase0.dataset import (
    ManifestIndexedDataset,
    build_raw_image_transform,
    load_hf_split,
)
from xai_ensemble.phase0.models import (
    ModelBuildRequest,
    create_model,
    get_model_definition,
    resolved_preprocessing,
)

from .config import ConditionConfig, DatasetConfig, ModelConfig
from .manifest_identity import dataset_manifest_identity_sha256
from .shared_cache import materialize_shared_tensor

_STREAMING_INPUTS_ENV = "XAI_SIMPLE_STREAMING_INPUTS"
_HOT_CACHE_GIB_ENV = "XAI_SIMPLE_HOT_CACHE_GIB"
_HOT_CACHE_ROOT_ENV = "XAI_SIMPLE_HOT_CACHE_ROOT"
_DEFAULT_HOT_CACHE_BYTES = 12 * 2**30


def streaming_inputs_enabled() -> bool:
    """Return whether full-split materialization is disabled for this worker."""

    return os.environ.get(_STREAMING_INPUTS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def hot_cache_bytes() -> int:
    """Resolve the bounded local hot-cache budget without changing experiment identity."""

    raw = os.environ.get(_HOT_CACHE_GIB_ENV)
    if raw is None:
        return _DEFAULT_HOT_CACHE_BYTES
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{_HOT_CACHE_GIB_ENV} must be a positive number") from error
    if value <= 0:
        raise ValueError(f"{_HOT_CACHE_GIB_ENV} must be positive")
    return int(value * 2**30)


def hot_cache_root(fallback: str | Path) -> Path:
    """Resolve one process-shared cache root for every generated component."""

    configured = os.environ.get(_HOT_CACHE_ROOT_ENV)
    return Path(configured or fallback).expanduser().resolve()


@dataclass(slots=True)
class BatchedTensor:
    """A row-addressable tensor view whose batches are materialized on demand.

    The loader returns CPU tensors.  Keeping the object row-addressable preserves
    the existing Phase 1 prediction/explanation APIs while avoiding a full-split
    CPU tensor and its corresponding tmpfs mmap.
    """

    _shape: tuple[int, ...]
    _loader: Callable[[int, int], Any]
    _max_cached_batches: int = 1
    _cache: OrderedDict[tuple[int, int], Any] = field(init=False, repr=False)
    _cache_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._shape or any(int(value) <= 0 for value in self._shape):
            raise ValueError("BatchedTensor shape must be positive")
        if self._max_cached_batches < 0:
            raise ValueError("BatchedTensor cache size cannot be negative")
        self._cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._cache_lock = threading.Lock()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __len__(self) -> int:
        return int(self._shape[0])

    def batch(self, start: int, stop: int) -> Any:
        if not 0 <= int(start) < int(stop) <= len(self):
            raise IndexError((start, stop))
        key = (int(start), int(stop))
        with self._cache_lock:
            cached = self._cache.pop(key, None)
        if cached is not None:
            with self._cache_lock:
                self._cache[key] = cached
            return cached
        value = self._loader(*key)
        if tuple(int(item) for item in value.shape) != (stop - start, *self._shape[1:]):
            raise ValueError(
                f"BatchedTensor loader returned {tuple(value.shape)}, "
                f"expected {(stop - start, *self._shape[1:])}"
            )
        if self._max_cached_batches:
            with self._cache_lock:
                self._cache[key] = value
                while len(self._cache) > self._max_cached_batches:
                    self._cache.popitem(last=False)
        return value

    def __getitem__(self, key: slice | int) -> Any:
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise ValueError("BatchedTensor only supports contiguous slices")
            start, stop, _ = key.indices(len(self))
            if start == stop:
                import torch

                return torch.empty((0, *self._shape[1:]), dtype=torch.float32)
            return self.batch(start, stop)
        index = int(key)
        if index < 0:
            index += len(self)
        return self.batch(index, index + 1)[0]


@dataclass(slots=True)
class DatasetBundle:
    """A deterministic split and, normally, its raw images cached in CPU RAM."""

    dataset: Any
    raw_images: Any | None
    labels: Any
    indices: Any
    sample_ids: tuple[str, ...]
    row_to_position: Mapping[int, int]
    raw_source_identity_digest: str
    raw_image_shape: tuple[int, ...] | None = None
    raw_batch_cache: ShardCache | None = None
    raw_cache_shard_size: int = 512

    def __post_init__(self) -> None:
        if self.raw_cache_shard_size <= 0:
            raise ValueError("raw_cache_shard_size must be positive")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def batch(self, start: int, stop: int) -> tuple[Any, Any, Any]:
        import torch

        if not 0 <= start < stop <= len(self):
            raise IndexError((start, stop))
        if self.raw_images is None:
            if self.raw_batch_cache is None:
                images = torch.stack(
                    [self.dataset[position]["image"] for position in range(start, stop)]
                )
            else:
                cache = self.raw_batch_cache
                pieces = []
                first_shard = start // self.raw_cache_shard_size
                final_shard = (stop - 1) // self.raw_cache_shard_size
                for shard_index in range(first_shard, final_shard + 1):
                    shard_start = shard_index * self.raw_cache_shard_size
                    shard_stop = min(len(self), shard_start + self.raw_cache_shard_size)
                    key = f"{self.raw_source_identity_digest}:{shard_start}:{shard_stop}.npy"

                    def fetch(
                        destination: Path,
                        *,
                        current_start: int = shard_start,
                        current_stop: int = shard_stop,
                    ) -> None:
                        values = torch.stack(
                            [
                                self.dataset[position]["image"]
                                for position in range(current_start, current_stop)
                            ]
                        ).to(dtype=torch.float32)
                        if (
                            not bool(torch.isfinite(values).all())
                            or float(values.min()) < 0.0
                            or float(values.max()) > 1.0
                        ):
                            raise ValueError(
                                "Dataset transform did not produce finite raw [0,1] images"
                            )
                        with destination.open("wb") as handle:
                            np.save(handle, values.numpy(), allow_pickle=False)

                    try:
                        with cache.use(key, fetch) as path:
                            values = np.load(path, mmap_mode="c", allow_pickle=False)
                    except (OSError, ValueError):
                        cache.remove(key)
                        with cache.use(key, fetch) as path:
                            values = np.load(path, mmap_mode="c", allow_pickle=False)
                    expected_shape = (
                        shard_stop - shard_start,
                        *(self.raw_image_shape or values.shape[1:]),
                    )
                    if tuple(values.shape) != expected_shape or values.dtype != np.dtype(
                        np.float32
                    ):
                        cache.remove(key)
                        raise ValueError(
                            f"Raw batch cache has shape/dtype {values.shape}/{values.dtype}, "
                            f"expected {expected_shape}/float32"
                        )
                    overlap_start = max(start, shard_start) - shard_start
                    overlap_stop = min(stop, shard_stop) - shard_start
                    pieces.append(torch.from_numpy(values[overlap_start:overlap_stop]))
                images = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        else:
            images = self.raw_images[start:stop]
        return images, self.labels[start:stop], self.indices[start:stop]

    def rows(self, row_indices: Any) -> tuple[Any, Any]:
        import torch

        values = torch.as_tensor(row_indices, dtype=torch.int64).tolist()
        try:
            positions = [self.row_to_position[int(value)] for value in values]
        except KeyError as error:
            raise KeyError(
                f"Artifact row index is absent from the configured split: {error}"
            ) from error
        if self.raw_images is None:
            # Phase 2 normally requests one manifest-ordered output shard. Load
            # each contiguous run through ``batch`` so the bounded raw cache is
            # used instead of reopening one provider row at a time.
            run_start = run_stop = positions[0]
            runs: list[tuple[int, int]] = []
            for position in positions[1:]:
                if position == run_stop + 1:
                    run_stop = position
                else:
                    runs.append((run_start, run_stop + 1))
                    run_start = run_stop = position
            runs.append((run_start, run_stop + 1))
            loaded_runs = {run: self.batch(*run)[0] for run in runs}
            position_to_run = {position: run for run in runs for position in range(run[0], run[1])}
            images = torch.stack(
                [
                    loaded_runs[position_to_run[position]][position - position_to_run[position][0]]
                    for position in positions
                ]
            )
        else:
            images = self.raw_images[positions]
        return images, self.labels[positions]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Any
    preprocessing: Mapping[str, Any]
    normalize: Callable[[Any], Any]


def _normalizer(preprocessing: Mapping[str, Any]) -> Callable[[Any], Any]:
    mean_values = tuple(float(item) for item in preprocessing["mean"])
    std_values = tuple(float(item) for item in preprocessing["std"])

    def normalize(images: Any) -> Any:
        import torch

        mean = torch.as_tensor(mean_values, device=images.device, dtype=images.dtype).view(
            1, -1, 1, 1
        )
        std = torch.as_tensor(std_values, device=images.device, dtype=images.dtype).view(
            1, -1, 1, 1
        )
        return (images - mean) / std

    return normalize


def load_model(
    config: ModelConfig,
    *,
    device: str | Any,
    include_checkpoint: bool = True,
) -> LoadedModel:
    """Load the exact experiment model, or the same random architecture for profiling."""

    import torch

    init_mode = config.init_mode if include_checkpoint else "random"
    checkpoint = (
        None
        if not include_checkpoint or config.checkpoint_path is None
        else resolve_full_matrix_runtime_path(config.checkpoint_path)
    )
    request = ModelBuildRequest(
        model_key=config.model_key,
        num_classes=config.num_classes,
        init_mode=init_mode,
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        strict_checkpoint=config.strict_checkpoint,
        disable_inplace_activations=True,
        # Seed the head initialization whenever no checkpoint overwrites it, so
        # that every Phase 1 task builds the identical model in its own
        # process (required for the Phase 2 prediction-alignment check).
        seed=0 if checkpoint is None else None,
        class_index_map=(config.class_index_map if init_mode == "imagenet1k_subset" else None),
    )
    model = create_model(request).to(device=torch.device(device), dtype=torch.float32)
    model.eval()
    preprocessing = resolved_preprocessing(model, get_model_definition(config.model_key))
    return LoadedModel(
        model=model, preprocessing=preprocessing, normalize=_normalizer(preprocessing)
    )


def load_relprop_model(
    config: ModelConfig,
    *,
    method: str,
    device: str | Any,
) -> LoadedModel:
    """Load the exact pinned Chefer implementation required by ``method``."""

    import torch

    from xai_ensemble.phase1.relprop import create_relprop_model_for_method

    model = create_relprop_model_for_method(
        method,
        model_key=config.model_key,
        num_classes=config.num_classes,
        init_mode=config.init_mode,
        checkpoint_path=(
            None
            if config.checkpoint_path is None
            else str(resolve_full_matrix_runtime_path(config.checkpoint_path))
        ),
        strict_checkpoint=config.strict_checkpoint,
    ).to(device=torch.device(device), dtype=torch.float32)
    model.eval()
    preprocessing = resolved_preprocessing(model, get_model_definition(config.model_key))
    return LoadedModel(
        model=model,
        preprocessing=preprocessing,
        normalize=_normalizer(preprocessing),
    )


def load_split(
    dataset_config: DatasetConfig,
    model: LoadedModel,
    *,
    split: str,
    workers: int,
    shared_cache_root: str | Path | None = None,
) -> DatasetBundle:
    """Load the manifest-ordered raw [0,1] split and optionally cache it in RAM."""

    import torch
    from torch.utils.data import DataLoader

    manifest_path = resolve_full_matrix_runtime_path(dataset_config.manifest_path)
    cache_directory = (
        None
        if dataset_config.cache_directory is None
        else resolve_full_matrix_runtime_path(dataset_config.cache_directory)
    )
    manifest = read_manifest(manifest_path)
    if dataset_config.registry_key is not None:
        spec = get_dataset_spec(dataset_config.registry_key)
        manifest.validate(spec, require_expected_counts=False)
        image_column = spec.image_column
        label_column = spec.label_column
    else:
        assert dataset_config.provider_factory is not None
        spec = None
        image_column = dataset_config.image_column
        label_column = dataset_config.label_column
    records = tuple(sorted(manifest.records_for_split(split), key=lambda item: item.row_index))
    if dataset_config.max_samples is not None:
        records = records[: dataset_config.max_samples]
    if not records:
        raise ValueError(f"No manifest records for {dataset_config.dataset_id}/{split}")
    transform = build_raw_image_transform(
        get_model_definition(model.model.model_key),
        preprocessing=model.preprocessing,
    )
    indexed = None

    def indexed_dataset() -> Any:
        nonlocal indexed
        if indexed is not None:
            return indexed
        if spec is not None:
            provider = load_hf_split(
                spec,
                split,
                cache_dir=cache_directory,
                keep_in_memory=dataset_config.keep_provider_in_memory,
            )
        else:
            assert dataset_config.provider_factory is not None
            provider = resolve_factory(dataset_config.provider_factory)(
                split=split,
                cache_directory=cache_directory,
                keep_in_memory=dataset_config.keep_provider_in_memory,
                **dict(dataset_config.provider_kwargs),
            )
        indexed = ManifestIndexedDataset(
            provider,
            records,
            image_column=image_column,
            label_column=label_column,
            transform=transform,
        )
        return indexed

    labels = torch.as_tensor([item.label for item in records], dtype=torch.int64)
    indices = torch.as_tensor([item.row_index for item in records], dtype=torch.int64)
    expected_shape = (
        len(records),
        3,
        int(model.preprocessing["input_size"]),
        int(model.preprocessing["input_size"]),
    )
    raw_source_identity = {
        "schema": "simple-raw-split-mmap-v1",
        "dataset_id": dataset_config.dataset_id,
        "registry_key": dataset_config.registry_key,
        "provider_factory": dataset_config.provider_factory,
        "provider_kwargs": dict(dataset_config.provider_kwargs),
        "manifest_sha256": dataset_manifest_identity_sha256(dataset_config.manifest_path),
        "split": split,
        "model_key": model.model.model_key,
        "rows_digest": object_sha256(
            [
                {
                    "row_index": item.row_index,
                    "sample_id": item.sample_id,
                    "label": item.label,
                }
                for item in records
            ]
        ),
        "preprocessing": dict(model.preprocessing),
        "transform": "build_raw_image_transform-v1",
        "shape": list(expected_shape),
    }
    raw_source_identity_digest = object_sha256(raw_source_identity)
    images = None
    cache_images_in_ram = dataset_config.cache_images_in_ram and not streaming_inputs_enabled()
    raw_batch_cache = None
    if streaming_inputs_enabled() and shared_cache_root is not None:
        raw_batch_cache = ShardCache(
            hot_cache_root(Path(shared_cache_root) / "bounded-hot-cache"),
            max_bytes=hot_cache_bytes(),
        )
    if cache_images_in_ram:

        def batches():
            return DataLoader(
                indexed_dataset(),
                batch_size=512,
                shuffle=False,
                num_workers=workers,
                pin_memory=False,
                persistent_workers=workers > 0,
            )

        if shared_cache_root is None:
            chunks = [batch["image"].to(dtype=torch.float32) for batch in batches()]
            images = torch.cat(chunks, dim=0)
        else:

            def populate(destination: Any) -> None:
                offset = 0
                for batch in batches():
                    values = batch["image"].to(dtype=torch.float32).numpy()
                    destination[offset : offset + len(values)] = values
                    offset += len(values)
                if offset != len(records):
                    raise RuntimeError("Dataset loader did not fill the shared split cache")

            def validate(values: np.ndarray) -> None:
                if (
                    not np.all(np.isfinite(values))
                    or float(np.min(values)) < 0.0
                    or float(np.max(values)) > 1.0
                ):
                    raise ValueError("Dataset transform did not produce finite raw [0,1] images")

            shared = materialize_shared_tensor(
                shared_cache_root,
                namespace="datasets",
                identity=raw_source_identity,
                shape=expected_shape,
                dtype=np.float32,
                populate=populate,
                validate=validate,
            )
            images = shared.tensor
            print(
                "SHARED_DATASET_CACHE "
                f"status={'hit' if shared.cache_hit else 'built'} "
                f"dataset={dataset_config.dataset_id} split={split} "
                f"gib={images.numel() * images.element_size() / 2**30:.2f} "
                f"seconds={shared.elapsed_seconds:.3f}",
                flush=True,
            )
        if tuple(images.shape) != expected_shape:
            raise RuntimeError(f"Unexpected cached image shape: {tuple(images.shape)}")
        if shared_cache_root is None:
            if (
                not bool(torch.isfinite(images).all())
                or float(images.min()) < 0.0
                or float(images.max()) > 1.0
            ):
                raise ValueError("Dataset transform did not produce finite raw [0,1] images")
    else:
        indexed_dataset()
    return DatasetBundle(
        dataset=indexed,
        raw_images=images,
        labels=labels,
        indices=indices,
        sample_ids=tuple(item.sample_id for item in records),
        row_to_position={item.row_index: position for position, item in enumerate(records)},
        raw_source_identity_digest=raw_source_identity_digest,
        raw_image_shape=expected_shape[1:],
        raw_batch_cache=raw_batch_cache,
    )


def _load_array(path: Path, key: str) -> np.ndarray:
    """Load a raw train-mean tensor from the accepted lightweight formats."""

    source = path
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if manifest_path.is_file():
            from xai_ensemble.core.manifest import load_manifest

            manifest = load_manifest(manifest_path)
            candidates = [entry.path for entry in manifest.files]
            preferred = [item for item in candidates if Path(item).name == "means.npz"]
            if not preferred:
                preferred = [
                    item
                    for item in candidates
                    if Path(item).suffix.lower() in {".npz", ".npy", ".pt", ".safetensors"}
                ]
            if not preferred:
                raise ValueError(f"No mean payload found in {manifest_path}")
            source = source / preferred[0]
        else:
            candidates = tuple(source.glob("means.*"))
            if len(candidates) != 1:
                raise ValueError(f"Cannot resolve one mean payload below {source}")
            source = candidates[0]
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f"{source} does not contain {key!r}")
            return np.asarray(archive[key])
    if suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if suffix == ".pt":
        import torch

        try:
            value = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(source, map_location="cpu")
        if isinstance(value, Mapping):
            value = value[key]
        return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    if suffix == ".safetensors":
        from safetensors.numpy import load_file

        values = load_file(source)
        if key not in values:
            raise KeyError(f"{source} does not contain {key!r}")
        return np.asarray(values[key])
    raise ValueError(f"Unsupported dataset mean format: {source}")


def load_raw_dataset_mean(config: ModelConfig, *, input_size: int) -> Any:
    import torch

    values = _load_array(
        resolve_full_matrix_runtime_path(config.mean_path), config.mean_key
    ).astype(np.float32, copy=False)
    if values.shape != (3, input_size, input_size):
        raise ValueError(
            f"Model-bound dataset mean must have shape "
            f"(3,{input_size},{input_size}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("Dataset mean must be finite and in raw [0,1] space")
    return torch.from_numpy(values.copy()).unsqueeze(0)


def load_raw_class_means(config: ModelConfig, *, input_size: int) -> tuple[Any, Any]:
    """Load the registered train-derived class means and their sample counts."""

    import torch

    mean_path = resolve_full_matrix_runtime_path(config.mean_path)
    means = _load_array(mean_path, "class_means").astype(np.float32, copy=False)
    counts = _load_array(mean_path, "class_counts")
    expected = (config.num_classes, 3, input_size, input_size)
    if means.shape != expected:
        raise ValueError(f"Class means must have shape {expected}, got {means.shape}")
    if counts.shape != (config.num_classes,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Class counts must be an integer vector aligned with class means")
    if np.any(counts <= 0):
        raise ValueError("Every configured class must occur in the training mean artifact")
    if not np.all(np.isfinite(means)) or float(means.min()) < 0.0 or float(means.max()) > 1.0:
        raise ValueError("Class means must be finite and in raw [0,1] space")
    return torch.from_numpy(means.copy()), torch.from_numpy(counts.astype(np.int64, copy=True))


def resolve_factory(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Factory must use module:function syntax, found {reference!r}")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Condition factory {reference!r} is not callable")
    return value


def apply_condition(
    condition: ConditionConfig,
    raw_images: Any,
    *,
    labels: Any,
    indices: Any,
    model: Any,
    normalize: Callable[[Any], Any],
    seed: int,
) -> Any:
    """Return conditioned raw images; clean is an exact no-op."""

    import torch

    if condition.kind == "clean":
        return raw_images
    if condition.kind == "adversarial":
        raise ValueError(
            "Adversarial conditions must load their immutable saved dataset; "
            "they cannot be regenerated as a condition factory"
        )
    assert condition.factory is not None
    factory = resolve_factory(condition.factory)
    # Invoke a custom condition one sample at a time with a row-derived seed.
    # This makes the scientific input independent of prediction/explanation/
    # evaluation batch sizes and of resume boundaries.
    pieces = []
    for position in range(int(raw_images.shape[0])):
        sample_index = int(indices[position].item())
        factory_kwargs = dict(condition.kwargs)
        seed_group = str(factory_kwargs.pop("seed_group", condition.condition_id))
        if not seed_group:
            raise ValueError("condition seed_group must be non-empty")
        derived_seed = int(stable_seed(seed, seed_group, sample_index))
        piece = factory(
            raw_images=raw_images[position : position + 1],
            labels=labels[position : position + 1],
            indices=indices[position : position + 1],
            model=model,
            normalize=normalize,
            seed=derived_seed,
            **factory_kwargs,
        )
        pieces.append(torch.as_tensor(piece, device=raw_images.device, dtype=torch.float32))
    result = torch.cat(pieces, dim=0)
    if result.shape != raw_images.shape:
        raise ValueError(
            f"Condition {condition.condition_id} returned {tuple(result.shape)}, "
            f"expected {tuple(raw_images.shape)}"
        )
    if (
        not bool(torch.isfinite(result).all())
        or float(result.min()) < 0.0
        or float(result.max()) > 1.0
    ):
        raise ValueError("Condition factory must return finite raw [0,1] images")
    return result


def materialize_shared_conditioned_raw_images(
    bundle: DatasetBundle,
    loaded_model: LoadedModel,
    condition: ConditionConfig,
    *,
    device: str | Any,
    batch_size: int,
    seed: int,
    shared_cache_root: str | Path,
) -> Any | None:
    """Return a shared mmap for deterministic natural corruptions when supported."""

    import torch

    if streaming_inputs_enabled():
        return None
    if condition.kind == "clean":
        return bundle.raw_images
    if condition.kind != "factory":
        return None
    if condition.factory != "xai_ensemble.simple.conditions:natural_corruption":
        # An arbitrary factory may depend on model parameters. Keep the generic
        # execution path instead of assuming that its output is model-agnostic.
        return None
    if bundle.raw_images is None:
        return None
    identity = {
        "schema": "simple-conditioned-raw-mmap-v1",
        "raw_source_identity_digest": bundle.raw_source_identity_digest,
        "rows_digest": object_sha256(
            {
                "indices": [int(value) for value in bundle.indices.tolist()],
                "labels": [int(value) for value in bundle.labels.tolist()],
            }
        ),
        "raw_shape": list(bundle.raw_images.shape),
        "preprocessing": dict(loaded_model.preprocessing),
        "condition": {
            "id": condition.condition_id,
            "factory": condition.factory,
            "kwargs": dict(condition.kwargs),
        },
        "seed": int(seed),
        "condition_application": "per-row-stable-seed-v1",
    }
    target_device = torch.device(device)

    def populate(destination: Any) -> None:
        for start in range(0, len(bundle), batch_size):
            stop = min(len(bundle), start + batch_size)
            raw, labels, indices = bundle.batch(start, stop)
            if target_device.type == "cuda" and not raw.is_pinned():
                raw = raw.pin_memory()
                labels = labels.pin_memory()
                indices = indices.pin_memory()
            conditioned = apply_condition(
                condition,
                raw.to(target_device, dtype=torch.float32, non_blocking=True),
                labels=labels.to(target_device, non_blocking=True),
                indices=indices.to(target_device, non_blocking=True),
                model=loaded_model.model,
                normalize=loaded_model.normalize,
                seed=seed,
            )
            destination[start:stop] = conditioned.detach().to("cpu").numpy()

    def validate(values: np.ndarray) -> None:
        if (
            not np.all(np.isfinite(values))
            or float(np.min(values)) < 0.0
            or float(np.max(values)) > 1.0
        ):
            raise ValueError("Condition cache contains values outside finite raw [0,1]")

    shared = materialize_shared_tensor(
        shared_cache_root,
        namespace="conditions",
        identity=identity,
        shape=tuple(int(value) for value in bundle.raw_images.shape),
        dtype=np.float32,
        populate=populate,
        validate=validate,
    )
    print(
        "SHARED_CONDITION_CACHE "
        f"status={'hit' if shared.cache_hit else 'built'} "
        f"condition={condition.condition_id} "
        f"gib={shared.tensor.numel() * shared.tensor.element_size() / 2**30:.2f} "
        f"seconds={shared.elapsed_seconds:.3f}",
        flush=True,
    )
    return shared.tensor


def materialize_model_inputs(
    bundle: DatasetBundle,
    loaded_model: LoadedModel,
    condition: ConditionConfig,
    *,
    device: str | Any,
    batch_size: int,
    seed: int,
    shared_cache_root: str | Path | None = None,
) -> Any:
    """Cache the exact conditioned, normalized FP32 inputs back in CPU RAM."""

    import torch

    target_device = torch.device(device)
    if streaming_inputs_enabled():
        input_shape = (
            3,
            int(loaded_model.preprocessing["input_size"]),
            int(loaded_model.preprocessing["input_size"]),
        )
        cacheable = condition.kind == "clean" or (
            condition.kind == "factory"
            and condition.factory == "xai_ensemble.simple.conditions:natural_corruption"
        )
        cache = (
            None
            if shared_cache_root is None or not cacheable
            else ShardCache(
                hot_cache_root(Path(shared_cache_root) / "bounded-hot-cache"),
                max_bytes=hot_cache_bytes(),
            )
        )
        # Clean input normalization is device-independent. Factory conditions
        # retain the target-device execution used by the original path so a
        # resumed task cannot generate different random corruption tensors.
        condition_device = torch.device("cpu") if condition.kind == "clean" else target_device
        input_identity_digest = object_sha256(
            {
                "schema": "simple-normalized-input-hot-cache-v1",
                "raw_source_identity_digest": bundle.raw_source_identity_digest,
                "preprocessing": dict(loaded_model.preprocessing),
                "condition": {
                    "id": condition.condition_id,
                    "kind": condition.kind,
                    "factory": condition.factory,
                    "kwargs": dict(condition.kwargs),
                },
                "seed": int(seed),
                "condition_application": "per-row-stable-seed-v1",
            }
        )

        def compute_batch(start: int, stop: int) -> Any:
            raw, labels, indices = bundle.batch(start, stop)
            if target_device.type == "cuda" and not raw.is_pinned():
                raw = raw.pin_memory()
                labels = labels.pin_memory()
                indices = indices.pin_memory()
            raw_device = raw.to(condition_device, dtype=torch.float32, non_blocking=True)
            conditioned = apply_condition(
                condition,
                raw_device,
                labels=labels.to(condition_device, non_blocking=True),
                indices=indices.to(condition_device, non_blocking=True),
                model=loaded_model.model,
                normalize=loaded_model.normalize,
                seed=seed,
            )
            return loaded_model.normalize(conditioned).detach().to("cpu", dtype=torch.float32)

        def load_batch(start: int, stop: int) -> Any:
            if cache is None:
                return compute_batch(start, stop)
            pieces = []
            shard_size = bundle.raw_cache_shard_size
            first_shard = start // shard_size
            final_shard = (stop - 1) // shard_size
            for shard_index in range(first_shard, final_shard + 1):
                shard_start = shard_index * shard_size
                shard_stop = min(len(bundle), shard_start + shard_size)
                key = f"inputs:{input_identity_digest}:{shard_start}:{shard_stop}.npy"

                def fetch(
                    destination: Path,
                    *,
                    current_start: int = shard_start,
                    current_stop: int = shard_stop,
                ) -> None:
                    values = compute_batch(current_start, current_stop)
                    with destination.open("wb") as handle:
                        np.save(handle, values.numpy(), allow_pickle=False)

                try:
                    with cache.use(key, fetch) as path:
                        values = np.load(path, mmap_mode="c", allow_pickle=False)
                except (OSError, ValueError):
                    cache.remove(key)
                    with cache.use(key, fetch) as path:
                        values = np.load(path, mmap_mode="c", allow_pickle=False)
                expected_shape = (shard_stop - shard_start, *input_shape)
                if tuple(values.shape) != expected_shape or values.dtype != np.dtype(np.float32):
                    cache.remove(key)
                    raise ValueError(
                        f"Input hot cache has shape/dtype {values.shape}/{values.dtype}, "
                        f"expected {expected_shape}/float32"
                    )
                overlap_start = max(start, shard_start) - shard_start
                overlap_stop = min(stop, shard_stop) - shard_start
                # Copy while ``cache.use`` pins the file against eviction;
                # callers may consume this batch asynchronously afterwards.
                pieces.append(torch.from_numpy(values[overlap_start:overlap_stop]).clone())
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

        return BatchedTensor((len(bundle), *input_shape), load_batch)

    shared_conditioned = (
        None
        if shared_cache_root is None
        else materialize_shared_conditioned_raw_images(
            bundle,
            loaded_model,
            condition,
            device=target_device,
            batch_size=batch_size,
            seed=seed,
            shared_cache_root=shared_cache_root,
        )
    )
    chunks = []
    for start in range(0, len(bundle), batch_size):
        stop = min(len(bundle), start + batch_size)
        raw, labels, indices = bundle.batch(start, stop)
        if shared_conditioned is not None:
            raw = shared_conditioned[start:stop]
        if target_device.type == "cuda" and not raw.is_pinned():
            raw = raw.pin_memory()
        raw = raw.to(target_device, dtype=torch.float32, non_blocking=True)
        conditioned = (
            raw
            if shared_conditioned is not None
            else apply_condition(
                condition,
                raw,
                labels=labels.to(target_device, non_blocking=True),
                indices=indices.to(target_device, non_blocking=True),
                model=loaded_model.model,
                normalize=loaded_model.normalize,
                seed=seed,
            )
        )
        normalized = loaded_model.normalize(conditioned)
        chunks.append(normalized.detach().to(device="cpu", dtype=torch.float32))
    return torch.cat(chunks, dim=0)


def materialize_clean_model_inputs(
    bundle: DatasetBundle,
    loaded_model: LoadedModel,
    *,
    batch_size: int,
) -> Any:
    """Cache normalized clean inputs without a GPU or condition factory."""

    import torch

    if streaming_inputs_enabled():
        target_shape = (
            3,
            int(loaded_model.preprocessing["input_size"]),
            int(loaded_model.preprocessing["input_size"]),
        )

        def load_batch(start: int, stop: int) -> Any:
            raw, _, _ = bundle.batch(start, stop)
            return loaded_model.normalize(raw.to(dtype=torch.float32)).to(
                device="cpu", dtype=torch.float32
            )

        return BatchedTensor((len(bundle), *target_shape), load_batch)

    chunks = []
    for start in range(0, len(bundle), batch_size):
        stop = min(len(bundle), start + batch_size)
        raw, _, _ = bundle.batch(start, stop)
        chunks.append(
            loaded_model.normalize(raw.to(dtype=torch.float32)).to(
                device="cpu", dtype=torch.float32
            )
        )
    return torch.cat(chunks, dim=0)


def fixed_baselines(
    raw_mean: Any,
    loaded_model: LoadedModel,
    *,
    device: str | Any,
    seed: int,
) -> tuple[Any, Any]:
    """Build batch-independent ESANN zero/Gaussian/dataset-mean baselines.

    Zero and Gaussian are in model-input space, matching the historical ESANN
    implementation.  The registered raw train mean is normalized into that
    same space.  The distribution always contains exactly three images and is
    therefore independent of the explanation batch size.
    """

    import torch

    target = torch.device(device)
    mean = loaded_model.normalize(raw_mean.to(target, dtype=torch.float32))
    zero = torch.zeros_like(mean)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    gaussian = torch.randn(mean.shape, generator=generator, dtype=torch.float32).to(target)
    distribution = torch.cat((zero, gaussian, mean), dim=0)
    return zero, distribution


__all__ = [
    "BatchedTensor",
    "DatasetBundle",
    "LoadedModel",
    "apply_condition",
    "fixed_baselines",
    "hot_cache_bytes",
    "hot_cache_root",
    "load_model",
    "load_raw_class_means",
    "load_raw_dataset_mean",
    "load_split",
    "materialize_clean_model_inputs",
    "materialize_model_inputs",
    "materialize_shared_conditioned_raw_images",
    "resolve_factory",
    "streaming_inputs_enabled",
]
