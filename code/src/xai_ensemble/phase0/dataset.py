"""Runtime adapters from a pinned HF split and manifest to PyTorch loaders."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.data.manifest import DatasetManifest, ManifestRecord
from xai_ensemble.data.partitions import SourcePartition
from xai_ensemble.data.specs import HFDatasetSpec

from .config import LoaderConfig
from .models import ModelDefinition


def load_hf_split(
    spec: HFDatasetSpec,
    split: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    keep_in_memory: bool = False,
    token: str | bool | None = None,
) -> Any:
    """Load a random-access split at the exact registered provider revision."""

    if split not in spec.splits:
        raise ValueError(f"Unknown split {split!r}")
    if spec.provider == "imagenet":
        from xai_ensemble.data.imagenet import load_imagenet_split

        return load_imagenet_split(spec, split, root=cache_dir)
    if spec.provider == "medmnist":
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Loading DermaMNIST requires the 'medmnist' package") from exc
        required_version = str(spec.provider_options["version"])
        installed_version = importlib.metadata.version("medmnist")
        if installed_version != required_version:
            raise RuntimeError(
                f"MedMNIST version mismatch: installed={installed_version}, "
                f"required={required_version}"
            )
        data_flag = str(spec.provider_options["data_flag"])
        info = INFO[data_flag]
        data_class = getattr(medmnist, info["python_class"])
        resolved_cache_dir = (
            None if cache_dir is None else resolve_full_matrix_runtime_path(cache_dir)
        )
        verified_local_source = None
        if resolved_cache_dir is not None and {
            "filename",
            "source_url",
            "md5",
            "zenodo_record",
        }.issubset(spec.provider_options):
            from xai_ensemble.data.medmnist_recovery import (
                ensure_verified_medmnist_source,
                has_verified_source_contract,
            )

            if has_verified_source_contract(spec):
                verified_local_source = ensure_verified_medmnist_source(
                    spec, resolved_cache_dir, download=True
                )
        kwargs: dict[str, Any] = {
            "split": "val" if split == "validation" else split,
            "download": verified_local_source is None,
            "as_rgb": bool(spec.provider_options.get("as_rgb", True)),
        }
        if "size" in inspect.signature(data_class.__init__).parameters:
            kwargs["size"] = int(spec.provider_options.get("size", 224))
        if resolved_cache_dir is not None:
            root = Path(resolved_cache_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            kwargs["root"] = str(root)
        dataset = data_class(**kwargs)
        spec.validate_split_size(split, len(dataset))
        return dataset
    if spec.provider == "composite_huggingface":
        from xai_ensemble.data.composite import load_composite_split

        dataset = load_composite_split(
            spec,
            split,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            token=token,
        )
        spec.validate_split_size(split, len(dataset))
        return dataset
    if spec.provider != "huggingface":
        raise ValueError(f"Unsupported dataset provider: {spec.provider}")
    from xai_ensemble.data.hf_parquet_recovery import (
        load_hf_parquet_recovery_source,
        recovery_source_for_config,
    )

    recovery_config = {
        "dataset_id": spec.dataset_id,
        "revision": spec.revision,
        "split": split,
        "image_column": spec.image_column,
        "label_column": spec.label_column,
        "text_column": spec.text_column,
    }
    if recovery_source_for_config(recovery_config) is not None:
        dataset = load_hf_parquet_recovery_source(recovery_config)
        spec.validate_split_size(split, len(dataset))
        return dataset
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Loading Hugging Face data requires the 'datasets' package") from exc
    kwargs: dict[str, Any] = {
        "path": spec.dataset_id,
        "revision": spec.revision,
        "split": split,
        "keep_in_memory": keep_in_memory,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    if token is not None:
        kwargs["token"] = token
    dataset = load_dataset(**kwargs)
    spec.validate_split_size(split, len(dataset))
    return dataset


def build_image_transform(
    definition: ModelDefinition,
    *,
    training: bool,
    preprocessing: Mapping[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """Create ImageNet augmentation/evaluation transforms lazily."""

    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Image transforms require torchvision") from exc

    values = dict(preprocessing or {})
    size = int(values.get("input_size", definition.input_size))
    mean = tuple(values.get("mean", definition.mean))
    std = tuple(values.get("std", definition.std))
    interpolation_name = str(values.get("interpolation", definition.interpolation)).upper()
    interpolation = getattr(transforms.InterpolationMode, interpolation_name, None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode: {interpolation_name}")

    if training:
        operations = [
            transforms.RandomResizedCrop(size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        crop_percentage = float(values.get("crop_percentage", definition.crop_percentage))
        resize_size = int(round(size / crop_percentage))
        operations = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(size),
        ]
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(operations)


def build_raw_image_transform(
    definition: ModelDefinition,
    *,
    preprocessing: Mapping[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """Create deterministic evaluation geometry without normalization.

    Phase 1 corruptions and Phase 2 masking are defined in raw ``[0, 1]``
    image space. This transform applies the exact resize/crop geometry used by
    the model but deliberately stops after ``ToTensor``.
    """

    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Image transforms require torchvision") from exc

    values = dict(preprocessing or {})
    size = int(values.get("input_size", definition.input_size))
    interpolation_name = str(values.get("interpolation", definition.interpolation)).upper()
    interpolation = getattr(transforms.InterpolationMode, interpolation_name, None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode: {interpolation_name}")
    crop_percentage = float(values.get("crop_percentage", definition.crop_percentage))
    resize_size = int(round(size / crop_percentage))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )


class ManifestIndexedDataset:
    """A map-style dataset whose public identity comes from the manifest."""

    def __init__(
        self,
        hf_dataset: Any,
        records: Sequence[ManifestRecord],
        *,
        image_column: str,
        label_column: str,
        transform: Callable[[Any], Any] | None,
    ) -> None:
        self._dataset = hf_dataset
        self.records = tuple(records)
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform
        if len({record.sample_id for record in self.records}) != len(self.records):
            raise ValueError("Dataset subset contains duplicate sample IDs")
        for record in self.records:
            if record.row_index >= len(hf_dataset):
                raise IndexError(
                    f"Manifest row {record.row_index} exceeds dataset length {len(hf_dataset)}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        row = self._dataset[record.row_index]
        if isinstance(row, Mapping):
            image = row[self.image_column]
            observed_label_value = row[self.label_column]
        else:
            image, observed_label_value = row[:2]
        if hasattr(observed_label_value, "item"):
            observed_label_value = observed_label_value.item()
        observed_label = int(observed_label_value)
        if observed_label != record.label:
            raise RuntimeError(
                f"Pinned row label changed at {record.split}[{record.row_index}]: "
                f"manifest={record.label}, dataset={observed_label}"
            )
        if getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": record.label,
            "sample_id": record.sample_id,
            "row_index": record.row_index,
        }

    def row_group_key(self, index: int) -> Any:
        """Forward an optional raw-source row-group key for training locality."""

        provider = getattr(self._dataset, "row_group_key", None)
        if not callable(provider):
            raise AttributeError("Dataset source has no Parquet row-group locality")
        return provider(self.records[index].row_index)


class DistributedEvalSampler:
    """Shard evaluation without DistributedSampler's duplicate padding."""

    def __init__(self, dataset: Any, *, rank: int, world_size: int) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Require world_size > 0 and rank in [0, world_size)")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Any:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


class RowGroupLocalDistributedSampler:
    """Deterministically shuffle Parquet row groups while retaining all rows.

    Large direct-Parquet sources cannot use a fully random index permutation:
    each image access otherwise pulls a multi-megabyte row group and evicts
    the previous group before another row can reuse it.  This sampler retains
    DistributedSampler's epoch seed, padding, and rank-sharding behavior, but
    permutes groups first and keeps each group's rows adjacent.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Require num_replicas > 0 and rank in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._group_keys, self._groups = self._build_groups(dataset)
        size = len(dataset)
        if drop_last and size % num_replicas:
            self.num_samples = math.ceil((size - num_replicas) / num_replicas)
        else:
            self.num_samples = math.ceil(size / num_replicas)
        self.total_size = self.num_samples * num_replicas

    @staticmethod
    def _build_groups(dataset: Any) -> tuple[tuple[Any, ...], tuple[tuple[int, ...], ...]]:
        key_for_index = getattr(dataset, "row_group_key", None)
        if not callable(key_for_index):
            raise TypeError("Row-group-local sampling requires dataset.row_group_key(index)")
        grouped: dict[Any, list[int]] = {}
        for index in range(len(dataset)):
            key = key_for_index(index)
            try:
                grouped.setdefault(key, []).append(index)
            except TypeError as error:
                raise TypeError("Dataset row_group_key values must be hashable") from error
        return tuple(grouped), tuple(tuple(indices) for indices in grouped.values())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _within_group_order(self, key: Any, indices: tuple[int, ...]) -> tuple[int, ...]:
        if len(indices) < 2:
            return indices
        payload = f"row-group-local-v1\0{self.seed}\0{self.epoch}\0{key!r}".encode()
        digest = hashlib.sha256(payload).digest()
        offset = int.from_bytes(digest[:8], "big") % len(indices)
        ordered = indices[offset:] + indices[:offset]
        return ordered[::-1] if digest[8] & 1 else ordered

    def __iter__(self) -> Any:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - DataLoader already requires torch
            raise RuntimeError("Row-group-local sampling requires PyTorch") from error

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        group_order = torch.randperm(len(self._groups), generator=generator).tolist()
        indices: list[int] = []
        for group_index in group_order:
            indices.extend(
                self._within_group_order(
                    self._group_keys[group_index],
                    self._groups[group_index],
                )
            )
        if not self.drop_last:
            padding = self.total_size - len(indices)
            if padding > 0:
                if not indices:
                    raise RuntimeError("Cannot pad an empty row-group-local dataset")
                indices.extend((indices * math.ceil(padding / len(indices)))[:padding])
        else:
            indices = indices[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


def _row_group_local_sampler(
    dataset: Any,
    *,
    num_replicas: int,
    rank: int,
    seed: int,
    drop_last: bool,
) -> RowGroupLocalDistributedSampler | None:
    """Construct a locality sampler only for direct-Parquet backed views."""

    if not len(dataset) or not callable(getattr(dataset, "row_group_key", None)):
        return None
    try:
        return RowGroupLocalDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            drop_last=drop_last,
        )
    except AttributeError:
        return None


def records_for_source(
    manifest: DatasetManifest,
    source: SourcePartition,
    *,
    split: str,
) -> tuple[ManifestRecord, ...]:
    records = {record.sample_id: record for record in manifest.records_for_split(split)}
    missing = set(source.sample_ids) - set(records)
    if missing:
        raise ValueError(f"Source references samples absent from {split}: {sorted(missing)[:3]}")
    return tuple(records[sample_id] for sample_id in source.sample_ids)


def make_worker_init_fn(base_seed: int, rank: int = 0) -> Callable[[int], None]:
    """Return a deterministic DataLoader worker initializer."""

    def initialize(worker_id: int) -> None:
        payload = f"worker-v1\0{base_seed}\0{rank}\0{worker_id}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:  # pragma: no cover - NumPy is a base dependency
            pass
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover - worker only exists with torch
            pass

    return initialize


def build_dataloader(
    dataset: Any,
    config: LoaderConfig,
    *,
    training: bool,
    seed: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    sampler_override: Any | None = None,
) -> Any:
    """Build a seeded single-process or DDP map-style DataLoader."""

    try:
        import torch
        from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Building a DataLoader requires PyTorch") from exc

    generator = torch.Generator()
    generator.manual_seed(seed)
    if sampler_override is not None:
        if not training:
            raise ValueError("sampler_override is only valid for training loaders")
        if distributed:
            raise ValueError("sampler_override is not supported with distributed training")
        sampler = sampler_override
    elif distributed:
        if training:
            sampler = _row_group_local_sampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                drop_last=config.drop_last,
            )
            if sampler is None:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=seed,
                    drop_last=config.drop_last,
                )
        else:
            sampler = DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
    elif training:
        # DistributedSampler with one replica gives the single-GPU path the
        # same epoch-addressable shuffle as DDP, which makes resume order
        # independent of a DataLoader generator's previously consumed state.
        sampler = _row_group_local_sampler(
            dataset,
            num_replicas=1,
            rank=0,
            seed=seed,
            drop_last=config.drop_last,
        )
        if sampler is None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=seed,
                drop_last=config.drop_last,
            )
    else:
        sampler = SequentialSampler(dataset)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last and training,
        "worker_init_fn": make_worker_init_fn(seed, rank),
        "generator": generator,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)
