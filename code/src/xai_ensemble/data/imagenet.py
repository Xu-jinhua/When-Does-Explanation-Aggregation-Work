"""Official ILSVRC2012 splits with a deterministic training-only holdout.

Point XAI_IMAGENET1K_ROOT (or the loader's root argument) at a torchvision
ImageNet directory: train/, val/, and meta.bin, or the official train, val and
devkit archives. The provider never downloads ImageNet. Logical ``test`` is
the labeled official validation set; logical ``validation`` is a 50-per-class
holdout from the official training set. Class indices follow sorted WNIDs.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .specs import HFDatasetSpec


@dataclass(frozen=True)
class ImageNetSplit:
    source: Any
    indices: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        image, label = self.source[source_index]
        filename = Path(self.source.samples[source_index][0]).relative_to(self.source.root)
        return {"image": image, "label": int(label), "filename": filename.as_posix()}

    @property
    def features(self) -> dict[str, Any]:
        @dataclass(frozen=True)
        class Labels:
            names: tuple[str, ...]

        return {"label": Labels(tuple(self.source.wnids))}


@lru_cache(maxsize=2)
def _official_split(root: str, split: str) -> Any:
    from torchvision.datasets import ImageNet

    source = ImageNet(root=root, split=split)
    if len(source.wnids) != 1000 or list(source.wnids) != sorted(source.wnids):
        raise ValueError("ImageNet-1K requires exactly 1000 classes in sorted WNID order")
    return source


@lru_cache(maxsize=2)
def _training_indices(
    root: str, seed: str, count_per_class: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = _official_split(root, "train")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(source.samples):
        by_class[int(label)].append(index)
    holdout = set()
    for label, indices in sorted(by_class.items()):
        if len(indices) <= count_per_class:
            raise ValueError(f"ImageNet holdout exhausts class {label}")

        def key(index: int) -> bytes:
            filename = Path(source.samples[index][0]).relative_to(root).as_posix()
            return hashlib.sha256(f"{seed}\0{filename}".encode()).digest()

        holdout.update(sorted(indices, key=key)[:count_per_class])
    return (
        tuple(index for index in range(len(source)) if index not in holdout),
        tuple(sorted(holdout)),
    )


def load_imagenet_split(
    spec: HFDatasetSpec, split: str, *, root: str | os.PathLike[str] | None = None
) -> ImageNetSplit:
    if spec.provider != "imagenet" or split not in spec.splits:
        raise ValueError(f"Invalid ImageNet provider/split: {spec.provider}/{split}")
    environment = str(spec.provider_options["root_environment"])
    configured = os.environ.get(environment) or root
    if not configured:
        raise ValueError(f"Set {environment} to the official ILSVRC2012 data directory")
    directory = Path(configured).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"ImageNet directory does not exist: {directory}")
    official_split = "val" if split == "test" else "train"
    source = _official_split(str(directory), official_split)
    count_key = "official_validation_size" if split == "test" else "official_train_size"
    if len(source) != int(spec.provider_options[count_key]):
        raise ValueError(f"Incomplete official ImageNet {official_split} split: {len(source)}")
    if split == "test":
        indices = tuple(range(len(source)))
        counts: dict[int, int] = defaultdict(int)
        for label in source.targets:
            counts[int(label)] += 1
        if set(counts) != set(range(1000)) or any(count != 50 for count in counts.values()):
            raise ValueError("Official ImageNet validation requires 50 images per class")
    else:
        train, validation = _training_indices(
            str(directory),
            str(spec.provider_options["holdout_seed"]),
            int(spec.provider_options["holdout_per_class"]),
        )
        indices = train if split == "train" else validation
    result = ImageNetSplit(source, indices)
    spec.validate_split_size(split, len(result))
    return result
