"""Support code for :mod:`xai_ensemble.phase1.compatibility`.

These helpers were relocated verbatim from the retired ``pilots``,
``phase1.generation``, ``phase1.ranking``, ``phase1.attack_generation``, and
``phase1.runner`` modules so that the explainer-compatibility pilot keeps its
exact historical behaviour without those legacy DAG modules.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ATTACK_BATCH_SIZE = 1
@dataclass
class PilotResult:
    name: str
    status: str
    selected: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    config_digest: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def digest(self) -> str:
        value = asdict(self)
        value.pop("created_at", None)
        return object_sha256(value)

    def require_passed(self) -> None:
        if not self.passed:
            raise RuntimeError(f"Pilot {self.name!r} has not passed: {self.failures}")

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, verify_digest: bool = True) -> PilotResult:
        result = cls(
            name=str(value["name"]),
            status=str(value["status"]),
            selected=dict(value.get("selected", {})),
            measurements=dict(value.get("measurements", {})),
            failures=[str(item) for item in value.get("failures", [])],
            config_digest=str(value.get("config_digest", "")),
            created_at=str(value.get("created_at", "")),
        )
        expected = value.get("digest")
        if verify_digest and expected is not None and str(expected) != result.digest:
            raise ValueError(
                f"Pilot result digest mismatch for {result.name!r}: "
                f"stored={expected} computed={result.digest}"
            )
        return result


def save_pilot_result(path: str | Path, result: PilotResult) -> Path:
    value = asdict(result)
    value["digest"] = result.digest
    return atomic_write_json(path, value)


def _resize_to_input(attributions: Any, height: int, width: int) -> Any:
    if tuple(attributions.shape[-2:]) == (height, width):
        return attributions
    import torch.nn.functional as functional

    return functional.interpolate(
        attributions,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


def _special_attack_attribution(
    model: Any,
    method: str,
    inputs: Any,
    targets: Any,
    *,
    create_graph: bool,
) -> Any | None:
    """Graph-preserving variants for the two common input-gradient sources."""

    if method not in {"Saliency", "InputXGradient"}:
        return None
    import torch

    logits = model(inputs)
    selected = logits.gather(1, targets.reshape(-1, 1)).sum()
    gradient = torch.autograd.grad(
        selected,
        inputs,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return gradient.abs() if method == "Saliency" else inputs * gradient


def _validate_patch_shape(height: int, width: int, patch_size: int) -> None:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Input shape {(height, width)} is not divisible by patch_size={patch_size}"
        )


def attribution_to_patch_scores(attributions: Any, patch_size: int) -> Any:
    """Return absolute-mean patch scores for BCHW attributions.

    Torch tensors retain their computation graph. NumPy arrays use an equivalent
    implementation for verification and offline conversion.
    """

    if getattr(attributions, "ndim", None) != 4:
        raise ValueError("attributions must have shape [batch, channels, height, width]")
    batch, channels, height, width = attributions.shape
    _validate_patch_shape(height, width, patch_size)
    if attributions.__class__.__module__.startswith("torch"):
        absolute = attributions.abs().mean(dim=1)
        patches = absolute.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        return patches.mean(dim=(-1, -2))
    values = np.asarray(attributions)
    absolute = np.abs(values).mean(axis=1)
    grid_h, grid_w = height // patch_size, width // patch_size
    return absolute.reshape(batch, grid_h, patch_size, grid_w, patch_size).mean(axis=(2, 4))


def sample_hash_priorities(token_count: int, sample_id: str | bytes) -> np.ndarray:
    """Match Phase 2's deterministic, machine-independent hash priority."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if isinstance(sample_id, str):
        prefix = sample_id.encode("utf-8")
    elif isinstance(sample_id, bytes):
        prefix = sample_id
    else:
        raise TypeError("sample_id must be str or bytes")
    digests = [
        hashlib.sha256(prefix + b":" + str(index).encode("ascii")).digest()
        for index in range(token_count)
    ]
    order = sorted(range(token_count), key=lambda index: (digests[index], index))
    priorities = np.empty(token_count, dtype=np.int64)
    priorities[np.asarray(order, dtype=np.int64)] = np.arange(token_count, dtype=np.int64)
    return priorities


def _tie_priorities(
    batch_size: int,
    token_count: int,
    *,
    tie_policy: str,
    sample_ids: Sequence[str] | None,
) -> np.ndarray:
    if tie_policy == "stable_patch_index":
        return np.broadcast_to(np.arange(token_count), (batch_size, token_count))
    if tie_policy not in {"deterministic_sample_hash", "sample_hash"}:
        raise ValueError(f"Unknown Phase-1 tie policy: {tie_policy}")
    if sample_ids is None or len(sample_ids) != batch_size:
        raise ValueError("deterministic_sample_hash requires one stable sample_id per score row")
    return np.stack([sample_hash_priorities(token_count, sample_id) for sample_id in sample_ids])


def _numpy_stable_ranks(
    scores: np.ndarray,
    base: int,
    *,
    tie_policy: str,
    sample_ids: Sequence[str] | None,
) -> np.ndarray:
    values = np.asarray(scores)
    flat = values.reshape(values.shape[0], -1)
    result = np.empty_like(flat, dtype=np.int32)
    priorities = _tie_priorities(
        flat.shape[0],
        flat.shape[1],
        tie_policy=tie_policy,
        sample_ids=sample_ids,
    )
    for row_index, row in enumerate(flat):
        order = np.lexsort((priorities[row_index], -row))
        result[row_index, order] = np.arange(base, flat.shape[1] + base, dtype=np.int32)
    return result.reshape(values.shape)


def scores_to_ranks(
    scores: Any,
    *,
    base: int = 1,
    tie_policy: str = "stable_patch_index",
    sample_ids: Sequence[str] | None = None,
) -> Any:
    """Rank scores descending with deterministic patch-index tie breaking."""

    if base not in {0, 1}:
        raise ValueError("rank base must be zero or one")
    if getattr(scores, "ndim", None) < 2:
        raise ValueError("scores must include a batch dimension")
    if scores.__class__.__module__.startswith("torch"):
        import torch

        flat = scores.reshape(scores.shape[0], -1)
        priorities = torch.as_tensor(
            _tie_priorities(
                flat.shape[0],
                flat.shape[1],
                tie_policy=tie_policy,
                sample_ids=sample_ids,
            ).copy(),
            device=flat.device,
            dtype=torch.int64,
        )
        secondary = torch.argsort(priorities, dim=1, stable=True)
        primary = torch.argsort(flat.gather(1, secondary), dim=1, descending=True, stable=True)
        order = secondary.gather(1, primary)
        ranks = torch.empty_like(order, dtype=torch.int32)
        values = torch.arange(base, flat.shape[1] + base, device=flat.device, dtype=torch.int32)
        ranks.scatter_(1, order, values.expand_as(order))
        return ranks.reshape(scores.shape)
    return _numpy_stable_ranks(
        np.asarray(scores),
        base,
        tie_policy=tie_policy,
        sample_ids=sample_ids,
    )


def _resolve_path(base: Path, value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise ValueError(f"Generation job field {key!r} cannot be empty")
    return value


@dataclass(frozen=True, slots=True)
class ModelArtifactSpec:
    model_key: str
    num_classes: int
    init_mode: str
    source_model_id: str
    checkpoint_sha256: str
    checkpoint_path: Path | None = None
    strict_checkpoint: bool = True
    factory: str | None = None
    factory_revision: str | None = None
    factory_source_digest: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, base: Path) -> ModelArtifactSpec:
        digest = str(values.get("checkpoint_sha256") or values.get("weights_sha256") or "").lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError("Every formal model requires a locked 64-character weights digest")
        checkpoint_path = _resolve_path(base, values.get("checkpoint_path"))
        init_mode = _required_string(values, "init_mode")
        if init_mode == "checkpoint" and checkpoint_path is None:
            raise ValueError("checkpoint initialization requires checkpoint_path")
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if checkpoint_path is not None and file_sha256(checkpoint_path) != digest:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_path}")
        factory = None if values.get("factory") is None else str(values["factory"])
        factory_revision = (
            None if values.get("factory_revision") is None else str(values["factory_revision"])
        )
        factory_source_digest = (
            None
            if values.get("factory_source_digest") is None
            else str(values["factory_source_digest"])
        )
        if factory_source_digest is not None and not _SHA256.fullmatch(factory_source_digest):
            raise ValueError("factory_source_digest must be a 64-character SHA-256 digest")
        from .relprop import (
            RELPROP_FACTORY_REFERENCES,
            RELPROP_REVISION,
            RELPROP_SOURCE_DIGEST,
        )

        if factory in RELPROP_FACTORY_REFERENCES and (
            factory_revision != RELPROP_REVISION
            or factory_source_digest != RELPROP_SOURCE_DIGEST
        ):
            raise ValueError(
                "RelProp model factory must lock the exact upstream revision and source digest"
            )
        return cls(
            model_key=_required_string(values, "model_key"),
            num_classes=int(values["num_classes"]),
            init_mode=init_mode,
            source_model_id=_required_string(values, "source_model_id"),
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint_path,
            strict_checkpoint=bool(values.get("strict_checkpoint", True)),
            factory=factory,
            factory_revision=factory_revision,
            factory_source_digest=factory_source_digest,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "num_classes": self.num_classes,
            "init_mode": self.init_mode,
            "source_model_id": self.source_model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "factory": self.factory,
            "factory_revision": self.factory_revision,
            "factory_source_digest": self.factory_source_digest,
        }


def _import_factory(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Model factory must use the form 'package.module:callable'")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Configured model factory is not callable: {reference}")
    return factory



def _build_model(spec: ModelArtifactSpec) -> Any:
    if (
        spec.checkpoint_path is not None
        and file_sha256(spec.checkpoint_path) != spec.checkpoint_sha256
    ):
        raise ValueError(f"Checkpoint changed after job validation: {spec.checkpoint_path}")
    if spec.factory is not None:
        return _import_factory(spec.factory)(
            model_key=spec.model_key,
            num_classes=spec.num_classes,
            init_mode=spec.init_mode,
            checkpoint_path=(None if spec.checkpoint_path is None else str(spec.checkpoint_path)),
            strict_checkpoint=spec.strict_checkpoint,
        )
    from xai_ensemble.phase0.models import ModelBuildRequest, create_model

    return create_model(
        ModelBuildRequest(
            model_key=spec.model_key,
            num_classes=spec.num_classes,
            init_mode=spec.init_mode,
            checkpoint_path=(None if spec.checkpoint_path is None else str(spec.checkpoint_path)),
            strict_checkpoint=spec.strict_checkpoint,
        )
    )


def _normalizer(mean: Sequence[float], std: Sequence[float]) -> Callable[[Any], Any]:
    def normalize(images: Any) -> Any:
        import torch

        means = torch.as_tensor(mean, device=images.device, dtype=images.dtype).reshape(1, -1, 1, 1)
        standard_deviations = torch.as_tensor(
            std, device=images.device, dtype=images.dtype
        ).reshape(1, -1, 1, 1)
        return (images - means) / standard_deviations

    return normalize



def _raw_eval_transform(preprocessing: Mapping[str, Any]) -> Any:
    try:
        from torchvision import transforms
    except ImportError as error:  # pragma: no cover - formal GPU dependency
        raise RuntimeError("Formal explanation generation requires torchvision") from error

    interpolation_name = str(preprocessing["interpolation"]).upper()
    interpolation = getattr(transforms.InterpolationMode, interpolation_name, None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation: {interpolation_name}")
    size = int(preprocessing["input_size"])
    resize = int(round(size / float(preprocessing["crop_percentage"])))
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )



def _load_distribution(path: Path, key: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key not in loaded.files:
                raise ValueError(f"Baseline distribution key {key!r} is absent from {path}")
            return np.asarray(loaded[key])
        finally:
            loaded.close()
    return np.asarray(loaded)


def canonical_attribution_map(values: Any) -> Any:
    """Return the signed one-channel map shared by every explainer family.

    Input-gradient explainers commonly return RGB-channel attributions while
    attention/LRP explainers return one channel.  SimpleAvg cannot average
    those objects without a declared common representation.  We preserve sign
    and take the arithmetic channel mean; patch ranking is still computed from
    each method's original attribution using the separately locked absolute
    channel/spatial reduction.
    """

    if getattr(values, "ndim", None) != 4:
        raise ValueError("attributions must have shape [batch,channels,height,width]")
    if int(values.shape[1]) <= 0:
        raise ValueError("attributions must contain at least one channel")
    return values if int(values.shape[1]) == 1 else values.mean(axis=1, keepdims=True)
