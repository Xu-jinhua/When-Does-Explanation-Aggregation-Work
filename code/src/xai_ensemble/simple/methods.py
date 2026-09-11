"""The fixed ESANN explainer settings used by the simplified experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.phase1.explainers import EXPLAINER_SPECS

Architecture = Literal["cnn", "vit"]

PAPER_CNN_METHODS = (
    "Saliency",
    "InputXGradient",
    "IntegratedGradients",
    "GuidedBackprop",
    "Deconvolution",
    "FeatureAblation",
    "Occlusion",
    "DeepLift",
    "GradientShap",
    "DeepLiftShap",
    "LRP",
)

PAPER_VIT_METHODS = (
    "Saliency",
    "InputXGradient",
    "IntegratedGradients",
    "FeatureAblation",
    "Occlusion",
    "GradientShap",
    "CheferTransformerAttribution",
    "PartialLRP",
    "FullLRP",
    "GradientAttentionRollout",
    "AttentionGradCAM",
)

PATCH_METHODS = frozenset({"FeatureAblation", "Occlusion"})
LOCKED_PATCH_SIZES = (8, 14, 16)


@dataclass(frozen=True, slots=True)
class MethodVariant:
    """One final explainer parameterization and its profiling identity."""

    family: str
    variant: str
    architecture: Architecture
    params: Mapping[str, Any]
    profile_start: int
    probe_perturbations: int

    @property
    def artifact_name(self) -> str:
        return self.family if not self.variant else f"{self.family}__{self.variant}"

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "family": self.family,
                "variant": self.variant,
                "architecture": self.architecture,
                "params": dict(self.params),
                "precision": "fp32",
            }
        )

    def profile_id(
        self,
        model_key: str,
        input_shape: Sequence[int],
        *,
        model_id: str | None = None,
        attribution_provider: Mapping[str, Any] | None = None,
    ) -> str:
        identity = {
            "schema": "simple-profile-v1",
            "model_architecture": model_key,
            "method": self.family,
            "params": dict(self.params),
            "precision": "fp32",
            "input_shape": [int(value) for value in input_shape],
        }
        if model_id is not None:
            identity["model_id"] = model_id
        if attribution_provider:
            identity["attribution_provider"] = dict(attribution_provider)
        digest = object_sha256(identity)[:16]
        return f"{model_key}--{self.artifact_name}--{digest}"


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    family: str
    architectures: tuple[Architecture, ...]
    params: Mapping[str, Any]
    variants: tuple[Mapping[str, Any], ...]
    profile_start: int = 256
    probe_perturbations: int = 3

    def __post_init__(self) -> None:
        if self.family not in EXPLAINER_SPECS:
            raise ValueError(f"Unknown explainer family {self.family!r}")
        if not self.architectures:
            raise ValueError(f"{self.family} must name at least one architecture")
        allowed = set(EXPLAINER_SPECS[self.family].candidate_architectures)
        if not set(self.architectures) <= allowed:
            raise ValueError(
                f"{self.family} cannot run on {sorted(set(self.architectures) - allowed)}"
            )
        if self.profile_start <= 0:
            raise ValueError("profile_start must be positive")
        if self.probe_perturbations <= 0:
            raise ValueError("probe_perturbations must be positive")
        if self.family in PATCH_METHODS:
            sizes = tuple(int(item.get("patch_size", -1)) for item in self.variants)
            if sizes != LOCKED_PATCH_SIZES:
                raise ValueError(
                    f"{self.family} variants must be the locked p={LOCKED_PATCH_SIZES}, "
                    f"found {sizes}"
                )
        elif self.variants:
            raise ValueError(f"Only patch methods may define variants: {self.family}")

    def instances(self, architecture: Architecture) -> tuple[MethodVariant, ...]:
        if architecture not in self.architectures:
            return ()
        raw_variants: tuple[Mapping[str, Any], ...] = self.variants or ({},)
        result = []
        for override in raw_variants:
            params = {**self.params, **dict(override)}
            variant = ""
            if self.family in PATCH_METHODS:
                variant = f"p{int(params['patch_size'])}"
            result.append(
                MethodVariant(
                    family=self.family,
                    variant=variant,
                    architecture=architecture,
                    params=params,
                    profile_start=self.profile_start,
                    probe_perturbations=self.probe_perturbations,
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class MethodCatalog:
    definitions: tuple[MethodDefinition, ...]
    source_digest: str

    def __post_init__(self) -> None:
        families = [item.family for item in self.definitions]
        if len(families) != len(set(families)):
            raise ValueError("Method catalog contains duplicate families")
        cnn = tuple(item.family for item in self.definitions if "cnn" in item.architectures)
        vit = tuple(item.family for item in self.definitions if "vit" in item.architectures)
        if cnn != PAPER_CNN_METHODS:
            raise ValueError(
                "CNN method roster/order differs from the paper lock: "
                f"expected={PAPER_CNN_METHODS}, found={cnn}"
            )
        if vit != PAPER_VIT_METHODS:
            raise ValueError(
                "ViT method roster/order differs from the paper lock: "
                f"expected={PAPER_VIT_METHODS}, found={vit}"
            )

    def for_architecture(self, architecture: Architecture) -> tuple[MethodDefinition, ...]:
        return tuple(item for item in self.definitions if architecture in item.architectures)

    def family(self, name: str) -> MethodDefinition:
        for definition in self.definitions:
            if definition.family == name:
                return definition
        raise KeyError(name)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def load_method_catalog(path: str | Path) -> MethodCatalog:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, context="method catalog")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("Method catalog schema_version must be 1")
    rows = root.get("methods")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Method catalog must contain a non-empty methods list")
    definitions = []
    for index, value in enumerate(rows):
        row = _mapping(value, context=f"methods[{index}]")
        architectures = tuple(str(item) for item in row.get("architectures", ()))
        if any(item not in {"cnn", "vit"} for item in architectures):
            raise ValueError(f"methods[{index}] has an invalid architecture")
        variants = row.get("variants", [])
        if not isinstance(variants, list):
            raise TypeError(f"methods[{index}].variants must be a list")
        definitions.append(
            MethodDefinition(
                family=str(row["family"]),
                architectures=architectures,  # type: ignore[arg-type]
                params=dict(_mapping(row.get("params", {}), context="params")),
                variants=tuple(dict(_mapping(item, context="variant")) for item in variants),
                profile_start=int(row.get("profile_start", 256)),
                probe_perturbations=int(row.get("probe_perturbations", 3)),
            )
        )
    return MethodCatalog(tuple(definitions), object_sha256(raw))


__all__ = [
    "Architecture",
    "LOCKED_PATCH_SIZES",
    "MethodCatalog",
    "MethodDefinition",
    "MethodVariant",
    "PAPER_CNN_METHODS",
    "PAPER_VIT_METHODS",
    "PATCH_METHODS",
    "load_method_catalog",
]
