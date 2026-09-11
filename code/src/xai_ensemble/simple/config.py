"""Configuration and deterministic task expansion for the simple pipeline."""

from __future__ import annotations

import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.data.specs import get_dataset_spec
from xai_ensemble.phase0.models import get_model_definition
from xai_ensemble.phase1.relprop import relprop_attribution_provider

from .methods import (
    PATCH_METHODS,
    Architecture,
    MethodCatalog,
    MethodDefinition,
    MethodVariant,
    load_method_catalog,
)

SEARCH_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_SPOOL_MAX_BYTES = 64 * 2**30
DEFAULT_SPOOL_MIN_FREE_BYTES = 32 * 2**30
_SPOOL_MAX_GIB_ENV = "XAI_SIMPLE_SPOOL_MAX_GIB"
_SPOOL_MIN_FREE_GIB_ENV = "XAI_SIMPLE_SPOOL_MIN_FREE_GIB"
_PHASE1_UPLOAD_WORKERS_ENV = "XAI_SIMPLE_PHASE1_UPLOAD_WORKERS"
_PHASE1_UPLOAD_GLOBAL_LIMIT_ENV = "XAI_SIMPLE_PHASE1_UPLOAD_GLOBAL_LIMIT"
_PHASE1_STAGE_WORKERS_ENV = "XAI_SIMPLE_PHASE1_STAGE_WORKERS"
_PHASE1_PREFETCH_WORKERS_ENV = "XAI_SIMPLE_PHASE1_PREFETCH_WORKERS"
_PHASE1_PREFETCH_MAX_GIB_ENV = "XAI_SIMPLE_PHASE1_PREFETCH_MAX_GIB"
_PHASE1_PREFETCH_MIN_FREE_GIB_ENV = "XAI_SIMPLE_PHASE1_PREFETCH_MIN_FREE_GIB"
_PHASE1_TELEMETRY_INTERVAL_ENV = "XAI_SIMPLE_PHASE1_TELEMETRY_INTERVAL_SECONDS"


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence")
    return value


def _identifier(value: Any, *, context: str) -> str:
    result = str(value)
    if not SAFE_ID.fullmatch(result):
        raise ValueError(f"{context} must match {SAFE_ID.pattern!r}; found {result!r}")
    return result


def _path(value: Any, *, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def _rclone_binary(value: Any, *, base: Path) -> Path:
    """Resolve the rclone executable, defaulting to a PATH lookup."""

    if value is not None:
        return _path(value, base=base)
    discovered = shutil.which("rclone")
    return Path(discovered) if discovered is not None else Path("rclone")


def _execution_gib_override(name: str, configured: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return configured
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative number") from error
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _execution_int_override(
    name: str,
    configured: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(configured)
    try:
        value = int(raw)
    except ValueError as error:
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer in {bound}") from error
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be in {bound}")
    return value


def _execution_float_override(name: str, configured: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(configured)
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number >= {minimum}") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _dataset_identity(item: DatasetConfig) -> Mapping[str, Any]:
    return {
        "id": item.dataset_id,
        "registry_key": item.registry_key,
        "provider_factory": item.provider_factory,
        "provider_kwargs": dict(item.provider_kwargs),
        "image_column": item.image_column,
        "label_column": item.label_column,
        "manifest_path": str(item.manifest_path),
        "max_samples": item.max_samples,
    }


def _model_identity(item: ModelConfig) -> Mapping[str, Any]:
    return {
        "id": item.model_id,
        "dataset": item.dataset_id,
        "model_key": item.model_key,
        "num_classes": item.num_classes,
        "init_mode": item.init_mode,
        "checkpoint_path": (None if item.checkpoint_path is None else str(item.checkpoint_path)),
        "strict_checkpoint": item.strict_checkpoint,
        "class_index_map": item.class_index_map,
        "mean_path": str(item.mean_path),
        "mean_key": item.mean_key,
        "architecture": item.architecture,
    }


@dataclass(frozen=True, slots=True)
class StorageConfig:
    remote_root: str
    scratch_root: Path
    rclone_binary: Path
    spool_root: Path
    spool_max_bytes: int
    spool_min_free_bytes: int

    def __post_init__(self) -> None:
        if not self.remote_root.strip():
            raise ValueError("storage.remote_root must be non-empty")
        if self.spool_max_bytes <= 0:
            raise ValueError("storage.spool_max_gib must be positive")
        if self.spool_min_free_bytes < 0:
            raise ValueError("storage.spool_min_free_gib cannot be negative")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    profile_directory: Path
    database_path: Path
    log_directory: Path
    gpu_ids: tuple[int, ...]
    search_grid: tuple[int, ...] = SEARCH_GRID
    default_profile_start: int = 256
    shard_size: int = 512
    prediction_batch_size: int = 512
    dataloader_workers: int = 8
    headroom_fraction: float = 0.10
    max_retries: int = 1
    seed: int = 20260714
    # These are execution-only controls. They are resolved from environment
    # overrides and intentionally do not enter the scientific or scheduler
    # identities stored in the YAML configuration.
    phase1_upload_workers: int = 2
    phase1_upload_global_limit: int = 4
    phase1_stage_workers: int = 2
    phase1_prefetch_workers: int = 2
    phase1_prefetch_max_gib: float = 64.0
    phase1_prefetch_min_free_gib: float = 40.0
    phase1_telemetry_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if self.search_grid != SEARCH_GRID:
            raise ValueError(f"runtime.search_grid is scientifically locked to {SEARCH_GRID}")
        if self.default_profile_start not in self.search_grid:
            raise ValueError("default_profile_start must occur in search_grid")
        if self.shard_size <= 0 or self.prediction_batch_size <= 0:
            raise ValueError("shard and prediction batch sizes must be positive")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers cannot be negative")
        if not 0.0 <= self.headroom_fraction < 0.5:
            raise ValueError("headroom_fraction must lie in [0, 0.5)")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if not 0 < self.phase1_upload_workers <= 16:
            raise ValueError("phase1_upload_workers must lie in [1, 16]")
        if not 0 < self.phase1_upload_global_limit <= 64:
            raise ValueError("phase1_upload_global_limit must lie in [1, 64]")
        if not 0 < self.phase1_stage_workers <= 16:
            raise ValueError("phase1_stage_workers must lie in [1, 16]")
        if not 0 < self.phase1_prefetch_workers <= 16:
            raise ValueError("phase1_prefetch_workers must lie in [1, 16]")
        if self.phase1_prefetch_max_gib <= 0:
            raise ValueError("phase1_prefetch_max_gib must be positive")
        if self.phase1_prefetch_min_free_gib < 0:
            raise ValueError("phase1_prefetch_min_free_gib cannot be negative")
        if self.phase1_telemetry_interval_seconds <= 0:
            raise ValueError("phase1_telemetry_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_id: str
    registry_key: str | None
    provider_factory: str | None
    provider_kwargs: Mapping[str, Any]
    image_column: str
    label_column: str
    manifest_path: Path
    splits: tuple[str, ...]
    cache_directory: Path | None
    keep_provider_in_memory: bool
    cache_images_in_ram: bool
    max_samples: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.dataset_id, context="dataset id")
        if (self.registry_key is None) == (self.provider_factory is None):
            raise ValueError(
                f"Dataset {self.dataset_id} must define exactly one of "
                "registry_key or provider_factory"
            )
        if not self.splits:
            raise ValueError(f"Dataset {self.dataset_id} must define at least one split")
        if self.registry_key is not None:
            spec = get_dataset_spec(self.registry_key)
            unknown = set(self.splits) - set(spec.splits)
            if unknown:
                raise ValueError(f"Dataset {self.dataset_id} has invalid splits: {sorted(unknown)}")
        else:
            assert self.provider_factory is not None
            module_name, separator, attribute = self.provider_factory.partition(":")
            if not separator or not module_name or not attribute:
                raise ValueError("provider_factory must use module:function syntax")
        if not self.image_column or not self.label_column:
            raise ValueError("Dataset image_column and label_column must be non-empty")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("dataset max_samples must be positive when set")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    dataset_id: str
    model_key: str
    num_classes: int
    init_mode: str
    checkpoint_path: Path | None
    strict_checkpoint: bool
    class_index_map: tuple[int, ...] | None
    mean_path: Path
    mean_key: str
    architecture: Architecture

    def __post_init__(self) -> None:
        _identifier(self.model_id, context="model id")
        definition = get_model_definition(self.model_key)
        if definition.family != self.architecture:
            raise ValueError(
                f"Model {self.model_id} architecture={self.architecture} contradicts "
                f"registry family={definition.family}"
            )
        if self.num_classes <= 1:
            raise ValueError("model num_classes must be greater than one")
        if self.init_mode == "checkpoint" and self.checkpoint_path is None:
            raise ValueError(f"Model {self.model_id} requires checkpoint_path")
        if not self.mean_key:
            raise ValueError(f"Model {self.model_id} mean_key must be non-empty")


@dataclass(frozen=True, slots=True)
class ConditionConfig:
    condition_id: str
    kind: Literal["clean", "factory", "adversarial"]
    factory: str | None
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.condition_id, context="condition id")
        if self.kind in {"clean", "adversarial"} and self.factory is not None:
            raise ValueError(f"{self.kind} condition cannot define a factory")
        if self.kind == "factory" and not self.factory:
            raise ValueError("factory condition requires module:function")
        if self.kind == "adversarial":
            for architecture in ("cnn", "vit"):
                self.adversarial_settings(architecture)

    def adversarial_settings(self, architecture: Architecture) -> Mapping[str, Any]:
        if self.kind != "adversarial":
            raise ValueError(f"Condition {self.condition_id} is not adversarial")
        expected = {
            "algorithm",
            "epsilon",
            "steps",
            "learning_rate",
            "classification_weight",
            "top_fraction",
            "source_by_architecture",
            "batch_size_by_architecture",
        }
        if set(self.kwargs) != expected:
            raise ValueError(
                f"Adversarial condition {self.condition_id} must define exactly "
                f"{sorted(expected)}; found {sorted(self.kwargs)}"
            )
        if self.kwargs["algorithm"] != "sara-repetto-v2":
            raise ValueError("Only the pilot-validated sara-repetto-v2 attack is supported")
        source_by_architecture = _mapping(
            self.kwargs["source_by_architecture"],
            context=f"condition {self.condition_id} source_by_architecture",
        )
        batch_by_architecture = _mapping(
            self.kwargs["batch_size_by_architecture"],
            context=f"condition {self.condition_id} batch_size_by_architecture",
        )
        if set(source_by_architecture) != {"cnn", "vit"}:
            raise ValueError("source_by_architecture must define exactly cnn and vit")
        if set(batch_by_architecture) != {"cnn", "vit"}:
            raise ValueError("batch_size_by_architecture must define exactly cnn and vit")
        # The ViT string is an immutable v1 artifact identifier. Its implemented
        # and reported scientific method is GradientAttentionRollout; retaining
        # the legacy id keeps the already generated full-test adversarial images
        # and every consuming Phase-1 task byte-for-byte reusable.
        expected_sources = {"cnn": "DeepLift", "vit": "TransformerAttribution"}
        observed_sources = {key: str(source_by_architecture[key]) for key in expected_sources}
        if observed_sources != expected_sources:
            raise ValueError(
                "The formal attack sources are locked to CNN=DeepLift and "
                "ViT=TransformerAttribution (legacy id for GradientAttentionRollout)"
            )
        batch_size = batch_by_architecture[architecture]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Adversarial batch sizes must be positive integers")
        epsilon = float(self.kwargs["epsilon"])
        steps_value = self.kwargs["steps"]
        if isinstance(steps_value, bool) or not isinstance(steps_value, int):
            raise ValueError("Adversarial steps must be a positive integer")
        steps = int(steps_value)
        learning_rate = float(self.kwargs["learning_rate"])
        classification_weight = float(self.kwargs["classification_weight"])
        top_fraction = float(self.kwargs["top_fraction"])
        numeric = (epsilon, learning_rate, classification_weight, top_fraction)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Adversarial parameters must be finite")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("Adversarial epsilon must lie in (0,1]")
        if steps <= 0 or learning_rate <= 0.0:
            raise ValueError("Adversarial steps and learning_rate must be positive")
        if classification_weight < 0.0:
            raise ValueError("Adversarial classification_weight cannot be negative")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("Adversarial top_fraction must lie in (0,1]")
        return {
            "algorithm": "sara-repetto-v2",
            "source_method": observed_sources[architecture],
            "batch_size": int(batch_size),
            "epsilon": epsilon,
            "steps": steps,
            "learning_rate": learning_rate,
            "classification_weight": classification_weight,
            "top_fraction": top_fraction,
        }


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    ensemble_id: str
    methods: tuple[str, ...] | Literal["architecture_default"]
    rules: tuple[str, ...]
    include_singles: bool

    def __post_init__(self) -> None:
        _identifier(self.ensemble_id, context="ensemble id")
        allowed = {"SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"}
        if not self.rules or not set(self.rules) <= allowed:
            raise ValueError(
                f"Ensemble {self.ensemble_id} has unsupported rules: "
                f"{sorted(set(self.rules) - allowed)}"
            )
        if self.methods != "architecture_default" and not self.methods:
            raise ValueError("ensemble methods cannot be empty")


@dataclass(frozen=True, slots=True)
class Phase2Config:
    patch_sizes: tuple[int, ...]
    primary_patch_size: int
    k: int
    inference_batch_size: int
    simpleavg_normalization: str
    rrf_c: float
    kemeny_starts: int
    kemeny_max_passes: int
    ensembles: tuple[EnsembleConfig, ...]

    def __post_init__(self) -> None:
        if not self.patch_sizes or not set(self.patch_sizes) <= {8, 14, 16}:
            raise ValueError("phase2.patch_sizes must be a non-empty subset of [8, 14, 16]")
        if self.primary_patch_size not in self.patch_sizes:
            raise ValueError("phase2.primary_patch_size must be present in phase2.patch_sizes")
        if self.k <= 0 or self.inference_batch_size <= 0:
            raise ValueError("phase2 k and inference_batch_size must be positive")
        if self.simpleavg_normalization not in {"minmax", "max", "l1", "none"}:
            raise ValueError("Unsupported SimpleAvg normalization")
        if self.rrf_c <= 0 or self.kemeny_starts <= 0 or self.kemeny_max_passes <= 0:
            raise ValueError("Phase 2 rule parameters must be positive")
        if not self.ensembles:
            raise ValueError("phase2 must define at least one ensemble")


@dataclass(frozen=True, slots=True)
class ProfileClass:
    profile_id: str
    model_id: str | None
    model_key: str
    num_classes: int
    architecture: Architecture
    input_size: int
    method: MethodVariant


@dataclass(frozen=True, slots=True)
class Phase2ProfileClass:
    profile_id: str
    model_key: str
    num_classes: int
    architecture: Architecture
    input_size: int
    inference_batch_size: int


@dataclass(frozen=True, slots=True)
class Phase1Task:
    task_id: str
    dataset: DatasetConfig
    model: ModelConfig
    split: str
    condition: ConditionConfig
    family: str
    variants: tuple[MethodVariant, ...]
    digest: str

    @property
    def profile_ids(self) -> tuple[str, ...]:
        input_size = get_model_definition(self.model.model_key).input_size
        profile_ids = []
        for variant in self.variants:
            provider = relprop_attribution_provider(
                variant.family,
                variant.architecture,
            )
            profile_ids.append(
                variant.profile_id(
                    self.model.model_key,
                    (3, input_size, input_size),
                    model_id=self.model.model_id if provider else None,
                    attribution_provider=provider or None,
                )
            )
        return tuple(profile_ids)


@dataclass(frozen=True, slots=True)
class AdversarialTask:
    task_id: str
    dataset: DatasetConfig
    model: ModelConfig
    split: str
    condition: ConditionConfig
    algorithm: str
    source_method: Literal["DeepLift", "TransformerAttribution"]
    batch_size: int
    epsilon: float
    steps: int
    learning_rate: float
    classification_weight: float
    top_fraction: float
    digest: str


@dataclass(frozen=True, slots=True)
class Phase2Task:
    task_id: str
    dataset: DatasetConfig
    model: ModelConfig
    split: str
    condition: ConditionConfig
    ensemble: EnsembleConfig
    patch_size: int
    digest: str


@dataclass(frozen=True, slots=True)
class SimpleExperiment:
    source_path: Path
    experiment_id: str
    precision: Literal["fp32"]
    methods: MethodCatalog
    storage: StorageConfig
    runtime: RuntimeConfig
    datasets: tuple[DatasetConfig, ...]
    models: tuple[ModelConfig, ...]
    conditions: tuple[ConditionConfig, ...]
    phase2: Phase2Config
    raw_config: Mapping[str, Any]
    # ``None`` preserves the full catalog variant set.  A configured value is
    # an explicit Phase-1 scope reduction, not a mutation of methods.yaml.
    phase1_patch_sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, context="experiment_id")
        if self.precision != "fp32":
            raise ValueError("The simplified experiment is locked to FP32")
        for collection, label in (
            (self.datasets, "dataset"),
            (self.models, "model"),
            (self.conditions, "condition"),
        ):
            identifiers = [getattr(item, f"{label}_id") for item in collection]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Duplicate {label} ids")
        dataset_ids = {item.dataset_id for item in self.datasets}
        for model in self.models:
            if model.dataset_id not in dataset_ids:
                raise ValueError(
                    f"Model {model.model_id} references unknown dataset {model.dataset_id}"
                )
        clean_conditions = [item for item in self.conditions if item.kind == "clean"]
        if len(clean_conditions) != 1:
            raise ValueError(
                "Exactly one clean condition is required so perturbed explanations "
                "can fix clean targets and compute paired robustness"
            )
        if self.phase1_patch_sizes is not None:
            allowed = {8, 14, 16}
            values = self.phase1_patch_sizes
            if not values or len(set(values)) != len(values) or not set(values) <= allowed:
                raise ValueError(
                    "phase1_patch_sizes must be a non-empty unique subset of [8, 14, 16]"
                )
            missing = set(self.phase2.patch_sizes) - set(values)
            if missing:
                raise ValueError(
                    "phase2.patch_sizes require unavailable Phase 1 patch variants: "
                    f"{sorted(missing)}"
                )

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-experiment-v2",
                "config": self.raw_config,
                "method_catalog_digest": self.methods.source_digest,
                "precision": self.precision,
            }
        )

    @property
    def phase1_digest(self) -> str:
        """Shared Phase 1 semantics, independent of the configured task roster."""

        pinned = self.raw_config.get("phase1_identity_digest")
        if pinned is not None:
            value = str(pinned).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("phase1_identity_digest must be a lowercase SHA-256")
            return value
        identity: dict[str, Any] = {
            "schema": "simple-phase1-experiment-v2",
            "precision": self.precision,
            "seed": self.runtime.seed,
            "method_catalog_digest": self.methods.source_digest,
        }
        # Existing experiments did not define a variant filter. Keep their
        # Phase-1 identity byte-stable while binding a newly explicit scope.
        if self.phase1_patch_sizes is not None:
            identity["phase1_patch_sizes"] = list(self.phase1_patch_sizes)
        return object_sha256(identity)

    @property
    def scheduler_digest(self) -> str:
        """Execution namespace identity that permits later Phase 2 additions."""

        return object_sha256(
            {
                "schema": "simple-scheduler-v2",
                "experiment_id": self.experiment_id,
                "phase1_digest": self.phase1_digest,
                "storage": self.raw_config["storage"],
                "runtime": self.raw_config["runtime"],
            }
        )

    def dataset(self, dataset_id: str) -> DatasetConfig:
        for item in self.datasets:
            if item.dataset_id == dataset_id:
                return item
        raise KeyError(dataset_id)

    def model(self, model_id: str) -> ModelConfig:
        for item in self.models:
            if item.model_id == model_id:
                return item
        raise KeyError(model_id)

    def condition(self, condition_id: str) -> ConditionConfig:
        for item in self.conditions:
            if item.condition_id == condition_id:
                return item
        raise KeyError(condition_id)

    def phase1_variants(
        self,
        definition: MethodDefinition,
        architecture: Architecture,
    ) -> tuple[MethodVariant, ...]:
        """Resolve the catalog variants that this experiment actually emits."""

        variants = definition.instances(architecture)
        if self.phase1_patch_sizes is None or definition.family not in PATCH_METHODS:
            return variants
        selected = tuple(
            variant
            for variant in variants
            if int(variant.params["patch_size"]) in self.phase1_patch_sizes
        )
        if not selected:
            raise ValueError(
                f"Phase 1 excludes every {definition.family} variant for {architecture}"
            )
        return selected

    def profiles(self) -> tuple[ProfileClass, ...]:
        by_id: dict[str, ProfileClass] = {}
        for model in self.models:
            definition = get_model_definition(model.model_key)
            for method in self.methods.for_architecture(model.architecture):
                for variant in self.phase1_variants(method, model.architecture):
                    attribution_provider = relprop_attribution_provider(
                        variant.family,
                        variant.architecture,
                    )
                    profile_id = variant.profile_id(
                        model.model_key,
                        (3, definition.input_size, definition.input_size),
                        model_id=model.model_id if attribution_provider else None,
                        attribution_provider=attribution_provider or None,
                    )
                    by_id.setdefault(
                        profile_id,
                        ProfileClass(
                            profile_id=profile_id,
                            model_id=model.model_id if attribution_provider else None,
                            model_key=model.model_key,
                            num_classes=model.num_classes,
                            architecture=model.architecture,
                            input_size=definition.input_size,
                            method=variant,
                        ),
                    )
        return tuple(by_id[key] for key in sorted(by_id))

    def phase2_profiles(self) -> tuple[Phase2ProfileClass, ...]:
        by_id: dict[str, Phase2ProfileClass] = {}
        for model in self.models:
            definition = get_model_definition(model.model_key)
            identity = {
                "schema": "simple-phase2-inference-profile-v1",
                "model_key": model.model_key,
                "num_classes": model.num_classes,
                "architecture": model.architecture,
                "precision": "fp32",
                "input_shape": [3, definition.input_size, definition.input_size],
                "forward_batch_size": self.phase2.inference_batch_size,
                "masked_variants": ["removed", "retained"],
            }
            digest = object_sha256(identity)
            profile_id = "--".join(
                (
                    "phase2-inference",
                    model.model_key,
                    f"b{self.phase2.inference_batch_size}",
                    digest[:12],
                )
            )
            by_id.setdefault(
                profile_id,
                Phase2ProfileClass(
                    profile_id=profile_id,
                    model_key=model.model_key,
                    num_classes=model.num_classes,
                    architecture=model.architecture,
                    input_size=definition.input_size,
                    inference_batch_size=self.phase2.inference_batch_size,
                ),
            )
        return tuple(by_id[key] for key in sorted(by_id))

    def phase2_profile_for_model(self, model: ModelConfig) -> Phase2ProfileClass:
        definition = get_model_definition(model.model_key)
        for profile in self.phase2_profiles():
            if (
                profile.model_key == model.model_key
                and profile.num_classes == model.num_classes
                and profile.architecture == model.architecture
                and profile.input_size == definition.input_size
                and profile.inference_batch_size == self.phase2.inference_batch_size
            ):
                return profile
        raise KeyError(f"No Phase 2 inference profile for model {model.model_id}")

    def adversarial_tasks(self) -> tuple[AdversarialTask, ...]:
        tasks = []
        for model in self.models:
            dataset = self.dataset(model.dataset_id)
            for split in dataset.splits:
                for condition in self.conditions:
                    if condition.kind != "adversarial":
                        continue
                    settings = condition.adversarial_settings(model.architecture)
                    identity = {
                        "schema": "simple-adversarial-task-v1",
                        "phase1_experiment_digest": self.phase1_digest,
                        "dataset": _dataset_identity(dataset),
                        "model": _model_identity(model),
                        "split": split,
                        "condition": asdict(condition),
                        "algorithm": settings["algorithm"],
                        "source_method": settings["source_method"],
                        "batch_size": settings["batch_size"],
                        "epsilon": settings["epsilon"],
                        "steps": settings["steps"],
                        "learning_rate": settings["learning_rate"],
                        "classification_weight": settings["classification_weight"],
                        "top_fraction": settings["top_fraction"],
                        "precision": "fp32",
                    }
                    digest = object_sha256(identity)
                    task_id = "--".join(
                        (
                            dataset.dataset_id,
                            model.model_id,
                            split,
                            condition.condition_id,
                            str(settings["source_method"]),
                            digest[:10],
                        )
                    )
                    tasks.append(
                        AdversarialTask(
                            task_id=task_id,
                            dataset=dataset,
                            model=model,
                            split=split,
                            condition=condition,
                            algorithm=str(settings["algorithm"]),
                            source_method=str(settings["source_method"]),  # type: ignore[arg-type]
                            batch_size=int(settings["batch_size"]),
                            epsilon=float(settings["epsilon"]),
                            steps=int(settings["steps"]),
                            learning_rate=float(settings["learning_rate"]),
                            classification_weight=float(settings["classification_weight"]),
                            top_fraction=float(settings["top_fraction"]),
                            digest=digest,
                        )
                    )
        return tuple(tasks)

    def adversarial_task_for(
        self,
        *,
        dataset_id: str,
        model_id: str,
        split: str,
        condition_id: str,
    ) -> AdversarialTask:
        matches = [
            task
            for task in self.adversarial_tasks()
            if task.dataset.dataset_id == dataset_id
            and task.model.model_id == model_id
            and task.split == split
            and task.condition.condition_id == condition_id
        ]
        if len(matches) != 1:
            raise KeyError(
                "Expected one adversarial task for "
                f"{dataset_id}/{model_id}/{split}/{condition_id}; found {len(matches)}"
            )
        return matches[0]

    def phase1_tasks(self) -> tuple[Phase1Task, ...]:
        tasks = []
        for model in self.models:
            dataset = self.dataset(model.dataset_id)
            for split in dataset.splits:
                for condition in self.conditions:
                    for definition in self.methods.for_architecture(model.architecture):
                        variants = self.phase1_variants(definition, model.architecture)
                        identity = {
                            "schema": "simple-phase1-task-v2",
                            "phase1_experiment_digest": self.phase1_digest,
                            "dataset": _dataset_identity(dataset),
                            "model": _model_identity(model),
                            "split": split,
                            "condition": asdict(condition),
                            "family": definition.family,
                            "variants": [
                                {"name": item.variant, "params": dict(item.params)}
                                for item in variants
                            ],
                            "precision": "fp32",
                        }
                        attribution_provider = relprop_attribution_provider(
                            definition.family,
                            model.architecture,
                        )
                        if attribution_provider:
                            identity["attribution_provider"] = attribution_provider
                        digest = object_sha256(identity)
                        task_id = "--".join(
                            (
                                dataset.dataset_id,
                                model.model_id,
                                split,
                                condition.condition_id,
                                definition.family,
                                digest[:10],
                            )
                        )
                        tasks.append(
                            Phase1Task(
                                task_id=task_id,
                                dataset=dataset,
                                model=model,
                                split=split,
                                condition=condition,
                                family=definition.family,
                                variants=variants,
                                digest=digest,
                            )
                        )
        return tuple(tasks)

    def phase2_tasks(self) -> tuple[Phase2Task, ...]:
        tasks = []
        for model in self.models:
            dataset = self.dataset(model.dataset_id)
            for split in dataset.splits:
                for condition in self.conditions:
                    for ensemble in self.phase2.ensembles:
                        for patch_size in self.phase2.patch_sizes:
                            resolved_methods = tuple(
                                item.family
                                for item in self.methods.for_architecture(model.architecture)
                            )
                            expanded_vit_roster = (
                                model.architecture == "vit" and len(resolved_methods) > 4
                            )
                            identity = {
                                "schema": (
                                    "simple-phase2-task-v3"
                                    if expanded_vit_roster
                                    else "simple-phase2-task-v2"
                                ),
                                "phase1_experiment_digest": self.phase1_digest,
                                "dataset": _dataset_identity(dataset),
                                "model": _model_identity(model),
                                "split": split,
                                "condition": asdict(condition),
                                "ensemble": asdict(ensemble),
                                "patch_size": patch_size,
                                "k": self.phase2.k,
                                "inference_batch_size": self.phase2.inference_batch_size,
                                "simpleavg_normalization": (self.phase2.simpleavg_normalization),
                                "rrf_c": self.phase2.rrf_c,
                                "kemeny_starts": self.phase2.kemeny_starts,
                                "kemeny_max_passes": self.phase2.kemeny_max_passes,
                            }
                            if expanded_vit_roster:
                                # architecture_default previously omitted its resolved
                                # roster from the task identity. Bind the expanded ViT
                                # result without invalidating completed CNN Phase 2 cells.
                                identity["resolved_methods"] = list(resolved_methods)
                                identity["attribution_providers"] = {
                                    family: provider
                                    for family in resolved_methods
                                    if (
                                        provider := relprop_attribution_provider(
                                            family,
                                            model.architecture,
                                        )
                                    )
                                }
                            digest = object_sha256(identity)
                            task_id = "--".join(
                                (
                                    dataset.dataset_id,
                                    model.model_id,
                                    split,
                                    condition.condition_id,
                                    ensemble.ensemble_id,
                                    f"p{patch_size}",
                                    digest[:10],
                                )
                            )
                            tasks.append(
                                Phase2Task(
                                    task_id=task_id,
                                    dataset=dataset,
                                    model=model,
                                    split=split,
                                    condition=condition,
                                    ensemble=ensemble,
                                    patch_size=patch_size,
                                    digest=digest,
                                )
                            )
        return tuple(tasks)


def _load_datasets(rows: Sequence[Any], *, base: Path) -> tuple[DatasetConfig, ...]:
    result = []
    for index, value in enumerate(rows):
        row = _mapping(value, context=f"datasets[{index}]")
        cache = row.get("cache_directory")
        max_samples = row.get("max_samples")
        result.append(
            DatasetConfig(
                dataset_id=_identifier(row["id"], context=f"datasets[{index}].id"),
                registry_key=(
                    None if row.get("registry_key") is None else str(row["registry_key"])
                ),
                provider_factory=(
                    None if row.get("provider_factory") is None else str(row["provider_factory"])
                ),
                provider_kwargs=dict(
                    _mapping(row.get("provider_kwargs", {}), context="provider_kwargs")
                ),
                image_column=str(row.get("image_column", "image")),
                label_column=str(row.get("label_column", "label")),
                manifest_path=_path(row["manifest_path"], base=base),
                splits=tuple(str(item) for item in _sequence(row["splits"], context="splits")),
                cache_directory=None if cache is None else _path(cache, base=base),
                keep_provider_in_memory=bool(row.get("keep_provider_in_memory", False)),
                cache_images_in_ram=bool(row.get("cache_images_in_ram", True)),
                max_samples=None if max_samples is None else int(max_samples),
            )
        )
    return tuple(result)


def _load_class_map(value: Any, *, base: Path) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    path = _path(value, base=base)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, Mapping):
        if "entries" in loaded:
            entries = _sequence(loaded["entries"], context="class_index_map.entries")
            indices = []
            for expected_index, entry in enumerate(entries):
                fields = _sequence(entry, context="class_index_map entry")
                if len(fields) < 4 or int(fields[0]) != expected_index:
                    raise ValueError(
                        "class_index_map entries must be ordered "
                        "[dataset_index, name, synset, imagenet1k_index] rows"
                    )
                indices.append(int(fields[3]))
            loaded = indices
        for key in ("indices", "class_indices", "mapping"):
            if isinstance(loaded, Mapping) and key in loaded:
                loaded = loaded[key]
                break
    return tuple(int(item) for item in _sequence(loaded, context="class_index_map"))


def _load_models(rows: Sequence[Any], *, base: Path) -> tuple[ModelConfig, ...]:
    result = []
    for index, value in enumerate(rows):
        row = _mapping(value, context=f"models[{index}]")
        checkpoint = row.get("checkpoint_path")
        model_key = str(row["model_key"])
        registry_architecture = get_model_definition(model_key).family
        architecture = str(row.get("architecture", registry_architecture))
        if architecture not in {"cnn", "vit"}:
            raise ValueError(f"models[{index}] architecture must be cnn or vit")
        result.append(
            ModelConfig(
                model_id=_identifier(row["id"], context=f"models[{index}].id"),
                dataset_id=str(row["dataset"]),
                model_key=model_key,
                num_classes=int(row["num_classes"]),
                init_mode=str(row.get("init_mode", "checkpoint")),
                checkpoint_path=None if checkpoint is None else _path(checkpoint, base=base),
                strict_checkpoint=bool(row.get("strict_checkpoint", True)),
                class_index_map=_load_class_map(row.get("class_index_map"), base=base),
                mean_path=_path(row["mean_path"], base=base),
                mean_key=str(row.get("mean_key", "dataset_mean")),
                architecture=architecture,  # type: ignore[arg-type]
            )
        )
    return tuple(result)


def _load_conditions(rows: Sequence[Any]) -> tuple[ConditionConfig, ...]:
    result = []
    for index, value in enumerate(rows):
        row = _mapping(value, context=f"conditions[{index}]")
        kind = str(row.get("kind", "clean"))
        if kind not in {"clean", "factory", "adversarial"}:
            raise ValueError(f"conditions[{index}] kind must be clean, factory, or adversarial")
        factory = row.get("factory")
        result.append(
            ConditionConfig(
                condition_id=_identifier(row["id"], context=f"conditions[{index}].id"),
                kind=kind,  # type: ignore[arg-type]
                factory=None if factory is None else str(factory),
                kwargs=dict(_mapping(row.get("kwargs", {}), context="condition kwargs")),
            )
        )
    return tuple(result)


def _load_phase2(value: Any) -> Phase2Config:
    row = _mapping(value, context="phase2")
    ensembles = []
    for index, value in enumerate(_sequence(row["ensembles"], context="phase2.ensembles")):
        item = _mapping(value, context=f"phase2.ensembles[{index}]")
        methods_value = item.get("methods", "architecture_default")
        methods: tuple[str, ...] | Literal["architecture_default"]
        if methods_value == "architecture_default":
            methods = "architecture_default"
        else:
            methods = tuple(
                str(method) for method in _sequence(methods_value, context="ensemble methods")
            )
        ensembles.append(
            EnsembleConfig(
                ensemble_id=_identifier(item["id"], context="ensemble id"),
                methods=methods,
                rules=tuple(
                    str(rule) for rule in _sequence(item["rules"], context="ensemble rules")
                ),
                include_singles=bool(item.get("include_singles", True)),
            )
        )
    return Phase2Config(
        patch_sizes=tuple(
            int(item) for item in _sequence(row.get("patch_sizes", [16]), context="patch_sizes")
        ),
        primary_patch_size=int(row.get("primary_patch_size", 16)),
        k=int(row.get("k", 20)),
        inference_batch_size=int(row.get("inference_batch_size", 512)),
        simpleavg_normalization=str(row.get("simpleavg_normalization", "minmax")),
        rrf_c=float(row.get("rrf_c", 60.0)),
        kemeny_starts=int(row.get("kemeny_starts", 1)),
        kemeny_max_passes=int(row.get("kemeny_max_passes", 1_024)),
        ensembles=tuple(ensembles),
    )


def load_experiment(path: str | Path) -> SimpleExperiment:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, context="experiment config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("Simple experiment schema_version must be 1")
    base = source.parent
    methods_path = _path(root["methods_file"], base=base)
    experiment_id = _identifier(root["experiment_id"], context="experiment_id")

    storage_row = _mapping(root["storage"], context="storage")
    spool_max_gib = _execution_gib_override(
        _SPOOL_MAX_GIB_ENV,
        float(storage_row.get("spool_max_gib", DEFAULT_SPOOL_MAX_BYTES / 2**30)),
    )
    spool_min_free_gib = _execution_gib_override(
        _SPOOL_MIN_FREE_GIB_ENV,
        float(
            storage_row.get(
                "spool_min_free_gib",
                DEFAULT_SPOOL_MIN_FREE_BYTES / 2**30,
            )
        ),
    )
    remote_root_raw = os.path.expandvars(str(storage_row["remote_root"])).rstrip("/")
    # A local (non-rclone) relative remote_root resolves against this file's
    # directory, like every other filesystem path in the configuration.
    if ":" not in remote_root_raw.split("/", 1)[0] and not os.path.isabs(remote_root_raw):
        remote_root_raw = str(_path(remote_root_raw, base=base))
    storage = StorageConfig(
        remote_root=remote_root_raw,
        scratch_root=_path(storage_row["scratch_root"], base=base),
        rclone_binary=_rclone_binary(storage_row.get("rclone_binary"), base=base),
        spool_root=_path(
            storage_row.get(
                "spool_root",
                f"/dev/shm/xai-simple/{experiment_id}",
            ),
            base=base,
        ),
        spool_max_bytes=int(spool_max_gib * 2**30),
        spool_min_free_bytes=int(spool_min_free_gib * 2**30),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    runtime = RuntimeConfig(
        profile_directory=_path(runtime_row["profile_directory"], base=base),
        database_path=_path(runtime_row["database_path"], base=base),
        log_directory=_path(runtime_row["log_directory"], base=base),
        gpu_ids=tuple(
            int(item) for item in _sequence(runtime_row.get("gpu_ids", (0,)), context="gpu_ids")
        ),
        search_grid=tuple(
            int(item)
            for item in _sequence(
                runtime_row.get("search_grid", SEARCH_GRID), context="search_grid"
            )
        ),
        default_profile_start=int(runtime_row.get("default_profile_start", 256)),
        shard_size=int(runtime_row.get("shard_size", 512)),
        prediction_batch_size=int(runtime_row.get("prediction_batch_size", 512)),
        dataloader_workers=int(runtime_row.get("dataloader_workers", 8)),
        headroom_fraction=float(runtime_row.get("headroom_fraction", 0.10)),
        max_retries=int(runtime_row.get("max_retries", 1)),
        seed=int(root.get("seed", 20260714)),
        phase1_upload_workers=_execution_int_override(
            _PHASE1_UPLOAD_WORKERS_ENV,
            2,
            maximum=16,
        ),
        phase1_upload_global_limit=_execution_int_override(
            _PHASE1_UPLOAD_GLOBAL_LIMIT_ENV,
            4,
            maximum=64,
        ),
        phase1_stage_workers=_execution_int_override(
            _PHASE1_STAGE_WORKERS_ENV,
            2,
            maximum=16,
        ),
        phase1_prefetch_workers=_execution_int_override(
            _PHASE1_PREFETCH_WORKERS_ENV,
            2,
            maximum=16,
        ),
        phase1_prefetch_max_gib=_execution_gib_override(
            _PHASE1_PREFETCH_MAX_GIB_ENV,
            64.0,
        ),
        phase1_prefetch_min_free_gib=_execution_gib_override(
            _PHASE1_PREFETCH_MIN_FREE_GIB_ENV,
            40.0,
        ),
        phase1_telemetry_interval_seconds=_execution_float_override(
            _PHASE1_TELEMETRY_INTERVAL_ENV,
            1.0,
            minimum=0.01,
        ),
    )
    datasets = _load_datasets(_sequence(root["datasets"], context="datasets"), base=base)
    models = _load_models(_sequence(root["models"], context="models"), base=base)
    conditions = _load_conditions(
        _sequence(root.get("conditions", [{"id": "clean"}]), context="conditions")
    )
    if not datasets or not models or not conditions:
        raise ValueError("datasets, models, and conditions must be non-empty")
    phase1_patch_sizes_value = root.get("phase1_patch_sizes")
    phase1_patch_sizes = (
        None
        if phase1_patch_sizes_value is None
        else tuple(
            int(item)
            for item in _sequence(
                phase1_patch_sizes_value,
                context="phase1_patch_sizes",
            )
        )
    )
    return SimpleExperiment(
        source_path=source,
        experiment_id=experiment_id,
        precision=str(root.get("precision", "fp32")),  # type: ignore[arg-type]
        methods=load_method_catalog(methods_path),
        storage=storage,
        runtime=runtime,
        datasets=datasets,
        models=models,
        conditions=conditions,
        phase2=_load_phase2(root["phase2"]),
        raw_config=dict(root),
        phase1_patch_sizes=phase1_patch_sizes,
    )


def find_phase1_task(experiment: SimpleExperiment, task_id: str) -> Phase1Task:
    for task in experiment.phase1_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown Phase 1 task {task_id!r}")


def find_adversarial_task(experiment: SimpleExperiment, task_id: str) -> AdversarialTask:
    for task in experiment.adversarial_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown adversarial task {task_id!r}")


def find_phase2_task(experiment: SimpleExperiment, task_id: str) -> Phase2Task:
    for task in experiment.phase2_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown Phase 2 task {task_id!r}")


def find_profile(experiment: SimpleExperiment, profile_id: str) -> ProfileClass:
    for profile in experiment.profiles():
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(f"Unknown profile class {profile_id!r}")


def find_phase2_profile(experiment: SimpleExperiment, profile_id: str) -> Phase2ProfileClass:
    for profile in experiment.phase2_profiles():
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(f"Unknown Phase 2 profile class {profile_id!r}")


__all__ = [
    "AdversarialTask",
    "ConditionConfig",
    "DatasetConfig",
    "EnsembleConfig",
    "ModelConfig",
    "Phase1Task",
    "Phase2Config",
    "Phase2ProfileClass",
    "Phase2Task",
    "ProfileClass",
    "RuntimeConfig",
    "SEARCH_GRID",
    "SimpleExperiment",
    "StorageConfig",
    "find_adversarial_task",
    "find_phase1_task",
    "find_phase2_task",
    "find_phase2_profile",
    "find_profile",
    "load_experiment",
]
