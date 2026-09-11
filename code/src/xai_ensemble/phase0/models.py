"""Lazy model registry and training-task construction for Phase 0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.data.partitions import PartitionPlan

IMAGENET1K_CLASS_COUNT = 1000
INIT_MODES = frozenset(
    {"random", "imagenet1k", "imagenet1k_subset", "checkpoint"}
)
INITIALIZATION_RECIPES = frozenset(
    {"timm_subset_logits", "timm_finetune", "random_scratch"}
)
RECIPE_INITIALIZATION = {
    "timm_subset_logits": "imagenet1k_subset",
    "timm_finetune": "imagenet1k",
    "random_scratch": "random",
}
MODEL_ROLES = frozenset({"source", "reference"})


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    key: str
    provider: str
    upstream_name: str
    family: str
    input_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bicubic"
    crop_percentage: float = 0.875

    def __post_init__(self) -> None:
        if self.provider not in {"timm", "torchvision"}:
            raise ValueError("provider must be timm or torchvision")
        if self.family not in {"cnn", "vit"}:
            raise ValueError("family must be cnn or vit")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")


MODEL_REGISTRY: Mapping[str, ModelDefinition] = {
    "resnet18": ModelDefinition("resnet18", "timm", "resnet18", "cnn"),
    "resnet50": ModelDefinition("resnet50", "timm", "resnet50", "cnn"),
    "densenet121": ModelDefinition("densenet121", "timm", "densenet121", "cnn"),
    "efficientnet_b0": ModelDefinition(
        "efficientnet_b0", "timm", "efficientnet_b0", "cnn"
    ),
    "mobilenetv3_large_100": ModelDefinition(
        "mobilenetv3_large_100", "timm", "mobilenetv3_large_100", "cnn"
    ),
    "vit_base_patch16_224": ModelDefinition(
        "vit_base_patch16_224", "timm", "vit_base_patch16_224", "vit"
    ),
    "deit_base_patch16_224": ModelDefinition(
        "deit_base_patch16_224", "timm", "deit_base_patch16_224", "vit"
    ),
    "swin_base_patch4_window7_224": ModelDefinition(
        "swin_base_patch4_window7_224",
        "timm",
        "swin_base_patch4_window7_224",
        "vit",
    ),
}


@dataclass(frozen=True, slots=True)
class ModelBuildRequest:
    model_key: str
    num_classes: int
    init_mode: str = "imagenet1k"
    checkpoint_path: str | None = None
    strict_checkpoint: bool = True
    disable_inplace_activations: bool = True
    seed: int | None = None
    class_index_map: tuple[int, ...] | None = None
    class_map_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {self.model_key!r}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if self.init_mode not in INIT_MODES:
            raise ValueError(f"Unknown initialization mode: {self.init_mode}")
        if self.init_mode == "checkpoint" and not self.checkpoint_path:
            raise ValueError("checkpoint initialization requires checkpoint_path")
        if self.init_mode != "checkpoint" and self.checkpoint_path is not None:
            raise ValueError("checkpoint_path is only valid for checkpoint initialization")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.init_mode == "imagenet1k_subset":
            if self.class_index_map is None:
                raise ValueError(
                    "imagenet1k_subset initialization requires class_index_map"
                )
            validate_imagenet1k_class_map(
                self.class_index_map,
                expected_classes=self.num_classes,
                expected_digest=self.class_map_sha256,
            )
        elif self.class_index_map is not None or self.class_map_sha256 is not None:
            raise ValueError(
                "class_index_map/class_map_sha256 are only valid for "
                "imagenet1k_subset initialization"
            )

    @property
    def definition(self) -> ModelDefinition:
        return MODEL_REGISTRY[self.model_key]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelTrainingTask:
    task_id: str
    role: str
    source_id: str
    partition_kind: str
    partition_digest: str
    model: ModelBuildRequest

    def __post_init__(self) -> None:
        if self.role not in MODEL_ROLES:
            raise ValueError(f"Unknown model role: {self.role}")
        if self.role == "reference" and self.partition_kind != "reference":
            raise ValueError("Reference model must use a full-data reference partition")


def get_model_definition(model_key: str) -> ModelDefinition:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {model_key!r}; available: {available}") from exc


def validate_imagenet1k_class_map(
    indices: tuple[int, ...] | list[int],
    *,
    expected_classes: int,
    expected_digest: str | None = None,
) -> str:
    """Validate and identify a dataset-label to ImageNet-1K class map."""

    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("ImageNet-1K class indices must be integers")
    normalized = tuple(indices)
    if len(normalized) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} class indices, found {len(normalized)}"
        )
    if len(set(normalized)) != len(normalized) or any(
        not 0 <= index < IMAGENET1K_CLASS_COUNT for index in normalized
    ):
        raise ValueError(
            "ImageNet-1K class indices must be unique values in [0, 1000)"
        )
    digest = object_sha256(list(normalized))
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            "ImageNet-1K class-map digest mismatch: "
            f"expected={expected_digest} computed={digest}"
        )
    return digest


def initialization_for_recipe(recipe: str) -> str:
    """Resolve one executable initialization recipe without aliases."""

    try:
        return RECIPE_INITIALIZATION[recipe]
    except KeyError as exc:
        raise ValueError(
            f"Unknown initialization recipe {recipe!r}; "
            f"expected one of {sorted(INITIALIZATION_RECIPES)}"
        ) from exc


def _linear_classifier_slot(model: Any) -> tuple[Any, str | int, Any]:
    """Return ``(owner, key, layer)`` for a common linear classifier head."""

    import torch.nn as nn

    for attribute in ("fc", "head"):
        layer = getattr(model, attribute, None)
        if isinstance(layer, nn.Linear):
            return model, attribute, layer

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return model, "classifier", classifier
    if isinstance(classifier, nn.Sequential):
        for index in range(len(classifier) - 1, -1, -1):
            layer = classifier[index]
            if isinstance(layer, nn.Linear):
                return classifier, index, layer

    heads = getattr(model, "heads", None)
    head = getattr(heads, "head", None)
    if isinstance(head, nn.Linear):
        return heads, "head", head
    raise ValueError(f"Cannot locate classification head on {type(model).__name__}")


def _set_classifier_slot(owner: Any, key: str | int, layer: Any) -> None:
    if isinstance(key, int):
        owner[key] = layer
    else:
        setattr(owner, key, layer)


def _replace_classifier(model: Any, num_classes: int) -> None:
    """Replace a common torchvision-style classifier without importing it eagerly."""

    import torch.nn as nn

    owner, key, layer = _linear_classifier_slot(model)
    if layer.out_features != num_classes:
        _set_classifier_slot(owner, key, nn.Linear(layer.in_features, num_classes))


def copy_imagenet1k_classifier_subset(
    model: Any,
    class_index_map: tuple[int, ...] | list[int],
) -> Any:
    """Replace a 1K linear head by an exactly equivalent ordered subset head.

    The resulting model has an ordinary task-specific ``nn.Linear`` classifier,
    so explainers see the same architecture they see for a trained 100-class
    model.  Its logits are exactly the selected rows of the original ImageNet-1K
    logits (up to the model's floating-point arithmetic).
    """

    import torch
    import torch.nn as nn

    indices = tuple(int(index) for index in class_index_map)
    validate_imagenet1k_class_map(indices, expected_classes=len(indices))
    owner, key, original = _linear_classifier_slot(model)
    if original.out_features != IMAGENET1K_CLASS_COUNT:
        raise ValueError(
            "Subset-head conversion requires an ImageNet-1K 1000-class head"
        )
    replacement = nn.Linear(
        original.in_features,
        len(indices),
        bias=original.bias is not None,
    ).to(device=original.weight.device, dtype=original.weight.dtype)
    selection = torch.as_tensor(indices, device=original.weight.device, dtype=torch.long)
    with torch.no_grad():
        replacement.weight.copy_(original.weight.index_select(0, selection))
        if original.bias is not None:
            assert replacement.bias is not None
            replacement.bias.copy_(original.bias.index_select(0, selection))
    replacement.train(original.training)
    _set_classifier_slot(owner, key, replacement)
    return model


def disable_inplace_activations(model: Any) -> Any:
    """Make common activations and residual additions explainer-safe."""

    import torch.nn as nn

    for module in model.modules():
        if hasattr(module, "inplace") and bool(module.inplace):
            module.inplace = False
    _patch_torchvision_residual_blocks(model, nn)
    _patch_timm_residual_blocks(model)
    return model


def _patch_torchvision_residual_blocks(model: Any, nn: Any) -> None:
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except ImportError:
        return

    def basic_forward(block: Any, inputs: Any) -> Any:
        shortcut = inputs
        hidden = block.conv1(inputs)
        hidden = block.bn1(hidden)
        hidden = block.relu(hidden)
        hidden = block.conv2(hidden)
        hidden = block.bn2(hidden)
        if block.downsample is not None:
            shortcut = block.downsample(inputs)
        hidden = hidden + shortcut
        return block.relu2(hidden)

    def bottleneck_forward(block: Any, inputs: Any) -> Any:
        shortcut = inputs
        hidden = block.conv1(inputs)
        hidden = block.bn1(hidden)
        hidden = block.relu(hidden)
        hidden = block.conv2(hidden)
        hidden = block.bn2(hidden)
        hidden = block.relu2(hidden)
        hidden = block.conv3(hidden)
        hidden = block.bn3(hidden)
        if block.downsample is not None:
            shortcut = block.downsample(inputs)
        hidden = hidden + shortcut
        return block.relu3(hidden)

    for module in model.modules():
        if isinstance(module, BasicBlock):
            module.relu.inplace = False
            if not hasattr(module, "relu2"):
                module.relu2 = nn.ReLU(inplace=False)
            module.relu2.inplace = False
            module.forward = MethodType(basic_forward, module)
        elif isinstance(module, Bottleneck):
            module.relu.inplace = False
            if not hasattr(module, "relu2"):
                module.relu2 = nn.ReLU(inplace=False)
            if not hasattr(module, "relu3"):
                module.relu3 = nn.ReLU(inplace=False)
            module.relu2.inplace = False
            module.relu3.inplace = False
            module.forward = MethodType(bottleneck_forward, module)


def _patch_timm_residual_blocks(model: Any) -> None:
    try:
        from timm.models.resnet import BasicBlock, Bottleneck
    except ImportError:
        return

    def basic_forward(block: Any, inputs: Any) -> Any:
        shortcut = inputs
        hidden = block.conv1(inputs)
        hidden = block.bn1(hidden)
        hidden = block.drop_block(hidden)
        hidden = block.act1(hidden)
        hidden = block.aa(hidden)
        hidden = block.conv2(hidden)
        hidden = block.bn2(hidden)
        if block.se is not None:
            hidden = block.se(hidden)
        if block.drop_path is not None:
            hidden = block.drop_path(hidden)
        if block.downsample is not None:
            shortcut = block.downsample(shortcut)
        hidden = hidden + shortcut
        return block.act2(hidden)

    def bottleneck_forward(block: Any, inputs: Any) -> Any:
        shortcut = inputs
        hidden = block.conv1(inputs)
        hidden = block.bn1(hidden)
        hidden = block.act1(hidden)
        hidden = block.conv2(hidden)
        hidden = block.bn2(hidden)
        hidden = block.drop_block(hidden)
        hidden = block.act2(hidden)
        hidden = block.aa(hidden)
        hidden = block.conv3(hidden)
        hidden = block.bn3(hidden)
        if block.se is not None:
            hidden = block.se(hidden)
        if block.drop_path is not None:
            hidden = block.drop_path(hidden)
        if block.downsample is not None:
            shortcut = block.downsample(shortcut)
        hidden = hidden + shortcut
        return block.act3(hidden)

    for module in model.modules():
        if isinstance(module, BasicBlock):
            module.forward = MethodType(basic_forward, module)
        elif isinstance(module, Bottleneck):
            module.forward = MethodType(bottleneck_forward, module)


def _load_model_state(model: Any, checkpoint_path: str, strict: bool) -> None:
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    state = payload
    if isinstance(payload, Mapping):
        for key in ("model_state", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {path} does not contain a model state mapping")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=strict)


def create_model(request: ModelBuildRequest) -> Any:
    """Instantiate a classifier using lazy optional imports.

    ``imagenet1k`` means transfer initialization with a fresh task-specific
    head.  ``imagenet1k_subset`` copies the locked ImageNet-1K classifier rows
    into an ordinary task-specific head and performs no random head reset.
    ``random`` and ``checkpoint`` never trigger a network download.
    """

    definition = request.definition
    pretrained = request.init_mode in {"imagenet1k", "imagenet1k_subset"}
    provider_num_classes = (
        IMAGENET1K_CLASS_COUNT
        if request.init_mode == "imagenet1k_subset"
        else request.num_classes
    )
    if request.seed is not None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - provider also requires torch
            raise RuntimeError("Model construction requires PyTorch") from exc
        torch.manual_seed(request.seed)
    if definition.provider == "timm":
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Model provider 'timm' is not installed") from exc
        model = timm.create_model(
            definition.upstream_name,
            pretrained=pretrained,
            num_classes=provider_num_classes,
        )
    else:
        try:
            from torchvision import models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Model provider 'torchvision' is not installed") from exc
        constructor = getattr(models, definition.upstream_name, None)
        if constructor is None:
            raise RuntimeError(
                f"Installed torchvision has no model {definition.upstream_name!r}"
            )
        weights: Any = None
        if pretrained:
            try:
                weights = models.get_model_weights(constructor).DEFAULT
            except (AttributeError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot resolve pretrained weights for {definition.upstream_name}"
                ) from exc
        model = constructor(weights=weights)
        _replace_classifier(model, provider_num_classes)

    if request.init_mode == "imagenet1k_subset":
        assert request.class_index_map is not None
        copy_imagenet1k_classifier_subset(model, request.class_index_map)

    if request.disable_inplace_activations:
        disable_inplace_activations(model)
    if request.init_mode == "checkpoint":
        assert request.checkpoint_path is not None
        _load_model_state(model, request.checkpoint_path, request.strict_checkpoint)

    model.model_key = definition.key
    model.model_family = definition.family
    model.model_provider = definition.provider
    model.initialization = request.init_mode
    model.initialization_seed = request.seed
    return model


def resolved_preprocessing(model: Any, definition: ModelDefinition) -> dict[str, Any]:
    """Read timm's pretrained data contract, with registry defaults as fallback."""

    config = getattr(model, "pretrained_cfg", {}) or {}
    input_shape = config.get("input_size", (3, definition.input_size, definition.input_size))
    return {
        "input_size": int(input_shape[-1]),
        "mean": tuple(float(value) for value in config.get("mean", definition.mean)),
        "std": tuple(float(value) for value in config.get("std", definition.std)),
        "interpolation": str(config.get("interpolation", definition.interpolation)),
        "crop_percentage": float(config.get("crop_pct", definition.crop_percentage)),
    }


def make_model_training_tasks(
    *,
    model_key: str,
    num_classes: int,
    source_plan: PartitionPlan,
    reference_plan: PartitionPlan,
    init_mode: str,
    base_seed: int = 20260714,
) -> tuple[ModelTrainingTask, ...]:
    """Register every source model plus the common full-data reference model."""

    if source_plan.kind not in {"ind", "overlap"}:
        raise ValueError("source_plan must be an IND or OVERLAP plan")
    if reference_plan.kind != "reference":
        raise ValueError("reference_plan must be a full-data reference plan")
    if source_plan.dataset_manifest_fingerprint != reference_plan.dataset_manifest_fingerprint:
        raise ValueError("Source and reference plans must use the same dataset manifest")

    tasks = [
        ModelTrainingTask(
            task_id=f"{source_plan.kind}-{model_key}-{source.source_id}",
            role="source",
            source_id=source.source_id,
            partition_kind=source_plan.kind,
            partition_digest=source_plan.digest,
            model=ModelBuildRequest(
                model_key=model_key,
                num_classes=num_classes,
                init_mode=init_mode,
                seed=int(stable_seed(base_seed, source_plan.kind, source.source_id, model_key)),
            ),
        )
        for source in source_plan.sources
    ]
    reference = reference_plan.sources[0]
    tasks.append(
        ModelTrainingTask(
            task_id=f"reference-{model_key}",
            role="reference",
            source_id=reference.source_id,
            partition_kind="reference",
            partition_digest=reference_plan.digest,
            model=ModelBuildRequest(
                model_key=model_key,
                num_classes=num_classes,
                init_mode=init_mode,
                seed=int(stable_seed(base_seed, "reference", reference.source_id, model_key)),
            ),
        )
    )
    return tuple(tasks)
