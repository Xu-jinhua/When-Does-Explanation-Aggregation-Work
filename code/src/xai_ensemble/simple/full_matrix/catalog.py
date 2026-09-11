"""Frozen dataset, model, and cell catalog for the complete result matrix."""

from __future__ import annotations

from dataclasses import dataclass

from xai_ensemble.data.specs import HFDatasetSpec, get_dataset_spec
from xai_ensemble.phase0.models import ModelDefinition, get_model_definition
from xai_ensemble.simple.methods import MethodCatalog

DATASET_IDS = (
    "imagenet1k",
    "food101",
    "places365",
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
)

MODEL_KEYS = (
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenetv3_large_100",
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "swin_base_patch4_window7_224",
)

MODEL_LABELS = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet-b0",
    "mobilenetv3_large_100": "mobilenetv3-large",
    "vit_base_patch16_224": "vit-b16",
    "deit_base_patch16_224": "deit-b16",
    "swin_base_patch4_window7_224": "swin-b",
}


@dataclass(frozen=True, slots=True)
class MatrixCell:
    dataset_id: str
    dataset: HFDatasetSpec
    model_key: str
    model: ModelDefinition

    @property
    def model_id(self) -> str:
        return f"{self.dataset_id}-{MODEL_LABELS[self.model_key]}"

    @property
    def cell_id(self) -> str:
        return f"{self.dataset_id}--{self.model_id}"

    @property
    def architecture(self) -> str:
        return self.model.family

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes


def matrix_cells(
    *,
    dataset_ids: tuple[str, ...] = DATASET_IDS,
    model_keys: tuple[str, ...] = MODEL_KEYS,
) -> tuple[MatrixCell, ...]:
    return tuple(
        MatrixCell(
            dataset_id=dataset_id,
            dataset=get_dataset_spec(dataset_id),
            model_key=model_key,
            model=get_model_definition(model_key),
        )
        for dataset_id in dataset_ids
        for model_key in model_keys
    )


def validate_method_rosters(
    methods: MethodCatalog,
    *,
    expected_count: int = 11,
) -> dict[str, tuple[str, ...]]:
    rosters = {
        architecture: tuple(
            method.family
            for method in methods.for_architecture(architecture)  # type: ignore[arg-type]
        )
        for architecture in ("cnn", "vit")
    }
    for architecture, roster in rosters.items():
        if len(roster) != expected_count or len(set(roster)) != expected_count:
            raise ValueError(
                f"{architecture} must have exactly {expected_count} unique explainers; "
                f"found {roster}"
            )
    return rosters


__all__ = [
    "DATASET_IDS",
    "MODEL_KEYS",
    "MODEL_LABELS",
    "MatrixCell",
    "matrix_cells",
    "validate_method_rosters",
]
