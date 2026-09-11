"""Configuration and stable paths for the complete experiment matrix."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.protocol import load_protocol
from xai_ensemble.simple.methods import MethodCatalog, load_method_catalog

from .catalog import DATASET_IDS, MODEL_KEYS, MatrixCell, matrix_cells, validate_method_rosters


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence")
    return value


def _path(value: Any, *, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def _gib(value: Any, *, context: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{context} must be a positive finite GiB value")
    return int(number * 2**30)


@dataclass(frozen=True, slots=True)
class MatrixStorage:
    asset_root: Path
    run_root: Path
    cache_root: Path
    selector_cache_root: Path
    cloudstorage_lock_root: Path
    rclone_binary: Path
    base_remote_root: str
    prefix_remote_root: str
    assumptions_remote_root: str


@dataclass(frozen=True, slots=True)
class MatrixRuntime:
    database_path: Path
    log_directory: Path
    gpu_ids: tuple[int, ...]
    headroom_fraction: float
    max_cpu_jobs: int
    max_retries: int
    poll_seconds: float
    training_reservation_bytes: Mapping[str, int]
    compatibility_reservation_bytes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class MatrixTraining:
    epochs: int
    batch_size: Mapping[str, int]
    validation_batch_size: Mapping[str, int]
    workers: int
    precision: str
    recipe: str
    initialization: str
    class_balance: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FullMatrixExperiment:
    source_path: Path
    experiment_id: str
    seed: int
    protocol_path: Path
    protocol_digest: str
    methods_path: Path
    methods: MethodCatalog
    dataset_ids: tuple[str, ...]
    model_keys: tuple[str, ...]
    storage: MatrixStorage
    runtime: MatrixRuntime
    training: MatrixTraining
    compatibility_samples: int
    raw_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.dataset_ids != DATASET_IDS:
            raise ValueError(f"full matrix datasets are frozen to {DATASET_IDS}")
        if self.model_keys != MODEL_KEYS:
            raise ValueError(f"full matrix models are frozen to {MODEL_KEYS}")
        if not self.runtime.gpu_ids or len(set(self.runtime.gpu_ids)) != len(self.runtime.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if not 0 <= self.runtime.headroom_fraction < 0.5:
            raise ValueError("runtime.headroom_fraction must lie in [0,0.5)")
        if self.runtime.max_cpu_jobs <= 0 or self.runtime.max_retries < 0:
            raise ValueError("runtime max_cpu_jobs/retries are invalid")
        if self.training.epochs != 100 or self.training.precision != "bf16":
            raise ValueError("reference training is frozen to 100 epochs and bf16")
        if self.training.recipe != "timm_finetune" or self.training.initialization != "imagenet1k":
            raise ValueError("reference models require ImageNet-1K initialized timm fine-tuning")
        if self.compatibility_samples < 2:
            raise ValueError("compatibility_samples must be at least two")
        validate_method_rosters(self.methods)
        if len(self.cells()) != 112:
            raise RuntimeError("full matrix must contain 14 x 8 = 112 cells")

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-full-matrix-v1",
                "config": self.raw_config,
                "protocol_digest": self.protocol_digest,
                "method_catalog_digest": self.methods.source_digest,
                "cells": [cell.cell_id for cell in self.cells()],
            }
        )

    @property
    def scheduler_digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-full-matrix-scheduler-v1",
                "experiment_digest": self.digest,
                "database": str(self.runtime.database_path),
            }
        )

    @property
    def generated_root(self) -> Path:
        return self.storage.run_root / "generated"

    @property
    def gate_config_path(self) -> Path:
        return self.generated_root / "compatibility-simple.yaml"

    @property
    def base_config_path(self) -> Path:
        return self.generated_root / "paper-main.yaml"

    @property
    def assumptions_config_path(self) -> Path:
        return self.generated_root / "paper-assumptions.yaml"

    @property
    def prefix_config_path(self) -> Path:
        return self.generated_root / "paper-noise-prefix.yaml"

    @property
    def active_cells_path(self) -> Path:
        return self.generated_root / "active-cells.json"

    @property
    def result_root(self) -> Path:
        return self.storage.run_root / "results"

    def cells(self) -> tuple[MatrixCell, ...]:
        return matrix_cells(dataset_ids=self.dataset_ids, model_keys=self.model_keys)

    def cell(self, cell_id: str) -> MatrixCell:
        for cell in self.cells():
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)

    def manifest_path(self, dataset_id: str) -> Path:
        return self.storage.asset_root / "datasets" / dataset_id / "manifest.json"

    def partition_path(self, dataset_id: str, split: str) -> Path:
        return (
            self.storage.asset_root
            / "datasets"
            / dataset_id
            / "partitions"
            / f"reference-{split}.json"
        )

    def sample_ids_path(self, dataset_id: str) -> Path:
        return self.storage.run_root / "compatibility" / "samples" / f"{dataset_id}.json"

    def mean_path(self, cell: MatrixCell) -> Path:
        return self.storage.asset_root / "means" / cell.dataset_id / cell.model_key

    def checkpoint_directory(self, cell: MatrixCell) -> Path:
        return self.storage.asset_root / "reference-models" / cell.cell_id / "checkpoints"

    def checkpoint_path(self, cell: MatrixCell) -> Path:
        return self.checkpoint_directory(cell) / "inference.pt"

    def training_scratch_directory(self, cell: MatrixCell) -> Path:
        return self.storage.run_root / "training-scratch" / cell.cell_id / "checkpoints"

    def compatibility_directory(self, cell: MatrixCell) -> Path:
        return self.storage.run_root / "compatibility" / "cells" / cell.cell_id

    def cache_directory(self, dataset_id: str) -> Path:
        return self.storage.cache_root / dataset_id


def load_full_matrix_experiment(path: str | Path) -> FullMatrixExperiment:
    source = Path(path).expanduser().resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")) or {}, context="config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("full matrix config schema_version must be 1")
    base = source.parent
    protocol_path = _path(root["protocol"], base=base)
    protocol = load_protocol(protocol_path)
    protocol_storage = _mapping(protocol.section("storage"), context="protocol storage")
    if protocol_storage.get("provider") != "cloudstorage":
        raise ValueError("full matrix requires the protocol CloudStorage provider")
    methods_path = _path(root["methods_file"], base=base)
    methods = load_method_catalog(methods_path)

    storage_row = _mapping(root["storage"], context="storage")
    storage = MatrixStorage(
        asset_root=_path(storage_row["asset_root"], base=base),
        run_root=_path(storage_row["run_root"], base=base),
        cache_root=_path(storage_row["cache_root"], base=base),
        selector_cache_root=_path(storage_row["selector_cache_root"], base=base),
        cloudstorage_lock_root=_path(protocol_storage["lock_root"], base=protocol_path.parent),
        rclone_binary=_path(storage_row["rclone_binary"], base=base),
        base_remote_root=str(storage_row["base_remote_root"]),
        prefix_remote_root=str(storage_row["prefix_remote_root"]),
        assumptions_remote_root=str(storage_row["assumptions_remote_root"]),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    runtime = MatrixRuntime(
        database_path=_path(runtime_row["database_path"], base=base),
        log_directory=_path(runtime_row["log_directory"], base=base),
        gpu_ids=tuple(
            int(value) for value in _sequence(runtime_row.get("gpu_ids", [0, 1]), context="gpu_ids")
        ),
        headroom_fraction=float(runtime_row.get("headroom_fraction", 0.05)),
        max_cpu_jobs=int(runtime_row.get("max_cpu_jobs", 2)),
        max_retries=int(runtime_row.get("max_retries", 1)),
        poll_seconds=float(runtime_row.get("poll_seconds", 2.0)),
        training_reservation_bytes={
            str(key): _gib(value, context=f"training reservation {key}")
            for key, value in _mapping(
                runtime_row.get("training_reservation_gib", {"cnn": 44, "vit": 44}),
                context="training reservations",
            ).items()
        },
        compatibility_reservation_bytes={
            str(key): _gib(value, context=f"compatibility reservation {key}")
            for key, value in _mapping(
                runtime_row.get("compatibility_reservation_gib", {"cnn": 44, "vit": 44}),
                context="compatibility reservations",
            ).items()
        },
    )
    training_row = _mapping(root["training"], context="training")
    training = MatrixTraining(
        epochs=int(training_row.get("epochs", 100)),
        batch_size={
            str(key): int(value)
            for key, value in _mapping(training_row["batch_size"], context="batch_size").items()
        },
        validation_batch_size={
            str(key): int(value)
            for key, value in _mapping(
                training_row["validation_batch_size"], context="validation_batch_size"
            ).items()
        },
        workers=int(training_row.get("workers", 8)),
        precision=str(training_row.get("precision", "bf16")),
        recipe=str(training_row.get("recipe", "timm_finetune")),
        initialization=str(training_row.get("initialization", "imagenet1k")),
        class_balance={
            str(key): str(value)
            for key, value in _mapping(
                training_row.get("class_balance", {}), context="class_balance"
            ).items()
        },
    )
    dataset_ids = tuple(
        str(value) for value in _sequence(root.get("datasets", DATASET_IDS), context="datasets")
    )
    model_keys = tuple(
        str(value) for value in _sequence(root.get("models", MODEL_KEYS), context="models")
    )
    return FullMatrixExperiment(
        source_path=source,
        experiment_id=str(root["experiment_id"]),
        seed=int(root.get("seed", 20260714)),
        protocol_path=protocol_path,
        protocol_digest=protocol.digest,
        methods_path=methods_path,
        methods=methods,
        dataset_ids=dataset_ids,
        model_keys=model_keys,
        storage=storage,
        runtime=runtime,
        training=training,
        compatibility_samples=int(root.get("compatibility_samples", 2)),
        raw_config=dict(root),
    )


__all__ = [
    "FullMatrixExperiment",
    "MatrixRuntime",
    "MatrixStorage",
    "MatrixTraining",
    "load_full_matrix_experiment",
]
