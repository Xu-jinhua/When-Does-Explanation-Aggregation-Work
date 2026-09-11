"""Configuration and deterministic DAG for the assumption experiments."""

from __future__ import annotations

import os
import posixpath
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from xai_ensemble.core.hashing import file_sha256, object_sha256, stable_seed
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path

from ..config import (
    ConditionConfig,
    DatasetConfig,
    ModelConfig,
    Phase1Task,
    SimpleExperiment,
    StorageConfig,
    load_experiment,
)
from ..manifest_identity import dataset_manifest_identity_sha256

Setting = Literal["ind", "matched-naive", "oracle-noise"]
DistanceModel = Literal["spearman", "kendall"]
SETTING_ORDER: tuple[Setting, ...] = ("ind", "matched-naive", "oracle-noise")
RULES = ("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze")


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
    result = int(float(value) * 2**30)
    if result <= 0:
        raise ValueError(f"{context} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class AssumptionRuntime:
    database_path: Path
    log_directory: Path
    gpu_ids: tuple[int, ...]
    inference_batch_size: int
    training_reservation_bytes: Mapping[str, int]
    phase1_reservation_bytes: Mapping[str, int]
    rank_reservation_bytes: int
    selection_reservation_bytes: int
    evaluation_reservation_bytes: int
    phase1_batch_caps: Mapping[str, int]
    headroom_fraction: float
    max_retries: int
    cpu_workers: int

    def __post_init__(self) -> None:
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if self.inference_batch_size <= 0 or self.cpu_workers <= 0:
            raise ValueError("runtime batch size and cpu_workers must be positive")
        if set(self.training_reservation_bytes) != {"cnn", "vit"}:
            raise ValueError("training reservations must define cnn and vit")
        if set(self.phase1_reservation_bytes) != {"cnn", "vit"}:
            raise ValueError("phase1 reservations must define cnn and vit")
        if not 0.0 <= self.headroom_fraction < 0.5:
            raise ValueError("runtime.headroom_fraction must lie in [0,0.5)")
        if self.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")
        if any(value <= 0 for value in self.phase1_batch_caps.values()):
            raise ValueError("phase1 batch caps must be positive")


@dataclass(frozen=True, slots=True)
class SourceTrainingConfig:
    train_split: str
    validation_split: str
    epochs: int
    batch_size: Mapping[str, int]
    validation_batch_size: Mapping[str, int]
    precision: Literal["off", "fp16", "bf16"]
    class_balance: Mapping[str, Literal["none", "weighted_loss", "balanced_sampler"]]

    def __post_init__(self) -> None:
        if self.train_split == self.validation_split:
            raise ValueError("training and validation splits must differ")
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if set(self.batch_size) != {"cnn", "vit"}:
            raise ValueError("training.batch_size must define cnn and vit")
        if set(self.validation_batch_size) != {"cnn", "vit"}:
            raise ValueError("training.validation_batch_size must define cnn and vit")
        if any(
            value <= 0
            for value in (*self.batch_size.values(), *self.validation_batch_size.values())
        ):
            raise ValueError("training batch sizes must be positive")

    def balance_for(self, model_id: str) -> str:
        return self.class_balance.get(model_id, self.class_balance.get("default", "none"))


@dataclass(frozen=True, slots=True)
class NoiseSelectionConfig:
    alpha: float
    bootstrap_replicates: int
    min_prefix: int
    selection_rule: Literal["largest_not_rejected"]
    spearman_artifact: Path
    spearman_calibration_config: Path

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("selection.alpha must lie in (0,1)")
        if self.bootstrap_replicates <= 0 or self.min_prefix < 2:
            raise ValueError("selection bootstrap count and min_prefix are invalid")
        if self.selection_rule != "largest_not_rejected":
            raise ValueError("Oracle NOISE is fixed to the largest non-rejected prefix")


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    dataset: DatasetConfig
    reference_model: ModelConfig
    methods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartitionTask:
    task_id: str
    digest: str
    cell: Cell
    source_count: int
    family_id: int = 0

    @property
    def artifact_root(self) -> str:
        return posixpath.join("partitions", self.cell.cell_id, self.digest)


@dataclass(frozen=True, slots=True)
class TrainingTask:
    task_id: str
    digest: str
    cell: Cell
    source_id: str
    partition_task_id: str
    family_id: int = 0

    @property
    def artifact_root(self) -> str:
        return posixpath.join("models", self.cell.cell_id, self.source_id, self.digest)


@dataclass(frozen=True, slots=True)
class SourcePhase1Task:
    task_id: str
    digest: str
    cell: Cell
    source_id: str
    condition: ConditionConfig
    training_task_id: str
    family_id: int = 0

    @property
    def artifact_root(self) -> str:
        return posixpath.join(
            "source-scopes",
            self.cell.cell_id,
            self.source_id,
            self.condition.condition_id,
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class SelectionTask:
    task_id: str
    digest: str
    cell: Cell
    distance_model: DistanceModel
    aggregation: Literal["borda", "kemeny"]

    @property
    def artifact_root(self) -> str:
        return posixpath.join("selection", self.cell.cell_id, self.distance_model, self.digest)


@dataclass(frozen=True, slots=True)
class RankTask:
    task_id: str
    digest: str
    cell: Cell
    setting: Setting
    condition: ConditionConfig
    source_id: str | None
    distance_model: DistanceModel | None
    methods: tuple[str, ...]
    source_method_pairs: tuple[tuple[str, str], ...]
    selection_task_id: str | None
    family_id: int | None = None

    @property
    def artifact_root(self) -> str:
        qualifier = self.source_id or self.distance_model or "combined"
        return posixpath.join(
            "ranks",
            self.setting,
            self.cell.cell_id,
            qualifier,
            self.condition.condition_id,
            "p16",
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    task_id: str
    digest: str
    rank_task_id: str
    cell: Cell
    setting: Setting
    condition: ConditionConfig
    source_id: str | None
    distance_model: DistanceModel | None
    family_id: int | None = None

    @property
    def artifact_root(self) -> str:
        qualifier = self.source_id or self.distance_model or "combined"
        return posixpath.join(
            "evaluations",
            self.setting,
            self.cell.cell_id,
            qualifier,
            self.condition.condition_id,
            "p16",
            "k20",
            "dataset-mean",
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class AssumptionExperiment:
    source_path: Path
    assumption_id: str
    base: SimpleExperiment
    storage: StorageConfig
    runtime: AssumptionRuntime
    training: SourceTrainingConfig
    selection: NoiseSelectionConfig
    settings: tuple[Setting, ...]
    split: str
    patch_size: int
    k: int
    assignment_seed: int
    matched_source_selection: Literal["all"]
    raw_config: Mapping[str, Any]
    partition_families: int = 3
    _task_cache: dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.storage.remote_root == self.base.storage.remote_root:
            raise ValueError("assumption artifacts must use an independent remote root")
        if self.runtime.database_path == self.base.runtime.database_path:
            raise ValueError("assumption jobs must use an independent SQLite database")
        if self.patch_size != 16 or self.k != 20:
            raise ValueError("assumption experiments are fixed to p=16 and k=20")
        if not self.settings or len(set(self.settings)) != len(self.settings):
            raise ValueError("science.settings must contain at least one unique setting")
        if any(setting not in SETTING_ORDER for setting in self.settings):
            raise ValueError(
                f"science.settings must be a subset of {SETTING_ORDER}; found {self.settings}"
            )
        if self.split != "test":
            raise ValueError("the accepted IND/NOISE protocol operates on the test split")
        if self.partition_families != 3:
            raise ValueError("The full IND design requires three partition families")
        if self.matched_source_selection != "all":
            raise ValueError("The full IND design requires all eleven matched sources per family")
        for dataset in self.base.datasets:
            if self.split not in dataset.splits:
                raise ValueError(f"Base dataset {dataset.dataset_id} lacks split {self.split}")

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-assumptions-full-ind-v2",
                "config": self.raw_config,
                "partition_families": self.partition_families,
                "matched_source_selection": self.matched_source_selection,
                "base_phase1_digest": self.base.phase1_digest,
                "method_catalog_digest": self.base.methods.source_digest,
                "spearman_calibration_config_sha256": file_sha256(
                    self.selection.spearman_calibration_config
                ),
            }
        )

    @property
    def scheduler_digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-assumptions-scheduler-v1",
                "assumption_id": self.assumption_id,
                "experiment_digest": self.digest,
                "database": str(self.runtime.database_path),
                "remote_root": self.storage.remote_root,
            }
        )

    def has_setting(self, setting: Setting) -> bool:
        return setting in self.settings

    @property
    def uses_source_bank(self) -> bool:
        return self.has_setting("ind") or self.has_setting("matched-naive")

    @property
    def uses_oracle_noise(self) -> bool:
        return self.has_setting("oracle-noise")

    def cells(self) -> tuple[Cell, ...]:
        if cached := self._task_cache.get("cells"):
            return cached
        result = []
        for model in self.base.models:
            methods = tuple(
                definition.family
                for definition in self.base.methods.for_architecture(model.architecture)
            )
            result.append(
                Cell(
                    cell_id=f"{model.dataset_id}--{model.model_id}",
                    dataset=self.base.dataset(model.dataset_id),
                    reference_model=model,
                    methods=methods,
                )
            )
        value = tuple(result)
        self._task_cache["cells"] = value
        return value

    def cell(self, cell_id: str) -> Cell:
        for item in self.cells():
            if item.cell_id == cell_id:
                return item
        raise KeyError(cell_id)

    def source_ids(self, cell: Cell, family_id: int | None = None) -> tuple[str, ...]:
        families = range(self.partition_families) if family_id is None else (family_id,)
        return tuple(
            f"family-{family:02d}-source-{index:02d}"
            for family in families
            for index in range(len(cell.methods))
        )

    def partition_seed(self, cell: Cell, family_id: int) -> int:
        return stable_seed("ind-partition-family-v2", self.assignment_seed, cell.cell_id, family_id)

    def training_seed(self, task: TrainingTask) -> int:
        return stable_seed("ind-source-training-v2", self.assignment_seed, task.cell.cell_id, task.source_id)

    def method_assignment(self, cell: Cell, family_id: int | None = None) -> tuple[tuple[str, str], ...]:
        if family_id is None:
            return tuple(pair for family in range(self.partition_families)
                         for pair in self.method_assignment(cell, family))
        cache_key = f"assignment:{cell.cell_id}:{family_id}"
        if cached := self._task_cache.get(cache_key):
            return cached
        methods = list(cell.methods)
        random.Random(
            stable_seed("simple-assumptions-method-assignment", self.assignment_seed, cell.cell_id, family_id)
        ).shuffle(methods)
        value = tuple(zip(self.source_ids(cell, family_id), methods, strict=True))
        self._task_cache[cache_key] = value
        return value

    def spearman_family_identity(self) -> Mapping[str, Any]:
        config = (
            yaml.safe_load(self.selection.spearman_calibration_config.read_text(encoding="utf-8"))
            or {}
        )
        if not isinstance(config, Mapping):
            raise TypeError("Spearman calibration config must be a mapping")
        return {
            "schema": "simple-spearman-mallows-p196-v2",
            "artifact_schema_version": 3,
            "n_items": 196,
            "theoretical_maximum_distance": 196 * (196 * 196 - 1) // 3,
            "support_completion": "exact_reverse_symmetry",
            "full_support": True,
            "seed": self.assignment_seed,
            "calibration_config_digest": object_sha256(config),
            "bootstrap_replicates": self.selection.bootstrap_replicates,
            "projected_prefix_count": max(len(cell.methods) - 1 for cell in self.cells()),
            "synthetic_validation_count": 4096,
            "synthetic_q": [0.05, 0.25, 0.5, 0.8, 1.0],
        }

    def partition_tasks(self) -> tuple[PartitionTask, ...]:
        if "partitions" in self._task_cache:
            return self._task_cache["partitions"]
        result = []
        if self.uses_source_bank:
            for cell in self.cells():
                for family_id in range(self.partition_families):
                    identity = {
                        "schema": "simple-assumptions-partition-v2",
                        "cell": cell.cell_id,
                        "family_id": family_id,
                        "manifest_sha256": dataset_manifest_identity_sha256(
                            resolve_full_matrix_runtime_path(cell.dataset.manifest_path)
                        ),
                        "train_split": self.training.train_split,
                        "validation_split": self.training.validation_split,
                        "source_count": len(cell.methods),
                        "seed": self.partition_seed(cell, family_id),
                        "strategy": "disjoint_stratified_full",
                    }
                    digest = object_sha256(identity)
                    result.append(PartitionTask(
                        f"partition--{cell.cell_id}--family-{family_id:02d}--{digest[:10]}",
                        digest, cell, len(cell.methods), family_id,
                    ))
        value = tuple(result)
        self._task_cache["partitions"] = value
        return value

    def training_tasks(self) -> tuple[TrainingTask, ...]:
        if "training" in self._task_cache:
            return self._task_cache["training"]
        result = []
        for partition in self.partition_tasks():
            cell = partition.cell
            for source_id in self.source_ids(cell, partition.family_id):
                identity = {
                    "schema": "simple-assumptions-source-training-v2",
                    "cell": cell.cell_id,
                    "family_id": partition.family_id,
                    "source_id": source_id,
                    "partition_digest": partition.digest,
                    "model_key": cell.reference_model.model_key,
                    "num_classes": cell.reference_model.num_classes,
                    "recipe": "random_scratch",
                    "epochs": self.training.epochs,
                    "batch_size": self.training.batch_size[cell.reference_model.architecture],
                    "validation_batch_size": self.training.validation_batch_size[
                        cell.reference_model.architecture
                    ],
                    "precision": self.training.precision,
                    "class_balance": self.training.balance_for(cell.reference_model.model_id),
                    "seed": stable_seed("ind-source-training-v2", self.assignment_seed, cell.cell_id, source_id),
                }
                digest = object_sha256(identity)
                result.append(TrainingTask(
                    f"train--{cell.cell_id}--{source_id}--{digest[:10]}",
                    digest, cell, source_id, partition.task_id, partition.family_id,
                ))
        value = tuple(result)
        self._task_cache["training"] = value
        return value

    def source_phase1_tasks(self) -> tuple[SourcePhase1Task, ...]:
        if "source_phase1" in self._task_cache:
            return self._task_cache["source_phase1"]
        result = []
        for train in self.training_tasks():
            cell = train.cell
            for condition in self.base.conditions:
                identity = {
                    "schema": "simple-assumptions-source-phase1-scope-v3",
                    "cell": cell.cell_id,
                    "family_id": train.family_id,
                    "source_id": train.source_id,
                    "condition": asdict(condition),
                    "training_digest": train.digest,
                    "reference_phase1_digest": self.base.phase1_digest,
                    "methods": list(cell.methods),
                    "patch_method_variant": "p16",
                    "artifact_representation": "p16-rank-and-simpleavg-patch-score",
                    "simpleavg_normalization": self.base.phase2.simpleavg_normalization,
                    "target_policy": "full_reference_clean_fp32_prediction",
                    "precision": "fp32",
                }
                digest = object_sha256(identity)
                result.append(SourcePhase1Task(
                    f"source-phase1--{cell.cell_id}--{train.source_id}--{condition.condition_id}--{digest[:10]}",
                    digest, cell, train.source_id, condition, train.task_id, train.family_id,
                ))
        value = tuple(result)
        self._task_cache["source_phase1"] = value
        return value

    def selection_tasks(self) -> tuple[SelectionTask, ...]:
        if cached := self._task_cache.get("selection"):
            return cached
        if not self.uses_oracle_noise:
            self._task_cache["selection"] = ()
            return ()
        result = []
        for cell in self.cells():
            for distance, aggregation in (("spearman", "borda"), ("kendall", "kemeny")):
                identity = {
                    "schema": "simple-oracle-noise-selection-v1",
                    "cell": cell.cell_id,
                    "distance": distance,
                    "aggregation": aggregation,
                    "base_phase2_task": self.base_phase2_task(cell, "clean").digest,
                    "methods": list(cell.methods),
                    "fidelity_scope": "complete_test_set",
                    "selection_scope": "complete_test_set",
                    "alpha": self.selection.alpha,
                    "bootstrap_replicates": self.selection.bootstrap_replicates,
                    "selection_rule": self.selection.selection_rule,
                    "gof_family": (
                        object_sha256(self.spearman_family_identity())
                        if distance == "spearman"
                        else "kendall-mallows-exact-distance-v1"
                    ),
                    "patch_size": self.patch_size,
                    "k": self.k,
                }
                digest = object_sha256(identity)
                result.append(
                    SelectionTask(
                        f"select--{cell.cell_id}--{distance}--{digest[:10]}",
                        digest,
                        cell,
                        distance,  # type: ignore[arg-type]
                        aggregation,  # type: ignore[arg-type]
                    )
                )
        value = tuple(result)
        self._task_cache["selection"] = value
        return value

    def rank_tasks(self) -> tuple[RankTask, ...]:
        if "ranks" in self._task_cache:
            return self._task_cache["ranks"]
        result = []
        phase1 = {
            (task.cell.cell_id, task.source_id, task.condition.condition_id): task
            for task in self.source_phase1_tasks()
        }
        selections = {
            (task.cell.cell_id, task.distance_model): task for task in self.selection_tasks()
        }
        for cell in self.cells():
            for condition in self.base.conditions:
                if self.uses_source_bank:
                    for family_id in range(self.partition_families):
                        source_ids = self.source_ids(cell, family_id)
                        assignment = self.method_assignment(cell, family_id)
                        scopes = [phase1[cell.cell_id, source, condition.condition_id]
                                  for source in source_ids]
                        if self.has_setting("ind"):
                            result.append(self._rank_task(
                                cell=cell, setting="ind", condition=condition,
                                source_id=f"family-{family_id:02d}", family_id=family_id,
                                distance_model=None, methods=tuple(method for _, method in assignment),
                                pairs=assignment, selection=None, source_scopes=scopes,
                            ))
                        if self.has_setting("matched-naive"):
                            for source_id in source_ids:
                                result.append(self._rank_task(
                                    cell=cell, setting="matched-naive", condition=condition,
                                    source_id=source_id, family_id=family_id, distance_model=None,
                                    methods=cell.methods,
                                    pairs=tuple((source_id, method) for method in cell.methods),
                                    selection=None,
                                    source_scopes=[phase1[cell.cell_id, source_id, condition.condition_id]],
                                ))
                if self.has_setting("oracle-noise"):
                    for distance in ("spearman", "kendall"):
                        result.append(self._rank_task(
                            cell=cell, setting="oracle-noise", condition=condition,
                            source_id=None, distance_model=distance, methods=cell.methods,
                            pairs=(), selection=selections[cell.cell_id, distance],
                            source_scopes=(),
                        ))
        value = tuple(result)
        self._task_cache["ranks"] = value
        return value

    def _rank_task(
        self,
        *,
        cell: Cell,
        setting: Setting,
        condition: ConditionConfig,
        source_id: str | None,
        distance_model: DistanceModel | None,
        methods: tuple[str, ...],
        pairs: tuple[tuple[str, str], ...],
        selection: SelectionTask | None,
        source_scopes: Sequence[SourcePhase1Task],
        family_id: int | None = None,
    ) -> RankTask:
        identity = {
            "schema": "simple-assumptions-rank-v1",
            "setting": setting,
            "cell": cell.cell_id,
            "condition": asdict(condition),
            "source_id": source_id,
            "partition_family_id": family_id,
            "distance_model": distance_model,
            "methods": list(methods),
            "source_method_pairs": [list(pair) for pair in pairs],
            "source_scope_digests": [task.digest for task in source_scopes],
            "selection_digest": None if selection is None else selection.digest,
            "base_phase2_digest": (
                self.base_phase2_task(cell, condition.condition_id).digest
                if setting == "oracle-noise"
                else None
            ),
            "rules": list(RULES),
            "patch_size": self.patch_size,
            "simpleavg_normalization": self.base.phase2.simpleavg_normalization,
            "rrf_c": self.base.phase2.rrf_c,
            "kemeny_starts": self.base.phase2.kemeny_starts,
            "kemeny_max_passes": self.base.phase2.kemeny_max_passes,
        }
        digest = object_sha256(identity)
        qualifier = source_id or distance_model or "combined"
        return RankTask(
            f"rank--{setting}--{cell.cell_id}--{qualifier}--{condition.condition_id}--{digest[:10]}",
            digest,
            cell,
            setting,
            condition,
            source_id,
            distance_model,
            methods,
            pairs,
            None if selection is None else selection.task_id,
            family_id,
        )

    def evaluation_tasks(self) -> tuple[EvaluationTask, ...]:
        if cached := self._task_cache.get("evaluations"):
            return cached
        result = []
        model_artifacts = {}
        for cell in self.cells():
            checkpoint_sha256 = (
                None
                if cell.reference_model.checkpoint_path is None
                else file_sha256(
                    resolve_full_matrix_runtime_path(cell.reference_model.checkpoint_path)
                )
            )
            mean_manifest = (
                resolve_full_matrix_runtime_path(cell.reference_model.mean_path) / "manifest.json"
                if resolve_full_matrix_runtime_path(cell.reference_model.mean_path).is_dir()
                else resolve_full_matrix_runtime_path(cell.reference_model.mean_path)
            )
            model_artifacts[cell.cell_id] = {
                "reference_checkpoint_sha256": checkpoint_sha256,
                "mean_artifact_sha256": file_sha256(mean_manifest),
            }
        for rank in self.rank_tasks():
            identity = {
                "schema": "simple-assumptions-evaluation-v1",
                "rank_task_digest": rank.digest,
                "setting": rank.setting,
                "reference_model": rank.cell.reference_model.model_id,
                "reference_model_key": rank.cell.reference_model.model_key,
                **model_artifacts[rank.cell.cell_id],
                "condition": asdict(rank.condition),
                "patch_size": self.patch_size,
                "k": self.k,
                "fill": "dataset_mean",
                "target_policy": "full_reference_clean_fp32_prediction",
            }
            digest = object_sha256(identity)
            result.append(
                EvaluationTask(
                    f"evaluate--{rank.task_id}--{digest[:10]}",
                    digest,
                    rank.task_id,
                    rank.cell,
                    rank.setting,
                    rank.condition,
                    rank.source_id,
                    rank.distance_model,
                    rank.family_id,
                )
            )
        value = tuple(result)
        self._task_cache["evaluations"] = value
        return value

    def base_phase2_task(self, cell: Cell, condition_id: str) -> Any:
        matches = [
            task
            for task in self.base.phase2_tasks()
            if task.dataset.dataset_id == cell.dataset.dataset_id
            and task.model.model_id == cell.reference_model.model_id
            and task.split == self.split
            and task.condition.condition_id == condition_id
            and task.patch_size == self.patch_size
            and task.ensemble.ensemble_id == "all-paper-methods"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one p=16 NAIVE Phase 2 source for {cell.cell_id}/{condition_id}; found {len(matches)}"
            )
        return matches[0]

    def source_model(self, task: TrainingTask, checkpoint_path: Path) -> ModelConfig:
        model = task.cell.reference_model
        return ModelConfig(
            model_id=f"{model.model_id}--{task.source_id}",
            dataset_id=model.dataset_id,
            model_key=model.model_key,
            num_classes=model.num_classes,
            init_mode="checkpoint",
            checkpoint_path=checkpoint_path,
            strict_checkpoint=True,
            class_index_map=None,
            mean_path=model.mean_path,
            mean_key=model.mean_key,
            architecture=model.architecture,
        )

    def method_phase1_tasks(
        self,
        scope: SourcePhase1Task,
        *,
        checkpoint_path: Path,
    ) -> tuple[Phase1Task, ...]:
        training = self.find_training_task(scope.training_task_id)
        model = self.source_model(training, checkpoint_path)
        tasks = []
        for definition in self.base.methods.for_architecture(model.architecture):
            variants = definition.instances(model.architecture)
            if definition.family in {"FeatureAblation", "Occlusion"}:
                variants = tuple(item for item in variants if item.variant == "p16")
            identity = {
                "schema": "simple-assumptions-source-phase1-method-v2",
                "scope_digest": scope.digest,
                "training_digest": training.digest,
                "model_id": model.model_id,
                "condition": asdict(scope.condition),
                "family": definition.family,
                "variants": [item.digest for item in variants],
                "artifact_representation": "p16-rank-and-simpleavg-patch-score",
                "patch_size": self.patch_size,
                "simpleavg_normalization": self.base.phase2.simpleavg_normalization,
                "target_policy": "full_reference_clean_fp32_prediction",
                "precision": "fp32",
            }
            digest = object_sha256(identity)
            tasks.append(
                Phase1Task(
                    f"{scope.cell.cell_id}--{scope.source_id}--{scope.condition.condition_id}--{definition.family}--{digest[:10]}",
                    scope.cell.dataset,
                    model,
                    self.split,
                    scope.condition,
                    definition.family,
                    variants,
                    digest,
                )
            )
        return tuple(tasks)

    def find_partition_task(self, task_id: str) -> PartitionTask:
        return _find(self.partition_tasks(), task_id)

    def find_training_task(self, task_id: str) -> TrainingTask:
        return _find(self.training_tasks(), task_id)

    def find_source_phase1_task(self, task_id: str) -> SourcePhase1Task:
        return _find(self.source_phase1_tasks(), task_id)

    def find_selection_task(self, task_id: str) -> SelectionTask:
        return _find(self.selection_tasks(), task_id)

    def find_rank_task(self, task_id: str) -> RankTask:
        return _find(self.rank_tasks(), task_id)

    def find_evaluation_task(self, task_id: str) -> EvaluationTask:
        return _find(self.evaluation_tasks(), task_id)


def _find(tasks: Sequence[Any], task_id: str) -> Any:
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown assumptions task {task_id!r}")


def load_assumption_experiment(path: str | Path) -> AssumptionExperiment:
    source = Path(path).expanduser().resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")) or {}, context="config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("Assumption config schema_version must be 1")
    base = load_experiment(_path(root["base_config"], base=source.parent))
    storage_row = _mapping(root["storage"], context="storage")
    assumption_id = str(root["assumption_id"])
    storage = StorageConfig(
        remote_root=os.path.expandvars(str(storage_row["remote_root"])).rstrip("/"),
        scratch_root=_path(storage_row["scratch_root"], base=source.parent),
        rclone_binary=_path(
            storage_row.get("rclone_binary", base.storage.rclone_binary), base=source.parent
        ),
        spool_root=_path(
            storage_row.get("spool_root", f"/dev/shm/xai-simple/{assumption_id}"),
            base=source.parent,
        ),
        spool_max_bytes=int(float(storage_row.get("spool_max_gib", 96)) * 2**30),
        spool_min_free_bytes=int(float(storage_row.get("spool_min_free_gib", 32)) * 2**30),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    train_res = _mapping(runtime_row["training_reservation_gib"], context="training reservations")
    phase1_res = _mapping(runtime_row["phase1_reservation_gib"], context="phase1 reservations")
    runtime = AssumptionRuntime(
        database_path=_path(runtime_row["database_path"], base=source.parent),
        log_directory=_path(runtime_row["log_directory"], base=source.parent),
        gpu_ids=tuple(
            int(item) for item in _sequence(runtime_row.get("gpu_ids", [0, 1]), context="gpu_ids")
        ),
        inference_batch_size=int(runtime_row.get("inference_batch_size", 512)),
        training_reservation_bytes={
            key: _gib(value, context=f"training.{key}") for key, value in train_res.items()
        },
        phase1_reservation_bytes={
            key: _gib(value, context=f"phase1.{key}") for key, value in phase1_res.items()
        },
        rank_reservation_bytes=_gib(
            runtime_row.get("rank_reservation_gib", 12), context="rank reservation"
        ),
        selection_reservation_bytes=_gib(
            runtime_row.get("selection_reservation_gib", 12), context="selection reservation"
        ),
        evaluation_reservation_bytes=_gib(
            runtime_row.get("evaluation_reservation_gib", 10), context="evaluation reservation"
        ),
        phase1_batch_caps={
            str(key): int(value)
            for key, value in _mapping(
                runtime_row.get("phase1_batch_caps", {}), context="phase1 caps"
            ).items()
        },
        headroom_fraction=float(runtime_row.get("headroom_fraction", 0.10)),
        max_retries=int(runtime_row.get("max_retries", 1)),
        cpu_workers=int(runtime_row.get("cpu_workers", 2)),
    )
    training_row = _mapping(root["training"], context="training")
    training = SourceTrainingConfig(
        train_split=str(training_row.get("train_split", "train")),
        validation_split=str(training_row.get("validation_split", "validation")),
        epochs=int(training_row.get("epochs", 100)),
        batch_size={
            str(key): int(value)
            for key, value in _mapping(
                training_row["batch_size"], context="training batch_size"
            ).items()
        },
        validation_batch_size={
            str(key): int(value)
            for key, value in _mapping(
                training_row["validation_batch_size"], context="validation batch_size"
            ).items()
        },
        precision=str(training_row.get("precision", "bf16")),  # type: ignore[arg-type]
        class_balance={
            str(key): str(value)
            for key, value in _mapping(
                training_row.get("class_balance", {"default": "none"}), context="class_balance"
            ).items()
        },  # type: ignore[arg-type]
    )
    selection_row = _mapping(root["selection"], context="selection")
    selection = NoiseSelectionConfig(
        alpha=float(selection_row.get("alpha", 0.05)),
        bootstrap_replicates=int(selection_row.get("bootstrap_replicates", 499)),
        min_prefix=int(selection_row.get("min_prefix", 2)),
        selection_rule=str(selection_row.get("selection_rule", "largest_not_rejected")),  # type: ignore[arg-type]
        spearman_artifact=_path(selection_row["spearman_artifact"], base=source.parent),
        spearman_calibration_config=_path(
            selection_row["spearman_calibration_config"], base=source.parent
        ),
    )
    science = _mapping(root.get("science", {}), context="science")
    raw_settings = _sequence(
        science.get("settings", list(SETTING_ORDER)),
        context="science.settings",
    )
    settings = tuple(str(value) for value in raw_settings)
    return AssumptionExperiment(
        source_path=source,
        assumption_id=assumption_id,
        base=base,
        storage=storage,
        runtime=runtime,
        training=training,
        selection=selection,
        settings=settings,  # type: ignore[arg-type]
        split=str(science.get("split", "test")),
        patch_size=int(science.get("patch_size", 16)),
        k=int(science.get("k", 20)),
        assignment_seed=int(science.get("assignment_seed", base.runtime.seed)),
        matched_source_selection=str(science.get("matched_source_selection", "all")),  # type: ignore[arg-type]
        raw_config=dict(root),
        partition_families=int(science.get("partition_families", 3)),
    )


__all__ = [
    "AssumptionExperiment",
    "AssumptionRuntime",
    "Cell",
    "DistanceModel",
    "EvaluationTask",
    "NoiseSelectionConfig",
    "PartitionTask",
    "RULES",
    "RankTask",
    "SelectionTask",
    "Setting",
    "SourcePhase1Task",
    "SourceTrainingConfig",
    "SETTING_ORDER",
    "TrainingTask",
    "load_assumption_experiment",
]
