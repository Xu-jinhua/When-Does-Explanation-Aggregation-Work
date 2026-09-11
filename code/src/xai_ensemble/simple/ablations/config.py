"""Configuration and scientific task identities for the NAIVE ablations."""

from __future__ import annotations

import math
import os
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from xai_ensemble.core.hashing import object_sha256

from ..config import (
    ConditionConfig,
    EnsembleConfig,
    Phase1Task,
    Phase2Task,
    RuntimeConfig,
    SimpleExperiment,
    StorageConfig,
    load_experiment,
)
from ..methods import PAPER_CNN_METHODS

NoiseType = Literal["gaussian", "salt_pepper", "speckle", "adversarial"]
FillKind = Literal["dataset_mean", "class_mean"]
RankSourceKind = Literal["existing_phase2", "constructed"]
StoreKind = Literal["base", "ablation"]

NOISE_ORDER: tuple[NoiseType, ...] = (
    "gaussian",
    "salt_pepper",
    "speckle",
    "adversarial",
)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence")
    return value


def _path(value: Any, *, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def _gib(value: Any, *, context: str) -> int:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be a positive finite GiB value")
    return int(result * 2**30)


def _severity_id(noise_type: NoiseType, value: float) -> str:
    if noise_type == "adversarial":
        numerator = round(value * 255.0)
        if not math.isclose(value, numerator / 255.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Adversarial levels must be exact integer multiples of 1/255")
        return f"adversarial-sara-{numerator}-255"
    prefix = "salt-pepper" if noise_type == "salt_pepper" else noise_type
    return f"{prefix}-{value:.2f}"


@dataclass(frozen=True, slots=True)
class NoiseLevel:
    noise_type: NoiseType
    severity: float
    condition_id: str
    center: bool


@dataclass(frozen=True, slots=True)
class AblationRuntime:
    database_path: Path
    log_directory: Path
    gpu_ids: tuple[int, ...]
    inference_batch_size: int
    evaluation_reservation_bytes: int
    rank_reservation_bytes: int
    phase1_batch_caps: Mapping[str, int]
    phase1_reservation_floors: Mapping[str, int]
    headroom_fraction: float
    max_retries: int

    def __post_init__(self) -> None:
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if self.inference_batch_size <= 0:
            raise ValueError("runtime.inference_batch_size must be positive")
        if not 0.0 <= self.headroom_fraction < 0.5:
            raise ValueError("runtime.headroom_fraction must lie in [0,0.5)")
        if self.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")
        for name, cap in self.phase1_batch_caps.items():
            if name not in PAPER_CNN_METHODS or cap <= 0:
                raise ValueError(f"Invalid Phase 1 batch cap {name}={cap}")
        unknown = set(self.phase1_reservation_floors) - set(PAPER_CNN_METHODS)
        if unknown:
            raise ValueError(f"Unknown Phase 1 reservation floors: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class ConstructionIdentity:
    setting: Literal["naive", "ind", "noise"]
    dataset_id: str
    model_id: str
    split: str
    condition_id: str
    patch_size: int
    methods: tuple[str, ...]
    rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankConstructionTask:
    task_id: str
    digest: str
    condition: ConditionConfig
    construction: ConstructionIdentity
    phase1_task_ids: tuple[str, ...]

    @property
    def artifact_root(self) -> str:
        return posixpath.join(
            "ranks",
            self.construction.setting,
            self.construction.dataset_id,
            self.construction.model_id,
            self.construction.split,
            self.construction.condition_id,
            f"p{self.construction.patch_size}",
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class RankSourceSpec:
    source_id: str
    kind: RankSourceKind
    store: StoreKind
    root: str
    digest: str
    condition: ConditionConfig
    patch_size: int


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    task_id: str
    digest: str
    table_id: Literal["table2-k", "table4-fill", "table5-noise"]
    parameter: str
    parameter_value: str
    condition: ConditionConfig
    rank_source: RankSourceSpec
    patch_size: int
    k: int
    fill: FillKind

    @property
    def artifact_root(self) -> str:
        return posixpath.join(
            "evaluations",
            "naive",
            self.table_id,
            self.parameter_value,
            self.condition.condition_id,
            f"p{self.patch_size}",
            f"k{self.k}",
            self.fill,
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class AblationExperiment:
    source_path: Path
    ablation_id: str
    base: SimpleExperiment
    output_storage: StorageConfig
    runtime: AblationRuntime
    dataset_id: str
    model_id: str
    split: str
    ensemble_id: str
    patch_size: int
    top_k_values: tuple[int, ...]
    fill_values: tuple[FillKind, ...]
    noise_levels: Mapping[NoiseType, tuple[NoiseLevel, ...]]
    additional_conditions: tuple[ConditionConfig, ...]
    raw_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.patch_size != 16:
            raise ValueError("The accepted ablation protocol is fixed to p=16")
        if self.top_k_values != (10, 20, 40, 80):
            raise ValueError("Table 2 must use k=(10,20,40,80)")
        if self.fill_values != ("dataset_mean", "class_mean"):
            raise ValueError("Table 4 must compare dataset_mean and class_mean")
        model = self.base.model(self.model_id)
        if model.dataset_id != self.dataset_id or model.architecture != "cnn":
            raise ValueError("The current ablation scope must be one CNN dataset/model cell")
        dataset = self.base.dataset(self.dataset_id)
        if self.split not in dataset.splits:
            raise ValueError(f"Ablation split {self.split!r} is not configured")
        if self.output_storage.remote_root == self.base.storage.remote_root:
            raise ValueError("Ablation output storage must not equal the immutable main root")
        if set(self.noise_levels) != set(NOISE_ORDER):
            raise ValueError("Table 5 must define all four perturbation types")

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-naive-ablations-v1",
                "base_experiment_digest": self.base.digest,
                "scope": {
                    "dataset": self.dataset_id,
                    "model": self.model_id,
                    "split": self.split,
                    "ensemble": self.ensemble_id,
                    "patch_size": self.patch_size,
                },
                "table2_top_k": list(self.top_k_values),
                "table4_fills": list(self.fill_values),
                "table5_levels": {
                    kind: [asdict(level) for level in levels]
                    for kind, levels in self.noise_levels.items()
                },
                "additional_conditions": [
                    asdict(condition) for condition in self.additional_conditions
                ],
            }
        )

    @property
    def scheduler_digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-ablation-scheduler-v1",
                "ablation_digest": self.digest,
                "database": str(self.runtime.database_path),
                "output_root": self.output_storage.remote_root,
            }
        )

    @property
    def model(self) -> Any:
        return self.base.model(self.model_id)

    @property
    def dataset(self) -> Any:
        return self.base.dataset(self.dataset_id)

    @property
    def ensemble(self) -> EnsembleConfig:
        matches = tuple(
            item for item in self.base.phase2.ensembles if item.ensemble_id == self.ensemble_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one base ensemble named {self.ensemble_id!r}; found {len(matches)}"
            )
        return matches[0]

    def center_levels(self) -> tuple[NoiseLevel, ...]:
        return tuple(
            next(level for level in self.noise_levels[kind] if level.center) for kind in NOISE_ORDER
        )

    def base_conditions(self) -> tuple[ConditionConfig, ...]:
        clean = next(condition for condition in self.base.conditions if condition.kind == "clean")
        centers = tuple(self.base.condition(level.condition_id) for level in self.center_levels())
        return (clean, *centers)

    def generation_experiment(self) -> SimpleExperiment:
        conditions = (*self.base_conditions(), *self.additional_conditions)
        base_runtime = self.base.runtime
        runtime = RuntimeConfig(
            profile_directory=base_runtime.profile_directory,
            database_path=self.runtime.database_path,
            log_directory=self.runtime.log_directory,
            gpu_ids=self.runtime.gpu_ids,
            search_grid=base_runtime.search_grid,
            default_profile_start=base_runtime.default_profile_start,
            shard_size=base_runtime.shard_size,
            prediction_batch_size=base_runtime.prediction_batch_size,
            dataloader_workers=base_runtime.dataloader_workers,
            headroom_fraction=self.runtime.headroom_fraction,
            max_retries=self.runtime.max_retries,
            seed=base_runtime.seed,
            phase1_upload_workers=base_runtime.phase1_upload_workers,
            phase1_upload_global_limit=base_runtime.phase1_upload_global_limit,
            phase1_stage_workers=base_runtime.phase1_stage_workers,
            phase1_prefetch_workers=base_runtime.phase1_prefetch_workers,
            phase1_prefetch_max_gib=base_runtime.phase1_prefetch_max_gib,
            phase1_prefetch_min_free_gib=base_runtime.phase1_prefetch_min_free_gib,
            phase1_telemetry_interval_seconds=base_runtime.phase1_telemetry_interval_seconds,
        )
        phase2 = replace(
            self.base.phase2,
            patch_sizes=(self.patch_size,),
            primary_patch_size=self.patch_size,
            k=20,
            inference_batch_size=self.runtime.inference_batch_size,
        )
        generated_config = {
            "schema": "simple-ablation-generation-v1",
            "ablation_digest": self.digest,
            "base_phase1_digest": self.base.phase1_digest,
            "phase1_identity_digest": self.base.phase1_digest,
            "conditions": [asdict(condition) for condition in conditions],
        }
        return SimpleExperiment(
            source_path=self.source_path,
            experiment_id=self.ablation_id,
            precision=self.base.precision,
            methods=self.base.methods,
            storage=self.output_storage,
            runtime=runtime,
            datasets=(self.dataset,),
            models=(self.model,),
            conditions=conditions,
            phase2=phase2,
            raw_config=generated_config,
        )

    def new_condition_ids(self) -> frozenset[str]:
        return frozenset(condition.condition_id for condition in self.additional_conditions)

    def phase1_tasks(self) -> tuple[Phase1Task, ...]:
        selected = self.new_condition_ids()
        return tuple(
            task
            for task in self.generation_experiment().phase1_tasks()
            if task.condition.condition_id in selected
        )

    def adversarial_tasks(self) -> tuple[Any, ...]:
        selected = self.new_condition_ids()
        return tuple(
            task
            for task in self.generation_experiment().adversarial_tasks()
            if task.condition.condition_id in selected
        )

    def base_phase2_task(self, condition_id: str) -> Phase2Task:
        matches = [
            task
            for task in self.base.phase2_tasks()
            if task.dataset.dataset_id == self.dataset_id
            and task.model.model_id == self.model_id
            and task.split == self.split
            and task.condition.condition_id == condition_id
            and task.ensemble.ensemble_id == self.ensemble_id
            and task.patch_size == self.patch_size
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one immutable main rank source for {condition_id}; found {len(matches)}"
            )
        return matches[0]

    def rank_tasks(self) -> tuple[RankConstructionTask, ...]:
        phase1 = self.phase1_tasks()
        ensemble = self.ensemble
        result = []
        for condition in self.additional_conditions:
            sources = tuple(
                task for task in phase1 if task.condition.condition_id == condition.condition_id
            )
            if tuple(task.family for task in sources) != PAPER_CNN_METHODS:
                raise RuntimeError(f"Incomplete method roster for {condition.condition_id}")
            construction = ConstructionIdentity(
                setting="naive",
                dataset_id=self.dataset_id,
                model_id=self.model_id,
                split=self.split,
                condition_id=condition.condition_id,
                patch_size=self.patch_size,
                methods=PAPER_CNN_METHODS,
                rules=ensemble.rules,
            )
            identity = {
                "schema": "simple-rank-construction-v1",
                "construction": asdict(construction),
                "phase1_sources": [
                    {
                        "task_id": task.task_id,
                        "digest": task.digest,
                        "family": task.family,
                        "variants": [variant.digest for variant in task.variants],
                    }
                    for task in sources
                ],
                "simpleavg_normalization": self.base.phase2.simpleavg_normalization,
                "rrf_c": self.base.phase2.rrf_c,
                "kemeny_starts": self.base.phase2.kemeny_starts,
                "kemeny_max_passes": self.base.phase2.kemeny_max_passes,
                "rank_semantics": "mean_over_patch_and_channels(abs(full_attribution))",
            }
            digest = object_sha256(identity)
            result.append(
                RankConstructionTask(
                    task_id=f"naive--{condition.condition_id}--p{self.patch_size}--{digest[:10]}",
                    digest=digest,
                    condition=condition,
                    construction=construction,
                    phase1_task_ids=tuple(task.task_id for task in sources),
                )
            )
        return tuple(result)

    def rank_source(self, condition: ConditionConfig) -> RankSourceSpec:
        rank_task = next(
            (
                task
                for task in self.rank_tasks()
                if task.condition.condition_id == condition.condition_id
            ),
            None,
        )
        if rank_task is not None:
            return RankSourceSpec(
                source_id=rank_task.task_id,
                kind="constructed",
                store="ablation",
                root=rank_task.artifact_root,
                digest=rank_task.digest,
                condition=condition,
                patch_size=self.patch_size,
            )
        from ..artifacts import phase2_artifact_root

        task = self.base_phase2_task(condition.condition_id)
        return RankSourceSpec(
            source_id=task.task_id,
            kind="existing_phase2",
            store="base",
            root=phase2_artifact_root(task),
            digest=task.digest,
            condition=condition,
            patch_size=self.patch_size,
        )

    def _evaluation_spec(
        self,
        *,
        table_id: Literal["table2-k", "table4-fill", "table5-noise"],
        parameter: str,
        parameter_value: str,
        condition: ConditionConfig,
        k: int,
        fill: FillKind,
    ) -> EvaluationSpec:
        source = self.rank_source(condition)
        identity = {
            "schema": "simple-rank-evaluation-v1",
            "construction_setting": "naive",
            "table": table_id,
            "parameter": parameter,
            "parameter_value": parameter_value,
            "rank_source": {
                "kind": source.kind,
                "root": source.root,
                "digest": source.digest,
            },
            "evaluator_model": self.model_id,
            "condition": asdict(condition),
            "patch_size": self.patch_size,
            "k": k,
            "fill": fill,
            "fill_class_policy": "fixed_clean_explanation_target" if fill == "class_mean" else None,
            "target_policy": "clean_model_fp32_prediction",
        }
        digest = object_sha256(identity)
        task_id = "--".join(
            (
                table_id,
                parameter_value,
                condition.condition_id,
                f"p{self.patch_size}",
                f"k{k}",
                fill,
                digest[:10],
            )
        )
        return EvaluationSpec(
            task_id=task_id,
            digest=digest,
            table_id=table_id,
            parameter=parameter,
            parameter_value=parameter_value,
            condition=condition,
            rank_source=source,
            patch_size=self.patch_size,
            k=k,
            fill=fill,
        )

    def evaluation_tasks(self) -> tuple[EvaluationSpec, ...]:
        base_conditions = self.base_conditions()
        tasks = []
        for k in self.top_k_values:
            if k == 20:
                continue
            for condition in base_conditions:
                tasks.append(
                    self._evaluation_spec(
                        table_id="table2-k",
                        parameter="k",
                        parameter_value=str(k),
                        condition=condition,
                        k=k,
                        fill="dataset_mean",
                    )
                )
        for fill in self.fill_values:
            if fill == "dataset_mean":
                continue
            for condition in base_conditions:
                tasks.append(
                    self._evaluation_spec(
                        table_id="table4-fill",
                        parameter="fill",
                        parameter_value=fill,
                        condition=condition,
                        k=20,
                        fill=fill,
                    )
                )
        for noise_type in NOISE_ORDER:
            for level in self.noise_levels[noise_type]:
                if level.center:
                    continue
                condition = next(
                    item
                    for item in self.additional_conditions
                    if item.condition_id == level.condition_id
                )
                tasks.append(
                    self._evaluation_spec(
                        table_id="table5-noise",
                        parameter=noise_type,
                        parameter_value=f"{noise_type}-{level.severity:.12g}",
                        condition=condition,
                        k=20,
                        fill="dataset_mean",
                    )
                )
        return tuple(tasks)

    def find_rank_task(self, task_id: str) -> RankConstructionTask:
        for task in self.rank_tasks():
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown rank task {task_id!r}")

    def find_evaluation_task(self, task_id: str) -> EvaluationSpec:
        for task in self.evaluation_tasks():
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown evaluation task {task_id!r}")


def _natural_center(
    base: SimpleExperiment, noise_type: NoiseType, severity: float
) -> ConditionConfig:
    matches = [
        condition
        for condition in base.conditions
        if condition.kind == "factory"
        and condition.kwargs.get("kind") == noise_type
        and math.isclose(float(condition.kwargs.get("severity", -1.0)), severity, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve one main {noise_type} center at {severity}")
    return matches[0]


def _adversarial_center(base: SimpleExperiment, severity: float) -> ConditionConfig:
    matches = [
        condition
        for condition in base.conditions
        if condition.kind == "adversarial"
        and math.isclose(float(condition.kwargs.get("epsilon", -1.0)), severity, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve one main adversarial center at {severity}")
    return matches[0]


def _load_noise_levels(
    row: Mapping[str, Any],
    *,
    base: SimpleExperiment,
) -> tuple[Mapping[NoiseType, tuple[NoiseLevel, ...]], tuple[ConditionConfig, ...]]:
    values: dict[NoiseType, tuple[NoiseLevel, ...]] = {}
    conditions = []
    for noise_type in NOISE_ORDER:
        key = "adversarial_numerators" if noise_type == "adversarial" else noise_type
        raw_levels = tuple(float(item) for item in _sequence(row[key], context=f"table5.{key}"))
        if noise_type == "adversarial":
            raw_levels = tuple(value / 255.0 for value in raw_levels)
        if len(raw_levels) != 3 or not raw_levels[0] < raw_levels[1] < raw_levels[2]:
            raise ValueError(f"Table 5 {noise_type} must contain three increasing levels")
        center_value = raw_levels[1]
        center_condition = (
            _adversarial_center(base, center_value)
            if noise_type == "adversarial"
            else _natural_center(base, noise_type, center_value)
        )
        levels = []
        for index, severity in enumerate(raw_levels):
            center = index == 1
            condition_id = (
                center_condition.condition_id if center else _severity_id(noise_type, severity)
            )
            levels.append(
                NoiseLevel(
                    noise_type=noise_type,
                    severity=severity,
                    condition_id=condition_id,
                    center=center,
                )
            )
            if center:
                continue
            kwargs = dict(center_condition.kwargs)
            if noise_type == "adversarial":
                kwargs["epsilon"] = severity
            else:
                kwargs["severity"] = severity
                kwargs["seed_group"] = center_condition.condition_id
            conditions.append(
                ConditionConfig(
                    condition_id=condition_id,
                    kind=center_condition.kind,
                    factory=center_condition.factory,
                    kwargs=kwargs,
                )
            )
        values[noise_type] = tuple(levels)
    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated ablation condition ids are not unique")
    return values, tuple(conditions)


def load_ablation_experiment(path: str | Path) -> AblationExperiment:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, context="ablation config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("Ablation config schema_version must be 1")
    base = load_experiment(_path(root["base_config"], base=source.parent))
    storage_row = _mapping(root["storage"], context="storage")
    ablation_id = str(root["ablation_id"])
    output_storage = StorageConfig(
        remote_root=os.path.expandvars(str(storage_row["remote_root"])).rstrip("/"),
        scratch_root=_path(storage_row["scratch_root"], base=source.parent),
        rclone_binary=_path(
            storage_row.get("rclone_binary", base.storage.rclone_binary), base=source.parent
        ),
        spool_root=_path(
            storage_row.get("spool_root", f"/dev/shm/xai-simple/{ablation_id}"),
            base=source.parent,
        ),
        spool_max_bytes=int(float(storage_row.get("spool_max_gib", 64)) * 2**30),
        spool_min_free_bytes=int(float(storage_row.get("spool_min_free_gib", 32)) * 2**30),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    caps = {
        str(key): int(value)
        for key, value in _mapping(
            runtime_row.get("phase1_batch_caps", {}), context="phase1_batch_caps"
        ).items()
    }
    floors = {
        str(key): _gib(value, context=f"phase1_reservation_floor_gib.{key}")
        for key, value in _mapping(
            runtime_row.get("phase1_reservation_floor_gib", {}),
            context="phase1_reservation_floor_gib",
        ).items()
    }
    runtime = AblationRuntime(
        database_path=_path(runtime_row["database_path"], base=source.parent),
        log_directory=_path(runtime_row["log_directory"], base=source.parent),
        gpu_ids=tuple(
            int(item) for item in _sequence(runtime_row.get("gpu_ids", [0, 1]), context="gpu_ids")
        ),
        inference_batch_size=int(runtime_row.get("inference_batch_size", 384)),
        evaluation_reservation_bytes=_gib(
            runtime_row.get("evaluation_reservation_gib", 9),
            context="evaluation_reservation_gib",
        ),
        rank_reservation_bytes=_gib(
            runtime_row.get("rank_reservation_gib", 9), context="rank_reservation_gib"
        ),
        phase1_batch_caps=caps,
        phase1_reservation_floors=floors,
        headroom_fraction=float(runtime_row.get("headroom_fraction", 0.10)),
        max_retries=int(runtime_row.get("max_retries", 1)),
    )
    scope = _mapping(root["scope"], context="scope")
    table2 = _mapping(root["table2"], context="table2")
    table4 = _mapping(root["table4"], context="table4")
    table5 = _mapping(root["table5"], context="table5")
    noise_levels, additional = _load_noise_levels(table5, base=base)
    fills = tuple(str(item) for item in _sequence(table4["fills"], context="table4.fills"))
    return AblationExperiment(
        source_path=source,
        ablation_id=ablation_id,
        base=base,
        output_storage=output_storage,
        runtime=runtime,
        dataset_id=str(scope["dataset"]),
        model_id=str(scope["model"]),
        split=str(scope.get("split", "test")),
        ensemble_id=str(scope.get("ensemble", "all-paper-methods")),
        patch_size=int(scope.get("patch_size", 16)),
        top_k_values=tuple(
            int(item) for item in _sequence(table2["top_k"], context="table2.top_k")
        ),
        fill_values=fills,  # type: ignore[arg-type]
        noise_levels=noise_levels,
        additional_conditions=additional,
        raw_config=dict(root),
    )


__all__ = [
    "AblationExperiment",
    "AblationRuntime",
    "ConstructionIdentity",
    "EvaluationSpec",
    "FillKind",
    "NOISE_ORDER",
    "NoiseLevel",
    "RankConstructionTask",
    "RankSourceSpec",
    "load_ablation_experiment",
]
