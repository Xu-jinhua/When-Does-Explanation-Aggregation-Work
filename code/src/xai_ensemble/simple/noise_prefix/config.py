"""Configuration and immutable task identities for the NOISE prefix sweep."""

from __future__ import annotations

import math
import os
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xai_ensemble.core.hashing import object_sha256

from ..assumptions.config import AssumptionExperiment, Cell, load_assumption_experiment
from ..config import ConditionConfig, StorageConfig

RULES = ("SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze")
RULE_IDS = tuple(value.lower() for value in RULES)
Q_VALUES = tuple(range(2, 12))
DIAGNOSTIC_SCOPE = "complete_test_set_in_sample_oracle_diagnostic"
SELECTION_INPUT_MODES = frozenset({"legacy_assumptions", "independent_geometry_deferred"})


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
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{context} must be a positive finite GiB value")
    return int(number * 2**30)


@dataclass(frozen=True, slots=True)
class NoisePrefixRuntime:
    database_path: Path
    log_directory: Path
    input_catalog_path: Path
    shared_cache_root: Path
    gpu_ids: tuple[int, ...]
    inference_batch_size: int
    evaluation_reservation_bytes: int
    aggregation_workspace_bytes: int
    headroom_fraction: float
    max_retries: int
    cpu_workers: int

    def __post_init__(self) -> None:
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if self.inference_batch_size <= 0 or self.cpu_workers <= 0:
            raise ValueError("runtime batch size and cpu_workers must be positive")
        if not 0.0 <= self.headroom_fraction < 0.5:
            raise ValueError("runtime.headroom_fraction must lie in [0,0.5)")
        if self.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")


@dataclass(frozen=True, slots=True)
class PrefixEvaluationTask:
    task_id: str
    digest: str
    cell: Cell
    condition: ConditionConfig

    @property
    def artifact_root(self) -> str:
        return posixpath.join(
            "evaluations",
            self.cell.cell_id,
            self.condition.condition_id,
            "p16",
            "k20",
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class NoisePrefixExperiment:
    source_path: Path
    sweep_id: str
    assumptions: AssumptionExperiment
    storage: StorageConfig
    runtime: NoisePrefixRuntime
    split: str
    patch_size: int
    k: int
    q_values: tuple[int, ...]
    rules: tuple[str, ...]
    selection_input_mode: str
    raw_science: Mapping[str, Any]
    _task_cache: dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.storage.remote_root in {
            self.assumptions.storage.remote_root,
            self.assumptions.base.storage.remote_root,
        }:
            raise ValueError("prefix sweep artifacts require an independent remote root")
        if self.runtime.database_path in {
            self.assumptions.runtime.database_path,
            self.assumptions.base.runtime.database_path,
        }:
            raise ValueError("prefix sweep jobs require an independent SQLite database")
        if self.split != "test" or self.patch_size != 16 or self.k != 20:
            raise ValueError("the formal prefix sweep is fixed to test, p=16, and k=20")
        if self.q_values != Q_VALUES:
            raise ValueError(f"the formal prefix sweep requires q={Q_VALUES}")
        if self.rules != RULES:
            raise ValueError(f"the formal prefix sweep requires rules={RULES}")
        if self.selection_input_mode not in SELECTION_INPUT_MODES:
            raise ValueError(
                "selection_input_mode must be legacy_assumptions or "
                "independent_geometry_deferred"
            )
        for cell in self.cells():
            if len(cell.methods) != max(self.q_values):
                raise ValueError(
                    f"{cell.cell_id} has {len(cell.methods)} methods; q=11 requires exactly 11"
                )

    @property
    def base(self) -> Any:
        return self.assumptions.base

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-noise-prefix-sweep-v1",
                "base_experiment_digest": self.base.digest,
                "base_phase1_digest": self.base.phase1_digest,
                "method_catalog_digest": self.base.methods.source_digest,
                "assumptions_digest": self.assumptions.digest,
                "science": dict(self.raw_science),
                "fidelity_order": {
                    "metric": "F",
                    "scope": "complete_test_set",
                    "direction": "descending",
                    "tie_break": "method_id_ascending",
                },
                "q11": "exact_immutable_naive_p16_reference",
                "diagnostic_scope": DIAGNOSTIC_SCOPE,
            }
        )

    @property
    def scheduler_digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-noise-prefix-scheduler-v1",
                "sweep_id": self.sweep_id,
                "experiment_digest": self.digest,
                "database": str(self.runtime.database_path),
                "remote_root": self.storage.remote_root,
            }
        )

    def cells(self) -> tuple[Cell, ...]:
        return self.assumptions.cells()

    def base_phase2_task(self, cell: Cell, condition_id: str) -> Any:
        return self.assumptions.base_phase2_task(cell, condition_id)

    def evaluation_tasks(self) -> tuple[PrefixEvaluationTask, ...]:
        if cached := self._task_cache.get("evaluations"):
            return cached
        tasks = []
        for cell in self.cells():
            clean_source = self.base_phase2_task(cell, "clean")
            for condition in self.base.conditions:
                source = self.base_phase2_task(cell, condition.condition_id)
                identity = {
                    "schema": "simple-noise-prefix-evaluation-v1",
                    "sweep_digest": self.digest,
                    "cell": cell.cell_id,
                    "condition": asdict(condition),
                    "source_phase2_digest": source.digest,
                    "fidelity_phase2_digest": clean_source.digest,
                    "methods": list(cell.methods),
                    "q_values": list(self.q_values),
                    "computed_q_values": list(self.q_values[:-1]),
                    "q11_source": "exact_immutable_naive_p16_reference",
                    "rules": list(self.rules),
                    "patch_size": self.patch_size,
                    "k": self.k,
                    "fill": "dataset_mean",
                    "target_policy": "full_reference_clean_fp32_prediction",
                }
                digest = object_sha256(identity)
                tasks.append(
                    PrefixEvaluationTask(
                        task_id=(
                            f"prefix--{cell.cell_id}--{condition.condition_id}--{digest[:10]}"
                        ),
                        digest=digest,
                        cell=cell,
                        condition=condition,
                    )
                )
        value = tuple(tasks)
        self._task_cache["evaluations"] = value
        return value

    def find_evaluation_task(self, task_id: str) -> PrefixEvaluationTask:
        for task in self.evaluation_tasks():
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown NOISE prefix task {task_id!r}")

    def clean_task(self, task: PrefixEvaluationTask) -> PrefixEvaluationTask:
        matches = tuple(
            candidate
            for candidate in self.evaluation_tasks()
            if candidate.cell.cell_id == task.cell.cell_id and candidate.condition.kind == "clean"
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one clean prefix task; found {len(matches)}")
        return matches[0]


def load_noise_prefix_experiment(path: str | Path) -> NoisePrefixExperiment:
    source = Path(path).expanduser().resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")) or {}, context="config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("NOISE prefix config schema_version must be 1")
    assumptions = load_assumption_experiment(_path(root["assumptions_config"], base=source.parent))
    sweep_id = str(root["sweep_id"])
    storage_row = _mapping(root["storage"], context="storage")
    storage = StorageConfig(
        remote_root=os.path.expandvars(str(storage_row["remote_root"])).rstrip("/"),
        scratch_root=_path(storage_row["scratch_root"], base=source.parent),
        rclone_binary=_path(
            storage_row.get("rclone_binary", assumptions.storage.rclone_binary),
            base=source.parent,
        ),
        spool_root=_path(
            storage_row.get("spool_root", f"/dev/shm/xai-simple/{sweep_id}"),
            base=source.parent,
        ),
        spool_max_bytes=_gib(storage_row.get("spool_max_gib", 96), context="spool max"),
        spool_min_free_bytes=int(float(storage_row.get("spool_min_free_gib", 32)) * 2**30),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    runtime = NoisePrefixRuntime(
        database_path=_path(runtime_row["database_path"], base=source.parent),
        log_directory=_path(runtime_row["log_directory"], base=source.parent),
        input_catalog_path=_path(runtime_row["input_catalog_path"], base=source.parent),
        shared_cache_root=_path(
            runtime_row.get("shared_cache_root", assumptions.storage.spool_root / "shared-cache"),
            base=source.parent,
        ),
        gpu_ids=tuple(
            int(value) for value in _sequence(runtime_row.get("gpu_ids", [0, 1]), context="gpu_ids")
        ),
        inference_batch_size=int(runtime_row.get("inference_batch_size", 512)),
        evaluation_reservation_bytes=_gib(
            runtime_row.get("evaluation_reservation_gib", 12),
            context="evaluation reservation",
        ),
        aggregation_workspace_bytes=_gib(
            runtime_row.get("aggregation_workspace_gib", 2),
            context="aggregation workspace",
        ),
        headroom_fraction=float(runtime_row.get("headroom_fraction", 0.05)),
        max_retries=int(runtime_row.get("max_retries", 1)),
        cpu_workers=int(runtime_row.get("cpu_workers", 3)),
    )
    science = _mapping(root.get("science", {}), context="science")
    q_values = tuple(
        int(value) for value in _sequence(science.get("q_values", Q_VALUES), context="q_values")
    )
    rules = tuple(str(value) for value in _sequence(science.get("rules", RULES), context="rules"))
    selection_input_mode = str(
        science.get("selection_input_mode", "legacy_assumptions")
    )
    raw_science = {
        "split": str(science.get("split", "test")),
        "patch_size": int(science.get("patch_size", 16)),
        "k": int(science.get("k", 20)),
        "q_values": list(q_values),
        "rules": list(rules),
        "fill": "dataset_mean",
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
    }
    if "selection_input_mode" in science:
        raw_science["selection_input_mode"] = selection_input_mode
    return NoisePrefixExperiment(
        source_path=source,
        sweep_id=sweep_id,
        assumptions=assumptions,
        storage=storage,
        runtime=runtime,
        split=str(raw_science["split"]),
        patch_size=int(raw_science["patch_size"]),
        k=int(raw_science["k"]),
        q_values=q_values,
        rules=rules,
        selection_input_mode=selection_input_mode,
        raw_science=raw_science,
    )


__all__ = [
    "DIAGNOSTIC_SCOPE",
    "Q_VALUES",
    "RULES",
    "RULE_IDS",
    "SELECTION_INPUT_MODES",
    "NoisePrefixExperiment",
    "NoisePrefixRuntime",
    "PrefixEvaluationTask",
    "load_noise_prefix_experiment",
]
