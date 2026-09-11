"""Configuration and deterministic task expansion for relative robustness."""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xai_ensemble.core.hashing import object_sha256, stable_seed

from ..config import SimpleExperiment, StorageConfig
from ..config import load_experiment as load_simple_experiment
from ..noise_prefix.config import (
    NoisePrefixExperiment,
    PrefixEvaluationTask,
    load_noise_prefix_experiment,
)

CONTROL_SCHEMA = "simple-relative-robustness-random-control-v1"


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
class RelativeRobustnessRuntime:
    database_path: Path
    log_directory: Path
    shared_cache_root: Path
    gpu_ids: tuple[int, ...]
    inference_batch_size: int
    control_reservation_bytes: int
    headroom_fraction: float
    max_retries: int
    cpu_workers: int

    def __post_init__(self) -> None:
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if self.inference_batch_size <= 0 or self.control_reservation_bytes <= 0:
            raise ValueError("runtime inference batch and control reservation must be positive")
        if self.cpu_workers <= 0:
            raise ValueError("runtime.cpu_workers must be positive")
        if not 0.0 <= self.headroom_fraction < 0.5:
            raise ValueError("runtime.headroom_fraction must lie in [0,0.5)")
        if self.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")


@dataclass(frozen=True, slots=True)
class RandomControlTask:
    task_id: str
    digest: str
    prefix_task: PrefixEvaluationTask
    patch_size: int
    k: int

    @property
    def cell_id(self) -> str:
        return self.prefix_task.cell.cell_id

    @property
    def artifact_root(self) -> str:
        return posixpath.join(
            "random-controls",
            self.cell_id,
            self.prefix_task.condition.condition_id,
            f"p{self.patch_size}",
            f"k{self.k}",
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class RelativeRobustnessExperiment:
    source_path: Path
    study_id: str
    base: SimpleExperiment
    prefix: NoisePrefixExperiment
    storage: StorageConfig
    runtime: RelativeRobustnessRuntime
    split: str
    patch_size: int
    k: int
    random_seed: int
    random_seed_count: int
    bootstrap_replicates: int
    bootstrap_batch_size: int
    confidence: float
    independent_noise_summary: Path
    raw_config: Mapping[str, Any]
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.base.digest != self.prefix.base.digest:
            raise ValueError(
                "relative robustness base config differs from prefix-sweep base config"
            )
        if self.storage.remote_root in {
            self.base.storage.remote_root,
            self.prefix.storage.remote_root,
        }:
            raise ValueError("relative robustness requires a separate immutable remote root")
        if self.runtime.database_path in {
            self.base.runtime.database_path,
            self.prefix.runtime.database_path,
        }:
            raise ValueError("relative robustness requires a separate scheduler database")
        if self.split != self.prefix.split or self.split != "test":
            raise ValueError("relative robustness is defined only on the shared test split")
        if self.patch_size != 16 or self.k != 20:
            raise ValueError("relative robustness is fixed to the paper setting p=16, k=20")
        if self.patch_size != self.prefix.patch_size or self.k != self.prefix.k:
            raise ValueError("relative robustness mask settings differ from the prefix sweep")
        if self.runtime.inference_batch_size != self.prefix.runtime.inference_batch_size:
            raise ValueError("relative robustness must use the frozen prefix inference batch size")
        if self.random_seed_count < 2:
            raise ValueError("random_seed_count must be at least two for Monte Carlo stability")
        if self.bootstrap_replicates <= 0 or self.bootstrap_batch_size <= 0:
            raise ValueError("bootstrap counts must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        if any(len(cell.methods) != 11 for cell in self.prefix.cells()):
            raise ValueError("relative robustness expects the complete eleven-method NOISE roster")

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-relative-robustness-v1",
                "config": self.raw_config,
                "base_digest": self.base.digest,
                "prefix_digest": self.prefix.digest,
            }
        )

    @property
    def scheduler_digest(self) -> str:
        return object_sha256(
            {
                "schema": "simple-relative-robustness-scheduler-v1",
                "study_id": self.study_id,
                "study_digest": self.digest,
                "database": str(self.runtime.database_path),
                "remote_root": self.storage.remote_root,
            }
        )

    @property
    def random_seed_bank(self) -> tuple[int, ...]:
        return tuple(
            stable_seed(CONTROL_SCHEMA, self.random_seed, "seed-bank", position)
            for position in range(self.random_seed_count)
        )

    def control_tasks(self) -> tuple[RandomControlTask, ...]:
        if cached := self._cache.get("control_tasks"):
            return cached
        result = []
        for prefix_task in self.prefix.evaluation_tasks():
            identity = {
                "schema": CONTROL_SCHEMA,
                "study_id": self.study_id,
                "study_digest": self.digest,
                "prefix_task_id": prefix_task.task_id,
                "prefix_task_digest": prefix_task.digest,
                "cell": prefix_task.cell.cell_id,
                "condition": prefix_task.condition.condition_id,
                "split": self.split,
                "patch_size": self.patch_size,
                "k": self.k,
                "random_mask_policy": "uniform_patch_permutation_shared_across_conditions_per_image_seed",
                "random_seed_bank": list(self.random_seed_bank),
            }
            digest = object_sha256(identity)
            result.append(
                RandomControlTask(
                    task_id=(
                        f"relative-random-control--{prefix_task.cell.cell_id}--"
                        f"{prefix_task.condition.condition_id}--{digest[:10]}"
                    ),
                    digest=digest,
                    prefix_task=prefix_task,
                    patch_size=self.patch_size,
                    k=self.k,
                )
            )
        value = tuple(result)
        self._cache["control_tasks"] = value
        return value

    def find_control_task(self, task_id: str) -> RandomControlTask:
        matches = tuple(task for task in self.control_tasks() if task.task_id == task_id)
        if len(matches) != 1:
            raise KeyError(f"Unknown relative robustness random-control task {task_id!r}")
        return matches[0]


def load_experiment(path: str | Path) -> RelativeRobustnessExperiment:
    source = Path(path).expanduser().resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")) or {}, context="config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("relative robustness config schema_version must be 1")
    base = load_simple_experiment(_path(root["base_config"], base=source.parent))
    prefix = load_noise_prefix_experiment(_path(root["noise_prefix_config"], base=source.parent))
    storage_row = _mapping(root["storage"], context="storage")
    storage = StorageConfig(
        remote_root=os.path.expandvars(str(storage_row["remote_root"])).rstrip("/"),
        scratch_root=_path(storage_row["scratch_root"], base=source.parent),
        rclone_binary=_path(
            storage_row.get("rclone_binary", base.storage.rclone_binary), base=source.parent
        ),
        spool_root=_path(
            storage_row.get("spool_root", f"/dev/shm/xai-simple/{root['study_id']}"),
            base=source.parent,
        ),
        spool_max_bytes=_gib(storage_row.get("spool_max_gib", 96), context="spool max"),
        spool_min_free_bytes=int(float(storage_row.get("spool_min_free_gib", 32)) * 2**30),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    runtime = RelativeRobustnessRuntime(
        database_path=_path(runtime_row["database_path"], base=source.parent),
        log_directory=_path(runtime_row["log_directory"], base=source.parent),
        shared_cache_root=_path(runtime_row["shared_cache_root"], base=source.parent),
        gpu_ids=tuple(
            int(value) for value in _sequence(runtime_row.get("gpu_ids", [0, 1]), context="gpu_ids")
        ),
        inference_batch_size=int(
            runtime_row.get("inference_batch_size", prefix.runtime.inference_batch_size)
        ),
        control_reservation_bytes=_gib(
            runtime_row.get("control_reservation_gib", 16), context="random control reservation"
        ),
        headroom_fraction=float(runtime_row.get("headroom_fraction", 0.05)),
        max_retries=int(runtime_row.get("max_retries", 1)),
        cpu_workers=int(runtime_row.get("cpu_workers", 3)),
    )
    science = _mapping(root["science"], context="science")
    return RelativeRobustnessExperiment(
        source_path=source,
        study_id=str(root["study_id"]),
        base=base,
        prefix=prefix,
        storage=storage,
        runtime=runtime,
        split=str(science.get("split", "test")),
        patch_size=int(science.get("patch_size", 16)),
        k=int(science.get("k", 20)),
        random_seed=int(science.get("random_seed", base.runtime.seed)),
        random_seed_count=int(science.get("random_seed_count", 8)),
        bootstrap_replicates=int(science.get("bootstrap_replicates", 19999)),
        bootstrap_batch_size=int(science.get("bootstrap_batch_size", 32)),
        confidence=float(science.get("confidence", 0.95)),
        independent_noise_summary=_path(science["independent_noise_summary"], base=source.parent),
        raw_config=dict(root),
    )


__all__ = [
    "CONTROL_SCHEMA",
    "RandomControlTask",
    "RelativeRobustnessExperiment",
    "RelativeRobustnessRuntime",
    "load_experiment",
]
