"""Configuration and immutable identities for random NOISE subsets."""

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

from ..assumptions.config import Cell
from ..config import ConditionConfig, StorageConfig
from ..noise_prefix.config import RULES, NoisePrefixExperiment, load_noise_prefix_experiment
from ..noise_prefix.independent_geometry import load_independent_geometry_selector

GEOMETRIES = ("spearman", "kendall")
GEOMETRY_RULES = {"spearman": "borda", "kendall": "kemeny"}
KS_CONTROL_MODE = "fixed_q_ks"
ANCHORED_CONTROL_MODE = "random_order_anchored"
FIT_POLICY = "exact_subset_mallows_empirical_cdf_ks_not_worse_than_reference"
CANDIDATE_POLICY = "fixed_seed_uniform_nonreference_fixed_q_candidate_pool"
SELECTION_POLICY = "uniform_without_replacement_from_noise_consistent_pool"
ANCHORED_FIT_POLICY = "exact_fixed_size_subset_mallows_mle"
ANCHORED_CANDIDATE_POLICY = "fixed_seed_unique_random_method_permutations"
ANCHORED_SELECTION_POLICY = "per_permutation_per_geometry_maximum_anchored_score_smallest_q_tie"


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
class NoiseSubsetRuntime:
    database_path: Path
    log_directory: Path
    shared_cache_root: Path
    gpu_ids: tuple[int, ...]
    inference_batch_size: int
    selection_reservation_bytes: int
    evaluation_reservation_bytes: int
    aggregation_workspace_bytes: int
    headroom_fraction: float
    max_retries: int
    cpu_workers: int

    def __post_init__(self) -> None:
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime.gpu_ids must be non-empty and unique")
        if self.inference_batch_size <= 0 or self.cpu_workers <= 0:
            raise ValueError("runtime inference batch and cpu_workers must be positive")
        if min(self.selection_reservation_bytes, self.evaluation_reservation_bytes) <= 0:
            raise ValueError("runtime reservations must be positive")
        if self.aggregation_workspace_bytes <= 0:
            raise ValueError("runtime aggregation workspace must be positive")
        if not 0.0 <= self.headroom_fraction < 0.5:
            raise ValueError("runtime.headroom_fraction must lie in [0,0.5)")
        if self.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")


@dataclass(frozen=True, slots=True)
class NoiseSubsetSelectionTask:
    task_id: str
    digest: str
    cell: Cell

    @property
    def artifact_root(self) -> str:
        return posixpath.join("selections", self.cell.cell_id, "p16", "k20", self.digest)


@dataclass(frozen=True, slots=True)
class NoiseSubsetEvaluationTask:
    task_id: str
    digest: str
    cell: Cell
    condition: ConditionConfig
    geometry: str
    selection_task_id: str

    @property
    def artifact_root(self) -> str:
        return posixpath.join(
            "evaluations",
            self.cell.cell_id,
            self.condition.condition_id,
            self.geometry,
            "p16",
            "k20",
            self.digest,
        )


@dataclass(frozen=True, slots=True)
class NoiseSubsetExperiment:
    source_path: Path
    study_id: str
    prefix: NoisePrefixExperiment
    selector_path: Path
    selector_digest: str
    storage: StorageConfig
    runtime: NoiseSubsetRuntime
    split: str
    patch_size: int
    k: int
    cell_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    geometries: tuple[str, ...]
    rules: tuple[str, ...]
    control_mode: str
    q_values: tuple[int, ...]
    random_seed: int
    candidate_draw_count: int
    selected_candidate_count: int
    minimum_accepted_candidates: int
    random_order_count: int
    fit_policy: str
    candidate_policy: str
    selection_policy: str
    raw_science: Mapping[str, Any]
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.storage.remote_root in {
            self.prefix.storage.remote_root,
            self.prefix.base.storage.remote_root,
            self.prefix.assumptions.storage.remote_root,
        }:
            raise ValueError("random-subset artifacts require an independent remote root")
        if self.runtime.database_path in {
            self.prefix.runtime.database_path,
            self.prefix.base.runtime.database_path,
            self.prefix.assumptions.runtime.database_path,
        }:
            raise ValueError("random-subset jobs require an independent SQLite database")
        if self.split != "test" or self.patch_size != 16 or self.k != 20:
            raise ValueError("random NOISE subsets are fixed to test, p=16, and k=20")
        if self.geometries != GEOMETRIES or self.rules != RULES:
            raise ValueError("random NOISE geometry or aggregation-rule coverage changed")
        if self.control_mode == KS_CONTROL_MODE:
            if self.fit_policy != FIT_POLICY:
                raise ValueError(f"random NOISE fit policy must be {FIT_POLICY}")
            if self.candidate_policy != CANDIDATE_POLICY:
                raise ValueError(f"random NOISE candidate policy must be {CANDIDATE_POLICY}")
            if self.selection_policy != SELECTION_POLICY:
                raise ValueError(f"random NOISE selection policy must be {SELECTION_POLICY}")
            if self.candidate_draw_count <= 0 or self.selected_candidate_count <= 0:
                raise ValueError("candidate draw and selected counts must be positive")
            if self.selected_candidate_count > self.candidate_draw_count:
                raise ValueError("selected candidate count exceeds the candidate draw bank")
            if not 1 <= self.minimum_accepted_candidates <= self.selected_candidate_count:
                raise ValueError("minimum_accepted_candidates lies outside the selected bank")
        elif self.control_mode == ANCHORED_CONTROL_MODE:
            expected = tuple(self.prefix.q_values)
            if self.q_values != expected or expected != tuple(range(2, 12)):
                raise ValueError("anchored random-order control requires the complete q=2..11 bank")
            if self.random_order_count <= 0:
                raise ValueError("anchored random-order control requires random_order_count > 0")
            expected_policies = (
                ANCHORED_FIT_POLICY,
                ANCHORED_CANDIDATE_POLICY,
                ANCHORED_SELECTION_POLICY,
            )
            if (self.fit_policy, self.candidate_policy, self.selection_policy) != expected_policies:
                raise ValueError("anchored random-order control policies are incompatible")
        else:
            raise ValueError(f"unknown random NOISE control mode: {self.control_mode}")
        if not self.cells() or not self.conditions():
            raise ValueError("random NOISE cells and conditions must be non-empty")
        selector = self.selector_value()
        if (
            selector.get("sweep_id") != self.prefix.sweep_id
            or selector.get("sweep_digest") != self.prefix.digest
        ):
            raise ValueError("independent-q selector does not belong to the prefix sweep")
        selector_cells = {str(row["cell"]): row for row in selector["cells"]}
        for cell in self.cells():
            if cell.cell_id not in selector_cells:
                raise ValueError(f"selector does not cover {cell.cell_id}")
            row = selector_cells[cell.cell_id]
            if row.get("ordered_methods") is None or len(row["ordered_methods"]) != 11:
                raise ValueError(f"selector method roster is invalid for {cell.cell_id}")
            if self.control_mode == ANCHORED_CONTROL_MODE:
                for geometry in self.geometries:
                    candidates = row["geometries"][geometry].get("candidates")
                    if not isinstance(candidates, Sequence) or [
                        int(candidate["q"]) for candidate in candidates
                    ] != list(self.q_values):
                        raise ValueError(
                            f"formal anchored selector q coverage is invalid for {cell.cell_id}/{geometry}"
                        )

    @property
    def base(self) -> Any:
        return self.prefix.base

    @property
    def digest(self) -> str:
        return object_sha256(
            {
                "schema": (
                    "simple-noise-random-order-anchored-v1"
                    if self.control_mode == ANCHORED_CONTROL_MODE
                    else "simple-noise-random-subset-v1"
                ),
                "study_id": self.study_id,
                "prefix_digest": self.prefix.digest,
                "selector_digest": self.selector_digest,
                "science": dict(self.raw_science),
            }
        )

    @property
    def scheduler_digest(self) -> str:
        return object_sha256(
            {
                "schema": (
                    "simple-noise-random-order-anchored-scheduler-v1"
                    if self.control_mode == ANCHORED_CONTROL_MODE
                    else "simple-noise-random-subset-scheduler-v1"
                ),
                "study_id": self.study_id,
                "study_digest": self.digest,
                "database": str(self.runtime.database_path),
                "remote_root": self.storage.remote_root,
            }
        )

    @property
    def artifact_schema_version(self) -> int:
        return 2 if self.control_mode == ANCHORED_CONTROL_MODE else 1

    def selector_value(self) -> Mapping[str, Any]:
        if cached := self._cache.get("selector"):
            return cached
        value = load_independent_geometry_selector(self.selector_path)
        if value.get("selector_digest") != self.selector_digest:
            raise ValueError("independent-q selector digest changed")
        self._cache["selector"] = value
        return value

    def selector_cell(self, cell_id: str) -> Mapping[str, Any]:
        matches = tuple(
            row for row in self.selector_value()["cells"] if str(row["cell"]) == cell_id
        )
        if len(matches) != 1:
            raise KeyError(f"selector cell is missing or duplicated: {cell_id}")
        return matches[0]

    def cells(self) -> tuple[Cell, ...]:
        by_id = {cell.cell_id: cell for cell in self.prefix.cells()}
        unknown = set(self.cell_ids).difference(by_id)
        if unknown:
            raise ValueError(f"unknown random NOISE cells: {sorted(unknown)}")
        return tuple(by_id[cell_id] for cell_id in self.cell_ids)

    def conditions(self) -> tuple[ConditionConfig, ...]:
        by_id = {condition.condition_id: condition for condition in self.base.conditions}
        unknown = set(self.condition_ids).difference(by_id)
        if unknown:
            raise ValueError(f"unknown random NOISE conditions: {sorted(unknown)}")
        return tuple(by_id[condition_id] for condition_id in self.condition_ids)

    def prefix_task(self, cell: Cell, condition_id: str) -> Any:
        matches = tuple(
            task
            for task in self.prefix.evaluation_tasks()
            if task.cell.cell_id == cell.cell_id and task.condition.condition_id == condition_id
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one prefix task for {cell.cell_id}/{condition_id}; found {len(matches)}"
            )
        return matches[0]

    def selection_tasks(self) -> tuple[NoiseSubsetSelectionTask, ...]:
        if cached := self._cache.get("selection_tasks"):
            return cached
        tasks = []
        for cell in self.cells():
            selector_cell = self.selector_cell(cell.cell_id)
            clean = self.prefix_task(cell, "clean")
            if self.control_mode == ANCHORED_CONTROL_MODE:
                identity = {
                    "schema": "simple-noise-random-order-anchored-selection-v1",
                    "study_digest": self.digest,
                    "cell": cell.cell_id,
                    "clean_prefix_task_digest": clean.digest,
                    "selector_digest": self.selector_digest,
                    "ordered_methods": list(selector_cell["ordered_methods"]),
                    "q_values": list(self.q_values),
                    "geometries": list(self.geometries),
                    "fit_policy": self.fit_policy,
                    "candidate_policy": self.candidate_policy,
                    "selection_policy": self.selection_policy,
                    "random_order_count": self.random_order_count,
                    "random_seed": self.random_seed,
                    "utility_anchor_scope": "all_ordered_individual_methods",
                }
            else:
                identity = {
                    "schema": "simple-noise-random-subset-selection-v1",
                    "study_digest": self.digest,
                    "cell": cell.cell_id,
                    "clean_prefix_task_digest": clean.digest,
                    "selector_digest": self.selector_digest,
                    "ordered_methods": list(selector_cell["ordered_methods"]),
                    "q_S": int(selector_cell["q_S"]),
                    "q_K": int(selector_cell["q_K"]),
                    "geometries": list(self.geometries),
                    "fit_policy": self.fit_policy,
                    "candidate_policy": self.candidate_policy,
                    "selection_policy": self.selection_policy,
                    "candidate_draw_count": self.candidate_draw_count,
                    "selected_candidate_count": self.selected_candidate_count,
                    "minimum_accepted_candidates": self.minimum_accepted_candidates,
                    "random_seed": self.random_seed,
                }
            digest = object_sha256(identity)
            tasks.append(
                NoiseSubsetSelectionTask(
                    task_id=(
                        f"noise-random-order-select--{cell.cell_id}--{digest[:10]}"
                        if self.control_mode == ANCHORED_CONTROL_MODE
                        else f"noise-random-select--{cell.cell_id}--{digest[:10]}"
                    ),
                    digest=digest,
                    cell=cell,
                )
            )
        value = tuple(tasks)
        self._cache["selection_tasks"] = value
        return value

    def evaluation_tasks(self) -> tuple[NoiseSubsetEvaluationTask, ...]:
        if cached := self._cache.get("evaluation_tasks"):
            return cached
        selection_by_cell = {task.cell.cell_id: task for task in self.selection_tasks()}
        tasks = []
        for cell in self.cells():
            selection = selection_by_cell[cell.cell_id]
            selector_cell = self.selector_cell(cell.cell_id)
            for condition in self.conditions():
                source = self.prefix_task(cell, condition.condition_id)
                for geometry in self.geometries:
                    if self.control_mode == ANCHORED_CONTROL_MODE:
                        identity = {
                            "schema": "simple-noise-random-order-anchored-evaluation-v1",
                            "study_digest": self.digest,
                            "selection_task_digest": selection.digest,
                            "source_prefix_task_digest": source.digest,
                            "cell": cell.cell_id,
                            "condition": asdict(condition),
                            "geometry": geometry,
                            "center_rule": GEOMETRY_RULES[geometry],
                            "q_values": list(self.q_values),
                            "random_order_count": self.random_order_count,
                            "rules": list(self.rules),
                            "patch_size": self.patch_size,
                            "k": self.k,
                            "fill": "dataset_mean",
                            "target_policy": "full_reference_clean_fp32_prediction",
                        }
                    else:
                        q = int(selector_cell["geometries"][geometry]["q"])
                        identity = {
                            "schema": "simple-noise-random-subset-evaluation-v1",
                            "study_digest": self.digest,
                            "selection_task_digest": selection.digest,
                            "source_prefix_task_digest": source.digest,
                            "cell": cell.cell_id,
                            "condition": asdict(condition),
                            "geometry": geometry,
                            "center_rule": GEOMETRY_RULES[geometry],
                            "q": q,
                            "rules": list(self.rules),
                            "patch_size": self.patch_size,
                            "k": self.k,
                            "fill": "dataset_mean",
                            "target_policy": "full_reference_clean_fp32_prediction",
                        }
                    digest = object_sha256(identity)
                    tasks.append(
                        NoiseSubsetEvaluationTask(
                            task_id=(
                                (
                                    "noise-random-order-evaluate"
                                    if self.control_mode == ANCHORED_CONTROL_MODE
                                    else "noise-random-evaluate"
                                )
                                + f"--{cell.cell_id}--{condition.condition_id}--{geometry}--"
                                + digest[:10]
                            ),
                            digest=digest,
                            cell=cell,
                            condition=condition,
                            geometry=geometry,
                            selection_task_id=selection.task_id,
                        )
                    )
        value = tuple(tasks)
        self._cache["evaluation_tasks"] = value
        return value

    def find_selection_task(self, task_id: str) -> NoiseSubsetSelectionTask:
        matches = tuple(task for task in self.selection_tasks() if task.task_id == task_id)
        if len(matches) != 1:
            raise KeyError(f"unknown random NOISE selection task {task_id!r}")
        return matches[0]

    def find_evaluation_task(self, task_id: str) -> NoiseSubsetEvaluationTask:
        matches = tuple(task for task in self.evaluation_tasks() if task.task_id == task_id)
        if len(matches) != 1:
            raise KeyError(f"unknown random NOISE evaluation task {task_id!r}")
        return matches[0]


def load_noise_subset_experiment(path: str | Path) -> NoiseSubsetExperiment:
    source = Path(path).expanduser().resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")) or {}, context="config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("random NOISE config schema_version must be 1")
    prefix = load_noise_prefix_experiment(_path(root["noise_prefix_config"], base=source.parent))
    selector_path = _path(root["independent_selector"], base=source.parent)
    selector = load_independent_geometry_selector(selector_path)
    storage_row = _mapping(root["storage"], context="storage")
    study_id = str(root["study_id"])
    storage = StorageConfig(
        remote_root=os.path.expandvars(str(storage_row["remote_root"])).rstrip("/"),
        scratch_root=_path(storage_row["scratch_root"], base=source.parent),
        rclone_binary=_path(
            storage_row.get("rclone_binary", prefix.storage.rclone_binary),
            base=source.parent,
        ),
        spool_root=_path(
            storage_row.get("spool_root", f"/dev/shm/xai-simple/{study_id}"),
            base=source.parent,
        ),
        spool_max_bytes=_gib(storage_row.get("spool_max_gib", 64), context="spool max"),
        spool_min_free_bytes=int(float(storage_row.get("spool_min_free_gib", 32)) * 2**30),
    )
    runtime_row = _mapping(root["runtime"], context="runtime")
    runtime = NoiseSubsetRuntime(
        database_path=_path(runtime_row["database_path"], base=source.parent),
        log_directory=_path(runtime_row["log_directory"], base=source.parent),
        shared_cache_root=_path(runtime_row["shared_cache_root"], base=source.parent),
        gpu_ids=tuple(
            int(value) for value in _sequence(runtime_row.get("gpu_ids", [0, 1]), context="gpu_ids")
        ),
        inference_batch_size=int(
            runtime_row.get("inference_batch_size", prefix.runtime.inference_batch_size)
        ),
        selection_reservation_bytes=_gib(
            runtime_row.get("selection_reservation_gib", 12), context="selection reservation"
        ),
        evaluation_reservation_bytes=_gib(
            runtime_row.get("evaluation_reservation_gib", 16),
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
    science = _mapping(root["science"], context="science")
    cell_ids = tuple(str(value) for value in _sequence(science["cells"], context="cells"))
    condition_ids = tuple(
        str(value)
        for value in _sequence(science.get("conditions", ["clean"]), context="conditions")
    )
    geometries = tuple(
        str(value)
        for value in _sequence(science.get("geometries", GEOMETRIES), context="geometries")
    )
    rules = tuple(str(value) for value in _sequence(science.get("rules", RULES), context="rules"))
    control_mode = str(science.get("control_mode", KS_CONTROL_MODE))
    common_science = {
        "split": str(science.get("split", "test")),
        "patch_size": int(science.get("patch_size", 16)),
        "k": int(science.get("k", 20)),
        "cells": list(cell_ids),
        "conditions": list(condition_ids),
        "geometries": list(geometries),
        "rules": list(rules),
        "random_seed": int(science.get("random_seed", 20260810)),
    }
    if control_mode == ANCHORED_CONTROL_MODE:
        q_values = tuple(
            int(value)
            for value in _sequence(science.get("q_values", prefix.q_values), context="q_values")
        )
        raw_science = {
            **common_science,
            "control_mode": control_mode,
            "q_values": list(q_values),
            "random_order_count": int(science.get("random_order_count", 10)),
            "fit_policy": str(science.get("fit_policy", ANCHORED_FIT_POLICY)),
            "candidate_policy": str(science.get("candidate_policy", ANCHORED_CANDIDATE_POLICY)),
            "selection_policy": str(science.get("selection_policy", ANCHORED_SELECTION_POLICY)),
        }
    else:
        q_values = tuple(prefix.q_values)
        raw_science = {
            **common_science,
            "candidate_draw_count": int(science.get("candidate_draw_count", 48)),
            "selected_candidate_count": int(science.get("selected_candidate_count", 10)),
            "minimum_accepted_candidates": int(science.get("minimum_accepted_candidates", 5)),
            "fit_policy": str(science.get("fit_policy", FIT_POLICY)),
            "candidate_policy": str(science.get("candidate_policy", CANDIDATE_POLICY)),
            "selection_policy": str(science.get("selection_policy", SELECTION_POLICY)),
        }
    return NoiseSubsetExperiment(
        source_path=source,
        study_id=study_id,
        prefix=prefix,
        selector_path=selector_path,
        selector_digest=str(selector["selector_digest"]),
        storage=storage,
        runtime=runtime,
        split=str(raw_science["split"]),
        patch_size=int(raw_science["patch_size"]),
        k=int(raw_science["k"]),
        cell_ids=cell_ids,
        condition_ids=condition_ids,
        geometries=geometries,
        rules=rules,
        control_mode=control_mode,
        q_values=q_values,
        random_seed=int(raw_science["random_seed"]),
        candidate_draw_count=int(raw_science.get("candidate_draw_count", 0)),
        selected_candidate_count=int(raw_science.get("selected_candidate_count", 0)),
        minimum_accepted_candidates=int(raw_science.get("minimum_accepted_candidates", 0)),
        random_order_count=int(raw_science.get("random_order_count", 0)),
        fit_policy=str(raw_science["fit_policy"]),
        candidate_policy=str(raw_science["candidate_policy"]),
        selection_policy=str(raw_science["selection_policy"]),
        raw_science=raw_science,
    )


__all__ = [
    "ANCHORED_CANDIDATE_POLICY",
    "ANCHORED_CONTROL_MODE",
    "ANCHORED_FIT_POLICY",
    "ANCHORED_SELECTION_POLICY",
    "CANDIDATE_POLICY",
    "FIT_POLICY",
    "GEOMETRIES",
    "GEOMETRY_RULES",
    "KS_CONTROL_MODE",
    "SELECTION_POLICY",
    "NoiseSubsetEvaluationTask",
    "NoiseSubsetExperiment",
    "NoiseSubsetRuntime",
    "NoiseSubsetSelectionTask",
    "load_noise_subset_experiment",
]
