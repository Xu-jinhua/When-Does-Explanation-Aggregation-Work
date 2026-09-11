"""Read-only readiness checks for the assumptions experiment inputs."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..adversarial import completed_adversarial_manifest
from ..artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactStore,
    completed_manifest,
    phase2_artifact_root,
)
from ..methods import PATCH_METHODS
from ..phase2 import _source_manifests
from ..profiler import load_phase2_profile
from ..runtime_dependencies import relprop_runtime_readiness, require_relprop_runtime
from .config import AssumptionExperiment
from .phase1 import _base_profile


def _base_queue_statuses(experiment: AssumptionExperiment) -> tuple[Mapping[str, str], str | None]:
    path = experiment.base.runtime.database_path
    if not path.is_file():
        return {}, None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='experiment_digest'"
            ).fetchone()
            if row is None or str(row[0]) != experiment.base.scheduler_digest:
                raise ValueError("base scheduler database digest does not match the current plan")
            values = {
                str(job_id): str(status)
                for job_id, status in connection.execute("SELECT job_id,status FROM jobs")
            }
    except (OSError, sqlite3.Error, ValueError) as error:
        return {}, str(error)
    return values, None


def _artifact_group(
    tasks: Sequence[Any],
    *,
    kind: str,
    statuses: Mapping[str, str],
    queue_authoritative: bool,
    probe: Callable[[Any], bool],
) -> Mapping[str, Any]:
    complete = []
    missing = []
    invalid = []
    evidence = Counter()
    queue_states = Counter()
    for task in tasks:
        job_id = f"{kind}:{task.task_id}"
        queue_status = statuses.get(job_id)
        queue_states[queue_status or "unregistered"] += 1
        if queue_status == "succeeded":
            complete.append(task.task_id)
            evidence["queue_succeeded"] += 1
            continue
        if queue_status is not None:
            missing.append(task.task_id)
            continue
        if queue_authoritative:
            missing.append(task.task_id)
            continue
        try:
            if probe(task):
                complete.append(task.task_id)
                evidence["verified_manifest"] += 1
            else:
                missing.append(task.task_id)
        except Exception as error:  # Report contradictions without hiding other missing inputs.
            invalid.append({"task_id": task.task_id, "error": str(error)})
    return {
        "ready": len(complete) == len(tasks) and not invalid,
        "required": len(tasks),
        "complete": len(complete),
        "missing": missing,
        "invalid": invalid,
        "evidence": dict(sorted(evidence.items())),
        "queue_states": dict(sorted(queue_states.items())),
    }


def _profile_group(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    if not experiment.uses_source_bank:
        return {
            "ready": True,
            "required_method_profiles": 0,
            "resolved_method_profiles": 0,
            "unique_profile_count": 0,
            "invalid": [],
            "skipped": "no_source_bank_setting",
        }
    requirements = []
    invalid = []
    resolved_profile_ids = set()
    scopes = {
        cell.cell_id: next(
            task
            for task in experiment.source_phase1_tasks()
            if task.cell.cell_id == cell.cell_id and task.condition.kind == "clean"
        )
        for cell in experiment.cells()
    }
    for cell in experiment.cells():
        scope = scopes[cell.cell_id]
        for definition in experiment.base.methods.for_architecture(
            cell.reference_model.architecture
        ):
            variants = definition.instances(cell.reference_model.architecture)
            if definition.family in PATCH_METHODS:
                variants = tuple(item for item in variants if item.variant == "p16")
            for variant in variants:
                requirement = f"{cell.cell_id}/{variant.artifact_name}"
                requirements.append(requirement)
                try:
                    profile, _ = _base_profile(experiment, scope, variant)
                    resolved_profile_ids.add(profile.profile_id)
                except Exception as error:
                    invalid.append({"requirement": requirement, "error": str(error)})
    return {
        "ready": not invalid,
        "required_method_profiles": len(requirements),
        "resolved_method_profiles": len(requirements) - len(invalid),
        "unique_profile_count": len(resolved_profile_ids),
        "invalid": invalid,
    }


def _phase2_profile_group(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    missing = []
    invalid = []
    profiles = experiment.base.phase2_profiles()
    for profile in profiles:
        try:
            result = load_phase2_profile(experiment.base, profile)
            if result is None:
                missing.append(profile.profile_id)
            elif result.selected_batch_size != experiment.runtime.inference_batch_size:
                invalid.append(
                    {
                        "profile_id": profile.profile_id,
                        "error": (
                            f"measured forward batch {result.selected_batch_size} does not match "
                            f"assumptions batch {experiment.runtime.inference_batch_size}"
                        ),
                    }
                )
        except Exception as error:
            invalid.append({"profile_id": profile.profile_id, "error": str(error)})
    return {
        "ready": not missing and not invalid,
        "required": len(profiles),
        "complete": len(profiles) - len(missing) - len(invalid),
        "missing": missing,
        "invalid": invalid,
    }


def _required_phase1_tasks(experiment: AssumptionExperiment) -> tuple[Any, ...]:
    expected = {
        (
            cell.dataset.dataset_id,
            cell.reference_model.model_id,
            condition.condition_id,
            method,
        )
        for cell in experiment.cells()
        for condition in experiment.base.conditions
        for method in cell.methods
    }
    result = tuple(
        task
        for task in experiment.base.phase1_tasks()
        if task.split == experiment.split
        and (
            task.dataset.dataset_id,
            task.model.model_id,
            task.condition.condition_id,
            task.family,
        )
        in expected
    )
    observed = {
        (
            task.dataset.dataset_id,
            task.model.model_id,
            task.condition.condition_id,
            task.family,
        )
        for task in result
    }
    if observed != expected or len(result) != len(expected):
        raise RuntimeError("The base plan does not contain exactly one required Phase 1 task")
    return result


def base_input_readiness(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    """Report whether every immutable NAIVE input required by this DAG exists."""

    statuses, queue_error = _base_queue_statuses(experiment)
    queue_authoritative = experiment.base.runtime.database_path.is_file() and queue_error is None
    base_store = ArtifactStore(experiment.base)
    base_phase2 = tuple(
        experiment.base_phase2_task(cell, condition.condition_id)
        for cell in experiment.cells()
        for condition in experiment.base.conditions
    )
    phase2_by_scope = {
        (task.dataset.dataset_id, task.model.model_id, task.condition.condition_id): task
        for task in base_phase2
    }

    def phase1_complete(task: Any) -> bool:
        source = phase2_by_scope[
            task.dataset.dataset_id,
            task.model.model_id,
            task.condition.condition_id,
        ]
        _source_manifests(experiment.base, source, base_store, (task.family,))
        return True

    phase1 = _artifact_group(
        _required_phase1_tasks(experiment),
        kind="phase1",
        statuses=statuses,
        queue_authoritative=queue_authoritative,
        probe=phase1_complete,
    )
    phase2 = _artifact_group(
        base_phase2,
        kind="phase2",
        statuses=statuses,
        queue_authoritative=queue_authoritative,
        probe=lambda task: (
            completed_manifest(
                base_store,
                phase2_artifact_root(task),
                expected_task_digest=task.digest,
                expected_schema_version=PHASE2_SCHEMA_VERSION,
            )
            is not None
        ),
    )
    adversarial_tasks = tuple(
        task for task in experiment.base.adversarial_tasks() if task.split == experiment.split
    )
    adversarial = _artifact_group(
        adversarial_tasks,
        kind="adversarial",
        statuses=statuses,
        queue_authoritative=queue_authoritative,
        probe=lambda task: (
            completed_adversarial_manifest(experiment.base, task, store=base_store) is not None
        ),
    )
    groups = {
        "phase1_profiles": _profile_group(experiment),
        "phase2_profiles": _phase2_profile_group(experiment),
        "adversarial_datasets": adversarial,
        "phase1_p16": phase1,
        "phase2_p16": phase2,
    }
    return {
        "schema_version": 1,
        "ready": all(bool(group["ready"]) for group in groups.values()),
        "base_experiment_id": experiment.base.experiment_id,
        "base_phase1_digest": experiment.base.phase1_digest,
        "base_scheduler_database": str(experiment.base.runtime.database_path),
        "base_scheduler_database_error": queue_error,
        "evidence_policy": "authoritative_current_queue_else_verified_immutable_manifest",
        "groups": groups,
    }


def _relprop_requirements(experiment: AssumptionExperiment) -> tuple[tuple[str, str], ...]:
    if not experiment.uses_source_bank:
        return ()
    return tuple(
        (method, cell.reference_model.architecture)
        for cell in experiment.cells()
        for method in cell.methods
    )


def runtime_dependency_readiness(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    relprop = relprop_runtime_readiness(_relprop_requirements(experiment))
    return {
        "schema_version": 1,
        "ready": bool(relprop["ready"]),
        "relprop": relprop,
    }


def assumption_readiness(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    base_inputs = base_input_readiness(experiment)
    runtime_dependencies = runtime_dependency_readiness(experiment)
    return {
        "schema_version": 1,
        "ready": bool(base_inputs["ready"] and runtime_dependencies["ready"]),
        "base_inputs": base_inputs,
        "runtime_dependencies": runtime_dependencies,
    }


def require_runtime_dependencies(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    relprop = require_relprop_runtime(_relprop_requirements(experiment))
    return {
        "schema_version": 1,
        "ready": True,
        "relprop": relprop,
    }


def require_base_inputs_ready(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    report = base_input_readiness(experiment)
    if report["ready"]:
        return report
    missing = {
        name: {
            key: group[key]
            for key in ("required", "complete", "missing", "invalid")
            if key in group
        }
        for name, group in report["groups"].items()
        if not group["ready"]
    }
    raise RuntimeError(f"Assumptions base inputs are not ready: {missing}")


__all__ = [
    "assumption_readiness",
    "base_input_readiness",
    "require_base_inputs_ready",
    "require_runtime_dependencies",
    "runtime_dependency_readiness",
]
