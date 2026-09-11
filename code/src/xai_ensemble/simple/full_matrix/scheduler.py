"""One SQLite scheduler for the complete GitHub result matrix.

The component schedulers in :mod:`xai_ensemble.simple` are useful when a
single study is run in isolation.  The complete matrix has a longer DAG and
must not create one queue per component, so this module collects their jobs
into one queue and extends it only after the preceding barrier succeeds.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from xai_ensemble.core.gpu import GpuProbeError, NvidiaSmiProbe
from xai_ensemble.core.paths import (
    FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV,
    FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV,
    FULL_MATRIX_LOGICAL_ASSET_ROOT_ENV,
    FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV,
    FULL_MATRIX_OVERLAY_DATASETS_ENV,
)
from xai_ensemble.simple.runtime import (
    GPU_RELEASE_JOB_ENV,
    GPU_RELEASE_PATH_ENV,
    GPU_RELEASE_TOKEN_ENV,
    gpu_release_signal_matches,
)
from xai_ensemble.simple.scheduler import (
    STATUSES,
    QueueJob,
    RunningProcess,
    SimpleJobStore,
    _gpu_release_blocks_admission,
    _launch,
    _outstanding_reservation,
    _process_is_zombie,
    _ready_jobs,
    _refresh_gpu_release_admission,
    _requires_exclusive_gpu,
    _reservation_fits,
    _resolve_reservation,
)

from .assets import (
    GPU_JOB_KINDS,
    compatibility_complete,
    compatibility_job_id,
    planned_asset_jobs,
)
from .catalog import MatrixCell
from .config import FullMatrixExperiment
from .planner import ExecutionScope, JobCollector, build_execution_scope, execution_paths

MATRIX_SCHEMA = "simple-full-matrix-scheduler-v1"
DEFERRED_DATASETS_SCHEMA = "simple-full-matrix-deferred-datasets-v1"
DEFERRED_DATASETS_FILENAME = "deferred-datasets.json"
CPU_KINDS = frozenset(
    {
        "matrix-manifest",
        "matrix-partition",
        "matrix-samples",
        "matrix-mean",
        "matrix-static-compatibility",
        "matrix-materialize",
        "matrix-barrier",
        "matrix-prefix-inputs",
        "matrix-selector",
        "matrix-selector-merge",
        "matrix-summary",
        "matrix-coverage",
        "partition",
    }
)
GPU_KINDS = frozenset(GPU_JOB_KINDS) | frozenset(
    {
        "profile",
        "phase2_profile",
        "phase1",
        "phase2",
        "adversarial",
        "training",
        "source-method",
        "rank",
        "evaluation",
        "noise-prefix-evaluation",
    }
)
# These jobs carry either a fixed, conservative 44 GiB declaration or a
# profile-derived estimate that can be slightly above the usable portion of a
# physical L40S. Resolve that declaration/estimate against the device ceiling
# before admission so the driver-reserved portion does not make a job
# permanently unrunnable. The ceiling is static and occupancy-independent, so
# the effective reservation stays truthful for outstanding-capacity
# accounting: a clamped job still blocks any co-admission on the same device.
_CAPACITY_CLAMPED_GPU_JOB_KINDS = frozenset(GPU_JOB_KINDS) | frozenset(
    {"training", "source-method"}
)
# A source-method estimate above this threshold is close enough to the usable
# capacity that it must not start beside an already-running worker.  The
# threshold is independent of the frozen queue/configuration digest.  It sits
# just below the measured IntegratedGradients reservation (~28.6 GiB): the
# 2026-09-05/06 IG failures showed that observed-memory packing can admit a
# second large job onto a device whose live accounting is momentarily stale,
# so every reservation from IG upward runs alone on a physically idle device.
_HIGH_RESERVATION_STRICT_THRESHOLD_BYTES = 28 * 2**30

MATERIALIZE_JOB_ID = "matrix-materialize-config"
BASE_SUMMARY_JOB_ID = "matrix-base-summary"
PREFIX_READY_JOB_ID = "matrix-prefix-ready"
PREFIX_COMPLETE_JOB_ID = "matrix-prefix-complete"
SELECTOR_MERGE_JOB_ID = "matrix-selector-merge"
PREFIX_SUMMARY_JOB_ID = "matrix-prefix-summary"
SELECTED_NOISE_SUMMARY_JOB_ID = "matrix-selected-noise-summary"
IND_SUMMARY_JOB_ID = "matrix-ind-summary"
COVERAGE_JOB_ID = "matrix-coverage"
_SCOPED_JOB_PREFIX = "scope"

# Execution-only stop floors. These are deliberately outside the frozen
# matrix configuration and scheduler digest.
LOCAL_DISK_FLOOR_BYTES = 30 * 2**30
SHM_FLOOR_BYTES = 40 * 2**30
_CAPACITY_STOP_GRACE_SECONDS = 5.0
_CAPACITY_STOP_KILL_WAIT_SECONDS = 5.0

_STAGE_ORDER = {
    "matrix-manifest": 0,
    "matrix-partition": 1,
    "matrix-samples": 1,
    "matrix-mean": 2,
    "matrix-reference-training": 3,
    "matrix-compatibility": 4,
    "matrix-static-compatibility": 0,
    "matrix-materialize": 5,
    "profile": 6,
    "phase2_profile": 6,
    "adversarial": 7,
    "phase1": 8,
    "matrix-summary": 9,
    "partition": 10,
    "training": 11,
    "source-method": 12,
    "rank": 13,
    "phase2": 14,
    "evaluation": 15,
    "noise-prefix-evaluation": 15,
    "matrix-prefix-inputs": 16,
    "matrix-barrier": 17,
    "matrix-selector": 18,
    "matrix-selector-merge": 19,
    "matrix-coverage": 20,
}


def _scope_stage_id(scope: ExecutionScope, stage: str) -> str:
    return f"{_SCOPED_JOB_PREFIX}:{scope.scope_id}:{stage}"


def _scope_component_prefix(scope: ExecutionScope, component: str) -> str:
    return f"{_SCOPED_JOB_PREFIX}:{scope.scope_id}:{component}"


def _scope_id_from_job(job: QueueJob) -> str | None:
    parts = job.job_id.split(":")
    if len(parts) >= 2 and parts[0] == _SCOPED_JOB_PREFIX:
        return parts[1]
    return None


def _component_for(job: QueueJob) -> str | None:
    parts = job.job_id.split(":")
    if len(parts) >= 3 and parts[0] == _SCOPED_JOB_PREFIX:
        return parts[2] if parts[2] in {"base", "prefix", "ind"} else None
    if not parts:
        return None
    return parts[0] if parts[0] in {"base", "prefix", "ind"} else None


def _stage_job(
    experiment: FullMatrixExperiment,
    *,
    job_id: str,
    kind: str,
    command: tuple[str, ...],
    dependencies: Sequence[str] = (),
    status: str = "pending",
) -> QueueJob:
    safe = job_id.replace(":", "--")
    return QueueJob(
        job_id=job_id,
        kind=kind,
        command=command,
        dependencies=tuple(dependencies),
        resource_ids=(),
        reservation_bytes=0 if kind in CPU_KINDS else None,
        status=status,
        attempts=0,
        max_retries=experiment.runtime.max_retries,
        pid=None,
        gpu_id=None,
        log_path=str(experiment.runtime.log_directory / "matrix" / f"{safe}.log"),
    )


def _prefix_jobs(
    jobs: Sequence[QueueJob],
    prefix: str,
    *,
    dependency: str | None = None,
    log_directory: Path,
) -> tuple[QueueJob, ...]:
    ids = {job.job_id: f"{prefix}:{job.job_id}" for job in jobs}
    result = []
    for job in jobs:
        dependencies = [ids.get(value, value) for value in job.dependencies]
        if dependency is not None and dependency not in dependencies:
            dependencies.append(dependency)
        result.append(
            replace(
                job,
                job_id=ids[job.job_id],
                dependencies=tuple(dependencies),
                log_path=str(log_directory / prefix / Path(job.log_path).name),
            )
        )
    return tuple(result)


def _component_job_ids(
    store: SimpleJobStore,
    component: str,
    *,
    scope: ExecutionScope | None = None,
) -> tuple[str, ...]:
    marker = (
        f"{_scope_component_prefix(scope, component)}:" if scope is not None else f"{component}:"
    )
    return tuple(job.job_id for job in store.jobs() if job.job_id.startswith(marker))


def _job_succeeded(store: SimpleJobStore, job_id: str) -> bool:
    return any(job.job_id == job_id and job.status == "succeeded" for job in store.jobs())


def _stage_complete(
    experiment: FullMatrixExperiment,
    job_id: str,
    *,
    scope: ExecutionScope | None = None,
) -> bool:
    component_paths = execution_paths(experiment, scope)
    materialize_id = (
        _scope_stage_id(scope, "materialize") if scope is not None else MATERIALIZE_JOB_ID
    )
    if job_id == materialize_id:
        generated = [
            component_paths.active_cells_path,
            component_paths.base_config_path,
            component_paths.assumptions_config_path,
            component_paths.prefix_config_path,
        ]
        if component_paths.scope_manifest_path is not None:
            generated.append(component_paths.scope_manifest_path)
        cells = (
            tuple(experiment.cell(cell_id) for cell_id in scope.cell_ids)
            if scope
            else experiment.cells()
        )
        return all(path.is_file() for path in generated) and all(
            compatibility_complete(experiment, cell) for cell in cells
        )
    control = component_paths.result_root / "control"
    if scope is not None:
        stage_ids = {
            "prefix-ready": _scope_stage_id(scope, "prefix-ready"),
            "prefix-complete": _scope_stage_id(scope, "prefix-complete"),
            "selector-merge": _scope_stage_id(scope, "selector-merge"),
            "base-summary": _scope_stage_id(scope, "base-summary"),
            "prefix-summary": _scope_stage_id(scope, "prefix-summary"),
            "selected-noise-summary": _scope_stage_id(scope, "selected-noise-summary"),
            "ind-summary": _scope_stage_id(scope, "ind-summary"),
            "coverage": _scope_stage_id(scope, "coverage"),
        }
    else:
        stage_ids = {
            "prefix-ready": PREFIX_READY_JOB_ID,
            "prefix-complete": PREFIX_COMPLETE_JOB_ID,
            "selector-merge": SELECTOR_MERGE_JOB_ID,
            "base-summary": BASE_SUMMARY_JOB_ID,
            "prefix-summary": PREFIX_SUMMARY_JOB_ID,
            "selected-noise-summary": SELECTED_NOISE_SUMMARY_JOB_ID,
            "ind-summary": IND_SUMMARY_JOB_ID,
            "coverage": COVERAGE_JOB_ID,
        }
    targets = {
        stage_ids["prefix-ready"]: control / "prefix-ready.json",
        stage_ids["prefix-complete"]: control / "prefix-complete.json",
        stage_ids["selector-merge"]: component_paths.result_root
        / "noise-prefix"
        / "selector"
        / "selector.json",
        stage_ids["base-summary"]: component_paths.result_root / "base" / "complete.json",
        stage_ids["prefix-summary"]: component_paths.result_root
        / "noise-prefix"
        / "summary"
        / "summary.json",
        stage_ids["selected-noise-summary"]: component_paths.result_root
        / "noise-prefix"
        / "selected"
        / "summary.json",
        stage_ids["ind-summary"]: component_paths.result_root / "ind" / "summary" / "summary.json",
        stage_ids["coverage"]: component_paths.result_root / "coverage" / "coverage.json",
    }
    target = targets.get(job_id)
    return target is not None and target.is_file()


def _command(
    experiment: FullMatrixExperiment,
    action: str,
    *values: str,
    scope: ExecutionScope | None = None,
) -> tuple[str, ...]:
    command = (
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "simple",
        "full-matrix",
        action,
        "--config",
        str(experiment.source_path),
    )
    if scope is None:
        return (*command, *values)
    if action == "materialize-config":
        scope_values: list[str] = ["--scope-digest", scope.scope_digest]
        for dataset in scope.deferred_datasets:
            scope_values.extend(("--deferred-dataset", dataset))
        for cell_id in scope.cell_ids:
            scope_values.extend(("--scope-cell", cell_id))
        return (*command, *values, *scope_values)
    scope_manifest = execution_paths(experiment, scope).scope_manifest_path
    assert scope_manifest is not None
    return (*command, *values, "--scope-manifest", str(scope_manifest))


def _submit_stage(
    store: SimpleJobStore,
    experiment: FullMatrixExperiment,
    *,
    job_id: str,
    kind: str,
    action: str,
    values: Sequence[str] = (),
    dependencies: Sequence[str] = (),
    scope: ExecutionScope | None = None,
) -> None:
    status = "succeeded" if _stage_complete(experiment, job_id, scope=scope) else "pending"
    store.submit(
        _stage_job(
            experiment,
            job_id=job_id,
            kind=kind,
            command=_command(experiment, action, *values, scope=scope),
            dependencies=dependencies,
            status=status,
        )
    )


def _submit_initial_plan(experiment: FullMatrixExperiment, store: SimpleJobStore) -> None:
    jobs = planned_asset_jobs(experiment)
    store.submit_many(jobs)
    _submit_stage(
        store,
        experiment,
        job_id=MATERIALIZE_JOB_ID,
        kind="matrix-materialize",
        action="materialize-config",
        dependencies=tuple(job.job_id for job in jobs),
    )


def _option_values(command: Sequence[str], option: str) -> tuple[str, ...]:
    values = []
    for index, value in enumerate(command):
        if value != option:
            continue
        if index + 1 >= len(command):
            raise ValueError(f"Queued command is missing a value for {option}")
        values.append(command[index + 1])
    return tuple(values)


def _scope_from_materialize_job(
    experiment: FullMatrixExperiment,
    job: QueueJob,
) -> ExecutionScope:
    parts = job.job_id.split(":")
    if len(parts) != 3 or parts[0] != _SCOPED_JOB_PREFIX or parts[2] != "materialize":
        raise ValueError(f"Queued job is not a scoped materialization: {job.job_id}")
    digests = _option_values(job.command, "--scope-digest")
    if len(digests) != 1:
        raise ValueError(f"Scoped materialization has an invalid digest: {job.job_id}")
    scope = build_execution_scope(
        experiment,
        deferred_datasets=_option_values(job.command, "--deferred-dataset"),
        cell_ids=_option_values(job.command, "--scope-cell"),
        expected_digest=digests[0],
    )
    if scope.scope_id != parts[1]:
        raise ValueError(f"Scoped materialization identity is contradictory: {job.job_id}")
    return scope


def _scopes_from_store(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
) -> Mapping[str, ExecutionScope]:
    scopes: dict[str, ExecutionScope] = {}
    for job in store.jobs():
        parts = job.job_id.split(":")
        if len(parts) != 3 or parts[0] != _SCOPED_JOB_PREFIX or parts[2] != "materialize":
            continue
        scope = _scope_from_materialize_job(experiment, job)
        if scope.scope_id in scopes:
            raise ValueError(f"Duplicate execution scope: {scope.scope_id}")
        scopes[scope.scope_id] = scope
    return scopes


def _scope_candidate_cells(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    deferred_datasets: frozenset[str],
    excluded_cell_ids: frozenset[str],
) -> tuple[MatrixCell, ...]:
    """Return unscoped cells that have passed the real compatibility gate."""

    jobs = store.jobs()
    statuses = {job.job_id: job.status for job in jobs}
    jobs_by_id = {job.job_id: job for job in jobs}
    candidates = []
    for cell in experiment.cells():
        if cell.dataset_id in deferred_datasets or cell.cell_id in excluded_cell_ids:
            continue
        training_id = f"matrix-reference-training:{cell.cell_id}"
        gate_id = compatibility_job_id(cell)
        gate = jobs_by_id.get(gate_id)
        if statuses.get(training_id) != "succeeded" or gate is None or gate.status != "succeeded":
            continue
        if compatibility_complete(experiment, cell, verify_checkpoint=False):
            gate_path = experiment.compatibility_directory(cell) / "gate.json"
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            if isinstance(gate_payload, Mapping) and gate_payload.get("status") == "passed":
                candidates.append(cell)
    return tuple(sorted(candidates, key=lambda cell: cell.cell_id))


def _scope_compatibility_prerequisites(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    deferred_datasets: frozenset[str],
    scopes: Mapping[str, ExecutionScope],
) -> frozenset[str]:
    """Return ready compatibility gates needed to unlock the next scope.

    The global stage order otherwise favors every remaining reference training
    before compatibility.  While a dataset is deferred, promote just these
    nondeferred gates so Phase 1/2 scopes can begin as soon as a cell is ready.
    """

    if not deferred_datasets:
        return frozenset()
    scoped_cells = frozenset(cell_id for scope in scopes.values() for cell_id in scope.cell_ids)
    jobs = store.jobs()
    statuses = {job.job_id: job.status for job in jobs}
    jobs_by_id = {job.job_id: job for job in jobs}
    prerequisites = set()
    for cell in experiment.cells():
        if cell.dataset_id in deferred_datasets or cell.cell_id in scoped_cells:
            continue
        training_id = f"matrix-reference-training:{cell.cell_id}"
        gate = jobs_by_id.get(compatibility_job_id(cell))
        if (
            statuses.get(training_id) == "succeeded"
            and gate is not None
            and gate.status == "pending"
            and all(statuses.get(dependency) == "succeeded" for dependency in gate.dependencies)
        ):
            prerequisites.add(gate.job_id)
    return frozenset(prerequisites)


def _scope_has_unresolved_compatibility(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    deferred_datasets: frozenset[str],
    scopes: Mapping[str, ExecutionScope],
) -> bool:
    """Keep a ready Phase-0 cohort together until every gate is terminal.

    The scheduler may run two compatibility gates at a time. Freezing a scope
    after each pair would fragment one coherent reference-training cohort into
    many small partial experiments. Gates blocked by normal dependency or
    retry handling are terminal for this purpose and are excluded later.
    """

    scoped_cells = frozenset(cell_id for scope in scopes.values() for cell_id in scope.cell_ids)
    jobs = {job.job_id: job for job in store.jobs()}
    for cell in experiment.cells():
        if cell.dataset_id in deferred_datasets or cell.cell_id in scoped_cells:
            continue
        training = jobs.get(f"matrix-reference-training:{cell.cell_id}")
        gate = jobs.get(compatibility_job_id(cell))
        if (
            training is not None
            and training.status == "succeeded"
            and gate is not None
            and gate.status in {"pending", "running"}
        ):
            return True
    return False


def _ensure_deferred_ready_scope(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    deferred_datasets: frozenset[str],
) -> Mapping[str, ExecutionScope]:
    """Append immutable downstream scopes without changing Phase-0 rows."""

    scopes = dict(_scopes_from_store(experiment, store))
    if not deferred_datasets:
        return scopes
    scoped_cells = frozenset(cell_id for scope in scopes.values() for cell_id in scope.cell_ids)
    if _scope_has_unresolved_compatibility(
        experiment,
        store,
        deferred_datasets=deferred_datasets,
        scopes=scopes,
    ):
        return scopes
    cells = _scope_candidate_cells(
        experiment,
        store,
        deferred_datasets=deferred_datasets,
        excluded_cell_ids=scoped_cells,
    )
    if not cells:
        return scopes
    scope = build_execution_scope(
        experiment,
        deferred_datasets=tuple(sorted(deferred_datasets)),
        cell_ids=tuple(cell.cell_id for cell in cells),
    )
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "materialize"),
        kind="matrix-materialize",
        action="materialize-config",
        dependencies=tuple(compatibility_job_id(cell) for cell in cells),
        scope=scope,
    )
    scopes[scope.scope_id] = scope
    return scopes


def _submit_base_component(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    scope: ExecutionScope | None = None,
) -> None:
    if _component_job_ids(store, "base", scope=scope):
        return
    from xai_ensemble.simple.config import load_experiment
    from xai_ensemble.simple.scheduler import submit_plan

    paths = execution_paths(experiment, scope)
    base = load_experiment(paths.base_config_path)
    collector = JobCollector()
    submit_plan(base, collector, include_phase2=True)
    jobs = _prefix_jobs(
        collector.jobs(),
        _scope_component_prefix(scope, "base") if scope is not None else "base",
        dependency=_scope_stage_id(scope, "materialize")
        if scope is not None
        else MATERIALIZE_JOB_ID,
        log_directory=(
            experiment.runtime.log_directory / "scopes" / scope.scope_id
            if scope is not None
            else experiment.runtime.log_directory
        ),
    )
    store.submit_many(jobs)
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "base-summary") if scope is not None else BASE_SUMMARY_JOB_ID,
        kind="matrix-summary",
        action="summarize-base",
        dependencies=tuple(job.job_id for job in jobs),
        scope=scope,
    )


def _submit_prefix_component(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    scope: ExecutionScope | None = None,
) -> None:
    if _component_job_ids(store, "prefix", scope=scope):
        return
    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment
    from xai_ensemble.simple.noise_prefix.scheduler import submit_plan

    paths = execution_paths(experiment, scope)
    prefix = load_noise_prefix_experiment(paths.prefix_config_path)
    collector = JobCollector()
    submit_plan(prefix, collector, scan_existing=True)
    jobs = _prefix_jobs(
        collector.jobs(),
        _scope_component_prefix(scope, "prefix") if scope is not None else "prefix",
        dependency=_scope_stage_id(scope, "prefix-ready")
        if scope is not None
        else PREFIX_READY_JOB_ID,
        log_directory=(
            experiment.runtime.log_directory / "scopes" / scope.scope_id
            if scope is not None
            else experiment.runtime.log_directory
        ),
    )
    store.submit_many(jobs)
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "prefix-complete")
        if scope is not None
        else PREFIX_COMPLETE_JOB_ID,
        kind="matrix-barrier",
        action="barrier",
        values=("prefix-complete",),
        dependencies=tuple(job.job_id for job in jobs),
        scope=scope,
    )
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "prefix-summary")
        if scope is not None
        else PREFIX_SUMMARY_JOB_ID,
        kind="matrix-summary",
        action="summarize-prefix",
        dependencies=tuple(job.job_id for job in jobs),
        scope=scope,
    )


def _submit_selector_component(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    scope: ExecutionScope | None = None,
) -> None:
    if _component_job_ids(store, "selector", scope=scope):
        return
    from .planner import load_active_cells, load_scoped_active_cells

    paths = execution_paths(experiment, scope)
    cells = (
        load_scoped_active_cells(experiment, scope)
        if scope is not None
        else load_active_cells(experiment)
    )
    jobs = []
    for cell in cells:
        job_id = (
            f"{_scope_component_prefix(scope, 'selector')}:{cell.cell_id}"
            if scope is not None
            else f"selector:{cell.cell_id}"
        )
        partial = (
            paths.result_root / "noise-prefix" / "selector" / "partials" / f"{cell.cell_id}.json"
        )
        jobs.append(
            _stage_job(
                experiment,
                job_id=job_id,
                kind="matrix-selector",
                command=_command(
                    experiment,
                    "selector-cell",
                    "--cell",
                    cell.cell_id,
                    scope=scope,
                ),
                dependencies=(
                    _scope_stage_id(scope, "prefix-complete")
                    if scope is not None
                    else PREFIX_COMPLETE_JOB_ID,
                ),
                status="succeeded" if partial.is_file() else "pending",
            )
        )
    store.submit_many(jobs)
    partial_ids = tuple(job.job_id for job in jobs)
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "selector-merge")
        if scope is not None
        else SELECTOR_MERGE_JOB_ID,
        kind="matrix-selector-merge",
        action="merge-selector",
        dependencies=partial_ids,
        scope=scope,
    )
    _submit_stage(
        store,
        experiment,
        job_id=(
            _scope_stage_id(scope, "selected-noise-summary")
            if scope is not None
            else SELECTED_NOISE_SUMMARY_JOB_ID
        ),
        kind="matrix-summary",
        action="summarize-selected-noise",
        dependencies=(
            _scope_stage_id(scope, "selector-merge")
            if scope is not None
            else SELECTOR_MERGE_JOB_ID,
            _scope_stage_id(scope, "prefix-summary")
            if scope is not None
            else PREFIX_SUMMARY_JOB_ID,
        ),
        scope=scope,
    )


def _submit_ind_component(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    scope: ExecutionScope | None = None,
) -> None:
    if _component_job_ids(store, "ind", scope=scope):
        return
    from xai_ensemble.simple.assumptions.config import load_assumption_experiment
    from xai_ensemble.simple.assumptions.table_priority import submit_plan

    paths = execution_paths(experiment, scope)
    assumptions = load_assumption_experiment(paths.assumptions_config_path)
    collector = JobCollector()
    from xai_ensemble.simple.assumptions.table_priority import table_priority_paths

    queue_paths = table_priority_paths(assumptions, root=paths.result_root / "ind" / "queue")
    submit_plan(assumptions, collector, paths=queue_paths)
    jobs = _prefix_jobs(
        collector.jobs(),
        _scope_component_prefix(scope, "ind") if scope is not None else "ind",
        dependency=(
            _scope_stage_id(scope, "selected-noise-summary")
            if scope is not None
            else SELECTED_NOISE_SUMMARY_JOB_ID
        ),
        log_directory=(
            experiment.runtime.log_directory / "scopes" / scope.scope_id
            if scope is not None
            else experiment.runtime.log_directory
        ),
    )
    store.submit_many(jobs)
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "ind-summary") if scope is not None else IND_SUMMARY_JOB_ID,
        kind="matrix-summary",
        action="summarize-ind",
        dependencies=tuple(job.job_id for job in jobs),
        scope=scope,
    )


def _submit_coverage(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    scope: ExecutionScope | None = None,
) -> None:
    _submit_stage(
        store,
        experiment,
        job_id=_scope_stage_id(scope, "coverage") if scope is not None else COVERAGE_JOB_ID,
        kind="matrix-coverage",
        action="coverage",
        dependencies=(
            _scope_stage_id(scope, "ind-summary") if scope is not None else IND_SUMMARY_JOB_ID,
        ),
        scope=scope,
    )


def _extend_plan(
    experiment: FullMatrixExperiment,
    store: SimpleJobStore,
    *,
    scopes: Mapping[str, ExecutionScope] | None = None,
) -> None:
    """Extend the queue exactly one barrier at a time.

    The checks are intentionally based on queue state rather than a process
    local flag.  A scheduler restart therefore reconstructs the same plan and
    never starts IND before the selected NOISE summary has succeeded.
    """

    if _job_succeeded(store, MATERIALIZE_JOB_ID):
        _submit_base_component(experiment, store)
    if _job_succeeded(store, BASE_SUMMARY_JOB_ID):
        _submit_stage(
            store,
            experiment,
            job_id=PREFIX_READY_JOB_ID,
            kind="matrix-prefix-inputs",
            action="prepare-prefix-inputs",
            dependencies=(BASE_SUMMARY_JOB_ID,),
        )
    if _job_succeeded(store, PREFIX_READY_JOB_ID):
        _submit_prefix_component(experiment, store)
    if _job_succeeded(store, PREFIX_COMPLETE_JOB_ID):
        _submit_selector_component(experiment, store)
    if _job_succeeded(store, SELECTED_NOISE_SUMMARY_JOB_ID):
        _submit_ind_component(experiment, store)
    if _job_succeeded(store, IND_SUMMARY_JOB_ID):
        _submit_coverage(experiment, store)
    for scope in (scopes or _scopes_from_store(experiment, store)).values():
        materialize_id = _scope_stage_id(scope, "materialize")
        base_summary_id = _scope_stage_id(scope, "base-summary")
        prefix_ready_id = _scope_stage_id(scope, "prefix-ready")
        prefix_complete_id = _scope_stage_id(scope, "prefix-complete")
        selected_noise_id = _scope_stage_id(scope, "selected-noise-summary")
        ind_summary_id = _scope_stage_id(scope, "ind-summary")
        if _job_succeeded(store, materialize_id):
            _submit_base_component(experiment, store, scope=scope)
        if _job_succeeded(store, base_summary_id):
            _submit_stage(
                store,
                experiment,
                job_id=prefix_ready_id,
                kind="matrix-prefix-inputs",
                action="prepare-prefix-inputs",
                dependencies=(base_summary_id,),
                scope=scope,
            )
        if _job_succeeded(store, prefix_ready_id):
            _submit_prefix_component(experiment, store, scope=scope)
        if _job_succeeded(store, prefix_complete_id):
            _submit_selector_component(experiment, store, scope=scope)
        if _job_succeeded(store, selected_noise_id):
            _submit_ind_component(experiment, store, scope=scope)
        if _job_succeeded(store, ind_summary_id):
            _submit_coverage(experiment, store, scope=scope)


def _resolve_job_reservation(
    experiment: FullMatrixExperiment,
    job: QueueJob,
    *,
    base: Any | None,
    scoped_bases: Mapping[str, Any] | None = None,
    device_total_bytes: int,
) -> int | None:
    if job.kind in CPU_KINDS or job.gpu_id == -1:
        return 0
    component = _component_for(job)
    if component == "base":
        scope_id = _scope_id_from_job(job)
        selected_base = (scoped_bases or {}).get(scope_id) if scope_id is not None else base
        if selected_base is None:
            return None
        parts = job.job_id.split(":")
        worker_job_id = ":".join(parts[3:]) if scope_id is not None else ":".join(parts[1:])
        return _resolve_reservation(
            selected_base,
            replace(job, job_id=worker_job_id),
            device_total_bytes=device_total_bytes,
        )
    if job.reservation_bytes is not None:
        return int(job.reservation_bytes)
    return None


def _measured_reference_training(job: QueueJob) -> bool:
    """Reference training with a measured reservation may share a device.

    The frozen 44 GiB declaration is a conservative placeholder that must run
    alone on a physically idle device.  Once a pending row is re-declared from
    NVML measurements below the strict threshold, reservation accounting is
    truthful enough to pack it beside other work, exactly like a measured
    ``source-method`` estimate.
    """

    return (
        job.kind == "matrix-reference-training"
        and 0 < (job.reservation_bytes or 0) <= _HIGH_RESERVATION_STRICT_THRESHOLD_BYTES
    )


def _requires_exclusive_matrix_gpu(job: QueueJob) -> bool:
    """Keep fixed full-memory jobs exclusive as well as profile-limited jobs."""

    # IND training uses the same fixed 44 GiB declaration as the static
    # reference jobs.  It must remain one-per-device after its declaration is
    # clamped to live capacity; otherwise two partially observed trainers could
    # be admitted before either allocator reaches its peak.
    high_source_method = (
        job.kind == "source-method"
        and (job.reservation_bytes or 0) > _HIGH_RESERVATION_STRICT_THRESHOLD_BYTES
    )
    return (
        (job.kind in GPU_JOB_KINDS and not _measured_reference_training(job))
        or job.kind == "training"
        or high_source_method
        or _requires_exclusive_gpu(job)
    )


def _requires_strict_matrix_gpu(job: QueueJob) -> bool:
    """Require a physically idle device for fixed full-memory GPU jobs."""

    high_source_method = (
        job.kind == "source-method"
        and (job.reservation_bytes or 0) > _HIGH_RESERVATION_STRICT_THRESHOLD_BYTES
    )
    return (
        (job.kind in GPU_JOB_KINDS and not _measured_reference_training(job))
        or job.kind == "training"
        or high_source_method
    )


def _effective_matrix_reservation(
    job: QueueJob,
    *,
    reservation_bytes: int,
    device_total_bytes: int,
    headroom_bytes: int,
) -> int | None:
    """Cap fixed full-matrix reservations at physical capacity after headroom.

    Static Phase-0 jobs and IND ``training`` jobs use a frozen 44 GiB
    declaration as a conservative upper bound. Some L40S boards report about
    45 GiB total, so that declaration cannot coexist with the separately
    frozen 5% admission headroom. A small number of profile-derived
    ``source-method`` estimates are also above that usable capacity. The queue
    specification remains unchanged while each launch is capped at the device
    ceiling.

    The cap is a static, occupancy-independent ceiling rather than the
    currently free portion of the device. Clamping against live free memory
    would shrink a reservation to the crumbs left by busy devices, and the
    outstanding-capacity accounting (``max(0, reservation - observed)``) then
    collapses once the job grows past its shrunken budget — further jobs keep
    being admitted until the device runs out of memory. A static ceiling
    keeps the effective reservation truthful: it either blocks co-admission
    outright or leaves the declared estimate untouched.

    Measured Phase 1/2 and downstream reservations are intentionally left
    untouched: those values are already below the physical admission limit.
    """

    if job.kind not in _CAPACITY_CLAMPED_GPU_JOB_KINDS:
        return reservation_bytes
    capacity = device_total_bytes - headroom_bytes
    if capacity <= 0:
        return None
    return min(reservation_bytes, capacity)


def _launch_cpu(
    store: SimpleJobStore,
    job: QueueJob,
    *,
    signal_directory: Path,
    environment: Mapping[str, str],
) -> RunningProcess:
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    child_environment = dict(os.environ)
    child_environment.update(environment)
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    child_environment["PYTHONUNBUFFERED"] = "1"
    signal_directory.mkdir(parents=True, exist_ok=True)
    release_token = uuid.uuid4().hex
    release_marker = signal_directory / f"{release_token}.json"
    child_environment[GPU_RELEASE_PATH_ENV] = str(release_marker)
    child_environment[GPU_RELEASE_TOKEN_ENV] = release_token
    child_environment[GPU_RELEASE_JOB_ENV] = job.job_id
    try:
        process = subprocess.Popen(
            job.command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=child_environment,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        release_marker.unlink(missing_ok=True)
        raise
    store.start(job.job_id, pid=process.pid, gpu_id=-1)
    refreshed = next(item for item in store.jobs(status="running") if item.job_id == job.job_id)
    return RunningProcess(
        refreshed,
        process,
        log_handle,
        0,
        release_marker,
        release_token,
    )


def _priority(
    job: QueueJob,
    *,
    scope_prerequisites: frozenset[str] = frozenset(),
    scopes: Mapping[str, ExecutionScope] | None = None,
) -> tuple[int, ...] | tuple[int, int, str]:
    """Prioritize a frozen partial scope ahead of unrelated Phase-0 work."""

    if scope_prerequisites or scopes:
        if job.job_id in scope_prerequisites:
            bucket = 0
        elif _scope_id_from_job(job) in (scopes or {}):
            bucket = 1
        else:
            bucket = 2
        return (
            bucket,
            _STAGE_ORDER.get(job.kind, 99),
            -int(job.reservation_bytes or 0),
            job.job_id,
        )
    return (
        _STAGE_ORDER.get(job.kind, 99),
        -int(job.reservation_bytes or 0),
        job.job_id,
    )


def _signal_directory(experiment: FullMatrixExperiment) -> Path:
    return experiment.storage.run_root / "signals"


def deferred_datasets_path(experiment: FullMatrixExperiment) -> Path:
    """Return the execution-only dataset deferral control path.

    This file deliberately lives beside run state rather than in the frozen
    configuration.  It changes claim eligibility only; it never changes a
    queued job, its ID, or the scheduler identity.
    """

    return experiment.storage.run_root / "control" / DEFERRED_DATASETS_FILENAME


def _deferred_datasets_payload(
    experiment: FullMatrixExperiment,
    *,
    datasets: Sequence[str],
    reason: str,
) -> Mapping[str, Any]:
    normalized = tuple(sorted(set(str(dataset) for dataset in datasets)))
    unknown = set(normalized) - set(experiment.dataset_ids)
    if unknown:
        raise ValueError(f"Unknown full-matrix datasets for deferral: {sorted(unknown)}")
    if not reason.strip():
        raise ValueError("Dataset deferral reason must be non-empty")
    return {
        "schema": DEFERRED_DATASETS_SCHEMA,
        "experiment_id": experiment.experiment_id,
        "scheduler_digest": experiment.scheduler_digest,
        "datasets": list(normalized),
        "reason": reason.strip(),
    }


def load_deferred_datasets(experiment: FullMatrixExperiment) -> frozenset[str]:
    """Load a fail-closed execution-only dataset claim filter.

    A malformed, foreign, or stale control file must not silently resume work
    under an ambiguous operator instruction.  An absent file means no dataset
    is deferred.
    """

    path = deferred_datasets_path(experiment)
    if not path.is_file():
        return frozenset()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Dataset deferral control is unreadable: {path}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Dataset deferral control is malformed: {path}")
    expected = {
        "schema": DEFERRED_DATASETS_SCHEMA,
        "experiment_id": experiment.experiment_id,
        "scheduler_digest": experiment.scheduler_digest,
    }
    contradictions = {
        key: (value.get(key), item) for key, item in expected.items() if value.get(key) != item
    }
    if contradictions:
        raise RuntimeError(f"Dataset deferral control identity is contradictory: {contradictions}")
    datasets = value.get("datasets")
    reason = value.get("reason")
    if (
        not isinstance(datasets, list)
        or any(not isinstance(dataset, str) for dataset in datasets)
        or datasets != sorted(set(datasets))
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise RuntimeError(f"Dataset deferral control is malformed: {path}")
    unknown = set(datasets) - set(experiment.dataset_ids)
    if unknown:
        raise RuntimeError(f"Dataset deferral control names unknown datasets: {sorted(unknown)}")
    return frozenset(datasets)


def write_deferred_datasets(
    experiment: FullMatrixExperiment,
    *,
    datasets: Sequence[str],
    reason: str,
) -> Mapping[str, Any]:
    """Atomically install or replace the runtime dataset deferral control."""

    from xai_ensemble.core.io import atomic_write_json

    payload = _deferred_datasets_payload(experiment, datasets=datasets, reason=reason)
    path = deferred_datasets_path(experiment)
    atomic_write_json(path, payload)
    return {**payload, "path": str(path)}


def clear_deferred_datasets(experiment: FullMatrixExperiment) -> Mapping[str, Any]:
    """Remove a validated runtime deferral control without touching SQLite."""

    path = deferred_datasets_path(experiment)
    deferred = sorted(load_deferred_datasets(experiment))
    if path.exists():
        path.unlink()
    return {
        "schema": DEFERRED_DATASETS_SCHEMA,
        "experiment_id": experiment.experiment_id,
        "scheduler_digest": experiment.scheduler_digest,
        "cleared_datasets": deferred,
        "path": str(path),
    }


def _job_dataset_id(job: QueueJob) -> str | None:
    """Extract a full-matrix dataset identity from a stable queued command."""

    command = job.command
    for index, value in enumerate(command[:-1]):
        if value == "--dataset":
            return command[index + 1]
        if value == "--cell":
            return command[index + 1].split("--", 1)[0]
        if value == "--task-id":
            task_id = command[index + 1]
            return task_id.split("--", 1)[0] if "--" in task_id else None
    # The complete matrix's own static job IDs retain the canonical dataset
    # before the first delimiter.  Dynamic component jobs carry --task-id.
    if ":" in job.job_id:
        task_id = job.job_id.split(":", 1)[1]
        if "--" in task_id:
            return task_id.split("--", 1)[0]
    return None


def _job_is_deferred(job: QueueJob, deferred_datasets: frozenset[str]) -> bool:
    if _job_dataset_id(job) in deferred_datasets:
        return True
    # Partition and shared-component IDs may encode the dataset in a colon
    # segment even when their argv intentionally carries only a manifest path.
    tokens = job.job_id.replace(":", "--").split("--")
    return any(
        token == dataset or token.startswith(f"{dataset}--")
        for dataset in deferred_datasets
        for token in tokens
    )


def _live_external_gpu_ids(
    store: SimpleJobStore,
    running: Mapping[str, RunningProcess],
) -> frozenset[int]:
    """Return GPUs held by live queue workers owned by an older scheduler.

    Normal orphan recovery leaves a live lease untouched.  A replacement
    scheduler therefore has no local ``RunningProcess`` object for it, but it
    must still exclude that GPU until the old worker exits.  This is especially
    important when a paused scheduler is replaced while a reference training
    process continues in its own session.
    """

    local_ids = frozenset(running)
    occupied: set[int] = set()
    for job in store.jobs(status="running"):
        if job.job_id in local_ids or job.pid is None or job.gpu_id is None or job.gpu_id < 0:
            continue
        try:
            os.kill(job.pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            occupied.add(job.gpu_id)
            continue
        if _process_is_zombie(job.pid):
            continue
        occupied.add(job.gpu_id)
    return frozenset(occupied)


def _worker_environment(experiment: FullMatrixExperiment) -> Mapping[str, str]:
    """Bind coherent HF cache roots and verified local recovery sources."""

    from xai_ensemble.data.hf_parquet_recovery import recovery_environment
    from xai_ensemble.data.places365_recovery import places365_recovery_environment

    configured_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hf_home = Path(configured_home).expanduser().resolve()
    configured_hub = os.environ.get("HF_HUB_CACHE")
    expected_hub = hf_home / "hub"
    if configured_hub is not None and Path(configured_hub).expanduser().resolve() != expected_hub:
        raise RuntimeError(
            "HF_HUB_CACHE must equal HF_HOME/hub for the full-matrix Places365 recovery "
            f"(HF_HOME={hf_home}, HF_HUB_CACHE={configured_hub})"
        )
    environment = {
        "XAI_CLOUD_STORAGE_LOCK_ROOT": str(experiment.storage.cloudstorage_lock_root),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(expected_hub),
        # Execution-only storage policy: keep full-matrix inputs in a verified,
        # bounded batch cache instead of materializing an entire split in tmpfs.
        # These variables are deliberately outside the frozen config/digest.
        "XAI_SIMPLE_STREAMING_INPUTS": "1",
        "XAI_SIMPLE_HOT_CACHE_GIB": "12",
        "XAI_SIMPLE_HOT_CACHE_ROOT": str(
            experiment.storage.selector_cache_root.parent / f"{experiment.experiment_id}-hot-cache"
        ),
        # Keep enough completed shards resident to cover temporary Drive
        # publication stalls without letting the bounded tmpfs spool consume
        # the 40 GiB physical free-space floor.
        "XAI_SIMPLE_SPOOL_MAX_GIB": "48",
        "XAI_SIMPLE_SPOOL_MIN_FREE_GIB": "40",
        # Each Phase-1 worker retains its default two publisher threads.  Up
        # to four released/active workers can therefore use all eight remote
        # publication slots without one worker creating unbounded contention.
        "XAI_SIMPLE_PHASE1_UPLOAD_GLOBAL_LIMIT": "8",
        "XAI_SIMPLE_PHASE1_STAGE_WORKERS": "2",
        "XAI_SIMPLE_PHASE1_PREFETCH_MAX_GIB": "64",
        "XAI_SIMPLE_PHASE1_PREFETCH_MIN_FREE_GIB": "40",
    }
    overlay_asset = os.environ.get(FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV)
    overlay_cache = os.environ.get(FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV)
    if bool(overlay_asset) != bool(overlay_cache):
        raise RuntimeError("full-matrix asset and cache overlays must be configured together")
    if overlay_asset is not None and overlay_cache is not None:
        environment.update(
            {
                FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV: str(Path(overlay_asset).expanduser().resolve()),
                FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV: str(Path(overlay_cache).expanduser().resolve()),
                FULL_MATRIX_LOGICAL_ASSET_ROOT_ENV: str(experiment.storage.asset_root),
                FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV: str(experiment.storage.cache_root),
                FULL_MATRIX_OVERLAY_DATASETS_ENV: os.environ.get(
                    FULL_MATRIX_OVERLAY_DATASETS_ENV,
                    "bloodmnist,breastmnist,dermamnist,food101,imagenet1k,octmnist,"
                    "organamnist,organcmnist,organsmnist,pathmnist,pneumoniamnist,"
                    "retinamnist,tissuemnist",
                ),
            }
        )
    environment.update(
        recovery_environment(
            hub_root=hf_home / "hub",
            marker_root=experiment.storage.run_root / "hf-parquet-recovery",
        )
    )
    environment.update(
        places365_recovery_environment(
            hf_cache_root=hf_home,
            marker_root=experiment.storage.run_root / "places365-recovery",
        )
    )
    return environment


def _preflight_places365_recovery(experiment: FullMatrixExperiment) -> Mapping[str, str]:
    """Fail before queue access unless all direct-Parquet markers validate."""

    from xai_ensemble.data.hf_parquet_recovery import (
        preflight_hf_parquet_recovery_environment,
    )
    from xai_ensemble.data.places365_recovery import preflight_places365_recovery_environment

    environment = _worker_environment(experiment)
    preflight_hf_parquet_recovery_environment(environment)
    preflight_places365_recovery_environment(environment)
    return environment


def _preflight_overlay_medmnist_sources(environment: Mapping[str, str]) -> None:
    """Fail before queue access unless every overlaid MedMNIST source is sealed.

    Overlaid datasets never fall back to the mounted cache at runtime, so a
    missing local archive or marker (for example after a tmpfs wipe emptied
    the volatile dataset cache) must stop the scheduler at start instead of
    failing every worker that later claims the dataset.
    """

    overlay_cache = environment.get(FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV)
    if not overlay_cache or not environment.get(FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV):
        return
    raw = environment.get(FULL_MATRIX_OVERLAY_DATASETS_ENV, "")
    names = {item.strip() for item in raw.split(",") if item.strip()}
    if not names:
        return
    from xai_ensemble.data.medmnist_recovery import (
        has_verified_source_contract,
        require_verified_medmnist_source,
    )
    from xai_ensemble.data.specs import DATASET_REGISTRY

    problems: list[str] = []
    for spec in DATASET_REGISTRY.values():
        options = spec.provider_options
        flag = str(options.get("data_flag", ""))
        if spec.provider != "medmnist" or flag not in names:
            continue
        if not has_verified_source_contract(spec):
            continue
        try:
            require_verified_medmnist_source(spec, Path(overlay_cache) / flag)
        except (FileNotFoundError, ValueError) as error:
            problems.append(f"{flag}: {error}")
    if problems:
        raise RuntimeError("Unsealed overlaid MedMNIST sources: " + "; ".join(sorted(problems)))


def _capacity_floor_failure(experiment: FullMatrixExperiment) -> str | None:
    """Return a durable-stop reason before local capacity is exhausted."""

    local_free = shutil.disk_usage(experiment.storage.run_root).free
    if local_free <= LOCAL_DISK_FLOOR_BYTES:
        return (
            f"local_disk_below_floor free_bytes={local_free} floor_bytes={LOCAL_DISK_FLOOR_BYTES}"
        )
    # The selector creates its private directory lazily. Probe an existing
    # ancestor so the scheduler can enforce the tmpfs floor before that stage.
    shm_path = experiment.storage.selector_cache_root
    while not shm_path.exists():
        parent = shm_path.parent
        if parent == shm_path:
            break
        shm_path = parent
    shm_free = shutil.disk_usage(shm_path).free
    if shm_free <= SHM_FLOOR_BYTES:
        return f"shm_below_floor free_bytes={shm_free} floor_bytes={SHM_FLOOR_BYTES}"
    return None


def _capacity_action(capacity_failure: str | None, *, running: bool) -> str:
    """Choose whether to proceed, pause admission, or stop durably."""

    if capacity_failure is None:
        return "proceed"
    return "pause" if running else "stop"


def _process_group_exists(pgid: int) -> bool:
    """Return whether a scheduler-owned process group still has a member."""

    if pgid <= 0:
        raise ValueError("scheduler-owned process group id must be positive")
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # This should not occur for scheduler-owned children, but it means the
        # group cannot safely be treated as gone.
        return True
    return True


def _signal_process_group(pgid: int, sig: signal.Signals) -> tuple[str, str | None]:
    """Signal one validated worker group and report a non-raising outcome."""

    if pgid <= 0:
        raise ValueError("scheduler-owned process group id must be positive")
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return "already_exited", None
    except OSError as error:
        return "failed", str(error)
    return "sent", None


def _record_capacity_event(
    store: SimpleJobStore,
    job_id: str,
    event: str,
    detail: str | None = None,
) -> None:
    """Record stop telemetry without allowing a degraded queue to block cleanup."""

    try:
        store.record_event(job_id, event, detail)
    except Exception:
        # Capacity recovery must take precedence when the queue itself has
        # become unable to accept another telemetry write.
        return


def _capacity_signal_detail(
    *,
    pid: int,
    outcome: str,
    error: str | None,
) -> tuple[str, str]:
    """Turn a non-raising process-group signal result into queue telemetry."""

    detail = f"pid={pid}"
    if outcome == "sent":
        return "", detail
    if outcome == "already_exited":
        return "", f"{detail},already_exited"
    if outcome == "failed":
        return "_failed", f"{detail},error={error}"
    return "_failed", f"{detail},error=unknown_outcome={outcome}"


def _wait_for_process_groups(
    active_processes: Sequence[RunningProcess],
    *,
    deadline: float,
) -> tuple[RunningProcess, ...]:
    """Poll worker roots while their independently launched groups remain alive."""

    survivors = tuple(
        active for active in active_processes if _process_group_exists(active.process.pid)
    )
    while survivors and time.monotonic() < deadline:
        for active in survivors:
            active.process.poll()
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        survivors = tuple(
            active for active in survivors if _process_group_exists(active.process.pid)
        )
    return survivors


def _stop_running_for_capacity(
    store: SimpleJobStore,
    running: Mapping[str, RunningProcess],
    *,
    reason: str,
    grace_seconds: float = _CAPACITY_STOP_GRACE_SECONDS,
) -> tuple[str, ...]:
    """Stop scheduler-owned worker groups without rewriting queue state.

    Every worker is launched in a separate session, so signalling its process
    group also reaches loader and publisher descendants.  The running SQLite
    rows are deliberately retained: the next single scheduler instance uses
    normal orphan recovery to reopen the same job IDs.
    """

    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")
    active_processes = tuple(running.values())
    # Validate every target before any signal.  In particular, killpg(0, ...)
    # would target this scheduler's own group rather than a worker.
    for active in active_processes:
        if active.process.pid <= 0:
            raise ValueError("scheduler-owned process group id must be positive")

    survivors = active_processes
    term_results: dict[str, tuple[str, str | None]] = {}
    kill_results: dict[str, tuple[str, str | None]] = {}
    unexpected_error: BaseException | None = None
    try:
        # Do the entire signal sequence before telemetry.  At the capacity
        # floor, SQLite may itself be slow or unable to accept event writes.
        for active in active_processes:
            term_results[active.job.job_id] = _signal_process_group(
                active.process.pid,
                signal.SIGTERM,
            )
        survivors = _wait_for_process_groups(
            active_processes,
            deadline=time.monotonic() + grace_seconds,
        )
    except BaseException as error:
        unexpected_error = error
    finally:
        # If any step above failed unexpectedly, ``survivors`` is still all
        # targets, so no owned group is left behind because of bookkeeping.
        try:
            for active in survivors:
                kill_results[active.job.job_id] = _signal_process_group(
                    active.process.pid,
                    signal.SIGKILL,
                )
            survivors = _wait_for_process_groups(
                survivors,
                deadline=time.monotonic() + _CAPACITY_STOP_KILL_WAIT_SECONDS,
            )
        finally:
            for active in active_processes:
                try:
                    active.process.poll()
                except OSError:
                    pass
                try:
                    active.log_handle.close()
                except OSError:
                    pass
                try:
                    active.release_marker.unlink(missing_ok=True)
                except OSError:
                    pass

    unreaped = tuple(active.job.job_id for active in survivors)
    for active in active_processes:
        _record_capacity_event(
            store,
            active.job.job_id,
            "scheduler_capacity_stop",
            reason,
        )
        outcome, error = term_results.get(active.job.job_id, ("failed", "not_attempted"))
        suffix, detail = _capacity_signal_detail(
            pid=active.process.pid,
            outcome=outcome,
            error=error,
        )
        event = f"scheduler_capacity_stop_sigterm{suffix}"
        _record_capacity_event(store, active.job.job_id, event, detail)
    for active in active_processes:
        result = kill_results.get(active.job.job_id)
        if result is None:
            continue
        outcome, error = result
        suffix, detail = _capacity_signal_detail(
            pid=active.process.pid,
            outcome=outcome,
            error=error,
        )
        event = f"scheduler_capacity_stop_sigkill{suffix}"
        _record_capacity_event(store, active.job.job_id, event, detail)
    for job_id in unreaped:
        _record_capacity_event(store, job_id, "scheduler_capacity_stop_unreaped", reason)
    if unexpected_error is not None:
        raise unexpected_error
    return unreaped


def _status_counts(store: SimpleJobStore) -> Mapping[str, int]:
    return {status: sum(job.status == status for job in store.jobs()) for status in STATUSES}


def _scheduler_worker_environment(
    experiment: FullMatrixExperiment,
    *,
    deferred_datasets: frozenset[str],
) -> Mapping[str, str]:
    """Validate the active direct sources before opening the queue."""

    environment = _worker_environment(experiment)
    _preflight_overlay_medmnist_sources(environment)
    if "places365" in deferred_datasets:
        from xai_ensemble.data.hf_parquet_recovery import (
            preflight_hf_parquet_recovery_environment,
        )

        preflight_hf_parquet_recovery_environment(environment)
        return environment
    return _preflight_places365_recovery(experiment)


def run_scheduler(
    experiment: FullMatrixExperiment,
    *,
    poll_seconds: float = 2.0,
) -> Mapping[str, int]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    from xai_ensemble.simple.runtime_dependencies import require_relprop_runtime

    relprop = require_relprop_runtime(
        (method.family, architecture)
        for architecture in ("cnn", "vit")
        for method in experiment.methods.for_architecture(architecture)
    )
    if relprop["required"]:
        print(
            "RELPROP_RUNTIME_READY "
            f"revision={relprop['revision']} source_digest={relprop['source_digest']} "
            f"repository={relprop['repository']}",
            flush=True,
        )
    # This intentionally precedes SimpleJobStore construction and orphan
    # recovery: a missing/invalid Places365 marker must never mutate the
    # authoritative queue as a side effect of a failed scheduler start. When
    # Places365 is deferred, no new Places365 worker can be claimed, so its
    # direct-Parquet preflight is deliberately skipped as well.
    deferred_at_start = load_deferred_datasets(experiment)
    worker_environment = _scheduler_worker_environment(
        experiment,
        deferred_datasets=deferred_at_start,
    )
    # Completion probes and workers must resolve the same execution overlay.
    # These variables are runtime-only and are intentionally absent from the
    # immutable experiment and scheduler digests.
    os.environ.update(worker_environment)
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    store.recover_orphans()
    if not store.jobs():
        _submit_initial_plan(experiment, store)
    elif not any(job.job_id == MATERIALIZE_JOB_ID for job in store.jobs()):
        _submit_initial_plan(experiment, store)

    from xai_ensemble.simple.config import load_experiment

    running: dict[str, RunningProcess] = {}
    probe = NvidiaSmiProbe(cache_seconds=0.1)
    gpu_probe_error: GpuProbeError | None = None
    last_heartbeat = 0.0
    while True:
        for job_id, active in tuple(running.items()):
            exit_code = active.process.poll()
            if exit_code is None:
                if not active.gpu_released and gpu_release_signal_matches(
                    active.release_marker,
                    job_id=active.job.job_id,
                    pid=active.process.pid,
                    token=active.release_token,
                ):
                    active.gpu_released = True
                    store.record_event(
                        job_id,
                        "gpu_released",
                        f"gpu={active.job.gpu_id},admission=awaiting_nvml",
                    )
                continue
            active.log_handle.close()
            active.release_marker.unlink(missing_ok=True)
            store.finish(job_id, exit_code=exit_code)
            del running[job_id]

        store.recover_orphans()
        now = time.monotonic()
        if now - last_heartbeat >= 30.0:
            for job_id in running:
                store.heartbeat(job_id)
            last_heartbeat = now

        store.block_failed_dependents()
        deferred_datasets = load_deferred_datasets(experiment)
        scopes = _ensure_deferred_ready_scope(
            experiment,
            store,
            deferred_datasets=deferred_datasets,
        )
        _extend_plan(experiment, store, scopes=scopes)
        jobs = store.jobs()
        if not any(job.status in {"pending", "running"} for job in jobs):
            break
        capacity_failure = _capacity_floor_failure(experiment)
        capacity_action = _capacity_action(capacity_failure, running=bool(running))
        if capacity_action == "pause":
            # A running worker may already be draining its publication spool.
            # Stopping it at the floor leaves its staged shards behind and can
            # turn a temporary pressure event into a repeated orphan cycle.
            # Pause admission while owned workers finish; their publisher
            # threads can release space and let the next iteration proceed.
            assert capacity_failure is not None
            print(f"CAPACITY_PAUSED {capacity_failure}", flush=True)
            time.sleep(poll_seconds)
            continue
        if capacity_action == "stop":
            assert capacity_failure is not None
            raise RuntimeError(f"Controlled full-matrix scheduler stop: {capacity_failure}")

        try:
            _refresh_gpu_release_admission(store, running, probe)
        except GpuProbeError as error:
            if gpu_probe_error is None:
                print(
                    "WARNING GPU telemetry unavailable "
                    f"({error}); running serially without GPU admission control",
                    flush=True,
                )
            gpu_probe_error = error

        base = None
        if _component_job_ids(store, "base") and experiment.base_config_path.is_file():
            base = load_experiment(experiment.base_config_path)
        scoped_bases = {
            scope.scope_id: load_experiment(execution_paths(experiment, scope).base_config_path)
            for scope in scopes.values()
            if _component_job_ids(store, "base", scope=scope)
            and execution_paths(experiment, scope).base_config_path.is_file()
        }
        try:
            snapshot = probe.snapshot(force=True)
            configured = {gpu_id: snapshot.device(gpu_id) for gpu_id in experiment.runtime.gpu_ids}
            gpu_probe_error = None
        except GpuProbeError as error:
            if gpu_probe_error is None:
                print(
                    "WARNING GPU telemetry unavailable "
                    f"({error}); running serially without GPU admission control",
                    flush=True,
                )
            gpu_probe_error = error
            configured = {}
        external_gpu_ids = _live_external_gpu_ids(store, running)
        scope_prerequisites = _scope_compatibility_prerequisites(
            experiment,
            store,
            deferred_datasets=deferred_datasets,
            scopes=scopes,
        )
        ready = sorted(
            (job for job in _ready_jobs(store) if not _job_is_deferred(job, deferred_datasets)),
            key=lambda job: _priority(
                job,
                scope_prerequisites=scope_prerequisites,
                scopes=scopes,
            ),
        )
        launched = False
        for job in ready:
            if job.job_id in running:
                continue
            if job.kind in CPU_KINDS or job.job_id.startswith("ind:partition:"):
                cpu_running = sum(item.job.gpu_id == -1 for item in running.values())
                if cpu_running >= experiment.runtime.max_cpu_jobs:
                    continue
                running[job.job_id] = _launch_cpu(
                    store,
                    job,
                    signal_directory=_signal_directory(experiment),
                    environment=worker_environment,
                )
                print(f"SCHEDULED job={job.job_id} cpu=1", flush=True)
                launched = True
                continue
            if job.kind not in GPU_KINDS and not _component_for(job):
                raise RuntimeError(f"Unknown full-matrix job kind: {job.kind}")
            if not configured:
                # Degraded mode: one serial worker on the CPU, no GPU
                # admission control.
                if running:
                    continue
                running[job.job_id] = _launch(
                    store,
                    job,
                    gpu_id=-1,
                    reservation_bytes=job.reservation_bytes or 0,
                    signal_directory=_signal_directory(experiment),
                    environment=worker_environment,
                )
                print(
                    f"SCHEDULED job={job.job_id} gpu=-1 mode=cpu-serial-fallback",
                    flush=True,
                )
                launched = True
                continue
            reservation = _resolve_job_reservation(
                experiment,
                job,
                base=base,
                scoped_bases=scoped_bases,
                device_total_bytes=min(device.total_bytes for device in configured.values()),
            )
            if reservation is None:
                continue
            if _component_for(job) == "base" and job.reservation_bytes != reservation:
                store.update_reservation(job.job_id, reservation)
                job = replace(job, reservation_bytes=reservation)
            snapshot = probe.snapshot(force=True)
            candidates = []
            for gpu_id, device in configured.items():
                if gpu_id in external_gpu_ids:
                    continue
                active = [item for item in running.values() if item.job.gpu_id == gpu_id]
                if _gpu_release_blocks_admission(active):
                    continue
                holders = [item for item in active if not item.gpu_admission_released]
                if _requires_strict_matrix_gpu(job) and active:
                    continue
                if _requires_exclusive_matrix_gpu(job) and holders:
                    continue
                if any(_requires_strict_matrix_gpu(item.job) for item in active):
                    continue
                if any(_requires_exclusive_matrix_gpu(item.job) for item in holders):
                    continue
                if _requires_strict_matrix_gpu(job) and any(
                    process.gpu_uuid == device.uuid for process in snapshot.processes
                ):
                    continue
                observed = 0
                outstanding = 0
                for item in active:
                    memory = probe.process_memory_bytes(item.process.pid, (gpu_id,)).get(gpu_id, 0)
                    observed += memory
                    outstanding += _outstanding_reservation(
                        reservation_bytes=item.reservation_bytes,
                        observed_process_bytes=memory,
                        gpu_admission_released=item.gpu_admission_released,
                    )
                headroom = int(device.total_bytes * experiment.runtime.headroom_fraction)
                live_free = snapshot.device(gpu_id).free_bytes
                effective_reservation = _effective_matrix_reservation(
                    job,
                    reservation_bytes=reservation,
                    device_total_bytes=device.total_bytes,
                    headroom_bytes=headroom,
                )
                if effective_reservation is None:
                    continue
                # A clamped reservation already sits at the static ceiling
                # (device total minus headroom); subtracting the headroom a
                # second time in the fits check would reject the job even on a
                # physically idle device, because the driver always occupies a
                # few hundred MiB below the reported total.
                clamped_to_ceiling = effective_reservation < reservation
                if not _reservation_fits(
                    job_kind=job.kind,
                    reservation_bytes=effective_reservation,
                    live_free_bytes=live_free,
                    outstanding_bytes=outstanding,
                    headroom_bytes=0 if clamped_to_ceiling else headroom,
                ):
                    continue
                available = live_free - outstanding - headroom
                candidates.append(
                    (
                        available - effective_reservation,
                        gpu_id,
                        effective_reservation,
                        live_free,
                        headroom,
                        outstanding,
                    )
                )
            if not candidates:
                continue
            (
                _,
                gpu_id,
                effective_reservation,
                live_free,
                headroom,
                outstanding,
            ) = min(candidates)
            if effective_reservation < reservation:
                store.record_event(
                    job.job_id,
                    "reservation_clamped_for_headroom",
                    json.dumps(
                        {
                            "gpu_id": gpu_id,
                            "headroom_bytes": headroom,
                            "live_free_bytes": live_free,
                            "outstanding_bytes": outstanding,
                            "requested_bytes": reservation,
                            "effective_bytes": effective_reservation,
                        },
                        sort_keys=True,
                    ),
                )
            running[job.job_id] = _launch(
                store,
                job,
                gpu_id=gpu_id,
                reservation_bytes=effective_reservation,
                signal_directory=_signal_directory(experiment),
                environment=worker_environment,
            )
            print(
                f"SCHEDULED job={job.job_id} gpu={gpu_id} "
                f"reservation_gib={effective_reservation / 2**30:.2f} "
                f"requested_gib={reservation / 2**30:.2f}",
                flush=True,
            )
            launched = True
        if not launched:
            time.sleep(poll_seconds)
    return _status_counts(store)


def scheduler_status(experiment: FullMatrixExperiment) -> Mapping[str, Any]:
    if not experiment.runtime.database_path.is_file():
        return {
            "schema": MATRIX_SCHEMA,
            "database": str(experiment.runtime.database_path),
            "exists": False,
            "counts": {},
        }
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    jobs = store.jobs()
    deferred_datasets = sorted(load_deferred_datasets(experiment))
    scopes = _scopes_from_store(experiment, store)
    return {
        "schema": MATRIX_SCHEMA,
        "database": str(experiment.runtime.database_path),
        "exists": True,
        "planned_job_count": len(jobs),
        "counts": _status_counts(store),
        "deferred_datasets": deferred_datasets,
        "deferred_pending_count": sum(
            job.status == "pending" and _job_is_deferred(job, frozenset(deferred_datasets))
            for job in jobs
        ),
        "execution_scopes": [
            {
                "scope_id": scope.scope_id,
                "scope_digest": scope.scope_digest,
                "deferred_datasets": list(scope.deferred_datasets),
                "planned_cells": len(scope.cell_ids),
                "materialize_job_id": _scope_stage_id(scope, "materialize"),
                "materialize_status": next(
                    job.status
                    for job in jobs
                    if job.job_id == _scope_stage_id(scope, "materialize")
                ),
            }
            for scope in sorted(scopes.values(), key=lambda item: item.scope_id)
        ],
        "by_kind": {
            kind: {
                status: sum(job.kind == kind and job.status == status for job in jobs)
                for status in STATUSES
            }
            for kind in sorted({job.kind for job in jobs})
        },
        "running": [
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "pid": job.pid,
                "gpu_id": job.gpu_id,
                "log_path": job.log_path,
            }
            for job in jobs
            if job.status == "running"
        ],
        "failed_or_blocked": [job.job_id for job in jobs if job.status in {"failed", "blocked"}],
    }


def retry_failed(
    experiment: FullMatrixExperiment,
    job_ids: Sequence[str],
) -> Mapping[str, Any]:
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    store.recover_orphans()
    result = dict(store.retry_failed(tuple(job_ids)))
    result["status"] = scheduler_status(experiment)
    return result


def requeue_succeeded(
    experiment: FullMatrixExperiment,
    job_ids: Sequence[str],
    *,
    reason: str,
) -> Mapping[str, Any]:
    store = SimpleJobStore(
        experiment.runtime.database_path,
        experiment_digest=experiment.scheduler_digest,
    )
    store.recover_orphans()
    result = dict(store.requeue_jobs(tuple(job_ids), reason=reason))
    result["status"] = scheduler_status(experiment)
    return result


__all__ = [
    "COVERAGE_JOB_ID",
    "DEFERRED_DATASETS_FILENAME",
    "DEFERRED_DATASETS_SCHEMA",
    "IND_SUMMARY_JOB_ID",
    "MATERIALIZE_JOB_ID",
    "MATRIX_SCHEMA",
    "PREFIX_COMPLETE_JOB_ID",
    "PREFIX_READY_JOB_ID",
    "PREFIX_SUMMARY_JOB_ID",
    "SELECTED_NOISE_SUMMARY_JOB_ID",
    "SELECTOR_MERGE_JOB_ID",
    "clear_deferred_datasets",
    "deferred_datasets_path",
    "_live_external_gpu_ids",
    "load_deferred_datasets",
    "requeue_succeeded",
    "run_scheduler",
    "retry_failed",
    "scheduler_status",
    "write_deferred_datasets",
]
