from __future__ import annotations

from collections.abc import Mapping

import pytest

from xai_ensemble.simple.scheduler import QueueJob

_TERMINAL_FAILURES = frozenset({"failed", "blocked"})


def _has_phase1_failure_ancestor(
    job_id: str,
    *,
    by_id: Mapping[str, QueueJob],
    visiting: set[str] | None = None,
) -> bool:
    visiting = set() if visiting is None else visiting
    if job_id in visiting:
        raise RuntimeError(f"Cycle in simple scheduler dependencies at {job_id}")
    visiting.add(job_id)
    try:
        job = by_id[job_id]
        for dependency_id in job.dependencies:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                continue
            if dependency.kind == "phase1" and dependency.status in _TERMINAL_FAILURES:
                return True
            if (
                dependency.kind == "phase2"
                and dependency.status == "blocked"
                and _has_phase1_failure_ancestor(
                    dependency_id,
                    by_id=by_id,
                    visiting=visiting,
                )
            ):
                return True
        return False
    finally:
        visiting.remove(job_id)


def _job(
    job_id: str,
    *,
    kind: str,
    status: str,
    dependencies: tuple[str, ...] = (),
) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        kind=kind,
        command=(),
        dependencies=dependencies,
        resource_ids=(),
        reservation_bytes=None,
        status=status,
        attempts=0,
        max_retries=3,
        pid=None,
        gpu_id=None,
        log_path="unused.log",
    )


def test_phase2_block_is_traced_to_phase1_failure() -> None:
    jobs = {
        "phase1:source": _job("phase1:source", kind="phase1", status="failed"),
        "phase2:clean": _job(
            "phase2:clean",
            kind="phase2",
            status="blocked",
            dependencies=("phase1:source",),
        ),
        "phase2:noise": _job(
            "phase2:noise",
            kind="phase2",
            status="blocked",
            dependencies=("phase2:clean",),
        ),
    }

    assert _has_phase1_failure_ancestor("phase2:clean", by_id=jobs)
    assert _has_phase1_failure_ancestor("phase2:noise", by_id=jobs)


def test_phase2_failure_is_not_mislabeled_as_phase1_skip() -> None:
    jobs = {
        "phase2:clean": _job("phase2:clean", kind="phase2", status="failed"),
        "phase2:noise": _job(
            "phase2:noise",
            kind="phase2",
            status="blocked",
            dependencies=("phase2:clean",),
        ),
    }

    assert not _has_phase1_failure_ancestor("phase2:noise", by_id=jobs)


def test_phase2_dependency_cycle_is_rejected() -> None:
    jobs = {
        "phase2:a": _job(
            "phase2:a",
            kind="phase2",
            status="blocked",
            dependencies=("phase2:b",),
        ),
        "phase2:b": _job(
            "phase2:b",
            kind="phase2",
            status="blocked",
            dependencies=("phase2:a",),
        ),
    }

    with pytest.raises(RuntimeError, match="Cycle"):
        _has_phase1_failure_ancestor("phase2:a", by_id=jobs)
