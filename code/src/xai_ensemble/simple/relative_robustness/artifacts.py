"""Immutable random-control artifacts for null-anchored relative robustness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..artifacts import ArtifactStore, completed_manifest
from ..assumptions.artifacts import (
    AssumptionShardPublisher,
    existing_shard_records,
    publish_manifest,
)
from .config import RandomControlTask, RelativeRobustnessExperiment

CONTROL_SCHEMA_VERSION = 1


def output_store(experiment: RelativeRobustnessExperiment) -> ArtifactStore:
    return ArtifactStore(experiment)  # type: ignore[arg-type]


def completed_control_manifest(
    experiment: RelativeRobustnessExperiment,
    task: RandomControlTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=CONTROL_SCHEMA_VERSION,
    )


def completed_control_shards(
    experiment: RelativeRobustnessExperiment,
    task: RandomControlTask,
    *,
    source_layout: Sequence[tuple[int, int, int]],
    store: ArtifactStore | None = None,
) -> dict[int, Mapping[str, Any]]:
    return existing_shard_records(
        store or output_store(experiment),
        task.artifact_root,
        task_digest=task.digest,
        schema_version=CONTROL_SCHEMA_VERSION,
        source_layout=source_layout,
    )


RandomControlShardPublisher = AssumptionShardPublisher


__all__ = [
    "CONTROL_SCHEMA_VERSION",
    "RandomControlShardPublisher",
    "completed_control_manifest",
    "completed_control_shards",
    "output_store",
    "publish_manifest",
]
