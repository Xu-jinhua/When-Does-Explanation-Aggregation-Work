"""Immutable artifacts for noise-consistent random subsets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..artifacts import ArtifactStore, completed_manifest
from ..assumptions.artifacts import (
    AssumptionShardPublisher,
    existing_shard_records,
    publish_manifest,
)
from .config import (
    NoiseSubsetEvaluationTask,
    NoiseSubsetExperiment,
    NoiseSubsetSelectionTask,
)

SELECTION_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1


def output_store(experiment: NoiseSubsetExperiment) -> ArtifactStore:
    return ArtifactStore(experiment)  # type: ignore[arg-type]


def completed_selection_manifest(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=experiment.artifact_schema_version,
    )


def completed_evaluation_manifest(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=experiment.artifact_schema_version,
    )


def completed_evaluation_shards(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetEvaluationTask,
    *,
    source_layout: Sequence[tuple[int, int, int]],
    store: ArtifactStore | None = None,
) -> dict[int, Mapping[str, Any]]:
    return existing_shard_records(
        store or output_store(experiment),
        task.artifact_root,
        task_digest=task.digest,
        schema_version=experiment.artifact_schema_version,
        source_layout=source_layout,
    )


NoiseSubsetShardPublisher = AssumptionShardPublisher


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "NoiseSubsetShardPublisher",
    "completed_evaluation_manifest",
    "completed_evaluation_shards",
    "completed_selection_manifest",
    "output_store",
    "publish_manifest",
]
