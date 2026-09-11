"""Immutable artifacts for the formal NOISE prefix sweep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..artifacts import ArtifactStore, completed_manifest
from ..assumptions.artifacts import (
    AssumptionShardPublisher,
    existing_shard_records,
    publish_manifest,
)
from .config import NoisePrefixExperiment, PrefixEvaluationTask

EVALUATION_SCHEMA_VERSION = 1


def output_store(experiment: NoisePrefixExperiment) -> ArtifactStore:
    return ArtifactStore(experiment)  # type: ignore[arg-type]


def completed_evaluation_manifest(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=EVALUATION_SCHEMA_VERSION,
    )


def completed_shards(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
    *,
    source_layout: Sequence[tuple[int, int, int]],
    store: ArtifactStore | None = None,
) -> dict[int, Mapping[str, Any]]:
    return existing_shard_records(
        store or output_store(experiment),
        task.artifact_root,
        task_digest=task.digest,
        schema_version=EVALUATION_SCHEMA_VERSION,
        source_layout=source_layout,
    )


PrefixShardPublisher = AssumptionShardPublisher


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "PrefixShardPublisher",
    "completed_evaluation_manifest",
    "completed_shards",
    "output_store",
    "publish_manifest",
]
