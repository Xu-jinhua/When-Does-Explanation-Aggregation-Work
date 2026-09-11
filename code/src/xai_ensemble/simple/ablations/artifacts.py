"""Immutable artifact helpers for rank construction and ablation evaluation."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json

from ..artifacts import ArtifactError, ArtifactStore, completed_manifest, write_phase2_shard
from .config import AblationExperiment, EvaluationSpec, RankConstructionTask

RANK_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1


def output_store(experiment: AblationExperiment) -> ArtifactStore:
    return ArtifactStore(experiment.generation_experiment())


def completed_rank_manifest(
    experiment: AblationExperiment,
    task: RankConstructionTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=RANK_SCHEMA_VERSION,
    )


def completed_evaluation_manifest(
    experiment: AblationExperiment,
    task: EvaluationSpec,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=EVALUATION_SCHEMA_VERSION,
    )


def shard_names(index: int) -> tuple[str, str]:
    stem = f"shard-{index:05d}"
    return f"shards/{stem}.safetensors", f"shards/{stem}.json"


def existing_shard_records(
    store: ArtifactStore,
    root: str,
    *,
    task_digest: str,
    schema_version: int,
    source_layout: Sequence[tuple[int, int, int]],
) -> dict[int, Mapping[str, Any]]:
    result = {}
    for shard_index, start, stop in source_layout:
        payload_name, record_name = shard_names(shard_index)
        relative_record = posixpath.join(root, record_name)
        if not store.exists(relative_record):
            continue
        record = store.read_json(relative_record)
        expected = {
            "schema_version": schema_version,
            "task_digest": task_digest,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ArtifactError(f"Contradictory ablation shard record: {relative_record}")
        payload = record.get("payload")
        expected_payload = posixpath.join(root, payload_name)
        if not isinstance(payload, Mapping) or payload.get("relative_path") != expected_payload:
            raise ArtifactError(f"Malformed ablation shard record: {relative_record}")
        if not store.exists(f"{expected_payload}.receipt.json"):
            raise ArtifactError(f"Unverified ablation shard payload: {expected_payload}")
        result[shard_index] = record
    return result


def publish_shard(
    experiment: AblationExperiment,
    store: ArtifactStore,
    *,
    root: str,
    task_id: str,
    task_digest: str,
    schema_version: int,
    shard_index: int,
    start: int,
    stop: int,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, str],
    record_fields: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload_name, record_name = shard_names(shard_index)
    scratch = experiment.output_storage.scratch_root / "ablations" / task_id
    scratch.mkdir(parents=True, exist_ok=True)
    local_payload = scratch / Path(payload_name).name
    write_phase2_shard(
        local_payload,
        tensors=tensors,
        metadata={
            "schema_version": str(schema_version),
            "task_id": task_id,
            "task_digest": task_digest,
            **{str(key): str(value) for key, value in metadata.items()},
        },
    )
    published = store.publish(local_payload, posixpath.join(root, payload_name))
    local_payload.unlink()
    record: Mapping[str, Any] = {
        "schema_version": schema_version,
        "task_digest": task_digest,
        "shard_index": shard_index,
        "start": start,
        "stop": stop,
        "count": stop - start,
        **dict(record_fields),
        "payload": {
            "relative_path": published.relative_path,
            "sha256": published.sha256,
            "size_bytes": published.size_bytes,
        },
    }
    local_record = scratch / Path(record_name).name
    atomic_write_json(local_record, record)
    store.publish(local_record, posixpath.join(root, record_name))
    local_record.unlink()
    return record


def publish_manifest(
    experiment: AblationExperiment,
    store: ArtifactStore,
    *,
    root: str,
    task_id: str,
    manifest: Mapping[str, Any],
) -> None:
    scratch = experiment.output_storage.scratch_root / "ablations" / task_id
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / "manifest.json"
    atomic_write_json(local, manifest)
    store.publish(local, posixpath.join(root, "manifest.json"), write_receipt=False)


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "RANK_SCHEMA_VERSION",
    "completed_evaluation_manifest",
    "completed_rank_manifest",
    "existing_shard_records",
    "output_store",
    "publish_manifest",
    "publish_shard",
    "shard_names",
]
