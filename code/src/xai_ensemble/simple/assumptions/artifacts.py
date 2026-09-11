"""Immutable artifacts for the independent assumption experiment namespace."""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from xai_ensemble.core.io import atomic_write_json

from ..artifacts import ArtifactError, ArtifactStore, completed_manifest, write_phase2_shard
from ..io_pipeline import AsyncSpoolWriter
from ..spool import SpoolQuota
from .config import AssumptionExperiment, EvaluationTask, RankTask, SelectionTask

PARTITION_SCHEMA_VERSION = 1
TRAINING_SCHEMA_VERSION = 1
SOURCE_SCOPE_SCHEMA_VERSION = 2
SPEARMAN_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1
RANK_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1


def output_store(experiment: AssumptionExperiment) -> ArtifactStore:
    return ArtifactStore(experiment)  # type: ignore[arg-type]


def task_spool_path(
    experiment: AssumptionExperiment,
    *,
    namespace: str,
    task_digest: str,
    relative_path: str,
) -> Path:
    """Return a process- and task-isolated tmpfs materialization path."""
    return (
        experiment.storage.spool_root
        / namespace
        / str(os.getpid())
        / task_digest
        / Path(relative_path).name
    )


def completed_task_manifest(
    experiment: AssumptionExperiment,
    task: Any,
    *,
    schema_version: int,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    manifest = completed_manifest(
        store or output_store(experiment),
        task.artifact_root,
        expected_task_digest=task.digest,
        expected_schema_version=schema_version,
    )
    if manifest is not None and hasattr(task, "family_id"):
        expected = {"cell": task.cell.cell_id, "family_id": task.family_id}
        if hasattr(task, "source_id"):
            expected["source_id"] = task.source_id
        mismatches = {
            key: {"artifact": manifest.get(key), "current": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ArtifactError(f"Assumption family identity changed: {mismatches}")
    return manifest


def completed_selection_manifest(
    experiment: AssumptionExperiment,
    task: SelectionTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_task_manifest(
        experiment, task, schema_version=SELECTION_SCHEMA_VERSION, store=store
    )


def completed_rank_manifest(
    experiment: AssumptionExperiment,
    task: RankTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_task_manifest(
        experiment, task, schema_version=RANK_SCHEMA_VERSION, store=store
    )


def completed_evaluation_manifest(
    experiment: AssumptionExperiment,
    task: EvaluationTask,
    *,
    store: ArtifactStore | None = None,
) -> Mapping[str, Any] | None:
    return completed_task_manifest(
        experiment, task, schema_version=EVALUATION_SCHEMA_VERSION, store=store
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
            raise ArtifactError(f"Contradictory assumption shard record: {relative_record}")
        payload = record.get("payload")
        expected_payload = posixpath.join(root, payload_name)
        if not isinstance(payload, Mapping) or payload.get("relative_path") != expected_payload:
            raise ArtifactError(f"Malformed assumption shard record: {relative_record}")
        if not store.exists(f"{expected_payload}.receipt.json"):
            raise ArtifactError(f"Unverified assumption shard payload: {expected_payload}")
        result[shard_index] = record
    return result


def publish_shard(
    experiment: AssumptionExperiment,
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
    scratch = experiment.storage.spool_root / "assumptions" / task_id
    scratch.mkdir(parents=True, exist_ok=True)
    local_payload = scratch / Path(payload_name).name
    try:
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
    finally:
        local_payload.unlink(missing_ok=True)
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
    try:
        atomic_write_json(local_record, record)
        store.publish(local_record, posixpath.join(root, record_name))
    finally:
        local_record.unlink(missing_ok=True)
    return record


def _estimated_tensor_bytes(tensors: Mapping[str, Any]) -> int:
    import torch

    total = 0
    for value in tensors.values():
        tensor = torch.as_tensor(value)
        total += int(tensor.numel()) * int(tensor.element_size())
    return total + 2**20


class AssumptionShardPublisher:
    """Stage assumption outputs in tmpfs and upload without holding the GPU."""

    def __init__(
        self,
        experiment: AssumptionExperiment,
        store: ArtifactStore,
        *,
        task_id: str,
    ) -> None:
        self.experiment = experiment
        self.store = store
        self.task_id = task_id
        self.quota = SpoolQuota(
            experiment.storage.spool_root,
            max_bytes=experiment.storage.spool_max_bytes,
            min_free_bytes=experiment.storage.spool_min_free_bytes,
        )
        self.writer = AsyncSpoolWriter(
            self.quota,
            namespace=f"assumptions-{task_id[:48]}",
        )

    def submit_shard(
        self,
        *,
        root: str,
        task_digest: str,
        schema_version: int,
        shard_index: int,
        start: int,
        stop: int,
        tensors: Mapping[str, Any],
        metadata: Mapping[str, str],
        record_fields: Mapping[str, Any],
    ) -> Future[Mapping[str, Any]]:
        payload_name, record_name = shard_names(shard_index)
        tensor_values = dict(tensors)
        metadata_values = {str(key): str(value) for key, value in metadata.items()}
        record_values = dict(record_fields)

        def stage(local_payload: Path) -> None:
            write_phase2_shard(
                local_payload,
                tensors=tensor_values,
                metadata={
                    "schema_version": str(schema_version),
                    "task_id": self.task_id,
                    "task_digest": task_digest,
                    **metadata_values,
                },
            )

        def upload(local_payload: Path, _: None) -> Mapping[str, Any]:
            published = self.store.publish(local_payload, posixpath.join(root, payload_name))
            record: Mapping[str, Any] = {
                "schema_version": schema_version,
                "task_digest": task_digest,
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "count": stop - start,
                **record_values,
                "payload": {
                    "relative_path": published.relative_path,
                    "sha256": published.sha256,
                    "size_bytes": published.size_bytes,
                },
            }
            local_record = local_payload.with_suffix(".json")
            atomic_write_json(local_record, record)
            self.store.publish(local_record, posixpath.join(root, record_name))
            return record

        return self.writer.submit(
            byte_count=_estimated_tensor_bytes(tensor_values),
            basename=Path(payload_name).name,
            stage=stage,
            upload=upload,
        )

    def check(self) -> None:
        self.writer.check()

    def shutdown(self) -> None:
        self.writer.shutdown()


def publish_manifest(
    experiment: AssumptionExperiment,
    store: ArtifactStore,
    *,
    root: str,
    task_id: str,
    manifest: Mapping[str, Any],
) -> None:
    scratch = experiment.storage.spool_root / "assumptions" / task_id
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / "manifest.json"
    try:
        atomic_write_json(local, manifest)
        store.publish(local, posixpath.join(root, "manifest.json"), write_receipt=False)
    finally:
        local.unlink(missing_ok=True)


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "PARTITION_SCHEMA_VERSION",
    "RANK_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "SOURCE_SCOPE_SCHEMA_VERSION",
    "SPEARMAN_SCHEMA_VERSION",
    "TRAINING_SCHEMA_VERSION",
    "AssumptionShardPublisher",
    "completed_evaluation_manifest",
    "completed_rank_manifest",
    "completed_selection_manifest",
    "completed_task_manifest",
    "existing_shard_records",
    "output_store",
    "publish_manifest",
    "publish_shard",
    "shard_names",
]
