"""Deterministic IND partitions and the p=16 Spearman-Mallows family."""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.data import read_manifest
from xai_ensemble.data.partitions import make_ind_partitions, write_partition_plan
from xai_ensemble.phase2.selection import locked_spearman_family_factory

from ..artifacts import ArtifactError, completed_manifest
from .artifacts import (
    PARTITION_SCHEMA_VERSION,
    SPEARMAN_SCHEMA_VERSION,
    completed_task_manifest,
    output_store,
    publish_manifest,
)
from .config import AssumptionExperiment, PartitionTask
from .spearman_calibration import (
    calibrate_spearman_mallows,
    synthetic_validation_distances,
)

SPEARMAN_ITEMS = 196
SPEARMAN_TASK_ID = "spearman-p196"


def partition_local_paths(
    experiment: AssumptionExperiment, task: PartitionTask
) -> tuple[Path, Path]:
    root = experiment.storage.scratch_root / "partitions" / task.digest
    return root / "train.json", root / "validation.json"


def run_partition_task(
    experiment: AssumptionExperiment,
    task: PartitionTask,
) -> Mapping[str, Any]:
    store = output_store(experiment)
    complete = completed_task_manifest(
        experiment, task, schema_version=PARTITION_SCHEMA_VERSION, store=store
    )
    train_path, validation_path = partition_local_paths(experiment, task)
    if complete is not None:
        for key, path in (("train", train_path), ("validation", validation_path)):
            payload = complete["payloads"][key]
            store.materialize(
                str(payload["relative_path"]), path, expected_sha256=str(payload["sha256"])
            )
        return complete

    manifest = read_manifest(resolve_full_matrix_runtime_path(task.cell.dataset.manifest_path))
    train = make_ind_partitions(
        manifest,
        split=experiment.training.train_split,
        num_sources=task.source_count,
        seed=experiment.partition_seed(task.cell, task.family_id),
        require_full_coverage=True,
    )
    validation = make_ind_partitions(
        manifest,
        split=experiment.training.validation_split,
        num_sources=task.source_count,
        seed=experiment.partition_seed(task.cell, task.family_id),
        require_full_coverage=True,
    )
    source_ids = experiment.source_ids(task.cell, task.family_id)
    train = replace(train, sources=tuple(
        replace(source, source_id=source_id)
        for source, source_id in zip(train.sources, source_ids, strict=True)
    ))
    validation = replace(validation, sources=tuple(
        replace(source, source_id=source_id)
        for source, source_id in zip(validation.sources, source_ids, strict=True)
    ))
    train_path.parent.mkdir(parents=True, exist_ok=True)
    write_partition_plan(train, train_path)
    write_partition_plan(validation, validation_path)
    payloads = {}
    for key, path in (("train", train_path), ("validation", validation_path)):
        published = store.publish(path, posixpath.join(task.artifact_root, f"{key}.json"))
        payloads[key] = {
            "relative_path": published.relative_path,
            "sha256": published.sha256,
            "size_bytes": published.size_bytes,
        }
    value: Mapping[str, Any] = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell.cell_id,
        "dataset_manifest_fingerprint": manifest.fingerprint,
        "source_count": task.source_count,
        "family_id": task.family_id,
        "partition_seed": experiment.partition_seed(task.cell, task.family_id),
        "train_partition_digest": train.digest,
        "validation_partition_digest": validation.digest,
        "source_sizes": {
            "train": {source.source_id: source.size for source in train.sources},
            "validation": {source.source_id: source.size for source in validation.sources},
        },
        "payloads": payloads,
    }
    publish_manifest(
        experiment, store, root=task.artifact_root, task_id=task.task_id, manifest=value
    )
    return value


def _calibration_config(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    value = (
        yaml.safe_load(experiment.selection.spearman_calibration_config.read_text(encoding="utf-8"))
        or {}
    )
    if not isinstance(value, Mapping):
        raise TypeError("Spearman calibration config must be a mapping")
    return value


def spearman_identity(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    return experiment.spearman_family_identity()


def spearman_local_artifact_path(experiment: AssumptionExperiment) -> Path:
    """Return the schema-versioned local cache without changing experiment identity."""

    configured = experiment.selection.spearman_artifact
    if configured.stem.endswith("-v2"):
        return configured
    return configured.with_name(f"{configured.stem}-v2{configured.suffix}")


def spearman_artifact_root(experiment: AssumptionExperiment) -> str:
    return posixpath.join(
        "statistics",
        "spearman-mallows-p196",
        object_sha256(spearman_identity(experiment)),
    )


def _validated_local_metadata(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    path = spearman_local_artifact_path(experiment)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("The local p=16 Spearman-Mallows artifact is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ArtifactError("The p=16 Spearman metadata is not a mapping")
    if metadata.get("identity") != spearman_identity(experiment):
        raise ArtifactError("The p=16 Spearman artifact has a different identity")
    if metadata.get("sha256") != file_sha256(path):
        raise ArtifactError("The p=16 Spearman artifact digest does not match its metadata")
    if not metadata.get("pilot_digest"):
        raise ArtifactError("The p=16 Spearman metadata has no pilot digest")
    return metadata


def completed_spearman_family(
    experiment: AssumptionExperiment,
    *,
    restore: bool = True,
) -> Mapping[str, Any] | None:
    identity = spearman_identity(experiment)
    digest = object_sha256(identity)
    store = output_store(experiment)
    root = spearman_artifact_root(experiment)
    manifest = completed_manifest(
        store,
        root,
        expected_task_digest=digest,
        expected_schema_version=SPEARMAN_SCHEMA_VERSION,
    )
    if manifest is None:
        return None
    if manifest.get("identity") != identity:
        raise ArtifactError("Published p=16 Spearman artifact has a different identity")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, Mapping):
        raise ArtifactError("Published p=16 Spearman manifest has no payload mapping")
    if restore:
        path = spearman_local_artifact_path(experiment)
        metadata_path = path.with_suffix(path.suffix + ".json")
        for key, destination in (("family", path), ("metadata", metadata_path)):
            payload = payloads.get(key)
            if not isinstance(payload, Mapping):
                raise ArtifactError(f"Published p=16 Spearman manifest lacks {key}")
            if not store.exists(f"{payload['relative_path']}.receipt.json"):
                raise ArtifactError(f"Published p=16 Spearman {key} has no verified receipt")
            store.materialize(
                str(payload["relative_path"]),
                destination,
                expected_sha256=str(payload["sha256"]),
            )
        _validated_local_metadata(experiment)
    return manifest


def prepare_spearman_family(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    """Generate and validate the one p=16 statistical family artifact."""

    complete = completed_spearman_family(experiment, restore=True)
    if complete is not None:
        return _validated_local_metadata(experiment)

    path = spearman_local_artifact_path(experiment)
    metadata_path = path.with_suffix(path.suffix + ".json")
    config = _calibration_config(experiment)
    seed = experiment.assignment_seed
    identity = spearman_identity(experiment)
    pilot_digest = object_sha256(identity)
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity or metadata.get("sha256") != file_sha256(path):
            raise ValueError("Existing p=16 Spearman artifact has a different identity")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        validation = synthetic_validation_distances(
            n_items=SPEARMAN_ITEMS,
            count=identity["synthetic_validation_count"],
            q_values=np.asarray(identity["synthetic_q"], dtype=np.float64),
            seed=seed,
        )
        outcome = calibrate_spearman_mallows(
            validation,
            n_items=SPEARMAN_ITEMS,
            config=config,
            artifact_path=path,
            seed=seed,
            bootstrap_replicates=experiment.selection.bootstrap_replicates,
            projected_prefix_count=identity["projected_prefix_count"],
        )
        if not outcome.converged:
            raise RuntimeError(f"p=16 Spearman-Mallows calibration failed: {outcome.failures}")
        actual_pilot_digest = str(outcome.measurement["spearman_pilot_digest"])
        atomic_write_json(
            metadata_path,
            {
                "schema_version": 1,
                "identity": identity,
                "identity_digest": pilot_digest,
                "pilot_digest": actual_pilot_digest,
                "sha256": file_sha256(path),
                "outcome": {
                    "artifact_path": str(outcome.artifact_path),
                    "measurement": outcome.measurement,
                    "converged": outcome.converged,
                    "failures": list(outcome.failures),
                },
            },
        )

    metadata = _validated_local_metadata(experiment)
    locked_spearman_family_factory(
        distance="spearman",
        n_items=SPEARMAN_ITEMS,
        seed=seed,
        pilot_digest=str(metadata["pilot_digest"]),
        config={
            "path": str(path),
            "sha256": file_sha256(path),
            **dict(config["spearman_mallows_approximation"]["pass"]),
        },
    )
    store = output_store(experiment)
    root = spearman_artifact_root(experiment)
    payloads = {}
    for key, source, name in (
        ("family", path, "spearman-mallows-p196-v2.npz"),
        ("metadata", metadata_path, "spearman-mallows-p196-v2.metadata.json"),
    ):
        published = store.publish(source, posixpath.join(root, name))
        payloads[key] = {
            "relative_path": published.relative_path,
            "sha256": published.sha256,
            "size_bytes": published.size_bytes,
        }
    manifest: Mapping[str, Any] = {
        "schema_version": SPEARMAN_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": SPEARMAN_TASK_ID,
        "task_digest": pilot_digest,
        "identity": identity,
        "pilot_digest": metadata["pilot_digest"],
        "payloads": payloads,
    }
    publish_manifest(
        experiment,
        store,
        root=root,
        task_id=SPEARMAN_TASK_ID,
        manifest=manifest,
    )
    return metadata


def spearman_family_config(experiment: AssumptionExperiment) -> Mapping[str, Any]:
    artifact_path = spearman_local_artifact_path(experiment)
    metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    if not artifact_path.is_file() or not metadata_path.is_file():
        if completed_spearman_family(experiment, restore=True) is None:
            raise FileNotFoundError(
                "Run the assumptions prepare step to create p=16 Spearman family"
            )
    metadata = _validated_local_metadata(experiment)
    config = _calibration_config(experiment)
    return {
        "path": str(artifact_path),
        "sha256": str(metadata["sha256"]),
        "pilot_digest": str(metadata["pilot_digest"]),
        **dict(config["spearman_mallows_approximation"]["pass"]),
    }


__all__ = [
    "SPEARMAN_ITEMS",
    "SPEARMAN_TASK_ID",
    "completed_spearman_family",
    "partition_local_paths",
    "prepare_spearman_family",
    "run_partition_task",
    "spearman_artifact_root",
    "spearman_family_config",
    "spearman_identity",
    "spearman_local_artifact_path",
]
