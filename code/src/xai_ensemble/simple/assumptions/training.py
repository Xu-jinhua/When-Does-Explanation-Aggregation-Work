"""Idempotent source-model training and compact checkpoint publication."""

from __future__ import annotations

import posixpath
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..manifest_identity import dataset_manifest_identity_sha256
from ..runtime import emit_gpu_release_signal
from .artifacts import (
    TRAINING_SCHEMA_VERSION,
    completed_task_manifest,
    output_store,
    publish_manifest,
)
from .config import AssumptionExperiment, TrainingTask
from .prepare import partition_local_paths, run_partition_task


def checkpoint_paths(experiment: AssumptionExperiment, task: TrainingTask) -> tuple[Path, Path]:
    checkpoint = (
        experiment.storage.scratch_root
        / "checkpoints"
        / task.cell.cell_id
        / task.source_id
        / task.digest
        / "inference.pt"
    )
    return checkpoint, checkpoint.with_suffix(checkpoint.suffix + ".json")


def _materialize_completed(
    experiment: AssumptionExperiment,
    task: TrainingTask,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    store = output_store(experiment)
    checkpoint, sidecar = checkpoint_paths(experiment, task)
    for key, destination in (("checkpoint", checkpoint), ("sidecar", sidecar)):
        payload = manifest["payloads"][key]
        store.materialize(
            str(payload["relative_path"]),
            destination,
            expected_sha256=str(payload["sha256"]),
        )
    return checkpoint, sidecar


def ensure_checkpoint(
    experiment: AssumptionExperiment, task: TrainingTask
) -> tuple[Path, Mapping[str, Any]]:
    manifest = completed_task_manifest(experiment, task, schema_version=TRAINING_SCHEMA_VERSION)
    if manifest is None:
        raise FileNotFoundError(f"Source checkpoint is not complete: {task.task_id}")
    checkpoint, _ = _materialize_completed(experiment, task, manifest)
    return checkpoint, manifest


def _training_command(
    experiment: AssumptionExperiment,
    task: TrainingTask,
    *,
    device: str,
    output: Path,
) -> tuple[str, ...]:
    partition = experiment.find_partition_task(task.partition_task_id)
    train_path, validation_path = partition_local_paths(experiment, partition)
    dataset_key = task.cell.dataset.registry_key
    if dataset_key is None:
        raise ValueError("Source training currently requires a registered dataset")
    architecture = task.cell.reference_model.architecture
    command = [
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "phase0",
        "train",
        "--dataset",
        dataset_key,
        "--manifest",
        str(task.cell.dataset.manifest_path),
        "--train-partition",
        str(train_path),
        "--validation-partition",
        str(validation_path),
        "--source-id",
        task.source_id,
        "--model",
        task.cell.reference_model.model_key,
        "--recipe",
        "random_scratch",
        "--initialization",
        "random",
        "--epochs",
        str(experiment.training.epochs),
        "--batch-size",
        str(experiment.training.batch_size[architecture]),
        "--validation-batch-size",
        str(experiment.training.validation_batch_size[architecture]),
        "--workers",
        str(task.cell.dataset.cache_images_in_ram and 8 or 4),
        "--precision",
        experiment.training.precision,
        "--seed",
        str(experiment.training_seed(task)),
        "--device",
        device,
        "--output",
        str(output),
        "--resume",
        "auto",
        "--class-balance",
        experiment.training.balance_for(task.cell.reference_model.model_id),
        "--run-id",
        experiment.assumption_id,
        "--task-id",
        task.task_id,
        "--protocol-digest",
        experiment.digest,
        "--project-root",
        str(experiment.source_path.parents[2]),
    ]
    if task.cell.dataset.cache_directory is not None:
        command.extend(("--cache-dir", str(task.cell.dataset.cache_directory)))
    if task.cell.dataset.keep_provider_in_memory:
        command.append("--keep-in-memory")
    return tuple(command)


def run_training_task(
    experiment: AssumptionExperiment,
    task: TrainingTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    store = output_store(experiment)
    complete = completed_task_manifest(
        experiment, task, schema_version=TRAINING_SCHEMA_VERSION, store=store
    )
    if complete is not None:
        _materialize_completed(experiment, task, complete)
        return complete

    partition = experiment.find_partition_task(task.partition_task_id)
    partition_manifest = run_partition_task(experiment, partition)
    checkpoint, sidecar = checkpoint_paths(experiment, task)
    output = checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        _training_command(experiment, task, device=device, output=output),
        check=True,
    )
    if not checkpoint.is_file() or not sidecar.is_file():
        raise RuntimeError("Phase 0 training did not publish an inference checkpoint")

    # The child process has exited and released all CUDA state. Let the scheduler
    # place another job while this small immutable checkpoint is verified remotely.
    emit_gpu_release_signal()
    payloads = {}
    for key, source, filename in (
        ("checkpoint", checkpoint, "inference.pt"),
        ("sidecar", sidecar, "inference.pt.json"),
    ):
        published = store.publish(source, posixpath.join(task.artifact_root, filename))
        payloads[key] = {
            "relative_path": published.relative_path,
            "sha256": published.sha256,
            "size_bytes": published.size_bytes,
        }
    value: Mapping[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "cell": task.cell.cell_id,
        "source_id": task.source_id,
        "family_id": task.family_id,
        "model_key": task.cell.reference_model.model_key,
        "num_classes": task.cell.reference_model.num_classes,
        "partition_task_id": partition.task_id,
        "partition_task_digest": partition.digest,
        "partition_manifest_digest": dataset_manifest_identity_sha256(
            task.cell.dataset.manifest_path
        ),
        "partition_artifact": {
            "task_digest": partition_manifest["task_digest"],
            "train_partition_digest": partition_manifest["train_partition_digest"],
            "validation_partition_digest": partition_manifest["validation_partition_digest"],
        },
        "training": {
            "recipe": "random_scratch",
            "epochs": experiment.training.epochs,
            "batch_size": experiment.training.batch_size[task.cell.reference_model.architecture],
            "validation_batch_size": experiment.training.validation_batch_size[
                task.cell.reference_model.architecture
            ],
            "precision": experiment.training.precision,
            "class_balance": experiment.training.balance_for(task.cell.reference_model.model_id),
            "seed": experiment.training_seed(task),
        },
        "payloads": payloads,
    }
    publish_manifest(
        experiment, store, root=task.artifact_root, task_id=task.task_id, manifest=value
    )

    # Exact training resume state is useful only until the committed inference
    # payload exists. Remove optimizer-heavy files but retain the compact local
    # inference cache consumed by source Phase 1.
    for candidate in output.glob("*.pt"):
        if candidate != checkpoint:
            candidate.unlink(missing_ok=True)
            candidate.with_suffix(candidate.suffix + ".json").unlink(missing_ok=True)
    for candidate in output.glob("epoch-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate)
    return value


__all__ = ["checkpoint_paths", "ensure_checkpoint", "run_training_task"]
