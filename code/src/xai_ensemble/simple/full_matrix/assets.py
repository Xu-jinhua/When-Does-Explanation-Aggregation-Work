"""Phase-0 assets and real-checkpoint compatibility jobs for the full matrix."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json, read_json
from xai_ensemble.core.manifest import load_manifest, validate_manifest_files
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.data import get_dataset_spec, read_manifest
from xai_ensemble.data.partitions import read_partition_plan
from xai_ensemble.phase1.compatibility import run_compatibility_pilot
from xai_ensemble.phase1.method_lock import LockedMethod
from xai_ensemble.phase1.relprop import relprop_support_error
from xai_ensemble.simple.methods import PATCH_METHODS, MethodVariant
from xai_ensemble.simple.scheduler import QueueJob

from .catalog import MatrixCell, validate_method_rosters
from .config import FullMatrixExperiment

CPU_JOB_KINDS = frozenset(
    {
        "matrix-manifest",
        "matrix-partition",
        "matrix-samples",
        "matrix-mean",
        "matrix-static-compatibility",
    }
)
GPU_JOB_KINDS = frozenset({"matrix-reference-training", "matrix-compatibility"})
ASSET_JOB_KINDS = CPU_JOB_KINDS | GPU_JOB_KINDS


def manifest_job_id(dataset_id: str) -> str:
    return f"matrix-manifest:{dataset_id}"


def partition_job_id(dataset_id: str, split: str) -> str:
    return f"matrix-partition:{dataset_id}:{split}"


def samples_job_id(dataset_id: str) -> str:
    return f"matrix-samples:{dataset_id}"


def mean_job_id(cell: MatrixCell) -> str:
    return f"matrix-mean:{cell.cell_id}"


def training_job_id(cell: MatrixCell) -> str:
    return f"matrix-reference-training:{cell.cell_id}"


def compatibility_job_id(cell: MatrixCell) -> str:
    return f"matrix-compatibility:{cell.cell_id}"


def _checkpoint_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def compatibility_variants(
    experiment: FullMatrixExperiment,
    cell: MatrixCell,
) -> tuple[MethodVariant, ...]:
    """Return the exact p=16 variants used by the released matrix."""

    variants: list[MethodVariant] = []
    for definition in experiment.methods.for_architecture(cell.architecture):
        candidates = definition.instances(cell.architecture)
        if definition.family in PATCH_METHODS:
            candidates = tuple(item for item in candidates if item.variant == "p16")
        if len(candidates) != 1:
            raise ValueError(
                "Full-matrix compatibility requires exactly one formal variant for "
                f"{cell.architecture}/{definition.family}; found {len(candidates)}"
            )
        variants.extend(candidates)
    if len(variants) != 11 or len({item.family for item in variants}) != len(variants):
        raise ValueError("Full-matrix compatibility must contain exactly eleven families")
    return tuple(variants)


def compatibility_candidates(
    experiment: FullMatrixExperiment,
    cell: MatrixCell,
) -> tuple[LockedMethod, ...]:
    return tuple(
        LockedMethod(
            family=variant.family,
            instance_id=f"{variant.artifact_name}-{variant.digest[:12]}",
            params=dict(variant.params),
            architecture=variant.architecture,
        )
        for variant in compatibility_variants(experiment, cell)
    )


def compatibility_candidate_payload(
    experiment: FullMatrixExperiment,
    cell: MatrixCell,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "family": candidate.family,
            "instance_id": candidate.instance_id,
            "params": dict(candidate.params),
            "architecture": candidate.architecture,
        }
        for candidate in compatibility_candidates(experiment, cell)
    )


def compatibility_candidate_digest(
    experiment: FullMatrixExperiment,
    cell: MatrixCell,
) -> str:
    return object_sha256(list(compatibility_candidate_payload(experiment, cell)))


def static_compatibility_failures(
    experiment: FullMatrixExperiment,
    cell: MatrixCell,
) -> tuple[Mapping[str, str], ...]:
    """Identify provider limitations that cannot be repaired by training a cell."""

    failures = []
    for candidate in compatibility_candidates(experiment, cell):
        error = relprop_support_error(
            candidate.family,
            model_key=cell.model_key,
            architecture=cell.architecture,
        )
        if error is not None:
            failures.append({"method": candidate.family, "error": error})
    return tuple(failures)


def manifest_complete(experiment: FullMatrixExperiment, dataset_id: str) -> bool:
    path = resolve_full_matrix_runtime_path(experiment.manifest_path(dataset_id))
    if not path.is_file():
        return False
    manifest = read_manifest(path)
    manifest.validate(get_dataset_spec(dataset_id), reject_cross_split_duplicates=False)
    return True


def partition_complete(
    experiment: FullMatrixExperiment,
    dataset_id: str,
    split: str,
) -> bool:
    path = resolve_full_matrix_runtime_path(experiment.partition_path(dataset_id, split))
    if not path.is_file():
        return False
    if not manifest_complete(experiment, dataset_id):
        return False
    manifest = read_manifest(resolve_full_matrix_runtime_path(experiment.manifest_path(dataset_id)))
    plan = read_partition_plan(path)
    plan.validate(manifest)
    return (
        plan.kind == "reference"
        and plan.split == split
        and len(plan.sources) == 1
        and plan.sources[0].source_id == "reference-full"
    )


def samples_complete(experiment: FullMatrixExperiment, dataset_id: str) -> bool:
    path = experiment.sample_ids_path(dataset_id)
    if not path.is_file():
        return False
    value = read_json(path)
    expected = {
        "schema": "simple-full-matrix-compatibility-samples-v1",
        "matrix_digest": experiment.digest,
        "dataset": dataset_id,
        "split": "validation",
        "count": experiment.compatibility_samples,
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise ValueError(f"Compatibility sample identity is contradictory: {path}")
    identifiers = value.get("sample_ids")
    if (
        not isinstance(identifiers, Sequence)
        or isinstance(identifiers, (str, bytes))
        or len(identifiers) != experiment.compatibility_samples
        or len(set(str(item) for item in identifiers)) != len(identifiers)
    ):
        raise ValueError(f"Compatibility sample coverage is invalid: {path}")
    return True


def mean_complete(experiment: FullMatrixExperiment, cell: MatrixCell) -> bool:
    path = resolve_full_matrix_runtime_path(experiment.mean_path(cell) / "manifest.json")
    if not path.is_file():
        return False
    manifest = load_manifest(path)
    validate_manifest_files(path.parent, manifest)
    metadata = manifest.metadata
    expected = {
        "dataset_id": cell.dataset.dataset_id,
        "dataset_revision": cell.dataset.revision,
        "model_id": cell.model_key,
        "protocol_digest": experiment.protocol_digest,
        "source_split": "train",
    }
    contradictions = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if contradictions:
        raise ValueError(f"Mean artifact identity is contradictory: {contradictions}")
    return True


def checkpoint_complete(experiment: FullMatrixExperiment, cell: MatrixCell) -> bool:
    path = resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell))
    sidecar = _checkpoint_sidecar(path)
    if not path.is_file() and not sidecar.is_file():
        return False
    if not path.is_file() or not sidecar.is_file():
        return False
    value = read_json(sidecar)
    metadata = value.get("metadata") if isinstance(value, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Reference checkpoint sidecar is malformed: {sidecar}")
    expected = {
        "role": "reference",
        "source_id": "reference-full",
        "dataset_id": cell.dataset.dataset_id,
        "dataset_revision": cell.dataset.revision,
        "model_key": cell.model_key,
        "num_classes": cell.num_classes,
        "protocol_digest": experiment.protocol_digest,
    }
    contradictions = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if contradictions:
        raise ValueError(f"Reference checkpoint identity is contradictory: {contradictions}")
    observed = file_sha256(path)
    if value.get("checkpoint_sha256") != observed:
        raise ValueError(f"Reference checkpoint SHA-256 is contradictory: {path}")
    return True


def compatibility_complete(
    experiment: FullMatrixExperiment,
    cell: MatrixCell,
    *,
    verify_checkpoint: bool = True,
) -> bool:
    """Validate a compatibility gate and, by default, its checkpoint bytes.

    The full validation remains the default for planning and worker gates.  A
    long-lived scheduler may pass ``verify_checkpoint=False`` while polling an
    already sealed gate; the gate's immutable checkpoint digest is still
    checked for presence and identity, but the remote payload is not hashed on
    every scheduling loop.
    """

    path = experiment.compatibility_directory(cell) / "gate.json"
    if not path.is_file():
        return False
    value = read_json(path)
    expected = {
        "schema": "simple-full-matrix-compatibility-gate-v1",
        "matrix_digest": experiment.digest,
        "cell": cell.cell_id,
        "method_catalog_digest": experiment.methods.source_digest,
        "candidate_digest": compatibility_candidate_digest(experiment, cell),
        "rank_patch_size": 16,
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise ValueError(f"Compatibility gate identity is contradictory: {path}")
    if value.get("status") not in {"passed", "blocked"}:
        raise ValueError(f"Compatibility gate has invalid status: {path}")
    gate_type = value.get("gate_type")
    if gate_type == "static_provider_preflight":
        if value.get("status") != "blocked" or value.get("checkpoint_sha256") is not None:
            raise ValueError(f"Static compatibility gate is malformed: {path}")
        failures = value.get("failures")
        if not isinstance(failures, list) or not failures:
            raise ValueError(f"Static compatibility gate has no failures: {path}")
        expected_failures = list(static_compatibility_failures(experiment, cell))
        if value.get("static_failure_digest") != object_sha256(expected_failures):
            raise ValueError(f"Static compatibility gate is stale: {path}")
    elif gate_type == "real_checkpoint":
        checkpoint = resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell))
        if not checkpoint.is_file():
            raise ValueError(f"Real compatibility checkpoint is missing: {checkpoint}")
        if verify_checkpoint and value.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError(f"Real compatibility checkpoint identity is contradictory: {path}")
    else:
        raise ValueError(f"Compatibility gate has unknown type: {path}")
    payload = {key: item for key, item in value.items() if key != "gate_digest"}
    if value.get("gate_digest") != object_sha256(payload):
        raise ValueError(f"Compatibility gate digest is invalid: {path}")
    return True


def write_static_compatibility_gate(
    experiment: FullMatrixExperiment,
    *,
    cell_id: str,
) -> Mapping[str, Any]:
    """Publish a terminal block for a known unavailable RelProp provider."""

    cell = experiment.cell(cell_id)
    failures = static_compatibility_failures(experiment, cell)
    if not failures:
        raise ValueError(f"{cell.cell_id} has no static provider compatibility failure")
    output = experiment.compatibility_directory(cell) / "gate.json"
    if output.is_file():
        if compatibility_complete(experiment, cell):
            return read_json(output)
        raise ValueError(f"Existing static compatibility gate is contradictory: {output}")
    candidates = compatibility_candidate_payload(experiment, cell)
    blocked = sorted(str(item["method"]) for item in failures)
    value: dict[str, Any] = {
        "schema": "simple-full-matrix-compatibility-gate-v1",
        "schema_version": 1,
        "matrix_digest": experiment.digest,
        "cell": cell.cell_id,
        "dataset": cell.dataset_id,
        "model": cell.model_key,
        "architecture": cell.architecture,
        "gate_type": "static_provider_preflight",
        "status": "blocked",
        "checkpoint_sha256": None,
        "method_catalog_digest": experiment.methods.source_digest,
        "candidate_digest": object_sha256(list(candidates)),
        "candidates": list(candidates),
        "rank_patch_size": 16,
        "required_methods": [candidate["family"] for candidate in candidates],
        "passed_methods": [],
        "blocked_methods": blocked,
        "pilot_status": "not_run_static_provider_unsupported",
        "pilot_result_digest": None,
        "pilot_result_path": None,
        "failures": list(failures),
        "static_failure_digest": object_sha256(list(failures)),
    }
    value["gate_digest"] = object_sha256(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)
    return value


def _job(
    experiment: FullMatrixExperiment,
    *,
    job_id: str,
    kind: str,
    command: tuple[str, ...],
    dependencies: tuple[str, ...],
    reservation_bytes: int | None,
    complete: bool,
) -> QueueJob:
    safe_name = job_id.replace(":", "--")
    return QueueJob(
        job_id=job_id,
        kind=kind,
        command=command,
        dependencies=dependencies,
        resource_ids=(),
        reservation_bytes=reservation_bytes,
        status="succeeded" if complete else "pending",
        attempts=0,
        max_retries=experiment.runtime.max_retries,
        pid=None,
        gpu_id=None,
        log_path=str(experiment.runtime.log_directory / "assets" / f"{safe_name}.log"),
    )


def planned_asset_jobs(
    experiment: FullMatrixExperiment,
    *,
    check_complete: bool = True,
) -> tuple[QueueJob, ...]:
    """Build the immutable Phase-0 queue projection.

    Scheduler submission verifies existing artifacts before it marks a job
    complete.  The read-only public plan command needs only the immutable DAG
    shape, so it can avoid opening every mounted CloudStorage artifact.
    """

    jobs: list[QueueJob] = []
    config = str(experiment.source_path)
    project_root = str(experiment.source_path.parents[3])
    for dataset_id in experiment.dataset_ids:
        manifest_id = manifest_job_id(dataset_id)
        jobs.append(
            _job(
                experiment,
                job_id=manifest_id,
                kind="matrix-manifest",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "phase0",
                    "build-manifest",
                    "--dataset",
                    dataset_id,
                    "--output",
                    str(experiment.manifest_path(dataset_id)),
                    "--cache-dir",
                    str(experiment.cache_directory(dataset_id)),
                    "--splits",
                    "train,validation,test",
                ),
                dependencies=(),
                reservation_bytes=None,
                complete=manifest_complete(experiment, dataset_id) if check_complete else False,
            )
        )
        for split in ("train", "validation"):
            jobs.append(
                _job(
                    experiment,
                    job_id=partition_job_id(dataset_id, split),
                    kind="matrix-partition",
                    command=(
                        sys.executable,
                        "-m",
                        "xai_ensemble.cli",
                        "phase0",
                        "build-partitions",
                        "--manifest",
                        str(experiment.manifest_path(dataset_id)),
                        "--output",
                        str(experiment.partition_path(dataset_id, split)),
                        "--kind",
                        "reference",
                        "--split",
                        split,
                        "--seed",
                        str(experiment.seed),
                    ),
                    dependencies=(manifest_id,),
                    reservation_bytes=None,
                    complete=partition_complete(experiment, dataset_id, split)
                    if check_complete
                    else False,
                )
            )
        jobs.append(
            _job(
                experiment,
                job_id=samples_job_id(dataset_id),
                kind="matrix-samples",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "simple",
                    "full-matrix",
                    "prepare-samples",
                    "--config",
                    config,
                    "--dataset",
                    dataset_id,
                ),
                dependencies=(manifest_id,),
                reservation_bytes=None,
                complete=samples_complete(experiment, dataset_id) if check_complete else False,
            )
        )

    for cell in experiment.cells():
        manifest_id = manifest_job_id(cell.dataset_id)
        static_failures = static_compatibility_failures(experiment, cell)
        if static_failures:
            jobs.append(
                _job(
                    experiment,
                    job_id=compatibility_job_id(cell),
                    kind="matrix-static-compatibility",
                    command=(
                        sys.executable,
                        "-m",
                        "xai_ensemble.cli",
                        "simple",
                        "full-matrix",
                        "static-compatibility",
                        "--config",
                        config,
                        "--cell",
                        cell.cell_id,
                    ),
                    dependencies=(),
                    reservation_bytes=None,
                    complete=compatibility_complete(experiment, cell) if check_complete else False,
                )
            )
            continue
        jobs.append(
            _job(
                experiment,
                job_id=mean_job_id(cell),
                kind="matrix-mean",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "phase0",
                    "compute-means",
                    "--dataset",
                    cell.dataset_id,
                    "--manifest",
                    str(experiment.manifest_path(cell.dataset_id)),
                    "--model",
                    cell.model_key,
                    "--split",
                    "train",
                    "--batch-size",
                    "128",
                    "--workers",
                    str(experiment.training.workers),
                    "--seed",
                    str(experiment.seed),
                    "--cache-dir",
                    str(experiment.cache_directory(cell.dataset_id)),
                    "--protocol-digest",
                    experiment.protocol_digest,
                    "--output",
                    str(experiment.mean_path(cell)),
                ),
                dependencies=(manifest_id,),
                reservation_bytes=None,
                complete=mean_complete(experiment, cell) if check_complete else False,
            )
        )
        jobs.append(
            _job(
                experiment,
                job_id=training_job_id(cell),
                kind="matrix-reference-training",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "simple",
                    "full-matrix",
                    "train-reference",
                    "--config",
                    config,
                    "--cell",
                    cell.cell_id,
                    "--project-root",
                    project_root,
                ),
                dependencies=(
                    manifest_id,
                    partition_job_id(cell.dataset_id, "train"),
                    partition_job_id(cell.dataset_id, "validation"),
                ),
                reservation_bytes=experiment.runtime.training_reservation_bytes[cell.architecture],
                complete=checkpoint_complete(experiment, cell) if check_complete else False,
            )
        )
        jobs.append(
            _job(
                experiment,
                job_id=compatibility_job_id(cell),
                kind="matrix-compatibility",
                command=(
                    sys.executable,
                    "-m",
                    "xai_ensemble.cli",
                    "simple",
                    "full-matrix",
                    "compatibility",
                    "--config",
                    config,
                    "--cell",
                    cell.cell_id,
                ),
                dependencies=(
                    training_job_id(cell),
                    mean_job_id(cell),
                    samples_job_id(cell.dataset_id),
                ),
                reservation_bytes=experiment.runtime.compatibility_reservation_bytes[
                    cell.architecture
                ],
                complete=compatibility_complete(experiment, cell) if check_complete else False,
            )
        )
    return tuple(jobs)


def prepare_compatibility_samples(
    experiment: FullMatrixExperiment,
    *,
    dataset_id: str,
) -> Mapping[str, Any]:
    if samples_complete(experiment, dataset_id):
        return read_json(experiment.sample_ids_path(dataset_id))
    manifest = read_manifest(resolve_full_matrix_runtime_path(experiment.manifest_path(dataset_id)))
    records = tuple(manifest.records_for_split("validation"))
    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                f"full-matrix-compatibility-v1\0{experiment.seed}\0{record.sample_id}".encode()
            ).hexdigest(),
            record.sample_id,
        ),
    )
    selected = []
    labels = set()
    for record in ordered:
        if record.label in labels:
            continue
        selected.append(record)
        labels.add(record.label)
        if len(selected) == experiment.compatibility_samples:
            break
    if len(selected) < experiment.compatibility_samples:
        chosen = {record.sample_id for record in selected}
        selected.extend(record for record in ordered if record.sample_id not in chosen)
        selected = selected[: experiment.compatibility_samples]
    if len(selected) != experiment.compatibility_samples:
        raise RuntimeError(f"Not enough validation samples for {dataset_id}")
    value: dict[str, Any] = {
        "schema": "simple-full-matrix-compatibility-samples-v1",
        "matrix_digest": experiment.digest,
        "dataset": dataset_id,
        "split": "validation",
        "count": len(selected),
        "sample_ids": [record.sample_id for record in selected],
        "labels": [record.label for record in selected],
        "selection": "sha256_order_distinct_labels_first_v1",
    }
    value["digest"] = object_sha256(value)
    atomic_write_json(experiment.sample_ids_path(dataset_id), value)
    return value


def _publish_checkpoint(source: Path, destination: Path) -> None:
    source_sidecar = _checkpoint_sidecar(source)
    if not source.is_file() or not source_sidecar.is_file():
        raise FileNotFoundError(f"Training did not create an inference checkpoint: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_sidecar = _checkpoint_sidecar(destination)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    temporary_sidecar = _checkpoint_sidecar(temporary)
    try:
        shutil.copy2(source, temporary)
        shutil.copy2(source_sidecar, temporary_sidecar)
        if file_sha256(temporary) != read_json(temporary_sidecar).get("checkpoint_sha256"):
            raise ValueError("Published reference checkpoint failed SHA-256 verification")
        os.replace(temporary, destination)
        os.replace(temporary_sidecar, destination_sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)


def train_reference_model(
    experiment: FullMatrixExperiment,
    *,
    cell_id: str,
    project_root: str | Path,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    cell = experiment.cell(cell_id)
    if checkpoint_complete(experiment, cell):
        return {
            "status": "complete",
            "cell": cell.cell_id,
            "checkpoint": str(resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell))),
            "checkpoint_sha256": file_sha256(
                resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell))
            ),
            "reused": True,
        }
    scratch = experiment.training_scratch_directory(cell)
    scratch.mkdir(parents=True, exist_ok=True)
    model_batch = experiment.training.batch_size.get(
        cell.model_key,
        experiment.training.batch_size[cell.architecture],
    )
    validation_batch = experiment.training.validation_batch_size.get(
        cell.model_key,
        experiment.training.validation_batch_size[cell.architecture],
    )
    class_balance = experiment.training.class_balance.get(cell.model_id, "none")
    command = (
        sys.executable,
        "-m",
        "xai_ensemble.cli",
        "phase0",
        "train",
        "--dataset",
        cell.dataset_id,
        "--manifest",
        str(experiment.manifest_path(cell.dataset_id)),
        "--train-partition",
        str(experiment.partition_path(cell.dataset_id, "train")),
        "--validation-partition",
        str(experiment.partition_path(cell.dataset_id, "validation")),
        "--source-id",
        "reference-full",
        "--model",
        cell.model_key,
        "--recipe",
        experiment.training.recipe,
        "--initialization",
        experiment.training.initialization,
        "--epochs",
        str(experiment.training.epochs),
        "--batch-size",
        str(model_batch),
        "--validation-batch-size",
        str(validation_batch),
        "--workers",
        str(experiment.training.workers),
        "--precision",
        experiment.training.precision,
        "--seed",
        str(experiment.seed),
        "--device",
        device,
        "--cache-dir",
        str(experiment.cache_directory(cell.dataset_id)),
        "--output",
        str(scratch),
        "--resume",
        "auto",
        "--run-id",
        experiment.experiment_id,
        "--task-id",
        f"reference-{cell.cell_id}",
        "--protocol-digest",
        experiment.protocol_digest,
        "--project-root",
        str(Path(project_root).resolve()),
        "--class-balance",
        class_balance,
    )
    subprocess.run(command, check=True)
    source = scratch / "inference.pt"
    destination = resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell))
    _publish_checkpoint(source, destination)
    if not checkpoint_complete(experiment, cell):
        raise RuntimeError(f"Published checkpoint did not validate: {destination}")
    shutil.rmtree(scratch)
    return {
        "status": "complete",
        "cell": cell.cell_id,
        "checkpoint": str(destination),
        "checkpoint_sha256": file_sha256(destination),
        "reused": False,
    }


def run_compatibility_gate(
    experiment: FullMatrixExperiment,
    *,
    cell_id: str,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    cell = experiment.cell(cell_id)
    if compatibility_complete(experiment, cell):
        return read_json(experiment.compatibility_directory(cell) / "gate.json")
    static_failures = static_compatibility_failures(experiment, cell)
    if static_failures:
        raise RuntimeError(
            f"{cell.cell_id} must use the static compatibility gate: {list(static_failures)}"
        )
    output_directory = experiment.compatibility_directory(cell)
    output_directory.mkdir(parents=True, exist_ok=True)
    candidates = compatibility_candidates(experiment, cell)
    candidate_payload = compatibility_candidate_payload(experiment, cell)
    output = run_compatibility_pilot(
        protocol_path=experiment.protocol_path,
        dataset_key=cell.dataset_id,
        dataset_manifest_path=resolve_full_matrix_runtime_path(
            experiment.manifest_path(cell.dataset_id)
        ),
        sample_ids_path=experiment.sample_ids_path(cell.dataset_id),
        split="validation",
        model_key=cell.model_key,
        checkpoint_path=resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell)),
        means_artifact=resolve_full_matrix_runtime_path(experiment.mean_path(cell)),
        output_directory=output_directory,
        batch_candidates=(1,),
        source_id="reference-full",
        device_name=device,
        cache_dir=resolve_full_matrix_runtime_path(experiment.cache_directory(cell.dataset_id)),
        seed=experiment.seed,
        dataloader_workers=experiment.training.workers,
        project_root=experiment.source_path.parents[3],
        candidate_methods=candidates,
        patch_size=16,
    )
    rosters = validate_method_rosters(experiment.methods)
    measurements = output.result.measurements.get("methods", [])
    passed = sorted(
        str(row["method"])
        for row in measurements
        if isinstance(row, Mapping) and bool(row.get("passed"))
    )
    expected = sorted(rosters[cell.architecture])
    status = "passed" if output.result.status == "passed" and passed == expected else "blocked"
    value: dict[str, Any] = {
        "schema": "simple-full-matrix-compatibility-gate-v1",
        "schema_version": 1,
        "matrix_digest": experiment.digest,
        "cell": cell.cell_id,
        "dataset": cell.dataset_id,
        "model": cell.model_key,
        "architecture": cell.architecture,
        "gate_type": "real_checkpoint",
        "status": status,
        "checkpoint_sha256": file_sha256(
            resolve_full_matrix_runtime_path(experiment.checkpoint_path(cell))
        ),
        "method_catalog_digest": experiment.methods.source_digest,
        "candidate_digest": object_sha256(list(candidate_payload)),
        "candidates": list(candidate_payload),
        "rank_patch_size": 16,
        "required_methods": expected,
        "passed_methods": passed,
        "blocked_methods": sorted(set(expected) - set(passed)),
        "pilot_status": output.result.status,
        "pilot_result_digest": output.result.digest,
        "pilot_result_path": str(output.result_path),
        "failures": list(output.result.failures),
    }
    value["gate_digest"] = object_sha256(value)
    atomic_write_json(output_directory / "gate.json", value)
    return value


__all__ = [
    "ASSET_JOB_KINDS",
    "CPU_JOB_KINDS",
    "GPU_JOB_KINDS",
    "checkpoint_complete",
    "compatibility_candidate_digest",
    "compatibility_candidate_payload",
    "compatibility_candidates",
    "compatibility_complete",
    "compatibility_job_id",
    "manifest_complete",
    "mean_complete",
    "planned_asset_jobs",
    "prepare_compatibility_samples",
    "run_compatibility_gate",
    "samples_complete",
    "static_compatibility_failures",
    "train_reference_model",
    "write_static_compatibility_gate",
]
