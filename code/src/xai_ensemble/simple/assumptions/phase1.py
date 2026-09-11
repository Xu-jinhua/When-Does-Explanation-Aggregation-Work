"""Compact source-model rank inputs with reference-model targets and outputs."""

from __future__ import annotations

import fcntl
import gc
import json
import posixpath
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.phase1.relprop import (
    relprop_attribution_provider,
    relprop_required,
)

from ..artifacts import (
    PHASE1_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    load_safetensors,
    phase1_artifact_root,
    write_phase2_shard,
)
from ..config import Phase1Task, SimpleExperiment
from ..data import (
    fixed_baselines,
    load_model,
    load_raw_dataset_mean,
    load_relprop_model,
    load_split,
    materialize_model_inputs,
)
from ..manifest_identity import dataset_manifest_identity_sha256
from ..phase1 import (
    FULL_REFERENCE_CLEAN_TARGET_POLICY,
    FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
    _digest_path,
    _generate_variant,
    _model_output_field_descriptions,
    _Phase1Publisher,
)
from ..profiler import load_profile
from ..rank_ready import rank_field, simpleavg_score_field
from ..relprop_equivalence import ensure_relprop_equivalence_certificate
from ..runtime import emit_gpu_release_signal
from .artifacts import (
    SOURCE_SCOPE_SCHEMA_VERSION,
    completed_task_manifest,
    output_store,
    publish_manifest,
)
from .config import AssumptionExperiment, Cell, SourcePhase1Task
from .training import ensure_checkpoint

SOURCE_RANK_INPUT_SCHEMA_VERSION = 1
SOURCE_RANK_INPUT_REPRESENTATION = "p16-rank-and-simpleavg-patch-score"
SOURCE_RANK_INPUT_SCHEMA = "simple-assumptions-source-rank-input-v1"


def _generation_experiment(
    experiment: AssumptionExperiment,
    task: SourcePhase1Task,
    checkpoint: Path,
) -> SimpleExperiment:
    training = experiment.find_training_task(task.training_task_id)
    model = experiment.source_model(training, checkpoint)
    runtime = replace(
        experiment.base.runtime,
        profile_directory=experiment.base.runtime.profile_directory,
        database_path=experiment.runtime.database_path,
        log_directory=experiment.runtime.log_directory,
        gpu_ids=experiment.runtime.gpu_ids,
        seed=experiment.assignment_seed,
    )
    raw_config = {
        "schema": "simple-assumptions-source-generation-v2",
        "assumption_digest": experiment.digest,
        "scope_digest": task.digest,
        "phase1_identity_digest": object_sha256(
            {
                "assumption_digest": experiment.digest,
                "training_task": training.digest,
                "target_policy": FULL_REFERENCE_CLEAN_TARGET_POLICY,
                "model_output_source": FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
                "artifact_representation": SOURCE_RANK_INPUT_REPRESENTATION,
                "patch_size": 16,
                "simpleavg_normalization": experiment.base.phase2.simpleavg_normalization,
            }
        ),
        "storage": {
            "remote_root": experiment.storage.remote_root,
            "scratch_root": str(experiment.storage.scratch_root),
        },
        "runtime": {"seed": experiment.assignment_seed},
    }
    return SimpleExperiment(
        source_path=experiment.source_path,
        experiment_id=experiment.assumption_id,
        precision="fp32",
        methods=experiment.base.methods,
        storage=experiment.storage,
        runtime=runtime,
        datasets=(task.cell.dataset,),
        models=(model,),
        conditions=experiment.base.conditions,
        phase2=experiment.base.phase2,
        raw_config=raw_config,
    )


def _reference_task(
    experiment: AssumptionExperiment,
    cell: Cell,
    condition_id: str,
) -> Phase1Task:
    matches = [
        task
        for task in experiment.base.phase1_tasks()
        if task.dataset.dataset_id == cell.dataset.dataset_id
        and task.model.model_id == cell.reference_model.model_id
        and task.split == experiment.split
        and task.condition.condition_id == condition_id
        and task.family == "Saliency"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one reference Saliency source for {cell.cell_id}/{condition_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _reference_cache_identity(
    task: Phase1Task,
    variant: Any,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    source_identity_digest = _reference_source_identity_digest(manifest)
    return {
        "schema": "simple-assumptions-reference-fields-cache-v1",
        "task_id": task.task_id,
        "task_digest": task.digest,
        "variant_digest": variant.digest,
        "sample_count": manifest["sample_count"],
        "source_identity_digest": source_identity_digest,
        "shards": [
            {
                "shard_index": record["shard_index"],
                "start": record["start"],
                "stop": record["stop"],
                "sha256": record["payload"]["sha256"],
            }
            for record in manifest["shards"]
        ],
    }


def _reference_source_identity_digest(manifest: Mapping[str, Any]) -> str:
    """Read the stable identity shared by legacy and current Phase 1 manifests."""

    value = manifest.get("source_identity_digest")
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactError("Reference Phase 1 manifest has no valid source identity digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ArtifactError(
            "Reference Phase 1 source identity digest is not hexadecimal"
        ) from error

    nested = manifest.get("source_identity")
    if nested is not None:
        if not isinstance(nested, Mapping) or nested.get("digest") != value:
            raise ArtifactError("Reference Phase 1 source identity digest is contradictory")

    for record in manifest.get("shards", ()):
        if not isinstance(record, Mapping) or record.get("source_identity_digest") != value:
            raise ArtifactError("Reference Phase 1 shard source identity is contradictory")
    return value


def _validated_reference_values(
    values: Mapping[str, Any],
    *,
    sample_count: int,
) -> Mapping[str, Any]:
    import torch

    required = ("indices", "labels", "predictions", "logits", "targets")
    if set(values) != set(required):
        raise ArtifactError("Compact reference cache has unexpected fields")
    if any(int(values[name].shape[0]) != sample_count for name in required):
        raise ArtifactError("Compact reference cache does not cover the registered split")
    if any(values[name].ndim != 1 for name in ("indices", "labels", "predictions", "targets")):
        raise ArtifactError("Compact reference vector fields have invalid shapes")
    if values["logits"].ndim != 2 or not bool(torch.isfinite(values["logits"]).all()):
        raise ArtifactError("Compact reference logits are invalid")
    if not torch.equal(values["predictions"], values["logits"].argmax(dim=1)):
        raise ArtifactError("Compact reference predictions differ from argmax(logits)")
    return values


def _reference_fields(
    experiment: AssumptionExperiment,
    cell: Cell,
    condition_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Phase1Task]:
    import torch

    task = _reference_task(experiment, cell, condition_id)
    variant = task.variants[0]
    store = ArtifactStore(experiment.base)
    root = phase1_artifact_root(task, variant.artifact_name)
    manifest = completed_manifest(
        store,
        root,
        expected_task_digest=task.digest,
        expected_schema_version=PHASE1_SCHEMA_VERSION,
    )
    if manifest is None:
        raise FileNotFoundError(f"Reference Phase 1 artifact is incomplete: {root}")
    identity = _reference_cache_identity(task, variant, manifest)
    cache_root = experiment.storage.scratch_root / "reference-fields" / task.digest
    cache_path = cache_root / f"{variant.digest}.safetensors"
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    cache_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            cached = None
            if cache_path.is_file() and metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("identity") == identity and metadata.get(
                        "sha256"
                    ) == file_sha256(cache_path):
                        cached = load_safetensors(cache_path)
                except (OSError, json.JSONDecodeError):
                    cached = None
            if cached is not None:
                values = _validated_reference_values(
                    cached, sample_count=int(manifest["sample_count"])
                )
            else:
                names = ("indices", "labels", "predictions", "logits", "targets")
                chunks: dict[str, list[Any]] = {name: [] for name in names}
                transient = experiment.storage.spool_root / "reference-extract" / task.digest
                transient.mkdir(parents=True, exist_ok=True)
                for record in manifest["shards"]:
                    payload = record["payload"]
                    local = transient / Path(str(payload["relative_path"])).name
                    store.materialize(
                        str(payload["relative_path"]),
                        local,
                        expected_sha256=str(payload["sha256"]),
                    )
                    try:
                        fields = load_safetensors(local)
                        for name in names:
                            chunks[name].append(fields[name].clone())
                    finally:
                        local.unlink(missing_ok=True)
                values = _validated_reference_values(
                    {name: torch.cat(parts, dim=0) for name, parts in chunks.items()},
                    sample_count=int(manifest["sample_count"]),
                )
                write_phase2_shard(
                    cache_path,
                    tensors=values,
                    metadata={
                        "schema": "simple-assumptions-reference-fields-cache-v1",
                        "identity_digest": object_sha256(identity),
                    },
                )
                atomic_write_json(
                    metadata_path,
                    {"identity": identity, "sha256": file_sha256(cache_path)},
                )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return values, manifest, task


def _source_identity(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    task: Phase1Task,
    *,
    training_manifest: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    reference_task: Phase1Task,
) -> Mapping[str, str]:
    assert task.model.checkpoint_path is not None
    values: dict[str, str] = {
        "source_id": scope.source_id,
        "family_id": str(scope.family_id),
        "dataset_manifest_sha256": dataset_manifest_identity_sha256(task.dataset.manifest_path),
        "checkpoint_sha256": file_sha256(
            resolve_full_matrix_runtime_path(task.model.checkpoint_path)
        ),
        "mean_artifact_digest": _digest_path(task.model.mean_path),
        "source_training_task_digest": str(training_manifest["task_digest"]),
        "source_phase1_scope_digest": scope.digest,
        "reference_phase1_task_digest": reference_task.digest,
        "reference_phase1_manifest_task_digest": str(reference_manifest["task_digest"]),
        "target_policy": FULL_REFERENCE_CLEAN_TARGET_POLICY,
        "model_output_source": FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
        "artifact_representation": SOURCE_RANK_INPUT_REPRESENTATION,
        "patch_size": "16",
        "simpleavg_normalization": experiment.base.phase2.simpleavg_normalization,
    }
    provider = relprop_attribution_provider(task.family, task.model.architecture)
    values.update({f"attribution_provider_{key}": str(value) for key, value in provider.items()})
    return {**values, "digest": object_sha256(values)}


def _base_profile(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    variant: Any,
) -> tuple[Any, int]:
    provider = relprop_attribution_provider(variant.family, scope.cell.reference_model.architecture)
    candidates = [
        profile
        for profile in experiment.base.profiles()
        if profile.model_key == scope.cell.reference_model.model_key
        and profile.architecture == scope.cell.reference_model.architecture
        and profile.method.digest == variant.digest
        and (profile.model_id is None or profile.model_id == scope.cell.reference_model.model_id)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot resolve one completed base profile for {scope.cell.cell_id}/"
            f"{variant.artifact_name}; found {len(candidates)}"
        )
    profile = candidates[0]
    result = load_profile(experiment.base, profile)
    if result is None:
        raise FileNotFoundError(f"Missing main-run profile {profile.profile_id}")
    cap = experiment.runtime.phase1_batch_caps.get(variant.family)
    if cap is None or result.selected_batch_size <= cap:
        return profile, int(result.selected_batch_size)
    valid = [
        measurement.batch_size
        for measurement in result.measurements
        if measurement.passed and measurement.batch_size <= cap
    ]
    if not valid:
        raise ValueError(f"Profile {profile.profile_id} has no successful batch <= {cap}")
    if provider and profile.model_id != scope.cell.reference_model.model_id:
        raise ArtifactError("RelProp profile belongs to another reference checkpoint")
    return profile, max(valid)


def compact_artifact_root(task: Phase1Task, artifact_name: str) -> str:
    return posixpath.join(
        "source-rank-inputs",
        task.dataset.dataset_id,
        task.model.model_id,
        task.split,
        task.condition.condition_id,
        artifact_name,
        task.digest,
    )


def _compact_manifest_value(
    experiment: SimpleExperiment,
    task: Phase1Task,
    variant: Any,
    *,
    records: list[Mapping[str, Any]],
    input_shape: tuple[int, int, int],
    checkpoint_sha256: str,
    mean_digest: str,
    source_identity: Mapping[str, str],
    dataset_manifest_sha256: str,
    profile_id: str,
    batch_size: int,
    attribution_provider: Mapping[str, Any],
    target_policy: str,
    model_output_source: str,
) -> Mapping[str, Any]:
    prediction_description, logit_description = _model_output_field_descriptions(
        model_output_source
    )
    patch_size = 16
    patch_count = (input_shape[-2] // patch_size) * (input_shape[-1] // patch_size)
    return {
        "schema": SOURCE_RANK_INPUT_SCHEMA,
        "schema_version": SOURCE_RANK_INPUT_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment.experiment_id,
        "experiment_digest": experiment.digest,
        "task_id": task.task_id,
        "task_digest": task.digest,
        "source_identity_digest": source_identity["digest"],
        "source_identity": dict(source_identity),
        "dataset": {
            "id": task.dataset.dataset_id,
            "manifest_sha256": dataset_manifest_sha256,
            "split": task.split,
            "condition": task.condition.condition_id,
        },
        "model": {
            "id": task.model.model_id,
            "model_key": task.model.model_key,
            "architecture": task.model.architecture,
            "checkpoint_sha256": checkpoint_sha256,
            "attribution_provider": dict(attribution_provider),
        },
        "method": {
            "family": variant.family,
            "artifact_name": variant.artifact_name,
            "variant": variant.variant,
            "variant_digest": variant.digest,
            "params": dict(variant.params),
        },
        "precision": "fp32",
        "runtime_profile": {"profile_id": profile_id, "batch_size": batch_size},
        "target_policy": target_policy,
        "model_output_source": model_output_source,
        "representation": SOURCE_RANK_INPUT_REPRESENTATION,
        "full_attribution_retained": False,
        "transient_attribution": "float32[N,C,H,W] discarded after compact reduction",
        "patch_size": patch_size,
        "patch_count": patch_count,
        "rank_base": 0,
        "tie_break": "stable_row_major_patch_index",
        "paper_rank_semantics": "mean_over_patch_and_channels(abs(full_attribution))",
        "simpleavg_semantics": {
            "channel_reduction": "mean(abs(attribution), channels)",
            "per_method_spatial_normalization": experiment.phase2.simpleavg_normalization,
            "patch_reduction": "arithmetic_mean",
            "method_reduction": "arithmetic_mean_across_saved_patch_scores",
        },
        "baseline": {
            "mean_artifact_digest": mean_digest,
            "zero_space": "model_input",
            "gaussian_space": "model_input",
            "gaussian_distribution": "N(0,1)",
            "gaussian_seed": experiment.runtime.seed,
            "distribution_members": ["zero", "gaussian", "dataset_mean"],
        },
        "safetensors_fields": {
            "indices": "int64[N] provider row indices",
            "labels": "int64[N]",
            "predictions": prediction_description,
            "logits": logit_description,
            "targets": "int64[N] fixed explanation targets",
            rank_field(patch_size): "int32[N,196] strict zero-based patch ranks",
            simpleavg_score_field(
                patch_size
            ): "float32[N,196] normalized per-method SimpleAvg patch scores",
        },
        "sample_count": sum(int(record["stop"]) - int(record["start"]) for record in records),
        "shard_size": experiment.runtime.shard_size,
        "shards": records,
    }


def _validate_compact_manifest(
    manifest: Mapping[str, Any],
    *,
    variant: Any,
    source_identity: Mapping[str, str],
    simpleavg_normalization: str,
) -> None:
    method = manifest.get("method")
    source = manifest.get("source_identity")
    semantics = manifest.get("simpleavg_semantics")
    fields = manifest.get("safetensors_fields")
    checks = {
        "schema": (manifest.get("schema"), SOURCE_RANK_INPUT_SCHEMA),
        "schema_version": (
            manifest.get("schema_version"),
            SOURCE_RANK_INPUT_SCHEMA_VERSION,
        ),
        "representation": (
            manifest.get("representation"),
            SOURCE_RANK_INPUT_REPRESENTATION,
        ),
        "full_attribution_retained": (manifest.get("full_attribution_retained"), False),
        "variant_digest": (
            method.get("variant_digest") if isinstance(method, Mapping) else None,
            variant.digest,
        ),
        "source_identity_digest": (
            manifest.get("source_identity_digest"),
            source_identity["digest"],
        ),
        "target_policy": (
            manifest.get("target_policy"),
            FULL_REFERENCE_CLEAN_TARGET_POLICY,
        ),
        "model_output_source": (
            manifest.get("model_output_source"),
            FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
        ),
        "patch_size": (manifest.get("patch_size"), 16),
        "patch_count": (manifest.get("patch_count"), 196),
        "simpleavg_normalization": (
            semantics.get("per_method_spatial_normalization")
            if isinstance(semantics, Mapping)
            else None,
            simpleavg_normalization,
        ),
        "rank_field": (
            rank_field(16) in fields if isinstance(fields, Mapping) else False,
            True,
        ),
        "simpleavg_score_field": (
            simpleavg_score_field(16) in fields if isinstance(fields, Mapping) else False,
            True,
        ),
        "attributions_absent": (
            "attributions" not in fields if isinstance(fields, Mapping) else False,
            True,
        ),
    }
    for name in (
        "source_id",
        "family_id",
        "dataset_manifest_sha256",
        "checkpoint_sha256",
        "mean_artifact_digest",
        "target_policy",
        "model_output_source",
    ):
        if name in source_identity:
            checks[f"source_identity.{name}"] = (
                source.get(name) if isinstance(source, Mapping) else None,
                source_identity[name],
            )
    mismatches = {
        name: {"artifact": observed, "current": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise ArtifactError(f"Compact source artifact identity changed: {mismatches}")


def source_scope_complete(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
) -> bool:
    scope_manifest = completed_task_manifest(
        experiment, scope, schema_version=SOURCE_SCOPE_SCHEMA_VERSION
    )
    training = experiment.find_training_task(scope.training_task_id)
    try:
        checkpoint, training_manifest = ensure_checkpoint(experiment, training)
    except FileNotFoundError:
        return False
    generated = _generation_experiment(experiment, scope, checkpoint)
    try:
        _, reference_manifest, reference_task = _reference_fields(
            experiment, scope.cell, scope.condition.condition_id
        )
    except FileNotFoundError:
        return False
    store = ArtifactStore(generated)
    method_tasks = experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
    identities = {
        task.task_id: _source_identity(
            experiment,
            scope,
            task,
            training_manifest=training_manifest,
            reference_manifest=reference_manifest,
            reference_task=reference_task,
        )
        for task in method_tasks
    }
    completed_values = _completed_method_manifests(
        experiment,
        method_tasks,
        identities=identities,
        store=store,
    )
    if completed_values is None or scope_manifest is None:
        return False
    _validate_scope_manifest(
        scope_manifest,
        scope=scope,
        method_artifacts=_scope_method_artifacts(method_tasks, completed_values),
    )
    return True


def _completed_method_manifests(
    experiment: AssumptionExperiment,
    method_tasks: tuple[Phase1Task, ...],
    *,
    identities: Mapping[str, Mapping[str, str]],
    store: ArtifactStore,
) -> tuple[Mapping[str, Any], ...] | None:
    completed_values = []
    for task in method_tasks:
        identity = identities[task.task_id]
        for variant in task.variants:
            manifest = completed_manifest(
                store,
                compact_artifact_root(task, variant.artifact_name),
                expected_task_digest=task.digest,
                expected_schema_version=SOURCE_RANK_INPUT_SCHEMA_VERSION,
            )
            if manifest is None:
                return None
            _validate_compact_manifest(
                manifest,
                variant=variant,
                source_identity=identity,
                simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
            )
            completed_values.append(manifest)
    return tuple(completed_values)


def _scope_method_artifacts(
    method_tasks: tuple[Phase1Task, ...],
    manifests: tuple[Mapping[str, Any], ...],
) -> list[Mapping[str, Any]]:
    tasks = {task.task_id: task for task in method_tasks}
    return [
        {
            "task_id": item["task_id"],
            "task_digest": item["task_digest"],
            "method": item["method"]["artifact_name"],
            "root": compact_artifact_root(
                tasks[str(item["task_id"])],
                str(item["method"]["artifact_name"]),
            ),
            "representation": SOURCE_RANK_INPUT_REPRESENTATION,
        }
        for item in manifests
    ]


def _scope_manifest_value(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    method_tasks: tuple[Phase1Task, ...],
    manifests: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    return {
        "schema_version": SOURCE_SCOPE_SCHEMA_VERSION,
        "status": "complete",
        "assumption_id": experiment.assumption_id,
        "assumption_digest": experiment.digest,
        "task_id": scope.task_id,
        "task_digest": scope.digest,
        "cell": scope.cell.cell_id,
        "source_id": scope.source_id,
        "family_id": scope.family_id,
        "condition": scope.condition.condition_id,
        "artifact_representation": SOURCE_RANK_INPUT_REPRESENTATION,
        "method_artifacts": _scope_method_artifacts(method_tasks, manifests),
    }


def _validate_scope_manifest(
    manifest: Mapping[str, Any],
    *,
    scope: SourcePhase1Task,
    method_artifacts: list[Mapping[str, Any]],
) -> None:
    expected = {
        "cell": scope.cell.cell_id,
        "source_id": scope.source_id,
        "family_id": scope.family_id,
        "condition": scope.condition.condition_id,
        "artifact_representation": SOURCE_RANK_INPUT_REPRESENTATION,
        "method_artifacts": method_artifacts,
    }
    mismatches = {
        key: {"artifact": manifest.get(key), "current": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Compact source scope manifest changed: {mismatches}")


def _publish_scope_if_complete(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    *,
    checkpoint: Path,
    training_manifest: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    reference_task: Phase1Task,
    store: ArtifactStore,
) -> Mapping[str, Any] | None:
    method_tasks = experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
    identities = {
        task.task_id: _source_identity(
            experiment,
            scope,
            task,
            training_manifest=training_manifest,
            reference_manifest=reference_manifest,
            reference_task=reference_task,
        )
        for task in method_tasks
    }
    manifests = _completed_method_manifests(
        experiment,
        method_tasks,
        identities=identities,
        store=store,
    )
    if manifests is None:
        return None
    value = _scope_manifest_value(experiment, scope, method_tasks, manifests)
    existing = completed_task_manifest(
        experiment,
        scope,
        schema_version=SOURCE_SCOPE_SCHEMA_VERSION,
    )
    if existing is not None:
        _validate_scope_manifest(
            existing,
            scope=scope,
            method_artifacts=list(value["method_artifacts"]),
        )
        return existing
    publish_manifest(
        experiment,
        output_store(experiment),
        root=scope.artifact_root,
        task_id=scope.task_id,
        manifest=value,
    )
    return value


def _method_task(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    *,
    checkpoint: Path,
    family: str,
) -> Phase1Task:
    matches = tuple(
        task
        for task in experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
        if task.family == family
    )
    if len(matches) != 1:
        raise KeyError(
            f"Expected one source method task for {scope.task_id}/{family}; found {len(matches)}"
        )
    return matches[0]


def source_method_complete(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    family: str,
) -> bool:
    training = experiment.find_training_task(scope.training_task_id)
    try:
        checkpoint, training_manifest = ensure_checkpoint(experiment, training)
        _, reference_manifest, reference_task = _reference_fields(
            experiment, scope.cell, scope.condition.condition_id
        )
    except FileNotFoundError:
        return False
    task = _method_task(experiment, scope, checkpoint=checkpoint, family=family)
    generated = _generation_experiment(experiment, scope, checkpoint)
    identity = _source_identity(
        experiment,
        scope,
        task,
        training_manifest=training_manifest,
        reference_manifest=reference_manifest,
        reference_task=reference_task,
    )
    return (
        _completed_method_manifests(
            experiment,
            (task,),
            identities={task.task_id: identity},
            store=ArtifactStore(generated),
        )
        is not None
    )


def _run_source_phase1_methods(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    method_tasks: tuple[Phase1Task, ...],
    *,
    device: str = "cuda:0",
) -> tuple[Mapping[str, Any], ...]:
    """Generate an explicit subset while retaining the full-run artifact identities."""

    import torch

    if not method_tasks:
        raise ValueError("Source Phase 1 requires at least one method task")
    training = experiment.find_training_task(scope.training_task_id)
    checkpoint, training_manifest = ensure_checkpoint(experiment, training)
    generated = _generation_experiment(experiment, scope, checkpoint)
    reference, reference_manifest, reference_task = _reference_fields(
        experiment, scope.cell, scope.condition.condition_id
    )
    store = ArtifactStore(generated)
    identities = {
        task.task_id: _source_identity(
            experiment,
            scope,
            task,
            training_manifest=training_manifest,
            reference_manifest=reference_manifest,
            reference_task=reference_task,
        )
        for task in method_tasks
    }
    completed_values = _completed_method_manifests(
        experiment,
        method_tasks,
        identities=identities,
        store=store,
    )
    if completed_values is not None:
        _publish_scope_if_complete(
            experiment,
            scope,
            checkpoint=checkpoint,
            training_manifest=training_manifest,
            reference_manifest=reference_manifest,
            reference_task=reference_task,
            store=store,
        )
        return completed_values

    from ..runtime_dependencies import require_relprop_runtime

    require_relprop_runtime(
        (task.family, scope.cell.reference_model.architecture) for task in method_tasks
    )

    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Source Phase 1 requires CUDA")
    loaded = load_model(method_tasks[0].model, device=target_device, include_checkpoint=True)
    bundle = load_split(
        scope.cell.dataset,
        loaded,
        split=experiment.split,
        workers=experiment.base.runtime.dataloader_workers,
        shared_cache_root=experiment.storage.spool_root / "shared-cache",
    )
    if not torch.equal(bundle.indices.to(torch.int64), reference["indices"].to(torch.int64)):
        raise ArtifactError("Reference outputs and source-model dataset rows are not aligned")
    if not torch.equal(bundle.labels.to(torch.int64), reference["labels"].to(torch.int64)):
        raise ArtifactError("Reference outputs and source-model labels are not aligned")

    if scope.condition.kind == "adversarial":
        from ..adversarial import materialize_adversarial_model_inputs

        adversarial = materialize_adversarial_model_inputs(
            experiment.base,
            reference_task,
            bundle,
            loaded,
            store=ArtifactStore(experiment.base),
        )
        model_inputs = adversarial.model_inputs
        if not torch.equal(adversarial.targets, reference["targets"]):
            raise ArtifactError("Adversarial target artifact differs from reference Phase 1")
    else:
        adversarial = None
        model_inputs = materialize_model_inputs(
            bundle,
            loaded,
            scope.condition,
            device=target_device,
            batch_size=experiment.base.runtime.prediction_batch_size,
            seed=experiment.base.runtime.seed,
            shared_cache_root=experiment.storage.spool_root / "shared-cache",
        )

    raw_mean = load_raw_dataset_mean(
        method_tasks[0].model,
        input_size=int(loaded.preprocessing["input_size"]),
    )
    zero, distribution = fixed_baselines(
        raw_mean,
        loaded,
        device=target_device,
        seed=experiment.base.runtime.seed,
    )
    publishers: list[_Phase1Publisher] = []
    futures: list[Future[Mapping[str, Any]]] = []
    publisher: _Phase1Publisher | None = None
    try:
        publisher = _Phase1Publisher(
            generated,
            method_tasks[0],
            store,
            compact_patch_size=16,
            artifact_schema_version=SOURCE_RANK_INPUT_SCHEMA_VERSION,
        )
        publishers.append(publisher)
        for task in method_tasks:
            publisher.task = task
            attribution_model = loaded.model
            attribution_provider: Mapping[str, Any] = {"kind": "timm"}
            relprop_model = None
            if relprop_required(task.family, task.model.architecture):
                relprop_model = load_relprop_model(
                    task.model, method=task.family, device=target_device
                )
                if object_sha256(relprop_model.preprocessing) != object_sha256(
                    loaded.preprocessing
                ):
                    raise ArtifactError("Source RelProp preprocessing differs from timm")
                equivalence = ensure_relprop_equivalence_certificate(
                    generated,
                    task.model,
                    method=task.family,
                    reference_model=loaded.model,
                    relprop_model=relprop_model.model,
                    preprocessing=loaded.preprocessing,
                    checkpoint_sha256=identities[task.task_id]["checkpoint_sha256"],
                    device=target_device,
                )
                attribution_model = relprop_model.model
                attribution_provider = {
                    **relprop_attribution_provider(task.family, task.model.architecture),
                    "equivalence_certificate": equivalence,
                }
            for variant in task.variants:
                profile, batch_size = _base_profile(experiment, scope, variant)
                futures.append(
                    _generate_variant(
                        generated,
                        task,
                        variant,
                        store=store,
                        publisher=publisher,
                        model=attribution_model,
                        model_inputs=model_inputs,
                        labels=bundle.labels,
                        indices=bundle.indices,
                        predictions=reference["predictions"],
                        logits=reference["logits"],
                        targets=reference["targets"],
                        baseline=zero,
                        distribution=distribution,
                        device=target_device,
                        batch_size=batch_size,
                        profile_id=profile.profile_id,
                        source_identity=identities[task.task_id],
                        checkpoint_sha256=identities[task.task_id]["checkpoint_sha256"],
                        mean_digest=identities[task.task_id]["mean_artifact_digest"],
                        attribution_provider=attribution_provider,
                        target_policy=FULL_REFERENCE_CLEAN_TARGET_POLICY,
                        model_output_source=FULL_REFERENCE_MODEL_OUTPUT_SOURCE,
                        artifact_root=compact_artifact_root(task, variant.artifact_name),
                        artifact_schema_version=SOURCE_RANK_INPUT_SCHEMA_VERSION,
                        completed_validator=lambda manifest, current_variant=variant, current_identity=identities[task.task_id]: (
                            _validate_compact_manifest(
                                manifest,
                                variant=current_variant,
                                source_identity=current_identity,
                                simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
                            )
                        ),
                        manifest_builder=_compact_manifest_value,
                    )
                )
            relprop_model = None
            attribution_model = None
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        adversarial = None
        model_inputs = None
        raw_mean = None
        zero = None
        distribution = None
        bundle = None
        loaded = None
        reference = {}
        torch.cuda.synchronize(target_device)
        gc.collect()
        torch.cuda.empty_cache()
        emit_gpu_release_signal()
        print(
            f"ASSUMPTIONS_PHASE1_GPU_RELEASED task={scope.task_id} "
            "pending_publications="
            f"{sum(not future.done() for future in publisher.futures) if publisher else 0}",
            flush=True,
        )
    try:
        if publisher is None:
            raise RuntimeError("Source Phase 1 publisher was not initialized")
        manifests = tuple(future.result() for future in futures)
        publisher.check()
        _publish_scope_if_complete(
            experiment,
            scope,
            checkpoint=checkpoint,
            training_manifest=training_manifest,
            reference_manifest=reference_manifest,
            reference_task=reference_task,
            store=store,
        )
        return manifests
    finally:
        for item in publishers:
            item.shutdown()


def run_source_phase1_method_task(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    family: str,
    *,
    device: str = "cuda:0",
) -> tuple[Mapping[str, Any], ...]:
    """Generate one assigned method without claiming the full source scope."""

    training = experiment.find_training_task(scope.training_task_id)
    checkpoint, _ = ensure_checkpoint(experiment, training)
    task = _method_task(experiment, scope, checkpoint=checkpoint, family=family)
    return _run_source_phase1_methods(experiment, scope, (task,), device=device)


def run_source_phase1_task(
    experiment: AssumptionExperiment,
    scope: SourcePhase1Task,
    *,
    device: str = "cuda:0",
) -> tuple[Mapping[str, Any], ...]:
    """Generate all compatible methods for one source model and condition."""

    training = experiment.find_training_task(scope.training_task_id)
    checkpoint, _ = ensure_checkpoint(experiment, training)
    method_tasks = experiment.method_phase1_tasks(scope, checkpoint_path=checkpoint)
    _run_source_phase1_methods(experiment, scope, method_tasks, device=device)
    manifest = completed_task_manifest(
        experiment,
        scope,
        schema_version=SOURCE_SCOPE_SCHEMA_VERSION,
    )
    if manifest is None:
        raise RuntimeError("All source methods finished without a complete scope manifest")
    return (manifest,)


__all__ = [
    "run_source_phase1_method_task",
    "run_source_phase1_task",
    "source_method_complete",
    "source_scope_complete",
]
