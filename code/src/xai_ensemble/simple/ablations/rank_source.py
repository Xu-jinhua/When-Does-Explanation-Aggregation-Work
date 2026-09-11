"""Uniform reader for existing Phase 2 ranks and newly constructed rank banks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..artifacts import PHASE2_SCHEMA_VERSION, ArtifactError, ArtifactStore, completed_manifest
from .artifacts import RANK_SCHEMA_VERSION, output_store
from .config import AblationExperiment, RankSourceSpec


@dataclass(frozen=True, slots=True)
class RankShard:
    shard_index: int
    start: int
    stop: int
    indices: Any
    labels: Any
    targets: Any
    predictions: Any
    ranks: Mapping[str, np.ndarray]


def source_store(experiment: AblationExperiment, source: RankSourceSpec) -> ArtifactStore:
    return ArtifactStore(experiment.base) if source.store == "base" else output_store(experiment)


def load_rank_manifest(
    experiment: AblationExperiment,
    source: RankSourceSpec,
) -> Mapping[str, Any]:
    store = source_store(experiment, source)
    schema = PHASE2_SCHEMA_VERSION if source.kind == "existing_phase2" else RANK_SCHEMA_VERSION
    manifest = completed_manifest(
        store,
        source.root,
        expected_task_digest=source.digest,
        expected_schema_version=schema,
    )
    if manifest is None:
        raise FileNotFoundError(f"Rank source is incomplete: {source.root}")
    expected = {
        "dataset": experiment.dataset_id,
        "model": experiment.model_id,
        "split": experiment.split,
        "condition": source.condition.condition_id,
        "patch_size": source.patch_size,
        "rank_base": 0,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"Rank source identity mismatch: {mismatches}")
    if (
        manifest.get("paper_rank_semantics")
        != "mean_over_patch_and_channels(abs(full_attribution))"
    ):
        raise ArtifactError("Rank source does not use the paper absolute attribution semantics")
    if source.kind == "existing_phase2":
        policy = manifest.get("rule_parameters")
        expected_policy = {
            "rrf_c": experiment.base.phase2.rrf_c,
            "kemeny_starts": experiment.base.phase2.kemeny_starts,
            "kemeny_max_passes": experiment.base.phase2.kemeny_max_passes,
        }
        if policy != expected_policy:
            raise ArtifactError(f"Existing rank source uses a different rule policy: {policy}")
    return manifest


def source_layout(manifest: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    try:
        return tuple(
            (int(item["shard_index"]), int(item["start"]), int(item["stop"]))
            for item in manifest["shards"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("Rank source has an invalid shard layout") from error


def load_rank_shard(
    experiment: AblationExperiment,
    source: RankSourceSpec,
    manifest: Mapping[str, Any],
    shard_index: int,
) -> RankShard:
    import torch

    try:
        record = next(
            item for item in manifest["shards"] if int(item["shard_index"]) == shard_index
        )
    except (KeyError, StopIteration, TypeError) as error:
        raise ArtifactError(f"Rank source has no shard {shard_index}") from error
    payload = record.get("payload")
    labels = record.get("rule_labels")
    if not isinstance(payload, Mapping) or not isinstance(labels, Mapping) or not labels:
        raise ArtifactError("Rank shard record is missing payload or rule labels")
    if len(set(str(value) for value in labels.values())) != len(labels):
        raise ArtifactError("Rank shard rule labels are not unique")
    scratch = (
        experiment.output_storage.scratch_root
        / "ablations"
        / "rank-source-cache"
        / str(os.getpid())
    )
    local = scratch / f"{source.digest[:12]}-{shard_index:05d}.safetensors"
    store = source_store(experiment, source)
    store.materialize(
        str(payload["relative_path"]),
        local,
        expected_sha256=str(payload["sha256"]),
    )
    from ..artifacts import load_safetensors

    try:
        fields = dict(load_safetensors(local))
    finally:
        local.unlink(missing_ok=True)
    prediction_key = "unmasked_predictions" if "unmasked_predictions" in fields else "predictions"
    required = {"indices", "labels", "targets", prediction_key}
    if not required <= set(fields):
        raise ArtifactError(f"Rank shard is missing fields: {sorted(required - set(fields))}")
    count = int(fields["indices"].shape[0])
    if count != int(record["stop"]) - int(record["start"]):
        raise ArtifactError("Rank shard size differs from its record")
    ranks = {}
    patch_count = (224 // source.patch_size) ** 2
    expected_permutation = np.arange(patch_count, dtype=np.int64)
    for field_id, rule_name in labels.items():
        key = f"rank__{field_id}"
        if key not in fields:
            raise ArtifactError(f"Rank shard is missing {key}")
        value = fields[key].numpy().astype(np.int64, copy=False)
        if value.shape != (count, patch_count):
            raise ArtifactError(f"Rank field {key} has invalid shape {value.shape}")
        if not np.array_equal(
            np.sort(value, axis=1), np.broadcast_to(expected_permutation, value.shape)
        ):
            raise ArtifactError(f"Rank field {key} is not a strict zero-based permutation")
        ranks[str(rule_name)] = value
    for name in required:
        if int(fields[name].shape[0]) != count:
            raise ArtifactError(f"Rank field {name} is not shard-aligned")
    for name in ("indices", "labels", "targets", prediction_key):
        if fields[name].dtype != torch.int64 or fields[name].ndim != 1:
            raise ArtifactError(f"Rank field {name} must be int64[N]")
    return RankShard(
        shard_index=shard_index,
        start=int(record["start"]),
        stop=int(record["stop"]),
        indices=fields["indices"],
        labels=fields["labels"],
        targets=fields["targets"],
        predictions=fields[prediction_key],
        ranks=ranks,
    )


__all__ = [
    "RankShard",
    "load_rank_manifest",
    "load_rank_shard",
    "source_layout",
    "source_store",
]
