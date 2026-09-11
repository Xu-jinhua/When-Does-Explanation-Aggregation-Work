"""Read-only input catalog and shard loader for the NOISE prefix sweep."""

from __future__ import annotations

import fcntl
import json
import posixpath
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.phase2.selection import order_methods_by_fidelity

from ..artifacts import (
    PHASE2_SCHEMA_VERSION,
    ArtifactError,
    ArtifactStore,
    completed_manifest,
    phase1_artifact_root,
    phase2_artifact_root,
)
from ..assumptions.artifacts import completed_selection_manifest
from ..methods import PATCH_METHODS
from ..phase2 import _phase1_task, _source_manifests
from ..rank_ready import (
    RankReadyPublisher,
    ensure_rank_ready_sidecar,
    existing_rank_ready_payload,
    load_existing_rank_ready_sidecar,
    rank_field,
    rank_ready_descriptor,
    simpleavg_spatial_field,
)
from .config import RULE_IDS, NoisePrefixExperiment, PrefixEvaluationTask

CATALOG_SCHEMA_VERSION = 1
CATALOG_SCHEMA = "simple-noise-prefix-input-catalog-v1"


@dataclass(frozen=True, slots=True)
class PrefixShardInput:
    reference: Mapping[str, Any]
    ballots: np.ndarray
    method_patch_scores: np.ndarray
    simple_scores_by_q: Mapping[int, np.ndarray]
    sidecars: tuple[Mapping[str, Any], ...]


class _ManifestSnapshotStore:
    """Serve manifest reads from one recursive metadata snapshot."""

    def __init__(
        self,
        store: ArtifactStore,
        manifests: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._store = store
        self._manifests = manifests

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def exists(self, relative_path: str) -> bool:
        if relative_path.endswith("/manifest.json"):
            return relative_path in self._manifests
        return self._store.exists(relative_path)

    def read_json(self, relative_path: str) -> Mapping[str, Any]:
        if relative_path.endswith("/manifest.json"):
            try:
                return self._manifests[relative_path]
            except KeyError as error:
                raise FileNotFoundError(relative_path) from error
        return self._store.read_json(relative_path)


def _manifest_snapshot(
    store: ArtifactStore,
    *,
    relative_paths: Sequence[str],
    workers: int = 16,
) -> Mapping[str, Mapping[str, Any]]:
    paths = tuple(sorted(set(relative_paths)))
    if not paths or workers <= 0:
        raise ValueError("Manifest snapshot paths and workers must be non-empty")

    def read(relative_path: str) -> tuple[str, Mapping[str, Any]]:
        return relative_path, store.read_json(relative_path)

    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
        return dict(executor.map(read, paths))


def _required_base_manifest_paths(
    experiment: NoisePrefixExperiment,
) -> tuple[str, ...]:
    paths = set()
    for task in experiment.evaluation_tasks():
        source_task = experiment.base_phase2_task(task.cell, task.condition.condition_id)
        paths.add(posixpath.join(phase2_artifact_root(source_task), "manifest.json"))
        for method in task.cell.methods:
            producer = _phase1_task(experiment.base, source_task, method)
            artifact_name = (
                f"{method}__p{source_task.patch_size}" if method in PATCH_METHODS else method
            )
            paths.add(
                posixpath.join(phase1_artifact_root(producer, artifact_name), "manifest.json")
            )
    return tuple(sorted(paths))


def _rank_ready_requirements(
    experiment: NoisePrefixExperiment,
    manifests: _ManifestSnapshotStore,
) -> Mapping[str, Mapping[str, Any]]:
    requirements = {}
    for task in experiment.evaluation_tasks():
        source_task = experiment.base_phase2_task(task.cell, task.condition.condition_id)
        sources = _source_manifests(
            experiment.base,
            source_task,
            manifests,  # type: ignore[arg-type]
            task.cell.methods,
        )
        for method, (_, manifest) in sources.items():
            for record in manifest["shards"]:
                source_payload = dict(record["payload"])
                descriptor = rank_ready_descriptor(
                    str(source_payload["sha256"]),
                    simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
                )
                row = {
                    "source_payload": source_payload,
                    "count": int(record["stop"]) - int(record["start"]),
                    "cell": task.cell.cell_id,
                    "condition": task.condition.condition_id,
                    "method": method,
                    "shard_index": int(record["shard_index"]),
                }
                previous = requirements.setdefault(descriptor.identity_digest, row)
                if (
                    previous["source_payload"] != source_payload
                    or previous["count"] != row["count"]
                ):
                    raise ArtifactError("One rank-ready identity maps to contradictory sources")
    return requirements


def materialize_missing_rank_ready(
    experiment: NoisePrefixExperiment,
) -> Mapping[str, Any]:
    """Explicitly derive legacy sidecars before the read-only formal sweep."""

    store = ArtifactStore(experiment.base)
    print("NOISE_PREFIX_SIDECARS stage=rank_ready_inventory", flush=True)
    inventory = _remote_rank_ready_inventory(store)
    paths = _required_base_manifest_paths(experiment)
    print(
        f"NOISE_PREFIX_SIDECARS stage=base_manifest_snapshot manifests={len(paths)}",
        flush=True,
    )
    snapshot = _manifest_snapshot(store, relative_paths=paths)
    manifests = _ManifestSnapshotStore(store, snapshot)
    requirements = _rank_ready_requirements(experiment, manifests)
    missing = []
    for identity_digest, row in requirements.items():
        descriptor = rank_ready_descriptor(
            str(row["source_payload"]["sha256"]),
            simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
        )
        if (
            descriptor.relative_path in inventory
            and f"{descriptor.relative_path}.receipt.json" in inventory
        ):
            continue
        try:
            existing_rank_ready_payload(
                store,
                source_payload=row["source_payload"],
                simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
            )
        except FileNotFoundError:
            missing.append((identity_digest, row))
    print(
        "NOISE_PREFIX_SIDECARS "
        f"stage=missing requirements={len(requirements)} missing={len(missing)}",
        flush=True,
    )
    if not missing:
        return {
            "status": "complete",
            "requirements": len(requirements),
            "generated": 0,
        }

    publisher = RankReadyPublisher(
        spool_root=experiment.storage.spool_root,
        spool_max_bytes=experiment.storage.spool_max_bytes,
        spool_min_free_bytes=experiment.storage.spool_min_free_bytes,
        namespace=f"noise-prefix-preflight-{experiment.digest[:16]}",
    )
    generated = 0
    try:
        with TemporaryDirectory(
            prefix="noise-prefix-sidecars-",
            dir=experiment.storage.spool_root,
        ) as temporary:
            work_root = Path(temporary)
            for position, (_, row) in enumerate(missing, start=1):
                compact = ensure_rank_ready_sidecar(
                    store,
                    source_payload=row["source_payload"],
                    work_directory=work_root / f"item-{position:05d}",
                    lock_root=experiment.storage.spool_root / "rank-ready-locks",
                    simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
                    count=int(row["count"]),
                    publisher=publisher,
                )
                generated += int(compact.generated)
                del compact
                publisher.check()
                print(
                    "NOISE_PREFIX_SIDECARS "
                    f"generated={position}/{len(missing)} cell={row['cell']} "
                    f"condition={row['condition']} method={row['method']} "
                    f"shard={row['shard_index']}",
                    flush=True,
                )
    finally:
        publisher.shutdown()
    return {
        "status": "complete",
        "requirements": len(requirements),
        "missing_before": len(missing),
        "generated": generated,
    }


def _sha256(hashes: Mapping[str, Any]) -> str | None:
    for name, value in hashes.items():
        if str(name).lower().replace("-", "").replace("_", "") == "sha256":
            return str(value).lower()
    return None


def _remote_rank_ready_inventory(store: ArtifactStore) -> Mapping[str, Mapping[str, Any]]:
    prefix = "derived/rank-ready-v2"
    if store.remote:
        command = (
            str(store.rclone),
            "lsjson",
            store.locator(prefix),
            "--recursive",
            "--files-only",
            "--hash",
        )
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", "") or ""
            raise ArtifactError(
                f"Cannot inventory rank-ready sidecars: {detail.strip()}"
            ) from error
        rows = json.loads(result.stdout)
        if not isinstance(rows, list):
            raise ArtifactError("rclone rank-ready inventory is not a list")
        inventory = {}
        for row in rows:
            if not isinstance(row, Mapping) or bool(row.get("IsDir")):
                continue
            relative = f"{prefix}/{str(row['Path']).lstrip('/')}"
            inventory[relative] = {
                "size_bytes": int(row["Size"]),
                "sha256": _sha256(row.get("Hashes", {})),
            }
        return inventory

    root = Path(store.locator(prefix))
    if not root.is_dir():
        raise FileNotFoundError(f"Rank-ready directory is missing: {root}")
    return {
        f"{prefix}/{path.relative_to(root).as_posix()}": {"size_bytes": path.stat().st_size}
        for path in root.rglob("*")
        if path.is_file()
    }


def _sidecar_from_inventory(
    store: ArtifactStore,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    source_payload: Mapping[str, Any],
    normalization: str,
) -> Mapping[str, Any]:
    descriptor = rank_ready_descriptor(
        str(source_payload["sha256"]),
        simpleavg_normalization=normalization,
    )
    receipt_path = f"{descriptor.relative_path}.receipt.json"
    payload_row = inventory.get(descriptor.relative_path)
    receipt_row = inventory.get(receipt_path)
    if payload_row is None or receipt_row is None:
        _, direct = existing_rank_ready_payload(
            store,
            source_payload=source_payload,
            simpleavg_normalization=normalization,
        )
        return {
            **dict(direct),
            "receipt_relative_path": receipt_path,
            "receipt_verified_by": "direct_receipt_validation",
        }
    if store.remote:
        digest = payload_row.get("sha256")
        size = int(payload_row.get("size_bytes", -1))
    else:
        receipt = store.read_json(receipt_path)
        digest = receipt.get("sha256")
        size = int(receipt.get("size_bytes", -1))
    if not isinstance(digest, str) or len(digest) != 64 or size <= 0:
        raise ArtifactError(
            f"Rank-ready inventory lacks SHA-256 metadata: {descriptor.relative_path}"
        )
    return {
        "relative_path": descriptor.relative_path,
        "sha256": digest.lower(),
        "size_bytes": size,
        "identity_digest": descriptor.identity_digest,
        "receipt_relative_path": receipt_path,
        "receipt_verified_by": "remote_recursive_inventory",
    }


def _base_phase2_manifest(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
    store: ArtifactStore,
) -> tuple[Any, str, Mapping[str, Any]]:
    source_task = experiment.base_phase2_task(task.cell, task.condition.condition_id)
    root = phase2_artifact_root(source_task)
    manifest = completed_manifest(
        store,
        root,
        expected_task_digest=source_task.digest,
        expected_schema_version=PHASE2_SCHEMA_VERSION,
    )
    if manifest is None:
        raise FileNotFoundError(f"Required NAIVE p=16 Phase 2 artifact is incomplete: {root}")
    expected = {
        "dataset": task.cell.dataset.dataset_id,
        "model": task.cell.reference_model.model_id,
        "split": experiment.split,
        "condition": task.condition.condition_id,
        "patch_size": experiment.patch_size,
        "k": experiment.k,
        "rank_base": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ArtifactError(f"NAIVE p=16 source identity mismatch: {mismatches}")
    if manifest.get("methods") != list(task.cell.methods):
        raise ArtifactError(f"NAIVE method roster mismatch for {task.cell.cell_id}")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ArtifactError("NAIVE p=16 source has no metric summary")
    required = {*RULE_IDS, *(f"single__{method}" for method in task.cell.methods)}
    if not required <= set(metrics):
        raise ArtifactError(f"NAIVE p=16 source lacks metrics: {sorted(required - set(metrics))}")
    return source_task, root, manifest


def _fidelity_order(
    cell: Any,
    clean_manifest: Mapping[str, Any],
) -> tuple[Mapping[str, float], tuple[str, ...]]:
    metrics = clean_manifest["metrics"]
    fidelity = {method: float(metrics[f"single__{method}"]["F"]) for method in cell.methods}
    _, ordered = order_methods_by_fidelity(cell.methods, fidelity)
    return fidelity, ordered


def _selection_records(
    experiment: NoisePrefixExperiment,
    cell: Any,
    *,
    fidelity: Mapping[str, float],
    ordered_methods: Sequence[str],
    store: Any | None = None,
) -> Mapping[str, Any]:
    result = {}
    for task in experiment.assumptions.selection_tasks():
        if task.cell.cell_id != cell.cell_id:
            continue
        manifest = completed_selection_manifest(experiment.assumptions, task, store=store)
        if manifest is None:
            raise FileNotFoundError(f"NOISE selection artifact is incomplete: {task.task_id}")
        selection = manifest.get("selection")
        if not isinstance(selection, Mapping):
            raise ArtifactError(f"Malformed NOISE selection artifact: {task.task_id}")
        if (
            manifest.get("fidelity_scope") != "complete_test_set"
            or manifest.get("oracle_scope") != "complete_test_set_in_sample"
            or manifest.get("fidelity") != dict(fidelity)
            or selection.get("ordered_methods") != list(ordered_methods)
        ):
            raise ArtifactError(
                f"NOISE selection does not match the sweep Fidelity ordering: {task.task_id}"
            )
        evaluations = selection.get("evaluations")
        if not isinstance(evaluations, list):
            raise ArtifactError(f"NOISE selection has no prefix evaluations: {task.task_id}")
        by_q = {str(int(row["size"])): row for row in evaluations if isinstance(row, Mapping)}
        if set(by_q) != {str(value) for value in experiment.q_values}:
            raise ArtifactError(f"NOISE selection q coverage is incomplete: {task.task_id}")
        result[task.distance_model] = {
            "task_id": task.task_id,
            "task_digest": task.digest,
            "manifest_content_digest": object_sha256(manifest),
            "aggregation": task.aggregation,
            "selection": selection,
        }
    if set(result) != {"spearman", "kendall"}:
        raise ArtifactError(f"Expected Spearman and Kendall selections for {cell.cell_id}")
    return result


def _deferred_selection_records(
    experiment: NoisePrefixExperiment,
    cell: Any,
    *,
    fidelity: Mapping[str, float],
    ordered_methods: Sequence[str],
) -> Mapping[str, Any]:
    """Record that q will be frozen by the independent-geometry selector."""

    return {
        "mode": "independent_geometry_deferred",
        "selector": "simple-independent-geometry-noise-selector-v1",
        "scope": "complete_test_set_post_hoc",
        "distance_models": ["spearman", "kendall"],
        "fidelity": dict(fidelity),
        "ordered_methods": list(ordered_methods),
        "cell": cell.cell_id,
    }


def _catalog_without_digest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: item for key, item in value.items() if key != "catalog_digest"}


def _validate_catalog(
    experiment: NoisePrefixExperiment,
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = {
        "schema": CATALOG_SCHEMA,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "sweep_id": experiment.sweep_id,
        "experiment_digest": experiment.digest,
        "base_remote_root": experiment.base.storage.remote_root,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ArtifactError("Input catalog identity differs from the current prefix sweep")
    if value.get("catalog_digest") != object_sha256(_catalog_without_digest(value)):
        raise ArtifactError("Input catalog digest is invalid")
    tasks = value.get("tasks")
    expected_tasks = {task.task_id for task in experiment.evaluation_tasks()}
    if not isinstance(tasks, Mapping) or set(tasks) != expected_tasks:
        raise ArtifactError("Input catalog task coverage differs from the current prefix sweep")
    return value


def load_input_catalog(experiment: NoisePrefixExperiment) -> Mapping[str, Any]:
    path = experiment.runtime.input_catalog_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Input catalog is missing: {path}; run `simple noise-prefix readiness` first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ArtifactError("Input catalog is not a mapping")
    return _validate_catalog(experiment, value)


def prepare_input_catalog(
    experiment: NoisePrefixExperiment,
    *,
    refresh: bool = False,
) -> Mapping[str, Any]:
    """Verify every immutable input once and cache only locators and metadata."""

    path = experiment.runtime.input_catalog_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file() and not refresh:
            return load_input_catalog(experiment)
        base_store = ArtifactStore(experiment.base)
        experiment.storage.spool_root.mkdir(parents=True, exist_ok=True)
        print("NOISE_PREFIX_READINESS stage=rank_ready_inventory", flush=True)
        inventory = _remote_rank_ready_inventory(base_store)
        with TemporaryDirectory(
            prefix="noise-prefix-manifests-",
            dir=experiment.storage.spool_root,
        ):
            base_paths = _required_base_manifest_paths(experiment)
            print(
                f"NOISE_PREFIX_READINESS stage=base_manifest_snapshot manifests={len(base_paths)}",
                flush=True,
            )
            base_snapshot = _manifest_snapshot(
                base_store,
                relative_paths=base_paths,
            )
            base_manifests = _ManifestSnapshotStore(base_store, base_snapshot)
            selection_manifests = None
            if experiment.selection_input_mode == "legacy_assumptions":
                selection_store = ArtifactStore(experiment.assumptions)  # type: ignore[arg-type]
                selection_paths = tuple(
                    posixpath.join(task.artifact_root, "manifest.json")
                    for task in experiment.assumptions.selection_tasks()
                )
                print(
                    "NOISE_PREFIX_READINESS "
                    f"stage=selection_manifest_snapshot manifests={len(selection_paths)}",
                    flush=True,
                )
                selection_snapshot = _manifest_snapshot(
                    selection_store,
                    relative_paths=selection_paths,
                )
                selection_manifests = _ManifestSnapshotStore(
                    selection_store,
                    selection_snapshot,
                )
            else:
                print(
                    "NOISE_PREFIX_READINESS "
                    "stage=selection_manifest_snapshot mode=independent_geometry_deferred",
                    flush=True,
                )
            phase2_cache: dict[str, tuple[Any, str, Mapping[str, Any]]] = {}
            order_by_cell = {}
            selections = {}
            for cell in experiment.cells():
                clean_task = next(
                    task
                    for task in experiment.evaluation_tasks()
                    if task.cell.cell_id == cell.cell_id and task.condition.kind == "clean"
                )
                clean = _base_phase2_manifest(
                    experiment,
                    clean_task,
                    base_manifests,  # type: ignore[arg-type]
                )
                phase2_cache[clean_task.task_id] = clean
                fidelity, ordered = _fidelity_order(cell, clean[2])
                order_by_cell[cell.cell_id] = (fidelity, ordered)
                if selection_manifests is None:
                    selections[cell.cell_id] = _deferred_selection_records(
                        experiment,
                        cell,
                        fidelity=fidelity,
                        ordered_methods=ordered,
                    )
                else:
                    selections[cell.cell_id] = _selection_records(
                        experiment,
                        cell,
                        fidelity=fidelity,
                        ordered_methods=ordered,
                        store=selection_manifests,
                    )

            catalog_tasks = {}
            required_sidecars = set()
            required_bytes = 0
            tasks = experiment.evaluation_tasks()
            for task_position, task in enumerate(tasks, start=1):
                source_task, source_root, source_manifest = phase2_cache.get(
                    task.task_id
                ) or _base_phase2_manifest(
                    experiment,
                    task,
                    base_manifests,  # type: ignore[arg-type]
                )
                fidelity, ordered_methods = order_by_cell[task.cell.cell_id]
                sources = _source_manifests(
                    experiment.base,
                    source_task,
                    base_manifests,  # type: ignore[arg-type]
                    ordered_methods,
                )
                first = next(iter(sources.values()))[1]
                layout = [
                    [int(row["shard_index"]), int(row["start"]), int(row["stop"])]
                    for row in first["shards"]
                ]
                source_rows = {}
                for method in ordered_methods:
                    root, manifest = sources[method]
                    shards = []
                    for record in manifest["shards"]:
                        sidecar = _sidecar_from_inventory(
                            base_store,
                            inventory,
                            source_payload=record["payload"],
                            normalization=experiment.base.phase2.simpleavg_normalization,
                        )
                        required_sidecars.add(sidecar["identity_digest"])
                        required_bytes += int(sidecar["size_bytes"])
                        shards.append(
                            {
                                "shard_index": int(record["shard_index"]),
                                "start": int(record["start"]),
                                "stop": int(record["stop"]),
                                "source_payload": dict(record["payload"]),
                                "rank_ready": sidecar,
                            }
                        )
                    source_rows[method] = {
                        "root": root,
                        "task_digest": manifest["task_digest"],
                        "variant_digest": manifest["method"]["variant_digest"],
                        "shards": shards,
                    }
                q11_metrics = {rule: dict(source_manifest["metrics"][rule]) for rule in RULE_IDS}
                q11_robustness = source_manifest.get("robustness")
                if q11_robustness is not None:
                    q11_robustness = {rule: q11_robustness[rule] for rule in RULE_IDS}
                catalog_tasks[task.task_id] = {
                    "task_digest": task.digest,
                    "cell": task.cell.cell_id,
                    "condition": task.condition.condition_id,
                    "fidelity": dict(fidelity),
                    "ordered_methods": list(ordered_methods),
                    "method_prefixes": {
                        str(q): list(ordered_methods[:q]) for q in experiment.q_values
                    },
                    "source_layout": layout,
                    "sources": source_rows,
                    "q11_reference": {
                        "task_id": source_task.task_id,
                        "task_digest": source_task.digest,
                        "artifact_root": source_root,
                        "manifest_locator": base_store.locator(f"{source_root}/manifest.json"),
                        "manifest_content_digest": object_sha256(source_manifest),
                        "sample_count": int(source_manifest["sample_count"]),
                        "metrics": q11_metrics,
                        "robustness": q11_robustness,
                    },
                }
                print(
                    "NOISE_PREFIX_READINESS "
                    f"stage=task task={task_position}/{len(tasks)} id={task.task_id}",
                    flush=True,
                )
        value: dict[str, Any] = {
            "schema": CATALOG_SCHEMA,
            "schema_version": CATALOG_SCHEMA_VERSION,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "sweep_id": experiment.sweep_id,
            "experiment_digest": experiment.digest,
            "base_remote_root": experiment.base.storage.remote_root,
            "rank_ready_inventory": {
                "strategy": "remote_recursive_inventory",
                "required_unique_sidecars": len(required_sidecars),
                "required_reference_bytes": required_bytes,
            },
            "noise_selections": selections,
            "tasks": catalog_tasks,
        }
        value["catalog_digest"] = object_sha256(value)
        atomic_write_json(path, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return _validate_catalog(experiment, value)


def catalog_task_input(
    experiment: NoisePrefixExperiment,
    task: PrefixEvaluationTask,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    value = catalog or load_input_catalog(experiment)
    row = value["tasks"][task.task_id]
    if row.get("task_digest") != task.digest:
        raise ArtifactError(f"Catalog task digest mismatch: {task.task_id}")
    return row


def source_layout(row: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(item[0]), int(item[1]), int(item[2])) for item in row["source_layout"])


def input_byte_count(row: Mapping[str, Any], shard_index: int) -> int:
    total = 0
    for method in row["ordered_methods"]:
        record = next(
            item
            for item in row["sources"][method]["shards"]
            if int(item["shard_index"]) == shard_index
        )
        total += int(record["rank_ready"]["size_bytes"])
    return max(1, total)


def _aligned(reference: Mapping[str, Any], candidate: Mapping[str, Any], method: str) -> None:
    import torch

    for field in ("indices", "labels", "predictions", "logits", "targets"):
        if not torch.equal(reference[field], candidate[field]):
            raise ArtifactError(f"Rank-ready {field} is not aligned for method {method}")


def load_prefix_shard(
    experiment: NoisePrefixExperiment,
    task_row: Mapping[str, Any],
    *,
    shard_index: int,
    work_directory: Path,
) -> PrefixShardInput:
    """Load only existing sidecars and derive all q=2..10 prefix inputs."""

    import torch

    store = ArtifactStore(experiment.base)
    reference = None
    ballots = []
    spatial_sum = None
    method_patch_scores = []
    simple_scores = {}
    sidecars = []
    methods = tuple(str(value) for value in task_row["ordered_methods"])
    computed_q = set(experiment.q_values[:-1])
    for position, method in enumerate(methods, start=1):
        record = next(
            item
            for item in task_row["sources"][method]["shards"]
            if int(item["shard_index"]) == shard_index
        )
        count = int(record["stop"]) - int(record["start"])
        compact = load_existing_rank_ready_sidecar(
            store,
            source_payload=record["source_payload"],
            work_directory=work_directory / f"method-{position:02d}",
            simpleavg_normalization=experiment.base.phase2.simpleavg_normalization,
            count=count,
            recorded_sidecar=record["rank_ready"],
        )
        fields = compact.fields
        fixed = {
            name: fields[name] for name in ("indices", "labels", "predictions", "logits", "targets")
        }
        if reference is None:
            reference = fixed
        else:
            _aligned(reference, fixed, method)
        rank = fields[rank_field(experiment.patch_size)]
        expected = torch.arange(rank.shape[1], dtype=torch.int32).expand(rank.shape[0], -1)
        if not torch.equal(torch.sort(rank, dim=1).values, expected):
            raise ArtifactError(f"Rank-ready ballot is not a strict permutation for {method}")
        ballots.append(rank.numpy().astype(np.int64, copy=False).copy())
        spatial = fields[simpleavg_spatial_field()].numpy().astype(np.float32, copy=False)
        spatial_sum = spatial.copy() if spatial_sum is None else spatial_sum + spatial
        height, width = spatial.shape[-2:]
        grid_h = height // experiment.patch_size
        grid_w = width // experiment.patch_size
        patch_scores = (
            spatial.reshape(
                spatial.shape[0],
                grid_h,
                experiment.patch_size,
                grid_w,
                experiment.patch_size,
            )
            .mean(axis=(2, 4), dtype=np.float32)
            .reshape(spatial.shape[0], -1)
        )
        method_patch_scores.append(patch_scores.copy())
        if position in computed_q:
            averaged = spatial_sum / float(position)
            simple_scores[position] = (
                averaged.reshape(
                    averaged.shape[0],
                    grid_h,
                    experiment.patch_size,
                    grid_w,
                    experiment.patch_size,
                )
                .mean(axis=(2, 4), dtype=np.float32)
                .reshape(averaged.shape[0], -1)
            )
        sidecars.append(
            {
                "method": method,
                "source_payload_sha256": record["source_payload"]["sha256"],
                **dict(compact.payload),
            }
        )
        del fields, fixed, rank, spatial
    assert reference is not None
    if reference["logits"].dtype != torch.float32 or reference["logits"].ndim != 2:
        raise ArtifactError("Reference logits must be an FP32 matrix")
    if not torch.equal(reference["predictions"], reference["logits"].argmax(dim=1)):
        raise ArtifactError("Reference predictions differ from argmax(logits)")
    return PrefixShardInput(
        reference=reference,
        ballots=np.stack(ballots, axis=1),
        method_patch_scores=np.stack(method_patch_scores, axis=1),
        simple_scores_by_q=simple_scores,
        sidecars=tuple(sidecars),
    )


def readiness_report(
    experiment: NoisePrefixExperiment, *, refresh: bool = False
) -> Mapping[str, Any]:
    catalog = prepare_input_catalog(experiment, refresh=refresh)
    return {
        "ready": True,
        "sweep_id": experiment.sweep_id,
        "experiment_digest": experiment.digest,
        "catalog_path": str(experiment.runtime.input_catalog_path),
        "catalog_digest": catalog["catalog_digest"],
        "tasks": len(catalog["tasks"]),
        **dict(catalog["rank_ready_inventory"]),
    }


__all__ = [
    "CATALOG_SCHEMA",
    "CATALOG_SCHEMA_VERSION",
    "PrefixShardInput",
    "catalog_task_input",
    "input_byte_count",
    "load_input_catalog",
    "load_prefix_shard",
    "materialize_missing_rank_ready",
    "prepare_input_catalog",
    "readiness_report",
    "source_layout",
]
