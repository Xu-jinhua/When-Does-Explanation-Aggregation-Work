"""Compact, content-bound Phase 1 sidecars for rank-only consumers."""

from __future__ import annotations

import fcntl
import os
import shutil
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256

from .artifacts import ArtifactError, ArtifactStore, load_safetensors, write_phase2_shard
from .spool import SpoolQuota, SpoolReservation

RANK_READY_SCHEMA_VERSION = 2
RANK_READY_PATCH_SIZES = (8, 14, 16)
RANK_READY_SCHEMA = "simple-rank-ready-v2"
COMPACT_SOURCE_RANK_SCHEMA = "simple-assumptions-source-rank-input-v1"
COMPACT_SCORE_ATOL_FP32_EPS = 8


def attribution_to_patch_scores(attributions: Any, patch_size: int) -> np.ndarray:
    """Paper score: mean(abs(attribution)) over channels and patch pixels."""

    values = np.asarray(attributions)
    if values.ndim != 4 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("Attributions must have numeric [N,C,H,W] shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("Attributions contain NaN or infinite values")
    _, _, height, width = values.shape
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError(f"Attribution shape {(height, width)} is not divisible by p={patch_size}")
    magnitude = np.mean(np.abs(values.astype(np.float32, copy=False)), axis=1)
    grid_h, grid_w = height // patch_size, width // patch_size
    return magnitude.reshape(magnitude.shape[0], grid_h, patch_size, grid_w, patch_size).mean(
        axis=(2, 4), dtype=np.float32
    )


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Vectorized strict rank, best=0, with stable row-major tie breaking."""

    values = np.asarray(scores)
    if values.ndim < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Scores must have finite [N,...] shape")
    flat = values.reshape(values.shape[0], -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    positions = np.broadcast_to(np.arange(flat.shape[1], dtype=np.int64), order.shape)
    np.put_along_axis(ranks, order, positions, axis=1)
    return ranks


def normalize_spatial(values: np.ndarray, method: str) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float32, copy=False)
    if method == "none":
        normalized = flat.copy()
    elif method == "minmax":
        minima = flat.min(axis=1, keepdims=True)
        spans = flat.max(axis=1, keepdims=True) - minima
        normalized = np.divide(
            flat - minima,
            spans,
            out=np.zeros_like(flat),
            where=spans > 0,
        )
    elif method == "max":
        maxima = np.max(np.abs(flat), axis=1, keepdims=True)
        normalized = np.divide(flat, maxima, out=np.zeros_like(flat), where=maxima > 0)
    elif method == "l1":
        totals = np.sum(np.abs(flat), axis=1, keepdims=True)
        normalized = np.divide(flat, totals, out=np.zeros_like(flat), where=totals > 0)
    else:
        raise ValueError(f"Unknown SimpleAvg normalization: {method}")
    return normalized.reshape(values.shape)


def supported_patch_sizes(
    height: int,
    width: int,
    candidates: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in candidates
        if int(value) > 0 and height % int(value) == 0 and width % int(value) == 0
    )


def _patch_field(prefix: str, patch_size: int) -> str:
    return f"{prefix}__p{patch_size:03d}"


def rank_ready_identity(
    source_payload_sha256: str,
    *,
    simpleavg_normalization: str,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> Mapping[str, Any]:
    values = tuple(int(item) for item in patch_sizes)
    if not values or len(set(values)) != len(values):
        raise ValueError("rank-ready patch sizes must be non-empty and unique")
    return {
        "schema": RANK_READY_SCHEMA,
        "schema_version": RANK_READY_SCHEMA_VERSION,
        "source_payload_sha256": str(source_payload_sha256),
        "patch_sizes": list(values),
        "paper_rank_semantics": "mean_over_patch_and_channels(abs(full_attribution))",
        "simpleavg_normalization": simpleavg_normalization,
        "simpleavg_order": "normalize_pixels_then_method_mean_then_patch_mean",
        "rank_base": 0,
        "tie_break": "stable_row_major_patch_index",
    }


@dataclass(frozen=True, slots=True)
class RankReadyDescriptor:
    identity: Mapping[str, Any]
    identity_digest: str
    relative_path: str


def rank_ready_descriptor(
    source_payload_sha256: str,
    *,
    simpleavg_normalization: str,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> RankReadyDescriptor:
    identity = rank_ready_identity(
        source_payload_sha256,
        simpleavg_normalization=simpleavg_normalization,
        patch_sizes=patch_sizes,
    )
    digest = object_sha256(identity)
    return RankReadyDescriptor(
        identity=identity,
        identity_digest=digest,
        relative_path=f"derived/rank-ready-v2/{digest[:2]}/{digest}.safetensors",
    )


def build_rank_ready_tensors(
    fields: Mapping[str, Any],
    *,
    simpleavg_normalization: str,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> Mapping[str, Any]:
    import torch

    required = ("indices", "labels", "predictions", "logits", "targets", "attributions")
    if any(name not in fields for name in required):
        raise ArtifactError("Phase 1 shard is missing fields required by rank-ready sidecars")
    fixed = {
        name: torch.as_tensor(fields[name]).detach().to("cpu").contiguous().clone()
        for name in required[:-1]
    }
    count = int(fixed["indices"].shape[0])
    if count <= 0 or any(int(value.shape[0]) != count for value in fixed.values()):
        raise ArtifactError("Rank-ready fixed fields are not aligned")
    attribution = (
        torch.as_tensor(fields["attributions"]).detach().to("cpu", dtype=torch.float32).contiguous()
    )
    if attribution.ndim != 4 or int(attribution.shape[0]) != count:
        raise ArtifactError("Rank-ready attribution must have [N,C,H,W] shape")
    values = attribution.numpy()
    magnitude = np.mean(np.abs(values), axis=1, dtype=np.float32)
    normalized = normalize_spatial(magnitude, simpleavg_normalization)
    tensors: dict[str, Any] = {
        **fixed,
        # Keep the normalized spatial map so method-wise FP32 addition occurs
        # before patch reduction, exactly matching the established artifacts.
        # Patch means alone are mathematically equivalent but can exchange
        # nearly tied ranks because FP32 addition is not associative.
        "simpleavg_spatial": torch.from_numpy(normalized).to(torch.float32),
    }
    for patch_size in patch_sizes:
        paper_scores = attribution_to_patch_scores(values, int(patch_size))
        grid_h = int(values.shape[-2]) // int(patch_size)
        grid_w = int(values.shape[-1]) // int(patch_size)
        simple_scores = normalized.reshape(
            count,
            grid_h,
            int(patch_size),
            grid_w,
            int(patch_size),
        ).mean(axis=(2, 4), dtype=np.float32)
        tensors[_patch_field("rank", int(patch_size))] = torch.from_numpy(
            scores_to_ranks(paper_scores)
        ).to(torch.int32)
        tensors[_patch_field("simpleavg_score", int(patch_size))] = torch.from_numpy(
            simple_scores.reshape(count, -1)
        ).to(torch.float32)
    return tensors


def write_rank_ready_sidecar(
    path: str | Path,
    *,
    fields: Mapping[str, Any],
    source_payload_sha256: str,
    simpleavg_normalization: str,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> RankReadyDescriptor:
    descriptor = rank_ready_descriptor(
        source_payload_sha256,
        simpleavg_normalization=simpleavg_normalization,
        patch_sizes=patch_sizes,
    )
    tensors = build_rank_ready_tensors(
        fields,
        simpleavg_normalization=simpleavg_normalization,
        patch_sizes=patch_sizes,
    )
    write_phase2_shard(
        path,
        tensors=tensors,
        metadata={
            "schema": RANK_READY_SCHEMA,
            "schema_version": str(RANK_READY_SCHEMA_VERSION),
            "identity_digest": descriptor.identity_digest,
            "source_payload_sha256": source_payload_sha256,
            "patch_sizes": ",".join(str(item) for item in patch_sizes),
            "simpleavg_normalization": simpleavg_normalization,
        },
    )
    return descriptor


def _validate_sidecar(
    path: Path,
    descriptor: RankReadyDescriptor,
    *,
    count: int,
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected_metadata = {
        "schema": RANK_READY_SCHEMA,
        "schema_version": str(RANK_READY_SCHEMA_VERSION),
        "identity_digest": descriptor.identity_digest,
        "source_payload_sha256": str(descriptor.identity["source_payload_sha256"]),
        "patch_sizes": ",".join(str(item) for item in descriptor.identity["patch_sizes"]),
        "simpleavg_normalization": str(descriptor.identity["simpleavg_normalization"]),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ArtifactError("Rank-ready sidecar metadata contradicts its content identity")
    fields = dict(load_safetensors(path))
    required = {
        "indices",
        "labels",
        "predictions",
        "logits",
        "targets",
        "simpleavg_spatial",
    }
    for patch_size in descriptor.identity["patch_sizes"]:
        required.add(_patch_field("rank", int(patch_size)))
        required.add(_patch_field("simpleavg_score", int(patch_size)))
    if set(fields) != required:
        raise ArtifactError("Rank-ready sidecar fields differ from its schema")
    if any(int(value.shape[0]) != count for value in fields.values()):
        raise ArtifactError("Rank-ready sidecar fields are not shard-aligned")
    if fields["logits"].ndim != 2 or not bool(torch.isfinite(fields["logits"]).all()):
        raise ArtifactError("Rank-ready logits are invalid")
    if not torch.equal(fields["predictions"], fields["logits"].argmax(dim=1)):
        raise ArtifactError("Rank-ready predictions differ from argmax(logits)")
    spatial = fields["simpleavg_spatial"]
    if spatial.ndim != 3 or spatial.dtype != torch.float32:
        raise ArtifactError("Rank-ready SimpleAvg spatial maps are invalid")
    if not bool(torch.isfinite(spatial).all()):
        raise ArtifactError("Rank-ready SimpleAvg spatial maps are not finite")
    for patch_size in descriptor.identity["patch_sizes"]:
        rank = fields[_patch_field("rank", int(patch_size))]
        score = fields[_patch_field("simpleavg_score", int(patch_size))]
        if rank.ndim != 2 or score.shape != rank.shape:
            raise ArtifactError("Rank-ready patch fields have incompatible shapes")
        if rank.dtype != torch.int32 or score.dtype != torch.float32:
            raise ArtifactError("Rank-ready patch fields have invalid dtypes")
        if not bool(torch.isfinite(score).all()):
            raise ArtifactError("Rank-ready SimpleAvg scores are not finite")
    return fields


@dataclass(frozen=True, slots=True)
class RankReadyShard:
    fields: Mapping[str, Any]
    descriptor: RankReadyDescriptor
    payload: Mapping[str, Any]
    generated: bool
    elapsed_seconds: float
    publication_future: Future[Mapping[str, Any]] | None = None


def existing_rank_ready_payload(
    store: ArtifactStore,
    *,
    source_payload: Mapping[str, Any],
    simpleavg_normalization: str,
    recorded_sidecar: Mapping[str, Any] | None = None,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> tuple[RankReadyDescriptor, Mapping[str, Any]]:
    """Resolve an existing sidecar without permitting attribution fallback.

    A caller may provide a payload from a preflight remote inventory. Such an
    entry must explicitly record the matching receipt; otherwise this function
    reads and validates the immutable receipt itself.
    """

    source_sha = str(source_payload["sha256"])
    descriptor = rank_ready_descriptor(
        source_sha,
        simpleavg_normalization=simpleavg_normalization,
        patch_sizes=patch_sizes,
    )
    receipt_path = f"{descriptor.relative_path}.receipt.json"
    if recorded_sidecar is None:
        if not store.exists(receipt_path):
            raise FileNotFoundError(
                f"Required rank-ready receipt is missing: {receipt_path}; "
                "the prefix sweep never falls back to full attribution"
            )
        receipt = store.read_json(receipt_path)
        if receipt.get("relative_path") != descriptor.relative_path:
            raise ArtifactError("Rank-ready receipt has a contradictory locator")
        payload: Mapping[str, Any] = {
            "relative_path": descriptor.relative_path,
            "sha256": str(receipt["sha256"]),
            "size_bytes": int(receipt["size_bytes"]),
            "identity_digest": descriptor.identity_digest,
            "receipt_relative_path": receipt_path,
        }
    else:
        payload = dict(recorded_sidecar)
        if payload.get("receipt_relative_path") != receipt_path or payload.get(
            "receipt_verified_by"
        ) not in {"remote_recursive_inventory", "direct_receipt_validation"}:
            raise ArtifactError("Rank-ready catalog entry has no verified matching receipt")
    expected = {
        "relative_path": descriptor.relative_path,
        "identity_digest": descriptor.identity_digest,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ArtifactError("Recorded rank-ready locator contradicts its source payload")
    digest = str(payload.get("sha256", ""))
    size = int(payload.get("size_bytes", -1))
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest.lower()):
        raise ArtifactError("Rank-ready payload has no valid SHA-256")
    if size <= 0:
        raise ArtifactError("Rank-ready payload has no valid byte size")
    return descriptor, payload


def load_existing_rank_ready_sidecar(
    store: ArtifactStore,
    *,
    source_payload: Mapping[str, Any],
    work_directory: str | Path,
    simpleavg_normalization: str,
    count: int,
    recorded_sidecar: Mapping[str, Any] | None = None,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
) -> RankReadyShard:
    """Materialize a validated sidecar and fail if it has not been published."""

    started = time.monotonic()
    descriptor, payload = existing_rank_ready_payload(
        store,
        source_payload=source_payload,
        simpleavg_normalization=simpleavg_normalization,
        recorded_sidecar=recorded_sidecar,
        patch_sizes=patch_sizes,
    )
    directory = Path(work_directory)
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / f"{descriptor.identity_digest}.safetensors"
    try:
        store.materialize(
            descriptor.relative_path,
            local,
            expected_sha256=str(payload["sha256"]),
        )
        fields = _validate_sidecar(local, descriptor, count=count)
    finally:
        local.unlink(missing_ok=True)
    return RankReadyShard(
        fields=fields,
        descriptor=descriptor,
        payload=payload,
        generated=False,
        elapsed_seconds=time.monotonic() - started,
    )


class RankReadyPublisher:
    """Take ownership of generated sidecars and upload them after prefetch returns."""

    def __init__(
        self,
        *,
        spool_root: str | Path,
        spool_max_bytes: int,
        spool_min_free_bytes: int,
        namespace: str,
    ) -> None:
        self.quota = SpoolQuota(
            spool_root,
            max_bytes=spool_max_bytes,
            min_free_bytes=spool_min_free_bytes,
        )
        self.work_directory = (
            self.quota.root
            / "publish"
            / "rank-ready"
            / namespace
            / f"{os.getpid()}-{uuid.uuid4().hex}"
        )
        self.work_directory.mkdir(parents=True, exist_ok=False)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rank-ready-upload")
        self.futures: list[Future[Mapping[str, Any]]] = []
        self._failure: BaseException | None = None
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False

    def _record_failure(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error

    def check(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise ArtifactError("Background rank-ready publication failed") from failure

    def submit(
        self,
        store: ArtifactStore,
        local_path: Path,
        descriptor: RankReadyDescriptor,
        *,
        lock_handle: Any,
    ) -> Future[Mapping[str, Any]]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rank-ready publisher is closed")
        self.check()
        reservation: SpoolReservation = self.quota.acquire(
            max(1, local_path.stat().st_size + 2**20),
            work_directory=self.work_directory,
            priority=10,
        )
        owned = self.work_directory / f"{descriptor.identity_digest}.safetensors"
        try:
            os.replace(local_path, owned)
        except BaseException:
            self.quota.release(reservation)
            raise

        def upload() -> Mapping[str, Any]:
            try:
                with ExitStack() as cleanup:
                    # Register in reverse execution order. ExitStack still runs
                    # every callback if an earlier cleanup operation fails.
                    cleanup.callback(lock_handle.close)
                    cleanup.callback(fcntl.flock, lock_handle.fileno(), fcntl.LOCK_UN)
                    cleanup.callback(self.quota.release, reservation)
                    cleanup.callback(owned.unlink, missing_ok=True)
                    published = store.publish(owned, descriptor.relative_path)
                    return {
                        "relative_path": published.relative_path,
                        "sha256": published.sha256,
                        "size_bytes": published.size_bytes,
                        "identity_digest": descriptor.identity_digest,
                    }
            except BaseException as error:
                self._record_failure(error)
                raise

        try:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("rank-ready publisher is closed")
                future = self.executor.submit(upload)
                self.futures.append(future)
        except BaseException:
            try:
                owned.unlink(missing_ok=True)
            finally:
                self.quota.release(reservation)
            raise
        return future

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        try:
            try:
                self.executor.shutdown(wait=True)
            finally:
                for future in self.futures:
                    try:
                        future.result()
                    except BaseException:
                        # ``upload`` records the first background failure. Resolve
                        # every future so shutdown drains all owned resources, then
                        # raise the stable ArtifactError from ``check`` below.
                        pass
                self.check()
        finally:
            shutil.rmtree(self.work_directory, ignore_errors=True)


def ensure_rank_ready_sidecar(
    store: ArtifactStore,
    *,
    source_payload: Mapping[str, Any],
    work_directory: str | Path,
    lock_root: str | Path,
    simpleavg_normalization: str,
    count: int,
    recorded_sidecar: Mapping[str, Any] | None = None,
    patch_sizes: Sequence[int] = RANK_READY_PATCH_SIZES,
    publisher: RankReadyPublisher | None = None,
) -> RankReadyShard:
    """Restore a compact sidecar, lazily deriving legacy Phase 1 shards once."""

    started = time.monotonic()
    source_sha = str(source_payload["sha256"])
    descriptor = rank_ready_descriptor(
        source_sha,
        simpleavg_normalization=simpleavg_normalization,
        patch_sizes=patch_sizes,
    )
    directory = Path(work_directory)
    directory.mkdir(parents=True, exist_ok=True)
    locks = Path(lock_root)
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f"{descriptor.identity_digest}.lock"
    local = directory / f"{descriptor.identity_digest}.safetensors"
    lock = lock_path.open("a+b")
    lock_transferred = False
    publication_future = None
    source_local = directory / f"source-{source_sha}.safetensors"
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload: Mapping[str, Any] | None = None
        if recorded_sidecar is not None:
            if (
                recorded_sidecar.get("relative_path") != descriptor.relative_path
                or recorded_sidecar.get("identity_digest") != descriptor.identity_digest
            ):
                raise ArtifactError("Recorded rank-ready locator contradicts its source payload")
            payload = recorded_sidecar
        else:
            receipt_path = f"{descriptor.relative_path}.receipt.json"
            if store.exists(receipt_path):
                receipt = store.read_json(receipt_path)
                if receipt.get("relative_path") != descriptor.relative_path:
                    raise ArtifactError("Rank-ready receipt has a contradictory locator")
                payload = {
                    "relative_path": descriptor.relative_path,
                    "sha256": str(receipt["sha256"]),
                    "size_bytes": int(receipt["size_bytes"]),
                    "identity_digest": descriptor.identity_digest,
                }
        generated = False
        if payload is not None:
            store.materialize(
                descriptor.relative_path,
                local,
                expected_sha256=str(payload["sha256"]),
            )
        else:
            try:
                store.materialize(
                    str(source_payload["relative_path"]),
                    source_local,
                    expected_sha256=source_sha,
                )
                source_fields = load_safetensors(source_local)
                write_rank_ready_sidecar(
                    local,
                    fields=source_fields,
                    source_payload_sha256=source_sha,
                    simpleavg_normalization=simpleavg_normalization,
                    patch_sizes=patch_sizes,
                )
                generated = True
            finally:
                source_local.unlink(missing_ok=True)
        fields = _validate_sidecar(local, descriptor, count=count)
        if payload is None:
            if publisher is None:
                published = store.publish(local, descriptor.relative_path)
                payload = {
                    "relative_path": published.relative_path,
                    "sha256": published.sha256,
                    "size_bytes": published.size_bytes,
                    "identity_digest": descriptor.identity_digest,
                }
            else:
                publication_future = publisher.submit(
                    store,
                    local,
                    descriptor,
                    lock_handle=lock,
                )
                lock_transferred = True
                payload = {
                    "relative_path": descriptor.relative_path,
                    "identity_digest": descriptor.identity_digest,
                    "publication_gate": "rank_task_completion",
                }
    finally:
        source_local.unlink(missing_ok=True)
        local.unlink(missing_ok=True)
        if not lock_transferred:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
    assert payload is not None
    return RankReadyShard(
        fields=fields,
        descriptor=descriptor,
        payload=payload,
        generated=generated,
        elapsed_seconds=time.monotonic() - started,
        publication_future=publication_future,
    )


def rank_field(patch_size: int) -> str:
    return _patch_field("rank", patch_size)


def simpleavg_score_field(patch_size: int) -> str:
    return _patch_field("simpleavg_score", patch_size)


def simpleavg_spatial_field() -> str:
    return "simpleavg_spatial"


def compact_rank_input_fp32_equivalence(source: Path, observed: Path) -> str | None:
    """Accept only rank-invariant FP32 reduction noise in compact source shards."""

    import torch
    from safetensors import safe_open

    def metadata(path: Path) -> Mapping[str, str]:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})

    source_metadata = metadata(source)
    observed_metadata = metadata(observed)
    if source_metadata != observed_metadata:
        return None
    if source_metadata.get("schema") != COMPACT_SOURCE_RANK_SCHEMA:
        return None
    try:
        patch_size = int(source_metadata["patch_size"])
    except (KeyError, TypeError, ValueError):
        return None
    rank_name = rank_field(patch_size)
    score_name = simpleavg_score_field(patch_size)
    required = {
        "indices",
        "labels",
        "predictions",
        "logits",
        "targets",
        rank_name,
        score_name,
    }
    source_fields = dict(load_safetensors(source))
    observed_fields = dict(load_safetensors(observed))
    if set(source_fields) != required or set(observed_fields) != required:
        return None
    for name in required - {score_name}:
        left = source_fields[name]
        right = observed_fields[name]
        if left.dtype != right.dtype or not torch.equal(left, right):
            return None
    left_score = source_fields[score_name]
    right_score = observed_fields[score_name]
    if (
        left_score.dtype != torch.float32
        or right_score.dtype != torch.float32
        or left_score.shape != right_score.shape
        or not bool(torch.isfinite(left_score).all())
        or not bool(torch.isfinite(right_score).all())
    ):
        return None
    tolerance = COMPACT_SCORE_ATOL_FP32_EPS * torch.finfo(torch.float32).eps
    if not bool(torch.allclose(left_score, right_score, rtol=0.0, atol=tolerance)):
        return None
    return "compact_rank_input_rank_exact_fp32_score_equivalence"


__all__ = [
    "COMPACT_SCORE_ATOL_FP32_EPS",
    "COMPACT_SOURCE_RANK_SCHEMA",
    "RANK_READY_PATCH_SIZES",
    "RANK_READY_SCHEMA_VERSION",
    "RankReadyDescriptor",
    "RankReadyPublisher",
    "RankReadyShard",
    "attribution_to_patch_scores",
    "build_rank_ready_tensors",
    "compact_rank_input_fp32_equivalence",
    "ensure_rank_ready_sidecar",
    "existing_rank_ready_payload",
    "load_existing_rank_ready_sidecar",
    "normalize_spatial",
    "rank_field",
    "rank_ready_descriptor",
    "scores_to_ranks",
    "simpleavg_score_field",
    "simpleavg_spatial_field",
    "supported_patch_sizes",
    "write_rank_ready_sidecar",
]
