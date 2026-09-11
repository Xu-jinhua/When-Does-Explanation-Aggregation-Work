"""Safetensors shards and verified direct-rclone publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import posixpath
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256
from xai_ensemble.core.io import atomic_write_json

from .config import AdversarialTask, Phase1Task, Phase2Task, SimpleExperiment

ADVERSARIAL_SCHEMA_VERSION = 1
PHASE1_SCHEMA_VERSION = 2
PHASE2_SCHEMA_VERSION = 2


class ArtifactError(RuntimeError):
    """A local or remote artifact violated the immutable transport contract."""


_RCLONE_EXISTS_ATTEMPTS = 4
_RCLONE_RETRY_DELAY_SECONDS = 1.0
_RCLONE_TRANSPORT_ATTEMPTS = 3
# Post-upload verification retries: Google Drive can take seconds to index a
# freshly created object, during which lsjson --stat reports "directory not
# found" even though the upload succeeded (observed on a Drive rclone remote).
_RCLONE_POST_UPLOAD_ATTEMPTS = 8

ExistingPayloadEquivalence = Callable[[Path, Path], str | None]


def _canonical_safetensors_header_bytes(header_bytes: bytes, target: Path) -> bytes:
    try:
        header = json.loads(header_bytes.rstrip(b" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Safetensors file has invalid JSON metadata: {target}") from error
    if not isinstance(header, Mapping):
        raise ArtifactError(f"Safetensors header is not a mapping: {target}")
    canonical = json.dumps(
        header,
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    if len(canonical) > len(header_bytes):
        raise ArtifactError(f"Canonical safetensors header grew unexpectedly: {target}")
    return canonical + b" " * (len(header_bytes) - len(canonical))


def _canonical_safetensors_header(path: str | Path) -> Path:
    """Canonicalize metadata ordering so equal tensors have stable file hashes."""

    target = Path(path)
    with target.open("r+b") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArtifactError(f"Safetensors file has no complete header length: {target}")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ArtifactError(f"Safetensors file has a truncated header: {target}")
        handle.seek(8)
        handle.write(_canonical_safetensors_header_bytes(header_bytes, target))
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _canonical_safetensors_sha256(path: str | Path) -> tuple[str, int]:
    """Hash a safetensors file after canonicalizing only its JSON header."""

    target = Path(path)
    size = target.stat().st_size
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArtifactError(f"Safetensors file has no complete header length: {target}")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ArtifactError(f"Safetensors file has a truncated header: {target}")
        digest.update(prefix)
        digest.update(_canonical_safetensors_header_bytes(header_bytes, target))
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), size


def _existing_safetensors_equivalence(
    source: Path,
    observed: Path,
    semantic_equivalence: ExistingPayloadEquivalence | None,
) -> str | None:
    if _canonical_safetensors_sha256(source) == _canonical_safetensors_sha256(observed):
        return "canonical_safetensors_header_equivalence"
    if semantic_equivalence is None:
        return None
    return semantic_equivalence(source, observed)


def _clean_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe artifact path component: {value!r}")
    return value


def phase1_artifact_root(
    task: Phase1Task,
    artifact_name: str,
) -> str:
    return posixpath.join(
        "phase1",
        *(
            _clean_component(value)
            for value in (
                task.dataset.dataset_id,
                task.model.model_id,
                task.split,
                task.condition.condition_id,
                artifact_name,
            )
        ),
    )


def adversarial_artifact_root(task: AdversarialTask) -> str:
    return posixpath.join(
        "adversarial",
        *(
            _clean_component(value)
            for value in (
                task.dataset.dataset_id,
                task.model.model_id,
                task.split,
                task.condition.condition_id,
            )
        ),
    )


def phase2_artifact_root(task: Phase2Task) -> str:
    return posixpath.join(
        "phase2",
        *(
            _clean_component(value)
            for value in (
                task.dataset.dataset_id,
                task.model.model_id,
                task.split,
                task.condition.condition_id,
                task.ensemble.ensemble_id,
                f"p{task.patch_size}",
                task.digest,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class PublishedFile:
    relative_path: str
    sha256: str
    size_bytes: int


class ArtifactStore:
    """A tiny immutable store supporting local paths and rclone remotes."""

    def __init__(self, experiment: SimpleExperiment) -> None:
        self.root = experiment.storage.remote_root.rstrip("/")
        self.rclone = experiment.storage.rclone_binary
        self.remote = ":" in self.root.split("/", 1)[0]
        self._ensured_remote_dirs: set[str] = set()
        self._ensure_lock = threading.Lock()
        spool_root = getattr(experiment.storage, "spool_root", None)
        lock_root = Path(spool_root) if spool_root is not None else Path(tempfile.gettempdir())
        self._remote_mkdir_lock = Path(lock_root) / ".remote-mkdir.lock"

    def locator(self, relative_path: str) -> str:
        clean = relative_path.lstrip("/")
        if clean != relative_path or any(part in {"", ".", ".."} for part in clean.split("/")):
            raise ValueError(f"Unsafe relative artifact path: {relative_path!r}")
        if self.remote:
            return f"{self.root}/{clean}"
        return str(Path(self.root) / clean)

    def _run(self, *arguments: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                (str(self.rclone), *arguments),
                check=True,
                capture_output=capture,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, "stderr", "") or ""
            raise ArtifactError(
                f"rclone {' '.join(arguments[:2])} failed: {stderr.strip()}"
            ) from error

    @staticmethod
    def _is_transient_transport_error(error: ArtifactError) -> bool:
        detail = str(error).lower()
        return any(
            marker in detail
            for marker in (
                "i/o error",
                "input/output error",
                "eio",
                "rate limit",
                "too many requests",
                "connection reset",
                "connection refused",
                "temporarily unavailable",
                "timed out",
                "timeout",
                "status code: 429",
                "status code: 500",
                "status code: 502",
                "status code: 503",
                "status code: 504",
            )
        )

    def _run_transport_with_retry(
        self,
        *arguments: str,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        for attempt in range(1, _RCLONE_TRANSPORT_ATTEMPTS + 1):
            try:
                return self._run(*arguments, capture=capture)
            except ArtifactError as error:
                if attempt == _RCLONE_TRANSPORT_ATTEMPTS or not self._is_transient_transport_error(
                    error
                ):
                    raise
                time.sleep(_RCLONE_RETRY_DELAY_SECONDS * attempt)
        raise AssertionError("unreachable")

    def exists(self, relative_path: str) -> bool:
        target = self.locator(relative_path)
        if not self.remote:
            return Path(target).is_file()
        command = (str(self.rclone), "lsjson", target, "--stat")
        for attempt in range(1, _RCLONE_EXISTS_ATTEMPTS + 1):
            try:
                result = subprocess.run(command, check=False, capture_output=True, text=True)
            except OSError as error:
                if attempt == _RCLONE_EXISTS_ATTEMPTS:
                    raise ArtifactError(
                        f"Cannot run rclone existence check for {target}"
                    ) from error
                time.sleep(_RCLONE_RETRY_DELAY_SECONDS * attempt)
                continue
            if result.returncode == 0:
                return bool(result.stdout.strip())
            stderr = result.stderr or ""
            if result.returncode == 3 and "not found" in stderr.lower():
                return False
            if attempt == _RCLONE_EXISTS_ATTEMPTS:
                raise ArtifactError(
                    f"rclone lsjson failed for {target} after {_RCLONE_EXISTS_ATTEMPTS} attempts "
                    f"(exit={result.returncode}): {stderr.strip()}"
                )
            time.sleep(_RCLONE_RETRY_DELAY_SECONDS * attempt)
        raise AssertionError("unreachable")

    def read_bytes(self, relative_path: str) -> bytes:
        target = self.locator(relative_path)
        if not self.remote:
            return Path(target).read_bytes()
        try:
            result = subprocess.run(
                (str(self.rclone), "cat", target),
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, "stderr", b"") or b""
            detail = (
                stderr.decode(errors="replace").strip()
                if isinstance(stderr, bytes)
                else str(stderr).strip()
            )
            suffix = f": {detail}" if detail else ""
            raise ArtifactError(f"Cannot read remote artifact {target}{suffix}") from error
        return result.stdout

    def read_json(self, relative_path: str) -> Mapping[str, Any]:
        value = json.loads(self.read_bytes(relative_path))
        if not isinstance(value, Mapping):
            raise ArtifactError(f"JSON artifact is not a mapping: {relative_path}")
        return value

    def _remote_sha256(self, relative_path: str) -> tuple[str, int]:
        target = self.locator(relative_path)
        if not self.remote:
            path = Path(target)
            return file_sha256(path), path.stat().st_size
        stat = self._run_transport_with_retry("lsjson", target, "--stat", "--hash")
        try:
            metadata = json.loads(stat.stdout)
        except json.JSONDecodeError as error:
            raise ArtifactError(f"Invalid rclone metadata for {target}") from error
        if not isinstance(metadata, Mapping):
            raise ArtifactError(f"rclone metadata is not a mapping for {target}")
        hashes = metadata.get("Hashes")
        if isinstance(hashes, Mapping):
            for name, value in hashes.items():
                normalized = str(name).lower().replace("-", "").replace("_", "")
                if normalized == "sha256" and value:
                    try:
                        return str(value).lower(), int(metadata["Size"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ArtifactError(
                            f"Invalid rclone SHA-256 metadata for {target}"
                        ) from error

        # Some remotes do not expose SHA-256 metadata. Keep the streaming
        # fallback for those providers, but Google Drive normally takes the
        # metadata path above and avoids downloading the uploaded shard.
        digest = hashlib.sha256()
        size = 0
        try:
            process = subprocess.Popen(
                (str(self.rclone), "cat", target),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise ArtifactError(f"Cannot start rclone verification for {target}") from error
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        _, stderr = process.communicate()
        if process.returncode:
            raise ArtifactError(
                f"Remote verification failed for {target}: {stderr.decode(errors='replace')}"
            )
        return digest.hexdigest(), size

    def _ensure_remote_parents(self, relative_path: str) -> None:
        """Create the remote parent chain once, serialized across local workers.

        ``rclone copyto`` creates missing parents implicitly, and two workers
        publishing into the same fresh prefix can each create the same Google
        Drive directory; later path lookups then resolve nondeterministically
        between the duplicates (the publish verification failures of
        2026-09-02 and 2026-09-03).  A host-local lock around the idempotent
        ``rclone mkdir`` removes the race while uploads themselves stay
        parallel.  Pre-existing duplicates are not repaired here; they still
        fail loudly and need a manual ``rclone dedupe --dedupe-mode newest``.
        """

        parent = posixpath.dirname(relative_path)
        if not parent:
            return
        with self._ensure_lock:
            if parent in self._ensured_remote_dirs:
                return
            self._remote_mkdir_lock.parent.mkdir(parents=True, exist_ok=True)
            with self._remote_mkdir_lock.open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    self._run_transport_with_retry("mkdir", self.locator(parent))
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            self._ensured_remote_dirs.add(parent)

    def _copy(self, local_path: Path, relative_path: str) -> None:
        target = self.locator(relative_path)
        if self.remote:
            self._ensure_remote_parents(relative_path)
            self._run_transport_with_retry("copyto", str(local_path), target, "--immutable")
            return
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_sha256(destination) != file_sha256(local_path):
                raise ArtifactError(f"Refusing to replace immutable artifact {destination}")
            return
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(local_path, temporary)
        os.replace(temporary, destination)

    def _remote_sha256_fresh_upload(self, relative_path: str) -> tuple[str, int]:
        """Verify a just-uploaded object, tolerating backend indexing lag.

        Google Drive can take seconds to index a freshly created object, and
        ``lsjson --stat`` meanwhile reports "directory not found" even though
        the upload succeeded (observed on a Drive rclone remote).  Retry that
        specific miss with backoff instead of failing the job; genuine
        duplicates or content mismatches still fail after the budget.
        """

        for attempt in range(1, _RCLONE_POST_UPLOAD_ATTEMPTS + 1):
            try:
                return self._remote_sha256(relative_path)
            except ArtifactError as error:
                if attempt == _RCLONE_POST_UPLOAD_ATTEMPTS or "not found" not in str(error).lower():
                    raise
                time.sleep(_RCLONE_RETRY_DELAY_SECONDS * attempt)
        raise AssertionError("unreachable")

    def publish(
        self,
        local_path: str | Path,
        relative_path: str,
        *,
        write_receipt: bool = True,
        existing_payload_equivalence: ExistingPayloadEquivalence | None = None,
    ) -> PublishedFile:
        source = Path(local_path)
        digest = file_sha256(source)
        size = source.stat().st_size
        transport_recovery = None
        receipt_path = f"{relative_path}.receipt.json"
        if write_receipt and self.exists(receipt_path):
            receipt = self.read_json(receipt_path)
            if not self.exists(relative_path):
                raise ArtifactError(f"Receipt exists but payload is missing: {relative_path}")
            observed_digest = str(receipt.get("sha256", ""))
            observed_size = int(receipt.get("size_bytes", -1))
            if observed_digest == digest and observed_size == size:
                return PublishedFile(relative_path, digest, size)
            if source.suffix == ".safetensors" and observed_size == size:
                with tempfile.TemporaryDirectory(prefix="xai-safetensors-verify-") as directory:
                    observed = Path(directory) / source.name
                    self.materialize(
                        relative_path,
                        observed,
                        expected_sha256=observed_digest,
                    )
                    recovery = _existing_safetensors_equivalence(
                        source,
                        observed,
                        existing_payload_equivalence,
                    )
                    if recovery is not None:
                        print(
                            f"ARTIFACT_RECOVERED path={relative_path} mode={recovery}",
                            flush=True,
                        )
                        return PublishedFile(relative_path, observed_digest, observed_size)
            raise ArtifactError(f"Existing receipt contradicts {relative_path}")

        payload_exists = self.exists(relative_path)
        if not payload_exists:
            self._copy(source, relative_path)
            remote_digest, remote_size = self._remote_sha256_fresh_upload(relative_path)
        else:
            remote_digest, remote_size = self._remote_sha256(relative_path)
        if remote_digest != digest or remote_size != size:
            recovery = None
            if payload_exists and source.suffix == ".safetensors" and remote_size == size:
                with tempfile.TemporaryDirectory(prefix="xai-safetensors-recover-") as directory:
                    observed = Path(directory) / source.name
                    self.materialize(
                        relative_path,
                        observed,
                        expected_sha256=remote_digest,
                    )
                    recovery = _existing_safetensors_equivalence(
                        source,
                        observed,
                        existing_payload_equivalence,
                    )
            if recovery is None:
                raise ArtifactError(
                    f"Uploaded artifact verification failed for {relative_path}: "
                    f"local=({digest},{size}) remote=({remote_digest},{remote_size})"
                )
            digest, size = remote_digest, remote_size
            transport_recovery = recovery
        if write_receipt:
            receipt = {
                "schema_version": 1,
                "relative_path": relative_path,
                "sha256": digest,
                "size_bytes": size,
                "verified_utc": datetime.now(UTC).isoformat(),
            }
            if transport_recovery is not None:
                receipt["transport_recovery"] = transport_recovery
            with tempfile.TemporaryDirectory(prefix="xai-receipt-") as directory:
                local_receipt = Path(directory) / "receipt.json"
                atomic_write_json(local_receipt, receipt)
                if not self.exists(receipt_path):
                    self._copy(local_receipt, receipt_path)
                observed = self.read_json(receipt_path)
                if observed.get("sha256") != digest:
                    raise ArtifactError(f"Remote receipt verification failed: {receipt_path}")
            if transport_recovery is not None:
                print(
                    f"ARTIFACT_RECOVERED path={relative_path} mode={transport_recovery}",
                    flush=True,
                )
        return PublishedFile(relative_path, digest, size)

    def materialize(
        self,
        relative_path: str,
        destination: str | Path,
        *,
        expected_sha256: str,
    ) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and file_sha256(target) == expected_sha256:
            return target
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        source = self.locator(relative_path)
        try:
            if self.remote:
                self._run_transport_with_retry("copyto", source, str(temporary))
            else:
                shutil.copyfile(source, temporary)
            observed = file_sha256(temporary)
            if observed != expected_sha256:
                raise ArtifactError(
                    f"Downloaded artifact digest mismatch: expected={expected_sha256}, "
                    f"observed={observed}"
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


def _cpu_contiguous(value: Any) -> Any:
    import torch

    # Safetensors rejects fields that alias one storage. Every schema field is
    # logically independent, even when (for clean inputs) targets equal
    # predictions exactly.
    return torch.as_tensor(value).detach().to("cpu").contiguous().clone()


def write_phase1_shard(
    path: str | Path,
    *,
    indices: Any,
    labels: Any,
    predictions: Any,
    logits: Any,
    targets: Any,
    attributions: Any,
    metadata: Mapping[str, str],
) -> Path:
    """Atomically write the full signed attribution map in FP32."""

    import torch
    from safetensors.torch import save_file

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "indices": _cpu_contiguous(indices).to(dtype=torch.int64),
        "labels": _cpu_contiguous(labels).to(dtype=torch.int64),
        "predictions": _cpu_contiguous(predictions).to(dtype=torch.int64),
        "logits": _cpu_contiguous(logits).to(dtype=torch.float32),
        "targets": _cpu_contiguous(targets).to(dtype=torch.int64),
        "attributions": _cpu_contiguous(attributions).to(dtype=torch.float32),
    }
    count = int(tensors["indices"].shape[0])
    if count <= 0 or any(int(value.shape[0]) != count for value in tensors.values()):
        raise ValueError("Every shard tensor must have the same non-empty leading dimension")
    if tensors["attributions"].ndim != 4:
        raise ValueError("attributions must have [N,C,H,W] shape")
    if tensors["logits"].ndim != 2 or int(tensors["logits"].shape[1]) <= 0:
        raise ValueError("logits must have [N,num_classes] shape")
    for field in ("indices", "labels", "predictions", "targets"):
        if tensors[field].ndim != 1:
            raise ValueError(f"{field} must have [N] shape")
    for field in ("logits", "attributions"):
        if not bool(torch.isfinite(tensors[field]).all()):
            raise ValueError(f"{field} contain NaN or infinite values")
    expected_predictions = torch.argmax(tensors["logits"], dim=1)
    if not torch.equal(tensors["predictions"], expected_predictions):
        raise ValueError("predictions must equal argmax(logits) for every sample")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".safetensors", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(
            tensors,
            str(temporary),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
        _canonical_safetensors_header(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_phase2_shard(
    path: str | Path,
    *,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, str],
) -> Path:
    import torch
    from safetensors.torch import save_file

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = {name: _cpu_contiguous(value) for name, value in tensors.items()}
    if "indices" not in values:
        raise ValueError("Phase 2 shard requires indices")
    count = int(values["indices"].shape[0])
    if count <= 0 or any(int(value.shape[0]) != count for value in values.values()):
        raise ValueError("Every Phase 2 tensor must align on its leading dimension")
    values["indices"] = values["indices"].to(dtype=torch.int64)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".safetensors", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(
            values,
            str(temporary),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
        _canonical_safetensors_header(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_safetensors(path: str | Path) -> Mapping[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def completed_manifest(
    store: ArtifactStore,
    root: str,
    *,
    expected_task_digest: str,
    expected_schema_version: int | None = None,
) -> Mapping[str, Any] | None:
    relative = posixpath.join(root, "manifest.json")
    if not store.exists(relative):
        return None
    manifest = store.read_json(relative)
    if manifest.get("status") != "complete":
        raise ArtifactError(f"Published manifest is not complete: {relative}")
    if manifest.get("task_digest") != expected_task_digest:
        raise ArtifactError(f"Published manifest belongs to a different task: {relative}")
    if (
        expected_schema_version is not None
        and manifest.get("schema_version") != expected_schema_version
    ):
        raise ArtifactError(
            f"Published manifest has schema_version={manifest.get('schema_version')!r}, "
            f"expected {expected_schema_version}: {relative}"
        )
    return manifest


__all__ = [
    "ADVERSARIAL_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactStore",
    "PHASE1_SCHEMA_VERSION",
    "PHASE2_SCHEMA_VERSION",
    "PublishedFile",
    "adversarial_artifact_root",
    "completed_manifest",
    "load_safetensors",
    "phase1_artifact_root",
    "phase2_artifact_root",
    "write_phase1_shard",
    "write_phase2_shard",
]
