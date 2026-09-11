"""Stable identity bridge for timestamp-only dataset-manifest regeneration."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.data import read_manifest

MANIFEST_IDENTITY_SCHEMA = "simple-dataset-manifest-identity-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def manifest_identity_lock_path(path: str | Path) -> Path:
    source = resolve_full_matrix_runtime_path(path)
    return Path(f"{source}.identity.json")


def _sha256_field(value: Mapping[str, Any], key: str, *, path: Path) -> str:
    result = value.get(key)
    if not isinstance(result, str) or _SHA256.fullmatch(result) is None:
        raise ValueError(f"{path}: {key} must be a lowercase SHA-256")
    return result


@lru_cache(maxsize=64)
def _validated_identity_sha256(
    source_value: str,
    source_size: int,
    source_mtime_ns: int,
    lock_size: int,
    lock_mtime_ns: int,
) -> str:
    del source_size, source_mtime_ns, lock_size, lock_mtime_ns
    source = Path(source_value)
    current_sha256 = file_sha256(source)
    lock_path = manifest_identity_lock_path(source)
    if not lock_path.is_file():
        return current_sha256

    value = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{lock_path}: manifest identity lock must be a mapping")
    if value.get("schema") != MANIFEST_IDENTITY_SCHEMA:
        raise ValueError(f"{lock_path}: unsupported manifest identity lock schema")
    materialized_sha256 = _sha256_field(value, "materialized_sha256", path=lock_path)
    identity_sha256 = _sha256_field(value, "identity_sha256", path=lock_path)
    if current_sha256 != materialized_sha256:
        raise ValueError(
            f"{lock_path}: materialized manifest SHA-256 changed: "
            f"{current_sha256} != {materialized_sha256}"
        )

    manifest = read_manifest(source)
    expected_fingerprint = _sha256_field(
        value,
        "dataset_manifest_fingerprint",
        path=lock_path,
    )
    if manifest.fingerprint != expected_fingerprint:
        raise ValueError(f"{lock_path}: stable dataset manifest fingerprint changed")
    if manifest.metadata.dataset_id != value.get("dataset_id"):
        raise ValueError(f"{lock_path}: dataset_id changed")
    if manifest.metadata.revision != value.get("dataset_revision"):
        raise ValueError(f"{lock_path}: dataset_revision changed")
    if len(manifest.records) != value.get("record_count"):
        raise ValueError(f"{lock_path}: manifest record count changed")
    expected_splits = value.get("split_sizes")
    if not isinstance(expected_splits, Mapping):
        raise ValueError(f"{lock_path}: split_sizes must be a mapping")
    normalized_splits = {str(key): int(count) for key, count in expected_splits.items()}
    if dict(manifest.audit().split_sizes) != normalized_splits:
        raise ValueError(f"{lock_path}: manifest split sizes changed")
    return identity_sha256


def dataset_manifest_identity_sha256(path: str | Path) -> str:
    """Return the immutable source SHA, validating any explicit recovery lock.

    Dataset manifest fingerprints deliberately exclude ``created_utc``, while
    the original simplified pipeline bound artifacts to the whole-file hash.
    A recovery lock permits a timestamp-only regenerated file to retain that
    already-published identity. It is intentionally strict: the exact current
    bytes and all stable manifest identity fields must match the lock.
    """

    source = resolve_full_matrix_runtime_path(path).resolve()
    lock_path = manifest_identity_lock_path(source)
    source_stat = source.stat()
    if lock_path.is_file():
        lock_stat = lock_path.stat()
        lock_size, lock_mtime_ns = lock_stat.st_size, lock_stat.st_mtime_ns
    else:
        lock_size, lock_mtime_ns = -1, -1
    return _validated_identity_sha256(
        str(source),
        source_stat.st_size,
        source_stat.st_mtime_ns,
        lock_size,
        lock_mtime_ns,
    )


__all__ = [
    "MANIFEST_IDENTITY_SCHEMA",
    "dataset_manifest_identity_sha256",
    "manifest_identity_lock_path",
]
