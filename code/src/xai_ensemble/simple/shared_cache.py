"""Process-shared, rebuildable tensor caches backed by mmap/tmpfs."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json


@dataclass(frozen=True, slots=True)
class SharedTensor:
    tensor: Any
    identity_digest: str
    path: Path
    cache_hit: bool
    elapsed_seconds: float


def _safe_namespace(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe cache namespace: {value!r}")
    return value


def _metadata_valid(
    metadata: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> bool:
    return (
        metadata.get("schema_version") == 1
        and metadata.get("identity") == dict(identity)
        and metadata.get("shape") == list(shape)
        and metadata.get("dtype") == dtype.str
        and metadata.get("status") == "complete"
    )


def _open_valid_cache(
    path: Path,
    metadata_path: Path,
    *,
    identity: Mapping[str, Any],
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.memmap[Any, Any] | None:
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping) or not _metadata_valid(
            metadata,
            identity=identity,
            shape=shape,
            dtype=dtype,
        ):
            return None
        values = np.load(path, mmap_mode="c", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if values.shape != shape or values.dtype != dtype:
        return None
    return values


def materialize_shared_tensor(
    root: str | Path,
    *,
    namespace: str,
    identity: Mapping[str, Any],
    shape: Sequence[int],
    dtype: Any,
    populate: Callable[[np.memmap[Any, Any]], None],
    validate: Callable[[np.ndarray[Any, Any]], None] | None = None,
) -> SharedTensor:
    """Open an immutable mmap cache, atomically constructing it once if absent."""

    import torch

    started = time.monotonic()
    cache_root = Path(root).expanduser().resolve() / _safe_namespace(namespace)
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError("shared tensor cache shape must be non-empty and positive")
    normalized_dtype = np.dtype(dtype)
    if normalized_dtype.hasobject:
        raise TypeError("shared tensor cache cannot contain object values")
    # Round-trip once so tuples and other JSON sequence forms compare exactly
    # after metadata is read back in another process.
    identity_value = json.loads(json.dumps(dict(identity), sort_keys=True))
    digest = object_sha256(identity_value)
    directory = cache_root / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.npy"
    metadata_path = path.with_suffix(".json")
    lock_path = path.with_suffix(".lock")
    cache_hit = False

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        values = _open_valid_cache(
            path,
            metadata_path,
            identity=identity_value,
            shape=normalized_shape,
            dtype=normalized_dtype,
        )
        if values is None:
            temporary = directory / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.npy"
            try:
                writable = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=normalized_dtype,
                    shape=normalized_shape,
                )
                populate(writable)
                writable.flush()
                if validate is not None:
                    validate(writable)
                del writable
                os.replace(temporary, path)
                atomic_write_json(
                    metadata_path,
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "identity": identity_value,
                        "identity_digest": digest,
                        "shape": list(normalized_shape),
                        "dtype": normalized_dtype.str,
                    },
                )
            finally:
                temporary.unlink(missing_ok=True)
            values = _open_valid_cache(
                path,
                metadata_path,
                identity=identity_value,
                shape=normalized_shape,
                dtype=normalized_dtype,
            )
            if values is None:
                raise RuntimeError(f"Failed to reopen shared tensor cache {path}")
        else:
            cache_hit = True
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # mmap_mode='c' is private-copy/writable from NumPy's perspective. Torch
    # therefore avoids the read-only warning while unchanged pages remain
    # physically shared by the kernel page cache across worker processes.
    tensor = torch.from_numpy(values)
    return SharedTensor(
        tensor=tensor,
        identity_digest=digest,
        path=path,
        cache_hit=cache_hit,
        elapsed_seconds=time.monotonic() - started,
    )


__all__ = ["SharedTensor", "materialize_shared_tensor"]
