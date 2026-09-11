"""Crash-safe local file helpers.

The artifact and job layers write small pieces of authoritative state locally.
Those writes must never expose a partially written file to another process.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from errno import ENOSYS, EOPNOTSUPP
from pathlib import Path
from typing import Any, BinaryIO, TextIO


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest for *path* without loading it in RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Persist a rename/link on POSIX; silently degrade on unsupported systems."""

    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _immutable_create_lock(destination: Path) -> Iterator[None]:
    """Serialize a FUSE create-if-absent fallback with local POSIX state.

    Some mounted object stores implement atomic rename but not hard links.  A
    configured local lock root lets cooperative workers retain immutable
    manifest publication without falling back to an unsafe check-and-replace.
    """

    configured = os.environ.get("XAI_CLOUD_STORAGE_LOCK_ROOT")
    if not configured:
        raise OSError(
            ENOSYS,
            "atomic immutable creation needs XAI_CLOUD_STORAGE_LOCK_ROOT when link is unsupported",
            str(destination),
        )
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - server workers are POSIX
        raise OSError(ENOSYS, "atomic immutable creation requires POSIX flock") from error

    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = os.path.abspath(destination)
    lock_name = hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".lock"
    with (root / lock_name).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _publish_immutable_without_link(temporary: Path, destination: Path) -> None:
    """Publish an immutable file on a FUSE mount lacking ``link(2)``.

    The local lock is shared by all configured experiment workers.  The final
    rename is atomic on the mounted filesystem, while the pre-rename check
    preserves create-if-absent semantics for cooperative publishers.
    """

    with _immutable_create_lock(destination):
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)


@contextmanager
def atomic_open(
    path: str | os.PathLike[str],
    mode: str = "wb",
    *,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> Iterator[BinaryIO | TextIO]:
    """Yield a temporary handle and atomically publish it when the context exits.

    ``overwrite=False`` uses a hard-link publication step, making the existence
    check race-free on the local POSIX filesystems used by the experiment
    servers.  CloudStorage FUSE mounts that lack ``link(2)`` use a configured
    local lock plus atomic rename instead.
    """

    if mode not in {"wb", "w"}:
        raise ValueError("atomic_open supports only 'wb' and 'w'")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    handle: BinaryIO | TextIO
    try:
        if mode == "wb":
            handle = os.fdopen(descriptor, "wb")
        else:
            handle = os.fdopen(descriptor, "w", encoding=encoding, newline="")
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

        if overwrite:
            os.replace(temporary, destination)
        else:
            # link() fails with FileExistsError without a check-then-write race.
            try:
                os.link(temporary, destination)
            except OSError as error:
                if error.errno not in {ENOSYS, EOPNOTSUPP}:
                    raise
                _publish_immutable_without_link(temporary, destination)
            else:
                temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_bytes(
    path: str | os.PathLike[str], data: bytes, *, overwrite: bool = True
) -> None:
    with atomic_open(path, "wb", overwrite=overwrite) as handle:
        handle.write(data)


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> None:
    with atomic_open(path, "w", encoding=encoding, overwrite=overwrite) as handle:
        handle.write(text)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for manifests and content hashes."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_json(
    path: str | os.PathLike[str], value: Any, *, overwrite: bool = True
) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def atomic_copy(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    overwrite: bool = True,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Copy a file while publishing the destination atomically."""

    with Path(source).open("rb") as source_handle, atomic_open(
        destination, "wb", overwrite=overwrite
    ) as destination_handle:
        while chunk := source_handle.read(chunk_size):
            destination_handle.write(chunk)
