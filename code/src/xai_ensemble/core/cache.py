"""Process-safe bounded LRU cache for remotely hosted artifact shards."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .atomic import atomic_copy, sha256_file


class CacheError(RuntimeError):
    pass


class CacheCapacityError(CacheError):
    pass


class CachedShardValidationError(CacheError):
    pass


@dataclass(frozen=True, slots=True)
class CacheStats:
    files: int
    bytes: int
    max_bytes: int


class ShardCache:
    """A cache whose published contents never exceed ``max_bytes``.

    Access time is represented by an explicitly touched mtime, avoiding
    dependence on filesystem atime mount options. A small advisory lock makes
    lookup, publication, and eviction safe across local worker processes.
    """

    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock_path = self.root / ".cache.lock"
        self._key_lock_dir = self.root / ".locks"
        self._key_lock_dir.mkdir(exist_ok=True)

    @contextmanager
    def _advisory_lock(self, path: Path) -> Iterator[None]:
        try:
            import fcntl
        except ImportError as error:
            raise CacheError("ShardCache requires POSIX fcntl locking") from error
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _lock(self) -> Iterator[None]:
        return self._advisory_lock(self._lock_path)

    def _key_lock(self, key: str) -> Iterator[None]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._advisory_lock(self._key_lock_dir / f"{digest}.lock")

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        suffix = PurePosixPath(key).suffix.lower()
        if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
            suffix = ".shard"
        return self.root / f"{digest}{suffix}"

    def _files(self) -> list[Path]:
        return [
            path for path in self.root.iterdir() if path.is_file() and not path.name.startswith(".")
        ]

    @staticmethod
    def _valid(
        path: Path,
        expected_sha256: str | None,
        expected_size: int | None,
    ) -> bool:
        if not path.is_file():
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return expected_sha256 is None or sha256_file(path) == expected_sha256

    def _evict(self, *, protected: Path | None = None) -> None:
        files = self._files()
        total = sum(path.stat().st_size for path in files)
        if total <= self.max_bytes:
            return
        for victim in sorted(files, key=lambda path: (path.stat().st_mtime_ns, path.name)):
            if protected is not None and victim == protected:
                continue
            size = victim.stat().st_size
            try:
                victim.unlink()
            except FileNotFoundError:
                continue
            total -= size
            if total <= self.max_bytes:
                break
        if total > self.max_bytes:
            raise CacheCapacityError("cache cannot satisfy its byte bound")

    def get(
        self,
        key: str,
        fetch: Callable[[Path], None],
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> Path:
        """Return a validated local shard, fetching and evicting on a miss."""

        if expected_size is not None and expected_size > self.max_bytes:
            raise CacheCapacityError(
                f"shard size {expected_size} exceeds cache capacity {self.max_bytes}"
            )
        destination = self._path_for_key(key)
        # Different shards may download concurrently. A per-key lock prevents
        # duplicate transfers, while the short global sections serialize only
        # publication and eviction so two GPU workers are not network-bound in
        # series.
        with self._key_lock(key):
            with self._lock():
                if self._valid(destination, expected_sha256, expected_size):
                    now = time.time_ns()
                    os.utime(destination, ns=(now, now))
                    return destination
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".part", dir=self.root
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                fetch(temporary)
                size = temporary.stat().st_size
                if size > self.max_bytes:
                    raise CacheCapacityError(
                        f"downloaded shard size {size} exceeds cache capacity {self.max_bytes}"
                    )
                if expected_size is not None and size != expected_size:
                    raise CachedShardValidationError(
                        f"downloaded shard has {size} bytes, expected {expected_size}"
                    )
                if expected_sha256 is not None:
                    actual_hash = sha256_file(temporary)
                    if actual_hash != expected_sha256:
                        raise CachedShardValidationError(
                            f"downloaded shard SHA-256 {actual_hash}, expected {expected_sha256}"
                        )
                with self._lock():
                    os.replace(temporary, destination)
                    now = time.time_ns()
                    os.utime(destination, ns=(now, now))
                    self._evict(protected=destination)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            return destination

    @contextmanager
    def use(
        self,
        key: str,
        fetch: Callable[[Path], None],
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> Iterator[Path]:
        """Yield a validated cache path while preventing an eviction race.

        The global lock is held only while the caller opens or copies the
        cached payload. Ordinary ``get`` misses still download concurrently;
        another miss briefly waits when a consumer is actively opening a hit.
        """

        while True:
            cached = self.get(
                key,
                fetch,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            with self._lock():
                if not self._valid(cached, expected_sha256, expected_size):
                    continue
                now = time.time_ns()
                os.utime(cached, ns=(now, now))
                yield cached
                return

    def remove(self, key: str) -> None:
        with self._lock():
            try:
                self._path_for_key(key).unlink()
            except FileNotFoundError:
                pass

    def materialize(
        self,
        key: str,
        fetch: Callable[[Path], None],
        destination: str | os.PathLike[str],
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> Path:
        """Atomically copy a validated entry without an eviction race.

        ``get()`` intentionally returns an ordinary cache path, which is
        convenient for short-lived reads but cannot pin that inode after the
        method returns.  Formal consumers materializing a shard must use this
        operation: it revalidates and copies while holding the global eviction
        lock.  If another process evicted the entry in the narrow interval
        after ``get()``, the loop simply fetches it again.
        """

        while True:
            cached = self.get(
                key,
                fetch,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            with self._lock():
                if not self._valid(cached, expected_sha256, expected_size):
                    continue
                now = time.time_ns()
                os.utime(cached, ns=(now, now))
                atomic_copy(cached, destination)
                return Path(destination)

    def clear(self) -> None:
        with self._lock():
            for path in self._files():
                path.unlink()

    def stats(self) -> CacheStats:
        with self._lock():
            files = self._files()
            return CacheStats(
                files=len(files),
                bytes=sum(path.stat().st_size for path in files),
                max_bytes=self.max_bytes,
            )
