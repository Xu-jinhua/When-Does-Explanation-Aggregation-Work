"""Bounded overlap between CloudStorage I/O, CPU staging, and GPU work."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections.abc import Callable, Hashable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from .spool import SpoolQuota, SpoolReservation

T = TypeVar("T")
S = TypeVar("S")


@dataclass(frozen=True, slots=True)
class PrefetchItem(Generic[T]):
    """One ordered input unit whose reservation lasts until GPU consumption."""

    key: Hashable
    byte_count: int
    load: Callable[[Path], T]
    cleanup: Callable[[T], None] | None = None

    def __post_init__(self) -> None:
        if self.byte_count <= 0:
            raise ValueError("prefetch byte_count must be positive")


@dataclass(frozen=True, slots=True)
class Prefetched(Generic[T]):
    value: T
    wait_seconds: float
    load_seconds: float
    reserved_bytes: int


@dataclass(slots=True)
class _Loaded(Generic[T]):
    value: T
    reservation: SpoolReservation
    load_seconds: float
    cleanup: Callable[[T], None] | None


class ByteBoundedPrefetcher(Generic[T]):
    """Prefetch every ordered item while a global byte quota bounds lookahead.

    All items are submitted immediately, but only ``workers`` loaders can run
    and each loader must first obtain a cross-process reservation.  Small
    inputs can therefore advance through the whole sequence, while large
    attribution shards naturally settle at N+1, N+2, or N+3 according to the
    available tmpfs budget.  Reservations use a higher priority than output
    publication so future downloads are not queued behind new uploads.
    """

    def __init__(
        self,
        quota: SpoolQuota,
        items: Iterable[PrefetchItem[T]],
        *,
        workers: int,
        namespace: str,
        priority: int = -1_000_000,
    ) -> None:
        if workers <= 0:
            raise ValueError("prefetch workers must be positive")
        self.quota = quota
        self.items = tuple(items)
        self.priority = int(priority)
        keys = [item.key for item in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("prefetch keys must be unique")
        self.work_directory = (
            quota.root / "prefetch" / namespace / f"{quota.pid}-{uuid.uuid4().hex}"
        )
        self.work_directory.mkdir(parents=True, exist_ok=False)
        self._cancel = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"prefetch-{namespace}",
        )
        self._futures: dict[Hashable, Future[_Loaded[T]]] = {}
        self._loaded: dict[Hashable, _Loaded[T]] = {}
        self._closed = False
        self._admission_events = [threading.Event() for _ in range(len(self.items) + 1)]
        self._admission_events[0].set()
        for position, item in enumerate(self.items):
            item_directory = self.work_directory / f"{position:05d}"
            self._futures[item.key] = self._executor.submit(
                self._load_one,
                item,
                item_directory,
                position,
            )

    def _load_one(
        self,
        item: PrefetchItem[T],
        directory: Path,
        position: int,
    ) -> _Loaded[T]:
        self._admission_events[position].wait()
        try:
            reservation = self.quota.acquire(
                item.byte_count,
                work_directory=self.work_directory,
                priority=self.priority + position,
                cancel_event=self._cancel,
            )
        finally:
            self._admission_events[position + 1].set()
        directory.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        try:
            value = item.load(directory)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            self.quota.release(reservation)
            raise
        return _Loaded(
            value=value,
            reservation=reservation,
            load_seconds=time.monotonic() - started,
            cleanup=item.cleanup,
        )

    def get(self, key: Hashable) -> Prefetched[T]:
        if self._closed:
            raise RuntimeError("prefetcher is closed")
        if key in self._loaded:
            raise RuntimeError(f"prefetch item {key!r} is already checked out")
        try:
            future = self._futures.pop(key)
        except KeyError as error:
            raise KeyError(f"Unknown prefetch item {key!r}") from error
        started = time.monotonic()
        loaded = future.result()
        self._loaded[key] = loaded
        return Prefetched(
            value=loaded.value,
            wait_seconds=time.monotonic() - started,
            load_seconds=loaded.load_seconds,
            reserved_bytes=loaded.reservation.byte_count,
        )

    def release(self, key: Hashable) -> None:
        try:
            loaded = self._loaded.pop(key)
        except KeyError as error:
            raise KeyError(f"Prefetch item {key!r} is not checked out") from error
        try:
            if loaded.cleanup is not None:
                loaded.cleanup(loaded.value)
        finally:
            self.quota.release(loaded.reservation)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        for future in self._futures.values():
            future.cancel()
        for loaded in self._loaded.values():
            try:
                if loaded.cleanup is not None:
                    loaded.cleanup(loaded.value)
            finally:
                self.quota.release(loaded.reservation)
        self._loaded.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for future in self._futures.values():
            if future.cancelled():
                continue
            try:
                loaded = future.result()
            except BaseException:
                continue
            try:
                if loaded.cleanup is not None:
                    loaded.cleanup(loaded.value)
            finally:
                self.quota.release(loaded.reservation)
        self._futures.clear()
        shutil.rmtree(self.work_directory, ignore_errors=True)

    def __enter__(self) -> ByteBoundedPrefetcher[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncSpoolWriter:
    """Serialize in tmpfs and upload on independent background threads."""

    def __init__(
        self,
        quota: SpoolQuota,
        *,
        namespace: str,
        stage_workers: int = 1,
        upload_workers: int = 1,
    ) -> None:
        if stage_workers <= 0 or upload_workers <= 0:
            raise ValueError("stage and upload worker counts must be positive")
        self.quota = quota
        self.work_directory = quota.root / "publish" / namespace / f"{quota.pid}-{uuid.uuid4().hex}"
        self.work_directory.mkdir(parents=True, exist_ok=False)
        self._stage = ThreadPoolExecutor(
            max_workers=stage_workers,
            thread_name_prefix=f"stage-{namespace}",
        )
        self._upload = ThreadPoolExecutor(
            max_workers=upload_workers,
            thread_name_prefix=f"upload-{namespace}",
        )
        self._futures: list[Future[Any]] = []
        self._failure: BaseException | None = None
        self._lock = threading.Lock()
        self._closed = False

    def _record_failure(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error

    def check(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("Background artifact staging or upload failed") from failure

    def submit(
        self,
        *,
        byte_count: int,
        basename: str,
        stage: Callable[[Path], S],
        upload: Callable[[Path, S], T],
    ) -> Future[T]:
        if self._closed:
            raise RuntimeError("async spool writer is closed")
        self.check()
        token = uuid.uuid4().hex
        path = self.work_directory / f"{token}-{Path(basename).name}"
        result: Future[T] = Future()

        def publish_staged(staged: S, reservation: SpoolReservation) -> None:
            try:
                self.check()
                value = upload(path, staged)
            except BaseException as error:
                self._record_failure(error)
                result.set_exception(error)
            else:
                result.set_result(value)
            finally:
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
                self.quota.release(reservation)

        def stage_then_enqueue() -> None:
            reservation: SpoolReservation | None = None
            try:
                self.check()
                reservation = self.quota.acquire(
                    byte_count,
                    work_directory=self.work_directory,
                    priority=10,
                )
                staged = stage(path)
                if not path.is_file():
                    raise RuntimeError("staging callback did not create its payload")
                if path.stat().st_size > reservation.byte_count:
                    raise RuntimeError("staged payload exceeded its spool reservation")
                self._upload.submit(publish_staged, staged, reservation)
            except BaseException as error:
                self._record_failure(error)
                path.unlink(missing_ok=True)
                if reservation is not None:
                    self.quota.release(reservation)
                result.set_exception(error)

        try:
            self._stage.submit(stage_then_enqueue)
        except BaseException:
            raise
        self._futures.append(result)
        return result

    @property
    def futures(self) -> tuple[Future[Any], ...]:
        return tuple(self._futures)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stage.shutdown(wait=True)
        self._upload.shutdown(wait=True)
        try:
            self.check()
        finally:
            shutil.rmtree(self.work_directory, ignore_errors=True)

    def __enter__(self) -> AsyncSpoolWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


__all__ = [
    "AsyncSpoolWriter",
    "ByteBoundedPrefetcher",
    "PrefetchItem",
    "Prefetched",
]
