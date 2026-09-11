"""Cross-process capacity control for the RAM-backed publication spool."""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class SpoolCapacityError(RuntimeError):
    """The configured spool cannot accept another staged artifact."""


def _process_start_time(pid: int) -> str | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, separator, suffix = value.rpartition(")")
    if not separator:
        return None
    fields = suffix.split()
    return fields[19] if len(fields) > 19 else None


def _process_is_stale(pid: int, expected_start_time: str) -> bool:
    """Return whether a recorded process identity is definitely stale.

    A transient procfs failure (for example ``EMFILE`` while a publisher is
    opening many files) is inconclusive.  Keep the record in that case and
    let a later pass retry; pruning it could release a live reservation.
    """

    if pid <= 0:
        return True
    current_start_time = _process_start_time(pid)
    if current_start_time is not None:
        return current_start_time != str(expected_start_time)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


# Per-attempt driver-level wait for a held write lock.  Quota release is
# bookkeeping, not latency-critical work: on 2026-09-08 a tissuemnist
# IntegratedGradients worker died at quota release after the old 2 s budget
# was exhausted by concurrent publication writers, discarding the finished
# GPU computation.  Waiting out publication bursts is strictly better than
# failing the job.
_SQLITE_BUSY_TIMEOUT_SECONDS = 30.0
_SQLITE_BUSY_RETRY_ATTEMPTS = 8
_SQLITE_BUSY_RETRY_MAX_DELAY_SECONDS = 0.5


def _is_sqlite_busy(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "database is locked" in message or "database is busy" in message


def _sqlite_retry_delay(attempt: int) -> float:
    return min(0.05 * (2**attempt), _SQLITE_BUSY_RETRY_MAX_DELAY_SECONDS)


def _run_sqlite_transaction(
    connect: Callable[[], sqlite3.Connection],
    operation: Callable[[sqlite3.Connection], object],
    *,
    immediate: bool = True,
) -> object:
    """Run one SQLite operation with bounded recovery from writer contention."""

    for attempt in range(_SQLITE_BUSY_RETRY_ATTEMPTS):
        connection = connect()
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            result = operation(connection)
            connection.commit()
        except sqlite3.OperationalError as error:
            try:
                connection.rollback()
            finally:
                connection.close()
            if not _is_sqlite_busy(error) or attempt + 1 >= _SQLITE_BUSY_RETRY_ATTEMPTS:
                raise
            time.sleep(_sqlite_retry_delay(attempt))
        except BaseException:
            try:
                connection.rollback()
            finally:
                connection.close()
            raise
        else:
            connection.close()
            return result
    raise AssertionError("unreachable SQLite retry loop")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


@dataclass(frozen=True, slots=True)
class SpoolReservation:
    token: str
    byte_count: int
    work_directory: Path


@dataclass(frozen=True, slots=True)
class UploadSlot:
    """One process-scoped permit for a remote publication operation."""

    token: str
    pid: int
    process_start_time: str


class UploadGate:
    """Bound concurrent remote uploads across all Phase 1 worker processes.

    The gate intentionally lives beside ``SpoolQuota`` so every publisher in
    one experiment shares the same bounded state. Claims include the Linux
    process start time; a crashed worker therefore cannot permanently consume
    a slot after the next claimant prunes its stale record.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_slots: int,
        poll_seconds: float = 0.05,
    ) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_slots = int(max_slots)
        self.poll_seconds = float(poll_seconds)
        self.database_path = self.root / ".quota.sqlite3"
        self.pid = os.getpid()
        self.process_start_time = _process_start_time(self.pid)
        if self.process_start_time is None:
            raise RuntimeError(f"Cannot identify upload-gate owner process {self.pid}")

        def create_schema(connection: sqlite3.Connection) -> None:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_slots (
                    token TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    process_start_time TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

        _run_sqlite_transaction(self._connect, create_schema, immediate=False)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=_SQLITE_BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        return connection

    def _prune_stale(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT token,pid,process_start_time FROM upload_slots"
        ).fetchall()
        stale = [
            str(token)
            for token, pid, process_start_time in rows
            if _process_is_stale(int(pid), str(process_start_time))
        ]
        if stale:
            connection.executemany(
                "DELETE FROM upload_slots WHERE token=?",
                ((token,) for token in stale),
            )

    def acquire(
        self,
        *,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> UploadSlot:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        token = uuid.uuid4().hex
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise SpoolCapacityError("Upload slot acquisition was cancelled")

            def claim(connection: sqlite3.Connection) -> bool:
                self._prune_stale(connection)
                count = int(connection.execute("SELECT COUNT(*) FROM upload_slots").fetchone()[0])
                if count < self.max_slots:
                    connection.execute(
                        """
                        INSERT INTO upload_slots(token,pid,process_start_time,created_at)
                        VALUES(?,?,?,?)
                        """,
                        (token, self.pid, self.process_start_time, time.time()),
                    )
                    return True
                return False

            acquired = bool(_run_sqlite_transaction(self._connect, claim))
            if acquired:
                return UploadSlot(token, self.pid, self.process_start_time)
            if deadline is not None and time.monotonic() >= deadline:
                raise SpoolCapacityError("Timed out waiting for a global upload slot")
            time.sleep(self.poll_seconds)

    def release(self, slot: UploadSlot) -> None:
        def remove(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                DELETE FROM upload_slots
                WHERE token=? AND pid=? AND process_start_time=?
                """,
                (slot.token, self.pid, self.process_start_time),
            )

        _run_sqlite_transaction(self._connect, remove)

    def active_slots(self) -> int:
        def count_active(connection: sqlite3.Connection) -> int:
            self._prune_stale(connection)
            return int(connection.execute("SELECT COUNT(*) FROM upload_slots").fetchone()[0])

        return int(_run_sqlite_transaction(self._connect, count_active))


class SpoolQuota:
    """Serialize a global byte budget shared by all Phase 1 worker processes."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int,
        min_free_bytes: int,
        poll_seconds: float = 0.25,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.poll_seconds = float(poll_seconds)
        self.database_path = self.root / ".quota.sqlite3"
        self.pid = os.getpid()
        self.process_start_time = _process_start_time(self.pid)
        if self.process_start_time is None:
            raise RuntimeError(f"Cannot identify spool owner process {self.pid}")

        def create_schema(connection: sqlite3.Connection) -> None:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    token TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    process_start_time TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    work_directory TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

        _run_sqlite_transaction(self._connect, create_schema, immediate=False)

        def create_waiters(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS waiters (
                    token TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    process_start_time TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

        _run_sqlite_transaction(self._connect, create_waiters, immediate=False)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=_SQLITE_BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        return connection

    def _stale_directories(self, connection: sqlite3.Connection) -> tuple[Path, ...]:
        rows = connection.execute(
            "SELECT token,pid,process_start_time,work_directory FROM reservations"
        ).fetchall()
        stale_tokens = []
        stale_directories = set()
        for token, pid, process_start_time, work_directory in rows:
            if not _process_is_stale(int(pid), str(process_start_time)):
                continue
            stale_tokens.append(str(token))
            stale_directories.add(Path(str(work_directory)).resolve())
        if stale_tokens:
            connection.executemany(
                "DELETE FROM reservations WHERE token=?",
                ((token,) for token in stale_tokens),
            )
        waiter_rows = connection.execute(
            "SELECT token,pid,process_start_time FROM waiters"
        ).fetchall()
        stale_waiters = [
            str(token)
            for token, pid, process_start_time in waiter_rows
            if _process_is_stale(int(pid), str(process_start_time))
        ]
        if stale_waiters:
            connection.executemany(
                "DELETE FROM waiters WHERE token=?",
                ((token,) for token in stale_waiters),
            )
        return tuple(stale_directories)

    def _remove_stale_directories(self, directories: tuple[Path, ...]) -> None:
        for directory in directories:
            if _is_within(directory, self.root):
                shutil.rmtree(directory, ignore_errors=True)

    def acquire(
        self,
        byte_count: int,
        *,
        work_directory: str | Path,
        timeout_seconds: float | None = None,
        priority: int = 10,
        cancel_event: threading.Event | None = None,
    ) -> SpoolReservation:
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        directory = Path(work_directory).expanduser().resolve()
        if not _is_within(directory, self.root):
            raise ValueError(f"Spool work directory must be below {self.root}: {directory}")
        if byte_count > self.max_bytes:
            raise SpoolCapacityError(
                f"One staged artifact needs {byte_count} bytes, above the "
                f"configured spool limit {self.max_bytes}"
            )
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        token = uuid.uuid4().hex
        registered = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise SpoolCapacityError(
                        f"Cancelled reservation for {byte_count} bytes below {self.root}"
                    )

                def reserve(
                    connection: sqlite3.Connection,
                    *,
                    waiter_registered: bool = registered,
                ) -> tuple[tuple[Path, ...], bool, bool]:
                    stale_directories = self._stale_directories(connection)
                    registered_now = False
                    if not waiter_registered:
                        connection.execute(
                            """
                            INSERT INTO waiters(
                                token,pid,process_start_time,byte_count,priority,created_at
                            ) VALUES(?,?,?,?,?,?)
                            """,
                            (
                                token,
                                self.pid,
                                self.process_start_time,
                                byte_count,
                                int(priority),
                                time.time(),
                            ),
                        )
                        registered_now = True
                    reserved = False
                    if not stale_directories:
                        used = int(
                            connection.execute(
                                "SELECT COALESCE(SUM(byte_count),0) FROM reservations"
                            ).fetchone()[0]
                        )
                        free = shutil.disk_usage(self.root).free
                        available = max(
                            0,
                            min(
                                self.max_bytes - used,
                                int(free) - self.min_free_bytes,
                            ),
                        )
                        # Preserve priority among requests that can make
                        # progress now. A large blocked prefetch must not stop
                        # a small staged result whose upload would free space.
                        first = connection.execute(
                            """
                            SELECT token FROM waiters
                            WHERE byte_count <= ?
                            ORDER BY priority ASC,created_at ASC,token ASC
                            LIMIT 1
                            """,
                            (available,),
                        ).fetchone()
                        if first is not None and str(first[0]) == token:
                            connection.execute(
                                """
                                INSERT INTO reservations(
                                    token,pid,process_start_time,byte_count,
                                    work_directory,created_at
                                ) VALUES(?,?,?,?,?,?)
                                """,
                                (
                                    token,
                                    self.pid,
                                    self.process_start_time,
                                    byte_count,
                                    str(directory),
                                    time.time(),
                                ),
                            )
                            connection.execute("DELETE FROM waiters WHERE token=?", (token,))
                            reserved = True
                    return stale_directories, reserved, registered_now

                stale_directories, reserved, registered_now = tuple(
                    _run_sqlite_transaction(self._connect, reserve)
                )
                if registered_now:
                    registered = True
                if stale_directories:
                    self._remove_stale_directories(stale_directories)
                    continue
                if reserved:
                    return SpoolReservation(token, byte_count, directory)
                if deadline is not None and time.monotonic() >= deadline:
                    raise SpoolCapacityError(
                        f"Timed out reserving {byte_count} bytes below {self.root}"
                    )
                time.sleep(self.poll_seconds)
        except BaseException:
            if registered:

                def remove_waiter(connection: sqlite3.Connection) -> None:
                    connection.execute("DELETE FROM waiters WHERE token=?", (token,))

                try:
                    _run_sqlite_transaction(self._connect, remove_waiter)
                except Exception:
                    # A later claimant will prune an orphaned waiter.  Keep
                    # the original acquisition error as the useful failure.
                    pass
            raise

    def release(self, reservation: SpoolReservation) -> None:
        def remove(connection: sqlite3.Connection) -> None:
            connection.execute(
                "DELETE FROM reservations WHERE token=? AND pid=? AND process_start_time=?",
                (reservation.token, self.pid, self.process_start_time),
            )

        _run_sqlite_transaction(self._connect, remove)

    def reserved_bytes(self) -> int:
        def count_reserved(connection: sqlite3.Connection) -> int:
            return int(
                connection.execute(
                    "SELECT COALESCE(SUM(byte_count),0) FROM reservations"
                ).fetchone()[0]
            )

        return int(_run_sqlite_transaction(self._connect, count_reserved, immediate=False))


__all__ = ["SpoolCapacityError", "SpoolQuota", "SpoolReservation", "UploadGate", "UploadSlot"]
