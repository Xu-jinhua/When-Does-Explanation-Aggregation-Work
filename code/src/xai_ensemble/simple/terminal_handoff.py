"""Classify a simple scheduler database for terminal experiment handoff."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueueTerminalState:
    database: str
    exists: bool
    total: int = 0
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    blocked: int = 0
    exhausted_failed: int = 0
    non_exhausted_failed: int = 0

    @property
    def retry_exhausted(self) -> bool:
        return (
            self.exists
            and self.total > 0
            and self.pending == 0
            and self.running == 0
            and self.failed > 0
            and self.exhausted_failed == self.failed
            and self.non_exhausted_failed == 0
        )

    @property
    def terminal_success(self) -> bool:
        return (
            self.exists
            and self.total > 0
            and self.pending == 0
            and self.running == 0
            and self.failed == 0
            and self.blocked == 0
            and self.succeeded == self.total
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "retry_exhausted": self.retry_exhausted,
            "terminal_success": self.terminal_success,
        }


def inspect_queue_terminal_state(database: str | Path) -> QueueTerminalState:
    path = Path(database).resolve()
    if not path.is_file():
        return QueueTerminalState(database=str(path), exists=False)

    uri = f"{path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(status = 'pending'), 0) AS pending,
                COALESCE(SUM(status = 'running'), 0) AS running,
                COALESCE(SUM(status = 'succeeded'), 0) AS succeeded,
                COALESCE(SUM(status = 'failed'), 0) AS failed,
                COALESCE(SUM(status = 'blocked'), 0) AS blocked,
                COALESCE(SUM(status = 'failed' AND attempts > max_retries), 0)
                    AS exhausted_failed,
                COALESCE(SUM(status = 'failed' AND attempts <= max_retries), 0)
                    AS non_exhausted_failed
            FROM jobs
            """
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not inspect scheduler database {path}")
    values = tuple(int(value) for value in row)
    return QueueTerminalState(str(path), True, *values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument(
        "--expect",
        choices=("retry-exhausted", "terminal-success"),
        default="retry-exhausted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state = inspect_queue_terminal_state(args.database)
    print(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    matched = state.retry_exhausted if args.expect == "retry-exhausted" else state.terminal_success
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
