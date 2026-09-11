from __future__ import annotations

import sqlite3

from xai_ensemble.simple.terminal_handoff import inspect_queue_terminal_state


def _database(tmp_path, rows):
    path = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                max_retries INTEGER NOT NULL
            )
            """
        )
        connection.executemany("INSERT INTO jobs VALUES (?, ?, ?)", rows)
    return path


def test_missing_queue_is_not_a_terminal_handoff(tmp_path) -> None:
    state = inspect_queue_terminal_state(tmp_path / "missing.sqlite3")

    assert not state.exists
    assert not state.retry_exhausted
    assert not state.terminal_success


def test_running_queue_is_not_a_terminal_handoff(tmp_path) -> None:
    state = inspect_queue_terminal_state(
        _database(tmp_path, [("succeeded", 1, 1), ("running", 1, 1)])
    )

    assert not state.retry_exhausted
    assert not state.terminal_success


def test_retry_exhausted_queue_allows_failed_handoff(tmp_path) -> None:
    state = inspect_queue_terminal_state(
        _database(
            tmp_path,
            [("succeeded", 1, 1), ("failed", 2, 1), ("blocked", 0, 1)],
        )
    )

    assert state.retry_exhausted
    assert not state.terminal_success


def test_failed_job_with_retry_remaining_is_not_exhausted(tmp_path) -> None:
    state = inspect_queue_terminal_state(_database(tmp_path, [("failed", 1, 1)]))

    assert not state.retry_exhausted


def test_all_succeeded_queue_allows_success_handoff(tmp_path) -> None:
    state = inspect_queue_terminal_state(
        _database(tmp_path, [("succeeded", 1, 1), ("succeeded", 0, 1)])
    )

    assert state.terminal_success
    assert not state.retry_exhausted
