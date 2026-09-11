"""Small scheduler-worker runtime signals for the simple pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from xai_ensemble.core.io import atomic_write_json

GPU_RELEASE_PATH_ENV = "XAI_SIMPLE_GPU_RELEASE_PATH"
GPU_RELEASE_TOKEN_ENV = "XAI_SIMPLE_GPU_RELEASE_TOKEN"
GPU_RELEASE_JOB_ENV = "XAI_SIMPLE_GPU_RELEASE_JOB"


def emit_gpu_release_signal() -> bool:
    """Tell a parent scheduler that this worker no longer owns GPU tensors."""

    values = {
        "path": os.environ.get(GPU_RELEASE_PATH_ENV),
        "token": os.environ.get(GPU_RELEASE_TOKEN_ENV),
        "job_id": os.environ.get(GPU_RELEASE_JOB_ENV),
    }
    if not any(values.values()):
        return False
    if not all(values.values()):
        raise RuntimeError("Incomplete scheduler GPU-release environment")
    path = Path(str(values["path"]))
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "job_id": values["job_id"],
            "pid": os.getpid(),
            "token": values["token"],
            "released_utc": datetime.now(UTC).isoformat(),
        },
    )
    return True


def gpu_release_signal_matches(
    path: str | Path,
    *,
    job_id: str,
    pid: int,
    token: str,
) -> bool:
    source = Path(path)
    if not source.is_file():
        return False
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("schema_version") == 1
        and value.get("job_id") == job_id
        and value.get("pid") == pid
        and value.get("token") == token
    )


__all__ = [
    "GPU_RELEASE_JOB_ENV",
    "GPU_RELEASE_PATH_ENV",
    "GPU_RELEASE_TOKEN_ENV",
    "emit_gpu_release_signal",
    "gpu_release_signal_matches",
]
