from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def collect_provenance(
    project_root: str | Path,
    packages: Iterable[str] = ("torch", "torchvision", "timm", "captum", "datasets"),
) -> dict:
    root = Path(project_root).resolve()
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_value(root, "status", "--porcelain")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": versions,
    }
