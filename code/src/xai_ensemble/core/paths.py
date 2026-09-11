"""Path resolution for runtime values embedded in immutable job argv."""

from __future__ import annotations

import os
from pathlib import Path

FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV = "XAI_FULL_MATRIX_ASSET_OVERLAY_ROOT"
FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV = "XAI_FULL_MATRIX_CACHE_OVERLAY_ROOT"
FULL_MATRIX_LOGICAL_ASSET_ROOT_ENV = "XAI_FULL_MATRIX_LOGICAL_ASSET_ROOT"
FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV = "XAI_FULL_MATRIX_LOGICAL_CACHE_ROOT"
FULL_MATRIX_OVERLAY_DATASETS_ENV = "XAI_FULL_MATRIX_OVERLAY_DATASETS"
FULL_MATRIX_FULL_ASSET_OVERLAY_DATASETS_ENV = "XAI_FULL_MATRIX_FULL_ASSET_OVERLAY_DATASETS"

FULL_MATRIX_OVERLAY_DATASETS = frozenset(
    {
        "bloodmnist",
        "breastmnist",
        "dermamnist",
        "food101",
        "imagenet100",
        "imagenet1k",
        "octmnist",
        "organamnist",
        "organcmnist",
        "organsmnist",
        "pathmnist",
        "pneumoniamnist",
        "retinamnist",
        "tissuemnist",
    }
)
FULL_MATRIX_FULL_ASSET_OVERLAY_DATASETS = frozenset(
    {"octmnist", "organamnist", "organcmnist", "organsmnist"}
)

_CANONICAL_PILOT_PATHS = frozenset(
    {
        Path("configs/pilots"),
        Path("code/configs/pilots"),
    }
)


def _package_code_root() -> Path:
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / "configs" / "pilots").is_dir():
            return parent
    return source.parents[3]


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    """Normalize a path without resolving symlinks or touching its contents."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _overlay_datasets() -> frozenset[str]:
    raw = os.environ.get(FULL_MATRIX_OVERLAY_DATASETS_ENV)
    if raw is None:
        return FULL_MATRIX_OVERLAY_DATASETS
    values = frozenset(item.strip() for item in raw.split(",") if item.strip())
    unknown = values - FULL_MATRIX_OVERLAY_DATASETS
    if unknown:
        raise ValueError(
            f"{FULL_MATRIX_OVERLAY_DATASETS_ENV} contains unsupported datasets: {sorted(unknown)}"
        )
    return values


def _full_asset_overlay_datasets() -> frozenset[str]:
    raw = os.environ.get(FULL_MATRIX_FULL_ASSET_OVERLAY_DATASETS_ENV)
    if raw is None:
        return FULL_MATRIX_FULL_ASSET_OVERLAY_DATASETS
    values = frozenset(item.strip() for item in raw.split(",") if item.strip())
    unknown = values - FULL_MATRIX_OVERLAY_DATASETS
    if unknown:
        raise ValueError(
            f"{FULL_MATRIX_FULL_ASSET_OVERLAY_DATASETS_ENV} contains unsupported datasets: "
            f"{sorted(unknown)}"
        )
    return values


def _asset_dataset(relative: Path) -> str | None:
    parts = relative.parts
    if len(parts) >= 2 and parts[0] in {"datasets", "means"}:
        return parts[1]
    if len(parts) >= 2 and parts[0] == "reference-models":
        return parts[1].split("--", 1)[0]
    return None


def resolve_full_matrix_runtime_path(value: str | os.PathLike[str]) -> Path:
    """Map selected full-matrix paths to an execution-only local overlay.

    The logical path remains in immutable configs and queue argv.  This
    function is called only at actual filesystem I/O sites, so enabling an
    overlay cannot change task IDs or scheduler identity.
    """

    source = _absolute_path(value)
    datasets = _overlay_datasets()
    full_asset_datasets = _full_asset_overlay_datasets()
    logical_asset = os.environ.get(FULL_MATRIX_LOGICAL_ASSET_ROOT_ENV)
    overlay_asset = os.environ.get(FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV)
    if logical_asset and overlay_asset:
        logical_root = _absolute_path(logical_asset)
        try:
            relative = source.relative_to(logical_root)
        except ValueError:
            pass
        else:
            dataset = _asset_dataset(relative)
            if dataset in datasets and (
                relative.parts[0] == "datasets" or dataset in full_asset_datasets
            ):
                return _absolute_path(overlay_asset) / relative

    logical_cache = os.environ.get(FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV)
    overlay_cache = os.environ.get(FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV)
    if logical_cache and overlay_cache:
        logical_root = _absolute_path(logical_cache)
        try:
            relative = source.relative_to(logical_root)
        except ValueError:
            pass
        else:
            if relative.parts and relative.parts[0] in datasets:
                return _absolute_path(overlay_cache) / relative
    return source


def resolve_pilot_definition_dir(
    value: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a pilot directory across planner and worker working directories.

    Formal argv may contain a relative path captured by the planner.  A worker
    can execute that argv from a different execution root, so existing paths
    are checked before the source tree's canonical pilot directory is used.
    Absolute paths remain authoritative, including when they do not exist.
    """

    raw = os.fspath(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates: list[Path] = []

    def add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    add(Path.cwd() / path)
    if project_root is not None:
        root = Path(project_root).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        else:
            root = root.resolve()
        add(root / path)
        add(root / "code" / path)

    package_code_root = _package_code_root()
    add(package_code_root / path)
    add(package_code_root.parent / path)
    add(package_code_root.parent / "code" / path)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    if path in _CANONICAL_PILOT_PATHS:
        return (package_code_root / "configs" / "pilots").resolve()
    return candidates[0]
