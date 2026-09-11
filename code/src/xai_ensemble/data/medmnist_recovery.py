"""Verified local transport for the pinned MedMNIST+ 224 archives.

The scientific identity remains the registered ``HFDatasetSpec``.  This module
only provides an execution-time transport cache when the normal mounted source
cannot be read.  A source is accepted only after its fixed size, MD5, and NPZ
member contract pass; the marker is content-derived and idempotent.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, read_json
from xai_ensemble.data.specs import HFDatasetSpec

MEDMNIST_SOURCE_SCHEMA = "medmnist-local-source-v1"
MEDMNIST_SOURCE_MARKER_SUFFIX = ".verified.json"

# These are execution contracts from the immutable Zenodo record.  They are
# deliberately kept outside HFDatasetSpec so enabling a transport cache cannot
# change the protocol or matrix digest.
MEDMNIST_224_FILE_SIZES: Mapping[str, int] = {
    "bloodmnist_224.npz": 1_540_731_655,
    "breastmnist_224.npz": 30_903_564,
    "dermamnist_224.npz": 1_091_112_502,
    "octmnist_224.npz": 3_959_197_818,
    "organamnist_224.npz": 1_803_859_544,
    "organcmnist_224.npz": 760_231_860,
    "organsmnist_224.npz": 802_713_625,
    "pathmnist_224.npz": 12_629_854_322,
    "pneumoniamnist_224.npz": 214_384_716,
    "retinamnist_224.npz": 127_992_567,
    "tissuemnist_224.npz": 3_433_703_243,
}


def _source_options(spec: HFDatasetSpec) -> tuple[str, str, str, int]:
    if spec.provider != "medmnist":
        raise ValueError(f"{spec.key} is not a MedMNIST specification")
    options = spec.provider_options
    filename = str(options["filename"])
    url = str(options["source_url"])
    expected_md5 = str(options["md5"]).lower()
    try:
        expected_size = MEDMNIST_224_FILE_SIZES[filename]
    except KeyError as error:
        raise ValueError(f"No fixed local-source size contract for {filename!r}") from error
    return filename, url, expected_md5, expected_size


def has_verified_source_contract(spec: HFDatasetSpec) -> bool:
    """Return whether ``spec`` has the fixed archive contract for local recovery."""

    if spec.provider != "medmnist":
        return False
    options = spec.provider_options
    required = {"filename", "source_url", "md5", "zenodo_record"}
    return required.issubset(options) and str(options["filename"]) in MEDMNIST_224_FILE_SIZES


def source_marker_path(root: str | os.PathLike[str], spec: HFDatasetSpec) -> Path:
    filename, _url, _md5, _size = _source_options(spec)
    return Path(root).expanduser().resolve() / f"{filename}{MEDMNIST_SOURCE_MARKER_SUFFIX}"


def _md5(path: Path, *, chunk_bytes: int = 16 * 2**20) -> str:
    digest = hashlib.md5()  # noqa: S324 - MD5 is the pinned upstream contract.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_members(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        names = tuple(sorted(archive.namelist()))
    expected = tuple(
        sorted(
            f"{split}_{suffix}.npy"
            for split in ("train", "val", "test")
            for suffix in ("images", "labels")
        )
    )
    if names != expected:
        raise ValueError(f"{path}: NPZ member contract differs: {names}")
    return names


def _source_identity(
    spec: HFDatasetSpec,
    *,
    filename: str,
    url: str,
    expected_md5: str,
    expected_size: int,
    observed_md5: str,
    observed_size: int,
    members: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema": MEDMNIST_SOURCE_SCHEMA,
        "dataset_key": spec.key,
        "dataset_id": spec.dataset_id,
        "dataset_revision": spec.revision,
        "provider": spec.provider,
        "provider_version": str(spec.provider_options["version"]),
        "data_flag": str(spec.provider_options["data_flag"]),
        "size": int(spec.provider_options["size"]),
        "zenodo_record": str(spec.provider_options["zenodo_record"]),
        "filename": filename,
        "source_url": url,
        "expected_md5": expected_md5,
        "expected_size_bytes": expected_size,
        "observed_md5": observed_md5,
        "observed_size_bytes": observed_size,
        "npz_members": list(members),
    }


def validate_medmnist_source(
    spec: HFDatasetSpec,
    root: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Validate one local archive and its immutable source marker."""

    filename, url, expected_md5, expected_size = _source_options(spec)
    directory = Path(root).expanduser().resolve()
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ValueError(
            f"{path}: size differs from pinned source: {observed_size} != {expected_size}"
        )
    observed_md5 = _md5(path)
    if observed_md5 != expected_md5:
        raise ValueError(
            f"{path}: MD5 differs from pinned source: {observed_md5} != {expected_md5}"
        )
    members = _npz_members(path)
    identity = _source_identity(
        spec,
        filename=filename,
        url=url,
        expected_md5=expected_md5,
        expected_size=expected_size,
        observed_md5=observed_md5,
        observed_size=observed_size,
        members=members,
    )
    marker = source_marker_path(directory, spec)
    if marker.is_file():
        existing = read_json(marker)
        if not isinstance(existing, Mapping):
            raise ValueError(f"{marker}: source marker must be an object")
        if existing.get("identity") != identity:
            raise ValueError(f"{marker}: source marker identity differs")
        return existing
    payload = {
        "identity": identity,
        "identity_digest": object_sha256(identity),
        "path": str(path),
    }
    atomic_write_json(marker, payload)
    return payload


def require_verified_medmnist_source(
    spec: HFDatasetSpec,
    root: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Check a previously sealed marker without rehashing multi-GB bytes."""

    filename, url, expected_md5, expected_size = _source_options(spec)
    directory = Path(root).expanduser().resolve()
    path = directory / filename
    marker = source_marker_path(directory, spec)
    if not marker.is_file() or not path.is_file():
        raise FileNotFoundError(
            f"verified MedMNIST source is missing: archive={path}, marker={marker}"
        )
    value = read_json(marker)
    if not isinstance(value, Mapping):
        raise ValueError(f"{marker}: verified source marker must be an object")
    identity = value.get("identity")
    expected = {
        "schema": MEDMNIST_SOURCE_SCHEMA,
        "dataset_key": spec.key,
        "dataset_id": spec.dataset_id,
        "dataset_revision": spec.revision,
        "provider": spec.provider,
        "provider_version": str(spec.provider_options["version"]),
        "data_flag": str(spec.provider_options["data_flag"]),
        "size": int(spec.provider_options["size"]),
        "zenodo_record": str(spec.provider_options["zenodo_record"]),
        "filename": filename,
        "source_url": url,
        "expected_md5": expected_md5,
        "expected_size_bytes": expected_size,
        "observed_md5": expected_md5,
        "observed_size_bytes": expected_size,
        "npz_members": list(
            sorted(
                f"{split}_{suffix}.npy"
                for split in ("train", "val", "test")
                for suffix in ("images", "labels")
            )
        ),
    }
    if identity != expected or value.get("identity_digest") != object_sha256(expected):
        raise ValueError(f"{marker}: verified source identity is invalid")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{path}: verified source size changed")
    return value


def ensure_verified_medmnist_source(
    spec: HFDatasetSpec,
    root: str | os.PathLike[str],
    *,
    download: bool,
) -> Mapping[str, Any]:
    """Reuse a sealed source or fetch exactly the pinned archive.

    A valid marker is always reused without a multi-gigabyte rehash.  A
    missing source may be downloaded only when the caller explicitly allows
    it; an existing but contradictory source is never replaced.
    """

    try:
        return require_verified_medmnist_source(spec, root)
    except FileNotFoundError:
        if not download:
            raise
        return download_medmnist_source(spec, root)


@contextmanager
def _download_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        handle.close()


def download_medmnist_source(
    spec: HFDatasetSpec,
    root: str | os.PathLike[str],
    *,
    timeout_seconds: float = 120.0,
    chunk_bytes: int = 16 * 2**20,
) -> Mapping[str, Any]:
    """Download or resume one pinned archive, then validate it atomically."""

    filename, url, _expected_md5, expected_size = _source_options(spec)
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    with _download_lock(destination):
        try:
            return validate_medmnist_source(spec, directory)
        except FileNotFoundError:
            pass
        except ValueError:
            if destination.exists():
                raise
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"{partial}: partial file exceeds pinned size")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = Request(url, headers=headers)
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            status = int(getattr(response, "status", response.getcode()))
            if offset and status != 206:
                # The server ignored the range request; restart safely.
                offset = 0
                partial.unlink(missing_ok=True)
                request = Request(url)
                response.close()
                with urlopen(request, timeout=timeout_seconds) as fresh:  # noqa: S310
                    with partial.open("wb") as handle:
                        while chunk := fresh.read(chunk_bytes):
                            handle.write(chunk)
            else:
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(chunk_bytes):
                        handle.write(chunk)
        if partial.stat().st_size != expected_size:
            raise ValueError(
                f"{partial}: download size differs from pinned source: "
                f"{partial.stat().st_size} != {expected_size}"
            )
        os.replace(partial, destination)
        return validate_medmnist_source(spec, directory)


__all__ = [
    "MEDMNIST_224_FILE_SIZES",
    "MEDMNIST_SOURCE_SCHEMA",
    "download_medmnist_source",
    "ensure_verified_medmnist_source",
    "has_verified_source_contract",
    "require_verified_medmnist_source",
    "source_marker_path",
    "validate_medmnist_source",
]
