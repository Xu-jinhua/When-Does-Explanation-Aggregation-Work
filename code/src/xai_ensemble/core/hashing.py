from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stream_sha256(handle: BinaryIO, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return stream_sha256(handle)


def stable_seed(*parts: Any, modulus: int = 2**63 - 1) -> int:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus
