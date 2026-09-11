"""Strict JSON parsing helpers for immutable experiment contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def loads_strict_json(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def load_strict_json(path: str | Path) -> Any:
    return loads_strict_json(Path(path).read_text(encoding="utf-8"))


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    allowed_optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - allowed_optional)
    if missing or unknown:
        raise ValueError(
            f"{context} schema mismatch: missing={missing}, unknown={unknown}"
        )


__all__ = ["load_strict_json", "loads_strict_json", "require_exact_keys"]
