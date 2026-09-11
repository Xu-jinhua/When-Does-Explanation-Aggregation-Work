"""Read-only checks for runtime-only experiment dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from xai_ensemble.phase1.relprop import (
    RelPropProviderError,
    default_relprop_repository,
    relprop_required,
    validate_relprop_repository,
)


def relprop_runtime_readiness(
    method_architectures: Iterable[tuple[str, str]],
) -> Mapping[str, Any]:
    required_methods = sorted(
        {
            method
            for method, architecture in method_architectures
            if relprop_required(method, architecture)
        }
    )
    if not required_methods:
        return {
            "ready": True,
            "required": False,
            "methods": [],
            "repository": None,
            "revision": None,
            "source_digest": None,
        }
    repository = default_relprop_repository()
    try:
        provenance = validate_relprop_repository(repository)
    except RelPropProviderError as error:
        return {
            "ready": False,
            "required": True,
            "methods": required_methods,
            "repository": str(repository),
            "revision": None,
            "source_digest": None,
            "error": str(error),
        }
    return {
        "ready": True,
        "required": True,
        "methods": required_methods,
        "repository": provenance["repository"],
        "revision": provenance["revision"],
        "source_digest": provenance["source_digest"],
    }


def require_relprop_runtime(
    method_architectures: Iterable[tuple[str, str]],
) -> Mapping[str, Any]:
    report = relprop_runtime_readiness(method_architectures)
    if not report["ready"]:
        raise RelPropProviderError(
            f"RelProp runtime dependency is not ready: {report.get('error', report)}"
        )
    return report


__all__ = ["relprop_runtime_readiness", "require_relprop_runtime"]
