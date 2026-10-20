from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json


@dataclass
class PilotResult:
    name: str
    status: str
    selected: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    config_digest: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def digest(self) -> str:
        value = asdict(self)
        value.pop("created_at", None)
        return object_sha256(value)

    def require_passed(self) -> None:
        if not self.passed:
            raise RuntimeError(f"Pilot {self.name!r} has not passed: {self.failures}")

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, verify_digest: bool = True) -> PilotResult:
        result = cls(
            name=str(value["name"]),
            status=str(value["status"]),
            selected=dict(value.get("selected", {})),
            measurements=dict(value.get("measurements", {})),
            failures=[str(item) for item in value.get("failures", [])],
            config_digest=str(value.get("config_digest", "")),
            created_at=str(value.get("created_at", "")),
        )
        expected = value.get("digest")
        if verify_digest and expected is not None and str(expected) != result.digest:
            raise ValueError(
                f"Pilot result digest mismatch for {result.name!r}: "
                f"stored={expected} computed={result.digest}"
            )
        return result


def save_pilot_result(path: str | Path, result: PilotResult) -> Path:
    value = asdict(result)
    value["digest"] = result.digest
    return atomic_write_json(path, value)
