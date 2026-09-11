from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, read_json


@dataclass(frozen=True)
class LockedMethod:
    family: str
    instance_id: str
    params: dict[str, Any]
    architecture: str


@dataclass(frozen=True)
class MethodLock:
    dataset_id: str
    model_id: str
    methods: tuple[LockedMethod, ...]
    pilot_digest: str

    @property
    def digest(self) -> str:
        return object_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "pilot_digest": self.pilot_digest,
            "methods": [
                {
                    "family": method.family,
                    "instance_id": method.instance_id,
                    "params": method.params,
                    "architecture": method.architecture,
                }
                for method in self.methods
            ],
        }

    def validate_unique_families(self) -> None:
        families = [method.family for method in self.methods]
        duplicates = sorted({name for name in families if families.count(name) > 1})
        if duplicates:
            raise ValueError(f"Method lock contains duplicate families: {duplicates}")


def save_method_lock(path: str | Path, lock: MethodLock) -> Path:
    lock.validate_unique_families()
    value = lock.to_dict()
    value["digest"] = lock.digest
    return atomic_write_json(path, value)


def load_method_lock(path: str | Path) -> MethodLock:
    value = read_json(path)
    methods = tuple(LockedMethod(**item) for item in value["methods"])
    lock = MethodLock(
        dataset_id=value["dataset_id"],
        model_id=value["model_id"],
        methods=methods,
        pilot_digest=value["pilot_digest"],
    )
    lock.validate_unique_families()
    expected = value.get("digest")
    if expected and expected != lock.digest:
        raise ValueError("Method lock digest does not match its content")
    return lock
