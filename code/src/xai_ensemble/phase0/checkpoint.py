"""Crash-safe, self-describing training checkpoints."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.core.io import atomic_write_json

CHECKPOINT_SCHEMA_VERSION = 1
RESUME_IDENTITY_FIELDS = (
    "run_id",
    "task_id",
    "role",
    "source_id",
    "dataset_id",
    "dataset_revision",
    "dataset_manifest_fingerprint",
    "partition_kind",
    "partition_digest",
    "model_key",
    "model_provider",
    "initialization",
    "num_classes",
    "training_config_digest",
    "protocol_digest",
)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    run_id: str
    task_id: str
    role: str
    source_id: str
    dataset_id: str
    dataset_revision: str
    dataset_spec_fingerprint: str
    dataset_manifest_fingerprint: str
    partition_kind: str
    partition_digest: str
    model_key: str
    model_provider: str
    initialization: str
    num_classes: int
    training_config_digest: str
    protocol_digest: str
    seed: int
    completed_epochs: int = 0
    global_step: int = 0
    best_validation_top1: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema {self.schema_version}")
        for field_name in (
            "run_id",
            "task_id",
            "role",
            "source_id",
            "dataset_id",
            "dataset_revision",
            "dataset_manifest_fingerprint",
            "partition_kind",
            "partition_digest",
            "model_key",
            "model_provider",
            "initialization",
            "training_config_digest",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"Checkpoint metadata field {field_name} cannot be empty")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if self.completed_epochs < 0 or self.global_step < 0:
            raise ValueError("Training progress cannot be negative")

    @property
    def identity_digest(self) -> str:
        values = asdict(self)
        return object_sha256({key: values[key] for key in RESUME_IDENTITY_FIELDS})

    def with_progress(
        self,
        *,
        completed_epochs: int,
        global_step: int,
        best_validation_top1: float | None,
        metrics: Mapping[str, float],
    ) -> CheckpointMetadata:
        return replace(
            self,
            completed_epochs=completed_epochs,
            global_step=global_step,
            best_validation_top1=best_validation_top1,
            metrics=dict(metrics),
            created_utc=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CheckpointMetadata:
        return cls(**dict(values))


def validate_resume_metadata(
    expected: CheckpointMetadata,
    observed: CheckpointMetadata,
) -> None:
    mismatches = []
    for name in RESUME_IDENTITY_FIELDS:
        expected_value = getattr(expected, name)
        observed_value = getattr(observed, name)
        if expected_value != observed_value:
            mismatches.append(f"{name}: expected={expected_value!r}, found={observed_value!r}")
    if mismatches:
        raise ValueError("Refusing incompatible checkpoint resume:\n" + "\n".join(mismatches))


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - NumPy is a base dependency
        pass
    return state


def _restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:  # pragma: no cover
            pass


def save_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    scaler: Any | None,
    trainer_state: Mapping[str, Any],
    metadata: CheckpointMetadata,
) -> Path:
    """Save model and exact resume state, then publish an inspectable sidecar."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Saving a training checkpoint requires PyTorch") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata.to_dict(),
        "model_state": _unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "trainer_state": dict(trainer_state),
        "rng_state": _capture_rng_state(torch),
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    sidecar = {
        "checkpoint": destination.name,
        "identity_digest": metadata.identity_digest,
        "metadata": metadata.to_dict(),
    }
    atomic_write_json(destination.with_suffix(destination.suffix + ".json"), sidecar)
    return destination


def export_inference_checkpoint(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> Path:
    """Export weights and identity only from a resumable training checkpoint.

    Formal explanation jobs never need optimizer, scheduler, scaler, or RNG
    state.  Keeping those states in every local source-model copy would exceed
    the server disk budget, especially for AdamW-trained ViTs.  The complete
    checkpoint remains in private remote storage for exact training resume;
    this compact derivative is the only file retained locally for inference.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Exporting an inference checkpoint requires PyTorch") from exc

    source_path = Path(source)
    output_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(source_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported training checkpoint schema in {source_path}")
    metadata = CheckpointMetadata.from_mapping(payload["metadata"])
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError(f"Training checkpoint has no model state: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_kind": "inference_weights_only",
                "metadata": metadata.to_dict(),
                "model_state": model_state,
            },
            temporary_name,
        )
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    atomic_write_json(
        output_path.with_suffix(output_path.suffix + ".json"),
        {
            "checkpoint": output_path.name,
            "checkpoint_kind": "inference_weights_only",
            "identity_digest": metadata.identity_digest,
            "metadata": metadata.to_dict(),
            "source_checkpoint": source_path.name,
            "source_checkpoint_sha256": file_sha256(source_path),
            "checkpoint_sha256": file_sha256(output_path),
        },
    )
    return output_path


def save_inference_checkpoint(
    destination: str | os.PathLike[str],
    *,
    model: Any,
    metadata: CheckpointMetadata,
) -> Path:
    """Save a compact model that has no optimizer/training state.

    This is used by the locked direct ImageNet-1K subset-logit recipe.  The
    copied 100-class head is already the final classifier, so manufacturing a
    fake optimizer checkpoint or running a nominal epoch would change the
    scientific recipe.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Saving an inference checkpoint requires PyTorch") from exc

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_kind": "inference_weights_only",
                "metadata": metadata.to_dict(),
                "model_state": _unwrap_model(model).state_dict(),
            },
            temporary_name,
        )
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    atomic_write_json(
        output_path.with_suffix(output_path.suffix + ".json"),
        {
            "checkpoint": output_path.name,
            "checkpoint_kind": "inference_weights_only",
            "identity_digest": metadata.identity_digest,
            "metadata": metadata.to_dict(),
            "checkpoint_sha256": file_sha256(output_path),
        },
    )
    return output_path


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_metadata: CheckpointMetadata | None = None,
    restore_rng: bool = True,
    strict_model: bool = True,
) -> tuple[CheckpointMetadata, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Loading a training checkpoint requires PyTorch") from exc

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(source, map_location="cpu")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema in {source}")
    metadata = CheckpointMetadata.from_mapping(payload["metadata"])
    if expected_metadata is not None:
        validate_resume_metadata(expected_metadata, metadata)
    _unwrap_model(model).load_state_dict(payload["model_state"], strict=strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        _restore_rng_state(torch, payload.get("rng_state", {}))
    return metadata, dict(payload.get("trainer_state", {}))


class CheckpointManager:
    """Conventional paths for resumable latest/best/epoch checkpoints."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def inference_path(self) -> Path:
        return self.directory / "inference.pt"

    def epoch_path(self, completed_epochs: int) -> Path:
        return self.directory / f"epoch-{completed_epochs:04d}.pt"

    def resolve_resume(self, value: str | os.PathLike[str] | None) -> Path | None:
        if value is None:
            return None
        if str(value) == "auto":
            return self.latest_path if self.latest_path.is_file() else None
        path = Path(value)
        return path

    def read_metadata(self, path: str | os.PathLike[str] | None = None) -> CheckpointMetadata:
        checkpoint_path = self.latest_path if path is None else Path(path)
        sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        with sidecar_path.open("r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        return CheckpointMetadata.from_mapping(sidecar["metadata"])


def make_checkpoint_metadata(
    *,
    run_id: str,
    task_id: str,
    role: str,
    source_id: str,
    dataset_id: str,
    dataset_revision: str,
    dataset_spec_fingerprint: str,
    dataset_manifest_fingerprint: str,
    partition_kind: str,
    partition_digest: str,
    model_key: str,
    model_provider: str,
    initialization: str,
    num_classes: int,
    training_config: Mapping[str, Any],
    protocol_digest: str,
    seed: int,
    provenance: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    """Build the immutable identity block embedded in every checkpoint."""

    return CheckpointMetadata(
        run_id=run_id,
        task_id=task_id,
        role=role,
        source_id=source_id,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        dataset_spec_fingerprint=dataset_spec_fingerprint,
        dataset_manifest_fingerprint=dataset_manifest_fingerprint,
        partition_kind=partition_kind,
        partition_digest=partition_digest,
        model_key=model_key,
        model_provider=model_provider,
        initialization=initialization,
        num_classes=num_classes,
        training_config_digest=object_sha256(dict(training_config)),
        protocol_digest=protocol_digest,
        seed=seed,
        provenance=dict(provenance or {}),
    )
