"""Deterministic, reusable equivalence certificates for RelProp classifiers."""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.phase1.relprop import (
    RELPROP_EQUIVALENCE_ATOL,
    RELPROP_EQUIVALENCE_RTOL,
    relprop_attribution_provider,
    verify_relprop_equivalence,
)

from .config import ModelConfig, SimpleExperiment

CERTIFICATE_SCHEMA_VERSION = 1
PROBE_BATCH_SIZE = 2
PROBE_GENERATOR = "torch-cpu-generator-randn-v1"


def _fixed_probe(*, seed: int, input_size: int) -> tuple[Any, str]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    probe = torch.randn(
        (PROBE_BATCH_SIZE, 3, input_size, input_size),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    digest = hashlib.sha256(probe.contiguous().numpy().tobytes()).hexdigest()
    return probe, digest


def certificate_identity(
    experiment: SimpleExperiment,
    model: ModelConfig,
    *,
    method: str,
    checkpoint_sha256: str,
    preprocessing: Mapping[str, Any],
    probe_sha256: str,
) -> Mapping[str, Any]:
    provider = relprop_attribution_provider(method, model.architecture)
    if not provider:
        raise ValueError(f"{method}/{model.architecture} does not use a RelProp provider")
    if not model.strict_checkpoint:
        raise ValueError("RelProp equivalence requires strict checkpoint loading")
    input_size = int(preprocessing["input_size"])
    return {
        "schema": "simple-relprop-equivalence-certificate-v1",
        "model": {
            "id": model.model_id,
            "model_key": model.model_key,
            "num_classes": model.num_classes,
            "init_mode": model.init_mode,
            "checkpoint_sha256": checkpoint_sha256,
            "strict_checkpoint": True,
        },
        "provider": provider,
        "preprocessing_digest": object_sha256(preprocessing),
        "precision": "fp32",
        "probe": {
            "generator": PROBE_GENERATOR,
            "seed": experiment.runtime.seed,
            "shape": [PROBE_BATCH_SIZE, 3, input_size, input_size],
            "space": "model_input",
            "sha256": probe_sha256,
        },
        "comparison": {
            "rtol": RELPROP_EQUIVALENCE_RTOL,
            "atol": RELPROP_EQUIVALENCE_ATOL,
            "finite_reference_logits": True,
            "finite_relprop_logits": True,
            "exact_argmax": True,
        },
    }


def certificate_path(
    experiment: SimpleExperiment,
    model: ModelConfig,
    *,
    method: str,
    identity_digest: str,
) -> Path:
    provider = relprop_attribution_provider(method, model.architecture)
    if not provider:
        raise ValueError(f"{method}/{model.architecture} does not use a RelProp provider")
    return (
        experiment.runtime.profile_directory
        / "relprop-equivalence"
        / f"{model.model_id}--{provider['implementation']}--{identity_digest[:16]}.json"
    )


def _validated_certificate(
    path: Path,
    *,
    identity: Mapping[str, Any],
    identity_digest: str,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable RelProp equivalence certificate: {path}") from error
    if (
        value.get("schema_version") != CERTIFICATE_SCHEMA_VERSION
        or value.get("status") != "passed"
        or value.get("identity") != identity
        or value.get("identity_digest") != identity_digest
    ):
        raise ValueError(f"Stale or contradictory RelProp equivalence certificate: {path}")
    outcome = value.get("outcome")
    if (
        not isinstance(outcome, Mapping)
        or not outcome.get("values_close")
        or not outcome.get("predictions_equal")
    ):
        raise ValueError(f"Invalid RelProp equivalence outcome: {path}")
    certificate_digest = value.get("certificate_digest")
    payload = {key: item for key, item in value.items() if key != "certificate_digest"}
    if certificate_digest != object_sha256(payload):
        raise ValueError(f"RelProp equivalence certificate digest mismatch: {path}")
    return value


def ensure_relprop_equivalence_certificate(
    experiment: SimpleExperiment,
    model: ModelConfig,
    *,
    method: str,
    reference_model: Any,
    relprop_model: Any,
    preprocessing: Mapping[str, Any],
    checkpoint_sha256: str,
    device: Any,
) -> Mapping[str, Any]:
    """Create or reuse one fixed-probe certificate per model/provider identity."""

    import torch

    input_size = int(preprocessing["input_size"])
    probe, probe_sha256 = _fixed_probe(seed=experiment.runtime.seed, input_size=input_size)
    identity = certificate_identity(
        experiment,
        model,
        method=method,
        checkpoint_sha256=checkpoint_sha256,
        preprocessing=preprocessing,
        probe_sha256=probe_sha256,
    )
    identity_digest = object_sha256(identity)
    path = certificate_path(
        experiment,
        model,
        method=method,
        identity_digest=identity_digest,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing = _validated_certificate(
                path,
                identity=identity,
                identity_digest=identity_digest,
            )
            if existing is not None:
                return existing
            target_device = torch.device(device)
            outcome = verify_relprop_equivalence(
                reference_model,
                relprop_model,
                probe.to(target_device, dtype=torch.float32, non_blocking=True),
                rtol=RELPROP_EQUIVALENCE_RTOL,
                atol=RELPROP_EQUIVALENCE_ATOL,
            )
            value: dict[str, Any] = {
                "schema_version": CERTIFICATE_SCHEMA_VERSION,
                "status": "passed",
                "created_utc": datetime.now(UTC).isoformat(),
                "identity": identity,
                "identity_digest": identity_digest,
                "outcome": outcome,
            }
            value["certificate_digest"] = object_sha256(value)
            atomic_write_json(path, value)
            verified = _validated_certificate(
                path,
                identity=identity,
                identity_digest=identity_digest,
            )
            assert verified is not None
            return verified
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "PROBE_BATCH_SIZE",
    "PROBE_GENERATOR",
    "certificate_identity",
    "certificate_path",
    "ensure_relprop_equivalence_certificate",
]
