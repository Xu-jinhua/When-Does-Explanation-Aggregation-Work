from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

import xai_ensemble.simple.relprop_equivalence as certificates
from xai_ensemble.simple.config import load_experiment

CODE_ROOT = Path(__file__).resolve().parents[1]


class _EquivalentModel:
    def __call__(self, inputs):
        score = inputs.flatten(1).mean(dim=1)
        return torch.stack((score, score + 1.0), dim=1)


def test_fixed_probe_certificate_is_concurrent_and_provider_scoped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = load_experiment(CODE_ROOT / "configs/simple/example.yaml")
    experiment = replace(
        original,
        runtime=replace(original.runtime, profile_directory=tmp_path / "profiles"),
    )
    model = next(item for item in experiment.models if item.architecture == "vit")
    preprocessing = {
        "input_size": 2,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "interpolation": "bicubic",
    }
    reference = _EquivalentModel()
    candidate = _EquivalentModel()
    calls = 0
    calls_lock = threading.Lock()
    original_verify = certificates.verify_relprop_equivalence

    def counted_verify(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(certificates, "verify_relprop_equivalence", counted_verify)

    def ensure(method: str):
        return certificates.ensure_relprop_equivalence_certificate(
            experiment,
            model,
            method=method,
            reference_model=reference,
            relprop_model=candidate,
            preprocessing=preprocessing,
            checkpoint_sha256="a" * 64,
            device="cpu",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        partial = executor.submit(ensure, "PartialLRP")
        full = executor.submit(ensure, "FullLRP")
        original_lrp = (partial.result(), full.result())

    assert calls == 1
    assert original_lrp[0] == original_lrp[1]
    assert original_lrp[0]["identity"]["provider"]["implementation"] == "original_lrp"
    assert original_lrp[0]["identity"]["comparison"]["atol"] == 2e-5
    assert original_lrp[0]["identity"]["probe"]["shape"] == [2, 3, 2, 2]

    transformer = ensure("CheferTransformerAttribution")
    assert calls == 2
    assert transformer["identity_digest"] != original_lrp[0]["identity_digest"]
    assert transformer["identity"]["provider"]["implementation"] == "transformer_attribution"
    assert len(tuple((tmp_path / "profiles/relprop-equivalence").glob("*.json"))) == 2
