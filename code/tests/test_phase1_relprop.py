from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from xai_ensemble.core.hashing import file_sha256, object_sha256
from xai_ensemble.phase1._compat_support import ModelArtifactSpec
from xai_ensemble.phase1.compatibility import (
    CompatibilityMeasurement,
    _compatibility_gate_failures,
    _cuda_headroom_passed,
)
from xai_ensemble.phase1.explainers import build_explainer, candidate_roster
from xai_ensemble.phase1.relprop import (
    ORIGINAL_LRP_FACTORY_REFERENCE,
    RELPROP_EQUIVALENCE_ATOL,
    RELPROP_EQUIVALENCE_RTOL,
    RELPROP_FACTORY_REFERENCE,
    RELPROP_FACTORY_REFERENCES,
    RELPROP_REQUIRED_FILES,
    RELPROP_REVISION,
    RELPROP_ROOT_ENV,
    RELPROP_SNAPSHOT_MANIFEST,
    RELPROP_SOURCE_DIGEST,
    RelPropProviderError,
    create_relprop_model_for_method,
    default_relprop_repository,
    relprop_attribution_provider,
    relprop_factory_metadata,
    relprop_implementation,
    relprop_required,
    strict_load_relprop_state,
    validate_relprop_repository,
    verify_relprop_equivalence,
)
from xai_ensemble.phase1.transformer import RelPropExplainer
from xai_ensemble.simple.runtime_dependencies import (
    relprop_runtime_readiness,
    require_relprop_runtime,
)


class _Value:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


class _StateModel:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"block.weight": _Value(2, 3), "head.bias": _Value(4)}

    def load_state_dict(self, state, *, strict: bool):
        self.loaded = (state, strict)


class _RelPropModel:
    def relprop(self, *_args, **_kwargs):
        raise AssertionError("not executed")


class _FixedLogitModel:
    def __init__(self, logits: tuple[float, ...]) -> None:
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def __call__(self, inputs):
        return self.logits.to(inputs.device).expand(len(inputs), -1)


def test_pinned_relprop_identity_is_complete() -> None:
    assert RELPROP_REVISION == "c3e578f76b954e8528afeaaee26de3f07e3fe559"
    assert len(RELPROP_REQUIRED_FILES) >= 6
    assert RELPROP_SOURCE_DIGEST == object_sha256(
        {"revision": RELPROP_REVISION, "files": dict(RELPROP_REQUIRED_FILES)}
    )
    assert all(len(value) == 64 for value in RELPROP_REQUIRED_FILES.values())
    assert RELPROP_FACTORY_REFERENCES == {
        RELPROP_FACTORY_REFERENCE,
        ORIGINAL_LRP_FACTORY_REFERENCE,
    }
    assert "baselines/ViT/ViT_orig_LRP.py" in RELPROP_REQUIRED_FILES
    assert "modules/layers_lrp.py" in RELPROP_REQUIRED_FILES


def test_relprop_equivalence_uses_fp32_absolute_tolerance_near_zero() -> None:
    inputs = torch.zeros(2, 1)
    reference = _FixedLogitModel((0.01, 1.0))
    candidate = _FixedLogitModel((0.0100143, 1.0))

    result = verify_relprop_equivalence(reference, candidate, inputs)

    assert result["values_close"] is True
    assert result["predictions_equal"] is True
    assert result["rtol"] == RELPROP_EQUIVALENCE_RTOL
    assert result["atol"] == RELPROP_EQUIVALENCE_ATOL
    assert result["max_abs_logit_difference"] == pytest.approx(1.43e-5, rel=1e-3)

    with pytest.raises(RelPropProviderError, match="values_close=False"):
        verify_relprop_equivalence(reference, candidate, inputs, rtol=1e-4, atol=1e-5)

    changed_prediction = _FixedLogitModel((1.01, 1.0))
    with pytest.raises(RelPropProviderError, match="predictions_equal=False"):
        verify_relprop_equivalence(reference, changed_prediction, inputs)


def test_repository_validation_checks_revision_and_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = {
        "LICENSE": "license text\n",
        "baselines/ViT/ViT_LRP.py": "model source\n",
    }
    expected = {}
    for relative, content in required.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        expected[relative] = file_sha256(path)
    monkeypatch.setattr("xai_ensemble.phase1.relprop._git_head", lambda _root: "a" * 40)
    result = validate_relprop_repository(
        tmp_path,
        expected_revision="a" * 40,
        required_files=expected,
    )
    assert result["revision"] == "a" * 40
    assert result["source_digest"] == object_sha256({"revision": "a" * 40, "files": expected})

    (tmp_path / "baselines/ViT/ViT_LRP.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RelPropProviderError, match="checksum mismatch"):
        validate_relprop_repository(
            tmp_path,
            expected_revision="a" * 40,
            required_files=expected,
        )
    monkeypatch.setattr("xai_ensemble.phase1.relprop._git_head", lambda _root: "b" * 40)
    with pytest.raises(RelPropProviderError, match="revision mismatch"):
        validate_relprop_repository(
            tmp_path,
            expected_revision="a" * 40,
            required_files=expected,
        )


def test_tracked_relprop_snapshot_is_the_valid_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RELPROP_ROOT_ENV, raising=False)
    monkeypatch.delenv("XAI_PROJECT_ROOT", raising=False)

    root = default_relprop_repository()
    result = validate_relprop_repository(root)

    assert (root / RELPROP_SNAPSHOT_MANIFEST).is_file()
    assert result["revision"] == RELPROP_REVISION
    assert result["source_digest"] == RELPROP_SOURCE_DIGEST


def test_vendored_snapshot_validation_does_not_require_nested_git(tmp_path: Path) -> None:
    required = {"LICENSE": "license text\n"}
    expected = {}
    for relative, content in required.items():
        path = tmp_path / relative
        path.write_text(content, encoding="utf-8")
        expected[relative] = file_sha256(path)
    revision = "a" * 40
    provenance = {
        "schema_version": 1,
        "repository_url": ("https://github.com/hila-chefer/Transformer-Explainability.git"),
        "revision": revision,
        "required_files": expected,
        "source_digest": object_sha256({"revision": revision, "files": expected}),
    }
    (tmp_path / RELPROP_SNAPSHOT_MANIFEST).write_text(json.dumps(provenance), encoding="utf-8")

    result = validate_relprop_repository(
        tmp_path,
        expected_revision=revision,
        required_files=expected,
    )

    assert result["source_digest"] == provenance["source_digest"]
    provenance["revision"] = "b" * 40
    (tmp_path / RELPROP_SNAPSHOT_MANIFEST).write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(RelPropProviderError, match="provenance"):
        validate_relprop_repository(
            tmp_path,
            expected_revision=revision,
            required_files=expected,
        )


def test_relprop_runtime_readiness_fails_closed_only_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RELPROP_ROOT_ENV, str(tmp_path / "missing"))

    unused = relprop_runtime_readiness((("Saliency", "vit"), ("LRP", "cnn")))
    required = relprop_runtime_readiness((("FullLRP", "vit"),))

    assert unused["ready"] is True
    assert unused["required"] is False
    assert required["ready"] is False
    assert required["methods"] == ["FullLRP"]
    with pytest.raises(RelPropProviderError, match="runtime dependency is not ready"):
        require_relprop_runtime((("FullLRP", "vit"),))


def test_strict_state_loader_accepts_only_exact_names_and_shapes() -> None:
    model = _StateModel()
    state = {"block.weight": _Value(2, 3), "head.bias": _Value(4)}
    strict_load_relprop_state(model, state)
    assert model.loaded == (state, True)

    with pytest.raises(RelPropProviderError, match="shape_mismatches"):
        strict_load_relprop_state(
            _StateModel(),
            {"block.weight": _Value(3, 2), "head.bias": _Value(4)},
        )
    with pytest.raises(RelPropProviderError, match="missing=.*unexpected"):
        strict_load_relprop_state(
            _StateModel(),
            {"block.weight": _Value(2, 3), "other": _Value(4)},
        )


def test_relprop_factory_selection_and_architecture_rosters() -> None:
    assert relprop_required("CheferTransformerAttribution", "vit")
    assert relprop_required("LRP", "vit")
    assert relprop_required("PartialLRP", "vit")
    assert relprop_required("FullLRP", "vit")
    assert not relprop_required("LRP", "cnn")
    assert not relprop_required("Saliency", "vit")
    assert relprop_attribution_provider("CheferTransformerAttribution", "vit") == {
        "kind": "chefer_relprop",
        "implementation": "transformer_attribution",
        "factory": RELPROP_FACTORY_REFERENCE,
        "factory_revision": RELPROP_REVISION,
        "factory_source_digest": RELPROP_SOURCE_DIGEST,
    }
    assert relprop_attribution_provider("LRP", "cnn") == {}
    assert relprop_attribution_provider("FullLRP", "vit") == {
        "kind": "chefer_relprop",
        "implementation": "original_lrp",
        "factory": ORIGINAL_LRP_FACTORY_REFERENCE,
        "factory_revision": RELPROP_REVISION,
        "factory_source_digest": RELPROP_SOURCE_DIGEST,
    }
    assert relprop_implementation("PartialLRP", "vit") == "original_lrp"
    assert relprop_implementation("FullLRP", "vit") == "original_lrp"
    assert {
        "CheferTransformerAttribution",
        "PartialLRP",
        "FullLRP",
        "GradientAttentionRollout",
        "AttentionGradCAM",
    }.issubset(candidate_roster("vit"))
    assert {
        "LRP",
        "GuidedBackprop",
        "DeepLift",
        "DeepLiftShap",
        "Rollout",
        "AttnLast",
    }.isdisjoint(candidate_roster("vit"))
    assert "LRP" in candidate_roster("cnn")
    assert "PartialLRP" not in candidate_roster("cnn")
    assert "FullLRP" not in candidate_roster("cnn")

    explainer = build_explainer(_RelPropModel(), "CheferTransformerAttribution", architecture="vit")
    assert isinstance(explainer, RelPropExplainer)
    with pytest.raises(ValueError, match="not a candidate"):
        build_explainer(_RelPropModel(), "PartialLRP", architecture="cnn")


def test_relprop_factory_metadata_is_part_of_formal_model_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "inference.pt"
    checkpoint.write_bytes(b"checkpoint")
    values = {
        "model_key": "vit_base_patch16_224",
        "num_classes": 100,
        "init_mode": "checkpoint",
        "source_model_id": "reference-full",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "strict_checkpoint": True,
        **relprop_factory_metadata("FullLRP", "vit"),
    }
    model = ModelArtifactSpec.from_mapping(values, base=tmp_path)
    assert model.identity()["factory"] == ORIGINAL_LRP_FACTORY_REFERENCE
    assert model.identity()["factory_revision"] == RELPROP_REVISION
    assert model.identity()["factory_source_digest"] == RELPROP_SOURCE_DIGEST

    values["factory_revision"] = "0" * 40
    with pytest.raises(ValueError, match="exact upstream revision"):
        ModelArtifactSpec.from_mapping(values, base=tmp_path)


def test_cuda_headroom_gate_is_hard_and_inclusive_at_boundary() -> None:
    assert _cuda_headroom_passed(None, None)
    assert _cuda_headroom_passed(85, 100)
    assert not _cuda_headroom_passed(86, 100)
    with pytest.raises(ValueError, match="Invalid CUDA"):
        _cuda_headroom_passed(1, 0)


def test_relprop_method_dispatch_uses_the_scientifically_matching_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def transformer_factory(**_kwargs):
        calls.append("transformer_attribution")
        return "transformer-model"

    def original_factory(**_kwargs):
        calls.append("original_lrp")
        return "original-model"

    monkeypatch.setattr("xai_ensemble.phase1.relprop.create_relprop_model", transformer_factory)
    monkeypatch.setattr("xai_ensemble.phase1.relprop.create_original_lrp_model", original_factory)
    kwargs = {
        "model_key": "vit_base_patch16_224",
        "num_classes": 100,
        "init_mode": "checkpoint",
        "checkpoint_path": "/unused/checkpoint.pt",
    }

    assert (
        create_relprop_model_for_method("CheferTransformerAttribution", **kwargs)
        == "transformer-model"
    )
    assert create_relprop_model_for_method("PartialLRP", **kwargs) == "original-model"
    assert create_relprop_model_for_method("FullLRP", **kwargs) == "original-model"
    assert calls == ["transformer_attribution", "original_lrp", "original_lrp"]


def _measurement(method: str, passed: bool) -> CompatibilityMeasurement:
    return CompatibilityMeasurement(
        method=method,
        passed=passed,
        selected_batch_size=1 if passed else None,
        batch_candidates=(),
        deterministic_repeat=passed,
        target_sensitive=passed,
        target_sensitivity_required=True,
        attack_batch_size=None,
        attack_seconds_per_sample=None,
        attack_peak_cuda_bytes=None,
        output_shape=(1, 1, 14, 14) if passed else None,
        finite_fraction=1.0 if passed else None,
        error=None if passed else "incompatible",
    )


def test_compatibility_gate_filters_optional_candidates_but_requires_relprop_set() -> None:
    protocol = {
        "explainers": {
            "compatibility_gate": {
                "required_by_architecture": {
                    "vit": [
                        "CheferTransformerAttribution",
                        "PartialLRP",
                        "FullLRP",
                        "GradientAttentionRollout",
                        "AttentionGradCAM",
                    ]
                },
                "minimum_generic_compatible_by_architecture": {"vit": 1},
            }
        }
    }
    passing = (
        _measurement("Saliency", True),
        _measurement("CheferTransformerAttribution", True),
        _measurement("PartialLRP", True),
        _measurement("FullLRP", True),
        _measurement("GradientAttentionRollout", True),
        _measurement("AttentionGradCAM", True),
        _measurement("DeepLift", False),
    )
    assert _compatibility_gate_failures(protocol, architecture="vit", measurements=passing) == []

    failed_required = passing[:4] + (_measurement("GradientAttentionRollout", False),) + passing[5:]
    failures = _compatibility_gate_failures(
        protocol, architecture="vit", measurements=failed_required
    )
    assert failures == ["required methods failed compatibility: ['GradientAttentionRollout']"]
