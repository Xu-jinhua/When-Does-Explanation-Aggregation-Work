"""Pinned Chefer model providers for transformer attribution and original LRP.

The upstream implementation changes the transformer layers themselves so a
normal timm model cannot be made RelProp-capable by attaching an explainer.
This module therefore reconstructs the matching Chefer architecture and
strictly loads the exact Phase-0 inference checkpoint.  No missing or extra
parameter is accepted and no attribution approximation is substituted.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import file_sha256, object_sha256

RELPROP_REPOSITORY_URL = "https://github.com/hila-chefer/Transformer-Explainability.git"
RELPROP_REVISION = "c3e578f76b954e8528afeaaee26de3f07e3fe559"
RELPROP_ROOT_ENV = "XAI_TRANSFORMER_EXPLAINABILITY_ROOT"
RELPROP_SNAPSHOT_MANIFEST = "UPSTREAM_PROVENANCE.json"
RELPROP_FACTORY_REFERENCE = "xai_ensemble.phase1.relprop:create_relprop_model"
ORIGINAL_LRP_FACTORY_REFERENCE = "xai_ensemble.phase1.relprop:create_original_lrp_model"
RELPROP_FACTORY_REFERENCES = frozenset({RELPROP_FACTORY_REFERENCE, ORIGINAL_LRP_FACTORY_REFERENCE})
RELPROP_EQUIVALENCE_RTOL = 1e-4
RELPROP_EQUIVALENCE_ATOL = 2e-5

# Both official Chefer implementations are pinned. ``ViT_LRP`` plus
# ``layers_ours`` implements Transformer Attribution, whereas the upstream
# evaluation uses ``ViT_orig_LRP`` plus ``layers_lrp`` for its Full- and
# Partial-LRP baselines. Hashing both transitive source sets prevents these
# scientifically distinct providers from being silently interchanged.
RELPROP_REQUIRED_FILES: Mapping[str, str] = {
    "LICENSE": "b00ff34230c492d8d0103a90bc8da180d2f5ce8032541f758402058cb82c18a1",
    "baselines/ViT/ViT_LRP.py": (
        "468e8af6757fa8175d6a6146b354aef52d3e563c1d6797eb377ccc1327a922d2"
    ),
    "baselines/ViT/ViT_orig_LRP.py": (
        "04a8beacfda0af716058e96c1ec36690b5954686e37c72210d42b731ae400f2e"
    ),
    "baselines/ViT/helpers.py": (
        "4db366adfa8772e203074f5f40192770a4ed97befad76db54bd9f4aedcc73d2a"
    ),
    "baselines/ViT/layer_helpers.py": (
        "94ff5295708e189838c957b2433f974e140d46fd67f8a1100ad5a2e50cf09401"
    ),
    "baselines/ViT/weight_init.py": (
        "7123469c03fe3470a6495b24a3a46697a3dae9fd84f1c791c7d7d0ea0ac3454e"
    ),
    "modules/layers_ours.py": ("4ef7d98d75c9b79e1a4ed0e96b92a8243ea01d478653b3fe96df1ca61a4841dc"),
    "modules/layers_lrp.py": ("d1b38f609c9ed989c87d9524df6b48166a5730e8a0bffc648f05bed04117c4da"),
}
RELPROP_SOURCE_DIGEST = object_sha256(
    {"revision": RELPROP_REVISION, "files": dict(RELPROP_REQUIRED_FILES)}
)

_PROVIDER_CONSTRUCTORS: Mapping[str, Mapping[str, str]] = {
    "transformer_attribution": {
        "vit_base_patch16_224": "vit_base_patch16_224",
        "deit_base_patch16_224": "deit_base_patch16_224",
    },
    # The pinned upstream ViT_orig_LRP.py exposes ViT-B/16, but no DeiT or
    # Swin constructor. Treat that as an explicit capability boundary rather
    # than allowing an import-time AttributeError after expensive training.
    "original_lrp": {
        "vit_base_patch16_224": "vit_base_patch16_224",
    },
}
_IMPORT_LOCK = threading.Lock()


class RelPropProviderError(RuntimeError):
    """The pinned RelProp provider is absent or incompatible."""


def relprop_required(method: str, architecture: str) -> bool:
    """Return whether this method/architecture needs the Chefer model.

    CNN ``LRP`` remains Captum's layer-wise relevance propagation. The legacy
    ViT name ``LRP`` is accepted only for older core locks; the simplified
    paper roster uses the scientifically explicit Chefer name.
    """

    return architecture == "vit" and method in {
        "LRP",
        "CheferTransformerAttribution",
        "PartialLRP",
        "FullLRP",
    }


def relprop_implementation(method: str, architecture: str) -> str | None:
    """Return the exact pinned upstream implementation required by a method."""

    if not relprop_required(method, architecture):
        return None
    if method in {"PartialLRP", "FullLRP"}:
        return "original_lrp"
    return "transformer_attribution"


def relprop_support_error(
    method: str,
    *,
    model_key: str,
    architecture: str,
) -> str | None:
    """Return a static provider limitation, if the exact method is unavailable."""

    implementation = relprop_implementation(method, architecture)
    if implementation is None:
        return None
    supported = _PROVIDER_CONSTRUCTORS[implementation]
    if model_key in supported:
        return None
    return (
        f"{method} requires RelProp provider {implementation!r}, which has no "
        f"scientifically equivalent constructor for {model_key!r}; supported models: "
        f"{', '.join(sorted(supported))}"
    )


def relprop_factory_metadata(method: str, architecture: str) -> dict[str, str]:
    """Return immutable job fields for a RelProp-backed source model."""

    if not relprop_required(method, architecture):
        return {}
    implementation = relprop_implementation(method, architecture)
    return {
        "factory": (
            ORIGINAL_LRP_FACTORY_REFERENCE
            if implementation == "original_lrp"
            else RELPROP_FACTORY_REFERENCE
        ),
        "factory_revision": RELPROP_REVISION,
        "factory_source_digest": RELPROP_SOURCE_DIGEST,
    }


def relprop_attribution_provider(method: str, architecture: str) -> dict[str, str]:
    """Return the immutable provider identity recorded by attribution artifacts."""

    factory = relprop_factory_metadata(method, architecture)
    if not factory:
        return {}
    return {
        "kind": "chefer_relprop",
        "implementation": str(relprop_implementation(method, architecture)),
        **factory,
    }


def default_relprop_repository() -> Path:
    configured = os.environ.get(RELPROP_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    project = os.environ.get("XAI_PROJECT_ROOT")
    if project:
        return (Path(project).expanduser() / "code/vendor/Transformer-Explainability").resolve()
    # Source checkout layout: code/src/xai_ensemble/phase1/relprop.py.
    return (Path(__file__).resolve().parents[3] / "vendor/Transformer-Explainability").resolve()


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RelPropProviderError(
            f"Cannot verify the RelProp Git revision under {root}"
        ) from error
    return result.stdout.strip().lower()


def _snapshot_revision(
    root: Path,
    *,
    expected_revision: str,
    required_files: Mapping[str, str],
) -> str:
    path = root / RELPROP_SNAPSHOT_MANIFEST
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RelPropProviderError(
            f"Cannot read the vendored RelProp provenance: {path}"
        ) from error
    expected_files = dict(required_files)
    expected = {
        "schema_version": 1,
        "repository_url": RELPROP_REPOSITORY_URL,
        "revision": expected_revision.lower(),
        "required_files": expected_files,
        "source_digest": object_sha256(
            {"revision": expected_revision.lower(), "files": expected_files}
        ),
    }
    if value != expected:
        raise RelPropProviderError(
            "Vendored RelProp provenance does not match the pinned provider identity"
        )
    return expected_revision.lower()


def validate_relprop_repository(
    root: str | os.PathLike[str],
    *,
    expected_revision: str = RELPROP_REVISION,
    required_files: Mapping[str, str] = RELPROP_REQUIRED_FILES,
) -> dict[str, Any]:
    """Verify Git revision and every imported upstream source file."""

    repository = Path(root).expanduser().resolve()
    if not repository.is_dir():
        raise RelPropProviderError(
            f"RelProp repository is absent: {repository}. Run the pinned setup script first."
        )
    snapshot_manifest = repository / RELPROP_SNAPSHOT_MANIFEST
    observed_revision = (
        _snapshot_revision(
            repository,
            expected_revision=expected_revision,
            required_files=required_files,
        )
        if snapshot_manifest.is_file()
        else _git_head(repository)
    )
    if observed_revision != expected_revision.lower():
        raise RelPropProviderError(
            "RelProp repository revision mismatch: "
            f"observed={observed_revision}, expected={expected_revision.lower()}"
        )
    observed_files: dict[str, str] = {}
    for relative, expected_sha256 in required_files.items():
        path = repository / relative
        if not path.is_file():
            raise RelPropProviderError(f"Pinned RelProp source is missing: {relative}")
        observed = file_sha256(path)
        observed_files[relative] = observed
        if observed != expected_sha256:
            raise RelPropProviderError(
                f"Pinned RelProp source checksum mismatch for {relative}: "
                f"observed={observed}, expected={expected_sha256}"
            )
    return {
        "repository": str(repository),
        "url": RELPROP_REPOSITORY_URL,
        "revision": observed_revision,
        "files": observed_files,
        "source_digest": object_sha256({"revision": observed_revision, "files": observed_files}),
    }


def _load_checkpoint_state(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    state: Any = payload
    if isinstance(payload, Mapping):
        for key in ("model_state", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping) or not state:
        raise RelPropProviderError(f"Checkpoint has no non-empty model state: {path}")
    normalized = {str(key): value for key, value in state.items()}
    if normalized and all(key.startswith("module.") for key in normalized):
        normalized = {key[7:]: value for key, value in normalized.items()}
    return normalized


def strict_load_relprop_state(model: Any, state: Mapping[str, Any]) -> None:
    """Fail before loading if names or shapes differ in any way."""

    expected = model.state_dict()
    expected_keys = set(expected)
    observed_keys = set(state)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    shape_mismatches = sorted(
        key
        for key in expected_keys & observed_keys
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append(f"missing={missing[:12]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:12]}")
        if shape_mismatches:
            details.append(f"shape_mismatches={shape_mismatches[:12]}")
        raise RelPropProviderError(
            "Phase-0 checkpoint is not structurally identical to the pinned RelProp model: "
            + "; ".join(details)
        )
    try:
        model.load_state_dict(dict(state), strict=True)
    except Exception as error:
        raise RelPropProviderError("Strict RelProp checkpoint loading failed") from error


def _pretrained_cfg(model_key: str) -> dict[str, Any]:
    try:
        import timm
    except ImportError as error:  # pragma: no cover - formal GPU dependency
        raise RelPropProviderError("The RelProp provider requires timm") from error
    getter = getattr(timm, "get_pretrained_cfg", None)
    if getter is None:
        getter = getattr(importlib.import_module("timm.models"), "get_pretrained_cfg", None)
    if getter is None:
        raise RelPropProviderError("Installed timm cannot expose the preprocessing contract")
    config = getter(model_key)
    if config is None:
        raise RelPropProviderError(f"timm has no preprocessing configuration for {model_key}")
    value = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return dict(value)


def _import_constructor(
    repository: Path,
    constructor_name: str,
    *,
    module_name: str,
) -> Any:
    with _IMPORT_LOCK:
        root_text = str(repository)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__).resolve()
        try:
            module_path.relative_to(repository)
        except ValueError as error:
            raise RelPropProviderError(
                "A different Transformer-Explainability checkout is already imported: "
                f"{module_path}"
            ) from error
        constructor = getattr(module, constructor_name, None)
        if not callable(constructor):
            raise RelPropProviderError(
                f"Pinned RelProp source has no constructor {constructor_name!r}"
            )
        return constructor


class RelPropModelAdapter:
    """Delegate a Chefer model while making prediction-only forwards safe.

    The upstream attention layer always registers gradient hooks.  PyTorch
    rightfully rejects that inside ``torch.no_grad()``.  Prediction calls are
    therefore evaluated in a short local gradient context and detached;
    explanation calls retain the caller's graph unchanged.
    """

    def __init__(self, model: Any, *, provenance: Mapping[str, Any]) -> None:
        self._model = model
        self.relprop_provenance = dict(provenance)
        self.pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
        self.default_cfg = dict(getattr(model, "default_cfg", {}) or {})

    def __call__(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        if torch.is_grad_enabled():
            return self._model(inputs, *args, **kwargs)
        with torch.enable_grad():
            proxy = inputs.detach().requires_grad_(True)
            output = self._model(proxy, *args, **kwargs)
        return output.detach()

    def relprop(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.relprop(*args, **kwargs)

    def eval(self) -> RelPropModelAdapter:
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> RelPropModelAdapter:
        self._model.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> RelPropModelAdapter:
        self._model.to(*args, **kwargs)
        return self

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self._model.zero_grad(*args, **kwargs)

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.parameters(*args, **kwargs)

    def modules(self) -> Any:
        return self._model.modules()

    def named_modules(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.named_modules(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.state_dict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "_model":
            raise AttributeError(name)
        return getattr(self._model, name)


def create_relprop_model(
    *,
    model_key: str,
    num_classes: int,
    init_mode: str,
    checkpoint_path: str | None,
    strict_checkpoint: bool = True,
    repository_root: str | os.PathLike[str] | None = None,
) -> RelPropModelAdapter:
    """Build the pinned Chefer Transformer Attribution model."""

    return _create_relprop_model(
        model_key=model_key,
        num_classes=num_classes,
        init_mode=init_mode,
        checkpoint_path=checkpoint_path,
        strict_checkpoint=strict_checkpoint,
        repository_root=repository_root,
        module_name="baselines.ViT.ViT_LRP",
        provider_name="chefer-transformer-attribution",
        implementation="transformer_attribution",
    )


def create_original_lrp_model(
    *,
    model_key: str,
    num_classes: int,
    init_mode: str,
    checkpoint_path: str | None,
    strict_checkpoint: bool = True,
    repository_root: str | os.PathLike[str] | None = None,
) -> RelPropModelAdapter:
    """Build the original-LRP model used by Chefer's official baselines."""

    return _create_relprop_model(
        model_key=model_key,
        num_classes=num_classes,
        init_mode=init_mode,
        checkpoint_path=checkpoint_path,
        strict_checkpoint=strict_checkpoint,
        repository_root=repository_root,
        module_name="baselines.ViT.ViT_orig_LRP",
        provider_name="chefer-original-lrp",
        implementation="original_lrp",
    )


def create_relprop_model_for_method(
    method: str,
    *,
    model_key: str,
    num_classes: int,
    init_mode: str,
    checkpoint_path: str | None,
    strict_checkpoint: bool = True,
    repository_root: str | os.PathLike[str] | None = None,
) -> RelPropModelAdapter:
    """Build the exact official provider associated with ``method``."""

    implementation = relprop_implementation(method, "vit")
    if implementation is None:
        raise RelPropProviderError(f"{method!r} is not a ViT RelProp method")
    factory = (
        create_original_lrp_model if implementation == "original_lrp" else create_relprop_model
    )
    return factory(
        model_key=model_key,
        num_classes=num_classes,
        init_mode=init_mode,
        checkpoint_path=checkpoint_path,
        strict_checkpoint=strict_checkpoint,
        repository_root=repository_root,
    )


def _create_relprop_model(
    *,
    model_key: str,
    num_classes: int,
    init_mode: str,
    checkpoint_path: str | None,
    strict_checkpoint: bool,
    repository_root: str | os.PathLike[str] | None,
    module_name: str,
    provider_name: str,
    implementation: str,
) -> RelPropModelAdapter:

    try:
        constructor_name = _PROVIDER_CONSTRUCTORS[implementation][model_key]
    except KeyError as error:
        supported = ", ".join(sorted(_PROVIDER_CONSTRUCTORS.get(implementation, {})))
        raise RelPropProviderError(
            f"RelProp provider {implementation!r} is unsupported for {model_key!r}; "
            f"supported models: {supported}"
        ) from error
    if init_mode != "checkpoint" or checkpoint_path is None:
        raise RelPropProviderError("RelProp formal models require a Phase-0 checkpoint")
    if not strict_checkpoint:
        raise RelPropProviderError("RelProp refuses non-strict checkpoint loading")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    repository = (
        default_relprop_repository()
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    provenance = validate_relprop_repository(repository)
    constructor = _import_constructor(
        repository,
        constructor_name,
        module_name=module_name,
    )
    model = constructor(pretrained=False, num_classes=int(num_classes))
    model.pretrained_cfg = _pretrained_cfg(model_key)
    strict_load_relprop_state(model, _load_checkpoint_state(checkpoint))
    model.model_key = model_key
    model.model_family = "vit"
    model.model_provider = provider_name
    model.initialization = init_mode
    model.relprop_revision = RELPROP_REVISION
    model.relprop_source_digest = RELPROP_SOURCE_DIGEST
    return RelPropModelAdapter(model, provenance=provenance)


def verify_relprop_equivalence(
    reference_model: Any,
    relprop_model: Any,
    inputs: Any,
    *,
    rtol: float = RELPROP_EQUIVALENCE_RTOL,
    atol: float = RELPROP_EQUIVALENCE_ATOL,
) -> dict[str, Any]:
    """GPU-pilot gate proving both architectures implement the same classifier."""

    import torch

    with torch.no_grad():
        reference = reference_model(inputs).detach()
        candidate = relprop_model(inputs).detach()
    if reference.shape != candidate.shape:
        raise RelPropProviderError(
            f"RelProp logits shape {tuple(candidate.shape)} differs from {tuple(reference.shape)}"
        )
    if not bool(torch.isfinite(reference).all()):
        raise RelPropProviderError("Reference model produced non-finite logits")
    if not bool(torch.isfinite(candidate).all()):
        raise RelPropProviderError("RelProp model produced non-finite logits")
    maximum = float((reference - candidate).abs().max().item())
    predictions_equal = bool(torch.equal(reference.argmax(dim=1), candidate.argmax(dim=1)))
    values_close = bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol))
    if not values_close or not predictions_equal:
        raise RelPropProviderError(
            "Strictly loaded RelProp model is not functionally equivalent to the Phase-0 model: "
            f"max_abs={maximum:.8g}, values_close={values_close}, "
            f"predictions_equal={predictions_equal}"
        )
    return {
        "max_abs_logit_difference": maximum,
        "values_close": values_close,
        "predictions_equal": predictions_equal,
        "rtol": rtol,
        "atol": atol,
        "revision": RELPROP_REVISION,
        "source_digest": RELPROP_SOURCE_DIGEST,
    }


__all__ = [
    "RELPROP_EQUIVALENCE_ATOL",
    "RELPROP_EQUIVALENCE_RTOL",
    "ORIGINAL_LRP_FACTORY_REFERENCE",
    "RELPROP_FACTORY_REFERENCE",
    "RELPROP_FACTORY_REFERENCES",
    "RELPROP_REPOSITORY_URL",
    "RELPROP_REQUIRED_FILES",
    "RELPROP_REVISION",
    "RELPROP_ROOT_ENV",
    "RELPROP_SNAPSHOT_MANIFEST",
    "RELPROP_SOURCE_DIGEST",
    "RelPropModelAdapter",
    "RelPropProviderError",
    "create_relprop_model",
    "create_original_lrp_model",
    "create_relprop_model_for_method",
    "default_relprop_repository",
    "relprop_factory_metadata",
    "relprop_implementation",
    "relprop_support_error",
    "relprop_attribution_provider",
    "relprop_required",
    "strict_load_relprop_state",
    "validate_relprop_repository",
    "verify_relprop_equivalence",
]
