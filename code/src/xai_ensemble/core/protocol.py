from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .hashing import object_sha256


class ProtocolError(ValueError):
    pass


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ProtocolError(f"Required environment variable is not set: {name}")

    return _ENV_PATTERN.sub(replace, value)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_dotted(root: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    current = root
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise ProtocolError(f"Cannot override {dotted}: {key} is not a mapping")
        current = child
    current[keys[-1]] = value


@dataclass(frozen=True)
class Protocol:
    source: Path
    data: dict[str, Any]
    digest: str

    @property
    def run_id(self) -> str:
        configured = self.data.get("run", {}).get("id")
        return str(configured or f"run-{self.digest[:12]}")

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ProtocolError(f"Protocol section {name!r} must be a mapping")
        return copy.deepcopy(value)

    def resolved_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def validate_protocol(data: Mapping[str, Any]) -> None:
    required = {"run", "datasets", "models", "explainers", "evaluation", "storage"}
    missing = sorted(required - set(data))
    if missing:
        raise ProtocolError(f"Missing protocol sections: {', '.join(missing)}")
    evaluation = data.get("evaluation", {})
    if evaluation.get("patch_size") != 16:
        raise ProtocolError("The locked primary protocol requires evaluation.patch_size=16")
    if evaluation.get("top_k") != 20:
        raise ProtocolError("The locked primary protocol requires evaluation.top_k=20")
    if evaluation.get("fill") != "dataset_mean":
        raise ProtocolError("The locked primary protocol requires evaluation.fill=dataset_mean")
    target = data.get("targets", {})
    if target.get("ind") != "reference_prediction":
        raise ProtocolError("IND must use targets.ind=reference_prediction")
    storage = data.get("storage", {})
    provider = storage.get("provider")
    if provider not in {"local", "cloudstorage"}:
        raise ProtocolError("storage.provider must be local or cloudstorage")
    if provider == "cloudstorage":
        root = storage.get("root")
        if not isinstance(root, str) or not root.strip():
            raise ProtocolError("CloudStorage requires a non-empty storage.root")
        revision = storage.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ProtocolError("CloudStorage requires a non-empty storage.revision namespace")
        if revision != revision.strip() or any(character.isspace() for character in revision):
            raise ProtocolError("CloudStorage storage.revision cannot contain whitespace")
        lock_root = storage.get("lock_root")
        if not isinstance(lock_root, str) or not lock_root.strip():
            raise ProtocolError("CloudStorage requires a non-empty storage.lock_root")
    noise = data.get("noise_selection", {})
    if noise and noise.get("patch_size") != evaluation.get("patch_size"):
        raise ProtocolError("noise_selection.patch_size must equal evaluation.patch_size")
    defaults = data.get("explainers", {}).get("locked_defaults", {})
    for method in ("FeatureAblation", "Occlusion"):
        options = defaults.get(method, {}) if isinstance(defaults, Mapping) else {}
        if options and options.get("patch_size") != evaluation.get("patch_size"):
            raise ProtocolError(
                f"{method} patch_size must equal evaluation.patch_size in the primary protocol"
            )
    class_balance = data.get("training", {}).get("class_balance", {})
    if class_balance:
        if not isinstance(class_balance, Mapping):
            raise ProtocolError("training.class_balance must be a dataset/model mapping")
        allowed_balance_modes = {"none", "weighted_loss", "balanced_sampler"}
        for dataset, models in class_balance.items():
            if not isinstance(models, Mapping):
                raise ProtocolError(
                    f"training.class_balance.{dataset} must be a model mapping"
                )
            for model, mode in models.items():
                if mode not in allowed_balance_modes:
                    raise ProtocolError(
                        f"training.class_balance.{dataset}.{model} has unsupported mode {mode!r}"
                    )
    rank = data.get("ranking", {})
    if int(rank.get("base", 1)) != 1:
        raise ProtocolError("New protocol rankings are canonicalized to one-based ranks")


def load_protocol(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Protocol:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ProtocolError("Protocol snapshot root must be a mapping")
            loaded = raw.get("protocol", raw)
            if not isinstance(loaded, dict):
                raise ProtocolError("Protocol snapshot 'protocol' must be a mapping")
            embedded_digest = raw.get("protocol_digest")
        else:
            loaded = yaml.safe_load(handle) or {}
            embedded_digest = None
    if not isinstance(loaded, dict):
        raise ProtocolError("Protocol root must be a mapping")

    includes = loaded.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (source.parent / include).resolve()
        with include_path.open("r", encoding="utf-8") as handle:
            fragment = yaml.safe_load(handle) or {}
        if not isinstance(fragment, dict):
            raise ProtocolError(f"Included protocol must be a mapping: {include_path}")
        merged = _deep_merge(merged, fragment)
    merged = _deep_merge(merged, loaded)
    for dotted, value in (overrides or {}).items():
        _set_dotted(merged, dotted, value)
    resolved = _expand_env(merged)
    explainers = resolved.get("explainers", {})
    if source.suffix.lower() != ".json" and explainers.get("methods_file"):
        if "locked_defaults" in explainers:
            raise ProtocolError("Use methods_file without a second locked_defaults parameter set")
        methods_path = (source.parent / explainers["methods_file"]).resolve()
        catalog = yaml.safe_load(methods_path.read_text(encoding="utf-8"))
        if not isinstance(catalog, Mapping) or catalog.get("schema_version") != 1:
            raise ProtocolError("Invalid explainer method catalog")
        defaults = {}
        patch_size = resolved["evaluation"]["patch_size"]
        for method in catalog["methods"]:
            family = str(method["family"])
            if family in defaults:
                raise ProtocolError(f"Duplicate method in catalog: {family}")
            params = dict(method.get("params", {}))
            variants = method.get("variants", [])
            if variants:
                primary = [v for v in variants if v.get("patch_size") == patch_size]
                if len(primary) != 1:
                    raise ProtocolError(f"{family} requires one primary patch-size variant")
                params.update(primary[0])
            defaults[family] = params
        explainers["locked_defaults"] = defaults
        explainers["method_catalog_digest"] = object_sha256(catalog)
    if validate:
        validate_protocol(resolved)
    digest = object_sha256(resolved)
    if embedded_digest is not None and str(embedded_digest) != digest:
        raise ProtocolError("Protocol snapshot digest does not match its contents")
    return Protocol(source=source, data=resolved, digest=digest)
