"""Streaming materialization and freezing of the independent NOISE selector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json, read_json
from xai_ensemble.simple.artifacts import ArtifactStore
from xai_ensemble.simple.noise_prefix.artifacts import (
    completed_evaluation_manifest,
    output_store,
)
from xai_ensemble.simple.noise_prefix.config import NoisePrefixExperiment
from xai_ensemble.simple.noise_prefix.independent_geometry import (
    build_independent_geometry_selector,
)
from xai_ensemble.simple.phase2 import phase2_artifact_root

from .config import FullMatrixExperiment


def _subset_experiment(
    experiment: NoisePrefixExperiment,
    cell_id: str,
) -> NoisePrefixExperiment:
    cell = next(item for item in experiment.cells() if item.cell_id == cell_id)
    base = experiment.base
    dataset = base.dataset(cell.dataset.dataset_id)
    model = base.model(cell.reference_model.model_id)
    subset_base = replace(
        base,
        datasets=(dataset,),
        models=(model,),
    )
    subset_assumptions = replace(
        experiment.assumptions,
        base=subset_base,
        _task_cache={},
    )
    subset = replace(experiment, assumptions=subset_assumptions, _task_cache={})
    if subset.digest != experiment.digest:
        raise RuntimeError("Cell-scoped selector changed the formal sweep identity")
    return subset


def _copy_manifest_and_shards(
    store: ArtifactStore,
    *,
    local_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(local_root / "manifest.json", manifest)
    for record in manifest.get("shards", ()):
        payload = record["payload"]
        relative = str(payload["relative_path"])
        destination = local_root / "shards" / Path(relative).name
        store.materialize(
            relative,
            destination,
            expected_sha256=str(payload["sha256"]),
        )


def materialize_selector_cell(
    experiment: FullMatrixExperiment,
    *,
    prefix_config: str | Path,
    cell_id: str,
    result_root: str | Path | None = None,
) -> Mapping[str, Any]:
    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment

    prefix = load_noise_prefix_experiment(prefix_config)
    cell = next(item for item in prefix.cells() if item.cell_id == cell_id)
    root = (
        experiment.result_root if result_root is None else Path(result_root).expanduser().resolve()
    )
    partial_path = root / "noise-prefix" / "selector" / "partials" / f"{cell_id}.json"
    if partial_path.is_file():
        return read_json(partial_path)
    base_task = prefix.base_phase2_task(cell, "clean")
    prefix_task = next(
        task
        for task in prefix.evaluation_tasks()
        if task.cell.cell_id == cell_id and task.condition.kind == "clean"
    )
    base_store = ArtifactStore(prefix.base)
    prefix_store = output_store(prefix)
    base_root = phase2_artifact_root(base_task)
    base_manifest = base_store.read_json(f"{base_root}/manifest.json")
    prefix_manifest = completed_evaluation_manifest(prefix, prefix_task, store=prefix_store)
    if prefix_manifest is None:
        raise FileNotFoundError(f"Clean prefix artifact is incomplete: {prefix_task.task_id}")

    experiment.storage.selector_cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f"selector-{cell_id}-",
        dir=experiment.storage.selector_cache_root,
    ) as temporary:
        root = Path(temporary)
        base_cache = root / "base"
        prefix_cache = root / "prefix"
        _copy_manifest_and_shards(
            base_store,
            local_root=base_cache / "phase2" / cell_id,
            manifest=base_manifest,
        )
        prefix_local_root = prefix_cache / cell_id / "clean" / "p16" / "k20" / prefix_task.digest
        _copy_manifest_and_shards(
            prefix_store,
            local_root=prefix_local_root,
            manifest=prefix_manifest,
        )
        selector = build_independent_geometry_selector(
            _subset_experiment(prefix, cell_id),
            base_clean_cache=base_cache,
            prefix_clean_cache=prefix_cache,
        )
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(partial_path, selector)
    return selector


def merge_selector_partials(
    experiment: FullMatrixExperiment,
    *,
    prefix_config: str | Path,
    result_root: str | Path | None = None,
) -> Mapping[str, Any]:
    from xai_ensemble.simple.noise_prefix.config import load_noise_prefix_experiment

    prefix = load_noise_prefix_experiment(prefix_config)
    root = (
        experiment.result_root if result_root is None else Path(result_root).expanduser().resolve()
    )
    partial_root = root / "noise-prefix" / "selector" / "partials"
    partials = [read_json(partial_root / f"{cell.cell_id}.json") for cell in prefix.cells()]
    if not partials:
        raise RuntimeError("No selector partials are available")
    common_keys = (
        "schema",
        "schema_version",
        "status",
        "experiment_id",
        "scope",
        "sweep_id",
        "sweep_digest",
        "selection_input_contract",
        "science",
    )
    first = partials[0]
    for value in partials:
        if any(value.get(key) != first.get(key) for key in common_keys):
            raise ValueError("Selector partials do not share one immutable identity")
    cells = []
    source_rows = []
    seen = set()
    for value in partials:
        rows = value.get("cells")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("Every selector partial must contain exactly one cell")
        row = rows[0]
        cell_id = str(row["cell"])
        if cell_id in seen:
            raise ValueError(f"Duplicate selector partial: {cell_id}")
        seen.add(cell_id)
        cells.append(row)
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"Selector partial has no source provenance: {cell_id}")
        source_rows.append({"cell": cell_id, **dict(source)})
    expected = {cell.cell_id for cell in prefix.cells()}
    if seen != expected:
        raise ValueError(f"Selector partial coverage differs from active cells: {seen ^ expected}")
    payload = {key: first[key] for key in common_keys}
    payload["selection_source_digest"] = object_sha256(
        sorted(source_rows, key=lambda row: str(row["cell"]))
    )
    payload["cells"] = sorted(cells, key=lambda row: str(row["cell"]))
    payload["selector_digest"] = object_sha256(payload)
    output = root / "noise-prefix" / "selector" / "selector.json"
    if output.is_file() and read_json(output) != payload:
        raise ValueError("Existing merged selector contradicts selector partials")
    atomic_write_json(output, payload)
    return payload


__all__ = ["merge_selector_partials", "materialize_selector_cell"]
