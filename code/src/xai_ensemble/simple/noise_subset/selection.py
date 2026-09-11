"""Rank-only selection of random subsets conforming to the NOISE model."""

from __future__ import annotations

import gc
import itertools
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.phase2.torch_aggregation import aggregate_rankings_torch

from ..artifacts import ArtifactError, ArtifactStore
from ..io_pipeline import ByteBoundedPrefetcher, PrefetchItem
from ..noise_prefix.anchored import (
    TopKSubsetMallowsGof,
    fidelity_anchored_subset,
    fit_topk_subset_mallows_gof,
    topk_set_distances,
)
from ..noise_prefix.inputs import (
    catalog_task_input,
    input_byte_count,
    load_input_catalog,
    load_prefix_shard,
    source_layout,
)
from ..runtime import emit_gpu_release_signal
from ..spool import SpoolQuota
from .artifacts import (
    SELECTION_SCHEMA_VERSION,
    completed_selection_manifest,
    output_store,
    publish_manifest,
)
from .config import (
    ANCHORED_CONTROL_MODE,
    GEOMETRY_RULES,
    NoiseSubsetExperiment,
    NoiseSubsetSelectionTask,
)

_KS_TOLERANCE = 1e-12


def draw_random_method_orders(
    method_count: int,
    count: int,
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Draw unique non-Fidelity method orders with one fixed RNG stream."""

    if method_count < 2 or count <= 0:
        raise ValueError("method_count must exceed one and count must be positive")
    if count >= math.factorial(method_count):
        raise ValueError("requested random-order count exhausts the permutation space")
    reference = tuple(range(method_count))
    generator = np.random.default_rng(seed)
    orders = []
    observed = {reference}
    while len(orders) < count:
        order = tuple(int(value) for value in generator.permutation(method_count).tolist())
        if order in observed:
            continue
        observed.add(order)
        orders.append(order)
    return tuple(orders)


def candidate_position_sets(
    method_count: int,
    q: int,
    *,
    reference: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    if not 1 <= q <= method_count:
        raise ValueError("q lies outside the method bank")
    frozen_reference = tuple(int(value) for value in reference)
    if len(frozen_reference) != q or len(set(frozen_reference)) != q:
        raise ValueError("reference positions must be one unique q-subset")
    values = tuple(
        positions
        for positions in itertools.combinations(range(method_count), q)
        if positions != frozen_reference
    )
    expected = math.comb(method_count, q) - 1
    if len(values) != expected:
        raise RuntimeError("candidate subset enumeration is incomplete")
    return values


def draw_candidate_position_sets(
    method_count: int,
    q: int,
    *,
    reference: Sequence[int],
    draw_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    all_positions = candidate_position_sets(method_count, q, reference=reference)
    count = min(int(draw_count), len(all_positions))
    if count <= 0:
        raise ValueError("draw_count must be positive")
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(all_positions), size=count, replace=False)
    return tuple(all_positions[int(position)] for position in selected.tolist())


def select_uniform_accepted_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_count: int,
    minimum_count: int,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    accepted = tuple(candidate for candidate in candidates if bool(candidate["accepted"]))
    if len(accepted) < minimum_count:
        raise RuntimeError(
            "Noise-consistent random subset pool is too small: "
            f"accepted={len(accepted)} minimum={minimum_count}"
        )
    count = min(selected_count, len(accepted))
    generator = np.random.default_rng(seed)
    positions = generator.choice(len(accepted), size=count, replace=False)
    return tuple(accepted[int(position)] for position in positions.tolist())


def _gof_record(value: TopKSubsetMallowsGof) -> Mapping[str, Any]:
    theta = value.fit.theta
    return {
        "theta": None if not math.isfinite(theta) else float(theta),
        "theta_is_infinite": bool(math.isinf(theta)),
        "boundary": bool(value.fit.boundary),
        "mean_distance": float(value.fit.mean_distance),
        "expected_distance": float(value.fit.expected_distance),
        "ks_statistic": float(value.ks_statistic),
        "total_variation": float(value.total_variation),
        "log_likelihood": float(value.log_likelihood),
        "empirical_probabilities": list(value.empirical_probabilities),
        "fitted_probabilities": list(value.fitted_probabilities),
    }


def _aggregate_center(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    ballots: np.ndarray,
    indices: np.ndarray,
    geometry: str,
    methods: Sequence[str],
    device: Any,
) -> np.ndarray:
    rule = GEOMETRY_RULES[geometry]
    requested = ("Borda",) if rule == "borda" else ("Kemeny",)
    seeds = tuple(
        stable_seed(
            "simple-noise-random-subset-center-v1",
            task.digest,
            geometry,
            tuple(methods),
            int(row_index),
        )
        for row_index in indices.tolist()
    )
    result = aggregate_rankings_torch(
        ballots,
        None,
        requested=requested,
        rrf_c=experiment.base.phase2.rrf_c,
        kemeny_starts=experiment.base.phase2.kemeny_starts,
        kemeny_max_passes=experiment.base.phase2.kemeny_max_passes,
        seeds=seeds,
        device=device,
        workspace_bytes=experiment.runtime.aggregation_workspace_bytes,
    )
    center = np.asarray(result[rule], dtype=np.int64)
    del result
    return center


def _candidate_record(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    all_ballots: np.ndarray,
    indices: np.ndarray,
    ordered_methods: Sequence[str],
    fidelity: Mapping[str, float],
    geometry: str,
    positions: Sequence[int],
    device: Any,
) -> Mapping[str, Any]:
    selected_positions = tuple(int(value) for value in positions)
    methods = tuple(str(ordered_methods[position]) for position in selected_positions)
    ballots = all_ballots[:, selected_positions, :]
    center = _aggregate_center(
        experiment,
        task,
        ballots=ballots,
        indices=indices,
        geometry=geometry,
        methods=methods,
        device=device,
    )
    top_indices = np.argsort(center, axis=1, kind="stable")[:, : experiment.k]
    distances = topk_set_distances(ballots, top_indices, k=experiment.k)
    gof = fit_topk_subset_mallows_gof(
        distances,
        n_items=all_ballots.shape[2],
        k=experiment.k,
    )
    qualities = tuple(float(fidelity[method]) for method in methods)
    identity = {
        "geometry": geometry,
        "positions": list(selected_positions),
        "methods": list(methods),
        "q": len(methods),
    }
    return {
        "candidate_digest": object_sha256(identity),
        **identity,
        "individual_clean_F": list(qualities),
        "mean_individual_clean_F": float(np.mean(qualities)),
        "gof": _gof_record(gof),
    }


def _load_q11_contribution_shard(
    store: ArtifactStore,
    record: Mapping[str, Any],
    *,
    ordered_methods: Sequence[str],
    expected_task_digest: str,
    directory: Path,
) -> tuple[np.ndarray, np.ndarray]:
    payload = record.get("payload")
    labels = record.get("rule_labels")
    if not isinstance(payload, Mapping) or not isinstance(labels, Mapping):
        raise ArtifactError("q=11 reference shard record is malformed")
    by_name = {str(label): str(field) for field, label in labels.items()}
    required = {f"single__{method}" for method in ordered_methods}
    if len(by_name) != len(labels) or not required.issubset(by_name):
        raise ArtifactError("q=11 reference shard lacks individual-method fields")
    path = store.materialize(
        str(payload["relative_path"]),
        directory / Path(str(payload["relative_path"])).name,
        expected_sha256=str(payload["sha256"]),
    )
    with safe_open(path, framework="np") as shard:
        metadata = shard.metadata() or {}
        if metadata.get("task_digest") != expected_task_digest:
            raise ArtifactError("q=11 reference shard task identity changed")
        indices = shard.get_tensor("indices").astype(np.int64, copy=False)
        labels_array = shard.get_tensor("labels").astype(np.int64, copy=False)
        predictions = shard.get_tensor("unmasked_predictions").astype(np.int64, copy=False)
        clean_correct = (predictions == labels_array).astype(np.int8)
        contributions = np.stack(
            [
                clean_correct
                - (
                    shard.get_tensor(f"removed_predictions__{by_name[f'single__{method}']}")
                    == labels_array
                ).astype(np.int8)
                for method in ordered_methods
            ],
            axis=1,
        )
    return indices.copy(), contributions


def _load_q11_contributions(
    experiment: NoiseSubsetExperiment,
    task_row: Mapping[str, Any],
    *,
    ordered_methods: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    reference = task_row.get("q11_reference")
    if not isinstance(reference, Mapping):
        raise ArtifactError("input catalog lacks the q=11 reference")
    artifact_root = str(reference["artifact_root"])
    expected_task_digest = str(reference["task_digest"])
    store = ArtifactStore(experiment.base)
    manifest = store.read_json(f"{artifact_root}/manifest.json")
    manifest_digest = object_sha256(manifest)
    if (
        manifest_digest != reference.get("manifest_content_digest")
        or manifest.get("task_digest") != expected_task_digest
        or int(manifest.get("sample_count", -1)) != int(reference["sample_count"])
    ):
        raise ArtifactError("q=11 reference manifest identity changed")
    records_value = manifest.get("shards")
    if not isinstance(records_value, Sequence) or isinstance(records_value, (str, bytes)):
        raise ArtifactError("q=11 reference manifest has no shard sequence")
    records = tuple(sorted(records_value, key=lambda row: int(row["shard_index"])))
    expected_start = 0
    items = []
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ArtifactError("q=11 reference shard record is malformed")
        start = int(record["start"])
        stop = int(record["stop"])
        payload = record.get("payload")
        if (
            int(record["shard_index"]) != expected_index
            or start != expected_start
            or stop <= start
            or not isinstance(payload, Mapping)
        ):
            raise ArtifactError("q=11 reference shard layout is invalid")
        expected_start = stop
        items.append(
            PrefetchItem(
                key=expected_index,
                byte_count=int(payload["size_bytes"]),
                load=lambda directory, row=record: _load_q11_contribution_shard(
                    store,
                    row,
                    ordered_methods=ordered_methods,
                    expected_task_digest=expected_task_digest,
                    directory=directory,
                ),
            )
        )
    if expected_start != int(reference["sample_count"]):
        raise ArtifactError("q=11 reference shard coverage is incomplete")

    quota = SpoolQuota(
        experiment.storage.spool_root,
        max_bytes=experiment.storage.spool_max_bytes,
        min_free_bytes=experiment.storage.spool_min_free_bytes,
    )
    indices = []
    contributions = []
    with ByteBoundedPrefetcher(
        quota,
        items,
        workers=experiment.runtime.cpu_workers,
        namespace=f"noise-random-q11-{expected_task_digest[:16]}",
    ) as prefetcher:
        for shard_index in range(len(items)):
            prefetched = prefetcher.get(shard_index)
            shard_indices, shard_contributions = prefetched.value
            indices.append(shard_indices)
            contributions.append(shard_contributions)
            prefetcher.release(shard_index)
    row_indices = np.concatenate(indices)
    values = np.concatenate(contributions)
    order = np.argsort(row_indices, kind="stable")
    if np.unique(row_indices).size != row_indices.size:
        raise ArtifactError("q=11 contribution sample ids are not unique")
    return row_indices[order], values[order], manifest_digest


def _anchored_candidate_record(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    all_ballots: np.ndarray,
    contributions: np.ndarray,
    indices: np.ndarray,
    ordered_methods: Sequence[str],
    fidelity: Mapping[str, float],
    geometry: str,
    positions: Sequence[int],
    device: Any,
) -> Mapping[str, Any]:
    selected_positions = tuple(sorted(int(value) for value in positions))
    if len(selected_positions) != len(set(selected_positions)):
        raise ValueError("anchored candidate positions must be unique")
    methods = tuple(str(ordered_methods[position]) for position in selected_positions)
    center = _aggregate_center(
        experiment,
        task,
        ballots=all_ballots[:, selected_positions, :],
        indices=indices,
        geometry=geometry,
        methods=methods,
        device=device,
    )
    top_indices = np.argsort(center, axis=1, kind="stable")[:, : experiment.k]
    distances = topk_set_distances(all_ballots, top_indices, k=experiment.k)
    anchored = fidelity_anchored_subset(
        distances,
        contributions,
        selected_positions=selected_positions,
        n_items=all_ballots.shape[2],
        k=experiment.k,
    )
    qualities = tuple(float(fidelity[method]) for method in methods)
    identity = {
        "schema": "simple-noise-random-order-candidate-v1",
        "cell": task.cell.cell_id,
        "geometry": geometry,
        "positions": list(selected_positions),
        "q": len(selected_positions),
    }
    return {
        "candidate_digest": object_sha256(identity),
        "geometry": geometry,
        "center_rule": GEOMETRY_RULES[geometry],
        "positions": list(selected_positions),
        "methods": list(methods),
        "q": len(methods),
        "individual_clean_F": list(qualities),
        "mean_individual_clean_F": float(np.mean(qualities)),
        "anchored_score": anchored.score,
        "theta": None if not math.isfinite(anchored.theta) else anchored.theta,
        "theta_is_infinite": bool(math.isinf(anchored.theta)),
        "mean_distance": anchored.mean_distance,
    }


def _load_clean_ballots(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    prefix_task = experiment.prefix_task(task.cell, "clean")
    catalog = load_input_catalog(experiment.prefix)
    task_row = catalog_task_input(experiment.prefix, prefix_task, catalog=catalog)
    layout = source_layout(task_row)
    quota = SpoolQuota(
        experiment.storage.spool_root,
        max_bytes=experiment.storage.spool_max_bytes,
        min_free_bytes=experiment.storage.spool_min_free_bytes,
    )
    items = [
        PrefetchItem(
            key=shard_index,
            byte_count=input_byte_count(task_row, shard_index),
            load=lambda directory, index=shard_index: load_prefix_shard(
                experiment.prefix,
                task_row,
                shard_index=index,
                work_directory=directory,
            ),
        )
        for shard_index, _, _ in layout
    ]
    indices = []
    ballots = []
    with ByteBoundedPrefetcher(
        quota,
        items,
        workers=experiment.runtime.cpu_workers,
        namespace=f"noise-random-select-{task.digest[:16]}",
    ) as prefetcher:
        for shard_index, _, _ in layout:
            prefetched = prefetcher.get(shard_index)
            shard = prefetched.value
            indices.append(shard.reference["indices"].numpy().astype(np.int64, copy=False).copy())
            ballots.append(shard.ballots)
            prefetcher.release(shard_index)
            del shard
    row_indices = np.concatenate(indices)
    rank_bank = np.concatenate(ballots)
    if np.unique(row_indices).size != row_indices.size:
        raise ArtifactError("random-subset selector input indices are not unique")
    if rank_bank.shape[0] != row_indices.size or rank_bank.shape[1] != 11:
        raise ArtifactError("random-subset selector rank bank is incomplete")
    return row_indices, rank_bank, catalog, task_row


def _run_ks_selection_task(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    import torch

    store = output_store(experiment)
    complete = completed_selection_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("random NOISE subset selection requires CUDA")

    selector_cell = experiment.selector_cell(task.cell.cell_id)
    ordered_methods = tuple(str(value) for value in selector_cell["ordered_methods"])
    row_indices = None
    rank_bank = None
    try:
        row_indices, rank_bank, catalog, task_row = _load_clean_ballots(experiment, task)
        if task_row.get("ordered_methods") != list(ordered_methods):
            raise ArtifactError("random-subset selector method order differs from input catalog")
        fidelity = {str(key): float(value) for key, value in task_row["fidelity"].items()}
        geometries = {}
        for geometry in experiment.geometries:
            geometry_source = selector_cell["geometries"][geometry]
            q = int(geometry_source["q"])
            reference_positions = tuple(range(q))
            reference = _candidate_record(
                experiment,
                task,
                all_ballots=rank_bank,
                indices=row_indices,
                ordered_methods=ordered_methods,
                fidelity=fidelity,
                geometry=geometry,
                positions=reference_positions,
                device=target_device,
            )
            reference_ks = float(reference["gof"]["ks_statistic"])
            if bool(reference["gof"]["boundary"]):
                raise RuntimeError(f"frozen NOISE reference lies on a Mallows boundary: {geometry}")
            candidates = []
            possible_candidate_count = len(
                candidate_position_sets(len(ordered_methods), q, reference=reference_positions)
            )
            position_sets = draw_candidate_position_sets(
                len(ordered_methods),
                q,
                reference=reference_positions,
                draw_count=experiment.candidate_draw_count,
                seed=stable_seed(
                    "simple-noise-random-subset-candidate-pool-v1",
                    experiment.random_seed,
                    task.digest,
                    geometry,
                ),
            )
            for candidate_index, positions in enumerate(position_sets, start=1):
                record = dict(
                    _candidate_record(
                        experiment,
                        task,
                        all_ballots=rank_bank,
                        indices=row_indices,
                        ordered_methods=ordered_methods,
                        fidelity=fidelity,
                        geometry=geometry,
                        positions=positions,
                        device=target_device,
                    )
                )
                gof = record["gof"]
                record["accepted"] = bool(
                    not gof["boundary"]
                    and float(gof["ks_statistic"]) <= reference_ks + _KS_TOLERANCE
                )
                candidates.append(record)
                if candidate_index % 16 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                print(
                    "NOISE_RANDOM_CANDIDATE "
                    f"geometry={geometry} candidate={candidate_index}/{len(position_sets)} "
                    f"accepted={record['accepted']} ks={float(gof['ks_statistic']):.8f} "
                    f"reference_ks={reference_ks:.8f}",
                    flush=True,
                )
            selected = select_uniform_accepted_candidates(
                candidates,
                selected_count=experiment.selected_candidate_count,
                minimum_count=experiment.minimum_accepted_candidates,
                seed=stable_seed(
                    "simple-noise-random-subset-selection-v1",
                    experiment.random_seed,
                    task.digest,
                    geometry,
                ),
            )
            selected_rows = [
                {**dict(record), "selection_position": position}
                for position, record in enumerate(selected)
            ]
            accepted_count = sum(bool(record["accepted"]) for record in candidates)
            geometries[geometry] = {
                "geometry": geometry,
                "center_rule": GEOMETRY_RULES[geometry],
                "q": q,
                "reference": reference,
                "possible_candidate_count": possible_candidate_count,
                "candidate_draw_count": len(candidates),
                "accepted_count": accepted_count,
                "acceptance_rate": float(accepted_count / len(candidates)),
                "selected_count": len(selected_rows),
                "selected": selected_rows,
                "candidates": candidates,
            }
        value: Mapping[str, Any] = {
            "schema": "simple-noise-random-subset-selection-v1",
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "study_id": experiment.study_id,
            "study_digest": experiment.digest,
            "task_id": task.task_id,
            "task_digest": task.digest,
            "cell": task.cell.cell_id,
            "dataset": task.cell.dataset.dataset_id,
            "model": task.cell.reference_model.model_id,
            "split": experiment.split,
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "sample_count": int(row_indices.size),
            "input_catalog_digest": catalog["catalog_digest"],
            "prefix_task_digest": experiment.prefix_task(task.cell, "clean").digest,
            "independent_selector_digest": experiment.selector_digest,
            "ordered_methods": list(ordered_methods),
            "fidelity": fidelity,
            "fit_policy": experiment.fit_policy,
            "acceptance_threshold": "candidate_ks_lte_frozen_reference_ks",
            "candidate_policy": experiment.candidate_policy,
            "selection_policy": experiment.selection_policy,
            "candidate_draw_count_requested": experiment.candidate_draw_count,
            "selected_candidate_count_requested": experiment.selected_candidate_count,
            "minimum_accepted_candidates": experiment.minimum_accepted_candidates,
            "random_seed": experiment.random_seed,
            "geometries": geometries,
        }
        publish_manifest(
            experiment,  # type: ignore[arg-type]
            store,
            root=task.artifact_root,
            task_id=task.task_id,
            manifest=value,
        )
        return value
    finally:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        row_indices = None
        rank_bank = None
        gc.collect()
        torch.cuda.empty_cache()
        emit_gpu_release_signal()


def _run_anchored_selection_task(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    device: str,
) -> Mapping[str, Any]:
    import torch

    store = output_store(experiment)
    complete = completed_selection_manifest(experiment, task, store=store)
    if complete is not None:
        return complete
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("anchored random-order selection requires CUDA")

    selector_cell = experiment.selector_cell(task.cell.cell_id)
    ordered_methods = tuple(str(value) for value in selector_cell["ordered_methods"])
    row_indices = None
    rank_bank = None
    contributions = None
    try:
        row_indices, rank_bank, catalog, task_row = _load_clean_ballots(experiment, task)
        if task_row.get("ordered_methods") != list(ordered_methods):
            raise ArtifactError("random-order selector method order differs from input catalog")
        rank_order = np.argsort(row_indices, kind="stable")
        row_indices = row_indices[rank_order]
        rank_bank = rank_bank[rank_order]
        contribution_indices, contributions, contribution_manifest_digest = _load_q11_contributions(
            experiment,
            task_row,
            ordered_methods=ordered_methods,
        )
        if not np.array_equal(row_indices, contribution_indices):
            raise ArtifactError("rank and clean-F contribution sample ids are not aligned")
        fidelity = {str(key): float(value) for key, value in task_row["fidelity"].items()}
        for position, method in enumerate(ordered_methods):
            observed = float(np.mean(contributions[:, position]))
            if not math.isclose(observed, fidelity[method], rel_tol=0.0, abs_tol=1e-12):
                raise ArtifactError(
                    f"clean-F utility marks disagree with the catalog for {method}: "
                    f"observed={observed} catalog={fidelity[method]}"
                )

        orders = draw_random_method_orders(
            len(ordered_methods),
            experiment.random_order_count,
            seed=stable_seed(
                "simple-noise-random-order-bank-v1",
                experiment.random_seed,
                task.digest,
            ),
        )
        geometries = {}
        for geometry in experiment.geometries:
            candidate_cache: dict[tuple[int, ...], Mapping[str, Any]] = {}
            order_rows = []
            for order_index, method_order in enumerate(orders):
                prefixes = []
                for q in experiment.q_values:
                    prefix_positions = tuple(method_order[:q])
                    canonical_positions = tuple(sorted(prefix_positions))
                    candidate = candidate_cache.get(canonical_positions)
                    if candidate is None:
                        candidate = {
                            **_anchored_candidate_record(
                                experiment,
                                task,
                                all_ballots=rank_bank,
                                contributions=contributions,
                                indices=row_indices,
                                ordered_methods=ordered_methods,
                                fidelity=fidelity,
                                geometry=geometry,
                                positions=canonical_positions,
                                device=target_device,
                            ),
                            "candidate_position": len(candidate_cache),
                        }
                        candidate_cache[canonical_positions] = candidate
                    prefixes.append(
                        {
                            "q": q,
                            "candidate_digest": candidate["candidate_digest"],
                            "candidate_position": candidate["candidate_position"],
                            "prefix_positions": list(prefix_positions),
                            "prefix_methods": [
                                ordered_methods[position] for position in prefix_positions
                            ],
                            "anchored_score": candidate["anchored_score"],
                            "theta": candidate["theta"],
                            "theta_is_infinite": candidate["theta_is_infinite"],
                            "mean_distance": candidate["mean_distance"],
                        }
                    )
                    print(
                        "NOISE_RANDOM_ORDER_CANDIDATE "
                        f"geometry={geometry} order={order_index + 1}/{len(orders)} "
                        f"q={q} unique={len(candidate_cache)} "
                        f"score={float(candidate['anchored_score']):.10f}",
                        flush=True,
                    )
                    if len(candidate_cache) % 16 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                selected_prefix = max(prefixes, key=lambda row: float(row["anchored_score"]))
                selected_candidate = candidate_cache[
                    tuple(sorted(int(value) for value in selected_prefix["prefix_positions"]))
                ]
                order_rows.append(
                    {
                        "random_order_index": order_index,
                        "order_positions": list(method_order),
                        "order_methods": [ordered_methods[position] for position in method_order],
                        "prefixes": prefixes,
                        "selected_q": int(selected_prefix["q"]),
                        "selected_candidate_digest": selected_candidate["candidate_digest"],
                        "selected_candidate_position": selected_candidate["candidate_position"],
                        "selected_anchored_score": selected_candidate["anchored_score"],
                    }
                )
            candidates = tuple(
                sorted(candidate_cache.values(), key=lambda row: int(row["candidate_position"]))
            )
            selected = []
            by_digest = {str(row["candidate_digest"]): row for row in candidates}
            for order_row in order_rows:
                candidate = by_digest[str(order_row["selected_candidate_digest"])]
                selected.append(
                    {
                        **dict(candidate),
                        "selection_position": int(order_row["random_order_index"]),
                        "random_order_index": int(order_row["random_order_index"]),
                    }
                )
            formal = selector_cell["geometries"][geometry]
            geometries[geometry] = {
                "geometry": geometry,
                "center_rule": GEOMETRY_RULES[geometry],
                "formal_fidelity_order_reference": {
                    "q": int(formal["q"]),
                    "score": float(formal["score"]),
                    "theta": float(formal["theta"]),
                    "mean_distance": float(formal["mean_distance"]),
                    "methods": list(formal["method_prefix"]),
                    "candidates": [dict(row) for row in formal["candidates"]],
                },
                "random_order_count": len(order_rows),
                "unique_candidate_count": len(candidates),
                "candidate_evaluation_policy": (
                    "all_unique_prefixes_geometry_center_rule_plus_selected_prefixes_all_rules"
                ),
                "orders": order_rows,
                "selected": selected,
                "candidates": list(candidates),
            }
        value: Mapping[str, Any] = {
            "schema": "simple-noise-random-order-anchored-selection-v1",
            "schema_version": experiment.artifact_schema_version,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "study_id": experiment.study_id,
            "study_digest": experiment.digest,
            "task_id": task.task_id,
            "task_digest": task.digest,
            "cell": task.cell.cell_id,
            "dataset": task.cell.dataset.dataset_id,
            "model": task.cell.reference_model.model_id,
            "split": experiment.split,
            "patch_size": experiment.patch_size,
            "k": experiment.k,
            "q_values": list(experiment.q_values),
            "sample_count": int(row_indices.size),
            "input_catalog_digest": catalog["catalog_digest"],
            "prefix_task_digest": experiment.prefix_task(task.cell, "clean").digest,
            "q11_contribution_manifest_content_digest": contribution_manifest_digest,
            "independent_selector_digest": experiment.selector_digest,
            "ordered_methods": list(ordered_methods),
            "fidelity": fidelity,
            "control_mode": experiment.control_mode,
            "fit_policy": experiment.fit_policy,
            "candidate_policy": experiment.candidate_policy,
            "selection_policy": experiment.selection_policy,
            "utility_mark": "per_sample_individual_clean_F_contribution",
            "utility_anchor_scope": "all_ordered_individual_methods",
            "aggregate_masked_forwards_used_for_selection": False,
            "random_order_count": experiment.random_order_count,
            "random_seed": experiment.random_seed,
            "geometries": geometries,
        }
        publish_manifest(
            experiment,  # type: ignore[arg-type]
            store,
            root=task.artifact_root,
            task_id=task.task_id,
            manifest=value,
        )
        return value
    finally:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        row_indices = None
        rank_bank = None
        contributions = None
        gc.collect()
        torch.cuda.empty_cache()
        emit_gpu_release_signal()


def run_selection_task(
    experiment: NoiseSubsetExperiment,
    task: NoiseSubsetSelectionTask,
    *,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    if experiment.control_mode == ANCHORED_CONTROL_MODE:
        return _run_anchored_selection_task(experiment, task, device=device)
    return _run_ks_selection_task(experiment, task, device=device)


__all__ = [
    "candidate_position_sets",
    "draw_random_method_orders",
    "draw_candidate_position_sets",
    "run_selection_task",
    "select_uniform_accepted_candidates",
]
