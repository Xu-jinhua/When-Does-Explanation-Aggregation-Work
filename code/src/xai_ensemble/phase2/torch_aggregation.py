"""Batched Torch backend for the formal rank aggregation rules."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MATRIX_WORKSPACE_COPIES = 12
_MATRIX_ELEMENT_BYTES = 4


class AggregationResult(Mapping[str, NDArray[np.int64]]):
    """Rank outputs with JSON-serializable aggregation diagnostics."""

    def __init__(
        self,
        rankings: Mapping[str, NDArray[np.int64]],
        statistics: Mapping[str, Mapping[str, int | float]],
    ) -> None:
        self._rankings = dict(rankings)
        self.statistics = {
            str(rule): {str(key): value for key, value in values.items()}
            for rule, values in statistics.items()
        }

    def __getitem__(self, key: str) -> NDArray[np.int64]:
        return self._rankings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rankings)

    def __len__(self) -> int:
        return len(self._rankings)


def _scores_to_ranks(scores: Any) -> Any:
    import torch

    if scores.ndim != 2:
        raise ValueError("scores must have [batch,items] shape")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order, dtype=torch.int64)
    positions = torch.arange(order.shape[1], device=order.device, dtype=torch.int64)
    ranks.scatter_(1, order, positions.expand_as(order))
    return ranks


def _pairwise_counts(ballots: Any) -> Any:
    import torch

    batch, methods, patches = ballots.shape
    counts = torch.zeros(
        (batch, patches, patches),
        dtype=torch.int32,
        device=ballots.device,
    )
    for method in range(methods):
        ranks = ballots[:, method]
        counts.add_(ranks.unsqueeze(2) < ranks.unsqueeze(1))
    return counts


def _matrix_batch_capacity(patches: int, workspace_bytes: int) -> int:
    bytes_per_instance = _MATRIX_WORKSPACE_COPIES * _MATRIX_ELEMENT_BYTES * patches * patches
    return max(1, workspace_bytes // bytes_per_instance)


def _schulze_ranks(ballots: Any, *, workspace_bytes: int) -> Any:
    import torch

    batch, _, patches = ballots.shape
    if patches == 1:
        return torch.zeros((batch, 1), dtype=torch.int64, device=ballots.device)
    capacity = _matrix_batch_capacity(patches, workspace_bytes)
    output = torch.empty((batch, patches), dtype=torch.int64, device=ballots.device)
    patch_ids = torch.arange(patches, dtype=torch.int64, device=ballots.device)
    for start in range(0, batch, capacity):
        stop = min(batch, start + capacity)
        chunk = ballots[start:stop]
        counts = _pairwise_counts(chunk)
        paths = torch.where(counts > counts.transpose(1, 2), counts, 0)
        paths.diagonal(dim1=1, dim2=2).zero_()
        for intermediate in range(patches):
            through = torch.minimum(
                paths[:, :, intermediate].unsqueeze(2),
                paths[:, intermediate, :].unsqueeze(1),
            )
            torch.maximum(paths, through, out=paths)
        paths.diagonal(dim1=1, dim2=2).zero_()

        relation = paths > paths.transpose(1, 2)
        indegree = relation.sum(dim=1, dtype=torch.int64)
        selected = torch.zeros_like(indegree, dtype=torch.bool)
        ranking = torch.empty_like(indegree)
        borda = ((patches - 1) - chunk).sum(dim=1, dtype=torch.int64)
        priority = borda * (patches + 1) + (patches - patch_ids)
        valid = torch.ones(stop - start, dtype=torch.bool, device=ballots.device)
        row_ids = torch.arange(stop - start, device=ballots.device)
        unavailable = torch.iinfo(torch.int64).min
        for position in range(patches):
            available = indegree.eq(0) & ~selected
            valid &= available.any(dim=1)
            chosen = priority.masked_fill(~available, unavailable).argmax(dim=1)
            ranking[row_ids, chosen] = position
            selected[row_ids, chosen] = True
            indegree -= relation[row_ids, chosen].to(torch.int64)
        if not bool(valid.all().item()):
            raise RuntimeError("Schulze strongest-path relation unexpectedly contains a cycle")
        output[start:stop] = ranking
    return output


def _initial_orders(
    ballots: NDArray[np.int64],
    borda_ranking: NDArray[np.int64],
    schulze_ranking: NDArray[np.int64] | None,
    *,
    n_starts: int,
    seed: int,
) -> NDArray[np.int64]:
    patches = ballots.shape[1]
    starts: list[NDArray[np.int64]] = []
    seen: set[tuple[int, ...]] = set()

    def add(order: NDArray[np.int64]) -> None:
        key = tuple(int(item) for item in order)
        if key not in seen and len(starts) < n_starts:
            seen.add(key)
            starts.append(order.copy())

    add(np.argsort(borda_ranking, kind="stable").astype(np.int64, copy=False))
    if schulze_ranking is not None:
        add(np.argsort(schulze_ranking, kind="stable").astype(np.int64, copy=False))
    for ballot in ballots:
        add(np.argsort(ballot, kind="stable").astype(np.int64, copy=False))
    generator = np.random.default_rng(seed)
    attempts = 0
    max_attempts = max(100, 100 * n_starts)
    while len(starts) < n_starts and attempts < max_attempts:
        add(generator.permutation(patches).astype(np.int64, copy=False))
        attempts += 1
    return np.stack(starts)


def _ordered_counts(counts: Any, orders: Any) -> Any:
    import torch

    patches = orders.shape[1]
    rows = orders.unsqueeze(2).expand(-1, -1, patches)
    row_counts = torch.gather(counts, 1, rows)
    columns = orders.unsqueeze(1).expand(-1, patches, -1)
    return torch.gather(row_counts, 2, columns)


def _destination_order(patches: int, device: Any) -> Any:
    import torch

    values = np.empty((patches, patches - 1), dtype=np.int64)
    for source in range(patches):
        values[source] = np.asarray(
            [*range(source + 1, patches), *range(source - 1, -1, -1)],
            dtype=np.int64,
        )
    return torch.from_numpy(values).to(device=device)


def _best_insertion_moves(orders: Any, counts: Any, destinations: Any) -> tuple[Any, Any, Any]:
    import torch

    instances, patches = orders.shape
    ordered = _ordered_counts(counts, orders)
    margins = ordered - ordered.transpose(1, 2)
    prefix = torch.cat(
        (
            torch.zeros(
                (instances, patches, 1),
                dtype=counts.dtype,
                device=orders.device,
            ),
            torch.cumsum(margins, dim=2, dtype=counts.dtype),
        ),
        dim=2,
    )
    positions = torch.arange(patches, dtype=torch.int64, device=orders.device)
    source = positions.view(1, patches, 1).expand(instances, -1, -1)
    source_after = prefix.gather(2, source + 1)
    source_before = prefix.gather(2, source)
    forward = prefix[:, :, 1:] - source_after
    backward = prefix[:, :, :patches] - source_before
    destination = positions.view(1, 1, patches)
    deltas = torch.where(destination > source, forward, backward)
    candidates = deltas.gather(
        2,
        destinations.unsqueeze(0).expand(instances, -1, -1),
    )
    best_delta, flat_index = candidates.flatten(start_dim=1).min(dim=1)
    best_source = torch.div(flat_index, patches - 1, rounding_mode="floor")
    local_index = flat_index.remainder(patches - 1)
    best_destination = destinations[best_source, local_index]
    return best_source, best_destination, best_delta


def _apply_insertions(orders: Any, source: Any, destination: Any) -> Any:
    import torch

    instances, patches = orders.shape
    positions = torch.arange(patches, dtype=torch.int64, device=orders.device).expand(instances, -1)
    source_column = source.unsqueeze(1)
    destination_column = destination.unsqueeze(1)
    gather = positions.clone()
    moving_forward = source_column < destination_column
    moving_backward = source_column > destination_column
    gather = torch.where(
        moving_forward & (positions >= source_column) & (positions < destination_column),
        positions + 1,
        gather,
    )
    gather = torch.where(
        moving_forward & positions.eq(destination_column),
        source_column,
        gather,
    )
    gather = torch.where(
        moving_backward & (positions > destination_column) & (positions <= source_column),
        positions - 1,
        gather,
    )
    gather = torch.where(
        moving_backward & positions.eq(destination_column),
        source_column,
        gather,
    )
    return orders.gather(1, gather)


def _kemeny_objectives(orders: Any, counts: Any) -> Any:
    import torch

    ordered = _ordered_counts(counts, orders)
    return torch.tril(ordered, diagonal=-1).sum(dim=(1, 2), dtype=torch.int64)


def _search_kemeny_orders(
    orders: Any,
    counts: Any,
    *,
    max_passes: int,
) -> tuple[Any, Any, Any, Any]:
    import torch

    if orders.shape[1] < 2:
        return (
            orders,
            _kemeny_objectives(orders, counts),
            torch.zeros(orders.shape[0], dtype=torch.int64, device=orders.device),
            torch.ones(orders.shape[0], dtype=torch.bool, device=orders.device),
        )
    destinations = _destination_order(orders.shape[1], orders.device)
    moves = torch.zeros(orders.shape[0], dtype=torch.int64, device=orders.device)
    for _ in range(max_passes):
        source, destination, delta = _best_insertion_moves(orders, counts, destinations)
        improving = delta < 0
        if not bool(improving.any().item()):
            break
        moved = _apply_insertions(orders, source, destination)
        orders = torch.where(improving.unsqueeze(1), moved, orders)
        moves.add_(improving.to(torch.int64))

    # A search that improves on its final allowed move may already be locally
    # optimal. Check once more so that such a row is not reported as a cap hit.
    _, _, remaining_delta = _best_insertion_moves(orders, counts, destinations)
    converged = remaining_delta >= 0
    return orders, _kemeny_objectives(orders, counts), moves, converged


def _kemeny_ranks(
    ballots: NDArray[np.int64],
    ballots_device: Any,
    borda_ranking: NDArray[np.int64],
    schulze_ranking: NDArray[np.int64] | None,
    *,
    seeds: Sequence[int],
    n_starts: int,
    max_passes: int,
    workspace_bytes: int,
) -> tuple[NDArray[np.int64], Mapping[str, int | float]]:
    import torch

    batch, _, patches = ballots.shape
    initial: list[NDArray[np.int64]] = []
    sample_ids: list[int] = []
    offsets = [0]
    for sample in range(batch):
        starts = _initial_orders(
            ballots[sample],
            borda_ranking[sample],
            None if schulze_ranking is None else schulze_ranking[sample],
            n_starts=n_starts,
            seed=int(seeds[sample]),
        )
        initial.append(starts)
        sample_ids.extend([sample] * len(starts))
        offsets.append(offsets[-1] + len(starts))
    initial_orders = np.concatenate(initial, axis=0)
    instance_samples = np.asarray(sample_ids, dtype=np.int64)
    final_orders = np.empty_like(initial_orders)
    initial_objectives = np.empty(len(initial_orders), dtype=np.int64)
    objectives = np.empty(len(initial_orders), dtype=np.int64)
    move_counts = np.empty(len(initial_orders), dtype=np.int64)
    converged = np.empty(len(initial_orders), dtype=np.bool_)
    capacity = _matrix_batch_capacity(patches, workspace_bytes)
    for start in range(0, len(initial_orders), capacity):
        stop = min(len(initial_orders), start + capacity)
        selected_samples = instance_samples[start:stop]
        unique_samples, inverse = np.unique(selected_samples, return_inverse=True)
        unique_tensor = torch.from_numpy(unique_samples).to(device=ballots_device.device)
        counts = _pairwise_counts(ballots_device.index_select(0, unique_tensor))
        inverse_tensor = torch.from_numpy(inverse).to(device=ballots_device.device)
        instance_counts = counts.index_select(0, inverse_tensor)
        orders = torch.from_numpy(initial_orders[start:stop]).to(device=ballots_device.device)
        initial_values = _kemeny_objectives(orders, instance_counts)
        searched, values, moves, chunk_converged = _search_kemeny_orders(
            orders,
            instance_counts,
            max_passes=max_passes,
        )
        final_orders[start:stop] = searched.cpu().numpy()
        initial_objectives[start:stop] = initial_values.cpu().numpy()
        objectives[start:stop] = values.cpu().numpy()
        move_counts[start:stop] = moves.cpu().numpy()
        converged[start:stop] = chunk_converged.cpu().numpy()

    secondary_scores = np.mean((patches - 1) - ballots, axis=1, dtype=np.float64)
    output = np.empty((batch, patches), dtype=np.int64)
    selected_indices = np.empty(batch, dtype=np.int64)
    positions = np.arange(patches, dtype=np.int64)
    for sample in range(batch):
        first, last = offsets[sample], offsets[sample + 1]
        best_objective = int(objectives[first:last].min())
        candidates = [
            index for index in range(first, last) if int(objectives[index]) == best_objective
        ]

        sample_scores = secondary_scores[sample]
        secondary_keys = {
            index: tuple((-float(sample_scores[item]), int(item)) for item in final_orders[index])
            for index in candidates
        }
        best = min(candidates, key=secondary_keys.__getitem__)
        selected_indices[sample] = best
        output[sample, final_orders[best]] = positions

    borda_indices = np.asarray(offsets[:-1], dtype=np.int64)
    borda_objectives = initial_objectives[borda_indices]
    selected_objectives = objectives[selected_indices]
    if np.any(selected_objectives > borda_objectives):
        raise RuntimeError("Kemeny local search returned an objective worse than Borda")
    converged_count = int(np.count_nonzero(converged))
    search_instances = int(len(initial_orders))
    statistics: Mapping[str, int | float] = {
        "sample_count": int(batch),
        "search_instance_count": search_instances,
        "starts_requested": int(n_starts),
        "max_passes": int(max_passes),
        "total_moves": int(np.sum(move_counts, dtype=np.int64)),
        "max_moves": int(np.max(move_counts, initial=0)),
        "cap_hit_count": search_instances - converged_count,
        "converged_count": converged_count,
        "converged_fraction": float(converged_count / search_instances),
        "borda_objective_sum": int(np.sum(borda_objectives, dtype=np.int64)),
        "final_objective_sum": int(np.sum(selected_objectives, dtype=np.int64)),
        "objective_improvement_sum": int(
            np.sum(borda_objectives - selected_objectives, dtype=np.int64)
        ),
    }
    return output, statistics


def aggregate_rankings_torch(
    ballots: NDArray[np.int64],
    simple_scores: NDArray[np.floating[Any]] | None,
    *,
    requested: Collection[str],
    rrf_c: float,
    kemeny_starts: int,
    kemeny_max_passes: int,
    seeds: Sequence[int],
    device: str | Any,
    workspace_bytes: int,
) -> AggregationResult:
    """Aggregate a shard on Torch while preserving the NumPy rule semantics."""

    import torch

    values = np.asarray(ballots)
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("ballots must contain integer [samples,methods,patches] ranks")
    if any(size <= 0 for size in values.shape):
        raise ValueError("ballots dimensions must be positive")
    samples, _, patches = values.shape
    max_objective = values.shape[1] * patches * (patches - 1) // 2
    if max_objective > np.iinfo(np.int32).max:
        raise ValueError("ballot profile exceeds the exact int32 aggregation range")
    expected = np.arange(patches, dtype=np.int64)
    canonical = values.astype(np.int64, copy=False)
    if not np.all(np.sort(canonical, axis=2) == expected):
        raise ValueError("every ballot must be a strict zero-based permutation")
    if len(seeds) != samples:
        raise ValueError("seeds must contain one deterministic seed per sample")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive")
    if not np.isfinite(rrf_c) or rrf_c <= 0:
        raise ValueError("rrf_c must be finite and positive")
    if kemeny_starts <= 0 or kemeny_max_passes <= 0:
        raise ValueError("Kemeny budgets must be positive")
    known = {"SimpleAvg", "Borda", "RRF", "Kemeny", "Schulze"}
    unknown = set(requested) - known
    if unknown:
        raise ValueError(f"unknown aggregation rules: {sorted(unknown)}")

    target = torch.device(device)
    ballot_tensor = torch.from_numpy(np.ascontiguousarray(canonical)).to(device=target)
    borda_scores = ((patches - 1) - ballot_tensor).sum(dim=1, dtype=torch.int64)
    borda_ranking = _scores_to_ranks(borda_scores)
    schulze_ranking = None
    if "Schulze" in requested or ("Kemeny" in requested and kemeny_starts > 1):
        schulze_ranking = _schulze_ranks(ballot_tensor, workspace_bytes=workspace_bytes)

    output: dict[str, NDArray[np.int64]] = {}
    statistics: dict[str, Mapping[str, int | float]] = {}
    if "SimpleAvg" in requested:
        if simple_scores is None:
            raise RuntimeError("SimpleAvg was requested without attribution score maps")
        scores = np.asarray(simple_scores)
        if scores.shape[0] != samples or scores.reshape(samples, -1).shape[1] != patches:
            raise ValueError("simple_scores must align with ballot samples and patches")
        if not np.all(np.isfinite(scores)):
            raise ValueError("simple_scores contain NaN or infinite values")
        score_tensor = torch.from_numpy(np.ascontiguousarray(scores.reshape(samples, patches))).to(
            device=target
        )
        output["simpleavg"] = _scores_to_ranks(score_tensor).cpu().numpy()
    if "Borda" in requested:
        output["borda"] = borda_ranking.cpu().numpy()
    if "RRF" in requested:
        rrf_scores = torch.zeros(
            (samples, patches),
            dtype=torch.float64,
            device=target,
        )
        for method in range(ballot_tensor.shape[1]):
            rrf_scores.add_(
                torch.reciprocal(ballot_tensor[:, method].to(torch.float64) + rrf_c + 1.0)
            )
        rrf_scores.div_(ballot_tensor.shape[1])
        output["rrf"] = _scores_to_ranks(rrf_scores).cpu().numpy()
    if "Kemeny" in requested:
        output["kemeny"], statistics["kemeny"] = _kemeny_ranks(
            canonical,
            ballot_tensor,
            borda_ranking.cpu().numpy(),
            None if schulze_ranking is None else schulze_ranking.cpu().numpy(),
            seeds=seeds,
            n_starts=kemeny_starts,
            max_passes=kemeny_max_passes,
            workspace_bytes=workspace_bytes,
        )
    if "Schulze" in requested:
        assert schulze_ranking is not None
        output["schulze"] = schulze_ranking.cpu().numpy()
    return AggregationResult(output, statistics)


__all__ = ["AggregationResult", "aggregate_rankings_torch"]
