"""Deterministic explanation and rank aggregation algorithms."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rankings import (
    canonicalize_ranking,
    canonicalize_rankings,
    order_to_ranking,
    pairwise_preference_counts,
    ranking_to_order,
    scores_to_ranking,
    scores_to_ranking_result,
)

Normalization = Literal["minmax", "max", "l1", "zscore", "none"]
Neighborhood = Literal["adjacent", "insertion"]


def _output_base(ranking: NDArray[np.int64], output_base: int) -> NDArray[np.int64]:
    if output_base not in (0, 1):
        raise ValueError("output_base must be 0 or 1")
    return ranking + output_base


def borda_scores(rankings: ArrayLike, *, index_base: int = 0) -> NDArray[np.float64]:
    """Return mean Borda scores; larger scores are better."""

    ranks = canonicalize_rankings(rankings, index_base=index_base)
    n_items = ranks.shape[1]
    return np.mean((n_items - 1) - ranks, axis=0, dtype=np.float64)


@dataclass(frozen=True)
class ScoreAggregationResult:
    ranking: NDArray[np.int64]
    scores: NDArray[np.float64]
    tie_rate: float
    tied_pairs: int
    total_pairs: int
    tie_break_policy: str


def borda_result(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> ScoreAggregationResult:
    scores = borda_scores(rankings, index_base=index_base)
    result = scores_to_ranking_result(
        scores, tie_break=tie_break, output_base=output_base
    )
    return ScoreAggregationResult(
        ranking=result.ranking,
        scores=scores,
        tie_rate=result.tie_rate,
        tied_pairs=result.tied_pairs,
        total_pairs=result.total_pairs,
        tie_break_policy=result.tie_break_policy,
    )


def borda(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Aggregate strict rankings with Borda count.

    Equal aggregate scores are resolved by ascending item identifier.
    """

    return borda_result(
        rankings,
        index_base=index_base,
        output_base=output_base,
        tie_break=tie_break,
    ).ranking


def rrf_scores(
    rankings: ArrayLike, *, c: float = 60.0, index_base: int = 0
) -> NDArray[np.float64]:
    """Return Reciprocal Rank Fusion scores using paper ranks ``1, ..., n``."""

    if not np.isfinite(c) or c <= 0:
        raise ValueError("RRF constant c must be finite and positive")
    ranks = canonicalize_rankings(rankings, index_base=index_base)
    return np.mean(1.0 / (c + ranks + 1.0), axis=0, dtype=np.float64)


def rrf_result(
    rankings: ArrayLike,
    *,
    c: float = 60.0,
    index_base: int = 0,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> ScoreAggregationResult:
    scores = rrf_scores(rankings, c=c, index_base=index_base)
    result = scores_to_ranking_result(
        scores, tie_break=tie_break, output_base=output_base
    )
    return ScoreAggregationResult(
        ranking=result.ranking,
        scores=scores,
        tie_rate=result.tie_rate,
        tied_pairs=result.tied_pairs,
        total_pairs=result.total_pairs,
        tie_break_policy=result.tie_break_policy,
    )


def rrf(
    rankings: ArrayLike,
    *,
    c: float = 60.0,
    index_base: int = 0,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Aggregate strict rankings with Reciprocal Rank Fusion."""

    return rrf_result(
        rankings,
        c=c,
        index_base=index_base,
        output_base=output_base,
        tie_break=tie_break,
    ).ranking


def _normalize_rows(values: NDArray[np.float64], method: Normalization) -> NDArray[np.float64]:
    if method == "none":
        return values.copy()
    if method == "minmax":
        minima = np.min(values, axis=1, keepdims=True)
        spans = np.max(values, axis=1, keepdims=True) - minima
        return np.divide(
            values - minima,
            spans,
            out=np.zeros_like(values),
            where=spans > 0,
        )
    if method == "max":
        denominators = np.max(np.abs(values), axis=1, keepdims=True)
        return np.divide(
            values,
            denominators,
            out=np.zeros_like(values),
            where=denominators > 0,
        )
    if method == "l1":
        denominators = np.sum(np.abs(values), axis=1, keepdims=True)
        return np.divide(
            values,
            denominators,
            out=np.zeros_like(values),
            where=denominators > 0,
        )
    if method == "zscore":
        means = np.mean(values, axis=1, keepdims=True)
        standard_deviations = np.std(values, axis=1, keepdims=True)
        return np.divide(
            values - means,
            standard_deviations,
            out=np.zeros_like(values),
            where=standard_deviations > 0,
        )
    raise ValueError(f"unknown normalization {method!r}")


def simple_average(
    score_maps: ArrayLike,
    *,
    normalization: Normalization = "minmax",
    absolute: bool = True,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Mean normalized score maps over the leading explainer axis.

    This is the intentionally direct SimpleAvg baseline.  Inputs have shape
    ``(n_explainers, ...)``.  Absolute attribution magnitude is used by
    default, each map is normalized independently, and then maps are averaged.
    Constant maps normalize to zero because they carry no ordering signal.
    """

    raw = np.asarray(score_maps)
    if raw.ndim < 2 or raw.shape[0] == 0 or np.prod(raw.shape[1:]) == 0:
        raise ValueError("score_maps must have shape (n_explainers, ...)")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise TypeError("score_maps must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("score_maps must contain real numeric values")
    maps = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(maps)):
        raise ValueError("score_maps contain NaN or infinite values")
    shape = maps.shape[1:]
    flat = maps.reshape(maps.shape[0], -1)
    if absolute:
        flat = np.abs(flat)
    normalized = _normalize_rows(flat, normalization)

    if weights is None:
        averaged = np.mean(normalized, axis=0)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (maps.shape[0],):
            raise ValueError(
                f"weights must have shape ({maps.shape[0]},), got {weight_array.shape}"
            )
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0):
            raise ValueError("weights must be finite and non-negative")
        if float(np.sum(weight_array)) <= 0:
            raise ValueError("at least one weight must be positive")
        averaged = np.average(normalized, axis=0, weights=weight_array)
    return averaged.reshape(shape)


def simple_average_ranking(
    score_maps: ArrayLike,
    *,
    normalization: Normalization = "minmax",
    absolute: bool = True,
    weights: ArrayLike | None = None,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Return the strict ranking induced by :func:`simple_average`."""

    mean_map = simple_average(
        score_maps,
        normalization=normalization,
        absolute=absolute,
        weights=weights,
    )
    return scores_to_ranking(
        mean_map.reshape(-1),
        higher_is_better=True,
        tie_break=tie_break,
        output_base=output_base,
    )


@dataclass(frozen=True)
class SimpleAttributionAggregationResult:
    """Protocol-locked SimpleAvg output before and after patch conversion."""

    mean_attribution: NDArray[np.float64]
    patch_scores: NDArray[np.float64]
    ranking: NDArray[np.int64]
    tie_rate: float


def simple_average_attributions(
    attributions: ArrayLike,
    *,
    patch_size: int = 14,
    tie_break: Sequence[int] | None = None,
    output_base: int = 0,
) -> SimpleAttributionAggregationResult:
    """Formal SimpleAvg: per-map z-score, mean attribution, then patch ranking.

    Input shape is ``(methods, channels, height, width)``.  Each signed spatial
    attribution map is z-scored over all of its entries before arithmetic
    averaging.  Only then is the consensus converted to patch importance using
    the manuscript's mean absolute channel/spatial reduction.  This deliberately
    differs from averaging already-absolute Phase 1 patch scores.
    """

    raw = np.asarray(attributions)
    if raw.ndim != 4 or any(size == 0 for size in raw.shape):
        raise ValueError("attributions must have shape (methods, channels, H, W)")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise TypeError("attributions must contain real numeric values")
    values = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("attributions contain NaN or infinite values")
    _, _, height, width = values.shape
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError("attribution H/W must be divisible by positive patch_size")
    flat = values.reshape(values.shape[0], -1)
    means = np.mean(flat, axis=1, keepdims=True)
    standard_deviations = np.std(flat, axis=1, keepdims=True)
    normalized = np.divide(
        flat - means,
        standard_deviations,
        out=np.zeros_like(flat),
        where=standard_deviations > 0,
    ).reshape(values.shape)
    mean_attribution = np.mean(normalized, axis=0)
    grid_height, grid_width = height // patch_size, width // patch_size
    absolute = np.abs(mean_attribution).mean(axis=0)
    patch_scores = absolute.reshape(
        grid_height, patch_size, grid_width, patch_size
    ).mean(axis=(1, 3))
    score_result = scores_to_ranking_result(
        patch_scores.reshape(-1),
        tie_break=tie_break,
        output_base=output_base,
    )
    return SimpleAttributionAggregationResult(
        mean_attribution=mean_attribution,
        patch_scores=patch_scores,
        ranking=score_result.ranking,
        tie_rate=score_result.tie_rate,
    )


@dataclass(frozen=True)
class SchulzeResult:
    ranking: NDArray[np.int64]
    pairwise_counts: NDArray[np.int64]
    strongest_paths: NDArray[np.int64]
    secondary_borda_scores: NDArray[np.float64]
    tie_rate: float


def schulze_result(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> SchulzeResult:
    """Run the manuscript's winning-votes Schulze rule deterministically.

    The strict strongest-path relation is extended to a total order with a
    deterministic topological sort.  Among currently unconstrained tied items,
    the smaller ``tie_break`` identifier (item id by default) is selected.
    """

    canonical = canonicalize_rankings(rankings, index_base=index_base)
    counts = pairwise_preference_counts(canonical)
    n_items = counts.shape[0]
    direct = np.where(counts > counts.T, counts, 0).astype(np.int64)
    np.fill_diagonal(direct, 0)
    paths = direct.copy()
    for intermediate in range(n_items):
        through = np.minimum(
            paths[:, intermediate, None], paths[intermediate, None, :]
        )
        paths = np.maximum(paths, through)
    np.fill_diagonal(paths, 0)

    relation = paths > paths.T
    secondary_scores = borda_scores(canonical)
    if tie_break is None:
        keys = np.arange(n_items, dtype=np.int64)
    else:
        keys = np.asarray(tie_break)
        if keys.shape != (n_items,) or not np.issubdtype(keys.dtype, np.integer):
            raise ValueError("tie_break must contain one integer id per item")
        keys = keys.astype(np.int64, copy=False)
        if np.unique(keys).size != n_items:
            raise ValueError("tie_break identifiers must be unique")

    indegree = np.sum(relation, axis=0, dtype=np.int64)
    available: list[tuple[float, int, int]] = [
        (-float(secondary_scores[item]), int(keys[item]), item)
        for item in range(n_items)
        if indegree[item] == 0
    ]
    heapq.heapify(available)
    order: list[int] = []
    while available:
        _, _, item = heapq.heappop(available)
        order.append(item)
        for successor in np.flatnonzero(relation[item]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    available,
                    (
                        -float(secondary_scores[successor]),
                        int(keys[successor]),
                        int(successor),
                    ),
                )
    if len(order) != n_items:
        raise RuntimeError("Schulze strongest-path relation unexpectedly contains a cycle")

    ranking = order_to_ranking(np.asarray(order, dtype=np.int64))
    tied_pairs = int(
        np.count_nonzero(np.triu(paths == paths.T, k=1))
    )
    total_pairs = n_items * (n_items - 1) // 2
    return SchulzeResult(
        ranking=_output_base(ranking, output_base),
        pairwise_counts=counts,
        strongest_paths=paths,
        secondary_borda_scores=secondary_scores,
        tie_rate=0.0 if total_pairs == 0 else tied_pairs / total_pairs,
    )


def schulze(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    output_base: int = 0,
    tie_break: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Return the deterministic Schulze consensus rank vector."""

    return schulze_result(
        rankings,
        index_base=index_base,
        output_base=output_base,
        tie_break=tie_break,
    ).ranking


def _objective_from_order(
    order: NDArray[np.int64], counts: NDArray[np.int64]
) -> int:
    objective = 0
    for position, first in enumerate(order[:-1]):
        after = order[position + 1 :]
        objective += int(np.sum(counts[after, first], dtype=np.int64))
    return objective


def kemeny_objective(
    candidate: ArrayLike,
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    candidate_base: int | None = None,
) -> int:
    """Total Kendall disagreement of a candidate with all ballots."""

    if candidate_base is None:
        candidate_base = index_base
    canonical = canonicalize_rankings(rankings, index_base=index_base)
    candidate_rank = canonicalize_ranking(candidate, index_base=candidate_base)
    if candidate_rank.size != canonical.shape[1]:
        raise ValueError("candidate and ballots rank different numbers of items")
    return _objective_from_order(ranking_to_order(candidate_rank), pairwise_preference_counts(canonical))


@dataclass(frozen=True)
class KemenyStartTrace:
    start_index: int
    initialization: str
    initial_objective: int
    final_objective: int
    moves: int
    converged: bool
    objective_path: tuple[int, ...]


@dataclass(frozen=True)
class KemenyResult:
    ranking: NDArray[np.int64]
    objective: int
    trace: tuple[KemenyStartTrace, ...]
    solver: str = "multi_start_local_search"
    neighborhood: str = "insertion"
    pairwise_tie_rate: float = 0.0
    tie_break_policy: str = "borda_then_stable_patch_index"


def _best_local_move(
    order: NDArray[np.int64],
    counts: NDArray[np.int64],
    neighborhood: Neighborhood,
) -> tuple[int, int, int] | None:
    n_items = order.size
    if n_items < 2:
        return None
    best_delta = 0
    best_move: tuple[int, int] | None = None
    if neighborhood == "adjacent":
        first = order[:-1]
        second = order[1:]
        deltas = counts[first, second] - counts[second, first]
        position = int(np.argmin(deltas))
        if int(deltas[position]) < 0:
            best_delta = int(deltas[position])
            best_move = (position, position + 1)
    elif neighborhood == "insertion":
        for source in range(n_items):
            item = order[source]
            after = order[source + 1 :]
            if after.size:
                forward = np.cumsum(
                    counts[item, after] - counts[after, item], dtype=np.int64
                )
                offset = int(np.argmin(forward))
                if int(forward[offset]) < best_delta:
                    best_delta = int(forward[offset])
                    best_move = (source, source + offset + 1)
            before_reversed = order[:source][::-1]
            if before_reversed.size:
                backward = np.cumsum(
                    counts[before_reversed, item] - counts[item, before_reversed],
                    dtype=np.int64,
                )
                offset = int(np.argmin(backward))
                if int(backward[offset]) < best_delta:
                    best_delta = int(backward[offset])
                    best_move = (source, source - offset - 1)
    else:
        raise ValueError(f"unknown Kemeny neighborhood {neighborhood!r}")
    if best_move is None:
        return None
    return best_move[0], best_move[1], best_delta


def _apply_insertion(
    order: NDArray[np.int64], source: int, destination: int
) -> NDArray[np.int64]:
    result = order.copy()
    item = result[source]
    if source < destination:
        result[source:destination] = result[source + 1 : destination + 1]
    elif source > destination:
        result[destination + 1 : source + 1] = result[destination:source]
    result[destination] = item
    return result


def _initial_orders(
    canonical: NDArray[np.int64],
    n_starts: int,
    seed: int,
    tie_break: Sequence[int] | None,
) -> list[tuple[NDArray[np.int64], str]]:
    starts: list[tuple[NDArray[np.int64], str]] = []
    seen: set[tuple[int, ...]] = set()

    def add(order: NDArray[np.int64], label: str) -> None:
        key = tuple(int(item) for item in order)
        if key not in seen and len(starts) < n_starts:
            seen.add(key)
            starts.append((order.copy(), label))

    add(ranking_to_order(borda(canonical, tie_break=tie_break)), "borda")
    add(ranking_to_order(schulze(canonical, tie_break=tie_break)), "schulze")
    for ballot_index, ballot in enumerate(canonical):
        add(ranking_to_order(ballot), f"ballot:{ballot_index}")
    rng = np.random.default_rng(seed)
    attempts = 0
    max_attempts = max(100, 100 * n_starts)
    while len(starts) < n_starts and attempts < max_attempts:
        add(rng.permutation(canonical.shape[1]), f"random:{attempts}")
        attempts += 1
    return starts


def kemeny_young(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    output_base: int = 0,
    n_starts: int = 16,
    max_passes: int = 10_000,
    neighborhood: Neighborhood = "insertion",
    seed: int = 0,
    tie_break: Sequence[int] | None = None,
) -> KemenyResult:
    """Approximate the true Kemeny objective with multi-start local search.

    This is a heuristic solver for the NP-hard Kemeny--Young problem, but it
    optimizes the actual total Kendall disagreement.  It never substitutes a
    Borda score for the Kemeny objective.  Every accepted local move and final
    objective is retained in the returned trace.
    """

    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    if neighborhood not in ("adjacent", "insertion"):
        raise ValueError(f"unknown Kemeny neighborhood {neighborhood!r}")
    canonical = canonicalize_rankings(rankings, index_base=index_base)
    counts = pairwise_preference_counts(canonical)
    n_items = canonical.shape[1]
    if tie_break is None:
        keys = np.arange(n_items, dtype=np.int64)
        tie_break_policy = "borda_then_stable_patch_index"
    else:
        keys = np.asarray(tie_break)
        if keys.shape != (n_items,) or not np.issubdtype(keys.dtype, np.integer):
            raise ValueError("tie_break must contain one integer id per item")
        keys = keys.astype(np.int64, copy=False)
        if np.unique(keys).size != n_items:
            raise ValueError("tie_break identifiers must be unique")
        tie_break_policy = "borda_then_deterministic_key"
    secondary_scores = borda_scores(canonical)
    traces: list[KemenyStartTrace] = []
    candidates: list[
        tuple[int, tuple[tuple[float, int], ...], NDArray[np.int64]]
    ] = []

    for start_index, (initial, label) in enumerate(
        _initial_orders(canonical, n_starts, seed, tie_break)
    ):
        order = initial
        objective = _objective_from_order(order, counts)
        initial_objective = objective
        path = [objective]
        moves = 0
        converged = False
        for _ in range(max_passes):
            move = _best_local_move(order, counts, neighborhood)
            if move is None:
                converged = True
                break
            source, destination, delta = move
            order = _apply_insertion(order, source, destination)
            objective += delta
            exact_objective = _objective_from_order(order, counts)
            if exact_objective != objective:
                raise RuntimeError("internal Kemeny move-delta invariant failed")
            path.append(objective)
            moves += 1
        traces.append(
            KemenyStartTrace(
                start_index=start_index,
                initialization=label,
                initial_objective=initial_objective,
                final_objective=objective,
                moves=moves,
                converged=converged,
                objective_path=tuple(path),
            )
        )
        secondary_key = tuple(
            (-float(secondary_scores[item]), int(keys[item])) for item in order
        )
        candidates.append((objective, secondary_key, order))

    objective, _, best_order = min(candidates, key=lambda entry: (entry[0], entry[1]))
    best_ranking = order_to_ranking(best_order)
    pairwise_ties = int(np.count_nonzero(np.triu(counts == counts.T, k=1)))
    total_pairs = n_items * (n_items - 1) // 2
    return KemenyResult(
        ranking=_output_base(best_ranking, output_base),
        objective=objective,
        trace=tuple(traces),
        neighborhood=neighborhood,
        pairwise_tie_rate=(
            0.0 if total_pairs == 0 else pairwise_ties / total_pairs
        ),
        tie_break_policy=tie_break_policy,
    )


def exact_kemeny_young(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    output_base: int = 0,
    max_items: int = 9,
    tie_break: Sequence[int] | None = None,
) -> KemenyResult:
    """Brute-force Kemeny oracle for tests and tiny diagnostic pilots."""

    canonical = canonicalize_rankings(rankings, index_base=index_base)
    n_items = canonical.shape[1]
    if n_items > max_items:
        raise ValueError(
            f"exact Kemeny is restricted to at most {max_items} items, got {n_items}"
        )
    counts = pairwise_preference_counts(canonical)
    if tie_break is None:
        keys = np.arange(n_items, dtype=np.int64)
        tie_break_policy = "borda_then_stable_patch_index"
    else:
        keys = np.asarray(tie_break)
        if keys.shape != (n_items,) or not np.issubdtype(keys.dtype, np.integer):
            raise ValueError("tie_break must contain one integer id per item")
        keys = keys.astype(np.int64, copy=False)
        if np.unique(keys).size != n_items:
            raise ValueError("tie_break identifiers must be unique")
        tie_break_policy = "borda_then_deterministic_key"
    secondary_scores = borda_scores(canonical)
    best_objective: int | None = None
    best_order: tuple[int, ...] | None = None
    best_secondary: tuple[tuple[float, int], ...] | None = None
    for candidate in permutations(range(n_items)):
        objective = _objective_from_order(np.asarray(candidate, dtype=np.int64), counts)
        secondary = tuple(
            (-float(secondary_scores[item]), int(keys[item])) for item in candidate
        )
        if best_objective is None or (objective, secondary) < (
            best_objective,
            best_secondary,
        ):
            best_objective = objective
            best_order = candidate
            best_secondary = secondary
    assert (
        best_objective is not None
        and best_order is not None
        and best_secondary is not None
    )
    ranking = order_to_ranking(np.asarray(best_order, dtype=np.int64))
    pairwise_ties = int(np.count_nonzero(np.triu(counts == counts.T, k=1)))
    total_pairs = n_items * (n_items - 1) // 2
    return KemenyResult(
        ranking=_output_base(ranking, output_base),
        objective=best_objective,
        trace=(),
        solver="exact_bruteforce",
        neighborhood="none",
        pairwise_tie_rate=(
            0.0 if total_pairs == 0 else pairwise_ties / total_pairs
        ),
        tie_break_policy=tie_break_policy,
    )


__all__ = [
    "KemenyResult",
    "KemenyStartTrace",
    "Normalization",
    "SchulzeResult",
    "ScoreAggregationResult",
    "SimpleAttributionAggregationResult",
    "borda",
    "borda_result",
    "borda_scores",
    "exact_kemeny_young",
    "kemeny_objective",
    "kemeny_young",
    "rrf",
    "rrf_result",
    "rrf_scores",
    "schulze",
    "schulze_result",
    "simple_average",
    "simple_average_attributions",
    "simple_average_ranking",
]
