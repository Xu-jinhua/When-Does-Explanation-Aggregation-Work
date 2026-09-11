"""Ranking representation and validation utilities.

The Phase 2 canonical representation is a *rank vector*: entry ``j`` is the
rank assigned to item ``j``.  Internally ranks are always zero based and form
an exact permutation of ``0, ..., n - 1``.  Public functions require callers
to state the input base; guessing between paper-style one-based ranks and
Phase-1-style zero-based ranks is deliberately avoided.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

TiePolicy = Literal["error", "item_id"]


def _validate_base(index_base: int) -> None:
    if index_base not in (0, 1):
        raise ValueError(f"index_base must be 0 or 1, got {index_base!r}")


def _numeric_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError(f"{name} must contain at least one item")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _tie_keys(n_items: int, tie_break: Sequence[int] | None) -> NDArray[np.int64]:
    if tie_break is None:
        return np.arange(n_items, dtype=np.int64)
    keys = np.asarray(tie_break)
    if keys.shape != (n_items,):
        raise ValueError(
            f"tie_break must have shape ({n_items},), got {keys.shape}"
        )
    if np.issubdtype(keys.dtype, np.bool_) or not np.issubdtype(
        keys.dtype, np.integer
    ):
        raise TypeError("tie_break must contain integer item identifiers")
    keys = keys.astype(np.int64, copy=False)
    if np.unique(keys).size != n_items:
        raise ValueError("tie_break identifiers must be unique")
    return keys


def order_to_ranking(order: ArrayLike, *, output_base: int = 0) -> NDArray[np.int64]:
    """Convert an ordered list of item identifiers to a rank vector."""

    _validate_base(output_base)
    raw = np.asarray(order)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("order must be a non-empty one-dimensional array")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError("order must contain integer item identifiers")
    order_i = raw.astype(np.int64, copy=False)
    expected = np.arange(order_i.size, dtype=np.int64)
    if not np.array_equal(np.sort(order_i), expected):
        raise ValueError("order must be a permutation of item ids 0, ..., n - 1")
    ranking = np.empty(order_i.size, dtype=np.int64)
    ranking[order_i] = expected
    return ranking + output_base


def ranking_to_order(ranking: ArrayLike, *, index_base: int = 0) -> NDArray[np.int64]:
    """Convert a strict rank vector to item identifiers in best-to-worst order."""

    canonical = canonicalize_ranking(ranking, index_base=index_base)
    return np.argsort(canonical, kind="stable").astype(np.int64, copy=False)


def canonicalize_ranking(
    ranking: ArrayLike,
    *,
    index_base: int = 0,
    tie_policy: TiePolicy = "error",
    tie_break: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Validate and return a zero-based strict rank vector.

    ``tie_policy="error"`` accepts only an exact permutation in the declared
    base.  ``tie_policy="item_id"`` treats the supplied values as ordinal rank
    keys and resolves equal keys by the deterministic ``tie_break`` item ids
    (the item index by default).  The latter policy is intended for importing
    explicitly tied external rankings, not for silently repairing corrupt
    permutation files.
    """

    _validate_base(index_base)
    if tie_policy not in ("error", "item_id"):
        raise ValueError(f"unknown tie_policy {tie_policy!r}")
    values = _numeric_vector(ranking, name="ranking")
    n_items = values.size

    if tie_policy == "error":
        if not np.all(values == np.floor(values)):
            raise ValueError("a strict ranking must contain integer ranks")
        integer = values.astype(np.int64)
        expected = np.arange(index_base, index_base + n_items, dtype=np.int64)
        if not np.array_equal(np.sort(integer), expected):
            raise ValueError(
                "ranking must be an exact permutation of "
                f"{index_base}, ..., {index_base + n_items - 1}; "
                "declare tie_policy='item_id' only for intentional ties"
            )
        return integer - index_base

    keys = _tie_keys(n_items, tie_break)
    order = np.lexsort((keys, values))
    return order_to_ranking(order)


def canonicalize_rankings(
    rankings: ArrayLike,
    *,
    index_base: int = 0,
    tie_policy: TiePolicy = "error",
    tie_break: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Validate a ``(n_rankings, n_items)`` matrix of rank vectors."""

    raw = np.asarray(rankings)
    if raw.ndim != 2:
        raise ValueError(
            "rankings must have shape (n_rankings, n_items), "
            f"got {raw.shape}"
        )
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("rankings must contain at least one ranking and one item")
    return np.stack(
        [
            canonicalize_ranking(
                row,
                index_base=index_base,
                tie_policy=tie_policy,
                tie_break=tie_break,
            )
            for row in raw
        ],
        axis=0,
    )


def scores_to_ranking(
    scores: ArrayLike,
    *,
    higher_is_better: bool = True,
    tie_policy: TiePolicy = "item_id",
    tie_break: Sequence[int] | None = None,
    output_base: int = 0,
) -> NDArray[np.int64]:
    """Rank scores with an explicit deterministic tie policy."""

    return scores_to_ranking_result(
        scores,
        higher_is_better=higher_is_better,
        tie_policy=tie_policy,
        tie_break=tie_break,
        output_base=output_base,
    ).ranking


@dataclass(frozen=True)
class ScoreRankingResult:
    ranking: NDArray[np.int64]
    tie_rate: float
    tied_pairs: int
    total_pairs: int
    tie_break_policy: str


def score_tie_rate(scores: ArrayLike) -> tuple[float, int, int]:
    """Return the fraction of unordered item pairs with exactly equal scores."""

    values = _numeric_vector(scores, name="scores")
    _, counts = np.unique(values, return_counts=True)
    tied_pairs = int(np.sum(counts * (counts - 1) // 2, dtype=np.int64))
    total_pairs = values.size * (values.size - 1) // 2
    rate = 0.0 if total_pairs == 0 else tied_pairs / total_pairs
    return float(rate), tied_pairs, total_pairs


def sample_hash_tie_break(n_items: int, sample_hash: str | bytes) -> NDArray[np.int64]:
    """Create a deterministic pseudo-random item priority from a sample hash.

    This optional policy avoids consistently favoring low patch indices (and
    therefore top-left image locations) when attribution maps contain many
    equal zeros.  It is deterministic across machines and does not use Python's
    process-randomized ``hash()``.
    """

    if not isinstance(n_items, (int, np.integer)) or n_items <= 0:
        raise ValueError("n_items must be a positive integer")
    if isinstance(sample_hash, str):
        prefix = sample_hash.encode("utf-8")
    elif isinstance(sample_hash, bytes):
        prefix = sample_hash
    else:
        raise TypeError("sample_hash must be str or bytes")
    digests = [
        hashlib.sha256(prefix + b":" + str(item).encode("ascii")).digest()
        for item in range(int(n_items))
    ]
    order = sorted(range(int(n_items)), key=lambda item: (digests[item], item))
    priorities = np.empty(int(n_items), dtype=np.int64)
    priorities[np.asarray(order, dtype=np.int64)] = np.arange(
        int(n_items), dtype=np.int64
    )
    return priorities


def scores_to_ranking_result(
    scores: ArrayLike,
    *,
    higher_is_better: bool = True,
    tie_policy: TiePolicy = "item_id",
    tie_break: Sequence[int] | None = None,
    output_base: int = 0,
) -> ScoreRankingResult:
    """Rank scores and retain diagnostics about exact score ties."""

    _validate_base(output_base)
    values = _numeric_vector(scores, name="scores")
    keys = _tie_keys(values.size, tie_break)
    sort_values = -values if higher_is_better else values
    if tie_policy == "error" and np.unique(sort_values).size != sort_values.size:
        raise ValueError("scores contain ties and tie_policy='error'")
    if tie_policy not in ("error", "item_id"):
        raise ValueError(f"unknown tie_policy {tie_policy!r}")
    order = np.lexsort((keys, sort_values))
    tie_rate, tied_pairs, total_pairs = score_tie_rate(values)
    policy = "stable_patch_index" if tie_break is None else "deterministic_key"
    return ScoreRankingResult(
        ranking=order_to_ranking(order, output_base=output_base),
        tie_rate=tie_rate,
        tied_pairs=tied_pairs,
        total_pairs=total_pairs,
        tie_break_policy=policy,
    )


def kendall_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    index_base: int = 0,
) -> int:
    """Return the number of discordant item pairs between strict rankings."""

    a = canonicalize_ranking(first, index_base=index_base)
    b = canonicalize_ranking(second, index_base=index_base)
    if a.shape != b.shape:
        raise ValueError(f"ranking shapes differ: {a.shape} and {b.shape}")
    n_items = a.size
    total = 0
    for item in range(n_items - 1):
        total += int(
            np.count_nonzero(
                (a[item] - a[item + 1 :])
                * (b[item] - b[item + 1 :])
                < 0
            )
        )
    return total


def spearman_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    index_base: int = 0,
) -> int:
    """Return squared Spearman rank distance between strict rankings."""

    a = canonicalize_ranking(first, index_base=index_base)
    b = canonicalize_ranking(second, index_base=index_base)
    if a.shape != b.shape:
        raise ValueError(f"ranking shapes differ: {a.shape} and {b.shape}")
    delta = a - b
    return int(np.dot(delta, delta))


def pairwise_preference_counts(rankings: ArrayLike, *, index_base: int = 0) -> NDArray[np.int64]:
    """Count ballots preferring row item ``a`` over column item ``b``."""

    canonical = canonicalize_rankings(rankings, index_base=index_base)
    counts = np.sum(
        canonical[:, :, None] < canonical[:, None, :], axis=0, dtype=np.int64
    )
    np.fill_diagonal(counts, 0)
    return counts


__all__ = [
    "TiePolicy",
    "ScoreRankingResult",
    "canonicalize_ranking",
    "canonicalize_rankings",
    "kendall_distance",
    "order_to_ranking",
    "pairwise_preference_counts",
    "ranking_to_order",
    "sample_hash_tie_break",
    "score_tie_rate",
    "scores_to_ranking",
    "scores_to_ranking_result",
    "spearman_distance",
]
