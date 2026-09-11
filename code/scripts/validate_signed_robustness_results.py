#!/usr/bin/env python3
"""Independently audit signed robustness summaries from their raw quality values."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

METRICS = ("F", "Fbar", "C", "Cbar")
MAXIMIZED = frozenset(("F", "C"))
ABSOLUTE_TOLERANCE = 1e-12
NON_RESULT_SUBTREES = frozenset(("standard_deviation",))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSON files or directories to audit recursively")
    return parser.parse_args()


def _json_paths(paths: Sequence[str]) -> tuple[Path, ...]:
    resolved = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            resolved.update(path.rglob("*.json"))
        elif path.is_file():
            resolved.add(path)
        else:
            raise FileNotFoundError(path)
    return tuple(sorted(resolved))


def _expected(metric: str, *, clean: float, perturbed: float) -> float:
    return perturbed - clean if metric in MAXIMIZED else clean - perturbed


def _is_metric_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(METRICS):
        return False
    try:
        numbers = tuple(float(value[metric]) for metric in METRICS)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(number) for number in numbers)


def _validate_result_row(value: Mapping[str, Any], *, context: str) -> dict[str, int]:
    quality = value.get("quality")
    perturbed = value.get("perturbed_quality")
    robustness = value.get("robustness")
    if not _is_metric_mapping(quality):
        return {"rows": 0, "values": 0, "positive": 0, "negative": 0, "zero": 0}
    if not isinstance(perturbed, Mapping) or not isinstance(robustness, Mapping):
        return {"rows": 0, "values": 0, "positive": 0, "negative": 0, "zero": 0}
    if _is_metric_mapping(perturbed) and _is_metric_mapping(robustness):
        conditions = (("perturbed", perturbed, robustness),)
    else:
        if not perturbed or set(perturbed) != set(robustness):
            raise ValueError(f"{context}: perturbation coverage is not aligned")
        conditions = tuple(
            (str(condition), perturbed[condition], robustness[condition]) for condition in perturbed
        )
    counts = {"rows": 1, "values": 0, "positive": 0, "negative": 0, "zero": 0}
    for condition, noisy_values, robustness_values in conditions:
        if not _is_metric_mapping(noisy_values) or not _is_metric_mapping(robustness_values):
            raise ValueError(f"{context}/{condition}: metric coverage is incomplete")
        for metric in METRICS:
            clean = float(quality[metric])
            noisy = float(noisy_values[metric])
            actual = float(robustness_values[metric])
            expected = _expected(metric, clean=clean, perturbed=noisy)
            if not all(math.isfinite(item) for item in (clean, noisy, actual)):
                raise ValueError(f"{context}/{condition}/{metric}: non-finite value")
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
                raise ValueError(
                    f"{context}/{condition}/{metric}: R={actual} but expected {expected}"
                )
            counts["values"] += 1
            counts["positive" if actual > 0.0 else "negative" if actual < 0.0 else "zero"] += 1
    return counts


def _validate_best_individual_sources(value: Mapping[str, Any], *, context: str) -> int:
    if str(value.get("method", "")).lower().replace(" ", "_") != "best_individual":
        return 0
    selected = value.get("selected_sources")
    if not isinstance(selected, Mapping):
        return 0
    if set(selected) == set(METRICS) and all(
        isinstance(selected[metric], str) and selected[metric] for metric in METRICS
    ):
        return 1
    clean = selected.get("quality")
    if not isinstance(clean, Mapping) or set(clean) != set(METRICS):
        raise ValueError(f"{context}: Best Individual clean sources are incomplete")
    for field in ("perturbed_quality", "robustness"):
        conditions = selected.get(field)
        if not isinstance(conditions, Mapping) or not conditions:
            raise ValueError(f"{context}: Best Individual {field} sources are incomplete")
        for condition, methods in conditions.items():
            if not isinstance(methods, Mapping) or set(methods) != set(METRICS):
                raise ValueError(f"{context}/{field}/{condition}: source coverage is incomplete")
            for metric in METRICS:
                if methods[metric] != clean[metric]:
                    raise ValueError(
                        f"{context}/{field}/{condition}/{metric}: source changed from clean"
                    )
    return 1


def _walk(value: Any, *, context: str, counts: dict[str, int]) -> None:
    if isinstance(value, Mapping):
        row_counts = _validate_result_row(value, context=context)
        for key, amount in row_counts.items():
            counts[key] += amount
        counts["best_individual_rows"] += _validate_best_individual_sources(value, context=context)
        directions = value.get("column_directions")
        if isinstance(directions, Mapping):
            for column, direction in directions.items():
                if str(column).startswith("R_"):
                    counts["direction_fields"] += 1
                    if direction != "max":
                        raise ValueError(
                            f"{context}/{column}: direction is {direction!r}, not 'max'"
                        )
        science = value.get("science")
        if isinstance(science, Mapping) and "robustness_direction" in science:
            counts["direction_fields"] += 1
            if science["robustness_direction"] != "max":
                raise ValueError(
                    f"{context}/science: robustness direction is "
                    f"{science['robustness_direction']!r}, not 'max'"
                )
        for key, child in value.items():
            if key in NON_RESULT_SUBTREES:
                counts["uncertainty_subtrees_skipped"] += 1
                continue
            _walk(child, context=f"{context}/{key}", counts=counts)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _walk(child, context=f"{context}/{index}", counts=counts)


def main() -> int:
    paths = _json_paths(_arguments().paths)
    counts = {
        "files": len(paths),
        "rows": 0,
        "values": 0,
        "positive": 0,
        "negative": 0,
        "zero": 0,
        "best_individual_rows": 0,
        "direction_fields": 0,
        "uncertainty_subtrees_skipped": 0,
    }
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        _walk(value, context=str(path), counts=counts)
    if counts["values"] == 0:
        raise ValueError("No signed robustness values were found")
    print(
        json.dumps(
            {
                "status": "valid",
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                **counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
