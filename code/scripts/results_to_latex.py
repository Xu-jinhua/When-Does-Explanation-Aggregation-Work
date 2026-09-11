#!/usr/bin/env python3
"""Convert released structured result files into dataset-organised LaTeX.

The result directories are intentionally listed below instead of discovered by
walking the whole workspace.  This keeps a release reproducible and makes the
manifest explicit about which historical snapshots were included.

Only the Python standard library is used.  This is a report/archive builder,
not an experiment runner; it never imports the project package or executes a
model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "latex"

# These are the result roots deliberately included in the public archive.
# Keeping the list here avoids accidentally publishing unrelated scheduler
# output from /mnt/data/jhu.
SOURCE_ROOTS = {
    "canonical_results": Path(
        "/mnt/data/jhu/xai-git-worktrees/full-matrix/code/results/simple"
    ),
    "canonical_live_results": Path(
        "/mnt/data/jhu/xai-full-matrix/github-complete-results-v1/results"
    ),
    # The compatibility export is kept beside the materialised result tree.
    # It contains real-checkpoint explainer and batch-size pilots, not Phase 2
    # quality metrics; those diagnostics are rendered in each dataset appendix.
    "full_matrix_compatibility": Path(
        "/mnt/data/jhu/xai-full-matrix/github-complete-results-v1"
    ),
    "signed_r_v2": Path("/mnt/data/jhu/xai-signed-r-v2-build"),
    "esann_archive": Path(
        "/mnt/data/jhu/xai-git-worktrees/full-matrix/OLD/experiment-archive/ESANN code/results"
    ),
    "paper_main": Path(
        "/mnt/data/jhu/xai-simple/paper-main-v1/scratch/summaries/table1"
    ),
    "naive_ablations": Path(
        "/mnt/data/jhu/xai-simple/paper-naive-ablations-v1/scratch/summaries"
    ),
    "noise_prefix_sweep": Path(
        "/mnt/data/jhu/xai-simple/paper-noise-prefix-sweep-v1/summaries"
    ),
    "random_subset": Path(
        "/mnt/data/jhu/xai-simple/paper-noise-random-subset-v2/summaries"
    ),
    "random_order": Path(
        "/mnt/data/jhu/xai-simple/paper-noise-random-order-anchored-v1/scratch/summaries"
    ),
    "mallows": Path(
        "/mnt/data/jhu/xai-simple/noise-model-boundary-v0/fidelity-anchored-topk-mallows-v1"
    ),
    "pathmnist_generalization": Path(
        "/mnt/data/jhu/xai-signed-r-v2-build/noise-generalization-pathmnist-densenet121"
    ),
    "relative_robustness": Path(
        "/mnt/data/jhu/xai-simple/paper-relative-robustness-v1"
    ),
}


DATASET_LABELS = {
    "bloodmnist": "BloodMNIST",
    "breastmnist": "BreastMNIST",
    "dermamnist": "DermaMNIST",
    "food101": "Food101",
    "imagenet": "ImageNet",
    "imagenet100": "ImageNet",
    "octmnist": "OCTMNIST",
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
    "pathmnist": "PathMNIST",
    "places365": "Places365",
    "pneumoniamnist": "PneumoniaMNIST",
    "retinamnist": "RetinaMNIST",
    "tissuemnist": "TissueMNIST",
}

MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit-b16": "ViT-B/16",
    "densenet121": "DenseNet-121",
    "efficientnet-b0": "EfficientNet-B0",
    "mobilenetv3-large": "MobileNetV3-Large",
    "deit-b16": "DeiT-B/16",
    "swin-b": "Swin-B",
}

@dataclass(frozen=True)
class SourceSpec:
    key: str
    root: Path
    patterns: tuple[str, ...]
    experiment: str
    default_dataset: str | None = None
    default_model: str | None = None
    historical_default: bool = True
    priority: int = 0


@dataclass
class TableRecord:
    dataset: str
    model: str
    experiment: str
    source_key: str
    source_path: str
    relative_source: str
    version: str
    historical: bool
    rows: list[dict[str, str]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    empty: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_path: str | None = None
    content_digest: str = ""
    duplicate_sources: list[dict[str, str]] = field(default_factory=list)


def source_specs() -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            "canonical_results",
            SOURCE_ROOTS["canonical_results"],
            ("**/*.csv",),
            "released result",
            historical_default=False,
            priority=100,
        ),
        SourceSpec(
            "canonical_live_results",
            SOURCE_ROOTS["canonical_live_results"],
            ("**/*.csv",),
            "released result",
            historical_default=True,
            priority=105,
        ),
        SourceSpec(
            "signed_r_v2",
            SOURCE_ROOTS["signed_r_v2"],
            ("**/*.csv",),
            "signed-r-v2 result",
            historical_default=True,
            priority=80,
        ),
        SourceSpec(
            "esann_archive",
            SOURCE_ROOTS["esann_archive"],
            ("**/*.csv",),
            "ESANN archive",
            historical_default=True,
            priority=10,
        ),
        SourceSpec(
            "paper_main",
            SOURCE_ROOTS["paper_main"],
            ("**/*.csv",),
            "main comparison",
            historical_default=True,
            priority=70,
        ),
        SourceSpec(
            "naive_ablations",
            SOURCE_ROOTS["naive_ablations"],
            ("Table_*/*/summary.csv",),
            "NAIVE ablation",
            historical_default=True,
            priority=60,
        ),
        SourceSpec(
            "noise_prefix_sweep",
            SOURCE_ROOTS["noise_prefix_sweep"],
            ("**/*.csv",),
            "NOISE q-sweep",
            historical_default=True,
            priority=60,
        ),
        SourceSpec(
            "random_subset",
            SOURCE_ROOTS["random_subset"],
            ("*.csv",),
            "random subset archive",
            "dermamnist",
            "dermamnist-resnet18",
            True,
            40,
        ),
        SourceSpec(
            "random_order",
            SOURCE_ROOTS["random_order"],
            ("*.csv",),
            "random order archive",
            "dermamnist",
            "dermamnist-resnet18",
            True,
            40,
        ),
        SourceSpec(
            "mallows",
            SOURCE_ROOTS["mallows"],
            ("*.csv",),
            "Fidelity-Anchored Top-k Mallows diagnostics",
            historical_default=True,
            priority=50,
        ),
        SourceSpec(
            "pathmnist_generalization",
            SOURCE_ROOTS["pathmnist_generalization"],
            ("**/*.csv",),
            "NOISE generalization",
            historical_default=True,
            priority=50,
        ),
    )


def clean_token(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("_", "-")
    return value


def normalise_dataset(value: str | None) -> str:
    token = clean_token(value or "")
    compact = re.sub(r"[-_ ]+", "", token)
    if compact == "imagenet100":
        return "imagenet"
    return token


def normalise_model(value: str | None) -> str:
    token = clean_token(value or "")
    token = re.sub(r"^(imagenet100|imagenet)-", "", token)
    token = re.sub(
        r"^(bloodmnist|breastmnist|dermamnist|food101|imagenet100|imagenet|places365|"
        r"octmnist|organamnist|organcmnist|organsmnist|pathmnist|pneumoniamnist|"
        r"retinamnist|tissuemnist)-",
        "",
        token,
    )
    aliases = {
        "resnet-18": "resnet18",
        "resnet18": "resnet18",
        "resnet-50": "resnet50",
        "resnet50": "resnet50",
        "densenet-121": "densenet121",
        "densenet121": "densenet121",
        "efficientnet-b0": "efficientnet-b0",
        "efficientnet-0": "efficientnet-b0",
        "mobilenetv3-large": "mobilenetv3-large",
        "mobilenet-v3-large": "mobilenetv3-large",
        "deit-b16": "deit-b16",
        "deit-b-16": "deit-b16",
        "swin-b": "swin-b",
        "vitb16": "vit-b16",
        "vit-b-16": "vit-b16",
        "vit-b16": "vit-b16",
        "vit-base-patch16-224": "vit-b16",
        "vit-base-patch-16-224": "vit-b16",
    }
    return aliases.get(token, token)


def split_cell(value: str | None) -> tuple[str, str]:
    token = clean_token(value or "")
    if "--" not in token:
        return "", ""
    dataset, model = token.split("--", 1)
    return normalise_dataset(dataset), normalise_model(model)


def display_dataset(value: str) -> str:
    return DATASET_LABELS.get(normalise_dataset(value), value or "Unknown")


def display_model(value: str) -> str:
    token = normalise_model(value)
    return MODEL_LABELS.get(token, token.replace("-", " ").title() or "Unknown")


def safe_name(value: str) -> str:
    token = clean_token(value)
    token = token.replace("/", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", token).strip("-") or "unknown"


def latex_escape(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("\n", " ")


def public_name(value: str) -> str:
    """Remove internal historical labels from public table text."""
    text = str(value)
    text = re.sub(
        r"(?i)full[-_ ]*(?:\d+[-_ ]*)?(?:cell[-_ ]*)?matrix",
        "released scope",
        text,
    )
    text = re.sub(r"(?i)matched[- ]?naive", "NAIVE", text)
    text = re.sub(r"(?i)oracle[-_ ]*noise", "NOISE", text)
    text = re.sub(r"(?i)oracle[-_ ]*clean[-_ ]*f(?:[-_ ]*(?:kendall|spearman))?", "NOISE", text)
    text = re.sub(r"(?i)compact[-_ ]*ind", "IND", text)
    text = re.sub(r"(?i)formal[-_ ]*noise", "NOISE", text)
    text = re.sub(r"(?i)q11[-_ ]*naive", "NAIVE q=11", text)
    text = re.sub(r"(?i)imagenet100", "ImageNet", text)
    text = re.sub(r"(?i)oracle[_-]?q", "NOISE q", text)
    text = re.sub(r"(?i)oracle[_-]?actual", "NOISE actual", text)
    # Treat underscores and hyphens as separators too, so identifiers such as
    # ``selected_q_matches_oracle`` receive the same public terminology.
    text = re.sub(r"(?i)(?<![A-Za-z])oracle(?![A-Za-z])", "NOISE", text)
    return text


def json_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return "" if value is None else str(value)


def format_cell(value: Any) -> str:
    text = json_cell(value)
    if not text:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return public_name(text)
    if number != number:
        return ""
    if abs(number) >= 1000 or (0 < abs(number) < 0.0001):
        return f"{number:.5g}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def infer_pair(row: dict[str, Any], metadata: dict[str, Any] | None = None) -> tuple[str, str]:
    metadata = metadata or {}
    dataset = row.get("dataset") or metadata.get("dataset") or ""
    model = row.get("model") or metadata.get("model") or ""
    cell = row.get("cell") or metadata.get("cell") or ""
    cell_dataset, cell_model = split_cell(str(cell))
    dataset = normalise_dataset(str(dataset or cell_dataset))
    model = normalise_model(str(model or cell_model))
    return dataset, model


def filename_pair(path: Path) -> tuple[str, str]:
    """Infer a legacy dataset/model pair from a result filename or parent."""
    text = "-".join(part.lower() for part in path.parts)
    dataset = ""
    # Match the longer ImageNet token first so ImageNet100 is normalised once.
    for candidate in (
        "imagenet100",
        "imagenet",
        "bloodmnist",
        "breastmnist",
        "dermamnist",
        "food101",
        "octmnist",
        "organamnist",
        "organcmnist",
        "organsmnist",
        "pathmnist",
        "places365",
        "pneumoniamnist",
        "retinamnist",
        "tissuemnist",
    ):
        if candidate in text:
            dataset = normalise_dataset(candidate)
            break
    model = ""
    if re.search(r"(?:^|[-_])(resnet[-_]?18)(?:$|[-_])", text):
        model = "resnet18"
    elif re.search(r"(?:^|[-_])(resnet[-_]?50)(?:$|[-_])", text):
        model = "resnet50"
    elif re.search(r"(?:^|[-_])(densenet[-_]?121)(?:$|[-_])", text):
        model = "densenet121"
    elif re.search(r"(?:^|[-_])efficientnet[-_]?b?[-_]?0(?:$|[-_])", text):
        model = "efficientnet-b0"
    elif re.search(r"(?:^|[-_])mobilenet(?:[-_]?v3)?[-_]?large(?:$|[-_])", text):
        model = "mobilenetv3-large"
    elif re.search(r"(?:^|[-_])deit(?:[-_]?b)?[-_]?16(?:$|[-_])", text):
        model = "deit-b16"
    elif re.search(r"(?:^|[-_])swin[-_]?b(?:$|[-_])", text):
        model = "swin-b"
    elif re.search(r"(?:^|[-_])vit(?:[-_]?b)?[-_]?16(?:$|[-_])", text):
        model = "vit-b16"
    return dataset, model


def metadata_pairs(metadata: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Collect all dataset/model pairs exposed by a summary JSON payload."""
    if not metadata:
        return []
    pairs: set[tuple[str, str]] = set()
    direct = infer_pair(metadata)
    if all(direct):
        pairs.add(direct)
    cells = metadata.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict):
                pair = infer_pair(cell, cell)
                if all(pair):
                    pairs.add(pair)
    analysis = metadata.get("analysis")
    if isinstance(analysis, dict):
        by_dataset_model = analysis.get("by_dataset_model")
        if isinstance(by_dataset_model, list):
            for cell in by_dataset_model:
                if isinstance(cell, dict):
                    pair = infer_pair(cell, cell)
                    if all(pair):
                        pairs.add(pair)
    # Some aggregate summaries keep the cell contract inside a large list
    # (for example quality-retention comparisons) rather than at the top
    # level.  Walk only JSON containers and collect explicit pairs.
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            pair = infer_pair(value, value)
            if all(pair):
                pairs.add(pair)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                if isinstance(nested, (dict, list)):
                    visit(nested)

    for key in ("comparisons", "quality_retention", "accuracies", "boundaries"):
        if key in metadata:
            visit(metadata[key])
    return sorted(pairs)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return rows, columns


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def metadata_for(path: Path, source_key: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    summary = path.with_name("summary.json")
    if summary.exists():
        metadata.update(load_json(summary))
    if source_key == "naive_ablations":
        # The ablation summary is the authoritative dataset/model contract.
        metadata.update(load_json(path.parent / "summary.json"))
    if source_key == "pathmnist_generalization":
        metadata.setdefault("dataset", "pathmnist")
        metadata.setdefault("model", "pathmnist-densenet121")
    return metadata


def version_for(path: Path, source: SourceSpec) -> str:
    relative = path.relative_to(source.root).as_posix()
    parts = relative.split("/")
    # Hash directories are intentional version identifiers in all historical
    # result exports.  Keep the complete directory token in the manifest.
    for token in parts:
        if re.fullmatch(r"[0-9a-f]{12,}", token):
            return token
    return "unversioned"


def is_historical(path: Path, source: SourceSpec) -> bool:
    if source.key == "paper_main":
        return version_for(path, source) != "80c3cc8eb7b5e7ade3ca8a96e32440f03d94b645f2f20b76023615d2af73c583"
    if source.key in {"canonical_results", "canonical_live_results"}:
        # The named paper and relative-robustness exports are the current
        # release families.  Scope and generalisation snapshots remain useful
        # but are presented in the archival appendix.
        rel = path.relative_to(source.root).parts
        return not rel or rel[0] not in {
            "paper-main-v1",
            "paper-noise-prefix-sweep-v1",
            "paper-assumptions-compact-v1",
            "paper-relative-robustness-v1",
        }
    return source.historical_default


def experiment_for(source: SourceSpec, path: Path) -> str:
    rel = path.relative_to(source.root).as_posix().lower()
    if source.key in {"canonical_results", "canonical_live_results"}:
        # The live full-matrix export is rooted at ``results/`` and therefore
        # starts at ``scopes/<scope-id>/...``; the older worktree mirror kept
        # the additional ``github-complete-results-v1`` directory in the
        # relative path.  Normalize both layouts before classifying tables.
        if not rel.startswith("github-complete-results-v1/"):
            rel = f"github-complete-results-v1/{rel}"
        if "paper-relative-robustness" in rel:
            return "relative robustness"
        if "effective-robustness" in rel:
            return "effective robustness"
        if "paper-noise-prefix-sweep" in rel or "noise-generalization" in rel:
            if "q-sweep" in rel or "per_q" in rel:
                return "NOISE q-sweep"
            if "fidelity-anchored-topk-mallows" in rel:
                return "NOISE selector diagnostics"
            return "NOISE diagnostics"
        if "paper-assumptions" in rel:
            if "/ind-table-priority/" in f"/{rel}":
                return "IND"
            return "NOISE"
        if "paper-main" in rel:
            return "main comparison"
        if "github-complete-results" in rel:
            if "/ind/" in f"/{rel}":
                return "IND"
            if "noise-prefix" in rel:
                return "NOISE diagnostics"
            if "/base/" in f"/{rel}":
                return "NAIVE"
            return "released scope"
    if source.key in {"signed_r_v2", "esann_archive"}:
        if source.key == "esann_archive":
            return "ESANN archive"
        if "q-sweep" in rel:
            return "NOISE q-sweep"
        if "effective-robustness" in rel:
            return "effective robustness"
        if "noise" in rel:
            return "NOISE"
        return source.experiment
    if source.key == "paper_main":
        if "analysis" in path.name:
            return "main comparison analysis"
        if "oracle" in path.name.lower():
            return "NOISE stability archive"
    if source.key == "naive_ablations":
        table = next((part for part in path.parts if part.lower().startswith("table_")), "NAIVE")
        return f"NAIVE {table.replace('_', ' ')}"
    if source.key == "noise_prefix_sweep":
        if "preview" in rel:
            return "NOISE preview"
        return "NOISE q-sweep diagnostics"
    if source.key == "random_subset":
        return "random subset archive"
    if source.key == "random_order":
        return "random order archive"
    if source.key == "mallows":
        return "NOISE selector diagnostics"
    if source.key == "pathmnist_generalization":
        if "dual-geometry" in rel:
            return "NOISE dual-geometry diagnostics"
        if "q-sweep" in rel:
            return "NOISE q-sweep diagnostics"
        if "oracle-noise" in rel:
            return "NOISE diagnostics"
        if "/table1/" in f"/{rel}":
            return "main comparison"
    return source.experiment


def discover_csvs(source: SourceSpec) -> list[Path]:
    paths: set[Path] = set()
    if not source.root.exists():
        return []
    for pattern in source.patterns:
        paths.update(path for path in source.root.glob(pattern) if path.is_file())
    return sorted(paths)


def add_compatibility_records(records: list[TableRecord]) -> None:
    """Render real-checkpoint compatibility pilots as archival diagnostics.

    These JSON files are deliberately kept separate from the CSV metric
    exports.  A compatibility pilot reports whether each explainer is finite,
    deterministic, and target-sensitive on one checkpoint; it does not report
    Phase 2 fidelity or robustness.  Keeping one table per cell makes partial
    model work discoverable without presenting it as a completed experiment.
    """
    source = SourceSpec(
        "full_matrix_compatibility",
        SOURCE_ROOTS["full_matrix_compatibility"],
        ("compatibility/cells/*/result.json",),
        "compatibility pilot",
        historical_default=True,
        priority=20,
    )
    columns = [
        "dataset",
        "model",
        "result_status",
        "gate_status",
        "method",
        "passed",
        "deterministic_repeat",
        "finite_fraction",
        "target_sensitive",
        "selected_batch_size",
        "attack_batch_size",
        "attack_seconds_per_sample",
        "attack_peak_cuda_bytes",
        "error",
    ]
    for path in sorted(source.root.glob("compatibility/cells/*/result.json")):
        payload = load_json(path)
        measurements = payload.get("measurements")
        if not isinstance(measurements, dict):
            continue
        methods = measurements.get("methods")
        if not isinstance(methods, list) or not methods:
            continue
        dataset, model = split_cell(path.parent.name)
        if not dataset or not model:
            dataset, model = infer_pair(measurements, measurements)
        if not dataset or not model:
            continue
        gate = load_json(path.parent / "gate.json")
        gate_status = gate.get("status", "")

        def scalar(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return str(value).lower()
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            return str(value)

        rows: list[dict[str, str]] = []
        for method in methods:
            if not isinstance(method, dict):
                continue
            rows.append(
                {
                    "dataset": display_dataset(dataset),
                    "model": display_model(model),
                    "result_status": scalar(payload.get("status")),
                    "gate_status": scalar(gate_status),
                    "method": scalar(method.get("method")),
                    "passed": scalar(method.get("passed")),
                    "deterministic_repeat": scalar(method.get("deterministic_repeat")),
                    "finite_fraction": scalar(method.get("finite_fraction")),
                    "target_sensitive": scalar(method.get("target_sensitive")),
                    "selected_batch_size": scalar(method.get("selected_batch_size")),
                    "attack_batch_size": scalar(method.get("attack_batch_size")),
                    "attack_seconds_per_sample": scalar(method.get("attack_seconds_per_sample")),
                    "attack_peak_cuda_bytes": scalar(method.get("attack_peak_cuda_bytes")),
                    "error": scalar(method.get("error")),
                }
            )
        if not rows:
            continue
        metadata = {
            "dataset": dataset,
            "model": model,
            "cell": path.parent.name,
            "status": payload.get("status"),
            "gate_status": gate_status,
            "config_digest": payload.get("config_digest"),
            "result_digest": payload.get("digest"),
            "failure_count": len(payload.get("failures") or []),
            "method_count": len(rows),
            "blocked_method_count": len(gate.get("blocked_methods") or []) if isinstance(gate, dict) else 0,
        }
        records.append(
            TableRecord(
                dataset,
                model,
                "compatibility pilot",
                source.key,
                str(path),
                path.relative_to(source.root).as_posix(),
                version_for(path, source),
                True,
                rows,
                columns,
                False,
                metadata,
            )
        )


def context_pairs(
    path: Path,
    source: SourceSpec,
    metadata: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Find the cell coverage for aggregate/empty companion files.

    Analysis CSVs intentionally aggregate across a scope and therefore leave
    their dataset/model columns blank.  The neighbouring base table is the
    authoritative coverage contract for those files.
    """
    pairs: set[tuple[str, str]] = set(metadata_pairs(metadata))
    candidates: list[Path] = []
    if source.key in {"canonical_results", "canonical_live_results"} and "github-complete-results-v1" in path.parts:
        try:
            relative = path.relative_to(source.root)
            scope_index = relative.parts.index("scopes")
            scope_dir = source.root.joinpath(*relative.parts[: scope_index + 2])
            candidates = sorted(scope_dir.glob("base/*/table1.csv"))
        except (ValueError, IndexError):
            candidates = []
    elif source.key == "paper_main":
        candidates = [path.parent / "table1.csv"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            rows, _ = read_csv(candidate)
        except (OSError, UnicodeError, csv.Error):
            continue
        for row in rows:
            dataset, model = infer_pair(row)
            if dataset and model:
                pairs.add((dataset, model))
    filename_dataset, filename_model = filename_pair(path)
    if filename_dataset and filename_model:
        pairs.add((filename_dataset, filename_model))
    if not pairs and source.default_dataset and source.default_model:
        pairs.add((normalise_dataset(source.default_dataset), normalise_model(source.default_model)))
    return sorted(pairs)


def add_csv_records(records: list[TableRecord], source: SourceSpec, path: Path) -> None:
    metadata = metadata_for(path, source.key)
    fallback_pairs = context_pairs(path, source, metadata)
    try:
        rows, columns = read_csv(path)
    except (OSError, UnicodeError, csv.Error):
        rows, columns = [], []
    relative = path.relative_to(source.root).as_posix()
    version = version_for(path, source)
    experiment = experiment_for(source, path)
    if not rows:
        dataset, model = infer_pair({}, metadata)
        filename_dataset, filename_model = filename_pair(path)
        pairs = fallback_pairs or [
            (
                normalise_dataset(str(source.default_dataset or dataset or filename_dataset or metadata.get("dataset") or "unknown")),
                normalise_model(str(source.default_model or model or filename_model or metadata.get("model") or "unknown")),
            )
        ]
        for dataset, model in pairs:
            records.append(
                TableRecord(
                    dataset,
                    model,
                    experiment,
                    source.key,
                    str(path),
                    relative,
                    version,
                    is_historical(path, source),
                    [],
                    columns,
                    True,
                    metadata,
                )
            )
        return

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    unresolved_rows: list[dict[str, str]] = []
    for row in rows:
        dataset, model = infer_pair(row, metadata)
        if not dataset or not model:
            unresolved_rows.append(row)
            continue
        enriched = dict(row)
        if dataset and not enriched.get("dataset"):
            enriched["dataset"] = dataset
        if model and not enriched.get("model"):
            enriched["model"] = model
        groups[(dataset, model)].append(enriched)
    if unresolved_rows:
        filename_dataset, filename_model = filename_pair(path)
        pairs = fallback_pairs or [
            (
                normalise_dataset(str(source.default_dataset or filename_dataset or "unknown")),
                normalise_model(str(source.default_model or filename_model or "unknown")),
            )
        ]
        for dataset, model in pairs:
            groups[(dataset, model)].extend(unresolved_rows)
    for (dataset, model), grouped_rows in sorted(groups.items()):
        records.append(
            TableRecord(
                dataset,
                model,
                experiment,
                source.key,
                str(path),
                relative,
                version,
                is_historical(path, source),
                grouped_rows,
                columns,
                False,
                metadata,
            )
        )


def parse_relative_log(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        start = text.find("{")
        if start < 0:
            return {}
        value = json.loads(text[start:])
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def add_relative_records(records: list[TableRecord]) -> None:
    source = SourceSpec("relative_robustness", SOURCE_ROOTS["relative_robustness"], (), "relative robustness")
    root = source.root / "logs"
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    columns = ["dataset", "model", "condition", "condition_kind", "status", "sample_count", "k", "F", "Fbar", "C", "Cbar"]
    for path in sorted(root.glob("*.log")):
        payload = parse_relative_log(path)
        if not payload:
            continue
        dataset, model = infer_pair(payload, payload)
        mean = payload.get("mean_metrics") or {}
        row = {
            "dataset": dataset,
            "model": model,
            "condition": str(payload.get("condition", "")),
            "condition_kind": str(payload.get("condition_kind", "")),
            "status": str(payload.get("status", "")),
            "sample_count": str(payload.get("sample_count", "")),
            "k": str(payload.get("k", "")),
            "F": json_cell(mean.get("F")),
            "Fbar": json_cell(mean.get("Fbar")),
            "C": json_cell(mean.get("C")),
            "Cbar": json_cell(mean.get("Cbar")),
        }
        groups[(dataset, model)].append(row)
    for (dataset, model), rows in sorted(groups.items()):
        records.append(
            TableRecord(
                dataset,
                model,
                "relative robustness",
                source.key,
                str(root),
                "logs/*.log",
                "structured-log-export",
                False,
                rows,
                columns,
                not rows,
                {"log_count": len(rows)},
            )
        )


def digest_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def record_digest(record: TableRecord) -> str:
    """Return a stable digest for a source table or structured log export."""
    path = Path(record.source_path)
    if path.is_file():
        return digest_bytes(path)
    payload = {
        "columns": record.columns,
        "rows": record.rows,
        "source_key": record.source_key,
        "relative_source": record.relative_source,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_priority(source_key: str) -> int:
    for source in source_specs():
        if source.key == source_key:
            return source.priority
    if source_key == "relative_robustness":
        return 90
    return 0


def deduplicate_records(records: list[TableRecord]) -> list[TableRecord]:
    """Collapse byte-identical exports while retaining their provenance."""
    grouped: dict[tuple[str, str, str], TableRecord] = {}
    aliases: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        record.content_digest = record.content_digest or record_digest(record)
        key = (record.content_digest, normalise_dataset(record.dataset), normalise_model(record.model))
        existing = grouped.get(key)
        alias = {
            "source_key": record.source_key,
            "relative_source": record.relative_source,
            "historical": str(record.historical).lower(),
        }
        if existing is None:
            grouped[key] = record
            aliases[key].append(alias)
            continue
        aliases[key].append(alias)
        existing_rank = (existing.historical, -source_priority(existing.source_key), existing.source_key, existing.relative_source)
        candidate_rank = (record.historical, -source_priority(record.source_key), record.source_key, record.relative_source)
        if candidate_rank < existing_rank:
            grouped[key] = record
    result = []
    for key, record in grouped.items():
        record.duplicate_sources = [
            alias for alias in aliases[key]
            if alias["source_key"] != record.source_key or alias["relative_source"] != record.relative_source
        ]
        result.append(record)
    return result


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep provenance-sized metadata without embedding raw summary payloads.

    Source summaries can contain millions of measurements or nested bootstrap
    diagnostics.  The generated tables already contain the published values;
    the manifest only needs scalar identifiers plus a compact description of
    omitted structures for auditability.
    """
    compact: dict[str, Any] = {}
    scalar_keys = {
        "dataset",
        "model",
        "cell",
        "status",
        "run_id",
        "protocol_digest",
        "result_digest",
        "q_summary_digest",
        "input_catalog_digest",
        "source_digest",
        "scope",
        "version",
        "generated_at",
    }
    for key in sorted(scalar_keys):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                compact[key] = value
    for key in sorted(metadata):
        if key in scalar_keys:
            continue
        value = metadata[key]
        if isinstance(value, list):
            compact[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            compact[f"{key}_key_count"] = len(value)
    return compact


def collect_records() -> list[TableRecord]:
    records: list[TableRecord] = []
    for source in source_specs():
        for path in discover_csvs(source):
            add_csv_records(records, source, path)
    add_compatibility_records(records)
    add_relative_records(records)
    return deduplicate_records(records)


def table_caption(record: TableRecord) -> str:
    coverage = f"{display_dataset(record.dataset)}--{display_model(record.model)}"
    if record.experiment == "compatibility pilot":
        return (
            f"Real-checkpoint explainer compatibility pilot for {coverage}. "
            "This archival diagnostic reports per-method execution checks and "
            "is not a Phase 2 quality or robustness result."
        )
    if record.historical:
        status = "a historical snapshot included for archival comparison"
    else:
        status = "the current released snapshot"
    return (
        f"{public_name(record.experiment)} results for {coverage}. "
        f"This table is {status}; it reports "
        "only the dataset--model rows present in the source file."
    )


def column_label(column: str) -> str:
    label = public_name(column)
    label = label.replace("_", " ")
    label = re.sub(r"(?i)perturbed quality", "perturbed quality", label)
    return label


def render_table(record: TableRecord, table_id: str) -> str:
    columns = record.columns or sorted({key for row in record.rows for key in row})
    columns = [column for column in columns if column]
    if not columns:
        return ""
    alignment = "".join("p{0.035\\linewidth}" if index else "p{0.08\\linewidth}" for index, _ in enumerate(columns))
    lines = [
        "% Generated by code/scripts/results_to_latex.py; do not edit by hand.",
        "\\begin{landscape}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{longtable}{@{}" + alignment + "@{}}",
        "\\caption{" + latex_escape(table_caption(record)) + "}\\label{tab:" + table_id + "}\\\\",
        "\\toprule",
        " & ".join(latex_escape(column_label(column)) for column in columns) + r" \\" + "",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(latex_escape(column_label(column)) for column in columns) + r" \\" + "",
        "\\midrule",
        "\\endhead",
    ]
    for row in record.rows:
        values = [latex_escape(format_cell(row.get(column, ""))) for column in columns]
        lines.append(" & ".join(values) + r" \\" + "")
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\end{landscape}", ""])
    return "\n".join(lines)


def write_report_preamble(dataset: str) -> str:
    label = display_dataset(dataset)
    note = ""
    if normalise_dataset(dataset) == "imagenet":
        note = (
            " The released ImageNet rows use the 100-class ImageNet-1k subset "
            "(ImageNet100); the label ImageNet is used consistently in this archive."
        )
    return "\n".join(
        [
            "\\documentclass[10pt,a4paper]{article}",
            "\\usepackage[margin=1.5cm]{geometry}",
            "\\usepackage{amsmath,amssymb,array,booktabs,caption,graphicx,longtable,pdflscape}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{lmodern}",
            "\\usepackage[hidelinks]{hyperref}",
            "\\captionsetup{hypcap=false}",
            "",
            f"\\title{{When Does Aggregating Explanations Work?\\\\\\large {latex_escape(label)} Results}}",
            "\\author{Jinhua Xu, Davide Anguita, Fabio Roli, Jing Yuan, and Luca Oneto}",
            "\\date{}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\noindent",
            "This report is generated from the structured result files listed in",
            "\\texttt{RESULT\\_MANIFEST.json}. It is organised by model and experiment",
            "and retains incomplete and historical snapshots as archival tables.",
            latex_escape(note),
            "",
        ]
    )


def write_outputs(records: list[TableRecord], output: Path) -> dict[str, Any]:
    by_dataset: dict[str, list[TableRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = output / "by-dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    # Remove the old manually assembled single-report directory.  The new
    # reports are all under by-dataset and are referenced from the top-level
    # index below.
    old_naive = output / "naive"
    if old_naive.exists():
        shutil.rmtree(old_naive)

    manifest: dict[str, Any] = {
        "schema": "xai-results-latex-manifest-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "code/scripts/results_to_latex.py",
        "imageNet_note": "ImageNet labels refer to the released 100-class ImageNet-1k subset (ImageNet100).",
        "records": [],
        "datasets": {},
    }
    counts: dict[str, int] = defaultdict(int)
    for dataset in sorted(by_dataset):
        dataset_records = sorted(
            by_dataset[dataset],
            key=lambda record: (normalise_model(record.model), record.experiment, record.source_key, record.relative_source, record.version),
        )
        dataset_dir = dataset_root / safe_name(dataset)
        tables_dir = dataset_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(dataset_records, start=1):
            if record.empty:
                continue
            table_id = f"{safe_name(dataset)}-{safe_name(record.model)}-{index:04d}"
            filename = f"{table_id}.tex"
            (tables_dir / filename).write_text(render_table(record, table_id), encoding="utf-8")
            record.generated_path = f"by-dataset/{safe_name(dataset)}/tables/{filename}"
            counts[dataset] += 1
        tables_tex: list[str] = ["% Generated by code/scripts/results_to_latex.py.", ""]
        current_records = [record for record in dataset_records if not record.historical and not record.empty]
        historical_records = [record for record in dataset_records if record.historical and not record.empty]

        def append_model_sections(records_for_sections: list[TableRecord], heading_suffix: str = "") -> None:
            grouped: dict[str, list[TableRecord]] = defaultdict(list)
            for record in records_for_sections:
                grouped[record.model].append(record)
            for model in sorted(grouped, key=normalise_model):
                title = display_model(model) + heading_suffix
                tables_tex.append(f"\\section{{{latex_escape(title)}}}")
                tables_tex.append("")
                for record in grouped[model]:
                    if record.generated_path is None:
                        continue
                    tables_tex.append(f"\\input{{tables/{Path(record.generated_path).name}}}")
                    tables_tex.append("")

        append_model_sections(current_records)
        if historical_records:
            tables_tex.extend([
                "\\appendix",
                "\\section*{Historical and superseded result tables}",
                "",
            ])
            append_model_sections(historical_records, " (archived)")
        (dataset_dir / "tables.tex").write_text("\n".join(tables_tex), encoding="utf-8")
        (dataset_dir / "main.tex").write_text(
            write_report_preamble(dataset)
            + "\\input{tables.tex}\n\n\\end{document}\n",
            encoding="utf-8",
        )
        dataset_records_meta = {
            "table_count": counts[dataset],
            "model_count": len({normalise_model(record.model) for record in dataset_records}),
            "models": sorted({display_model(record.model) for record in dataset_records}),
        }
        manifest["datasets"][display_dataset(dataset)] = dataset_records_meta
        readme = [
            f"# {display_dataset(dataset)} results",
            "",
            f"This report contains {counts[dataset]} generated LaTeX tables from released structured result and diagnostic exports.",
            "Tables are grouped by model in `tables.tex`; incomplete snapshots are retained and historical/superseded snapshots appear after the LaTeX appendix marker.",
            "",
        ]
        if normalise_dataset(dataset) == "imagenet":
            readme.append("The ImageNet-labelled rows use the 100-class ImageNet-1k subset (ImageNet100).")
            readme.append("")
        (dataset_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    for record in records:
        row_count = len(record.rows)
        manifest["records"].append(
            {
                "dataset": display_dataset(record.dataset),
                "dataset_key": normalise_dataset(record.dataset),
                "model": display_model(record.model),
                "model_key": normalise_model(record.model),
                "experiment": public_name(record.experiment),
                "source_key": record.source_key,
                # Keep the public manifest portable: absolute workspace paths
                # are implementation details and must not enter the release.
                "source_path": f"{record.source_key}/{record.relative_source}",
                "relative_source": record.relative_source,
                "version": record.version,
                "historical": record.historical,
                "content_sha256": record.content_digest,
                "duplicate_source_count": len(record.duplicate_sources),
                "duplicate_sources": record.duplicate_sources,
                "row_count": row_count,
                "columns": [public_name(column) for column in record.columns],
                "empty": record.empty,
                "coverage": f"{display_dataset(record.dataset)}--{display_model(record.model)}",
                "generated_latex": record.generated_path,
                "metadata": compact_metadata(record.metadata),
            }
        )
    manifest["table_counts"] = {display_dataset(key): value for key, value in sorted(counts.items())}
    (output / "RESULT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top_lines = [
        "# Released result LaTeX",
        "",
        "Generated by `code/scripts/results_to_latex.py`. Each dataset has a standalone report under `by-dataset/<dataset>/`; model sections and experiment tables are included from `tables.tex`.",
        "",
        "The manifest records every discovered CSV/log export, including empty files and historical snapshots. Empty sources are recorded but do not produce empty LaTeX tables.",
        "",
        "ImageNet-labelled results use the 100-class ImageNet-1k subset (ImageNet100). A report only claims the dataset/model rows present in each source table.",
        "The count includes archival real-checkpoint compatibility diagnostics where available; those tables report execution checks, not completed Phase 2 quality or robustness metrics.",
        "Historical configurations are retained because their result rows remain useful; the archive is not filtered to one target protocol.",
        "",
        "## Table counts",
        "",
    ]
    for dataset, count in sorted(manifest["table_counts"].items()):
        top_lines.append(f"- {dataset}: {count}")
    top_lines.extend(["", "See `RESULT_MANIFEST.json` for source paths, versions, columns, row counts, and coverage.", ""])
    (output / "README.md").write_text("\n".join(top_lines), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-counts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = collect_records()
    manifest = write_outputs(records, args.output)
    if args.print_counts:
        for dataset, count in sorted(manifest["table_counts"].items()):
            print(f"{dataset}: {count}")
        print(f"total tables: {sum(manifest['table_counts'].values())}")
        print(f"manifest records: {len(manifest['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
