from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from xai_ensemble.core.strict_json import load_strict_json

from .base import PilotResult


def _load_config(path: str) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"pilot config must be a mapping: {path}")
    return value


def _save_measured_result(path: str, result: PilotResult) -> int:
    from .base import save_pilot_result

    save_pilot_result(path, result)
    print(
        f"{result.status.upper()} {result.name} digest={result.digest} output={path}"
    )
    return 0 if result.passed else 2


def _check(args: argparse.Namespace) -> int:
    value = load_strict_json(args.result)
    result = PilotResult.from_dict(value)
    result.require_passed()
    print(f"PASS {result.name} digest={result.digest}")
    return 0


def _list(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    for path in sorted(root.glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        print(f"{value.get('name', path.stem):24} {value.get('description', '')}")
    return 0


def _show(args: argparse.Namespace) -> int:
    path = Path(args.config)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(yaml.safe_dump(value, sort_keys=False))
    return 0


def _json_reports(path: str) -> list[dict[str, object]]:
    value = load_strict_json(path)
    if isinstance(value, dict):
        value = value.get("reports", value.get("candidates", value.get("arms")))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError("measurement file must contain an array of JSON objects")
    return value


def _kemeny_solver(args: argparse.Namespace) -> int:
    from .kemeny_solver import kemeny_benchmark

    result = kemeny_benchmark(
        config=_load_config(args.config),
        seed=args.seed,
        max_passes=args.max_passes,
    )
    return _save_measured_result(args.output, result)


def _noise_gof_cost(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.reports:
        from .noise_gof_cost import noise_gof_cost_pilot

        result = noise_gof_cost_pilot(_json_reports(args.reports), config=config)
    else:
        if not args.artifact:
            raise ValueError("--artifact is required when --reports is not given")
        from .noise_gof_cost import run_noise_gof_cost_pilot

        proposal_q = (
            [float(item) for item in args.synthetic_proposal_q.split(",") if item.strip()]
            if args.synthetic_proposal_q
            else None
        )
        result = run_noise_gof_cost_pilot(
            config=config,
            seed=args.seed,
            artifact_path=args.artifact,
            n_items=args.n_items,
            synthetic_validation_count=args.synthetic_validation_count,
            synthetic_proposal_q=proposal_q,
            bootstrap_replicates=args.bootstrap_replicates,
            projected_prefix_count=args.projected_prefix_count,
        )
    return _save_measured_result(args.output, result)


def _ig_step_budget(args: argparse.Namespace) -> int:
    from xai_ensemble.simple.config import load_experiment

    from .ig_step_budget import run_ig_step_budget_pilot

    experiment = load_experiment(args.experiment)
    result = run_ig_step_budget_pilot(
        experiment,
        model_id=args.model_id,
        config=_load_config(args.config),
        split=args.split,
        sample_count=args.sample_count,
        device=args.device,
    )
    return _save_measured_result(args.output, result)


def _baseline_selection(args: argparse.Namespace) -> int:
    from xai_ensemble.simple.config import load_experiment

    from .baseline_selection import run_baseline_selection_pilot

    experiment = load_experiment(args.experiment)
    result = run_baseline_selection_pilot(
        experiment,
        model_id=args.model_id,
        config=_load_config(args.config),
        split=args.split,
        sample_count=args.sample_count,
        device=args.device,
    )
    return _save_measured_result(args.output, result)


def register_subcommands(subparsers: argparse._SubParsersAction) -> None:
    check_parser = subparsers.add_parser(
        "check", help="Verify a pilot result digest and require a passed status"
    )
    check_parser.add_argument("result")
    check_parser.set_defaults(handler=_check)

    list_parser = subparsers.add_parser("list", help="List all pilot definitions")
    list_parser.add_argument("--directory", default="configs/pilots")
    list_parser.set_defaults(handler=_list)

    show_parser = subparsers.add_parser("show", help="Print a pilot definition")
    show_parser.add_argument("config")
    show_parser.set_defaults(handler=_show)

    kemeny_parser = subparsers.add_parser(
        "kemeny-solver",
        help="Benchmark the Kemeny local-search grid against an exact n=16 subset-DP oracle",
    )
    kemeny_parser.add_argument(
        "--config", default="configs/pilots/kemeny_solver.yaml"
    )
    kemeny_parser.add_argument("--seed", type=int, default=20260714)
    kemeny_parser.add_argument("--max-passes", type=int, default=10_000)
    kemeny_parser.add_argument("--output", required=True)
    kemeny_parser.set_defaults(handler=_kemeny_solver)

    noise_parser = subparsers.add_parser(
        "noise-gof-cost",
        help="Calibrate the Spearman distance family and adjudicate the refitted-bootstrap budget",
    )
    noise_parser.add_argument(
        "--config", default="configs/pilots/noise_gof_cost.yaml"
    )
    noise_parser.add_argument(
        "--reports",
        help="adjudicate pre-measured cost reports instead of running the calibration",
    )
    noise_parser.add_argument("--artifact", help="output Spearman family NPZ")
    noise_parser.add_argument(
        "--n-items",
        type=int,
        default=196,
        help="ranked item count; the primary p=16 grid on 224x224 inputs has 14x14=196 patches",
    )
    noise_parser.add_argument("--synthetic-validation-count", type=int)
    noise_parser.add_argument(
        "--synthetic-proposal-q",
        help="comma-separated exact Kendall--Mallows q values for the validation profile",
    )
    noise_parser.add_argument("--bootstrap-replicates", type=int)
    noise_parser.add_argument("--projected-prefix-count", type=int)
    noise_parser.add_argument("--seed", type=int, default=0)
    noise_parser.add_argument("--output", required=True)
    noise_parser.set_defaults(handler=_noise_gof_cost)

    ig_parser = subparsers.add_parser(
        "ig-step-budget",
        help="Sweep the Integrated Gradients approximation depth on validation Fidelity",
    )
    ig_parser.add_argument(
        "--config", default="configs/pilots/ig_step_budget.yaml"
    )
    ig_parser.add_argument("--experiment", required=True)
    ig_parser.add_argument("--model-id", required=True)
    ig_parser.add_argument("--split")
    ig_parser.add_argument("--sample-count", type=int)
    ig_parser.add_argument("--device", default="cuda:0")
    ig_parser.add_argument("--output", required=True)
    ig_parser.set_defaults(handler=_ig_step_budget)

    baseline_parser = subparsers.add_parser(
        "baseline-selection",
        help="Select each baseline-dependent method's reference by validation Fidelity",
    )
    baseline_parser.add_argument(
        "--config", default="configs/pilots/baseline_selection.yaml"
    )
    baseline_parser.add_argument("--experiment", required=True)
    baseline_parser.add_argument("--model-id", required=True)
    baseline_parser.add_argument("--split")
    baseline_parser.add_argument("--sample-count", type=int)
    baseline_parser.add_argument("--device", default="cuda:0")
    baseline_parser.add_argument("--output", required=True)
    baseline_parser.set_defaults(handler=_baseline_selection)
