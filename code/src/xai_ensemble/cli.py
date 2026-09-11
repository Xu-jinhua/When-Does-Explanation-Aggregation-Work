from __future__ import annotations

import argparse
import sys
import traceback

import yaml

from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.core.protocol import load_protocol
from xai_ensemble.core.provenance import collect_provenance


def _protocol_validate(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.path)
    print(f"PASS protocol={protocol.run_id} digest={protocol.digest}")
    return 0


def _protocol_show(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.path)
    print(yaml.safe_dump(protocol.resolved_dict(), sort_keys=False))
    return 0


def _protocol_snapshot(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.path)
    value = {
        "schema_version": 1,
        "run_id": protocol.run_id,
        "protocol_digest": protocol.digest,
        "protocol": protocol.resolved_dict(),
        "provenance": collect_provenance(args.project_root),
    }
    atomic_write_json(args.output, value)
    print(f"WROTE {args.output} digest={protocol.digest}")
    return 0


def _register_protocol(root: argparse._SubParsersAction) -> None:
    parser = root.add_parser("protocol", help="Validate and snapshot the machine-readable protocol")
    commands = parser.add_subparsers(dest="protocol_command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("path")
    validate.set_defaults(handler=_protocol_validate)
    show = commands.add_parser("show")
    show.add_argument("path")
    show.set_defaults(handler=_protocol_show)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("path")
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--project-root", default=".")
    snapshot.set_defaults(handler=_protocol_snapshot)


def _register_group(
    root: argparse._SubParsersAction,
    name: str,
    help_text: str,
    register: object,
) -> None:
    parser = root.add_parser(name, help=help_text)
    commands = parser.add_subparsers(dest=f"{name}_command", required=True)
    register(commands)  # type: ignore[operator]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xai-exp")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    root = parser.add_subparsers(dest="command", required=True)
    _register_protocol(root)

    optional_groups = [
        ("simple", "Run the paper-first two-stage experiment", "xai_ensemble.simple.cli"),
        ("phase0", "Prepare data and train models", "xai_ensemble.phase0.cli"),
    ]
    import importlib

    for name, description, module_name in optional_groups:
        try:
            register = importlib.import_module(module_name).register_subcommands
            _register_group(root, name, description, register)
        except (ImportError, AttributeError) as error:
            root._name_parser_map.pop(name, None)
            root._choices_actions = [a for a in root._choices_actions if a.dest != name]
            missing = getattr(error, "name", None)
            hint = f" (missing optional dependency: {missing})" if missing else ""
            print(
                f"note: '{name}' command group unavailable{hint}; "
                "install the gpu extra: pip install -e '.[gpu]'",
                file=sys.stderr,
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command handler was registered")
    try:
        return int(handler(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        if getattr(args, "debug", False):
            raise
        chain = []
        current: BaseException | None = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        print(f"ERROR {' <- '.join(chain)}", file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
