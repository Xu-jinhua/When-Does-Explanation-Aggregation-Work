#!/usr/bin/env python3
"""Fetch and verify the exact Chefer RelProp source used by Phase 1."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = CODE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from xai_ensemble.phase1.relprop import (  # noqa: E402
    RELPROP_REPOSITORY_URL,
    RELPROP_REVISION,
    validate_relprop_repository,
)


def _git(*arguments: str) -> None:
    subprocess.run(["git", *arguments], check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        default=str(CODE_ROOT / "vendor/Transformer-Explainability"),
        help="checkout destination (default: code/vendor/Transformer-Explainability)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate an existing checkout without network or filesystem changes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination = Path(args.destination).expanduser().resolve()
    if args.verify_only:
        provenance = validate_relprop_repository(destination)
    else:
        if destination.exists() and any(destination.iterdir()):
            # Existing valid checkouts are idempotent.  Wrong or modified
            # directories are never overwritten by this setup command.
            provenance = validate_relprop_repository(destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            _git("-C", str(destination), "init", "--quiet")
            _git(
                "-C",
                str(destination),
                "remote",
                "add",
                "origin",
                RELPROP_REPOSITORY_URL,
            )
            _git(
                "-C",
                str(destination),
                "fetch",
                "--depth=1",
                "origin",
                RELPROP_REVISION,
            )
            _git("-C", str(destination), "checkout", "--detach", "FETCH_HEAD")
            provenance = validate_relprop_repository(destination)
    print(
        "PASS pinned RelProp provider "
        f"revision={provenance['revision']} source_digest={provenance['source_digest']} "
        f"root={provenance['repository']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
