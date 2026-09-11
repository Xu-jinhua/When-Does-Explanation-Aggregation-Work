#!/usr/bin/env python3
"""Read-only environment self-check for the XAI ensemble pipelines.

Run this before `simple validate` / `simple run` to catch missing packages,
absent CUDA runtimes, unwritable scratch space, or a missing rclone binary.
Every check is read-only except for creating (and deleting) one small probe
file per writable-directory check.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "yaml",
    "timm",
    "captum",
    "einops",
    "datasets",
    "huggingface_hub",
    "safetensors",
    "pyarrow",
    "pandas",
    "medmnist",
)

PACKAGE_DISTRIBUTIONS = {
    "yaml": "PyYAML",
    "huggingface_hub": "huggingface-hub",
}


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    required: bool
    detail: Any


def _writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".xai-doctor-", dir=path)
        os.close(descriptor)
        Path(temporary).unlink()
    except OSError as error:
        return False, str(error)
    return True, str(path.resolve())


def _gpus() -> tuple[list[dict[str, Any]], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return [], "nvidia-smi not found"
    command = [
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        return [], str(error)
    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            return [], f"unexpected nvidia-smi row: {line!r}"
        devices.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mib": int(parts[2]),
                "driver": parts[3],
            }
        )
    return devices, None


def _package_status(package: str) -> tuple[bool, dict[str, str]]:
    try:
        importlib.import_module(package)
    except Exception as error:  # Import-time binary/runtime failures matter here.
        return False, {"status": "import failed", "error": f"{type(error).__name__}: {error}"}
    distribution = PACKAGE_DISTRIBUTIONS.get(package, package)
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = "unknown"
    return True, {"status": "imported", "version": version}


def _torch_cuda() -> tuple[dict[str, Any], str | None]:
    try:
        import torch
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"
    try:
        count = int(torch.cuda.device_count())
        devices = [
            {
                "index": index,
                "name": str(torch.cuda.get_device_name(index)),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(count)
        ]
        return {
            "available": bool(torch.cuda.is_available()),
            "device_count": count,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if count else False,
            "devices": devices,
        }, None
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[Check] = []
    python_supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(
        Check(
            "python",
            python_supported,
            True,
            {
                "version": platform.python_version(),
                "executable": sys.executable,
                "required": ">=3.11,<3.13",
            },
        )
    )
    work_dir = Path(args.work_dir)
    writable, detail = _writable(work_dir)
    checks.append(Check("work_dir_writable", writable, True, detail))
    usage = shutil.disk_usage(work_dir if work_dir.exists() else work_dir.parent)
    free_gib = usage.free / (1024**3)
    checks.append(
        Check(
            "scratch_disk",
            free_gib >= args.min_free_gib,
            True,
            {"path": str(work_dir.resolve()), "free_gib": round(free_gib, 2)},
        )
    )
    if args.cache_dir:
        cache_writable, cache_detail = _writable(Path(args.cache_dir))
        checks.append(Check("cache_dir_writable", cache_writable, True, cache_detail))

    devices, gpu_error = _gpus()
    enough_gpus = len(devices) >= args.min_gpus
    enough_memory = enough_gpus and all(
        device["memory_mib"] >= args.min_gpu_memory_mib
        for device in devices[: args.min_gpus]
    )
    checks.append(
        Check(
            "nvidia_gpus",
            enough_gpus and enough_memory,
            args.require_gpu,
            gpu_error or devices,
        )
    )

    cuda, cuda_error = _torch_cuda()
    cuda_passed = (
        not cuda_error
        and bool(cuda.get("available"))
        and int(cuda.get("device_count", 0) or 0) >= args.min_gpus
        and (not args.require_bf16 or bool(cuda.get("bf16_supported")))
    )
    checks.append(Check("torch_cuda_runtime", cuda_passed, args.require_gpu, cuda_error or cuda))

    rclone = shutil.which(args.rclone_binary)
    checks.append(
        Check(
            "rclone_binary",
            rclone is not None,
            args.require_rclone,
            rclone or f"{args.rclone_binary!r} not found on PATH",
        )
    )

    required_packages = set(args.require_package)
    packages = tuple(dict.fromkeys((*args.package, *args.require_package)))
    for package in packages:
        installed, package_detail = _package_status(package)
        checks.append(
            Check(
                f"package:{package}",
                installed,
                package in required_packages,
                package_detail,
            )
        )

    required_failures = [check.name for check in checks if check.required and not check.passed]
    return {
        "schema_version": 1,
        "passed": not required_failures,
        "required_failures": required_failures,
        "host": platform.node(),
        "platform": platform.platform(),
        "checks": [asdict(check) for check in checks],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="runtime")
    parser.add_argument("--cache-dir")
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--min-gpu-memory-mib", type=int, default=8 * 1024)
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail when no suitable NVIDIA GPU / CUDA runtime is present",
    )
    parser.add_argument(
        "--require-bf16",
        action="store_true",
        help="also require bf16 support (only needed for bf16 training recipes)",
    )
    parser.add_argument(
        "--rclone-binary",
        default="rclone",
        help="rclone executable name looked up on PATH",
    )
    parser.add_argument(
        "--require-rclone",
        action="store_true",
        help="fail when the rclone binary is missing (only needed for rclone remotes)",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--require-package",
        action="append",
        default=list(DEFAULT_REQUIRED_PACKAGES),
    )
    parser.add_argument("--output")
    return parser


def _atomic_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_checks(args)
    content = json.dumps(report, sort_keys=True, indent=2)
    print(content)
    if args.output:
        _atomic_report(Path(args.output), report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
