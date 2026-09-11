"""Low-overhead stage timing and device-level NVML telemetry."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StageTimings:
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.add(stage, time.monotonic() - started)

    def add(self, stage: str, seconds: float) -> None:
        if not stage:
            raise ValueError("telemetry stage must be non-empty")
        value = float(seconds)
        if value < 0:
            raise ValueError("telemetry duration cannot be negative")
        with self._lock:
            self.values[stage].append(value)

    def summary(self) -> Mapping[str, Mapping[str, float | int]]:
        with self._lock:
            values = {stage: tuple(samples) for stage, samples in self.values.items()}
        return {
            stage: {
                "count": len(values),
                "total_seconds": float(sum(values)),
                "mean_seconds": float(sum(values) / len(values)),
                "max_seconds": float(max(values)),
            }
            for stage, values in sorted(values.items())
            if values
        }


class GpuUtilizationSampler:
    """Sample nvidia-smi's NVML-backed device utilization in the background."""

    def __init__(self, *, requested_device: str, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU telemetry interval must be positive")
        visible = tuple(
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        )
        if ":" in requested_device:
            requested = requested_device.rpartition(":")[2]
        elif requested_device.isdigit():
            requested = requested_device
        else:
            requested = "0"
        try:
            logical_index = int(requested)
        except ValueError:
            logical_index = -1
        self.device_id = (
            visible[logical_index] if visible and 0 <= logical_index < len(visible) else requested
        )
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, int, int]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                (
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=max(2.0, self.interval_seconds),
            )
            row = next(line for line in result.stdout.splitlines() if line.strip())
            utilization, memory_mib = (int(value.strip()) for value in row.split(","))
            with self._lock:
                self._samples.append((time.time(), utilization, memory_mib * 2**20))
        except (OSError, StopIteration, ValueError, subprocess.SubprocessError) as error:
            with self._lock:
                if len(self._errors) < 4:
                    self._errors.append(f"{type(error).__name__}: {error}")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self) -> GpuUtilizationSampler:
        if self._thread is not None:
            raise RuntimeError("GPU utilization sampler is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="simple-gpu-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> Mapping[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.interval_seconds * 2))
        with self._lock:
            samples = tuple(self._samples)
            errors = tuple(self._errors)
        utilizations = [item[1] for item in samples]
        memory = [item[2] for item in samples]
        return {
            "source": "nvidia-smi_nvml_device_level",
            "scope": "physical_device_including_co_scheduled_processes",
            "device_id": self.device_id,
            "interval_seconds": self.interval_seconds,
            "sample_count": len(samples),
            "utilization_percent": (
                None
                if not utilizations
                else {
                    "mean": float(sum(utilizations) / len(utilizations)),
                    "min": min(utilizations),
                    "max": max(utilizations),
                }
            ),
            "memory_used_bytes": (
                None
                if not memory
                else {
                    "mean": float(sum(memory) / len(memory)),
                    "min": min(memory),
                    "max": max(memory),
                }
            ),
            "errors": list(errors),
        }


__all__ = ["GpuUtilizationSampler", "StageTimings"]
