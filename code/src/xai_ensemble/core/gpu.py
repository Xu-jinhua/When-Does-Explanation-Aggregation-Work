"""Low-overhead NVIDIA GPU and per-process memory telemetry."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass


class GpuProbeError(RuntimeError):
    """The local NVIDIA telemetry command failed or returned invalid data."""


@dataclass(frozen=True, slots=True)
class GpuDeviceState:
    index: int
    uuid: str
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class GpuProcessState:
    gpu_uuid: str
    pid: int
    used_bytes: int


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    devices: tuple[GpuDeviceState, ...]
    processes: tuple[GpuProcessState, ...]
    captured_at: float

    def device(self, index: int) -> GpuDeviceState:
        for device in self.devices:
            if device.index == index:
                return device
        raise GpuProbeError(f"nvidia-smi did not report configured GPU {index}")


def _mib_to_bytes(value: str) -> int:
    try:
        return int(value.strip()) * 1024 * 1024
    except ValueError as error:
        raise GpuProbeError(f"invalid nvidia-smi memory value: {value!r}") from error


def _is_descendant(pid: int, root_pid: int) -> bool:
    current = pid
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        try:
            fields = open(f"/proc/{current}/stat", encoding="utf-8").read().split()
            current = int(fields[3])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            return False
    return False


class NvidiaSmiProbe:
    """Cache one shared ``nvidia-smi`` snapshot across scheduler threads."""

    def __init__(self, *, cache_seconds: float = 0.2) -> None:
        if cache_seconds < 0:
            raise ValueError("cache_seconds cannot be negative")
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._cached: GpuSnapshot | None = None

    @staticmethod
    def _query(arguments: tuple[str, ...]) -> str:
        try:
            result = subprocess.run(
                ("nvidia-smi", *arguments),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise GpuProbeError(f"nvidia-smi query failed: {error}") from error
        return result.stdout

    def snapshot(self, *, force: bool = False) -> GpuSnapshot:
        now = time.monotonic()
        with self._lock:
            if (
                not force
                and self._cached is not None
                and now - self._cached.captured_at <= self.cache_seconds
            ):
                return self._cached
            gpu_output = self._query(
                (
                    "--query-gpu=index,uuid,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                )
            )
            devices: list[GpuDeviceState] = []
            for line in gpu_output.splitlines():
                if not line.strip():
                    continue
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 4:
                    raise GpuProbeError(f"invalid nvidia-smi GPU row: {line!r}")
                try:
                    index = int(fields[0])
                except ValueError as error:
                    raise GpuProbeError(f"invalid nvidia-smi GPU index: {fields[0]!r}") from error
                devices.append(
                    GpuDeviceState(
                        index=index,
                        uuid=fields[1],
                        total_bytes=_mib_to_bytes(fields[2]),
                        free_bytes=_mib_to_bytes(fields[3]),
                    )
                )
            process_output = self._query(
                (
                    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                )
            )
            processes: list[GpuProcessState] = []
            for line in process_output.splitlines():
                if not line.strip():
                    continue
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 3 or fields[2].upper() == "N/A":
                    continue
                try:
                    pid = int(fields[1])
                except ValueError as error:
                    raise GpuProbeError(f"invalid nvidia-smi process PID: {fields[1]!r}") from error
                processes.append(
                    GpuProcessState(
                        gpu_uuid=fields[0],
                        pid=pid,
                        used_bytes=_mib_to_bytes(fields[2]),
                    )
                )
            self._cached = GpuSnapshot(
                devices=tuple(sorted(devices, key=lambda item: item.index)),
                processes=tuple(processes),
                captured_at=now,
            )
            return self._cached

    def process_memory_bytes(
        self,
        root_pid: int,
        gpu_ids: tuple[int, ...],
    ) -> dict[int, int]:
        snapshot = self.snapshot()
        uuid_to_index = {device.uuid: device.index for device in snapshot.devices}
        requested = set(gpu_ids)
        result = {gpu_id: 0 for gpu_id in gpu_ids}
        for process in snapshot.processes:
            gpu_id = uuid_to_index.get(process.gpu_uuid)
            if gpu_id not in requested:
                continue
            belongs = process.pid == root_pid or _is_descendant(process.pid, root_pid)
            if not belongs:
                try:
                    belongs = os.getpgid(process.pid) == root_pid
                except (ProcessLookupError, PermissionError):
                    belongs = False
            if belongs:
                result[gpu_id] += process.used_bytes
        return result


__all__ = [
    "GpuDeviceState",
    "GpuProbeError",
    "GpuProcessState",
    "GpuSnapshot",
    "NvidiaSmiProbe",
]
