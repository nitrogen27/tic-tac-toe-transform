"""GPU detection and lightweight runtime telemetry for WebSocket monitoring."""

from __future__ import annotations

import csv
import shutil
import subprocess
from typing import Any


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == "[N/A]":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value == "[N/A]":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _query_nvidia_smi() -> dict[str, Any] | None:
    if not shutil.which("nvidia-smi"):
        return None

    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
    if not rows:
        return None

    row = rows[0]
    if len(row) < 10:
        return None

    return {
        "timestamp": row[0].strip(),
        "utilizationGpu": _parse_int(row[1]),
        "utilizationMemory": _parse_int(row[2]),
        "memoryUsedMB": _parse_int(row[3]),
        "memoryTotalMB": _parse_int(row[4]),
        "powerDrawW": _parse_float(row[5]),
        "powerLimitW": _parse_float(row[6]),
        "clockSmMHz": _parse_int(row[7]),
        "clockMemMHz": _parse_int(row[8]),
        "temperatureC": _parse_int(row[9]),
    }


def get_gpu_info() -> dict:
    """Return GPU info matching the legacy server protocol."""
    try:
        import torch

        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
                telemetry = _query_nvidia_smi() or {}
                allocated = 0
                reserved = 0
                try:
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                except Exception:
                    pass

                vram = {
                    "total": total,
                    "totalMB": round(total / 1024 / 1024),
                    "allocated": allocated,
                    "allocatedMB": round(allocated / 1024 / 1024),
                    "reserved": reserved,
                    "reservedMB": round(reserved / 1024 / 1024),
                }
                if telemetry.get("memoryUsedMB") is not None:
                    vram["usedMB"] = telemetry["memoryUsedMB"]
                if telemetry.get("memoryTotalMB") is not None:
                    vram["totalMB"] = telemetry["memoryTotalMB"]
                return {
                    "available": True,
                    "backend": "cuda",
                    "name": props.name,
                    "vram": vram,
                    "telemetry": telemetry,
                }
            except Exception:
                pass
    except ImportError:
        pass
    return {"available": False, "backend": "cpu", "name": "CPU", "vram": {}}
