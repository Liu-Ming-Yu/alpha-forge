"""Host hardware + process status for the operator console System dashboard.

Reads live CPU / memory / disk / process metrics via ``psutil`` and GPU metrics
via ``nvidia-smi`` (falling back to ``torch.cuda`` for VRAM). Everything is
guarded so the endpoint degrades to whatever is available rather than failing.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess  # noqa: S404 - only used for the fixed nvidia-smi query below
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

_NVIDIA_SMI_QUERY = "name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"
# GPU metrics change slowly relative to the dashboard poll cadence (down to 1 s),
# and nvidia-smi spawns a process each call — cache the read for a short TTL.
_GPU_TTL_SECONDS = 2.0
_gpu_cache: tuple[float, list[dict[str, Any]]] | None = None
_gpu_lock = threading.Lock()


def _int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def _gpus_via_nvidia_smi() -> list[dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            [smi, f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
    except Exception:  # noqa: BLE001 - missing/slow GPU tooling must not break the endpoint
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_used_mb": _int(parts[1]),
                "memory_total_mb": _int(parts[2]),
                "utilization_pct": _int(parts[3]),
                "temperature_c": _int(parts[4]),
                "power_w": _int(parts[5]) if len(parts) > 5 else None,
            }
        )
    return gpus


def _gpus_via_torch() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        out: list[dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out.append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_used_mb": int((total - free) / 1024 / 1024),
                    "memory_total_mb": int(total / 1024 / 1024),
                    "utilization_pct": None,
                    "temperature_c": None,
                    "power_w": None,
                }
            )
        return out
    except Exception:  # noqa: BLE001
        return []


def _gpus_cached() -> list[dict[str, Any]]:
    global _gpu_cache
    now = time.monotonic()
    with _gpu_lock:
        if _gpu_cache is not None and now - _gpu_cache[0] < _GPU_TTL_SECONDS:
            return _gpu_cache[1]
    gpus = _gpus_via_nvidia_smi() or _gpus_via_torch()
    with _gpu_lock:
        _gpu_cache = (now, gpus)
    return gpus


def system_status(settings: PlatformSettings) -> dict[str, Any]:
    """Return a JSON-able snapshot of host hardware + this process."""
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "hostname": platform.node(),
        "gpus": _gpus_cached(),
    }

    try:
        import psutil
    except Exception:  # noqa: BLE001 - psutil optional; report what we can
        payload["psutil_available"] = False
        return payload

    payload["psutil_available"] = True
    payload["cpu"] = {
        "percent": psutil.cpu_percent(interval=None),
        "per_core": psutil.cpu_percent(interval=None, percpu=True),
        "logical": psutil.cpu_count(),
        "physical": psutil.cpu_count(logical=False),
    }
    vm = psutil.virtual_memory()
    payload["memory"] = {
        "total": int(vm.total),
        "used": int(vm.used),
        "available": int(vm.available),
        "percent": float(vm.percent),
    }
    root = settings.storage.object_store_root or "."
    target = root if os.path.isdir(root) else "."
    try:
        usage = shutil.disk_usage(target)
        payload["disk"] = {
            "total": int(usage.total),
            "used": int(usage.used),
            "free": int(usage.free),
            "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
        }
    except Exception:  # noqa: BLE001
        payload["disk"] = None

    try:
        proc = psutil.Process()
        with proc.oneshot():
            payload["process"] = {
                "pid": proc.pid,
                "rss": int(proc.memory_info().rss),
                "threads": proc.num_threads(),
                "cpu_percent": proc.cpu_percent(interval=None),
                "create_time": proc.create_time(),
            }
        payload["boot_time"] = psutil.boot_time()
    except Exception:  # noqa: BLE001
        payload["process"] = None
    return payload
