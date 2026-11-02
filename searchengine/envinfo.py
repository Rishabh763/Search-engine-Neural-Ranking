"""Hardware/software fingerprint attached to every benchmark and evaluation report.

Latency and throughput numbers are meaningless without the machine they came
from: the same reranker runs ~2 s/query on a 6-core laptop CPU and ~20 ms on a
desktop GPU. Every report therefore carries this block, so results from
different machines can sit side by side instead of being confused.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _gpu_adapters() -> list[str]:
    """Graphics adapters as the OS sees them (catches AMD/Intel GPUs that torch can't use)."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=8,
            ).stdout
            return [x.strip() for x in out.splitlines() if x.strip()]
        if sys.platform == "darwin":
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=8).stdout
            return [ln.split(":", 1)[1].strip() for ln in out.splitlines() if "Chipset Model" in ln]
    except Exception:
        pass
    return []


def _version(module: str) -> str | None:
    try:
        mod = __import__(module)
        v = getattr(mod, "__version__", "unknown")
        return ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
    except Exception:
        return None


def environment_info() -> dict:
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        batt = psutil.sensors_battery()
        # Laptops on battery often throttle CPU hard; record it so slow runs are explainable.
        info["on_battery"] = None if batt is None else (not batt.power_plugged)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())  # Apple-silicon GPU (Metal)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    adapters = _gpu_adapters()
    if adapters:
        info["gpu_adapters"] = adapters
    try:
        import onnxruntime as ort

        info["onnx_providers"] = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
    except Exception:
        pass
    for mod in ("numpy", "transformers", "sentence_transformers", "onnxruntime", "faiss", "elasticsearch"):
        v = _version(mod)
        if v is not None:
            info[mod] = v
    return info


def short_label(info: dict) -> str:
    """One line for printing: 'Intel i5-1235U, 10 cores (on battery), no GPU'."""
    cores = info.get("physical_cores") or info.get("logical_cores")
    batt = " (on battery)" if info.get("on_battery") else ""
    if info.get("gpu"):
        gpu = info["gpu"]
    elif info.get("mps_available"):
        gpu = "Apple GPU (MPS)"
    elif "DmlExecutionProvider" in info.get("onnx_providers", []):
        gpu = f"{(info.get('gpu_adapters') or ['GPU'])[0]} via DirectML"
    else:
        gpu = "no GPU usable"
    return f"{info.get('cpu', '?')}, {cores} cores{batt}, {gpu}"
