"""
LLM Router — System Metrics

Collects GPU (nvidia-smi), CPU, RAM stats.
Returns plain dicts suitable for JSON serialization.
"""
import subprocess
import time
import psutil
from typing import Optional


def get_gpu_metrics() -> dict:
    """Query nvidia-smi for RTX 3050 stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            vram_used = int(parts[1]) if parts[1].isdigit() else 0
            vram_total = int(parts[2]) if parts[2].isdigit() else 6144
            return {
                "available": True,
                "name": parts[0],
                "vram_used_mb": vram_used,
                "vram_total_mb": vram_total,
                "vram_pct": round(vram_used / vram_total * 100, 1) if vram_total else 0,
                "utilization_pct": int(parts[3]) if parts[3].isdigit() else 0,
                "temperature_c": int(parts[4]) if parts[4].isdigit() else 0,
                "power_w": float(parts[5]) if _is_float(parts[5]) else 0,
                "power_limit_w": float(parts[6]) if _is_float(parts[6]) else 0,
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return {
        "available": False,
        "name": "RTX 3050 (unavailable)",
        "vram_used_mb": 0,
        "vram_total_mb": 6144,
        "vram_pct": 0,
        "utilization_pct": 0,
        "temperature_c": 0,
        "power_w": 0,
        "power_limit_w": 0,
    }


def get_system_metrics() -> dict:
    """Collect CPU, RAM, GPU metrics."""
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
    cpu_avg = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0

    return {
        "timestamp": time.time(),
        "cpu": {
            "avg_pct": round(cpu_avg, 1),
            "per_core": [round(c, 1) for c in cpu_per_core],
            "core_count": len(cpu_per_core),
            "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else 0,
        },
        "ram": {
            "used_gb": round(ram.used / 1024**3, 2),
            "total_gb": round(ram.total / 1024**3, 2),
            "pct": ram.percent,
            "available_gb": round(ram.available / 1024**3, 2),
        },
        "swap": {
            "used_gb": round(swap.used / 1024**3, 2),
            "total_gb": round(swap.total / 1024**3, 2),
            "pct": swap.percent,
        },
        "gpu": get_gpu_metrics(),
    }


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
