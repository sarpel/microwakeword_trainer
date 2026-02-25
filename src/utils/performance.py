"""
Performance utilities for microwakeword_trainer v2.0
"""

import psutil
import GPUtil
from typing import Dict, Optional


def get_system_info() -> Dict[str, any]:
    """Get system resource information."""
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()

    info = {
        "cpu_count": cpu_count,
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
    }

    # GPU info
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            info["gpu"] = {
                "name": gpus[0].name,
                "memory_total_mb": gpus[0].memoryTotal,
                "memory_free_mb": gpus[0].memoryFree,
                "memory_used_mb": gpus[0].memoryUsed,
                "load_percent": gpus[0].load * 100,
            }
    except Exception:
        pass

    return info


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        gpus = GPUtil.getGPUs()
        return len(gpus) > 0
    except Exception:
        return False


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"
