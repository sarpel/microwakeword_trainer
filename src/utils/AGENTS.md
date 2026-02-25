# performance.py

## Description

TensorFlow GPU configuration, mixed precision training, threading, and system info utilities.

## Key Functions

| Function | Purpose |
|----------|---------|
| `configure_tensorflow_gpu()` | Configure GPU memory growth and limits |
| `configure_mixed_precision()` | Enable FP16 training for 2-3x speedup |
| `set_threading_config()` | Set inter/intra-op parallelism threads |
| `get_system_info()` | Get CPU, memory, GPU stats |
| `check_gpu_available()` | Check if GPU is present |
| `setup_gpu_environment()` | Set CUDA/TF env vars before import |

## Notes

- `setup_gpu_environment()` must be called BEFORE importing TensorFlow
- Mixed precision only effective on Tensor Core GPUs (Volta+)
- Memory growth prevents OOM by allocating on demand

## Canonical Mapping / Duplication Checklist

Quick repository check: `src/training/gpu_monitor.py`, `src/training/memory_monitor.py`, and `src/training/optimizations.py` are currently **not present** in this codebase, so `performance.py` remains the canonical location for these TF/system utilities.

Current function ownership:

- `configure_tensorflow_gpu()` → `src/utils/performance.py` (canonical, TensorFlow-specific)
- `check_gpu_available()` → `src/utils/performance.py` (canonical, GPUtil-based availability)
- `get_system_info()` → `src/utils/performance.py` (canonical, psutil + GPUtil)
- Global GPU/CPU tuning (`configure_mixed_precision`, `set_threading_config`, `setup_gpu_environment`) → `src/utils/performance.py`

If training-specific monitor/optimization modules are introduced later, prefer delegating from `performance.py` wrappers instead of duplicating behavior.
