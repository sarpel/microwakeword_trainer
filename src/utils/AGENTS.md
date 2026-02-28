# src/utils/ - Performance & System Utilities

## Overview

TensorFlow GPU configuration, mixed precision training, threading, and system info utilities.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `performance.py` | 246 | GPU config, mixed precision, threading, system info |
| `__init__.py` | 8 | Package init |

## Key Functions

| Function | Purpose |
|----------|---------|
| `configure_tensorflow_gpu()` | Configure GPU memory growth and limits |
| `configure_mixed_precision()` | Enable FP16 training for 2-3x speedup |
| `set_threading_config()` | Set inter/intra-op parallelism threads |
| `get_system_info()` | Get CPU, memory, GPU stats (psutil + GPUtil) |
| `check_gpu_available()` | Check if GPU is present (GPUtil-based) |
| `setup_gpu_environment()` | Set CUDA/TF env vars before import |
| `format_bytes()` | Human-readable byte formatting |

## Usage

```python
from src.utils.performance import configure_tensorflow_gpu, configure_mixed_precision

# MUST call before importing TensorFlow
setup_gpu_environment()

# After TF import
configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=8192)
configure_mixed_precision(enabled=True)
set_threading_config(inter_op_parallelism=16, intra_op_parallelism=16)
```

## Notes

- `setup_gpu_environment()` must be called **BEFORE** importing TensorFlow
- Mixed precision only effective on Tensor Core GPUs (Volta+)
- Memory growth prevents OOM by allocating on demand

## Canonical Mapping / Duplication Checklist

`src/training/gpu_monitor.py`, `src/training/memory_monitor.py`, and `src/training/optimizations.py` are **not present** in this codebase — `performance.py` remains the canonical location for all TF/system utilities.

Current function ownership:

- `configure_tensorflow_gpu()` → `src/utils/performance.py` (canonical, TensorFlow-specific)
- `check_gpu_available()` → `src/utils/performance.py` (canonical, GPUtil-based availability)
- `get_system_info()` → `src/utils/performance.py` (canonical, psutil + GPUtil)
- Global GPU/CPU tuning (`configure_mixed_precision`, `set_threading_config`, `setup_gpu_environment`) → `src/utils/performance.py`

If training-specific monitor/optimization modules are introduced later, prefer delegating from `performance.py` wrappers instead of duplicating behavior.
