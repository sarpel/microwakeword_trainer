# src/utils/ - Performance & System Utilities

TensorFlow GPU configuration, mixed precision training, threading, and system info.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `performance.py` | 246 | GPU config, mixed precision, threading |

## Key Functions

| Function | Purpose |
|----------|---------|
| `configure_tensorflow_gpu()` | GPU memory growth and limits |
| `configure_mixed_precision()` | Enable FP16 for 2-3x speedup |
| `set_threading_config()` | Inter/intra-op parallelism |
| `get_system_info()` | CPU, memory, GPU stats |

## Usage

```python
from src.utils.performance import configure_tensorflow_gpu, configure_mixed_precision

configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=8192)
configure_mixed_precision(enabled=True)
```

## Notes

- `setup_gpu_environment()` must be called **BEFORE** importing TensorFlow
- Mixed precision requires Tensor Core GPUs (Volta+)
- Memory growth prevents OOM by allocating on demand

## Related Documentation

- [Configuration Reference](../../docs/CONFIGURATION.md)
