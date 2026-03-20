# Environment Variables Reference

Complete reference for all environment variables used in the project.

<!-- AUTO-GENERATED: Generated from config presets, source code analysis, and config/loader.py -->
<!-- Last updated: 2026-03-20 -->

---

## Overview

The project uses environment variables for:
- **Configuration paths** — Override default dataset, checkpoint, and export directories
- **GPU setup** — Control GPU visibility, memory allocation, and device ordering
- **TensorFlow optimization** — Configure logging, XLA flags, and oneDNN options
- **Debugging** — Enable detailed tracebacks and TensorFlow logs

**No .env file is required** — all variables are set via shell environment or in YAML configs using `${VAR:-default}` syntax.

---

## Configuration Path Variables

Used in config presets (`fast_test.yaml`, `standard.yaml`, `max_quality.yaml`) with `${VAR:-default}` syntax.

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `DATASET_DIR` | No | `./dataset` | Base directory for audio datasets (positive, negative, hard_negative, background, rirs) | `export DATASET_DIR=/data/audio_dataset` |
| `DATA_DIR` | No | `./data` | Base directory for processed data and caches | `export DATA_DIR=/tmp/processed` |
| `CHECKPOINT_DIR` | No | `./models/checkpoints` | Directory for saving model checkpoints during training | `export CHECKPOINT_DIR=/models/checkpoints` |
| `MODEL_EXPORT_DIR` | No | `./models/exported` | Directory for exported TFLite models | `export MODEL_EXPORT_DIR=/models/production` |

### Usage in YAML Configs

```yaml
paths:
  dataset_dir: ${DATASET_DIR:-./dataset}
  data_dir: ${DATA_DIR:-./data}
  checkpoints_dir: ${CHECKPOINT_DIR:-./models/checkpoints}
  model_export_dir: ${MODEL_EXPORT_DIR:-./models/exported}
```

### Path Override Examples

**Standard production setup:**
```bash
export DATASET_DIR=/data/audio
export DATA_DIR=/data/processed
export CHECKPOINT_DIR=/models/checkpoints
export MODEL_EXPORT_DIR=/models/production

mww-train --config standard
```

**Fast test with temporary directories:**
```bash
export DATASET_DIR=/tmp/test_dataset
export DATA_DIR=/tmp/test_data
export CHECKPOINT_DIR=/tmp/test_checkpoints
export MODEL_EXPORT_DIR=/tmp/test_models

mww-train --config fast_test
```

---

## GPU and Performance Variables

Control GPU visibility, memory allocation, and device ordering.

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | No | (all GPUs) | Limit which GPUs are visible to TensorFlow (0-indexed, comma-separated) | `export CUDA_VISIBLE_DEVICES=0,1` |
| `TF_FORCE_GPU_ALLOW_GROWTH` | No | (not set) | Allow GPU memory to grow dynamically instead of allocating all at startup | `export TF_FORCE_GPU_ALLOW_GROWTH=true` |
| `TF_GPU_ALLOCATOR` | No | (not set) | GPU memory allocator to use (e.g., cuda_malloc_async for reduced fragmentation) | `export TF_GPU_ALLOCATOR=cuda_malloc_async` |
| `CUDA_DEVICE_ORDER` | No | (not set) | Order in which CUDA devices are numbered (PCI_BUS_ID recommended for consistency) | `export CUDA_DEVICE_ORDER=PCI_BUS_ID` |

### GPU Setup Examples

**Single GPU training:**
```bash
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

mww-train --config standard
```

**Multi-GPU training (using first 2 GPUs):**
```bash
export CUDA_VISIBLE_DEVICES=0,1
export TF_GPU_ALLOCATOR=cuda_malloc_async

mww-train --config max_quality
```

**Memory-optimized training (reduces OOM errors):**
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

mww-train --config standard
```

---

## TensorFlow Optimization Variables

Configure TensorFlow logging, XLA compilation, and oneDNN optimizations.

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `TF_CPP_MIN_LOG_LEVEL` | No | (not set) | TensorFlow C++ logging level: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL | `export TF_CPP_MIN_LOG_LEVEL=3` |
| `TF_XLA_FLAGS` | No | (not set) | XLA (Accelerated Linear Algebra) compiler flags | `export TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false` |
| `TF_ENABLE_ONEDNN_OPTS` | No | (not set) | Enable oneDNN (MKL-DNN) optimizations for CPU operations | `export TF_ENABLE_ONEDNN_OPTS=0` |
| `TF_DETERMINISTIC_OPS` | No | (not set) | Force deterministic operations for reproducibility (reduces performance) | `export TF_DETERMINISTIC_OPS=1` |

### TensorFlow Logging Examples

**Suppress TensorFlow info messages (recommended):**
```bash
export TF_CPP_MIN_LOG_LEVEL=3  # Only show errors and fatal

mww-train --config standard
```

**Enable full TensorFlow logging for debugging:**
```bash
export TF_CPP_MIN_LOG_LEVEL=0  # Show all logs
export MWW_DEBUG_TRACEBACKS=1

mww-train --config fast_test
```

**Disable XLA (if encountering XLA-related errors):**
```bash
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false

mww-export --checkpoint models/checkpoints/best_weights.weights.h5 --output models/exported/
```

---

## Debugging Variables

Enable enhanced debugging output.

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `MWW_DEBUG_TRACEBACKS` | No | (not set) | Enable detailed traceback information in logging (shows full stack traces) | `export MWW_DEBUG_TRACEBACKS=1` |

### Debugging Examples

**Enable detailed tracebacks for error diagnosis:**
```bash
export MWW_DEBUG_TRACEBACKS=1
export TF_CPP_MIN_LOG_LEVEL=0

mww-train --config fast_test
```

---

## Reproducibility Setup

For reproducible training runs (reduced performance):

```bash
# Force deterministic operations
export TF_DETERMINISTIC_OPS=1

# Consistent GPU device ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Train with reproducible seed (set in config)
mww-train --config standard
```

---

## Production Deployment Setup

Optimal configuration for production training:

```bash
# GPU setup
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# TensorFlow optimization
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=1

# Paths
export DATASET_DIR=/data/audio
export DATA_DIR=/data/processed
export CHECKPOINT_DIR=/models/checkpoints
export MODEL_EXPORT_DIR=/models/production

# Run training
mww-train --config max_quality
```

---

## Environment Variable Sources

| Variable | Read By |
|----------|-----------|
| `DATASET_DIR`, `DATA_DIR`, `CHECKPOINT_DIR`, `MODEL_EXPORT_DIR` | `config/loader.py` (via ${VAR:-default} in YAML) |
| `CUDA_VISIBLE_DEVICES`, `TF_FORCE_GPU_ALLOW_GROWTH`, `TF_GPU_ALLOCATOR` | `src/utils/performance.py` |
| `CUDA_DEVICE_ORDER` | `src/training/trainer.py` |
| `TF_CPP_MIN_LOG_LEVEL`, `TF_XLA_FLAGS`, `TF_ENABLE_ONEDNN_OPTS` | `src/export/tflite.py`, `src/training/trainer.py`, `src/tuning/cli.py` |
| `TF_DETERMINISTIC_OPS` | `src/utils/seed.py`, `src/data/tfdata_pipeline.py`, `src/tuning/orchestrator.py` |
| `MWW_DEBUG_TRACEBACKS` | `src/utils/logging_config.py` |

---

## Related Documentation

- [Configuration Reference](CONFIGURATION.md) — Complete config system documentation
- [Training Guide](TRAINING.md) — Training workflow and GPU setup
- [Performance Guide](PERFORMANCE.md) — Performance tuning and optimization
- [Troubleshooting](TROUBLESHOOTING.md) — Common issues and solutions
