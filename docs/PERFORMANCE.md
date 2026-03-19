# Performance Tuning Guide

This document provides performance optimization strategies and configuration guidelines for the microwakeword_trainer project.

## Performance Status

**Overall Performance Grade: C+**

All Critical performance blockers have been addressed. High priority optimizations remain.

## Addressed Performance Issues

### ✅ Fixed (March 2026)

| Issue | Impact | Fix |
|-------|--------|-----|
| PERF-001 | 5-15% throughput loss from prints | ~18 prints → logger |
| PERF-002 | OOM crashes from unbounded cache | Memory-aware cache eviction |
| PERF-003 | 20-40% I/O bottleneck in audio loading | Parallel loading with ThreadPoolExecutor |
| PERF-004 | 10-20% overhead from frontend per call | Cached MicroFrontend instances |
| PERF-005 | 15-25% slowdown from CuPy transfers | Default to TF-native SpecAugment |
| PERF-006 | OOM from unbounded validation metrics | Reservoir sampling |

## Performance Configuration

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 4GB+ VRAM (for training)
- Storage: SSD (for faster I/O)

### Environment Variables

```bash
# TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=3
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_ENABLE_ONEDNN_OPTS=0

# GPU memory management
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Training Performance

### Data Pipeline Optimization

**Config presets (config/presets/):**

| Preset | Batch Size | GPU Memory | Throughput |
|-------|------------|------------|------------|
| `fast_test` | 16 | 2GB | Low (debugging) |
| `standard` | 32 | 4GB | Medium |
| `max_quality` | 64 | 8GB+ | High |

**Optimization strategies:**

1. **Use tf.data pipeline** (default)
   - Automatic prefetching
   - Parallel feature extraction
   - In-graph SpecAugment

2. **Enable parallel audio loading**
   - Automatically enabled in `src/data/ingestion.py`
   - Uses ThreadPoolExecutor with 32 workers

3. **Optimize I/O patterns**
   - Use RAM disk for dataset cache if possible
   - Disable swap during training

### Memory Management

**RaggedMmap Cache (src/data/dataset.py):**

```python
# Configuration
max_cache_memory_mb: int = 1024  # Default: 1GB

# Tuning for large datasets:
max_cache_memory_mb: int = 2048  # 2GB for large datasets
max_cache_memory_mb: int = 512   # 512MB for memory-constrained
```

**Validation Metrics:**

```python
# In config/presets/[preset].yaml under training:
validation:
  max_samples: 10000  # Limits memory usage for validation
```

### GPU Utilization

**Mixed Precision Training:**

```yaml
# In config preset under performance:
performance:
  mixed_precision: true  # Default: false (15-25% speedup)
```

**SpecAugment Backend:**

```yaml
# In config preset under augmentation:
spec_augment_backend: "tf"  # Default: "tf" (fastest)
# "cupy" is slower due to CPU-GPU transfers
```

## Export Performance

### TFLite Export Optimization

**Calibration data quality:**

```bash
# Use real data for best quantization accuracy
mww-export --checkpoint best_weights.weights.h5 \
  --data-dir /path/to/train \
  --output models/exported/
```

**Quantization settings:**

```yaml
# In export config under optimization:
quantization:
  optimizer: "default"  # Full integer quantization
  target_ops: [INT8]     # INT8 activations
  representative_dataset_real_size: 4000
```

### Export Checklist

- [ ] Use real calibration data (not random)
- [ ] Verify with `python scripts/verify_esphome.py`
- [ ] Test on target hardware before deployment
- [ ] Measure latency on target device

## Profiling Tools

### Built-in Profiling

```bash
# Profile training step
python -m cProfile -o profile.stats -m src.training.trainer

# Profile with snakeviz
pip install snakeviz
python -m cProfile -o - | snakeviz -
```

### Performance Monitoring

**TensorBoard:**

```python
# Automatically enabled in standard preset
# View at: http://localhost:6006
tensorboard --logdir logs/
```

**Progress tracking:**

- Rich console output with real-time metrics
- ETA calculation based on step/time
- Phase-based progress tracking

## Bottleneck Analysis

### Common Bottlenecks

1. **I/O Bound**
   - Symptom: GPU utilization < 50%
   - Fix: Increase `batch_size` in config
   - Fix: Use faster storage (SSD)

2. **CPU Bound**
   - Symptom: GPU utilization > 90%, CPU 100%
   - Fix: Enable `mixed_precision: true`
   - Fix: Reduce audio file count

3. **Memory Bound**
   - Symptom: OOM crashes
   - Fix: Reduce `batch_size`
   - Fix: Reduce `max_samples` in validation

### Performance Debugging

**Enable performance logging:**

```yaml
# In config preset:
performance:
  log_level: DEBUG  # Shows detailed timing info
```

**Check cache hit rate:**

```python
# RaggedMmap cache statistics (after training):
logger.info(f"Cache hits: {cache_hits}, misses: {cache_misses}")
```

## Optimization Priority

### High Impact, Low Effort

| Change | Impact | Effort | Status |
|--------|--------|-------|--------|
| Enable `mixed_precision: true` | 15-25% faster | 1 line | ✅ Config |
| Set `spec_augment_backend: "tf"` | 15-25% faster | 1 line | ✅ Fixed |
| Parallel audio loading | 20-40% faster | 10 lines | ✅ Fixed |

### Medium Impact, Medium Effort

| Change | Impact | Effort | Status |
|--------|--------|-------|--------|
| Frontend caching | 10-20% faster | 5 lines | ✅ Fixed |
| Increase batch size | Variable | 1 line config | ⏳ Config dependent |

### Low Impact, High Effort

| Change | Impact | Effort | Status |
|--------|--------|-------|--------|
| Refactor large files | 5-10% faster | 40h+ | ⏳ Pending |

## Monitoring in Production

### Key Metrics to Track

1. **Training throughput** - Steps/second
2. **GPU utilization** - Target: 80-90%
3. **Memory usage** - Monitor for OOM
4. **Latency** - TFLite inference time
5. **FAH/Recall** - Model quality metrics

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| GPU memory | >90% | >98% |
| Training throughput | <10 steps/s | <5 steps/s |
| FAH (False Alarms/Hour) | >5.0 | >10.0 |
| Recall | <80% | <70% |

---

**Last Updated:** March 19, 2026
**Next Review:** After next performance optimization
