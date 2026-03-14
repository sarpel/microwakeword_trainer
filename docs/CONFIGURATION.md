# Configuration Reference

Complete reference for the microwakeword_trainer configuration system.

## Overview

The configuration system uses **12 dataclasses** with YAML-based presets. Configuration is loaded via `config/loader.py` (736 lines) which supports:

- **Preset inheritance**: Start from `fast_test`, `standard`, or `max_quality`
- **Environment variable substitution**: `${VAR:-default}` syntax
- **Custom overrides**: Merge preset with user-provided YAML
- **Validation**: Dataclass constraints enforce valid values

## Configuration Loading Flow

```
1. Load preset YAML (fast_test/standard/max_quality)
2. Apply environment variable substitution
3. Merge with custom override YAML (if provided)
4. Validate dataclass constraints
5. Return FullConfig object
```

## Environment Variable Substitution

Use `${VAR:-default}` syntax in any YAML value:

```yaml
paths:
  positive_dir: "${DATASET_DIR:-./dataset}/positive"
  checkpoint_dir: "${CHECKPOINT_DIR:-./checkpoints}"
```

Available environment variables:
- `DATASET_DIR` - Base directory for datasets
- `DATA_DIR` - Base directory for processed data
- `CHECKPOINT_DIR` - Directory for model checkpoints
- `MODEL_EXPORT_DIR` - Directory for exported models

## Configuration Sections

### 1. HardwareConfig

**File**: `config/loader.py` - Immutable audio frontend parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate_hz` | int | 16000 | Audio sample rate (ESPHome requirement) |
| `mel_bins` | int | 40 | Number of mel-frequency bins |
| `window_size_ms` | int | 30 | FFT window size in milliseconds |
| `window_step_ms` | int | 10 | Hop length between windows |
| `clip_duration_ms` | int | 1000 | Training clip duration |

> **Note**: These values are immutable per ARCHITECTURAL_CONSTITUTION.md

### 2. PathsConfig

**File**: `config/loader.py` - Directory paths

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `positive_dir` | str | "./dataset/positive" | Wake word samples |
| `negative_dir` | str | "./dataset/negative" | Background speech |
| `hard_negative_dir` | str | "./dataset/hard_negative" | False positives |
| `background_dir` | str | "./dataset/background" | Noise/ambient |
| `rir_dir` | str | "./dataset/rirs" | Room impulse responses |
| `processed_dir` | str | "./data/processed" | Preprocessed features |
| `checkpoint_dir` | str | "./checkpoints" | Model checkpoints |
| `export_dir` | str | "./models/exported" | TFLite exports |

### 3. TrainingConfig

**File**: `config/loader.py` - Training hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training_steps` | list[int] | [20000, 10000] | Steps per phase |
| `learning_rates` | list[float] | [0.001, 0.0001] | LR per phase |
| `batch_size` | int | 128 | Training batch size |
| `eval_step_interval` | int | 500 | Steps between validation |
| `steps_per_epoch` | int | 1000 | Approximate steps per epoch |
| `positive_class_weight` | list[float] | [1.0, 1.0] | Weight per phase |
| `negative_class_weight` | list[float] | [20.0, 20.0] | Weight per phase |
| `hard_negative_class_weight` | list[float] | [40.0, 40.0] | Weight per phase |
| `time_mask_max_size` | list[int] | [5, 3] | SpecAugment time mask size |
| `time_mask_count` | list[int] | [2, 1] | Number of time masks |
| `freq_mask_max_size` | list[int] | [3, 2] | SpecAugment freq mask size |
| `freq_mask_count` | list[int] | [2, 1] | Number of freq masks |
| `minimization_metric` | str | "ambient_false_positives_per_hour" | Metric to minimize |
| `target_minimization` | float | 0.2 | Target FAH value |
| `maximization_metric` | str | "average_viable_recall" | Metric to maximize |
| `ambient_duration_hours` | float | 10.0 | Hours for FAH calculation |
| `train_split` | float | 0.8 | Training data fraction |
| `val_split` | float | 0.1 | Validation data fraction |
| `test_split` | float | 0.1 | Test data fraction |
| `split_seed` | int | 42 | RNG seed for splits |
| `strict_content_hash_leakage_check` | bool | true | Verify no train/test leakage |
| `optimizer` | str | "adam" | Optimizer type |
| `label_smoothing` | float | 0.1 | Label smoothing factor |
| `gradient_clipnorm` | float | 5.0 | Gradient clipping norm |
 | `ema_decay` | float | null | EMA decay (null = disabled). Default in `max_quality.yaml`: 0.999. See ARCHITECTURAL_CONSTITUTION.md Article IX for EMA behavior and checkpoint usage. |
| `random_seed` | int | null | Global RNG seed |
| `auto_tune_on_poor_fah` | bool | false | Auto-run mww-autotune if FAH high |

### 4. ModelConfig

**File**: `config/loader.py` - MixedNet architecture

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `architecture` | str | "mixednet" | Model architecture type |
| `first_conv_filters` | int | 32 | Initial Conv2D filters |
| `first_conv_kernel_size` | int | 5 | Initial Conv2D kernel size |
| `stride` | int | 3 | Global stride (immutable) |
| `pointwise_filters` | str | "64,64,64,64" | Filters per MixConv block |
| `mixconv_kernel_sizes` | str | "[5],[7,11],[9,15],[23]" | Kernels per block |
| `repeat_in_block` | str | "1,1,1,1" | Repeats per block |
| `residual_connection` | str | "0,1,1,1" | Residual flags per block |
| `dropout_rate` | float | 0.2 | Dropout rate |
| `l2_regularization` | float | 0.0001 | L2 regularization |

### 5. AugmentationConfig

**File**: `config/loader.py` - Data augmentation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `SevenBandParametricEQ` | float | 0.1 | EQ probability |
| `TanhDistortion` | float | 0.1 | Distortion probability |
| `PitchShift` | float | 0.1 | Pitch shift probability |
| `BandStopFilter` | float | 0.1 | Band stop probability |
| `AddColorNoise` | float | 0.1 | Color noise probability |
| `AddBackgroundNoiseFromFile` | float | 0.3 | Background noise probability |
| `Gain` | float | 1.0 | Gain probability (always) |
| `ApplyImpulseResponse` | float | 0.2 | RIR probability |
| `background_min_snr_db` | float | 0.0 | Min SNR for background |
| `background_max_snr_db` | float | 10.0 | Max SNR for background |
| `min_jitter_s` | float | 0.195 | Min timing jitter |
| `max_jitter_s` | float | 0.205 | Max timing jitter |
| `eq_min_gain_db` | float | -6.0 | EQ min gain |
| `eq_max_gain_db` | float | 6.0 | EQ max gain |
| `distortion_min` | float | 0.1 | Min distortion |
| `distortion_max` | float | 0.5 | Max distortion |
| `pitch_shift_min_semitones` | float | -2.0 | Min pitch shift |
| `pitch_shift_max_semitones` | float | 2.0 | Max pitch shift |
| `band_stop_min_center_freq` | float | 100.0 | Min band stop freq |
| `band_stop_max_center_freq` | float | 5000.0 | Max band stop freq |
| `band_stop_min_bandwidth_fraction` | float | 0.5 | Min bandwidth |
| `band_stop_max_bandwidth_fraction` | float | 1.99 | Max bandwidth |
| `gain_min_db` | float | -3.0 | Min gain |
| `gain_max_db` | float | 3.0 | Max gain |
| `color_noise_min_snr_db` | float | -5.0 | Min color noise SNR |
| `color_noise_max_snr_db` | float | 10.0 | Max color noise SNR |
| `impulse_paths` | list[str] | ["./dataset/rirs"] | RIR directories |
| `background_paths` | list[str] | ["./dataset/background"] | Background dirs |
| `augmentation_duration_s` | float | 3.2 | Target duration |

### 6. PerformanceConfig

**File**: `config/loader.py` - Performance settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_only` | bool | false | Require GPU |
| `mixed_precision` | bool | true | Enable FP16 training |
| `spec_augment_backend` | str | "tf" | Backend for SpecAugment |
| `async_mining` | bool | false | Async hard negative mining |
| `num_workers` | int | 12 | Data loading workers |
| `num_threads_per_worker` | int | 2 | Threads per worker |
| `prefetch_factor` | int | 8 | Prefetch factor |
| `pin_memory` | bool | true | Pin GPU memory |
| `max_memory_gb` | int | 32 | Max RAM usage |
| `inter_op_parallelism` | int | 12 | TF inter-op threads |
| `intra_op_parallelism` | int | 12 | TF intra-op threads |
| `enable_profiling` | bool | true | Enable profiling |
| `profile_every_n_steps` | int | 100 | Profile interval |
| `profile_output_dir` | str | "./profiles" | Profile output |
| `tf_profile_start_step` | int | 100 | TF Profiler start |
| `gpu_memory_log_interval` | int | 1000 | GPU memory log interval |
| `tensorboard_enabled` | bool | true | Enable TensorBoard |
| `tensorboard_log_dir` | str | "./logs" | TensorBoard logs |
| `tensorboard_log_histograms` | bool | true | Log score histograms |
| `tensorboard_log_images` | bool | true | Log images (curves, confusion matrix) |
| `tensorboard_log_pr_curves` | bool | true | Log interactive PR curves |
| `tensorboard_log_graph` | bool | true | Log model graph |
| `tensorboard_log_advanced_scalars` | bool | true | Log advanced scalar metrics |
| `tensorboard_log_weight_histograms` | bool | false | Log weight histograms (slow) |
| `tensorboard_image_interval` | int | 5000 | Steps between image logs |
| `tensorboard_histogram_interval` | int | 5000 | Steps between histogram logs |
| `prefetch_buffer` | int | null | Prefetch buffer (null = AUTOTUNE) |
| `use_tfdata` | bool | true | Use tf.data pipeline |
| `tfdata_cache_dir` | str | null | Cache directory |
| `mmap_readonly` | bool | true | Read-only mmap |
| `tfdata_prefetch_to_device` | bool | true | Prefetch to GPU |
| `tfdata_prefetch_device` | str | "/GPU:0" | Target device |
| `benchmark_pipeline` | bool | false | Benchmark data pipeline |
| `log_throughput` | bool | true | Log throughput |
| `log_throughput_interval` | int | 1000 | Throughput log interval |
| `disable_mmap` | bool | false | Disable memory mapping |

### 7. SpeakerClusteringConfig

**File**: `config/loader.py` - Speaker diversity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | true | Enable clustering |
| `method` | str | "agglomerative" | Clustering algorithm |
| `embedding_model` | str | "speechbrain/spkrec-ecapa-voxceleb" | Model for embeddings |
| `similarity_threshold` | float | 0.72 | Cluster merge threshold |
| `n_clusters` | int | null | Fixed cluster count (null = auto) |
| `leakage_audit_enabled` | bool | true | Check for data leakage |
| `leakage_similarity_threshold` | float | 0.9 | Leakage detection threshold |
| `use_embedding_cache` | bool | true | Cache embeddings |
| `cache_dir` | str | null | Cache directory |
| `batch_size` | int | null | Inference batch size |
| `num_io_workers` | int | 8 | I/O workers |
| `use_mixed_precision` | bool | true | FP16 inference |
| `use_dataloader` | bool | false | Use DataLoader |
| `use_adaptive_clustering` | bool | true | Auto-select algorithm |
| `hdbscan_min_cluster_size` | int | 5 | HDBSCAN min cluster |
| `hdbscan_min_samples` | int | 3 | HDBSCAN min samples |
| `adaptive_threshold_small` | int | 5000 | Small dataset threshold |
| `adaptive_threshold_large` | int | 50000 | Large dataset threshold |

### 8. HardNegativeMiningConfig

**File**: `config/loader.py` - Hard negative mining

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | true | Enable mining |
| `fp_threshold` | float | 0.8 | False positive threshold |
| `max_samples` | int | 5000 | Max hard negatives to keep |
| `mining_interval_epochs` | int | 5 | Mining frequency |
| `collection_mode` | str | "mine_immediately" | "log_only" or "mine_immediately" |
| `log_predictions` | bool | true | Log false predictions |
| `log_file` | str | "logs/false_predictions.json" | Log file path |
| `enable_post_training_mining` | bool | true | Mine after training |
| `mined_subdirectory` | str | "mined" | Subdir for mined files |
| `min_epochs_before_mining` | int | 10 | Minimum epochs before mining |
| `top_k_per_epoch` | int | 100 | Top false positives per epoch |
| `deduplicate_by_hash` | bool | true | Remove duplicates |

### 9. ExportConfig

**File**: `config/loader.py` - TFLite export

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wake_word` | str | "Hey Katya" | Wake word name |
| `author` | str | "Sarpel GURAY" | Model author |
| `website` | str | "..." | Project website |
| `trained_languages` | list[str] | ["en"] | Languages |
| `quantize` | bool | true | Enable quantization |
| `inference_input_type` | str | "int8" | Input dtype (immutable) |
| `inference_output_type` | str | "uint8" | Output dtype (immutable) |
| `probability_cutoff` | float | 0.97 | Detection threshold |
| `sliding_window_size` | int | 5 | Smoothing window |
| `tensor_arena_size` | int | 0 | Arena size in bytes (`0` = auto-calculate from exported TFLite + margin) |
| `minimum_esphome_version` | str | "2024.7.0" | Min ESPHome version |
| `representative_dataset_size` | int | 500 | Calibration samples |
| `representative_dataset_real_size` | int | 2000 | Actual samples to use |
| `arena_size_margin` | float | 1.3 | Arena safety margin |

### 10. PreprocessingConfig

**File**: `config/loader.py` - Audio preprocessing pipeline

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_duration_ms` | float | 300.0 | Minimum clip duration |
| `max_duration_ms` | float | 2000.0 | Maximum clip duration |
| `discarded_dir` | str | "./discarded" | Directory for discarded clips |
| `vad_aggressiveness` | int | 2 | VAD aggressiveness (0-3) |
| `vad_pad_ms` | int | 200 | Silence padding around speech |
| `vad_frame_ms` | int | 30 | VAD frame size |
| `split_max_chunk_ms` | float | 2000.0 | Max chunk size for splitting |
| `split_min_chunk_ms` | float | 500.0 | Min chunk size for splitting |
| `split_target_chunk_ms` | float | 2000.0 | Target chunk size |

### 11. QualityConfig

**File**: `config/loader.py` - Audio quality scoring thresholds

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clip_threshold` | float | 0.001 | Clipping detection threshold |
| `max_clip_ratio` | float | 0.01 | Maximum allowed clipping ratio |
| `discard_bottom_pct` | float | 5.0 | Discard lowest N% by WQI score |
| `min_wqi` | float | 0.0 | Minimum WQI threshold |
| `discarded_quality_dir` | str | "./discarded/quality" | Quality discard directory |
| `min_snr_db` | float | -10.0 | Minimum SNR in dB |
| `vad_speech_threshold` | float | 0.3 | Speech fraction threshold |
| `dnsmos_min_ovrl` | float | 0.0 | Minimum DNSMOS OVRL score |
| `dnsmos_min_sig` | float | 0.0 | Minimum DNSMOS SIG score |
| `dnsmos_cache_dir` | str | "~/.cache/dnsmos" | DNSMOS model cache |

### 12. EvaluationConfig

**File**: `config/loader.py` - Evaluation and metrics configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_threshold` | float | 0.5 | Default probability threshold |
| `n_thresholds` | int | 101 | Number of thresholds for curves |
| `max_fah` | float | 10.0 | Maximum FAH for viable recall |
| `target_fah` | float | 0.5 | Target FAH for metrics |
| `target_recall` | float | 0.95 | Target recall for metrics |
| `gain_window_steps` | int | 1000 | Step window for gain metrics |
| `plateau_window_evals` | int | 5 | Rolling evals for plateau detection |
| `plateau_min_delta` | float | 0.001 | Minimum improvement delta |
| `plateau_slope_eps` | float | 0.0001 | Slope epsilon for plateau |
| `warmup_runs` | int | 10 | Warmup runs for latency |
| `n_latency_runs` | int | 100 | Number of latency runs |

`scripts/evaluate_model.py` consumes these settings and writes `evaluation_artifacts/` output:
- `evaluation_report.json`
- `executive_report.md`
- `executive_report.html`
- PNG plots (ROC/PR/DET/confusion/calibration/threshold/cost)

Use `--output-dir` to choose artifact location, and `--n-thresholds` to override sweep density per run.

### 13. AutoTuningConfig

**File**: `config/loader.py` - Auto-tuning hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `search_eval_fraction` | float | 0.30 (standard/fast_test), 0.35 (max_quality) | Fraction of search data reserved for evaluation during auto-tuning. Prevents train-on-test contamination by ensuring FocusedSampler trains on `search_train` while evaluation uses the held-out `search_eval` split. Higher values provide more robust evaluation but reduce training data for the sampler. |

> **Added in**: Phase 7 auto-tuner fix

## Preset Comparison

### fast_test.yaml (~1 hour)

For: Debugging, CI/CD, smoke tests

```yaml
training:
  training_steps: [2000, 1000]
  batch_size: 32

speaker_clustering:
  enabled: false

hard_negative_mining:
  enabled: false

augmentation:
  # Most augmentations disabled
  SevenBandParametricEQ: 0.0
  TanhDistortion: 0.0
  # ...

performance:
  tensorboard_enabled: false
  enable_profiling: false
```

### standard.yaml (~8 hours)

For: Production training with good quality/speed balance

```yaml
training:
  training_steps: [20000, 10000]
  batch_size: 128

speaker_clustering:
  enabled: true
  method: "agglomerative"

hard_negative_mining:
  enabled: true

augmentation:
  # Full augmentation enabled
  SevenBandParametricEQ: 0.1
  TanhDistortion: 0.1
  # ...

performance:
  tensorboard_enabled: true
  enable_profiling: true
```

### max_quality.yaml (~24 hours)

For: Maximum accuracy

```yaml
training:
  training_steps: [40000, 25000, 10000]  # 3 phases
  batch_size: 64

speaker_clustering:
  enabled: true
  method: "adaptive"

hard_negative_mining:
  enabled: true
  async_mining: true

augmentation:
  # Aggressive augmentation
  SevenBandParametricEQ: 0.20
  TanhDistortion: 0.15
  # ...
```

## Creating Custom Configurations

### Method 1: Override Preset

Create `my_config.yaml`:

```yaml
# Override standard preset
training:
  batch_size: 64  # Smaller for limited VRAM

model:
  first_conv_filters: 20  # Smaller model

export:
  wake_word: "Hey Jarvis"
  author: "Your Name"
```

Load with:
```bash
mww-train --config standard --override my_config.yaml
```

### Method 2: Custom Preset

Create `config/presets/custom.yaml` (copy from standard.yaml, modify values).

### Method 3: Python API

```python
from config.loader import load_full_config

# Load preset with override
config = load_full_config("standard", "my_config.yaml")

# Access values
print(config.training.batch_size)
print(config.model.first_conv_filters)
```

## Common Configuration Patterns

### Limited GPU Memory

```yaml
training:
  batch_size: 32  # Reduce from 128

performance:
  mixed_precision: true  # 2-3x speedup, less memory
  max_memory_gb: 16
```

### Single-Speaker Dataset

```yaml
speaker_clustering:
  enabled: false  # Skip clustering for single speaker
```

### Quick Iteration

```yaml
training:
  training_steps: [5000, 2000]  # Faster convergence
  eval_step_interval: 100  # More frequent validation
```

### Production Quality

```yaml
training:
  training_steps: [40000, 20000, 10000]  # 3 phases
  label_smoothing: 0.1
  gradient_clipnorm: 5.0

speaker_clustering:
## Validation Rules

The config loader enforces these validations:

1. **Hardware**: sample_rate_hz must be 16000
2. **Splits**: train_split + val_split + test_split must equal 1.0
3. **Model**: stride must be 3 (per ARCHITECTURAL_CONSTITUTION.md)
4. **Model**: architecture must be "mixednet"
5. **Training**: training_steps and learning_rates must have same length
6. **Training**: batch_size must be greater than 0
7. **Paths**: All directories must exist or be creatable
8. **Export**: inference_input_type must be "int8", inference_output_type must be "uint8"

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - Model architecture details
- [Training Guide](TRAINING.md) - Training workflow
- [Export Guide](EXPORT.md) - TFLite export process
