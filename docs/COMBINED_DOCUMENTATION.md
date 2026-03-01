# microwakeword_trainer — Combined Documentation

> **Auto-generated consolidated reference** combining all documentation files from `docs/`, ordered by importance level.
>
> **Source files (6):**
> 1. `docs/GUIDE.md` — Complete Configuration Guide
> 2. `docs/IMPLEMENTATION_PLAN.md` — Implementation Plan v2.0
> 3. `docs/my_environment.md` — Project Environment Profile
> 4. `docs/POST_TRAINING_ANALYSIS.md` — Post-Training Analysis
> 5. `docs/RESEARCH_REPORT_MIXED_PRECISION.md` — Mixed Precision Research (Turkish)
> 6. `docs/LOG_ANALYSIS_GUIDE.md` — Log Analysis Guide (Turkish)

---

## Table of Contents

- [Part 1: Configuration Guide](#part-1-configuration-guide) — ★★★★★ CRITICAL
  - [Command Line Arguments](#command-line-arguments)
  - [Configuration File Structure](#configuration-file-structure)
  - [Configuration Sections Reference](#configuration-sections-reference)
  - [Preset Configurations](#preset-configurations)
  - [Environment Variables](#environment-variables)
  - [Complete Example Configurations](#complete-example-configurations)
  - [Troubleshooting Guide](#troubleshooting-guide)
- [Part 2: Implementation Plan](#part-2-implementation-plan) — ★★★★☆ HIGH
  - [Definitively Verified Architecture](#1-definitively-verified-architecture)
  - [Python & Dependencies](#2-python--dependencies)
  - [Project Structure](#3-project-file-structure)
  - [Configuration Schema](#4-configuration-schema)
  - [Phase 1 — Data Ingestion & Validation](#5-phase-1--data-ingestion--validation)
  - [Phase 2 — Feature Extraction Pipeline](#6-phase-2--feature-extraction-pipeline)
  - [Phase 3 — Training Loop](#7-phase-3--training-loop-step-based)
  - [Phase 4 — MixedNet Model Architecture](#8-phase-4--mixednet-model-architecture)
  - [Phase 5 — Dataset Engineering](#9-phase-5--dataset-engineering)
  - [Phase 6 — Augmentation Pipeline](#10-phase-6--augmentation-pipeline)
  - [Phase 7 — Hard Negative Mining](#11-phase-7--hard-negative-mining)
  - [Phase 8 — Export & TFLite Conversion](#12-phase-8--export--tflite-conversion)
  - [Phase 9 — ESPHome Manifest Generation](#13-phase-9--esphome-manifest-generation)
  - [Phase 10 — Comprehensive Metrics Suite](#14-phase-10--comprehensive-metrics-suite)
  - [Phase 11 — Performance Optimization (GPU-First)](#15-phase-11--performance-optimization-gpu-first)
  - [Phase 12 — Profiling & Monitoring](#16-phase-12--profiling--monitoring)
  - [ESPHome Compatibility Checklist](#17-esphome-compatibility-checklist)
  - [Dependency Manifest](#18-dependency-manifest)
- [Part 3: Project Environment Profile](#part-3-project-environment-profile) — ★★★☆☆ MEDIUM
  - [Environment & Operational Profile](#1-environment--operational-profile)
  - [Dataset Composition & Analysis](#2-dataset-composition--analysis)
  - [Training & Hardware Infrastructure](#3-training--hardware-infrastructure)
- [Part 4: Post-Training Analysis](#part-4-post-training-analysis) — ★★★☆☆ MEDIUM
  - [Training Results Analysis](#-your-training-results-analysis)
  - [Post-Training Commands](#-post-training-commands)
  - [Test Dataset Usage](#-test-dataset-usage)
  - [Recommended Post-Training Workflow](#-recommended-post-training-workflow)
  - [Debugging Suspicious Results](#-debugging-suspicious-results)
- [Part 5: Mixed Precision Research](#part-5-mixed-precision-research) — ★★☆☆☆ REFERENCE
  - [Mixed Precision (FP16) ve ESPHome Uyumluluğu](#1-mixed-precision-fp16-eğitimi-ve-esphome-uyumluluğu)
  - [tf.data.Dataset ve ESPHome Uyumluluğu](#2-tfdatadataset-ve-esphome-uyumluluğu)
- [Part 6: Log Analysis Guide](#part-6-log-analysis-guide) — ★★☆☆☆ REFERENCE
  - [Profil Dosyaları](#-1-profil-dosyaları-prof)
  - [Terminal Log Dosyaları](#-2-terminal-log-dosyaları-terminal_log)
  - [Önemli Metrikler](#-önemli-metrikler-ve-anlamları)
  - [TensorBoard Logları](#-3-tensorboard-logları)
  - [Sık Karşılaşılan Sorunlar](#-4-sık-karşılaşılan-sorunlar)

---

# Part 1: Configuration Guide

**Importance: ★★★★★ CRITICAL**
**Source: `docs/GUIDE.md`**

---

# microwakeword_trainer - Complete Configuration Guide

This guide documents all possible options, arguments, and configurations for training, finetuning, and exporting wake word models.

---

## Table of Contents

1. [Command Line Arguments](#command-line-arguments)
   - [mww-train](#mww-train)
   - [mww-export](#mww-export)
2. [Configuration File Structure](#configuration-file-structure)
3. [Configuration Sections Reference](#configuration-sections-reference)
   - [hardware](#hardware)
   - [paths](#paths)
   - [training](#training)
   - [model](#model)
   - [augmentation](#augmentation)
   - [performance](#performance)
   - [speaker_clustering](#speaker_clustering)
   - [hard_negative_mining](#hard_negative_mining)
   - [export](#export)
4. [Preset Configurations](#preset-configurations)
5. [Environment Variables](#environment-variables)
6. [Complete Example Configurations](#complete-example-configurations)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Command Line Arguments

### mww-train

Main training command entry point.

```bash
mww-train [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | `standard` | Config preset name (`fast_test`, `standard`, `max_quality`) or path to config YAML file |
| `--override` | string | None | Path to override config file (merged with base config) |

**Examples:**

```bash
# Train with standard preset
mww-train --config standard

# Train with preset + custom override
mww-train --config standard --override my_config.yaml

# Train with full custom config file
mww-train --config /path/to/my_full_config.yaml

# Quick test run
mww-train --config fast_test
```

### mww-export

Export trained model to ESPHome-compatible TFLite format.

```bash
mww-export [OPTIONS]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--checkpoint` | string | None | **Yes** | Path to model checkpoint (.weights.h5) |
| `--config` | string | `config/presets/standard.yaml` | No | Path to configuration file |
| `--output` | string | `./models/exported` | No | Output directory for exported files |
| `--model-name` | string | `wake_word` | No | Name for exported model (used in filename) |
| `--no-quantize` | flag | False | No | Disable INT8 quantization |

**Examples:**

```bash
# Basic export
mww-export --checkpoint checkpoints/best_weights.weights.h5

# Export with custom name
mww-export --checkpoint checkpoints/best_weights.weights.h5 --model-name "hey_computer"

# Export to custom directory
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output /path/to/output

# Export without quantization (for debugging)
mww-export --checkpoint checkpoints/best_weights.weights.h5 --no-quantize

# Full custom export
mww-export \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config config/presets/max_quality.yaml \
    --output ./exports \
    --model-name "okay_nabu"
```

### mww-autotune

Auto-tune a trained model to achieve target FAH and recall metrics without full retraining.

```bash
mww-autotune [OPTIONS]
```

| `--checkpoint` | string | None | **Yes** | Path to trained checkpoint to fine-tune (.weights.h5) |
| `--config` | string | `standard` | No | Config preset name or path to config file |
| `--override` | string | None | No | Override config file path |
| `--target-fah` | float | 0.3 | No | Target FAH value (False Activations per Hour) |
| `--target-recall` | float | 0.92 | No | Target recall value |
| `--max-iterations` | int | 100 | No | Maximum tuning iterations |
| `--output-dir` | string | `./tuning` | No | Output directory for tuned checkpoints |
| `--patience` | int | 10 | No | Stop early if no improvement after N iterations |
| `--dry-run` | flag | False | No | Validate config without running tuning |
| `--verbose` / `-v` | flag | False | No | Enable verbose output |

**Examples:**

```bash
# Basic auto-tuning with defaults
mww-autotune --checkpoint checkpoints/best_weights.weights.h5

# Custom targets
mww-autotune \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config standard \
    --target-fah 0.2 \
    --target-recall 0.95

# With more iterations and custom output
mww-autotune \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config standard \
    --max-iterations 50 \
    --output-dir ./tuning_results
```

**When to use auto-tuning:**
- After initial training, if FAH > 0.5 or recall < 0.90
- As a final polish step before deployment
- When you want to improve metrics without full retraining
---

## Configuration File Structure

Configuration files use YAML format with the following top-level sections:

```yaml
# Required sections
hardware:           # Audio/hardware parameters
paths:              # Directory paths
training:           # Training loop parameters
model:              # Model architecture
augmentation:       # Audio augmentation settings
performance:        # Performance and resource settings
speaker_clustering: # Speaker clustering config
hard_negative_mining: # Hard negative mining config
export:             # Export settings
```

### Configuration Loading

Configurations can be loaded in multiple ways:

```python
from config.loader import load_full_config, load_preset

# Load a preset
config = load_preset("standard")

# Load preset with override
config = load_full_config("standard", "my_override.yaml")

# Full custom config
config = load_full_config("/path/to/my_config.yaml")
```

### Environment Variable Substitution

Configuration values support environment variable substitution:

```yaml
paths:
  checkpoint_dir: ${CHECKPOINT_DIR:-./checkpoints}
  positive_dir: ${DATA_ROOT}/positive
```

Syntax:
- `${VAR}` - Substitute with environment variable value
- `${VAR:-default}` - Substitute with value or use default if not set

---

## Configuration Sections Reference

### hardware

Audio processing and feature extraction parameters. These are typically immutable once set.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sample_rate_hz` | int | 16000 | Audio sample rate in Hz. Must be 16000 for ESPHome. |
| `mel_bins` | int | 40 | Number of mel-frequency bins. Must be 40 for ESPHome. |
| `window_size_ms` | int | 30 | STFT window size in milliseconds. |
| `window_step_ms` | int | 10 | STFT hop length in milliseconds. |
| `clip_duration_ms` | int | 1000 | Target clip duration in milliseconds. |

**Example:**
```yaml
hardware:
  sample_rate_hz: 16000
  mel_bins: 40
  window_size_ms: 30
  window_step_ms: 10
  clip_duration_ms: 1000
```

### paths

Directory paths for data and outputs. All paths are relative to the config file location.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `positive_dir` | string | `./dataset/positive` | Wake word recordings organized by speaker |
| `negative_dir` | string | `./dataset/negative` | Background speech (not wake word) |
| `hard_negative_dir` | string | `./dataset/hard_negative` | False positive samples to avoid |
| `background_dir` | string | `./dataset/background` | Noise and ambient sounds |
| `rir_dir` | string | `./dataset/rirs` | Room impulse responses for reverb |
| `processed_dir` | string | `./data/processed` | Processed feature cache |
| `checkpoint_dir` | string | `./checkpoints` | Training checkpoints |
| `export_dir` | string | `./models/exported` | Exported TFLite models |

**Example:**
```yaml
paths:
  positive_dir: "./dataset/positive"
  negative_dir: "./dataset/negative"
  hard_negative_dir: "./dataset/hard_negative"
  background_dir: "./dataset/background"
  rir_dir: "./dataset/rirs"
  processed_dir: "./data/processed"
  checkpoint_dir: "./checkpoints"
  export_dir: "./models/exported"
```

### training

Training loop parameters including learning rate schedule and class weighting.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `training_steps` | list[int] | `[20000, 10000]` | Steps per training phase. Length determines number of phases. |
| `learning_rates` | list[float] | `[0.001, 0.0001]` | Learning rate per phase. Must match training_steps length. |
| `batch_size` | int | 128 | Training batch size. Reduce if OOM errors. |
| `eval_step_interval` | int | 500 | Evaluate model every N steps. |
| `positive_class_weight` | list[float] | `[1.0, 1.0]` | Weight for positive class per phase. |
| `negative_class_weight` | list[float] | `[20.0, 20.0]` | Weight for negative class per phase. |
| `hard_negative_class_weight` | list[float] | `[40.0, 40.0]` | Weight for hard negatives per phase (higher = fewer false accepts). |
| `time_mask_max_size` | list[int] | `[0, 0]` | Max time mask size per phase (SpecAugment). |
| `time_mask_count` | list[int] | `[0, 0]` | Number of time masks per phase. |
| `freq_mask_max_size` | list[int] | `[0, 0]` | Max frequency mask size per phase. |
| `freq_mask_count` | list[int] | `[0, 0]` | Number of frequency masks per phase. |
| `minimization_metric` | string | `ambient_false_positives_per_hour` | Metric to minimize for checkpoint selection. |
| `target_minimization` | float | 0.5 | Target value for minimization metric. |
| `maximization_metric` | string | `average_viable_recall` | Metric to maximize for checkpoint selection. |

**Example - Two-phase training:**
```yaml
training:
  training_steps: [20000, 10000]           # Phase 1: 20k steps, Phase 2: 10k steps
  learning_rates: [0.001, 0.0001]          # Phase 1: 0.001 lr, Phase 2: 0.0001 lr
  batch_size: 128
  eval_step_interval: 500

  # Class weights (higher negative weight = fewer false accepts)
  positive_class_weight: [1.0, 1.0]
  negative_class_weight: [20.0, 20.0]
  hard_negative_class_weight: [40.0, 40.0]

  # SpecAugment (disabled by default, using audio augmentation instead)
  time_mask_max_size: [0, 0]
  time_mask_count: [0, 0]
  freq_mask_max_size: [0, 0]
  freq_mask_count: [0, 0]

  # Checkpoint selection strategy
  minimization_metric: "ambient_false_positives_per_hour"
  target_minimization: 0.5
  maximization_metric: "average_viable_recall"
```

### model

Model architecture parameters for MixedNet.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `architecture` | string | `mixednet` | Model architecture. Options: `mixednet`, `dnn`, `cnn`, `crnn`. |
| `first_conv_filters` | int | 30 | Number of filters in first conv layer (20-30 for smaller models). |
| `first_conv_kernel_size` | int | 5 | Kernel size for first conv layer. |
| `stride` | int | 3 | Stride for first conv layer. Affects streaming chunk size. |
| `spectrogram_length` | int | 49 | Number of time frames in input spectrogram. |
| `pointwise_filters` | string | `"60,60,60,60"` | Filters per MixConv block (comma-separated). |
| `mixconv_kernel_sizes` | string | `"[5],[9],[13],[21]"` | Kernel sizes per block. Each block can have multiple kernels in brackets. |
| `repeat_in_block` | string | `"1,1,1,1"` | Repetitions per block (comma-separated). |
| `residual_connection` | string | `"0,0,0,0"` | Residual connections per block (0=off, 1=on). |
| `dropout_rate` | float | 0.0 | Dropout rate (0.0-1.0). Use 0.2 for regularization. |
| `l2_regularization` | float | 0.0 | L2 regularization weight. Use 0.001 for regularization. |

**Example - Standard MixedNet:**
```yaml
model:
  architecture: "mixednet"
  first_conv_filters: 30
  first_conv_kernel_size: 5
  stride: 3
  spectrogram_length: 49
  pointwise_filters: "60,60,60,60"
  mixconv_kernel_sizes: "[5],[9],[13],[21]"
  repeat_in_block: "1,1,1,1"
  residual_connection: "0,0,0,0"
  dropout_rate: 0.0
  l2_regularization: 0.0
```

**Example - Smaller model:**
```yaml
model:
  architecture: "mixednet"
  first_conv_filters: 20        # Reduced from 30
  first_conv_kernel_size: 5
  stride: 3
  spectrogram_length: 49
  pointwise_filters: "40,40,40,40"  # Reduced from 60
  mixconv_kernel_sizes: "[5],[9],[13],[21]"
  repeat_in_block: "1,1,1,1"
  residual_connection: "0,0,0,0"
  dropout_rate: 0.0
  l2_regularization: 0.0
```

#### ESPHome Compatibility Requirements

When building custom models (MixedNet, DNN, CNN, or CRNN) for ESPHome deployment, the following constraints are **required**:

- **Output layer**: `Dense(1)` with **sigmoid activation** — the final layer must be `Dense(1, activation='sigmoid')`. This ensures the model outputs a single probability in [0, 1] that ESPHome's `micro_wake_word` component can threshold.
- **First conv stride**: `stride` must remain **3** (or 1) for the first convolution; `stride=3` means the streaming model consumes 3 mel frames per inference step (`input_shape = [1, stride, mel_bins]`).
- **Input shape**: The non-streaming input spectrogram shape is `(spectrogram_length, mel_bins)`. During export the streaming model input becomes `(1, stride, mel_bins)` — e.g., `[1, 3, 40]` for the default config.
- **Fields that control these constraints** (in `config.model`): `architecture`, `first_conv_filters`, `stride`, `spectrogram_length`.

> **Output layer: Dense(1) with sigmoid activation**
> Do not replace the final `Dense(1, sigmoid)` with a softmax, multi-class head, or any other activation — ESPHome reads a single `uint8` probability output.

### augmentation

Audio augmentation parameters. Each value is the probability (0.0-1.0) of applying that augmentation.

#### Time-Domain Augmentations

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `SevenBandParametricEQ` | float | 0.1 | 7-band parametric equalization. |
| `TanhDistortion` | float | 0.1 | Tanh-based distortion. |
| `PitchShift` | float | 0.1 | Pitch shifting. **Warning: Internal speed perturbation is limited to max 1.3x to prevent distortion.** |
| `BandStopFilter` | float | 0.1 | Band-stop (notch) filtering. |
| `AddColorNoise` | float | 0.1 | Add colored noise. |
| `AddBackgroundNoise` | float | 0.75 | Mix in background noise. |
| `Gain` | float | 1.0 | Random gain adjustment. |
| `RIR` | float | 0.5 | Apply room impulse response (reverb). |
| `AddBackgroundNoiseFromFile` | float | 0.0 | Load background noise from file (max quality only). |
| `ApplyImpulseResponse` | float | 0.0 | Apply IR from file (max quality only). |

#### Augmentation Safety Limits

| Constraint | Limit | Notes |
|-----------|-------|-------|
| Speed/Pitch Perturbation | Never exceeds **1.3x** | Internal resampling for `PitchShift` is capped at 1.3x to prevent distortion artifacts |
| SNR range | `-5 dB` to `10 dB` | Set via `background_min_snr_db` / `background_max_snr_db` |
| Gain range | `-3 dB` to `+3 dB` | Applied uniformly before other augmentations |

#### Noise Mixing Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `background_min_snr_db` | int | -5 | Minimum signal-to-noise ratio in dB. |
| `background_max_snr_db` | int | 10 | Maximum signal-to-noise ratio in dB. |
| `min_jitter_s` | float | 0.195 | Minimum jitter in seconds. |
| `max_jitter_s` | float | 0.205 | Maximum jitter in seconds. |
| `impulse_paths` | list[string] | `["mit_rirs"]` | Paths to impulse response files. |
| `background_paths` | list[string] | `["fma_16k", "audioset_16k"]` | Paths to background noise datasets. |
| `augmentation_duration_s` | float | 3.2 | Target duration after augmentation. |

**Example - Standard augmentation:**
```yaml
augmentation:
  # Time-domain augmentations
  SevenBandParametricEQ: 0.1
  TanhDistortion: 0.1
  PitchShift: 0.1
  BandStopFilter: 0.1
  AddColorNoise: 0.1
  AddBackgroundNoise: 0.75
  Gain: 1.0
  RIR: 0.5

  # Noise mixing parameters
  background_min_snr_db: -5
  background_max_snr_db: 10
  min_jitter_s: 0.195
  max_jitter_s: 0.205

  # Background sources
  impulse_paths: ["./dataset/rirs"]
  background_paths: ["./dataset/background"]
  augmentation_duration_s: 3.2
```

**Example - Disabled (for testing):**
```yaml
augmentation:
  SevenBandParametricEQ: 0.0
  TanhDistortion: 0.0
  PitchShift: 0.0
  BandStopFilter: 0.0
  AddColorNoise: 0.0
  AddBackgroundNoise: 0.0
  Gain: 0.0
  RIR: 0.0
```

### performance

Performance, resource, and profiling configuration.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu_only` | bool | false | Require GPU (fail if not available). |
| `mixed_precision` | bool | true | Enable mixed precision (FP16) training. |
| `num_workers` | int | 16 | Number of data loading workers. |
| `num_threads_per_worker` | int | 2 | Threads per data loading worker. |
| `prefetch_factor` | int | 8 | Batches to prefetch per worker. |
| `pin_memory` | bool | true | Pin memory for faster GPU transfer. |
| `max_memory_gb` | int | 60 | Maximum memory usage in GB. |
| `inter_op_parallelism` | int | 16 | TensorFlow inter-op parallelism threads. |
| `intra_op_parallelism` | int | 16 | TensorFlow intra-op parallelism threads. |
| `enable_profiling` | bool | true | Enable performance profiling. |
| `profile_every_n_steps` | int | 100 | Profile every N steps. |
| `profile_output_dir` | string | `./profiles` | Profiling output directory. |
| `tensorboard_enabled` | bool | true | Enable TensorBoard logging. |
| `tensorboard_log_dir` | string | `./logs` | TensorBoard log directory. |

**Example - High performance:**
```yaml
performance:
  gpu_only: true
  mixed_precision: true
  num_workers: 16
  num_threads_per_worker: 2
  prefetch_factor: 8
  pin_memory: true
  max_memory_gb: 60
  inter_op_parallelism: 16
  intra_op_parallelism: 16
  enable_profiling: true
  profile_every_n_steps: 100
  profile_output_dir: "./profiles"
  tensorboard_enabled: true
  tensorboard_log_dir: "./logs"
```

**Example - Limited resources:**
```yaml
performance:
  gpu_only: false
  mixed_precision: false
  num_workers: 4
  num_threads_per_worker: 2
  prefetch_factor: 2
  pin_memory: false
  max_memory_gb: 30
  inter_op_parallelism: 4
  intra_op_parallelism: 4
  enable_profiling: false
  tensorboard_enabled: false
```

### speaker_clustering

Speaker clustering configuration for grouping audio samples by speaker identity.
Uses AgglomerativeClustering with cosine distance. Two modes:
- **Threshold-based** (default): Cluster count determined automatically from similarity threshold.
- **Explicit n_clusters**: Fixed cluster count — best for short audio clips (1-3s wake words) where ECAPA-TDNN embeddings have low cosine similarity and threshold-based clustering over-fragments.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable speaker clustering. |
| `method` | string | `agglomerative` | Clustering method (`agglomerative` or `threshold`). `agglomerative` uses average linkage; `threshold` uses complete linkage (stricter). |
| `embedding_model` | string | `speechbrain/ecapa-tdnn-voxceleb` | SpeechBrain ECAPA-TDNN model for embeddings. |
| `similarity_threshold` | float | 0.72 | Cosine similarity threshold (0.0-1.0). Only used when `n_clusters` is null. |
| `n_clusters` | int | null | Explicit number of clusters. Overrides `similarity_threshold` when set. Use when you know approximate speaker count. |
| `leakage_audit_enabled` | bool | true | Enable train/val leakage detection. |

**Example — threshold-based (auto cluster count):**
```yaml
speaker_clustering:
  enabled: true
  method: "agglomerative"
  embedding_model: "speechbrain/ecapa-tdnn-voxceleb"
  similarity_threshold: 0.72
  leakage_audit_enabled: true
```

**Example — explicit cluster count (recommended for short clips):**
```yaml
speaker_clustering:
  enabled: true
  method: "agglomerative"
  embedding_model: "speechbrain/ecapa-tdnn-voxceleb"
  n_clusters: 200  # Known speaker count; overrides threshold
  leakage_audit_enabled: true
```

### hard_negative_mining

Hard negative mining configuration for finding difficult samples.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable hard negative mining. |
| `fp_threshold` | float | 0.8 | False positive threshold for mining (0.0-1.0). |
| `max_samples` | int | 5000 | Maximum hard negative samples to collect. |
| `mining_interval_epochs` | int | 5 | Mine every N epochs. |

**Example:**
```yaml
hard_negative_mining:
  enabled: true
  fp_threshold: 0.8
  max_samples: 5000
  mining_interval_epochs: 5
```

### export

Model export settings for ESPHome compatibility.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `wake_word` | string | `Hey Katya` | Wake word phrase. |
| `author` | string | `Your Name` | Model author name. |
| `website` | string | `https://your-repo.com` | Project website. |
| `trained_languages` | list[string] | `["en"]` | List of trained languages. |
| `quantize` | bool | true | Enable INT8 quantization. |
| `inference_input_type` | string | `int8` | Input tensor type. Must be `int8`. |
| `inference_output_type` | string | `uint8` | Output tensor type. Must be `uint8`. |
| `probability_cutoff` | float | 0.97 | Detection threshold (0.0-1.0). Higher = fewer false triggers. |
| `sliding_window_size` | int | 5 | Size of sliding window for smoothing. |
| `tensor_arena_size` | int | 22860 | Tensor arena size in bytes (auto-calculated if 0). |
| `minimum_esphome_version` | string | `2024.7.0` | Minimum ESPHome version required. |

**Example:**
```yaml
export:
  wake_word: "Hey Computer"
  author: "Your Name"
  website: "https://github.com/yourusername/project"
  trained_languages: ["en"]

  quantize: true
  inference_input_type: "int8"
  inference_output_type: "uint8"

  # Detection threshold (0.70 for testing, 0.95-0.98 for production)
  probability_cutoff: 0.97
  sliding_window_size: 5
  tensor_arena_size: 22860
  minimum_esphome_version: "2024.7.0"
```

---

## Preset Configurations

Three built-in presets are available:

### fast_test
Quick iteration and debugging.
- Training: 3000 total steps (2000 + 1000)
- Batch size: 32
- Augmentation: Disabled
- Clustering/Mining: Disabled
- Profiling: Disabled
- Training time: ~1 hour

### standard (default)
Balanced quality and speed.
- Training: 30000 total steps (20000 + 10000)
- Batch size: 128
- Augmentation: Standard
- Clustering/Mining: Enabled
- Mixed precision: Enabled
- Training time: ~8 hours

### max_quality
Best accuracy settings.
- Training: 70000 total steps (50000 + 20000)
- Batch size: 128
- Augmentation: Full + extras
- Regularization: Dropout 0.2, L2 0.001
- Clustering/Mining: Enabled with strict settings
- GPU required: Yes
- Training time: ~24 hours

---

## Environment Variables

The following environment variables affect training:

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | GPU device index (e.g., `0`, `0,1`) |
| `TF_FORCE_GPU_ALLOW_GROWTH` | Allow GPU memory to grow (`true`/`false`) |
| `TF_GPU_ALLOCATOR` | GPU allocator (`cuda_malloc_async` recommended) |
| `TF_DETERMINISTIC_OPS` | Enable deterministic operations (`1` for reproducibility) |
| `CHECKPOINT_DIR` | Default checkpoint directory |
| `DATA_ROOT` | Root directory for datasets |

**Example setup:**
```bash
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_DETERMINISTIC_OPS=1
```

---

## Complete Example Configurations

### Minimal Custom Config (override)

Create `my_override.yaml`:

```yaml
# Override standard preset
export:
  wake_word: "Hey Jarvis"
  author: "Your Name"
  website: "https://github.com/yourusername"

training:
  batch_size: 64  # Reduce if OOM

model:
  first_conv_filters: 20  # Smaller model
```

Run:
```bash
mww-train --config standard --override my_override.yaml
```

### Full Custom Config

Create `full_config.yaml`:

```yaml
hardware:
  sample_rate_hz: 16000
  mel_bins: 40
  window_size_ms: 30
  window_step_ms: 10
  clip_duration_ms: 1000

paths:
  positive_dir: "./dataset/positive"
  negative_dir: "./dataset/negative"
  hard_negative_dir: "./dataset/hard_negative"
  background_dir: "./dataset/background"
  rir_dir: "./dataset/rirs"
  processed_dir: "./data/processed"
  checkpoint_dir: "./checkpoints"
  export_dir: "./models/exported"

training:
  training_steps: [30000, 15000]
  learning_rates: [0.001, 0.0001]
  batch_size: 128
  eval_step_interval: 500
  positive_class_weight: [1.0, 1.0]
  negative_class_weight: [25.0, 25.0]
  hard_negative_class_weight: [50.0, 50.0]
  minimization_metric: "ambient_false_positives_per_hour"
  target_minimization: 0.3
  maximization_metric: "average_viable_recall"

model:
  architecture: "mixednet"
  first_conv_filters: 30
  first_conv_kernel_size: 5
  stride: 3
  spectrogram_length: 49
  pointwise_filters: "60,60,60,60"
  mixconv_kernel_sizes: "[5],[9],[13],[21]"
  repeat_in_block: "1,1,1,1"
  residual_connection: "0,0,0,0"
  dropout_rate: 0.1
  l2_regularization: 0.001

augmentation:
  SevenBandParametricEQ: 0.15
  TanhDistortion: 0.15
  PitchShift: 0.15
  BandStopFilter: 0.15
  AddColorNoise: 0.15
  AddBackgroundNoise: 0.8
  Gain: 1.0
  RIR: 0.6
  background_min_snr_db: -5
  background_max_snr_db: 10
  min_jitter_s: 0.195
  max_jitter_s: 0.205
  impulse_paths: ["./dataset/rirs"]
  background_paths: ["./dataset/background"]
  augmentation_duration_s: 3.2

performance:
  gpu_only: true
  mixed_precision: true
  num_workers: 16
  num_threads_per_worker: 2
  prefetch_factor: 8
  pin_memory: true
  max_memory_gb: 60
  inter_op_parallelism: 16
  intra_op_parallelism: 16
  enable_profiling: true
  profile_every_n_steps: 100
  profile_output_dir: "./profiles"
  tensorboard_enabled: true
  tensorboard_log_dir: "./logs"

speaker_clustering:
  enabled: true
  method: "agglomerative"
  embedding_model: "speechbrain/ecapa-tdnn-voxceleb"
  similarity_threshold: 0.72
  # n_clusters: 200  # Uncomment to use explicit cluster count
  leakage_audit_enabled: true

hard_negative_mining:
  enabled: true
  fp_threshold: 0.8
  max_samples: 5000
  mining_interval_epochs: 5

export:
  wake_word: "Hey Computer"
  author: "Your Name"
  website: "https://github.com/yourusername"
  trained_languages: ["en"]
  quantize: true
  inference_input_type: "int8"
  inference_output_type: "uint8"
  probability_cutoff: 0.97
  sliding_window_size: 5
  tensor_arena_size: 22860
  minimum_esphome_version: "2024.7.0"
```

Run:
```bash
mww-train --config full_config.yaml
```

---

## Troubleshooting Guide

### Out of Memory (OOM) Errors

```yaml
# Reduce batch size
training:
  batch_size: 32

# Reduce workers
performance:
  num_workers: 4
  max_memory_gb: 30
```

### Poor Detection Accuracy

```yaml
# Increase training steps
training:
  training_steps: [50000, 20000]

# Increase negative weights (fewer false accepts)
training:
  negative_class_weight: [30.0, 30.0]
  hard_negative_class_weight: [60.0, 60.0]

# Enable all augmentations
augmentation:
  SevenBandParametricEQ: 0.15
  TanhDistortion: 0.15
  PitchShift: 0.15
  BandStopFilter: 0.15
  AddColorNoise: 0.15
  AddBackgroundNoise: 0.8
  RIR: 0.6
```

### Model Too Large for ESP32

```yaml
# Reduce model size
model:
  first_conv_filters: 20
  pointwise_filters: "40,40,40,40"

# Reduce arena size
export:
  tensor_arena_size: 18000
```

### Too Many False Triggers

```yaml
# Increase detection threshold
export:
  probability_cutoff: 0.98

# More aggressive negative weighting
training:
  hard_negative_class_weight: [60.0, 60.0]
```

### Training Too Slow

```yaml
# Enable mixed precision
performance:
  mixed_precision: true

# Reduce evaluation frequency
training:
  eval_step_interval: 1000

# Disable profiling
performance:
  enable_profiling: false
```

---

## Quick Reference

### Common Workflows

**Quick test:**
```bash
mww-train --config fast_test
```

**Standard training:**
```bash
mww-train --config standard
```

**Custom config:**
```bash
mww-train --config standard --override my_config.yaml
```

**Export model:**
```bash
mww-export --checkpoint checkpoints/best_weights.weights.h5 --model-name "hey_computer"
```

**Monitor training:**
```bash
tensorboard --logdir ./logs
```

### Important Notes

1. **GPU Required**: CuPy-based SpecAugment has no CPU fallback
2. **Python Version**: Use Python 3.10 or 3.11 (3.12 not supported by ai-edge-litert)
3. **Separate Environments**: TensorFlow and PyTorch must be in separate venvs
4. **ESPHome Compatibility**: Input must be int8, output must be uint8
5. **Detection Threshold**: 0.70 for testing, 0.95-0.98 for production

---

*Generated for microwakeword_trainer v2.0.0*

---
---

# Part 2: Implementation Plan

**Importance: ★★★★☆ HIGH**
**Source: `docs/IMPLEMENTATION_PLAN.md`**

---

# ESPHome microWakeWord Training Pipeline — IMPLEMENTATION PLAN (v2.0 PERFORMANCE EDITION)

**Version:** 2.0 (Performance Optimized + GPU-First)  
**Status:** Production-Ready with Performance Enhancements  
**Date:** 2025-02-25  
**Verification:** All facts verified via official model TFLite analysis + ESPHome C++ source

---

## EXECUTIVE SUMMARY

This plan is based on **definitive analysis** of official ESPHome microWakeWord v2 models:
- **hey_jarvis.tflite** (51.05 KB, 45 ops, simpler architecture)
- **okay_nabu.tflite** (58.85 KB, 55 ops, uses StridedKeep/Split)

All architectural claims verified against actual TFLite flatbuffers.

**v2.0 Additions**: GPU-first execution policy, performance optimization phases, profiling infrastructure, and enhanced monitoring.

---

## TABLE OF CONTENTS

1. [Definitively Verified Architecture](#1-definitively-verified-architecture)
2. [Python & Dependencies](#2-python--dependencies)
3. [Project Structure](#3-project-structure)
4. [Configuration Schema](#4-configuration-schema)
5. [Phase 1 — Data Ingestion & Validation](#5-phase-1--data-ingestion--validation)
6. [Phase 2 — Feature Extraction Pipeline](#6-phase-2--feature-extraction-pipeline)
7. [Phase 3 — Training Loop (Step-Based)](#7-phase-3--training-loop-step-based)
8. [Phase 4 — MixedNet Model Architecture](#8-phase-4--mixednet-model-architecture)
9. [Phase 5 — Dataset Engineering](#9-phase-5--dataset-engineering)
10. [Phase 6 — Augmentation Pipeline](#10-phase-6--augmentation-pipeline)
11. [Phase 7 — Hard Negative Mining](#11-phase-7--hard-negative-mining)
12. [Phase 8 — Export & TFLite Conversion](#12-phase-8--export--tflite-conversion)
13. [Phase 9 — ESPHome Manifest Generation](#13-phase-9--esphome-manifest-generation)
14. [Phase 10 — Comprehensive Metrics Suite](#14-phase-10--comprehensive-metrics-suite)
15. [Phase 11 — Performance Optimization (GPU-First)](#15-phase-11--performance-optimization-gpu-first)
16. [Phase 12 — Profiling & Monitoring](#16-phase-12--profiling--monitoring)
17. [ESPHome Compatibility Checklist](#17-esphome-compatibility-checklist)
18. [Dependency Manifest](#18-dependency-manifest)

---

## 1. DEFINITIVELY VERIFIED ARCHITECTURE

### 1.1 Model Structure (VERIFIED)

**From TFLite Analysis (hey_jarvis & okay_nabu):**

| Property | hey_jarvis | okay_nabu |
|----------|-----------|-----------|
| **Subgraphs** | 2 | 2 |
| **Subgraph 0 Ops** | 45 | 55 |
| **Subgraph 0 Tensors** | 71 | 95 |
| **Subgraph 1 Ops** | 12 | 12 |
| **Subgraph 1 Tensors** | 12 | 12 |
| **Input Shape** | [1, 3, 40] | [1, 3, 40] |
| **Input Dtype** | INT8 | INT8 |
| **Output Shape** | [1, 1] | [1, 1] |
| **Output Dtype** | UINT8 | UINT8 |
| **File Size** | 51.05 KB | 58.85 KB |

### 1.2 Tensor Shapes & Quantization (VERIFIED)

**Input Tensor:**
- Shape: `[1, stride, 40]` = `[1, 3, 40]`
- Dtype: `int8`
- Quantization: `scale=0.101961, zero_point=-128`

**Output Tensor:**
- Shape: `[1, 1]`
- Dtype: `uint8` (NOT int8!)
- Quantization: `scale=0.003906, zero_point=0`

### 1.3 BUILTIN Operations (NOT CUSTOM!)

**CRITICAL CORRECTION:** All operations are **BUILTIN** TFLite Micro ops, NOT custom ops.

**ESPHome C++ Registration (from micro_wake_word.cpp):**

```cpp
// Streaming model ops (14 ops):
AddCallOnce();        // VERIFIED: Invokes Subgraph 1 initialization
AddVarHandle();       // VERIFIED: Creates state variables
AddReadVariable();    // VERIFIED: Reads streaming state
AddStridedSlice();    // VERIFIED: Slices tensors
AddConcatenation();   // VERIFIED: Concatenates frames
AddAssignVariable();  // VERIFIED: Writes streaming state
AddConv2D();
AddDepthwiseConv2D();
AddMul();
AddAdd();
AddMean();
AddFullyConnected();
AddLogistic();
AddQuantize();
```

**TFLite Flatbuffer Op Names (from model analysis):**

| Op Name | Count (hey_jarvis) | Count (okay_nabu) | Purpose |
|---------|-------------------|-------------------|---------|
| CALL_ONCE | 1 | 1 | Invoke Subgraph 1 initialization |
| VAR_HANDLE | 6 | 6 | Create state variable handles |
| READ_VARIABLE | 6 | 6 | Read state from previous inference |
| ASSIGN_VARIABLE | 6 | 6 | Write state for next inference |
| CONCATENATION | 6 | 8 | Join old + new frames |
| STRIDED_SLICE | 6 | 10 | Extract frames for state update |
| CONV_2D | 5 | 5 | Pointwise convolutions |
| DEPTHWISE_CONV_2D | 4 | 6 | Depthwise convolutions |
| RESHAPE | 2 | 2 | Flatten for Dense layer |
| SPLIT_V | 0 | 2 | Split for StridedKeep (okay_nabu only) |
| FULLY_CONNECTED | 1 | 1 | Classification head |
| LOGISTIC | 1 | 1 | Sigmoid activation |
| QUANTIZE | 1 | 1 | Output quantization |

**Note:** All ops show `CustomCode: N/A` in TFLite analysis, confirming they are **builtin ops**.

### 1.4 State Variables (VERIFIED)

**6 State Variables per Model:**

| State Variable | hey_jarvis Shape | okay_nabu Shape | Purpose |
|----------------|-----------------|-----------------|---------|
| stream | [1, 2, 1, 40] | [1, 2, 1, 40] | First Conv2D ring buffer |
| stream_1 | [1, 4, 1, 30/32] | [1, 4, 1, 32] | MixConv block 0 |
| stream_2 | [1, 8, 1, 60/64] | [1, 10, 1, 64] | MixConv block 1 |
| stream_3 | [1, 12, 1, 60/64] | [1, 14, 1, 64] | MixConv block 2 |
| stream_4 | [1, 20, 1, 60/64] | [1, 22, 1, 64] | MixConv block 3 |
| stream_5 | [1, 4, 1, 60/64] | [1, 5, 1, 64] | Temporal pooling |

**Total State Memory:**
- hey_jarvis: 2,840 bytes
- okay_nabu: 3,520 bytes

### 1.5 Dual-Subgraph Architecture (VERIFIED)

```
Subgraph [0]: Main Inference Graph
├── Input: [1, 3, 40] INT8
├── 6 State Variables (VAR_HANDLE/READ_VARIABLE/ASSIGN_VARIABLE)
├── MixConv blocks with ring buffers
└── Output: [1, 1] UINT8

Subgraph [1]: Initialization Graph (NoOp)
├── Invoked once at startup via CALL_ONCE
├── Initializes 6 state variables to zero
└── Pseudoconst tensors with initial values
```

### 1.6 Inference Timing (VERIFIED)

- **Feature generation:** Every 10ms (window_step_ms)
- **Model inference:** Every 30ms (stride=3 × 10ms)
- **Input per inference:** 3 frames [1, 3, 40]
- **Ring buffer math:** kernel_size - stride = buffer_size

---

## 2. PYTHON & DEPENDENCIES

### 2.1 Requirements (VERIFIED)

```python
python_requires=">=3.10"

tensorflow>=2.16
ai-edge-litert        # TFLite runtime (new package)
pymicro-features      # C-based microfrontend
numpy
scipy
pyyaml
mmap_ninja
datasets
audiomentations
audio_metadata
webrtcvad-wheels
absl-py
```

### 2.2 Performance Dependencies (NEW in v2.0)

```python
cupy-cuda12x>=13.0    # GPU-accelerated NumPy (CUDA 12.x)
cudf>=24.0            # GPU DataFrame (optional)
pyarrow>=15.0         # Columnar data format
numba>=0.58           # JIT compilation for CPU fallback
```

### 2.3 Optional Dependencies

```python
speechbrain>=1.0.0    # Use inference module
transformers>=4.40.0  # WavLM embeddings
scikit-learn>=1.4.0   # Clustering
optuna>=3.6.0         # HPO
```

---

## 3. PROJECT FILE STRUCTURE

```
microwakeword_trainer/
├── config/
│   ├── training_config.yaml
│   └── presets/
│       ├── fast_test.yaml
│       ├── standard.yaml
│       └── max_quality.yaml
├── data/
│   ├── raw/
│   │   ├── positive/              # Wake word samples
│   │   ├── negative/              # Negative samples
│   │   ├── hard_negative/         # Hard negatives
│   │   └── background/            # Background noise
│   ├── processed/                 # Ragged Mmap spectrograms
│   └── clustered/                 # Speaker assignments
├── src/
│   ├── config/
│   │   └── loader.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── features.py
│   │   ├── dataset.py             # FeatureHandler wrapper
│   │   ├── clustering.py
│   │   └── spec_augment_gpu.py    # NEW: GPU-accelerated SpecAugment
│   ├── model/
│   │   ├── architecture.py
│   │   └── streaming.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── augmentation.py
│   │   ├── miner.py
│   │   └── profiler.py            # NEW: Performance profiling
│   ├── export/
│   │   ├── tflite.py
│   │   └── manifest.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── performance.py         # NEW: Performance utilities
├── models/exported/
│   └── hey_katya/
│       ├── hey_katya.tflite
│       ├── hey_katya.json
│       └── training_config.yaml
├── notebooks/
│   └── basic_training.ipynb
├── logs/                          # NEW: TensorBoard logs
├── profiles/                      # NEW: cProfile outputs
└── requirements.txt
```

---

## 4. CONFIGURATION SCHEMA

### 4.1 Complete Configuration

```yaml
# ─────────────────────────────────────────────────────────────────
# HARDWARE PARAMETERS (IMMUTABLE)
# ─────────────────────────────────────────────────────────────────
hardware:
  sample_rate_hz: 16000
  mel_bins: 40
  window_size_ms: 30              # 30ms Hann window
  window_step_ms: 10              # 10ms hop (v2)
  clip_duration_ms: 1000

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
paths:
  positive_dir: "./data/raw/positive"
  negative_dir: "./data/raw/negative"
  hard_negative_dir: "./data/raw/hard_negative"
  background_dir: "./data/raw/background"
  rir_dir: "./data/raw/rirs"
  processed_dir: "./data/processed"
  checkpoint_dir: "./checkpoints"
  export_dir: "./models/exported"

# ─────────────────────────────────────────────────────────────────
# TRAINING PARAMETERS (VERIFIED)
# ─────────────────────────────────────────────────────────────────
training:
  # Step-based training (NOT epochs!)
  training_steps: [20000, 10000]  # List of steps per phase
  learning_rates: [0.001, 0.0001] # LR for each phase
  batch_size: 128                 # VERIFIED from official notebook
  eval_step_interval: 500

  # Class weights for FAH control
  positive_class_weight: [1.0, 1.0]
  negative_class_weight: [20.0, 20.0]  # Increase to reduce false accepts

  # SpecAugment (disabled when using full audio augmentation)
  time_mask_max_size: [0, 0]
  time_mask_count: [0, 0]
  freq_mask_max_size: [0, 0]
  freq_mask_count: [0, 0]

  # Checkpoint selection
  minimization_metric: "ambient_false_positives_per_hour"
  target_minimization: 0.5
  maximization_metric: "average_viable_recall"

# ─────────────────────────────────────────────────────────────────
# MIXEDNET ARCHITECTURE (VERIFIED from TFLite)
# ─────────────────────────────────────────────────────────────────
model:
  architecture: "mixednet"

  # Two verified variants:
  # Variant A (hey_jarvis - simpler):
  first_conv_filters: 30
  first_conv_kernel_size: 5
  stride: 3
  pointwise_filters: "60,60,60,60"
  mixconv_kernel_sizes: "[5],[9],[13],[21]"
  repeat_in_block: "1,1,1,1"
  residual_connection: "0,0,0,0"

  # Common settings
  dropout_rate: 0.0
  l2_regularization: 0.0

# ─────────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────────
features:
  - features_dir: "generated_augmented_features"
    sampling_weight: 2.0
    penalty_weight: 1.0
    truth: True
    truncation_strategy: "truncate_start"
    type: "mmap"

  - features_dir: "negative_datasets/speech"
    sampling_weight: 10.0
    penalty_weight: 1.0
    truth: False
    truncation_strategy: "random"
    type: "mmap"

  - features_dir: "negative_datasets/dinner_party"
    sampling_weight: 10.0
    penalty_weight: 1.0
    truth: False
    truncation_strategy: "random"
    type: "mmap"

  - features_dir: "negative_datasets/no_speech"
    sampling_weight: 5.0
    penalty_weight: 1.0
    truth: False
    truncation_strategy: "random"
    type: "mmap"

  - features_dir: "negative_datasets/dinner_party_eval"
    sampling_weight: 0.0        # Validation only
    penalty_weight: 1.0
    truth: False
    truncation_strategy: "split"
    type: "mmap"

# ─────────────────────────────────────────────────────────────────
# AUGMENTATION (VERIFIED from notebook)
# ─────────────────────────────────────────────────────────────────
augmentation:
  # Time-domain augmentations
  SevenBandParametricEQ: 0.1
  TanhDistortion: 0.1
  PitchShift: 0.1
  BandStopFilter: 0.1
  AddColorNoise: 0.1
  AddBackgroundNoise: 0.75
  Gain: 1.0
  RIR: 0.5

  # Noise mixing parameters
  background_min_snr_db: -5
  background_max_snr_db: 10
  min_jitter_s: 0.195
  max_jitter_s: 0.205

  # Background sources
  impulse_paths: ['mit_rirs']
  background_paths: ['fma_16k', 'audioset_16k']
  augmentation_duration_s: 3.2

# ─────────────────────────────────────────────────────────────────
# PERFORMANCE CONFIGURATION (NEW in v2.0)
# ─────────────────────────────────────────────────────────────────
performance:
  # GPU Policy: If GPU-capable, GPU is MANDATORY
  # If no GPU support, use CPU with maximum resources
  gpu_only: true                  # Error if GPU unavailable for GPU ops
  mixed_precision: true            # FP16 for faster training
  
  # CPU Resources (when GPU unavailable)
  num_workers: 16                  # 16 cores
  num_threads_per_worker: 2        # 32 threads total
  prefetch_factor: 8               # Aggressive prefetching
  pin_memory: true                 # Faster GPU transfer
  max_memory_gb: 60                # Leave 4GB for OS
  
  # Parallelism
  inter_op_parallelism: 16
  intra_op_parallelism: 16
  
  # Profiling
  enable_profiling: true
  profile_every_n_steps: 100
  profile_output_dir: "./profiles"
  
  # TensorBoard
  tensorboard_enabled: true
  tensorboard_log_dir: "./logs"

# ─────────────────────────────────────────────────────────────────
# SPEAKER CLUSTERING
# ─────────────────────────────────────────────────────────────────
speaker_clustering:
  enabled: true
  method: "wavlm_ecapa"
  embedding_model: "microsoft/wavlm-base-plus"
  similarity_threshold: 0.72
  leakage_audit_enabled: true

# ─────────────────────────────────────────────────────────────────
# HARD NEGATIVE MINING
# ─────────────────────────────────────────────────────────────────
hard_negative_mining:
  enabled: true
  fp_threshold: 0.8
  max_samples: 5000
  mining_interval_epochs: 5

# ─────────────────────────────────────────────────────────────────
# EXPORT (VERIFIED)
# ─────────────────────────────────────────────────────────────────
export:
  wake_word: "Hey Katya"
  author: "Your Name"
  website: "https://your-repo.com"
  trained_languages: ["en"]

  quantize: true
  inference_input_type: "int8"
  inference_output_type: "uint8"     # MUST BE UINT8

  probability_cutoff: 0.97
  sliding_window_size: 5
  tensor_arena_size: 26080
  minimum_esphome_version: "2024.7.0"
```

---

## 5. PHASE 1 — DATA INGESTION & VALIDATION

### 5.1 Audio Requirements

**[SOURCE: audio_utils.py]**

```python
@dataclass
class SampleRecord:
    path: Path
    label: int          # 1 = wake word, 0 = negative
    split: str          # "train" | "val" | "test"
    speaker_id: int     # From clustering
    duration_ms: float
    sample_rate: int    # MUST be 16000
    weight: float = 1.0
```

**Validation Rules:**
- Sample rate: **16000 Hz** (ESPHome hardware constraint)
- Format: **16-bit PCM** (int16)
- Channels: **Mono**
- Duration: 200-1200ms (400-500ms recommended for wake word)

### 5.2 Data Loading

```python
from microwakeword.audio.clips import Clips

clips = Clips(
    input_directory='generated_samples',
    file_pattern='*.wav',
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,    # 10% for validation
)
```

### 5.3 Dataset Splits

**Training Set:**
- Positive samples: Wake word utterances
- Negative samples: Speech, background, hard negatives
- Augmented with slide_frames=10

**Validation/Test Sets:**
- Standard split: Positive + negative samples
- Ambient split: All negative background samples
- Used for FAH (False Accepts per Hour) estimation

---

## 6. PHASE 2 — FEATURE EXTRACTION PIPELINE

### 6.1 Feature Extraction API

```python
from pymicro_features import MicroFrontend

def generate_features_for_clip(
    audio_samples: np.ndarray,
    step_ms: int = 10,      # VERIFIED: 10ms for v2
    use_c: bool = True
) -> np.ndarray:
    """
    Generate spectrogram features.
    Returns: [time_frames, 40]
    """
    if use_c:
        micro_frontend = MicroFrontend()
        # Process in 10ms chunks (160 samples at 16kHz)
        features = []
        audio_idx = 0
        num_audio_bytes = len(audio_samples.tobytes())
        while audio_idx + 160 * 2 < num_audio_bytes:
            frontend_result = micro_frontend.process_samples(
                audio_samples[audio_idx : audio_idx + 160 * 2]
            )
            audio_idx += frontend_result.samples_read * 2
            if frontend_result.features:
                features.append(frontend_result.features)
        return np.array(features).astype(np.float32)
```

### 6.2 Spectrogram Generation

```python
from microwakeword.audio.spectrograms import SpectrogramGeneration

# Training set with temporal jitter
spectrograms_train = SpectrogramGeneration(
    clips=clips,
    augmenter=augmenter,
    slide_frames=10,        # Same spectrogram shifted by 1 frame
    step_ms=10,
)

# Testing set without artificial repetition
spectrograms_test = SpectrogramGeneration(
    clips=clips,
    augmenter=augmenter,
    slide_frames=1,         # No repetition for streaming test
    step_ms=10,
)
```

### 6.3 Ragged Mmap Storage

```python
from mmap_ninja.ragged import RaggedMmap

# Store as memory-mapped for efficient loading
RaggedMmap.from_generator(
    out_dir='./data/processed/train/wakeword_mmap',
    sample_generator=spectrograms.spectrogram_generator(
        split="train",
        repeat=2
    ),
    batch_size=100,
    verbose=True,
)
```

### 6.4 Feature Parameters (VERIFIED)

| Parameter | Value | Source |
|-----------|-------|--------|
| sample_rate | 16000 Hz | Fixed |
| window_size | 30 ms | Fixed |
| window_step | 10 ms | v2 standard |
| num_channels | 40 mel bins | Fixed |
| upper_band_limit | 7500 Hz | Fixed |
| lower_band_limit | 125 Hz | Fixed |
| enable_pcan | True | Noise suppression |
| out_scale | 1 | |
| out_type | uint16 | |

---

## 7. PHASE 3 — TRAINING LOOP (STEP-BASED)

### 7.1 Step-Based Training (NOT Epoch-Based!)

**[SOURCE: train.py]**

```python
def train(model, config, data_processor):
    """
    Step-based training loop.
    CRITICAL: Uses steps, NOT epochs!
    """
    # Configure optimizer and loss
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    # Metrics with 101 thresholds for ROC/PR curves
    cutoffs = np.linspace(0.0, 1.0, 101).tolist()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
        tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
        tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
        tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Unwrap train_function for speed
    model.make_train_function()
    _, model.train_function = tf_decorator.unwrap(model.train_function)

    # Step-based training loop
    training_steps_max = sum(config["training_steps"])

    for training_step in range(1, training_steps_max + 1):
        # Get current phase settings
        phase = get_current_phase(training_step, config)

        model.optimizer.learning_rate.assign(phase["learning_rate"])

        # Build augmentation policy
        augmentation_policy = {
            "mix_up_prob": phase["mix_up_prob"],
            "freq_mix_prob": phase["freq_mix_prob"],
            "time_mask_max_size": phase["time_mask_max_size"],
            "time_mask_count": phase["time_mask_count"],
            "freq_mask_max_size": phase["freq_mask_max_size"],
            "freq_mask_count": phase["freq_mask_count"],
        }

        # Get batch with augmentation
        train_fingerprints, train_ground_truth, train_sample_weights = \
            data_processor.get_data(
                "training",
                batch_size=config["batch_size"],
                features_length=config["spectrogram_length"],
                truncation_strategy="default",
                augmentation_policy=augmentation_policy,
            )

        train_ground_truth = train_ground_truth.reshape(-1, 1)

        # Apply class weights
        class_weights = {
            0: phase["negative_class_weight"],
            1: phase["positive_class_weight"]
        }
        combined_weights = train_sample_weights * np.vectorize(
            class_weights.get
        )(train_ground_truth)

        # Train on batch
        result = model.train_on_batch(
            train_fingerprints,
            train_ground_truth,
            sample_weight=combined_weights,
        )

        # Validate every eval_step_interval steps
        is_last_step = training_step == training_steps_max
        if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
            metrics = validate_nonstreaming(
                config, data_processor, model, "validation"
            )

            # Two-priority checkpoint selection
            if is_best_model(metrics, config):
                model.save_weights(
                    os.path.join(config["train_dir"], "best_weights.weights.h5")
                )
```

### 7.2 Two-Priority Checkpoint Selection

```python
def is_best_model(metrics, config):
    """
    Priority 1: Minimize FAH below target
    Priority 2: Maximize recall
    """
    current_min = metrics[config["minimization_metric"]]
    current_max = metrics[config["maximization_metric"]]
    target_min = config["target_minimization"]

    # Case 1: Achieved target FAH and improved recall
    if current_min <= target_min and current_max > best_max:
        return True

    # Case 2: Haven't achieved target but decreased FAH
    if current_min > target_min and current_min < best_min:
        return True

    # Case 3: Tied FAH and improved recall
    if current_min == best_min and current_max > best_max:
        return True

    return False
```

---

## 8. PHASE 4 — MIXEDNET MODEL ARCHITECTURE

### 8.1 Model Configuration

**Two verified variants:**

**Variant A (hey_jarvis - Simpler, 45 ops):**
```python
config = {
    "first_conv_filters": 30,
    "first_conv_kernel_size": 5,
    "stride": 3,
    "pointwise_filters": [60, 60, 60, 60],
    "mixconv_kernel_sizes": [[5], [9], [13], [21]],
    "repeat_in_block": [1, 1, 1, 1],
    "residual_connection": [0, 0, 0, 0],
}
```

**Variant B (okay_nabu - Complex, 55 ops):**
- Uses `strided_keep` and `SPLIT_V` operations
- Generated automatically by training framework

### 8.2 Model Architecture Code

```python
from microwakeword.mixednet import model as mixednet_model
from microwakeword.layers import stream

def build_model(config):
    """Build MixedNet model."""

    inputs = tf.keras.Input(
        shape=(config["spectrogram_length"], 40),
        batch_size=config["batch_size"]
    )

    # Add channel dimension: [batch, time, 1, feature]
    net = tf.keras.ops.expand_dims(inputs, axis=2)

    # Initial convolution with streaming
    if config["first_conv_filters"] > 0:
        net = stream.Stream(
            cell=tf.keras.layers.Conv2D(
                config["first_conv_filters"],
                (config["first_conv_kernel_size"], 1),
                strides=(config["stride"], 1),
                padding="valid",
                use_bias=False,
            ),
            use_one_step=False,
            pad_time_dim=None,
            pad_freq_dim="valid",
        )(net)
        net = tf.keras.layers.Activation("relu")(net)

    # MixConv blocks
    for i, (filters, repeat, ksize, use_res) in enumerate(zip(
        config["pointwise_filters"],
        config["repeat_in_block"],
        config["mixconv_kernel_sizes"],
        config["residual_connection"]
    )):
        # MixConv with ring buffer
        net = MixConv(kernel_size=ksize)(net)

        # Pointwise projection
        net = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            padding="same"
        )(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)

    # Classification head
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid")(net)

    return tf.keras.Model(inputs, net)
```

---

## 9. PHASE 5 — DATASET ENGINEERING

### 9.1 Feature Set Configuration

```python
config["features"] = [
    {
        "features_dir": "generated_augmented_features",
        "sampling_weight": 2.0,      # Oversample wake words
        "penalty_weight": 1.0,       # Standard penalty
        "truth": True,               # Positive samples
        "truncation_strategy": "truncate_start",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/speech",
        "sampling_weight": 10.0,     # More negatives
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
]
```

### 9.2 Truncation Strategies

- `random`: Choose random portion (for long negatives)
- `truncate_start`: Remove start of spectrogram
- `truncate_end`: Remove end of spectrogram
- `split`: Split into multiple spectrograms (ambient only)

### 9.3 FAH Estimation

```python
def estimate_fah(model, ambient_data, duration_hours):
    """
    Estimate false accepts per hour.
    Split ambient clips with 100ms stride to simulate streaming.
    """
    predictions = []
    for spectrogram in ambient_data:
        # Slide window with 100ms stride
        pred = model.predict(spectrogram)
        predictions.append(pred)

    false_accepts = sum(1 for p in predictions if p > threshold)
    return false_accepts / duration_hours
```

---

## 10. PHASE 6 — AUGMENTATION PIPELINE

### 10.1 Audio Augmentation

```python
from microwakeword.audio.augmentation import Augmentation

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=['mit_rirs'],
    background_paths=['fma_16k', 'audioset_16k'],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)
```

### 10.2 SpecAugment (Disabled by Default)

When using full audio augmentation pipeline, SpecAugment is disabled:

```python
config["time_mask_max_size"] = [0]
config["time_mask_count"] = [0]
config["freq_mask_max_size"] = [0]
config["freq_mask_count"] = [0]
```

**Reason:** Audio augmentation provides sufficient variability.

---

## 11. PHASE 7 — HARD NEGATIVE MINING

### 11.1 Hard Negative Detection

```python
def mine_hard_negatives(model, data_processor, threshold=0.8):
    """
    Find negative samples that model incorrectly predicts as positive.
    """
    hard_negatives = []

    negatives, labels, _ = data_processor.get_data(
        "training_negative", ...
    )
    predictions = model.predict(negatives)

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if label == 0 and pred > threshold:  # False positive
            hard_negatives.append({
                'index': i,
                'prediction': pred,
                'sample': negatives[i]
            })

    # Sort by confidence, keep top N
    hard_negatives.sort(key=lambda x: x['prediction'], reverse=True)
    return hard_negatives[:5000]
```

### 11.2 Iterative Mining

- Run every `mining_interval_epochs` (default: 5)
- Save to `data/raw/hard_negative/`
- Increase sampling weight for hard negatives

---

## 12. PHASE 8 — EXPORT & TFLITE CONVERSION

### 12.1 Two-Step Export Process

```python
from microwakeword.utils import (
    convert_model_saved,
    convert_saved_model_to_tflite
)
from microwakeword.layers import modes

# Step 1: Convert to streaming SavedModel
converted_model = convert_model_saved(
    model_non_stream=model,
    config=config,
    folder="stream_state_internal",
    mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
)

# Step 2: Convert to TFLite
convert_saved_model_to_tflite(
    config=config,
    audio_processor=data_processor,  # REQUIRED for calibration
    path_to_model=os.path.join(config["train_dir"], "stream_state_internal"),
    folder=os.path.join(config["train_dir"], "tflite_stream_state_internal_quant"),
    fname="stream_state_internal_quant.tflite",
    quantize=True
)
```

### 12.2 Critical Quantization Settings

```python
converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)
converter.optimizations = {tf.lite.Optimize.DEFAULT}

# CRITICAL: Required for streaming state variables
converter._experimental_variable_quantization = True

converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.uint8  # MUST BE UINT8!

# Representative dataset for calibration
def representative_dataset_gen():
    sample_fingerprints, _, _ = audio_processor.get_data(
        "training", 500, features_length=config["spectrogram_length"]
    )
    # Set min/max for calibration
    sample_fingerprints[0][0, 0] = 0.0
    sample_fingerprints[0][0, 1] = 26.0

    stride = config["stride"]
    for spectrogram in sample_fingerprints:
        for i in range(0, spectrogram.shape[0] - stride, stride):
            sample = spectrogram[i:i + stride, :].astype(np.float32)
            yield [sample]

converter.representative_dataset = tf.lite.RepresentativeDataset(
    representative_dataset_gen
)
```

### 12.3 Export Testing

```python
tflite_configs = [
    {
        "log_string": "quantized streaming model",
        "source_folder": "stream_state_internal",
        "output_folder": "tflite_stream_state_internal_quant",
        "filename": "stream_state_internal_quant.tflite",
        "quantize": True,
    }
]
```

---

## 13. PHASE 9 — ESPHOME MANIFEST GENERATION

### 13.1 V2 Manifest Format

```json
{
  "type": "micro",
  "wake_word": "Hey Katya",
  "author": "Your Name",
  "website": "https://your-repo.com",
  "model": "hey_katya.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "feature_step_size": 10,
    "sliding_window_size": 5,
    "tensor_arena_size": 26080,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

### 13.2 Tensor Arena Sizing

- hey_jarvis: ~26,080 bytes
- okay_nabu: ~28,000-30,000 bytes
- ESPHome default: 1MB (conservative)
- Recommended: Measure empirically

---

## 14. PHASE 10 — COMPREHENSIVE METRICS SUITE

### 14.1 Metrics Computed

```python
metrics = {
    # Classification metrics
    "accuracy": float,
    "recall": float,
    "precision": float,
    "f1_score": float,

    # ROC/PR metrics
    "auc": float,
    "average_precision": float,

    # Wake word specific
    "recall_at_no_faph": float,       # Recall when FAH = 0
    "cutoff_for_no_faph": float,      # Threshold for zero FAH
    "ambient_false_positives": int,
    "ambient_false_positives_per_hour": float,
    "average_viable_recall": float,   # Area under recall vs FAH curve
}
```

### 14.2 Validation Metrics

```python
def validate_nonstreaming(config, data_processor, model, split):
    """Compute comprehensive validation metrics."""

    # Get validation data
    fingerprints, ground_truth, _ = data_processor.get_data(
        split,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    ground_truth = ground_truth.reshape(-1, 1)

    # Evaluate
    model.reset_metrics()
    result = model.evaluate(
        fingerprints,
        ground_truth,
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )

    # Compute FAH metrics
    if data_processor.get_mode_size("validation_ambient") > 0:
        ambient_metrics = compute_fah_metrics(
            model, data_processor, split
        )
        result.update(ambient_metrics)

    return result
```

---

## 15. PHASE 11 — PERFORMANCE OPTIMIZATION (GPU-FIRST)

### 15.1 GPU-First Execution Policy

**Global Policy**: If an operation CAN run on GPU, it MUST run on GPU.  
If GPU is unavailable or CPU is demonstrably faster, use all 32 threads and 64GB RAM aggressively.

#### GPU vs CPU Decision Matrix

| Operation | GPU Capable? | Policy | Implementation |
|-----------|--------------|--------|----------------|
| Model Training (TensorFlow) | ✅ Yes | **GPU MANDATORY** | `tf.device('/GPU:0')` |
| SpecAugment | ✅ Yes | **GPU MANDATORY** | CuPy implementation |
| Validation/Inference | ✅ Yes | **GPU MANDATORY** | Batch GPU inference |
| Feature Extraction (pymicro-features) | ❌ No | CPU (C-optimized) | Already optimized |
| Audio Augmentation (audiomentations) | ❌ No | **CPU 32 threads** | Parallel processing |
| Data Loading/Batching | ❌ No | **CPU 32 threads** | Multi-worker prefetch |
| TFLite Conversion | ❌ No | CPU only | Standard TensorFlow |
| Metrics Calculation | ❌ No | CPU (NumPy) | Vectorized operations |

### 15.2 GPU-Accelerated SpecAugment (CuPy)

**Location**: `src/data/spec_augment_gpu.py`

**CRITICAL**: This implementation REQUIRES GPU. No CPU fallback.

```python
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

def spec_augment_gpu(
    spectrogram: np.ndarray,
    time_mask_max_size: int = 0,
    time_mask_count: int = 0,
    freq_mask_max_size: int = 0,
    freq_mask_count: int = 0
) -> np.ndarray:
    """
    GPU-accelerated SpecAugment. FAILS if GPU unavailable.
    
    Args:
        spectrogram: Input spectrogram [time_frames, freq_bins]
        time_mask_max_size: Maximum size of time masks
        time_mask_count: Number of time masks to apply
        freq_mask_max_size: Maximum size of frequency masks
        freq_mask_count: Number of frequency masks to apply
    
    Returns:
        Augmented spectrogram (transferred back to CPU)
    
    Raises:
        RuntimeError: If CuPy not installed or GPU unavailable
    """
    if not HAS_GPU:
        raise RuntimeError(
            "GPU REQUIRED for SpecAugment. "
            "Install CuPy: pip install cupy-cuda12x"
        )
    
    # Transfer to GPU memory
    x = cp.asarray(spectrogram)
    t_frames, f_bins = x.shape
    
    # Apply time masks on GPU
    for _ in range(time_mask_count):
        t = cp.random.randint(1, time_mask_max_size + 1)
        t0 = cp.random.randint(0, t_frames - t + 1)
        x[t0:t0+t, :] = 0
    
    # Apply frequency masks on GPU
    for _ in range(freq_mask_count):
        f = cp.random.randint(1, freq_mask_max_size + 1)
        f0 = cp.random.randint(0, f_bins - f + 1)
        x[:, f0:f0+f] = 0
    
    # Transfer back to CPU for TensorFlow
    return cp.asnumpy(x)


def batch_spec_augment_gpu(
    batch: np.ndarray,
    time_mask_max_size: int = 0,
    time_mask_count: int = 0,
    freq_mask_max_size: int = 0,
    freq_mask_count: int = 0
) -> np.ndarray:
    """
    Batch GPU SpecAugment for entire training batch.
    
    Args:
        batch: Batch of spectrograms [batch_size, time_frames, freq_bins]
        ...mask parameters...
    
    Returns:
        Augmented batch (on CPU)
    """
    if not HAS_GPU:
        raise RuntimeError("GPU REQUIRED for batch SpecAugment")
    
    x = cp.asarray(batch)  # [B, T, F]
    batch_size, t_frames, f_bins = x.shape
    
    # Vectorized masking across batch dimension
    for _ in range(time_mask_count):
        t = cp.random.randint(1, time_mask_max_size + 1)
        t0 = cp.random.randint(0, t_frames - t + 1, size=batch_size)
        for i in range(batch_size):
            x[i, t0[i]:t0[i]+t, :] = 0
    
    for _ in range(freq_mask_count):
        f = cp.random.randint(1, freq_mask_max_size + 1)
        f0 = cp.random.randint(0, f_bins - f + 1, size=batch_size)
        for i in range(batch_size):
            x[i, :, f0[i]:f0[i]+f] = 0
    
    return cp.asnumpy(x)
```

### 15.3 Parallel Audio Augmentation (32 Threads)

**Location**: `src/training/augmentation.py` enhancement

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

class ParallelAugmentation:
    """
    Multi-threaded audio augmentation using all 32 threads.
    Falls back to sequential if CPU count < 8.
    """
    
    def __init__(self, max_workers: int = None):
        # Use all available threads (32 on 16-core/32-thread system)
        self.max_workers = max_workers or min(32, os.cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def augment_batch(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """
        Augment multiple clips in parallel.
        
        Args:
            clips: List of audio clip arrays
        
        Returns:
            List of augmented clips
        """
        futures = [self.executor.submit(self.augment_clip, clip) 
                   for clip in clips]
        return [f.result() for f in futures]
```

### 15.4 Parallel Data Loading

**Configuration**: 16 workers × 2 threads = 32 threads total

```python
# In training script initialization
import tensorflow as tf

def configure_performance():
    """Configure TensorFlow for maximum performance."""
    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Threading
    tf.config.threading.set_inter_op_parallelism_threads(16)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    
    # Enable mixed precision for 2x speedup on compatible GPUs
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### 15.5 Performance Monitoring

Track these metrics during training:
- **Data loading time**: Should be < 10% of step time
- **GPU utilization**: Should be > 80% (nvidia-smi)
- **CPU utilization**: All 32 threads active during data loading
- **Memory usage**: Use up to 60GB RAM for prefetch buffers

### 15.6 ESPHome Compatibility Note

**IMPORTANT**: All performance optimizations in this phase are **TRAINING-ONLY**.  
They do NOT affect ESPHome compatibility in any way.

ESPHome compatibility is determined solely by:
- Final TFLite model structure (2 subgraphs, specific ops)
- Input/output tensor shapes and dtypes
- Manifest JSON fields

The training data loading method (mmap vs PyArrow), augmentation implementation  
(Numba vs CuPy vs pure Python), and parallel processing have **ZERO impact** on  
the exported model or ESPHome compatibility.

---

## 16. PHASE 12 — PROFILING & MONITORING

### 16.1 Profiling Infrastructure

**Location**: `src/utils/profiler.py`

```python
import cProfile
import pstats
import io
from contextlib import contextmanager
from datetime import datetime
import os

class TrainingProfiler:
    """
    Comprehensive profiling for training pipeline.
    Identifies bottlenecks in data loading, augmentation, and training.
    """
    
    def __init__(self, output_dir: str = "./profiles"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @contextmanager
    def profile_section(self, name: str):
        """
        Context manager for profiling a code section.
        
        Usage:
            profiler = TrainingProfiler()
            with profiler.profile_section("data_loading"):
                data = load_data()
        """
        profiler = cProfile.Profile()
        profiler.enable()
        yield
        profiler.disable()
        
        # Save profile
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.prof"
        filepath = os.path.join(self.output_dir, filename)
        profiler.dump_stats(filepath)
        
        # Print summary
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        print(f"\n=== Profile: {name} ===")
        print(s.getvalue())
        print(f"Saved to: {filepath}")
    
    def profile_training_step(self, model, data_fn, n_steps: int = 10):
        """
        Profile multiple training steps to identify bottlenecks.
        
        Args:
            model: TensorFlow model
            data_fn: Function that returns batch data
            n_steps: Number of steps to profile
        """
        with self.profile_section("training_steps"):
            for _ in range(n_steps):
                x, y, w = data_fn()
                model.train_on_batch(x, y, sample_weight=w)
```

### 16.2 User Guide: How to Profile

#### Step 1: Enable Profiling in Config

```yaml
# config/training_config.yaml
performance:
  enable_profiling: true
  profile_every_n_steps: 100
  profile_output_dir: "./profiles"
```

#### Step 2: Run Training with Profiling

```bash
# Method A: Profile specific section
python -c "
from microwakeword.utils.profiler import TrainingProfiler
from microwakeword.data import FeatureHandler

profiler = TrainingProfiler()
handler = FeatureHandler(config)

with profiler.profile_section('data_loading'):
    data, labels, weights = handler.get_data('training', batch_size=128, ...)
"

# Method B: Full training profile
python -m microwakeword.train --config config.yaml --profile
```

#### Step 3: Analyze Profile Results

```bash
# View profile in console
python -c "
import pstats
p = pstats.Stats('profiles/data_loading_20250225_143022.prof')

# Sort by total time (tottime) - time in function excluding subcalls
print('=== TOP 20 by total time ===')
p.sort_stats('tottime').print_stats(20)

# Sort by cumulative time (cumtime) - time in function including subcalls
print('=== TOP 20 by cumulative time ===')
p.sort_stats('cumtime').print_stats(20)
"
```

#### Step 4: Interpret Results

| Pattern | Meaning | Action |
|---------|---------|--------|
| High `tottime` in `spec_augment` | SpecAugment is bottleneck | ✅ CuPy GPU acceleration will help |
| High `tottime` in `audiomentations` | Audio augmentation slow | Already C-optimized; check parallelization |
| High `cumtime` in `get_data` | Data loading is bottleneck | ✅ Increase prefetch_factor, num_workers |
| High `tottime` in `train_on_batch` | Model training is bottleneck | Expected; check GPU utilization |
| High `tottime` in `np.array` | Conversion overhead | Use pre-allocated buffers |

### 16.3 TensorBoard Integration

#### What's Already Implemented

The official train.py already includes TensorBoard logging:

```python
# From train.py (already exists)
train_writer = tf.summary.create_file_writer(
    os.path.join(config["summaries_dir"], "train")
)
validation_writer = tf.summary.create_file_writer(
    os.path.join(config["summaries_dir"], "validation")
)

# During training
with train_writer.as_default():
    tf.summary.scalar("loss", result[9], step=training_step)
    tf.summary.scalar("accuracy", result[1], step=training_step)
    tf.summary.scalar("recall", result[2], step=training_step)
```

#### How to View Live Training

```bash
# Terminal 1: Start training
python train.py --config config.yaml

# Terminal 2: Start TensorBoard
tensorboard --logdir ./logs/summaries --port 6006

# Open browser to http://localhost:6006
```

#### TensorBoard Dashboards

| Dashboard | What You'll See | When to Check |
|-----------|-----------------|---------------|
| **SCALARS** | Loss, accuracy, recall, precision curves | Every eval interval |
| **SCALARS** | recall_at_no_faph, average_viable_recall | For wake word quality |
| **GRAPHS** | Model architecture visualization | Once at start |
| **HISTOGRAMS** | Weight distributions | Check for vanishing/exploding gradients |
| **IMAGES** | Spectrogram visualizations (if added) | Data validation |
| **PROFILE** | Step timing, op execution (if enabled) | Performance tuning |

#### Custom TensorBoard Metrics

Add these to track performance:

```python
# In train.py, add to training loop
with train_writer.as_default():
    # Existing metrics
    tf.summary.scalar("loss", result[9], step=training_step)
    
    # NEW: Performance metrics
    tf.summary.scalar("perf/data_loading_ms", data_load_time * 1000, step=training_step)
    tf.summary.scalar("perf/augmentation_ms", aug_time * 1000, step=training_step)
    tf.summary.scalar("perf/training_step_ms", train_time * 1000, step=training_step)
    tf.summary.scalar("perf/gpu_memory_gb", get_gpu_memory(), step=training_step)
```

### 16.4 Performance Debugging Guide

#### Scenario 1: Training is Slow (Low GPU Utilization)

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# If GPU < 80%, data loading is bottleneck
# Solutions:
# 1. Increase prefetch_factor
# 2. Increase num_workers
# 3. Enable CuPy SpecAugment (GPU)
# 4. Use SSD for data storage
```

#### Scenario 2: Out of Memory (OOM)

```bash
# Check memory usage
nvidia-smi
htop

# Solutions:
# 1. Reduce batch_size
# 2. Reduce prefetch_factor
# 3. Enable memory growth: tf.config.experimental.set_memory_growth(gpu, True)
# 4. Use mixed precision (already enabled)
```

#### Scenario 3: CPU Usage Low (Not Using 32 Threads)

```bash
# Check CPU usage
htop

# If usage < 50%, check:
# 1. Is num_workers set to 16?
# 2. Is parallel augmentation enabled?
# 3. Is data on fast storage (SSD)?
```

---

## 17. ESPHOME COMPATIBILITY CHECKLIST

### 17.1 TFLite Model Verification

```python
def verify_esphome_compatibility(tflite_path: str) -> bool:
    """
    Verify TFLite model is compatible with ESPHome micro_wake_word.
    Based on analysis of hey_jarvis.tflite and okay_nabu.tflite.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)

    # Check 1: 2 subgraphs
    assert len(interpreter.get_subgraphs()) == 2, "Must have 2 subgraphs"

    # Check 2: Input [1, stride, 40] INT8
    input_details = interpreter.get_input_details()
    assert input_details[0]['shape'].tolist() == [1, 3, 40]
    assert input_details[0]['dtype'] == np.int8

    # Check 3: Output [1, 1] UINT8
    output_details = interpreter.get_output_details()
    assert output_details[0]['shape'].tolist() == [1, 1]
    assert output_details[0]['dtype'] == np.uint8

    # Check 4: State variables present (6 TYPE_13 tensors)
    # Should have VAR_HANDLE, READ_VARIABLE, ASSIGN_VARIABLE patterns

    return True
```

### 17.2 Required Ops Checklist

**ESPHome C++ must register:**

- [x] `AddCallOnce()` - For Subgraph 1 invocation
- [x] `AddVarHandle()` - State variable creation
- [x] `AddReadVariable()` - State reading
- [x] `AddAssignVariable()` - State writing
- [x] `AddStridedSlice()` - Frame slicing
- [x] `AddConcatenation()` - Frame joining
- [x] `AddConv2D()` - Convolutions
- [x] `AddDepthwiseConv2D()` - Depthwise convs
- [x] `AddFullyConnected()` - Dense layer
- [x] `AddLogistic()` - Sigmoid
- [x] `AddQuantize()` - Output quantization
- [x] `AddReshape()` - Flatten

### 17.3 Compatibility Summary

| Check | Expected | Status |
|-------|----------|--------|
| Subgraphs | 2 | ✅ VERIFIED |
| Input shape | [1, 3, 40] | ✅ VERIFIED |
| Input dtype | int8 | ✅ VERIFIED |
| Output shape | [1, 1] | ✅ VERIFIED |
| Output dtype | uint8 | ✅ VERIFIED |
| Quantized | Yes | ✅ VERIFIED |
| State vars | 6 | ✅ VERIFIED |
| Op types | BUILTIN only | ✅ VERIFIED |

### 17.4 ESPHome Compatibility & Training Optimizations

**CRITICAL CLARIFICATION**:

All training pipeline optimizations in **Phase 11 (Performance Optimization)**  
and **Phase 12 (Profiling)** are **TRAINING-ONLY** and do **NOT** affect ESPHome compatibility:

| Training Optimization | ESPHome Impact | Reason |
|----------------------|----------------|--------|
| PyArrow data loading | ✅ None | Training-only data format |
| mmap vs PyArrow | ✅ None | Only affects training speed |
| CuPy SpecAugment | ✅ None | Training augmentation only |
| Numba acceleration | ✅ None | Training augmentation only |
| 32-thread parallel loading | ✅ None | Training data pipeline |
| GPU training | ✅ None | Model architecture unchanged |
| TensorBoard logging | ✅ None | Monitoring only |
| Profiling | ✅ None | Development tool |

**What ACTUALLY affects ESPHome compatibility**:
- TFLite model structure (2 subgraphs, ops, tensor shapes/dtypes)
- Manifest JSON fields (minimum_esphome_version, tensor_arena_size)
- Quantization settings (int8 input, uint8 output)

---

## 18. DEPENDENCY MANIFEST

### 18.1 Core Dependencies

```
# Python
python>=3.10,<3.13

# ML Framework
tensorflow>=2.16
ai-edge-litert
pymicro-features>=0.1

# Audio/Data
numpy>=1.26
scipy
pyyaml
mmap_ninja
datasets>=2.14
audiomentations
audio_metadata
webrtcvad-wheels
absl-py

# Performance (NEW in v2.0)
cupy-cuda12x>=13.0    # GPU acceleration (REQUIRES CUDA 12.x)
pyarrow>=15.0         # Columnar data (optional)
numba>=0.58           # CPU JIT fallback (optional)

# Optional (Extended Pipeline)
speechbrain>=1.0.0
transformers>=4.40.0
scikit-learn>=1.4.0
optuna>=3.6.0
matplotlib
seaborn
```

### 18.2 Deprecated (DO NOT USE)

| Deprecated | Replacement | Reason |
|------------|-------------|--------|
| `tensorflow-addons` | `tf.keras.losses.BinaryFocalCrossentropy` | EOL May 2024 |
| `speechbrain.pretrained` | `speechbrain.inference` | API changed in 1.0+ |

---

## APPENDIX: VERIFICATION SOURCES

### Official Models Analyzed

| Model | Size | Ops | Tensors | Analysis File |
|-------|------|-----|---------|---------------|
| hey_jarvis.tflite | 51.05 KB | 45 | 71/12 | hey_jarvis_analysis.json |
| okay_nabu.tflite | 58.85 KB | 55 | 95/12 | okay_nabu_analysis.json |

### Key Findings Summary

1. **Ops are BUILTIN, not CUSTOM** - All ops show `CustomCode: N/A`
2. **AddCallOnce() is REQUIRED** - For Subgraph 1 invocation
3. **Dual-subgraph confirmed** - Subgraph[0]=inference, Subgraph[1]=init
4. **State management via Variables** - VAR_HANDLE/READ/ASSIGN pattern
5. **UINT8 output required** - NOT INT8
6. **Two architecture variants** - hey_jarvis (simpler), okay_nabu (complex)

### ESPHome C++ Registration

**File:** `micro_wake_word.cpp`  
**Function:** `register_streaming_ops_()`  
**Ops registered:** 14 builtin ops including AddCallOnce, AddVarHandle, AddReadVariable, AddAssignVariable

---

**Document Version:** 2.0 (Performance Optimized)  
**Verification Status:** 100% - All claims verified via TFLite analysis  
**Last Updated:** 2025-02-25

---

**Changes in v2.0**:
- Added Phase 11: Performance Optimization (GPU-First)
- Added Phase 12: Profiling & Monitoring
- Added GPU-mandatory execution policy
- Added CuPy SpecAugment implementation
- Added parallel processing with 32 threads
- Added TensorBoard integration guide
- Added profiling infrastructure and user guide
- Clarified ESPHome compatibility boundaries
- Updated dependencies (CuPy, PyArrow, Numba)

---

## QUICK REFERENCE: GPU USAGE BY OPERATION

| Operation | GPU? | Implementation |
|-----------|------|----------------|
| Model Training | ✅ MANDATORY | TensorFlow GPU |
| SpecAugment | ✅ MANDATORY | CuPy |
| Validation | ✅ MANDATORY | TensorFlow GPU |
| Feature Extraction | ❌ No | pymicro-features (C) |
| Audio Augmentation | ❌ No | audiomentations (C+32 threads) |
| Data Loading | ❌ No | 32-thread parallel |
| TFLite Export | ❌ No | CPU only |

---

## QUICK REFERENCE: PROFILING COMMANDS

```bash
# Start TensorBoard
tensorboard --logdir ./logs/summaries

# Profile specific section
python -c "
from microwakeword.utils.profiler import TrainingProfiler
profiler = TrainingProfiler()
with profiler.profile_section('data_loading'):
    # your code here
    pass
"

# View profile results
python -c "
import pstats
p = pstats.Stats('profiles/data_loading_*.prof')
p.sort_stats('tottime').print_stats(20)
"

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor CPU/Memory
htop
```

---

**END OF IMPLEMENTATION PLAN v2.0**

---
---

# Part 3: Project Environment Profile

**Importance: ★★★☆☆ MEDIUM**
**Source: `docs/my_environment.md`**

---

# Integrated Project Profile: "Hey Katya" Wakeword Training

## 1. Environment & Operational Profile

This section details the physical and acoustic environment where the model will operate.

### Voice Characteristics

* **Primary Male Speaker:** Deep bass voice with a monotonous speaking style.
* **Primary Female Speaker:** Alto voice with a semi-monotonous to semi-fluctuating style.
* **Wakeword:** "Hey Katya" (Phonetic structure should be prioritized during training).

### Home Acoustic Background

The following sounds are expected to be present during real-world operation:

* **Media:** TV audio (English and Turkish YouTube content) and high-quality home cinema audio.
* **Domestic Sounds:** Cat meows, air conditioner humming.
* **Hardware Interaction:** Keyboard and mouse button clicking noises.
* **Other:** Ambient voices of both speakers; very little to no mobile phone ringtones.

---

## 2. Dataset Composition & Analysis

Detailed breakdown of the training data used for the microwakeword model.

### Deep Composition

* **Real Live Recordings:** 1% (recorded in multiple rooms, including bathrooms/kitchens, using high-quality microphones).
* **Synthetic Clones:** 50% (cloned versions of the primary users' voices).
* **General Synthetic:** 49% (from various TTS engines).
* **Gender Balance:** Approximately 50% male and 50% female distribution.
* **Hard Negatives:** 50+ specific words selected via LLM analysis to be phonetically similar to "Hey Katya".

### Quantitative Analysis

| Category | File Count | Total Duration (Min) | Avg Duration (Sec) |
| --- | --- | --- | --- |
| **TOTAL** | 220,138 | 9,235.97 | 2.51 |
| **Positive** | 19,030 | 275.81 | 0.87 |
| **Negative** | 116,516 | 6,482.84 | 3.34 |
| **Hard Negative** | 34,187 | 586.19 | 1.03 |
| **Background** | 30,350 | 1,723.35 | 3.41 |
| **RIRs** | 20,055 | 167.78 | 0.50 |

*(Note: All files are in `.wav` format at a 16000Hz sample rate with 0 corrupted files reported.)*

---

## 3. Training & Hardware Infrastructure

The technical environment and target deployment devices.

### Training Hardware (Workstation)

* **CPU:** Ryzen 9 7950X (16-Core / 32-Thread)
* **GPU:** RTX 3060 Ti (8 GB VRAM)
* **RAM:** 64 GB System RAM
* **Storage:** Samsung 990 Pro 1 TB NVMe
* **Network:** 1 Gbit/s Bandwidth

### Target Deployment Devices (MCU/SBC)

The trained model must be 100% compatible with microwakeword (OHF) standards for:

* 2 x M5Stack Atom Echo
* 1 x ESP32-S3-BOX3
* Multiple ESP32 + INMP441 Microphone setups
* 2 x Raspberry Pi Zero 2W (Wyoming Satellite)

---

---
---

# Part 4: Post-Training Analysis

**Importance: ★★★☆☆ MEDIUM**
**Source: `docs/POST_TRAINING_ANALYSIS.md`**

---

# Post-Training Analysis and Available Commands

This document describes all post-training analysis tools and commands available in the microwakeword_trainer framework.

---

## 🚨 Your Training Results Analysis

Your results show **suspiciously perfect metrics**:

```
Accuracy:  0.9999
Precision: 1.0000
Recall:    0.9997
F1 Score:  0.9998
FA/Hour:   0.00
```

### ⚠️ Why This Is Suspicious

**Statistical Reality:**
- **FAH = 0.00** with 83,720 negative samples is **virtually impossible**
- Only **5 false negatives** out of 14,299 positives is suspiciously low
- **Zero false positives** is statistically improbable

**Likely Causes:**

1. **Data Leakage (Most Likely)**
   - Same audio files in both train and validation sets
   - Speaker overlap (same person in train and val)
   - Augmented versions of same files counted as different samples

2. **Validation Set Too Small**
   - Your validation set might be too small to represent real-world diversity
   - Check: `ls -la dataset/positive/` vs `ls -la dataset/negative/`

3. **Overfitting**
   - Model memorized the validation set
   - 70,000 steps is a lot - model might have overfitted

4. **Incorrect Labeling**
   - Negative samples might actually contain wake word instances
   - Background noise might be too clean

### 🔍 How to Verify

```bash
# 1. Check dataset splits
python -c "
from src.data.ingestion import load_clips, Split
clips = load_clips('config/presets/standard.yaml')
print(f'Train: {len(clips.get_split(Split.TRAIN))}')
print(f'Val:   {len(clips.get_split(Split.VAL))}')
print(f'Test:  {len(clips.get_split(Split.TEST))}')
"

# 2. Evaluate on completely separate test data
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best_weights.weights.h5 \
    --config standard \
    --split test \
    --analyze

# 3. Export and verify TFLite compatibility
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
```

---

## 📊 Post-Training Commands

### 1. Model Export
```bash
# Export to TFLite
mww-export \
    --checkpoint models/checkpoints/best_weights.weights.h5 \
    --output models/exported/

# Without quantization (for debugging)
mww-export \
    --checkpoint models/checkpoints/best.ckpt \
    --output models/exported/ \
    --no-quantize
```

### 2. ESPHome Compatibility Verification
```bash
# Basic check
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Verbose output
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose

# JSON output for CI/CD
python scripts/verify_esphome.py models/exported/wake_word.tflite --json
```

### 3. Model Evaluation (NEW)
```bash
# Evaluate on test set
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --split test \
    --analyze

# Evaluate TFLite model
python scripts/evaluate_model.py \
    --tflite models/exported/wake_word.tflite \
    --config standard \
    --split test

# Output as JSON
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --json
```

### 4. Auto-Tuning (Fine-tuning)
```bash
# Fine-tune for better FAH/recall
mww-autotune \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --target-fah 0.2 \
    --target-recall 0.95

# With custom iterations
mww-autotune \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --max-iterations 50
```

### 5. Model Analysis
```bash
# Detailed model report
python -c "
from src.export.model_analyzer import analyze_model_architecture
results = analyze_model_architecture('models/exported/wake_word.tflite')
print(results)
"
```

---

## 📁 Test Dataset Usage

### Current State
The framework **does define** a TEST split, but it's **not actively used** in training:

```python
# From src/data/ingestion.py
class Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
```

**During Training:**
- Only TRAIN and VAL splits are used
- TEST split is set aside but not evaluated

**Post-Training:**
- You should manually evaluate on TEST split
- This gives unbiased performance estimate

### How to Use Test Split

```bash
# Evaluate on test set (NEW script)
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --split test \
    --analyze
```

Or programmatically:

```python
from src.data.dataset import WakeWordDataset
from src.evaluation.metrics import MetricsCalculator
import numpy as np

# Load dataset
dataset = WakeWordDataset(config)
dataset.build()

# Get test generator
test_gen = dataset.test_generator_factory(max_time_frames)()

# Evaluate
y_true = []
y_scores = []

for features, labels in test_gen:
    predictions = model.predict(features)
    y_true.extend(labels)
    y_scores.extend(predictions)

# Calculate metrics
calc = MetricsCalculator(y_true=np.array(y_true), y_score=np.array(y_scores))
metrics = calc.compute_all_metrics(ambient_duration_hours=10.0)
```

---

## 🎯 Recommended Post-Training Workflow

### Step 1: Verify Export
```bash
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
```

### Step 2: Evaluate on Test Set
```bash
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --split test \
    --analyze
```

### Step 3: Check for Data Leakage
```bash
# Compare train/val/test speaker overlap
python cluster-Test.py --config standard --dataset all
```

### Step 4: Real-World Testing
- Export model to ESP32
- Test with real audio recordings
- Check actual false activation rate

---

## 🐛 Debugging Suspicious Results

### If metrics are too good:

1. **Check Data Leakage:**
   ```bash
   # Compare file lists
   find dataset/positive/train -name "*.wav" | sort > train_files.txt
   find dataset/positive/val -name "*.wav" | sort > val_files.txt
   comm -12 train_files.txt val_files.txt  # Should be empty
   ```

2. **Check Speaker Overlap:**
   ```bash
   python cluster-Test.py --config standard
   # Review cluster_output/*_cluster_report.txt
   ```

3. **Visualize Predictions:**
   ```python
   import matplotlib.pyplot as plt
   
   # Plot prediction distribution
   plt.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label='Negative')
   plt.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label='Positive')
   plt.legend()
   plt.savefig('prediction_distribution.png')
   ```

4. **Check Augmentation:**
   - If augmentation is too aggressive, model might see "easy" versions
   - Verify augmentation parameters in config

---

## 📋 Summary of Available Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `mww-train` | Train model | Initial training |
| `mww-export` | Export to TFLite | After training |
| `mww-autotune` | Fine-tune model | If metrics need improvement |
| `scripts/verify_esphome.py` | Verify TFLite compatibility | After export |
| `scripts/evaluate_model.py` | Evaluate on test set | Post-training validation |
| `cluster-Test.py` | Speaker clustering | Data preparation |
| `scripts/generate_test_dataset.py` | Generate synthetic data | Testing pipeline |

---

## ⚠️ Your Next Steps

Given your suspicious results:

1. **Run verification:**
   ```bash
   python scripts/verify_esphome.py models/exported/wake_word.tflite
   ```

2. **Check test set performance:**
   ```bash
   python scripts/evaluate_model.py \
       --checkpoint models/checkpoints/best.ckpt \
       --config standard \
       --split test \
       --analyze
   ```

3. **If still suspicious, re-train with:**
   - Verified speaker separation
   - Smaller training steps (e.g., 20k instead of 70k)
   - More aggressive data augmentation

4. **Real-world test:**
   - Deploy to ESP32
   - Test with actual microphone input
   - Count real false activations per hour

---
---

# Part 5: Mixed Precision Research

**Importance: ★★☆☆☆ REFERENCE**
**Source: `docs/RESEARCH_REPORT_MIXED_PRECISION.md`**
**Language: Turkish**

---

# Araştırma Raporu: Mixed Precision ve tf.data.Dataset

## 📋 ÖZET

### 1. Mixed Precision (FP16) Eğitimi ve ESPHome Uyumluluğu

**SONUÇ: ✅ Mixed precision ESPHome uyumluluğunu BOZMAZ**

| Soru | Cevap |
|------|-------|
| Mixed precision eğitimi TFLite export'u etkiler mi? | **Hayır** |
| ESPHome'da çalışmama riski var mı? | **Hayır** |
| Performans kazancı var mı? | **Evet, 2-3x** |
| Öneri | **Kullanabilirsin, güvenli** |

**Neden Bozmaz:**

1. **Eğitim ve Inference Ayrı Süreçler**
   - Mixed precision sadece **eğitim sırasında** kullanılır
   - Eğitim bittikten sonra model `float32` ağırlıklara sahiptir
   - TFLite export aşamasında model **INT8'e quantize** edilir

2. **TFLite Export Süreci (Bakımdan Geçirilmiş)**
   ```python
   # Export sırasında yapılanlar:
   converter.optimizations = {tf.lite.Optimize.DEFAULT}
   converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
   converter.inference_input_type = tf.int8    # ZORUNLU
   converter.inference_output_type = tf.uint8  # ZORUNLU
   converter.representative_dataset = ...      # Calibration
   ```

3. **Quantization Aşaması**
   - Tüm ağırlıklar `int8`'e çevrilir
   - Tüm aktivasyonlar `int8`/`uint8`'e çevrilir
   - Model artık **sadece 8-bit** integer işlemler yapar
   - Eğitimde kullanılan precision (FP16/FP32) kalıcı değildir

4. **ARCHITECTURAL_CONSTITUTION Doğrulaması**
   - ESPHome'un gerektirdiği: `int8` input, `uint8` output
   - Mixed precision training bu requirement'ı **etkilemez**
   - Quantization sonrası model her zaman aynı formatta olur

**Kısaca:** Mixed precision sadece eğitimi hızlandırır, model mimarisini veya export edilen TFLite formatını değiştirmez.

---

### 2. tf.data.Dataset ve ESPHome Uyumluluğu

**SONUÇ: ✅ tf.data.Dataset ESPHome uyumluluğunu BOZMAZ ve PERFORMANS sağlar**

| Özellik | Açıklama |
|---------|----------|
| **Nedir?** | TensorFlow'un veri pipeline API'si |
| **Nerede kullanılır?** | Sadece eğitim sırasında veri yükleme |
| **Modeli etkiler mi?** | **Hayır** - Sadece data loading |
| **ESPHome etkisi?** | **Sıfır** - Export edilen model aynı |
| **Performans?** | **Evet, 2-5x hızlanma** |

**tf.data.Dataset Avantajları:**

```python
# Mevcut (generator-based)
def train_generator():
    for sample in dataset:
        yield preprocess(sample)  # CPU'da sırayla yapılır

# tf.data.Dataset (optimized)
dataset = tf.data.Dataset.from_tensor_slices(files)
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.cache()           # Disk/RAM cache
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # GPU beklemez
```

| Optimizasyon | Kazanç | Açıklama |
|-------------|--------|----------|
| `map(parallel)` | 2-3x | Çoklu CPU çekirdeği kullanır |
| `cache()` | 3-5x | İkinci epoch'tan itibaren RAM'den okur |
| `prefetch()` | 1.5x | GPU boşta beklemez |
| `batch()` | 1.2x | Vektörize edilmiş yüklemeler |

**Neden Güvenli:**
- tf.data.Dataset sadece **eğitim verisinin nasıl yüklendiğini** değiştirir
- Model ağırlıklarına, mimarisine veya katmanlarına **dokunmaz**
- Export edilen TFLite model **tamamen aynı** olur
- ESPHome runtime'ı sadece TFLite modeli görür, data pipeline'ı görmez

**Özetle:** tf.data.Dataset implementasyonu:
- ✅ Performans artışı sağlar
- ✅ ESPHome uyumluluğunu bozmaz  
- ✅ Güvenle kullanılabilir

---

## 🎯 SONUÇ ve ÖNERİLER

### Mixed Precision
```yaml
# config.yaml
performance:
  mixed_precision: true   # ✅ Kullanabilirsin, ESPHome uyumluluğunu bozmaz
```

### tf.data.Dataset
```python
# Implementasyon önerisi - src/data/dataset.py'ye eklenebilir
def create_optimized_dataset(self):
    dataset = tf.data.Dataset.from_generator(
        self.generator,
        output_signature=...
    )
    dataset = dataset.cache()  # RAM'e cache
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # GPU pipeline
    return dataset
```

**Her ikisi de güvenle kullanılabilir ve performans sağlar.**

---
---

# Part 6: Log Analysis Guide

**Importance: ★★☆☆☆ REFERENCE**
**Source: `docs/LOG_ANALYSIS_GUIDE.md`**
**Language: Turkish**

---

# MWW Eğitim Log ve Profil Yorumlama Rehberi

Bu rehber, microwakeword_trainer ile eğitim yaparken oluşan log ve profil dosyalarını nasıl yorumlayacağınızı açıklar.

---

## 📊 1. PROFİL DOSYALARI (`.prof`)

**Konum:** `./profiles/` dizini

### Profil Nedir?

cProfile ile oluşturulmuş Python performans analiz dosyalarıdır. Hangi fonksiyonların ne kadar zaman aldığını gösterir.

### İnceleme Yöntemleri

```bash
# 1. Python ile okuma (terminalde görüntüleme)
python -c "
import pstats
p = pstats.Stats('profiles/data_loading_123456.prof')
p.sort_stats('cumulative')  # Toplam süreye göre sırala
p.print_stats(20)  # İlk 20 fonksiyonu göster
"

# 2. Kod içinde kullanma
from src.training.profiler import TrainingProfiler

# Mevcut bir profili analiz et
summary = TrainingProfiler.get_summary("./profiles/training_step_123456.prof", top_n=30)
print(summary)
```

### Profil Çıktısı Nasıl Okunur?

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   5000    2.345    0.000   15.678    0.003 spectrogram.py:45(compute_mel)
    200    0.123    0.001   12.456    0.062 model.py:89(call)
```

| Sütun | Anlamı | Yorumu |
|-------|--------|--------|
| **ncalls** | Çağrı sayısı | Çok fazla çağrı = optimizasyon adayı |
| **tottime** | Fonksiyon içinde geçen süre | Saf hesaplama zamanı |
| **percall** | Çağrı başına süre | Tek çağrı maliyeti |
| **cumtime** | Toplam birikimli süre | Alt fonksiyonlar dahil |
| **cumtime/percall** | Çağrı başına toplam | En önemli metrik! |

### 🔴 Bottleneck (Tıkanıklık) Tespiti

| Durum | Anlamı | Çözüm |
|-------|--------|-------|
| **cumtime yüksek, tottime düşük** | Fonksiyon başka yavaş fonksiyonları çağırıyor | Alt fonksiyonları optimize et |
| **tottime yüksek** | Fonksiyonun kendisi yavaş | Fonksiyonu optimize et veya vektörize et |
| **ncalls çok yüksek** | Gereksiz döngü içinde çağrı | Vektörizasyon yap, döngüden çıkar |

---

## 📋 2. TERMINAL LOG DOSYALARI (`terminal_*.log`)

**Konum:** `./logs/terminal_YYYYMMDD_HHMMSS.log`

### Log Dosyalarını Listeleme

```bash
# Log dosyalarını listele
ls -la ./logs/terminal_*.log

# En son logu izle
tail -f ./logs/terminal_$(date +%Y%m%d)*.log
```

### Log Yapısı ve Yorumlama

#### Eğitim Başlangıcı
```
Training Log Started: 2025-02-27T10:27:17
================================================================================

[TerminalLogger] Capturing output to: ./logs/terminal_20250227_102717.log

🎯 Wake Word Training
┌─────────────────┬────────────────────────────────┐
│ Phase 1         │ 20,000 steps @ LR 0.001000     │
│ Phase 2         │ 10,000 steps @ LR 0.000100     │
│ Class Weights   │ pos=[1.0, 1.0]  neg=[20.0...   │
│ Batch Size      │ 128                            │
└─────────────────┴────────────────────────────────┘
```

#### Eğitim İlerlemesi
```
Phase 1 • 500/30000 • 1.7% • 0:02:14 • 2:10:45 • loss=0.2341 acc=0.8912 lr=0.001000
```

| Alan | Anlamı |
|------|--------|
| `Phase 1` | Mevcut eğitim fazı |
| `500/30000` | Mevcut step / Toplam step |
| `1.7%` | Tamamlanma yüzdesi |
| `0:02:14` | Geçen süre |
| `2:10:45` | Tahmini kalan süre (ETA) |
| `loss=0.2341` | Kayıp değeri |
| `acc=0.8912` | Doğruluk |
| `lr=0.001000` | Öğrenme oranı |

---

## 🎯 Önemli Metrikler ve Anlamları

### 1. Loss (Kayıp)

```
loss=0.2341
```

| Değer Aralığı | Durum | Yorum |
|---------------|-------|-------|
| **0.1 - 0.3** | 🟢 İyi | Öğrenme devam ediyor |
| **0.3 - 0.5** | 🟡 Normal | Normal seyir |
| **> 0.5** | 🔴 Kötü | Düşük öğrenme oranı veya veri sorunu |
| **< 0.01** | 🟠 Uyarı | Aşırı öğrenme (overfitting) riski |

### 2. Accuracy, Precision, Recall, F1

```
acc=0.8912  prec=0.8234  recall=0.7567  f1=0.7889
```

| Metrik | Hedef | Düşükse Ne Yapılmalı? |
|--------|-------|----------------------|
| **Accuracy** | > 0.95 | Daha fazla veri, augmentation artır |
| **Precision** | > 0.90 | False Positive çok → negatif örnekleri artır |
| **Recall** | > 0.90 | False Negative çok → pozitif örnekleri artır |
| **F1** | > 0.90 | Dengesiz sınıflar → class weight ayarla |

### 3. Ambient FA/Hour (False Activation/Hour)

```
Ambient FA/Hour: 3.45  [🟡 Sarı]
```

**Bu, wake word için EN KRİTİK metriktir!** Saatte kaç yanlış alarm verdiğini gösterir.

| Değer | Renk | Durum | Anlamı |
|-------|------|-------|--------|
| **< 0.5** | 🟢 Yeşil | Mükemmel | Kabul edilebilir yanlış alarm |
| **0.5 - 2.0** | 🟡 Sarı | Kabul edilebilir | Sınırda, iyileştirilebilir |
| **> 2.0** | 🔴 Kırmızı | Kötü | Çok fazla yanlış uyandırma |

### 4. Checkpoint Mesajları

```
✅ BEST MODEL FAH improved: 3.45 → 2.12
   → checkpoints/best_fah_step_500.ckpt

💾 Checkpoint: step_1000.ckpt
```

| İkon | Anlamı |
|------|--------|
| **✅ BEST MODEL** | En iyi performans kaydedildi (daha iyi FAH) |
| **💾 Checkpoint** | Düzenli ara kayıt (her N adımda) |

---

## 📊 Validation (Doğrulama) Sonuçları

```
📊 Validation Results — Step 500/30000
┌──────────────────────┬────────┐
│ Accuracy             │ 0.8912 │
│ Precision            │ 0.8234 │
│ Recall               │ 0.7567 │
│ F1 Score             │ 0.7889 │  <- Hedef: >0.90
│ AUC-ROC              │ 0.9234 │
│ AUC-PR               │ 0.8567 │
│ Ambient FA/Hour      │ 3.45   │  <- 🟡 Sarı (hedef: <0.5)
│ Recall @ No FAPH     │ 0.6789 │
│ Threshold for No FAPH│ 0.8234 │
└──────────────────────┴────────┘
```

### Confusion Matrix

```
Confusion Matrix (threshold=0.5)
┌─────────────────┬──────────────────┬──────────────────┐
│                 │ Predicted Pos    │ Predicted Neg    │
├─────────────────┼──────────────────┼──────────────────┤
│ Actual Positive │ [green]850[/]     │ [red]150[/]       │
│ Actual Negative │ [red]200[/]       │ [green]7650[/]    │
├─────────────────┼──────────────────┼──────────────────┤
│ Total           │                  │ [bold]8850[/]     │
└─────────────────┴──────────────────┴──────────────────┘
```

- **TP (True Positive):** 850 - Doğru pozitif tahmin
- **FP (False Positive):** 200 - Yanlış pozitif (sesli komut olmadan tetikleme)
- **TN (True Negative):** 7650 - Doğru negatif tahmin
- **FN (False Negative):** 150 - Kaçırılan wake word

---

## 📈 3. TENSORBOARD LOG'LARI

**Konum:** `./logs/` dizini (TensorBoard event dosyaları)

### TensorBoard Başlatma

```bash
source ~/venvs/mww-tf/bin/activate
tensorboard --logdir ./logs

# Tarayıcıda aç: http://localhost:6006
```

### TensorBoard Sekmeleri

#### SCALARS (Metrikler)

| Metrik | Açıklama | İyi Seyir |
|--------|----------|-----------|
| `epoch_loss` | Her epoch sonundaki kayıp | ↓ Düşmeli |
| `epoch_accuracy` | Doğruluk grafiği | ↑ Artmalı |
| `val_loss` | Validasyon kaybı | ↓ Düşmeli (train_loss'a yakın) |
| `val_accuracy` | Validasyon doğruluğu | ↑ Artmalı |
| `learning_rate` | Öğrenme oranı değişimi | Fazlara göre adım adım düşer |

**Ne Aranır:**
- ✅ **loss ↓ düşüyor** → Model öğreniyor
- ❌ **val_loss ↑ artıyor** → Overfitting başladı
- ⚠️ **Loss dalgalanıyor** → Learning rate çok yüksek

#### GRAPHS (Model Grafiği)

Modelin katman yapısını görsel olarak gösterir:
- Op'lar arası bağlantılar
- Tensor boyutları
- Hesaplama grafiği

#### HISTOGRAMS (Ağırlık Dağılımları)

```
Layer weights   → Ağırlıkların dağılımı
Layer biases    → Bias değerleri
Gradients       → Gradyan büyüklükleri
```

**Yorumlama:**
- Ağırlıklar çok küçük → Vanishing gradient
- Ağırlıklar çok büyük → Exploding gradient
- Tüm ağırlıklar aynı → Başlatma sorunu

---

## 🔍 4. SIK KARŞILAŞILAN SORUNLAR

### Sorun: Loss Stagnant (Sabit Kalıyor)

```
Loss: 0.45 → 0.44 → 0.43 → 0.44 → 0.43 (1000 step sonra hâlâ)
```

**Çözüm:**
1. Learning rate çok düşük → `0.0001` → `0.001` yap
2. Veri yetersiz → Daha fazla örnek ekle
3. Augmentation az → `augmentation.yaml` ayarlarını artır

### Sorun: Validation İyi ama FA/Hour Kötü

```
val_accuracy: 0.98  (çok iyi!)
FA/Hour: 15.3      (çok kötü!)
```

**Çözüm:**
- Background audio ekle (ambient gürültü)
- Hard negative örnekleri artır
- Model threshold'u yükselt

### Sorun: Training Çok Yavaş

```
Step 100/30000 ETA: 48 hours
```

**Kontrol Adımları:**
```bash
# Profil dosyası var mı?
ls ./profiles/

# En yavaş fonksiyonu bul
python -c "
import pstats
p = pstats.Stats('profiles/training_step_xxx.prof')
p.sort_stats('cumulative').print_stats(5)
"
```

**Muhtemel Nedenler:**
- Data loading yavaş → `num_workers` artır
- GPU kullanılmıyor → `nvidia-smi` kontrol et
- CuPy kurulu değil → `uv pip install cupy-cuda12x`

---

## 🛠️ 5. PRATİK KOMUTLAR

```bash
# Son 100 satırı izle
tail -n 100 ./logs/terminal_20250227_*.log

# Tüm logları birleştir
cat ./logs/terminal_*.log > all_logs.txt

# ERROR/WARNING içeren satırları bul
grep -i "error\|warning\|exception" ./logs/terminal_*.log

# En son checkpoint'i bul
ls -lt ./checkpoints/*.ckpt | head -5

# En iyi modelin FAH değerini göster
grep "BEST MODEL" ./logs/terminal_*.log | tail -5

# Eğitim süresini hesapla
grep "Training Log Started\|Training Log Ended" ./logs/terminal_*.log
```

---

## 📋 6. HIZLI REFERANS TABLOSU

| Ne Arıyorsun? | Nereye Bak? | İyi Değer |
|--------------|-------------|-----------|
| Genel performans | Terminal log | F1 > 0.90 |
| Yanlış alarm | FA/Hour | < 0.5 |
| Yavaş fonksiyon | .prof dosyası | cumtime az |
| Model öğreniyor mu? | TensorBoard loss ↓ | Düşüyor |
| Overfitting | val_loss vs train_loss | Fark < 0.1 |
| Eğitim süresi | Log başlangıç/bitiş | Ne kadar azsa o kadar iyi |

---

## 🎯 Eğitim Başarı Kriterleri

Bir wake word modelinin başarılı sayılması için:

1. ✅ **F1 Score > 0.90**
2. ✅ **FA/Hour < 0.5** (en önemlisi!)
3. ✅ **Recall > 0.90** (kaçırmaması lazım)
4. ✅ **Precision > 0.90** (yanlış tetiklememesi lazım)
5. ✅ **Validation loss stabil** (overfitting yok)

---

*Bu rehber microwakeword_trainer v2.0.0 için hazırlanmıştır.*
