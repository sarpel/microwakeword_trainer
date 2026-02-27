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
| `--checkpoint` | string | None | **Yes** | Path to model checkpoint (.h5 or .ckpt) |
| `--config` | string | `config/presets/standard.yaml` | No | Path to configuration file |
| `--output` | string | `./models/exported` | No | Output directory for exported files |
| `--model-name` | string | `wake_word` | No | Name for exported model (used in filename) |
| `--no-quantize` | flag | False | No | Disable INT8 quantization |

**Examples:**

```bash
# Basic export
mww-export --checkpoint checkpoints/best.ckpt

# Export with custom name
mww-export --checkpoint checkpoints/best.ckpt --model-name "hey_computer"

# Export to custom directory
mww-export --checkpoint checkpoints/best.ckpt --output /path/to/output

# Export without quantization (for debugging)
mww-export --checkpoint checkpoints/best.ckpt --no-quantize

# Full custom export
mww-export \
    --checkpoint checkpoints/best.ckpt \
    --config config/presets/max_quality.yaml \
    --output ./exports \
    --model-name "hey_jarvis"
```

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

Speaker clustering configuration for data leakage detection.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable speaker clustering. |
| `method` | string | `agglomerative` | Clustering method. |
| `embedding_model` | string | `speechbrain/ecapa-tdnn-voxceleb` | SpeechBrain ECAPA-TDNN model for embeddings. |
| `similarity_threshold` | float | 0.72 | Similarity threshold for clustering (0.0-1.0). |
| `leakage_audit_enabled` | bool | true | Enable train/val leakage detection. |

**Example:**
```yaml
speaker_clustering:
  enabled: true
  method: "agglomerative"
  embedding_model: "speechbrain/ecapa-tdnn-voxceleb"
  similarity_threshold: 0.72
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
mww-export --checkpoint checkpoints/best.ckpt --model-name "hey_computer"
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
