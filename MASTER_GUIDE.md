# microwakeword_trainer — MASTER GUIDE

**Complete reference for training, exporting, and deploying custom wake word models to ESPHome.**

> Version: 2.0.0 | Framework: TensorFlow + CuPy GPU | Target: ESP32 via ESPHome micro_wake_word | Architecture verified: 2026-03-13

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Speaker Clustering (Optional but Recommended)](#4-speaker-clustering-optional-but-recommended)
5. [Configuration System](#5-configuration-system)
6. [Training](#6-training)
7. [Monitoring Training](#7-monitoring-training)
8. [Post-Training: Export to TFLite](#8-post-training-export-to-tflite)
9. [Post-Training: Evaluation](#9-post-training-evaluation)
10. [Post-Training: Auto-Tuning](#10-post-training-auto-tuning)
11. [ESPHome Compatibility Verification](#11-esphome-compatibility-verification)
12. [ESPHome Deployment](#12-esphome-deployment)
13. [Performance Optimization](#13-performance-optimization)
14. [Troubleshooting](#14-troubleshooting)
15. [API Reference](#15-api-reference)
16. [Architectural Constants (IMMUTABLE)](#16-architectural-constants-immutable)
17. [v1 vs v2 Model Differences](#17-v1-vs-v2-model-differences)
18. [Violation Consequence Matrix](#18-violation-consequence-matrix)

---

## 1. Overview & Architecture

### What This Framework Does

microwakeword_trainer trains MixedNet models that detect a custom wake word (e.g., "Hey Computer") and run on ESP32 microcontrollers via ESPHome's `micro_wake_word` component.

### Complete Pipeline

```
Audio Files (WAV)
    │
    ▼
[Speaker Clustering]  ← PyTorch env (optional, prevents data leakage)
    │
    ▼
[Data Ingestion]      ← Validates audio, assigns train/val/test splits
    │
    ▼
[Feature Extraction]  ← 40-bin mel spectrograms at 16kHz
    │
    ▼
[Augmentation]        ← EQ, pitch shift, background noise, RIR, SpecAugment
    │
    ▼
[Training Loop]       ← Two-phase step-based training with class weighting
    │
    ▼
[Hard Negative Mining]← Finds false positives, adds to training
    │
    ▼
[TFLite Export]       ← INT8 quantization, streaming model, 2 subgraphs
    │
    ▼
[ESPHome Manifest]    ← JSON manifest for micro_wake_word component
    │
    ▼
ESP32 Device          ← Real-time wake word detection
```

### Model Architecture: MixedNet

The default model is a MixedNet — a lightweight CNN optimized for edge deployment:

```
Input: [1, 3, 40]  (int8)  ← 3 mel frames × 40 bins
    │
    ▼
Conv2D(32 filters, kernel=5, stride=3)
    │
    ▼
MixConvBlock × 4  (parallel depthwise convs with kernels [5,9,13,21])
    │
    ▼
Dense(1, sigmoid)
    │
    ▼
Output: [1, 1]  (uint8)  ← Wake word probability
```

**Streaming inference**: The exported TFLite model has 2 subgraphs and 6 ring buffer state variables. ESPHome feeds 3 mel frames per inference step (every 30ms).

---

## 2. Environment Setup

### Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 or 3.11 | 3.12 NOT supported |
| CUDA | 12.x | Required for CuPy |
| GPU | NVIDIA (Volta+) | No CPU fallback for SpecAugment |
| RAM | 16GB+ | For max_quality training |
| Storage | 10GB+ | Datasets + checkpoints |

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg

# Verify CUDA
nvidia-smi
nvcc --version  # Should show 12.x
```

### Environment 1: TensorFlow (Training, Export, Evaluation)

```bash
# Create TF environment
python3.11 -m venv ~/venvs/mww-tf
source ~/venvs/mww-tf/bin/activate

# Install dependencies
cd /path/to/microwakeword_trainer
pip install -r requirements.txt

# Verify GPU access
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### Environment 2: PyTorch (Speaker Clustering Only)

```bash
# Create PyTorch environment
python3.11 -m venv ~/venvs/mww-torch
source ~/venvs/mww-torch/bin/activate

# Install dependencies
pip install -r requirements-torch.txt

# Verify
python -c "import torch; print(torch.__version__)"
python -c "import speechbrain; print(speechbrain.__version__)"
```

### Shell Aliases (Add to ~/.bashrc or ~/.zshrc)

```bash
alias mww-tf='source ~/.venvs/mww-tf/bin/activate && cd $PROJECT_DIR'
alias mww-torch='source ~/.venvs/mww-torch/bin/activate && cd $PROJECT_DIR'
```

### GPU Environment Variables (Set Before Training)

```bash
export CUDA_VISIBLE_DEVICES=0           # Use first GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true   # Prevent OOM by growing memory
export TF_GPU_ALLOCATOR=cuda_malloc_async  # Faster allocator
export TF_DETERMINISTIC_OPS=1           # Reproducibility
```

---

## 3. Dataset Preparation

### Directory Structure

```
dataset/
├── positive/           # Wake word recordings (REQUIRED)
│   ├── speaker_001/    # Organize by speaker (prevents data leakage)
│   │   ├── rec_001.wav
│   │   ├── rec_002.wav
│   │   └── ...
│   ├── speaker_002/
│   └── ...
├── negative/           # Background speech — NOT the wake word (REQUIRED)
│   └── speech/
│       ├── conv_001.wav
│       └── ...
├── hard_negative/      # Sounds similar to wake word (RECOMMENDED)
│   ├── false_positive_001.wav
│   └── ...
├── background/         # Ambient noise (RECOMMENDED)
│   ├── noise_001.wav
│   └── ...
└── rirs/              # Room impulse responses for reverb (OPTIONAL)
    └── reverb_001.wav
```

### Audio Requirements

| Property | Requirement | Notes |
|----------|-------------|-------|
| Format | WAV, 16-bit PCM | Other formats auto-converted |
| Sample rate | 16kHz | Will be resampled if needed |
| Duration | 1–3 seconds | Per clip |
| Channels | Mono | Stereo auto-converted |

### Minimum Dataset Sizes

| Type | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| Positive | 100 | 1000+ | More = better recall |
| Negative | 1000 | 10000+ | More = fewer false triggers |
| Hard Negative | 50 | 500+ | Sounds similar to wake word |

### Recording Tips

- Record from **5+ different speakers** for diversity
- Record at **various distances** (0.5–3 meters)
- Record in **different rooms** (kitchen, bedroom, office)
- Include **variations in tone and speed**
- Record at **different times of day** (morning voice vs evening)
- Include **background noise** during recording

### Create Dataset Structure

```bash
mkdir -p dataset/{positive,negative,hard_negative,background,rirs}
```

### Generate Synthetic Test Dataset (for pipeline testing)

```bash
mww-tf
python scripts/generate_test_dataset.py
# Creates synthetic audio in dataset/ for testing the pipeline
```

---

## 4. Speaker Clustering (Optional but Recommended)

Speaker clustering prevents **train/validation data leakage** — the same speaker appearing in both splits. Uses SpeechBrain ECAPA-TDNN embeddings to group audio by speaker identity.

**Requires PyTorch environment.**

### Prerequisites

```bash
mww-torch

# Login to Hugging Face (free account required)
huggingface-cli login
```

### Step 1: Analyze Clusters (Dry Run — No Files Moved)

```bash
# Cluster positive dataset (default)
mww-cluster-analyze --config max_quality

# Cluster all datasets at once
mww-cluster-analyze --config max_quality --dataset all

# Cluster specific dataset
mww-cluster-analyze --config max_quality --dataset negative

# Use explicit speaker count (RECOMMENDED for short clips 1-3s)
# Threshold-based clustering over-fragments short wake word clips
mww-cluster-analyze --config max_quality --n-clusters 200

# Combine options
mww-cluster-analyze --config max_quality --dataset all --n-clusters 200 --threshold 0.65
```

**Output per dataset:**
- `cluster_output/{dataset}_namelist.json` — file → speaker mapping
- `cluster_output/{dataset}_cluster_report.txt` — human-readable report

**Review the report** before proceeding. Check that speakers are grouped correctly.

### mww-cluster-analyze Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | **required** | Config preset name or path to YAML |
| `--override` | string | None | Override config file |
| `--dataset` | string | `positive` | `positive`, `negative`, `hard_negative`, or `all` |
| `--n-clusters` | int | None | Explicit cluster count (overrides threshold) |
| `--threshold` | float | from config | Override similarity threshold |
| `--output-dir` | string | `./cluster_output` | Output directory |
| `--max-files` | int | None | Limit files (for testing) |

### Step 2: Organize Files by Speaker

```bash
# Preview first (ALWAYS do this first)
mww-cluster-apply --namelist cluster_output/positive_namelist.json --dry-run

# Organize a single dataset
mww-cluster-apply --namelist cluster_output/positive_namelist.json

# Organize all datasets at once
mww-cluster-apply --namelist-dir cluster_output

# Undo if something looks wrong
mww-cluster-apply --undo cluster_output/positive_backup_manifest.json
```

### mww-cluster-apply Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--namelist` | string | None | Path to single namelist JSON |
| `--namelist-dir` | string | None | Directory with `*_namelist.json` files |
| `--undo` | string | None | Path to backup manifest to reverse |
| `--output-dir` | string | `./cluster_output` | Backup manifest directory |
| `--dry-run` | flag | off | Preview without moving files |

> **Note:** `--namelist`, `--namelist-dir`, and `--undo` are mutually exclusive.
> A backup manifest is saved automatically before any files are moved.

### When to Skip Clustering

- You already organized files by speaker into subdirectories
- Single-user wake word (only one speaker)
- You prefer directory-based speaker detection

---

## 5. Configuration System

### Preset Configurations

| Preset | Training Steps | Batch Size | Augmentation | Time | Use Case |
|--------|---------------|------------|--------------|------|----------|
| `fast_test` | 3000 (2k+1k) | 32 | Disabled | ~1 hour | Quick iteration |
| `max_quality` | 30000 (20k+10k) | 128 | max_quality | ~8 hours | Production |
| `max_quality` | 70000 (50k+20k) | 128 | Full | ~24 hours | Best accuracy |

### Loading Configurations

```bash
# Use preset by name
mww-train --config max_quality

# Use preset file path
mww-train --config config/presets/max_quality.yaml

# Use preset + override file
mww-train --config max_quality --override my_config.yaml

# Full custom config
mww-train --config my_full_config.yaml
```

```python
# Programmatic loading
from config.loader import load_full_config, load_preset

config = load_preset("max_quality")
config = load_full_config("max_quality", "my_override.yaml")
config = load_full_config("/path/to/my_config.yaml")
```

### Configuration Sections

#### hardware (IMMUTABLE — Do Not Change)

```yaml
hardware:
  sample_rate_hz: 16000    # IMMUTABLE: ESPHome hardware clock
  mel_bins: 40             # IMMUTABLE: Feature tensor width
  window_size_ms: 30       # IMMUTABLE: 480 samples per FFT window
  window_step_ms: 10       # IMMUTABLE: 160 samples per hop
  clip_duration_ms: 1000   # Configurable: training clip length
```

> ⛔ **WARNING**: `sample_rate_hz`, `mel_bins`, `window_size_ms`, `window_step_ms` are burned into ESPHome firmware. Changing them produces a model that silently fails on device.

#### paths

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

Supports environment variable substitution:
```yaml
paths:
  checkpoint_dir: ${CHECKPOINT_DIR:-./checkpoints}
  positive_dir: ${DATA_ROOT}/positive
```

#### training

```yaml
training:
  # Two-phase training: [phase1_steps, phase2_steps]
  training_steps: [20000, 10000]
  learning_rates: [0.001, 0.0001]
  batch_size: 128
  eval_step_interval: 500

  # Class weights (higher = penalize more)
  positive_class_weight: [1.0, 1.0]
  negative_class_weight: [20.0, 20.0]
  hard_negative_class_weight: [40.0, 40.0]

  # SpecAugment (GPU-only, disabled by default)
  time_mask_max_size: [0, 0]
  time_mask_count: [0, 0]
  freq_mask_max_size: [0, 0]
  freq_mask_count: [0, 0]

  # Checkpoint selection strategy
  minimization_metric: "ambient_false_positives_per_hour"
  target_minimization: 0.5
  maximization_metric: "average_viable_recall"
```

#### model

```yaml
model:
  architecture: "mixednet"
  first_conv_filters: 32
  first_conv_kernel_size: 5
  stride: 3                      # MUST be 3 for ESPHome streaming
  pointwise_filters: "64,64,64,64"
  mixconv_kernel_sizes: "[5],[7,11],[9,15],[23]"
  repeat_in_block: "1,1,1,1"
  residual_connection: "0,1,1,1"
  dropout_rate: 0.2              # 0.2 for regularization
  l2_regularization: 0.001        # 0.001 for regularization
```

> ⛔ **ESPHome Requirements**: Output layer MUST be `Dense(1, sigmoid)`. `stride` MUST be 3. Input shape MUST be `[1, 3, 40]`.

#### augmentation

```yaml
augmentation:
  # Probability of applying each augmentation (0.0 = disabled)
  SevenBandParametricEQ: 0.1
  TanhDistortion: 0.1
  PitchShift: 0.1              # Capped at 1.3x speed internally
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

  # Source paths
  impulse_paths: ["./dataset/rirs"]
  background_paths: ["./dataset/background"]
  augmentation_duration_s: 3.2
```

#### performance

```yaml
performance:
  gpu_only: true
  mixed_precision: true          # FP16 for 2-3x speedup
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

#### speaker_clustering

```yaml
speaker_clustering:
  enabled: true
  method: "agglomerative"        # agglomerative or threshold
  embedding_model: "speechbrain/ecapa-tdnn-voxceleb"
  similarity_threshold: 0.72    # Used when n_clusters is null
  n_clusters: null              # Set to known speaker count for short clips
  leakage_audit_enabled: true
```

#### hard_negative_mining

```yaml
hard_negative_mining:
  enabled: true
  fp_threshold: 0.8             # False positive detection threshold
  max_samples: 5000
  mining_interval_epochs: 5
```

#### export

```yaml
export:
  wake_word: "Hey Computer"
  author: "Your Name"
  website: "https://github.com/yourusername"
  trained_languages: ["en"]
  quantize: true
  inference_input_type: "int8"   # IMMUTABLE: ESPHome requirement
  inference_output_type: "uint8" # IMMUTABLE: ESPHome requirement (NOT int8!)
  probability_cutoff: 0.97       # 0.70 for testing, 0.95-0.98 for production
  sliding_window_size: 5
  tensor_arena_size: 0           # Auto-calculate from exported TFLite (recommended)
  minimum_esphome_version: "2024.7.0"
```

### Minimal Override Example

Create `my_override.yaml`:

```yaml
# Override max_quality preset with your settings
export:
  wake_word: "Hey Jarvis"
  author: "Your Name"
  website: "https://github.com/yourusername"

training:
  batch_size: 64  # Reduce if OOM errors

model:
  first_conv_filters: 20  # Smaller model for limited RAM
```

Run:
```bash
mww-train --config max_quality --override my_override.yaml
```

---

## 6. Training

### Switch to TF Environment

```bash
mww-tf
# or: source ~/venvs/mww-tf/bin/activate && cd /path/to/project
```

### mww-train Command

```bash
mww-train [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | `max_quality` | Preset name or path to YAML config |
| `--override` | string | None | Override config file (merged with base) |

### Training Examples

```bash
# Quick test (validates pipeline, ~1 hour)
mww-train --config fast_test

# max_quality production training (~8 hours)
mww-train --config max_quality

# Best quality training (~24 hours)
mww-train --config max_quality

# Custom wake word with override
mww-train --config max_quality --override my_config.yaml

# Full custom config
mww-train --config /path/to/my_full_config.yaml
```

### What Happens During Training

1. **Config validation** — All paths and parameters validated
2. **Data ingestion** — Audio files discovered, validated, split into train/val/test
3. **Feature extraction** — Mel spectrograms computed and cached to `data/processed/`
4. **Phase 1 training** — High learning rate (0.001), 20k steps
5. **Evaluation** — Every 500 steps: FAH, recall, precision, F1
6. **Hard negative mining** — Every 5 epochs: finds false positives, adds to training
7. **Phase 2 training** — Low learning rate (0.0001), 10k steps
8. **Best model selection** — Checkpoint with lowest FAH + highest recall saved

### Output Files

```
checkpoints/
├── best_weights.weights.h5           # Best model (lowest FAH + highest recall)
├── best_fah_step_XXXX.weights.h5     # Best FAH checkpoint
└── checkpoint_step_XXXX.weights.h5   # Regular interval checkpoints

logs/
├── terminal_YYYYMMDD_HHMMSS.log  # Full training log
└── events.out.tfevents.*          # TensorBoard events

profiles/
└── training_step_XXXX.prof  # cProfile performance data
```

### Training Metrics Explained

| Metric | Target | Description |
|--------|--------|-------------|
| `loss` | Decreasing | Binary cross-entropy loss |
| `accuracy` | > 0.95 | Overall classification accuracy |
| `precision` | > 0.90 | Fraction of detections that are correct |
| `recall` | > 0.90 | Fraction of wake words detected |
| `F1` | > 0.90 | Harmonic mean of precision and recall |
| **`FA/Hour`** | **< 0.5** | **False activations per hour — most critical!** |

**FA/Hour thresholds:**
- 🟢 < 0.5 — Excellent, production-ready
- 🟡 0.5–2.0 — Acceptable, can improve
- 🔴 > 2.0 — Too many false triggers

---

## 7. Monitoring Training

### TensorBoard

```bash
# In a separate terminal (TF environment)
tensorboard --logdir ./logs
# Open: http://localhost:6006
```

**Key metrics to watch:**
- `epoch_loss` — Should decrease steadily
- `val_loss` — Should track `epoch_loss` (if diverging = overfitting)
- `learning_rate` — Should step down between phases

### Terminal Log Analysis

```bash
# Watch live training output
tail -f ./logs/terminal_$(date +%Y%m%d)*.log

# Find best model checkpoints
grep "BEST MODEL" ./logs/terminal_*.log | tail -10

# Find errors/warnings
grep -i "error\|warning\|exception" ./logs/terminal_*.log

# Calculate training duration
grep "Training Log Started\|Training Log Ended" ./logs/terminal_*.log
```

### Training Progress Format

```
Phase 1 • 500/30000 • 1.7% • 0:02:14 elapsed • 2:10:45 remaining • loss=0.2341 acc=0.8912 lr=0.001000
```

### Validation Results Format

```
📊 Validation Results — Step 500/30000
┌──────────────────────┬────────┐
│ Accuracy             │ 0.8912 │
│ Precision            │ 0.8234 │
│ Recall               │ 0.7567 │
│ F1 Score             │ 0.7889 │
│ AUC-ROC              │ 0.9234 │
│ Ambient FA/Hour      │ 3.45   │  ← Most important!
│ Recall @ No FAPH     │ 0.6789 │
└──────────────────────┴────────┘
```

### Profile Analysis

```bash
# Analyze performance bottlenecks
python -c "
import pstats
p = pstats.Stats('profiles/training_step_XXXX.prof')
p.sort_stats('cumulative')
p.print_stats(20)
"

# Using TrainingProfiler
from src.training.profiler import TrainingProfiler
summary = TrainingProfiler.get_summary('./profiles/training_step_XXXX.prof', top_n=30)
print(summary)
```

---

## 8. Post-Training: Export to TFLite

### mww-export Command

```bash
mww-export [OPTIONS]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--checkpoint` | string | None | **Yes** | Path to checkpoint (.h5 or .ckpt) |
| `--config` | string | `config/presets/max_quality.yaml` | No | Config file path |
| `--output` | string | `./models/exported` | No | Output directory |
| `--model-name` | string | `wake_word` | No | Model filename (without extension) |

### Export Examples

```bash
# Basic export (uses best checkpoint)
mww-export --checkpoint checkpoints/best.ckpt

# Export with custom name
mww-export --checkpoint checkpoints/best.ckpt --model-name "hey_computer"

mww-export --checkpoint models/checkpoints/best_weights.weights.h5 --output models/exported/ --config config/presets/max_quality.yaml
# Export to custom directory
mww-export --checkpoint checkpoints/best.ckpt --output /path/to/output

# Export with explicit preset path
mww-export --checkpoint checkpoints/best.ckpt --config config/presets/max_quality.yaml

# Full custom export
mww-export \
    --checkpoint checkpoints/best.ckpt \
    --config config/presets/max_quality.yaml \
    --output ./exports \
    --model-name "okay_nabu"
```

### Generated Files

```
models/exported/
├── hey_computer.tflite      # The model file (deploy this to ESPHome)
├── manifest.json            # ESPHome manifest
└── streaming/               # Streaming SavedModel (for debugging)
```

### Manifest Format

```json
{
  "type": "micro",
  "wake_word": "Hey Computer",
  "author": "Your Name",
  "website": "https://github.com/sarpel/microwakeword-training-platform",
  "model": "hey_computer.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 0,  // auto-calculated during export
    "minimum_esphome_version": "2024.7.0"
  }
}
```

#### Migrating from V1 (flat) to V2 (nested `micro`) Format

| V1 flat field | V2 location | Notes |
|---|---|---|
| `probability_cutoff` | `micro.probability_cutoff` | Moved inside `micro` object |
| `sliding_window_size` | `micro.sliding_window_size` | Moved inside `micro` object |
| `feature_step_size` | `micro.feature_step_size` | Moved inside `micro` object |
| `tensor_arena_size` | `micro.tensor_arena_size` | Moved inside `micro` object; set `0` for auto-resolve |
| `minimum_esphome_version` | `micro.minimum_esphome_version` | Moved inside `micro` object |
| `wake_word`, `author`, `website`, `model` | Top-level (unchanged) | Already at top level in V1 |
| `trained_languages` | Top-level (new in V2) | **NEW field in V2** — add if missing; recommended default: `["en"]` |
| `version` | Top-level `"version": 2` | **Must be updated from `1` to `2`** |

### Mandatory Export Pipeline Flags

> ⛔ **CRITICAL — These flags are not optional. Omitting any of them produces a model that fails to load on device or produces silently wrong predictions.**

The export pipeline has two mandatory stages (matching `OHF-Voice/micro-wake-word/microwakeword/utils.py`):

**Stage 1 — Non-Streaming → Streaming SavedModel:**
```python
# MANDATORY: materialize ring buffer VAR_HANDLE/READ_VARIABLE/ASSIGN_VARIABLE ops
converted_model = model_to_saved(
    model_non_stream=model,
    config=config,
    mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
)
# MANDATORY: use tf.keras.export.ExportArchive — NOT model.export() (causes quantization errors)
```

**Stage 2 — Streaming SavedModel → Quantized TFLite:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)
converter.optimizations = {tf.lite.Optimize.DEFAULT}

# MANDATORY: without this, state variable payloads remain float32
# and TFLite Micro int8-only kernel resolver fails at load time on device
converter._experimental_variable_quantization = True

converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.uint8    # UINT8. ALWAYS. NOT int8.
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
```

> ⛔ **Use `tf.lite.TFLiteConverter` from max_quality `tensorflow`. Do NOT use `ai_edge_litert` for export.** `ai_edge_litert` is only for inference/testing.

### Representative Dataset Requirements

The calibration representative dataset must satisfy:
1. **Minimum 500 training spectrograms** — fewer samples increase quantization noise
2. **Boundary anchor points on first sample** — `sample[0][0,0] = 0.0` (minimum) and `sample[0][0,1] = 26.0` (maximum) — pins the quantization scale to the correct range
3. **Slice each spectrogram by `stride=3`** — each yielded sample must have shape `[3, 40]` matching the runtime input cadence

Without the boundary anchors, the quantizer may choose a different scale that compresses the dynamic range, making predictions effectively non-functional even though the model loads successfully.

### Programmatic Export

```python
from src.export.tflite import convert_model_saved
from config.loader import load_full_config

config = load_full_config("max_quality")
convert_model_saved(
    checkpoint_path="checkpoints/best.ckpt",
    output_dir="models/exported/",
    model_name="hey_computer",
    config=config,
    quantize=True
)
```

---

## 9. Post-Training: Evaluation

### Evaluate on Test Set

```bash
# Evaluate checkpoint on test split
python scripts/evaluate_model.py \
    --checkpoint tuning_results/checkpoints/tuned_fah0.000_rec1.000_iter3.weights.h5 \
    --config max_quality \
    --split test \
    --analyze

# Evaluate TFLite model
python scripts/evaluate_model.py \
    --tflite models/exported/wake_word.tflite \
    --config max_quality \
    --split test

# JSON output for CI/CD
python scripts/evaluate_model.py \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config max_quality \
    --json
```

### Programmatic Evaluation

```python
from src.data.dataset import WakeWordDataset
from src.evaluation.metrics import MetricsCalculator
import numpy as np

# Load dataset
dataset = WakeWordDataset(config)
dataset.build()

# Get test generator
test_gen = dataset.test_generator_factory(max_time_frames)()

# Collect predictions
y_true, y_scores = [], []
for features, labels in test_gen:
    predictions = model.predict(features)
    y_true.extend(labels)
    y_scores.extend(predictions)

# Calculate metrics
calc = MetricsCalculator(
    y_true=np.array(y_true),
    y_score=np.array(y_scores)
)
metrics = calc.compute_all_metrics(ambient_duration_hours=10.0)
print(metrics)
```

### Check for Data Leakage

If your metrics look suspiciously perfect (accuracy > 0.999, FA/Hour = 0.00):

```bash
# Check speaker overlap between splits
mww-cluster-analyze --config max_quality --dataset all
# Review cluster_output/*_cluster_report.txt

# Check file overlap
find dataset/positive/train -name "*.wav" | sort > train_files.txt
find dataset/positive/val -name "*.wav" | sort > val_files.txt
comm -12 train_files.txt val_files.txt  # Should be empty
```

---

## 10. Post-Training: Auto-Tuning

Auto-tuning iteratively fine-tunes a trained model to achieve target metrics without retraining from scratch.

**Targets:**
- FAH (False Activations per Hour) < 0.3
- Recall > 0.92

### mww-autotune Command

```bash
mww-autotune [OPTIONS]
```

Argument | Type | Default | Description
----------|------|---------|-------------
| `--checkpoint` | string | **required** | Path to trained checkpoint (.weights.h5) |
| `--config` | string | `max_quality` | Config preset or path |
| `--override` | string | None | Override config file |
| `--target-fah` | float | 0.3 | Target FAH value |
| `--target-recall` | float | 0.92 | Target recall value |
| `--max-iterations` | int | 100 | Maximum tuning iterations |
| `--output-dir` | string | `./tuning` | Output directory for tuned checkpoints |
| `--patience` | int | 10 | Stop early if no improvement after N iterations |
| `--dry-run` | flag | off | Validate config without running tuning |
| `--verbose` / `-v` | flag | off | Enable verbose output |

### Auto-Tuning Examples

```bash
# Basic auto-tuning with defaults
mww-autotune --checkpoint ./models/checkpoints/best_weights.weights.h5

# Custom targets
mww-autotune \
    --checkpoint models/checkpoints/best_weights.weights.h5  \
    --config max_quality \
    --target-fah 0.2 \
    --target-recall 0.95

# With more iterations
mww-autotune \
    --checkpoint models/checkpoints/best_weights.weights.h5  \
    --config max_quality \
    --max-iterations 20 \
    --output-dir ./tuning_results
```

### When to Use Auto-Tuning

- After initial training, if FAH > 0.5 or recall < 0.90
- When you want to improve metrics without full retraining
- As a final polish step before deployment

### Programmatic Auto-Tuning

```python
from src.tuning.autotuner import AutoTuner, TuningTarget
from config.loader import load_full_config

config = load_full_config("max_quality")
target = TuningTarget(max_fah=0.2, min_recall=0.95, max_iterations=50)

tuner = AutoTuner(config=config, target=target)
result = tuner.tune(checkpoint_path="checkpoints/best.ckpt")
print(f"Final FAH: {result.current_fah}, Recall: {result.current_recall}")
```

---

## 11. ESPHome Compatibility Verification

### Verify TFLite Model

```bash
# Basic check
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Verbose output (shows all tensor details)
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose

# JSON output for CI/CD
python scripts/verify_esphome.py models/exported/wake_word.tflite --json
```

### Expected Output

```
✓ Subgraphs: 2 (correct)
✓ Input shape: [1, 3, 40] (correct)
✓ Input dtype: int8 (correct)
✓ Output shape: [1, 1] (correct)
✓ Output dtype: uint8 (correct)
✓ Quantization: enabled (correct)
✓ ESPHome compatible: YES
```

### Common Compatibility Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Wrong output dtype (int8) | Used int8 instead of uint8 | Set `inference_output_type: "uint8"` |
| Wrong subgraph count | Export failed | Re-export with correct config |
| Missing quantization | Export pipeline misconfigured | Re-export with default INT8 settings (`inference_input_type=int8`, `inference_output_type=uint8`) |
| Wrong input shape | Wrong stride | Set `stride: 3` in model config |

### Programmatic Verification

```python
from src.export.model_analyzer import analyze_model_architecture

results = analyze_model_architecture("models/exported/wake_word.tflite")
print(results)
```

---

## 12. ESPHome Deployment

### Copy Files to ESPHome

```bash
mkdir -p /config/esphome/models
cp models/exported/hey_computer.tflite /config/esphome/models/
cp models/exported/manifest.json /config/esphome/models/
```

### ESPHome YAML Configuration

```yaml
# Basic wake word detection
micro_wake_word:
  models:
    - model: models/hey_katya.tflite
      probability_cutoff: 0.97

# With voice assistant
voice_assistant:
  wake_word: "Hey Katya"
  on_wake_word_detected:
    - logger.log: "Wake word detected!"
```

### Probability Cutoff Tuning

| Cutoff | Use Case | Trade-off |
|--------|----------|-----------|
| 0.70 | Testing/development | More false triggers |
| 0.90 | Balanced | Good for most use cases |
| 0.95–0.97 | Production | Fewer false triggers, may miss some |
| 0.98–0.99 | Strict | Very few false triggers, may miss more |

---

## 13. Performance Optimization

### Mixed Precision Training (2-3x Speedup)

```yaml
performance:
  mixed_precision: true  # Enable FP16 training
```

```python
from src.utils.performance import configure_mixed_precision
configure_mixed_precision(enabled=True)
```

### GPU Memory Configuration

```python
from src.utils.performance import configure_tensorflow_gpu

# Allow memory growth (prevents OOM)
configure_tensorflow_gpu(memory_growth=True)

# Limit to specific amount
configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=8192)
```

### Threading Configuration

```python
from src.utils.performance import set_threading_config

# Maximize CPU utilization for data loading
set_threading_config(inter_op_parallelism=16, intra_op_parallelism=16)
```

### Optimized tf.data Pipeline

```python
from src.data.tfdata_pipeline import OptimizedDataPipeline, create_optimized_dataset
from src.data.dataset import WakeWordDataset

dataset = WakeWordDataset(config)
pipeline = OptimizedDataPipeline(dataset, config, batch_size=128)

# Get optimized training dataset
train_ds = pipeline.create_training_pipeline()

# Benchmark pipeline performance
from src.data.tfdata_pipeline import benchmark_pipeline
benchmark_pipeline(train_ds, num_batches=100)
```

### PerformanceOptimizer (High-Level)

```python
from src.training.performance_optimizer import PerformanceOptimizer
from src.data.dataset import WakeWordDataset

dataset = WakeWordDataset(config)
optimizer = PerformanceOptimizer(config)
optimizer.enable_all()

# Get optimized datasets
train_ds, val_ds = optimizer.create_datasets(dataset)

# Use with model
model.fit(train_ds, validation_data=val_ds)
```

### Performance Benchmarks

| Optimization | Speedup | Trade-off |
|--------------|---------|-----------|
| Mixed precision (FP16) | 2–3x | Minimal accuracy loss |
| Larger batch size | 1.5x | More VRAM needed |
| CuPy SpecAugment | 5–10x | Requires GPU |
| tf.data pipeline | 1.2–2x | More memory |

---

## 14. Troubleshooting

### GPU Out of Memory (OOM)

```yaml
# Reduce batch size
training:
  batch_size: 256  # Default: 128

# Reduce workers
performance:
  num_workers: 16
  max_memory_gb: 32
```

```python
# Limit GPU memory
from src.utils.performance import configure_tensorflow_gpu
configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=4096)
```

### CuPy Not Found / GPU Not Available

```bash
# Verify CUDA version
nvcc --version  # Must show 12.x

# Reinstall CuPy
pip uninstall cupy-cuda12x
pip install cupy-cuda12x>=13.0

# Verify CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### Model Not Detecting Wake Word

1. Check dataset balance: positive:negative ratio should be 1:10+
2. Verify audio quality: no clipping, good SNR
3. Increase training steps: `training_steps: [50000, 20000]`
4. Add more speaker diversity (5+ speakers)
5. Enable all augmentations
6. Increase negative class weight: `negative_class_weight: [30.0, 30.0]`

### Too Many False Triggers

```yaml
# Increase detection threshold
export:
  probability_cutoff: 0.98

# More aggressive negative weighting
training:
  hard_negative_class_weight: [60.0, 60.0]

# Add more hard negative samples
hard_negative_mining:
  enabled: true
  max_samples: 10000
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

### Loss Not Decreasing

1. Learning rate too low → increase to 0.001
2. Insufficient data → add more samples
3. Augmentation too aggressive → reduce probabilities
4. Check GPU is actually being used: `nvidia-smi`

### Validation Metrics Too Good (Suspicious)

If accuracy > 0.999 or FA/Hour = 0.00:
1. **Data leakage** — same speaker in train and val
2. **Overfitting** — too many training steps
3. **Incorrect labeling** — negative samples contain wake word

```bash
# Check for data leakage
mww-cluster-analyze --config max_quality --dataset all
# Review cluster_output/*_cluster_report.txt
```

### ESPHome Model Not Working

```bash
# Verify compatibility
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose

# Common issues:
# - output dtype must be uint8 (NOT int8)
# - must have exactly 2 subgraphs
# - must have quantization enabled
```

### Model Too Large for ESP32

```yaml
# Reduce model size
model:
  first_conv_filters: 20          # Reduced from 32
  pointwise_filters: "40,40,40,40"  # Reduced from 64

# Auto-calculate arena size from exported model (set > 0 to override with explicit bytes)
export:
  tensor_arena_size: 0
```

---

## 15. API Reference

### Training API

```python
from src.training.trainer import Trainer, train, main
from config.loader import load_full_config

# Load config
config = load_full_config("max_quality", "my_override.yaml")

# Create trainer
trainer = Trainer(config)

# Train (provide data factories)
trainer.train(
    train_data_factory=train_factory,
    val_data_factory=val_factory,
    input_shape=(49, 40)
)
```

### Dataset API

```python
from src.data.dataset import WakeWordDataset, load_dataset
from src.data.ingestion import load_clips, Split, Label

# Load clips (discovers audio files)
clips = load_clips("config/presets/max_quality.yaml")
train_clips = clips.get_split(Split.TRAIN)
val_clips = clips.get_split(Split.VAL)

# Build dataset with features
dataset = WakeWordDataset(config)
dataset.build()

# Get generators
train_gen = dataset.train_generator_factory(max_time_frames=49)
val_gen = dataset.val_generator_factory(max_time_frames=49)
```

### Feature Extraction API

```python
from src.data.features import MicroFrontend, SpectrogramGeneration, FeatureConfig

# Configure features
feature_config = FeatureConfig(
    sample_rate_hz=16000,
    mel_bins=40,
    window_size_ms=30,
    window_step_ms=10
)

# Extract features from file
gen = SpectrogramGeneration(feature_config)
spectrogram = gen.generate_from_file("audio.wav")  # Shape: [time_frames, 40]
```

### Model API

```python
from src.model.architecture import build_model
from config.loader import load_full_config

config = load_full_config("max_quality")

# Build model
model = build_model(config.model, input_shape=(49, 40))
model.summary()
```

### Evaluation API

```python
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.fah_estimator import FAHEstimator
import numpy as np

# Calculate all metrics
calc = MetricsCalculator(
    y_true=np.array(y_true),
    y_score=np.array(y_scores)
)
metrics = calc.compute_all_metrics(ambient_duration_hours=10.0)

# Key metrics
print(f"FAH: {metrics['ambient_false_positives_per_hour']}")
print(f"Recall: {metrics['recall_at_threshold_0.5']}")
print(f"AUC-ROC: {metrics['auc_roc']}")
```

### Export API

```python
from src.export.tflite import convert_model_saved
from src.export.manifest import generate_manifest
from src.export.model_analyzer import analyze_model_architecture

# Export to TFLite
convert_model_saved(
    checkpoint_path="checkpoints/best.ckpt",
    output_dir="models/exported/",
    model_name="hey_computer",
    config=config,
    quantize=True
)

# Generate manifest
generate_manifest(
    model_path="models/exported/hey_computer.tflite",
    config=config,
    output_path="models/exported/manifest.json"
)

# Analyze model
results = analyze_model_architecture("models/exported/hey_computer.tflite")
```

### Config API

```python
from config.loader import load_full_config, load_preset, ConfigLoader

# Load preset
config = load_preset("max_quality")

# Load with override
config = load_full_config("max_quality", "my_override.yaml")

# Access config sections
print(config.training.batch_size)
print(config.model.first_conv_filters)
print(config.export.wake_word)
print(config.hardware.sample_rate_hz)
```

---

## 16. Architectural Constants (IMMUTABLE)

> ⛔ These values are burned into ESPHome firmware. **DO NOT CHANGE THEM.**
> Source: `ARCHITECTURAL_CONSTITUTION.md` — verified 2026-03-13 from TFLite flatbuffer binary, ESPHome C++ source, and OHF-Voice training pipeline.

### Audio Frontend Constants

| Constant | Value | Why Immutable |
|----------|-------|---------------|
| `sample_rate_hz` | 16,000 Hz | ESPHome ADC hardware clock; hardcoded in firmware |
| `mel_bins` | 40 | Defines feature tensor width; changing it changes model input shape |
| `window_size_ms` | 30 ms | 480 samples per FFT window; baked into audio frontend C code |
| `window_step_ms` | 10 ms | 160 samples per hop; v2 value (v1 was 20 ms) |
| `upper_band_limit_hz` | 7,500 Hz | Nyquist constraint for 16 kHz with margin |
| `lower_band_limit_hz` | 125 Hz | DC rejection floor |
| `enable_pcan` | True | Per-Channel Amplitude Normalization; disabling changes entire feature distribution |

### PCAN and Noise Reduction Parameters

These are compiled into `pymicro-features` (`rhasspy/pymicro-features`) and cannot be changed at runtime.

| Parameter | Value |
|-----------|-------|
| `pcan_strength` | 0.95 |
| `pcan_offset` | 80.0 |
| `pcan_gain_bits` | 21 |
| `noise_even_smoothing` | 0.025 |
| `noise_odd_smoothing` | 0.06 |
| `noise_min_signal_remaining` | 0.05 |
| `log_scale_shift` | 6 |

### Model I/O Contract

| Property | Value | Notes |
|----------|-------|-------|
| Input shape | `[1, 3, 40]` | 3 mel frames × 40 bins |
| Input dtype | `int8` | ESPHome runtime check — model fails to load if violated |
| Input quantization scale | `0.10196078568696976` (≈ 26/255) | Maps int8[-128,127] → float[0.0, ~26.0] |
| Input quantization zero_point | `-128` | |
| Output shape | `[1, 1]` | Single probability |
| Output dtype | **`uint8`** | ESPHome reads `output->data.uint8[0]` — **NOT int8, NOT float32** |
| Output quantization scale | `0.00390625` (= 1/256) | Maps uint8[0,255] → float[0.0, ~1.0] |
| Output quantization zero_point | `0` | |
| Subgraphs | 2 | Main inference (Subgraph 0) + initialization (Subgraph 1) |
| Quantization | INT8 | Required for micro_wake_word |
| Subgraph 0 tensors (canonical) | **94** | Using `tf.lite` interpreter path |
| Subgraph 1 tensors | **12** | |

> **Note on tensor counts:** `ai_edge_litert` may report 95 tensors for Subgraph 0 due to an exposed runtime scratch tensor. That is not an architectural difference. This project's canonical convention is **94 / 12**.

### Inference Timing

| Constant | Value | Derivation |
|----------|-------|------------|
| New frames per inference call | 3 | `stride` (read from input tensor dim[1] at runtime) |
| Feature frame period | 10 ms | `window_step_ms` |
| Inference period | 30 ms | `stride × window_step_ms = 3 × 10` |
| Samples consumed per inference | 480 | `stride × 160 = 3 × 160` |

### Streaming State Variables (6 total — exact shapes required)

| Variable | Shape | Bytes | Notes |
|----------|-------|-------|-------|
| `stream` | `[1, 2, 1, 40]` | 80 | Ring buffer before first Conv2D; `kernel(5) - global_stride(3) = 2` |
| `stream_1` | `[1, 4, 1, 32]` | 128 | MixConv block 0; single kernel=5, `5-1=4` |
| `stream_2` | `[1, 10, 1, 64]` | 640 | MixConv block 1; dual kernels [7,11], `max(7,11)-1=10` |
| `stream_3` | `[1, 14, 1, 64]` | 896 | MixConv block 2; dual kernels [9,15], `max(9,15)-1=14` |
| `stream_4` | `[1, 22, 1, 64]` | 1,408 | MixConv block 3; single kernel=23, `23-1=22` |
| `stream_5` | `[1, 5, 1, 64]` | 320 | Temporal flatten buffer; pre-flatten dim(6) minus 1 |
| **Total state memory** | | **3,472 bytes** | |

> **Ring buffer laws:** For `stream` (before strided conv): `buffer_frames = kernel_size - global_stride`. For `stream_1` through `stream_4` (after strided conv, internal stride=1): `buffer_frames = max_kernel_size - 1`. Do NOT mix these two laws.

### Permitted TFLite Operations (20 exactly — ESPHome 2025.12.5)

Any op NOT in this list causes a fatal `kTfLiteError` at model load time. The wake word engine will never start.

```
AddCallOnce()          AddVarHandle()         AddReshape()
AddReadVariable()      AddStridedSlice()      AddConcatenation()
AddAssignVariable()    AddConv2D()            AddMul()
AddAdd()               AddMean()              AddFullyConnected()
AddLogistic()          AddQuantize()          AddDepthwiseConv2D()
AddAveragePool2D()     AddMaxPool2D()         AddPad()
AddPack()              AddSplitV()
```

**Used in okay_nabu reference model:** CALL_ONCE, VAR_HANDLE, READ_VARIABLE, ASSIGN_VARIABLE, CONCATENATION, STRIDED_SLICE, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SPLIT_V, FULLY_CONNECTED, LOGISTIC, QUANTIZE
**Available but unused in reference:** MUL, ADD, MEAN, AVERAGE_POOL_2D, MAX_POOL_2D, PAD, PACK
**Zero custom ops** — all ops show `CustomCode: N/A`.

> **Repository default note:** The default `residual_connection = [0, 1, 1, 1]` configuration adds `ADD` ops and produces **58 ops** in the main subgraph (vs the reference model's 55). This is ESPHome-compatible because `ADD` is registered.

### MixedNet Default Configuration

```python
first_conv_filters     = 32
first_conv_kernel_size = 5          # → stream shape [1, 2, 1, 40]
stride                 = 3          # GLOBAL IMMUTABLE CONSTANT
pointwise_filters      = [64, 64, 64, 64]
mixconv_kernel_sizes   = [[5], [7, 11], [9, 15], [23]]
repeat_in_block        = [1, 1, 1, 1]
residual_connection    = [0, 1, 1, 1]  # repository default; ADD op used
```

**Structural rules (all variants):**
- All temporal convolutions must be wrapped in `stream.Stream`
- `padding="valid"` on the time axis
- `use_bias=False` on all Conv2D/DepthwiseConv2D
- Exactly one `Dense(1, activation="sigmoid")` as the last layer
- BatchNormalization after every depthwise/pointwise conv block
- No LSTM, GRU, attention, recurrent layers, custom ops, or `tf.py_function`

### ESPHome Manifest Required Fields

| Field | Required Value |
|-------|---------------|
| `type` | `"micro"` |
| `version` | `2` |
| `micro.feature_step_size` | `10` (ms) — **NOT 20** |
| `micro.minimum_esphome_version` | `"2024.7.0"` |
| `micro.tensor_arena_size` | `0` for auto-resolve (recommended) |

---

## 17. v1 vs v2 Model Differences

This project targets **v2**. v2 models will NOT work on ESPHome firmware older than 2024.7.0.

| Property | v1 | v2 (this project) |
|----------|----|--------------------|
| `feature_step_size` | 20 ms | **10 ms** |
| JSON manifest `version` | 1 | **2** |
| `minimum_esphome_version` | older | **2024.7.0** |
| Inference period | 60 ms (stride 3 × 20ms) | **30 ms** (stride 3 × 10ms) |
| Temporal resolution | Lower (20ms frames) | **Higher (10ms frames)** |
| Model architecture | Same MixedNet | Same MixedNet |
| Fields location | Flat top-level JSON | Nested inside `"micro": {}` |
| `trained_languages` field | Not present | **Required** (new in v2) |

The key v2 improvement is halving the feature step size from 20ms to 10ms, doubling temporal resolution without changing the model architecture.

---

## 18. Violation Consequence Matrix

> This table describes what breaks **on real hardware** when architectural rules are violated. "Works in Python" is NOT a definition of correctness.

| What You Might Change | What Actually Breaks |
|-----------------------|----------------------|
| `mel_bins`, `window_step_ms`, `sample_rate_hz` | Input tensor shape mismatch; model receives wrong feature dimensions; garbage predictions |
| Output dtype `uint8` → `int8` | ESPHome reads signed bytes as unsigned; every prediction ≥ 128 is misinterpreted; wake word never triggers or always triggers |
| Input dtype `int8` → `float32` | `load_model_()` fails: "Streaming model tensor input is not int8"; model never loads |
| Calibration dataset < 500 samples or missing boundary anchors | Scale/zero_point shift; dynamic range wrong; predictions compressed into tiny range; effectively non-functional |
| Remove `_experimental_variable_quantization` | State payload tensors remain float32; TFLite Micro int8-only kernel resolver fails at load time; device halts |
| Use op outside the 20 registered ops | Op resolver returns `kTfLiteError` at load time; device halts; wake word engine never starts |
| Export without `STREAM_INTERNAL_STATE_INFERENCE` mode | No VAR_HANDLE/state ops in graph; model has no memory; per-frame prediction only; accuracy is random |
| Missing or corrupted Subgraph 1 | State variables not initialized at boot; undefined initial behavior from first inference |
| Wrong ring buffer size (kernel/stride mismatch) | Ring buffer reads from wrong temporal offset; scrambled temporal context; no crash, just permanently wrong predictions |
| Change `stride` in code but not in export | Input tensor slicing misaligned with ring buffer writes; state corruption accumulates; model degrades after first second |
| Add LSTM/GRU/attention or custom ops | Op not registered; device halts at model load |
| `inference_output_type = tf.int8` instead of `tf.uint8` | Output read as signed; probabilities inverted vs ESPHome's uint8 threshold |
| `feature_step_size ≠ 10` in v2 manifest | ESPHome feeds frames at wrong cadence; temporal context 1.5× or 2× too long/short |
| `version ≠ 2` in v2 manifest | v1 loader path taken; state variables may not be handled correctly |
| TensorFlow < 2.16 | `_experimental_variable_quantization` may not be available; `ExportArchive` API may differ; export fails or produces incompatible model |

---

## Complete Workflow Summary

```bash
# ═══════════════════════════════════════════════════════
# STEP 1: Prepare Dataset
# ═══════════════════════════════════════════════════════
mkdir -p dataset/{positive,negative,hard_negative,background,rirs}
# Add your audio files...

# ═══════════════════════════════════════════════════════
# STEP 2: Speaker Clustering (PyTorch env, optional)
# ═══════════════════════════════════════════════════════
mww-torch
mww-cluster-analyze --config max_quality --dataset all --n-clusters 200
# Review cluster_output/*_cluster_report.txt
mww-cluster-apply --namelist-dir cluster_output --dry-run
mww-cluster-apply --namelist-dir cluster_output

# ═══════════════════════════════════════════════════════
# STEP 3: Configure
# ═══════════════════════════════════════════════════════
# Create my_config.yaml with your wake word name, etc.

# ═══════════════════════════════════════════════════════
# STEP 4: Train (TF env)
# ═══════════════════════════════════════════════════════
mww-tf
mww-train --config max_quality --override my_config.yaml
# Monitor: tensorboard --logdir ./logs

# ═══════════════════════════════════════════════════════
# STEP 5: Evaluate
# ═══════════════════════════════════════════════════════
python scripts/evaluate_model.py \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config max_quality --split test --analyze

# ═══════════════════════════════════════════════════════
# STEP 6: Auto-Tune (if needed)
# ═══════════════════════════════════════════════════════
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config max_quality

# ═══════════════════════════════════════════════════════
# STEP 7: Export
# ═══════════════════════════════════════════════════════
mww-export --checkpoint checkpoints/best_weights.weights.h5 --model-name "hey_computer"

# ═══════════════════════════════════════════════════════
# STEP 8: Verify
# ═══════════════════════════════════════════════════════
python scripts/verify_esphome.py models/exported/hey_computer.tflite --verbose

# ═══════════════════════════════════════════════════════
# STEP 9: Deploy to ESPHome
# ═══════════════════════════════════════════════════════
cp models/exported/hey_computer.tflite /config/esphome/models/
cp models/exported/manifest.json /config/esphome/models/
```

---

## Resources

- [ESPHome micro_wake_word](https://esphome.io/components/micro_wake_word.html)
- [Original microWakeWord](https://github.com/kahrendt/microWakeWord)
- [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word) — Official training pipeline (reference implementation)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [ARCHITECTURAL_CONSTITUTION.md](./ARCHITECTURAL_CONSTITUTION.md) — **Authoritative source of architectural truth** (verified 2026-03-13 from TFLite flatbuffer binary + ESPHome C++ source)

---

*microwakeword_trainer v2.0.0 — GPU-Accelerated Wake Word Training Framework*
