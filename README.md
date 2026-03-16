# microwakeword_trainer v2.1.0

GPU-accelerated wake word training framework for ESPHome. Train custom "Hey Siri" or "OK Google" style wake words and deploy them to ESP32 devices.

**Status**: ✅ Production Ready - All features implemented

## Overview

This framework provides a complete pipeline for training wake word detection models:

- **Feature extraction** with 40-bin mel spectrograms
- **MixedNet architecture** optimized for edge deployment
- **GPU-accelerated training** with CuPy SpecAugment
- **ESPHome-compatible export** with INT8 quantization
- **Adaptive thresholding** for production-ready evaluation metrics
- **Streaming inference** support for real-time detection

## Project Status

✅ **Production Ready** - All features implemented and tested

| Component | Status | Details |
|-----------|--------|---------|
| Configuration System | ✅ Complete | Config dataclasses for all pipeline sections, 3 presets, env var substitution |
| Training Pipeline | ✅ Complete | GPU SpecAugment, mixed precision, 2-phase training |
| Model Architecture | ✅ Complete | MixedNet with MixConv blocks |
| Streaming Export | ✅ Complete | TFLite with INT8 quantization |
| Speaker Clustering | ✅ Complete | ECAPA-TDNN embeddings |
| Hard Negative Mining | ✅ Complete | During & post-training mining |
| Auto-Tuning | ✅ Complete | FAH/recall optimization |
| Test Suite | ✅ Complete | Unit and integration tests |

**Total**: Comprehensive Python codebase with modular architecture

### Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - MixedNet architecture details
- [Configuration Reference](docs/CONFIGURATION.md) - Complete config reference
- [Training Guide](docs/TRAINING.md) - Training workflow and optimization
- [Export Guide](docs/EXPORT.md) - TFLite export and ESPHome deployment
- [Implementation Status](specs/implementation_status.md) - Detailed component status

## Requirements

- **Python**: 3.10 or 3.11
- **GPU**: CUDA-capable NVIDIA GPU (training requires GPU)
- **CUDA**: Version 12.x (required for CuPy compatibility)
- **RAM**: 16GB+ recommended for standard training
- **Storage**: 10GB+ for datasets and checkpoints

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg

# Verify CUDA installation
nvidia-smi
```

## Environment Setup (CRITICAL)

**⚠️ This project requires TWO separate virtual environments.** TensorFlow and PyTorch cannot coexist in the same environment without conflicts.

### Environment 1: TensorFlow (Main Training)

Used for: Training, export, inference, data processing

```bash
# Adjust paths to match your installation directory
# Example: Replace $PROJECT_DIR and ~/.venvs/mww-tf with your actual paths

# Create environment
python3.11 -m venv ~/.venvs/mww-tf
source ~/.venvs/mww-tf/bin/activate

# Install TensorFlow dependencies
cd $PROJECT_DIR
pip install -r requirements.txt
```

### Environment 2: PyTorch (Speaker Clustering)

Used for: Speaker clustering, ECAPA-TDNN embeddings, hard negative mining (optional)

```bash
# Adjust paths to match your installation directory
# Example: Replace $PROJECT_DIR and ~/.venvs/mww-torch with your actual paths

# Create environment
python3.11 -m venv ~/.venvs/mww-torch
source ~/.venvs/mww-torch/bin/activate

# Install PyTorch dependencies
cd $PROJECT_DIR
pip install -r requirements-torch.txt
```

### Quick Environment Switching

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Adjust paths to match your installation directory
# Example: Replace $PROJECT_DIR with your actual project path

# TensorFlow environment (default for training)
alias mww-tf='source ~/.venvs/mww-tf/bin/activate && cd $PROJECT_DIR'

# PyTorch environment (for clustering)
alias mww-torch='source ~/.venvs/mww-torch/bin/activate && cd $PROJECT_DIR'
```

Then use:
```bash
mww-tf        # Switch to TF env
mww-torch     # Switch to PyTorch env
```

### Verify Installation

**In TensorFlow environment:**
```bash
# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

**In PyTorch environment:**
```bash
# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check speechbrain
python -c "import speechbrain; print(speechbrain.__version__)"
```

---

## User Guide

### Complete Workflow

Follow these steps to train and deploy a custom wake word:

#### Step 1: Prepare Your Dataset

Create the dataset directory structure:

```bash
mkdir -p dataset/{positive,negative,hard_negative,background,rirs}
```

Organize your audio files:

```
dataset/
├── positive/           # Your wake word recordings
│   ├── speaker_001/    # Organize by speaker
│   │   ├── rec_001.wav
│   │   ├── rec_002.wav
│   │   └── ...
│   ├── speaker_002/
│   └── ...
├── negative/           # Background speech (not wake word)
│   └── speech/
│       ├── conv_001.wav
│       └── ...
├── hard_negative/      # Sounds similar to wake word
│   ├── false_positive_001.wav
│   └── ...
├── background/         # Ambient noise
│   ├── noise_001.wav
│   └── ...
└── rirs/              # Room impulse responses (optional)
    └── reverb_001.wav
```

**Audio Requirements:**
- Format: WAV, 16-bit PCM
- Sample rate: 16kHz (will be resampled if needed)
- Length: 1-3 seconds per clip
- Channels: Mono

**Recording Tips:**
- Record at least 100 wake word samples (1000+ recommended)
- Use 5+ different speakers for diversity
- Record at various distances (1-3 meters)
- Record in different rooms/environments
- Include variations in tone and speed

#### Step 2: Run Speaker Clustering (Optional but Recommended)

To prevent train/test data leakage from the same speaker, use ML-based speaker clustering. This analyzes your audio samples and groups them by speaker voice characteristics.

**Prerequisites:**
- Requires PyTorch environment (`mww-torch`)
- Hugging Face account (free) with accepted model terms
- Run: `huggingface-cli login` after creating account

**Analyze clusters (dry-run):**
```bash
mww-torch

# Cluster positive dataset (default)
mww-cluster-analyze --config standard

# Cluster all datasets at once
mww-cluster-analyze --config standard --dataset all

# Cluster specific dataset
mww-cluster-analyze --config standard --dataset negative

# If you know your speaker count (~200 speakers), use --n-clusters
# (recommended for short wake word clips where threshold-based clustering over-fragments)
mww-cluster-analyze --config standard --n-clusters 200

# Combine options
mww-cluster-analyze --config standard --dataset all --n-clusters 200 --threshold 0.65

# Generates per dataset:
#   - cluster_output/{dataset}_namelist.json (file → speaker mapping)
#   - cluster_output/{dataset}_cluster_report.txt (human-readable report)
```

Review the report. Check if speakers are grouped correctly. Files stay in place—this is read-only.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | **required** | Config preset name or path to YAML file |
| `--override` | string | None | Override config file (optional) |
| `--dataset` | string | `positive` | Which dataset(s) to cluster: `positive`, `negative`, `hard_negative`, `all` |
| `--n-clusters` | int | None | Explicit cluster count (overrides threshold). Use when you know approximate speaker count. |
| `--threshold` | float | from config | Override similarity threshold |
| `--output-dir` | string | `./cluster_output` | Directory for output files |
| `--max-files` | int | None | Limit number of files to process (for testing) |

**Organize files by speaker (after reviewing clusters):**
```bash
# Organize a single dataset
mww-cluster-apply --namelist cluster_output/positive_namelist.json

# Organize all datasets at once
mww-cluster-apply --namelist-dir cluster_output

# Preview first (recommended)
mww-cluster-apply --namelist cluster_output/positive_namelist.json --dry-run

# Undo if something looks wrong
mww-cluster-apply --undo cluster_output/positive_backup_manifest.json
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--namelist` | string | None | Path to a single namelist JSON from mww-cluster-analyze |
| `--namelist-dir` | string | None | Directory containing `*_namelist.json` files (processes all) |
| `--undo` | string | None | Path to backup manifest JSON to reverse a previous organization |
| `--output-dir` | string | `./cluster_output` | Directory for backup manifests |
| `--dry-run` | flag | off | Preview changes without moving files |

> **Note:** `--namelist`, `--namelist-dir`, and `--undo` are mutually exclusive (pick one).
> A backup manifest is saved automatically before any files are moved.

**Skip this step if:**
- You already organized files by speaker into subdirectories
- You don't have multiple speakers (single-user wake word)
- You prefer to use directory-based speaker detection

#### Step 3: Configure Your Training

Choose a preset configuration:

| Preset | Use Case | Training Time | Accuracy |
|--------|----------|---------------|----------|
| `fast_test.yaml` | Quick iteration | ~1 hour | Basic |
| `standard.yaml` | Production | ~8 hours | Good |
| `max_quality.yaml` | Best accuracy | ~24 hours | Excellent |

Create a custom configuration override:

```yaml
# my_config.yaml
# This overrides the standard preset

export:
  wake_word: "Hey Computer"    # Your wake word name
  author: "Your Name"
  website: "https://github.com/yourusername"

training:
  batch_size: 64                 # Reduce if OOM errors

model:
  first_conv_filters: 30         # Model size (20-30 for smaller models)
```

#### Step 4: Run Training

```bash
# Switch to TensorFlow environment
mww-tf

# Train with standard preset
mww-train --config config/presets/standard.yaml

# Or with custom override
mww-train --config config/presets/standard.yaml --override my_config.yaml

# Resume from checkpoint (if interrupted)
# (No --resume flag — restart training from scratch or use latest checkpoint)
# Note: --resume and --dry-run are not supported flags in mww-train
```

**During Training:**
- Checkpoints saved to `./checkpoints/`
- Logs saved to `./logs/`
- Profiles saved to `./profiles/` (if enabled)
- Monitor with TensorBoard: `tensorboard --logdir ./logs`

#### Step 5: Export to TFLite

```bash
# Export the best checkpoint
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/

# Export with custom name
mww-export \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --output models/exported/ \
    --model-name "hey_computer"

# Export with explicit config override (example)
mww-export \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --output models/exported/ \
    --config config/presets/standard.yaml
```

**Generated Files:**
```
models/exported/
├── hey_computer.tflite      # The model file
├── manifest.json            # ESPHome manifest
└── streaming/               # Streaming SavedModel (for debugging)
```

#### Step 5b: Auto-Tune (Optional — Improve FAH/Recall)

If your model's FAH > 0.5 or recall < 0.90, use auto-tuning to improve metrics without full retraining.

**What it does:** The `mww-autotune` tool adjusts probability thresholds and hyperparameters through iterative fine-tuning. It does **not** perform full model retraining — instead, it optimizes your existing checkpoint to achieve target FAH and recall metrics.

**Duration:** Typically 5–10 minutes depending on dataset size.

**Output:** Writes a new tuned checkpoint (does NOT overwrite the original) and saves results to the specified output directory.

```bash
# Auto-tune with default targets (FAH < 0.3, recall > 0.92)
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard

# Custom targets
mww-autotune \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config standard \
    --target-fah 0.2 \
    --target-recall 0.95


mww-autotune \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config standard \
    --output-dir checkpoints/tuned/

# With user-defined hard negatives directory
mww-autotune \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config standard \
    --users-hard-negs /path/to/custom_hard_negatives/
```

The tool will:
- Iterate through hyperparameters to find optimal thresholds
- Use hard negative mining to improve model discrimination
- Use user-provided hard negatives for better specificity (if specified)
- Save the tuned checkpoint to your output directory
- Log metrics and final configuration
#### Step 6: Verify ESPHome Compatibility

```bash
# Verify the exported model
python scripts/verify_esphome.py models/exported/hey_computer.tflite

# Verbose output
python scripts/verify_esphome.py models/exported/hey_computer.tflite --verbose

# JSON output for CI/CD
python scripts/verify_esphome.py models/exported/hey_computer.tflite --json
```

Expected output:
```
✓ Subgraphs: 2 (correct)
✓ Input shape: [1, 3, 40] (correct)
✓ Input dtype: int8 (correct)
✓ Output shape: [1, 1] (correct)
✓ Output dtype: uint8 (correct)
✓ Quantization: enabled (correct)
✓ ESPHome compatible: YES
```

#### Step 7: Deploy to ESPHome

Copy the model and manifest to your ESPHome configuration:

```bash
mkdir -p /config/esphome/models
cp models/exported/hey_computer.tflite /config/esphome/models/
cp models/exported/manifest.json /config/esphome/models/
```

Add to your ESPHome YAML:

```yaml
micro_wake_word:
  models:
    - model: models/hey_computer.tflite
      probability_cutoff: 0.97

# Optional: Use the wake word
voice_assistant:
  wake_word: "Hey Computer"
```

---

## Quick Start (TL;DR)

Train your first wake word model:

```bash
# 1. Prepare dataset in dataset/positive/, dataset/negative/, etc.

# 2. (Optional but recommended) Run speaker clustering to prevent data leakage
mww-torch
mww-cluster-analyze --config standard --dataset all --n-clusters 200
# Review cluster_output/*_cluster_report.txt
mww-cluster-apply --namelist-dir cluster_output --dry-run  # Preview first
mww-cluster-apply --namelist-dir cluster_output            # Execute

# 3. Switch to TF environment
mww-tf

# 4. Train
# Note: You can use either the preset name (e.g., "standard") or full path (e.g., "config/presets/standard.yaml")
mww-train --config config/presets/standard.yaml

# 5. (Optional) Auto-tune if FAH > 0.5 or recall < 0.90
# Note: You can use either the preset name (e.g., "standard") or full path (e.g., "config/presets/standard.yaml")
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config config/presets/standard.yaml

# 6. Export
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/

# 7. Verify
python scripts/verify_esphome.py models/exported/wake_word.tflite
```

---

## GPU Setup

This framework requires a GPU for training. The CuPy-based SpecAugment has no CPU fallback.

### Environment Variables

Set these before training for optimal GPU performance:

```bash
# Limit GPU visibility (if you have multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Allow GPU memory to grow instead of allocating all at once
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Use cuda_malloc_async allocator (recommended)
export TF_GPU_ALLOCATOR=cuda_malloc_async

# For reproducibility
export TF_DETERMINISTIC_OPS=1
```

### Memory Configuration

For GPUs with limited VRAM:

```python
from src.utils.performance import configure_tensorflow_gpu, configure_mixed_precision

# Enable memory growth (prevents OOM errors)
configure_tensorflow_gpu(memory_growth=True)

# Optional: limit to 8GB
configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=8192)

# Enable mixed precision for 2-3x speedup
configure_mixed_precision(enabled=True)
```

### Threading Configuration

For CPU-bound operations (data loading, augmentation):

```python
from src.utils.performance import set_threading_config

# Use 16 threads for data loading
set_threading_config(inter_op_parallelism=16, intra_op_parallelism=16)
```

---

## Usage Examples

### Training

```bash
# Basic training with standard preset
# Note: You can use either the preset name (e.g., "standard") or full path (e.g., "config/presets/standard.yaml")
mww-train --config standard
# Or: mww-train --config config/presets/standard.yaml

# Training with custom config override
mww-train --config standard --override my_settings.yaml

# Resume from checkpoint
# (No --resume flag)

# (No --dry-run flag in mww-train)
```

### Export

```bash
# Export trained model to TFLite
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/

# Export with custom name
mww-export \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --output models/exported/ \
    --model-name "hey_computer"

# Export with explicit preset path
mww-export \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --output models/exported/ \
    --config config/presets/standard.yaml
```

### Verification

```bash
# Verify TFLite model is ESPHome compatible
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Verbose output
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose

# JSON output for CI/CD
python scripts/verify_esphome.py models/exported/wake_word.tflite --json
```

### Utility Scripts

```bash
# Analyze audio files
python scripts/audio_analyzer.py <audio_file>

# Detect duplicate/similar audio files
python scripts/audio_similarity_detector.py <directory>

# Count dataset samples
python scripts/count_dataset.py <dataset_directory>

# Score audio quality (fast)
python scripts/score_quality_fast.py <audio_file>

# Score audio quality (full)
python scripts/score_quality_full.py <audio_file>

# Split audio files
python scripts/split_audio.py <audio_file>

# VAD-based audio trimming
python scripts/vad_trim.py <audio_file>

# Comprehensive model evaluation (JSON + images + executive reports)
python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard --output-dir logs/

# Build interactive dashboard from evaluation_report.json
python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json
```

---

## Dataset Preparation

### Directory Structure

```
dataset/
├── positive/           # Wake word samples (e.g., "hey_computer")
│   ├── speaker_001/
│   ├── speaker_002/
│   └── ...
├── negative/           # Background speech
│   └── speech/
├── hard_negative/      # False positives to avoid
├── background/         # Noise/ambient sounds
└── rirs/              # Room impulse responses for reverb
```

### Audio Requirements

- **Format**: WAV, 16-bit PCM
- **Sample rate**: 16kHz (will be resampled if needed)
- **Length**: 1-3 seconds per clip
- **Channels**: Mono

### Minimum Dataset Sizes

| Dataset Type | Minimum | Recommended |
|--------------|---------|-------------|
| Positive     | 100     | 1000+       |
| Negative     | 1000    | 10000+      |
| Hard Negative| 50      | 500+        |

### Speaker Diversity

For best results, record wake word samples from:

- Multiple speakers (5+ different voices)
- Various distances from microphone (1-3 meters)
- Different rooms/environments
- Various times of day (morning/evening voice)

---

## Configuration

### Presets

Three built-in presets cover most use cases:

| Preset | Use Case | Training Time | Accuracy |
|--------|----------|---------------|----------|
| `fast_test.yaml` | Quick iteration | ~1 hour | Basic |
| `standard.yaml` | Production | ~8 hours | Good |
| `max_quality.yaml` | Best accuracy | ~24 hours | Excellent |

### Configuration File Structure

```yaml
# hardware.yaml
hardware:
  sample_rate_hz: 16000
  mel_bins: 40
  window_size_ms: 30
  window_step_ms: 10
  clip_duration_ms: 1000

training:
  training_steps: [20000, 10000]
  learning_rates: [0.001, 0.0001]
  batch_size: 128

model:
  architecture: "mixednet"
  first_conv_filters: 30
  first_conv_kernel_size: 5
  stride: 3
  pointwise_filters: "60,60,60,60"
  mixconv_kernel_sizes: "[5],[9],[13],[21]"

export:
  wake_word: "Hey Computer"
  quantize: true
  inference_input_type: "int8"
  inference_output_type: "uint8"
  tensor_arena_size: 0  # auto-calculate from exported model
```

### Custom Configuration

Create a custom YAML that overrides preset values:

```yaml
# my_config.yaml
# Loads standard preset, then applies these overrides
training:
  batch_size: 64  # Smaller batch for limited VRAM

model:
  first_conv_filters: 20  # Smaller model

export:
  wake_word: "Hey Jarvis"
  author: "Your Name"
```

Load it with:

```python
from config.loader import load_full_config

config = load_full_config("standard", "my_config.yaml")
```

---

## ESPHome Integration

### Generated Files

After export, you will have:

```
models/exported/
├── wake_word.tflite       # The model file
├── manifest.json          # ESPHome manifest
└── streaming/             # Streaming SavedModel (for debugging)
```

### Manifest Format

```json
{
  "type": "micro",
  "wake_word": "Hey Computer",
  "author": "Your Name",
  "website": "https://github.com/sarpel/microwakeword-trainer",
  "model": "wake_word.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 0,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

`tensor_arena_size` is auto-resolved during export when `export.tensor_arena_size: 0`.

### ESPHome Configuration

```yaml
micro_wake_word:
  models:
    - model: wake_word.tflite
      probability_cutoff: 0.97
```

---

## Architecture

### MixedNet Model

The default architecture uses:

- **Initial Conv2D**: 30 filters, kernel size 5, stride 3
- **4 MixConv blocks**: Parallel depthwise convolutions with different kernel sizes
- **Classification head**: Single dense layer with sigmoid activation

### Input/Output Spec

| Tensor | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| Input | [1, 3, 40] | int8 | 3 frames of 40 mel bins |
| Output | [1, 1] | uint8 | Wake word probability |

### Streaming Inference

The exported TFLite model uses:

- **2 subgraphs**: Main inference + initialization
- **6 state variables**: Ring buffers for streaming
- **Internal state**: No external state management needed

---

## Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size
training:
  batch_size: 32  # Default is 128

# Or limit GPU memory
configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=4096)
```

### CuPy Not Found

```bash
# Verify CUDA version
nvcc --version  # Should show 12.x

# Reinstall cupy for your CUDA version
pip uninstall cupy-cuda12x
pip install cupy-cuda12x>=14.0
```

### Model Not Detecting Wake Word

1. Check dataset balance (positive:negative ratio should be 1:10+)
2. Verify audio quality (no clipping, good SNR)
3. Increase training steps
4. Add more speaker diversity

### ESPHome Compatibility Errors

```bash
# Run verification script
python scripts/verify_esphome.py model.tflite --verbose

# Common issues:
# - Wrong input/output dtype (must be int8/uint8)
# - Wrong number of subgraphs (must be 2)
# - Missing quantization
```

---

## Performance Tips

### Training Speed

| Optimization | Speedup | Trade-off |
|--------------|---------|-----------|
| Mixed precision (FP16) | 2-3x | Minimal accuracy loss |
| Larger batch size | 1.5x | More VRAM needed |
| CuPy SpecAugment | 5-10x | Requires GPU |

### Model Size

For smaller models (ESP32-S3 with limited RAM):

```yaml
model:
  first_conv_filters: 20
  pointwise_filters: "40,40,40,40"  # Smaller than default 64

export:
  tensor_arena_size: 0  # keep auto-calculated unless you measured a custom value
```

---

## Project Structure

```
./
├── config/
│   ├── presets/           # standard, max_quality, fast_test
│   └── loader.py          # Configuration loading (736 lines)
├── src/
│   ├── training/          # Training loop, augmentation, mining, profiling
│   ├── tuning/            # Auto-tuning (mww-autotune)
│   ├── data/              # Dataset, ingestion, features, augmentation, clustering, preprocessing, quality
│   ├── model/             # MixedNet architecture + streaming layers
│   ├── export/            # TFLite export, analysis, verification, manifests
│   ├── evaluation/        # Metrics, FAH estimation, calibration, test evaluation
│   ├── utils/             # GPU config, performance, terminal logger, optional deps
│   ├── tools/             # CLI entry points (cluster-analyze, cluster-apply)
│   └── config/            # Config package init
├── scripts/
│   ├── verify_esphome.py          # ESPHome compatibility checker
│   ├── generate_test_dataset.py   # Synthetic dataset generator
│   ├── evaluate_model.py          # Post-training model evaluation
│   ├── audio_analyzer.py          # Audio file analysis
│   ├── audio_similarity_detector.py  # Duplicate/similar audio detection
│   ├── count_dataset.py           # Dataset sample counter
│   ├── score_quality_fast.py      # Fast audio quality scoring
│   ├── score_quality_full.py      # Full audio quality scoring
│   ├── split_audio.py             # Audio splitting utility
│   └── vad_trim.py                # VAD-based audio trimming
├── dataset/               # Audio data (user-provided)
├── requirements.txt       # TensorFlow environment
├── requirements-torch.txt # PyTorch environment
├── models/                # Checkpoints and exports
├── docs/                  # Documentation
└── ARCHITECTURAL_CONSTITUTION.md  # Immutable source truth
```

---

## Contributing

This is a community training framework for ESPHome wake word detection. The export format matches the official ESPHome micro_wake_word component requirements.

---

## License

MIT License - See LICENSE file for details.

---

## Resources

- [ESPHome micro_wake_word](https://esphome.io/components/micro_wake_word.html)
- [Original microWakeWord](https://github.com/kahrendt/microWakeWord)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
