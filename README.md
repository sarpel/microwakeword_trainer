# microwakeword_trainer

GPU-accelerated wake word training framework for ESPHome. Train custom "Hey Siri" or "OK Google" style wake words and deploy them to ESP32 devices.

## Overview

This framework provides a complete pipeline for training wake word detection models:

- **Feature extraction** with 40-bin mel spectrograms
- **MixedNet architecture** optimized for edge deployment
- **GPU-accelerated training** with CuPy SpecAugment
- **ESPHome-compatible export** with INT8 quantization
- **Streaming inference** support for real-time detection

## Requirements

- **Python**: 3.10, 3.11, or 3.12 (3.12 not yet supported by ai-edge-litert)
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

## Installation

### 1. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv --python 3.11 ~/venvs/mww-tf
source ~/venvs/mww-tf/bin/activate

# Or using venv
python3.11 -m venv ~/venvs/mww-tf
source ~/venvs/mww-tf/bin/activate
```

### 2. Install Dependencies

```bash
cd /home/sarpel/mww/microwakeword_trainer
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

## Quick Start

Train your first wake word model in three commands:

```bash
# 1. Prepare your dataset (see Dataset Preparation below)

# 2. Train with standard preset
mww-train --config config/presets/standard.yaml

# 3. Export to TFLite
mww-export --checkpoint checkpoints/best.ckpt --output models/exported/
```

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

For GPUs with limited VRAM, configure memory limits in your training script:

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

## Usage Examples

### Training

```bash
# Basic training with standard preset
mww-train --config config/presets/standard.yaml

# Training with custom config override
mww-train --config config/presets/standard.yaml --override my_settings.yaml

# Resume from checkpoint
mww-train --config config/presets/standard.yaml --resume checkpoints/last.ckpt

# Dry run (validate config without training)
mww-train --config config/presets/standard.yaml --dry-run
```

### Export

```bash
# Export trained model to TFLite
mww-export --checkpoint checkpoints/best.ckpt --output models/exported/

# Export with custom name
mww-export \
    --checkpoint checkpoints/best.ckpt \
    --output models/exported/ \
    --model-name "hey_computer"

# Export without quantization (for debugging)
mww-export \
    --checkpoint checkpoints/best.ckpt \
    --output models/exported/ \
    --no-quantize
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
  tensor_arena_size: 26080
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
  "name": "Hey Computer",
  "author": "Your Name",
  "version": "1.0.0",
  "model": "wake_word.tflite",
  "minimum_esphome_version": "2024.7.0",
  "tensor_arena_size": 26080,
  "probability_cutoff": 0.97,
  "sliding_window_size": 5
}
```

### ESPHome Configuration

```yaml
micro_wake_word:
  models:
    - model: wake_word.tflite
      probability_cutoff: 0.97
```

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
pip install cupy-cuda12x>=13.0
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

## Performance Tips

### Training Speed

| Optimization | Speedup | Trade-off |
|--------------|---------|-----------|
| Mixed precision (FP16) | 2-3x | Minimal accuracy loss |
| Larger batch size | 1.5x | More VRAM needed |
| CuPy SpecAugment | 5-10x | Requires GPU |
| PyArrow data loading | 2x | More RAM needed |

### Model Size

For smaller models (ESP32-S3 with limited RAM):

```yaml
model:
  first_conv_filters: 20
  pointwise_filters: "40,40,40,40"  # Smaller than default 60
  
export:
  tensor_arena_size: 20000  # Smaller arena
```

## Project Structure

```
/home/sarpel/mww/microwakeword_trainer/
├── config/
│   ├── presets/           # standard, max_quality, fast_test
│   └── loader.py          # Configuration loading
├── src/
│   ├── training/          # Training loop
│   ├── data/              # Dataset and augmentation
│   ├── model/             # MixedNet architecture
│   ├── export/            # TFLite conversion
│   └── utils/             # Performance utilities
├── scripts/
│   └── verify_esphome.py  # Compatibility checker
├── dataset/               # Audio data (user-provided)
└── models/                # Checkpoints and exports
```

## Contributing

This is a community training framework for ESPHome wake word detection. The export format matches the official ESPHome micro_wake_word component requirements.

## License

MIT License - See LICENSE file for details.

## Resources

- [ESPHome micro_wake_word](https://esphome.io/components/micro_wake_word.html)
- [Original microWakeWord](https://github.com/kahrendt/microWakeWord)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
