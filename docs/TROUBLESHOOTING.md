# Troubleshooting Guide — microwakeword_trainer v2.0

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Environment & Installation Issues](#environment--installation-issues)
3. [GPU & CUDA Issues](#gpu--cuda-issues)
4. [Dataset & Data Issues](#dataset--data-issues)
5. [Training Issues](#training-issues)
6. [Export & ESPHome Compatibility Issues](#export--esphome-compatibility-issues)
7. [Configuration Issues](#configuration-issues)
8. [Performance & Optimization](#performance--optimization)
9. [Speaker Clustering Issues](#speaker-clustering-issues)
10. [Diagnostic Tools & Commands](#diagnostic-tools--commands)

---

## Quick Diagnostic Checklist

Run these commands in order when something goes wrong:

```bash
# 1. Verify virtual environment is correct
which python
# Should show: /home/YOURNAME/.venvs/mww-tf/bin/python

# 2. Check GPU availability
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# 3. Check CuPy (required for SpecAugment)
python -c "import cupy; print('CuPy version:', cupy.__version__); print('CUDA version:', cupy.cuda.runtime.getDeviceCount(), 'devices')"

# 4. Verify dataset structure
python scripts/count_dataset.py dataset/positive
python scripts/count_dataset.py dataset/negative

# 5. Check config validity
python -c "from config.loader import load_full_config; c = load_full_config('standard'); print('Config loaded successfully')"

# 6. Check disk space
df -h .

# 7. Check memory usage
free -h
```

---

## Environment & Installation Issues

### Issue: TensorFlow and PyTorch Conflict

**Symptoms:**
- Import errors with TensorFlow or PyTorch
- `Illegal instruction (core dumped)`
- `ModuleNotFoundError` for packages that are installed

**Root Cause:**
TensorFlow and PyTorch have incompatible CUDA dependencies. They **cannot** coexist in the same virtual environment.

**Solution:**

```bash
# Step 1: Completely remove existing environments
rm -rf ~/.venvs/mww-tf
rm -rf ~/.venvs/mww-torch

# Step 2: Create TensorFlow environment
python3.11 -m venv ~/.venvs/mww-tf
source ~/.venvs/mww-tf/bin/activate
pip install -r requirements.txt

# Step 3: Create separate PyTorch environment
python3.11 -m venv ~/.venvs/mww-torch
source ~/.venvs/mww-torch/bin/activate
pip install -r requirements-torch.txt
```

**Prevention:**
- Always use `mww-tf` alias for training/export
- Always use `mww-torch` alias for clustering
- Never install both requirements files in the same environment

---

### Issue: Python Version Mismatch

**Symptoms:**
- `SyntaxError` on modern Python features
- TensorFlow/PyTorch installation fails
- `ModuleNotFoundError` for standard library modules

**Root Cause:**
This project requires Python 3.10 or 3.11.

**Solution:**

```bash
# Check Python version
python --version  # Should be 3.10.x or 3.11.x

# If wrong version, use specific Python version
python3.11 -m venv ~/.venvs/mww-tf
source ~/.venvs/mww-tf/bin/activate
```

**Verification:**
```bash
python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'; print('✓ Python version OK')"
```

---

### Issue: Missing System Dependencies

**Symptoms:**
- `ImportError: libsndfile.so.1: cannot open shared object file`
- Audio file loading fails with "Unknown format"
- `audiomentations` import errors

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg

# Verify installation
python -c "import soundfile; print('✓ libsndfile OK')"
```

---

## GPU & CUDA Issues

### Issue: CUDA Out of Memory (OOM)

**Symptoms:**
- `CUDA_ERROR_OUT_OF_MEMORY`
- `ResourceExhaustedError: OOM when allocating tensor`
- Training crashes at first batch

**Diagnostic Commands:**

```bash
# Check GPU memory usage
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi

# Check available memory
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    details = tf.config.experimental.get_memory_info(gpu)
    print(f'GPU memory: {details}')
"
```

**Solutions (try in order):**

1. **Reduce batch size** (most common fix):
   ```yaml
   # my_config.yaml
   training:
     batch_size: 32  # Default is 128
   ```

2. **Enable memory growth** (prevents TF from allocating all GPU memory):
   ```yaml
   # my_config.yaml
   performance:
     memory_growth: true
   ```

3. **Limit GPU memory**:
   ```python
   # In your config or script
   from src.utils.performance import configure_tensorflow_gpu
   configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=4096)
   ```

4. **Use smaller model**:
   ```yaml
   # my_config.yaml
   model:
     first_conv_filters: 20  # Default is 30
     pointwise_filters: "40,40,40,40"  # Default is "60,60,60,60"
   ```

5. **Use gradient checkpointing** (trade memory for speed):
   ```yaml
   training:
     gradient_checkpointing: true
   ```

---

### Issue: CuPy Not Found / GPU Not Available

**Symptoms:**
- `RuntimeError: CuPy is not available. Install cupy package`
- `RuntimeError: GPU is not available. This module requires GPU acceleration.`
- Training runs extremely slowly (CPU fallback would be slow, but there's no CPU fallback)

**Root Cause:**
CuPy requires specific CUDA version matching. Training **requires** GPU.

**Diagnostic Commands:**

```bash
# Check CUDA version
nvcc --version

# Check if CuPy can see GPU
python -c "
import cupy
try:
    print('CuPy version:', cupy.__version__)
    print('CUDA version:', cupy.cuda.runtime.getDeviceCount(), 'devices')
    print('Device 0:', cupy.cuda.runtime.getDeviceProperties(0)['name'])
except Exception as e:
    print('Error:', e)
"
```

**Solutions:**

1. **Verify CUDA 12.x is installed:**
   ```bash
   nvcc --version  # Should show "release 12.x"
   ```

2. **Reinstall CuPy for correct CUDA version:**
   ```bash
   pip uninstall cupy-cuda12x
   pip install cupy-cuda12x==13.6.0
   ```

3. **Verify NVIDIA drivers:**
   ```bash
   nvidia-smi
   # Should show driver version and GPU details
   ```

4. **If using WSL2:**
   - Install NVIDIA drivers on **Windows host**, not in WSL
   - WSL uses Windows drivers via WSLg
   - Run: `nvidia-smi.exe` from WSL to verify

---

### Issue: TensorFlow Cannot See GPU

**Symptoms:**
- `[]` when running `tf.config.list_physical_devices('GPU')`
- Training uses CPU (very slow)

**Diagnostic Commands:**

```bash
# Full GPU diagnostic
python -c "
import tensorflow as tf
print('TF version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPUs available:', tf.config.list_physical_devices('GPU'))
print('GPU devices:', tf.config.experimental.list_physical_devices('GPU'))
"
```

**Solutions:**

1. **Ensure CUDA 12.x compatibility:**
   TensorFlow 2.16 requires CUDA 12.x. Check:
   ```bash
   nvcc --version  # Should be 12.x
   ```

2. **Install TensorFlow with CUDA wheels:**
   ```bash
   pip install tensorflow[and-cuda]==2.16.2
   ```

3. **Set environment variables:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

4. **Check for conflicting CUDA installations:**
   ```bash
   # Look for multiple CUDA versions
   ls -la /usr/local/ | grep cuda
   
   # Ensure correct version is in PATH
   export PATH=/usr/local/cuda-12.0/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
   ```

---

## Dataset & Data Issues

### Issue: Dataset Not Found

**Symptoms:**
- `FileNotFoundError: [Errno 2] No such file or directory: 'dataset/positive'`
- `ValueError: No audio files found in dataset/positive`
- `Dataset is empty` errors during training

**Solution:**

```bash
# 1. Verify directory structure
ls -la dataset/
# Should show: positive/  negative/  hard_negative/  background/  rirs/

# 2. Create structure if missing
mkdir -p dataset/{positive,negative,hard_negative,background,rirs}

# 3. Check for audio files
find dataset/positive -name "*.wav" -o -name "*.mp3" -o -name "*.flac" | head -10

# 4. Count dataset samples
python scripts/count_dataset.py dataset/positive
python scripts/count_dataset.py dataset/negative
```

---

### Issue: Audio Format Problems

**Symptoms:**
- `RuntimeError: Error loading audio file`
- `ValueError: Audio file is empty or corrupt`
- Silent failures during feature extraction

**Diagnostic Commands:**

```bash
# Check audio file properties
python scripts/audio_analyzer.py dataset/positive/speaker_001/sample.wav

# Check for corrupt files
python -c "
import soundfile as sf
import os

def check_audio(filepath):
    try:
        data, sr = sf.read(filepath)
        print(f'✓ {filepath}: {len(data)} samples @ {sr}Hz')
        return True
    except Exception as e:
        print(f'✗ {filepath}: {e}')
        return False

# Check all files
for root, dirs, files in os.walk('dataset/positive'):
    for f in files:
        if f.endswith('.wav'):
            check_audio(os.path.join(root, f))
"
```

**Solutions:**

1. **Convert to correct format:**
   ```bash
   # Convert all files to 16kHz mono WAV
   for f in dataset/positive/**/*.wav; do
     ffmpeg -i "$f" -ar 16000 -ac 1 -acodec pcm_s16le "${f%.wav}_16k.wav"
   done
   ```

2. **Remove corrupt files:**
   ```bash
   # Find and list corrupt files
   find dataset/ -name "*.wav" -exec sh -c 'python -c "import soundfile; soundfile.read(\"{}\")" 2>/dev/null || echo "CORRUPT: {}"' \;
   ```

3. **Verify after preprocessing:**
   ```bash
   python scripts/vad_trim.py dataset/positive/speaker_001/sample.wav --output trimmed.wav
   ```

---

### Issue: Low Negative:Positive Ratio

**Symptoms:**
- Warning: `Low negative:positive ratio (2.5:1) in train — recommend at least 5:1`
- High false positive rate (low FAH)
- Model overfits to wake word

**Root Cause:**
Not enough negative samples compared to positive samples.

**Solution:**

```bash
# Check current ratio
python scripts/count_dataset.py dataset/positive
python scripts/count_dataset.py dataset/negative

# Calculate: negative_count / positive_count should be >= 5
```

**Add more negative samples:**

1. **Record background speech:**
   - Record conversations, podcasts, TV audio
   - Must NOT contain wake word
   - Target: 10x more negative than positive

2. **Generate synthetic negatives:**
   ```bash
   python scripts/generate_test_dataset.py --output dataset/synthetic_negative/
   ```

3. **Use audio augmentation to multiply samples:**
   ```yaml
   # config.yaml
   augmentation:
     pitch_shift: true
     time_stretch: true
     volume: true
   ```

---

### Issue: Data Leakage (Same Speaker in Train/Test)

**Symptoms:**
- Training accuracy much higher than validation accuracy
- Model performs well on validation but poorly on real-world audio
- Overfitting to specific speakers

**Diagnostic Commands:**

```bash
# Run speaker clustering analysis
source ~/.venvs/mww-torch/bin/activate
mww-cluster-analyze --config standard --dataset all

# Review the report
cat cluster_output/positive_cluster_report.txt

# Check for duplicate speakers across splits
python -c "
import json

# Load speaker mappings
with open('cluster_output/positive_namelist.json') as f:
    positive = json.load(f)

# Check if same speaker appears in multiple dataset splits
print('Unique speakers:', len(set(positive.values())))
print('Total files:', len(positive))
"
```

**Solution:**

1. **Organize by speaker before training:**
   ```bash
   mww-cluster-apply --namelist-dir cluster_output --dry-run  # Preview first
   mww-cluster-apply --namelist-dir cluster_output            # Execute
   ```

2. **Verify splits are speaker-independent:**
   ```bash
   # Check that no speaker appears in both train and test
   python -c "
   import json
   
   with open('cluster_output/positive_namelist.json') as f:
       speakers = json.load(f)
   
   # TODO: Compare with actual train/test splits
   print('Files organized by speaker')
   "
   ```

---

## Training Issues

### Issue: Training Not Starting / Hangs

**Symptoms:**
- Stuck at "Building model..."
- No GPU utilization shown in `nvidia-smi`
- Process appears frozen

**Diagnostic Commands:**

```bash
# Check if process is running
ps aux | grep mww-train

# Check system resources
top
free -h

# Check for deadlock (in another terminal)
py-spy dump --pid $(pgrep -f mww-train)

# Check logs
tail -f logs/terminal_*.log
```

**Solutions:**

1. **Reduce data loading workers:**
   ```yaml
   # config.yaml
   training:
     num_workers: 2  # Default might be too high
   ```

2. **Disable multiprocessing:**
   ```yaml
   training:
     use_multiprocessing: false
   ```

3. **Check disk I/O:**
   ```bash
   # If using HDD instead of SSD, I/O might be bottleneck
   iostat -x 1
   ```

4. **Reduce dataset size for testing:**
   ```bash
   # Use fast_test preset for quick validation
   mww-train --config config/presets/fast_test.yaml
   ```

---

### Issue: Loss is NaN

**Symptoms:**
- `loss: nan` in training logs
- Model weights become NaN
- Training diverges immediately

**Root Causes:**
- Learning rate too high
- Numerical instability in loss function
- Bad data samples (inf/nan values)
- Mixed precision overflow

**Solutions:**

1. **Reduce learning rate:**
   ```yaml
   # config.yaml
   training:
     learning_rates: [0.0001, 0.00001]  # Instead of [0.001, 0.0001]
   ```

2. **Disable mixed precision temporarily:**
   ```yaml
   performance:
     mixed_precision: false
   ```

3. **Check for bad data:**
   ```python
   # Add to training script temporarily
   import tensorflow as tf
   
   # Check a few batches
   for batch in dataset.take(10):
       features, labels = batch
       print(f"Features range: [{tf.reduce_min(features):.4f}, {tf.reduce_max(features):.4f}]")
       print(f"Features has NaN: {tf.reduce_any(tf.math.is_nan(features))}")
   ```

4. **Add gradient clipping:**
   ```yaml
   training:
     gradient_clip_value: 1.0
   ```

---

### Issue: Model Not Learning (Stuck at ~50% accuracy)

**Symptoms:**
- Accuracy stays around 50% (random guessing)
- Loss not decreasing
- Validation metrics flat

**Root Causes:**
- Dataset too small
- Data augmentation too aggressive
- Learning rate too low
- Class imbalance

**Solutions:**

1. **Verify dataset size:**
   ```bash
   # Minimum recommended:
   # - Positive: 1000 samples
   # - Negative: 10000 samples
   python scripts/count_dataset.py dataset/positive
   python scripts/count_dataset.py dataset/negative
   ```

2. **Check class weights:**
   ```yaml
   # config.yaml - ensure weights are reasonable
   training:
     class_weights:
       positive: 1.0
       negative: 20.0
       hard_negative: 40.0
   ```

3. **Reduce augmentation intensity:**
   ```yaml
   augmentation:
     time_mask_width: 10  # Instead of 20
     freq_mask_width: 5   # Instead of 10
   ```

4. **Increase learning rate temporarily:**
   ```yaml
   training:
     learning_rates: [0.01, 0.001]  # Start higher, then decay
   ```

---

### Issue: Poor FAH (False Accepts per Hour)

**Symptoms:**
- FAH > 1.0 (target is < 0.3)
- Too many false activations
- Model triggers on background speech

**Solutions:**

1. **Use auto-tuning:**
   ```bash
   mww-autotune \
       --checkpoint checkpoints/best_weights.weights.h5 \
       --config standard \
       --target-fah 0.3 \
       --target-recall 0.92
   ```

2. **Add more hard negatives:**
   ```bash
   # Mine hard negatives from your ambient audio
   python scripts/mine_hard_negatives.py \
       --checkpoint checkpoints/best_weights.weights.h5 \
       --audio-dir /path/to/ambient/audio \
       --output dataset/hard_negative/
   ```

3. **Increase hard negative weight:**
   ```yaml
   training:
     class_weights:
       hard_negative: 60.0  # Increase from default 40.0
   ```

4. **Add more diverse negative samples:**
   - Record more background speech
   - Include various accents/languages
   - Add TV/radio/news audio

---

### Issue: Poor Recall (Missing Wake Word)

**Symptoms:**
- Recall < 0.90 (target is > 0.92)
- Model doesn't detect wake word consistently
- Missed activations in real-world use

**Solutions:**

1. **Reduce positive class weight (less penalty on false negatives):**
   ```yaml
   training:
     class_weights:
       positive: 0.5  # Reduce from default 1.0
   ```

2. **Increase training data diversity:**
   - Record from more speakers (5+)
   - Record at different distances
   - Include various room acoustics
   - Record at different times of day

3. **Use longer training:**
   ```bash
   # Use max_quality preset
   mww-train --config config/presets/max_quality.yaml
   ```

4. **Lower detection threshold (in ESPHome config):**
   ```yaml
   micro_wake_word:
     models:
       - model: wake_word.tflite
         probability_cutoff: 0.95  # Lower from 0.97
   ```

---

### Issue: Training Resumes from Wrong Checkpoint

**Symptoms:**
- Model accuracy drops after resuming
- Loss jumps unexpectedly
- Optimizer state seems lost

**Root Cause:**
The framework does not support `--resume` flag. Training must be restarted from scratch or from a specific checkpoint.

**Solution:**

```bash
# Don't use --resume flag (not supported)

# To restart from a specific checkpoint, specify it directly:
mww-train --config standard --checkpoint checkpoints/checkpoint_step_50000.weights.h5

# Or let it find the latest automatically (if implemented in your version)
# Check available checkpoints:
ls -la checkpoints/*.h5
```

---

## Export & ESPHome Compatibility Issues

### Issue: Export Fails

**Symptoms:**
- `KeyError: 'some_key'` during export
- `AttributeError` on model methods
- `ValueError` during quantization

**Diagnostic Commands:**

```bash
# Check checkpoint exists and is valid
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('checkpoints/best_weights.weights.h5')
print('Model loaded successfully')
print('Input shape:', model.input_shape)
print('Output shape:', model.output_shape)
"

# Check model architecture
python -c "
from src.model.architecture import build_model
from config.loader import load_full_config

config = load_full_config('standard')
model = build_model(config.model)
model.summary()
"
```

**Solutions:**

1. **Verify checkpoint compatibility:**
   ```bash
   # Ensure checkpoint matches current model architecture
   python scripts/evaluate_model.py --checkpoint checkpoints/best_weights.weights.h5
   ```

2. **Export without quantization (for debugging):**
   ```bash
   mww-export \
       --checkpoint checkpoints/best_weights.weights.h5 \
       --output models/exported/ \
       --no-quantize
   ```

3. **Check config consistency:**
   ```bash
   # Ensure export config matches training config
   python -c "
   from config.loader import load_full_config
   config = load_full_config('standard')
   print('Model config:', config.model)
   print('Export config:', config.export)
   "
   ```

---

### Issue: ESPHome Verification Fails

**Symptoms:**
- `✗ Subgraphs: 3 (expected: 2)`
- `✗ Input shape: [1, 4, 40] (expected: [1, 3, 40])`
- `✗ Input dtype: float32 (expected: int8)`
- `ESPHome compatible: NO`

**Root Causes:**
Violations of the Architectural Constitution (immutable constraints).

**Diagnostic Commands:**

```bash
# Run verification with verbose output
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose

# Check model structure
python scripts/verify_esphome.py models/exported/wake_word.tflite --json

# Compare with official model
python scripts/verify_esphome.py official_models/okay_nabu.tflite --verbose
```

**Critical Requirements (DO NOT VIOLATE):**

| Parameter | Required Value | Consequence if Wrong |
|-----------|---------------|---------------------|
| Input shape | `[1, 3, 40]` | Model receives wrong data |
| Input dtype | `int8` | Quantization mismatch |
| Output shape | `[1, 1]` | Wrong output tensor |
| Output dtype | `uint8` | Probability scaling wrong |
| Subgraphs | `2` | Streaming inference broken |
| Sample rate | `16000 Hz` | Feature extraction wrong |
| Mel bins | `40` | Feature dimension mismatch |

**Solutions:**

1. **Check audio frontend constants:**
   ```yaml
   # config.yaml - must match these values EXACTLY
   hardware:
     sample_rate_hz: 16000
     mel_bins: 40
     window_size_ms: 30
     window_step_ms: 10
   ```

2. **Verify export quantization:**
   ```yaml
   export:
     quantize: true
     inference_input_type: int8
     inference_output_type: uint8
   ```

3. **Check streaming layer configuration:**
   ```yaml
   model:
     streaming: true  # Must be enabled for ESPHome
   ```

4. **If all else fails, re-read ARCHITECTURAL_CONSTITUTION.md:**
   ```bash
   cat ARCHITECTURAL_CONSTITUTION.md | head -100
   ```

---

### Issue: Model Works in Python but Not on ESP32

**Symptoms:**
- Model passes verification
- Works in test scripts
- No detection on actual ESP32 device

**Diagnostic Steps:**

1. **Check tensor arena size:**
   ```bash
   # ESP32 has limited RAM
   python scripts/verify_esphome.py model.tflite
   # Look for "tensor_arena_size"
   
   # Should be < 30000 for ESP32, < 50000 for ESP32-S3
   ```

2. **Verify probability scaling:**
   ```python
   # Check output quantization
   import tensorflow as tf
   interpreter = tf.lite.Interpreter(model_path="model.tflite")
   output_details = interpreter.get_output_details()
   print("Output quantization:", output_details[0]['quantization'])
   # Should show scale and zero_point for uint8
   ```

3. **Test with official ESPHome:**
   ```yaml
   # In ESPHome config, enable verbose logging
   logger:
     level: VERBOSE
   
   micro_wake_word:
     models:
       - model: wake_word.tflite
         probability_cutoff: 0.5  # Very low for testing
   ```

4. **Check ESPHome version:**
   ```yaml
   # Requires ESPHome 2024.7.0+
esphome:
     name: my-device
     min_version: 2024.7.0
   ```

---

## Configuration Issues

### Issue: Config Not Loading

**Symptoms:**
- `ValueError: Invalid configuration`
- `KeyError: 'missing_key'`
- Default values not applied

**Diagnostic Commands:**

```bash
# Validate config
python -c "
from config.loader import load_full_config
try:
    config = load_full_config('standard', 'my_override.yaml')
    print('Config loaded successfully!')
    print('Training steps:', config.training.training_steps)
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('my_config.yaml'))"
```

**Common Issues:**

1. **Wrong preset name:**
   ```bash
   # Wrong
   mww-train --config standard  # Without .yaml
   
   # Correct
   mww-train --config config/presets/standard.yaml
   # OR use just the name if implemented:
   mww-train --config standard
   ```

2. **YAML syntax errors:**
   ```yaml
   # Wrong - mixed tabs and spaces
   training:
     batch_size: 64
   	learning_rate: 0.001  # Tab instead of spaces
   
   # Correct - use spaces only
   training:
     batch_size: 64
     learning_rate: 0.001
   ```

3. **Wrong data types:**
   ```yaml
   # Wrong - string instead of list
   training:
     training_steps: "10000, 5000"
   
   # Correct
   training:
     training_steps: [10000, 5000]
   ```

---

### Issue: Environment Variables Not Substituted

**Symptoms:**
- `${HOME}` appears literally in paths
- `${USER}` not expanded
- Default values not used

**Solution:**

```yaml
# config.yaml - use ${VAR} or ${VAR:-default}
paths:
  dataset_dir: ${HOME}/datasets/wakeword
  output_dir: ${MWW_OUTPUT_DIR:-./models}
```

**Verification:**
```bash
# Set environment variable
export MWW_OUTPUT_DIR=/mnt/storage/models

# Check substitution
python -c "
from config.loader import load_full_config
config = load_full_config('standard', 'my_config.yaml')
print('Output dir:', config.paths.output_dir)
"
```

---

## Performance & Optimization

### Issue: Training is Too Slow

**Symptoms:**
- < 10 steps/second
- GPU utilization < 50%
- CPU bottleneck

**Diagnostic Commands:**

```bash
# Monitor GPU utilization
nvidia-smi dmon -s u

# Check CPU usage
top
htop

# Profile training
python -m cProfile -o profile.stats -m src.training.trainer --config standard
```

**Solutions:**

1. **Enable mixed precision:**
   ```yaml
   performance:
     mixed_precision: true  # 2-3x speedup
   ```

2. **Increase batch size (if VRAM allows):**
   ```yaml
   training:
     batch_size: 256  # Increase from 128
   ```

3. **Use more data loading workers:**
   ```yaml
   training:
     num_workers: 8  # Adjust based on CPU cores
   ```

4. **Enable XLA compilation:**
   ```bash
   export TF_XLA_FLAGS=--tf_xla_auto_jit=2
   ```

5. **Use CuPy SpecAugment (already default):**
   ```yaml
   augmentation:
     use_cupy: true  # 5-10x faster than CPU
   ```

---

### Issue: High CPU Usage During Training

**Symptoms:**
- CPU at 100%
- GPU utilization low
- Data loading bottleneck

**Solutions:**

1. **Preprocess dataset:**
   ```bash
   # Convert to TFRecords for faster loading
   python -c "
   from src.data.dataset import create_tfrecord_dataset
   create_tfrecord_dataset('dataset/', 'dataset.tfrecord')
   "
   ```

2. **Cache features to disk:**
   ```yaml
   training:
     cache_dataset: true
   ```

3. **Optimize audio loading:**
   ```yaml
   data:
     preload_audio: true
     num_workers: 4
   ```

---

## Speaker Clustering Issues

### Issue: Clustering Takes Too Long

**Symptoms:**
- Hours to cluster dataset
- ECAPA-TDNN model download fails
- Out of memory during clustering

**Solutions:**

1. **Limit number of files for testing:**
   ```bash
   mww-cluster-analyze \
       --config standard \
       --max-files 100  # Test with small subset
   ```

2. **Use explicit cluster count:**
   ```bash
   # Faster than threshold-based clustering
   mww-cluster-analyze \
       --config standard \
       --n-clusters 200
   ```

3. **Check HuggingFace authentication:**
   ```bash
   huggingface-cli login
   # Must accept model terms at https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
   ```

4. **Process datasets separately:**
   ```bash
   # Instead of --dataset all
   mww-cluster-analyze --config standard --dataset positive
   mww-cluster-analyze --config standard --dataset negative
   ```

---

### Issue: Wrong Speaker Groupings

**Symptoms:**
- Same speaker split into multiple clusters
- Different speakers grouped together
- Too many or too few clusters

**Solutions:**

1. **Adjust similarity threshold:**
   ```bash
   # Lower threshold = fewer, larger clusters
   mww-cluster-analyze --config standard --threshold 0.65  # Default is 0.70
   
   # Higher threshold = more, smaller clusters
   mww-cluster-analyze --config standard --threshold 0.75
   ```

2. **Use explicit cluster count:**
   ```bash
   # If you know you have ~50 speakers
   mww-cluster-analyze --config standard --n-clusters 50
   ```

3. **Review clusters before applying:**
   ```bash
   # Always review the report first
   cat cluster_output/positive_cluster_report.txt
   
   # Preview changes
   mww-cluster-apply --namelist cluster_output/positive_namelist.json --dry-run
   ```

---

## Diagnostic Tools & Commands

### System Health Checks

```bash
# Check overall system health
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}% used')
print(f'Disk: {psutil.disk_usage(\"/\").percent}% used')
"

# Check GPU health
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv

# Check process resources
ps aux | grep python
```

### TensorFlow Diagnostics

```bash
# Full TF system report
python -c "
import tensorflow as tf
print('=== TensorFlow System Report ===')
print(f'TF version: {tf.__version__}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')
print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CUDA devices: {tf.config.experimental.list_physical_devices(\"GPU\")}')

# Memory info
if tf.config.list_physical_devices('GPU'):
    print('\n=== GPU Memory ===')
    for gpu in tf.config.list_physical_devices('GPU'):
        details = tf.config.experimental.get_memory_info(gpu)
        print(f'{gpu}: {details}')
"
```

### Dataset Diagnostics

```bash
# Comprehensive dataset check
python scripts/count_dataset.py dataset/positive
python scripts/count_dataset.py dataset/negative
python scripts/count_dataset.py dataset/hard_negative

# Audio quality check
python scripts/score_quality_full.py dataset/positive/

# Find duplicates
python scripts/audio_similarity_detector.py dataset/positive/ --threshold 0.95

# Analyze specific file
python scripts/audio_analyzer.py dataset/positive/speaker_001/sample.wav
```

### Model Diagnostics

```bash
# Verify checkpoint
python scripts/evaluate_model.py --checkpoint checkpoints/best_weights.weights.h5

# Compare two models
python scripts/compare_models.py \
    checkpoints/model_a.weights.h5 \
    checkpoints/model_b.weights.h5

# Verify streaming
python scripts/verify_streaming.py --checkpoint checkpoints/best_weights.weights.h5

# Check ESPHome compatibility
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
```

### Log Analysis

```bash
# Find errors in logs
grep -i "error\|exception\|fail" logs/terminal_*.log

# Find warnings
grep -i "warning" logs/terminal_*.log

# Check training progress
tail -100 logs/terminal_*.log | grep -E "step|loss|accuracy|fah"

# Extract metrics
python -c "
import re
with open('logs/terminal_20260306_061107.log') as f:
    content = f.read()
    
# Find all FAH values
fah_values = re.findall(r'FAH:\\s*([0-9.]+)', content)
print('FAH values:', fah_values[-10:])  # Last 10
"
```

---

## Emergency Procedures

### Complete Reset

If nothing works, start fresh:

```bash
# 1. Backup important data
cp -r checkpoints/ checkpoints_backup_$(date +%Y%m%d)/
cp -r models/ models_backup_$(date +%Y%m%d)/

# 2. Clean up
cd /path/to/microwakeword_trainer
rm -rf .mypy_cache .ruff_cache __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 3. Clean pip cache
pip cache purge

# 4. Recreate virtual environments
rm -rf ~/.venvs/mww-tf
rm -rf ~/.venvs/mww-torch

# 5. Reinstall
python3.11 -m venv ~/.venvs/mww-tf
source ~/.venvs/mww-tf/bin/activate
pip install -r requirements.txt

# 6. Verify
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Recover from Training Crash

```bash
# Find latest checkpoint
ls -lt checkpoints/*.h5 | head -5

# Evaluate checkpoint
python scripts/evaluate_model.py --checkpoint checkpoints/checkpoint_step_50000.weights.h5

# Export if model is usable
mww-export --checkpoint checkpoints/checkpoint_step_50000.weights.h5 --output models/recovered/

# Resume training from checkpoint (if supported)
mww-train --config standard --checkpoint checkpoints/checkpoint_step_50000.weights.h5
```

### Debug Mode

Enable verbose logging:

```bash
# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=0  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
export PYTHONUNBUFFERED=1

# Run with verbose output
python -u -m src.training.trainer --config standard 2>&1 | tee debug.log
```

---

## Getting Help

### Information to Include in Bug Reports

When reporting an issue, include:

1. **System Info:**
   ```bash
   python -c "
   import platform, tensorflow, sys
   print(f'Python: {sys.version}')
   print(f'OS: {platform.platform()}')
   print(f'TF: {tensorflow.__version__}')
   print(f'GPUs: {tensorflow.config.list_physical_devices(\"GPU\")}')
   "
   ```

2. **Command run:**
   ```bash
   # Exact command that failed
   mww-train --config config/presets/standard.yaml
   ```

3. **Full error message** (copy-paste, don't paraphrase)

4. **Logs:**
   ```bash
   # Attach relevant log file
   cat logs/terminal_*.log | tail -100
   ```

5. **Config:**
   ```bash
   # Show active config
   python -c "
   from config.loader import load_full_config
   import yaml
   config = load_full_config('standard')
   print(yaml.dump(config.__dict__))
   "
   ```

---

## Quick Reference: Common Errors

| Error | Quick Fix |
|-------|-----------|
| `CUDA_ERROR_OUT_OF_MEMORY` | Reduce `batch_size` in config |
| `CuPy is not available` | `pip install cupy-cuda12x==13.6.0` |
| `No audio files found` | Check `dataset/positive/` exists |
| `ESPHome compatible: NO` | Re-read ARCHITECTURAL_CONSTITUTION.md |
| `NaN loss` | Reduce learning rate, check data |
| `GPU not found` | `export CUDA_VISIBLE_DEVICES=0` |
| `ImportError: torch` | Switch to PyTorch env: `mww-torch` |
| `Permission denied` | `chmod +x scripts/*.py` |
| `Config not found` | Use full path: `config/presets/standard.yaml` |

---

**Last Updated:** 2026-03-06  
**Version:** v2.0.0  
**Maintainer:** Project Team
