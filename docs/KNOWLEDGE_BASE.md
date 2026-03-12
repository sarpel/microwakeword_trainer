# microwakeword_trainer — Knowledge Base

> **Comprehensive reference for the GPU-accelerated wake word training framework**  
> Version 2.0.0 | Last Updated: 2026-03-12

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Concepts](#2-architecture-concepts)
3. [Configuration Knowledge](#3-configuration-knowledge)
4. [Training Concepts](#4-training-concepts)
5. [Export & Deployment](#5-export--deployment)
6. [Data Management](#6-data-management)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Troubleshooting Patterns](#8-troubleshooting-patterns)
9. [Cross-Reference Index](#9-cross-reference-index)

---

## 1. Overview

### 1.1 What is microwakeword_trainer?

A complete pipeline for training custom wake word detection models that run on ESP32 microcontrollers via ESPHome. The framework trains a **MixedNet** neural network and exports it as an INT8-quantized TFLite model compatible with ESPHome's `micro_wake_word` component.

**Key Value Propositions:**
- 🎯 Train custom "Hey Siri" / "OK Google" style wake words
- ⚡ GPU-accelerated training with CuPy SpecAugment (no CPU fallback)
- 📱 ESPHome-compatible export with INT8 quantization
- 🔄 Streaming inference for real-time detection
- 🔊 Speaker clustering to prevent data leakage

### 1.2 Complete Workflow

```
Audio Collection → Speaker Clustering → Data Ingestion → Feature Extraction →
Augmentation → Two-Phase Training → Hard Negative Mining → TFLite Export →
ESPHome Deployment
```

### 1.3 Project Structure

| Directory | Purpose |
|-----------|---------|
| `config/` | Configuration system with YAML presets |
| `src/training/` | Training loop, augmentation, mining |
| `src/tuning/` | Auto-tuning for FAH/recall optimization |
| `src/data/` | Dataset, features, augmentation, clustering |
| `src/model/` | MixedNet architecture, streaming layers |
| `src/export/` | TFLite export, verification, manifest |
| `src/evaluation/` | Metrics, FAH estimation, calibration |
| `src/utils/` | GPU config, performance, logging |
| `src/tools/` | CLI tools (cluster-analyze, cluster-apply) |
| `scripts/` | Standalone utilities |
| `docs/` | Documentation |
| `specs/` | Implementation status and testing plans |

### 1.4 Environment Requirements

**CRITICAL: Two Separate Virtual Environments Required**

| Environment | Purpose | Key Dependencies |
|-------------|---------|------------------|
| `mww-tf` | Training, export, inference | TensorFlow, CuPy, tflite |
| `mww-torch` | Speaker clustering | PyTorch, SpeechBrain, ECAPA-TDNN |

> **Why two environments?** TensorFlow and PyTorch have incompatible CUDA dependencies and cannot coexist.

**System Requirements:**
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.x
- 16GB+ RAM
- 10GB+ storage

---

## 2. Architecture Concepts

### 2.1 MixedNet Architecture

The default model architecture optimized for edge deployment.

```
Input: [batch, time_frames, 40]  (mel spectrogram)
    ↓
Conv2D (first_conv_filters=30, kernel=5, stride=3)
    ↓
MixConvBlock × 4  (parallel depthwise convolutions)
    ↓
Flatten
    ↓
Dense(1, sigmoid)  (classification head)
    ↓
Output: [batch, 1]  (wake word probability)
```

**Key Concepts:**
- **MixConvBlock**: Parallel depthwise convolutions with different kernel sizes
- **Stride = 3**: Determines input cadence (3 frames = 30ms)
- **No recurrent layers**: LSTM/GRU not supported in ESPHome

### 2.2 Immutable Architectural Constants

These values are **burned into ESPHome firmware** and cannot be changed:

| Constant | Value | Impact if Changed |
|----------|-------|-------------------|
| `sample_rate_hz` | 16000 Hz | Audio feature mismatch |
| `mel_bins` | 40 | Input tensor shape mismatch |
| `window_size_ms` | 30 ms | Feature extraction mismatch |
| `window_step_ms` | 10 ms | Temporal resolution mismatch |
| `stride` | 3 | Inference timing desync |
| Input shape | [1, 3, 40] | ESPHome buffer mismatch |
| Input dtype | int8 | Quantization mismatch |
| Output dtype | uint8 | **Critical**: ESPHome reads as unsigned |

> **See:** [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md) for complete specification

### 2.3 Streaming Inference Model

The exported TFLite model uses **streaming inference** with ring buffers:

**Dual-Subgraph Structure:**
- **Subgraph 0**: Main inference (55 ops, 95 tensors)
- **Subgraph 1**: Initialization (12 ops, 12 tensors)

**6 State Variables (Ring Buffers):**
| Variable | Shape | Purpose |
|----------|-------|---------|
| `stream_0` | [1, 2, 1, 40] | Initial conv buffer |
| `stream_1` | [1, 4, 1, 32] | MixConv block 0 |
| `stream_2` | [1, 10, 1, 64] | MixConv block 1 |
| `stream_3` | [1, 14, 1, 64] | MixConv block 2 |
| `stream_4` | [1, 22, 1, 64] | MixConv block 3 |
| `stream_5` | [1, 5, 1, 64] | Temporal flatten buffer |

**Ring Buffer Law:**
```
buffer_frames = kernel_size - stride
```

### 2.4 Quantization

**Input Quantization:**
- Scale: 0.101961 (26/255)
- Zero point: -128
- Range: int8[-128, 127] → float[0.0, ~26.0]

**Output Quantization:**
- Scale: 0.00390625 (1/256)
- Zero point: 0
- Range: uint8[0, 255] → float[0.0, ~1.0]

**Critical Export Flags:**
```python
converter.optimizations = {tf.lite.Optimize.DEFAULT}
converter._experimental_variable_quantization = True  # REQUIRED
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.uint8  # MUST be uint8
```

---

## 3. Configuration Knowledge

### 3.1 Configuration System Architecture

**Loading Priority (later wins):**
```
Preset YAML → Override YAML → Environment Variables
```

**12 Configuration Dataclasses:**
1. `HardwareConfig` — Immutable audio frontend
2. `PathsConfig` — Directory paths
3. `TrainingConfig` — Hyperparameters, 2-phase settings
4. `ModelConfig` — MixedNet architecture
5. `AugmentationConfig` — Data augmentation settings
6. `PerformanceConfig` — GPU, threading, profiling
7. `SpeakerClusteringConfig` — ECAPA-TDNN clustering
8. `HardNegativeMiningConfig` — Mining parameters
9. `ExportConfig` — TFLite export settings
10. `PreprocessingConfig` — VAD, resampling
11. `QualityConfig` — Audio quality thresholds
12. `EvaluationConfig` — Metrics and targets

### 3.2 Preset Comparison

| Preset | Steps | Batch | Time | Use Case |
|--------|-------|-------|------|----------|
| `fast_test` | [2000, 1000] | 32 | ~1hr | Quick iteration |
| `standard` | [20000, 10000] | 128 | ~8hr | Production |
| `max_quality` | [40000, 25000, 10000] | 64 | ~24hr | Best accuracy |

### 3.3 Environment Variable Substitution

```yaml
paths:
  dataset_dir: ${DATASET_DIR:-./dataset}
  checkpoint_dir: ${CHECKPOINT_DIR:-./checkpoints}
```

**Usage:**
```bash
export DATASET_DIR=/mnt/data/wakeword
mww-train --config standard
```

### 3.4 Configuration Validation Rules

1. `sample_rate_hz` must be 16000
2. `stride` must be 3
3. `train_split + val_split + test_split` must equal 1.0
4. `inference_input_type` must be "int8"
5. `inference_output_type` must be "uint8"
6. `training_steps` and `learning_rates` must have same length

---

## 4. Training Concepts

### 4.1 Two-Phase Training

**Phase 1 — Feature Learning:**
- Steps: 20,000 (default)
- Learning rate: 0.001
- Focus: Basic feature extraction
- Augmentation: Full pipeline

**Phase 2 — Fine-tuning:**
- Steps: 10,000 (default)
- Learning rate: 0.0001
- Focus: Precision optimization
- Augmentation: Reduced intensity

### 4.2 Class Weighting

| Class | Weight | Purpose |
|-------|--------|---------|
| Positive | 1.0 | Wake word samples |
| Negative | 20.0 | Compensate for imbalance |
| Hard Negative | 40.0 | Emphasize difficult examples |

> Target ratio: 1 positive : 10+ negative samples

### 4.3 Checkpoint Selection Strategy

**Two-Stage Strategy (2026-03-12 update):**

**Stage 1 — Warm-up:**
- Until any epoch meets `FAH ≤ target_fah × 1.1`
- Saves by **PR-AUC** (`auc_pr`) improvement
- Threshold-free, robust to imbalance

**Stage 2 — Operational:**
- Once FAH budget is met
- Saves by **constrained recall** (`recall_at_target_fah`)
- Only when current epoch also satisfies FAH budget

### 4.4 Hard Negative Mining

**Two Modes:**

1. **Synchronous** (`AsyncHardExampleMiner` disabled):
   - Blocking during training
   - Traditional `HardExampleMiner`

2. **Asynchronous** (`performance.async_mining: true`):
   - Background thread
   - Non-blocking to training
   - Better GPU utilization

**Configuration:**
```yaml
performance:
  async_mining: true

hard_negative_mining:
  enabled: true
  collection_mode: "mine_immediately"  # or "log_only"
  fp_threshold: 0.8
  max_samples: 5000
  mining_interval_epochs: 5
```

### 4.5 Augmentation Pipeline

**Waveform-Level (8 types):**
1. `SevenBandParametricEQ` — Random equalization
2. `TanhDistortion` — Nonlinear distortion
3. `PitchShift` — ±2 semitones
4. `BandStopFilter` — Frequency band removal
5. `AddColorNoise` — SNR-controlled noise
6. `AddBackgroundNoiseFromFile` — Real background audio
7. `ApplyImpulseResponse` — Room reverberation
8. `Gain` — ±3dB amplitude

**Spectrogram-Level (GPU):**
- `time_mask_max_size`: [5, 3] (phase 1, phase 2)
- `freq_mask_max_size`: [3, 2]
- Requires CuPy (no CPU fallback)

---

## 5. Export & Deployment

### 5.1 Export Pipeline

```
Keras Checkpoint
    ↓
Build StreamingExportModel
    ↓
Fold BatchNorm (eliminates ReadVariableOp)
    ↓
Convert to SavedModel (ExportArchive, NOT model.export())
    ↓
TFLite Conversion + INT8 Quantization
    ↓
Generate Manifest
    ↓
Verification
```

### 5.2 TFLite Export Requirements

**Critical Settings:**
```python
converter.optimizations = {tf.lite.Optimize.DEFAULT}
converter._experimental_variable_quantization = True  # REQUIRED for state vars
converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.uint8  # NOT int8!
```

**Representative Dataset:**
- Minimum 500 samples (recommended 2000+)
- Must include boundary anchors (0.0 and 26.0)

### 5.3 ESPHome Manifest

```json
{
  "type": "micro",
  "version": 2,
  "micro": {
    "feature_step_size": 10,
    "sliding_window_size": 5,
    "tensor_arena_size": 26080,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

**Tensor Arena Sizing:**
- `okay_nabu` reference: ~135,873 bytes
- Add 10% margin to measured value
- Underestimating causes silent memory corruption

### 5.4 Verification Checklist

| Check | Expected | Critical |
|-------|----------|----------|
| Subgraphs | 2 | ✅ Yes |
| Input shape | [1, 3, 40] | ✅ Yes |
| Input dtype | int8 | ✅ Yes |
| Output shape | [1, 1] | ✅ Yes |
| Output dtype | uint8 | ✅ Yes |
| State variables | 6 | ✅ Yes |
| Quantization | enabled | ✅ Yes |

> **Command:** `python scripts/verify_esphome.py model.tflite --verbose`

---

## 6. Data Management

### 6.1 Dataset Structure

```
dataset/
├── positive/           # Wake word recordings (REQUIRED)
│   ├── speaker_001/    # Organize by speaker
│   └── speaker_002/
├── negative/           # Background speech (REQUIRED)
├── hard_negative/      # False positives (RECOMMENDED)
├── background/         # Ambient noise (RECOMMENDED)
└── rirs/              # Room impulse responses (OPTIONAL)
```

### 6.2 Audio Requirements

| Property | Requirement |
|----------|-------------|
| Format | WAV, 16-bit PCM |
| Sample rate | 16kHz |
| Duration | 1-3 seconds per clip |
| Channels | Mono |

### 6.3 Minimum Dataset Sizes

| Type | Minimum | Recommended |
|------|---------|-------------|
| Positive | 100 | 1000+ |
| Negative | 1000 | 10000+ |
| Hard Negative | 50 | 500+ |

### 6.4 Speaker Clustering Workflow

**Purpose:** Prevent train/test data leakage from same speaker

```bash
# Step 1: Analyze (dry-run)
mww-cluster-analyze --config standard --dataset all --n-clusters 200

# Step 2: Review report
cat cluster_output/positive_cluster_report.txt

# Step 3: Organize files
mww-cluster-apply --namelist-dir cluster_output --dry-run  # Preview
mww-cluster-apply --namelist-dir cluster_output            # Execute
```

**Clustering Algorithms:**
- `agglomerative` — Default, threshold-based
- `hdbscan` — For large datasets
- Explicit `n_clusters` — Recommended for short clips

---

## 7. Evaluation & Metrics

### 7.1 Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **FAH** | False Activations per Hour | < 0.5 |
| **Recall** | TP / (TP + FN) | > 0.90 |
| **Precision** | TP / (TP + FP) | > 0.90 |
| **F1** | 2 × (P × R) / (P + R) | > 0.90 |
| **AUC-ROC** | Area under ROC curve | > 0.95 |
| **AUC-PR** | Area under PR curve | > 0.90 |

### 7.2 FAH (False Activations per Hour)

**Definition:** Number of false positive detections per hour of ambient audio

**Calculation:**
```
FAH = (false_positives / total_ambient_duration_hours)
```

**Targets:**
- 🟢 < 0.5 — Production ready
- 🟡 0.5-2.0 — Acceptable, can improve
- 🔴 > 2.0 — Too many false triggers

### 7.3 Auto-Tuning

**Purpose:** Optimize FAH/recall without full retraining

**Targets:**
- FAH < 0.3 (configurable)
- Recall > 0.92 (configurable)

**Usage:**
```bash
mww-autotune \
    --checkpoint checkpoints/best_weights.weights.h5 \
    --config standard \
    --target-fah 0.2 \
    --target-recall 0.95
```

**Features:**
- Pareto archive for multi-objective optimization
- Thompson sampling for strategy selection
- 7 strategy arms (threshold, architecture, learning rate, etc.)
- User-defined hard negatives support

---

## 8. Troubleshooting Patterns

### 8.1 Common Error Patterns

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| `CUDA_ERROR_OUT_OF_MEMORY` | Batch size too large | Reduce `batch_size` to 32 |
| `CuPy is not available` | Wrong CUDA version | `pip install cupy-cuda12x` |
| `ESPHome compatible: NO` | Violated architecture rules | Read ARCHITECTURAL_CONSTITUTION.md |
| `NaN loss` | Learning rate too high | Reduce to 0.0001 |
| `loss not decreasing` | Insufficient data | Add more samples |
| `FAH > 2.0` | Not enough negatives | Add 10x more negative samples |
| `Recall < 0.80` | Not enough positives | Add more speaker diversity |

### 8.2 Training-Export AUC Gap

**Symptom:** Training AUC 0.99, Export AUC 0.85

**Root Cause (Fixed 2026-03-11):**
- Training used `GlobalAveragePooling2D`
- Export used `Flatten`
- Dense layer weight mismatch (64 vs 384 inputs)

**Solution:**
- Retrain with current code (post-2026-03-11)
- Both now use `Flatten` consistently

### 8.3 Auto-Tune Confirmation Failure

**Symptom:** FAH 0 → 129+ between tuning and confirmation

**Root Cause (Fixed 2026-03-10):**
- `_serialize_weights()` used `model.trainable_weights`
- Excluded BatchNorm moving statistics
- Weights mis-assigned during deserialization

**Solution:**
- Use `model.get_weights()`/`model.set_weights()` for full state
- Update to code post-2026-03-10

### 8.4 State Tensor Order Mismatch

**Symptom:** State shapes in wrong order in TFLite

**Root Cause (Fixed 2026-03-10):**
- TFLite sorts variables alphabetically
- `stream` vs `stream_1` ordering issue

**Solution:**
- Renamed `stream` → `stream_0`
- Now sorts correctly: `stream_0` < `stream_1` < ...

---

## 9. Cross-Reference Index

### 9.1 By Topic

**Audio Frontend:**
- Architecture: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md) Article I
- Implementation: `src/data/features.py` (`MicroFrontend`)
- Configuration: `HardwareConfig` in [CONFIGURATION.md](CONFIGURATION.md)

**BatchNorm Folding:**
- Export: `src/export/tflite.py`
- Architecture: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md) Article VIII
- Verification: `scripts/verify_esphome.py`

**Class Weights:**
- Training: `src/training/trainer.py` (`_apply_class_weights`)
- Configuration: `TrainingConfig` in [CONFIGURATION.md](CONFIGURATION.md)
- Best Practices: [TRAINING.md](TRAINING.md)

**Configuration:**
- System: [CONFIGURATION.md](CONFIGURATION.md)
- Loader: `config/loader.py`
- Presets: `config/presets/`

**Data Augmentation:**
- Waveform: `src/training/augmentation.py`
- Spectrogram: `src/data/spec_augment_gpu.py`
- Configuration: [CONFIGURATION.md](CONFIGURATION.md) §AugmentationConfig

**Dataset:**
- Structure: [MASTER_GUIDE.md](MASTER_GUIDE.md) §3
- Loading: `src/data/dataset.py` (`WakeWordDataset`)
- Requirements: [README.md](README.md)

**ESPHome Compatibility:**
- Requirements: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md)
- Verification: `scripts/verify_esphome.py`
- Deployment: [EXPORT.md](EXPORT.md)

**Evaluation:**
- Metrics: `src/evaluation/metrics.py` (`MetricsCalculator`)
- FAH: `src/evaluation/fah_estimator.py`
- Usage: [TRAINING.md](TRAINING.md) §Monitoring

**Export:**
- Pipeline: `src/export/tflite.py`
- Manifest: `src/export/manifest.py`
- Documentation: [EXPORT.md](EXPORT.md)

**Feature Extraction:**
- Implementation: `src/data/features.py`
- Configuration: `HardwareConfig` ([CONFIGURATION.md](CONFIGURATION.md))
- ESPHome Match: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md)

**GPU Configuration:**
- Setup: `src/utils/performance.py`
- Environment: [MASTER_GUIDE.md](MASTER_GUIDE.md) §2
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) §GPU

**Hard Negative Mining:**
- Sync: `src/training/miner.py` (`HardExampleMiner`)
- Async: `src/training/async_miner.py` (`AsyncHardExampleMiner`)
- Configuration: `HardNegativeMiningConfig`

**Metrics:**
- Implementation: `src/evaluation/metrics.py`
- FAH: `src/evaluation/fah_estimator.py`
- Calibration: `src/evaluation/calibration.py`

**MixedNet:**
- Architecture: `src/model/architecture.py`
- Streaming: `src/model/streaming.py`
- Documentation: [ARCHITECTURE.md](ARCHITECTURE.md)

**Model Architecture:**
- MixedNet: `src/model/architecture.py`
- Streaming: `src/model/streaming.py`
- Constitution: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md)

**Quantization:**
- Parameters: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md) Article III
- Export: `src/export/tflite.py`
- Representative Dataset: [EXPORT.md](EXPORT.md)

**Speaker Clustering:**
- Analysis: `src/tools/cluster_analyze.py`
- Application: `src/tools/cluster_apply.py`
- Implementation: `src/data/clustering.py`

**Streaming:**
- Layers: `src/model/streaming.py`
- Export: `src/export/tflite.py` (`StreamingExportModel`)
- State Variables: [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md) Article VI

**TFLite:**
- Export: `src/export/tflite.py`
- Verification: `scripts/verify_esphome.py`
- Manifest: `src/export/manifest.py`

**Training:**
- Loop: `src/training/trainer.py`
- Configuration: [TRAINING.md](TRAINING.md)
- Checkpoint Selection: [specs/implementation_status.md](specs/implementation_status.md)

**Troubleshooting:**
- Guide: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Common Errors: §8.1 above
- Quick Fix Table: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) §Quick Reference

### 9.2 By File

| File | Primary Topics |
|------|----------------|
| `config/loader.py` | Configuration system, 12 dataclasses |
| `src/training/trainer.py` | Training loop, checkpoint selection |
| `src/tuning/autotuner.py` | Auto-tuning, Pareto optimization |
| `src/data/dataset.py` | Dataset loading, RaggedMmap |
| `src/data/features.py` | Mel spectrogram extraction |
| `src/model/architecture.py` | MixedNet, MixConv blocks |
| `src/model/streaming.py` | Streaming layers, ring buffers |
| `src/export/tflite.py` | TFLite export, quantization |
| `src/evaluation/metrics.py` | FAH, recall, AUC |
| `src/evaluation/fah_estimator.py` | False activation estimation |

### 9.3 CLI Commands Quick Reference

| Command | Environment | Purpose |
|---------|-------------|---------|
| `mww-train` | TF | Training pipeline |
| `mww-export` | TF | Export to TFLite |
| `mww-autotune` | TF | Post-training tuning |
| `mww-cluster-analyze` | PyTorch | Speaker clustering |
| `mww-cluster-apply` | PyTorch | Apply cluster organization |
| `mww-mine` | TF | Hard negative mining |

### 9.4 Critical Anti-Patterns

| Anti-Pattern | Why Wrong | Correct Approach |
|--------------|-----------|------------------|
| Using `model.trainable_weights` | Excludes BatchNorm stats | Use `model.get_weights()` |
| `inference_output_type = int8` | ESPHome reads as uint8 | Must be `uint8` |
| `model.export()` | Fails with ring buffers | Use `ExportArchive` |
| Changing `mel_bins` | Input shape mismatch | Keep at 40 |
| Mixing TF/PyTorch envs | CUDA conflicts | Use separate venvs |

---

## Related Documentation

- [README.md](README.md) — Project overview and quickstart
- [MASTER_GUIDE.md](MASTER_GUIDE.md) — Complete operational guide
- [ARCHITECTURAL_CONSTITUTION.md](ARCHITECTURAL_CONSTITUTION.md) — Immutable architectural truth
- [ARCHITECTURE.md](ARCHITECTURE.md) — MixedNet architecture details
- [CONFIGURATION.md](CONFIGURATION.md) — Configuration reference
- [TRAINING.md](TRAINING.md) — Training workflow
- [EXPORT.md](EXPORT.md) — Export process
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Problem solving
- [PHONETIC_SCORER.md](PHONETIC_SCORER.md) — Hard negative scoring
- [specs/implementation_status.md](specs/implementation_status.md) — Component status

---

*microwakeword_trainer v2.0.0 — Knowledge Base*  
*Generated: 2026-03-12*
