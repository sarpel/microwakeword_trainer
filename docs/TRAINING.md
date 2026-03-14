# Training Documentation

## Overview

This document provides comprehensive guidance for training wake word detection models using the microwakeword_trainer framework. The training pipeline includes data loading, augmentation, two-phase training, hard negative mining, and performance profiling.

## Data Flow

```
Raw Audio → Data Ingestion → Feature Extraction → Dataset Creation → Training Loop → Model Export
```

### Detailed Pipeline

1. **Raw Audio**: WAV files organized by speaker in dataset/positive/, dataset/negative/, etc.
2. **Data Ingestion**: Audio validation, VAD trimming, splitting, quality scoring
3. **Feature Extraction**: pymicro-features library converts to 40 mel-bin spectrograms
4. **Dataset Creation**: RaggedMmap storage for variable-length sequences
5. **Training Loop**: Two-phase training with augmentation and hard negative mining
6. **Model Export**: TFLite conversion with ESPHome compatibility verification

## Data Preparation and Ingestion

### Audio Organization

The dataset should be organized as follows:

```
dataset/
├── positive/          # Wake word samples (by speaker)
├── negative/          # Background speech
├── hard_negative/     # False positives
├── background/        # Noise/ambient
└── rirs/             # Room impulse responses
```

### Ingestion Process

1. **Validation**: Check sample rate (16kHz), format, duration
2. **VAD Trimming**: Remove silence using voice activity detection
3. **Splitting**: Break long audio into segments
4. **Quality Scoring**: Assess audio quality for filtering

### Key Configuration

```yaml
data:
  sample_rate: 16000
  min_duration: 0.3
  max_duration: 2.0
  vad_trim: true
  quality_threshold: 0.95
```

## Feature Extraction

### Parameters

- **Library**: pymicro-features (ESPHome-compatible)
- **Sample Rate**: 16000 Hz
- **Window**: 30ms Hamming window
- **Step**: 10ms overlap (~67% overlap)
- **Mel Bins**: 40 (ESPHome requirement)
- **Output Shape**: [time_frames, 40]

### Process

```python
# Feature extraction pipeline
micro_frontend = MicroFrontend(
    sample_rate=16000,
    window_size_ms=30,
    window_step_ms=10,
    num_channels=40
)
features = micro_frontend.process(audio)
```

### ESPHome Compatibility

Features are extracted to match ESPHome's micro_wake_word component exactly:
- 40 mel bins
- Log mel spectrogram
- Compatible with streaming inference

## Augmentation Pipeline

### Waveform-Level Augmentation (8 Types)

1. **SevenBandParametricEQ**: Random equalization across frequency bands
2. **TanhDistortion**: Nonlinear distortion for robustness
3. **PitchShift**: ±2 semitones pitch variation
4. **BandStopFilter**: Random frequency band removal
5. **AddColorNoise**: SNR-controlled noise addition
6. **AddBackgroundNoiseFromFile**: Real background audio mixing
7. **ApplyImpulseResponse**: Room reverberation simulation
8. **Gain**: ±3dB amplitude variation

### Spectrogram-Level Augmentation

**GPU SpecAugment** (CuPy-based):
- **Time Masking**: Random time segment masking
- **Frequency Masking**: Random frequency band masking

```yaml
augmentation:
  waveform:
    enabled: true
    types: [eq, distortion, pitch_shift, band_stop, color_noise, background_noise, rir, gain]
  spectrogram:
    time_mask_max_size: [5, 3]
    freq_mask_max_size: [2, 1]
```

## Training Loop Structure

### Two-Phase Training

```
Phase 1: Feature Learning (20000 steps)
├── Learning Rate: 0.001
├── Focus: Basic feature extraction
└── Augmentation: Full pipeline

Phase 2: Fine-tuning (10000 steps)
├── Learning Rate: 0.0001
├── Focus: Precision optimization
└── Augmentation: Reduced intensity
```

### Training Loop Diagram

```
for epoch in training:
    for batch in dataset:
        # Data loading
        features, labels = load_batch()

        # Augmentation
        if phase_1:
            features = apply_waveform_aug(features)
        features = apply_spectrogram_aug(features)

        # Forward pass
        predictions = model(features)

        # Loss calculation
        loss = binary_crossentropy(predictions, labels)
        loss = apply_class_weights(loss, weights)

        # Hard negative mining
        if step % mining_interval == 0:
            hard_negatives = mine_hard_negatives(predictions)
            update_dataset(hard_negatives)

        # Optimization
        optimizer.apply_gradients(loss)

        # Logging
        log_metrics(step, loss, accuracy)
```

### Class Weights

- **Positive**: 1.0 (wake word samples)
- **Negative**: 20.0 (compensates for class imbalance)
- **Hard Negative**: 40.0 (emphasizes difficult false positives)

### Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| training_steps | [20000, 10000] | Total training duration |
| batch_size | 128 | Speed vs VRAM trade-off |
| negative_class_weight | 20.0 | Class balance compensation |
| time_mask_max_size | [5, 3] | SpecAugment time masking |
| ambient_duration_hours | 10.0 | FAH calculation baseline |

## Hard Negative Mining

### Process

1. **Detection**: Identify false positives with confidence > threshold
2. **Collection**: Gather hard negative samples during training
3. **Integration**: Add to training dataset with high class weight
4. **Iteration**: Repeat mining throughout training

### Configuration

```yaml
hard_negative_mining:
  enabled: true
  fp_threshold: 0.8
  max_samples: 5000
  mining_interval_epochs: 5
```

### Benefits

- Improves model robustness
- Reduces false positive rate
- Adapts to training data distribution

### Async Mining (Background Mining)

For improved throughput, the framework provides `AsyncHardExampleMiner` which runs hard negative mining in a background thread. Async mining is enabled by setting two configuration options:

```yaml
performance:
  async_mining: true  # Enable async mining mode

hard_negative_mining:
  enabled: true
  collection_mode: "mine_immediately"  # Use async mode
  fp_threshold: 0.8
  max_samples: 5000
  mining_interval_epochs: 5
```

**Benefits**:
- Runs mining in parallel with training
- No training interruptions for mining
- Faster hard negative updates
- Better GPU utilization

**Configuration Relationship**:
- **`performance.async_mining`**: Global flag to enable async mining (default: `false`)
- **`hard_negative_mining.collection_mode`**: Determines when mining occurs
  - `"log_only"`: Only log false predictions; mining happens later (async mining not used)
  - `"mine_immediately"`: Mine immediately during training; uses async miner if `performance.async_mining=true`

**When Both Settings Apply**:
- If `performance.async_mining=true` AND `hard_negative_mining.collection_mode="mine_immediately"`: AsyncHardExampleMiner is used (background thread, non-blocking)
- If `performance.async_mining=false` OR `hard_negative_mining.collection_mode="log_only"`: Traditional HardExampleMiner is used (synchronous, blocking during mining operations)

**Common Parameters** (same for both sync and async):
- `fp_threshold` / `confidence_threshold`: Minimum confidence for hard negative candidates (0.0-1.0)
- `max_samples`: Maximum hard negatives to collect per mining operation
- `mining_interval_epochs`: Number of epochs between mining operations (used by both modes to determine timing)
- `class_weight`: Weight assigned to hard negatives during training (inherited from `training.hard_negative_class_weight`)

> **Note**: Async mining is enabled by default in the `standard` and `max_quality` presets for better performance. See [docs/GUIDE.md](GUIDE.md) for all available presets and their configurations.

## Performance Optimization

### GPU Acceleration

- **CuPy SpecAugment**: GPU-based spectrogram augmentation
- **Mixed Precision**: Automatic mixed precision training
- **TF Data Pipeline**: Optimized data loading with prefetching

### Memory Management

- **RaggedMmap**: Efficient variable-length sequence storage
- **Batch Size Tuning**: Balance speed and VRAM usage
- **Gradient Accumulation**: Handle large batches

### Profiling

```yaml
performance:
  mixed_precision: true
  prefetch_buffer_size: 2
  num_parallel_calls: 4
  profile_training: true
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptoms**: Training crashes with OOM error
**Solutions**:
- Reduce batch_size in config
- Enable gradient_checkpointing
- Use mixed precision training

#### 2. Slow Training
**Symptoms**: Training takes too long
**Solutions**:
- Increase batch_size if VRAM allows
- Enable XLA compilation
- Check GPU utilization with nvidia-smi

#### 3. Poor Model Performance
**Symptoms**: Low accuracy, high FAH
**Solutions**:
- Check data quality and quantity
- Adjust augmentation intensity
- Increase training steps
- Enable hard negative mining

#### 4. ESPHome Compatibility Issues
**Symptoms**: Model works in training but fails on device
**Solutions**:
- Verify feature extraction matches ESPHome
- Check tensor shapes in ARCHITECTURAL_CONSTITUTION.md
- Use verify_esphome.py script

### Performance Monitoring

Use TensorBoard for monitoring:
```bash
tensorboard --logdir logs/
```

Key metrics to watch:
- Training loss
- Validation accuracy
- False Acceptance per Hour (FAH)
- False Rejection Rate (FRR)

### Configuration Validation

Always validate config before training:
```python
from config.loader import load_full_config
config = load_full_config('standard', 'custom.yaml')
```

## Best Practices

1. **Data Quality**: Use high-quality, diverse audio data
2. **Augmentation Balance**: Don't over-augment (can hurt performance)
3. **Class Weights**: Tune based on your data distribution
4. **Hard Negative Mining**: Enable for production models
5. **Regular Checkpointing**: Save models frequently
6. **Cross-Validation**: Use separate validation set
7. **ESPHome Testing**: Always verify device compatibility

## Configuration Examples

### Fast Test Configuration
```yaml
training:
  steps: [1000, 500]
  batch_size: 64
augmentation:
  waveform:
    enabled: false
  spectrogram:
    time_mask_max_size: [1, 1]
```

### Production Configuration
```yaml
training:
  steps: [20000, 10000]
  batch_size: 128
  negative_class_weight: 20.0
hard_negative_mining:
  enabled: true
  confidence_threshold: 0.8
performance:
  mixed_precision: true
```

## Next Steps

After training:
1. Export model with `mww-export`
2. Verify ESPHome compatibility
3. Test on device
4. Consider auto-tuning for better FAH/recall balance

## Checkpoint Selection Strategy

The training loop uses a **two-stage checkpoint strategy** in `_is_best_model()` to save the best epoch:

### Stage 1 — Warm-up
Active until any epoch first meets `FAH ≤ target_fah × 1.1`.

Saves by **PR-AUC** (`auc_pr`) improvement. PR-AUC is:
- Threshold-free — captures model quality across all operating points
- Robust to class imbalance — wake word audio is massively imbalanced (ambient >> wake word)
- Aligned with the autotuner's Pareto objective (which also uses `auc_pr`)

### Stage 2 — Operational
Active once any epoch has met the FAH budget.

Saves by **constrained recall** (`recall_at_target_fah`) improvement, **only** when the current epoch also satisfies `FAH ≤ target_fah × 1.1`. This directly encodes the production constraint:
> *"Best recall of all models that will deploy within the FAH budget."*

### Logged but not used for selection
The composite `quality_score = (0.7 × operating_recall + 0.3 × AVR) × Lorentzian_FAH_penalty` is still computed and logged (visible in TensorBoard and plateau metrics) but does **not** drive checkpoint decisions.

### Configuration
The FAH budget comes from `evaluation.target_fah` in your config preset. A 10% tolerance (`× 1.1`) is applied to avoid rejecting epochs that are marginally over budget.

```yaml
# In your config preset or override:
evaluation:
  target_fah: 0.5        # Maximum acceptable false activations per hour
  target_recall: 0.90    # Minimum acceptable recall (used by recall_at_target_fah metric)
```

For detailed configuration options, see `docs/GUIDE.md`.
