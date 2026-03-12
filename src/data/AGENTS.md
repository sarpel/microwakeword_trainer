# src/data/

Audio data pipeline: ingestion, feature extraction, augmentation, dataset storage, clustering, preprocessing, and quality scoring for wake word training.

## Files

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `dataset.py` | 962 | Training dataset with mmap storage | `RaggedMmap`, `FeatureStore`, `WakeWordDataset`, `load_dataset()`, `get_split_file_paths()` |
| `ingestion.py` | 777 | Audio loading, validation, metadata | `SampleRecord`, `Clips`, `ClipsLoaderConfig`, `Split`, `Label`, `load_clips()` |
| `clustering.py` | 1,212 | Speaker diversity via embeddings | `extract_speaker_embeddings`, `cluster_samples`, `SpeakerClustering`, `SpeakerClusteringConfig` |
| `features.py` | 513 | Mel spectrogram generation | `FeatureConfig`, `MicroFrontend`, `SpectrogramGeneration`, `extract_features()` |
| `augmentation.py` | 405 | Data-level audio augmentation (8 types) | `AudioAugmentation`, `AugmentationConfig`, `apply_spec_augment_gpu()` |
| `spec_augment_gpu.py` | 150 | CuPy SpecAugment (GPU-only) | `spec_augment_gpu`, `batch_spec_augment_gpu` |
| `tfdata_pipeline.py` | 364 | TF data pipeline optimization | `OptimizedDataPipeline`, `benchmark_pipeline`, `create_optimized_dataset` |
| `preprocessing.py` | 598 | Audio preprocessing pipeline | `SpeechPreprocessConfig`, `PreprocessResult`, `SplitResult` |
| `quality.py` | 660 | Audio quality scoring | `QualityScoreConfig`, `FileScore` |

## Data Flow

```
Audio (WAV) → ingestion.py → features.py → dataset.py → Training
                   ↓              ↓              ↓
            SampleRecord    Spectrogram    RaggedMmap
            (metadata)      (mel bins)     (mmap storage)
```

Augmentation happens at two levels:
1. **Waveform-level**: Before spectrogram conversion
2. **Spectrogram-level**: After spectrogram conversion (GPU SpecAugment)

## Augmentation Pipeline

- **Waveform-level augmentations**: `src/training/augmentation.py` — `AudioAugmentationPipeline` / `ParallelAugmenter`. Handles all time-domain transforms (EQ, pitch shift, background noise mixing, RIR) on raw waveform samples **before** spectrogram conversion. Primary augmentation entrypoint for training loop.
- **Data-level augmentations**: `src/data/augmentation.py` — `AudioAugmentation` class with 8 augment types: `apply_gain`, `apply_eq`, `apply_distortion`, `apply_pitch_shift`, `apply_band_stop`, `apply_color_noise`, `apply_background_noise`, `apply_rir`. Called via `__call__()`.
- **Spectrogram-level augmentations**: `src/data/spec_augment_gpu.py` — CuPy GPU time/freq masking. Applied **after** spectrogram conversion. Optional. Controlled by `augmentation.SpecAugment.*` config fields.

**Typical usage flow**:
```
Raw audio (WAV)
  → src/training/augmentation.py   # waveform-level augmentations (EQ, pitch, noise, RIR)
  → src/data/features.py           # spectrogram conversion (40 mel bins, 10ms stride, 30ms window)
  → src/data/spec_augment_gpu.py   # optional GPU spectrogram augmentations (time/freq masking)
  → Training batch
```

## Ingestion Module (`ingestion.py`)

Key types:
- `Split` enum: TRAIN, VAL, TEST
- `Label` enum: POSITIVE, NEGATIVE, HARD_NEGATIVE
- `SampleRecord` dataclass: path, label, speaker_id, split, audio metadata
- `ClipsLoaderConfig` dataclass: directory paths, split ratios
- `Clips` class: discovers samples, assigns splits, provides iterators
- `AudioValidationError`: raised for invalid audio files

Audio validation constants: VALIDATION_SAMPLE_RATE=16000, VALIDATION_SAMPLE_WIDTH=2, VALIDATION_CHANNELS=1

## Features Module (`features.py`)

- `FeatureConfig`: validates params on `__post_init__`, warns if sample_rate != 16000
- `MicroFrontend`: pymicro-features wrapper for mel spectrogram computation
- `SpectrogramGeneration`: batch processing with `generate()`, `generate_from_file()`, `process_batch()`
- Output shape: `[time_frames, 40]` — 40 mel bins, 10ms stride, 30ms window (ESPHome requirement)

### PCAN (Per-Channel Amplitude Normalization)

PCAN is **always ON** in the pymicro-features C++ backend. There is no Python flag to enable or disable it. This matches the official ESPHome okay_nabu model configuration. Do NOT add an `enable_pcan` config field — it would have no effect.

## Dataset Module (`dataset.py`)

- `RaggedMmap`: Variable-length arrays stored as memory-mapped files (data.bin + offsets.bin + lengths.bin)
- `FeatureStore`: Wraps RaggedMmap with add/get semantics for processed features
- `WakeWordDataset`: Full dataset with `train_generator_factory()` / `val_generator_factory()`
- `load_dataset()`: Convenience function to load and return ready dataset

## Preprocessing Module (`preprocessing.py`)

- `SpeechPreprocessConfig`: Configuration for preprocessing pipeline
- `PreprocessResult`: Result container for preprocessed audio
- `SplitResult`: Result container for split audio segments
- Handles VAD trimming, audio splitting, and normalization

## Quality Module (`quality.py`)

- `QualityScoreConfig`: Configuration for quality scoring
- `FileScore`: Score container for individual audio files
- Evaluates audio quality metrics (SNR, clipping, silence ratio, etc.)

## Key Patterns

- **RaggedMmap**: Variable-length audio stored as memory-mapped files (data.bin + offsets.bin + lengths.bin)
- **16kHz mono**: All audio normalized to 16-bit PCM, 16kHz, single channel
- **40 mel bins, 10ms stride, 30ms window**: Standard feature output shape [time_frames, 40] — required by ESPHome
- **Optional clustering**: SpeechBrain ECAPA-TDNN embeddings for speaker diversity (requires torch+speechbrain)
- **HAS_CUPY flag**: `augmentation.py` checks for CuPy availability at import time

## Anti-Patterns

- **Don't resample in training loop**: Preprocess to 16kHz once, store in RaggedMmap
- **Don't batch without padding**: Variable-length clips need careful collation
- **Don't use CPU augmentation**: SpecAugment has no CPU fallback, GPU is mandatory
- **Don't skip speaker clustering**: Leads to train/test leakage from same speaker

## Notes

- CuPy SpecAugment raises RuntimeError if GPU unavailable, no fallback
- Clustering requires optional deps: `pip install speechbrain torch scikit-learn`
- RaggedMmap creates `.data`, `.offsets`, `.lengths` files in storage directory
- FeatureConfig validates on init, warns if sample rate != 16000 Hz
- `load_clips()` is the main entry for audio discovery/loading


## Related Documentation

- [Training Guide](../../docs/TRAINING.md) - Complete training workflow
- [Configuration Reference](../../docs/CONFIGURATION.md) - All config options
- [Export Guide](../../docs/EXPORT.md) - TFLite export process