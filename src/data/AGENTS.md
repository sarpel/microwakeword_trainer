# src/data/

Audio data pipeline: ingestion, feature extraction, augmentation, dataset storage, clustering, preprocessing, and quality scoring.

## Files

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `dataset.py` | 962 | Training dataset with mmap storage | `RaggedMmap`, `FeatureStore`, `WakeWordDataset` |
| `ingestion.py` | 777 | Audio loading, validation, metadata | `SampleRecord`, `Clips`, `load_clips()` |
| `clustering.py` | 1,212 | Speaker diversity via embeddings | `SpeakerClustering`, `extract_speaker_embeddings` |
| `features.py` | 513 | Mel spectrogram generation | `FeatureConfig`, `MicroFrontend`, `extract_features()` |
| `augmentation.py` | 405 | Data-level audio augmentation | `AudioAugmentation` |
| `spec_augment_gpu.py` | 150 | CuPy SpecAugment (GPU-only) | `spec_augment_gpu` |
| `tfdata_pipeline.py` | 364 | TF data pipeline optimization | `OptimizedDataPipeline` |
| `preprocessing.py` | 598 | Audio preprocessing pipeline | `SpeechPreprocessConfig` |
| `quality.py` | 660 | Audio quality scoring | `QualityScoreConfig`, `FileScore` |

## Data Flow

```
Audio (WAV) → ingestion.py → features.py → dataset.py → Training
                   ↓              ↓              ↓
            SampleRecord    Spectrogram    RaggedMmap
```

## Augmentation Pipeline

- **Waveform-level**: `src/training/augmentation.py` — `AudioAugmentationPipeline` (EQ, pitch, noise, RIR)
- **Spectrogram-level**: `src/data/spec_augment_gpu.py` — CuPy GPU time/freq masking

## Key Patterns

- **RaggedMmap**: Variable-length audio stored as memory-mapped files
- **16kHz mono**: All audio normalized to 16-bit PCM, 16kHz, single channel
- **40 mel bins, 10ms stride, 30ms window**: Standard feature output shape [time_frames, 40]
- **PCAN always ON**: Enabled in pymicro-features C++ backend (no Python flag)
- **HAS_CUPY flag**: Checked at import time for GPU availability

## Anti-Patterns

- **Don't resample in training loop**: Preprocess to 16kHz once, store in RaggedMmap
- **Don't batch without padding**: Variable-length clips need careful collation
- **Don't use CPU augmentation**: SpecAugment has no CPU fallback
- **Don't skip speaker clustering**: Leads to train/test leakage

## Related Documentation

- [Training Guide](../../docs/TRAINING.md)
- [Configuration Reference](../../docs/CONFIGURATION.md)
- [Export Guide](../../docs/EXPORT.md)
