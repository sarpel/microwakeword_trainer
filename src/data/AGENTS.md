# src/data/

Audio data pipeline: ingestion, feature extraction, augmentation, and clustering for wake word training.

## Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `ingestion.py` | Audio loading and metadata | `SampleRecord`, `Clips`, `Split`, `Label` |
| `features.py` | Mel spectrogram generation | `FeatureConfig`, `MicroFrontend`, `SpectrogramGeneration` — **40 mel bins, 10ms stride, 30ms window** (ESPHome requirement) |
| `dataset.py` | Training dataset with mmap | `RaggedMmap`, `WakeWordDataset`, `FeatureStore` |
| `clustering.py` | Speaker diversity via embeddings | `extract_speaker_embeddings`, `cluster_samples` |
| `spec_augment_gpu.py` | CuPy SpecAugment (GPU-only) | `spec_augment_gpu`, `batch_spec_augment_gpu` |

## Data Flow

```
Audio (WAV) → ingestion.py → features.py → dataset.py → Training
                   ↓              ↓              ↓
            SampleRecord    Spectrogram    RaggedMmap
            (metadata)      (mel bins)     (mmap storage)
```

Augmentation happens after features: spectrograms → `spec_augment_gpu.py` → augmented batches.

## Augmentation Pipeline

- **Waveform-level augmentations**: `src/training/augmentation.py` — implements `AugmentationPipeline` / `augment_audio`. Handles all time-domain transforms (EQ, pitch shift, background noise mixing, RIR) on raw waveform samples **before** spectrogram conversion. This is the primary augmentation entrypoint for the training loop.
- **Spectrogram-level augmentations**: `src/data/spec_augment_gpu.py` — implements `spec_augment` / `apply_spec_augment` routines that run on GPU (via CuPy). These are applied **after** spectrogram conversion and are optional. The switch between the two is controlled by config: `augmentation.SpecAugment.*` fields trigger GPU spectrogram masking, while all other augmentation fields run via `src/training/augmentation.py`.

**Typical usage flow**:
```
Raw audio (WAV)
  → src/training/augmentation.py   # waveform-level augmentations (EQ, pitch, noise, RIR)
  → src/data/features.py           # spectrogram conversion (40 mel bins, 10ms stride, 30ms window)
  → src/data/spec_augment_gpu.py   # optional GPU spectrogram augmentations (time/freq masking)
  → Training batch
```

## Key Patterns

- **RaggedMmap**: Variable-length audio stored as memory-mapped files (data.bin + offsets.bin + lengths.bin)
- **16kHz mono**: All audio normalized to 16-bit PCM, 16kHz, single channel
- **40 mel bins, 10ms stride, 30ms window**: Standard feature output shape [time_frames, 40] — these parameters are required by ESPHome (`FeatureConfig` / `MicroFrontend` / `SpectrogramGeneration`)
- **Optional clustering**: SpeechBrain ECAPA-TDNN embeddings for speaker diversity (requires torch+speechbrain)

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
