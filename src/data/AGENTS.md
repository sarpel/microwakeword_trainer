# src/data/

Audio data pipeline: ingestion, feature extraction, augmentation, and clustering for wake word training.

## Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `ingestion.py` | Audio loading and metadata | `SampleRecord`, `Clips`, `Split`, `Label` |
| `features.py` | Mel spectrogram generation | `FeatureConfig`, `MicroFrontend`, `SpectrogramGeneration` |
| `dataset.py` | Training dataset with mmap | `RaggedMmap`, `WakeWordDataset`, `FeatureStore` |
| `clustering.py` | Speaker diversity via embeddings | `extract_wavlm_embeddings`, `cluster_samples` |
| `spec_augment_gpu.py` | CuPy SpecAugment (GPU-only) | `spec_augment_gpu`, `batch_spec_augment_gpu` |

## Data Flow

```
Audio (WAV) → ingestion.py → features.py → dataset.py → Training
                   ↓              ↓              ↓
            SampleRecord    Spectrogram    RaggedMmap
            (metadata)      (mel bins)     (mmap storage)
```

Augmentation happens after features: spectrograms → `spec_augment_gpu.py` → augmented batches.

## Key Patterns

- **RaggedMmap**: Variable-length audio stored as memory-mapped files (data.bin + offsets.bin + lengths.bin)
- **16kHz mono**: All audio normalized to 16-bit PCM, 16kHz, single channel
- **40 mel bins**: Standard feature output shape [time_frames, 40]
- **Optional clustering**: WavLM embeddings for speaker diversity (requires torch+transformers)

## Anti-Patterns

- **Don't resample in training loop**: Preprocess to 16kHz once, store in RaggedMmap
- **Don't batch without padding**: Variable-length clips need careful collation
- **Don't use CPU augmentation**: SpecAugment has no CPU fallback, GPU is mandatory
- **Don't skip speaker clustering**: Leads to train/test leakage from same speaker

## Notes

- CuPy SpecAugment raises RuntimeError if GPU unavailable, no fallback
- Clustering requires optional deps: `pip install transformers torch scikit-learn`
- RaggedMmap creates `.data`, `.offsets`, `.lengths` files in storage directory
- FeatureConfig validates on init, warns if sample rate != 16000 Hz
