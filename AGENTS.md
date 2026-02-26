# microwakeword_trainer

**GPU-Accelerated Wake Word Training Framework** | v2.0.0

## Overview
TensorFlow-based wake word detection model training pipeline with GPU-accelerated SpecAugment and TFLite export for edge deployment.

## Project Structure
```
./
├── src/                    # Source code
│   ├── training/          # Training loop & CLI (mww-train)
│   ├── data/              # Dataset, augmentation, features
│   ├── model/             # Architecture definitions
│   ├── export/            # TFLite export (mww-export)
│   ├── utils/             # Performance, helpers
│   ├── evaluation/        # Metrics
│   └── config/            # Config loading
├── config/                # YAML presets & loader
│   ├── presets/           # standard.yaml, max_quality.yaml, fast_test.yaml
│   └── loader.py          # Complex config system (625 lines)
├── dataset/               # Audio data
│   ├── positive/          # Wake word samples
│   ├── negative/          # Background speech
│   ├── hard_negative/     # False positives
│   ├── background/        # Noise/ambient
│   └── rirs/              # Room impulse responses
├── models/                # Checkpoints & exports
│   └── exported/          # TFLite models
├── logs/                  # Training logs
├── profiles/              # Performance profiles
└── notebooks/             # Analysis notebooks
```

## Entry Points
| Command | Module | Purpose |
|---------|--------|---------|
| `mww-train` | `src.training.trainer:main` | Train wake word model |
| `mww-export` | `src.export.tflite:main` | Export to TFLite |

## Key Dependencies
- **tensorflow>=2.16** - Core ML framework
- **cupy-cuda12x>=13.0** - GPU SpecAugment (no CPU fallback)
- **ai-edge-litert** - TFLite export (formerly TF Lite)
- **pymicro-features** - Audio feature extraction

## Configuration System
Heavy YAML-based config with presets in `config/presets/`:
- `standard.yaml` - Balanced quality/speed
- `max_quality.yaml` - Best accuracy
- `fast_test.yaml` - Quick iteration

Loader supports env var substitution (`${VAR}`) and preset merging.

## Critical Constraints
- **GPU Required**: CuPy SpecAugment has no CPU fallback
- **CUDA 12.x**: Required for CuPy compatibility
- **Python 3.10-3.11**: ai-edge-litert 2.x does not support Python 3.12 (use 3.10 or 3.11)
- **Separate venvs for TF/PyTorch**: If using speechbrain, use different environments

## Commands
```bash
# Install
uv venv --python 3.10 ~/venvs/mww-tf
source ~/venvs/mww-tf/bin/activate
uv pip install -r requirements.txt

# Train
mww-train --config config/presets/standard.yaml

# Export
mww-export --checkpoint models/best.ckpt --output models/exported/

# With preset + override
python -c "from config.loader import load_full_config; load_full_config('standard', 'custom.yaml')"
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Training loop | `src/training/trainer.py` | Trainer class, main() |
| Data pipeline | `src/data/` | 7 modules, GPU augmentation |
| Model arch | `src/model/architecture.py` | tf.keras.Model subclasses |
| Config loading | `config/loader.py` | Dataclass-based loader |
| TFLite export | `src/export/tflite.py` | ai-edge-litert usage |
| Performance | `src/utils/performance.py` | Profiling utilities |
| Speaker clustering | `src/data/clustering.py` | WavLM embeddings, leakage audit |
| Hard negative mining | `src/data/hard_negatives.py` | FP detection, auto-mining |
| Audio augmentation | `src/data/augmentation.py` | 8 augmentation types |

## Implemented Configurations

| Config | Status | Implementation |
|--------|--------|----------------|
| PathsConfig | ✅ Complete | `src/data/ingestion.py` - Individual dirs |
| TrainingConfig | ✅ Complete | `src/training/trainer.py` |
| ModelConfig | ✅ Complete | `src/model/architecture.py` |
| AugmentationConfig | ✅ Complete | `src/data/augmentation.py` |
| PerformanceConfig | ✅ Complete | `src/training/trainer.py` |
| SpeakerClusteringConfig | ✅ Complete | `src/data/clustering.py` |
| HardNegativeMiningConfig | ✅ Complete | `src/data/hard_negatives.py` |
| ExportConfig | ✅ Complete | `src/export/manifest.py` |

## Notes
- ✅ **ALL PHASES COMPLETE** - All config variables implemented and connected
- Large config loader (625 lines) - complex validation and merging
- Uses custom RaggedMmap storage for efficient audio data loading
- Speaker clustering and hard negative mining fully implemented
- Audio augmentation pipeline complete with 8 augmentation types
## Anti-Patterns (This Project)
## Anti-Patterns (This Project)
- **Don't install nvidia-driver inside WSL** - Install on Windows host only
- **Don't mix TF and PyTorch in same venv** - Use separate environments
- **Don't use CPU-only CuPy** - SpecAugment requires GPU, no fallback
- **Don't use Python 3.12 yet** - ai-edge-litert 2.1.2 lacks support
- **Don't pin ai-edge-litert without version** - Pin to `<3.0`

## Aliases (User Configured)
```bash
alias mww-tf='source ~/venvs/mww-tf/bin/activate && cd /home/sarpel/mww/microwakeword_trainer'
alias mww-torch='source ~/venvs/mww-torch/bin/activate && cd /home/sarpel/mww/microwakeword_trainer'
```

## Development Notes
- Project is in active development (v2.0.0, Beta status)
- Large config loader (625 lines) - complex validation and merging
- Uses custom RaggedMmap storage for efficient audio data loading
- Supports speaker clustering and hard negative mining
