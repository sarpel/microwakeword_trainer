# microwakeword_trainer

**GPU-Accelerated Wake Word Training Framework** | v2.0.0

## 🏛️ SOURCE TRUTH

**@ARCHITECTURAL_CONSTITUTION.md** is the **supreme governing document** for this project. It contains immutable architectural constants, tensor shapes, dtypes, and timing values verified from official ESPHome TFLite flatbuffers. **NO CODE MAY CONTRADICT THIS DOCUMENT** — not bug fixes, not features, not refactors. Read it before any change that touches constants, shapes, or model architecture.

> ⛔ **VIOLATION = NUCLEAR WASTE**: Code that breaks ARCHITECTURAL_CONSTITUTION rules produces models that are physically incompatible with ESPHome runtime. The device will silently fail. There is no error message. There is no recovery.

---

## Overview

TensorFlow-based wake word detection model training pipeline with GPU-accelerated SpecAugment and TFLite export for edge deployment. Trains MixedNet models that run on ESP32 via ESPHome's micro_wake_word component.

## Project Structure
```
./
├── src/                    # Source code (~11,000 lines Python)
│   ├── training/          # Training loop, logging, mining, augmentation (mww-train)
│   ├── data/              # Dataset, ingestion, features, augmentation, clustering
│   ├── model/             # MixedNet architecture + streaming layers
│   ├── export/            # TFLite export, model analysis, manifests (mww-export)
│   ├── utils/             # GPU config, performance helpers
│   ├── evaluation/        # Metrics, FAH estimation, calibration
│   └── config/            # Package init (loader lives in config/)
├── config/                # YAML presets & loader
│   ├── presets/           # standard.yaml, max_quality.yaml, fast_test.yaml
│   └── loader.py          # Complex config system (666 lines)
├── scripts/               # Standalone tools
│   ├── verify_esphome.py  # TFLite ESPHome compatibility checker (406 lines)
│   └── generate_test_dataset.py  # Synthetic dataset generator (190 lines)
├── cluster-Test.py        # Speaker clustering dry-run analysis (458 lines)
├── cluster_output/       # Output from cluster-Test.py
│   ├── {dataset}_namelist.json     # File → speaker mappings (per dataset)
│   └── {dataset}_cluster_report.txt # Human-readable report (per dataset)
├── dataset/               # Audio data
│   ├── positive/          # Wake word samples (by speaker)
│   ├── negative/          # Background speech
│   ├── hard_negative/     # False positives
│   ├── background/        # Noise/ambient
│   └── rirs/              # Room impulse responses
├── checkpoints/           # Training checkpoints
├── models/                # Exports
│   └── exported/          # TFLite models + manifests
├── data/processed/        # Preprocessed feature stores (train/val)
├── logs/                  # Training logs (TensorBoard)
├── profiles/              # Performance profiles
├── notebooks/             # Analysis notebooks
└── ARCHITECTURAL_CONSTITUTION.md  # ⛔ IMMUTABLE SOURCE TRUTH (530 lines)
```

## Entry Points
| Command | Module | Purpose |
|---------|--------|----------|
| `mww-train` | `src.training.trainer:main` | Train wake word model |
| `mww-export` | `src.export.tflite:main` | Export to TFLite |
| `cluster-Test.py` | Speaker clustering analysis (dry-run, supports positive/negative/hard_negative/all) | |
| `Start-Clustering.py` | Move files into speaker directories (uses cluster-Test.py output) | |

## Key Dependencies
- **tensorflow>=2.16** - Core ML framework
- **cupy-cuda12x>=13.0** - GPU SpecAugment (no CPU fallback)
- **ai-edge-litert** - TFLite export (formerly TF Lite)
- **pymicro-features** - Audio feature extraction (40 mel bins, ESPHome-compatible)
- **rich** - Training progress display (RichTrainingLogger)
- **optuna** - Hyperparameter optimization (optional)
- **tensorboard** - Training visualization

## Configuration System
Heavy YAML-based config with presets in `config/presets/`:
- `standard.yaml` - Balanced quality/speed
- `max_quality.yaml` - Best accuracy
- `fast_test.yaml` - Quick iteration

Loader (666 lines) supports:
- 9 dataclass sections: Hardware, Paths, Training, Model, Augmentation, Performance, SpeakerClustering, HardNegativeMining, Export
- Env var substitution (`${VAR}` or `${VAR:-default}`)
- Preset merging with custom overrides
- Path resolution relative to project root

## Critical Constraints
- **GPU Required**: CuPy SpecAugment has no CPU fallback
- **CUDA 12.x**: Required for CuPy compatibility
- **Python 3.10-3.11**: ai-edge-litert 2.x does not support Python 3.12 (use 3.10 or 3.11)
- **Separate venvs for TF/PyTorch**: If using speechbrain, use different environments
- **ARCHITECTURAL_CONSTITUTION.md is immutable**: No exceptions, no overrides, no "quick tweaks"
- **No test infrastructure**: No pytest/unittest; use `scripts/verify_esphome.py` for export validation
- **Strict typing**: mypy.ini enforces `disallow_untyped_defs=True`, `no_implicit_optional=True`

## Commands
```bash
# Install
uv venv --python 3.10 ~/venvs/mww-tf
source ~/venvs/mww-tf/bin/activate
uv pip install -r requirements.txt

# Train
mww-train --config config/presets/standard.yaml

# Export
mww-export --checkpoint checkpoints/best.ckpt --output models/exported/

# Verify ESPHome compatibility
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Generate synthetic test dataset
python scripts/generate_test_dataset.py

# With preset + override
python -c "from config.loader import load_full_config; load_full_config('standard', 'custom.yaml')"
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Training loop | `src/training/trainer.py` (874 lines) | Trainer class, EvaluationMetrics, train(), main() |
| Training logging | `src/training/rich_logger.py` (312 lines) | RichTrainingLogger — Rich-based progress display |
| Hard example mining | `src/training/miner.py` (305 lines) | HardExampleMiner — negative sample selection |
| Waveform augmentation | `src/training/augmentation.py` (266 lines) | AudioAugmentationPipeline, ParallelAugmenter |
| Training profiling | `src/training/profiler.py` (176 lines) | TrainingProfiler — section-based timing |
| Audio ingestion | `src/data/ingestion.py` (734 lines) | SampleRecord, Clips, ClipsLoaderConfig, audio validation |
| Feature extraction | `src/data/features.py` (525 lines) | FeatureConfig, MicroFrontend, SpectrogramGeneration |
| Dataset storage | `src/data/dataset.py` (831 lines) | RaggedMmap, FeatureStore, WakeWordDataset |
| Speaker clustering | `src/data/clustering.py` (595 lines) | SpeechBrain ECAPA-TDNN embeddings, leakage audit |
| Audio augmentation | `src/data/augmentation.py` (437 lines) | AudioAugmentation — 8 augmentation types (EQ, pitch, RIR, etc.) |
| Hard negative mining | `src/data/hard_negatives.py` (328 lines) | FP detection, auto-mining pipeline |
| GPU SpecAugment | `src/data/spec_augment_gpu.py` (148 lines) | CuPy GPU-only time/freq masking |
| Model architecture | `src/model/architecture.py` (757 lines) | MixedNet, MixConvBlock, ResidualBlock, build_model() |
| Streaming layers | `src/model/streaming.py` (831 lines) | Stream, RingBuffer, Modes, StridedDrop/Keep, StreamingMixedNet |
| TFLite export | `src/export/tflite.py` (817 lines) | convert_model_saved(), INT8 quantization, main() |
| Model analysis | `src/export/model_analyzer.py` (568 lines) | analyze_model_architecture(), validate_model_quality() |
| ESPHome manifest | `src/export/manifest.py` (327 lines) | generate_manifest(), calculate_tensor_arena_size() |
| Evaluation metrics | `src/evaluation/metrics.py` (397 lines) | MetricsCalculator — FAH, ROC/PR, recall |
| FAH estimation | `src/evaluation/fah_estimator.py` (74 lines) | FAHEstimator class |
| Calibration | `src/evaluation/calibration.py` (94 lines) | calibration curves, Brier score |
| Config loading | `config/loader.py` (666 lines) | ConfigLoader, 9 dataclasses, FullConfig |
| GPU/performance | `src/utils/performance.py` (246 lines) | TF GPU config, mixed precision, threading |
| ESPHome verification | `scripts/verify_esphome.py` (406 lines) | TFLite compatibility checker |

## Implemented Configurations

| Config | Status | Implementation |
|--------|--------|----------------|
| PathsConfig | ✅ Complete | `src/data/ingestion.py` - Individual dirs |
| TrainingConfig | ✅ Complete | `src/training/trainer.py` |
| ModelConfig | ✅ Complete | `src/model/architecture.py` |
| AugmentationConfig | ✅ Complete | `src/data/augmentation.py` + `src/training/augmentation.py` |
| PerformanceConfig | ✅ Complete | `src/training/trainer.py` + `src/utils/performance.py` |
| SpeakerClusteringConfig | ✅ Complete | `src/data/clustering.py` |
| HardNegativeMiningConfig | ✅ Complete | `src/data/hard_negatives.py` + `src/training/miner.py` |
| ExportConfig | ✅ Complete | `src/export/manifest.py` + `src/export/tflite.py` |

## Notes
- ✅ **ALL PHASES COMPLETE** - All config variables implemented and connected
- ~11,053 lines of Python across ~30 files
- Config loader (666 lines) - complex validation and merging with 9 dataclass sections
- Uses custom RaggedMmap storage for efficient variable-length audio data loading
- Speaker clustering and hard negative mining fully implemented
- Audio augmentation: waveform-level (8 types in `src/data/augmentation.py`) + spectrogram-level (GPU SpecAugment)
- Two-phase training with class weighting (positive=1.0, negative=20.0, hard_neg=40.0)
- Rich-based training logger for formatted progress display
- Model analyzer for architecture verification and quality validation
- **ARCHITECTURAL_CONSTITUTION.md is the supreme source of truth** - all constants verified from TFLite flatbuffers
- **No CI/CD pipeline** - no .github/workflows, Makefile, or Dockerfile
- **No test suite** - no pytest/unittest infrastructure; validation via scripts/

## Anti-Patterns (This Project)
- **Don't install nvidia-driver inside WSL** - Install on Windows host only
- **Don't mix TF and PyTorch in same venv** - Use separate environments
- **Don't use CPU-only CuPy** - SpecAugment requires GPU, no fallback
- **Don't use Python 3.12 yet** - ai-edge-litert 2.1.2 lacks support
- **Don't pin ai-edge-litert without version** - Pin to `<3.0`
- **Don't contradict ARCHITECTURAL_CONSTITUTION.md** - Not even "small tweaks" to constants
- **Don't use `model.export()`** - Fails with ring buffer states; use `tf.keras.export.ExportArchive`
- **Don't use int8 output dtype** - ESPHome requires uint8; model silently broken on device

## Aliases (User Configured)
```bash
alias mww-tf='source ~/venvs/mww-tf/bin/activate && cd /home/sarpel/mww/microwakeword_trainer'
alias mww-torch='source ~/venvs/mww-torch/bin/activate && cd /home/sarpel/mww/microwakeword_trainer'
```

## Development Notes
- Project is in active development (v2.0.0, Beta status)
- Branch: `v1.3.0` — setuptools packaging via `setup.py` (no pyproject.toml)
- Config loader (666 lines) - complex validation and merging
- Uses custom RaggedMmap storage for efficient audio data loading
- Supports speaker clustering (ECAPA-TDNN) and hard negative mining
- Strict mypy typing enforced (mypy.ini: `disallow_untyped_defs=True`)
- **When in doubt, re-read ARCHITECTURAL_CONSTITUTION.md from the top**

---

## 🤖 AI Agent Editing Rules (For Automated Tools)

When modifying any file in this project, AI agents MUST obey these rules:

### 1. The 5-Second Rule
**Re-read the file immediately before editing.** Never use LINE#IDs older than 5 seconds. Tags are volatile fingerprints, not stable coordinates.

### 2. One Edit Per File
**Batch ALL changes to a single file into ONE edit() call.** No sequential edits to the same file. If you need to edit a file twice, you failed rule #1.

### 3. Hash Mismatch Protocol
If you get a hash mismatch:
1. STOP immediately
2. RE-READ the file to get fresh LINE#IDs
3. Re-build your edits with the new tags
4. Try again

### 4. No Guessing
Never guess LINE#IDs, line numbers, or tags. Always use the exact tags from the most recent read.

### 5. ARCHITECTURAL_CONSTITUTION.md Check
Before any change touching constants, shapes, dtypes, or timing:
1. Re-read ARCHITECTURAL_CONSTITUTION.md
2. Verify your change doesn't contradict any Article
3. If in doubt, the change is wrong

**Failure to follow these rules causes file corruption. No exceptions.**
