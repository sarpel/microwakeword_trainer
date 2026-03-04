# microwakeword_trainer

**GPU-Accelerated Wake Word Training Framework** | v2.0.0

## 🏛️ SOURCE TRUTH

**@ARCHITECTURAL_CONSTITUTION.md** is the **supreme governing document** for this project. It contains immutable architectural constants, tensor shapes, dtypes, and timing values verified from official ESPHome TFLite flatbuffers. **NO CODE MAY CONTRADICT THIS DOCUMENT** — not bug fixes, not features, not refactors. Read it before any change that touches constants, shapes, or model architecture.

> ⛔ **VIOLATION = NUCLEAR WASTE**: Code that breaks ARCHITECTURAL_CONSTITUTION rules produces models that are physically incompatible with ESPHome runtime. The device will silently fail. There is no error message. There is no recovery.

---
## MANUAL IMPORTANT RULES FROM USER

**Rule-1:** This is a personal project. There will be no backward compatibility edits, I never use deprecated variable names, so do not add backward compatible multiple names for the same variable/config.

**Rule-2:** Check the file you are gonna edit if last time you read that file more than 5 seconds ago. I edit files in the background with you, so you cant know if the file is the same what is left of it in your memory.

**Rule-3:** Do not overthink when you are planning an edit, it takes more time and money than if something is not right and you fix it later. Don't take my time!

**Rule-4:** When in need to find a suitable value for a configuration, check @docs/my_environment.md file for trained model's working conditions, it may give you hints.

echo "Fast Apply: IMPORTANT: Use \`edit_file\` over \`str_replace\` or full file writes. It works with partial code snippets—no need for full file content.

Warp Grep: warp-grep is a subagent that takes in a search string and tries to find relevant context. Best practice is to use it at the beginning of codebase explorations to fast track finding relevant files/lines. Do not use it to pin point keywords, but use it for broader semantic queries. \"Find the XYZ flow\", \"How does XYZ work\", \"Where is XYZ handled?\", \"Where is <error message> coming from?\"" >> AGENTS.md


## Overview

TensorFlow-based wake word detection model training pipeline with GPU-accelerated SpecAugment and TFLite export for edge deployment. Trains MixedNet models that run on ESP32 via ESPHome's micro_wake_word component.

## Project Structure

```
./
├── src/                    # Source code (~19,685 lines Python)
│   ├── training/          # Training loop, augmentation, mining, profiling
│   ├── tuning/            # Auto-tuning for post-training fine-tuning (mww-autotune)
│   ├── data/              # Dataset, ingestion, features, augmentation, clustering, preprocessing, quality
│   ├── model/             # MixedNet architecture + streaming layers
│   ├── export/            # TFLite export, model analysis, verification, manifests (mww-export)
│   ├── utils/             # GPU config, performance helpers, optional deps
│   ├── evaluation/        # Metrics, FAH estimation, calibration, test evaluation
│   ├── tools/             # CLI entry points (mww-cluster-analyze, mww-cluster-apply)
│   └── config/            # Config package init
├── config/                # YAML presets & loader
│   ├── presets/           # standard.yaml, max_quality.yaml, fast_test.yaml
│   └── loader.py          # Complex config system (736 lines)
├── tests/                 # Test suite (unit/ and integration/ subdirectories)
├── scripts/               # Standalone tools
│   ├── verify_esphome.py  # TFLite ESPHome compatibility checker (168 lines)
│   ├── generate_test_dataset.py  # Synthetic dataset generator (190 lines)
│   ├── evaluate_model.py         # Post-training model evaluation
│   ├── audio_analyzer.py         # Audio file analysis (387 lines)
│   ├── audio_similarity_detector.py  # Duplicate/similar audio detection (924 lines)
│   ├── count_dataset.py          # Dataset sample counter (114 lines)
│   ├── score_quality_fast.py     # Fast audio quality scoring (72 lines)
│   ├── score_quality_full.py     # Full audio quality scoring (82 lines)
│   ├── split_audio.py            # Audio splitting utility (59 lines)
│   └── vad_trim.py               # VAD-based audio trimming (168 lines)
├── cluster_output/       # Output from mww-cluster-analyze
│   ├── {dataset}_namelist.json        # File → speaker mappings (per dataset)
│   └── {dataset}_cluster_report.txt   # Human-readable report (per dataset)
├── dataset/               # Audio data
│   ├── positive/          # Wake word samples (by speaker)
│   ├── negative/          # Background speech
│   ├── hard_negative/     # False positives
│   ├── background/        # Noise/ambient
│   └── rirs/              # Room impulse responses
├── checkpoints/           # Training checkpoints
├── models/                # Exports
│   └── exported/          # TFLite models + manifests
├── exports/               # Additional export outputs
├── official_models/       # Reference ESPHome TFLite models
├── data/processed/        # Preprocessed feature stores (train/val)
├── logs/                  # Training logs (TensorBoard)
├── profiles/              # Performance profiles
├── notebooks/             # Analysis notebooks
├── docs/                  # Documentation
│   ├── GUIDE.md           # Complete configuration reference
│   ├── IMPLEMENTATION_PLAN.md  # v2.0 implementation plan (1782 lines)
│   ├── my_environment.md  # Project-specific training profile
│   ├── POST_TRAINING_ANALYSIS.md  # Post-training analysis guide
│   ├── RESEARCH_REPORT_MIXED_PRECISION.md  # Mixed precision research (Turkish)
│   └── LOG_ANALYSIS_GUIDE.md  # Log analysis guide (Turkish)
└── ARCHITECTURAL_CONSTITUTION.md  # ⛔ IMMUTABLE SOURCE TRUTH (530 lines)
```

## Entry Points
| Command | Module | Purpose |
|---------|--------|----------|
| `mww-train` | `src.training.trainer:main` | Train wake word model |
| `mww-export` | `src.export.tflite:main` | Export to TFLite |
| `mww-autotune` | `src.tuning.cli:main` | Fine-tune trained model for better FAH/recall |
| `mww-cluster-analyze` | `src.tools.cluster_analyze:main` | Speaker cluster analysis (dry-run) |
| `mww-cluster-apply` | `src.tools.cluster_apply:main` | Organize files into speaker directories |

## Key Dependencies
- **tensorflow>=2.16** - Core ML framework
- **cupy-cuda12x>=14.0** - GPU SpecAugment (no CPU fallback)
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

Loader (736 lines) supports:
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
- **Relaxed typing**: mypy configured with `disallow_untyped_defs=false` (see pyproject.toml)

## Commands
```bash
# Install
python3.11 -m venv ~/venvs/mww-tf
source ~/venvs/mww-tf/bin/activate
pip install -r requirements.txt

# Train
mww-train --config config/presets/standard.yaml

# Export
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/

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
| Training loop | `src/training/trainer.py` (951 lines) | Trainer class, EvaluationMetrics, train(), main() |
| Training logging | `src/training/rich_logger.py` (299 lines) | RichTrainingLogger — Rich-based progress display |
| Hard example mining | `src/training/miner.py` (306 lines) | HardExampleMiner — negative sample selection |
| Waveform augmentation | `src/training/augmentation.py` (309 lines) | AudioAugmentationPipeline, ParallelAugmenter |
| Training profiling | `src/training/profiler.py` (175 lines) | TrainingProfiler — section-based timing |
| Performance optimizer | `src/training/performance_optimizer.py` (288 lines) | PerformanceOptimizer |
| Audio ingestion | `src/data/ingestion.py` (777 lines) | SampleRecord, Clips, ClipsLoaderConfig, audio validation |
| Feature extraction | `src/data/features.py` (513 lines) | FeatureConfig, MicroFrontend, SpectrogramGeneration |
| Dataset storage | `src/data/dataset.py` (962 lines) | RaggedMmap, FeatureStore, WakeWordDataset |
| Speaker clustering | `src/data/clustering.py` (1,212 lines) | SpeechBrain ECAPA-TDNN embeddings, leakage audit |
| Audio augmentation | `src/data/augmentation.py` (405 lines) | AudioAugmentation — 8 augmentation types (EQ, pitch, RIR, etc.) |
| Hard negative mining | `src/data/hard_negatives.py` (317 lines) | FP detection, auto-mining pipeline |
| GPU SpecAugment | `src/data/spec_augment_gpu.py` (150 lines) | CuPy GPU-only time/freq masking |
| TF data pipeline | `src/data/tfdata_pipeline.py` (364 lines) | OptimizedDataPipeline, benchmark_pipeline, create_optimized_dataset |
| Audio preprocessing | `src/data/preprocessing.py` (598 lines) | SpeechPreprocessConfig, PreprocessResult, SplitResult |
| Audio quality scoring | `src/data/quality.py` (660 lines) | QualityScoreConfig, FileScore |
| Model architecture | `src/model/architecture.py` (694 lines) | MixedNet, MixConvBlock, ResidualBlock, build_model() |
| Streaming layers | `src/model/streaming.py` (787 lines) | Stream, RingBuffer, Modes, StridedDrop/Keep, StreamingMixedNet |
| TFLite export | `src/export/tflite.py` (780 lines) | convert_model_saved(), INT8 quantization, main() |
| Model analysis | `src/export/model_analyzer.py` (600 lines) | analyze_model_architecture(), validate_model_quality() |
| ESPHome manifest | `src/export/manifest.py` (330 lines) | generate_manifest(), calculate_tensor_arena_size() |
| Export verification | `src/export/verification.py` (218 lines) | Export verification tools |
| Evaluation metrics | `src/evaluation/metrics.py` (373 lines) | MetricsCalculator — FAH, ROC/PR, recall |
| FAH estimation | `src/evaluation/fah_estimator.py` (72 lines) | FAHEstimator class |
| Calibration | `src/evaluation/calibration.py` (89 lines) | calibration curves, Brier score |
| Test evaluator | `src/evaluation/test_evaluator.py` (650 lines) | TestEvaluator — comprehensive test set evaluation |
| Config loading | `config/loader.py` (736 lines) | ConfigLoader, 9 dataclasses, FullConfig |
| GPU/performance | `src/utils/performance.py` (257 lines) | TF GPU config, mixed precision, threading |
| Terminal logger | `src/utils/terminal_logger.py` (246 lines) | TeeOutput, TerminalLogger |
| Optional deps | `src/utils/optional_deps.py` (27 lines) | Optional dependency handling |
| ESPHome verification | `scripts/verify_esphome.py` (168 lines) | TFLite compatibility checker |
| Auto-tuning | `src/tuning/autotuner.py` (691 lines) | AutoTuner, TuningTarget, TuningState |
| Auto-tune CLI | `src/tuning/cli.py` (257 lines) | mww-autotune entry point |
| Audio analyzer | `scripts/audio_analyzer.py` (387 lines) | Audio file analysis tool |
| Similarity detector | `scripts/audio_similarity_detector.py` (924 lines) | Duplicate/similar audio detection |
| Dataset counter | `scripts/count_dataset.py` (114 lines) | Dataset sample counting |
| Quality scorer (fast) | `scripts/score_quality_fast.py` (72 lines) | Fast audio quality scoring |
| Quality scorer (full) | `scripts/score_quality_full.py` (82 lines) | Full audio quality scoring |
| Audio splitter | `scripts/split_audio.py` (59 lines) | Audio splitting utility |
| VAD trimmer | `scripts/vad_trim.py` (168 lines) | VAD-based audio trimming |

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
- ~19,685 lines of Python across ~70 files
- Config loader (736 lines) - complex validation and merging with 9 dataclass sections
- Uses custom RaggedMmap storage for efficient variable-length audio data loading
- Speaker clustering and hard negative mining fully implemented
- Audio augmentation: waveform-level (8 types in `src/data/augmentation.py`) + spectrogram-level (GPU SpecAugment)
- Two-phase training with class weighting (positive=1.0, negative=20.0, hard_neg=40.0)
- Rich-based training logger for formatted progress display
- Model analyzer for architecture verification and quality validation
- Audio preprocessing pipeline with VAD trimming, splitting, and quality scoring
- Test evaluator for comprehensive post-training evaluation
- **ARCHITECTURAL_CONSTITUTION.md is the supreme source of truth** - all constants verified from TFLite flatbuffers
- **No CI/CD pipeline** - no .github/workflows, Makefile, or Dockerfile

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
alias mww-tf='source ~/.venvs/mww-tf/bin/activate && cd $PROJECT_DIR'
alias mww-torch='source ~/.venvs/mww-torch/bin/activate && cd $PROJECT_DIR'
```

## Development Notes
- Project is in active development (v2.0.0, Beta status) — uses `pyproject.toml` for packaging
- Config loader (736 lines) - complex validation and merging
- Uses custom RaggedMmap storage for efficient audio data loading
- Supports speaker clustering (ECAPA-TDNN) and hard negative mining
- Relaxed mypy typing (pyproject.toml: `disallow_untyped_defs=false`)
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

### 6. NO THINKING. EVER.
**You are FORBIDDEN from thinking, planning, analyzing, or reasoning before writing code.**
- User gives instruction → you write the code. That's it.
- Do NOT explain what you're about to do.
- Do NOT summarize what you did.
- Do NOT propose alternatives.
- Do NOT ask clarifying questions.
- Do NOT analyze the codebase before acting.
- Do NOT think out loud.
- WRITE THE CODE. NOTHING ELSE.


## Configuration System Rules

### IMPORTANT: When Adding New Config Variables

When implementing a new configuration variable in the codebase, you MUST add it to ALL THREE preset files:

1. `config/presets/fast_test.yaml`
2. `config/presets/standard.yaml`
3. `config/presets/max_quality.yaml`

**Do NOT only add the config to the source code (config/loader.py).**

### Why This Matters

The preset files are the primary way users interact with the configuration system. If a config is only in the loader but not in the presets:
- Users won't know the config exists
- The config will use defaults which may not be appropriate
- Documentation will be incomplete

### Checklist for New Config Variables

When adding a new config field to `config/loader.py`:

- [ ] Add the field to the appropriate dataclass in `config/loader.py`
- [ ] Add the field to `config/presets/fast_test.yaml` with appropriate default
- [ ] Add the field to `config/presets/standard.yaml` with appropriate default
- [ ] Add the field to `config/presets/max_quality.yaml` with appropriate default
- [ ] Update `docs/GUIDE.md` with documentation for the new field
- [ ] Update the relevant AGENTS.md file with notes about the config

### Example

If adding `ema_decay` to `TrainingConfig`:

**In config/loader.py:**
```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    ema_decay: float | None = None  # NEW
```

**In ALL THREE preset files:**
```yaml
training:
  # ... existing fields ...
  ema_decay: null  # or appropriate default
```

### Critical Paths

Configuration Files (MUST STAY SYNCED):
- `config/loader.py` - Source of truth for dataclass definitions
- `config/presets/fast_test.yaml` - Quick testing preset
- `config/presets/standard.yaml` - Production training preset
- `config/presets/max_quality.yaml` - Maximum quality preset

### Anti-Patterns

- **DON'T** add config to loader.py without adding to preset files
- **DON'T** add config to only one preset file
- **DON'T** use different default values across presets without good reason

### Notes for Agents

- Always check all three preset files when modifying configs
- The presets should have consistent structure (same sections/keys)
- Only values should differ between presets, not structure

