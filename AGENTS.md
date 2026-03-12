# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-08
**Commit:** consolidation
**Project:** microwakeword_trainer v2.0.0

## OVERVIEW
GPU-accelerated wake word training framework for ESPHome. TensorFlow-based pipeline with MixedNet architecture, CuPy GPU SpecAugment, TFLite INT8 export.

## STRUCTURE
```
./
├── src/                  # Source code (8 modules, ~19,685 lines Python)
├── config/                # YAML presets + Python loader (dual structure)
├── tests/                 # Unit + integration tests
├── scripts/               # Standalone tools (~10 utilities)
├── docs/                  # User documentation
├── specs/                 # Implementation specs & status (NEW)
├── ARCHITECTURAL_CONSTITUTION.md  # Immutable architectural truth
└── AGENTS.md             # This file
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Training loop | `src/training/trainer.py` (951 lines) | Two-phase, hard negative mining |
| Mining & FP extraction | `src/training/mining.py` (1859 lines) | Unified: HardExampleMiner, AsyncMiner, FP logging, top-FP extraction, consolidation |
| Config system | `config/loader.py` (736 lines) | 14 dataclasses, env var substitution |
| Model architecture | `src/model/architecture.py` (694 lines) | MixedNet, streaming layers, `build_core_layers()` factory |
| TFLite export | `src/export/tflite.py` (780 lines) | INT8 quantization, dual subgraphs |
| Data pipeline | `src/data/dataset.py` (962 lines) | RaggedMmap, WakeWordDataset |
| Speaker clustering | `src/data/clustering.py` (1,212 lines) | ECAPA-TDNN embeddings |
| Auto-tuning | `src/tuning/autotuner.py` (2333 lines) | MaxQualityAutoTuner: Pareto archive, Thompson sampling, 7 strategy arms |
| Evaluation metrics | `src/evaluation/metrics.py` (373 lines) | FAH, ROC-AUC, calibration |

## CONVENTIONS

### Configuration System
- **14 dataclass sections** in loader.py (Hardware, Paths, Training, Model, Augmentation, Performance, SpeakerClustering, Mining, Export, Preprocessing, Quality, Evaluation)
- **Three presets**: fast_test, standard, max_quality
- **Environment variable substitution**: `${VAR}` or `${VAR:-default}`
- **Immutable hardware section**: Enforced by ARCHITECTURAL_CONSTITUTION.md
- **Two separate venvs**: TF (main) + PyTorch (clustering)

### Code Style
- **Relaxed typing**: mypy `disallow_untyped_defs=false`
- **Line count**: 70 files with ~19,685 lines total
- **Large files**: 18 files >500 lines (cluster.py, dataset.py, etc.)
- **Directory structure**: src/ layout despite root __init__.py

### Testing
- **Unit tests**: 5 modules (async_miner, config, test_evaluator, vectorized_metrics, spec_augment)
- **Integration tests**: 1 module (training pipeline)
- **No CI/CD**: Manual testing, Makefile for automation

## ANTI-PATTERNS (THIS PROJECT)

### Critical Violations (Silent Failure on Device)
- **⛔ Don't contradict ARCHITECTURAL_CONSTITUTION.md** - Silent device failure
- **⛔ Don't use int8 output dtype** - ESPHome requires uint8; model silently broken
- **⛔ Don't use `model.export()`** - Fails with ring buffer states; use `tf.keras.export.ExportArchive`
- **⛔ Don't modify immutable constants** - No exceptions, no "quick tweaks"
- **⛔ Don't use `model.trainable_weights` for serialization** - Excludes BatchNorm moving statistics (non-trainable). Use `model.get_weights()`/`model.set_weights()` for full state. See `src/tuning/AGENTS.md` for details.


### Environment & Dependencies
- **⛔ Don't install nvidia-driver inside WSL** - Install on Windows host only
- **⛔ Don't mix TF and PyTorch in same venv** - Use separate environments (mww-tf, mww-torch)
- **⛔ Don't use CPU-only CuPy** - SpecAugment requires GPU, no fallback
- **⛔ Don't import scripts as a module** - scripts/ has no __init__.py. Run as `python scripts/<script>.py` or use subprocess
  - Exception: `trainer.py` line 1614 wraps optional import in try/except for top FP extraction
### Configuration Rules
- **⛔ Don't add config only to loader.py** - Must add to ALL THREE presets (fast_test, standard, max_quality)
- **⛔ Don't use deprecated variable names** - No backward compatibility (Rule-1)

### Editing Rules (MANDATORY)
- **⛔ Don't use LINE#IDs older than 10 seconds** - Re-read file immediately before editing
- **⛔ Don't make sequential edits to same file** - Batch ALL changes into ONE edit() call
- **⛔ Don't guess LINE#IDs** - Always use exact tags from most recent read

## UNIQUE STYLES

### Data Management
- **RaggedMmap**: Custom memory-mapped storage for variable-length audio
- **Separate data directories**: dataset/ (raw) vs data/processed/ (features)
- **Quality scoring**: Pre-filtering pipeline (SNR, clipping, WQI)

### PCAN (Per-Channel Normalization)
- PCAN is always ON in the pymicro-features C++ backend; there is no Python flag to enable or disable.
- This matches the ESPHome okay_nabu model configuration.

### Training Pipeline
- **AsyncHardExampleMiner**: Background hard negative mining (no training interruption)
- **Two-phase training**: Phase 1 (feature learning) + Phase 2 (fine-tuning)
- **Class weighting**: positive=1.0, negative=20.0, hard_neg=40.0
- **Rich-based logger**: RichTrainingLogger for formatted progress

### Export System
- **Dual subgraphs**: Main inference + initialization
- **State variables**: 6 ring buffers named stream_0 through stream_5 (int8-quantized)
- **BatchNorm folding**: Critical for streaming export
- **Representative dataset**: Required for INT8 quantization calibration

### CLI Organization
- **Console scripts**: Defined in pyproject.toml (mww-train, mww-export, mww-autotune, mww-cluster-analyze, mww-cluster-apply)
- **Standalone scripts**: 14 utilities in scripts/ directory
- **Makefile targets**: 20+ targets for common workflows

## COMMANDS
```bash
# Training
mww-train --config config/presets/standard.yaml

# Export
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/

# Auto-tune
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard --users-hard-negs /path/to/custom_hard_negatives/

# Speaker clustering (PyTorch env)
mww-cluster-analyze --config standard --dataset all --n-clusters 200
mww-cluster-apply --namelist-dir cluster_output

# Verification
python scripts/verify_esphome.py models/exported/wake_word.tflite
```

## NOTES

### Critical Files
- **ARCHITECTURAL_CONSTITUTION.md**: Supreme source of truth - read before any architectural changes
- **AGENTS.md** (root): This file - main project patterns
- **src/*/AGENTS.md**: Per-module patterns and anti-patterns

### Recent Enhancements
- **User-defined hard negatives in AutoTuner** (commit 2fa00e22e): `--users-hard-negs` CLI flag, `users_hard_negs_dir` parameter
- **Configuration validation enhancements** (commit f62bb69a3): Improved validation in config/loader.py, better error messages
- **Critical Bug Fix** (2026-03-10): Auto-tuner weight serialization now uses `model.get_weights()`/`model.set_weights()` instead of `model.trainable_weights` to include BatchNorm moving statistics. The tuning model runs in NON_STREAM mode (no streaming states), but BatchNorm moving_mean/moving_variance are non-trainable and were lost. Prevents confirmation failures (FAH 0→129).
- **Two-Stage Checkpoint Strategy** (2026-03-12): Replaced `quality_score`-based checkpoint selection with a principled two-stage approach. Stage 1 (warm-up): saves by PR-AUC (`auc_pr`) until FAH budget is first met. Stage 2 (operational): saves by `recall_at_target_fah` when FAH ≤ `target_fah × 1.1`. `quality_score` is retained for logging/display only. Eliminates arbitrary 0.7/0.3 weights, Lorentzian config-sensitivity, and AGENTS.md/code contradictions. See `src/training/trainer.py::_is_best_model()`.
- **Ground Truth Audit** (2026-03-12): Verified all documentation against `official_okay_nabu_analysis.txt`. Key corrections: 95 tensors in Subgraph 0 (not 94), 13 unique op types used by okay_nabu, 20 registered op resolvers in ESPHome (not 14), MUL/ADD ops registered but unused. Fixed in ARCHITECTURAL_CONSTITUTION.md and docs/ARCHITECTURE.md.
- **Pipeline Alignment** (2026-03-12): `build_core_layers()` factory extracted to architecture.py — both `MixedNet` and `StreamingExportModel` use it. `convert_to_tflite()` dead code removed from tflite.py. INT8 shadow evaluation removed from AutoTuner (float32 eval only). PCAN hardcoded ON in pymicro-features C++ backend (no Python flag). All residual_connections defaults aligned to `[0,1,1,1]`.

### Gotchas
- **CuPy no CPU fallback**: Must have GPU for training (SpecAugment)
- **uint8 output mandatory**: ESPHome rejects int8 outputs
- **ExportArchive required**: `model.export()` fails with streaming state variables
- **BatchNorm state in serialization**: BatchNorm moving_mean/moving_variance are NON-TRAINABLE. Use `model.get_weights()`/`model.set_weights()` for serialization, NOT `model.trainable_weights`
- **Immutable constants**: Never override ARCHITECTURAL_CONSTITUTION.md values
- **Old checkpoint incompatibility**: Checkpoints trained before the Flatten architecture fix (2026-03-11) have Dense layer shape `(64, 1)` and are incompatible with the current export pipeline which expects `(temporal_frames × 64, 1)`. Must retrain with current code.
- **temporal_frames inference**: Export pipeline infers `temporal_frames = dense_input_features // 64` from checkpoint Dense kernel shape. Dense layer input size = `temporal_frames × 64` (64 = last pointwise filter count).
- **State variable naming**: Variables are named `stream_0` through `stream_5` (not `stream`, `stream_1`...). The `_0` suffix ensures correct alphabetical ordering in TFLite flatbuffer.
- **convert_to_tflite() removed**: Legacy function deleted in pipeline-alignment. Use `export_streaming_tflite()` or `convert_model_saved()` instead.

### Module AGENTS.md Files
- `src/data/AGENTS.md` - Data pipeline, augmentation, clustering, quality
- `src/training/AGENTS.md` - Training loop, mining, profiling, augmentation
- `src/model/AGENTS.md` - Architecture, streaming layers, state management
- `src/export/AGENTS.md` - TFLite export, manifest, verification
- `src/evaluation/AGENTS.md` - Metrics, FAH estimation, calibration
- `src/utils/AGENTS.md` - GPU config, performance, logging
- `src/tools/AGENTS.md` - CLI tools (cluster-analyze, cluster-apply)
- `src/tuning/AGENTS.md` - Auto-tuning (NEW)
- `config/AGENTS.md` - Configuration system
- `scripts/AGENTS.md` - Standalone utilities (NEW)

---

## Documentation References
- `docs/INDEX.md` - Documentation index
- `docs/ARCHITECTURE.md` - MixedNet architecture
- `docs/CONFIGURATION.md` - Config reference
- `docs/TRAINING.md` - Training workflow
- `docs/EXPORT.md` - Export guide
- `specs/implementation_status.md` - Implementation status
- `specs/phase1_complete.yaml` - Phase 1 summary
- `specs/testing_plan.md` - Testing strategy
