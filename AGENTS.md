# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-13
**Project:** microwakeword_trainer v2.0.0

## OVERVIEW
GPU-accelerated wake word training framework for ESPHome. TensorFlow-based pipeline with MixedNet architecture, CuPy GPU SpecAugment, TFLite INT8 export.

## STRUCTURE
```
./
├── src/                  # Source code (8 modules, ~19,685 lines Python)
├── config/                # YAML presets + Python loader
├── tests/                 # Unit + integration tests
├── scripts/               # Standalone tools
├── docs/                  # User documentation
├── specs/                 # Implementation specs
├── ARCHITECTURAL_CONSTITUTION.md  # Immutable architectural truth
└── AGENTS.md             # This file
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Training loop | `src/training/trainer.py` | Two-phase, hard negative mining |
| Mining & FP extraction | `src/training/mining.py` | HardExampleMiner, AsyncMiner |
| Config system | `config/loader.py` | 14 dataclasses, env var substitution |
| Model architecture | `src/model/architecture.py` | MixedNet, streaming layers |
| TFLite export | `src/export/tflite.py` | INT8 quantization, dual subgraphs |
| Data pipeline | `src/data/dataset.py` | RaggedMmap, WakeWordDataset |
| Speaker clustering | `src/data/clustering.py` | ECAPA-TDNN embeddings |
| Auto-tuning | `src/tuning/autotuner.py` | MaxQualityAutoTuner |
| Evaluation metrics | `src/evaluation/metrics.py` | FAH, ROC-AUC |

## CONVENTIONS

### Configuration System
- **14 dataclass sections** in loader.py (Hardware, Paths, Training, Model, Augmentation, Performance, SpeakerClustering, Mining, Export, Preprocessing, Quality, Evaluation)
- **Three presets**: fast_test, standard, max_quality
- **Environment variable substitution**: `${VAR}` or `${VAR:-default}`
- **Immutable hardware section**: Enforced by ARCHITECTURAL_CONSTITUTION.md
- **Two separate venvs**: TF (main) + PyTorch (clustering)
- **`search_eval_fraction`** in AutoTuningConfig controls search data split (default 0.30)

### Code Style
- **Relaxed typing**: mypy `disallow_untyped_defs=false`
- **Directory structure**: src/ layout

## ANTI-PATTERNS (THIS PROJECT)

### Critical Violations (Silent Device Failure)
- **Don't contradict ARCHITECTURAL_CONSTITUTION.md** - Silent device failure
- **Don't use int8 output dtype** - ESPHome requires uint8
- **Don't use `model.export()`** - Use `tf.keras.export.ExportArchive`
- **Don't use `model.trainable_weights` for serialization** - Excludes BatchNorm moving stats. Use `model.get_weights()`/`model.set_weights()`
- **Don't evaluate auto-tuner on same data used for FocusedSampler training** — split search into search_train/search_eval via `search_eval_fraction` config. Training and evaluating on same data causes train-on-test contamination that degrades model quality.
- **Don't hardcode okay_nabu state shapes in export verification** — use `compute_expected_state_shapes()` for config-aware validation. Models with different `clip_duration_ms` produce different `temporal_frames`, changing stream_5 shape.
- **Don't reload checkpoints after EMA finalize** - At end of training, EMA may have cleared optimizer state; reloading triggers "optimizer has 2 variables whereas saved has 92 variables" warning without benefit. Only reload when resuming from interruption (see ARCHITECTURAL_CONSTITUTION.md Article IX).

### Environment & Dependencies
- **Don't mix TF and PyTorch in same venv** - Use separate environments
- **Don't use CPU-only CuPy** - SpecAugment requires GPU
- **Don't import scripts as a module** - Run as `python scripts/<script>.py`

### Configuration
- **Don't add config only to loader.py** - Must add to ALL THREE presets
- **Don't use deprecated variable names** - No backward compatibility

### Editing Rules
- **Don't use LINE#IDs older than 10 seconds** - Re-read file before editing
- **Don't make sequential edits to same file** - Batch into ONE edit

## UNIQUE STYLES

### PCAN (Per-Channel Normalization)
- PCAN is always ON in pymicro-features C++ backend; no Python flag exists

### Training Pipeline
- **AsyncHardExampleMiner**: Background hard negative mining
- **Two-phase training**: Phase 1 (feature learning) + Phase 2 (fine-tuning)
- **Class weighting**: positive=1.0, negative=20.0, hard_neg=40.0
- **Checkpoint selection**: Two-stage (PR-AUC warm-up → recall@target_FAH)

### Export System
- **Dual subgraphs**: Main inference + initialization
- **6 state variables**: `stream` through `stream_5` (see ARCHITECTURAL_CONSTITUTION.md)
- **Input**: int8 [1,3,40], **Output**: uint8 [1,1]

## COMMANDS
```bash
# Training
mww-train --config config/presets/standard.yaml

# Export
mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/

# Auto-tune
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard

# Speaker clustering (PyTorch env)
mww-cluster-analyze --config standard --dataset all --n-clusters 200
mww-cluster-apply --namelist-dir cluster_output

# Verification
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Advanced evaluation + reports
python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard --output-dir logs/

# Interactive dashboard from report
python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json
```

## NOTES

### Critical Files
- **ARCHITECTURAL_CONSTITUTION.md**: Supreme source of truth for all architectural constants
- **src/*/AGENTS.md**: Per-module patterns

### Gotchas
- CuPy no CPU fallback: Must have GPU for training
- uint8 output mandatory: ESPHome rejects int8 outputs
- Old checkpoint incompatibility: Pre-2026-03-11 checkpoints have incompatible Dense shapes
- BatchNorm state: moving_mean/moving_variance are NON-TRAINABLE
- Auto-tuner search split: search data is split into search_train (70%) and search_eval (30%) — FocusedSampler trains on search_train only
- Export state shapes vary: stream_5 shape depends on temporal_frames derived from clip_duration_ms, not always (1,5,1,64)
- `verify_esphome.py` JSON mode sanitizes NumPy values before serialization; use `--json` for CI and `--verbose` for detailed local diagnostics
- `evaluate_model.py` writes `evaluation_report.json`, PNG plots, and executive reports (`executive_report.md` / `.html`) under `evaluation_artifacts/`
- `eval_dashboard.py` builds `interactive_dashboard.html` from `evaluation_report.json` (keep dashboard in same folder as report/images)
- `DELEGATE` visibility is runtime/delegate-path dependent in analyzers; compatibility checks should focus on static-graph invariants and ESPHome-registered op set

### Module AGENTS.md Files
- `src/data/AGENTS.md` - Data pipeline
- `src/training/AGENTS.md` - Training loop
- `src/model/AGENTS.md` - Architecture
- `src/export/AGENTS.md` - TFLite export
- `src/evaluation/AGENTS.md` - Metrics
- `src/utils/AGENTS.md` - GPU config
- `src/tools/AGENTS.md` - CLI tools
- `src/tuning/AGENTS.md` - Auto-tuning
- `config/AGENTS.md` - Configuration

---

## Documentation References
- `docs/INDEX.md` - Documentation index
- `docs/ARCHITECTURE.md` - MixedNet architecture
- `docs/CONFIGURATION.md` - Config reference
- `ARCHITECTURAL_CONSTITUTION.md` - Immutable constants and constraints
