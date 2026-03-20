# Commands Reference

Complete reference for all console scripts, Makefile targets, and standalone utility scripts.

<!-- AUTO-GENERATED: Generated from pyproject.toml, setup.py, Makefile, and scripts/ directory -->
<!-- Last updated: 2026-03-20 -->

---

## Console Scripts (mww-*)

These commands are installed globally via `pip install -e .` and available from any directory.

| Command | Purpose | Entry Point |
|---------|---------|-------------|
| `mww-train` | Full training pipeline with 2-phase training, hard negative mining, and checkpointing | `src.training.trainer:main` |
| `mww-export` | Export trained model to ESPHome-compatible streaming TFLite with INT8 quantization | `src.export.tflite:main` |
| `mww-autotune` | Population-based post-training auto-tuning to optimize FAH/recall trade-off | `src.tuning.cli:main` |
| `mww-cluster-analyze` | Speaker clustering analysis using ECAPA-TDNN embeddings (DRY-RUN, PyTorch env) | `src.tools.cluster_analyze:main` |
| `mww-cluster-apply` | Apply speaker clustering results to reorganize files (PyTorch env) | `src.tools.cluster_apply:main` |
| `mww-mine-hard-negatives` | Unified hard negative mining and false prediction extraction | `src.training.mining:main` |
| `mww-pipeline` | End-to-end pipeline: train → autotune → export → verify → gate → promote | `src.pipeline:main` |
| `mww-help` | Show post-training command reference panel with auto-detected checkpoint | `src.tools.help_panel:main` |

### Common Command Patterns

**Training:**
```bash
# Train with standard preset
mww-train --config config/presets/standard.yaml

# Train with custom config
mww-train --config my_config.yaml
```

**Export:**
```bash
# Export best checkpoint
mww-export --checkpoint models/checkpoints/best_weights.weights.h5 --output models/exported/

# Export with custom name
mww-export --checkpoint models/checkpoints/best_weights.weights.h5 --output models/exported/ --model-name "hey_siri"
```

**Auto-tune:**
```bash
# Tune checkpoint with standard config
mww-autotune --checkpoint models/checkpoints/best_weights.weights.h5 --config standard

# Tune with specific preset
mww-autotune --checkpoint models/checkpoints/best_weights.weights.h5 --config max_quality
```

**Speaker Clustering (PyTorch env):**
```bash
# Analyze clusters (read-only)
mww-cluster-analyze --config standard --dataset all --n-clusters 200

# Apply cluster results (MUTATES FILES)
mww-cluster-apply --namelist-dir cluster_output --dry-run  # Preview first
mww-cluster-apply --namelist-dir cluster_output            # Execute

# Undo cluster reorganization
mww-cluster-apply --undo cluster_output/positive_backup_manifest.json
```

**Full Pipeline:**
```bash
# Run complete pipeline with standard config
mww-pipeline --config standard

# Pipeline with custom checkpoint location
mww-pipeline --config standard --checkpoint-dir models/checkpoints
```

---

## Makefile Targets

Development and build targets for the project.

| Target | Description |
|---------|-------------|
| `make help` | Show all available targets |
| `make install` | Install production dependencies |
| `make install-dev` | Install development dependencies and pre-commit hooks |
| `make lint` | Run ruff linter on src/, config/, scripts/, tests/ |
| `make format` | Run ruff formatter on all source files |
| `make format-check` | Check formatting without making changes |
| `make type-check` | Run mypy type checker on src/ and config/ |
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only (fast, no GPU required) |
| `make test-integration` | Run integration tests only (GPU required) |
| `make test-parallel` | Run all tests in parallel with pytest-xdist |
| `make test-fast` | Run fast tests only (excludes slow and gpu tests) |
| `make coverage` | Run tests with coverage report (HTML and XML) |
| `make check` | Run all checks: lint, format-check, type-check, test |
| `make clean` | Clean build artifacts, caches, and temporary files |
| `make pre-commit` | Install and run pre-commit hooks on all files |
| `make build` | Build distribution package |
| `make check-dist` | Check package distribution with twine |

### Common Makefile Patterns

**Full development setup:**
```bash
make install-dev
make pre-commit  # Install git hooks
```

**Run checks before commit:**
```bash
make check  # Runs lint, format-check, type-check, test
```

**Clean and rebuild:**
```bash
make clean
make build
```

---

## Standalone Scripts

Utility scripts in `scripts/` directory run with `python scripts/<script>.py`.

### Evaluation Scripts

| Script | Purpose |
|---------|---------|
| `evaluate_model.py` | Full test-set evaluation with JSON/image/executive reports |
| `eval_dashboard.py` | Build interactive HTML dashboard from `evaluation_report.json` |
| `compare_models.py` | Compare two wake word models side-by-side on test dataset |
| `verify_esphome.py` | ESPHome compatibility verification with optional strict mode |
| `verify_streaming.py` | Streaming equivalence gate for TFLite models |
| `check_esphome_compat.py` | Check TFLite model compatibility with ESPHome (source-derived constraints) |

### Dataset Scripts

| Script | Purpose |
|---------|---------|
| `vad_trim.py` | VAD-trim speech files and split background files |
| `split_audio.py` | Split long audio files into shorter clips |
| `count_dataset.py` | Count dataset samples by category |
| `count_audio_hours.py` | Calculate total audio duration in dataset |
| `audio_analyzer.py` | Wakeword dataset audio file analyzer |
| `generate_test_dataset.py` | Generate synthetic test dataset for pipeline |

### Quality Analysis Scripts

| Script | Purpose |
|---------|---------|
| `score_quality_full.py` | Full audio quality scoring (DNSMOS + Silero + WADA-SNR + clipping) |
| `score_quality_fast.py` | Fast audio quality scoring (WADA-SNR + clipping) |
| `audio_similarity_detector.py` | Audio similarity detection using CLAP + GPU |

### Hard Negative Mining Scripts

| Script | Purpose |
|---------|---------|
| `phonetic_scorer.py` | Phonetic similarity scorer for identifying hard negatives |

### Debugging Scripts

| Script | Purpose |
|---------|---------|
| `debug_state_reset.py` | Debug state reset tool |
| `debug_state_reset2.py` | Debug state reset tool (version 2) |
| `diagnose_savedmodel.py` | SavedModel diagnostic tool |
| `full_reset_test.py` | Full reset test tool |
| `benchmark_reset.py` | Benchmark reset tool |
| `debug_streaming_gap.py` | Debug streaming vs training model gap |

### Utility Scripts

| Script | Purpose |
|---------|---------|
| `cleanup_tfdata_cache.py` | Clean up TensorFlow tf.data cache tempstate files |
| `code_review_report.py` | Code review report generator |
| `find_rich_tables.py` | Search for 'box=None' expressions in Python files |

### Common Script Patterns

**Evaluate model:**
```bash
python scripts/evaluate_model.py \
  --model models/exported/wake_word.tflite \
  --config standard \
  --output-dir logs/
```

**Create evaluation dashboard:**
```bash
python scripts/eval_dashboard.py \
  --report logs/evaluation_artifacts/evaluation_report.json
```

**Verify ESPHome compatibility:**
```bash
# Basic verification
python scripts/verify_esphome.py models/exported/wake_word.tflite

# JSON mode for CI
python scripts/verify_esphome.py models/exported/wake_word.tflite --json

# Verbose mode for diagnostics
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
```

**Quality scoring:**
```bash
# Fast scoring (recommended for large datasets)
python scripts/score_quality_fast.py --input-dir dataset/positive/

# Full scoring with multiple metrics
python scripts/score_quality_full.py --input-dir dataset/positive/
```

**Audio preprocessing:**
```bash
# VAD trim speech files
python scripts/vad_trim.py --input-dir dataset/positive/ --output-dir dataset/trimmed_positive/

# Split long background files
python scripts/vad_trim.py --input-dir dataset/background/ --output-dir dataset/split_background/ --split
```

---

## Quick Reference

| Task | Command |
|-------|---------|
| Train model | `mww-train --config standard` |
| Export model | `mww-export --checkpoint models/checkpoints/best_weights.weights.h5 --output models/exported/` |
| Auto-tune | `mww-autotune --checkpoint models/checkpoints/best_weights.weights.h5 --config standard` |
| Evaluate | `python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard` |
| Run all checks | `make check` |
| Run tests | `make test` |
| Format code | `make format` |
| Clean artifacts | `make clean` |

---

## Related Documentation

- [Configuration Reference](CONFIGURATION.md) — Complete config system documentation
- [Training Guide](TRAINING.md) — Training workflow and optimization
- [Export Guide](EXPORT.md) — TFLite export and ESPHome deployment
- [Project Index](INDEX.md) — Complete project overview
