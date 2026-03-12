# Auto-Tuning Module

**Module:** `src/tuning/`
**Purpose:** Post-training auto-tuning for FAH/recall optimization using MaxQualityAutoTuner

## OVERVIEW
MaxQualityAutoTuner — a sophisticated post-training auto-tuning system that iteratively improves wake word model quality to achieve target FAH/recall. Uses 7 surgical strategy arms with Thompson sampling, 3-pass threshold optimization, temperature scaling, Pareto archive, and optional INT8 shadow evaluation. Quality is prioritized over speed.

**Replaces** the previous Optuna-based AutoTuner (which was broken — every iteration produced identical results due to full retraining destroying optimizer state).

## STRUCTURE
```
src/tuning/
├── autotuner.py        # MaxQualityAutoTuner + all components (2624 lines)
├── cli.py             # mww-autotune CLI entry point (308 lines)
└── __init__.py        # Module init, exports AutoTuner + autotune
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main auto-tuner class | `autotuner.py` (AutoTuner) | Entry point: `tune()` method |
| Data structures | `autotuner.py` (TuneMetrics, CandidateState) | Dataclasses for metrics/state |
| Pareto archive | `autotuner.py` (ParetoArchive) | Multi-objective, crowding distance |
| Error memory | `autotuner.py` (ErrorMemory) | Persistent FP/miss tracking |
| Focused sampler | `autotuner.py` (FocusedSampler) | 7 strategy-specific batch builders |
| Temperature scaling | `autotuner.py` (TemperatureScaler) | Platt scaling, ECE computation |
| Threshold optimizer | `autotuner.py` (ThresholdOptimizer) | 3-pass: quantile→exact→CV |
| Thompson sampling | `autotuner.py` (ThompsonSampler) | Beta posterior + regime bonuses |
| Stir controller | `autotuner.py` (StirController) | 5-level stagnation escape |
| Annealing controller | `autotuner.py` (AnnealingController) | Simulated annealing + reheat |
| CLI entry point | `cli.py` (main) | Argument parsing, config overrides |
| User hard negatives | `autotuner.py` (users_hard_negs_dir param) | Custom hard negatives support |

## ARCHITECTURE

### Key Components

1. **ParetoArchive** — Multi-objective archive tracking non-dominated solutions. Uses crowding distance for diversity. Never regresses on any objective.

2. **ErrorMemory** — Tracks per-sample false positive/miss history across iterations. Identifies persistent errors for focused sampling.

3. **FocusedSampler** — 7 strategy-specific batch builders:
   - `threshold_only` — No gradient, threshold sweep only
   - `hard_negative_focus` — Emphasizes hard negatives
   - `false_positive_focus` — Targets persistent false positives
   - `boundary_refinement` — Decision boundary samples
   - `recall_recovery` — Boosts positive class
   - `balanced_tune` — Equal class representation
   - `adversarial_mix` — Mixed difficult samples

4. **TemperatureScaler** — Platt scaling (logistic regression on logits) for probability calibration. Reports ECE (Expected Calibration Error).

5. **ThresholdOptimizer** — 3-pass threshold search:
   - Pass 1: Coarse quantile sweep
   - Pass 2: Fine exact sweep around best quantile
   - Pass 3: Cross-validated refinement with bootstrap confidence intervals

6. **ThompsonSampler** — Beta-distribution posterior over strategy success. Adds regime-dependent bonuses (e.g., boost `false_positive_focus` when FAH is high).

7. **StirController** — 5-level stagnation escape:
   - Level 1: LR perturbation
   - Level 2: Strategy diversity boost
   - Level 3: Weight noise injection
   - Level 4: Partial layer reset
   - Level 5: Full reset to best checkpoint

8. **AnnealingController** — Simulated annealing with automatic reheat on prolonged stagnation.

### Training Flow
1. Load checkpoint, build model, prepare data partitions (train/val/confirm)
2. Evaluate baseline with 3-pass threshold optimization
3. Main loop (max_iterations):
   a. Thompson sampling selects strategy arm
   b. FocusedSampler builds targeted batch
   c. `train_burst()` executes gradient steps (supports SAM + SWA)
   d. 3-pass threshold optimization on new weights
   e. Temperature scaling for calibration
   f. Simulated annealing acceptance criterion
   g. Optional INT8 shadow evaluation at intervals
   h. Pareto archive update, error memory update
   i. Stir controller checks for stagnation
4. Confirmation phase: validate final candidate on held-out data
5. Export best checkpoint

## CONVENTIONS

### Configuration
- **AutoTuningConfig** in `config/loader.py` — Core config (max_iterations, target_fah, target_recall, max_gradient_steps, cv_folds, int8_shadow, etc.)
- **AutoTuningExpertConfig** in `config/loader.py` — Expert-only knobs (22 fields: annealing, stir, SAM, SWA, Thompson prior, etc.)
- **All three presets** have both sections: fast_test (reduced), standard, max_quality (enhanced)

### API Contract (MUST PRESERVE)
```python
# Constructor (drop-in compatible with old AutoTuner)
AutoTuner(
    checkpoint_path: str,
    config: dict,
    auto_tuning_config: dict | None = None,
    console: Console | None = None,
    users_hard_negs_dir: str | None = None,
)

# Main method
tune() -> dict
# Returns: best_fah, best_recall, final_fah, final_recall,
#          iterations, best_checkpoint, target_met, pareto_frontier

# Convenience function
autotune(checkpoint_path, config, output_dir, target_fah,
         target_recall, max_iterations) -> dict
```

### Key Technical Details
- **Weight format**: `.weights.h5` via `model.save_weights()` / `model.load_weights()`
- **BN freezing**: Set `trainable=False` on BatchNormalization layers
- **Loss**: BinaryCrossentropy(from_logits=False, label_smoothing=...)
- **Optimizer**: Adam with `optimizer.learning_rate.assign(lr)` for LR changes
- **Labels**: 0=negative, 1=positive, 2=hard_negative
- **FAH**: false_positives / ambient_duration_hours
- **INT8 export**: ExportArchive → SavedModel → TFLiteConverter with uint8 output

### Critical Bug Fix: Weight Serialization (2026-03-10)
**Issue**: `_serialize_weights()` originally used `model.trainable_weights` which does NOT include non-trainable variables like BatchNorm moving statistics (moving_mean, moving_variance). This caused models to appear excellent during tuning (FAH=0.00) but fail catastrophically during confirmation (FAH=129+) because BatchNorm running statistics were lost.

**Root Cause**: The tuning model runs in NON_STREAM mode (no streaming state variables exist). However, BatchNorm layers have 4 variables: gamma, beta (trainable) + moving_mean, moving_variance (non-trainable). The non-trainable BN state was excluded by `model.trainable_weights`, causing feature normalization to use uninitialized defaults during confirmation.

**Fix**: Changed to `model.get_weights()` / `model.set_weights()` which includes ALL weights in layer creation order:
```python
# WRONG - only saves trainable weights (missing BN moving stats)
weights = [w.numpy() for w in model.trainable_weights]

# ALSO WRONG - alphabetical ordering doesn't match get_weights()
weights = [w.numpy() for w in model.variables]

# CORRECT - saves all weights in layer creation order
weights = model.get_weights()  # serialize
model.set_weights(weights)      # deserialize
```

## ANTI-PATTERNS (THIS MODULE)

- **⛔ Don't perform full retraining** — Auto-tuning uses surgical gradient bursts (20-200 steps), NOT full epochs
- **⛔ Don't destroy optimizer state** — Use `optimizer.learning_rate.assign()`, don't recreate optimizer
- **⛔ Don't skip threshold optimization** — Always run 3-pass threshold search after weight changes
- **⛔ Don't ignore Pareto archive** — Accept candidates only if Pareto-improving or annealing accepts
- **⛔ Don't use int8 output dtype** — ESPHome requires uint8 (applies to INT8 shadow eval)
- **⛔ Don't ignore user-defined hard negatives** — Always use `users_hard_negs_dir` if provided
- **⛔ Don't set aggressive targets** — Realistic: FAH < 0.5, recall > 0.90
- **⛔ Don't skip confirmation phase** — Final validation on held-out data prevents overfitting
- **⛔ Don't mix TF and PyTorch** — This module is TensorFlow only
- **⛔ Don't use `trainable_weights` for serialization** — Use `model.get_weights()`/`model.set_weights()` to include BatchNorm moving statistics. See Critical Bug Fix section above.
- **⛔ Don't use `model.variables` for serialization** — Variables are in alphabetical order, not layer creation order. Causes weight misassignment when paired with `model.set_weights()`.

## CLI USAGE
```bash
# Auto-tune with default targets
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard

# With custom targets
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard \
    --target-fah 0.2 --target-recall 0.95

# With user-defined hard negatives
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard \
    --users-hard-negs /path/to/custom_hard_negatives/

# With expert overrides
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard \
    --max-gradient-steps 100 --cv-folds 5 --no-int8-shadow

# Dry run (config check only)
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard --dry-run
```

## INTEGRATION POINTS
- **Model**: `src/model/architecture.py` — `build_model()` for model construction
- **Data**: `src/data/dataset.py` — `WakeWordDataset` for data loading, `val_generator_factory()`
- **Evaluation**: `src/evaluation/metrics.py` — `MetricsCalculator` for FAH/recall computation
- **Export**: `src/export/tflite.py` — INT8 shadow evaluation via `convert_model_saved()` + `convert_saved_model_to_tflite()`
- **Config**: `config/loader.py` — `AutoTuningConfig` + `AutoTuningExpertConfig`
- **GPU**: `src/utils/performance.py` — `set_threading_config()` for GPU setup

## NOTES

### Performance Characteristics
- **Duration**: Minutes to hours depending on max_iterations and gradient steps
- **GPU required**: All training + evaluation runs on GPU
- **Output**: Tuned checkpoint saved to new location, original preserved
- **Quality focus**: Time cost does not matter — one successful model over many mediocre ones

### Gotchas
- `WakeWordDataset.val_generator_factory()` returns a factory (callable), not a generator. Call `factory()` to get actual generator.
- `build_model()` takes the full config dict, not just model config
- `model.load_weights()` prints UserWarning about optimizer state — suppress with `warnings.catch_warnings()`
- Ambient duration hours must be scaled by validation split fraction for correct FAH on validation subset
- SciPy required for temperature scaling (Platt scaling via `scipy.optimize.minimize`)
