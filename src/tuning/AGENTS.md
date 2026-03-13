# Auto-Tuning Module

**Module:** `src/tuning/`
**Purpose:** Post-training auto-tuning for FAH/recall optimization using MaxQualityAutoTuner

## OVERVIEW

MaxQualityAutoTuner - Post-training optimization to achieve target FAH/recall without full retraining. Uses 7 strategy arms with Thompson sampling, Pareto archive, and temperature scaling.

## STRUCTURE

```
src/tuning/
├── autotuner.py        # MaxQualityAutoTuner (2689 lines)
├── cli.py             # mww-autotune CLI entry point
└── __init__.py        # Module exports
```

## Key Components

1. **ParetoArchive** - Multi-objective archive for non-dominated solutions
2. **ErrorMemory** - Tracks persistent false positives/misses
3. **FocusedSampler** - 7 strategy-specific batch builders
4. **TemperatureScaler** - Platt scaling for probability calibration
5. **ThompsonSampler** - Strategy selection via Beta posteriors

## API Contract

```python
AutoTuner(
    checkpoint_path: str,
    config: dict,
    auto_tuning_config: dict | None = None,
    users_hard_negs_dir: str | None = None,
)

tune() -> dict  # Returns best_fah, best_recall, best_checkpoint, etc.
```

## Weight Serialization

**CRITICAL**: Use `model.get_weights()` / `model.set_weights()` (NOT `trainable_weights`) to include BatchNorm moving statistics.

## CLI Usage

```bash
mww-autotune --checkpoint checkpoints/best.weights.h5 --config standard
mww-autotune --checkpoint checkpoints/best.weights.h5 --config standard \
    --target-fah 0.2 --target-recall 0.95 --users-hard-negs /path/to/negs/
```

## ANTI-PATTERNS

- **Don't perform full retraining** - Uses surgical gradient bursts (20-200 steps)
- **Don't destroy optimizer state** - Use `optimizer.learning_rate.assign()`
- **Don't use `trainable_weights` for serialization** - Use `get_weights()`/`set_weights()`
- **Don't skip confirmation phase** - Final validation on held-out data

## Related Documentation

- [Configuration Reference](../../docs/CONFIGURATION.md)
