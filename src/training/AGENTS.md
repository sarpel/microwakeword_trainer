# src/training/ - Training Pipeline

Complete training loop with Rich logging, profiling, hard example mining, and waveform augmentation.

## Files
| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `trainer.py` | ~960 | Main training loop | `Trainer`, `EvaluationMetrics`, `train()` |
| `mining.py` | ~1860 | Unified mining & FP extraction | `HardExampleMiner`, `AsyncHardExampleMiner` |
| `rich_logger.py` | 312 | Rich-based progress display | `RichTrainingLogger` |
| `augmentation.py` | 266 | Waveform-level augmentation | `AudioAugmentationPipeline` |
| `profiler.py` | 176 | Section-based profiling | `TrainingProfiler` |

## Entry Points
```python
mww-train = src.training.trainer:main
mww-mine-hard-negatives = src.training.mining:main
```

## Trainer Class
```python
class Trainer:
    def __init__(self, config: FullConfig)
    def train(self, train_data_factory, val_data_factory, ...)
    def train_step(self, fingerprints, ground_truth, sample_weights)
    def validate(self, data_factory)
```

## Mining Module

Unified hard negative mining with two modes:
- **AsyncHardExampleMiner**: Background mining (non-blocking)
- **HardExampleMiner**: Synchronous mining

## Training Flow
```
Config → Trainer.__init__() → _build_model() → train()
  │                                                │
  │  For each phase:
  │    train_step() → validate() → mining
  │    _save_checkpoint() if _is_best_model()
  └── log_completion()
```

## Checkpoint Selection

Two-stage strategy:
1. **Stage 1 (warm-up)**: Saves by PR-AUC until FAH budget first met
2. **Stage 2 (operational)**: Saves by recall@target_fah when FAH ≤ target

## EMA Weight Management

When EMA is enabled (`training.ema_decay` configured, default in `max_quality.yaml`):

**Checkpoint types and EMA status:**

| Checkpoint File | EMA Weights? | Purpose |
|---------------|-------------|---------|
| `final_weights.weights.h5` | ✅ Yes | End of training, export/inference (preferred) |
| `best_weights.weights.h5` | ✅ Yes | Best model during training, resume training |
| `checkpoint_step_NNNN.weights.h5` | ✅ Yes | Periodic recovery checkpoints |

**Training flow with EMA:**
1. `_swap_to_ema_weights()` called before validation/checkpointing
2. Optimizer EMA weights used for evaluation
3. `_restore_training_weights()` called after validation
4. Original training weights restored for gradient updates

## Class Weights (Default)
- Positive: 1.0
- Negative: 20.0
- Hard negative: 40.0

## Anti-Patterns
- **Don't instantiate Trainer directly** - Use `main()` or `train()` helper
- **Don't ignore config validation** - Pass validated config from loader
- **Don't use `trainable_weights` for serialization** - Use `get_weights()`/`set_weights()`
- **Don't export with pre-Flatten checkpoints** - Pre-2026-03-11 checkpoints incompatible
- **Don't reload `best_weights` at end of training** - Training complete; `final_weights` already has EMA-smoothed weights
- **Never load checkpoint after EMA finalize** - Triggers optimizer state warnings without benefit; only reload when resuming from interruption

## Related Documentation

- [Training Guide](../../docs/TRAINING.md)
- [Configuration Reference](../../docs/CONFIGURATION.md)
