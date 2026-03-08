# Auto-Tuning Module

**Module:** `src/tuning/`
**Purpose:** Post-training fine-tuning for FAH/recall optimization

## OVERVIEW
Optuna-based hyperparameter optimization that adjusts probability thresholds and fine-tunes existing checkpoints to achieve target FAH and recall metrics without full retraining.

## STRUCTURE
```
src/tuning/
├── autotuner.py        # AutoTuner class, tuning logic (691 lines)
├── cli.py             # mww-autotune CLI entry point (257 lines)
└── __init__.py        # Module init
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Auto-tuning logic | `autotuner.py` (AutoTuner class) | Optuna studies, threshold tuning |
| CLI entry point | `cli.py` (main function) | Argument parsing, orchestration |
| User-defined hard negatives | `autotuner.py` (users_hard_negs_dir param) | Load custom hard negatives |

## CONVENTIONS

### Auto-Tuning Workflow
1. **Load baseline model** from checkpoint
2. **Iterate through hyperparameters** using Optuna
3. **Evaluate on validation set** for each iteration
4. **Fine-tune best configuration** with hard negative mining
5. **Export tuned checkpoint** to output directory

### Hyperparameters Optimized
- Probability threshold (probability_cutoff)
- Learning rate (fine_tuning_lr)
- Training steps (fine_tuning_steps)
- Batch size (fine_tuning_batch_size)

### User-Defined Hard Negatives
- **Parameter**: `users_hard_negs_dir` in AutoTuner
- **CLI flag**: `--users-hard-negs /path/to/hard_negatives/`
- **Purpose**: Use custom hard negatives for better specificity
- **Behavior**: Loads user-provided samples during fine-tuning iterations

## ANTI-PATTERNS (THIS MODULE)

- **Don't perform full retraining** - Auto-tuning is for fine-tuning existing models
- **Don't ignore user-defined hard negatives** - Always use `users_hard_negs_dir` if provided
- **Don't use aggressive targets** - Set realistic FAH (<0.5) and recall (>0.90) goals
- **Don't skip baseline evaluation** - Always evaluate original model first

## NOTES

### Recent Enhancements
- **User-defined hard negatives support** (commit 2fa00e22e): Users can provide custom hard negative directories for auto-tuning to improve model discrimination against known false positives

### CLI Usage
```bash
# Auto-tune with default targets
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard

# With custom targets
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard --target-fah 0.2 --target-recall 0.95

# With user-defined hard negatives
mww-autotune --checkpoint checkpoints/best_weights.weights.h5 --config standard --users-hard-negs /path/to/custom_hard_negatives/
```

### Integration Points
- **Dependencies**: Optuna for hyperparameter optimization
- **Data loading**: Uses WakeWordDataset from `src/data/dataset.py`
- **Model loading**: Loads checkpoints from `src/model/architecture.py`
- **Evaluation**: Uses metrics from `src/evaluation/metrics.py`
- **Hard negatives**: Integrates with `src/data/hard_negatives.py`

### Performance
- **Duration**: Typically 5-10 minutes per auto-tuning session
- **Output**: Tuned checkpoint (does NOT overwrite original)
- **Optimization**: FAH/recall trade-off via threshold adjustment

### Gotchas
- **Optuna required**: Must install with `pip install optuna`
- **No GPU fallback**: Requires GPU for fine-tuning
- **Separate checkpoint**: Tuned model saved to new location, preserving original
