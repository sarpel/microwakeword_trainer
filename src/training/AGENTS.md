# src/training/ - Training Pipeline

## Overview
Complete training loop with Rich logging, profiling, hard example mining, and waveform augmentation.

## Files
| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `trainer.py` | ~960 | Main training loop | `Trainer`, `EvaluationMetrics`, `train()`, `main()` |
| `mining.py` | ~1860 | Unified mining & FP extraction | `HardExampleMiner`, `AsyncHardExampleMiner`, `log_false_predictions_to_json()`, `run_top_fp_extraction()`, `consolidate_prediction_logs()`, `mine_from_prediction_logs()` |
| `rich_logger.py` | 312 | Rich-based progress display | `RichTrainingLogger` |
| `augmentation.py` | 266 | Waveform-level augmentation pipeline | `AudioAugmentationPipeline`, `ParallelAugmenter` |
| `profiler.py` | 176 | Section-based training profiling | `TrainingProfiler` |
| `__init__.py` | 16 | Package exports | Exports: `Trainer`, `TrainingMetrics` (alias for `EvaluationMetrics`), `train()`, `main()` |

## Entry Points
```python
# From pyproject.toml console_scripts
mww-train = src.training.trainer:main
mww-mine-hard-negatives = src.training.mining:main
```

## Trainer Class
```python
class Trainer:
    def __init__(self, config: FullConfig)
    def train(self, train_data_factory, val_data_factory, mining_data_factory=None, test_data_factory=None, input_shape=None, val_file_paths=None)
    def train_step(self, fingerprints, ground_truth, sample_weights)
    def validate(self, data_factory)   # Accepts callable factory or generator
    # Internal:
    def _get_current_phase_settings(self)
    def _build_model(self)
    def _apply_class_weights(self)
    def _is_best_model(self)
    def _save_checkpoint(self)
```

## EvaluationMetrics
```python
class EvaluationMetrics:
    def __init__(self)
    def update(self, y_true, y_pred)
    def compute_metrics(self) -> dict
    def reset(self)
```

## RichTrainingLogger
```python
class RichTrainingLogger:
    def log_header(self)
    def create_progress(self)
    def update_step(self, step, metrics)
    def log_validation_results(self, metrics)
    def log_confusion_matrix(self, matrix)
    def log_phase_transition(self, phase)
    def log_checkpoint(self, path)
    def log_completion(self, summary)
    def log_mining(self, stats)
    def log_warning(self, msg)
    def log_info(self, msg)
```

## Mining Module (`mining.py`)

Unified module consolidating ALL hard negative mining, false prediction logging, and extraction.

### Classes:

```python
class HardExampleMiner:
    """Heap-based in-training mining on feature-level data"""
    def __init__(self, strategy, fp_threshold, max_samples, mining_interval_epochs, output_dir)
    def get_hard_samples(self, labels, predictions) -> np.ndarray
    def mine_from_dataset(self, model, data_generator, epoch) -> dict
    def save_hard_negatives(self, features, labels, predictions, filepath=None)
    def load_hard_negatives(self, filepath)
    def get_all_hard_negatives(self) -> list[dict]

class AsyncHardExampleMiner:
    """Thread-safe wrapper for non-blocking background mining"""
    def __init__(self, strategy, fp_threshold, max_samples, mining_interval_epochs, output_dir)
    def start_mining(self, model, data_generator, epoch)  # Clones model for thread safety
    def wait_for_completion(self, timeout=None) -> bool
    def get_result(self) -> dict | None
    def is_mining(self) -> bool
```

### Module-level functions:

```python
# Called from trainer per-epoch:
log_false_predictions_to_json(epoch, y_true, y_scores, fp_threshold, top_k, log_file, val_paths, best_weights_path, logger)

# Called at end of training:
run_top_fp_extraction(config, checkpoint_path=None, top_percent_override=None, threshold_override=None, log_file_override=None)

# Post-training consolidation:
consolidate_prediction_logs(epoch_logs, all_files) -> dict
mine_from_prediction_logs(prediction_log, output_dir, min_epoch, top_k, deduplicate, dry_run) -> int
compute_file_hash(file_path, chunk_size=8192) -> str
```

### CLI subcommands (via `mww-mine-hard-negatives`):
- `mine` — Post-training mining from prediction logs
- `extract-top-fps` — Top N% false positive extraction via model inference
- `consolidate-logs` — Merge per-epoch logs with file path mapping

## Mining Configuration

All mining is configured via a single `MiningConfig` section in `config/loader.py`:

```yaml
mining:
  enabled: true
  async_mining: true        # Background mining (non-blocking)
  fp_threshold: 0.8         # Min score for false positive detection
  max_samples: 5000
  mining_interval_epochs: 5
  collection_mode: "log_only"  # or "mine_immediately"
  log_predictions: true
  log_file: "logs/false_predictions.json"
  top_k_per_epoch: 100
  # Post-training extraction
  extract_top_fps: true
  top_fp_percent: 5.0
  extraction_confidence_threshold: 0.5
  extraction_output_dir: "dataset/top5fps"
```

**Note**: `performance.async_mining` is DEPRECATED. Use `mining.async_mining` instead.

**Selection Criteria**:
- **Use AsyncHardExampleMiner** (`mining.async_mining: true`):
  - Large datasets (>10k samples)
  - Need minimal training interruptions
  - Production training runs

- **Use HardExampleMiner** (`mining.async_mining: false`):
  - Small datasets (<5k samples)
  - Debugging and development
  - Reproducible behavior needed

## AudioAugmentationPipeline
Waveform-level augmentations applied **before** spectrogram conversion. This is separate from `src/data/augmentation.py` (which is the data-level AudioAugmentation class with 8 aug types).
```python
class AudioAugmentationPipeline:
    def __init__(self, config: AugmentationConfig)
    def augment(self, waveform) -> waveform
    def __call__(self, waveform) -> waveform

class ParallelAugmenter:
    def __init__(self, pipeline, num_workers)
    def augment_batch(self, batch) -> batch
```

## Training Flow
```
Config → Trainer.__init__() → _build_model() → train()
  │                                                │
  │  For each phase (learning_rates × training_steps):
  │    For each step:
  │      train_step() → validate() every N steps
  │      HardExampleMiner.mine_from_dataset() periodically
  │      RichTrainingLogger.update_step()
  │    _save_checkpoint() if _is_best_model()
  └── log_completion()
```

## Class Weights (Default)
- Positive: 1.0
- Negative: 20.0
- Hard negative: 40.0

## Anti-Patterns
- **Don't instantiate Trainer directly for production** - Use `main()` or `train()` helper
- **Don't ignore config validation** - Pass validated config from loader
- **Don't call augmentation after spectrogram** - Waveform augs are pre-spectrogram only
- **Don't import from deleted files** - `miner.py`, `async_miner.py` no longer exist; import from `mining.py`
- **Don't use `hard_negative_mining` config key** - Use `mining` instead (consolidated)
- **Don't use `performance.async_mining`** - Deprecated; use `mining.async_mining`

## Notes
- Integrates with `src/data/` for dataset loading
- Uses `src/model/` for model architecture via `build_model()`
- Uses `src/evaluation/` for validation metrics via `MetricsCalculator`
- **`EvaluationMetrics`** (in `trainer.py`) wraps `MetricsCalculator` with a batch-accumulation interface and is the canonical class used inside the training loop.
- **`TrainingMetrics`** (exported from `__init__.py`) is a backward-compatibility alias: `TrainingMetrics = EvaluationMetrics`. Prefer `EvaluationMetrics` for new code; use `TrainingMetrics` only when you need a package-level import for backward compat.
- External callers that only need offline/post-training metrics should import `MetricsCalculator` directly from `src.evaluation.metrics`.
- Supports TensorBoard logging (controlled via config)
- Two-phase training: typically [20000, 10000] steps with [0.001, 0.0001] LR
- **Checkpoint selection uses a two-stage strategy** (implemented 2026-03-12):
  - **Stage 1 — Warm-up** (no epoch has yet met the FAH budget): saves by `auc_pr` (PR-AUC) improvement. Threshold-free and robust to class imbalance. Provides a stable training signal before the model learns to meet the FAH constraint.
  - **Stage 2 — Operational** (≥1 epoch has met FAH ≤ `target_fah × 1.1`): saves by `recall_at_target_fah` improvement, ONLY when the current epoch also meets the FAH budget. Directly maps to production semantics: best recall of all models that deploy within FAH budget.
  - Transition is automatic via `self.fah_budget_ever_met` flag.
  - `quality_score` (composite) is still computed and logged for display/plateau tracking but no longer drives checkpoint decisions.
  - New instance vars: `best_auc_pr`, `best_constrained_recall`, `fah_budget_ever_met`
- **Validation file path tracking**: `WakeWordDataset.get_split_file_paths('val')` provides ordered file paths for mapping prediction indices to files. Passed to `Trainer.train()` via `val_file_paths` parameter.

## Related Documentation

- [Training Guide](../../docs/TRAINING.md) - Complete training documentation
- [Configuration Reference](../../docs/CONFIGURATION.md) - TrainingConfig options
- [Architecture Guide](../../docs/ARCHITECTURE.md) - Model architecture
