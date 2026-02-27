# src/training/ - Training Pipeline

## Overview
Complete training loop with Rich logging, profiling, hard example mining, and waveform augmentation.

## Files
| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `trainer.py` | 874 | Main training loop | `Trainer`, `EvaluationMetrics`, `train()`, `main()` |
| `rich_logger.py` | 312 | Rich-based progress display | `RichTrainingLogger` |
| `miner.py` | 305 | Hard negative mining during training | `HardExampleMiner`, `mine_hard_examples()` |
| `augmentation.py` | 266 | Waveform-level augmentation pipeline | `AudioAugmentationPipeline`, `ParallelAugmenter` |
| `profiler.py` | 176 | Section-based training profiling | `TrainingProfiler` |
| `__init__.py` | 16 | Package exports | Exports: `Trainer`, `TrainingMetrics`, `train()`, `main()` |

## Entry Point
```python
# From setup.py console_scripts
mww-train = src.training.trainer:main
```

## Trainer Class
```python
class Trainer:
    def __init__(self, config: FullConfig)
    def train(self, train_data_factory, val_data_factory, input_shape=(49, 40))
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

## HardExampleMiner
```python
class HardExampleMiner:
    def __init__(self, config)
    def get_hard_samples(self, model, dataset)
    def mine_from_dataset(self, model, data_factory)
    def save_hard_negatives(self, path)
    def load_hard_negatives(self, path)
    def get_all_hard_negatives(self)
```

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

## Configuration
Expects FullConfig from `config.loader` with:
- `training.*` - batch_size, learning_rates, training_steps, eval_step_interval
- `performance.*` - gpu_only, mixed_precision, profiling
- `hardware.*` - sample_rate, mel_bins, window/step sizes
- `augmentation.*` - waveform augmentation params

## Anti-Patterns
- **Don't instantiate Trainer directly for production** - Use `main()` or `train()` helper
- **Don't ignore config validation** - Pass validated config from loader
- **Don't call augmentation after spectrogram** - Waveform augs are pre-spectrogram only

## Notes
- Integrates with `src/data/` for dataset loading
- Uses `src/model/` for model architecture via `build_model()`
- Uses `src/evaluation/` for validation metrics (MetricsCalculator)
- Supports TensorBoard logging (controlled via config)
- Two-phase training: typically [20000, 10000] steps with [0.001, 0.0001] LR
- Best model selection by FAH (false activations/hour) then recall
