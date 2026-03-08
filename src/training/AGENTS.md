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
| `__init__.py` | 16 | Package exports | Exports: `Trainer`, `TrainingMetrics` (alias for `EvaluationMetrics`), `train()`, `main()` |

## Entry Point
```python
# From setup.py console_scripts
mww-train = src.training.trainer:main
```

## Trainer Class
```python
class Trainer:
    def __init__(self, config: FullConfig)
    def train(self, train_data_factory, val_data_factory, input_shape=None)
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

## AsyncHardExampleMiner
```python
class AsyncHardExampleMiner:
    def __init__(self, strategy, fp_threshold, max_samples, mining_interval_epochs, output_dir)
    def start_mining(self, model, data_generator, epoch)
    def wait_for_completion(self)
    def get_result(self)
    def is_mining(self)
```

**Async Hard Negative Mining** - Background mining for non-blocking training operations:

**Public Methods**:
- `start_mining(model, data_generator, epoch)`: Start mining in background thread
- `wait_for_completion()`: Block until mining finishes (called at epoch end)
- `get_result()`: Retrieve mining results (dict with stats)
- `is_mining()`: Check if mining is currently running

**Usage Example**:
```python
# In Trainer._train_epoch()
if performance.async_mining and collection_mode == "mine_immediately":
    async_miner = AsyncHardExampleMiner(
        fp_threshold=0.8,
        max_samples=5000,
        mining_interval_epochs=5
    )
    async_miner.start_mining(model, data_generator, epoch)

    # Continue training...

    # At epoch end
    async_miner.wait_for_completion()
    result = async_miner.get_result()
```

**Key Differences from HardExampleMiner**:
- Runs in background thread (non-blocking)
- Model is cloned for thread safety
- Uses thread-safe locking for result access
- Better GPU utilization but requires careful sync

**Presets**:
- `standard` and `max_quality` presets in `config/presets/` enable async mining by default
- `fast_test` preset uses synchronous mining for simplicity

**Selection Criteria**:
- **Use AsyncHardExampleMiner** (`performance.async_mining=true`):
  - Large datasets (>10k samples)
  - High GPU availability
  - Need minimal training interruptions
  - Production training runs

- **Use HardExampleMiner** (`performance.async_mining=false`):
  - Small datasets (<5k samples)
  - Debugging and development
  - Reproducible behavior needed
  - Limited GPU resources

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
- `training.async_hard_neg_mining.*` - enabled, queue_size, confidence_threshold (for async mining)
- `performance.*` - gpu_only, mixed_precision, profiling
- `hardware.*` - sample_rate, mel_bins, window/step sizes
- `augmentation.*` - waveform augmentation params
- `hard_negative_mining.*` - fp_threshold, mining_interval_epochs, max_samples, collection_mode (root-level config for mining behavior)

## Hard Negative Mining Configuration

The framework provides two mechanisms for hard negative mining:

1. **`hard_negative_mining.*`** (root-level): Configuration for hard negative mining behavior
   - Controls mining parameters: `fp_threshold`, `max_samples`, `mining_interval_epochs`
   - Determines when mining occurs via `collection_mode`:
     - `"log_only"`: Only log false predictions; no mining during training
     - `"mine_immediately"`: Mine immediately during training (uses async if enabled)
   - Applies to both synchronous and async mining modes

2. **`performance.async_mining`**: Boolean flag to enable.background async mining
   - `false` (default): Uses `HardExampleMiner` (synchronous, blocking)
   - `true`: Uses `AsyncHardExampleMiner` when `collection_mode="mine_immediately"` (background thread, non-blocking)

**Relationship and Selection**:
- These are NOT mutually exclusive; they work together to control mining behavior
- `performance.async_mining` is a performance optimization flag
- `hard_negative_mining.collection_mode` controls the mining strategy
- **Use `performance.async_mining=true`** for: High-throughput training, non-blocking mining, better GPU utilization
- **Use `performance.async_mining=false`** for: Deterministic behavior, simpler debugging, reproduction

**Example Configurations**:

```yaml
# Async mining (high throughput)
performance:
  async_mining: true
hard_negative_mining:
  enabled: true
  collection_mode: "mine_immediately"

# Sync mining (deterministic)
performance:
  async_mining: false
hard_negative_mining:
  enabled: true
  collection_mode: "mine_immediately"
```

## Anti-Patterns
- **Don't instantiate Trainer directly for production** - Use `main()` or `train()` helper
- **Don't ignore config validation** - Pass validated config from loader
- **Don't call augmentation after spectrogram** - Waveform augs are pre-spectrogram only

## Notes
- Integrates with `src/data/` for dataset loading
- Uses `src/model/` for model architecture via `build_model()`
- Uses `src/evaluation/` for validation metrics via `MetricsCalculator`
- **`EvaluationMetrics`** (in `trainer.py`) wraps `MetricsCalculator` with a batch-accumulation interface and is the canonical class used inside the training loop.
- **`TrainingMetrics`** (exported from `__init__.py`) is a backward-compatibility alias: `TrainingMetrics = EvaluationMetrics`. Prefer `EvaluationMetrics` for new code; use `TrainingMetrics` only when you need a package-level import for backward compat.
- External callers that only need offline/post-training metrics should import `MetricsCalculator` directly from `src.evaluation.metrics`.
- Supports TensorBoard logging (controlled via config)
- Two-phase training: typically [20000, 10000] steps with [0.001, 0.0001] LR
- Best model selection by FAH (false activations/hour) then recall
- **AsyncHardExampleMiner** is enabled by default in standard and max_quality presets for better throughput


## Related Documentation

- [Training Guide](../../docs/TRAINING.md) - Complete training documentation
- [Configuration Reference](../../docs/CONFIGURATION.md) - TrainingConfig options
- [Architecture Guide](../../docs/ARCHITECTURE.md) - Model architecture