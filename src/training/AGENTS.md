# src/training/ - Training Pipeline

## Overview
Training loop implementation with profiling and hard negative mining.

## Files
| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `trainer.py` | Main training loop | `Trainer`, `train()`, `main()` |
| `profiler.py` | Performance profiling | 176 lines - TensorBoard, timing |
| `augmentation.py` | Audio augmentation | Pipeline setup |
| `miner.py` | Hard negative mining | Sample selection |

## Entry Point
```python
# From setup.py console_scripts
mww-train = src.training.trainer:main
```

## Trainer Class
```python
class Trainer:
    def __init__(self, config: dict)
    def train(self, train_data_factory, val_data_factory, input_shape=(49, 40))
    def train_step(self, fingerprints, ground_truth, sample_weights)
    def validate(self, data_factory)   # Accepts callable factory or generator
```

## Configuration
Expects config dict from `config.loader` with:
- `training.*` - batch_size, learning_rates, weights
- `performance.*` - gpu_only, mixed_precision, profiling
- `hardware.*` - sample_rate, feature dims

## Anti-Patterns
- **Don't instantiate Trainer directly** - Use `main()` or `train()` helper
- **Don't ignore config validation** - Pass validated config from loader

## Notes
- Integrates with `src/data/` for dataset loading
- Uses `src/model/` for model architecture
- Supports TensorBoard logging (controlled via config)
