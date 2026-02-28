# config/ - Configuration System

## Overview
YAML-based configuration with dataclass validation, preset management, and env var substitution.

## Structure
```
config/
├── presets/           # Predefined configurations
│   ├── standard.yaml
│   ├── max_quality.yaml
│   └── fast_test.yaml
├── loader.py          # ConfigLoader class (666 lines)
└── __init__.py
```

## ConfigLoader (`loader.py`)
Heavyweight config system supporting:
- **Dataclass validation**: 9 config sections + FullConfig container
- **Preset loading**: Load base config from `presets/`
- **Config merging**: Override presets with custom YAML (recursive deep merge)
- **Env var substitution**: `${VAR}` or `${VAR:-default}`
- **Path resolution**: Relative paths resolved against base dir

### Config Dataclasses
| Section | Dataclass | Key Fields |
|---------|-----------|------------|
| hardware | HardwareConfig | sample_rate_hz, mel_bins, window_size_ms, window_step_ms, clip_duration_ms |
| paths | PathsConfig | positive_dir, negative_dir, checkpoint_dir, export_dir |
| training | TrainingConfig | learning_rates, training_steps, batch_size, eval_step_interval |
| model | ModelConfig | architecture, first_conv_filters, pointwise_filters, mixconv_kernel_sizes |
| augmentation | AugmentationConfig | EQ, distortion, pitch shift, background noise, RIR params |
| performance | PerformanceConfig | gpu_only, mixed_precision, profiling, memory_growth |
| speaker_clustering | SpeakerClusteringConfig | embedding model, similarity threshold, n_clusters |
| hard_negative_mining | HardNegativeMiningConfig | fp_threshold, mining_interval, max_samples |
| export | ExportConfig | wake_word name, author, quantize, tensor_arena_size |

### FullConfig
Container dataclass aggregating all 9 sections. Accessed as `config.training.batch_size`, `config.model.architecture`, etc.

### ConfigLoader Methods
| Method | Purpose |
|--------|---------|
| `load()` | Load raw YAML file |
| `load_preset()` | Load from presets/ directory |
| `merge()` | Deep-merge two config dicts |
| `load_and_merge()` | Load preset + apply overrides |
| `validate()` | Validate all fields (architecture names, list lengths, ranges) |
| `to_dataclass()` | Convert dict to FullConfig dataclass tree |

### Convenience Functions
```python
from config.loader import load_full_config, load_preset, load_config

config = load_preset("standard")                      # Preset only
config = load_full_config("standard", "override.yaml") # Preset + override
config = load_config("path/to/config.yaml")            # Direct YAML load
```

## Preset Files
| Preset | Use Case |
|--------|----------|
| `standard.yaml` | Default balanced config (20k+10k steps) |
| `max_quality.yaml` | Highest accuracy, heavy augmentation, slower training |
| `fast_test.yaml` | Quick iteration, lower quality |

## Validation Rules
- `hardware.sample_rate_hz` >= 1000
- `training.training_steps` and `learning_rates` must have same length
- `training.batch_size` > 0
- `model.architecture` must be in `["mixednet"]`
- Path fields resolved and validated for existence

## Env Var Substitution
```yaml
paths:
  checkpoint_dir: ${CHECKPOINT_DIR:-./checkpoints}
  positive_dir: ${DATA_ROOT}/positive
```

## Anti-Patterns
- **Don't modify presets directly** - Create override files
- **Don't hardcode paths in code** - Use config loader
- **Don't skip validation** - Loader validates architecture names, list lengths, etc.
- **Don't import config dataclasses from src/config/** - They live in `config/loader.py`

## Notes
- Loader is 666 lines - complex but feature-complete
- Supports recursive config merging for nested dicts
- Validates against known architectures (defined in loader)
- `_substitute_env_vars()` and `_resolve_path()` handle all path/env processing
