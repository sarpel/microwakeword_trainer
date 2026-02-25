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
└── loader.py          # ConfigLoader class (733 lines)
```

## ConfigLoader (`loader.py`)
Heavyweight config system supporting:
- **Dataclass validation**: 11 config sections (Hardware, Paths, Training, Model, etc.)
- **Preset loading**: Load base config from `presets/`
- **Config merging**: Override presets with custom YAML
- **Env var substitution**: `${VAR}` or `${VAR:-default}`
- **Path resolution**: Relative paths resolved against base dir

### Sections
| Section | Dataclass | Key Fields |
|---------|-----------|------------|
| hardware | HardwareConfig | sample_rate_hz, mel_bins, window_size_ms |
| paths | PathsConfig | positive_dir, negative_dir, checkpoint_dir |
| training | TrainingConfig | learning_rates, batch_size, augmentation params |
| model | ModelConfig | architecture, filters, dropout_rate |
| features | FeaturesConfig | feature sets with sampling weights |
| augmentation | AugmentationConfig | EQ, distortion, pitch shift params |
| performance | PerformanceConfig | gpu_only, mixed_precision, profiling |
| speaker_clustering | SpeakerClusteringConfig | embedding model, similarity threshold |
| hard_negative_mining | HardNegativeMiningConfig | fp_threshold, mining_interval |
| export | ExportConfig | wake_word name, quantize, tensor_arena_size |

## Usage
```python
from config.loader import load_full_config, load_preset

# Load preset only
config = load_preset("standard")

# Load preset + override
config = load_full_config("standard", "my_override.yaml")

# Direct loader
from config.loader import ConfigLoader
loader = ConfigLoader("/path/to/project")
config = loader.load_and_merge("standard", "override.yaml")
```

## Preset Files
| Preset | Use Case |
|--------|----------|
| `standard.yaml` | Default balanced config |
| `max_quality.yaml` | Highest accuracy, slower training |
| `fast_test.yaml` | Quick iteration, lower quality |

## Env Var Substitution
```yaml
paths:
  checkpoint_dir: ${CHECKPOINT_DIR:-./models}
  positive_dir: ${DATA_ROOT}/positive
```

## Anti-Patterns
- **Don't modify presets directly** - Create override files
- **Don't hardcode paths in code** - Use config loader
- **Don't skip validation** - Loader validates architecture names, etc.

## Notes
- Loader is 733 lines - complex but feature-complete
- Supports recursive config merging for nested dicts
- Validates against known architectures (defined in loader)
