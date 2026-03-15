# config/ - Configuration System

YAML-based configuration with dataclass validation, preset management, and env var substitution.

## Structure

```
config/
├── presets/
│   ├── standard.yaml
│   ├── max_quality.yaml
│   └── fast_test.yaml
└── loader.py          # ConfigLoader class
```

## ConfigLoader

- **15 dataclass sections**: Hardware, Paths, Training, Model, Augmentation, Performance, SpeakerClustering, Mining, Export, Preprocessing, Quality, Evaluation, AutoTune, AutoTuningExpert (FullConfig is the container)
- **Preset loading**: Load base from `presets/`
- **Config merging**: Override with custom YAML
- **Env var substitution**: `${VAR}` or `${VAR:-default}`

## Config Sections

| Section | Key Fields |
|---------|------------|
| hardware | sample_rate_hz, mel_bins, window_size_ms, window_step_ms |
| training | learning_rates, training_steps, batch_size |
| model | architecture, first_conv_filters, mixconv_kernel_sizes |
| export | wake_word, quantize, tensor_arena_size |
| auto_tuning | target_fah, target_recall, max_iterations, fine_tuning_lr, fine_tuning_steps |

### Notes
- **AutoTuneConfig vs AutoTuningExpertConfig**: AutoTuneConfig contains basic tuning parameters (targets, iterations, cross-validation). AutoTuningExpertConfig contains advanced parameters for burst steps, learning rates, SAM/SWA, simulated annealing, and curriculum learning. FullConfig is the container that combines all 15 dataclasses.
- **fine_tuning_lr** (conceptual): This concept is used in AutoTuningExpertConfig's min_lr, max_lr, and default_lr fields for gradient burst steps. The old `initial_lr` field name referred to the same concept.

## Convenience Functions

```python
from config.loader import load_full_config

config = load_full_config("standard", "override.yaml")
```

## Anti-Patterns

- **Don't modify presets directly** - Create override files
- **Don't add config only to loader.py** - Must add to ALL THREE presets
- **Don't skip validation** - Loader validates architecture names, ranges

## Related Documentation

- [Configuration Reference](../docs/CONFIGURATION.md)
