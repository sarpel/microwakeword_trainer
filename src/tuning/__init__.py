"""Auto-tuning package for wake word model fine-tuning.

This package provides sophisticated post-training fine-tuning to achieve
target quality metrics:
- FAH (False Activations per Hour) < 0.3
- Recall > 0.92

Example:
    from src.tuning import AutoTuner

    tuner = AutoTuner(
        checkpoint_path="checkpoints/best.ckpt",
        config=config_dict,
        target_fah=0.3,
        target_recall=0.92,
    )
    result = tuner.tune()
"""

from src.tuning.autotuner import (
    AutoTuner,
    FAHReductionStrategy,
    MicroConfigAdjuster,
    TuningState,
    TuningTarget,
    autotune,
)
from src.tuning.cli import main

__all__ = [
    "AutoTuner",
    "FAHReductionStrategy",
    "MicroConfigAdjuster",
    "TuningState",
    "TuningTarget",
    "autotune",
    "main",
]
