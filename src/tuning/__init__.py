"""Auto-tuning package for wake word model fine-tuning.

This package provides sophisticated post-training fine-tuning using:
- Multi-phase optimization (AGGRESSIVE_FAH → BALANCED → PRECISION_RECALL → POLISH)
- Adaptive knob selection with impact memory
- Pareto frontier tracking
- Multi-threshold evaluation

Example:
    from src.tuning import AutoTuner

    tuner = AutoTuner(
        checkpoint_path="checkpoints/best.weights.h5",
        config=config_dict,
    )
    result = tuner.tune()
"""

from src.tuning.autotuner import (
    AdaptiveKnobController,
    AutoTuner,
    ImpactMemory,
    ParetoFrontier,
    ParetoPoint,
    Phase,
    PhaseController,
    TrendAnalyzer,
    TuneMetrics,
    autotune,
)
from src.tuning.cli import main

__all__ = [
    "AdaptiveKnobController",
    "AutoTuner",
    "ImpactMemory",
    "ParetoFrontier",
    "ParetoPoint",
    "Phase",
    "PhaseController",
    "TrendAnalyzer",
    "TuneMetrics",
    "autotune",
    "main",
]
