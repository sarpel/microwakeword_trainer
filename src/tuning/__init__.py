"""Auto-tuning module for wake word model post-training optimization."""

from src.tuning.autotuner import (
    AutoTuner,
    ParetoArchive,
    TuneMetrics,
    CandidateState,
    StrategyArm,
    STRATEGY_ARMS,
    ThompsonSampler,
    ErrorMemory,
    FocusedSampler,
    ThresholdOptimizer,
    AnnealingController,
    StirController,
    autotune,
    main,
)

__all__ = [
    "AutoTuner",
    "ParetoArchive",
    "TuneMetrics",
    "CandidateState",
    "StrategyArm",
    "STRATEGY_ARMS",
    "ThompsonSampler",
    "ErrorMemory",
    "FocusedSampler",
    "ThresholdOptimizer",
    "AnnealingController",
    "StirController",
    "autotune",
    "main",
]
