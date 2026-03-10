"""Auto-tuning module for wake word model post-training optimization."""

from src.tuning.autotuner import (
    STRATEGY_ARMS,
    AnnealingController,
    AutoTuner,
    CandidateState,
    ErrorMemory,
    FocusedSampler,
    ParetoArchive,
    StirController,
    StrategyArm,
    ThompsonSampler,
    ThresholdOptimizer,
    TuneMetrics,
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
