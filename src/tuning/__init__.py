"""Auto-tuning module for wake word models."""

from src.tuning.dashboard import TuningDashboard, save_artifacts
from src.tuning.knobs import (
    KnobCycle,
    LabelSmoothingKnob,
    LRKnob,
    SamplingMixKnob,
    TemperatureKnob,
    ThresholdKnob,
    WeightPerturbationKnob,
)
from src.tuning.metrics import (
    ErrorMemory,
    ParetoArchive,
    TuneMetrics,
    compute_hypervolume,
)
from src.tuning.orchestrator import MicroAutoTuner
from src.tuning.population import Candidate, Population, partition_data


def autotune(checkpoint_path, config, auto_tuning_config=None, **kwargs) -> dict:
    """Convenience wrapper around MicroAutoTuner.tune()."""
    tuner = MicroAutoTuner(
        checkpoint_path,
        config,
        auto_tuning_config or {},
        **kwargs,
    )
    return tuner.tune()


__all__ = [
    "MicroAutoTuner",
    "autotune",
    "TuneMetrics",
    "ParetoArchive",
    "ErrorMemory",
    "compute_hypervolume",
    "Candidate",
    "Population",
    "partition_data",
    "KnobCycle",
    "LRKnob",
    "ThresholdKnob",
    "TemperatureKnob",
    "SamplingMixKnob",
    "WeightPerturbationKnob",
    "LabelSmoothingKnob",
    "TuningDashboard",
    "save_artifacts",
]
