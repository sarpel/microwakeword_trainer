"""
Data processing utilities for microwakeword_trainer.

Handles audio data loading, preprocessing, augmentation,
and dataset creation for wake word model training.
"""

__version__ = "2.0.0"


try:
    from .tfdata_pipeline import (
        OptimizedDataPipeline,
        PrefetchGenerator,
        benchmark_pipeline,
        create_optimized_dataset,
    )
except ImportError:
    OptimizedDataPipeline = None  # type: ignore[assignment]
    PrefetchGenerator = None  # type: ignore[assignment]
    benchmark_pipeline = None  # type: ignore[assignment]
    create_optimized_dataset = None  # type: ignore[assignment]

__all__ = [
    "OptimizedDataPipeline",
    "PrefetchGenerator",
    "benchmark_pipeline",
    "create_optimized_dataset",
]
