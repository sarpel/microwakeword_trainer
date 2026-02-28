"""
Data processing utilities for microwakeword_trainer.

Handles audio data loading, preprocessing, augmentation,
and dataset creation for wake word model training.
"""

__version__ = "2.0.0"


from src.data.tfdata_pipeline import (
    OptimizedDataPipeline,
    PrefetchGenerator,
    benchmark_pipeline,
    create_optimized_dataset,
)

__all__ = [
    "OptimizedDataPipeline",
    "PrefetchGenerator",
    "benchmark_pipeline",
    "create_optimized_dataset",
]
