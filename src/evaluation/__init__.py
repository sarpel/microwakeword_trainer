"""
Evaluation utilities for microwakeword_trainer.

Provides metrics, evaluation pipelines, and analysis tools
for assessing wake word model performance.
"""

__version__ = "2.0.0"
from .metrics import MetricsCalculator
from .test_evaluator import TestEvaluator

__all__ = ["MetricsCalculator", "TestEvaluator"]
