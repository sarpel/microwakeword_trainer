"""
Training utilities for microwakeword_trainer.

Provides training loops, callbacks, optimizers, and
checkpoint management for wake word model training.
"""

__version__ = "2.0.0"

# Import main classes for convenience
from .trainer import Trainer as Trainer, TrainingMetrics as TrainingMetrics, train as train, main as main
