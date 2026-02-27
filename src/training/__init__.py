"""
Training utilities for microwakeword_trainer.

Provides training loops, callbacks, optimizers, and
checkpoint management for wake word model training.
"""

__version__ = "2.0.0"

# Import main classes for convenience
from .trainer import (
    Trainer as Trainer,
)
from .trainer import (
    TrainingMetrics as TrainingMetrics,
)
from .trainer import (
    main as main,
)
from .trainer import (
    train as train,
)
