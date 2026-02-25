"""Evaluation metrics module."""

import numpy as np
from typing import Dict


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    pass


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC score.

    Args:
        y_true: True labels
        y_scores: Prediction scores

    Returns:
        AUC score
    """
    pass


def compute_latency(model, input_shape: tuple, n_runs: int = 100) -> Dict[str, float]:
    """Compute inference latency.

    Args:
        model: Model to evaluate
        input_shape: Input shape
        n_runs: Number of runs

    Returns:
        Latency statistics
    """
    pass
