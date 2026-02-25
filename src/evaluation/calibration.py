"""Calibration helpers for wake-word probability outputs."""

from typing import Dict

import numpy as np


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute a simple reliability diagram curve."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    prob_true = np.zeros(n_bins, dtype=float)
    prob_pred = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_ids == i
        counts[i] = int(np.sum(mask))
        if counts[i] > 0:
            prob_true[i] = float(np.mean(y_true[mask]))
            prob_pred[i] = float(np.mean(y_prob[mask]))

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "counts": counts,
        "bin_edges": bins,
    }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score for probabilistic binary predictions."""
    return float(np.mean((y_prob - y_true) ** 2))


def calibrate_probabilities(
    y_prob: np.ndarray,
    scale: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """Apply a lightweight logistic calibration transform."""
    logits = np.log(np.clip(y_prob, 1e-7, 1 - 1e-7) / np.clip(1 - y_prob, 1e-7, 1))
    calibrated_logits = scale * logits + bias
    return 1.0 / (1.0 + np.exp(-calibrated_logits))
