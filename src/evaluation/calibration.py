"""Calibration helpers for wake-word probability outputs."""

from typing import Dict

import numpy as np
from scipy.special import expit


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute a simple reliability diagram curve.

    Raises:
        ValueError: If n_bins < 1, shapes mismatch, or values out of range
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob must have the same length, "
            f"got {len(y_true)} vs {len(y_prob)}"
        )
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob values must be in [0, 1]")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
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
    """Compute Brier score for probabilistic binary predictions.

    Raises:
        ValueError: If shapes mismatch, or values out of range
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob must have the same length, "
            f"got {len(y_true)} vs {len(y_prob)}"
        )
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob values must be in [0, 1]")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

    return float(np.mean((y_prob - y_true) ** 2))


def calibrate_probabilities(
    y_prob: np.ndarray,
    scale: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """Apply a lightweight logistic calibration transform."""
    # Use consistent epsilon for both y_prob and its complement
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    y_prob_complement_clipped = np.clip(1 - y_prob, eps, 1 - eps)

    logits = np.log(y_prob_clipped / y_prob_complement_clipped)
    calibrated_logits = scale * logits + bias

    return expit(calibrated_logits)
