"""False-accepts-per-hour (FAH) estimation utilities."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FAHEstimator:
    """Estimate false activations per hour for wake-word evaluation."""

    def __init__(self, ambient_duration_hours: float | None = None):
        """Initialize FAHEstimator.

        Args:
            ambient_duration_hours: Hours of ambient audio used for FAH calculation.
                If None, callers must pass ambient_duration_hours to compute_fah_metrics.
        """
        self.ambient_duration_hours: float | None = float(ambient_duration_hours) if ambient_duration_hours is not None else None

    def compute_fah_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
        ambient_duration_hours: float | None = None,
    ) -> dict[str, Any]:
        """Compute FAH metrics at a threshold."""
        if ambient_duration_hours is not None:
            duration_hours = float(ambient_duration_hours)
        elif self.ambient_duration_hours is not None:
            duration_hours = self.ambient_duration_hours
        else:
            raise ValueError("ambient_duration_hours must be provided either at construction " "or as an argument to compute_fah_metrics.")

        if duration_hours == 0.0:
            logger.warning("ambient_duration_hours is 0.0 \u2014 FAH will be reported as 0. " "Provide a non-zero duration for meaningful false-activation-per-hour estimates.")

        y_pred = (y_scores >= threshold).astype(int)
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fah = fp / duration_hours if duration_hours > 0 else 0.0

        return {
            "ambient_false_positives": fp,
            "ambient_false_positives_per_hour": float(fah),
            "ambient_duration_hours": float(duration_hours),
        }

    def estimate_false_activations_per_hour(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
        ambient_duration_hours: float | None = None,
    ) -> dict[str, Any]:
        """Alias for compute_fah_metrics for clearer external naming."""
        return self.compute_fah_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=threshold,
            ambient_duration_hours=ambient_duration_hours,
        )
