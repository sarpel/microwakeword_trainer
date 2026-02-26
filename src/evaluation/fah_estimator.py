"""False-accepts-per-hour (FAH) estimation utilities."""

from typing import Any, Dict, Optional

import numpy as np


class FAHEstimator:
    """Estimate false activations per hour for wake-word evaluation."""

    def __init__(self, ambient_duration_hours: float = 0.0):
        self.ambient_duration_hours = float(ambient_duration_hours)

    def compute_fah_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
        ambient_duration_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute FAH metrics at a threshold."""
        duration_hours = (
            float(ambient_duration_hours)
            if ambient_duration_hours is not None
            else self.ambient_duration_hours
        )

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
        ambient_duration_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Alias for compute_fah_metrics for clearer external naming."""
        return self.compute_fah_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=threshold,
            ambient_duration_hours=ambient_duration_hours,
        )
