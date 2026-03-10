"""Unit tests for target-FAH operating-point selection."""

import numpy as np

from src.evaluation.metrics import compute_recall_at_target_fah


def test_compute_recall_at_target_fah_picks_max_recall_feasible_threshold():
    """Select threshold that maximizes recall under FAH constraint."""
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2], dtype=np.float32)

    recall, threshold, fah = compute_recall_at_target_fah(
        y_true=y_true,
        y_scores=y_scores,
        ambient_duration_hours=1.0,
        target_fah=1.0,
        n_thresholds=6,  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    assert np.isclose(recall, 1.0)
    assert np.isclose(threshold, 0.6)
    assert np.isclose(fah, 1.0)


def test_compute_recall_at_target_fah_breaks_ties_with_lower_fah():
    """When recall ties, choose point with lower FAH."""
    y_true = np.array([1, 1, 0, 0], dtype=np.int32)
    y_scores = np.array([0.95, 0.55, 0.85, 0.65], dtype=np.float32)

    recall, threshold, fah = compute_recall_at_target_fah(
        y_true=y_true,
        y_scores=y_scores,
        ambient_duration_hours=1.0,
        target_fah=1.0,
        n_thresholds=11,  # [0.0, 0.1, ..., 1.0]
    )

    assert np.isclose(recall, 0.5)
    assert np.isclose(threshold, 0.9)
    assert np.isclose(fah, 0.0)
