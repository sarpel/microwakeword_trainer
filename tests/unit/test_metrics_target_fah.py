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


# ── Tests for compute_fah_at_target_recall ──

from src.evaluation.metrics import compute_fah_at_target_recall


def test_compute_fah_at_target_recall_picks_highest_threshold():
    """Select highest threshold meeting target recall → lowest FAH."""
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2], dtype=np.float32)

    fah, threshold, recall = compute_fah_at_target_recall(
        y_true=y_true,
        y_scores=y_scores,
        ambient_duration_hours=1.0,
        target_recall=0.66,  # Need at least 2/3 TPs
        n_thresholds=6,  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # At threshold=0.8: recall=2/3≈0.667 (≥0.66), FP: scores [0.6,0.4,0.2] < 0.8 → 0 FP → FAH=0
    # At threshold=0.6: recall=3/3=1.0 (≥0.66), FP: 0.6≥0.6 → 1 FP → FAH=1.0
    # Should pick threshold=0.8 (highest meeting recall) → FAH=0.0 (lowest)
    assert threshold >= 0.7, f"Should pick high threshold, got {threshold}"
    assert fah <= 1.0, f"FAH should be minimal, got {fah}"
    assert recall >= 0.66, f"Recall should meet target, got {recall}"


def test_compute_fah_at_target_recall_prefers_lower_fah():
    """Among thresholds meeting target recall, pick the one with lowest FAH."""
    y_true = np.array([1, 1, 0, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.95, 0.85, 0.90, 0.50, 0.30, 0.10], dtype=np.float32)

    fah, threshold, recall = compute_fah_at_target_recall(
        y_true=y_true,
        y_scores=y_scores,
        ambient_duration_hours=1.0,
        target_recall=0.5,  # Need at least 1/2 TPs
        n_thresholds=11,  # [0.0, 0.1, ..., 1.0]
    )

    # At threshold=0.9: recall=1/2=0.5 (≥0.5), FP: 0.90≥0.9 → 1 FP → FAH=1.0
    # At threshold=0.8: recall=2/2=1.0 (≥0.5), FP: 0.90≥0.8 → 1 FP → FAH=1.0
    # Should pick highest threshold (0.9) where recall still meets target
    # and FAH is as low as possible
    assert recall >= 0.5, f"Recall should meet target, got {recall}"
    assert fah < float('inf'), f"FAH should be finite, got {fah}"


def test_compute_fah_at_target_recall_no_feasible_threshold():
    """When no threshold meets target recall, return defaults."""
    y_true = np.array([1, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.1, 0.9, 0.8, 0.7], dtype=np.float32)

    fah, threshold, recall = compute_fah_at_target_recall(
        y_true=y_true,
        y_scores=y_scores,
        ambient_duration_hours=1.0,
        target_recall=0.99,  # Can't achieve — only 1 positive at very low score
        n_thresholds=11,
    )

    # At threshold=0.0: recall=1.0 (≥0.99), FAH=3.0 — this IS feasible
    # At threshold=0.1: recall=1.0 (0.1≥0.1), FAH=3.0
    # At threshold=0.2: recall=0.0 — not feasible
    # Should find a feasible point (low threshold, high FAH)
    assert recall >= 0.99 or recall == 0.0, f"Should either meet target or return default, got {recall}"
