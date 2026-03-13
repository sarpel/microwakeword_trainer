"""Unit tests for FAH estimator module."""

import numpy as np
import pytest

from src.evaluation.fah_estimator import FAHEstimator


class TestFAHEstimatorInit:
    """Tests for FAHEstimator initialization."""

    def test_init_with_valid_hours(self):
        """Test initialization with valid ambient_duration_hours."""
        estimator = FAHEstimator(ambient_duration_hours=24.0)
        assert estimator.ambient_duration_hours == 24.0

    def test_init_with_none(self):
        """Test initialization with None (deferred)."""
        estimator = FAHEstimator(ambient_duration_hours=None)
        assert estimator.ambient_duration_hours is None

    def test_init_with_zero(self):
        """Test initialization with zero hours."""
        estimator = FAHEstimator(ambient_duration_hours=0.0)
        assert estimator.ambient_duration_hours == 0.0

    def test_init_with_negative_hours_raises(self):
        """Test that negative hours raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            FAHEstimator(ambient_duration_hours=-1.0)

    def test_init_with_negative_zero_hours_raises(self):
        """Test that negative zero-equivalent raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            FAHEstimator(ambient_duration_hours=-0.1)


class TestFAHEstimatorComputeFahMetrics:
    """Tests for compute_fah_metrics method."""

    def test_basic_fah_computation(self):
        """Test basic FAH computation."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.9, 0.8])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

        assert "ambient_false_positives" in result
        assert "ambient_false_positives_per_hour" in result
        assert "ambient_duration_hours" in result
        # 2 false positives over 1 hour = 2 FAH
        assert result["ambient_false_positives"] == 2
        assert result["ambient_false_positives_per_hour"] == 2.0
        assert result["ambient_duration_hours"] == 1.0

    def test_zero_false_positives(self):
        """Test with zero false positives."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.9, 0.8])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

        assert result["ambient_false_positives"] == 0
        assert result["ambient_false_positives_per_hour"] == 0.0

    def test_zero_duration_hours(self):
        """Test with zero duration hours."""
        estimator = FAHEstimator(ambient_duration_hours=0.0)
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.1, 0.1])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

        # Should report 0 FAH when duration is 0
        assert result["ambient_false_positives_per_hour"] == 0.0

    def test_different_thresholds(self):
        """Test FAH computation with different thresholds."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_scores = np.array([0.6, 0.5, 0.4, 0.3, 0.9, 0.8])

        # At threshold 0.55: 1 FP (score 0.6 >= 0.55)
        result_high = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.55)
        assert result_high["ambient_false_positives"] == 1

        # At threshold 0.45: 2 FPs (scores 0.6, 0.5 >= 0.45)
        result_low = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.45)
        assert result_low["ambient_false_positives"] == 2

    def test_overridden_duration_hours(self):
        """Test that method parameter overrides init value."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.9, 0.9, 0.9, 0.9])

        # Override with 2.0 hours
        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5, ambient_duration_hours=2.0)

        # 4 FPs over 2 hours = 2 FAH
        assert result["ambient_false_positives"] == 4
        assert result["ambient_false_positives_per_hour"] == 2.0
        assert result["ambient_duration_hours"] == 2.0

    def test_deferred_duration_hours(self):
        """Test providing duration at method call time."""
        estimator = FAHEstimator(ambient_duration_hours=None)
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.1, 0.1])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5, ambient_duration_hours=1.0)

        assert result["ambient_false_positives_per_hour"] == 2.0

    def test_missing_duration_hours_raises(self):
        """Test that missing duration raises ValueError."""
        estimator = FAHEstimator(ambient_duration_hours=None)
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.1, 0.1])

        with pytest.raises(ValueError, match="ambient_duration_hours must be provided"):
            estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

    def test_negative_override_duration_raises(self):
        """Test that negative override duration raises ValueError."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.1, 0.1])

        with pytest.raises(ValueError, match="must be >= 0"):
            estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5, ambient_duration_hours=-1.0)

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 1])
        y_scores = np.array([0.9, 0.9])

        with pytest.raises(ValueError, match="same length"):
            estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

    def test_all_negatives(self):
        """Test with all negative labels."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

        # All predictions are FPs
        assert result["ambient_false_positives"] == 4
        assert result["ambient_false_positives_per_hour"] == 4.0

    def test_all_positives(self):
        """Test with all positive labels."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

        # No negative samples, so no FPs
        assert result["ambient_false_positives"] == 0
        assert result["ambient_false_positives_per_hour"] == 0.0

    def test_2d_array_input(self):
        """Test with 2D arrays that get flattened."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([[0, 0], [1, 1]])
        y_scores = np.array([[0.9, 0.9], [0.1, 0.1]])

        result = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)

        # 2 false positives in the flattened array
        assert result["ambient_false_positives"] == 2


class TestFAHEstimatorEstimateFalseActivations:
    """Tests for estimate_false_activations_per_hour alias."""

    def test_alias_returns_same_result(self):
        """Test that alias returns same result as compute_fah_metrics."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.9, 0.8])

        result1 = estimator.compute_fah_metrics(y_true, y_scores, threshold=0.5)
        result2 = estimator.estimate_false_activations_per_hour(y_true, y_scores, threshold=0.5)

        assert result1 == result2

    def test_alias_with_all_parameters(self):
        """Test alias with all parameters provided."""
        estimator = FAHEstimator(ambient_duration_hours=1.0)
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.1, 0.1])

        result = estimator.estimate_false_activations_per_hour(y_true, y_scores, threshold=0.5, ambient_duration_hours=2.0)

        assert result["ambient_false_positives_per_hour"] == 1.0  # 2 FPs over 2 hours
