"""Unit tests for calibration module."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    calibrate_probabilities,
    compute_brier_score,
    compute_calibration_curve,
)


class TestComputeCalibrationCurve:
    """Tests for compute_calibration_curve function."""

    def test_basic_calibration_curve(self):
        """Test basic calibration curve computation."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])

        result = compute_calibration_curve(y_true, y_prob, n_bins=5)

        assert "prob_true" in result
        assert "prob_pred" in result
        assert "counts" in result
        assert "bin_edges" in result
        assert len(result["prob_true"]) == 5
        assert len(result["counts"]) == 5

    def test_perfect_calibration(self):
        """Test with perfectly calibrated probabilities."""
        y_true = np.array([1, 0, 1, 0])
        # Probabilities match true labels exactly
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])

        result = compute_calibration_curve(y_true, y_prob, n_bins=2)

        # For perfect calibration, prob_true should match prob_pred in each bin
        assert isinstance(result["prob_true"], np.ndarray)
        assert isinstance(result["prob_pred"], np.ndarray)

    def test_n_bins_validation(self):
        """Test that n_bins must be >= 1."""
        y_true = np.array([1, 0])
        y_prob = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_calibration_curve(y_true, y_prob, n_bins=0)

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_calibration_curve(y_true, y_prob, n_bins=-1)

    def test_length_mismatch(self):
        """Test error on length mismatch."""
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="same length"):
            compute_calibration_curve(y_true, y_prob)

    def test_prob_range_validation(self):
        """Test that probabilities must be in [0, 1]."""
        y_true = np.array([1, 0])

        with pytest.raises(ValueError, match="must be in"):
            compute_calibration_curve(y_true, np.array([1.1, 0.5]))

        with pytest.raises(ValueError, match="must be in"):
            compute_calibration_curve(y_true, np.array([-0.1, 0.5]))

    def test_y_true_binary_validation(self):
        """Test that y_true must contain only 0 and 1."""
        y_prob = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
            compute_calibration_curve(np.array([1, 2]), y_prob)

        with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
            compute_calibration_curve(np.array([0.5, 0.5]), y_prob)

    def test_empty_bins(self):
        """Test handling of empty bins."""
        y_true = np.array([1, 0])
        y_prob = np.array([0.9, 0.1])

        result = compute_calibration_curve(y_true, y_prob, n_bins=10)

        # Most bins should be empty (count=0)
        assert np.sum(result["counts"] > 0) <= 2
        # Empty bins should have zero values
        empty_bins = result["counts"] == 0
        assert np.all(result["prob_true"][empty_bins] == 0)
        assert np.all(result["prob_pred"][empty_bins] == 0)

    def test_single_bin(self):
        """Test with single bin."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        result = compute_calibration_curve(y_true, y_prob, n_bins=1)

        assert len(result["prob_true"]) == 1
        assert result["counts"][0] == 4
        # All in one bin, should have 50% positive rate
        assert result["prob_true"][0] == 0.5

    def test_2d_array_input(self):
        """Test with 2D arrays that get flattened."""
        y_true = np.array([[1, 0], [0, 1]])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8]])

        result = compute_calibration_curve(y_true, y_prob, n_bins=5)

        assert isinstance(result["prob_true"], np.ndarray)
        assert np.sum(result["counts"]) == 4


class TestComputeBrierScore:
    """Tests for compute_brier_score function."""

    def test_perfect_brier_score(self):
        """Test Brier score with perfect predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])

        score = compute_brier_score(y_true, y_prob)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_worst_brier_score(self):
        """Test Brier score with worst predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])

        score = compute_brier_score(y_true, y_prob)
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_random_brier_score(self):
        """Test Brier score is between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_prob = np.random.uniform(0, 1, size=100)

        score = compute_brier_score(y_true, y_prob)
        assert 0 <= score <= 1

    def test_uncertain_predictions(self):
        """Test Brier score with uncertain (0.5) predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        score = compute_brier_score(y_true, y_prob)
        # MSE of 0.25 for each, mean = 0.25
        assert score == pytest.approx(0.25, abs=1e-10)

    def test_length_mismatch(self):
        """Test error on length mismatch."""
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="same length"):
            compute_brier_score(y_true, y_prob)

    def test_prob_range_validation(self):
        """Test that probabilities must be in [0, 1]."""
        y_true = np.array([1, 0])

        with pytest.raises(ValueError, match="must be in"):
            compute_brier_score(y_true, np.array([1.5, 0.5]))

        with pytest.raises(ValueError, match="must be in"):
            compute_brier_score(y_true, np.array([-0.5, 0.5]))

    def test_y_true_binary_validation(self):
        """Test that y_true must contain only 0 and 1."""
        y_prob = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
            compute_brier_score(np.array([1, 2]), y_prob)

    def test_2d_array_input(self):
        """Test with 2D arrays that get flattened."""
        y_true = np.array([[1, 0], [0, 1]])
        y_prob = np.array([[1.0, 0.0], [0.0, 1.0]])

        score = compute_brier_score(y_true, y_prob)
        assert score == pytest.approx(0.0, abs=1e-10)


class TestCalibrateProbabilities:
    """Tests for calibrate_probabilities function."""

    def test_identity_calibration(self):
        """Test with scale=1.0, bias=0.0 (identity transform)."""
        y_prob = np.array([0.1, 0.5, 0.9])

        calibrated = calibrate_probabilities(y_prob, scale=1.0, bias=0.0)

        # Should be approximately the same (within numerical tolerance)
        np.testing.assert_array_almost_equal(calibrated, y_prob, decimal=5)

    def test_output_range(self):
        """Test that output is always in [0, 1]."""
        y_prob = np.array([0.01, 0.5, 0.99])

        # Test various scales and biases
        for scale in [0.5, 1.0, 2.0]:
            for bias in [-1.0, 0.0, 1.0]:
                calibrated = calibrate_probabilities(y_prob, scale=scale, bias=bias)
                assert np.all(calibrated >= 0)
                assert np.all(calibrated <= 1)

    def test_monotonicity_preservation(self):
        """Test that monotonicity is preserved."""
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        calibrated = calibrate_probabilities(y_prob, scale=1.0, bias=0.0)

        # Should still be monotonically increasing
        assert np.all(np.diff(calibrated) >= -1e-10)

    def test_extreme_values(self):
        """Test with extreme probability values."""
        y_prob = np.array([0.0, 1.0])

        # Should handle extreme values without error
        calibrated = calibrate_probabilities(y_prob)

        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

    def test_near_extreme_values(self):
        """Test with values very close to 0 and 1."""
        y_prob = np.array([1e-8, 1 - 1e-8])

        calibrated = calibrate_probabilities(y_prob)

        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

    def test_prob_range_validation(self):
        """Test that input probabilities must be in [0, 1]."""
        with pytest.raises(ValueError, match="must be in"):
            calibrate_probabilities(np.array([1.1, 0.5]))

        with pytest.raises(ValueError, match="must be in"):
            calibrate_probabilities(np.array([-0.1, 0.5]))

    def test_2d_array_input(self):
        """Test with 2D arrays."""
        y_prob = np.array([[0.1, 0.5], [0.9, 0.3]])

        calibrated = calibrate_probabilities(y_prob)

        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_scale_bias_effects(self):
        """Test that scale and bias have expected effects."""
        y_prob = np.array([0.5])

        # With scale > 1, should push towards extremes
        high_scale = calibrate_probabilities(y_prob, scale=2.0, bias=0.0)
        # With scale < 1, should pull towards center
        low_scale = calibrate_probabilities(y_prob, scale=0.5, bias=0.0)

        # At 0.5, bias should shift the output
        positive_bias = calibrate_probabilities(y_prob, scale=1.0, bias=1.0)
        negative_bias = calibrate_probabilities(y_prob, scale=1.0, bias=-1.0)

        assert positive_bias[0] > 0.5
        assert negative_bias[0] < 0.5
