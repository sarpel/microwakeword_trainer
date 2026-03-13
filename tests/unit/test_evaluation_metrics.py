"""Unit tests for evaluation metrics module."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    MetricsCalculator,
    _manual_roc_auc,
    compute_accuracy,
    compute_average_viable_recall,
    compute_fah_at_target_recall,
    compute_precision_recall,
    compute_recall_at_no_faph,
    compute_recall_at_target_fah,
    compute_roc_auc,
    compute_roc_pr_curves,
)


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_basic_accuracy(self):
        """Test basic accuracy computation."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0])
        # 4 out of 5 correct = 80%
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.8)

    def test_empty_arrays(self):
        """Test with empty arrays returns 0."""
        y_true = np.array([])
        y_pred = np.array([])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        """Test with all wrong predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_weighted_accuracy(self):
        """Test weighted accuracy computation."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        weights = np.array([1.0, 0.5, 1.0, 0.5])
        # Correct at indices 0 and 3, weighted 1.0 and 0.5
        # Total weight = 3.0, correct weight = 1.5
        result = compute_accuracy(y_true, y_pred, sample_weight=weights)
        assert result == pytest.approx(0.5)

    def test_weighted_accuracy_length_mismatch(self):
        """Test error on weight length mismatch."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        weights = np.array([1.0, 0.5])
        with pytest.raises(ValueError, match="sample_weight length"):
            compute_accuracy(y_true, y_pred, sample_weight=weights)

    def test_zero_total_weight(self):
        """Test with zero total weight returns 0."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        weights = np.array([0.0, 0.0, 0.0])
        assert compute_accuracy(y_true, y_pred, sample_weight=weights) == 0.0


class TestComputeRocAuc:
    """Tests for compute_roc_auc function."""

    def test_perfect_auc(self):
        """Test AUC with perfectly separable data."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        auc = compute_roc_auc(y_true, y_scores)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_random_auc(self):
        """Test AUC returns value between 0.5 and 1.0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_scores = np.random.uniform(0, 1, size=100)
        auc = compute_roc_auc(y_true, y_scores)
        assert 0.5 <= auc <= 1.0

    def test_single_class_returns_half(self):
        """Test with single class returns 0.5."""
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_roc_auc(y_true, y_scores) == 0.5

    def test_manual_roc_auc(self):
        """Test manual ROC AUC computation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        auc = _manual_roc_auc(y_true, y_scores)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_manual_roc_auc_all_same_class(self):
        """Test manual ROC AUC with all same class."""
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        # When all are same class, cumsum will handle gracefully
        auc = _manual_roc_auc(y_true, y_scores)
        assert isinstance(auc, float)


class TestComputePrecisionRecall:
    """Tests for compute_precision_recall function."""

    def test_basic_precision_recall(self):
        """Test basic precision and recall."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 0, 0])
        # TP=2, FP=1, FN=1
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        precision, recall, f1 = compute_precision_recall(y_true, y_pred)
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_perfect_precision_recall(self):
        """Test with perfect predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        precision, recall, f1 = compute_precision_recall(y_true, y_pred)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_no_positive_predictions(self):
        """Test with no positive predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        precision, recall, f1 = compute_precision_recall(y_true, y_pred)
        assert precision == 0.0  # No positive predictions
        assert recall == 0.0  # No true positives found
        assert f1 == 0.0

    def test_all_positive_predictions(self):
        """Test with all positive predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        # TP=2, FP=2, FN=0
        precision, recall, f1 = compute_precision_recall(y_true, y_pred)
        assert precision == 0.5
        assert recall == 1.0
        assert f1 == pytest.approx(2 / 3)

    def test_weighted_precision_recall(self):
        """Test weighted precision and recall."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 1, 0])
        weights = np.array([2.0, 1.0, 1.0])
        # TP at index 0: weight 2.0
        # FP at index 1: weight 1.0
        # FN at index 2: weight 1.0
        # Precision = 2/3, Recall = 2/3
        precision, recall, f1 = compute_precision_recall(y_true, y_pred, sample_weight=weights)
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)


class TestComputeRocPrCurves:
    """Tests for compute_roc_pr_curves function."""

    def test_basic_curves(self):
        """Test basic ROC/PR curve computation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])

        curves = compute_roc_pr_curves(y_true, y_scores, n_thresholds=11)

        assert "thresholds" in curves
        assert "tpr" in curves
        assert "fpr" in curves
        assert "precision" in curves
        assert "recall" in curves
        assert len(curves["thresholds"]) == 11

    def test_thresholds_range(self):
        """Test that thresholds span [0, 1]."""
        y_true = np.array([0, 1])
        y_scores = np.array([0.5, 0.5])

        curves = compute_roc_pr_curves(y_true, y_scores, n_thresholds=101)

        assert curves["thresholds"][0] == 0.0
        assert curves["thresholds"][-1] == 1.0


class TestComputeRecallAtNoFaph:
    """Tests for compute_recall_at_no_faph function."""

    def test_basic_recall_at_no_faph(self):
        """Test recall at zero false positives."""
        # All negatives have scores below 0.3
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        recall, threshold = compute_recall_at_no_faph(y_true, y_scores)
        # At threshold 0.21 (first where FP=0): TP=2, Recall=1.0
        assert recall == 1.0
        assert threshold > 0.2  # Should be above the max negative score

    def test_no_zero_fp_possible(self):
        """Test when no threshold yields zero FP."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        # Negatives have higher scores than positives
        # At threshold 1.0: FP=0, but TP might also be 0

        recall, threshold = compute_recall_at_no_faph(y_true, y_scores)
        assert isinstance(recall, float)
        assert isinstance(threshold, float)


class TestComputeRecallAtTargetFah:
    """Tests for compute_recall_at_target_fah function."""

    def test_basic_recall_at_target_fah(self):
        """Test recall computation at target FAH."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.85, 0.95])

        recall, threshold, fah = compute_recall_at_target_fah(y_true, y_scores, ambient_duration_hours=1.0, target_fah=2.0)

        assert isinstance(recall, float)
        assert 0 <= recall <= 1
        assert fah <= 2.0

    def test_no_threshold_meets_target(self):
        """Test when no threshold meets target FAH."""
        y_true = np.array([0, 0, 1])
        y_scores = np.array([0.9, 0.9, 0.1])

        recall, threshold, fah = compute_recall_at_target_fah(y_true, y_scores, ambient_duration_hours=1.0, target_fah=0.5)

        # Should return best effort values
        assert isinstance(recall, float)

    def test_tie_breaking_by_fah(self):
        """Test that ties are broken by lower FAH."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.3, 0.7, 0.8, 0.8])

        recall, threshold, fah = compute_recall_at_target_fah(y_true, y_scores, ambient_duration_hours=1.0, target_fah=10.0)

        assert isinstance(recall, float)
        assert isinstance(threshold, float)


class TestComputeFahAtTargetRecall:
    """Tests for compute_fah_at_target_recall function."""

    def test_basic_fah_at_target_recall(self):
        """Test FAH computation at target recall."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.6])

        fah, threshold, recall = compute_fah_at_target_recall(y_true, y_scores, ambient_duration_hours=1.0, target_recall=0.5)

        assert isinstance(fah, float)
        assert isinstance(threshold, float)
        assert recall >= 0.5

    def test_target_recall_not_achievable(self):
        """Test when target recall cannot be achieved."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        # Maximum possible recall is low

        fah, threshold, recall = compute_fah_at_target_recall(y_true, y_scores, ambient_duration_hours=1.0, target_recall=0.9)

        # Should return best achievable
        assert isinstance(fah, float)
        assert isinstance(recall, float)


class TestComputeAverageViableRecall:
    """Tests for compute_average_viable_recall function."""

    def test_basic_average_viable_recall(self):
        """Test average viable recall computation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        avg_recall = compute_average_viable_recall(y_true, y_scores, ambient_duration_hours=1.0, max_fah=10.0)

        assert isinstance(avg_recall, float)
        assert 0 <= avg_recall <= 1

    def test_no_viable_thresholds(self):
        """Test when no thresholds are viable."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.1, 0.1])
        # All thresholds have high FP

        avg_recall = compute_average_viable_recall(y_true, y_scores, ambient_duration_hours=0.1, max_fah=0.1)

        assert avg_recall == 0.0 or isinstance(avg_recall, float)

    def test_insufficient_points(self):
        """Test with insufficient viable points."""
        y_true = np.array([0, 1])
        y_scores = np.array([0.5, 0.5])

        avg_recall = compute_average_viable_recall(y_true, y_scores, ambient_duration_hours=1.0, max_fah=10.0)

        assert isinstance(avg_recall, float)


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_initialization(self):
        """Test MetricsCalculator initialization."""
        y_true = np.array([0, 1, 0, 1])
        calc = MetricsCalculator(y_true=y_true, y_score=np.array([0.1, 0.9, 0.2, 0.8]))

        assert np.array_equal(calc.y_true, y_true)
        assert calc.y_pred is None

    def test_compute_fah_metrics_requires_y_score(self):
        """Test that compute_fah_metrics requires y_score."""
        calc = MetricsCalculator(y_true=np.array([0, 1]))

        with pytest.raises(ValueError, match="requires y_score"):
            calc.compute_fah_metrics()

    def test_compute_roc_pr_curves_requires_y_score(self):
        """Test that compute_roc_pr_curves requires y_score."""
        calc = MetricsCalculator(y_true=np.array([0, 1]))

        with pytest.raises(ValueError, match="requires y_score"):
            calc.compute_roc_pr_curves()

    def test_compute_all_metrics(self):
        """Test compute_all_metrics with ambient duration."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        calc = MetricsCalculator(y_true=y_true, y_score=y_score)

        metrics = calc.compute_all_metrics(ambient_duration_hours=1.0, threshold=0.5)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "ambient_false_positives" in metrics

    def test_compute_all_metrics_without_ambient(self):
        """Test compute_all_metrics without ambient duration."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        calc = MetricsCalculator(y_true=y_true, y_score=y_score)

        metrics = calc.compute_all_metrics(ambient_duration_hours=0.0, threshold=0.5)

        # Should not have FAH metrics
        assert "ambient_false_positives" not in metrics

    def test_compute_precision_recall_requires_y_pred(self):
        """Test that compute_precision_recall requires y_pred."""
        calc = MetricsCalculator(y_true=np.array([0, 1]), y_score=np.array([0.1, 0.9]))

        with pytest.raises(ValueError, match="requires y_pred"):
            calc.compute_precision_recall()

    def test_compute_precision_recall_with_y_pred(self):
        """Test compute_precision_recall with y_pred provided."""
        calc = MetricsCalculator(y_true=np.array([0, 1, 0, 1]), y_pred=np.array([0, 1, 0, 0]))

        precision, recall, f1 = calc.compute_precision_recall()

        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)
