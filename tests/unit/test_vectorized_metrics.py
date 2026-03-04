"""Unit tests for vectorized metrics in EvaluationMetrics."""

import numpy as np

from src.training.trainer import EvaluationMetrics


class TestVectorizedMetrics:
    """Tests for vectorized metrics computation in EvaluationMetrics."""

    def test_vectorized_metrics_correctness(self):
        """Test that vectorized metrics produce correct TP/FP/TN/FN counts."""
        metrics = EvaluationMetrics(cutoffs=[0.0, 0.5, 1.0])

        # Test data: 6 samples
        # Ground truth: [1, 1, 0, 0, 1, 0] (3 positive, 3 negative)
        # Scores:       [0.9, 0.7, 0.4, 0.2, 0.6, 0.8]
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
        y_scores = np.array([0.9, 0.7, 0.4, 0.2, 0.6, 0.8], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # At threshold 0.5:
        # Scores >= 0.5: [0.9, 0.7, 0.6, 0.8] -> predicted positive
        # Scores < 0.5:  [0.4, 0.2] -> predicted negative
        #
        # True labels:
        # [1, 1, 0, 0, 1, 0]
        #
        # At 0.5:
        # TP: true=1 and pred=1 -> indices 0, 1, 4 = 3
        # FP: true=0 and pred=1 -> index 5 = 1
        # TN: true=0 and pred=0 -> indices 2, 3 = 2
        # FN: true=1 and pred=0 -> none = 0

        assert metrics.tp_at_threshold[0.5] == 3
        assert metrics.fp_at_threshold[0.5] == 1
        assert metrics.tn_at_threshold[0.5] == 2
        assert metrics.fn_at_threshold[0.5] == 0

    def test_vectorized_metrics_equivalence_to_loop(self):
        """Test that vectorized update matches loop-based computation."""
        cutoffs = [0.0, 0.25, 0.5, 0.75, 1.0]
        metrics_vectorized = EvaluationMetrics(cutoffs=cutoffs)

        # Random test data
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100).astype(np.float32)
        y_scores = np.random.uniform(0, 1, size=100).astype(np.float32)

        # Vectorized update
        metrics_vectorized.update(y_true, y_scores)

        # Manual loop-based computation
        loop_tp = dict.fromkeys(cutoffs, 0)
        loop_fp = dict.fromkeys(cutoffs, 0)
        loop_tn = dict.fromkeys(cutoffs, 0)
        loop_fn = dict.fromkeys(cutoffs, 0)

        for cutoff in cutoffs:
            y_pred = (y_scores >= cutoff).astype(np.int32)
            for i in range(len(y_true)):
                if y_pred[i] == 1 and y_true[i] == 1:
                    loop_tp[cutoff] += 1
                elif y_pred[i] == 1 and y_true[i] == 0:
                    loop_fp[cutoff] += 1
                elif y_pred[i] == 0 and y_true[i] == 0:
                    loop_tn[cutoff] += 1
                else:  # y_pred[i] == 0 and y_true[i] == 1
                    loop_fn[cutoff] += 1

        # Compare results
        for cutoff in cutoffs:
            assert metrics_vectorized.tp_at_threshold[cutoff] == loop_tp[cutoff], f"TP mismatch at {cutoff}"
            assert metrics_vectorized.fp_at_threshold[cutoff] == loop_fp[cutoff], f"FP mismatch at {cutoff}"
            assert metrics_vectorized.tn_at_threshold[cutoff] == loop_tn[cutoff], f"TN mismatch at {cutoff}"
            assert metrics_vectorized.fn_at_threshold[cutoff] == loop_fn[cutoff], f"FN mismatch at {cutoff}"

    def test_vectorized_metrics_edge_cases_empty(self):
        """Test vectorized metrics with empty arrays."""
        metrics = EvaluationMetrics(cutoffs=[0.5])

        y_true = np.array([], dtype=np.float32)
        y_scores = np.array([], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # Should have zeros for all metrics
        assert metrics.tp_at_threshold[0.5] == 0
        assert metrics.fp_at_threshold[0.5] == 0
        assert metrics.tn_at_threshold[0.5] == 0
        assert metrics.fn_at_threshold[0.5] == 0

    def test_vectorized_metrics_edge_cases_all_positive(self):
        """Test vectorized metrics with all positive labels."""
        metrics = EvaluationMetrics(cutoffs=[0.5])

        y_true = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        y_scores = np.array([0.9, 0.7, 0.3, 0.8, 0.4], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # At 0.5: predictions [1, 1, 0, 1, 0]
        # TP: indices 0, 1, 3 = 3
        # FN: indices 2, 4 = 2 (true=1 but pred=0)
        # FP: 0 (no true negatives)
        # TN: 0 (no true negatives)

        assert metrics.tp_at_threshold[0.5] == 3
        assert metrics.fn_at_threshold[0.5] == 2
        assert metrics.fp_at_threshold[0.5] == 0
        assert metrics.tn_at_threshold[0.5] == 0

    def test_vectorized_metrics_edge_cases_all_negative(self):
        """Test vectorized metrics with all negative labels."""
        metrics = EvaluationMetrics(cutoffs=[0.5])

        y_true = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        y_scores = np.array([0.9, 0.7, 0.3, 0.8, 0.4], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # At 0.5: predictions [1, 1, 0, 1, 0]
        # FP: indices 0, 1, 3 = 3
        # TN: indices 2, 4 = 2 (true=0 and pred=0)
        # TP: 0 (no true positives)
        # FN: 0 (no true positives)

        assert metrics.fp_at_threshold[0.5] == 3
        assert metrics.tn_at_threshold[0.5] == 2
        assert metrics.tp_at_threshold[0.5] == 0
        assert metrics.fn_at_threshold[0.5] == 0

    def test_vectorized_metrics_multiple_updates(self):
        """Test that multiple update calls accumulate correctly."""
        metrics = EvaluationMetrics(cutoffs=[0.5])

        # First update
        y_true1 = np.array([1, 0, 1], dtype=np.float32)
        y_scores1 = np.array([0.8, 0.3, 0.9], dtype=np.float32)
        metrics.update(y_true1, y_scores1)

        # Second update
        y_true2 = np.array([0, 1, 0], dtype=np.float32)
        y_scores2 = np.array([0.6, 0.4, 0.2], dtype=np.float32)
        metrics.update(y_true2, y_scores2)

        # Combined:
        # y_true: [1, 0, 1, 0, 1, 0]
        # y_scores: [0.8, 0.3, 0.9, 0.6, 0.4, 0.2]
        # At 0.5: predictions [1, 0, 1, 1, 0, 0]
        # TP: indices 0, 2 = 2
        # FP: index 3 = 1
        # TN: indices 1, 5 = 2
        # FN: index 4 = 1

        assert metrics.tp_at_threshold[0.5] == 2
        assert metrics.fp_at_threshold[0.5] == 1
        assert metrics.tn_at_threshold[0.5] == 2
        assert metrics.fn_at_threshold[0.5] == 1

    def test_vectorized_metrics_flattening(self):
        """Test that update correctly flattens 2D arrays."""
        metrics = EvaluationMetrics(cutoffs=[0.5])

        # 2D arrays that should be flattened
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_scores = np.array([[0.8, 0.3], [0.4, 0.9]], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # Flattened: [1, 0, 0, 1] and [0.8, 0.3, 0.4, 0.9]
        # At 0.5: predictions [1, 0, 0, 1]
        # TP: indices 0, 3 = 2
        # TN: index 1 = 1
        # FN: index 2 = 1 (true=0 but wait, that's wrong - index 2 is true=0, pred=0 so TN)
        # Let me recalculate:
        # y_true: [1, 0, 0, 1]
        # y_pred (>=0.5): [1, 0, 0, 1]
        # TP: true=1 and pred=1 -> indices 0, 3 = 2
        # TN: true=0 and pred=0 -> indices 1, 2 = 2
        # FP: 0
        # FN: 0

        assert metrics.tp_at_threshold[0.5] == 2
        assert metrics.tn_at_threshold[0.5] == 2
        assert metrics.fp_at_threshold[0.5] == 0
        assert metrics.fn_at_threshold[0.5] == 0

    def test_vectorized_metrics_threshold_zero(self):
        """Test vectorized metrics at threshold 0.0 (all predicted positive)."""
        metrics = EvaluationMetrics(cutoffs=[0.0])

        y_true = np.array([1, 0, 1, 0], dtype=np.float32)
        y_scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # At 0.0: all scores >= 0.0, so all predicted positive
        # TP: true=1 -> indices 0, 2 = 2
        # FP: true=0 -> indices 1, 3 = 2
        # TN: 0
        # FN: 0

        assert metrics.tp_at_threshold[0.0] == 2
        assert metrics.fp_at_threshold[0.0] == 2
        assert metrics.tn_at_threshold[0.0] == 0
        assert metrics.fn_at_threshold[0.0] == 0

    def test_vectorized_metrics_threshold_one(self):
        """Test vectorized metrics at threshold 1.0 (only scores >= 1.0 predicted positive)."""
        metrics = EvaluationMetrics(cutoffs=[1.0])

        y_true = np.array([1, 0, 1, 0], dtype=np.float32)
        y_scores = np.array([0.9, 1.0, 0.5, 1.0], dtype=np.float32)

        metrics.update(y_true, y_scores)

        # At 1.0: only scores >= 1.0 are predicted positive
        # y_scores: [0.9, 1.0, 0.5, 1.0]
        # Predictions: [0, 1, 0, 1]
        # TP: true=1 and pred=1 -> index 3 = 1 (y_true[3]=0, so actually 0)
        # Let me recalculate:
        # y_true: [1, 0, 1, 0]
        # y_pred: [0, 1, 0, 1]
        # TP: index 0 is true=1 but pred=0, index 2 is true=1 but pred=0
        #     So TP = 0
        # FP: index 1 is true=0 but pred=1, index 3 is true=0 but pred=1
        #     So FP = 2
        # TN: index 0 is true=1 but pred=0 - no, TN requires true=0
        #     index 2 is true=1 but pred=0
        #     So TN = 0
        # FN: index 0 is true=1 and pred=0, index 2 is true=1 and pred=0
        #     So FN = 2

        assert metrics.tp_at_threshold[1.0] == 0
        assert metrics.fp_at_threshold[1.0] == 2
        assert metrics.tn_at_threshold[1.0] == 0
        assert metrics.fn_at_threshold[1.0] == 2

    def test_vectorized_metrics_compute_metrics(self):
        """Test that compute_metrics produces valid results."""
        metrics = EvaluationMetrics(cutoffs=[0.5], ambient_duration_hours=1.0)

        y_true = np.array([1, 1, 0, 0], dtype=np.float32)
        y_scores = np.array([0.9, 0.6, 0.4, 0.2], dtype=np.float32)

        metrics.update(y_true, y_scores)
        result = metrics.compute_metrics()

        # Should return a dictionary with metrics
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_vectorized_metrics_reset(self):
        """Test that reset clears all accumulated metrics."""
        metrics = EvaluationMetrics(cutoffs=[0.5])

        y_true = np.array([1, 0], dtype=np.float32)
        y_scores = np.array([0.8, 0.2], dtype=np.float32)

        metrics.update(y_true, y_scores)
        assert len(metrics.all_y_true) > 0

        metrics.reset()

        assert len(metrics.all_y_true) == 0
        assert len(metrics.all_y_scores) == 0
        assert metrics.tp_at_threshold[0.5] == 0
        assert metrics.fp_at_threshold[0.5] == 0
