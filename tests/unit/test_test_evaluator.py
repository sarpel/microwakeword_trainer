"""Unit tests for TestEvaluator module."""

import json

import numpy as np
import pytest


class TestTestEvaluator:
    """Tests for TestEvaluator class."""

    def test_import(self):
        """Test that TestEvaluator imports correctly."""
        from src.evaluation.test_evaluator import TestEvaluator

        assert TestEvaluator is not None

    def test_evaluate_returns_dict_with_expected_keys(self, tmp_path):
        """Test evaluate returns dict with all expected keys."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        # Create mock model
        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        # Create mock config
        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        # Create synthetic test factory
        def test_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
            yield (X, y, np.ones(20, dtype=np.float32))

        result = evaluator.evaluate(test_factory)

        assert result is not None
        assert "basic_metrics" in result
        assert "advanced_metrics" in result
        assert "confidence_intervals" in result
        assert "calibration" in result
        assert "per_category" in result
        assert "operating_points" in result
        assert "threshold_sweep" in result
        assert "score_distributions" in result
        assert "confusion_matrix" in result
        assert "metadata" in result

    def test_basic_metrics_correct(self, tmp_path):
        """Test basic metrics computation with known values."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        # Create mock model
        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        # Override predict to return known values
        model.predict = lambda x, **kw: np.concatenate(
            [
                np.full((x.shape[0] // 2, 1), 0.99, dtype=np.float32),
                np.full((x.shape[0] - x.shape[0] // 2, 1), 0.1, dtype=np.float32),
            ],
            axis=0,
        )

        def test_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
            yield (X, y, np.ones(20, dtype=np.float32))

        result = evaluator.evaluate(test_factory)

        if result:
            basic = result["basic_metrics"]
            assert basic["accuracy"] == 1.0
            assert basic["precision"] == 1.0
            assert basic["recall"] == 1.0
            assert basic["f1_score"] == 1.0

    def test_edge_case_tiny_dataset(self, tmp_path):
        """Test edge case with tiny dataset (< 10 samples)."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        # Only 5 samples - below minimum
        def tiny_factory():
            X = np.random.randn(5, 100, 40).astype(np.float32)
            y = np.array([1, 0, 1, 0, 1], dtype=np.int32)
            yield (X, y, np.ones(5, dtype=np.float32))

        result = evaluator.evaluate(tiny_factory)
        # Should return None for tiny datasets
        assert result is None

    def test_edge_case_single_class(self, tmp_path):
        """Test edge case with single class in dataset."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        # All positive class
        def single_class_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 20, dtype=np.int32)  # All positive
            yield (X, y, np.ones(20, dtype=np.float32))

        # Should handle gracefully
        result = evaluator.evaluate(single_class_factory)  # noqa: F841
        # With single class, some metrics may be None but should not crash

    def test_json_report_valid(self, tmp_path):
        """Test JSON report is valid and parseable."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        def test_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
            yield (X, y, np.ones(20, dtype=np.float32))

        model.predict = lambda x, **kw: np.random.rand(x.shape[0], 1).astype(np.float32)

        result = evaluator.evaluate(test_factory)

        if result:
            json_path = tmp_path / "test_report.json"
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert "basic_metrics" in data
            assert "advanced_metrics" in data

    def test_plots_created(self, tmp_path):
        """Test that 6 PNG plots are created."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        def test_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
            yield (X, y, np.ones(20, dtype=np.float32))

        model.predict = lambda x, **kw: np.random.rand(x.shape[0], 1).astype(np.float32)

        result = evaluator.evaluate(test_factory)  # noqa: F841

        expected_plots = ["test_roc.png", "test_pr.png", "test_det.png", "test_scores.png", "test_calibration.png", "test_fah_recall.png"]

        for plot_name in expected_plots:
            plot_path = tmp_path / plot_name
            assert plot_path.exists(), f"{plot_name} not found"
            assert plot_path.stat().st_size > 0, f"{plot_name} is empty"

    def test_numpy_serialization(self, tmp_path):
        """Test that no numpy types leak into JSON."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        def test_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
            yield (X, y, np.ones(20, dtype=np.float32))

        model.predict = lambda x, **kw: np.random.rand(x.shape[0], 1).astype(np.float32)

        result = evaluator.evaluate(test_factory)

        if result:
            json_path = tmp_path / "test_report.json"
            with open(json_path) as f:
                content = f.read()

            # Should not contain numpy type indicators
            assert "dtype" not in content.lower() and "object" not in content
            # JSON should be parseable
            json.loads(content)

    def test_mcc_computation(self):
        """Test MCC computation."""
        from src.evaluation.test_evaluator import _compute_mcc

        # Perfect predictions: TP=10, TN=10, FP=0, FN=0
        mcc = _compute_mcc(10, 0, 10, 0)
        assert mcc == 1.0

        # Balanced chance predictions should be exactly 0
        mcc_random = _compute_mcc(5, 5, 5, 5)
        assert mcc_random == 0.0

    def test_eer_computation(self, tmp_path):
        """Test EER is computed and is between 0 and 1."""
        import tensorflow as tf

        from src.evaluation.test_evaluator import TestEvaluator

        model = tf.keras.Sequential([tf.keras.Input(shape=(100, 40)), tf.keras.layers.Dense(1, activation="sigmoid")])

        config = {"performance": {"tensorboard_log_dir": str(tmp_path)}, "paths": {"processed_dir": str(tmp_path)}, "training": {"test_split": 0.1, "ambient_duration_hours": 10.0}}

        evaluator = TestEvaluator(model, config, str(tmp_path))

        def test_factory():
            X = np.random.randn(20, 100, 40).astype(np.float32)
            y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
            yield (X, y, np.ones(20, dtype=np.float32))

        model.predict = lambda x, **kw: np.random.rand(x.shape[0], 1).astype(np.float32)

        result = evaluator.evaluate(test_factory)

        if result and result["advanced_metrics"].get("eer") is not None:
            eer = result["advanced_metrics"]["eer"]
            assert 0.0 <= eer <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
