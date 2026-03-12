"""Unit tests for weighted evaluation metrics behavior."""

import numpy as np

from src.evaluation.metrics import (
    MetricsCalculator,
    compute_accuracy,
    compute_all_metrics,
    compute_precision_recall,
)


def test_compute_accuracy_uses_sample_weights():
    y_true = np.array([1, 1, 0], dtype=np.int32)
    y_pred = np.array([1, 0, 0], dtype=np.int32)
    sample_weight = np.array([1.0, 10.0, 1.0], dtype=np.float32)

    acc_unweighted = compute_accuracy(y_true, y_pred)
    acc_weighted = compute_accuracy(y_true, y_pred, sample_weight=sample_weight)

    assert np.isclose(acc_unweighted, 2 / 3)
    assert np.isclose(acc_weighted, 2 / 12)


def test_compute_precision_recall_uses_sample_weights():
    y_true = np.array([1, 1, 0, 0], dtype=np.int32)
    y_pred = np.array([1, 0, 1, 0], dtype=np.int32)
    sample_weight = np.array([1.0, 3.0, 2.0, 1.0], dtype=np.float32)

    precision, recall, f1 = compute_precision_recall(y_true, y_pred, sample_weight=sample_weight)

    # Weighted counts: TP=1, FP=2, FN=3 -> precision=1/3, recall=1/4
    assert np.isclose(precision, 1 / 3)
    assert np.isclose(recall, 1 / 4)
    assert np.isclose(f1, 2 * (1 / 3) * (1 / 4) / ((1 / 3) + (1 / 4)))


def test_compute_all_metrics_propagates_sample_weight():
    y_true = np.array([1, 1, 0], dtype=np.int32)
    y_scores = np.array([0.9, 0.6, 0.7], dtype=np.float32)
    sample_weight = np.array([1.0, 10.0, 1.0], dtype=np.float32)

    unweighted = compute_all_metrics(y_true=y_true, y_scores=y_scores, threshold=0.65)
    weighted = compute_all_metrics(
        y_true=y_true,
        y_scores=y_scores,
        threshold=0.65,
        sample_weight=sample_weight,
    )

    # threshold 0.65 => y_pred = [1,0,1]
    # Unweighted: correct=[1,0,0] => 1/3
    assert np.isclose(unweighted["accuracy"], 1 / 3)
    # Weighted: correct*weights=[1,0,0], sum(weights)=12 => 1/12
    assert np.isclose(weighted["accuracy"], 1 / 12)
    # Weighted precision/recall as in test above with first 3 items -> TP=1, FP=1, FN=10
    assert np.isclose(weighted["precision"], 1 / 2)
    assert np.isclose(weighted["recall"], 1 / 11)


def test_metrics_calculator_precision_recall_uses_sample_weight():
    y_true = np.array([1, 1, 0], dtype=np.int32)
    y_pred = np.array([1, 0, 1], dtype=np.int32)
    sample_weight = np.array([1.0, 4.0, 2.0], dtype=np.float32)

    calculator = MetricsCalculator(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    precision, recall, _ = calculator.compute_precision_recall()

    # TP=1, FP=2, FN=4 => precision=1/3, recall=1/5
    assert np.isclose(precision, 1 / 3)
    assert np.isclose(recall, 1 / 5)


def test_weighted_metrics_raise_on_length_mismatch():
    y_true = np.array([1, 0], dtype=np.int32)
    y_pred = np.array([1, 0], dtype=np.int32)
    bad_weights = np.array([1.0], dtype=np.float32)

    try:
        compute_accuracy(y_true, y_pred, sample_weight=bad_weights)
        raise AssertionError("Expected ValueError for sample_weight length mismatch")
    except ValueError:
        pass

    try:
        compute_precision_recall(y_true, y_pred, sample_weight=bad_weights)
        raise AssertionError("Expected ValueError for sample_weight length mismatch")
    except ValueError:
        pass
