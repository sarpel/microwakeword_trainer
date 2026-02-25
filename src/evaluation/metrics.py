"""Evaluation metrics module for wake word detection models.

Provides comprehensive metrics including:
- Standard classification metrics (accuracy, precision, recall, F1)
- ROC/PR curve computation
- Wake word specific metrics (FAH, average_viable_recall)
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)

    Returns:
        Accuracy score (0 to 1)
    """
    if len(y_true) == 0:
        return 0.0

    return np.mean((y_true == y_pred).astype(float))


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC score.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores/probabilities (0 to 1)

    Returns:
        AUC-ROC score (0 to 1)
    """
    if len(np.unique(y_true)) < 2:
        # Only one class present - AUC is undefined
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(y_true, y_scores)
    except ImportError:
        # Fallback: manual ROC computation
        return _manual_roc_auc(y_true, y_scores)


def _manual_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Manually compute ROC AUC using trapezoidal rule."""
    # Sort by scores descending
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    # Calculate TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    # Add origin point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Tuple of (precision, recall, f1)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def compute_latency(
    model: Any,
    input_shape: Tuple[int, ...],
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Compute inference latency statistics.

    Args:
        model: Model to evaluate (TensorFlow/Keras or similar with __call__)
        input_shape: Input shape for a single inference
        n_runs: Number of runs for timing
        warmup_runs: Number of warmup runs before timing

    Returns:
        Latency statistics including mean, std, min, max (in milliseconds)
    """
    import tensorflow as tf

    # Prepare dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)

    # Synchronize GPU
    if tf.config.list_physical_devices("GPU"):
        tf.keras.backend.sync_to_session()

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = model(dummy_input, training=False)

        # Synchronize for accurate timing
        if tf.config.list_physical_devices("GPU"):
            tf.keras.backend.sync_to_session()

        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def compute_fah_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute false accepts per hour (FAH) metrics.

    Args:
        y_true: True labels for ambient/negative audio
        y_scores: Prediction scores
        ambient_duration_hours: Total duration of ambient audio in hours
        threshold: Classification threshold

    Returns:
        Dict with FAH metrics
    """
    # Predict at threshold
    y_pred = (y_scores >= threshold).astype(int)

    # Count false positives
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # Calculate FAH
    fah = fp / ambient_duration_hours if ambient_duration_hours > 0 else 0.0

    return {
        "ambient_false_positives": int(fp),
        "ambient_false_positives_per_hour": float(fah),
        "ambient_duration_hours": float(ambient_duration_hours),
    }


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 101,
) -> Dict[str, np.ndarray]:
    """Compute ROC and PR curves at multiple thresholds.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        n_thresholds: Number of threshold values to compute

    Returns:
        Dict with ROC and PR curve data
    """
    thresholds = np.linspace(0, 1, n_thresholds)

    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)

    return {
        "thresholds": thresholds,
        "tpr": np.array(tpr_list),  # True positive rate (recall)
        "fpr": np.array(fpr_list),  # False positive rate
        "precision": np.array(precision_list),
        "recall": np.array(recall_list),
    }


def compute_recall_at_no_faph(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
) -> Tuple[float, float]:
    """Compute recall when FAH = 0 (no false accepts).

    Finds the highest threshold that results in zero false accepts,
    then returns the recall at that threshold.

    Args:
        y_true: True labels (positive samples)
        y_scores: Prediction scores
        ambient_duration_hours: Total duration of ambient audio in hours

    Returns:
        Tuple of (recall_at_no_faph, threshold_for_no_faph)
    """
    # Get thresholds in descending order
    thresholds = np.linspace(0, 1, 101)

    # Find threshold with zero false accepts
    for thresh in reversed(thresholds):
        # Count false positives at this threshold on negative samples
        neg_mask = y_true == 0
        fp = np.sum((y_scores[neg_mask] >= thresh))

        if fp == 0:
            # This threshold gives zero FAH
            # Calculate recall for positive samples
            pos_mask = y_true == 1
            tp = np.sum((y_scores[pos_mask] >= thresh))
            recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
            return float(recall), float(thresh)

    # If no threshold gives zero FAH, return lowest threshold result
    thresh = thresholds[0]
    pos_mask = y_true == 1
    tp = np.sum((y_scores[pos_mask] >= thresh))
    recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
    return float(recall), float(thresh)


def compute_average_viable_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    max_fah: float = 10.0,
) -> float:
    """Compute average viable recall (area under recall vs FAH curve).

    This metric summarizes model quality by measuring recall at various
    FAH levels up to max_fah. Higher is better.

    Args:
        y_true: True labels
        y_scores: Prediction scores
        ambient_duration_hours: Total duration of ambient audio in hours
        max_fah: Maximum FAH to consider (default: 10 FAH)

    Returns:
        Average viable recall (area under curve)
    """
    thresholds = np.linspace(0, 1, 101)

    recalls = []
    fahs = []

    for thresh in thresholds:
        # Get predictions at threshold
        y_pred = (y_scores >= thresh).astype(int)

        # Calculate TP, FP, FN
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate FAH
        fah = fp / ambient_duration_hours if ambient_duration_hours > 0 else 0.0

        # Only consider up to max_fah
        if fah <= max_fah:
            recalls.append(recall)
            fahs.append(fah)

    if len(recalls) < 2:
        return 0.0

    # Integrate using trapezoidal rule
    fahs = np.array(fahs)
    recalls = np.array(recalls)

    # Normalize FAH to [0, 1] for integration
    avg_recall = np.trapz(recalls, fahs / max_fah)

    return float(avg_recall)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float = 0.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        ambient_duration_hours: Duration of ambient audio (for FAH)
        threshold: Classification threshold

    Returns:
        Dictionary of all computed metrics
    """
    # Binary predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Basic classification metrics
    accuracy = compute_accuracy(y_true, y_pred)
    precision, recall, f1 = compute_precision_recall(y_true, y_pred)

    # ROC/PR metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
    except ImportError:
        auc_roc = _manual_roc_auc(y_true, y_scores)
        auc_pr = 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }

    # FAH metrics if ambient duration provided
    if ambient_duration_hours > 0:
        fah_metrics = compute_fah_metrics(
            y_true, y_scores, ambient_duration_hours, threshold
        )
        metrics.update(fah_metrics)

        # Recall at no FAH
        recall_no_faph, thresh_no_faph = compute_recall_at_no_faph(
            y_true, y_scores, ambient_duration_hours
        )
        metrics["recall_at_no_faph"] = recall_no_faph
        metrics["threshold_for_no_faph"] = thresh_no_faph

        # Average viable recall
        avg_viable_recall = compute_average_viable_recall(
            y_true, y_scores, ambient_duration_hours
        )
        metrics["average_viable_recall"] = avg_viable_recall

    return metrics
