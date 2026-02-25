"""Evaluation metrics module for wake word detection models."""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .fah_estimator import FAHEstimator

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_true == y_pred).astype(float)))


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC score."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_scores))
    except ImportError:
        return _manual_roc_auc(y_true, y_scores)


def _manual_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Manually compute ROC AUC using trapezoidal rule."""
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    return float(np.trapz(tpr, fpr))


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score."""
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

    return float(precision), float(recall), float(f1)


def _synchronize_output(output: Any) -> None:
    """Force eager execution completion for timing accuracy."""
    if hasattr(output, "numpy"):
        _ = output.numpy()
        return
    if isinstance(output, (list, tuple)):
        for item in output:
            _synchronize_output(item)


def compute_latency(
    model: Any,
    input_shape: Tuple[int, ...],
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Compute inference latency statistics."""
    import tensorflow as tf

    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    has_gpu = bool(tf.config.list_physical_devices("GPU"))

    for _ in range(warmup_runs):
        output = model(dummy_input, training=False)
        if has_gpu:
            _synchronize_output(output)

    latencies = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        output = model(dummy_input, training=False)

        if has_gpu:
            _synchronize_output(output)

        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

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


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 101,
) -> Dict[str, np.ndarray]:
    """Compute ROC and PR curves at multiple thresholds."""
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
        "tpr": np.array(tpr_list),
        "fpr": np.array(fpr_list),
        "precision": np.array(precision_list),
        "recall": np.array(recall_list),
    }


def compute_recall_at_no_faph(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute recall at the lowest threshold yielding zero false positives."""
    thresholds = np.linspace(0, 1, 101)

    for thresh in thresholds:
        neg_mask = y_true == 0
        fp = np.sum(y_scores[neg_mask] >= thresh)

        if fp == 0:
            pos_mask = y_true == 1
            tp = np.sum(y_scores[pos_mask] >= thresh)
            recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
            return float(recall), float(thresh)

    thresh = float(thresholds[-1])
    pos_mask = y_true == 1
    tp = np.sum(y_scores[pos_mask] >= thresh)
    recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
    return float(recall), thresh


def compute_average_viable_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    max_fah: float = 10.0,
) -> float:
    """Compute average viable recall (AUC of recall vs normalized FAH)."""
    thresholds = np.linspace(0, 1, 101)

    recalls = []
    fahs = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fah = fp / ambient_duration_hours if ambient_duration_hours > 0 else 0.0

        if fah <= max_fah:
            recalls.append(recall)
            fahs.append(fah)

    if len(recalls) < 2:
        return 0.0

    fahs = np.array(fahs)
    recalls = np.array(recalls)

    sort_idx = np.argsort(fahs)
    fahs = fahs[sort_idx]
    recalls = recalls[sort_idx]

    avg_recall = np.trapz(recalls, fahs / max_fah)
    return float(avg_recall)


class MetricsCalculator:
    """Class-based entry point for evaluation metrics computation."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        **opts: Any,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score
        self.sample_weight = sample_weight
        self.opts = opts
        self.fah_estimator = FAHEstimator(
            ambient_duration_hours=float(opts.get("ambient_duration_hours", 0.0))
        )

    def compute_fah_metrics(
        self,
        threshold: float = 0.5,
        ambient_duration_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_fah_metrics requires y_score")
        return self.fah_estimator.compute_fah_metrics(
            self.y_true,
            self.y_score,
            threshold=threshold,
            ambient_duration_hours=ambient_duration_hours,
        )

    def compute_roc_pr_curves(self, n_thresholds: int = 101) -> Dict[str, np.ndarray]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_roc_pr_curves requires y_score")
        return compute_roc_pr_curves(self.y_true, self.y_score, n_thresholds)

    def compute_average_viable_recall(
        self,
        ambient_duration_hours: float,
        max_fah: float = 10.0,
    ) -> float:
        if self.y_score is None:
            raise ValueError(
                "MetricsCalculator.compute_average_viable_recall requires y_score"
            )
        return compute_average_viable_recall(
            self.y_true,
            self.y_score,
            ambient_duration_hours=ambient_duration_hours,
            max_fah=max_fah,
        )

    def compute_recall_at_no_faph(self) -> Tuple[float, float]:
        if self.y_score is None:
            raise ValueError(
                "MetricsCalculator.compute_recall_at_no_faph requires y_score"
            )
        return compute_recall_at_no_faph(self.y_true, self.y_score)

    def compute_all_metrics(
        self,
        ambient_duration_hours: float = 0.0,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_all_metrics requires y_score")

        y_pred = (self.y_score >= threshold).astype(int)

        accuracy = compute_accuracy(self.y_true, y_pred)
        precision, recall, f1 = compute_precision_recall(self.y_true, y_pred)
        auc_roc = compute_roc_auc(self.y_true, self.y_score)

        auc_pr: Optional[float] = None
        unique_classes = np.unique(self.y_true)
        if len(unique_classes) >= 2:
            try:
                from sklearn.metrics import average_precision_score

                auc_pr = float(average_precision_score(self.y_true, self.y_score))
            except ImportError:
                logger.warning(
                    "sklearn not available; setting auc_pr=None in compute_all_metrics"
                )
        else:
            logger.warning(
                "Only one class present in y_true; auc_pr is undefined and set to None"
            )

        metrics: Dict[str, Any] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
        }

        if ambient_duration_hours > 0:
            self.fah_estimator.ambient_duration_hours = float(ambient_duration_hours)
            metrics.update(self.compute_fah_metrics(threshold=threshold))

            recall_no_faph, thresh_no_faph = self.compute_recall_at_no_faph()
            metrics["recall_at_no_faph"] = recall_no_faph
            metrics["threshold_for_no_faph"] = thresh_no_faph

            metrics["average_viable_recall"] = self.compute_average_viable_recall(
                ambient_duration_hours=ambient_duration_hours
            )

        return metrics

    def compute_latency(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        n_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        return compute_latency(
            model, input_shape, n_runs=n_runs, warmup_runs=warmup_runs
        )

    def compute_precision_recall(self) -> Tuple[float, float, float]:
        if self.y_pred is None:
            raise ValueError(
                "MetricsCalculator.compute_precision_recall requires y_pred"
            )
        return compute_precision_recall(self.y_true, self.y_pred)


# Backward-compatible wrappers


def compute_fah_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    calculator = MetricsCalculator(
        y_true=y_true,
        y_score=y_scores,
        ambient_duration_hours=ambient_duration_hours,
    )
    return calculator.compute_fah_metrics(threshold=threshold)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float = 0.0,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    calculator = MetricsCalculator(y_true=y_true, y_score=y_scores)
    return calculator.compute_all_metrics(
        ambient_duration_hours=ambient_duration_hours,
        threshold=threshold,
    )
