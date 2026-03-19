"""Evaluation metrics module for wake word detection models."""

import logging
import time
from typing import Any

import numpy as np

from .fah_estimator import FAHEstimator

logger = logging.getLogger(__name__)


def apply_sliding_window_detection(
    y_scores: np.ndarray,
    threshold: float,
    sliding_window_size: int,
    clip_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ESPHome-style sliding-window detection semantics to score sequence.

    ESPHome triggers detection when:
        sum(recent_probabilities) > probability_cutoff * sliding_window_size
    which is equivalent to sliding-average > probability_cutoff.

    This helper returns a binary detection sequence where each element indicates
    whether the detection condition is satisfied at that timestep using
    trailing window ending at that timestep.

    Args:
        y_scores: Score sequence for detection
        threshold: Detection threshold
        sliding_window_size: Size of sliding window in timesteps
        clip_ids: Optional array indicating clip boundaries. If provided, resets
            sliding window at each clip boundary to prevent cross-contamination.

    Returns:
        Binary detection sequence.
    """
    # NOTE: Uses strict > comparison to match ESPHome's detection condition:
    # sum(recent_probs) > probability_cutoff * sliding_window_size
    scores = np.asarray(y_scores, dtype=float).reshape(-1)
    if sliding_window_size <= 1:
        return (scores > threshold).astype(int)

    if scores.size == 0:
        return np.array([], dtype=int)

    if clip_ids is None:
        cumsum = np.zeros(scores.size + 1, dtype=float)
        cumsum[1:] = np.cumsum(scores)
        i = np.arange(scores.size)
        start = np.maximum(0, i - sliding_window_size + 1)
        window_sum = cumsum[i + 1] - cumsum[start]
        window_len = i - start + 1
        return np.asarray((window_sum > float(threshold) * window_len).astype(int), dtype=int)

    # Validate clip_ids length matches scores
    if len(clip_ids) != scores.size:
        raise ValueError(f"clip_ids length ({len(clip_ids)}) must equal scores size ({scores.size})")

    clip_ids_arr = np.asarray(clip_ids).reshape(-1)
    detections = np.zeros(scores.size, dtype=int)
    boundaries = np.where(np.diff(clip_ids_arr) != 0)[0] + 1
    segment_starts = np.concatenate(([0], boundaries))
    segment_ends = np.concatenate((boundaries, [scores.size]))

    for seg_start, seg_end in zip(segment_starts, segment_ends, strict=False):
        seg_scores = scores[seg_start:seg_end]
        if seg_scores.size == 0:
            continue
        cumsum = np.zeros(seg_scores.size + 1, dtype=float)
        cumsum[1:] = np.cumsum(seg_scores)
        i = np.arange(seg_scores.size)
        start = np.maximum(0, i - sliding_window_size + 1)
        window_sum = cumsum[i + 1] - cumsum[start]
        window_len = i - start + 1
        detections[seg_start:seg_end] = (window_sum > float(threshold) * window_len).astype(int)

    return detections


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute classification accuracy, optionally weighted per sample."""
    if len(y_true) == 0:
        return 0.0
    correct = (y_true == y_pred).astype(float)
    if sample_weight is None:
        return float(np.mean(correct))

    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(weights) != len(correct):
        raise ValueError(f"sample_weight length ({len(weights)}) must match number of samples ({len(correct)})")

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 0.0
    return float(np.sum(correct * weights) / total_weight)


def _binarize_labels(y_true: np.ndarray) -> np.ndarray:
    """Binarize labels: treat label==1 as positive, everything else as negative.

    This ensures consistent handling of hard-negative label 2 across all metrics.
    """
    return np.asarray((y_true == 1).astype(np.int32))


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC score."""
    # Use helper for consistent binarization
    y_true = _binarize_labels(y_true)
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
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score, optionally weighted per sample."""
    if sample_weight is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != len(y_true):
            raise ValueError(f"sample_weight length ({len(weights)}) must match number of samples ({len(y_true)})")

    tp = np.sum(weights * ((y_true == 1) & (y_pred == 1)).astype(float))
    fp = np.sum(weights * ((y_true == 0) & (y_pred == 1)).astype(float))
    fn = np.sum(weights * ((y_true == 1) & (y_pred == 0)).astype(float))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(precision), float(recall), float(f1)


def _synchronize_output(output: Any) -> None:
    """Force eager execution completion for timing accuracy."""
    if hasattr(output, "numpy"):
        _ = output.numpy()
        return
    if isinstance(output, dict):
        for item in output.values():
            _synchronize_output(item)
        return
    if isinstance(output, (list, tuple)):
        for item in output:
            _synchronize_output(item)


def compute_latency(
    model: Any,
    input_shape: tuple[int, ...],
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> dict[str, float]:
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

    latencies_arr = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies_arr)),
        "std_ms": float(np.std(latencies_arr)),
        "min_ms": float(np.min(latencies_arr)),
        "max_ms": float(np.max(latencies_arr)),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
    }


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 101,
    sliding_window_size: int = 1,
    clip_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute ROC and PR curves at multiple thresholds."""
    thresholds = np.linspace(0, 1, n_thresholds)

    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    for thresh in thresholds:
        if sliding_window_size > 1:
            y_pred = apply_sliding_window_detection(
                y_scores,
                float(thresh),
                sliding_window_size,
                clip_ids=clip_ids,
            )
        else:
            y_pred = (y_scores > thresh).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))  # noqa: F841
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


def _compute_thresholds(y_scores: np.ndarray, n_thresholds: int | None = None) -> np.ndarray:
    """Compute threshold sweep values.

    Args:
        y_scores: Prediction scores.
        n_thresholds: If provided, use fixed linspace for backward compatibility.
            If None, use score-adaptive thresholds derived from unique score values.

    Returns:
        Sorted threshold array in [0, 1].
    """
    if n_thresholds is not None:
        return np.linspace(0, 1, n_thresholds)

    scores = np.asarray(y_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    unique_scores = np.unique(np.clip(scores, 0.0, 1.0))

    # Keep adaptive mode bounded for dense float outputs:
    # 333 unique values * 3 offsets + 2 boundaries = 1001 (pre-dedup).
    max_unique_values = 333
    if unique_scores.size > max_unique_values:
        sample_idx = np.linspace(0, unique_scores.size - 1, max_unique_values).astype(int)
        unique_scores = unique_scores[np.unique(sample_idx)]

    eps = 1e-7
    thresholds = np.concatenate(
        [
            np.array([0.0, 1.0], dtype=float),
            unique_scores - eps,
            unique_scores,
            unique_scores + eps,
        ]
    )
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))

    max_thresholds = 1000
    if thresholds.size > max_thresholds:
        sample_idx = np.linspace(0, thresholds.size - 1, max_thresholds).astype(int)
        thresholds = thresholds[np.unique(sample_idx)]

    return np.asarray(thresholds)


def compute_recall_at_no_faph(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int | None = None,
    sliding_window_size: int = 1,
    clip_ids: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute recall at the lowest threshold yielding zero false positives."""
    thresholds = _compute_thresholds(y_scores, n_thresholds=n_thresholds)

    y_pred_curr = np.array([], dtype=int)

    for thresh in thresholds:
        neg_mask = y_true == 0
        if sliding_window_size > 1:
            y_pred_curr = apply_sliding_window_detection(
                y_scores,
                float(thresh),
                sliding_window_size,
                clip_ids=clip_ids,
            )
            fp = np.sum((y_pred_curr == 1) & neg_mask)
        else:
            fp = np.sum(y_scores[neg_mask] > thresh)

        if fp == 0:
            pos_mask = y_true == 1
            if sliding_window_size > 1:
                tp = np.sum((y_pred_curr == 1) & pos_mask)
            else:
                tp = np.sum(y_scores[pos_mask] > thresh)
            recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
            return float(recall), float(thresh)

    thresh = float(thresholds[-1])
    pos_mask = y_true == 1
    if sliding_window_size > 1:
        y_pred = apply_sliding_window_detection(
            y_scores,
            float(thresh),
            sliding_window_size,
            clip_ids=clip_ids,
        )
        tp = np.sum((y_pred == 1) & pos_mask)
    else:
        tp = np.sum(y_scores[pos_mask] > thresh)
    recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
    return float(recall), thresh


def compute_recall_at_target_fah(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    target_fah: float,
    n_thresholds: int | None = None,
    sliding_window_size: int = 1,
    clip_ids: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute the best recall achievable while meeting target FAH.

    This selects the operating point with maximum recall among all thresholds
    satisfying ``fah <= target_fah``.
    """
    thresholds = _compute_thresholds(y_scores, n_thresholds=n_thresholds)
    neg_mask = y_true == 0
    pos_mask = y_true == 1
    pos_total = np.sum(pos_mask)
    best_threshold = float(thresholds[-1])
    best_recall = 0.0
    best_fah = float("inf")

    y_pred_curr = np.array([], dtype=int)

    for thresh in thresholds:
        if sliding_window_size > 1:
            y_pred_curr = apply_sliding_window_detection(
                y_scores,
                float(thresh),
                sliding_window_size,
                clip_ids=clip_ids,
            )
            fp = np.sum((y_pred_curr == 1) & neg_mask)
        else:
            fp = np.sum(y_scores[neg_mask] > thresh)
        fah = fp / ambient_duration_hours if ambient_duration_hours > 0 else float("inf")
        if fah <= target_fah:
            if sliding_window_size > 1:
                tp = np.sum((y_pred_curr == 1) & pos_mask)
            else:
                tp = np.sum(y_scores[pos_mask] > thresh)
            recall = tp / pos_total if pos_total > 0 else 0.0
            recall_f = float(recall)
            fah_f = float(fah)
            thresh_f = float(thresh)

            if recall_f > best_recall:
                best_threshold = thresh_f
                best_recall = recall_f
                best_fah = fah_f
            elif np.isclose(recall_f, best_recall) and fah_f < best_fah:
                best_threshold = thresh_f
                best_fah = fah_f

    return float(best_recall), float(best_threshold), float(best_fah)


def compute_fah_at_target_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    target_recall: float,
    n_thresholds: int = 101,
    sliding_window_size: int = 1,
    clip_ids: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute FAH at the best (highest) threshold meeting target recall.

    Sweeps thresholds from low to high. Among all thresholds where
    recall >= target_recall, selects the one with the HIGHEST threshold
    (which gives the LOWEST / best FAH). This is the optimal operating
    point for deployment: meeting the recall target with fewest false alarms.

    Returns:
        (best_fah, best_threshold, best_recall) tuple.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    neg_mask = y_true == 0
    pos_mask = y_true == 1
    pos_total = np.sum(pos_mask)
    best_threshold = float(thresholds[-1])
    best_recall = 0.0
    best_fah = float("inf")

    y_pred_curr = np.array([], dtype=int)

    for thresh in thresholds:
        if sliding_window_size > 1:
            y_pred_curr = apply_sliding_window_detection(
                y_scores,
                float(thresh),
                sliding_window_size,
                clip_ids=clip_ids,
            )
            tp = np.sum((y_pred_curr == 1) & pos_mask)
        else:
            tp = np.sum(y_scores[pos_mask] > thresh)
        recall = tp / pos_total if pos_total > 0 else 0.0
        if recall >= target_recall:
            if sliding_window_size > 1:
                fp = np.sum((y_pred_curr == 1) & neg_mask)
            else:
                fp = np.sum(y_scores[neg_mask] > thresh)
            fah = fp / ambient_duration_hours if ambient_duration_hours > 0 else float("inf")
            # Always update — later (higher) thresholds have lower FAH,
            # so the last feasible threshold gives the best operating point
            best_threshold = float(thresh)
            best_recall = float(recall)
            best_fah = float(fah)
            # Do NOT break — continue to find higher threshold with lower FAH

    return float(best_fah), float(best_threshold), float(best_recall)


def compute_average_viable_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    max_fah: float = 10.0,
    n_thresholds: int | None = None,
    sliding_window_size: int = 1,
    clip_ids: np.ndarray | None = None,
) -> float:
    """Compute average viable recall (AUC of recall vs normalized FAH)."""
    thresholds = _compute_thresholds(y_scores, n_thresholds=n_thresholds)

    recalls = []
    fahs = []

    for thresh in thresholds:
        if sliding_window_size > 1:
            y_pred = apply_sliding_window_detection(
                y_scores,
                float(thresh),
                sliding_window_size,
                clip_ids=clip_ids,
            )
        else:
            y_pred = (y_scores > thresh).astype(int)

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

    fahs_arr = np.array(fahs)
    recalls_arr = np.array(recalls)

    sort_idx = np.argsort(fahs_arr)
    fahs_arr = fahs_arr[sort_idx]
    recalls_arr = recalls_arr[sort_idx]

    avg_recall = np.trapz(recalls_arr, fahs_arr / max_fah)
    return float(avg_recall)


class MetricsCalculator:
    """Class-based entry point for evaluation metrics computation."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray | None = None,
        y_score: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        clip_ids: np.ndarray | None = None,
        **opts: Any,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score
        self.sample_weight = sample_weight
        self.clip_ids = clip_ids
        self.opts = opts
        self.fah_estimator = FAHEstimator(ambient_duration_hours=opts.get("ambient_duration_hours"))
        self.sliding_window_size = int(opts.get("sliding_window_size", 1) or 1)

    def compute_fah_metrics(
        self,
        threshold: float = 0.5,
        ambient_duration_hours: float | None = None,
    ) -> dict[str, Any]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_fah_metrics requires y_score")
        effective_scores = self.y_score
        if self.sliding_window_size > 1:
            effective_scores = apply_sliding_window_detection(
                self.y_score,
                threshold,
                self.sliding_window_size,
                clip_ids=self.clip_ids,
            )
            threshold = 0.5

        return self.fah_estimator.compute_fah_metrics(
            self.y_true,
            effective_scores,
            threshold=threshold,
            ambient_duration_hours=ambient_duration_hours,
        )

    def compute_roc_pr_curves(self, n_thresholds: int = 101) -> dict[str, np.ndarray]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_roc_pr_curves requires y_score")
        return compute_roc_pr_curves(
            self.y_true,
            self.y_score,
            n_thresholds,
            sliding_window_size=self.sliding_window_size,
            clip_ids=self.clip_ids,
        )

    def compute_average_viable_recall(
        self,
        ambient_duration_hours: float,
        max_fah: float = 10.0,
        n_thresholds: int | None = None,
    ) -> float:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_average_viable_recall requires y_score")
        return compute_average_viable_recall(
            self.y_true,
            self.y_score,
            ambient_duration_hours=ambient_duration_hours,
            max_fah=max_fah,
            n_thresholds=n_thresholds,
            sliding_window_size=self.sliding_window_size,
            clip_ids=self.clip_ids,
        )

    def compute_recall_at_no_faph(self, n_thresholds: int | None = None) -> tuple[float, float]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_recall_at_no_faph requires y_score")
        return compute_recall_at_no_faph(
            self.y_true,
            self.y_score,
            n_thresholds=n_thresholds,
            sliding_window_size=self.sliding_window_size,
            clip_ids=self.clip_ids,
        )

    def compute_recall_at_target_fah(
        self,
        ambient_duration_hours: float,
        target_fah: float,
        n_thresholds: int | None = None,
    ) -> tuple[float, float, float]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_recall_at_target_fah requires y_score")
        return compute_recall_at_target_fah(
            self.y_true,
            self.y_score,
            ambient_duration_hours=ambient_duration_hours,
            target_fah=target_fah,
            n_thresholds=n_thresholds,
            sliding_window_size=self.sliding_window_size,
            clip_ids=self.clip_ids,
        )

    def compute_fah_at_target_recall(
        self,
        ambient_duration_hours: float,
        target_recall: float,
        n_thresholds: int = 101,
    ) -> tuple[float, float, float]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_fah_at_target_recall requires y_score")
        return compute_fah_at_target_recall(
            self.y_true,
            self.y_score,
            ambient_duration_hours=ambient_duration_hours,
            target_recall=target_recall,
            n_thresholds=n_thresholds,
            sliding_window_size=self.sliding_window_size,
            clip_ids=self.clip_ids,
        )

    def compute_all_metrics(
        self,
        ambient_duration_hours: float = 0.0,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        if self.y_score is None:
            raise ValueError("MetricsCalculator.compute_all_metrics requires y_score")

        if self.sliding_window_size > 1:
            y_pred = apply_sliding_window_detection(
                self.y_score,
                threshold,
                self.sliding_window_size,
                clip_ids=self.clip_ids,
            )
        else:
            y_pred = (self.y_score > threshold).astype(int)

        accuracy = compute_accuracy(self.y_true, y_pred, sample_weight=self.sample_weight)
        precision, recall, f1 = compute_precision_recall(
            self.y_true,
            y_pred,
            sample_weight=self.sample_weight,
        )
        auc_roc = compute_roc_auc(self.y_true, self.y_score)

        auc_pr: float | None = None
        valid_mask = self.y_true != 2
        y_true_pr = _binarize_labels(self.y_true[valid_mask])
        y_score_pr = self.y_score[valid_mask]

        if len(np.unique(y_true_pr)) >= 2:
            try:
                from sklearn.metrics import average_precision_score

                auc_pr = float(average_precision_score(y_true_pr, y_score_pr))
            except ImportError:
                logger.warning("sklearn not available; setting auc_pr=None in compute_all_metrics")
        else:
            logger.warning("Only one class present in PR-AUC labels; auc_pr is undefined and set to None")

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
        }

        if ambient_duration_hours > 0:
            # Pass ambient_duration_hours explicitly instead of mutating estimator state
            metrics.update(
                self.compute_fah_metrics(
                    threshold=threshold,
                    ambient_duration_hours=ambient_duration_hours,
                )
            )

            recall_no_faph, thresh_no_faph = self.compute_recall_at_no_faph()
            metrics["recall_at_no_faph"] = recall_no_faph
            metrics["threshold_for_no_faph"] = thresh_no_faph

            metrics["average_viable_recall"] = self.compute_average_viable_recall(ambient_duration_hours=ambient_duration_hours)

            # Guardrail: surface clearly poor deployment operating points.
            fah = metrics.get("false_activations_per_hour")
            recall_at_threshold = metrics.get("recall_at_threshold")
            if isinstance(fah, (float, int)) and isinstance(recall_at_threshold, (float, int)):
                if fah > 0.5 or recall_at_threshold < 0.90:
                    logger.warning(
                        "Potential soft-break risk for deployment threshold %.4f: FAH=%.4f, recall_at_threshold=%.4f. "
                        "Consider re-tuning probability_cutoff/sliding_window_size with stream-ordered evaluation data.",
                        threshold,
                        float(fah),
                        float(recall_at_threshold),
                    )

        return metrics

    def compute_latency(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        n_runs: int = 100,
        warmup_runs: int = 10,
    ) -> dict[str, float]:
        return compute_latency(model, input_shape, n_runs=n_runs, warmup_runs=warmup_runs)

    def compute_precision_recall(self) -> tuple[float, float, float]:
        if self.y_pred is None:
            raise ValueError("MetricsCalculator.compute_precision_recall requires y_pred")
        return compute_precision_recall(self.y_true, self.y_pred, sample_weight=self.sample_weight)


# Backward-compatible wrappers


def compute_fah_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float,
    threshold: float = 0.5,
) -> dict[str, Any]:
    calculator = MetricsCalculator(
        y_true=y_true,
        y_score=y_scores,
        ambient_duration_hours=ambient_duration_hours,
        sliding_window_size=1,
    )
    return calculator.compute_fah_metrics(threshold=threshold)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ambient_duration_hours: float = 0.0,
    threshold: float = 0.5,
    sample_weight: np.ndarray | None = None,
    sliding_window_size: int = 1,
) -> dict[str, Any]:
    calculator = MetricsCalculator(
        y_true=y_true,
        y_score=y_scores,
        sample_weight=sample_weight,
        sliding_window_size=sliding_window_size,
    )
    return calculator.compute_all_metrics(
        ambient_duration_hours=ambient_duration_hours,
        threshold=threshold,
    )
