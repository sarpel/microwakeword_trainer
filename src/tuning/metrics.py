"""Tuning metrics module.

Provides:
- TuneMetrics: Evaluation metrics dataclass with Pareto dominance and target-meeting
- ParetoArchive: Multi-objective non-dominated archive with NEW API (metrics + candidate_id)
- ErrorMemory: Tracks persistent false-alarms and misses across iterations
- ThresholdOptimizer: 3-pass threshold optimization returning (float32, uint8, TuneMetrics)
- compute_hypervolume: Hypervolume indicator for Pareto front quality measurement
- fit_temperature / apply_temperature / compute_ece: Probability calibration utilities
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TuneMetrics
# ============================================================================


@dataclass
class TuneMetrics:
    """Evaluation metrics for a tuning candidate."""

    fah: float = float("inf")
    recall: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    ece: float = 1.0
    threshold: float = 0.5
    threshold_uint8: int = 128
    precision: float = 0.0
    f1: float = 0.0
    total_positives: int = 0
    total_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ambient_duration_hours: float = 0.0

    def dominates(self, other: "TuneMetrics") -> bool:
        """Pareto dominance: self dominates other if better-or-equal in all objectives
        and strictly better in at least one.
        Objectives: minimize fah, maximize recall, maximize auc_pr.
        """
        at_least_as_good = self.fah <= other.fah and self.recall >= other.recall and self.auc_pr >= other.auc_pr
        strictly_better = self.fah < other.fah or self.recall > other.recall or self.auc_pr > other.auc_pr
        return at_least_as_good and strictly_better

    def meets_target(self, target_fah: float, target_recall: float) -> bool:
        return self.fah <= target_fah and self.recall >= target_recall

    def to_dict(self) -> dict:
        return {
            "fah": self.fah,
            "recall": self.recall,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "ece": self.ece,
            "threshold": self.threshold,
            "threshold_uint8": self.threshold_uint8,
            "precision": self.precision,
            "f1": self.f1,
        }


# ============================================================================
# ParetoArchive — NEW API: try_add(metrics, candidate_id)
# ============================================================================


class ParetoArchive:
    """Multi-objective Pareto archive with diversity filtering.

    NEW API: stores (TuneMetrics, candidate_id) pairs.
    - try_add(metrics: TuneMetrics, candidate_id: str) -> bool
    - get_best(target_fah, target_recall) -> Optional[tuple[TuneMetrics, str]]
    - get_frontier_points() -> list[dict]
    - len(archive) -> int
    """

    def __init__(self, max_size: int = 24, diversity_threshold: float = 0.01):
        # Each entry is (TuneMetrics, candidate_id)
        self._archive: list[tuple[TuneMetrics, str]] = []
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold

    def try_add(self, metrics: TuneMetrics, candidate_id: str) -> bool:
        """Add candidate if non-dominated. Returns True if added, False if dominated."""
        # Remove entries dominated by the new metrics
        new_archive = [(m, cid) for (m, cid) in self._archive if not metrics.dominates(m)]

        # Check if new metrics is dominated by any remaining — dominated = ALWAYS rejected
        is_dominated = any(m.dominates(metrics) for (m, _) in new_archive)

        if is_dominated:
            self._archive = new_archive
            return False

        # Not dominated — add to archive
        new_archive.append((metrics, candidate_id))

        # If over capacity, prune by crowding distance
        if len(new_archive) > self.max_size:
            new_archive = self._prune_by_crowding(new_archive)

        self._archive = new_archive
        return True

    def _is_diverse(
        self,
        metrics: TuneMetrics,
        archive: list[tuple[TuneMetrics, str]],
    ) -> bool:
        if not archive:
            return True
        for m, _ in archive:
            dist = math.sqrt((metrics.fah - m.fah) ** 2 + (metrics.recall - m.recall) ** 2 + (metrics.auc_pr - m.auc_pr) ** 2)
            if dist < self.diversity_threshold:
                return False
        return True

    def _prune_by_crowding(self, archive: list[tuple[TuneMetrics, str]]) -> list[tuple[TuneMetrics, str]]:
        """Remove entries with smallest crowding distance until at max_size."""
        while len(archive) > self.max_size:
            distances = self._crowding_distances(archive)
            worst_idx = int(np.argmin(distances))
            archive.pop(worst_idx)
        return archive

    def _crowding_distances(self, archive: list[tuple[TuneMetrics, str]]) -> np.ndarray:
        n = len(archive)
        if n <= 2:
            return np.full(n, float("inf"))

        # objectives: minimize fah (negate), maximize recall, maximize auc_pr
        objectives = np.array([[-m.fah, m.recall, m.auc_pr] for (m, _) in archive])
        distances = np.zeros(n)

        for obj_idx in range(objectives.shape[1]):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range < 1e-12:
                continue

            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (objectives[sorted_indices[i + 1], obj_idx] - objectives[sorted_indices[i - 1], obj_idx]) / obj_range

        return distances

    def get_best(self, target_fah: float, target_recall: float) -> Optional[tuple[TuneMetrics, str]]:
        """Best candidate: meets targets with max recall, else closest to targets."""
        meeting = [(m, cid) for (m, cid) in self._archive if m.meets_target(target_fah, target_recall)]
        if meeting:
            return max(meeting, key=lambda x: x[0].recall)

        if not self._archive:
            return None

        # Closest to targets by scalarized distance
        def score(entry: tuple[TuneMetrics, str]) -> float:
            m = entry[0]
            fah_excess = max(0, m.fah - target_fah) / max(target_fah, 1e-8)
            recall_deficit = max(0, target_recall - m.recall) / max(target_recall, 1e-8)
            return 2.0 * fah_excess + 1.0 * recall_deficit

        return min(self._archive, key=score)

    def get_frontier_points(self) -> list[dict]:
        """Return frontier as list of dicts with fah/recall/auc_pr/id keys."""
        points = []
        for m, cid in sorted(self._archive, key=lambda x: x[0].fah):
            points.append(
                {
                    "id": cid,
                    "fah": m.fah,
                    "recall": m.recall,
                    "auc_pr": m.auc_pr,
                    "threshold": m.threshold,
                    "threshold_uint8": m.threshold_uint8,
                }
            )
        return points

    def __len__(self) -> int:
        return len(self._archive)


# ============================================================================
# ErrorMemory
# ============================================================================


class ErrorMemory:
    """Tracks persistent false alarms and misses across iterations."""

    def __init__(self, max_history: int = 10):
        self.persistent_false_alarms: dict[int, int] = {}
        self.persistent_misses: dict[int, int] = {}
        self.recent_scores: dict[int, list[float]] = {}
        self.max_history = max_history

    def update(
        self,
        indices: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> None:
        """Update error tracking after evaluation."""
        predictions = (y_pred >= threshold).astype(int)
        labels = (y_true >= 0.5).astype(int)

        for i, idx in enumerate(indices):
            idx_key = int(idx)

            # Track scores
            if idx_key not in self.recent_scores:
                self.recent_scores[idx_key] = []
            self.recent_scores[idx_key].append(float(y_pred[i]))
            if len(self.recent_scores[idx_key]) > self.max_history:
                self.recent_scores[idx_key] = self.recent_scores[idx_key][-self.max_history :]

            # Track false alarms (predicted positive, actually negative)
            if predictions[i] == 1 and labels[i] == 0:
                self.persistent_false_alarms[idx_key] = self.persistent_false_alarms.get(idx_key, 0) + 1
            # Track misses (predicted negative, actually positive)
            elif predictions[i] == 0 and labels[i] == 1:
                self.persistent_misses[idx_key] = self.persistent_misses.get(idx_key, 0) + 1

        # Cap dictionary sizes to prevent unbounded growth
        max_entries = 10000
        if len(self.persistent_false_alarms) > max_entries:
            # Remove oldest entries (lowest indices)
            keys_to_remove = sorted(self.persistent_false_alarms.keys())[: len(self.persistent_false_alarms) - max_entries]
            for k in keys_to_remove:
                del self.persistent_false_alarms[k]
        if len(self.persistent_misses) > max_entries:
            keys_to_remove = sorted(self.persistent_misses.keys())[: len(self.persistent_misses) - max_entries]
            for k in keys_to_remove:
                del self.persistent_misses[k]

    def get_persistent_fa_indices(self, min_count: int = 3) -> list[int]:
        return [idx for idx, count in self.persistent_false_alarms.items() if count >= min_count]

    def get_persistent_miss_indices(self, min_count: int = 3) -> list[int]:
        return [idx for idx, count in self.persistent_misses.items() if count >= min_count]

    def get_near_boundary_indices(self, threshold: float, margin: float = 0.05) -> list[int]:
        """Samples whose recent scores are consistently near the threshold."""
        result = []
        for idx, scores in self.recent_scores.items():
            if scores:
                mean_score = float(np.mean(scores))
                if abs(mean_score - threshold) < margin:
                    result.append(idx)
        return result


# ============================================================================
# Temperature Calibration
# ============================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), dtype=np.float64)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.asarray(np.log(p / (1.0 - p)), dtype=np.float64)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Platt scaling: find temperature T minimizing NLL on calibration set."""
    try:
        from scipy.optimize import OptimizeResult, minimize_scalar
    except ImportError:
        return 1.0

    logits = _logit(probs)
    y = labels.astype(np.float64)

    def nll(t: float) -> float:
        if t <= 0:
            return 1e10
        scaled = _sigmoid(logits / t)
        scaled = np.clip(scaled, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled)))

    result: OptimizeResult = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    optimal_t = float(result.x)

    # Only use if ECE improves
    ece_before = compute_ece(y, probs)
    scaled_probs = apply_temperature(probs, optimal_t)
    ece_after = compute_ece(y, scaled_probs)

    if ece_after >= ece_before:
        return 1.0
    return float(optimal_t)


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to model output probabilities."""
    if abs(temperature - 1.0) < 1e-6:
        return scores
    logits = _logit(scores)
    return _sigmoid(logits / temperature)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        n_bin = int(np.sum(mask))
        if n_bin == 0:
            continue
        avg_confidence = float(np.mean(y_prob[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        ece += (n_bin / total) * abs(avg_confidence - avg_accuracy)
    return ece


# ============================================================================
# ThresholdOptimizer
# ============================================================================


class ThresholdOptimizer:
    """3-pass threshold optimization.

    Returns: (threshold_float32, threshold_uint8, TuneMetrics)
    """

    def __init__(self) -> None:
        self._threshold_cache: dict[int, float] = {}

    @staticmethod
    def _float_to_uint8(threshold: float) -> int:
        return int(round(max(0.0, min(1.0, threshold)) * 255))

    def optimize(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        ambient_duration_hours: float,
        target_fah: float,
        target_recall: float,
        cv_folds: int = 5,
        fold_indices: Optional[list] = None,
        use_binary_search: bool = False,
        sample_weights: Optional[np.ndarray] = None,
    ) -> tuple[float, int, TuneMetrics]:
        """Threshold optimization.

        Returns: (threshold_float32, threshold_uint8, TuneMetrics)
        """
        if len(y_true) == 0 or len(y_scores) == 0:
            logger.warning("Threshold optimization received empty arrays; using default threshold=0.5")
            best_threshold = 0.5
            threshold_uint8 = self._float_to_uint8(best_threshold)
            metrics = self._compute_metrics_at_threshold(
                y_true,
                y_scores,
                best_threshold,
                ambient_duration_hours,
                sample_weights=sample_weights,
            )
            return best_threshold, threshold_uint8, metrics

        if use_binary_search:
            best_threshold = self._optimize_threshold_cached(
                y_scores=y_scores,
                y_true=y_true,
                target_fah=target_fah,
                val_ambient_duration_hours=ambient_duration_hours,
            )
            threshold_uint8 = self._float_to_uint8(best_threshold)
            metrics = self._compute_metrics_at_threshold(
                y_true,
                y_scores,
                best_threshold,
                ambient_duration_hours,
                sample_weights=sample_weights,
            )
            return best_threshold, threshold_uint8, metrics

        # Pass 1: Coarse quantile sweep
        region = self._coarse_sweep(y_true, y_scores, ambient_duration_hours, target_fah, target_recall)

        # Pass 2: Fine sweep within promising region
        best_threshold = self._fine_sweep(y_true, y_scores, ambient_duration_hours, target_fah, target_recall, region)

        # Pass 3: Robust CV refinement
        if cv_folds > 1 and fold_indices is not None and len(fold_indices) >= cv_folds:
            best_threshold = self._cv_refine(
                y_true,
                y_scores,
                ambient_duration_hours,
                target_fah,
                target_recall,
                best_threshold,
                fold_indices,
            )

        threshold_uint8 = self._float_to_uint8(best_threshold)
        metrics = self._compute_metrics_at_threshold(
            y_true,
            y_scores,
            best_threshold,
            ambient_duration_hours,
            sample_weights=sample_weights,
        )

        return best_threshold, threshold_uint8, metrics

    def _compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
        ambient_duration_hours: float,
        sample_weights: Optional[np.ndarray] = None,
    ) -> TuneMetrics:
        if len(y_true) == 0:
            return TuneMetrics(
                threshold=threshold,
                threshold_uint8=self._float_to_uint8(threshold),
                ambient_duration_hours=ambient_duration_hours,
            )

        labels = (y_true >= 0.5).astype(int)
        preds = (y_scores >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        total_pos = int(np.sum(labels == 1))
        total_neg = int(np.sum(labels == 0))

        recall = tp / max(total_pos, 1) if total_pos > 0 else 0.0
        precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-8) if (precision + recall) > 0 else 0.0
        fah = fp / max(ambient_duration_hours, 1e-8)

        # AUC metrics (simple trapezoidal)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            if total_pos > 0 and total_neg > 0:
                auc_roc = float(roc_auc_score(labels, y_scores))
                auc_pr = float(average_precision_score(labels, y_scores))
            else:
                auc_roc = 0.0
                auc_pr = 0.0
        except Exception:
            auc_roc = 0.0
            auc_pr = 0.0

        ece = compute_ece(y_true, y_scores)

        return TuneMetrics(
            fah=fah,
            recall=recall,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            ece=ece,
            threshold=threshold,
            threshold_uint8=self._float_to_uint8(threshold),
            precision=precision,
            f1=f1,
            total_positives=total_pos,
            total_negatives=total_neg,
            false_positives=fp,
            false_negatives=fn,
            ambient_duration_hours=ambient_duration_hours,
        )

    def _optimize_threshold_cached(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        target_fah: float,
        val_ambient_duration_hours: float,
    ) -> float:
        if len(y_scores) == 0 or len(y_true) == 0:
            return 0.5

        n_samples = min(1000, len(y_scores))
        sample_indices = np.linspace(0, len(y_scores) - 1, n_samples, dtype=int)
        sampled_scores = tuple(np.round(y_scores[sample_indices], 4))
        sampled_labels = tuple((y_true[sample_indices] >= 0.5).astype(np.int32))

        cache_key = hash(
            (
                sampled_scores,
                sampled_labels,
                round(float(target_fah), 4),
                round(float(val_ambient_duration_hours), 4),
            )
        )

        if cache_key in self._threshold_cache:
            return float(self._threshold_cache[cache_key])

        result = self._optimize_threshold_binary_search(
            y_scores=y_scores,
            y_true=y_true,
            target_fah=target_fah,
            val_ambient_duration_hours=val_ambient_duration_hours,
        )

        self._threshold_cache[cache_key] = float(result)

        if len(self._threshold_cache) > 50:
            keys = list(self._threshold_cache.keys())
            for key in keys[: len(keys) // 2]:
                del self._threshold_cache[key]

        return float(result)

    def _optimize_threshold_binary_search(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        target_fah: float,
        val_ambient_duration_hours: float,
    ) -> float:
        if len(y_scores) == 0 or len(y_true) == 0:
            return 0.5

        labels = (y_true >= 0.5).astype(np.int32)
        total_positives = int(np.sum(labels == 1))

        low, high = 0.0, 1.0
        best_threshold = 0.5
        best_metric = float("-inf")
        lowest_fah = float("inf")
        lowest_fah_threshold = 1.0

        for _ in range(15):
            mid = (low + high) / 2.0
            predictions = (y_scores >= mid).astype(np.int32)

            false_positives = int(np.sum((predictions == 1) & (labels == 0)))
            fah = false_positives / val_ambient_duration_hours if val_ambient_duration_hours > 0 else float("inf")
            true_positives = int(np.sum((predictions == 1) & (labels == 1)))
            recall = true_positives / total_positives if total_positives > 0 else 0.0

            if fah < lowest_fah:
                lowest_fah = fah
                lowest_fah_threshold = mid

            if fah <= target_fah:
                if recall > best_metric:
                    best_metric = recall
                    best_threshold = mid
                high = mid
            else:
                low = mid

        if best_metric == float("-inf"):
            best_threshold = lowest_fah_threshold

        return float(best_threshold)

    def _coarse_sweep(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        ambient_hours: float,
        target_fah: float,
        target_recall: float,
    ) -> tuple:
        """Pass 1: Sweep 4096 quantile thresholds, find promising region."""
        n_thresholds = 4096
        thresholds = np.quantile(y_scores, np.linspace(0, 1, n_thresholds))
        thresholds = np.unique(thresholds)

        labels = (y_true >= 0.5).astype(int)
        n_pos = int(np.sum(labels))
        feasible_thresholds = []

        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            fp = int(np.sum((preds == 1) & (labels == 0)))
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fah = fp / max(ambient_hours, 1e-8)
            recall = tp / max(n_pos, 1) if n_pos > 0 else 0.0

            if fah <= target_fah:
                feasible_thresholds.append((t, recall, fah))

        if feasible_thresholds:
            feas_t = [f[0] for f in feasible_thresholds]
            margin = (max(feas_t) - min(feas_t)) * 0.1
            region_low = min(feas_t) - margin
            region_high = max(feas_t) + margin
        else:
            region_low = float(np.quantile(y_scores, 0.5))
            region_high = float(np.quantile(y_scores, 0.99))

        return (max(0.0, region_low), min(1.0, region_high))

    def _fine_sweep(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        ambient_hours: float,
        target_fah: float,
        target_recall: float,
        region: tuple,
    ) -> float:
        """Pass 2: Exact unique scores within promising region."""
        mask = (y_scores >= region[0]) & (y_scores <= region[1])
        unique_scores = np.unique(y_scores[mask])

        if len(unique_scores) == 0:
            unique_scores = np.unique(y_scores)

        labels = (y_true >= 0.5).astype(int)
        n_pos = int(np.sum(labels))
        best_threshold = 0.5
        best_recall = -1.0

        for t in unique_scores:
            preds = (y_scores >= t).astype(int)
            fp = int(np.sum((preds == 1) & (labels == 0)))
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fah = fp / max(ambient_hours, 1e-8)
            recall = tp / max(n_pos, 1) if n_pos > 0 else 0.0

            if fah <= target_fah and recall > best_recall:
                best_recall = recall
                best_threshold = float(t)

        return best_threshold

    def _cv_refine(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        ambient_hours: float,
        target_fah: float,
        target_recall: float,
        initial_threshold: float,
        fold_indices: list,
    ) -> float:
        """Pass 3: Cross-validation refinement around initial threshold."""
        candidate_thresholds = np.linspace(
            max(0.0, initial_threshold - 0.1),
            min(1.0, initial_threshold + 0.1),
            50,
        )

        threshold_scores: dict[float, list[float]] = {float(t): [] for t in candidate_thresholds}

        for fold_idx in fold_indices:
            fold_mask = np.zeros(len(y_true), dtype=bool)
            fold_mask[fold_idx] = True

            y_val = y_true[fold_mask]
            s_val = y_scores[fold_mask]

            if len(y_val) == 0:
                continue

            labels_val = (y_val >= 0.5).astype(int)
            n_pos_val = int(np.sum(labels_val))

            for t in candidate_thresholds:
                preds = (s_val >= t).astype(int)
                fp = int(np.sum((preds == 1) & (labels_val == 0)))
                tp = int(np.sum((preds == 1) & (labels_val == 1)))
                fah = fp / max(ambient_hours / max(len(fold_indices), 1), 1e-8)
                recall = tp / max(n_pos_val, 1) if n_pos_val > 0 else 0.0

                if fah <= target_fah:
                    threshold_scores[float(t)].append(recall)
                else:
                    threshold_scores[float(t)].append(-fah)

        best_threshold = initial_threshold
        best_score = float("-inf")

        for t, scores in threshold_scores.items():
            if scores:
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_threshold = t

        return best_threshold


# ============================================================================
# Hypervolume Indicator
# ============================================================================


def compute_hypervolume(
    pareto_points: list[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    """Compute 2D hypervolume dominated by Pareto front relative to reference point.

    Args:
        pareto_points: List of (fah, recall) tuples on the Pareto front.
        reference: (ref_fah, ref_recall) reference point (should be dominated by all points).

    Returns:
        Hypervolume indicator value (non-negative float).
    """
    if not pareto_points:
        return 0.0

    ref_fah, ref_recall = reference

    # Transform to maximization problem:
    # objective 1: maximize (ref_fah - fah)   [i.e. lower fah is better]
    # objective 2: maximize (recall - ref_recall) [i.e. higher recall is better]
    transformed = []
    for fah, recall in pareto_points:
        x = ref_fah - fah
        y = recall - ref_recall
        if x > 0 and y > 0:
            transformed.append((x, y))

    if not transformed:
        # Try to be lenient - compute even if points don't strictly dominate reference
        transformed = []
        for fah, recall in pareto_points:
            x = max(0.0, ref_fah - fah)
            y = max(0.0, recall - ref_recall)
            if x > 0 or y > 0:
                transformed.append((max(x, 0.0), max(y, 0.0)))

        if not transformed:
            return 0.0

    # Sort by first objective descending, second objective ascending (for sweep)
    transformed.sort(key=lambda p: (-p[0], p[1]))

    # 2D hypervolume via sweep

    # Keep non-dominated front (for correct 2D hypervolume)
    # Sort by x descending; remove any point with y <= previous max y
    front: list[tuple[float, float]] = []
    max_y = float("-inf")
    for p in sorted(transformed, key=lambda p: p[0], reverse=True):
        if p[1] > max_y:
            front.append(p)
            max_y = p[1]

    # Compute hypervolume via sweep line
    hv = 0.0

    # height sweep (x from 0 to max, y is the contribution)
    # For each segment between consecutive x values, area += width * height
    # Process from largest x to smallest to accumulate
    front_sorted_desc = sorted(front, key=lambda p: p[0], reverse=True)
    prev_x_val = front_sorted_desc[0][0]
    cur_max_y = 0.0

    for i, (x, y) in enumerate(front_sorted_desc):
        cur_max_y = max(cur_max_y, y)
        if i + 1 < len(front_sorted_desc):
            next_x = front_sorted_desc[i + 1][0]
            hv += (prev_x_val - next_x) * cur_max_y
            prev_x_val = next_x
        else:
            hv += x * cur_max_y

    return float(hv)
