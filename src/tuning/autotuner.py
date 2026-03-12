"""MaxQualityAutoTuner — Post-training auto-tuning for wake word models.

Replaces the legacy AutoTuner with a sophisticated branch-and-confirm system
using surgical gradient bursts, 7 strategy arms with Thompson sampling,
3-pass threshold optimization, temperature scaling, simulated annealing
acceptance, Pareto archive, INT8 shadow evaluation, and confirmation phase.

Quality over speed. Time budget does not matter — only final model quality.
"""

import contextlib
import io
import json
import logging
import math
import os
import pickle
import re
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResilientFileHandler(logging.FileHandler):
    """FileHandler that recovers if underlying stream gets closed unexpectedly."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except ValueError as exc:
            if "closed file" not in str(exc).lower() and "i/o operation on closed file" not in str(exc).lower():
                raise
            self.acquire()
            try:
                if self.stream is not None:
                    self.stream = self._open()
            finally:
                self.release()
            super().emit(record)


# ============================================================================
# Section 1: Data Structures
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
        """Pareto dominance: self dominates other if better in all objectives."""
        return self.fah <= other.fah and self.recall >= other.recall and self.auc_pr >= other.auc_pr and (self.fah < other.fah or self.recall > other.recall or self.auc_pr > other.auc_pr)

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


@dataclass
class CandidateState:
    """Complete state of a tuning candidate."""

    id: str
    weights_bytes: bytes
    optimizer_state_bytes: bytes
    batchnorm_state: dict
    swa_buffers: Optional[bytes] = None
    temperature: float = 1.0
    threshold_float32: float = 0.5
    threshold_uint8: int = 128
    eval_results: Optional[TuneMetrics] = None
    eval_results_int8: Optional[TuneMetrics] = None
    sharpness_score: float = 0.0
    curriculum_stage: int = 0
    strategy_arm: int = -1
    parent_id: str = ""
    iteration: int = 0
    stagnation_count: int = 0
    history: list = field(default_factory=list)
    lr: float = 1e-5


@dataclass
class StrategyArm:
    """Definition of a strategy arm."""

    name: str
    min_steps: int
    max_steps: int
    default_steps: int
    lr_range: tuple
    default_lr: float
    description: str
    use_sam: bool = False
    use_swa: bool = False
    use_cosine_schedule: bool = False
    consecutive_ban: bool = False


STRATEGY_ARMS = [
    StrategyArm(
        "boundary_polish",
        500,
        1000,
        750,
        (1e-6, 1e-4),
        0.0,
        "50% near-threshold + 50% replay",
    ),
    StrategyArm(
        "fa_suppression",
        750,
        1500,
        1000,
        (5e-6, 2e-5),
        1e-5,
        "60% FA + 20% hard_neg + 20% replay",
    ),
    StrategyArm(
        "recall_recovery",
        750,
        1500,
        1000,
        (1e-5, 5e-5),
        2e-5,
        "60% misses + 20% easy_pos + 20% replay",
    ),
    StrategyArm(
        "sam_flatten",
        1500,
        3000,
        2000,
        (1e-5, 5e-5),
        3e-5,
        "SAM flatten + SWA",
        use_sam=True,
        use_swa=True,
    ),
    StrategyArm(
        "cyclic_op_sweep",
        1500,
        3000,
        2000,
        (1e-6, 5e-5),
        0.0,
        "Cosine warm restarts",
        use_cosine_schedule=True,
    ),
    StrategyArm(
        "macro_refine",
        2000,
        5000,
        3000,
        (1e-6, 3e-5),
        1e-5,
        "Full proportional + curriculum",
        use_cosine_schedule=True,
    ),
    StrategyArm(
        "hardest_only_shock",
        250,  # Increased from 200 to be more conservative
        350,  # Reduced from 500 to cap aggression
        300,
        (5e-6, 1.5e-5),  # Reduced max LR from 2e-5 to 1.5e-5
        1e-5,
        "100% hardest examples (reduced aggression)",
        consecutive_ban=True,
    ),
]


# ============================================================================
# Section 2: Pareto Archive
# ============================================================================


class ParetoArchive:
    """Multi-objective Pareto archive with diversity filtering."""

    def __init__(self, max_size: int = 24, diversity_threshold: float = 0.01):
        self.archive: list[CandidateState] = []
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold

    def try_add(self, candidate: CandidateState) -> bool:
        """Add candidate if non-dominated or diverse. Returns True if added."""
        if candidate.eval_results is None:
            return False

        c_metrics = candidate.eval_results
        # Remove dominated entries
        new_archive = []
        for entry in self.archive:
            if entry.eval_results is not None and not c_metrics.dominates(entry.eval_results):
                new_archive.append(entry)

        # Check if candidate is dominated by any remaining
        is_dominated = any(e.eval_results is not None and e.eval_results.dominates(c_metrics) for e in new_archive)

        if is_dominated:
            # Still add if diverse enough
            if not self._is_diverse(candidate, new_archive):
                self.archive = new_archive
                return False

        # Check diversity
        if not is_dominated or self._is_diverse(candidate, new_archive):
            new_archive.append(candidate)

        # If over capacity, drop by crowding distance
        if len(new_archive) > self.max_size:
            new_archive = self._prune_by_crowding(new_archive)

        self.archive = new_archive
        return True

    def _is_diverse(self, candidate: CandidateState, archive: list[CandidateState]) -> bool:
        if not archive:
            return True
        c = candidate.eval_results
        if c is None:
            return False
        for entry in archive:
            e = entry.eval_results
            if e is None:
                continue
            dist = math.sqrt((c.fah - e.fah) ** 2 + (c.recall - e.recall) ** 2 + (c.auc_pr - e.auc_pr) ** 2)
            if dist < self.diversity_threshold:
                return False
        return True

    def _prune_by_crowding(self, archive: list[CandidateState]) -> list[CandidateState]:
        """Remove entries with smallest crowding distance until at max_size."""
        while len(archive) > self.max_size:
            distances = self._crowding_distances(archive)
            worst_idx = int(np.argmin(distances))
            archive.pop(worst_idx)
        return archive

    def _crowding_distances(self, archive: list[CandidateState]) -> np.ndarray:
        n = len(archive)
        if n <= 2:
            return np.full(n, float("inf"))

        # Track valid indices (entries with eval_results)
        valid_indices = [i for i, a in enumerate(archive) if a.eval_results is not None]
        m = len(valid_indices)

        if m <= 2:
            # Not enough valid entries for crowding distance
            distances = np.zeros(n)
            for i in valid_indices:
                distances[i] = float("inf")
            return distances

        objectives = np.array([[-archive[i].eval_results.fah, archive[i].eval_results.recall, archive[i].eval_results.auc_pr] for i in valid_indices])
        distances = np.zeros(n)

        for obj_idx in range(objectives.shape[1]):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            # Map back to original archive indices
            sorted_orig = [valid_indices[i] for i in sorted_indices]

            distances[sorted_orig[0]] = float("inf")
            distances[sorted_orig[-1]] = float("inf")

            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range < 1e-12:
                continue

            for i in range(1, m - 1):
                # Map index to original archive position
                orig_idx = valid_indices[sorted_indices[i]]
                distances[orig_idx] += (objectives[sorted_indices[i + 1], obj_idx] - objectives[sorted_indices[i - 1], obj_idx]) / obj_range
        return distances

    def get_best(self, target_fah: float, target_recall: float) -> Optional[CandidateState]:
        """Best candidate: meets targets with max recall, else closest."""
        meeting = [c for c in self.archive if c.eval_results is not None and c.eval_results.meets_target(target_fah, target_recall)]
        if meeting:
            return max(meeting, key=lambda c: c.eval_results.recall)

        if not self.archive:
            return None

        # Closest to targets by scalarized distance
        def score(c):
            m = c.eval_results
            if m is None:
                return float("inf")
            fah_excess = max(0, m.fah - target_fah) / max(target_fah, 1e-8)
            recall_deficit = max(0, target_recall - m.recall) / max(target_recall, 1e-8)
            return 2.0 * fah_excess + 1.0 * recall_deficit

        return min(self.archive, key=score)

    def get_frontier_points(self) -> list[dict]:
        """Return frontier as list of dicts for logging."""
        points = []
        for c in sorted(
            self.archive,
            key=lambda x: x.eval_results.fah if x.eval_results else float("inf"),
        ):
            if c.eval_results is not None:
                points.append(
                    {
                        "id": c.id,
                        "fah": c.eval_results.fah,
                        "recall": c.eval_results.recall,
                        "auc_pr": c.eval_results.auc_pr,
                        "threshold": c.eval_results.threshold,
                        "threshold_uint8": c.eval_results.threshold_uint8,
                        "arm": STRATEGY_ARMS[c.strategy_arm].name if 0 <= c.strategy_arm < len(STRATEGY_ARMS) else "initial",
                        "iteration": c.iteration,
                    }
                )
        return points

    def __len__(self) -> int:
        return len(self.archive)


# ============================================================================
# Section 3: Error Memory
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
    ):
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

    def get_persistent_fa_indices(self, min_count: int = 3) -> list[int]:
        return [idx for idx, count in self.persistent_false_alarms.items() if count >= min_count]

    def get_persistent_miss_indices(self, min_count: int = 3) -> list[int]:
        return [idx for idx, count in self.persistent_misses.items() if count >= min_count]

    def get_near_boundary_indices(self, threshold: float, margin: float = 0.05) -> list[int]:
        """Samples whose recent scores are consistently near the threshold."""
        result = []
        for idx, scores in self.recent_scores.items():
            if scores:
                mean_score = np.mean(scores)
                if abs(mean_score - threshold) < margin:
                    result.append(idx)
        return result


# ============================================================================
# Section 4: Focused Sampler
# ============================================================================


class FocusedSampler:
    """Builds focused mini-batches for each strategy arm."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        error_memory: ErrorMemory,
    ):
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights
        self.error_memory = error_memory
        self._index_by_label()

    def _index_by_label(self):
        labels_int = (self.labels >= 0.5).astype(int)
        self.pos_indices = np.where(labels_int == 1)[0]
        self.neg_indices = np.where(labels_int == 0)[0]
        self.all_indices = np.arange(len(self.labels))

    def build_batch(
        self,
        strategy_arm: int,
        threshold: float,
        batch_size: int = 128,
        curriculum_stage: int = 0,
        recent_scores: Optional[np.ndarray] = None,
    ) -> tuple:
        """Return (batch_features, batch_labels, batch_weights)."""
        if strategy_arm == 0:
            return self._boundary_polish_batch(threshold, batch_size, recent_scores)
        elif strategy_arm == 1:
            return self._fa_suppression_batch(threshold, batch_size, recent_scores)
        elif strategy_arm == 2:
            return self._recall_recovery_batch(threshold, batch_size, recent_scores)
        elif strategy_arm == 3:
            return self._standard_batch(batch_size)
        elif strategy_arm == 4:
            return self._standard_batch(batch_size)
        elif strategy_arm == 5:
            return self._curriculum_batch(batch_size, curriculum_stage, recent_scores)
        elif strategy_arm == 6:
            return self._hardest_only_batch(threshold, batch_size, recent_scores)
        else:
            return self._standard_batch(batch_size)

    def _boundary_polish_batch(self, threshold: float, batch_size: int, scores: Optional[np.ndarray]) -> tuple:
        """50% near-threshold + 50% replay."""
        half = batch_size // 2
        near = self._get_near_boundary(threshold, half, scores)
        replay = self._random_sample(batch_size - len(near))
        indices = np.concatenate([near, replay])
        return self._gather(indices)

    def _fa_suppression_batch(self, threshold: float, batch_size: int, scores: Optional[np.ndarray]) -> tuple:
        """60% FA + 20% hard_neg + 20% replay."""
        n_fa = int(batch_size * 0.6)
        n_hard = int(batch_size * 0.2)
        n_replay = batch_size - n_fa - n_hard

        fa_indices = self._get_false_alarm_indices(threshold, n_fa, scores)
        hard_neg = self._random_from(self.neg_indices, n_hard)
        replay = self._random_sample(n_replay)
        indices = np.concatenate([fa_indices, hard_neg, replay])
        return self._gather(indices)

    def _recall_recovery_batch(self, threshold: float, batch_size: int, scores: Optional[np.ndarray]) -> tuple:
        """60% misses + 20% easy_pos + 20% replay."""
        n_miss = int(batch_size * 0.6)
        n_easy = int(batch_size * 0.2)
        n_replay = batch_size - n_miss - n_easy

        miss_indices = self._get_miss_indices(threshold, n_miss, scores)
        easy_pos = self._random_from(self.pos_indices, n_easy)
        replay = self._random_sample(n_replay)
        indices = np.concatenate([miss_indices, easy_pos, replay])
        return self._gather(indices)

    def _curriculum_batch(
        self,
        batch_size: int,
        curriculum_stage: int,
        scores: Optional[np.ndarray],
    ) -> tuple:
        """Curriculum-aware batch based on stage."""
        if curriculum_stage == 0:
            # 70% near-boundary, 30% replay
            n_focus = int(batch_size * 0.7)
        elif curriculum_stage == 1:
            n_focus = int(batch_size * 0.6)
        elif curriculum_stage == 2:
            n_focus = int(batch_size * 0.5)
        else:
            n_focus = int(batch_size * 0.25)

        n_replay = batch_size - n_focus
        persistent = self.error_memory.get_persistent_fa_indices(min_count=2) + self.error_memory.get_persistent_miss_indices(min_count=2)

        if persistent:
            focus_pool = np.array(persistent)
            focus_pool = focus_pool[focus_pool < len(self.features)]
            focus = self._random_from(focus_pool, n_focus)
        else:
            focus = self._random_sample(n_focus)

        replay = self._random_sample(n_replay)
        indices = np.concatenate([focus, replay])
        return self._gather(indices)

    def _hardest_only_batch(self, threshold: float, batch_size: int, scores: Optional[np.ndarray]) -> tuple:
        """100% hardest examples."""
        persistent = self.error_memory.get_persistent_fa_indices(min_count=1) + self.error_memory.get_persistent_miss_indices(min_count=1)

        if persistent and len(persistent) >= batch_size:
            pool = np.array(persistent)
            pool = pool[pool < len(self.features)]
            indices = self._random_from(pool, batch_size)
        elif scores is not None:
            # Pick samples closest to threshold
            distances = np.abs(scores - threshold)
            indices = np.argsort(distances)[:batch_size]
        else:
            indices = self._random_sample(batch_size)
        return self._gather(indices)

    def _standard_batch(self, batch_size: int) -> tuple:
        """Standard proportional batch."""
        indices = self._random_sample(batch_size)
        return self._gather(indices)

    def _get_near_boundary(self, threshold: float, n: int, scores: Optional[np.ndarray]) -> np.ndarray:
        near_indices = self.error_memory.get_near_boundary_indices(threshold)
        if near_indices:
            pool = np.array(near_indices)
            pool = pool[pool < len(self.features)]
            return self._random_from(pool, n)
        if scores is not None:
            distances = np.abs(scores - threshold)
            sorted_idx = np.argsort(distances)
            return sorted_idx[:n]
        return self._random_sample(n)

    def _get_false_alarm_indices(self, threshold: float, n: int, scores: Optional[np.ndarray]) -> np.ndarray:
        fa_indices = self.error_memory.get_persistent_fa_indices(min_count=1)
        if fa_indices:
            pool = np.array(fa_indices)
            pool = pool[pool < len(self.features)]
            return self._random_from(pool, n)
        # Fallback: negative samples with high scores
        if scores is not None:
            neg_scores = scores[self.neg_indices]
            top_fa = np.argsort(-neg_scores)[:n]
            return self.neg_indices[top_fa]
        return self._random_from(self.neg_indices, n)

    def _get_miss_indices(self, threshold: float, n: int, scores: Optional[np.ndarray]) -> np.ndarray:
        miss_indices = self.error_memory.get_persistent_miss_indices(min_count=1)
        if miss_indices:
            pool = np.array(miss_indices)
            pool = pool[pool < len(self.features)]
            return self._random_from(pool, n)
        # Fallback: positive samples with low scores
        if scores is not None:
            pos_scores = scores[self.pos_indices]
            low_recall = np.argsort(pos_scores)[:n]
            return self.pos_indices[low_recall]
        return self._random_from(self.pos_indices, n)

    def _random_from(self, pool: np.ndarray, n: int) -> np.ndarray:
        if len(pool) == 0:
            return self._random_sample(n)
        indices = np.random.choice(pool, size=min(n, len(pool)), replace=len(pool) < n)
        return indices

    def _random_sample(self, n: int) -> np.ndarray:
        return np.random.choice(self.all_indices, size=n, replace=n > len(self.all_indices))

    def _gather(self, indices: np.ndarray) -> tuple:
        indices = indices.astype(int)
        indices = np.clip(indices, 0, len(self.features) - 1)
        return (
            self.features[indices],
            self.labels[indices],
            self.sample_weights[indices],
        )


# ============================================================================
# Section 5: Temperature Scaling
# ============================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Platt scaling: find T minimizing NLL on calibration set."""
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return 1.0

    logits = _logit(probs)
    y = labels.astype(np.float64)

    def nll(t):
        if t <= 0:
            return 1e10
        scaled = _sigmoid(logits / t)
        scaled = np.clip(scaled, 1e-7, 1.0 - 1e-7)
        return -np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    optimal_t = result.x

    # Verify ECE improves
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
        n_bin = np.sum(mask)
        if n_bin == 0:
            continue
        avg_confidence = np.mean(y_prob[mask])
        avg_accuracy = np.mean(y_true[mask])
        ece += (n_bin / total) * abs(avg_confidence - avg_accuracy)
    return ece


# ============================================================================
# Section 6: 3-Pass Threshold Optimizer
# ============================================================================


class ThresholdOptimizer:
    """3-pass threshold optimization with cross-validation."""

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
    ) -> tuple:
        """Threshold optimization.

        Uses existing 3-pass optimization by default. Optionally enables a
        fast binary-search path with caching.

        Returns: (threshold_float32, threshold_uint8, TuneMetrics)
        """
        if len(y_true) == 0 or len(y_scores) == 0:
            logger.warning("Threshold optimization received empty arrays; using default threshold=0.5")
            best_threshold = 0.5
            threshold_uint8 = self._float_to_uint8(best_threshold)
            metrics = self._compute_metrics_at_threshold(y_true, y_scores, best_threshold, ambient_duration_hours)
            return best_threshold, threshold_uint8, metrics

        if use_binary_search:
            best_threshold = self._optimize_threshold_cached(
                y_scores=y_scores,
                y_true=y_true,
                target_fah=target_fah,
                val_ambient_duration_hours=ambient_duration_hours,
            )

            threshold_uint8 = self._float_to_uint8(best_threshold)
            metrics = self._compute_metrics_at_threshold(y_true, y_scores, best_threshold, ambient_duration_hours)
            return best_threshold, threshold_uint8, metrics

        # Pass 1: Coarse quantile sweep
        region = self._coarse_sweep(y_true, y_scores, ambient_duration_hours, target_fah, target_recall)

        # Pass 2: Fine sweep within promising region
        best_threshold = self._fine_sweep(
            y_true,
            y_scores,
            ambient_duration_hours,
            target_fah,
            target_recall,
            region,
        )

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
        metrics = self._compute_metrics_at_threshold(y_true, y_scores, best_threshold, ambient_duration_hours)

        return best_threshold, threshold_uint8, metrics

    def _optimize_threshold_binary_search(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        target_fah: float,
        val_ambient_duration_hours: float,
    ) -> float:
        """Optimize threshold using binary search instead of linear sweep.

        Reduces coarse threshold evaluations dramatically while preserving
        objective: maximize recall subject to FAH <= target_fah.

        Args:
            y_scores: Model prediction scores.
            y_true: Ground-truth labels.
            target_fah: Target false accepts per hour.
            val_ambient_duration_hours: Ambient validation duration in hours.

        Returns:
            Optimal threshold value.
        """
        if len(y_scores) == 0 or len(y_true) == 0:
            logger.warning("Binary search threshold optimization received empty arrays; using default threshold=0.5")
            return 0.5

        low, high = 0.0, 1.0
        labels = (y_true >= 0.5).astype(np.int32)
        total_positives = int(np.sum(labels == 1))

        best_threshold = 0.5
        best_metric = float("-inf")
        lowest_fah = float("inf")
        lowest_fah_threshold = 1.0

        for iteration in range(15):  # ~3e-5 resolution in [0, 1]
            mid = (low + high) / 2.0
            predictions = (y_scores >= mid).astype(np.int32)

            false_positives = int(np.sum((predictions == 1) & (labels == 0)))
            fah = false_positives / val_ambient_duration_hours if val_ambient_duration_hours > 0 else float("inf")

            true_positives = int(np.sum((predictions == 1) & (labels == 1)))
            recall = true_positives / total_positives if total_positives > 0 else 0.0

            if fah < lowest_fah:
                lowest_fah = fah
                lowest_fah_threshold = mid

            # Maximize recall subject to FAH constraint.
            if fah <= target_fah:
                if recall > best_metric:
                    best_metric = recall
                    best_threshold = mid
                # Lower threshold tends to increase recall.
                high = mid
            else:
                # Increase threshold to suppress false accepts.
                low = mid

            logger.debug(
                "Binary search iter %d: threshold=%.6f, FAH=%.6f, recall=%.6f",
                iteration,
                mid,
                fah,
                recall,
            )

        if best_metric == float("-inf"):
            logger.warning(
                "Binary search found no feasible threshold for target_fah=%.6f; falling back to lowest-FAH threshold=%.6f (FAH=%.6f)",
                target_fah,
                lowest_fah_threshold,
                lowest_fah,
            )
            best_threshold = lowest_fah_threshold
            best_metric = 0.0 if total_positives == 0 else best_metric

        logger.info(
            "Binary search threshold: %.6f (best_recall=%.6f)",
            best_threshold,
            best_metric,
        )
        return float(best_threshold)

    def _optimize_threshold_cached(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        target_fah: float,
        val_ambient_duration_hours: float,
    ) -> float:
        """Optimize threshold with result caching.

        Caches optimization results to avoid recomputing for effectively
        identical score/label distributions.
        """
        if not hasattr(self, "_threshold_cache"):
            self._threshold_cache = {}

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
            logger.info("Using cached threshold optimization result")
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
        n_pos = np.sum(labels)
        best_score = -1.0
        region_low = 0.0
        region_high = 1.0

        feasible_thresholds = []
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            fp = np.sum((preds == 1) & (labels == 0))
            tp = np.sum((preds == 1) & (labels == 1))
            fah = fp / max(ambient_hours, 1e-8)
            recall = tp / max(n_pos, 1) if n_pos > 0 else 0.0

            if fah <= target_fah:
                feasible_thresholds.append((t, recall, fah))
                if recall > best_score:
                    best_score = recall

        if feasible_thresholds:
            feas_t = [f[0] for f in feasible_thresholds]
            margin = (max(feas_t) - min(feas_t)) * 0.1
            region_low = min(feas_t) - margin
            region_high = max(feas_t) + margin
        else:
            # No feasible point found — sweep around the middle
            region_low = np.quantile(y_scores, 0.5)
            region_high = np.quantile(y_scores, 0.99)

        return (max(0, region_low), min(1, region_high))

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
        n_pos = np.sum(labels)
        best_threshold = 0.5
        best_recall = -1.0
        best_fah = float("inf")

        for t in unique_scores:
            preds = (y_scores >= t).astype(int)
            fp = np.sum((preds == 1) & (labels == 0))
            tp = np.sum((preds == 1) & (labels == 1))
            fah = fp / max(ambient_hours, 1e-8)
            recall = tp / max(n_pos, 1) if n_pos > 0 else 0.0

            if fah <= target_fah and recall > best_recall:
                best_recall = recall
                best_fah = fah
                best_threshold = t
            elif fah <= target_fah and recall == best_recall and fah < best_fah:
                best_fah = fah
                best_threshold = t

        if best_recall < 0:
            # No feasible point — pick threshold with lowest FAH
            for t in sorted(unique_scores, reverse=True):
                preds = (y_scores >= t).astype(int)
                fp = np.sum((preds == 1) & (labels == 0))
                fah = fp / max(ambient_hours, 1e-8)
                if fah < best_fah:
                    best_fah = fah
                    best_threshold = t

        return float(best_threshold)

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
        """Pass 3: Cross-validation refinement around the initial threshold."""
        # Search window around initial threshold
        window = 0.02
        n_search = 201
        candidates = np.linspace(
            max(0, initial_threshold - window),
            min(1, initial_threshold + window),
            n_search,
        )

        labels = (y_true >= 0.5).astype(int)
        best_threshold = initial_threshold
        best_worst_recall = -1.0

        for t in candidates:
            fold_recalls = []
            fold_fahs = []
            for fold_idx in fold_indices:
                fold_labels = labels[fold_idx]
                fold_scores = y_scores[fold_idx]
                n_pos = np.sum(fold_labels)
                if n_pos == 0:
                    continue

                preds = (fold_scores >= t).astype(int)
                fp = np.sum((preds == 1) & (fold_labels == 0))
                tp = np.sum((preds == 1) & (fold_labels == 1))
                fold_fah = fp / max(ambient_hours / len(fold_indices), 1e-8)
                fold_recall = tp / max(n_pos, 1)
                fold_recalls.append(fold_recall)
                fold_fahs.append(fold_fah)

            if not fold_recalls:
                continue

            worst_recall = min(fold_recalls)
            worst_fah = max(fold_fahs)

            if worst_fah <= target_fah * 1.1 and worst_recall > best_worst_recall:
                best_worst_recall = worst_recall
                best_threshold = t

        # INT8 quantization margin: shift by ±1/255
        uint8_val = self._float_to_uint8(best_threshold)
        # Prefer slightly higher threshold for INT8 safety margin
        adjusted = (uint8_val + 1) / 255.0
        if adjusted <= 1.0:
            test_metrics = self._compute_metrics_at_threshold(y_true, y_scores, adjusted, ambient_hours)
            if test_metrics.fah <= target_fah:
                best_threshold = adjusted

        return float(best_threshold)

    def _float_to_uint8(self, threshold: float) -> int:
        return max(0, min(255, int(round(threshold * 255))))

    def _compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
        ambient_hours: float,
    ) -> TuneMetrics:
        """Compute full metrics at a given threshold."""
        labels = (y_true >= 0.5).astype(int)
        preds = (y_scores >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        n_pos = int(np.sum(labels))
        n_neg = int(np.sum(1 - labels))

        recall = tp / max(n_pos, 1) if n_pos > 0 else 0.0
        precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-8) if (precision + recall) > 0 else 0.0
        fah = fp / max(ambient_hours, 1e-8)

        # AUC metrics
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            auc_roc = float(roc_auc_score(labels, y_scores))
            auc_pr = float(average_precision_score(labels, y_scores))
        except (ValueError, TypeError) as e:
            import logging

            logging.getLogger(__name__).warning("AUC computation failed: %s", e)
            auc_roc = 0.0
            auc_pr = 0.0

        ece = compute_ece(labels, y_scores)

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
            total_positives=n_pos,
            total_negatives=n_neg,
            false_positives=fp,
            false_negatives=fn,
            ambient_duration_hours=ambient_hours,
        )


# ============================================================================
# Section 7: Thompson Sampling
# ============================================================================


class ThompsonSampler:
    """Thompson sampling for strategy arm selection with regime bonuses."""

    def __init__(self, n_arms: int = 7):
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)
        self.last_arm: int = -1
        self.n_arms = n_arms

    # Regime → arm affinity mapping
    REGIME_BONUSES = {
        "near_feasible": {0, 3},  # boundary_polish, sam_flatten
        "fah_dominated": {1, 6},  # fa_suppression, hardest_only_shock
        "recall_dominated": {2, 4},  # recall_recovery, cyclic_op_sweep
        "balanced": {3, 5},  # sam_flatten, macro_refine
    }

    def select_arm(self, regime: str) -> int:
        """Sample from Beta distributions with regime bonus."""
        bonus_arms = self.REGIME_BONUSES.get(regime, set())

        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            # Ban consecutive hardest_only_shock
            if i == 6 and self.last_arm == 6:
                samples[i] = -1.0
                continue

            sample = np.random.beta(self.successes[i], self.failures[i])
            if i in bonus_arms:
                sample += 0.3
            samples[i] = sample

        selected = int(np.argmax(samples))
        self.last_arm = selected
        return selected

    def update(self, arm: int, success: bool):
        self.successes[arm] += float(success)
        self.failures[arm] += float(not success)


def diagnose_regime(metrics: TuneMetrics, target_fah: float, target_recall: float) -> str:
    """Classify current state into regime."""
    fah_ok = metrics.fah <= target_fah * 1.5
    recall_ok = metrics.recall >= target_recall * 0.9
    if fah_ok and recall_ok:
        return "near_feasible"
    elif not fah_ok and recall_ok:
        return "fah_dominated"
    elif fah_ok and not recall_ok:
        return "recall_dominated"
    else:
        return "balanced"


# ============================================================================
# Section 8: Stir Controller
# ============================================================================


class StirController:
    """Manages 5 levels of stir for stagnation escape."""

    def __init__(self, thresholds: Optional[list] = None):
        self.thresholds = thresholds or [3, 5, 7, 9, 12]

    def get_stir_level(self, stagnation_count: int) -> int:
        for i, thresh in enumerate(self.thresholds):
            if stagnation_count < thresh:
                return i
        return 5


# ============================================================================
# Section 9: Annealing Controller
# ============================================================================


class AnnealingController:
    """Simulated annealing acceptance with Pareto awareness."""

    def __init__(
        self,
        initial_temperature: float = 0.5,
        cooling_rate: float = 0.95,
        reheat_factor: float = 1.3,
        reheat_after: int = 5,
    ):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.reheat_factor = reheat_factor
        self.reheat_after = reheat_after
        self.consecutive_rejections = 0

    def should_accept(
        self,
        candidate_metrics: TuneMetrics,
        parent_metrics: TuneMetrics,
        target_fah: float,
        target_recall: float,
    ) -> bool:
        """Simulated annealing acceptance."""
        if candidate_metrics.dominates(parent_metrics):
            self._on_accept()
            return True

        candidate_cost = self._scalarized_cost(candidate_metrics, target_fah, target_recall)
        parent_cost = self._scalarized_cost(parent_metrics, target_fah, target_recall)
        delta = candidate_cost - parent_cost

        if delta <= 0:
            self._on_accept()
            return True

        p = np.exp(-delta / max(self.temperature, 1e-8))
        accepted = np.random.random() < p
        if accepted:
            self._on_accept()
        else:
            self._on_reject()
        return accepted

    def _scalarized_cost(self, m: TuneMetrics, target_fah: float, target_recall: float) -> float:
        fah_excess = max(0, m.fah - target_fah) / max(target_fah, 1e-8)
        recall_deficit = max(0, target_recall - m.recall) / max(target_recall, 1e-8)
        return 2.0 * fah_excess + 1.0 * recall_deficit

    def _on_accept(self):
        self.temperature *= self.cooling_rate
        self.consecutive_rejections = 0

    def _on_reject(self):
        self.consecutive_rejections += 1
        if self.consecutive_rejections >= self.reheat_after:
            self.temperature *= self.reheat_factor
            self.consecutive_rejections = 0


# ============================================================================
# Section 10: Main AutoTuner Class
# ============================================================================


class AutoTuner:
    """MaxQualityAutoTuner — Sophisticated post-training auto-tuning system.

    Uses surgical gradient bursts, 7 strategy arms with Thompson sampling,
    3-pass threshold optimization, temperature scaling, simulated annealing,
    Pareto archive, INT8 shadow evaluation, and confirmation phase.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        auto_tuning_config: dict | None = None,
        console=None,
        users_hard_negs_dir: str | None = None,
    ):
        from rich.console import Console

        self.checkpoint_path = Path(checkpoint_path)
        self.config = config

        at = auto_tuning_config or config.get("auto_tuning", {})
        self.target_fah = at.get("target_fah", 2.0)
        self.target_recall = at.get("target_recall", 0.90)
        self.max_iterations = at.get("max_iterations", 50)
        self.max_gradient_steps = at.get("max_gradient_steps", 250_000)
        self.cv_folds = at.get("cv_folds", 3)
        self.confirmation_fraction = at.get("confirmation_fraction", 0.40)
        self.bootstrap_samples = at.get("bootstrap_samples", 2000)
        self.int8_shadow_enabled = at.get("int8_shadow", True)
        self.int8_shadow_interval = at.get("int8_shadow_interval", 10)
        self.require_int8_pass = at.get("require_int8_pass", True)
        self.require_confirmation = at.get("require_confirmation", True)
        self.group_key = at.get("group_key", "speaker_id")
        self.patience = at.get("patience", 15)
        self.output_dir = Path(at.get("output_dir", "./tuning_output"))
        self.users_hard_negs_dir = users_hard_negs_dir

        # Expert params
        expert = config.get("auto_tuning_expert", {})
        self.burst_steps_range = (
            expert.get("min_burst_steps", 200),
            expert.get("max_burst_steps", 25000),
        )
        self.lr_range = (
            expert.get("min_lr", 1e-7),
            expert.get("max_lr", 1e-4),
        )
        self.default_lr = expert.get("default_lr", 1e-5)
        self.sam_rho = expert.get("sam_rho", 0.05)
        self.swa_interval = expert.get("swa_collection_interval", 100)
        self.initial_annealing_temp = expert.get("initial_temperature", 0.5)
        self.cooling_rate = expert.get("cooling_rate", 0.97)
        self.reheat_after = expert.get("reheat_after", 5)
        self.reheat_factor = expert.get("reheat_factor", 1.3)
        self.active_pool_size = expert.get("active_pool_size", 16)
        self.archive_size = expert.get("pareto_archive_size", 32)
        stir_defaults = [3, 5, 7, 9, 12]
        self.stir_thresholds = [expert.get(f"stir_level_{i}", d) for i, d in enumerate(stir_defaults, 1)]
        self.curriculum_threshold = expert.get("curriculum_advance_threshold", 0.3)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
        (self.output_dir / "confirmation").mkdir(exist_ok=True)

        # Console and logging
        self.console = console or Console()
        self.file_logger = self._setup_file_logger()

        # Campaign state
        self.archive = ParetoArchive(max_size=self.archive_size)
        self.thompson = ThompsonSampler(n_arms=len(STRATEGY_ARMS))
        self.annealing = AnnealingController(
            initial_temperature=self.initial_annealing_temp,
            cooling_rate=self.cooling_rate,
            reheat_factor=self.reheat_factor,
            reheat_after=self.reheat_after,
        )
        self.stir = StirController(thresholds=self.stir_thresholds)
        self.error_memory = ErrorMemory()
        self.threshold_optimizer = ThresholdOptimizer()

        self.total_gradient_steps = 0
        self.best_checkpoint_path: Optional[str] = None

    def _setup_file_logger(self) -> logging.Logger:
        flogger = logging.getLogger(f"autotuner.{id(self)}")
        flogger.setLevel(logging.DEBUG)
        flogger.handlers.clear()
        fh = ResilientFileHandler(self.output_dir / "logs" / "autotune.log", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        flogger.addHandler(fh)
        flogger.propagate = False
        return flogger

    # ------------------------------------------------------------------
    # Data loading and partitioning
    # ------------------------------------------------------------------

    def _load_evaluation_data(self) -> tuple:
        """Load validation data and return (features, labels, sample_weights, indices).

        Returns arrays ready for in-memory evaluation.
        Data is cached after first load to avoid redundant reloading.
        """
        # Check cache first
        if hasattr(self, "_cached_eval_data") and self._cached_eval_data is not None:
            self.file_logger.info("Using cached validation data")
            return self._cached_eval_data

        import tensorflow as tf

        from src.data.dataset import WakeWordDataset

        self.file_logger.info("Loading validation data...")

        dataset = WakeWordDataset(config=self.config, split="val")
        gen_factory = dataset.val_generator_factory()
        generator = gen_factory()

        all_features = []
        all_labels = []
        all_weights = []

        for batch in generator:
            features, labels, weights = batch[0], batch[1], batch[2]
            if isinstance(features, tf.Tensor):
                features = features.numpy()
            if isinstance(labels, tf.Tensor):
                labels = labels.numpy()
            if isinstance(weights, tf.Tensor):
                weights = weights.numpy()

            all_features.append(features)
            all_labels.append(labels)
            all_weights.append(weights)

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        weights = np.concatenate(all_weights, axis=0)
        indices = np.arange(len(labels))

        group_ids: Optional[np.ndarray] = None
        if self.group_key:
            val_paths_file = dataset.data_path / "val" / "file_paths.json"
            if val_paths_file.exists():
                try:
                    with open(val_paths_file, "r", encoding="utf-8") as f:
                        file_paths = json.load(f)

                    if len(file_paths) == len(labels):
                        if self.group_key == "speaker_id":
                            parsed_groups = []
                            for p in file_paths:
                                p_str = str(p)
                                match = re.search(r"(?:^|/)speaker_[^/]+(?:/|$)", p_str)
                                if match:
                                    parsed_groups.append(match.group(0).strip("/"))
                                else:
                                    parsed_groups.append("unknown")
                            group_ids = np.array(parsed_groups, dtype=object)
                            known = int(np.sum(group_ids != "unknown"))
                            self.file_logger.info(f"Loaded group ids for partitioning ({self.group_key}): known={known}, unknown={len(group_ids) - known}, unique_groups={len(np.unique(group_ids))}")
                        else:
                            self.file_logger.warning(f"Unsupported group_key='{self.group_key}' for autotuner split; falling back to random partition")
                    else:
                        self.file_logger.warning(f"Group metadata length mismatch: file_paths={len(file_paths)} vs labels={len(labels)}; falling back to random partition")
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    self.file_logger.warning(f"Failed loading group metadata for partitioning: {e}")

        dataset.close()

        self.file_logger.info(f"Loaded {len(labels)} samples: {int(np.sum(labels >= 0.5))} positive, {int(np.sum(labels < 0.5))} negative")

        # Cache the loaded data for future iterations
        self._cached_eval_data = (features, labels, weights, indices, group_ids)
        return self._cached_eval_data

    def _partition_data(self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray, group_ids: Optional[np.ndarray] = None) -> dict:
        """Partition data: calibration 15%, search 60%, confirmation 20%, representative 5%.

        Returns dict with keys: cal, search, confirm, repr, fold_indices.
        Each value is (features, labels, weights, indices).
        """
        n = len(labels)
        indices = np.arange(n)

        n_cal = max(1, int(n * 0.15))
        n_repr = max(1, int(n * 0.05))
        n_confirm = max(1, int(n * self.confirmation_fraction))
        n_search = n - n_cal - n_repr - n_confirm

        # Guard against negative or zero search partition for small datasets
        if n_search < 1:
            # Adjust allocations deterministically: reduce confirmation first
            while n_search < 1 and n_confirm > 1:
                n_confirm -= 1
                n_search = n - n_cal - n_repr - n_confirm
            # Then reduce representative
            while n_search < 1 and n_repr > 1:
                n_repr -= 1
                n_search = n - n_cal - n_repr - n_confirm
            # Finally reduce calibration
            while n_search < 1 and n_cal > 1:
                n_cal -= 1
                n_search = n - n_cal - n_repr - n_confirm
            if n_search < 1:
                raise ValueError(f"Dataset too small to partition: n={n}, need at least 4 samples for all partitions")

        seed = int(self.config.get("training", {}).get("split_seed", 42))
        rng = np.random.RandomState(seed)

        use_group_partition = False
        if group_ids is not None and len(group_ids) == n:
            non_unknown = group_ids[group_ids != "unknown"]
            if len(np.unique(non_unknown)) >= 2:
                use_group_partition = True

        if use_group_partition:
            target_sizes = {
                "cal": n_cal,
                "search": n_search,
                "confirm": n_confirm,
                "repr": n_repr,
            }
            bins: dict[str, list[int]] = {k: [] for k in target_sizes}

            group_to_indices: dict[str, list[int]] = {}
            for idx, gid in enumerate(group_ids):
                key = str(gid)
                group_to_indices.setdefault(key, []).append(idx)

            groups = list(group_to_indices.keys())
            rng.shuffle(groups)

            # Largest groups first for more stable balancing
            groups.sort(key=lambda g: len(group_to_indices[g]), reverse=True)

            for gid in groups:
                gidx = group_to_indices[gid]
                candidate_bins = []
                for b in target_sizes:
                    target = max(target_sizes[b], 1)
                    ratio = len(bins[b]) / target
                    candidate_bins.append((ratio, len(bins[b]), b))
                candidate_bins.sort()
                chosen = candidate_bins[0][2]
                bins[chosen].extend(gidx)

            cal_idx = np.array(bins["cal"], dtype=np.int64)
            search_idx = np.array(bins["search"], dtype=np.int64)
            confirm_idx = np.array(bins["confirm"], dtype=np.int64)
            repr_idx = np.array(bins["repr"], dtype=np.int64)

            if min(len(cal_idx), len(search_idx), len(confirm_idx), len(repr_idx)) == 0:
                self.file_logger.warning("Group-aware partition created an empty bin; falling back to random partition")
                use_group_partition = False
            else:
                self.file_logger.info(f"Group-aware partition enabled ({self.group_key}): cal={len(cal_idx)}, search={len(search_idx)}, confirm={len(confirm_idx)}, repr={len(repr_idx)}")

        if not use_group_partition:
            indices = rng.permutation(indices)
            cal_idx = indices[:n_cal]
            search_idx = indices[n_cal : n_cal + n_search]
            confirm_idx = indices[n_cal + n_search : n_cal + n_search + n_confirm]
            repr_idx = indices[n_cal + n_search + n_confirm :]

        n_cal = len(cal_idx)
        n_search = len(search_idx)
        n_confirm = len(confirm_idx)
        n_repr = len(repr_idx)

        # Create CV fold indices for the search partition
        fold_indices = []
        fold_size = len(search_idx) // max(self.cv_folds, 1)
        for i in range(self.cv_folds):
            start = i * fold_size
            end = start + fold_size if i < self.cv_folds - 1 else len(search_idx)
            fold_indices.append(np.arange(start, end))

        partition = {
            "cal": (features[cal_idx], labels[cal_idx], weights[cal_idx], cal_idx),
            "search": (
                features[search_idx],
                labels[search_idx],
                weights[search_idx],
                search_idx,
            ),
            "confirm": (
                features[confirm_idx],
                labels[confirm_idx],
                weights[confirm_idx],
                confirm_idx,
            ),
            "repr": (
                features[repr_idx],
                labels[repr_idx],
                weights[repr_idx],
                repr_idx,
            ),
            "fold_indices": fold_indices,
        }

        self.file_logger.info(f"Data partitioned: cal={n_cal}, search={n_search}, confirm={n_confirm}, repr={n_repr}")
        return partition

    # ------------------------------------------------------------------
    # Model serialization helpers
    # ------------------------------------------------------------------

    def _serialize_weights(self, model) -> bytes:
        """Serialize all model weights including non-trainable state variables."""
        # Use model.get_weights() to preserve correct ordering
        # This includes both trainable and non-trainable weights
        # (e.g., BatchNorm moving_mean/moving_variance)
        weights = model.get_weights()
        return pickle.dumps(weights)

    def _deserialize_weights(self, model, weights_bytes: bytes):
        """Restore all model weights including non-trainable state variables."""
        weights = pickle.loads(weights_bytes)
        # Use model.set_weights() to restore in correct order
        # This matches model.get_weights() ordering (layer creation order)
        model.set_weights(weights)

    def _serialize_optimizer_state(self, optimizer) -> bytes:
        try:
            state = [v.numpy() for v in optimizer.variables]
            return pickle.dumps(state)
        except (AttributeError, TypeError, ValueError, RuntimeError, pickle.PickleError) as e:
            logger.warning(f"Could not serialize optimizer state, continuing without it: {e}")
            return pickle.dumps([])

    def _deserialize_optimizer_state(self, optimizer, state_bytes: bytes):
        try:
            state = pickle.loads(state_bytes)
            if state:
                for v, val in zip(optimizer.variables, state):
                    v.assign(val)
        except (AttributeError, TypeError, ValueError, RuntimeError, EOFError, pickle.PickleError) as e:
            logger.warning(f"Could not restore optimizer state, continuing without it: {e}")

    def _save_bn_state(self, model) -> dict:
        import tensorflow as tf

        bn_state = {}
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                bn_state[layer.name] = {
                    "moving_mean": layer.moving_mean.numpy().copy(),
                    "moving_variance": layer.moving_variance.numpy().copy(),
                }
        return bn_state

    def _restore_bn_state(self, model, bn_state: dict):
        import tensorflow as tf

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                if layer.name in bn_state:
                    layer.moving_mean.assign(bn_state[layer.name]["moving_mean"])
                    layer.moving_variance.assign(bn_state[layer.name]["moving_variance"])

    def _freeze_bn(self, model):
        import tensorflow as tf

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    def _unfreeze_bn(self, model):
        import tensorflow as tf

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # ------------------------------------------------------------------
    # Training: gradient burst
    # ------------------------------------------------------------------

    def _train_burst(
        self,
        model,
        optimizer,
        sampler: FocusedSampler,
        strategy_arm: int,
        candidate: CandidateState,
        n_steps: int,
        lr: float,
        use_sam: bool = False,
        use_swa: bool = False,
        recent_scores: Optional[np.ndarray] = None,
    ) -> dict:
        """Execute a short gradient burst with optional SAM/SWA.

        Returns: {'steps': int, 'final_loss': float, 'mean_loss': float,
                  'swa_snapshots': list (if SWA)}
        """
        import tensorflow as tf

        self._freeze_bn(model)
        optimizer.learning_rate.assign(lr)

        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=self.config.get("training", {}).get("label_smoothing", 0.01),
        )
        clipnorm = self.config.get("training", {}).get("gradient_clipnorm", None)
        batch_size = self.config.get("training", {}).get("batch_size", 384)

        losses = []
        swa_snapshots = []
        burst_start = time.time()
        last_heartbeat = burst_start
        heartbeat_steps = 500
        heartbeat_seconds = 60.0

        # Early stopping setup for aggressive arms
        arm_config = STRATEGY_ARMS[strategy_arm]
        early_stopping_patience = 50 if "hardest" in arm_config.name else 0
        best_loss = float("inf")
        patience_counter = 0
        min_steps_before_stop = max(50, n_steps // 4)  # Don't stop too early

        for step in range(n_steps):
            batch_features, batch_labels, batch_weights = sampler.build_batch(
                strategy_arm,
                candidate.threshold_float32,
                batch_size=batch_size,
                curriculum_stage=candidate.curriculum_stage,
                recent_scores=recent_scores,
            )

            batch_features = tf.constant(batch_features, dtype=tf.float32)
            batch_labels = tf.constant(batch_labels.reshape(-1, 1), dtype=tf.float32)
            batch_weights = tf.constant(batch_weights.reshape(-1), dtype=tf.float32)

            if use_sam:
                loss_val = self._sam_step(
                    model,
                    optimizer,
                    loss_fn,
                    batch_features,
                    batch_labels,
                    batch_weights,
                    clipnorm,
                )
            else:
                loss_val = self._standard_step(
                    model,
                    optimizer,
                    loss_fn,
                    batch_features,
                    batch_labels,
                    batch_weights,
                    clipnorm,
                )

            losses.append(float(loss_val))

            # Early stopping check for aggressive arms
            if early_stopping_patience > 0 and step >= min_steps_before_stop:
                current_loss = float(loss_val)
                if current_loss < best_loss - 1e-5:  # Improved
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.file_logger.info(f"Early stopping burst at step {step + 1}/{n_steps} (no improvement for {patience_counter} steps)")
                        break

            now = time.time()
            should_log_step = n_steps >= heartbeat_steps and (step + 1) % heartbeat_steps == 0
            should_log_time = now - last_heartbeat >= heartbeat_seconds
            if should_log_step or should_log_time:
                mean_recent = float(np.mean(losses[-100:])) if losses else 0.0
                elapsed = now - burst_start
                self.file_logger.info(f"Burst progress: step={step + 1}/{n_steps}, elapsed={elapsed:.1f}s, mean_recent_loss={mean_recent:.6f}, lr={float(optimizer.learning_rate.numpy()):.8f}")
                last_heartbeat = now

            # SWA snapshot collection
            if use_swa and (step + 1) % self.swa_interval == 0:
                snapshot = [w.numpy().copy() for w in model.trainable_weights]
                swa_snapshots.append(snapshot)

            # Cosine schedule
            if STRATEGY_ARMS[strategy_arm].use_cosine_schedule:
                arm = STRATEGY_ARMS[strategy_arm]
                lr_max = arm.lr_range[1]
                lr_min = arm.lr_range[0]
                cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / max(n_steps, 1)))
                optimizer.learning_rate.assign(cosine_lr)

        self._unfreeze_bn(model)

        return {
            "steps": len(losses),  # Actual steps completed (may be less due to early stopping)
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "swa_snapshots": swa_snapshots,
        }

    def _standard_step(self, model, optimizer, loss_fn, features, labels, weights, clipnorm):
        import tensorflow as tf

        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            per_sample_loss = loss_fn(labels, predictions)
            # Apply sample weights
            weighted_loss = tf.reduce_mean(per_sample_loss * weights)

        gradients = tape.gradient(weighted_loss, model.trainable_variables)
        if clipnorm is not None:
            gradients = [tf.clip_by_norm(g, clipnorm) if g is not None else g for g in gradients]
        optimizer.apply_gradients([(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None])
        return weighted_loss

    def _sam_step(self, model, optimizer, loss_fn, features, labels, weights, clipnorm):
        """Sharpness-Aware Minimization step."""
        import tensorflow as tf

        trainable_vars = model.trainable_variables

        # Step 1: Compute gradient at current point
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            per_sample_loss = loss_fn(labels, predictions)
            loss_val = tf.reduce_mean(per_sample_loss * weights)

        grad1 = tape.gradient(loss_val, trainable_vars)

        # Step 2: Compute epsilon = rho * grad / ||grad||
        grad_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in grad1 if g is not None))
        epsilon = []
        old_values = []
        for g, v in zip(grad1, trainable_vars):
            old_values.append(v.numpy().copy())
            if g is not None:
                e = self.sam_rho * g / (grad_norm + 1e-12)
                v.assign_add(e)
                epsilon.append(e)
            else:
                epsilon.append(None)

        # Step 3: Compute gradient at perturbed point
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            per_sample_loss = loss_fn(labels, predictions)
            loss_perturbed = tf.reduce_mean(per_sample_loss * weights)

        grad2 = tape.gradient(loss_perturbed, trainable_vars)

        # Step 4: Restore original weights
        for v, old_val in zip(trainable_vars, old_values):
            v.assign(old_val)

        # Step 5: Apply SAM gradient
        if clipnorm is not None:
            grad2 = [tf.clip_by_norm(g, clipnorm) if g is not None else g for g in grad2]
        optimizer.apply_gradients([(g, v) for g, v in zip(grad2, trainable_vars) if g is not None])
        return loss_val

    # ------------------------------------------------------------------
    # SWA averaging
    # ------------------------------------------------------------------

    def _apply_swa(self, model, swa_snapshots: list):
        """Average SWA snapshots and assign to model."""
        if not swa_snapshots:
            return
        averaged = []
        for weight_group in zip(*swa_snapshots):
            averaged.append(np.mean(weight_group, axis=0))
        for w, avg_val in zip(model.trainable_weights, averaged):
            w.assign(avg_val)

    # ------------------------------------------------------------------
    # Refresh BN statistics
    # ------------------------------------------------------------------

    def _refresh_bn_statistics(self, model, features: np.ndarray, n_batches: int = 50):
        """Run forward passes to refresh BatchNorm running statistics."""
        import tensorflow as tf

        self._unfreeze_bn(model)
        batch_size = self.config.get("training", {}).get("batch_size", 384)
        n = len(features)

        for i in range(min(n_batches, n // batch_size + 1)):
            start = (i * batch_size) % n
            end = min(start + batch_size, n)
            batch = tf.constant(features[start:end], dtype=tf.float32)
            _ = model(batch, training=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model,
        features: np.ndarray,
        labels: np.ndarray,
        ambient_hours: float,
        temperature: float = 1.0,
        fold_indices: Optional[list] = None,
    ) -> TuneMetrics:
        """Full evaluation: predict → temperature scale → threshold optimize → metrics."""
        import tensorflow as tf

        # Predict in batches
        batch_size = self.config.get("training", {}).get("batch_size", 384)
        all_scores = []
        for i in range(0, len(features), batch_size):
            batch = tf.constant(features[i : i + batch_size], dtype=tf.float32)
            preds = model(batch, training=False)
            all_scores.append(preds.numpy())

        y_scores = np.concatenate(all_scores, axis=0).flatten()

        # Apply temperature scaling
        y_scores = apply_temperature(y_scores, temperature)
        y_true = labels.flatten()

        # 3-pass threshold optimization
        threshold, threshold_uint8, metrics = self.threshold_optimizer.optimize(
            y_true,
            y_scores,
            ambient_hours,
            self.target_fah,
            self.target_recall,
            cv_folds=self.cv_folds,
            fold_indices=fold_indices,
        )

        return metrics

    def _predict_scores(self, model, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get model prediction scores with temperature scaling."""
        import tensorflow as tf

        batch_size = self.config.get("training", {}).get("batch_size", 384)
        all_scores = []
        for i in range(0, len(features), batch_size):
            batch = tf.constant(features[i : i + batch_size], dtype=tf.float32)
            preds = model(batch, training=False)
            all_scores.append(preds.numpy())

        scores = np.concatenate(all_scores, axis=0).flatten()
        return apply_temperature(scores, temperature)

    # ------------------------------------------------------------------
    # INT8 shadow evaluation
    # ------------------------------------------------------------------

    def _evaluate_int8(
        self,
        model,
        search_features: np.ndarray,
        search_labels: np.ndarray,
        repr_features: np.ndarray,
        ambient_hours: float,
    ) -> Optional[TuneMetrics]:
        """Export model to INT8 TFLite and evaluate."""
        import tensorflow as tf

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Export current candidate model to SavedModel for quantization
                saved_model_dir = os.path.join(tmpdir, "saved_model")
                if search_features.ndim != 3:
                    self.file_logger.warning(f"Unexpected search feature rank for INT8 eval: {search_features.shape}; skipping")
                    return None

                num_input_frames = int(search_features.shape[1])
                mel_bins = int(search_features.shape[2])

                export_archive = tf.keras.export.ExportArchive()
                export_archive.track(model)

                export_input_sig = [
                    tf.TensorSpec(
                        shape=(1, num_input_frames, mel_bins),
                        dtype=tf.float32,
                        name="inputs",
                    )
                ]

                def serve_fn(inputs: tf.Tensor) -> tf.Tensor:
                    return model(inputs, training=False)

                export_archive.add_endpoint(
                    name="serve",
                    fn=serve_fn,
                    input_signature=export_input_sig,
                )
                with io.StringIO() as _stdout_buf, io.StringIO() as _stderr_buf:
                    with contextlib.redirect_stdout(_stdout_buf), contextlib.redirect_stderr(_stderr_buf):
                        export_archive.write_out(saved_model_dir)

                # Build representative dataset from in-memory features for INT8 calibration

                def _representative_dataset():
                    for i in range(min(len(repr_features), 500)):
                        sample = repr_features[i].astype(np.float32)
                        if sample.ndim == 1:
                            sample = sample.reshape(num_input_frames, mel_bins)
                        elif sample.ndim == 2:
                            if sample.shape[1] != mel_bins:
                                sample = sample[:, :mel_bins]
                            if sample.shape[0] != num_input_frames:
                                if sample.shape[0] > num_input_frames:
                                    sample = sample[:num_input_frames, :]
                                else:
                                    pad = np.zeros((num_input_frames - sample.shape[0], sample.shape[1]), dtype=np.float32)
                                    sample = np.concatenate([sample, pad], axis=0)
                        sample = sample.reshape(1, num_input_frames, mel_bins)
                        yield [sample]

                # Convert SavedModel to INT8 TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = _representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.uint8
                converter._experimental_variable_quantization = True
                tflite_bytes = converter.convert()

                # Save TFLite for inference
                tflite_path = os.path.join(tmpdir, "model.tflite")
                with open(tflite_path, "wb") as f:
                    f.write(tflite_bytes)

                # Run TFLite inference
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                all_scores = []
                for i in range(len(search_features)):
                    input_data = search_features[i : i + 1].astype(np.float32)
                    # Handle potential int8 input quantization
                    if input_details[0]["dtype"] == np.int8:
                        scale = input_details[0]["quantization_parameters"]["scales"][0]
                        zp = input_details[0]["quantization_parameters"]["zero_points"][0]
                        input_data = (input_data / scale + zp).astype(np.int8)

                    interpreter.set_tensor(input_details[0]["index"], input_data)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details[0]["index"])

                    # Handle uint8 output
                    if output_details[0]["dtype"] == np.uint8:
                        scale = output_details[0]["quantization_parameters"]["scales"][0]
                        zp = output_details[0]["quantization_parameters"]["zero_points"][0]
                        score = (output.astype(np.float32) - zp) * scale
                    else:
                        score = output.astype(np.float32)

                    all_scores.append(float(score.flatten()[0]))

                y_scores = np.array(all_scores)
                y_true = search_labels.flatten()

                # Compute metrics
                threshold, threshold_uint8, metrics = self.threshold_optimizer.optimize(y_true, y_scores, ambient_hours, self.target_fah, self.target_recall)

                self.file_logger.info(f"INT8 eval: FAH={metrics.fah:.4f}, Recall={metrics.recall:.4f}, Threshold={metrics.threshold:.4f}")
                return metrics

        except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as e:
            self.file_logger.error(f"INT8 evaluation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Confirmation phase
    # ------------------------------------------------------------------

    def _confirmation_phase(
        self,
        model,
        confirm_data: tuple,
        repr_data: tuple,
        ambient_hours: float,
    ) -> tuple[Optional["CandidateState"], Optional[dict]]:
        """Final confirmation on held-out data."""
        from rich.table import Table

        self.file_logger.info("=" * 60)
        self.file_logger.info("CONFIRMATION PHASE")
        self.file_logger.info("=" * 60)

        if len(self.archive) == 0:
            self.file_logger.warning("No candidates in archive for confirmation")
            return None, None

        # Shortlist top 5 (or fewer)
        shortlist = sorted(
            self.archive.archive,
            key=lambda c: self._scalarized_score(c.eval_results) if c.eval_results else float("inf"),
        )[:5]

        confirm_features, confirm_labels, confirm_weights, _ = confirm_data
        repr_features = repr_data[0]
        best_confirmed = None
        best_attempt = None
        best_attempt_metrics = None
        best_score = float("inf")

        table = Table(title="🔬 Confirmation Results")
        table.add_column("Candidate", style="cyan")
        table.add_column("Float32 FAH", justify="right")
        table.add_column("Float32 Recall", justify="right")
        table.add_column("INT8 FAH", justify="right")
        table.add_column("INT8 Recall", justify="right")
        table.add_column("Status", justify="center")

        for candidate in shortlist:
            # Restore weights
            self._deserialize_weights(model, candidate.weights_bytes)
            self._restore_bn_state(model, candidate.batchnorm_state)

            # Evaluate on confirmation set (no re-optimization)
            scores = self._predict_scores(model, confirm_features, candidate.temperature)
            threshold = candidate.threshold_float32
            metrics = self.threshold_optimizer._compute_metrics_at_threshold(confirm_labels.flatten(), scores, threshold, ambient_hours)

            # Diagnostic: show what metrics would be with re-optimized threshold (for analysis only)
            diag_metrics = self.threshold_optimizer.optimize(confirm_labels.flatten(), scores, ambient_hours)
            if diag_metrics and diag_metrics.recall != metrics.recall:
                self.file_logger.info(
                    f"  {candidate.id} diagnostic: fixed-threshold Recall={metrics.recall:.4f} FAH={metrics.fah:.4f} | "
                    f"re-optimized Recall={diag_metrics.recall:.4f} FAH={diag_metrics.fah:.4f} thr={diag_metrics.threshold:.4f}"
                )

            # INT8 shadow evaluation (diagnostic only — current INT8 export uses
            # non-streaming batch model which doesn't match production StreamingExportModel.
            # INT8 results are logged but do NOT gate confirmation pass/fail.)
            int8_metrics = None
            if self.int8_shadow_enabled:
                int8_metrics = self._evaluate_int8(model, confirm_features, confirm_labels, repr_features, ambient_hours)
                if int8_metrics:
                    self.file_logger.info(f"  INT8 diagnostic (non-gating): FAH={int8_metrics.fah:.4f}, Recall={int8_metrics.recall:.4f}")

            # Check pass/fail — Float32 only (INT8 is diagnostic until streaming export is implemented)
            float_pass = metrics.meets_target(self.target_fah, self.target_recall)
            passed = float_pass

            status = "✅ PASS" if passed else "❌ FAIL"
            table.add_row(
                candidate.id,
                f"{metrics.fah:.4f}",
                f"{metrics.recall:.4f}",
                f"{int8_metrics.fah:.4f}" if int8_metrics else "N/A",
                f"{int8_metrics.recall:.4f}" if int8_metrics else "N/A",
                status,
            )

            if passed:
                score = self._scalarized_score(metrics)
                if score < best_score:
                    best_score = score
                    candidate.eval_results = metrics
                    candidate.eval_results_int8 = int8_metrics
                    best_confirmed = candidate
            else:
                # Track best failed attempt for reporting
                score = self._scalarized_score(metrics)
                if best_attempt is None or score < self._scalarized_score(best_attempt_metrics):
                    best_attempt = candidate
                    best_attempt_metrics = metrics

            self.file_logger.info(f"Confirmation {candidate.id}: FAH={metrics.fah:.4f}, Recall={metrics.recall:.4f}, Status={status}")

        self.console.print(table)

        if best_confirmed:
            self.file_logger.info(f"Best confirmed: {best_confirmed.id} — FAH={best_confirmed.eval_results.fah:.4f}, Recall={best_confirmed.eval_results.recall:.4f}")
        else:
            self.file_logger.warning("No candidate passed confirmation phase")
            if best_attempt and best_attempt_metrics:
                self.file_logger.info(f"Best failed attempt: {best_attempt.id} — FAH={best_attempt_metrics.fah:.4f}, Recall={best_attempt_metrics.recall:.4f}")

        return best_confirmed, best_attempt_metrics

    def _scalarized_score(self, m: Optional[TuneMetrics]) -> float:
        if m is None:
            return float("inf")
        fah_excess = max(0, m.fah - self.target_fah) / max(self.target_fah, 1e-8)
        recall_deficit = max(0, self.target_recall - m.recall) / max(self.target_recall, 1e-8)
        return 2.0 * fah_excess + 1.0 * recall_deficit

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model, metrics: TuneMetrics, iteration: int) -> str:
        """Save model checkpoint and return path."""
        name = f"tuned_fah{metrics.fah:.3f}_rec{metrics.recall:.3f}_iter{iteration}.weights.h5"
        path = self.output_dir / "checkpoints" / name
        model.save_weights(str(path))
        self.file_logger.info(f"Checkpoint saved: {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Stir mechanisms
    # ------------------------------------------------------------------

    def _apply_stir(
        self,
        model,
        optimizer,
        candidate: CandidateState,
        stir_level: int,
        sampler: FocusedSampler,
        recent_scores: Optional[np.ndarray],
    ) -> dict:
        """Apply stir mechanism based on level. Returns info dict."""
        if stir_level <= 0:
            return {"level": 0, "action": "none"}

        info = {"level": stir_level, "action": ""}

        if stir_level >= 1:
            # L1: SWA collection will happen in train_burst
            info["action"] = "swa_collection"

        if stir_level >= 2:
            # L2: SAM will be enabled in train_burst
            info["action"] = "sam_flatten"

        if stir_level >= 3:
            # L3: Loss landscape probing — try random weight perturbations
            info["action"] = "landscape_probe"
            best_loss = float("inf")
            best_weights = None
            original_weights = self._serialize_weights(model)

            for probe in range(4):
                self._deserialize_weights(model, original_weights)
                for w in model.trainable_weights:
                    noise = np.random.normal(0, 0.001, w.shape)
                    w.assign_add(noise)

                # Quick eval
                batch = sampler.build_batch(
                    candidate.strategy_arm if candidate.strategy_arm >= 0 else 5,
                    candidate.threshold_float32,
                    batch_size=64,
                    recent_scores=recent_scores,
                )
                import tensorflow as tf

                preds = model(tf.constant(batch[0], dtype=tf.float32), training=False)
                loss = float(tf.keras.losses.binary_crossentropy(batch[1].reshape(-1, 1), preds).numpy().mean())
                if loss < best_loss:
                    best_loss = loss
                    best_weights = self._serialize_weights(model)

            if best_weights is not None:
                self._deserialize_weights(model, best_weights)
            else:
                self._deserialize_weights(model, original_weights)

        if stir_level >= 5:
            # L5: Gaussian noise perturbation + reheat annealing
            info["action"] = "diversify"
            for w in model.trainable_weights:
                noise = np.random.normal(0, 0.001, w.shape)
                w.assign_add(noise)
            self.annealing.temperature *= self.reheat_factor

        self.file_logger.info(f"Stir L{stir_level} applied: {info['action']}")
        return info

    # ------------------------------------------------------------------
    # Parent selection
    # ------------------------------------------------------------------

    def _select_parent(self, active_pool: list[CandidateState]) -> CandidateState:
        """Tournament selection from active pool."""
        if len(active_pool) <= 1:
            return active_pool[0]

        # Tournament of 3
        tournament_size = min(3, len(active_pool))
        contenders = np.random.choice(len(active_pool), size=tournament_size, replace=False)

        best = None
        best_score = float("inf")
        for idx in contenders:
            c = active_pool[idx]
            score = self._scalarized_score(c.eval_results)
            if score < best_score:
                best_score = score
                best = c

        return best

    # ------------------------------------------------------------------
    # Rich logging
    # ------------------------------------------------------------------

    def _log_header(self):
        from rich.panel import Panel

        header = (
            f"[bold]MaxQualityAutoTuner[/bold]\n"
            f"Target: FAH ≤ {self.target_fah:.2f}, Recall ≥ {self.target_recall:.2f}\n"
            f"Max iterations: {self.max_iterations}, "
            f"Max gradient steps: {self.max_gradient_steps:,}\n"
            f"Checkpoint: {self.checkpoint_path}\n"
            f"Output: {self.output_dir}"
        )
        self.console.print(Panel(header, title="🎯 Auto-Tuning Campaign", border_style="blue"))

    def _log_iteration(
        self,
        iteration: int,
        arm: StrategyArm,
        metrics: TuneMetrics,
        accepted: bool,
        stir_level: int,
        burst_info: dict,
    ):

        status = "✅" if accepted else "❌"
        stir_str = f"⚡L{stir_level}" if stir_level > 0 else ""
        target_met = "🎯" if metrics.meets_target(self.target_fah, self.target_recall) else ""

        self.console.print(
            f"  [{iteration:3d}/{self.max_iterations}] "
            f"{arm.name:20s} │ "
            f"FAH={metrics.fah:8.4f} │ "
            f"Recall={metrics.recall:.4f} │ "
            f"AUC-PR={metrics.auc_pr:.4f} │ "
            f"Thr={metrics.threshold:.4f} │ "
            f"Loss={burst_info.get('mean_loss', 0):.4f} │ "
            f"{status} {stir_str} {target_met}"
        )

        self.file_logger.info(
            f"Iter {iteration}: arm={arm.name}, FAH={metrics.fah:.4f}, "
            f"Recall={metrics.recall:.4f}, AUC-PR={metrics.auc_pr:.4f}, "
            f"threshold={metrics.threshold:.4f}, accepted={accepted}, "
            f"stir={stir_level}, steps={burst_info.get('steps', 0)}, "
            f"total_steps={self.total_gradient_steps}"
        )

    def _log_final_summary(self, result: dict):
        from rich.panel import Panel
        from rich.table import Table

        # Summary panel
        met = "✅ YES" if result["target_met"] else "❌ NO"
        elapsed = result.get("elapsed_seconds", 0)
        minutes = elapsed / 60

        # Check if confirmation was attempted but failed
        confirmation_attempted = result.get("confirmation_attempted", False)
        confirmation_failed = confirmation_attempted and not result["target_met"]

        if confirmation_failed:
            # Show confirmation metrics (best failed attempt) instead of search-set metrics
            conf_fah = result.get("confirmation_best_fah")
            conf_recall = result.get("confirmation_best_recall")
            summary = (
                f"Target met: {met}\n"
                f"⚠️ Confirmation failed — showing best confirmation attempt:\n"
                f"Best FAH: {conf_fah:.4f}\n"
                f"Best Recall: {conf_recall:.4f}\n"
                f"Search-set metrics (overfit): FAH={result['best_fah']:.4f}, Recall={result['best_recall']:.4f}\n"
                f"Total iterations: {result['iterations']}\n"
                f"Total gradient steps: {self.total_gradient_steps:,}\n"
                f"Wall clock: {minutes:.1f} min\n"
                f"Best checkpoint: {result.get('best_checkpoint', 'N/A')}"
            )

        else:
            summary = (
                f"Target met: {met}\n"
                f"Best FAH: {result['best_fah']:.4f}\n"
                f"Best Recall: {result['best_recall']:.4f}\n"
                f"Total iterations: {result['iterations']}\n"
                f"Total gradient steps: {self.total_gradient_steps:,}\n"
                f"Wall clock: {minutes:.1f} min\n"
                f"Best checkpoint: {result.get('best_checkpoint', 'N/A')}"
            )

        self.console.print(Panel(summary, title="📊 Auto-Tuning Results", border_style="green" if result["target_met"] else "red"))

        # Pareto frontier table
        frontier = result.get("pareto_frontier", [])
        if frontier:
            table = Table(title="🏔️ Pareto Frontier")
            table.add_column("ID", style="cyan")
            table.add_column("FAH", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("AUC-PR", justify="right")
            table.add_column("Threshold", justify="right")
            table.add_column("Arm", style="yellow")
            table.add_column("Iter", justify="right")

            for p in frontier:
                table.add_row(
                    p["id"],
                    f"{p['fah']:.4f}",
                    f"{p['recall']:.4f}",
                    f"{p['auc_pr']:.4f}",
                    f"{p['threshold']:.4f}",
                    p["arm"],
                    str(p["iteration"]),
                )
            self.console.print(table)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def tune(self) -> dict:
        """Main orchestration loop. Returns results dict."""
        import tensorflow as tf

        from src.model.architecture import build_model
        from src.utils.performance import set_threading_config

        hw = self.config.get("hardware", {})
        set_threading_config(
            inter_op_parallelism=hw.get("inter_op_parallelism", 16),
            intra_op_parallelism=hw.get("intra_op_parallelism", 16),
            num_threads=hw.get("num_threads_per_worker", None),
        )
        wall_start = time.time()

        self._log_header()
        self.file_logger.info("=" * 60)
        self.file_logger.info("MaxQualityAutoTuner campaign started")
        self.file_logger.info("=" * 60)

        # 1. Build model and load base checkpoint
        model_cfg = self.config.get("model", {})
        hardware_cfg = self.config.get("hardware", {})
        clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        stride = model_cfg.get("stride", 3)
        num_time_frames = int(clip_duration_ms / window_step_ms)
        model = build_model(
            input_shape=(num_time_frames, mel_bins),
            num_classes=2,
            first_conv_filters=model_cfg.get("first_conv_filters", 32),
            first_conv_kernel_size=model_cfg.get("first_conv_kernel_size", 5),
            stride=stride,
            pointwise_filters=model_cfg.get("pointwise_filters", "64,64,64,64"),
            mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
            repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
            residual_connection=model_cfg.get("residual_connection", "0,1,1,1"),
            dropout_rate=model_cfg.get("dropout_rate", 0.08),
            l2_regularization=model_cfg.get("l2_regularization", 0.00003),
        )
        # Build model by running a forward pass before loading weights
        model(tf.zeros((1, num_time_frames, mel_bins), dtype=tf.float32), training=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.load_weights(str(self.checkpoint_path))
        self.file_logger.info(f"Model loaded from {self.checkpoint_path}")

        # 2. Load and partition data
        features, labels, weights, indices, group_ids = self._load_evaluation_data()
        ambient_hours = self.config.get("training", {}).get("ambient_duration_hours", 42.02)
        # Scale ambient hours by validation split fraction
        val_split = self.config.get("training", {}).get("val_split", 0.1)
        ambient_hours_val = ambient_hours * val_split

        partition = self._partition_data(features, labels, weights, group_ids=group_ids)
        search_features, search_labels, search_weights, search_indices = partition["search"]
        cal_features, cal_labels = partition["cal"][0], partition["cal"][1]
        confirm_data = partition["confirm"]
        repr_data = partition["repr"]
        fold_indices = partition["fold_indices"]

        # Scale ambient hours for search partition
        search_fraction = len(search_labels) / max(len(labels), 1)
        ambient_hours_search = ambient_hours_val * search_fraction

        # Scale ambient hours for confirmation partition
        confirm_labels_for_fraction = confirm_data[1]
        confirm_fraction = len(confirm_labels_for_fraction) / max(len(labels), 1)
        ambient_hours_confirm = ambient_hours_val * confirm_fraction

        # 3. Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.default_lr)
        # Build optimizer state by doing a dummy step
        dummy_grads = [tf.zeros_like(v) for v in model.trainable_variables]
        optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))

        # 4. Evaluate base model
        self.file_logger.info("Evaluating base model...")
        base_temperature = fit_temperature(
            self._predict_scores(model, cal_features),
            (cal_labels >= 0.5).astype(float).flatten(),
        )

        base_metrics = self._evaluate_model(
            model,
            search_features,
            search_labels,
            ambient_hours_search,
            temperature=base_temperature,
            fold_indices=fold_indices,
        )

        self.file_logger.info(f"Base model: FAH={base_metrics.fah:.4f}, Recall={base_metrics.recall:.4f}, AUC-PR={base_metrics.auc_pr:.4f}, Temperature={base_temperature:.4f}")
        self.console.print(f"  [bold]Base model[/bold]: FAH={base_metrics.fah:.4f}, Recall={base_metrics.recall:.4f}, AUC-PR={base_metrics.auc_pr:.4f}")

        # 5. Create initial candidate
        initial_candidate = CandidateState(
            id="c_000",
            weights_bytes=self._serialize_weights(model),
            optimizer_state_bytes=self._serialize_optimizer_state(optimizer),
            batchnorm_state=self._save_bn_state(model),
            temperature=base_temperature,
            threshold_float32=base_metrics.threshold,
            threshold_uint8=base_metrics.threshold_uint8,
            eval_results=base_metrics,
            iteration=0,
            lr=self.default_lr,
        )
        self.archive.try_add(initial_candidate)

        # Active pool
        active_pool = [initial_candidate]

        # Get initial scores for sampler
        recent_scores = self._predict_scores(model, search_features, base_temperature)

        # 6. Main loop
        consecutive_target_met = 0
        last_iteration = 0

        try:
            for iteration in range(1, self.max_iterations + 1):
                last_iteration = iteration

                # Check gradient budget
                if self.total_gradient_steps >= self.max_gradient_steps:
                    self.file_logger.info(f"Gradient step budget exhausted: {self.total_gradient_steps}/{self.max_gradient_steps}")
                    break

                # a. Select parent
                parent = self._select_parent(active_pool)

                # b. Restore parent state
                self._deserialize_weights(model, parent.weights_bytes)
                self._restore_bn_state(model, parent.batchnorm_state)
                self._deserialize_optimizer_state(optimizer, parent.optimizer_state_bytes)

                # c. Diagnose regime and select strategy arm
                regime = diagnose_regime(parent.eval_results, self.target_fah, self.target_recall)
                arm_idx = self.thompson.select_arm(regime)
                arm = STRATEGY_ARMS[arm_idx]

                # d. Determine learning rate
                lr = arm.default_lr if arm.default_lr > 0 else parent.lr
                lr = max(self.lr_range[0], min(lr, self.lr_range[1]))

                # e. Determine burst steps
                n_steps = arm.default_steps
                n_steps = max(
                    self.burst_steps_range[0],
                    min(n_steps, self.burst_steps_range[1]),
                )
                # Don't exceed remaining budget
                remaining = self.max_gradient_steps - self.total_gradient_steps
                n_steps = min(n_steps, remaining)

                # f. Check and apply stir
                stir_level = self.stir.get_stir_level(parent.stagnation_count)
                use_sam = arm.use_sam or stir_level >= 2
                use_swa = arm.use_swa or stir_level >= 1

                self._apply_stir(
                    model,
                    optimizer,
                    parent,
                    stir_level,
                    FocusedSampler(search_features, search_labels, search_weights, self.error_memory),
                    recent_scores,
                )

                # g. Build focused sampler
                sampler = FocusedSampler(
                    search_features,
                    search_labels,
                    search_weights,
                    self.error_memory,
                )

                # h. Execute gradient burst
                burst_info = self._train_burst(
                    model,
                    optimizer,
                    sampler,
                    arm_idx,
                    parent,
                    n_steps,
                    lr,
                    use_sam=use_sam,
                    use_swa=use_swa,
                    recent_scores=recent_scores,
                )
                self.total_gradient_steps += n_steps

                # i. SWA averaging if snapshots collected
                swa_snapshots = burst_info.get("swa_snapshots", [])
                if swa_snapshots and len(swa_snapshots) >= 2:
                    # Save pre-SWA weights
                    _ = self._serialize_weights(model)
                    self._apply_swa(model, swa_snapshots)

                # j. Refresh BN statistics
                self._refresh_bn_statistics(model, search_features)

                # k. Recalibrate temperature
                cal_scores = self._predict_scores(model, cal_features)
                temperature = fit_temperature(
                    cal_scores,
                    (cal_labels >= 0.5).astype(float).flatten(),
                )

                # l. Update recent_scores
                recent_scores = self._predict_scores(model, search_features, temperature)

                # m. Full evaluation
                eval_metrics = self._evaluate_model(
                    model,
                    search_features,
                    search_labels,
                    ambient_hours_search,
                    temperature=temperature,
                    fold_indices=fold_indices,
                )
                # Diagnostic: per-set metrics after search evaluation
                self.file_logger.info(f"Iter {iteration}: Search FAH={eval_metrics.fah:.4f}, Recall={eval_metrics.recall:.4f}, AUC-PR={eval_metrics.auc_pr:.4f}")

                # n. INT8 shadow evaluation
                int8_metrics = None
                if self.int8_shadow_enabled and iteration % self.int8_shadow_interval == 0:
                    int8_metrics = self._evaluate_int8(
                        model,
                        search_features,
                        search_labels,
                        repr_data[0],
                        ambient_hours_search,
                    )
                    if int8_metrics is not None:
                        self.file_logger.info(f"Iter {iteration}: INT8 shadow FAH={int8_metrics.fah:.4f}, Recall={int8_metrics.recall:.4f}")

                # o. Build new candidate
                new_candidate = CandidateState(
                    id=f"c_{iteration:03d}",
                    weights_bytes=self._serialize_weights(model),
                    optimizer_state_bytes=self._serialize_optimizer_state(optimizer),
                    batchnorm_state=self._save_bn_state(model),
                    temperature=temperature,
                    threshold_float32=eval_metrics.threshold,
                    threshold_uint8=eval_metrics.threshold_uint8,
                    eval_results=eval_metrics,
                    eval_results_int8=int8_metrics,
                    strategy_arm=arm_idx,
                    parent_id=parent.id,
                    iteration=iteration,
                    lr=lr,
                    history=parent.history
                    + [
                        {
                            "iter": iteration,
                            "arm": arm.name,
                            "fah": eval_metrics.fah,
                            "recall": eval_metrics.recall,
                        }
                    ],
                )

                # p. Accept/reject
                accepted = self.annealing.should_accept(
                    eval_metrics,
                    parent.eval_results,
                    self.target_fah,
                    self.target_recall,
                )

                # q. Update Thompson sampling
                improvement = eval_metrics.dominates(parent.eval_results)
                self.thompson.update(arm_idx, improvement)

                # r. Update error memory
                search_all_indices = np.arange(len(search_labels))
                self.error_memory.update(
                    search_all_indices,
                    search_labels.flatten(),
                    recent_scores,
                    eval_metrics.threshold,
                )

                # s. Manage active pool and archive
                if accepted:
                    new_candidate.stagnation_count = 0
                    # Replace parent in active pool
                    for i, c in enumerate(active_pool):
                        if c.id == parent.id:
                            active_pool[i] = new_candidate
                            break
                    else:
                        if len(active_pool) < self.active_pool_size:
                            active_pool.append(new_candidate)
                        else:
                            # Replace worst
                            worst_idx = max(
                                range(len(active_pool)),
                                key=lambda i: self._scalarized_score(active_pool[i].eval_results),
                            )
                            active_pool[worst_idx] = new_candidate
                else:
                    parent.stagnation_count += 1

                # Try to add to Pareto archive
                self.archive.try_add(new_candidate)

                # t. Save checkpoint if target met
                if eval_metrics.meets_target(self.target_fah, self.target_recall):
                    ckpt_path = self._save_checkpoint(model, eval_metrics, iteration)
                    if self.best_checkpoint_path is None:
                        self.best_checkpoint_path = ckpt_path
                    elif eval_metrics.recall > base_metrics.recall:
                        self.best_checkpoint_path = ckpt_path
                    consecutive_target_met += 1
                else:
                    consecutive_target_met = 0

                # u. Log iteration
                self._log_iteration(iteration, arm, eval_metrics, accepted, stir_level, burst_info)

                # v. Early termination: stable target met
                if consecutive_target_met >= 3:
                    self.file_logger.info("Target met for 3 consecutive iterations — stable convergence")
                    self.console.print(f"  [bold green]✅ Stable target reached after {iteration} iterations[/bold green]")
                    break

        except KeyboardInterrupt:
            self.file_logger.warning("Campaign interrupted by user")
            self.console.print("\n  [bold yellow]⚠️ Campaign interrupted[/bold yellow]")

        # 7. Confirmation phase
        confirmed = None
        best_attempt_metrics = None
        if self.require_confirmation and len(self.archive) > 0:
            confirmed, best_attempt_metrics = self._confirmation_phase(model, confirm_data, repr_data, ambient_hours_confirm)
            if confirmed is not None:
                # Save confirmed checkpoint
                self._deserialize_weights(model, confirmed.weights_bytes)
                self._restore_bn_state(model, confirmed.batchnorm_state)
                self.best_checkpoint_path = self._save_checkpoint(model, confirmed.eval_results, confirmed.iteration)

        # 8. Build final results
        best = confirmed or self.archive.get_best(self.target_fah, self.target_recall)
        if best is None and len(self.archive) > 0:
            best = self.archive.archive[0]

        # Ensure we have a checkpoint for the best
        if best is not None and self.best_checkpoint_path is None:
            self._deserialize_weights(model, best.weights_bytes)
            self._restore_bn_state(model, best.batchnorm_state)
            self.best_checkpoint_path = self._save_checkpoint(model, best.eval_results, best.iteration)

        elapsed = time.time() - wall_start
        if self.require_confirmation:
            target_met = confirmed is not None and confirmed.eval_results is not None and confirmed.eval_results.meets_target(self.target_fah, self.target_recall)
        else:
            target_met = best is not None and best.eval_results is not None and best.eval_results.meets_target(self.target_fah, self.target_recall)

        result = {
            "best_fah": best.eval_results.fah if best and best.eval_results else float("inf"),
            "best_recall": best.eval_results.recall if best and best.eval_results else 0.0,
            "final_fah": best.eval_results.fah if best and best.eval_results else float("inf"),
            "final_recall": best.eval_results.recall if best and best.eval_results else 0.0,
            "iterations": last_iteration,
            "best_checkpoint": self.best_checkpoint_path or "",
            "target_met": target_met,
            "pareto_frontier": self.archive.get_frontier_points(),
            "elapsed_seconds": elapsed,
            "total_gradient_steps": self.total_gradient_steps,
            "confirmation_attempted": self.require_confirmation and len(self.archive) > 0,
            "confirmation_best_fah": best_attempt_metrics.fah if best_attempt_metrics else None,
            "confirmation_best_recall": best_attempt_metrics.recall if best_attempt_metrics else None,
        }

        self._log_final_summary(result)
        return result


# ============================================================================
# Section 11: Convenience Functions
# ============================================================================


def autotune(
    checkpoint_path: str,
    config: dict,
    output_dir: str = "./tuning_output",
    target_fah: float = 0.3,
    target_recall: float = 0.92,
    max_iterations: int = 100,
) -> dict:
    """Convenience function for auto-tuning a trained wake word model."""
    at_config = config.get("auto_tuning", {}).copy()
    at_config["output_dir"] = output_dir
    at_config["target_fah"] = target_fah
    at_config["target_recall"] = target_recall
    at_config["max_iterations"] = max_iterations

    tuner = AutoTuner(checkpoint_path, config, at_config)
    return tuner.tune()


def main() -> int:
    """CLI entry point — delegates to cli.py."""
    from src.tuning.cli import main as cli_main

    return cli_main()
