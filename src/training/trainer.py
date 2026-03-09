"""Trainer module for wake word model training.

Step-based training loop with:
- Class weighting for imbalanced data
- Two-priority checkpoint selection
- Evaluation at regular intervals
- Mixed precision training support
- Integration with TrainingProfiler
"""

import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# XLA JIT compilation: let TF decide (don't disable)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.spec_augment_gpu import batch_spec_augment_gpu
from src.evaluation.calibration import compute_brier_score
from src.evaluation.metrics import MetricsCalculator
from src.model.architecture import build_model
from src.training.mining import AsyncHardExampleMiner, HardExampleMiner, log_false_predictions_to_json, run_top_fp_extraction
from src.training.profiler import TFProfiler, TrainingProfiler
from src.training.rich_logger import RichTrainingLogger
from src.training.tensorboard_logger import TensorBoardLogger
from src.utils.performance import get_system_info
from src.utils.terminal_logger import TerminalLogger


class EvaluationMetrics:
    """Wrapper around MetricsCalculator that supports batch accumulation.

    This provides backward compatibility with TrainingMetrics' update pattern
    while using the evaluation package for computations.
    """

    def __init__(
        self,
        cutoffs: list[float] | None = None,
        ambient_duration_hours: float = 0.0,
        default_threshold: float = 0.5,
    ):
        """Initialize metrics tracker.

        Args:
            cutoffs: Array of threshold values (default: 101 points from 0 to 1)
        """
        if cutoffs is None:
            cutoffs_list = np.linspace(0.0, 1.0, 101).tolist()
        else:
            cutoffs_list = cutoffs if isinstance(cutoffs, list) else list(cutoffs)
        self.cutoffs: list[float] = [float(cutoff) for cutoff in cutoffs_list]
        self.ambient_duration_hours = ambient_duration_hours
        self.default_threshold = default_threshold

        # Accumulated predictions and labels
        self.all_y_true: list = []
        self.all_y_scores: list = []

        # Per-threshold counts stored as numpy arrays (indexed by cutoff position).
        # Using arrays avoids 101-iteration Python loops during accumulation.
        n = len(self.cutoffs)
        self._tp_arr = np.zeros(n, dtype=np.int64)
        self._fp_arr = np.zeros(n, dtype=np.int64)
        self._tn_arr = np.zeros(n, dtype=np.int64)
        self._fn_arr = np.zeros(n, dtype=np.int64)

        # Backward-compatible dict views for tests/consumers
        self.tp_at_threshold: dict[float, int] = dict.fromkeys(self.cutoffs, 0)
        self.fp_at_threshold: dict[float, int] = dict.fromkeys(self.cutoffs, 0)
        self.tn_at_threshold: dict[float, int] = dict.fromkeys(self.cutoffs, 0)
        self.fn_at_threshold: dict[float, int] = dict.fromkeys(self.cutoffs, 0)

    def update(self, y_true: np.ndarray, y_scores: np.ndarray) -> None:
        """Update metrics with batch predictions.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_scores: Prediction scores (0 to 1)
        """
        # Flatten arrays to prevent broadcasting issues
        y_true_flat = np.ravel(y_true)
        y_scores_flat = np.ravel(y_scores)

        self.all_y_true.extend(y_true_flat.tolist())
        self.all_y_scores.extend(y_scores_flat.tolist())

        # Vectorized update across all thresholds using numpy broadcasting
        cutoffs_arr = np.array(self.cutoffs, dtype=np.float32).reshape(-1, 1)  # [101, 1]
        y_true_2d = y_true_flat.reshape(1, -1)  # [1, N]
        y_scores_2d = y_scores_flat.reshape(1, -1)  # [1, N]

        # [101, N] boolean masks — no dtype cast needed
        y_pred_2d = y_scores_2d >= cutoffs_arr
        pos_mask = y_true_2d == 1
        neg_mask = ~pos_mask

        # Vectorized sum along axis=1 (samples) — replaces 101-iteration Python dict loop
        self._tp_arr += np.sum(y_pred_2d & pos_mask, axis=1)
        self._fp_arr += np.sum(y_pred_2d & neg_mask, axis=1)
        self._tn_arr += np.sum(~y_pred_2d & neg_mask, axis=1)
        self._fn_arr += np.sum(~y_pred_2d & pos_mask, axis=1)

        # Keep dict views in sync
        for i, cutoff in enumerate(self.cutoffs):
            self.tp_at_threshold[cutoff] = int(self._tp_arr[i])
            self.fp_at_threshold[cutoff] = int(self._fp_arr[i])
            self.tn_at_threshold[cutoff] = int(self._tn_arr[i])
            self.fn_at_threshold[cutoff] = int(self._fn_arr[i])

    def compute_metrics(self) -> dict[str, float]:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dictionary of computed metrics
        """
        if not self.all_y_true:
            return {}

        y_true = np.array(self.all_y_true)
        y_scores = np.array(self.all_y_scores)

        # Use MetricsCalculator for main metrics computation
        calc = MetricsCalculator(
            y_true=y_true,
            y_score=y_scores,
            ambient_duration_hours=self.ambient_duration_hours,
        )

        # Get basic metrics using default threshold
        metrics = calc.compute_all_metrics(
            ambient_duration_hours=self.ambient_duration_hours,
            threshold=self.default_threshold,
        )

        # Add per-threshold metrics for backward compatibility (ROC/PR curves)
        for i, cutoff in enumerate(self.cutoffs):
            tp = int(self._tp_arr[i])
            fp = int(self._fp_arr[i])
            tn = int(self._tn_arr[i])
            fn = int(self._fn_arr[i])
            total = tp + fp + tn + fn
            metrics[f"precision_{cutoff:.2f}"] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics[f"recall_{cutoff:.2f}"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"accuracy_{cutoff:.2f}"] = (tp + tn) / total if total > 0 else 0

        return metrics

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.all_y_true = []
        self.all_y_scores = []
        n = len(self.cutoffs)
        self._tp_arr = np.zeros(n, dtype=np.int64)
        self._fp_arr = np.zeros(n, dtype=np.int64)
        self._tn_arr = np.zeros(n, dtype=np.int64)
        self._fn_arr = np.zeros(n, dtype=np.int64)
        self.tp_at_threshold = dict.fromkeys(self.cutoffs, 0)
        self.fp_at_threshold = dict.fromkeys(self.cutoffs, 0)
        self.tn_at_threshold = dict.fromkeys(self.cutoffs, 0)
        self.fn_at_threshold = dict.fromkeys(self.cutoffs, 0)

    def get_counts_at_threshold(self, threshold: float) -> tuple[int, int, int, int]:
        """Return (tp, fp, tn, fn) counts at the nearest stored cutoff to threshold."""
        cutoffs_arr = np.array(self.cutoffs)
        idx = int(np.argmin(np.abs(cutoffs_arr - threshold)))
        return int(self._tp_arr[idx]), int(self._fp_arr[idx]), int(self._tn_arr[idx]), int(self._fn_arr[idx])


# Backward-compatible alias
TrainingMetrics = EvaluationMetrics


class Trainer:
    """Training orchestrator with step-based training loop."""

    def __init__(self, config: dict[str, Any]):
        """Initialize trainer.

        Args:
            config: Training configuration from config loader
        """
        self.config = config
        self.model: tf.keras.Model | None = None
        self.profiler: TrainingProfiler | None = None
        self.tf_profiler: TFProfiler | None = None

        # Rich terminal logger - initialize early for seed logging
        self.logger = RichTrainingLogger()

        # Extract training config
        training = config.get("training", {})
        self.training_steps_list = training.get("training_steps", [20000, 10000])
        self.learning_rates = training.get("learning_rates", [0.001, 0.0001])
        self.batch_size = training.get("batch_size", 128)
        self.eval_step_interval = training.get("eval_step_interval", 500)
        self.eval_basic_step_interval = training.get("eval_basic_step_interval", 1000)
        self.materialize_metrics_interval = int(training.get("materialize_metrics_interval", self.eval_basic_step_interval) or self.eval_basic_step_interval)
        if self.materialize_metrics_interval <= 0:
            self.materialize_metrics_interval = self.eval_basic_step_interval
        self.eval_advanced_step_interval = training.get("eval_advanced_step_interval", 5000)
        self.eval_confusion_matrix_interval = training.get("eval_confusion_matrix_interval", 5000)
        self.eval_checkpoints_interval = training.get("eval_checkpoints_interval", 5000)
        self.eval_log_every_step = bool(training.get("eval_log_every_step", True))
        self._last_basic_eval_step = 0
        self._last_advanced_eval_step = 0
        self.steps_per_epoch = training.get("steps_per_epoch", 1000)  # For mining epoch calculation

        # Seed control for reproducibility
        self.random_seed = training.get("random_seed", None)
        self._seed_applied = False
        if self.random_seed is not None:
            from src.utils.seed import seed_everything

            seed_everything(int(self.random_seed))
            self._seed_applied = True
        # Class weights (positive upweighted to compensate for class imbalance)
        self.positive_weights = training.get("positive_class_weight", [1.0, 1.0])
        self.negative_weights = training.get("negative_class_weight", [20.0, 20.0])
        self.hard_negative_weights = training.get("hard_negative_class_weight", [40.0, 40.0])

        # Ensure all per-phase lists have the same length as training_steps_list
        n_phases = len(self.training_steps_list)

        def _pad_or_trim(lst, default):
            """Pad (repeating last element) or trim lst to n_phases elements."""
            if not lst:
                return [default] * n_phases
            if len(lst) >= n_phases:
                return lst[:n_phases]
            return lst + [lst[-1]] * (n_phases - len(lst))

        self.learning_rates = _pad_or_trim(self.learning_rates, 0.0001)
        self.positive_weights = _pad_or_trim(self.positive_weights, 1.0)
        self.negative_weights = _pad_or_trim(self.negative_weights, 20.0)
        self.hard_negative_weights = _pad_or_trim(self.hard_negative_weights, 40.0)

        # SpecAugment configuration (per-phase)
        self.time_mask_max_size = _pad_or_trim(training.get("time_mask_max_size", [0, 0]), 0)
        self.time_mask_count = _pad_or_trim(training.get("time_mask_count", [0, 0]), 0)
        self.freq_mask_max_size = _pad_or_trim(training.get("freq_mask_max_size", [0, 0]), 0)
        self.freq_mask_count = _pad_or_trim(training.get("freq_mask_count", [0, 0]), 0)
        self.spec_augment_enabled = any(self.time_mask_max_size + self.time_mask_count + self.freq_mask_max_size + self.freq_mask_count)

        # Checkpoint selection config
        self.minimization_metric = training.get("minimization_metric", "ambient_false_positives_per_hour")
        self.target_minimization = training.get("target_minimization", 0.5)
        self.maximization_metric = training.get("maximization_metric", "average_viable_recall")

        # Performance config
        performance = config.get("performance", {})
        self.mixed_precision = performance.get("mixed_precision", True)
        self.enable_profiling = performance.get("enable_profiling", True)
        self.profile_every_n = performance.get("profile_every_n_steps", 100)
        self.profile_output_dir = performance.get("profile_output_dir", "./profiles")
        self.num_workers = performance.get("num_workers", 16)
        self.prefetch_factor = performance.get("prefetch_factor", 8)
        self.pin_memory = performance.get("pin_memory", True)
        self.inter_op_parallelism = performance.get("inter_op_parallelism", 16)
        self.intra_op_parallelism = performance.get("intra_op_parallelism", 16)
        self.tensorboard_enabled = performance.get("tensorboard_enabled", True)
        self.tensorboard_log_dir = performance.get("tensorboard_log_dir", "./logs")
        self.prefetch_buffer = performance.get("prefetch_buffer", 12)
        self.use_tfdata = performance.get("use_tfdata", True)
        self.log_throughput = performance.get("log_throughput", True)
        mining_cfg = config.get("mining", {})
        self.async_mining = mining_cfg.get("async_mining", performance.get("async_mining", False))
        self.spec_augment_backend = performance.get("spec_augment_backend", "tf")
        self.log_throughput_interval = int(performance.get("log_throughput_interval", 1000) or 1000)
        self.tf_profile_start_step = int(performance.get("tf_profile_start_step", 100) or 0)
        self.gpu_memory_log_interval = int(performance.get("gpu_memory_log_interval", 1000) or 0)
        self.tensorboard_writer: tf.summary.SummaryWriter | None = None
        self.tensorboard_logger: TensorBoardLogger | None = None

        # Log seed after logger is initialized
        if self._seed_applied:
            self.logger.log_info(f"Reproducibility: random_seed={self.random_seed} applied (Python, NumPy, TensorFlow, TF_DETERMINISTIC_OPS=1)")

        # Calculate input shape from hardware config
        hardware = config.get("hardware", {})
        clip_duration_ms = hardware.get("clip_duration_ms", 1000)
        window_step_ms = hardware.get("window_step_ms", 10)
        mel_bins = hardware.get("mel_bins", 40)
        self.input_shape = (int(clip_duration_ms / window_step_ms), mel_bins)

        # FAH calculation - prefer evaluation config, fall back to training config.
        # Only overwrite if evaluation block explicitly provides a non-zero value.
        self.ambient_duration_hours = training.get("ambient_duration_hours", 10.0)
        if self.ambient_duration_hours > 0:
            self.logger.log_info(f"FAH calculation enabled with {self.ambient_duration_hours:.2f} hours of ambient audio")
        evaluation = config.get("evaluation", {})
        eval_ambient = evaluation.get("ambient_duration_hours")
        if eval_ambient is not None and eval_ambient > 0:
            self.ambient_duration_hours = eval_ambient
            self.logger.log_info(f"FAH calculation overridden by evaluation config: {self.ambient_duration_hours:.2f} hours of ambient audio")

        # Scale ambient duration for validation split so FAH is comparable to test evaluation.
        # Without scaling, validation FAH divides by the FULL ambient hours but only sees
        # val_split fraction of the negative data, understating FAH by ~1/val_split.
        self.val_split = float(training.get("val_split", 0.1))
        self.val_ambient_duration_hours = self.ambient_duration_hours * self.val_split
        if self.val_ambient_duration_hours > 0:
            self.logger.log_info(f"Validation FAH uses scaled duration: {self.val_ambient_duration_hours:.2f}h (full={self.ambient_duration_hours:.2f}h × val_split={self.val_split})")

        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = paths.get("checkpoint_dir", "./checkpoints")

        # Training state
        self.current_step = 0
        self.best_fah = float("inf")
        self.best_recall = 0.0
        self.best_quality_score = float("-inf")
        self.best_weights_path: str | None = None
        self._last_assigned_lr: float | None = None  # Guard redundant LR assigns

        # SpecAugment warning flag to prevent log flooding
        self._spec_augment_warning_shown = False

        # Metrics trackers
        self.evaluation_config = evaluation
        default_threshold = float(self.evaluation_config.get("default_threshold", 0.5) or 0.5)
        self.eval_target_fah = float(self.evaluation_config.get("target_fah", self.target_minimization) or self.target_minimization)
        self.eval_target_recall = float(self.evaluation_config.get("target_recall", 0.95) or 0.95)
        self.eval_gain_window_steps = int(self.evaluation_config.get("gain_window_steps", 1000) or 1000)
        self.eval_plateau_window_evals = int(self.evaluation_config.get("plateau_window_evals", 5) or 5)
        self.eval_plateau_min_delta = float(self.evaluation_config.get("plateau_min_delta", 0.001) or 0.001)
        self.eval_plateau_slope_eps = float(self.evaluation_config.get("plateau_slope_eps", 0.0001) or 0.0001)
        self._eval_history: list[tuple[int, dict[str, float]]] = []
        self.train_metrics = TrainingMetrics(
            cutoffs=self._get_cutoffs(),
            ambient_duration_hours=self.ambient_duration_hours,
            default_threshold=default_threshold,
        )
        self.val_metrics = TrainingMetrics(
            cutoffs=self._get_cutoffs(),
            ambient_duration_hours=self.val_ambient_duration_hours,
            default_threshold=default_threshold,
        )
        self._last_val_raw_labels: list[int] = []
        self._last_materialized_accuracy: float = 0.0  # Keep last accuracy for display between materialization intervals

        # TensorBoard metric selection (keep focused, high-signal metrics)
        self._tb_train_metric_keys = ["loss", "precision", "recall", "auc", "step_time_ms", "data_loading_ms", "spec_augment_ms", "train_step_ms"]
        self._tb_val_metric_keys = [
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "auc_pr",
            "ambient_false_positives_per_hour",
            "recall_at_no_faph",
            "threshold_for_no_faph",
            "average_viable_recall",
            "quality_score",
            "recall_at_target_fah",
            "threshold_for_target_fah",
            "fah_at_target_recall",
            "threshold_for_target_recall",
            "quality_plateau_score",
            "quality_plateau_slope",
            "quality_plateau_gap",
            "gain_f1_score_per_1k_steps",
            "gain_average_viable_recall_per_1k_steps",
            "gain_recall_at_no_faph_per_1k_steps",
            "gain_auc_pr_per_1k_steps",
            "gain_recall_at_target_fah_per_1k_steps",
            "gain_fah_at_target_recall_per_1k_steps",
        ]
        self._prev_quality_metrics: dict[str, float] | None = None
        self._prev_quality_step: int | None = None

        # Advanced TensorBoard logger configuration
        self.tensorboard_log_histograms = bool(performance.get("tensorboard_log_histograms", True))
        self.tensorboard_log_images = bool(performance.get("tensorboard_log_images", True))
        self.tensorboard_log_pr_curves = bool(performance.get("tensorboard_log_pr_curves", True))
        self.tensorboard_log_graph = bool(performance.get("tensorboard_log_graph", True))
        self.tensorboard_log_advanced_scalars = bool(performance.get("tensorboard_log_advanced_scalars", True))
        self.tensorboard_log_weight_histograms = bool(performance.get("tensorboard_log_weight_histograms", False))
        self.tensorboard_image_interval = int(performance.get("tensorboard_image_interval", 5000) or 5000)
        self.tensorboard_histogram_interval = int(performance.get("tensorboard_histogram_interval", 5000) or 5000)

        # Hard negative mining
        hn_config = config.get("mining", {})
        self.hard_negative_mining_enabled = hn_config.get("enabled", False)
        self.hn_config = hn_config  # Store config for collection mode access
        self.hard_negative_miner: HardExampleMiner | None = None
        self._async_miner: AsyncHardExampleMiner | None = None  # Async miner instance
        self.false_predictions_log: list[dict] = []  # In-memory log for current epoch
        if self.hard_negative_mining_enabled:
            collection_mode = hn_config.get("collection_mode", "log_only")
            if self.async_mining and collection_mode == "mine_immediately":
                # Use async miner for non-blocking background mining
                self._async_miner = AsyncHardExampleMiner(
                    fp_threshold=hn_config.get("fp_threshold", 0.8),
                    max_samples=hn_config.get("max_samples", 5000),
                    mining_interval_epochs=hn_config.get("mining_interval_epochs", 5),
                    output_dir=paths.get("hard_negative_dir", "./dataset/hard_negative"),
                )
                self.logger.log_info(
                    f"Async hard negative mining enabled: threshold={hn_config.get('fp_threshold', 0.8)}, max_samples={hn_config.get('max_samples', 5000)}, collection_mode={collection_mode}"
                )
            else:
                # Use synchronous miner
                self.hard_negative_miner = HardExampleMiner(
                    fp_threshold=hn_config.get("fp_threshold", 0.8),
                    max_samples=hn_config.get("max_samples", 5000),
                    mining_interval_epochs=hn_config.get("mining_interval_epochs", 5),
                    output_dir=paths.get("hard_negative_dir", "./dataset/hard_negative"),
                )
                self.logger.log_info(
                    f"Hard negative mining enabled: threshold={hn_config.get('fp_threshold', 0.8)}, max_samples={hn_config.get('max_samples', 5000)}, collection_mode={collection_mode}"
                )
        # Check for EMA configuration
        training_cfg = config.get("training", {})
        ema_decay = training_cfg.get("ema_decay")
        if ema_decay is not None:
            self.logger.log_info(f"EMA configured with decay={ema_decay} (will be applied during optimizer creation)")

        # Pre-compute phase boundaries for fast lookup
        self._phase_boundaries: list[int] = []
        cumulative = 0
        for steps in self.training_steps_list:
            cumulative += steps
            self._phase_boundaries.append(cumulative)
        self._cached_phase: int = -1
        self._cached_phase_settings: dict[str, Any] = {}

    def _get_cutoffs(self) -> list[float]:
        """Generate cutoff thresholds for evaluation metrics."""
        n_thresholds = int(self.evaluation_config.get("n_thresholds", 101) or 101)
        if n_thresholds < 2:
            n_thresholds = 2
        return np.linspace(0.0, 1.0, n_thresholds).tolist()

    def _init_tensorboard_logger(self, log_dir: Path) -> None:
        if not self.tensorboard_enabled:
            return
        self.tensorboard_logger = TensorBoardLogger(
            log_dir=str(log_dir),
            enabled=True,
            log_histograms=self.tensorboard_log_histograms,
            log_images=self.tensorboard_log_images,
            log_pr_curves=self.tensorboard_log_pr_curves,
            log_graph=self.tensorboard_log_graph,
            log_advanced_scalars=self.tensorboard_log_advanced_scalars,
            image_interval=self.tensorboard_image_interval,
            histogram_interval=self.tensorboard_histogram_interval,
            log_weight_histograms=self.tensorboard_log_weight_histograms,
        )

    def _log_advanced_tensorboard_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.tensorboard_logger is None:
            return
        if not self.val_metrics.all_y_true:
            return

        y_true = np.array(self.val_metrics.all_y_true)
        y_score = np.array(self.val_metrics.all_y_scores)
        raw_labels = None
        if self._last_val_raw_labels:
            raw_labels = np.array(self._last_val_raw_labels)

        hist_due = self.tensorboard_histogram_interval > 0 and step % self.tensorboard_histogram_interval == 0
        image_due = self.tensorboard_image_interval > 0 and step % self.tensorboard_image_interval == 0

        if hist_due:
            self.tensorboard_logger.log_score_histograms(y_true, y_score, step, raw_labels)

        tp, fp, tn, fn = self.val_metrics.get_counts_at_threshold(self.val_metrics.default_threshold)
        if image_due:
            self.tensorboard_logger.log_score_distribution_image(y_true, y_score, step, raw_labels)
            self.tensorboard_logger.log_confusion_matrix_image(tp, fp, tn, fn, step, threshold=self.val_metrics.default_threshold)
            self.tensorboard_logger.log_roc_pr_curves(y_true, y_score, step)
            self.tensorboard_logger.log_pr_curve_interactive(y_true, y_score, step)
            self.tensorboard_logger.log_calibration_curve(y_true, y_score, step)

        if image_due and self.val_ambient_duration_hours > 0:
            self.tensorboard_logger.log_fah_recall_curve(y_true, y_score, self.val_ambient_duration_hours, step)

        self.tensorboard_logger.log_advanced_scalars(metrics, step)

        if hist_due and self.tensorboard_log_weight_histograms and self.model is not None:
            self.tensorboard_logger.log_weight_histograms(self.model, step)

        self.tensorboard_logger.flush()

    def _get_current_phase_settings(self, step: int) -> dict[str, Any]:
        """Get training settings for current step (cached per phase)."""
        # Fast phase lookup via pre-computed boundaries
        current_phase = 0
        for i, boundary in enumerate(self._phase_boundaries):
            if step < boundary:
                current_phase = i
                break
        else:
            current_phase = len(self.training_steps_list) - 1

        if current_phase == self._cached_phase:
            return self._cached_phase_settings

        self._cached_phase = current_phase
        self._cached_phase_settings = {
            "phase": current_phase,
            "learning_rate": self.learning_rates[current_phase],
            "positive_weight": self.positive_weights[current_phase],
            "negative_weight": self.negative_weights[current_phase],
            "hard_negative_weight": self.hard_negative_weights[current_phase],
        }
        return self._cached_phase_settings

    def _log_tensorboard_metrics(
        self,
        prefix: str,
        metrics: dict[str, float],
        step: int,
        keys: list[str],
    ) -> None:
        if self.tensorboard_writer is None:
            return
        with self.tensorboard_writer.as_default():
            for name in keys:
                if name not in metrics:
                    continue
                value = metrics[name]
                if isinstance(value, (int, float, np.floating, np.integer)):
                    tf.summary.scalar(f"{prefix}/{name}", float(value), step=step)

    def _augment_quality_metrics(self, metrics: dict[str, float], step: int) -> dict[str, float]:
        """Add derived quality metrics and gains per 1k steps."""
        quality_metrics = dict(metrics)
        quality_score, _, _, _ = self._compute_checkpoint_quality_score(metrics, self.eval_target_fah)
        if isinstance(quality_score, (int, float, np.floating, np.integer)):
            quality_metrics["quality_score"] = float(quality_score)

        if self._prev_quality_metrics is not None and self._prev_quality_step is not None:
            step_delta = step - self._prev_quality_step
            if step_delta > 0:
                scale = float(self.eval_gain_window_steps) / float(step_delta)
                gain_sources = [
                    "f1_score",
                    "average_viable_recall",
                    "recall_at_no_faph",
                    "auc_pr",
                    "recall_at_target_fah",
                    "fah_at_target_recall",
                ]
                for name in gain_sources:
                    current = metrics.get(name)
                    previous = self._prev_quality_metrics.get(name)
                    if current is None or previous is None:
                        continue
                    if isinstance(current, (int, float, np.floating, np.integer)) and isinstance(
                        previous,
                        (int, float, np.floating, np.integer),
                    ):
                        gain_name = f"gain_{name}_per_1k_steps"
                        quality_metrics[gain_name] = (float(current) - float(previous)) * scale

        quality_metrics.update(self._compute_plateau_metrics(step, quality_metrics))

        self._prev_quality_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating, np.integer))}
        self._prev_quality_step = step
        return quality_metrics

    def _compute_plateau_metrics(self, step: int, metrics: dict[str, float]) -> dict[str, float]:
        """Compute plateau indicators over recent evals."""
        plateau_metrics: dict[str, float] = {}
        quality_score = metrics.get("quality_score")
        if not isinstance(quality_score, (int, float, np.floating, np.integer)):
            return plateau_metrics

        self._eval_history.append((step, {"quality_score": float(quality_score)}))
        max_len = 5
        if len(self._eval_history) > max_len:
            self._eval_history = self._eval_history[-max_len:]

        scores = [entry[1]["quality_score"] for entry in self._eval_history]
        if len(scores) < 2:
            return plateau_metrics

        score_min = float(min(scores))
        score_max = float(max(scores))
        plateau_gap = score_max - score_min
        slope = 0.0
        if len(scores) >= 2:
            x_vals = np.arange(len(scores), dtype=float)
            y_vals = np.array(scores, dtype=float)
            x_mean = float(np.mean(x_vals))
            y_mean = float(np.mean(y_vals))
            denom = float(np.sum((x_vals - x_mean) ** 2))
            if denom > 0:
                slope = float(np.sum((x_vals - x_mean) * (y_vals - y_mean)) / denom)

        plateau_score = 1.0 if (plateau_gap < self.eval_plateau_min_delta or abs(slope) < self.eval_plateau_slope_eps) else 0.0
        plateau_metrics["quality_plateau_score"] = float(plateau_score)
        plateau_metrics["quality_plateau_slope"] = float(slope)
        plateau_metrics["quality_plateau_gap"] = float(plateau_gap)
        return plateau_metrics

    def _compute_checkpoint_quality_score(self, metrics: dict[str, float], target_fah: float) -> tuple[float, float, float, float]:
        """Compute composite quality score used by both logging and checkpointing.

        Returns:
            Tuple of (quality_score, current_recall, fah, fah_penalty)
        """
        if "ambient_false_positives_per_hour" in metrics:
            fah = float(metrics.get("ambient_false_positives_per_hour", float("inf")))
        else:
            fp = metrics.get("fp", 0)
            if isinstance(fp, (int, float, np.floating, np.integer)) and self.val_ambient_duration_hours > 0:
                fah = float(fp) / max(self.val_ambient_duration_hours, 0.001)
            else:
                fah = float("inf")

        operating_recall_raw = metrics.get("operating_recall")
        if not isinstance(operating_recall_raw, (int, float, np.floating, np.integer)):
            operating_recall_raw = metrics.get("recall_at_target_fah")
        if not isinstance(operating_recall_raw, (int, float, np.floating, np.integer)):
            operating_recall_raw = metrics.get("recall")
        operating_recall = float(operating_recall_raw) if isinstance(operating_recall_raw, (int, float, np.floating, np.integer)) else 0.0

        avr_raw = metrics.get("average_viable_recall")
        avr = float(avr_raw) if isinstance(avr_raw, (int, float, np.floating, np.integer)) else operating_recall

        # FAH-aware recall quality: blend operating-point recall and average viable recall.
        # Keeps checkpointing robust while rewarding broad-threshold quality (AVR).
        recall_quality = (0.7 * operating_recall) + (0.3 * avr)

        # Smooth decay: always positive, never zero
        # At FAH=0 → 1.0, at FAH=target → 0.5, at FAH=10×target → 0.01
        fah_penalty = 1.0 / (1.0 + (fah / max(float(target_fah), 0.01)) ** 2)

        quality_score = recall_quality * fah_penalty
        return quality_score, operating_recall, fah, fah_penalty

    def _build_model(self, input_shape: tuple[int, ...]) -> tf.keras.Model:
        """Build the model architecture.

        Args:
            input_shape: Input feature shape (timesteps, mel_bins)

        Returns:
            Compiled Keras model
        """
        # Extract model config params to forward to build_model
        model_cfg = self.config.get("model", {})

        # Build model using architecture module with all config params
        model = build_model(
            input_shape=input_shape,
            num_classes=2,
            first_conv_filters=model_cfg.get("first_conv_filters", 32),
            first_conv_kernel_size=model_cfg.get("first_conv_kernel_size", 5),
            stride=model_cfg.get("stride", 3),
            pointwise_filters=model_cfg.get("pointwise_filters", "64,64,64,64"),
            mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
            repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
            residual_connection=model_cfg.get("residual_connection", "0,0,0,0"),
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            l2_regularization=model_cfg.get("l2_regularization", 0.0),
        )

        # Get current phase settings
        phase_settings = self._get_current_phase_settings(0)

        # Compile model with BinaryCrossentropy loss
        training = self.config.get("training", {})
        optimizer_name = str(training.get("optimizer", "adam")).lower()
        if optimizer_name != "adam":
            self.logger.log_warning(f"Unsupported optimizer '{optimizer_name}', falling back to Adam")
        optimizer = keras.optimizers.Adam(
            learning_rate=phase_settings["learning_rate"],
            clipnorm=training.get("gradient_clipnorm"),
        )

        # Apply EMA if configured - use native optimizer EMA support
        ema_decay = training.get("ema_decay")
        if ema_decay is not None:
            optimizer_kwargs = {
                "use_ema": True,
                "ema_momentum": float(ema_decay),
                "ema_overwrite_frequency": 1,
            }
            self.logger.log_info(f"EMA enabled with decay={ema_decay}")
            optimizer = keras.optimizers.Adam(
                learning_rate=phase_settings["learning_rate"],
                clipnorm=training.get("gradient_clipnorm"),
                **optimizer_kwargs,
            )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(
                from_logits=False,
                label_smoothing=float(training.get("label_smoothing", 0.0) or 0.0),
            ),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.TruePositives(name="tp"),
                keras.metrics.FalsePositives(name="fp"),
                keras.metrics.TrueNegatives(name="tn"),
                keras.metrics.FalseNegatives(name="fn"),
                keras.metrics.AUC(name="auc", curve="ROC"),
            ],
        )

        return model

    def _apply_class_weights(
        self,
        y_true: np.ndarray,
        sample_weights: np.ndarray,
        positive_weight: float,
        negative_weight: float,
        hard_negative_weight: float,
        is_hard_negative: np.ndarray | None = None,
    ) -> tf.Tensor:
        """Apply class weights to sample weights (GPU-accelerated)."""
        y_true_t = tf.cast(tf.constant(y_true), tf.float32)
        sw_t = tf.cast(tf.constant(sample_weights), tf.float32)
        if is_hard_negative is not None:
            hn_t = tf.cast(tf.constant(is_hard_negative), tf.bool)
            class_weights = tf.where(
                y_true_t == 1,
                positive_weight,
                tf.where(hn_t, hard_negative_weight, negative_weight),
            )
        else:
            class_weights = tf.where(y_true_t == 1, positive_weight, negative_weight)
        return sw_t * class_weights

    def _is_best_model(self, metrics: dict[str, float], target_fah: float, target_recall: float) -> tuple[bool, str]:
        """Determine if current model is best using composite quality score.

        Composite score = recall_quality × fah_penalty
        - recall_quality: 0.7×operating_recall + 0.3×average_viable_recall (with fallbacks)
        - fah_penalty: 1 / (1 + (fah/target_fah)²) — smooth Lorentzian decay, always positive

        When FAH = 0 → penalty = 1.0 (perfect)
        When FAH = target → penalty = 0.5
        When FAH = 10×target → penalty ≈ 0.01

        Args:
            metrics: Computed validation metrics
            target_fah: Target false accepts per hour
            target_recall: Current best recall (unused, kept for signature compat)

        Returns:
            Tuple of (is_best, reason)
        """
        quality_score, current_recall, fah, fah_penalty = self._compute_checkpoint_quality_score(metrics, target_fah)
        avr_raw = metrics.get("average_viable_recall")
        avr = float(avr_raw) if isinstance(avr_raw, (int, float, np.floating, np.integer)) else current_recall
        recall_quality = (0.7 * current_recall) + (0.3 * avr)

        if quality_score > self.best_quality_score:
            reason = (
                f"Quality score improved: {quality_score:.4f} > {self.best_quality_score:.4f} "
                f"(operating_recall={current_recall:.4f}, avr={avr:.4f}, recall_quality={recall_quality:.4f}, "
                f"FAH={fah:.2f}, fah_penalty={fah_penalty:.4f})"
            )
            return True, reason

        return (
            False,
            (
                f"No improvement: quality={quality_score:.4f} (best={self.best_quality_score:.4f}), "
                f"operating_recall={current_recall:.4f}, avr={avr:.4f}, recall_quality={recall_quality:.4f}, FAH={fah:.2f}"
            ),
        )

    def _save_checkpoint(self, metrics: dict[str, float], is_best: bool, reason: str) -> None:
        """Save model checkpoint.

        Args:
            metrics: Current validation metrics
            is_best: Whether this is the best model
            reason: Reason for saving/not saving
        """
        assert self.model is not None, "_save_checkpoint called before model was built"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if is_best:
            # Save best weights
            self.best_weights_path = os.path.join(self.checkpoint_dir, "best_weights.weights.h5")
            self.model.save_weights(self.best_weights_path)
            self.logger.log_checkpoint(reason, True, self.best_weights_path)

            # Update best metrics
            quality_score, current_recall, fah, _ = self._compute_checkpoint_quality_score(metrics, self.eval_target_fah)
            self.best_fah = float(fah)
            self.best_recall = float(current_recall)
            self.best_quality_score = float(quality_score)

        # Save periodic checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.current_step}.weights.h5")
        self.model.save_weights(checkpoint_path)

    def _write_run_metadata(self) -> None:
        """Write run_metadata.json to checkpoint dir for reproducibility tracking."""
        import hashlib
        import json as _json
        import platform

        # Git commit hash (best-effort)
        git_commit = "unknown"
        try:
            import subprocess

            result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception as exc:
            self.logger.log_warning(f"Could not resolve git commit: {exc}")

        # Stable config hash (ignore run-time-only fields)
        try:
            config_str = _json.dumps(self.config, sort_keys=True, default=str)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        except Exception:
            config_hash = "unknown"

        metadata = {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "random_seed": self.random_seed,
            "python_version": platform.python_version(),
            "total_steps": sum(self.training_steps_list),
            "batch_size": self.batch_size,
            "learning_rates": self.learning_rates,
            "training_steps": self.training_steps_list,
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(self.checkpoint_dir, "run_metadata.json")
        try:
            with open(meta_path, "w") as f:
                _json.dump(metadata, f, indent=2)
            self.logger.log_info(f"Run metadata written to {meta_path}")
        except Exception as exc:
            self.logger.log_warning(f"Could not write run metadata: {exc}")

    def train_step(
        self,
        train_fingerprints: np.ndarray,
        train_ground_truth: np.ndarray,
        train_sample_weights: np.ndarray,
        is_hard_negative: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Execute single training step.

        Materialization policy:
        - Keep training step TF-only; no .numpy() calls in hot path.
        - Scalars may be materialized only at logging intervals.
        - Validation owns numpy conversion for metrics.

        Args:
            train_fingerprints: Input features
            train_ground_truth: Ground truth labels
            train_sample_weights: Sample weights from data processor
            is_hard_negative: Optional boolean array indicating hard negative samples

        Returns:
            Dictionary of training metrics
        """
        assert self.model is not None, "train_step called before model was built"
        # Get current phase settings
        phase_settings = self._get_current_phase_settings(self.current_step)

        # Update learning rate only when it changes (avoid per-step GPU sync)
        current_lr = phase_settings["learning_rate"]
        optimizer = self.model.optimizer
        if optimizer is None:
            raise RuntimeError("Model optimizer is not set")
        if current_lr != self._last_assigned_lr:
            optimizer.learning_rate.assign(current_lr)
            self._last_assigned_lr = current_lr

        # Apply class weights (GPU-accelerated) - works for both tfdata and non-tfdata paths
        combined_weights: Any = self._apply_class_weights(
            y_true=train_ground_truth,
            sample_weights=train_sample_weights,
            positive_weight=phase_settings["positive_weight"],
            negative_weight=phase_settings["negative_weight"],
            hard_negative_weight=phase_settings["hard_negative_weight"],
            is_hard_negative=is_hard_negative,
        )

        result = self.model.train_on_batch(
            train_fingerprints,
            tf.reshape(train_ground_truth, [-1, 1]),
            sample_weight=combined_weights,
            return_dict=True,
        )

        # Build metrics dict
        metrics_dict: dict[str, float] = {}
        if isinstance(result, dict):
            metrics_dict = result
        else:
            # Fallback path for environments not supporting return_dict
            metric_names = getattr(self.model, "metrics_names", [])
            if not isinstance(result, (list, tuple)):
                result = [result]
            for i, value in enumerate(result):
                name = metric_names[i] if i < len(metric_names) else f"metric_{i}"
                metrics_dict[name] = value

        return metrics_dict

    def validate(self, data_generator) -> dict[str, float]:
        """Validate model on validation set.

        Args:
            data_generator: Factory callable (invoked to yield batches) or generator.
                When a callable is provided it is called to get a fresh iterator;
                this avoids exhausting a one-shot generator.
                Must yield 3-tuples: (fingerprints, ground_truth, metadata), where
                fingerprints is a NumPy/Tensor array of model inputs, ground_truth
                is the label array, and metadata can be any auxiliary batch info.

        Returns:
            Dictionary of validation metrics
        """
        assert self.model is not None, "validate called before model was built"
        # Accept both a callable factory and a plain generator
        self.val_metrics.reset()

        iterator = data_generator() if callable(data_generator) else data_generator
        if not isinstance(iterator, Iterable):
            raise ValueError("Trainer.validate() expected an iterable or generator from data_generator")

        batch_count = 0
        score_samples: list[float] = []
        score_sample_limit = 2000
        all_scores: list[tf.Tensor] = []
        all_labels: list[tf.Tensor] = []
        self._last_val_paths: list[str] = self._val_file_paths or []  # Use pre-loaded paths if available
        self._last_val_raw_labels: list[int] = []
        for batch in iterator:
            if not isinstance(batch, tuple) or len(batch) != 3:
                raise ValueError(f"Trainer.validate() expects data_generator to yield (fingerprints, ground_truth, metadata) 3-tuples. Got: {type(batch).__name__} with value {batch!r}")
            fingerprints, ground_truth, metadata = batch
            # Get predictions (TF-only; materialize once at end)
            predictions = self.model(fingerprints, training=False)

            # Handle both binary [N] and multi-class [N, C] output shapes.
            # For multi-class (softmax), column 1 is the positive-class probability.
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                scores = predictions[:, 1]
            else:
                scores = tf.reshape(predictions, [-1])

            all_scores.append(scores)
            all_labels.append(tf.cast(ground_truth, tf.int32))
            # File paths are now pre-loaded via val_file_paths (populated from dataset.get_split_file_paths)
            # Legacy metadata-based path collection removed — was always broken (metadata was sample_weights numpy array)
            if isinstance(metadata, dict) and "raw_labels" in metadata:
                raw_labels = metadata["raw_labels"]
                if isinstance(raw_labels, list):
                    self._last_val_raw_labels.extend(raw_labels)
                else:
                    self._last_val_raw_labels.extend(np.asarray(raw_labels).ravel().tolist())
            else:
                self._last_val_raw_labels.extend(np.asarray(ground_truth).ravel().tolist())
            batch_count += 1

        if all_scores:
            scores_all = tf.concat(all_scores, axis=0)
            labels_all = tf.concat(all_labels, axis=0)
            scores_np = scores_all.numpy()
            labels_np = labels_all.numpy()
            # Clear lists to free memory before metrics computation
            all_scores.clear()
            all_labels.clear()
            del scores_all, labels_all
            self.val_metrics.update(labels_np, scores_np)

            if score_sample_limit > 0:
                sample_count = min(score_sample_limit, scores_np.shape[0])
                if sample_count < scores_np.shape[0]:
                    rng = np.random.default_rng(seed=42)
                    indices = rng.choice(scores_np.shape[0], size=sample_count, replace=False)
                    score_samples = scores_np[indices].tolist()
                else:
                    score_samples = scores_np.tolist()

        metrics = self.val_metrics.compute_metrics()
        if self.val_metrics.all_y_true:
            y_true = np.array(self.val_metrics.all_y_true)
            y_score = np.array(self.val_metrics.all_y_scores)
            pos_count = int(np.sum(y_true == 1))
            neg_count = int(np.sum(y_true == 0))
            total_count = int(y_true.shape[0])
            metrics["val_positive_count"] = float(pos_count)
            metrics["val_negative_count"] = float(neg_count)
            metrics["val_total_count"] = float(total_count)
            metrics["score_mean"] = float(np.mean(y_score))
            metrics["score_std"] = float(np.std(y_score))
            metrics["score_median"] = float(np.median(y_score))
            try:
                metrics["brier_score"] = compute_brier_score(y_true, y_score)
            except ValueError as exc:
                self.logger.log_warning(f"Brier score computation skipped: {exc}")
            if len(np.unique(y_true)) >= 2:
                calc = MetricsCalculator(
                    y_true=y_true,
                    y_score=y_score,
                    ambient_duration_hours=self.val_ambient_duration_hours,
                )
                curves = calc.compute_roc_pr_curves(n_thresholds=len(self.val_metrics.cutoffs))
                fpr = curves["fpr"]
                tpr = curves["tpr"]
                thresholds = curves["thresholds"]
                if fpr.size > 0:
                    fnr = 1.0 - tpr
                    idx = int(np.argmin(np.abs(fpr - fnr)))
                    metrics["eer"] = float(fpr[idx])
                    metrics["eer_threshold"] = float(thresholds[idx])
        if score_samples:
            scores_arr = np.array(score_samples, dtype=np.float32)
            metrics["score_min"] = float(np.min(scores_arr))
            metrics["score_p05"] = float(np.percentile(scores_arr, 5))
            metrics["score_p50"] = float(np.percentile(scores_arr, 50))
            metrics["score_p95"] = float(np.percentile(scores_arr, 95))
            metrics["score_max"] = float(np.max(scores_arr))
            metrics["score_sample_count"] = float(scores_arr.shape[0])

        if self.val_ambient_duration_hours > 0 and self.eval_target_fah > 0:
            calc = MetricsCalculator(
                y_true=np.array(self.val_metrics.all_y_true),
                y_score=np.array(self.val_metrics.all_y_scores),
                ambient_duration_hours=self.val_ambient_duration_hours,
            )
            recall_at_fah, threshold_at_fah, _ = calc.compute_recall_at_target_fah(
                ambient_duration_hours=self.val_ambient_duration_hours,
                target_fah=self.eval_target_fah,
                n_thresholds=len(self.val_metrics.cutoffs),
            )
            metrics["recall_at_target_fah"] = float(recall_at_fah)
            metrics["threshold_for_target_fah"] = float(threshold_at_fah)
            metrics["operating_threshold"] = float(threshold_at_fah)
            metrics["operating_recall"] = float(recall_at_fah)
            metrics["operating_target_fah"] = float(self.eval_target_fah)

            fah_at_recall, threshold_at_recall, _ = calc.compute_fah_at_target_recall(
                ambient_duration_hours=self.val_ambient_duration_hours,
                target_recall=self.eval_target_recall,
                n_thresholds=len(self.val_metrics.cutoffs),
            )
            metrics["fah_at_target_recall"] = float(fah_at_recall)
            metrics["threshold_for_target_recall"] = float(threshold_at_recall)

        return metrics

    def _log_false_predictions_to_json(self, epoch: int) -> None:
        """Log false positive predictions to JSON file for post-training mining.

        Delegates to the unified mining module function.

        Args:
            epoch: Current training epoch
        """
        if not self.hard_negative_mining_enabled:
            return

        hn_config = self.hn_config
        if not hn_config.get("log_predictions", True):
            return

        # Access validation data from metrics accumulator
        if not hasattr(self.val_metrics, "all_y_true") or not hasattr(self.val_metrics, "all_y_scores"):
            self.logger.log_warning("Cannot log false predictions: validation metrics not available")
            return

        y_true = np.array(self.val_metrics.all_y_true)
        y_scores = np.array(self.val_metrics.all_y_scores)
        val_paths = self._last_val_paths if hasattr(self, "_last_val_paths") else None

        _ = log_false_predictions_to_json(
            epoch=epoch,
            y_true=y_true,
            y_scores=y_scores,
            fp_threshold=hn_config.get("fp_threshold", 0.8),
            top_k=hn_config.get("top_k_per_epoch", 100),
            log_file=hn_config.get("log_file", "logs/false_predictions.json"),
            val_paths=val_paths,
            best_weights_path=self.best_weights_path,
            logger=self.logger,
        )
        self.false_predictions_log.clear()

    def train(
        self,
        train_data_factory,
        val_data_factory,
        mining_data_factory=None,
        test_data_factory=None,
        input_shape: tuple[int, ...] | None = None,
        weights_path: str | None = None,
        val_file_paths: list[str] | None = None,
    ) -> tf.keras.Model:
        """Main training loop.

        Args:
            train_data_factory: Factory for training dataset
            val_data_factory: Factory for validation dataset
            mining_data_factory: Optional factory for mining dataset
            test_data_factory: Optional factory for test dataset
            weights_path: Optional path to model weights to load after building model (for fine-tuning)
            val_file_paths: Optional ordered list of validation file paths for FP tracking
        """
        self._val_file_paths = val_file_paths
        input_shape = self.input_shape if input_shape is None else input_shape

        self.logger.log_info("Building model...")
        self.model = self._build_model(input_shape)
        # Build model weights before summary so params are visible
        _ = self.model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)

        # Load pre-trained weights if provided (for fine-tuning)
        if weights_path is not None:
            self.logger.log_info(f"Loading weights from: {weights_path}")
            self.model.load_weights(weights_path)

        self.model.summary(print_fn=self.logger.console.print)

        # Setup cProfile profiler
        if self.enable_profiling:
            self.profiler = TrainingProfiler(self.profile_output_dir)

        # Setup TF Profiler (GPU kernel/memory traces for TensorBoard)
        if self.tf_profile_start_step > 0 and self.tensorboard_enabled:
            self.tf_profiler = TFProfiler(log_dir=self.tensorboard_log_dir)

        # Write run metadata for reproducibility tracking
        self._write_run_metadata()

        # Calculate total training steps
        total_steps = sum(self.training_steps_list)
        self.logger.log_header(self.config, total_steps)
        # Create progress bar
        progress, progress_task = self.logger.create_progress(total_steps)
        progress.start()
        prev_phase = -1

        # Training loop
        start_time = time.time()
        last_throughput_time = start_time
        last_throughput_step = 0
        last_throughput_samples = 0
        # Create initial generator from factory so we can restart it if exhausted
        train_data_generator = train_data_factory()
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        if self.tensorboard_enabled:
            log_dir = Path(self.tensorboard_log_dir) / run_name
            log_dir.mkdir(parents=True, exist_ok=True)
            self._init_tensorboard_logger(log_dir)
            if self.tensorboard_logger is not None and self.tensorboard_logger.writer is not None:
                self.tensorboard_writer = self.tensorboard_logger.writer
            else:
                self.tensorboard_writer = tf.summary.create_file_writer(str(log_dir))
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.log_model_graph(self.model, input_shape, step=0)

        try:
            for step in range(1, total_steps + 1):
                self.current_step = step
                step_start = time.perf_counter()
                _t0 = step_start

                # Get training batch
                _t_data_end = step_start  # fallback; overwritten on success
                try:
                    train_fingerprints, train_ground_truth, train_sample_weights, train_is_hard_neg = next(train_data_generator)
                    _t_data_end = time.perf_counter()
                except StopIteration:
                    # Restart by calling the factory again
                    train_data_generator = train_data_factory()
                    try:
                        train_fingerprints, train_ground_truth, train_sample_weights, train_is_hard_neg = next(train_data_generator)
                        _t_data_end = time.perf_counter()
                    except StopIteration as exc:
                        raise RuntimeError("train_data_factory() returned an empty generator after restart. Cannot continue training without batches.") from exc
                _t_aug_start = time.perf_counter()
                # Apply SpecAugment if enabled (GPU-accelerated, requires numpy)
                # Skip if using TF-native SpecAugment in pipeline
                if self.spec_augment_enabled and self.spec_augment_backend != "tf":
                    phase_settings = self._get_current_phase_settings(step)
                    current_phase = phase_settings["phase"]
                    if self.time_mask_count[current_phase] > 0 or self.freq_mask_count[current_phase] > 0:
                        try:
                            # Convert to numpy for CuPy SpecAugment if needed
                            if hasattr(train_fingerprints, "numpy"):
                                train_fingerprints = train_fingerprints.numpy()
                            train_fingerprints = batch_spec_augment_gpu(
                                train_fingerprints,
                                time_mask_max_size=self.time_mask_max_size[current_phase],
                                time_mask_count=self.time_mask_count[current_phase],
                                freq_mask_max_size=self.freq_mask_max_size[current_phase],
                                freq_mask_count=self.freq_mask_count[current_phase],
                            )
                        except RuntimeError as e:
                            # GPU not available or CuPy not installed, skip SpecAugment
                            if not self._spec_augment_warning_shown:
                                self.logger.log_warning(f"SpecAugment skipped: {e}")
                                self._spec_augment_warning_shown = True
                _t_aug_end = time.perf_counter()
                # Profile train_step with cProfile when enabled
                _t_train_start = time.perf_counter()
                if self.profiler and step % self.profile_every_n == 0:
                    with self.profiler.profile_section(f"full_step_{step}"):
                        train_metrics = self.train_step(train_fingerprints, train_ground_truth, train_sample_weights, train_is_hard_neg)
                else:
                    train_metrics = self.train_step(train_fingerprints, train_ground_truth, train_sample_weights, train_is_hard_neg)
                _t_train_end = time.perf_counter()
                materialize_due = step % self.materialize_metrics_interval == 0
                if not materialize_due:
                    loss_value = float(train_metrics.get("loss", 0.0))
                    # Keep last materialized accuracy for display (don't show 0 when not materialized)
                    train_metrics = {"loss": loss_value, "accuracy": self._last_materialized_accuracy}
                else:
                    # Update cached accuracy when materialized
                    self._last_materialized_accuracy = float(train_metrics.get("accuracy", 0.0))
                train_metrics["step_time_ms"] = (time.perf_counter() - step_start) * 1000
                train_metrics["data_loading_ms"] = (_t_data_end - _t0) * 1000
                train_metrics["spec_augment_ms"] = (_t_aug_end - _t_aug_start) * 1000
                train_metrics["train_step_ms"] = (_t_train_end - _t_train_start) * 1000

                if self.log_throughput and step % self.log_throughput_interval == 0:
                    now = time.time()
                    elapsed = now - last_throughput_time
                    if elapsed > 0:
                        samples_seen = step * self.batch_size
                        delta_samples = samples_seen - last_throughput_samples
                        delta_steps = step - last_throughput_step
                        steps_per_sec = delta_steps / elapsed if delta_steps > 0 else 0.0
                        samples_per_sec = delta_samples / elapsed if delta_samples > 0 else 0.0
                        self.logger.log_info(f"Throughput: {steps_per_sec:.2f} steps/s, {samples_per_sec:.1f} samples/s over last {delta_steps} steps")
                    last_throughput_time = now
                    last_throughput_step = step
                    last_throughput_samples = step * self.batch_size

                # Update progress bar and detect phase transitions
                phase_settings_display = self._get_current_phase_settings(step)
                approx_epoch = step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
                phase_settings_display["epoch"] = approx_epoch
                if self.eval_log_every_step or step % self.eval_basic_step_interval == 0:
                    self.logger.update_step(progress, progress_task, step, train_metrics, phase_settings_display)

                if self.tensorboard_writer is not None and step % self.eval_basic_step_interval == 0:
                    self._log_tensorboard_metrics("train", train_metrics, step, self._tb_train_metric_keys)
                    with self.tensorboard_writer.as_default():
                        tf.summary.scalar("train/learning_rate", float(phase_settings_display["learning_rate"]), step=step)

                # Log GPU memory + utilization to TensorBoard
                if self.tensorboard_writer is not None and self.gpu_memory_log_interval > 0 and step % self.gpu_memory_log_interval == 0 and self.tf_profiler is not None:
                    mem = self.tf_profiler.get_gpu_memory_info()
                    sys_info = get_system_info()
                    gpu_info = sys_info.get("gpu", {}) if isinstance(sys_info, dict) else {}
                    gpu_load = float(gpu_info.get("load_percent", 0.0)) if isinstance(gpu_info, dict) else 0.0
                    with self.tensorboard_writer.as_default():
                        tf.summary.scalar("gpu/peak_memory_mb", mem["peak_mb"], step=step)
                        tf.summary.scalar("gpu/current_memory_mb", mem["current_mb"], step=step)
                        tf.summary.scalar("gpu/utilization_percent", gpu_load, step=step)

                # Start TF Profiler trace (fires once, captures a short window)
                if self.tf_profiler is not None and step == self.tf_profile_start_step:
                    self.tf_profiler.start_trace(step=step)
                elif self.tf_profiler is not None and self.tf_profiler.is_tracing() and step == self.tf_profile_start_step + self.tf_profiler.warmup_steps + self.tf_profiler.active_steps:
                    self.tf_profiler.stop_trace()

                # Detect and announce phase transitions
                current_phase = phase_settings_display["phase"]
                if current_phase != prev_phase:
                    if prev_phase >= 0:  # Skip announcement for initial phase
                        progress.stop()
                        self.logger.log_phase_transition(
                            current_phase,
                            len(self.training_steps_list),
                            phase_settings_display["learning_rate"],
                            phase_settings_display["positive_weight"],
                            phase_settings_display["negative_weight"],
                        )
                        progress.start()
                    prev_phase = current_phase

                # Evaluate every N steps (regardless of phase transition)
                basic_due = step % self.eval_basic_step_interval == 0
                advanced_due = step % self.eval_advanced_step_interval == 0
                if basic_due or advanced_due:
                    val_metrics = self.validate(val_data_factory)
                    val_metrics = self._augment_quality_metrics(val_metrics, step)

                    if basic_due:
                        self.logger.log_validation_results(val_metrics, step, total_steps)

                    if advanced_due:
                        if self.eval_confusion_matrix_interval and step % self.eval_confusion_matrix_interval == 0:
                            tp, fp, tn, fn = self.val_metrics.get_counts_at_threshold(self.val_metrics.default_threshold)
                            self.logger.log_confusion_matrix(tp, fp, tn, fn, threshold=self.val_metrics.default_threshold)
                            self.logger.log_per_class_analysis(tp, fp, tn, fn, threshold=self.val_metrics.default_threshold)

                    if self.tensorboard_writer is not None:
                        self._log_tensorboard_metrics("val", val_metrics, step, self._tb_val_metric_keys)
                        self.tensorboard_writer.flush()
                    if advanced_due and self.tensorboard_logger is not None:
                        self._log_advanced_tensorboard_metrics(val_metrics, step)

                    if advanced_due and self.eval_checkpoints_interval and step % self.eval_checkpoints_interval == 0:
                        is_best, reason = self._is_best_model(val_metrics, self.eval_target_fah, self.best_recall)
                        self._save_checkpoint(val_metrics, is_best, reason)
                        if self.tensorboard_logger is not None:
                            summary_text = "\n".join(
                                [
                                    f"Checkpoint at step {step}",
                                    f"Best FAH: {self.best_fah:.4f}",
                                    f"Best Recall: {self.best_recall:.4f}",
                                    f"Best Quality Score: {self.best_quality_score:.4f}",
                                    f"Best weights path: {self.best_weights_path or 'N/A'}",
                                    f"Reason: {reason}",
                                ]
                            )
                            self.tensorboard_logger.log_text_summary("checkpoints/summary", summary_text, step)

                    # Hard negative mining or logging (based on collection_mode)
                    if advanced_due and step % self.eval_advanced_step_interval == 0:
                        approx_epoch = step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
                        collection_mode = self.hn_config.get("collection_mode", "log_only")
                        mining_interval = self.hn_config.get("mining_interval_epochs", 5)
                        min_epochs_before_mining = int(self.hn_config.get("min_epochs_before_mining", 5) or 5)

                        if approx_epoch >= min_epochs_before_mining and approx_epoch % mining_interval == 0 and approx_epoch != getattr(self, "_last_mined_epoch", -1):
                            if collection_mode == "log_only":
                                # Log false predictions to JSON for post-training mining
                                self._log_false_predictions_to_json(approx_epoch)
                                self._last_mined_epoch = approx_epoch
                            elif collection_mode == "mine_immediately":
                                if self.async_mining and self._async_miner is not None:
                                    # Async mining path
                                    if not self._async_miner.is_mining():
                                        # Start async mining
                                        self.logger.log_mining(f"Starting async mining at epoch {approx_epoch}...")
                                        mining_source_factory = mining_data_factory if mining_data_factory is not None else val_data_factory
                                        self._async_miner.start_mining(self.model, mining_source_factory, approx_epoch)
                                    else:
                                        # Poll for results
                                        mining_result = self._async_miner.get_result()
                                        if mining_result is not None:
                                            self.logger.log_mining(
                                                f"Completed async mining at epoch {approx_epoch}",
                                                count=mining_result["num_hard_negatives"],
                                            )
                                            self._last_mined_epoch = approx_epoch
                                elif self.hard_negative_miner is not None:
                                    # Synchronous mining path
                                    self.logger.log_mining(f"Mining at epoch {approx_epoch}...")
                                    try:
                                        mining_source_factory = mining_data_factory if mining_data_factory is not None else val_data_factory
                                        mining_result = self.hard_negative_miner.mine_from_dataset(self.model, mining_source_factory, approx_epoch)
                                        self.logger.log_mining(
                                            f"Completed at epoch {approx_epoch}",
                                            count=mining_result["num_hard_negatives"],
                                        )
                                        self._last_mined_epoch = approx_epoch
                                    except OSError as e:
                                        self.logger.log_warning(f"Hard negative mining failed (IOError): {e}")
                                    except RuntimeError as e:
                                        self.logger.log_warning(f"Hard negative mining failed (RuntimeError): {e}")
                                    except Exception:
                                        raise
                        # (duplicate mining block removed)

                    # Resume progress bar
                    progress.start()
        finally:
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.close()
                self.tensorboard_logger = None
                self.tensorboard_writer = None
            elif self.tensorboard_writer is not None:
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
                self.tensorboard_writer = None
            # Wait for async miner to complete if running
            if self._async_miner is not None and self._async_miner.is_mining():
                self.logger.log_info("Waiting for async hard negative mining to complete...")
                self._async_miner.wait_for_completion()
                self.logger.log_info("Async hard negative mining completed")

        # Training complete
        total_time = time.time() - start_time
        progress.stop()

        # Always save final weights as fallback for auto-tuner and post-training tools
        final_path = os.path.join(self.checkpoint_dir, "final_weights.weights.h5")
        self.model.save_weights(final_path)
        self.logger.log_info(f"Final weights saved to {final_path}")

        self.logger.log_completion(
            total_time,
            self.best_weights_path or "N/A",
            self.best_fah,
            self.best_recall,
        )

        # Load best weights for return
        if self.best_weights_path and os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)

        if test_data_factory is not None:
            from src.evaluation.test_evaluator import TestEvaluator

            log_dir = self.config.get("performance", {}).get("tensorboard_log_dir", "./logs")
            test_feature_store_path = os.path.join(self.config.get("paths", {}).get("processed_dir", "./data/processed"), "test")
            evaluator = TestEvaluator(self.model, self.config, str(log_dir))
            evaluator.evaluate(test_data_factory, test_feature_store_path=test_feature_store_path)
            self.logger.log_info("Running held-out test evaluation...")
            test_metrics = self.validate(test_data_factory)
            self.logger.log_validation_results(test_metrics, total_steps, total_steps)
            tp, fp, tn, fn = self.val_metrics.get_counts_at_threshold(self.val_metrics.default_threshold)
            self.logger.log_confusion_matrix(tp, fp, tn, fn, threshold=self.val_metrics.default_threshold)
            self.logger.log_per_class_analysis(tp, fp, tn, fn, threshold=self.val_metrics.default_threshold)

        return self.model


def train(config: dict) -> tf.keras.Model:
    """Train wake word model.

    Main entry point for training. Sets up the data generators,
    initializes the Trainer, and runs the training loop.

    Args:
        config: Training configuration from config loader

    Returns:
        Trained model
    """
    from src.data.dataset import WakeWordDataset

    # Calculate input shape from hardware config (not from model.spectrogram_length)
    hardware_cfg = config.get("hardware", {})
    clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
    window_step_ms = hardware_cfg.get("window_step_ms", 10)
    mel_bins = hardware_cfg.get("mel_bins", 40)
    input_shape = (int(clip_duration_ms / window_step_ms), mel_bins)
    max_time_frames = input_shape[0]

    # Input shape already calculated above from hardware config
    # No need to check for spectrogram_length in model_cfg - it's deprecated

    dataset = WakeWordDataset(config)
    try:
        dataset.build()

        trainer = Trainer(config)

        # Validate dataset class distribution
        _LABEL_NAMES = {0: "negative", 1: "positive", 2: "hard_negative"}
        for split_name in ["train", "val", "test"]:
            dist = dataset.get_label_distribution(split_name)
            if not dist:
                if split_name == "train":
                    raise RuntimeError(f"No samples found in {split_name} split!")
                continue
            dist_str = ", ".join(f"{_LABEL_NAMES.get(k, f'label_{k}')}={v}" for k, v in sorted(dist.items()))
            trainer.logger.log_info(f"Dataset {split_name}: {dist_str} (total={sum(dist.values())})")
            pos_count = dist.get(1, 0)
            neg_count = dist.get(0, 0) + dist.get(2, 0)
            if pos_count == 0 and split_name == "train":
                raise RuntimeError("No positive samples in training set!")
            if pos_count > 0 and neg_count / pos_count < 5.0 and split_name == "train":
                trainer.logger.log_warning(f"Low negative:positive ratio ({neg_count / pos_count:.1f}:1) in {split_name} — recommend at least 5:1")
        performance = config.get("performance", {})
        use_tfdata = performance.get("use_tfdata", True)
        if performance.get("benchmark_pipeline", False):
            from src.data.tfdata_pipeline import benchmark_pipeline

            trainer.logger.log_info("Benchmarking data pipeline...")
            benchmark_pipeline(dataset, config, n_batches=100)

        if use_tfdata:
            from src.data.tfdata_pipeline import OptimizedDataPipeline

            # Build SpecAugment config to pass to pipeline
            training_cfg = config.get("training", {})
            spec_augment_config = None
            if trainer.spec_augment_backend == "tf":
                spec_augment_config = {
                    "enabled": trainer.spec_augment_enabled,
                    "backend": "tf",
                    "time_mask_max_size": training_cfg.get("time_mask_max_size", [0, 0]),
                    "time_mask_count": training_cfg.get("time_mask_count", [0, 0]),
                    "freq_mask_max_size": training_cfg.get("freq_mask_max_size", [0, 0]),
                    "freq_mask_count": training_cfg.get("freq_mask_count", [0, 0]),
                    "seed": training_cfg.get("split_seed", 42),
                }

            pipeline = OptimizedDataPipeline(dataset, config, max_time_frames=max_time_frames, spec_augment_config=spec_augment_config)
            shuffle_seed = int(config.get("training", {}).get("split_seed", 42))
            train_ds = pipeline.create_training_pipeline(shuffle_seed=shuffle_seed)
            val_ds = pipeline.create_validation_pipeline()
            test_ds = pipeline.create_test_pipeline()

            # Pipeline already yields 4-tuples (features, labels, sample_weights, is_hard_neg)
            # Just need to cast labels to int32
            def train_factory():
                for features, labels, sample_weights, is_hard_neg in train_ds:
                    labels_int = tf.cast(labels, tf.int32)
                    yield features, labels_int, sample_weights, is_hard_neg

            def val_factory():
                for features, labels, _, _ in val_ds:
                    labels_int = tf.cast(labels, tf.int32)
                    yield features, labels_int, None

            def test_factory():
                for features, labels, _, _ in test_ds:
                    labels_int = tf.cast(labels, tf.int32)
                    metadata = {"raw_labels": labels_int.numpy().tolist()}
                    yield features, labels_int, metadata

            model = trainer.train(
                train_data_factory=train_factory,
                val_data_factory=val_factory,
                mining_data_factory=dataset.train_mining_generator_factory(max_time_frames=max_time_frames),
                test_data_factory=test_factory,
                input_shape=input_shape,
                val_file_paths=dataset.get_split_file_paths("val"),
            )
        else:
            model = trainer.train(
                train_data_factory=dataset.train_generator_factory(max_time_frames=max_time_frames),
                val_data_factory=dataset.val_generator_factory(max_time_frames=max_time_frames),
                mining_data_factory=dataset.train_mining_generator_factory(max_time_frames=max_time_frames),
                test_data_factory=dataset.test_generator_factory(max_time_frames=max_time_frames),
                input_shape=input_shape,
                val_file_paths=dataset.get_split_file_paths("val"),
            )
        # Auto-tune post-training if configured and final FAH > target
        at_config = config.get("auto_tuning", {})
        auto_tune_enabled = at_config.get("enabled", False)
        # Fallback to legacy flag
        if not auto_tune_enabled:
            auto_tune_enabled = config.get("training", {}).get("auto_tune_on_poor_fah", False)
        target_min = config.get("training", {}).get("target_minimization", 0.5)

        # Track if auto-tuning was run to skip top FP extraction (config doesn't propagate back)
        auto_tune_was_run = False

        def _resolve_checkpoint(trainer):
            """Resolve best available checkpoint with fallback chain."""
            import glob as _glob

            candidates = [
                trainer.best_weights_path,
                os.path.join(trainer.checkpoint_dir, "best_weights.weights.h5"),
                os.path.join(trainer.checkpoint_dir, "final_weights.weights.h5"),
            ]
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    return candidate
            # Last resort: latest periodic checkpoint
            periodic = sorted(_glob.glob(os.path.join(trainer.checkpoint_dir, "checkpoint_step_*.weights.h5")))
            return periodic[-1] if periodic else None

        if auto_tune_enabled and trainer.best_fah > target_min:
            auto_tune_was_run = True
            trainer.logger.log_info(f"auto_tuning.enabled: final FAH={trainer.best_fah:.3f} > target={target_min:.3f} — launching auto-tuner")
            checkpoint = _resolve_checkpoint(trainer)
            if checkpoint is None:
                trainer.logger.log_warning("No checkpoint found for auto-tuner — skipping auto-tuning")
            else:
                import subprocess

                config_arg = config.get("training", {}).get("_config_preset", "standard")
                cmd = [
                    sys.executable,
                    "-m",
                    "src.tuning.cli",
                    "--checkpoint",
                    checkpoint,
                    "--config",
                    config_arg,
                ]
                # Pass target overrides from auto_tuning config
                if "target_fah" in at_config:
                    cmd.extend(["--target-fah", str(at_config["target_fah"])])
                if "target_recall" in at_config:
                    cmd.extend(["--target-recall", str(at_config["target_recall"])])
                try:
                    subprocess.run(cmd, check=True)  # noqa: S603
                except subprocess.CalledProcessError as e:
                    trainer.logger.log_warning(f"Auto-tune finished with non-zero exit ({e.returncode}) — see above for details")
                except Exception as e:
                    trainer.logger.log_warning(f"Auto-tune failed: {e}")
        mining_config = config.get("mining", {})
        if mining_config.get("extract_top_fps", False) and mining_config.get("run_extraction_at_training_end", True) and not auto_tune_was_run:
            try:
                trainer.logger.log_info("Running top FP extraction from hard negatives...")
                checkpoint = _resolve_checkpoint(trainer)
                if checkpoint is None:
                    trainer.logger.log_warning("No checkpoint found for top FP extraction — skipping extraction")
                else:
                    result = run_top_fp_extraction(config, checkpoint_path=checkpoint)
                    trainer.logger.log_info(
                        f"Top FP extraction: {result.get('top_fp_count', 0)} files logged "
                        f"(out of {result.get('total_false_positives', 0)} FPs). "
                        f"Run 'mww-mine-hard-negatives extract-top-fps' to move them."
                    )
            except Exception as e:
                trainer.logger.log_warning(f"Top FP extraction failed: {e}")
    finally:
        dataset.close()
    return model


def main():
    """Main entry point for mww-train command."""
    # Env vars for TF log suppression are set at module level (top of file).

    import argparse

    from config.loader import load_full_config

    parser = argparse.ArgumentParser(description="Train wake word model")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name or path to config file",
    )
    parser.add_argument("--override", type=str, default=None, help="Override config file path")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log filename (default: auto-generated timestamp)",
    )
    args = parser.parse_args()

    # Start terminal logging with default directory first (update after loading config)
    terminal_logger = TerminalLogger(
        log_dir="./logs",
        log_filename=args.log_file,
    )

    try:
        with terminal_logger:
            # Load config
            config = load_full_config(args.config, args.override)

            # Convert dataclass to dict since train() expects a dict
            import dataclasses

            config_dict = dataclasses.asdict(config)

            # Store actual config preset name so auto-tuner subprocess can use it
            config_dict.setdefault("training", {})["_config_preset"] = args.config
            perf_dict = config_dict.get("performance", {})
            log_dir = perf_dict.get("tensorboard_log_dir", "./logs")
            terminal_logger.log_dir = Path(log_dir)
            terminal_logger.log_dir.mkdir(parents=True, exist_ok=True)

            # Train model
            train(config_dict)

            # Final message
            logger = RichTrainingLogger()
            logger.log_info("Training completed successfully!")

    except KeyboardInterrupt:
        print("\n[TerminalLogger] Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n[TerminalLogger] Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
