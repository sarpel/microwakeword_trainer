"""Trainer module for wake word model training.

Step-based training loop with:
- Class weighting for imbalanced data
- Two-priority checkpoint selection
- Evaluation at regular intervals
- Mixed precision training support
- Integration with TrainingProfiler
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model.architecture import build_model
from src.training.profiler import TrainingProfiler
from src.training.miner import HardExampleMiner
from src.data.spec_augment_gpu import batch_spec_augment_gpu
from src.evaluation.metrics import MetricsCalculator
from src.training.rich_logger import RichTrainingLogger


class EvaluationMetrics:
    """Wrapper around MetricsCalculator that supports batch accumulation.

    This provides backward compatibility with TrainingMetrics' update pattern
    while using the evaluation package for computations.
    """

    def __init__(
        self,
        cutoffs: Optional[list] = None,
        ambient_duration_hours: float = 0.0,
    ):
        """Initialize metrics tracker.

        Args:
            cutoffs: Array of threshold values (default: 101 points from 0 to 1)
        """
        if cutoffs is None:
            cutoffs = np.linspace(0.0, 1.0, 101).tolist()
        elif hasattr(cutoffs, "tolist"):
            cutoffs = cutoffs.tolist()
        self.cutoffs = cutoffs
        self.ambient_duration_hours = ambient_duration_hours

        # Accumulated predictions and labels
        self.all_y_true: list = []
        self.all_y_scores: list = []

        # Per-threshold metrics
        self.tp_at_threshold: Dict[float, int] = {c: 0 for c in self.cutoffs}
        self.fp_at_threshold: Dict[float, int] = {c: 0 for c in self.cutoffs}
        self.tn_at_threshold: Dict[float, int] = {c: 0 for c in self.cutoffs}
        self.fn_at_threshold: Dict[float, int] = {c: 0 for c in self.cutoffs}

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

        # Update per-threshold metrics
        for cutoff in self.cutoffs:
            y_pred = (y_scores_flat >= cutoff).astype(np.int32)

            tp = int(np.sum((y_pred == 1) & (y_true_flat == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true_flat == 0)))
            tn = int(np.sum((y_pred == 0) & (y_true_flat == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true_flat == 1)))

            self.tp_at_threshold[cutoff] += tp
            self.fp_at_threshold[cutoff] += fp
            self.tn_at_threshold[cutoff] += tn
            self.fn_at_threshold[cutoff] += fn

    def compute_metrics(self) -> Dict[str, float]:
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

        # Get basic metrics using default threshold 0.5
        metrics = calc.compute_all_metrics(
            ambient_duration_hours=self.ambient_duration_hours,
            threshold=0.5,
        )

        # Add per-threshold metrics for backward compatibility (ROC/PR curves)
        for cutoff in self.cutoffs:
            tp = self.tp_at_threshold[cutoff]
            fp = self.fp_at_threshold[cutoff]
            tn = self.tn_at_threshold[cutoff]
            fn = self.fn_at_threshold[cutoff]

            total = tp + fp + tn + fn
            if total > 0:
                acc = (tp + tn) / total
            else:
                acc = 0

            prec_denom = tp + fp
            rec_denom = tp + fn

            metrics[f"precision_{cutoff:.2f}"] = (
                tp / prec_denom if prec_denom > 0 else 0
            )
            metrics[f"recall_{cutoff:.2f}"] = tp / rec_denom if rec_denom > 0 else 0
            metrics[f"accuracy_{cutoff:.2f}"] = acc

        return metrics

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.all_y_true = []
        self.all_y_scores = []
        self.tp_at_threshold = {c: 0 for c in self.cutoffs}
        self.fp_at_threshold = {c: 0 for c in self.cutoffs}
        self.tn_at_threshold = {c: 0 for c in self.cutoffs}
        self.fn_at_threshold = {c: 0 for c in self.cutoffs}


# Backward-compatible alias
TrainingMetrics = EvaluationMetrics


class Trainer:
    """Training orchestrator with step-based training loop."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.

        Args:
            config: Training configuration from config loader
        """
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.profiler: Optional[TrainingProfiler] = None

        # Extract training config
        training = config.get("training", {})
        self.training_steps_list = training.get("training_steps", [20000, 10000])
        self.learning_rates = training.get("learning_rates", [0.001, 0.0001])
        self.batch_size = training.get("batch_size", 128)
        self.eval_step_interval = training.get("eval_step_interval", 500)
        self.steps_per_epoch = training.get(
            "steps_per_epoch", 1000
        )  # For mining epoch calculation

        # Class weights (positive=1.0, negative=20.0, hard_negative=40.0 typical for wake word)
        self.positive_weights = training.get("positive_class_weight", [1.0, 1.0])
        self.negative_weights = training.get("negative_class_weight", [20.0, 20.0])
        self.hard_negative_weights = training.get(
            "hard_negative_class_weight", [40.0, 40.0]
        )

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
        self.time_mask_max_size = _pad_or_trim(
            training.get("time_mask_max_size", [0, 0]), 0
        )
        self.time_mask_count = _pad_or_trim(training.get("time_mask_count", [0, 0]), 0)
        self.freq_mask_max_size = _pad_or_trim(
            training.get("freq_mask_max_size", [0, 0]), 0
        )
        self.freq_mask_count = _pad_or_trim(training.get("freq_mask_count", [0, 0]), 0)
        self.spec_augment_enabled = any(
            self.time_mask_max_size
            + self.time_mask_count
            + self.freq_mask_max_size
            + self.freq_mask_count
        )

        # Checkpoint selection config
        self.minimization_metric = training.get(
            "minimization_metric", "ambient_false_positives_per_hour"
        )
        self.target_minimization = training.get("target_minimization", 0.5)
        self.maximization_metric = training.get(
            "maximization_metric", "average_viable_recall"
        )

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
        
        # Rich terminal logger
        self.logger = RichTrainingLogger()
        # FAH calculation - get ambient duration hours from training config
        self.ambient_duration_hours = training.get("ambient_duration_hours", 10.0)
        if self.ambient_duration_hours > 0:
            self.logger.log_info(
                f"FAH calculation enabled with {self.ambient_duration_hours:.2f} hours of ambient audio"
            )
        evaluation = config.get("evaluation", {})
        self.ambient_duration_hours = evaluation.get("ambient_duration_hours", 0.0)
        if self.ambient_duration_hours > 0:
            self.logger.log_info(
                f"FAH calculation enabled with {self.ambient_duration_hours:.2f} hours of ambient audio"
            )

        # Apply threading config
        if self.inter_op_parallelism > 0:
            tf.config.threading.set_inter_op_parallelism_threads(
                self.inter_op_parallelism
            )
        if self.intra_op_parallelism > 0:
            tf.config.threading.set_intra_op_parallelism_threads(
                self.intra_op_parallelism
            )

        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = paths.get("checkpoint_dir", "./checkpoints")

        # Training state
        self.current_step = 0
        self.best_fah = float("inf")
        self.best_recall = 0.0
        self.best_weights_path: Optional[str] = None

        # SpecAugment warning flag to prevent log flooding
        self._spec_augment_warning_shown = False

        # Metrics trackers
        self.train_metrics = TrainingMetrics(
            ambient_duration_hours=self.ambient_duration_hours
        )
        self.val_metrics = TrainingMetrics(
            ambient_duration_hours=self.ambient_duration_hours
        )

        # Hard negative mining
        hn_config = config.get("hard_negative_mining", {})
        self.hard_negative_mining_enabled = hn_config.get("enabled", False)
        self.hard_negative_miner: Optional[HardExampleMiner] = None
        if self.hard_negative_mining_enabled:
            self.hard_negative_miner = HardExampleMiner(
                fp_threshold=hn_config.get("fp_threshold", 0.8),
                max_samples=hn_config.get("max_samples", 5000),
                mining_interval_epochs=hn_config.get("mining_interval_epochs", 5),
                output_dir=paths.get("hard_negative_dir", "./dataset/hard_negative"),
            )
            self.logger.log_info(
                f"Hard negative mining enabled: threshold={hn_config.get('fp_threshold', 0.8)}, "
                f"max_samples={hn_config.get('max_samples', 5000)}"
            )
    def _get_current_phase_settings(self, step: int) -> Dict[str, Any]:
        """Get training settings for current step.

        Args:
            step: Current training step

        Returns:
            Dictionary with learning_rate and class weights for current phase
        """
        # Calculate cumulative steps per phase
        cumulative_steps = 0
        current_phase = 0

        for i, phase_steps in enumerate(self.training_steps_list):
            if step < cumulative_steps + phase_steps:
                current_phase = i
                break
            cumulative_steps += phase_steps
        else:
            # If beyond total steps, use last phase
            current_phase = len(self.training_steps_list) - 1

        return {
            "phase": current_phase,
            "learning_rate": self.learning_rates[current_phase],
            "positive_weight": self.positive_weights[current_phase],
            "negative_weight": self.negative_weights[current_phase],
            "hard_negative_weight": self.hard_negative_weights[current_phase],
        }

    def _build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
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
            first_conv_filters=model_cfg.get("first_conv_filters", 30),
            first_conv_kernel_size=model_cfg.get("first_conv_kernel_size", 5),
            stride=model_cfg.get("stride", 3),
            pointwise_filters=model_cfg.get("pointwise_filters", "60,60,60,60"),
            mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[9],[13],[21]"),
            repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
            residual_connection=model_cfg.get("residual_connection", "0,0,0,0"),
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            l2_regularization=model_cfg.get("l2_regularization", 0.0),
        )

        # Get current phase settings
        phase_settings = self._get_current_phase_settings(0)

        # Compile model with BinaryCrossentropy loss
        optimizer = keras.optimizers.Adam(learning_rate=phase_settings["learning_rate"])

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
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
        is_hard_negative: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply class weights to sample weights.

        Args:
            y_true: Ground truth labels (1=positive, 0=negative)
            sample_weights: Existing sample weights
            positive_weight: Weight for positive class
            negative_weight: Weight for regular negative samples
            hard_negative_weight: Weight for hard negative samples (false positives)
            is_hard_negative: Optional boolean array indicating hard negative samples

        Returns:
            Combined weights
        """
        if is_hard_negative is not None:
            # Three-class weighting: positive, hard_negative, regular_negative
            class_weights = np.where(
                y_true == 1,
                positive_weight,
                np.where(is_hard_negative, hard_negative_weight, negative_weight),
            )
        else:
            # Binary weighting: positive vs negative (hard negatives get negative_weight)
            class_weights = np.where(y_true == 1, positive_weight, negative_weight)
        return sample_weights * class_weights

    def _is_best_model(
        self, metrics: Dict[str, float], target_fah: float, target_recall: float
    ) -> Tuple[bool, str]:
        """Determine if current model is best based on two-priority checkpoint selection.

        Priority 1: Minimize FAH (false accepts per hour) below target
        Priority 2: Maximize recall

        Args:
            metrics: Computed validation metrics
            target_fah: Target false accepts per hour
            target_recall: Current best recall

        Returns:
            Tuple of (is_best, reason)
        """
        # Use proper FAH from evaluation package if available
        if "ambient_false_positives_per_hour" in metrics:
            fah = metrics.get("ambient_false_positives_per_hour", float("inf"))
        else:
            # Fallback: estimate FAH based on FP
            fp = metrics.get("fp", 0)
            fah = (
                fp / max(self.ambient_duration_hours, 0.001)
                if self.ambient_duration_hours > 0
                else float("inf")
            )

        current_recall = metrics.get("recall", 0)

        # Case 1: Achieved target FAH and improved recall
        if fah <= target_fah and current_recall > self.best_recall:
            return (
                True,
                f"Case 1: FAH={fah:.2f} <= target={target_fah}, recall improved to {current_recall:.4f}",
            )

        # Case 2: Haven't achieved target but decreased FAH
        if fah < self.best_fah:
            return True, f"Case 2: FAH decreased from {self.best_fah:.2f} to {fah:.2f}"

        # Case 3: Tied FAH and improved recall
        if fah == self.best_fah and current_recall > self.best_recall:
            return (
                True,
                f"Case 3: FAH tied at {fah:.2f}, recall improved to {current_recall:.4f}",
            )

        return (
            False,
            f"No improvement: FAH={fah:.2f} (best={self.best_fah:.2f}), recall={current_recall:.4f} (best={self.best_recall:.4f})",
        )

    def _save_checkpoint(
        self, metrics: Dict[str, float], is_best: bool, reason: str
    ) -> None:
        """Save model checkpoint.

        Args:
            metrics: Current validation metrics
            is_best: Whether this is the best model
            reason: Reason for saving/not saving
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if is_best:
            # Save best weights
            self.best_weights_path = os.path.join(
                self.checkpoint_dir, "best_weights.weights.h5"
            )
            self.model.save_weights(self.best_weights_path)
            self.logger.log_checkpoint(reason, True, self.best_weights_path)

            # Update best metrics
            if "ambient_false_positives_per_hour" in metrics:
                self.best_fah = metrics.get(
                    "ambient_false_positives_per_hour", float("inf")
                )
            else:
                fp = metrics.get("fp", 0)
                self.best_fah = (
                    fp / max(self.ambient_duration_hours, 0.001)
                    if self.ambient_duration_hours > 0
                    else float("inf")
                )
            self.best_recall = metrics.get("recall", 0)

        # Save periodic checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{self.current_step}.weights.h5"
        )
        self.model.save_weights(checkpoint_path)

    def train_step(
        self,
        train_fingerprints: np.ndarray,
        train_ground_truth: np.ndarray,
        train_sample_weights: np.ndarray,
        is_hard_negative: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Execute single training step.

        Args:
            train_fingerprints: Input features
            train_ground_truth: Ground truth labels
            train_sample_weights: Sample weights from data processor
            is_hard_negative: Optional boolean array indicating hard negative samples

        Returns:
            Dictionary of training metrics
        """
        # Get current phase settings
        phase_settings = self._get_current_phase_settings(self.current_step)

        # Update learning rate
        self.model.optimizer.learning_rate.assign(phase_settings["learning_rate"])

        # Apply class weights
        combined_weights = self._apply_class_weights(
            train_ground_truth,
            train_sample_weights,
            phase_settings["positive_weight"],
            phase_settings["negative_weight"],
            phase_settings["hard_negative_weight"],
            is_hard_negative,
        )

        # Train on batch
        result = self.model.train_on_batch(
            train_fingerprints,
            train_ground_truth,
            sample_weight=combined_weights,
            return_dict=True,
        )

        # Build metrics dict
        metrics_dict = {}
        if isinstance(result, dict):
            metrics_dict = result
        else:
            # Fallback path for environments not supporting return_dict
            metric_names = getattr(self.model, "metrics_names", [])
            for i, value in enumerate(result):
                name = metric_names[i] if i < len(metric_names) else f"metric_{i}"
                metrics_dict[name] = value

        return metrics_dict

    def validate(self, data_generator) -> Dict[str, float]:
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
        self.val_metrics.reset()

        # Accept both a callable factory and a plain generator
        iterator = data_generator() if callable(data_generator) else data_generator

        for batch in iterator:
            if not isinstance(batch, tuple) or len(batch) != 3:
                raise ValueError(
                    "Trainer.validate() expects data_generator to yield "
                    "(fingerprints, ground_truth, metadata) 3-tuples. "
                    f"Got: {type(batch).__name__} with value {batch!r}"
                )
            fingerprints, ground_truth, _ = batch
            # Get predictions
            predictions = self.model(fingerprints, training=False).numpy()

            # Handle both binary [N] and multi-class [N, C] output shapes.
            # For multi-class (softmax), column 1 is the positive-class probability.
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                scores = predictions[:, 1]
            else:
                scores = predictions.flatten()

            # Update metrics
            self.val_metrics.update(ground_truth, scores)

        return self.val_metrics.compute_metrics()

    def train(
        self,
        train_data_factory,
        val_data_factory,
        input_shape: Tuple[int, ...] = (49, 40),
    ) -> tf.keras.Model:
        """Execute full step-based training loop.

        Args:
            train_data_factory: Callable that returns a generator yielding
                (features, labels, weights) tuples.  Passed as a factory so the
                generator can be restarted when exhausted.
            val_data_factory: Same pattern for validation data.
            input_shape: Input feature shape

        Returns:
            Trained model
        """
        # Build and compile model
        self.logger.log_info("Building model...")
        self.model = self._build_model(input_shape)
        self.model.summary(print_fn=self.logger.console.print)

        # Setup profiler
        if self.enable_profiling:
            self.profiler = TrainingProfiler(self.profile_output_dir)

        # Calculate total training steps
        total_steps = sum(self.training_steps_list)
        self.logger.log_header(self.config, total_steps)
        # Create progress bar
        progress, progress_task = self.logger.create_progress(total_steps)
        progress.start()
        prev_phase = -1

        # Training loop
        # Training loop
        start_time = time.time()
        # Create initial generator from factory so we can restart it if exhausted
        train_data_generator = train_data_factory()

        for step in range(1, total_steps + 1):
            self.current_step = step

            # Get training batch
            try:
                train_fingerprints, train_ground_truth, train_sample_weights = next(
                    train_data_generator
                )
            except StopIteration:
                # Restart by calling the factory again
                train_data_generator = train_data_factory()
                try:
                    train_fingerprints, train_ground_truth, train_sample_weights = next(
                        train_data_generator
                    )
                except StopIteration as exc:
                    raise RuntimeError(
                        "train_data_factory() returned an empty generator after restart. "
                        "Cannot continue training without batches."
                    ) from exc

            # Apply SpecAugment if enabled (GPU-accelerated)
            if self.spec_augment_enabled:
                phase_settings = self._get_current_phase_settings(step)
                current_phase = phase_settings["phase"]
                if (
                    self.time_mask_count[current_phase] > 0
                    or self.freq_mask_count[current_phase] > 0
                ):
                    try:
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
            # Profile data loading if enabled
            if self.profiler and step % self.profile_every_n == 0:
                with self.profiler.profile_section(f"step_{step}"):
                    train_metrics = self.train_step(
                        train_fingerprints, train_ground_truth, train_sample_weights
                    )
            else:
                train_metrics = self.train_step(
                    train_fingerprints, train_ground_truth, train_sample_weights
                )

            # Update progress bar and detect phase transitions
            phase_settings_display = self._get_current_phase_settings(step)
            self.logger.update_step(progress, progress_task, step, train_metrics, phase_settings_display)

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

            # Evaluate every N steps
                val_metrics = self.validate(val_data_factory)

                # Log validation results with Rich table and confusion matrix
                self.logger.log_validation_results(val_metrics, step, total_steps)

                # Confusion matrix at threshold 0.5
                tp = self.val_metrics.tp_at_threshold.get(0.5, 0)
                fp = self.val_metrics.fp_at_threshold.get(0.5, 0)
                tn = self.val_metrics.tn_at_threshold.get(0.5, 0)
                fn = self.val_metrics.fn_at_threshold.get(0.5, 0)
                self.logger.log_confusion_matrix(tp, fp, tn, fn)

                # Check if best model
                is_best, reason = self._is_best_model(
                    val_metrics, self.target_minimization, self.best_recall
                )

                self._save_checkpoint(val_metrics, is_best, reason)

                # Hard negative mining (during evaluation)
                if self.hard_negative_miner and step % self.eval_step_interval == 0:
                    # Calculate approximate epoch from steps per epoch
                    approx_epoch = (
                        step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
                    )
                    if (
                        approx_epoch > 0
                        and approx_epoch
                        % self.hard_negative_miner.mining_interval_epochs
                        == 0
                        and approx_epoch != getattr(self, "_last_mined_epoch", -1)
                    ):
                        self.logger.log_mining(f"Mining at epoch {approx_epoch}...")
                        try:
                            mining_result = self.hard_negative_miner.mine_from_dataset(
                                self.model, val_data_factory, approx_epoch
                            )
                            self.logger.log_mining(
                                f"Completed at epoch {approx_epoch}",
                                count=mining_result['num_hard_negatives'],
                            )
                            self._last_mined_epoch = approx_epoch
                        except IOError as e:
                            self.logger.log_warning(f"Hard negative mining failed (IOError): {e}")
                        except RuntimeError as e:
                            self.logger.log_warning(f"Hard negative mining failed (RuntimeError): {e}")
                        except Exception:
                            raise

                # Resume progress bar
                progress.start()

        # Training complete
        progress.stop()
        self.logger.log_completion(
            total_time, self.best_weights_path or "N/A",
            self.best_fah, self.best_recall,
        )

        # Load best weights for return
        if self.best_weights_path and os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)

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

    model_cfg = config.get("model", {})
    if "spectrogram_length" not in model_cfg:
        raise ValueError(
            "Model config must define 'spectrogram_length' under config['model']."
        )

    # mel_bins: prefer hardware config (the canonical location)
    mel_bins = config.get("hardware", {}).get("mel_bins")
    if mel_bins is None:
        mel_bins = model_cfg.get("mel_bins", 40)

    input_shape = (
        model_cfg.get("spectrogram_length", 49),
        mel_bins,
    )

    dataset = WakeWordDataset(config)  # type: ignore[arg-type]
    dataset.build()  # type: ignore[attr-defined]

    trainer = Trainer(config)
    model = trainer.train(
        train_data_factory=dataset.train_generator_factory(),  # type: ignore[attr-defined]
        val_data_factory=dataset.val_generator_factory(),  # type: ignore[attr-defined]
        input_shape=input_shape,
    )
    return model


def main():
    """Main entry point for mww-train command."""
    import argparse
    from config.loader import load_full_config

    parser = argparse.ArgumentParser(description="Train wake word model")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name or path to config file",
    )
    parser.add_argument(
        "--override", type=str, default=None, help="Override config file path"
    )
    args = parser.parse_args()

    # Load config
    config = load_full_config(args.config, args.override)

    # Convert dataclass to dict since train() expects a dict
    import dataclasses
    config_dict = dataclasses.asdict(config)

    # Train model
    train(config_dict)

    # Final message
    logger = RichTrainingLogger()
    logger.log_info("Training completed successfully!")


if __name__ == "__main__":
    main()
