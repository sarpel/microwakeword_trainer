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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model.architecture import WakeWordModel, build_model
from src.training.profiler import TrainingProfiler


class TrainingMetrics:
    """Track training metrics at multiple thresholds for ROC/PR curves."""

    def __init__(self, cutoffs: Optional[np.ndarray] = None):
        """Initialize metrics tracker.

        Args:
            cutoffs: Array of threshold values (default: 101 points from 0 to 1)
        """
        if cutoffs is None:
            cutoffs = np.linspace(0.0, 1.0, 101)
        self.cutoffs = cutoffs.tolist()

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
        self.all_y_true.extend(y_true.tolist())
        self.all_y_scores.extend(y_scores.tolist())

        # Update per-threshold metrics
        for cutoff in self.cutoffs:
            y_pred = (y_scores >= cutoff).astype(np.int32)

            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

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

        # Basic metrics
        tp_total = np.sum((y_scores >= 0.5) & (y_true == 1))
        fp_total = np.sum((y_scores >= 0.5) & (y_true == 0))
        tn_total = np.sum((y_scores < 0.5) & (y_true == 0))
        fn_total = np.sum((y_scores < 0.5) & (y_true == 1))

        # Avoid division by zero
        precision_denom = tp_total + fp_total
        recall_denom = tp_total + fn_total

        metrics = {
            "accuracy": (tp_total + tn_total) / len(y_true) if len(y_true) > 0 else 0,
            "precision": tp_total / precision_denom if precision_denom > 0 else 0,
            "recall": tp_total / recall_denom if recall_denom > 0 else 0,
            "tp": int(tp_total),
            "fp": int(fp_total),
            "tn": int(tn_total),
            "fn": int(fn_total),
        }

        # AUC calculation
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            if len(np.unique(y_true)) > 1:
                metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
                metrics["auc_pr"] = average_precision_score(y_true, y_scores)
        except ImportError:
            pass

        # Compute metrics at all thresholds for ROC/PR curves
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

        # Class weights (positive=1.0, negative=20.0 typical for wake word)
        self.positive_weights = training.get("positive_class_weight", [1.0, 1.0])
        self.negative_weights = training.get("negative_class_weight", [20.0, 20.0])

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

        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = paths.get("checkpoint_dir", "./checkpoints")

        # Training state
        self.current_step = 0
        self.best_fah = float("inf")
        self.best_recall = 0.0
        self.best_weights_path: Optional[str] = None

        # Metrics trackers
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()

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
        }

    def _build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build the model architecture.

        Args:
            input_shape: Input feature shape (timesteps, mel_bins)

        Returns:
            Compiled Keras model
        """
        model_config = self.config.get("model", {})

        # Build model using architecture module
        model = build_model(input_shape=input_shape, num_classes=2)

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
    ) -> np.ndarray:
        """Apply class weights to sample weights.

        Args:
            y_true: Ground truth labels
            sample_weights: Existing sample weights
            positive_weight: Weight for positive class
            negative_weight: Weight for negative class

        Returns:
            Combined weights
        """
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
        # Calculate false accept rate (using FP at 0.5 threshold)
        fp = metrics.get("fp", 0)
        # Estimate FAH based on FP - this is a simplified version
        # In practice, you'd calculate based on hours of negative audio
        fah = fp / max(metrics.get("tn", 1), 1) * 1000  # Normalized FAH estimate

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
            print(f"[CHECKPOINT] Saved best weights: {reason}")

            # Update best metrics
            fp = metrics.get("fp", 0)
            self.best_fah = fp / max(metrics.get("tn", 1), 1) * 1000
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
    ) -> Dict[str, float]:
        """Execute single training step.

        Args:
            train_fingerprints: Input features
            train_ground_truth: Ground truth labels
            train_sample_weights: Sample weights from data processor

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
        )

        # Train on batch
        result = self.model.train_on_batch(
            train_fingerprints, train_ground_truth, sample_weight=combined_weights
        )

        # Build metrics dict
        metrics_dict = {}
        if isinstance(result, dict):
            metrics_dict = result
        else:
            # result is a list [loss, metrics...]
            metrics_dict["loss"] = result[0]
            for i, name in enumerate(
                ["accuracy", "recall", "precision", "tp", "fp", "tn", "fn", "auc"]
            ):
                if i + 1 < len(result):
                    metrics_dict[name] = result[i + 1]

        return metrics_dict

    def validate(self, data_generator) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            data_generator: Generator yielding (features, labels, weights) tuples

        Returns:
            Dictionary of validation metrics
        """
        self.val_metrics.reset()

        for fingerprints, ground_truth, _ in data_generator:
            # Get predictions
            predictions = self.model(fingerprints, training=False)
            scores = predictions.numpy().flatten()

            # Update metrics
            self.val_metrics.update(ground_truth, scores)

        return self.val_metrics.compute_metrics()

    def train(
        self,
        train_data_generator,
        val_data_generator,
        input_shape: Tuple[int, ...] = (49, 40),
    ) -> tf.keras.Model:
        """Execute full step-based training loop.

        Args:
            train_data_generator: Generator yielding (features, labels, weights) tuples
            val_data_generator: Generator yielding (features, labels, weights) tuples
            input_shape: Input feature shape

        Returns:
            Trained model
        """
        # Build and compile model
        print("[TRAINER] Building model...")
        self.model = self._build_model(input_shape)
        self.model.summary(print_fn=print)

        # Setup profiler
        if self.enable_profiling:
            self.profiler = TrainingProfiler(self.profile_output_dir)

        # Calculate total training steps
        total_steps = sum(self.training_steps_list)
        print(f"[TRAINER] Starting step-based training for {total_steps} steps...")
        print(f"[TRAINER] Training phases: {self.training_steps_list}")
        print(f"[TRAINER] Learning rates: {self.learning_rates}")
        print(
            f"[TRAINER] Class weights (positive/negative): {self.positive_weights}/{self.negative_weights}"
        )
        print(f"[TRAINER] Evaluation interval: {self.eval_step_interval} steps")

        # Training loop
        start_time = time.time()

        for step in range(1, total_steps + 1):
            self.current_step = step

            # Get training batch
            try:
                train_fingerprints, train_ground_truth, train_sample_weights = next(
                    train_data_generator
                )
            except StopIteration:
                # Restart generator if exhausted
                train_data_generator = train_data_generator()
                train_fingerprints, train_ground_truth, train_sample_weights = next(
                    train_data_generator
                )

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

            # Log training progress
            if step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta = (total_steps - step) / steps_per_sec

                lr = self.model.optimizer.learning_rate.numpy()
                print(
                    f"[STEP {step}/{total_steps}] "
                    f"loss={train_metrics.get('loss', 0):.4f} "
                    f"acc={train_metrics.get('accuracy', 0):.4f} "
                    f"lr={lr:.6f} "
                    f"({steps_per_sec:.1f} steps/s, ETA: {eta / 60:.1f} min)"
                )

            # Evaluate every N steps
            if step % self.eval_step_interval == 0:
                print(f"\n[EVALUATING at step {step}]")

                val_metrics = self.validate(val_data_generator)

                print(
                    f"Validation - "
                    f"loss: N/A (eval mode) "
                    f"acc: {val_metrics.get('accuracy', 0):.4f} "
                    f"recall: {val_metrics.get('recall', 0):.4f} "
                    f"precision: {val_metrics.get('precision', 0):.4f}"
                )

                # Check if best model
                is_best, reason = self._is_best_model(
                    val_metrics, self.target_minimization, self.best_recall
                )

                self._save_checkpoint(val_metrics, is_best, reason)
                print()

        # Training complete
        total_time = time.time() - start_time
        print(f"[TRAINER] Training complete in {total_time / 60:.1f} minutes")
        print(f"[TRAINER] Best weights saved to: {self.best_weights_path}")

        # Load best weights for return
        if self.best_weights_path and os.path.exists(self.best_weights_path):
            self.model.load_weights(self.best_weights_path)

        return self.model


def train(config: dict) -> tf.keras.Model:
    """Train wake word model.

    This is the main entry point for training. It sets up the data generators,
    initializes the trainer, and runs the training loop.

    Args:
        config: Training configuration from config loader

    Returns:
        Trained model
    """
    # This is a placeholder - actual implementation would:
    # 1. Create data generators from processed features
    # 2. Initialize Trainer with config
    # 3. Call trainer.train()

    # For now, return a simple stub
    print("[TRAIN] Training function called with config:")
    print(f"  Training steps: {config.get('training', {}).get('training_steps', [])}")
    print(f"  Learning rates: {config.get('training', {}).get('learning_rates', [])}")
    print(f"  Batch size: {config.get('training', {}).get('batch_size', 128)}")

    # Build a simple model for testing
    model = build_model(input_shape=(49, 40), num_classes=2)
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

    # Train model
    model = train(config)

    print("\n[TRAIN] Training completed successfully!")


if __name__ == "__main__":
    main()
