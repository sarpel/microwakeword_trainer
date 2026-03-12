"""Advanced TensorBoard logging module for wake word training.

This module provides sophisticated TensorBoard visualizations including:
- Histogram logging for score distributions and model statistics
- Image logging for confusion matrices and curves
- Interactive PR curves
- Advanced scalar metrics
- Model graph visualization
"""

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from ..evaluation.calibration import compute_calibration_curve
from ..evaluation.metrics import compute_roc_pr_curves

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Advanced TensorBoard logger with sophisticated visualization capabilities.

    Features:
    - Histogram logging for score distributions and weight statistics
    - Image logging for confusion matrices, ROC/PR curves, score distributions
    - Interactive PR curves via tf.summary.pr_curve (if available)
    - Advanced scalar metrics (EER, calibration, per-class breakdown)
    - Model graph visualization
    - Text summaries for operating points and confusion matrices
    """

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        log_histograms: bool = True,
        log_images: bool = True,
        log_pr_curves: bool = True,
        log_graph: bool = True,
        log_advanced_scalars: bool = True,
        image_interval: int = 5000,
        histogram_interval: int = 5000,
        log_weight_histograms: bool = False,
        log_learning_rate: bool = True,
        log_gradient_norms: bool = False,
        log_activation_stats: bool = False,
        log_confidence_drift: bool = True,
        log_per_class_accuracy: bool = True,
        sophisticated_interval: int = 2000,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether logging is enabled
            log_histograms: Enable histogram logging
            log_images: Enable image logging
            log_pr_curves: Enable PR curve logging
            log_graph: Enable model graph logging
            log_advanced_scalars: Enable advanced scalar metrics
            image_interval: Steps between image logging
            histogram_interval: Steps between histogram logging
            log_weight_histograms: Enable model weight histograms
            log_learning_rate: Track learning rate schedule
            log_gradient_norms: Log gradient norm histograms (expensive)
            log_activation_stats: Log per-layer activation statistics
            log_confidence_drift: Track prediction confidence over time
            log_per_class_accuracy: Log accuracy breakdown by class
            sophisticated_interval: Steps between sophisticated metrics
        """
        self.enabled = enabled
        self.log_histograms = log_histograms
        self.log_images = log_images
        self.log_pr_curves = log_pr_curves
        self.log_graph = log_graph
        self.log_advanced_scalars_enabled = log_advanced_scalars
        self.image_interval = image_interval
        self.histogram_interval = histogram_interval
        self.log_weight_histograms_enabled = log_weight_histograms

        # Sophisticated metrics (Phase 4)
        # Boolean flags use _enabled suffix to avoid conflict with method names
        self.log_learning_rate_enabled = log_learning_rate
        self.log_gradient_norms_enabled = log_gradient_norms
        self.log_activation_stats_enabled = log_activation_stats
        self.log_confidence_drift_enabled = log_confidence_drift
        self.log_per_class_accuracy_enabled = log_per_class_accuracy
        self.sophisticated_interval = sophisticated_interval

        self.writer: tf.summary.SummaryWriter | None = None
        self._log_dir = log_dir
        self._graph_logged = False

        # Confidence drift tracking
        self._confidence_history: list[tuple[int, float]] = []
        self._max_confidence_history = 100

        if enabled:
            self._setup_writer()

    def _setup_writer(self) -> None:
        """Create TensorBoard summary writer."""
        try:
            log_path = Path(self._log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = tf.summary.create_file_writer(str(log_path))
            logger.info(f"TensorBoard logger initialized: {log_path}")
        except Exception as e:
            logger.warning(f"Failed to create TensorBoard writer: {e}")
            self.enabled = False

    def log_model_graph(
        self,
        model: tf.keras.Model,
        input_shape: tuple[int, ...],
        step: int = 0,
    ) -> None:
        """Log model architecture graph to TensorBoard.

        Args:
            model: Keras model to visualize
            input_shape: Shape of input tensor (batch_size, ...)
            step: Global step
        """
        if not self.enabled or not self.log_graph or self._graph_logged or self.writer is None:
            return

        try:
            # Create dummy input for tracing
            dummy_input = tf.zeros((1, *input_shape), dtype=tf.float32)

            with self.writer.as_default():
                tf.summary.trace_on(graph=True, profiler=False)
                _ = model(dummy_input, training=False)
                tf.summary.trace_export(name="model_graph", step=step)
                tf.summary.trace_off()

            self._graph_logged = True
            logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")

    def log_score_histograms(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        step: int,
        raw_labels: np.ndarray | None = None,
    ) -> None:
        """Log score distribution histograms.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            step: Global step
            raw_labels: Optional raw labels for hard negative separation
        """
        if not self.enabled or not self.log_histograms or self.writer is None:
            return

        try:
            with self.writer.as_default():
                # Positive scores
                pos_scores = y_score[y_true == 1]
                if len(pos_scores) > 0:
                    tf.summary.histogram(
                        "scores/positive",
                        pos_scores,
                        step=step,
                        buckets=50,
                    )

                # Negative scores
                neg_scores = y_score[y_true == 0]
                if len(neg_scores) > 0:
                    tf.summary.histogram(
                        "scores/negative",
                        neg_scores,
                        step=step,
                        buckets=50,
                    )

                # Hard negative scores
                if raw_labels is not None:
                    hn_scores = y_score[raw_labels == 2]
                    if len(hn_scores) > 0:
                        tf.summary.histogram(
                            "scores/hard_negative",
                            hn_scores,
                            step=step,
                            buckets=50,
                        )

                # All scores
                tf.summary.histogram("scores/all", y_score, step=step, buckets=50)

        except Exception as e:
            logger.warning(f"Failed to log score histograms: {e}")

    def log_confusion_matrix_image(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        step: int,
        threshold: float = 0.5,
    ) -> None:
        """Log confusion matrix as an image.

        Args:
            tp: True positives
            fp: False positives
            tn: True negatives
            fn: False negatives
            step: Global step
            threshold: Classification threshold used
        """
        if not self.enabled or not self.log_images or self.writer is None:
            return

        fig = None
        buf = None
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(6, 5))

            # Create confusion matrix array
            cm = np.array([[tn, fp], [fn, tp]])
            total = cm.sum()

            # Plot
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
            ax.figure.colorbar(im, ax=ax)

            # Labels
            classes = ["Negative", "Positive"]
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=classes,
                yticklabels=classes,
                xlabel="Predicted",
                ylabel="Actual",
                title=f"Confusion Matrix (threshold={threshold:.2f})",
            )

            # Add text annotations
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    value = cm[i, j]
                    percentage = (value / total * 100) if total > 0 else 0
                    ax.text(
                        j,
                        i,
                        f"{value}\n({percentage:.1f}%)",
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10,
                    )

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image = plt.imread(buf)

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image(
                    "confusion_matrix",
                    image[np.newaxis, ...],
                    step=step,
                )

        except Exception as e:
            logger.warning(f"Failed to log confusion matrix image: {e}")
        finally:
            if buf is not None:
                buf.close()
            if fig is not None:
                plt.close(fig)

    def log_roc_pr_curves(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        step: int,
    ) -> None:
        """Log ROC and PR curves as images.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            step: Global step
        """
        if not self.enabled or not self.log_images or self.writer is None:
            return

        fig = None
        buf = None
        try:
            curves = compute_roc_pr_curves(y_true, y_score, n_thresholds=101)
            fpr = curves["fpr"]
            tpr = curves["tpr"]
            precision = curves["precision"]
            recall = curves["recall"]
            roc_auc = float(np.trapz(tpr, fpr)) if len(fpr) > 1 else 0.0
            pr_auc = float(np.trapz(precision, recall)) if len(recall) > 1 else 0.0

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # ROC Curve
            ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
            ax1.plot([0, 1], [0, 1], "k--", label="Random")
            ax1.set_xlabel("False Positive Rate")
            ax1.set_ylabel("True Positive Rate")
            ax1.set_title("ROC Curve")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # PR Curve
            ax2.plot(recall, precision, "b-", linewidth=2, label=f"PR (AUC = {pr_auc:.3f})")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Precision-Recall Curve")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image = plt.imread(buf)

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image(
                    "curves/roc_pr",
                    image[np.newaxis, ...],
                    step=step,
                )

        except Exception as e:
            logger.warning(f"Failed to log ROC/PR curves: {e}")
        finally:
            if buf is not None:
                buf.close()
            if fig is not None:
                plt.close(fig)

    def log_score_distribution_image(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        step: int,
        raw_labels: np.ndarray | None = None,
    ) -> None:
        """Log score distribution as an image.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            step: Global step
            raw_labels: Optional raw labels for hard negative separation
        """
        if not self.enabled or not self.log_images or self.writer is None:
            return

        fig = None
        buf = None
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Positive scores
            pos_scores = y_score[y_true == 1]
            if len(pos_scores) > 0:
                ax.hist(
                    pos_scores,
                    bins=50,
                    alpha=0.6,
                    label=f"Positive (n={len(pos_scores)})",
                    color="green",
                    density=True,
                )

            # Negative scores
            neg_scores = y_score[y_true == 0]
            if len(neg_scores) > 0:
                ax.hist(
                    neg_scores,
                    bins=50,
                    alpha=0.6,
                    label=f"Negative (n={len(neg_scores)})",
                    color="red",
                    density=True,
                )

            # Hard negative scores
            if raw_labels is not None:
                hn_scores = y_score[raw_labels == 2]
                if len(hn_scores) > 0:
                    ax.hist(
                        hn_scores,
                        bins=50,
                        alpha=0.6,
                        label=f"Hard Negative (n={len(hn_scores)})",
                        color="orange",
                        density=True,
                    )

            ax.set_xlabel("Model Score")
            ax.set_ylabel("Density")
            ax.set_title("Score Distribution by Class")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image = plt.imread(buf)

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image(
                    "distributions/scores",
                    image[np.newaxis, ...],
                    step=step,
                )

        except Exception as e:
            logger.warning(f"Failed to log score distribution image: {e}")
        finally:
            if buf is not None:
                buf.close()
            if fig is not None:
                plt.close(fig)

    def log_pr_curve_interactive(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        step: int,
    ) -> None:
        """Log interactive PR curve using tf.summary.pr_curve.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            step: Global step
        """
        if not self.enabled or not self.log_pr_curves or self.writer is None:
            return

        # tf.summary.pr_curve was removed in TF2; PR curve is already logged as
        # an image via log_roc_pr_curves(). Nothing to do here.
        logger.debug("log_pr_curve_interactive: tf.summary.pr_curve unavailable in TF2, skipping.")

    def log_calibration_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        step: int,
        n_bins: int = 10,
    ) -> None:
        """Log calibration curve as image.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            step: Global step
            n_bins: Number of calibration bins
        """
        if not self.enabled or not self.log_images or self.writer is None:
            return

        fig = None
        buf = None
        try:
            cal_curve = compute_calibration_curve(y_true, y_score, n_bins=n_bins)
            prob_true = cal_curve["prob_true"]
            prob_pred = cal_curve["prob_pred"]

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(prob_pred, prob_true, "bo-", label="Model")
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration Curve (Reliability Diagram)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image = plt.imread(buf)

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image(
                    "calibration/reliability_diagram",
                    image[np.newaxis, ...],
                    step=step,
                )

        except Exception as e:
            logger.warning(f"Failed to log calibration curve: {e}")
        finally:
            if buf is not None:
                buf.close()
            if fig is not None:
                plt.close(fig)

    def log_fah_recall_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        ambient_duration_hours: float,
        step: int,
    ) -> None:
        """Log FAH vs Recall operating curve as image.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            ambient_duration_hours: Duration of ambient audio
            step: Global step
        """
        if not self.enabled or not self.log_images or self.writer is None:
            return

        fig = None
        buf = None
        try:
            thresholds = np.linspace(0.01, 0.99, 100)

            recalls = []
            fahs = []

            for thresh in thresholds:
                y_pred = (y_score >= thresh).astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fah = fp / ambient_duration_hours if ambient_duration_hours > 0 else 0.0

                recalls.append(recall)
                fahs.append(fah)

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(fahs, recalls, "b-", linewidth=2, label="Operating Curve")
            ax.axhline(y=0.9, color="g", linestyle="--", alpha=0.5, label="90% Recall Target")
            ax.axvline(x=0.5, color="r", linestyle="--", alpha=0.5, label="0.5 FAH Target")
            ax.set_xlabel("False Activations per Hour")
            ax.set_ylabel("Recall")
            ax.set_title("FAH vs Recall Operating Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image = plt.imread(buf)

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image(
                    "operating/fah_recall_curve",
                    image[np.newaxis, ...],
                    step=step,
                )

        except Exception as e:
            logger.warning(f"Failed to log FAH/recall curve: {e}")
        finally:
            if buf is not None:
                buf.close()
            if fig is not None:
                plt.close(fig)

    def log_advanced_scalars(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """Log advanced scalar metrics.

        Args:
            metrics: Dictionary of metrics
            step: Global step
        """
        if not self.enabled or not self.log_advanced_scalars_enabled or self.writer is None:
            return

        try:
            with self.writer.as_default():
                # EER metrics
                if "eer" in metrics:
                    tf.summary.scalar("advanced/eer", float(metrics["eer"]), step=step)
                if "eer_threshold" in metrics:
                    tf.summary.scalar(
                        "advanced/eer_threshold",
                        float(metrics["eer_threshold"]),
                        step=step,
                    )

                # Calibration metrics
                if "brier_score" in metrics:
                    tf.summary.scalar(
                        "calibration/brier_score",
                        float(metrics["brier_score"]),
                        step=step,
                    )

                # Per-category metrics
                prefix_map = {
                    "positive_": "per_class/positive/",
                    "negative_": "per_class/negative/",
                    "hard_negative_": "per_class/hard_negative/",
                }

                for old_prefix, new_prefix in prefix_map.items():
                    for key, value in metrics.items():
                        if key.startswith(old_prefix):
                            metric_name = key[len(old_prefix) :]
                            if isinstance(value, (int, float, np.floating, np.integer)):
                                tf.summary.scalar(
                                    f"{new_prefix}{metric_name}",
                                    float(value),
                                    step=step,
                                )

                # Score distribution statistics
                for key in ["score_mean", "score_std", "score_median"]:
                    if key in metrics:
                        tf.summary.scalar(
                            f"scores/{key}",
                            float(metrics[key]),
                            step=step,
                        )

        except Exception as e:
            logger.warning(f"Failed to log advanced scalars: {e}")

    def log_weight_histograms(
        self,
        model: tf.keras.Model,
        step: int,
    ) -> None:
        """Log model weight histograms.

        Args:
            model: Keras model
            step: Global step
        """
        if not self.enabled or not self.log_histograms or not self.log_weight_histograms_enabled or self.writer is None:
            return

        try:
            with self.writer.as_default():
                for layer in model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(":", "_")
                        tf.summary.histogram(
                            f"weights/{weight_name}",
                            weight,
                            step=step,
                        )

                        # Log weight statistics
                        tf.summary.scalar(
                            f"weight_stats/{weight_name}_mean",
                            tf.reduce_mean(weight),
                            step=step,
                        )
                        tf.summary.scalar(
                            f"weight_stats/{weight_name}_std",
                            tf.math.reduce_std(weight),
                            step=step,
                        )

        except Exception as e:
            logger.warning(f"Failed to log weight histograms: {e}")

    # ==========================================================================
    # SOPHISTICATED METRICS (Phase 4)
    # ==========================================================================

    def log_learning_rate(self, lr: float, step: int) -> None:
        """Log current learning rate.

        Args:
            lr: Current learning rate value
            step: Global step
        """
        if not self.enabled or not self.log_learning_rate_enabled or self.writer is None:
            return
        try:
            with self.writer.as_default():
                tf.summary.scalar("train/learning_rate", float(lr), step=step)
        except Exception as e:
            logger.warning(f"Failed to log learning rate: {e}")

    def log_gradient_norms(
        self,
        gradients: list[tf.Tensor],
        step: int,
        model: tf.keras.Model | None = None,
    ) -> None:
        """Log gradient norm histograms and statistics.

        Args:
            gradients: List of gradient tensors from tape.gradients()
            step: Global step
            model: Optional model for layer name mapping
        """
        if not self.enabled or not self.log_gradient_norms_enabled or self.writer is None:
            return
        try:
            with self.writer.as_default():
                # Global gradient norm
                global_norm = tf.linalg.global_norm(gradients)
                tf.summary.scalar("gradients/global_norm", global_norm, step=step)

                # Per-layer gradient norms (if model provided)
                if model is not None:
                    for grad, var in zip(gradients, model.trainable_variables, strict=False):
                        if grad is not None:
                            var_name = var.name.replace(":", "_")
                            grad_norm = tf.norm(grad)
                            tf.summary.scalar(f"gradients/norm/{var_name}", grad_norm, step=step)
                # Gradient norm histogram (all gradients flattened)
                grad_list = [tf.reshape(g, [-1]) for g in gradients if g is not None]
                if grad_list:
                    all_grads = tf.concat(grad_list, axis=0)
                    tf.summary.histogram("gradients/all", all_grads, step=step, buckets=50)

                    # Gradient statistics
                    tf.summary.scalar("gradients/mean", tf.reduce_mean(all_grads), step=step)
                    tf.summary.scalar("gradients/std", tf.math.reduce_std(all_grads), step=step)
                    tf.summary.scalar("gradients/max", tf.reduce_max(tf.abs(all_grads)), step=step)

        except Exception as e:
            logger.warning(f"Failed to log gradient norms: {e}")

    def log_activation_stats(
        self,
        activations: dict[str, tf.Tensor],
        step: int,
    ) -> None:
        """Log per-layer activation statistics.

        Args:
            activations: Dictionary mapping layer names to activation tensors
            step: Global step
        """
        if not self.enabled or not self.log_activation_stats_enabled or self.writer is None:
            return
        try:
            with self.writer.as_default():
                for layer_name, activation in activations.items():
                    # Clean layer name for TensorBoard
                    clean_name = layer_name.replace(":", "_").replace("/", "_")

                    # Sparsity (% of zeros)
                    sparsity = tf.reduce_mean(tf.cast(tf.equal(activation, 0), tf.float32))
                    tf.summary.scalar(f"activations/{clean_name}/sparsity", sparsity, step=step)

                    # Activation statistics
                    tf.summary.scalar(f"activations/{clean_name}/mean", tf.reduce_mean(activation), step=step)
                    tf.summary.scalar(f"activations/{clean_name}/std", tf.math.reduce_std(activation), step=step)
                    tf.summary.scalar(
                        f"activations/{clean_name}/saturation",
                        tf.reduce_mean(tf.cast(tf.abs(activation) > 0.95, tf.float32)),
                        step=step,
                    )

                    # Histogram (occasionally)
                    if step % (self.sophisticated_interval * 5) == 0:
                        tf.summary.histogram(
                            f"activations/{clean_name}/distribution",
                            activation,
                            step=step,
                            buckets=50,
                        )

        except Exception as e:
            logger.warning(f"Failed to log activation stats: {e}")

    def log_confidence_drift(
        self,
        y_score: np.ndarray,
        step: int,
    ) -> None:
        """Track prediction confidence distribution drift over time.

        Args:
            y_score: Predicted scores
            step: Global step
        """
        if not self.enabled or not self.log_confidence_drift_enabled or self.writer is None:
            return
        try:
            # Compute confidence metrics
            mean_confidence = float(np.mean(y_score))
            high_confidence_ratio = float(np.mean(y_score > 0.9))
            low_confidence_ratio = float(np.mean(y_score < 0.1))
            uncertainty = float(np.mean(np.abs(y_score - 0.5)))

            # Store in history
            self._confidence_history.append((step, mean_confidence))
            if len(self._confidence_history) > self._max_confidence_history:
                self._confidence_history.pop(0)

            with self.writer.as_default():
                tf.summary.scalar("confidence/mean", mean_confidence, step=step)
                tf.summary.scalar("confidence/high_ratio", high_confidence_ratio, step=step)
                tf.summary.scalar("confidence/low_ratio", low_confidence_ratio, step=step)
                tf.summary.scalar("confidence/uncertainty", uncertainty, step=step)

                # Compute drift if we have history
                if len(self._confidence_history) >= 20:
                    recent_mean = np.mean([c for _, c in self._confidence_history[-10:]])
                    older_mean = np.mean([c for _, c in self._confidence_history[:10]])
                    drift = abs(recent_mean - older_mean)
                    tf.summary.scalar("confidence/drift", drift, step=step)

        except Exception as e:
            logger.warning(f"Failed to log confidence drift: {e}")

    def log_per_class_accuracy(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.5,
        step: int = 0,
    ) -> None:
        """Log accuracy metrics broken down by class.

        Args:
            y_true: Ground truth labels
            y_score: Predicted scores
            threshold: Classification threshold
            step: Global step
        """
        if not self.enabled or not self.log_per_class_accuracy_enabled or self.writer is None:
            return
        try:
            y_pred = (y_score >= threshold).astype(int)

            # Per-class metrics
            pos_mask = y_true == 1
            neg_mask = y_true == 0

            pos_correct = np.sum((y_pred == 1) & pos_mask) if np.any(pos_mask) else 0
            pos_total = np.sum(pos_mask)
            pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0.0

            neg_correct = np.sum((y_pred == 0) & neg_mask) if np.any(neg_mask) else 0
            neg_total = np.sum(neg_mask)
            neg_accuracy = neg_correct / neg_total if neg_total > 0 else 0.0

            with self.writer.as_default():
                tf.summary.scalar("per_class/positive_accuracy", pos_accuracy, step=step)
                tf.summary.scalar("per_class/negative_accuracy", neg_accuracy, step=step)

                # Class balance in batch
                pos_ratio = np.mean(y_true)
                tf.summary.scalar("train/positive_ratio", float(pos_ratio), step=step)

                # Prediction distribution
                pred_pos_ratio = np.mean(y_pred)
                tf.summary.scalar("train/predicted_positive_ratio", float(pred_pos_ratio), step=step)

        except Exception as e:
            logger.warning(f"Failed to log per-class accuracy: {e}")

    def log_sophisticated_metrics(
        self,
        step: int,
        learning_rate: float | None = None,
        gradients: list[tf.Tensor] | None = None,
        model: tf.keras.Model | None = None,
        activations: dict[str, tf.Tensor] | None = None,
        y_true: np.ndarray | None = None,
        y_score: np.ndarray | None = None,
    ) -> None:
        """Batch log all sophisticated metrics at the configured interval.

        This is the main entry point for Phase 4 sophisticated metrics.
        Call this method at the sophisticated_interval to log all enabled metrics.

        Args:
            step: Global step
            learning_rate: Current learning rate (optional)
            gradients: Gradient tensors from tape.gradients() (optional)
            model: Keras model for weight/gradient mapping (optional)
            activations: Layer activations dictionary (optional)
            y_true: Ground truth labels for per-class metrics (optional)
            y_score: Predicted scores for confidence drift (optional)
        """
        if not self.enabled or self.writer is None:
            return

        # Only log at interval
        if step % self.sophisticated_interval != 0:
            return

        try:
            if learning_rate is not None:
                self.log_learning_rate(learning_rate, step)

            if gradients is not None:
                self.log_gradient_norms(gradients, step, model)

            if activations is not None:
                self.log_activation_stats(activations, step)

            if y_score is not None:
                self.log_confidence_drift(y_score, step)

            if y_true is not None and y_score is not None:
                self.log_per_class_accuracy(y_true, y_score, step=step)

        except Exception as e:
            logger.warning(f"Failed to log sophisticated metrics: {e}")

    def log_text_summary(
        self,
        title: str,
        content: str,
        step: int,
    ) -> None:
        """Log text summary to TensorBoard.

        Args:
            title: Summary title
            content: Text content
            step: Global step
        """
        if not self.enabled or self.writer is None:
            return

        try:
            with self.writer.as_default():
                tf.summary.text(title, content, step=step)
        except Exception as e:
            logger.warning(f"Failed to log text summary: {e}")

    def flush(self) -> None:
        """Flush pending writes to disk."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close the writer and flush remaining data."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def as_default(self):
        """Context manager for default writer."""
        if self.writer is not None:
            return self.writer.as_default()
        # Return a no-op context manager
        import contextlib

        return contextlib.nullcontext()
