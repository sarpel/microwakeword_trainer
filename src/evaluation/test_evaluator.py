"""Test evaluation module for post-training held-out test analysis."""

# Force Agg backend BEFORE any pyplot import
import matplotlib

matplotlib.use("Agg")

import json
import logging
import os
from datetime import datetime
from typing import Any, Callable

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .calibration import compute_brier_score, compute_calibration_curve
from .fah_estimator import FAHEstimator
from .metrics import compute_roc_pr_curves

logger = logging.getLogger(__name__)


def _compute_mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """Compute Matthews Correlation Coefficient."""
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _compute_cohens_kappa(tp: int, fp: int, tn: int, fn: int, total: int) -> float:
    """Compute Cohen's Kappa."""
    accuracy = (tp + tn) / total if total > 0 else 0.0
    p_pos = (tp + fp) / total if total > 0 else 0.0
    p_neg = (fn + tn) / total if total > 0 else 0.0
    p_observed = accuracy
    p_expected = (p_pos * (tp + fn) / total + p_neg * (fp + tn) / total) if total > 0 else 0.0
    if p_expected == 1.0:
        return 1.0
    return float((p_observed - p_expected) / (1.0 - p_expected))


def _compute_eer_manual(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate manually from ROC data."""
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    min_idx = np.argmin(diff)
    eer = float(fpr[min_idx])
    eer_threshold = float(thresholds[min_idx])
    return eer, eer_threshold


class TestEvaluator:
    """Comprehensive test evaluator for held-out test set analysis."""

    def __init__(self, model, config: dict, log_dir: str):
        """Initialize TestEvaluator.

        Args:
            model: Trained Keras model
            config: Full config dict
            log_dir: Directory for output files
        """
        self.model = model
        self.config = config
        self.console = Console()

        perf_config = config.get("performance", {})
        paths_config = config.get("paths", {})
        training_config = config.get("training", {})

        self.log_dir = perf_config.get("tensorboard_log_dir", log_dir)
        self.processed_dir = paths_config.get("processed_dir", "./data/processed")
        evaluation_config = config.get("evaluation", {})
        self.test_split = training_config.get("test_split", 0.1)
        self.ambient_duration_hours = training_config.get("ambient_duration_hours", 10.0)
        self.default_threshold = float(evaluation_config.get("default_threshold", 0.5) or 0.5)
        self.n_thresholds = int(evaluation_config.get("n_thresholds", 101) or 101)

    def evaluate(self, test_data_factory: Callable, test_feature_store_path: str | None = None) -> dict[str, Any] | None:
        """Run comprehensive test evaluation."""
        logger.info("Starting test evaluation...")
        self.console.print("\n[bold cyan]Running Comprehensive Test Evaluation...[/bold cyan]\n")

        y_true, y_score, raw_labels = self._run_inference(test_data_factory)

        if y_true is None or len(y_true) < 10:
            logger.warning("Test set too small (< 10 samples), skipping evaluation")
            self.console.print("[yellow]Test set too small, skipping evaluation[/yellow]")
            return None

        unique_classes = np.unique(y_true)
        has_both_classes = len(unique_classes) >= 2

        if raw_labels is None and test_feature_store_path:
            raw_labels = self._load_raw_labels(test_feature_store_path)

        results: dict[str, Any] = {}
        if y_score is None:
            logger.warning("No predictions available; skipping metric computation")
            self.console.print("[yellow]No predictions available; skipping metrics[/yellow]")
            return None

        results["basic_metrics"] = self._compute_basic_metrics(y_true, y_score)
        results["advanced_metrics"] = self._compute_advanced_metrics(y_true, y_score, has_both_classes)
        results["confidence_intervals"] = self._compute_bootstrap_cis(y_true, y_score)
        results["calibration"] = self._compute_calibration(y_true, y_score)
        results["per_category"] = self._compute_per_category(y_true, y_score, raw_labels)
        results["operating_points"] = self._compute_operating_points(y_true, y_score)
        results["threshold_sweep"] = self._compute_threshold_sweep(y_true, y_score)
        results["score_distributions"] = self._compute_score_distributions(y_true, y_score, raw_labels)
        results["confusion_matrix"] = self._compute_confusion_matrix(y_true, y_score)
        results["metadata"] = {
            "test_samples": int(len(y_true)),
            "timestamp": datetime.now().isoformat(),
            "test_split": float(self.test_split),
            "ambient_duration_hours": float(self.ambient_duration_hours * self.test_split),
        }

        self._display_results(results, has_both_classes)
        self._save_json_report(results)
        self._generate_plots(y_true, y_score, raw_labels)

        logger.info("Test evaluation complete")
        return results

    def _run_inference(self, test_data_factory):
        """Run inference on test data."""
        y_true_list = []
        y_score_list = []
        raw_labels_list = []

        try:
            data_gen = test_data_factory()
            for batch in data_gen:
                if len(batch) >= 2:
                    features = batch[0]
                    labels = batch[1]
                    metadata = batch[2] if len(batch) >= 3 else None
                else:
                    continue

                predictions = self.model.predict(features, verbose=0)
                if predictions.ndim > 1:
                    predictions = predictions.reshape(predictions.shape[0], -1)
                    if predictions.shape[1] == 1:
                        predictions = predictions[:, 0]
                    else:
                        predictions = np.mean(predictions, axis=1)

                y_true_list.append(labels)
                y_score_list.append(predictions)
                if metadata is not None and isinstance(metadata, dict) and "raw_labels" in metadata:
                    raw_labels_list.append(np.array(metadata["raw_labels"], dtype=np.int32))

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return None, None, None

        if not y_true_list:
            logger.warning("No test data processed")
            return None, None, None

        y_true = np.concatenate(y_true_list).astype(np.int32)
        y_score = np.concatenate(y_score_list).astype(np.float32)
        raw_labels = None
        if raw_labels_list:
            raw_labels = np.concatenate(raw_labels_list).astype(np.int32)

        if np.any(np.isnan(y_score)) or np.any(np.isinf(y_score)):
            logger.error("NaN/Inf detected in predictions")
            return None, None, None

        return y_true, y_score, raw_labels

    def _load_raw_labels(self, store_path: str) -> np.ndarray | None:
        """Load raw labels from FeatureStore."""
        try:
            from ..data.dataset import FeatureStore

            store = FeatureStore(store_path)
            try:
                store.open()
                raw_labels = []
                for i in range(len(store)):
                    _, label = store.get(i)
                    raw_labels.append(label)
                return np.array(raw_labels, dtype=np.int32)
            finally:
                store.close()
            store.open()
            raw_labels = []
            for i in range(len(store)):
                _, label = store.get(i)
                raw_labels.append(label)
            store.close()
            return np.array(raw_labels, dtype=np.int32)
        except Exception as e:
            logger.warning(f"Could not load raw labels: {e}")
            return None

    def _compute_basic_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> dict:
        """Compute basic classification metrics at threshold 0.5."""
        threshold = self.default_threshold
        y_pred = (y_score >= threshold).astype(np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        mcc = _compute_mcc(tp, fp, tn, fn)
        kappa = _compute_cohens_kappa(tp, fp, tn, fn, len(y_true))

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "npv": float(npv),
            "mcc": float(mcc),
            "cohens_kappa": float(kappa),
        }

    def _compute_advanced_metrics(self, y_true: np.ndarray, y_score: np.ndarray, has_both_classes: bool) -> dict:
        """Compute advanced metrics including EER."""
        results: dict[str, float | int | None] = {}

        if has_both_classes:
            curves = compute_roc_pr_curves(y_true, y_score, n_thresholds=self.n_thresholds)
            fpr = curves["fpr"]
            tpr = curves["tpr"]
            thresholds = curves["thresholds"]

            # Reverse arrays: compute_roc_pr_curves sweeps thresholds 0→1,
            # so FPR/recall are DECREASING. np.trapz needs ascending x-values.
            results["auc_roc"] = float(np.trapz(tpr[::-1], fpr[::-1]))
            results["auc_pr"] = float(np.trapz(curves["precision"][::-1], curves["recall"][::-1])) if "precision" in curves else None

            eer, eer_threshold = _compute_eer_manual(fpr, tpr, thresholds)
            results["eer"] = eer
            results["eer_threshold"] = eer_threshold
        else:
            results["auc_roc"] = None
            results["auc_pr"] = None
            results["eer"] = None
            results["eer_threshold"] = None

        scaled_duration = self.ambient_duration_hours * self.test_split
        fah_estimator = FAHEstimator(ambient_duration_hours=scaled_duration)
        fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_score, threshold=self.default_threshold)
        results["fah"] = fah_metrics.get("ambient_false_positives_per_hour", 0.0)
        results["false_positives"] = fah_metrics.get("ambient_false_positives", 0)

        return results

    def _compute_bootstrap_cis(self, y_true: np.ndarray, y_score: np.ndarray, n_iterations: int = 1000) -> dict:
        """Compute bootstrap confidence intervals."""
        rng = np.random.RandomState(42)
        n_samples = len(y_true)

        def _bootstrap_sample():
            idx = rng.randint(0, n_samples, n_samples)
            y_t = y_true[idx]
            y_s = y_score[idx]

            y_p = (y_s >= self.default_threshold).astype(np.int32)
            tp = np.sum((y_t == 1) & (y_p == 1))
            fp = np.sum((y_t == 0) & (y_p == 1))
            fn = np.sum((y_t == 1) & (y_p == 0))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            fp_count = int(fp)
            scaled_duration = self.ambient_duration_hours * self.test_split
            fah = fp_count / scaled_duration if scaled_duration > 0 else 0.0

            return recall, precision, f1, fah

        recalls: list[float] = []
        precisions: list[float] = []
        f1s: list[float] = []
        fahs: list[float] = []

        for _ in range(n_iterations):
            r, p, f, fah = _bootstrap_sample()
            recalls.append(r)
            precisions.append(p)
            f1s.append(f)
            fahs.append(fah)

        recalls_arr: np.ndarray = np.array(recalls)
        precisions_arr: np.ndarray = np.array(precisions)
        f1s_arr: np.ndarray = np.array(f1s)
        fahs_arr: np.ndarray = np.array(fahs)

        return {
            "recall_ci": [float(np.percentile(recalls_arr, 2.5)), float(np.percentile(recalls_arr, 97.5))],
            "precision_ci": [float(np.percentile(precisions_arr, 2.5)), float(np.percentile(precisions_arr, 97.5))],
            "f1_ci": [float(np.percentile(f1s_arr, 2.5)), float(np.percentile(f1s_arr, 97.5))],
            "fah_ci": [float(np.percentile(fahs_arr, 2.5)), float(np.percentile(fahs_arr, 97.5))],
        }

    def _compute_calibration(self, y_true: np.ndarray, y_score: np.ndarray) -> dict:
        """Compute calibration metrics."""
        brier = compute_brier_score(y_true, y_score)
        curve = compute_calibration_curve(y_true, y_score, n_bins=10)

        return {
            "brier_score": float(brier),
            "calibration_curve": {
                "prob_true": curve["prob_true"].tolist(),
                "prob_pred": curve["prob_pred"].tolist(),
                "counts": curve["counts"].tolist(),
                "bin_edges": curve["bin_edges"].tolist(),
            },
        }

    def _compute_per_category(self, y_true: np.ndarray, y_score: np.ndarray, raw_labels: np.ndarray | None) -> dict:
        """Compute per-category breakdown."""
        if raw_labels is None:
            return {"available": False}

        threshold = self.default_threshold
        y_pred = (y_score >= threshold).astype(np.int32)

        results: dict[str, dict[str, int | float] | bool] = {"available": True}

        pos_mask = raw_labels == 1
        if np.sum(pos_mask) > 0:
            pos_tp = np.sum((y_true[pos_mask] == 1) & (y_pred[pos_mask] == 1))
            pos_fn = np.sum((y_true[pos_mask] == 1) & (y_pred[pos_mask] == 0))
            pos_tpr = pos_tp / (pos_tp + pos_fn) if (pos_tp + pos_fn) > 0 else 0.0
            results["positive"] = {"count": int(np.sum(pos_mask)), "true_positive_rate": float(pos_tpr)}

        neg_mask = raw_labels == 0
        if np.sum(neg_mask) > 0:
            neg_tn = np.sum((y_true[neg_mask] == 0) & (y_pred[neg_mask] == 0))
            neg_fp = np.sum((y_true[neg_mask] == 0) & (y_pred[neg_mask] == 1))
            neg_tnr = neg_tn / (neg_tn + neg_fp) if (neg_tn + neg_fp) > 0 else 0.0
            results["negative"] = {
                "count": int(np.sum(neg_mask)),
                "true_negative_rate": float(neg_tnr),
                "false_positive_rate": float(1.0 - neg_tnr),
            }

        hn_mask = raw_labels == 2
        if np.sum(hn_mask) > 0:
            hn_tn = np.sum((y_true[hn_mask] == 0) & (y_pred[hn_mask] == 0))
            hn_fp = np.sum((y_true[hn_mask] == 0) & (y_pred[hn_mask] == 1))
            hn_tnr = hn_tn / (hn_tn + hn_fp) if (hn_tn + hn_fp) > 0 else 0.0
            results["hard_negative"] = {
                "count": int(np.sum(hn_mask)),
                "true_negative_rate": float(hn_tnr),
                "false_positive_rate": float(1.0 - hn_tnr),
            }

        return results

    def _compute_operating_points(self, y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
        """Find optimal thresholds at target FAH rates."""
        target_fahs = [0.1, 0.5, 1.0, 2.0]
        scaled_duration = self.ambient_duration_hours * self.test_split

        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        fah_estimator = FAHEstimator(ambient_duration_hours=scaled_duration)

        operating_points = []
        for target_fah in target_fahs:
            best_threshold = self.default_threshold
            best_recall = 0.0
            best_fah = float("inf")

            for thresh in thresholds:
                fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_score, threshold=thresh)
                fah = fah_metrics.get("ambient_false_positives_per_hour", float("inf"))
                recall = fah_metrics.get("recall", 0.0)

                if fah <= target_fah and recall > best_recall:
                    best_fah = fah
                    best_recall = recall
                    best_threshold = thresh

            operating_points.append(
                {
                    "target_fah": target_fah,
                    "achieved_fah": float(best_fah),
                    "threshold": float(best_threshold),
                    "recall": float(best_recall),
                }
            )

        return operating_points

    def _compute_threshold_sweep(self, y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
        """Sweep thresholds and compute metrics."""
        thresholds = np.arange(0.1, 1.0, 0.1)
        scaled_duration = self.ambient_duration_hours * self.test_split
        fah_estimator = FAHEstimator(ambient_duration_hours=scaled_duration)

        results = []
        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(np.int32)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_score, threshold=thresh)
            fah = fah_metrics.get("ambient_false_positives_per_hour", 0.0)

            results.append(
                {
                    "threshold": float(thresh),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "fah": float(fah),
                }
            )

        return results

    def _compute_score_distributions(self, y_true: np.ndarray, y_score: np.ndarray, raw_labels: np.ndarray | None) -> dict:
        """Compute score distribution statistics."""
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]

        results = {
            "positive": {
                "mean": float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0,
                "std": float(np.std(pos_scores)) if len(pos_scores) > 0 else 0.0,
                "median": float(np.median(pos_scores)) if len(pos_scores) > 0 else 0.0,
                "min": float(np.min(pos_scores)) if len(pos_scores) > 0 else 0.0,
                "max": float(np.max(pos_scores)) if len(pos_scores) > 0 else 0.0,
            },
            "negative": {
                "mean": float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0,
                "std": float(np.std(neg_scores)) if len(neg_scores) > 0 else 0.0,
                "median": float(np.median(neg_scores)) if len(neg_scores) > 0 else 0.0,
                "min": float(np.min(neg_scores)) if len(neg_scores) > 0 else 0.0,
                "max": float(np.max(neg_scores)) if len(neg_scores) > 0 else 0.0,
            },
        }

        if raw_labels is not None:
            hn_scores = y_score[raw_labels == 2]
            if len(hn_scores) > 0:
                results["hard_negative"] = {
                    "mean": float(np.mean(hn_scores)),
                    "std": float(np.std(hn_scores)),
                    "median": float(np.median(hn_scores)),
                    "min": float(np.min(hn_scores)),
                    "max": float(np.max(hn_scores)),
                }

        return results

    def _compute_confusion_matrix(self, y_true: np.ndarray, y_score: np.ndarray) -> dict:
        """Compute confusion matrix at threshold 0.5."""
        threshold = self.default_threshold
        y_pred = (y_score >= threshold).astype(np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def _display_results(self, results: dict, has_both_classes: bool):
        """Display results with Rich console."""
        self.console.print(Panel.fit("[bold cyan]Test Evaluation Results[/bold cyan]", border_style="cyan"))

        basic = results["basic_metrics"]
        basic_table = Table(title="Basic Metrics", show_header=True)
        basic_table.add_column("Metric", style="cyan")
        basic_table.add_column("Value", style="green")
        for key, value in basic.items():
            if value is not None:
                basic_table.add_row(key, f"{value:.4f}")
        self.console.print(basic_table)

        advanced = results["advanced_metrics"]
        adv_table = Table(title="Advanced Metrics", show_header=True)
        adv_table.add_column("Metric", style="cyan")
        adv_table.add_column("Value", style="yellow")

        if has_both_classes:
            if advanced.get("auc_roc") is not None:
                adv_table.add_row("AUC-ROC", f"{advanced['auc_roc']:.4f}")
            if advanced.get("eer") is not None:
                adv_table.add_row("EER", f"{advanced['eer']:.4f}")
                adv_table.add_row("EER Threshold", f"{advanced['eer_threshold']:.4f}")

        adv_table.add_row("FAH", f"{advanced.get('fah', 0):.4f}")
        adv_table.add_row("False Positives", str(advanced.get("false_positives", 0)))
        self.console.print(adv_table)

        cm = results["confusion_matrix"]
        cm_table = Table(title=f"Confusion Matrix (threshold={self.default_threshold:.2f})", show_header=True)
        cm_table.add_column("", style="cyan")
        cm_table.add_column("Pred: Negative", style="red")
        cm_table.add_column("Pred: Positive", style="green")
        cm_table.add_row("Actual: Negative", str(cm["tn"]), str(cm["fp"]))
        cm_table.add_row("Actual: Positive", str(cm["fn"]), str(cm["tp"]))
        self.console.print(cm_table)

        per_cat = results.get("per_category", {})
        if per_cat.get("available"):
            cat_table = Table(title="Per-Category Breakdown", show_header=True)
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="yellow")
            cat_table.add_column("TNR/TPR", style="green")
            for cat_name in ["positive", "negative", "hard_negative"]:
                if cat_name in per_cat:
                    cat = per_cat[cat_name]
                    rate = cat.get("true_positive_rate") or cat.get("true_negative_rate")
                    cat_table.add_row(cat_name, str(cat["count"]), f"{rate:.4f}" if rate else "N/A")
            self.console.print(cat_table)

        cis = results.get("confidence_intervals", {})
        if cis:
            ci_table = Table(title="Bootstrap 95% Confidence Intervals", show_header=True)
            ci_table.add_column("Metric", style="cyan")
            ci_table.add_column("CI Range", style="yellow")
            for metric in ["recall", "precision", "f1", "fah"]:
                key = f"{metric}_ci"
                if key in cis:
                    lo, hi = cis[key]
                    ci_table.add_row(metric, f"[{lo:.4f}, {hi:.4f}]")
            self.console.print(ci_table)

        self.console.print(f"\n[bold green]JSON report and plots saved to {self.log_dir}/[/bold green]\n")

    def _save_json_report(self, results: dict):
        """Save metrics to JSON report."""
        os.makedirs(self.log_dir, exist_ok=True)
        json_path = os.path.join(self.log_dir, "test_report.json")

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        converted = convert(results)

        with open(json_path, "w") as f:
            json.dump(converted, f, indent=2)

        logger.info(f"JSON report saved to {json_path}")

    def _generate_plots(self, y_true: np.ndarray, y_score: np.ndarray, raw_labels: np.ndarray | None):
        """Generate 6 matplotlib plots."""
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        os.makedirs(self.log_dir, exist_ok=True)

        curves = compute_roc_pr_curves(y_true, y_score)
        fpr, tpr = curves["fpr"], curves["tpr"]

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, "b-", linewidth=2, label="ROC")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "test_roc.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(curves["recall"], curves["precision"], "b-", linewidth=2, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "test_pr.png"), dpi=150)
        plt.close()

        fnr = 1 - tpr
        with np.errstate(divide="ignore", invalid="ignore"):
            fpr_det = norm.ppf(np.clip(fpr, 1e-6, 1 - 1e-6))
            fnr_det = norm.ppf(np.clip(fnr, 1e-6, 1 - 1e-6))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_det, fnr_det, "b-", linewidth=2, label="DET")
        plt.xlabel("False Positive Rate (normal deviate)")
        plt.ylabel("False Negative Rate (normal deviate)")
        plt.title("Detection Error Tradeoff (DET) Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "test_det.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        plt.hist(pos_scores, bins=30, alpha=0.5, label="Positive", color="blue")
        plt.hist(neg_scores, bins=30, alpha=0.5, label="Negative", color="orange")
        if raw_labels is not None:
            hn_scores = y_score[raw_labels == 2]
            if len(hn_scores) > 0:
                plt.hist(hn_scores, bins=30, alpha=0.5, label="Hard Negative", color="red")
        plt.xlabel("Model Score")
        plt.ylabel("Count")
        plt.title("Score Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "test_scores.png"), dpi=150)
        plt.close()

        cal_curve = compute_calibration_curve(y_true, y_score, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(cal_curve["prob_pred"], cal_curve["prob_true"], "bo-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Reliability Diagram)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "test_calibration.png"), dpi=150)
        plt.close()

        thresholds = np.linspace(0.01, 0.99, 100)
        scaled_duration = self.ambient_duration_hours * self.test_split
        fah_estimator = FAHEstimator(ambient_duration_hours=scaled_duration)

        recalls, fahs_list = [], []
        for thresh in thresholds:
            fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_score, threshold=thresh)
            recalls.append(fah_metrics.get("recall", 0))
            fahs_list.append(fah_metrics.get("ambient_false_positives_per_hour", 0))

        plt.figure(figsize=(8, 6))
        plt.plot(fahs_list, recalls, "b-", linewidth=2, label="Operating Curve")
        plt.axhline(y=0.9, color="g", linestyle="--", alpha=0.5, label="90% Recall")
        plt.axvline(x=0.5, color="r", linestyle="--", alpha=0.5, label="0.5 FAH")
        plt.xlabel("False Activations per Hour")
        plt.ylabel("Recall")
        plt.title("FAH vs Recall Operating Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "test_fah_recall.png"), dpi=150)
        plt.close()

        logger.info(f"Plots saved to {self.log_dir}")
