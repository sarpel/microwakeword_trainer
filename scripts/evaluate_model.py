"""Post-training evaluation script with advanced model quality diagnostics.

Features:
- Basic and advanced binary classification metrics
- FAH-aware threshold analysis for wake-word deployment
- Bootstrap confidence intervals
- Calibration analysis (Brier score + ECE + reliability bins)
- Confusion matrix (raw + normalized)
- Plot artifacts (ROC, PR, DET, distributions, calibration, threshold/cost curves)

Usage examples:
    python scripts/evaluate_model.py --checkpoint checkpoints/best_weights.weights.h5 --config standard
    python scripts/evaluate_model.py --tflite models/exported/wake_word.tflite --config standard --json
    python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard --output-dir logs/eval
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib
import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.calibration import compute_brier_score, compute_calibration_curve
from src.evaluation.fah_estimator import FAHEstimator
from src.evaluation.metrics import MetricsCalculator


def evaluate_model(
    model_path: str,
    config: dict,
    split: str = "test",
    output_dir: str | None = None,
    n_thresholds: int | None = None,
    bootstrap_iterations: int = 400,
    fp_cost: float = 20.0,
    fn_cost: float = 1.0,
    save_plots: bool = True,
    console: Console | None = None,
) -> dict:
    """Evaluate model on specified dataset split.

    Args:
        model_path: Path to checkpoint or TFLite model
        config: Training configuration
        split: Dataset split to evaluate ('train', 'val', 'test')
        output_dir: Directory to save JSON report and plot artifacts
        n_thresholds: Override threshold sweep resolution
        bootstrap_iterations: Number of bootstrap resamples
        fp_cost: Cost weight for false positives (cost-aware thresholding)
        fn_cost: Cost weight for false negatives (cost-aware thresholding)
        save_plots: Whether to generate plot artifacts
        console: Rich console for output

    Returns:
        Evaluation metrics dictionary
    """
    console = console or Console()
    eval_n_thresholds = int(n_thresholds or config.get("evaluation", {}).get("n_thresholds", 201) or 201)

    # Load model
    if model_path.endswith(".tflite"):
        metrics = _evaluate_tflite(
            tflite_path=model_path,
            config=config,
            split=split,
            output_dir=output_dir,
            n_thresholds=eval_n_thresholds,
            bootstrap_iterations=bootstrap_iterations,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            save_plots=save_plots,
            console=console,
        )
    else:
        metrics = _evaluate_checkpoint(
            checkpoint_path=model_path,
            config=config,
            split=split,
            output_dir=output_dir,
            n_thresholds=eval_n_thresholds,
            bootstrap_iterations=bootstrap_iterations,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            save_plots=save_plots,
            console=console,
        )

    return metrics


def _load_manifest_threshold(tflite_path: str) -> float | None:
    """Load probability_cutoff from manifest.json adjacent to TFLite file."""
    manifest_path = Path(tflite_path).with_name("manifest.json")
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    micro = data.get("micro") if isinstance(data, dict) else None
    cutoff = micro.get("probability_cutoff") if isinstance(micro, dict) else None
    if isinstance(cutoff, (int, float)) and 0.0 <= float(cutoff) <= 1.0:
        return float(cutoff)
    return None


def _resolve_eval_threshold(model_path: str, config: dict) -> tuple[float, str]:
    """Resolve threshold with deployment-aware precedence.

    For exported TFLite evaluation, prefer the adjacent manifest cutoff so reported
    precision/recall/F1/FAH match actual deployment behavior.
    """
    if model_path.endswith(".tflite"):
        manifest_cutoff = _load_manifest_threshold(model_path)
        if manifest_cutoff is not None:
            return manifest_cutoff, "manifest.micro.probability_cutoff"

    export_cutoff = config.get("export", {}).get("probability_cutoff")
    if isinstance(export_cutoff, (int, float)):
        return float(export_cutoff), "config.export.probability_cutoff"

    eval_cutoff = config.get("evaluation", {}).get("default_threshold")
    if isinstance(eval_cutoff, (int, float)):
        return float(eval_cutoff), "config.evaluation.default_threshold"

    return 0.5, "fallback=0.5"


def _build_and_load_model(checkpoint_path: str, config: dict) -> tuple[tf.keras.Model, tuple[int, int]]:
    """Build model architecture and load weights.

    Args:
        checkpoint_path: Path to weights file
        config: Training configuration

    Returns:
        Tuple of (built model with loaded weights, input_shape)
    """
    from src.model.architecture import build_model

    # Get model parameters from config
    hardware_cfg = config.get("hardware", {})
    clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
    window_step_ms = hardware_cfg.get("window_step_ms", 10)
    mel_bins = hardware_cfg.get("mel_bins", 40)
    input_shape = (int(clip_duration_ms / window_step_ms), mel_bins)

    # Build model
    model = build_model(
        input_shape=input_shape,
        first_conv_filters=config.get("model", {}).get("first_conv_filters", 32),
        first_conv_kernel_size=config.get("model", {}).get("first_conv_kernel_size", 5),
        stride=config.get("model", {}).get("stride", 3),
        pointwise_filters=config.get("model", {}).get("pointwise_filters", "64,64,64,64"),
        mixconv_kernel_sizes=config.get("model", {}).get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
        repeat_in_block=config.get("model", {}).get("repeat_in_block", "1,1,1,1"),
        residual_connection=config.get("model", {}).get("residual_connection", "0,0,0,0"),
        dropout_rate=config.get("model", {}).get("dropout_rate", 0.0),
        mode="non_stream",
    )

    # Build model by calling it on dummy data
    # This is required before loading weights in Keras 3
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    _ = model(dummy_input, training=False)

    # Now load weights
    model.load_weights(checkpoint_path)

    return model, input_shape


def _evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    split: str,
    output_dir: str | None,
    n_thresholds: int,
    bootstrap_iterations: int,
    fp_cost: float,
    fn_cost: float,
    save_plots: bool,
    console: Console,
) -> dict:
    """Evaluate Keras checkpoint."""
    from src.data.dataset import WakeWordDataset

    # Build and load model
    model, input_shape = _build_and_load_model(checkpoint_path, config)
    threshold, threshold_source = _resolve_eval_threshold(checkpoint_path, config)

    # Load dataset
    dataset = WakeWordDataset(config)
    dataset.build()

    # Get appropriate generator
    max_time_frames = input_shape[0]
    gen_factory = _get_generator_factory(dataset, split, max_time_frames, console)

    # Collect predictions
    y_true = []
    y_scores = []

    gen = gen_factory()
    for batch in gen:
        batch_features = batch[0]
        batch_labels = batch[1]
        predictions = model.predict(batch_features, verbose=0)
        y_true.extend(batch_labels.flatten().tolist())
        y_scores.extend(predictions.flatten().tolist())
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Debug: Print prediction distribution
    console.print(f"\n[dim]Debug: y_true unique values: {np.unique(y_true)}[/]")
    console.print(f"[dim]Debug: y_scores range: [{y_scores.min():.4f}, {y_scores.max():.4f}], mean: {y_scores.mean():.4f}[/]")
    console.print(f"[dim]Debug: Positive samples: {y_true.sum()}/{len(y_true)} ({100 * y_true.mean():.1f}%)[/]")
    console.print(f"[dim]Debug: Threshold source: {threshold_source}[/]")
    console.print(f"[dim]Debug: Predictions > {threshold:.2f}: {(y_scores > threshold).sum()}/{len(y_scores)}[/]")

    return _compute_metrics(
        y_true=y_true,
        y_scores=y_scores,
        config=config,
        threshold=threshold,
        threshold_source=threshold_source,
        split=split,
        model_path=checkpoint_path,
        model_type="checkpoint",
        output_dir=output_dir,
        n_thresholds=n_thresholds,
        bootstrap_iterations=bootstrap_iterations,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        save_plots=save_plots,
        console=console,
    )


def _get_generator_factory(dataset, split: str, max_time_frames: int, console: Console):
    """Get generator factory for specified split.

    Args:
        dataset: WakeWordDataset instance
        split: Dataset split name
        max_time_frames: Maximum time frames
        console: Rich console

    Returns:
        Generator factory function
    """
    # Try different generator methods based on split
    if split == "test":
        # Try test generator first
        if hasattr(dataset, "test_generator_factory"):
            try:
                return dataset.test_generator_factory(max_time_frames)
            except (AttributeError, ValueError):
                pass
        console.print("[yellow]Warning: No test split found, using validation split[/]")
        return dataset.val_generator_factory(max_time_frames)
    elif split == "val":
        return dataset.val_generator_factory(max_time_frames)
    else:
        return dataset.train_generator_factory(max_time_frames)


def _evaluate_tflite(
    tflite_path: str,
    config: dict,
    split: str,
    output_dir: str | None,
    n_thresholds: int,
    bootstrap_iterations: int,
    fp_cost: float,
    fn_cost: float,
    save_plots: bool,
    console: Console,
) -> dict:
    """Evaluate streaming TFLite model.

    The exported TFLite model is a streaming model with input shape [1, 3, 40]
    (Article II). Full spectrograms must be fed in stride-sized chunks, and the
    final output after processing the entire clip is the prediction.

    State variables (Article VI) must be explicitly zeroed between samples
    because allocate_tensors() only invokes CALL_ONCE on the first call.
    """
    from src.data.dataset import WakeWordDataset

    # Get evaluation threshold from config (instead of hardcoding 0.5)
    threshold, threshold_source = _resolve_eval_threshold(tflite_path, config)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Architectural constants (Article II, Article VII)
    stride = 3
    mel_bins = 40
    input_dtype = input_details[0]["dtype"]
    input_quant = input_details[0].get("quantization_parameters", {})
    input_scale = input_quant.get("scales", np.array([1.0]))[0]
    input_zero_point = input_quant.get("zero_points", np.array([0]))[0]

    output_dtype = output_details[0]["dtype"]
    output_quant = output_details[0].get("quantization_parameters", {})
    output_scale = output_quant.get("scales", np.array([1.0]))[0]
    output_zero_point = output_quant.get("zero_points", np.array([0]))[0]

    # Discover streaming state tensors (Article VI: exactly 6 state variables)
    # State tensors in TFLite are identified by 'ReadVariableOp' in their name.
    # These are the ring buffer variables that persist across inference calls.
    # allocate_tensors() does NOT re-invoke CALL_ONCE (state zeroing subgraph)
    # after the first call, so we must zero them manually between samples.
    input_indices = {d["index"] for d in input_details}
    output_indices = {d["index"] for d in output_details}

    state_tensors = []  # list of (index, shape, dtype)
    for tensor in interpreter.get_tensor_details():
        idx = tensor["index"]
        if idx in input_indices or idx in output_indices:
            continue
        name = tensor.get("name", "")
        # State variables have 'ReadVariableOp' in their name from streaming.py tf.Variable usage
        if "ReadVariableOp" in name:
            shape = tuple(tensor.get("shape", ()))
            state_tensors.append((idx, shape, tensor["dtype"]))

    console.print(f"  Found {len(state_tensors)} state tensors to reset between samples")
    if len(state_tensors) != 6:
        console.print(f"[yellow]  Warning: Expected 6 state variables, found {len(state_tensors)}[/]")

    # Load dataset
    dataset = WakeWordDataset(config)
    dataset.build()

    hardware_cfg = config.get("hardware", {})
    clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
    window_step_ms = hardware_cfg.get("window_step_ms", 10)
    max_time_frames = int(clip_duration_ms / window_step_ms)

    # Get generator
    gen_factory = _get_generator_factory(dataset, split, max_time_frames, console)

    # Collect predictions
    y_true = []
    y_scores = []
    sample_count = 0

    gen = gen_factory()
    for batch in gen:
        batch_features = batch[0]
        batch_labels = batch[1]
        # Process each sample in batch
        for i in range(len(batch_features)):
            full_spectrogram = batch_features[i]  # shape: [time_frames, 40]
            label = batch_labels[i]
            num_frames = full_spectrogram.shape[0]

            # Reset all streaming state variables to zero (Article VI)
            # allocate_tensors() does NOT re-invoke CALL_ONCE after the first call,
            # so state from previous samples leaks into the next prediction.
            for idx, shape, dtype in state_tensors:
                interpreter.set_tensor(idx, np.zeros(shape, dtype=dtype))

            # Streaming inference: slide stride-sized chunks across spectrogram
            prediction = 0.0
            for t in range(0, num_frames - stride + 1, stride):
                chunk = full_spectrogram[t : t + stride]  # [stride, mel_bins]
                chunk = chunk.reshape(1, stride, mel_bins)  # [1, 3, 40]

                # Quantize float32 -> int8 (Article II, Article III)
                if input_dtype != np.float32:
                    chunk = np.clip(chunk / input_scale + input_zero_point, -128, 127).astype(input_dtype)

                interpreter.set_tensor(input_details[0]["index"], chunk)
                interpreter.invoke()

                raw_out = interpreter.get_tensor(output_details[0]["index"])
                # Dequantize uint8 -> float (Article II, Article III)
                if output_dtype != np.float32:
                    prediction = (raw_out.astype(np.float32) - output_zero_point) * output_scale
                    prediction = prediction.flatten()[0]
                else:
                    prediction = raw_out.flatten()[0]

            y_true.append(label)
            y_scores.append(prediction)
            sample_count += 1

            if sample_count % 200 == 0:
                console.print(f"  Processed {sample_count} samples...")

    console.print(f"  Total: {sample_count} samples")
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Debug: Print prediction distribution
    console.print(f"\n[dim]Debug: y_true unique values: {np.unique(y_true)}[/]")
    console.print(f"[dim]Debug: y_scores range: [{y_scores.min():.4f}, {y_scores.max():.4f}], mean: {y_scores.mean():.4f}[/]")
    console.print(f"[dim]Debug: Positive samples: {y_true.sum()}/{len(y_true)} ({100 * y_true.mean():.1f}%)[/]")
    console.print(f"[dim]Debug: Threshold source: {threshold_source}[/]")
    console.print(f"[dim]Debug: Predictions > {threshold:.2f}: {(y_scores > threshold).sum()}/{len(y_scores)}[/]")

    return _compute_metrics(
        y_true=y_true,
        y_scores=y_scores,
        config=config,
        threshold=threshold,
        threshold_source=threshold_source,
        split=split,
        model_path=tflite_path,
        model_type="tflite",
        output_dir=output_dir,
        n_thresholds=n_thresholds,
        bootstrap_iterations=bootstrap_iterations,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        save_plots=save_plots,
        console=console,
    )


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


def _bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    ambient_hours: float,
    iterations: int,
) -> dict[str, Any]:
    if iterations <= 0 or len(y_true) < 2:
        return {"iterations": 0}

    rng = np.random.RandomState(42)
    n = len(y_true)

    metric_samples: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "specificity": [],
        "auc_roc": [],
        "auc_pr": [],
        "ambient_false_positives_per_hour": [],
        "brier_score": [],
    }

    for _ in range(iterations):
        idx = rng.randint(0, n, n)
        ys = y_scores[idx]
        yt = y_true[idx]

        calc = MetricsCalculator(y_true=yt, y_score=ys)
        m = calc.compute_all_metrics(ambient_duration_hours=ambient_hours, threshold=threshold)

        yp = (ys >= threshold).astype(int)
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        metric_samples["accuracy"].append(float(m.get("accuracy", 0.0)))
        metric_samples["precision"].append(float(m.get("precision", 0.0)))
        metric_samples["recall"].append(float(m.get("recall", 0.0)))
        metric_samples["f1_score"].append(float(m.get("f1_score", 0.0)))
        metric_samples["specificity"].append(_safe_div(tn, tn + fp))
        metric_samples["auc_roc"].append(float(m.get("auc_roc", 0.5)))
        auc_pr = m.get("auc_pr")
        if auc_pr is not None:
            metric_samples["auc_pr"].append(float(auc_pr))
        metric_samples["ambient_false_positives_per_hour"].append(float(m.get("ambient_false_positives_per_hour", 0.0)))
        metric_samples["brier_score"].append(float(compute_brier_score(yt, ys)))

    cis: dict[str, Any] = {"iterations": iterations}
    for name, values in metric_samples.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        cis[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }

    return cis


def _compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    raw = [[tn, fp], [fn, tp]]
    row0 = tn + fp
    row1 = fn + tp
    normalized = [
        [_safe_div(tn, row0), _safe_div(fp, row0)],
        [_safe_div(fn, row1), _safe_div(tp, row1)],
    ]

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "raw": raw,
        "normalized": normalized,
    }


def _compute_ece(calibration_curve: dict[str, np.ndarray]) -> dict[str, float]:
    counts = np.asarray(calibration_curve.get("counts", []), dtype=float)
    prob_true = np.asarray(calibration_curve.get("prob_true", []), dtype=float)
    prob_pred = np.asarray(calibration_curve.get("prob_pred", []), dtype=float)
    if counts.size == 0 or counts.sum() <= 0:
        return {"ece": 0.0, "mce": 0.0}

    mask = counts > 0
    gaps = np.abs(prob_true[mask] - prob_pred[mask])
    weights = counts[mask] / counts.sum()
    ece = float(np.sum(weights * gaps)) if gaps.size else 0.0
    mce = float(np.max(gaps)) if gaps.size else 0.0
    return {"ece": ece, "mce": mce}


def _threshold_sweep(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
    ambient_hours: float,
    fp_cost: float,
    fn_cost: float,
) -> dict[str, Any]:
    fah_estimator = FAHEstimator(ambient_duration_hours=ambient_hours)
    points: list[dict[str, float]] = []

    best_f1: tuple[float, float] = (-1.0, 0.5)
    best_bal_acc: tuple[float, float] = (-1.0, 0.5)
    best_cost: tuple[float, float] = (float("inf"), 0.5)

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        accuracy = _safe_div(tp + tn, len(y_true))
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
        mcc = _safe_div((tp * tn) - (fp * fn), mcc_den)
        balanced_accuracy = 0.5 * (recall + specificity)
        expected_cost = (fp_cost * fp + fn_cost * fn) / max(len(y_true), 1)

        fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_scores, threshold=float(threshold))
        fah = float(fah_metrics.get("ambient_false_positives_per_hour", 0.0))

        if f1 > best_f1[0]:
            best_f1 = (f1, float(threshold))
        if balanced_accuracy > best_bal_acc[0]:
            best_bal_acc = (balanced_accuracy, float(threshold))
        if expected_cost < best_cost[0]:
            best_cost = (expected_cost, float(threshold))

        points.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "f1": float(f1),
                "mcc": float(mcc),
                "balanced_accuracy": float(balanced_accuracy),
                "expected_cost": float(expected_cost),
                "ambient_false_positives_per_hour": float(fah),
            }
        )

    return {
        "points": points,
        "best_by_f1": {"f1": float(best_f1[0]), "threshold": float(best_f1[1])},
        "best_by_balanced_accuracy": {
            "balanced_accuracy": float(best_bal_acc[0]),
            "threshold": float(best_bal_acc[1]),
        },
        "best_by_expected_cost": {
            "expected_cost": float(best_cost[0]),
            "threshold": float(best_cost[1]),
            "fp_cost": float(fp_cost),
            "fn_cost": float(fn_cost),
        },
    }


def _plot_and_save_all(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    confusion: dict[str, Any],
    calibration: dict[str, Any],
    threshold_sweep: dict[str, Any],
    output_dir: Path,
    threshold_used: float,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    calc = MetricsCalculator(y_true=y_true, y_score=y_scores)
    curves = calc.compute_roc_pr_curves(n_thresholds=401)
    fpr = np.asarray(curves.get("fpr", []), dtype=float)
    tpr = np.asarray(curves.get("tpr", []), dtype=float)
    precision = np.asarray(curves.get("precision", []), dtype=float)
    recall = np.asarray(curves.get("recall", []), dtype=float)

    # 1) Confusion matrix (raw + normalized)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw = np.asarray(confusion["raw"], dtype=float)
    norm = np.asarray(confusion["normalized"], dtype=float)
    im0 = axes[0].imshow(raw, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Raw)")
    axes[0].set_xticks([0, 1], ["Pred Neg", "Pred Pos"])
    axes[0].set_yticks([0, 1], ["True Neg", "True Pos"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{int(raw[i, j])}", ha="center", va="center")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(norm, cmap="Greens", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xticks([0, 1], ["Pred Neg", "Pred Pos"])
    axes[1].set_yticks([0, 1], ["True Neg", "True Pos"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{norm[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = output_dir / "eval_confusion_matrix.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    artifacts.append(str(p))

    # 2) ROC curve
    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = output_dir / "eval_roc_curve.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    artifacts.append(str(p))

    # 3) PR curve
    fig = plt.figure(figsize=(7, 6))
    plt.step(recall, precision, where="post", linewidth=2, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.ylim([0.0, 1.02])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = output_dir / "eval_pr_curve.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    artifacts.append(str(p))

    # 4) DET curve
    fnr = 1.0 - tpr
    with np.errstate(divide="ignore", invalid="ignore"):
        from scipy.stats import norm

        det_x = norm.ppf(np.clip(fpr, 1e-6, 1 - 1e-6))
        det_y = norm.ppf(np.clip(fnr, 1e-6, 1 - 1e-6))

    fig = plt.figure(figsize=(7, 6))
    plt.plot(det_x, det_y, linewidth=2, label="DET")
    plt.xlabel("False Positive Rate (normal deviate)")
    plt.ylabel("False Negative Rate (normal deviate)")
    plt.title("DET Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = output_dir / "eval_det_curve.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    artifacts.append(str(p))

    # 5) Score distributions
    fig = plt.figure(figsize=(8, 6))
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    if len(pos_scores) > 0:
        plt.hist(pos_scores, bins=40, alpha=0.55, density=True, label="Positive", color="tab:blue")
    if len(neg_scores) > 0:
        plt.hist(neg_scores, bins=40, alpha=0.55, density=True, label="Negative", color="tab:orange")
    plt.axvline(x=threshold_used, linestyle="--", linewidth=1.8, color="tab:red", label=f"threshold={threshold_used:.3f}")
    plt.yscale("log")
    plt.xlabel("Model score")
    plt.ylabel("Density (log)")
    plt.title("Score Distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = output_dir / "eval_score_distribution.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    artifacts.append(str(p))

    # 6) Calibration / reliability diagram
    cal_curve = calibration["calibration_curve"]
    prob_pred = np.asarray(cal_curve.get("prob_pred", []), dtype=float)
    prob_true = np.asarray(cal_curve.get("prob_true", []), dtype=float)
    counts = np.asarray(cal_curve.get("counts", []), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(prob_pred, prob_true, "o-", linewidth=2, label="Model")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction positives")
    axes[0].set_title("Reliability Diagram")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if counts.size > 0:
        centers = np.linspace(0.05, 0.95, len(counts))
        axes[1].bar(centers, counts, width=0.08, color="tab:gray", alpha=0.8)
    axes[1].set_xlabel("Probability bin")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    p = output_dir / "eval_calibration.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    artifacts.append(str(p))

    # 7) Threshold sweep (precision/recall/f1)
    points = threshold_sweep.get("points", [])
    if points:
        th = np.array([pnt["threshold"] for pnt in points], dtype=float)
        prs = np.array([pnt["precision"] for pnt in points], dtype=float)
        rcs = np.array([pnt["recall"] for pnt in points], dtype=float)
        f1s = np.array([pnt["f1"] for pnt in points], dtype=float)
        fig = plt.figure(figsize=(8, 6))
        plt.plot(th, prs, label="Precision", linewidth=2)
        plt.plot(th, rcs, label="Recall", linewidth=2)
        plt.plot(th, f1s, label="F1", linewidth=2)
        plt.xlabel("Threshold")
        plt.ylabel("Metric value")
        plt.title("Threshold Sweep")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = output_dir / "eval_threshold_sweep.png"
        plt.savefig(p, dpi=160)
        plt.close(fig)
        artifacts.append(str(p))

        # 8) FAH-Recall operating curve
        fahs = np.array([pnt["ambient_false_positives_per_hour"] for pnt in points], dtype=float)
        fig = plt.figure(figsize=(8, 6))
        plt.plot(fahs, rcs, linewidth=2, label="Operating curve")
        plt.xlabel("False Activations per Hour")
        plt.ylabel("Recall")
        plt.title("FAH vs Recall")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = output_dir / "eval_fah_recall_curve.png"
        plt.savefig(p, dpi=160)
        plt.close(fig)
        artifacts.append(str(p))

        # 9) Cost curve
        costs = np.array([pnt["expected_cost"] for pnt in points], dtype=float)
        fig = plt.figure(figsize=(8, 6))
        plt.plot(th, costs, linewidth=2)
        plt.xlabel("Threshold")
        plt.ylabel("Expected cost")
        plt.title("Cost-Aware Threshold Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = output_dir / "eval_cost_curve.png"
        plt.savefig(p, dpi=160)
        plt.close(fig)
        artifacts.append(str(p))

    return artifacts


def _metric_status_line(name: str, value: float | None, good: float | None = None, bad: float | None = None, higher_is_better: bool = True) -> str:
    if value is None:
        return f"- {name}: N/A"
    icon = "🟢"
    if good is not None and bad is not None:
        if higher_is_better:
            if value < bad:
                icon = "🔴"
            elif value < good:
                icon = "🟡"
        else:
            if value > bad:
                icon = "🔴"
            elif value > good:
                icon = "🟡"
    return f"- {icon} **{name}**: {value:.4f}"


def _compose_executive_summary(report: dict[str, Any]) -> dict[str, str]:
    model_type = str(report.get("model_type", "unknown"))
    split = str(report.get("split", "unknown"))
    n_samples = int(report.get("n_samples", 0))
    threshold = float(report.get("threshold_used", 0.5))
    threshold_source = str(report.get("threshold_source", "unknown"))
    precision = float(report.get("precision", 0.0))
    recall = float(report.get("recall", 0.0))
    f1 = float(report.get("f1_score", 0.0))
    auc_roc = float(report.get("auc_roc", 0.0))
    auc_pr_raw = report.get("auc_pr")
    auc_pr = float(auc_pr_raw) if auc_pr_raw is not None else None
    fah = float(report.get("ambient_false_positives_per_hour", 0.0))
    adv = report.get("advanced_metrics", {})
    specificity = float(adv.get("specificity", 0.0))
    ece = float(report.get("calibration", {}).get("ece", 0.0))
    brier = float(report.get("calibration", {}).get("brier_score", 0.0))

    if recall >= 0.9 and fah <= 0.5 and precision >= 0.9:
        readiness = "Production-ready profile for many wake-word deployments (subject to real-world validation)."
    elif recall >= 0.85 and fah <= 1.0:
        readiness = "Near-production profile; likely needs threshold tuning on representative ambient data."
    else:
        readiness = "Not production-ready yet; improve training data quality/coverage and retune threshold."

    md_lines = [
        "# Executive Evaluation Report",
        "",
        "## Summary",
        f"- Model Type: **{model_type}**",
        f"- Split: **{split}**",
        f"- Samples: **{n_samples}**",
        f"- Decision Threshold: **{threshold:.4f}** ({threshold_source})",
        f"- Overall Assessment: **{readiness}**",
        "",
        "## Key KPIs",
        _metric_status_line("Recall", recall, good=0.90, bad=0.80, higher_is_better=True),
        _metric_status_line("Precision", precision, good=0.90, bad=0.80, higher_is_better=True),
        _metric_status_line("F1 Score", f1, good=0.90, bad=0.80, higher_is_better=True),
        _metric_status_line("FAH", fah, good=0.5, bad=2.0, higher_is_better=False),
        _metric_status_line("AUC-ROC", auc_roc, good=0.98, bad=0.90, higher_is_better=True),
        _metric_status_line("AUC-PR", auc_pr, good=0.95, bad=0.80, higher_is_better=True),
        _metric_status_line("Specificity", specificity, good=0.97, bad=0.90, higher_is_better=True),
        _metric_status_line("Calibration ECE", ece, good=0.05, bad=0.12, higher_is_better=False),
        _metric_status_line("Brier Score", brier, good=0.08, bad=0.20, higher_is_better=False),
        "",
        "## Confusion Matrix",
        f"- TN: **{report.get('confusion_matrix', {}).get('tn', 0)}**",
        f"- FP: **{report.get('confusion_matrix', {}).get('fp', 0)}**",
        f"- FN: **{report.get('confusion_matrix', {}).get('fn', 0)}**",
        f"- TP: **{report.get('confusion_matrix', {}).get('tp', 0)}**",
        "",
        "## Operating Point Guidance",
    ]

    op = report.get("operating_points", {})
    target_fah = op.get("target_fah", {})
    target_recall = op.get("target_recall", {})
    if target_fah:
        md_lines.append(f"- At target FAH={float(target_fah.get('target', 0.0)):.2f}: recall={float(target_fah.get('recall', 0.0)):.4f}, threshold={float(target_fah.get('threshold', 0.0)):.4f}")
    if target_recall:
        md_lines.append(f"- At target Recall={float(target_recall.get('target', 0.0)):.2f}: FAH={float(target_recall.get('fah', 0.0)):.4f}, threshold={float(target_recall.get('threshold', 0.0)):.4f}")
    best_cost = op.get("best_by_expected_cost", {})
    if best_cost:
        md_lines.append(
            f"- Cost-optimal threshold (fp_cost={float(best_cost.get('fp_cost', 0.0)):.1f}, fn_cost={float(best_cost.get('fn_cost', 0.0)):.1f}) = {float(best_cost.get('threshold', 0.0)):.4f}"
        )

    md_lines.extend(
        [
            "",
            "## Recommended Next Actions",
            "1. Validate this threshold on a chronologically separate ambient test set.",
            "2. Review false positives near threshold from high-score negatives/hard-negatives.",
            "3. If FAH remains high, enrich hard negatives and re-run autotune.",
            "4. If recall is low, expand positive speaker/environment diversity.",
            "",
            "## Artifacts",
            f"- JSON: `{report.get('artifacts', {}).get('directory', '')}/evaluation_report.json`",
            "- Images: confusion matrix, ROC, PR, DET, score distribution, calibration, threshold/cost curves",
        ]
    )

    model_type_h = html.escape(model_type, quote=True)
    split_h = html.escape(split, quote=True)
    threshold_source_h = html.escape(threshold_source, quote=True)

    op_lines: list[str] = []
    if target_fah:
        r = target_fah.get("recall")
        t = target_fah.get("threshold")
        r_txt = f"{float(r):.4f}" if isinstance(r, (int, float)) and math.isfinite(float(r)) else "N/A"
        t_txt = f"{float(t):.4f}" if isinstance(t, (int, float)) and math.isfinite(float(t)) else "N/A"
        op_lines.append(f"<li>Recall at target FAH: {r_txt} (threshold={t_txt})</li>")
    if target_recall:
        f = target_recall.get("fah")
        t = target_recall.get("threshold")
        f_txt = f"{float(f):.4f}" if isinstance(f, (int, float)) and math.isfinite(float(f)) else "N/A"
        t_txt = f"{float(t):.4f}" if isinstance(t, (int, float)) and math.isfinite(float(t)) else "N/A"
        op_lines.append(f"<li>FAH at target Recall: {f_txt} (threshold={t_txt})</li>")
    bc_t = best_cost.get("threshold")
    bc_txt = f"{float(bc_t):.4f}" if isinstance(bc_t, (int, float)) and math.isfinite(float(bc_t)) else "N/A"
    op_lines.append(f"<li>Cost-optimal threshold: {bc_txt}</li>")

    image_blocks: list[str] = []
    for img in report.get("artifacts", {}).get("images", []):
        name = Path(str(img)).name
        safe_src = quote(name)
        safe_cap = html.escape(name, quote=True)
        image_blocks.append(f'<div><img src="{safe_src}" alt="{safe_cap}"><div>{safe_cap}</div></div>')

    html_out = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Executive Evaluation Report</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b1220; color: #e8eefc; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin-bottom: 6px; font-size: 32px; }}
    .sub {{ color: #9bb0d3; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ background: #121c2f; border: 1px solid #1f2d4a; border-radius: 12px; padding: 14px; }}
    .k {{ color: #8fa7d2; font-size: 12px; text-transform: uppercase; letter-spacing: .07em; }}
    .v {{ font-size: 30px; font-weight: 700; margin-top: 6px; }}
    .status {{ margin: 16px 0; padding: 14px; border-radius: 10px; background: #13233e; border-left: 5px solid #58a6ff; }}
    .section {{ margin-top: 26px; }}
    table {{ width: 100%; border-collapse: collapse; background: #121c2f; border-radius: 10px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #213154; text-align: left; }}
    th {{ color: #a7bee7; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    tr:last-child td {{ border-bottom: none; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .images img {{ width: 100%; border-radius: 10px; border: 1px solid #1f2d4a; background: white; }}
    a {{ color: #7cc6ff; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Executive Evaluation Report</h1>
    <p class=\"sub\">Model: {model_type_h} · Split: {split_h} · Samples: {n_samples}</p>

    <div class=\"status\"><strong>Overall Assessment:</strong> {readiness}</div>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"k\">Recall</div><div class=\"v\">{recall:.4f}</div></div>
      <div class=\"card\"><div class=\"k\">Precision</div><div class=\"v\">{precision:.4f}</div></div>
      <div class=\"card\"><div class=\"k\">F1 Score</div><div class=\"v\">{f1:.4f}</div></div>
      <div class=\"card\"><div class=\"k\">FAH</div><div class=\"v\">{fah:.4f}</div></div>
      <div class=\"card\"><div class=\"k\">AUC-ROC</div><div class=\"v\">{auc_roc:.4f}</div></div>
      <div class=\"card\"><div class=\"k\">AUC-PR</div><div class=\"v\">{(f"{auc_pr:.4f}" if auc_pr is not None else "N/A")}</div></div>
      <div class=\"card\"><div class=\"k\">Specificity</div><div class=\"v\">{specificity:.4f}</div></div>
      <div class=\"card\"><div class=\"k\">ECE</div><div class=\"v\">{ece:.4f}</div></div>
    </div>

    <div class=\"section\">
      <h2>Confusion Matrix</h2>
      <table>
        <thead><tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr></thead>
        <tbody>
          <tr><td>True Neg</td><td>{int(report.get("confusion_matrix", {}).get("tn", 0))}</td><td>{int(report.get("confusion_matrix", {}).get("fp", 0))}</td></tr>
          <tr><td>True Pos</td><td>{int(report.get("confusion_matrix", {}).get("fn", 0))}</td><td>{int(report.get("confusion_matrix", {}).get("tp", 0))}</td></tr>
        </tbody>
      </table>
      <p>Threshold: <strong>{threshold:.4f}</strong> ({threshold_source_h})</p>
    </div>

    <div class=\"section\">
      <h2>Operating Point Guidance</h2>
      <ul>
        {"".join(op_lines)}
      </ul>
    </div>

    <div class=\"section\">
      <h2>Generated Plots</h2>
      <div class=\"images\">
        {"".join(image_blocks)}
      </div>
    </div>

    <div class=\"section\">
      <h2>Recommendations</h2>
      <ol>
        <li>Validate selected threshold on chronologically separate ambient recordings.</li>
        <li>Inspect top-scored negatives to discover acoustic confusers and augment hard negatives.</li>
        <li>Re-run autotune if FAH/recall trade-off remains outside target.</li>
      </ol>
    </div>
  </div>
</body>
</html>"""

    return {"markdown": "\n".join(md_lines).strip() + "\n", "html": html_out}


def _write_executive_reports(report: dict[str, Any], artifact_dir: Path) -> dict[str, str]:
    composed = _compose_executive_summary(report)
    md_path = artifact_dir / "executive_report.md"
    html_path = artifact_dir / "executive_report.html"
    md_path.write_text(composed["markdown"], encoding="utf-8")
    html_path.write_text(composed["html"], encoding="utf-8")
    return {
        "markdown": str(md_path),
        "html": str(html_path),
    }


def _compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    config: dict,
    split: str,
    model_path: str,
    model_type: str,
    output_dir: str | None,
    n_thresholds: int,
    bootstrap_iterations: int,
    fp_cost: float,
    fn_cost: float,
    save_plots: bool,
    threshold: float = 0.5,
    threshold_source: str = "unknown",
    console: Console | None = None,
) -> dict:
    """Compute exhaustive metrics, save artifacts, and display summary."""
    console = console or Console()
    ambient_hours = float(config.get("training", {}).get("ambient_duration_hours", 10.0) or 10.0)
    eval_cfg = config.get("evaluation", {})
    target_fah = float(eval_cfg.get("target_fah", 0.5) or 0.5)
    target_recall = float(eval_cfg.get("target_recall", 0.9) or 0.9)

    calc = MetricsCalculator(y_true=y_true, y_score=y_scores)
    basic = calc.compute_all_metrics(ambient_duration_hours=ambient_hours, threshold=threshold)

    y_pred = (y_scores >= threshold).astype(int)
    cm = _compute_confusion(y_true, y_pred)
    tp = cm["tp"]
    fp = cm["fp"]
    tn = cm["tn"]
    fn = cm["fn"]

    prevalence = _safe_div(int(np.sum(y_true == 1)), len(y_true))
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    balanced_accuracy = 0.5 * (float(basic.get("recall", 0.0)) + specificity)

    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = _safe_div((tp * tn) - (fp * fn), mcc_den)

    po = _safe_div(tp + tn, len(y_true))
    pe = _safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), len(y_true) * len(y_true))
    cohens_kappa = _safe_div(po - pe, 1.0 - pe) if pe < 1.0 else 1.0

    informedness = float(basic.get("recall", 0.0)) + specificity - 1.0
    markedness = float(basic.get("precision", 0.0)) + npv - 1.0
    lr_pos = _safe_div(float(basic.get("recall", 0.0)), fpr)
    lr_neg = _safe_div(fnr, specificity)
    dor = _safe_div(lr_pos, lr_neg)

    curves = calc.compute_roc_pr_curves(n_thresholds=max(401, n_thresholds))
    fpr_curve = np.asarray(curves.get("fpr", []), dtype=float)
    tpr_curve = np.asarray(curves.get("tpr", []), dtype=float)
    thr_curve = np.asarray(curves.get("thresholds", []), dtype=float)

    if fpr_curve.size > 0 and tpr_curve.size > 0:
        fnr_curve = 1.0 - tpr_curve
        diff = np.abs(fpr_curve - fnr_curve)
        eer_idx = int(np.argmin(diff))
        eer = float(fpr_curve[eer_idx])
        eer_threshold = float(thr_curve[eer_idx]) if thr_curve.size > eer_idx else float(threshold)
        ks_stat = float(np.max(np.abs(tpr_curve - fpr_curve)))
    else:
        eer = 0.0
        eer_threshold = float(threshold)
        ks_stat = 0.0

    cal_curve = compute_calibration_curve(y_true, y_scores, n_bins=12)
    brier = float(compute_brier_score(y_true, y_scores))
    cal_err = _compute_ece(cal_curve)

    thresholds = np.linspace(0.0, 1.0, max(51, n_thresholds))
    sweep = _threshold_sweep(
        y_true=y_true,
        y_scores=y_scores,
        thresholds=thresholds,
        ambient_hours=ambient_hours,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )

    rec_target_fah, thr_target_fah, achieved_fah = calc.compute_recall_at_target_fah(
        ambient_duration_hours=ambient_hours,
        target_fah=target_fah,
        n_thresholds=max(101, n_thresholds),
    )
    fah_target_recall, thr_target_recall, achieved_recall = calc.compute_fah_at_target_recall(
        ambient_duration_hours=ambient_hours,
        target_recall=target_recall,
        n_thresholds=max(101, n_thresholds),
    )

    cis = _bootstrap_confidence_intervals(
        y_true=y_true,
        y_scores=y_scores,
        threshold=threshold,
        ambient_hours=ambient_hours,
        iterations=bootstrap_iterations,
    )

    output_root = Path(output_dir) if output_dir else Path(config.get("performance", {}).get("tensorboard_log_dir", "./logs"))
    artifact_dir = output_root / "evaluation_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []
    if save_plots:
        artifacts = _plot_and_save_all(
            y_true=y_true,
            y_scores=y_scores,
            confusion=cm,
            calibration={"calibration_curve": cal_curve, "brier_score": brier, "ece": cal_err["ece"], "mce": cal_err["mce"]},
            threshold_sweep=sweep,
            output_dir=artifact_dir,
            threshold_used=threshold,
        )

    top_k = [0.01, 0.05, 0.10]
    ranked_idx = np.argsort(-y_scores)
    ranked_true = y_true[ranked_idx]
    lift_analysis: dict[str, Any] = {}
    n_pos_total = int(np.sum(y_true == 1))
    for frac in top_k:
        k = max(1, int(len(y_true) * frac))
        captured = int(np.sum(ranked_true[:k]))
        capture_rate = _safe_div(captured, n_pos_total)
        precision_at_k = _safe_div(captured, k)
        lift = _safe_div(precision_at_k, prevalence)
        lift_analysis[f"top_{int(frac * 100)}pct"] = {
            "k": int(k),
            "captured_positives": captured,
            "capture_rate": float(capture_rate),
            "precision": float(precision_at_k),
            "lift": float(lift),
        }

    report = {
        # Keep top-level compatibility for pipeline quality gate
        **basic,
        "threshold_used": float(threshold),
        "threshold_source": threshold_source,
        "model_path": model_path,
        "model_type": model_type,
        "split": split,
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "prevalence": float(prevalence),
        "advanced_metrics": {
            "specificity": float(specificity),
            "npv": float(npv),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "balanced_accuracy": float(balanced_accuracy),
            "mcc": float(mcc),
            "cohens_kappa": float(cohens_kappa),
            "informedness": float(informedness),
            "markedness": float(markedness),
            "likelihood_ratio_positive": float(lr_pos),
            "likelihood_ratio_negative": float(lr_neg),
            "diagnostic_odds_ratio": float(dor),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "ks_statistic": float(ks_stat),
        },
        "confusion_matrix": cm,
        "calibration": {
            "brier_score": float(brier),
            "ece": float(cal_err["ece"]),
            "mce": float(cal_err["mce"]),
            "calibration_curve": {
                "prob_true": cal_curve["prob_true"],
                "prob_pred": cal_curve["prob_pred"],
                "counts": cal_curve["counts"],
                "bin_edges": cal_curve["bin_edges"],
            },
        },
        "operating_points": {
            "target_fah": {
                "target": float(target_fah),
                "recall": float(rec_target_fah),
                "threshold": float(thr_target_fah),
                "achieved_fah": float(achieved_fah),
            },
            "target_recall": {
                "target": float(target_recall),
                "fah": float(fah_target_recall),
                "threshold": float(thr_target_recall),
                "achieved_recall": float(achieved_recall),
            },
            "best_by_f1": sweep["best_by_f1"],
            "best_by_balanced_accuracy": sweep["best_by_balanced_accuracy"],
            "best_by_expected_cost": sweep["best_by_expected_cost"],
        },
        "threshold_sweep": sweep,
        "curves": {
            "roc_pr": {
                "thresholds": curves.get("thresholds", np.array([])),
                "fpr": curves.get("fpr", np.array([])),
                "tpr": curves.get("tpr", np.array([])),
                "precision": curves.get("precision", np.array([])),
                "recall": curves.get("recall", np.array([])),
            }
        },
        "lift_analysis": lift_analysis,
        "bootstrap_confidence_intervals": cis,
        "artifacts": {
            "directory": str(artifact_dir),
            "images": artifacts,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "ambient_duration_hours": float(ambient_hours),
            "n_thresholds": int(n_thresholds),
            "bootstrap_iterations": int(bootstrap_iterations),
            "fp_cost": float(fp_cost),
            "fn_cost": float(fn_cost),
            "config_eval_target_fah": float(target_fah),
            "config_eval_target_recall": float(target_recall),
        },
    }

    executive_paths = _write_executive_reports(report, artifact_dir)
    report["artifacts"]["executive_report_markdown"] = executive_paths["markdown"]
    report["artifacts"]["executive_report_html"] = executive_paths["html"]

    report_path = artifact_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(_to_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Samples", f"{len(y_true)}")
    table.add_row("Threshold", f"{threshold:.4f} ({threshold_source})")
    table.add_row("Accuracy", f"{float(report.get('accuracy', 0.0)):.4f}")
    table.add_row("Precision", f"{float(report.get('precision', 0.0)):.4f}")
    table.add_row("Recall", f"{float(report.get('recall', 0.0)):.4f}")
    table.add_row("F1", f"{float(report.get('f1_score', 0.0)):.4f}")
    table.add_row("AUC-ROC", f"{float(report.get('auc_roc', 0.0)):.4f}")
    auc_pr = report.get("auc_pr")
    table.add_row("AUC-PR", f"{float(auc_pr):.4f}" if auc_pr is not None else "N/A")
    table.add_row("FA/Hour", f"{float(report.get('ambient_false_positives_per_hour', 0.0)):.4f}")
    table.add_row("Specificity", f"{specificity:.4f}")
    table.add_row("MCC", f"{mcc:.4f}")
    table.add_row("EER", f"{eer:.4f}")
    table.add_row("ECE", f"{float(cal_err['ece']):.4f}")
    console.print(table)

    console.print(f"[green]Saved evaluation report:[/] {report_path}")
    console.print(f"[green]Saved executive report (Markdown):[/] {executive_paths['markdown']}")
    console.print(f"[green]Saved executive report (HTML):[/] {executive_paths['html']}")
    if artifacts:
        console.print(f"[green]Saved {len(artifacts)} plot artifacts under:[/] {artifact_dir}")

    return report


def analyze_model_quality(metrics: dict) -> dict:
    """Analyze model quality and provide warnings.

    Args:
        metrics: Evaluation metrics

    Returns:
        Analysis results with warnings
    """
    warnings = []
    concerns = []

    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    # f1 = metrics.get("f1_score", 0)  # noqa: F841  # Available if needed
    fah = metrics.get("ambient_false_positives_per_hour", float("inf"))

    # Check for "too good to be true" metrics
    if accuracy > 0.999 and precision > 0.999 and recall > 0.999:
        warnings.append("⚠️  Metrics are suspiciously perfect (>99.9%)")
        warnings.append("   Possible causes:")
        warnings.append("   - Data leakage (train/validation overlap)")
        warnings.append("   - Overfitting (validation set too small)")
        warnings.append("   - Duplicate samples in train and validation")

    if fah == 0.0:
        warnings.append("⚠️  FAH is exactly 0.0 - statistically unlikely")
        warnings.append("   Verify negative samples are truly negative")

    # Check for common issues
    if recall < 0.8:
        concerns.append(f"❌ Low recall ({recall:.2f}) - model misses too many wake words")

    if fah > 2.0:
        concerns.append(f"❌ High FAH ({fah:.2f}) - too many false activations")

    if precision < 0.9:
        concerns.append(f"❌ Low precision ({precision:.2f}) - many false positives")

    return {
        "warnings": warnings,
        "concerns": concerns,
        "is_suspicious": len(warnings) > 0,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate wake word model with advanced diagnostics")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model (.weights.h5 or .tflite). Alias for --checkpoint/--tflite",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.weights.h5 file)",
    )
    parser.add_argument(
        "--tflite",
        type=str,
        help="Path to TFLite model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name or path",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Optional override YAML path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON report and plot artifacts",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=None,
        help="Number of thresholds for sweep-based analyses",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=400,
        help="Bootstrap iterations for confidence intervals (default: 400)",
    )
    parser.add_argument(
        "--fp-cost",
        type=float,
        default=20.0,
        help="Cost weight for false positives in cost-aware thresholding",
    )
    parser.add_argument(
        "--fn-cost",
        type=float,
        default=1.0,
        help="Cost weight for false negatives in cost-aware thresholding",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation (JSON-only evaluation)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze model quality and show warnings",
    )

    args = parser.parse_args()

    model_path_from_alias = args.model
    checkpoint = args.checkpoint
    tflite = args.tflite

    if model_path_from_alias and not checkpoint and not tflite:
        if model_path_from_alias.endswith(".tflite"):
            tflite = model_path_from_alias
        else:
            checkpoint = model_path_from_alias

    if not checkpoint and not tflite:
        print("Error: Must specify --model, --checkpoint, or --tflite")
        return 1

    console = Console()

    # Load config
    try:
        from config.loader import load_full_config

        config = load_full_config(args.config, args.override)
        config_dict = dataclasses.asdict(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/]")
        return 1

    # Evaluate
    model_path = checkpoint or tflite
    try:
        metrics = evaluate_model(
            model_path=model_path,
            config=config_dict,
            split=args.split,
            output_dir=args.output_dir,
            n_thresholds=args.n_thresholds,
            bootstrap_iterations=args.bootstrap_iterations,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            save_plots=not args.no_plots,
            console=console,
        )

        # Analyze if requested
        if args.analyze:
            analysis = analyze_model_quality(metrics)

            if analysis["warnings"]:
                console.print("\n[bold yellow]Quality Analysis Warnings:[/]")
                for warning in analysis["warnings"]:
                    console.print(f"[yellow]{warning}[/]")

            if analysis["concerns"]:
                console.print("\n[bold red]Quality Concerns:[/]")
                for concern in analysis["concerns"]:
                    console.print(f"[red]{concern}[/]")

            if not analysis["warnings"] and not analysis["concerns"]:
                console.print("\n[green]✓ Model quality looks good[/]")

        # Output JSON if requested
        if args.json:
            print(json.dumps(_to_json_safe(metrics), indent=2, allow_nan=False))

        return 0

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/]")
        import traceback

        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
