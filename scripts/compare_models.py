#!/usr/bin/env python3
"""Compare two wake word models side-by-side on the same test dataset.

Accepts .weights.h5 Keras checkpoints or .tflite models.
Prints and saves a side-by-side comparison of FAH, recall, precision, and
operating points.

Usage:
    python scripts/compare_models.py model_a.tflite model_b.tflite --config standard
    python scripts/compare_models.py ckpt_a.weights.h5 ckpt_b.weights.h5 --config standard
    python scripts/compare_models.py model_a.tflite ckpt_b.weights.h5 --config standard --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_keras_model(checkpoint_path: str, config: dict):
    """Load a Keras model from a .weights.h5 checkpoint."""
    import tensorflow as tf

    from src.model.architecture import build_model

    hardware = config.get("hardware", {})
    clip_duration_ms = hardware.get("clip_duration_ms", 1000)
    window_step_ms = hardware.get("window_step_ms", 10)
    mel_bins = hardware.get("mel_bins", 40)
    input_shape = (int(clip_duration_ms / window_step_ms), mel_bins)

    model_cfg = config.get("model", {})
    model = build_model(
        input_shape=input_shape,
        num_classes=2,
        first_conv_filters=model_cfg.get("first_conv_filters", 32),
        first_conv_kernel_size=model_cfg.get("first_conv_kernel_size", 5),
        stride=model_cfg.get("stride", 3),
        pointwise_filters=model_cfg.get("pointwise_filters", "64,64,64,64"),
        mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
        repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
        residual_connection=model_cfg.get("residual_connection", "0,1,1,1"),
        dropout_rate=model_cfg.get("dropout_rate", 0.08),
        l2_regularization=model_cfg.get("l2_regularization", 0.00003),
    )
    _ = model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)

    # Try Keras 3 checkpoint loader first
    try:
        from src.export.tflite import load_weights_from_keras3_checkpoint

        n = load_weights_from_keras3_checkpoint(model, checkpoint_path)
        console.print(f"  Loaded {n} weights from {checkpoint_path}")
    except Exception:
        model.load_weights(checkpoint_path)
        console.print(f"  Loaded weights (fallback) from {checkpoint_path}")

    return model, "keras"


def _load_tflite_model(tflite_path: str):
    """Load a TFLite model and return model bytes for fresh-interpreter-per-sample inference.

    TFLite resource variables (streaming state) cannot be reliably reset via
    reset_all_variables() or set_tensor() — those only affect ReadVariableOp read copies,
    not the underlying resource variable buffers.  Creating a fresh interpreter per sample
    from in-memory bytes is the ONLY reliable method (~0.6ms overhead, negligible).
    """
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    return model_bytes, "tflite"


def _infer_keras(model, batch: np.ndarray) -> np.ndarray:
    """Run inference with a Keras model."""
    preds = model(batch.astype(np.float32), training=False)
    return np.array(preds).ravel()


def _infer_tflite(model_bytes: bytes, batch: np.ndarray, config: dict) -> np.ndarray:
    """Run streaming inference with a TFLite model using fresh interpreter per sample.

    The exported TFLite model is a streaming model with input shape [1, stride, 40].
    Full spectrograms must be fed in stride-sized chunks.  A fresh interpreter is created
    per sample because TFLite resource variables cannot be reliably reset via
    reset_all_variables() or set_tensor() on ReadVariableOp tensors.
    """
    import tensorflow as tf

    # Extract quantization params from a template interpreter (once per batch)
    _tmpl = tf.lite.Interpreter(model_content=model_bytes)
    _tmpl.allocate_tensors()

    input_details = _tmpl.get_input_details()
    output_details = _tmpl.get_output_details()

    in_dtype = input_details[0]["dtype"]
    out_dtype = output_details[0]["dtype"]

    # Quantization parameters
    in_quant = input_details[0].get("quantization_parameters", {})
    in_scale = in_quant.get("scales", np.array([1.0]))[0]
    in_zero_point = in_quant.get("zero_points", np.array([0]))[0]

    out_quant = output_details[0].get("quantization_parameters", {})
    out_scale = out_quant.get("scales", np.array([1.0]))[0]
    out_zero_point = out_quant.get("zero_points", np.array([0]))[0]

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    del _tmpl

    # Model config for stride and mel_bins
    model_cfg = config.get("model", {})
    hardware_cfg = config.get("hardware", {})
    stride = model_cfg.get("stride", 3)
    mel_bins = hardware_cfg.get("mel_bins", 40)

    preds = []
    for sample in batch:
        num_frames = sample.shape[0]

        # Fresh interpreter per sample — the ONLY reliable way to reset streaming state
        interpreter = tf.lite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()

        # Streaming inference: slide stride-sized chunks across spectrogram
        prediction = 0.0
        for t in range(0, num_frames - stride + 1, stride):
            chunk = sample[t : t + stride]  # [stride, mel_bins]
            chunk = chunk.reshape(1, stride, mel_bins)  # [1, stride, mel_bins]

            # Quantize float32 -> int8 using proper scale/zero_point
            if in_dtype != np.float32:
                chunk = np.clip(chunk / in_scale + in_zero_point, -128, 127).astype(in_dtype)

            interpreter.set_tensor(input_index, chunk)
            interpreter.invoke()

            raw_out = interpreter.get_tensor(output_index)
            # Dequantize output to float [0, 1]
            if out_dtype != np.float32:
                prediction = (raw_out.astype(np.float32) - out_zero_point) * out_scale
                prediction = prediction.flatten()[0]
            else:
                prediction = raw_out.flatten()[0]

        preds.append(float(prediction))

    return np.array(preds)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_model(model_path: str, config: dict, data_factory) -> dict:
    """Evaluate a single model on data from data_factory.

    Returns a dict with:
        y_true, y_scores, metrics (from MetricsCalculator)
    """
    from src.evaluation.metrics import MetricsCalculator

    path = Path(model_path)
    if not path.exists():
        console.print(f"[red]✗ Model not found: {model_path}[/]")
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5") or "weights" in path.name:
        model, kind = _load_keras_model(str(path), config)

        def infer_fn(batch):
            return _infer_keras(model, batch)

    elif suffix == ".tflite":
        model_bytes, kind = _load_tflite_model(str(path))

        def infer_fn(batch):
            return _infer_tflite(model_bytes, batch, config)

    else:
        console.print(f"[red]✗ Unsupported model format: {suffix}[/]")
        sys.exit(1)

    console.print(f"  Loaded [{kind}] model: {path.name}")

    all_y_true = []
    all_y_scores = []

    for batch_data in data_factory():
        if len(batch_data) == 4:
            features, labels, _, _ = batch_data
        elif len(batch_data) == 3:
            features, labels, _ = batch_data
        else:
            features, labels = batch_data[0], batch_data[1]

        if hasattr(features, "numpy"):
            features = features.numpy()
        if hasattr(labels, "numpy"):
            labels = labels.numpy()

        scores = infer_fn(np.array(features))
        all_y_true.extend(np.ravel(labels).tolist())
        all_y_scores.extend(scores.tolist())

    y_true = np.array(all_y_true)
    y_scores = np.array(all_y_scores)

    ambient_hours = float(config.get("training", {}).get("ambient_duration_hours", 10.0))
    test_split = float(config.get("training", {}).get("test_split", 0.1))
    scaled_hours = ambient_hours * test_split

    calc = MetricsCalculator(y_true=y_true, y_score=y_scores)
    metrics = calc.compute_all_metrics(ambient_duration_hours=scaled_hours, threshold=0.5)

    return {
        "y_true": y_true,
        "y_scores": y_scores,
        "metrics": metrics,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_DISPLAY_METRICS = [
    ("ambient_false_positives_per_hour", "FAH", ".3f"),
    ("recall", "Recall", ".4f"),
    ("precision", "Precision", ".4f"),
    ("f1_score", "F1", ".4f"),
    ("auc_roc", "AUC-ROC", ".4f"),
    ("auc_pr", "AUC-PR", ".4f"),
    ("recall_at_no_faph", "Recall@0FAH", ".4f"),
    ("average_viable_recall", "AvgViableRecall", ".4f"),
    ("recall_at_target_fah", "Recall@TargetFAH", ".4f"),
]


def _delta_str(a_val, b_val, fmt: str, lower_is_better: bool = False) -> str:
    """Compute delta string A-B, coloring green when A is better."""
    try:
        delta = float(a_val) - float(b_val)
    except (TypeError, ValueError):
        return "N/A"
    sign = "+" if delta >= 0 else ""
    delta_str = f"{sign}{delta:{fmt}}"
    # Green if A improved (lower is better → negative delta is good)
    if lower_is_better:
        color = "green" if delta < 0 else "red" if delta > 0 else "white"
    else:
        color = "green" if delta > 0 else "red" if delta < 0 else "white"
    return f"[{color}]{delta_str}[/{color}]"


def _build_comparison_table(name_a: str, name_b: str, metrics_a: dict, metrics_b: dict) -> Table:
    table = Table(title="Model Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold cyan", width=22)
    table.add_column(f"A: {Path(name_a).stem}", justify="right", width=18)
    table.add_column(f"B: {Path(name_b).stem}", justify="right", width=18)
    table.add_column("Δ (A-B)", justify="right", width=14)

    for key, label, fmt in _DISPLAY_METRICS:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        str_a = f"{va:{fmt}}" if va is not None else "N/A"
        str_b = f"{vb:{fmt}}" if vb is not None else "N/A"
        lower_better = key == "ambient_false_positives_per_hour"
        delta = _delta_str(va, vb, fmt, lower_is_better=lower_better)
        table.add_row(label, str_a, str_b, delta)

    return table


def _build_operating_points_table(name: str, metrics: dict) -> "Table | None":
    """Show threshold-recall-FAH table from operating_points if available."""
    ops = metrics.get("operating_points", [])
    if not ops:
        return None

    t = Table(
        title=f"Operating Points: {Path(name).stem}",
        show_header=True,
        header_style="bold blue",
    )
    t.add_column("Target FAH", justify="right")
    t.add_column("Threshold", justify="right")
    t.add_column("Actual FAH", justify="right")
    t.add_column("Recall", justify="right")

    for op in ops:
        t.add_row(
            f"{op.get('target_fah', 'N/A')}",
            f"{op.get('threshold', 0.0):.4f}",
            f"{op.get('fah', 0.0):.3f}",
            f"{op.get('recall', 0.0):.4f}",
        )
    return t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_models",
        description="Compare two wake word models (h5 or tflite) side-by-side on the test dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_a", help="Path to model A (.weights.h5 or .tflite)")
    parser.add_argument("model_b", help="Path to model B (.weights.h5 or .tflite)")
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name or path (default: standard)",
    )
    parser.add_argument("--override", type=str, default=None, help="Override config YAML path")
    parser.add_argument("--json", action="store_true", help="Output results as JSON to stdout")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON results to this file",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    import dataclasses

    from config.loader import load_full_config

    config_dc = load_full_config(args.config, args.override)
    config = dataclasses.asdict(config_dc)

    from src.data.dataset import WakeWordDataset

    hardware_cfg = config.get("hardware", {})
    clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
    window_step_ms = hardware_cfg.get("window_step_ms", 10)
    max_time_frames = int(clip_duration_ms / window_step_ms)

    console.print(Panel.fit("[bold]Wake Word Model Comparison[/bold]", border_style="magenta"))

    dataset = WakeWordDataset(config)
    dataset.build()
    test_factory = dataset.test_generator_factory(max_time_frames=max_time_frames)

    console.print("\n[bold cyan]Evaluating Model A...[/bold cyan]")
    result_a = _evaluate_model(args.model_a, config, test_factory)

    # Rebuild factory for model B (generators are exhausted after one pass)
    test_factory = dataset.test_generator_factory(max_time_frames=max_time_frames)

    console.print("\n[bold cyan]Evaluating Model B...[/bold cyan]")
    result_b = _evaluate_model(args.model_b, config, test_factory)

    metrics_a = result_a["metrics"]
    metrics_b = result_b["metrics"]

    if args.json:
        out = {
            "model_a": {
                "path": args.model_a,
                "metrics": {k: v for k, v in metrics_a.items() if isinstance(v, (int, float, type(None)))},
            },
            "model_b": {
                "path": args.model_b,
                "metrics": {k: v for k, v in metrics_b.items() if isinstance(v, (int, float, type(None)))},
            },
            "delta": {},
        }
        for key, _, _fmt in _DISPLAY_METRICS:
            va = metrics_a.get(key)
            vb = metrics_b.get(key)
            if va is not None and vb is not None:
                out["delta"][key] = float(va) - float(vb)
        print(json.dumps(out, indent=2))
        if args.output:
            output_path = Path(args.output)
            # Create backup if output file already exists
            if output_path.exists():
                from datetime import datetime

                backup_path = output_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                output_path.rename(backup_path)
                console.print(f"[yellow]Backup created:[/] {backup_path}")
            output_path.write_text(json.dumps(out, indent=2))
        return

    # Rich comparison table
    console.print()
    console.print(_build_comparison_table(args.model_a, args.model_b, metrics_a, metrics_b))

    # Operating points
    for name, result in [(args.model_a, result_a), (args.model_b, result_b)]:
        ops_table = _build_operating_points_table(name, result["metrics"])
        if ops_table is not None:
            console.print()
            console.print(ops_table)

    # Summary verdict
    fah_a = metrics_a.get("ambient_false_positives_per_hour", float("inf"))
    fah_b = metrics_b.get("ambient_false_positives_per_hour", float("inf"))
    recall_a = metrics_a.get("recall", 0)
    recall_b = metrics_b.get("recall", 0)

    console.print()
    if fah_a < fah_b and recall_a >= recall_b:
        verdict = f"[green]Model A is better[/green] (lower FAH: {fah_a:.3f} vs {fah_b:.3f}, recall: {recall_a:.4f} vs {recall_b:.4f})"
    elif fah_b < fah_a and recall_b >= recall_a:
        verdict = f"[green]Model B is better[/green] (lower FAH: {fah_b:.3f} vs {fah_a:.3f}, recall: {recall_b:.4f} vs {recall_a:.4f})"
    else:
        verdict = f"[yellow]Trade-off[/yellow] — A: FAH={fah_a:.3f}/recall={recall_a:.4f}  B: FAH={fah_b:.3f}/recall={recall_b:.4f}"
    console.print(Panel.fit(verdict, title="Verdict"))

    if args.output:
        out = {
            "model_a": {
                "path": args.model_a,
                "metrics": {k: v for k, v in metrics_a.items() if isinstance(v, (int, float, type(None)))},
            },
            "model_b": {
                "path": args.model_b,
                "metrics": {k: v for k, v in metrics_b.items() if isinstance(v, (int, float, type(None)))},
            },
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        console.print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
