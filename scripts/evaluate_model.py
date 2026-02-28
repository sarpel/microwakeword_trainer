"""Post-training analysis and evaluation script for wake word models.

Provides comprehensive evaluation on test datasets and model analysis.

Usage:
    python scripts/evaluate_model.py --checkpoint models/checkpoints/best.ckpt --config standard
    python scripts/evaluate_model.py --tflite models/exported/wake_word.tflite --test-dir ./dataset/test
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table


def evaluate_model(
    model_path: str,
    config: dict,
    split: str = "test",
    console: Console | None = None,
) -> dict:
    """Evaluate model on specified dataset split.

    Args:
        model_path: Path to checkpoint or TFLite model
        config: Training configuration
        split: Dataset split to evaluate ('train', 'val', 'test')
        console: Rich console for output

    Returns:
        Evaluation metrics dictionary
    """
    console = console or Console()

    # Load model
    if model_path.endswith(".tflite"):
        metrics = _evaluate_tflite(model_path, config, split, console)
    else:
        metrics = _evaluate_checkpoint(model_path, config, split, console)

    return metrics


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
        model_config=config.get("model", {}),
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
    console: Console,
) -> dict:
    """Evaluate Keras checkpoint."""
    from src.data.dataset import WakeWordDataset

    # Build and load model
    model, input_shape = _build_and_load_model(checkpoint_path, config)

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
    for batch_features, batch_labels in gen:
        predictions = model.predict(batch_features, verbose=0)
        y_true.extend(batch_labels.flatten().tolist())
        y_scores.extend(predictions.flatten().tolist())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Calculate metrics
    return _compute_metrics(y_true, y_scores, config, console)


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
    console: Console,
) -> dict:
    """Evaluate TFLite model."""
    from src.data.dataset import WakeWordDataset

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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

    gen = gen_factory()
    for batch_features, batch_labels in gen:
        # Process each sample in batch
        for i in range(len(batch_features)):
            features = batch_features[i : i + 1]
            label = batch_labels[i]

            # Set input
            interpreter.set_tensor(input_details[0]["index"], features)
            interpreter.invoke()

            # Get output
            prediction = interpreter.get_tensor(output_details[0]["index"])

            y_true.append(label)
            y_scores.append(prediction.flatten()[0])

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    return _compute_metrics(y_true, y_scores, config, console)


def _compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    config: dict,
    console: Console,
) -> dict:
    """Compute and display metrics."""
    from src.evaluation.metrics import MetricsCalculator

    calc = MetricsCalculator(y_true=y_true, y_score=y_scores)

    ambient_hours = config.get("training", {}).get("ambient_duration_hours", 10.0)

    metrics = calc.compute_all_metrics(
        ambient_duration_hours=ambient_hours,
        threshold=0.5,
    )

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    table.add_row("Precision", f"{metrics.get('precision', 0):.4f}")
    table.add_row("Recall", f"{metrics.get('recall', 0):.4f}")
    table.add_row("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    table.add_row("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")
    table.add_row("AUC-PR", f"{metrics.get('auc_pr', 0):.4f}")
    table.add_row("FA/Hour", f"{metrics.get('ambient_false_positives_per_hour', 0):.2f}")

    console.print(table)

    return metrics


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
    parser = argparse.ArgumentParser(description="Evaluate wake word model on test dataset")
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
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)",
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

    if not args.checkpoint and not args.tflite:
        print("Error: Must specify either --checkpoint or --tflite")
        return 1

    console = Console()

    # Load config
    try:
        import dataclasses

        from config.loader import load_full_config

        config = load_full_config(args.config)
        config_dict = dataclasses.asdict(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/]")
        return 1

    # Evaluate
    model_path = args.checkpoint or args.tflite
    try:
        metrics = evaluate_model(model_path, config_dict, args.split, console)

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
            print(json.dumps(metrics, indent=2))

        return 0

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/]")
        import traceback

        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
