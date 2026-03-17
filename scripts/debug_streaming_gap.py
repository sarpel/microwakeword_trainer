#!/usr/bin/env python3
"""Numerical comparison: Training model (non-streaming) vs StreamingExportModel (float32).

This script diagnoses the AUC gap between the training Keras model and the
TFLite export by comparing outputs on the SAME input data in float32
(before quantization), isolating whether the gap comes from:
  1. Architecture mismatch (streaming vs non-streaming computation)
  2. BN folding numerical error
  3. INT8 quantization

Usage:
    python scripts/debug_streaming_gap.py \
        --checkpoint models/checkpoints/best_weights.weights.h5 \
        --config config/presets/max_quality.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_full_config
from src.export.tflite import StreamingExportModel, load_weights_from_keras3_checkpoint
from src.model.architecture import build_model


def parse_kernel_sizes(
    s: str | list[str] | list[list[int]] | list[int],
) -> list[list[int]]:
    """Parse mixconv_kernel_sizes string like '[5],[7,11],[9,15],[23]' or already-parsed list."""
    import ast

    if isinstance(s, str):
        return [list(x) if isinstance(x, (list, tuple)) else [x] for x in ast.literal_eval(f"[{s}]")]
    elif isinstance(s, list):
        # Handle list of ints, list of strings, or already-parsed structure
        result = []
        for item in s:
            if isinstance(item, int):
                result.append([item])
            elif isinstance(item, str):
                # Parse string like '5' or '[5,7]'
                if item.startswith("["):
                    result.append(ast.literal_eval(item))
                else:
                    result.append([int(item)])
            elif isinstance(item, list):
                result.append(item)
        return result


def parse_int_list(s: str | list[int] | list[str] | int) -> list[int]:
    """Parse comma-separated int string like '64,64,64,64' or already-parsed list."""
    if isinstance(s, str):
        return [int(x.strip()) for x in s.split(",")]
    elif isinstance(s, int):
        return [s]
    elif isinstance(s, list):
        return [int(x) for x in s]


def build_training_model(config, checkpoint_path: str):
    """Build and load the training (non-streaming) model."""
    clip_frames = int(config.hardware.clip_duration_ms / config.hardware.window_step_ms)
    mel_bins = config.hardware.mel_bins

    model = build_model(
        input_shape=(clip_frames, mel_bins),
        first_conv_filters=config.model.first_conv_filters,
        first_conv_kernel_size=config.model.first_conv_kernel_size,
        stride=config.model.stride,
        pointwise_filters=config.model.pointwise_filters,
        mixconv_kernel_sizes=config.model.mixconv_kernel_sizes,
        repeat_in_block=config.model.repeat_in_block,
        residual_connection=config.model.residual_connection,
        dropout_rate=0.0,
        mode="non_stream",
    )
    # Build by forward pass
    dummy = np.zeros((1, clip_frames, mel_bins), dtype=np.float32)
    _ = model(dummy, training=False)
    model.load_weights(checkpoint_path)
    return model


def build_streaming_model(config, checkpoint_path: str, fold_bn: bool = True):
    """Build and load the streaming export model (float32)."""
    pf = parse_int_list(config.model.pointwise_filters)
    ks = parse_kernel_sizes(config.model.mixconv_kernel_sizes)
    rc = parse_int_list(config.model.residual_connection)

    model = StreamingExportModel(
        first_conv_filters=config.model.first_conv_filters,
        first_conv_kernel=config.model.first_conv_kernel_size,
        stride=config.model.stride,
        pointwise_filters=pf,
        mixconv_kernel_sizes=ks,
        residual_connections=rc,
        mel_bins=config.hardware.mel_bins,
    )
    # Build by forward pass
    dummy = tf.zeros((1, config.model.stride, config.hardware.mel_bins), dtype=tf.float32)
    _ = model(dummy, training=False)

    n_loaded = load_weights_from_keras3_checkpoint(model, checkpoint_path)
    print(f"Streaming model: loaded {n_loaded} weights")

    if fold_bn:
        model.fold_batch_norms()
        print("BN folded into conv weights")

    return model


def run_non_streaming(model, spectrogram: np.ndarray) -> float:
    """Run non-streaming model on full spectrogram. Returns probability."""
    x = spectrogram[np.newaxis, ...]  # (1, T, F)
    pred = model(x, training=False)
    return float(pred.numpy().squeeze())


def run_streaming(model, spectrogram: np.ndarray, stride: int = 3) -> tuple[list[float], float]:
    """Run streaming model frame-by-frame. Returns all intermediate predictions and final."""
    T, F = spectrogram.shape
    predictions = []

    # Reset state variables to zero
    for sv in model.state_vars:
        sv.assign(tf.zeros_like(sv))

    # Process stride-sized chunks
    n_chunks = T // stride
    for i in range(n_chunks):
        chunk = spectrogram[i * stride : (i + 1) * stride]  # (stride, F)
        chunk = chunk[np.newaxis, ...]  # (1, stride, F)
        pred = model(tf.constant(chunk, dtype=tf.float32), training=False)
        predictions.append(float(pred.numpy().squeeze()))

    return predictions, predictions[-1] if predictions else 0.0


def generate_test_inputs(n_samples: int = 10, T: int = 100, F: int = 40, seed: int = 42) -> list[np.ndarray]:
    """Generate random test spectrograms."""
    rng = np.random.RandomState(seed)
    return [rng.randn(T, F).astype(np.float32) for _ in range(n_samples)]


def compare_models(config, checkpoint_path: str, n_samples: int = 50):
    """Compare non-streaming vs streaming model outputs."""
    print("=" * 70)
    print("NUMERICAL COMPARISON: Training vs Streaming Export (Float32)")
    print("=" * 70)

    # Build both models
    print("\n--- Building models ---")
    training_model = build_training_model(config, checkpoint_path)
    streaming_model_no_fold = build_streaming_model(config, checkpoint_path, fold_bn=False)
    streaming_model_folded = build_streaming_model(config, checkpoint_path, fold_bn=True)

    clip_frames = int(config.hardware.clip_duration_ms / config.hardware.window_step_ms)
    mel_bins = config.hardware.mel_bins
    stride = config.model.stride

    # Generate test inputs
    test_inputs = generate_test_inputs(n_samples, T=clip_frames, F=mel_bins)

    # Compare outputs
    print(f"\n--- Comparing on {n_samples} random spectrograms ---")
    print(f"{'Sample':>8} | {'Training':>10} | {'Stream(noBN)':>12} | {'Stream(fold)':>12} | {'Δ noBN':>10} | {'Δ fold':>10}")
    print("-" * 78)

    diffs_no_fold = []
    diffs_folded = []

    for i, spec in enumerate(test_inputs):
        p_train = run_non_streaming(training_model, spec)
        _, p_stream_no_fold = run_streaming(streaming_model_no_fold, spec, stride)
        _, p_stream_folded = run_streaming(streaming_model_folded, spec, stride)

        d_no_fold = abs(p_train - p_stream_no_fold)
        d_folded = abs(p_train - p_stream_folded)
        diffs_no_fold.append(d_no_fold)
        diffs_folded.append(d_folded)

        if i < 20:  # Print first 20
            print(f"{i:>8} | {p_train:>10.6f} | {p_stream_no_fold:>12.6f} | {p_stream_folded:>12.6f} | {d_no_fold:>10.6f} | {d_folded:>10.6f}")

    print("\n--- Summary Statistics ---")
    diffs_no_fold_arr = np.array(diffs_no_fold)
    diffs_folded_arr = np.array(diffs_folded)

    print(f"{'Metric':>20} | {'No BN fold':>12} | {'With BN fold':>12}")
    print("-" * 50)
    print(f"{'Mean abs diff':>20} | {diffs_no_fold_arr.mean():>12.6f} | {diffs_folded_arr.mean():>12.6f}")
    print(f"{'Max abs diff':>20} | {diffs_no_fold_arr.max():>12.6f} | {diffs_folded_arr.max():>12.6f}")
    print(f"{'Median abs diff':>20} | {np.median(diffs_no_fold_arr):>12.6f} | {np.median(diffs_folded_arr):>12.6f}")
    print(f"{'Std abs diff':>20} | {diffs_no_fold_arr.std():>12.6f} | {diffs_folded_arr.std():>12.6f}")

    print("\n--- Diagnosis ---")
    if diffs_no_fold_arr.mean() < 0.01:
        print("✅ Streaming (no BN fold) closely matches training → architecture is correct")
        if diffs_folded_arr.mean() < 0.01:
            print("✅ BN folding is numerically accurate")
            print("→ AUC gap is likely due to INT8 quantization")
        else:
            print("❌ BN folding introduces significant error")
            print("→ Investigate fold_batch_norms() math")
    else:
        print(f"❌ Streaming architecture DIVERGES from training (mean diff: {diffs_no_fold_arr.mean():.4f})")
        print("→ Possible causes:")
        print("   1. Temporal pooling: training=GlobalAvgPool(33 frames), streaming=Mean(5 frames)")
        print("   2. Residual connection computation differs")
        print("   3. Ring buffer / concat_update issue")

    # Extra: show streaming prediction curve for sample 0
    print("\n--- Streaming prediction curve (sample 0) ---")
    preds, _ = run_streaming(streaming_model_folded, test_inputs[0], stride)
    for j, p in enumerate(preds):
        bar = "█" * int(p * 50)
        print(f"  Frame {j:>3}: {p:.4f} |{bar}")

    return diffs_no_fold, diffs_folded


def main():
    parser = argparse.ArgumentParser(description="Debug streaming vs training model gap")
    parser.add_argument("--checkpoint", required=True, help="Path to .weights.h5 checkpoint")
    parser.add_argument(
        "--config",
        default="config/presets/max_quality.yaml",
        help="Config YAML",
    )
    parser.add_argument("--n-samples", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    config = load_full_config(override_path=args.config)
    compare_models(config, args.checkpoint, args.n_samples)


if __name__ == "__main__":
    main()
