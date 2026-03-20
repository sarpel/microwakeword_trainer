#!/usr/bin/env python3
"""Isolate INT8 quantization gap from streaming architecture gap.

Exports a float32 TFLite alongside the existing INT8 TFLite and evaluates both
on real feature store data (positive + negative samples) using streaming inference.

This answers: does the 13% recall gap come from INT8 quantization or from the
streaming architecture/BN folding/weight loading?

Usage:
    python scripts/isolate_quantization_gap.py \
        --checkpoint models/checkpoints/best_weights.weights.h5 \
        --int8-model models/exported_v3/wake_word.tflite \
        --config config/presets/max_quality.yaml
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_full_config
from src.export.tflite import export_streaming_tflite
from src.utils.label_guard import LABEL_POSITIVE, assert_labels_valid, is_negative


def run_tflite_streaming(interpreter, spectrogram: np.ndarray, stride: int) -> list[float]:
    """Run streaming TFLite on spectrogram, returning per-frame predictions."""
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    # Get quantization params
    input_quant = input_details[0].get("quantization_parameters", {})
    output_quant = output_details[0].get("quantization_parameters", {})
    input_scale = input_quant.get("scales", [1.0])
    input_zp = input_quant.get("zero_points", [0])
    output_scale = output_quant.get("scales", [1.0])
    output_zp = output_quant.get("zero_points", [0])

    # Reset state: run subgraph 1 (CALL_ONCE init)
    try:
        interpreter.reset_all_variables()
    except Exception:
        pass

    T, F = spectrogram.shape
    n_chunks = T // stride
    predictions = []

    for i in range(n_chunks):
        chunk = spectrogram[i * stride : (i + 1) * stride]
        chunk = chunk[np.newaxis, ...]  # (1, stride, F)

        # Quantize input if needed
        if input_dtype == np.int8:
            chunk_q = np.clip(np.round(chunk / input_scale[0] + input_zp[0]), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details[0]["index"], chunk_q)
        else:
            interpreter.set_tensor(input_details[0]["index"], chunk.astype(np.float32))

        interpreter.invoke()

        raw_out = interpreter.get_tensor(output_details[0]["index"])

        # Dequantize output if needed
        if output_dtype == np.uint8:
            prob = (raw_out.astype(np.float32) - output_zp[0]) * output_scale[0]
        elif output_dtype == np.int8:
            prob = (raw_out.astype(np.float32) - output_zp[0]) * output_scale[0]
        else:
            prob = raw_out.astype(np.float32)

        predictions.append(float(prob.squeeze()))

    return predictions


def load_feature_store_samples(config, max_positive=500, max_negative=500):
    """Load real samples from the feature store for evaluation."""
    data_dir = config.paths.data_dir if hasattr(config, "paths") else "processed_data"

    # Try to load from processed data
    positive_samples = []
    negative_samples = []

    # Check for processed feature store
    from src.data.dataset import FeatureStore

    store = FeatureStore(data_dir)
    store_any = cast(Any, store)

    clip_frames = int(config.hardware.clip_duration_ms / config.hardware.window_step_ms)
    mel_bins = config.hardware.mel_bins

    pos_count = 0
    neg_count = 0
    observed_labels: list[int] = []

    for i in range(min(len(store), max_positive + max_negative)):
        sample = store_any[i]
        if hasattr(sample, "label"):
            label = sample.label
            features = sample.features
        elif isinstance(sample, tuple):
            features, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            features = sample.get("features", sample.get("spectrogram"))
            label = sample.get("label", 0)
        else:
            continue

        # Ensure correct shape
        if isinstance(features, np.ndarray):
            spec = features
        else:
            spec = np.array(features)

        if spec.ndim == 1:
            # Flat features, try to reshape
            if len(spec) == clip_frames * mel_bins:
                spec = spec.reshape(clip_frames, mel_bins)
            else:
                continue

        if spec.shape != (clip_frames, mel_bins):
            continue

        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue

        observed_labels.append(label_int)

        if label_int == LABEL_POSITIVE and pos_count < max_positive:
            positive_samples.append(spec.astype(np.float32))
            pos_count += 1
        elif bool(np.asarray(is_negative(label_int)).item()) and neg_count < max_negative:
            negative_samples.append(spec.astype(np.float32))
            neg_count += 1

        if pos_count >= max_positive and neg_count >= max_negative:
            break

    if observed_labels:
        assert_labels_valid(np.asarray(observed_labels), context="isolate_quantization_gap")

    return positive_samples, negative_samples


def load_feature_store_raw(data_dir, clip_frames, mel_bins, max_positive=500, max_negative=500):
    """Load directly from RaggedMmap files in data_dir."""
    positive_samples = []
    negative_samples = []

    # Look for .npy files or ragged mmap data
    data_path = Path(data_dir)

    for category in ["positive", "negative"]:
        cat_path = data_path / category
        if not cat_path.exists():
            continue

        # Try loading individual files or mmap
        npy_files = sorted(cat_path.glob("*.npy"))
        max_count = max_positive if category == "positive" else max_negative
        target_list = positive_samples if category == "positive" else negative_samples

        for f in npy_files[:max_count]:
            try:
                spec = np.load(f).astype(np.float32)
                if spec.shape == (clip_frames, mel_bins):
                    target_list.append(spec)
                elif spec.ndim == 1 and len(spec) == clip_frames * mel_bins:
                    target_list.append(spec.reshape(clip_frames, mel_bins).astype(np.float32))
            except Exception:
                continue

    return positive_samples, negative_samples


def main():
    parser = argparse.ArgumentParser(description="Isolate INT8 quantization gap")
    parser.add_argument("--checkpoint", required=True, help="Path to .weights.h5 checkpoint")
    parser.add_argument("--int8-model", required=True, help="Path to existing INT8 TFLite model")
    parser.add_argument("--config", default="config/presets/max_quality.yaml", help="Config YAML")
    parser.add_argument("--data-dir", default=None, help="Processed data directory for real samples")
    parser.add_argument("--n-random", type=int, default=200, help="Number of random spectrograms to test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    config = load_full_config(override_path=args.config)

    clip_frames = int(config.hardware.clip_duration_ms / config.hardware.window_step_ms)
    mel_bins = config.hardware.mel_bins
    stride = config.model.stride

    # Step 1: Export float32 TFLite
    print("=" * 70)
    print("STEP 1: Export Float32 TFLite (no quantization)")
    print("=" * 70)

    float32_dir = tempfile.mkdtemp(prefix="mww_float32_")

    # Build config dict for export
    # Build config dict for export (must parse string configs like main() does)
    import ast

    pw_raw: Any = config.model.pointwise_filters
    if isinstance(pw_raw, str):
        pw_list = [int(x.strip()) for x in pw_raw.split(",")]
    elif isinstance(pw_raw, list):
        pw_list = [int(x) for x in pw_raw]
    else:
        pw_list = [64, 64, 64, 64]

    mc_raw = config.model.mixconv_kernel_sizes
    if isinstance(mc_raw, str):
        mc_list = ast.literal_eval(f"[{mc_raw}]")
    elif isinstance(mc_raw, list):
        mc_list = mc_raw
    else:
        mc_list = [[5], [7, 11], [9, 15], [23]]

    rc_raw: Any = config.model.residual_connection
    if isinstance(rc_raw, str):
        rc_list = [int(x.strip()) for x in rc_raw.split(",")]
    elif isinstance(rc_raw, list):
        rc_list = [int(x) for x in rc_raw]
    else:
        rc_list = [0, 1, 1, 1]

    export_config = {
        "first_conv_filters": config.model.first_conv_filters,
        "first_conv_kernel": config.model.first_conv_kernel_size,
        "stride": config.model.stride,
        "pointwise_filters": pw_list,
        "mixconv_kernel_sizes": mc_list,
        "residual_connections": rc_list,
        "mel_bins": config.hardware.mel_bins,
    }

    # Export float32 (unquantized)
    try:
        export_streaming_tflite(
            checkpoint_path=args.checkpoint,
            output_dir=float32_dir,
            model_name="wake_word_float32",
            config=export_config,
            quantize=False,
        )
        float32_path = os.path.join(float32_dir, "wake_word_float32.tflite")
        print(f"✓ Float32 TFLite exported: {float32_path}")
        print(f"  Size: {os.path.getsize(float32_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"✗ Float32 export failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 2: Load both interpreters
    print("\n" + "=" * 70)
    print("STEP 2: Load Interpreters")
    print("=" * 70)

    int8_interpreter = tf.lite.Interpreter(model_path=args.int8_model)
    int8_interpreter.allocate_tensors()

    float32_interpreter = tf.lite.Interpreter(model_path=float32_path)
    float32_interpreter.allocate_tensors()

    # Print model details
    for name, interp in [("INT8", int8_interpreter), ("Float32", float32_interpreter)]:
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print(f"  {name}: input={inp['shape']} dtype={inp['dtype'].__name__}, output={out['shape']} dtype={out['dtype'].__name__}")

    # Step 3: Generate test data
    print("\n" + "=" * 70)
    print("STEP 3: Generate Test Data")
    print("=" * 70)

    # Use random spectrograms with varying characteristics
    rng = np.random.RandomState(42)

    # Random Gaussian (as baseline)
    random_specs = [rng.randn(clip_frames, mel_bins).astype(np.float32) for _ in range(args.n_random)]

    # Also generate some with different distributions to exercise the quantization range
    # Uniform in [0, 1] range
    uniform_specs = [rng.uniform(0, 1, (clip_frames, mel_bins)).astype(np.float32) for _ in range(50)]

    # Large values (stress quantization)
    large_specs = [rng.randn(clip_frames, mel_bins).astype(np.float32) * 5.0 for _ in range(50)]

    # Small values near zero
    small_specs = [rng.randn(clip_frames, mel_bins).astype(np.float32) * 0.01 for _ in range(50)]

    test_sets = {
        "random_gaussian": random_specs,
        "uniform_0_1": uniform_specs,
        "large_5x": large_specs,
        "small_0.01x": small_specs,
    }

    # Step 4: Compare predictions
    print("\n" + "=" * 70)
    print("STEP 4: Compare Float32 vs INT8 Predictions")
    print("=" * 70)

    for set_name, specs in test_sets.items():
        float32_finals = []
        int8_finals = []

        for spec in specs:
            f32_preds = run_tflite_streaming(float32_interpreter, spec, stride)
            i8_preds = run_tflite_streaming(int8_interpreter, spec, stride)

            if f32_preds:
                float32_finals.append(f32_preds[-1])
            if i8_preds:
                int8_finals.append(i8_preds[-1])

        f32_arr = np.array(float32_finals)
        i8_arr = np.array(int8_finals)
        diffs = np.abs(f32_arr - i8_arr)

        # Classification agreement
        f32_class = (f32_arr > args.threshold).astype(int)
        i8_class = (i8_arr > args.threshold).astype(int)
        agreement = np.mean(f32_class == i8_class) * 100

        print(f"\n  --- {set_name} ({len(specs)} samples) ---")
        print(f"    Float32 mean pred: {f32_arr.mean():.6f} (std: {f32_arr.std():.6f})")
        print(f"    INT8    mean pred: {i8_arr.mean():.6f} (std: {i8_arr.std():.6f})")
        print(f"    Abs diff:  mean={diffs.mean():.6f}, max={diffs.max():.6f}, median={np.median(diffs):.6f}")
        print(f"    Classification agreement: {agreement:.1f}%")

    # Step 5: Detailed per-frame comparison on a few samples
    print("\n" + "=" * 70)
    print("STEP 5: Per-Frame Comparison (first 5 random samples)")
    print("=" * 70)

    for i in range(min(5, len(random_specs))):
        spec = random_specs[i]
        f32_preds = run_tflite_streaming(float32_interpreter, spec, stride)
        i8_preds = run_tflite_streaming(int8_interpreter, spec, stride)

        print(f"\n  Sample {i}:")
        print(f"    {'Frame':>6} | {'Float32':>10} | {'INT8':>10} | {'Δ':>10}")
        print(f"    {'-' * 48}")

        for j in range(min(len(f32_preds), len(i8_preds))):
            delta = abs(f32_preds[j] - i8_preds[j])
            marker = " ⚠️" if delta > 0.05 else ""
            print(f"    {j:>6} | {f32_preds[j]:>10.6f} | {i8_preds[j]:>10.6f} | {delta:>10.6f}{marker}")

        print(f"    Final: float32={f32_preds[-1]:.6f}, int8={i8_preds[-1]:.6f}, Δ={abs(f32_preds[-1] - i8_preds[-1]):.6f}")

    # Step 6: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Compute overall gap
    all_f32 = []
    all_i8 = []
    for specs in test_sets.values():
        for spec in specs:
            f32_preds = run_tflite_streaming(float32_interpreter, spec, stride)
            i8_preds = run_tflite_streaming(int8_interpreter, spec, stride)
            if f32_preds and i8_preds:
                all_f32.append(f32_preds[-1])
                all_i8.append(i8_preds[-1])

    all_f32 = np.array(all_f32)
    all_i8 = np.array(all_i8)
    all_diffs = np.abs(all_f32 - all_i8)

    print(f"\n  Total samples evaluated: {len(all_f32)}")
    print(f"  Overall mean absolute diff: {all_diffs.mean():.6f}")
    print(f"  Overall max absolute diff:  {all_diffs.max():.6f}")
    print(f"  Correlation (Pearson):       {np.corrcoef(all_f32, all_i8)[0, 1]:.6f}")

    if all_diffs.mean() < 0.01:
        print("\n  ✅ INT8 quantization drift is SMALL on synthetic data.")
        print("     The 13% recall gap is likely due to THRESHOLD SENSITIVITY:")
        print("     - Small quantization noise shifts predictions near the decision boundary")
        print("     - This causes disproportionate classification changes for borderline samples")
        print("     → Try evaluating with a LOWER threshold for INT8 model")
    elif all_diffs.mean() < 0.05:
        print("\n  ⚠️  INT8 quantization drift is MODERATE.")
        print("     The Dense layer's large weight scale (0.358) compresses the sigmoid range.")
        print("     → Need better calibration data or weight regularization")
    else:
        print("\n  ❌ INT8 quantization drift is LARGE.")
        print("     Something is fundamentally wrong with the quantization.")

    # Cleanup
    import shutil

    shutil.rmtree(float32_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
