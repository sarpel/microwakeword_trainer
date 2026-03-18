#!/usr/bin/env python3
"""Full test: F32 TFLite vs Python streaming on ALL 991 positives with proper reset.

Key question: Does the 12% gap disappear when we properly reset state between samples?
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.export.tflite import StreamingExportModel, load_weights_from_keras3_checkpoint


def load_all_positives(test_dir, mel_bins=40):
    """Load all positive test samples."""
    offsets = np.fromfile(os.path.join(test_dir, "features.offsets"), dtype=np.int64)
    lengths = np.fromfile(os.path.join(test_dir, "features.lengths"), dtype=np.int64)
    labels = np.fromfile(os.path.join(test_dir, "labels.data"), dtype=np.int32)
    features_data = np.fromfile(os.path.join(test_dir, "features.data"), dtype=np.float32)

    positive_indices = np.where(labels == 1)[0]
    samples = []
    for idx in positive_indices:
        byte_offset = offsets[idx]
        float_offset = byte_offset // 4
        n_floats = lengths[idx] // 4
        spec = features_data[float_offset : float_offset + n_floats].reshape(-1, mel_bins)
        samples.append(spec)
    return samples, positive_indices


def run_tflite(tflite_path, spectrogram, stride=3):
    """Run TFLite with FRESH interpreter per sample (guarantees clean state)."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    try:
        interp.reset_all_variables()
    except Exception:
        pass

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_dtype = input_details[0]["dtype"]

    n_frames = spectrogram.shape[0]
    pred = 0.0

    for start in range(0, n_frames - stride + 1, stride):
        chunk = spectrogram[start : start + stride]
        inp = chunk[np.newaxis, :, :].astype(np.float32)

        if input_dtype == np.int8:
            quant = input_details[0].get("quantization_parameters", {})
            scale = quant["scales"][0]
            zp = quant["zero_points"][0]
            inp = np.clip(np.round(inp / scale) + zp, -128, 127).astype(np.int8)

        interp.set_tensor(input_details[0]["index"], inp)
        interp.invoke()
        raw_out = interp.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.uint8:
            quant = output_details[0].get("quantization_parameters", {})
            scale = quant["scales"][0]
            zp = quant["zero_points"][0]
            pred = float((float(raw_out.flatten()[0]) - zp) * scale)
        else:
            pred = float(raw_out.flatten()[0])

    return pred


def run_tflite_reuse(interp, spectrogram, stride=3):
    """Run TFLite reusing interpreter but resetting variables."""
    try:
        interp.reset_all_variables()
    except Exception:
        pass

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_dtype = input_details[0]["dtype"]

    n_frames = spectrogram.shape[0]
    pred = 0.0

    for start in range(0, n_frames - stride + 1, stride):
        chunk = spectrogram[start : start + stride]
        inp = chunk[np.newaxis, :, :].astype(np.float32)

        if input_dtype == np.int8:
            quant = input_details[0].get("quantization_parameters", {})
            scale = quant["scales"][0]
            zp = quant["zero_points"][0]
            inp = np.clip(np.round(inp / scale) + zp, -128, 127).astype(np.int8)

        interp.set_tensor(input_details[0]["index"], inp)
        interp.invoke()
        raw_out = interp.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.uint8:
            quant = output_details[0].get("quantization_parameters", {})
            scale = quant["scales"][0]
            zp = quant["zero_points"][0]
            pred = float((float(raw_out.flatten()[0]) - zp) * scale)
        else:
            pred = float(raw_out.flatten()[0])

    return pred


def run_python(model, spectrogram, stride=3):
    """Run Python streaming model."""
    for sv in model.state_vars:
        sv.assign(tf.zeros_like(sv))

    n_frames = spectrogram.shape[0]
    pred = 0.0
    for start in range(0, n_frames - stride + 1, stride):
        chunk = spectrogram[start : start + stride]
        inp = tf.constant(chunk[np.newaxis, :, :], dtype=tf.float32)
        out = model(inp, training=False)
        pred = float(out.numpy().flatten()[0])
    return pred


def main():
    test_dir = "data/processed/test"
    f32_tflite = "/tmp/wake_word_float32.tflite"
    int8_tflite = "models/exported_v3/wake_word.tflite"
    checkpoint = "models/checkpoints/best_weights.weights.h5"

    print("Loading positive samples...")
    samples, indices = load_all_positives(test_dir)
    print(f"  {len(samples)} positive samples")

    # Build Python model
    print("Building Python model...")
    model = StreamingExportModel(
        first_conv_filters=32,
        first_conv_kernel=5,
        stride=3,
        pointwise_filters=[64, 64, 64, 64],
        mixconv_kernel_sizes=[[5], [7, 11], [9, 15], [23]],
        residual_connections=[0, 1, 1, 1],
        mel_bins=40,
        temporal_frames=32,
    )
    _ = model(tf.zeros((1, 3, 40), dtype=tf.float32))
    load_weights_from_keras3_checkpoint(model, checkpoint)
    model.fold_batch_norms()
    print("  Ready")

    # Test 1: F32 TFLite with reset_all_variables (reusing interpreter)
    print("\n=== Test 1: F32 TFLite with reset_all_variables() ===")
    f32_interp = tf.lite.Interpreter(model_path=f32_tflite)
    f32_interp.allocate_tensors()

    f32_reset_preds = []
    for i, spec in enumerate(samples):
        pred = run_tflite_reuse(f32_interp, spec, stride=3)
        f32_reset_preds.append(pred)
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}...")
    f32_reset = np.array(f32_reset_preds)
    print(f"  F32 (reset_all_variables): mean={f32_reset.mean():.4f}, recall@0.5={(f32_reset > 0.5).mean():.4f}")

    # Test 2: F32 TFLite with FRESH interpreter per sample
    print("\n=== Test 2: F32 TFLite with fresh interpreter per sample (first 100) ===")
    f32_fresh_preds = []
    for i, spec in enumerate(samples[:100]):
        pred = run_tflite(f32_tflite, spec, stride=3)
        f32_fresh_preds.append(pred)
        if i % 20 == 0:
            print(f"  {i}/100...")
    f32_fresh = np.array(f32_fresh_preds)
    print(f"  F32 (fresh interp): mean={f32_fresh.mean():.4f}, recall@0.5={(f32_fresh > 0.5).mean():.4f}")

    # Test 3: Python streaming (first 100 for speed)
    print("\n=== Test 3: Python streaming (first 100) ===")
    py_preds = []
    for i, spec in enumerate(samples[:100]):
        pred = run_python(model, spec, stride=3)
        py_preds.append(pred)
        if i % 20 == 0:
            print(f"  {i}/100...")
    py_arr = np.array(py_preds)
    print(f"  Python: mean={py_arr.mean():.4f}, recall@0.5={(py_arr > 0.5).mean():.4f}")

    # Compare first 100
    print("\n=== First 100 comparison ===")
    f32_reset_100 = f32_reset[:100]
    print(f"MAE Python vs F32 (reset):  {np.abs(py_arr - f32_reset_100).mean():.6f}")
    print(f"MAE Python vs F32 (fresh):  {np.abs(py_arr - f32_fresh).mean():.6f}")
    print(f"MAE F32 reset vs F32 fresh: {np.abs(f32_reset_100 - f32_fresh).mean():.6f}")

    # Check if reset is actually working
    print("\n=== State reset check ===")
    print("Comparing first 10 predictions between methods:")
    for i in range(min(10, len(py_arr))):
        diff_reset = abs(py_arr[i] - f32_reset_preds[i])
        diff_fresh = abs(py_arr[i] - f32_fresh_preds[i])
        flag = " ***" if diff_reset > 0.01 else ""
        print(f"  [{i:3d}] Python={py_arr[i]:.4f}  F32_reset={f32_reset_preds[i]:.4f}  F32_fresh={f32_fresh_preds[i]:.4f}  diff_reset={diff_reset:.6f}  diff_fresh={diff_fresh:.6f}{flag}")

    # Test 4: INT8 TFLite with reset
    print("\n=== Test 4: INT8 TFLite with reset_all_variables() ===")
    i8_interp = tf.lite.Interpreter(model_path=int8_tflite)
    i8_interp.allocate_tensors()

    i8_preds = []
    for i, spec in enumerate(samples):
        pred = run_tflite_reuse(i8_interp, spec, stride=3)
        i8_preds.append(pred)
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}...")
    i8_arr = np.array(i8_preds)
    print(f"  INT8 (reset): mean={i8_arr.mean():.4f}, recall@0.5={(i8_arr > 0.5).mean():.4f}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Python streaming (100):      recall@0.5 = {(py_arr > 0.5).mean():.4f}")
    print(f"F32 TFLite reset (all 991):  recall@0.5 = {(f32_reset > 0.5).mean():.4f}")
    print(f"F32 TFLite fresh (100):      recall@0.5 = {(f32_fresh > 0.5).mean():.4f}")
    print(f"INT8 TFLite reset (all 991): recall@0.5 = {(i8_arr > 0.5).mean():.4f}")

    # Save
    np.savez("/tmp/full_reset_test.npz", py_preds=py_arr, f32_reset_preds=f32_reset, f32_fresh_preds=f32_fresh, i8_preds=i8_arr)
    print("\nSaved to /tmp/full_reset_test.npz")


if __name__ == "__main__":
    main()
