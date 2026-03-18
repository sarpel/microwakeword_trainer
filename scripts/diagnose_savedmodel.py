#!/usr/bin/env python3
"""Diagnose WHERE in the export pipeline the model gets corrupted.

Tests:
1. Python StreamingExportModel (eager)
2. SavedModel (tf.saved_model.load)
3. TFLite float32
4. TFLite INT8

On the same divergent sample to find the exact corruption point.
"""

import os
import shutil
import sys
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.export.tflite import (
    StreamingExportModel,
    load_weights_from_keras3_checkpoint,
)


def load_test_sample(test_dir, sample_idx):
    """Load a single test sample from RaggedMmap."""
    offsets = np.fromfile(os.path.join(test_dir, "features.offsets"), dtype=np.int64)
    lengths = np.fromfile(os.path.join(test_dir, "features.lengths"), dtype=np.int64)
    labels = np.fromfile(os.path.join(test_dir, "labels.data"), dtype=np.int32)

    mel_bins = 40
    byte_offset = offsets[sample_idx]
    float_offset = byte_offset // 4
    n_floats = lengths[sample_idx] // 4

    features_data = np.fromfile(os.path.join(test_dir, "features.data"), dtype=np.float32, count=float_offset + n_floats, offset=0)
    spec = features_data[float_offset : float_offset + n_floats].reshape(-1, mel_bins)
    return spec, labels[sample_idx]


def run_python_streaming(model, spectrogram, stride=3):
    """Run model in Python eager mode (known good)."""
    n_frames = spectrogram.shape[0]
    # Reset state
    for sv in model.state_vars:
        sv.assign(tf.zeros_like(sv))

    pred = 0.0
    for start in range(0, n_frames - stride + 1, stride):
        chunk = spectrogram[start : start + stride]
        inp = tf.constant(chunk[np.newaxis, :, :], dtype=tf.float32)
        out = model(inp, training=False)
        pred = float(out.numpy().flatten()[0])
    return pred


def run_savedmodel_streaming(saved_model_dir, spectrogram, stride=3):
    """Run SavedModel in graph mode."""
    loaded = tf.saved_model.load(saved_model_dir)
    serve_fn = loaded.signatures.get("serving_default")
    if serve_fn is None:
        # Try the 'serve' endpoint
        serve_fn = getattr(loaded, "serve", None)
        if serve_fn is None:
            print("  Available signatures:", list(loaded.signatures.keys()))
            raise ValueError("No serving function found")

    # Get variable handles from the loaded model
    variables = loaded.variables
    print(f"  SavedModel has {len(variables)} variables")

    # Reset all variables to zero
    for var in variables:
        if "stream" in var.name.lower() or hasattr(var, "shape"):
            try:
                var.assign(tf.zeros_like(var))
            except Exception:
                pass

    n_frames = spectrogram.shape[0]
    pred = 0.0

    for start in range(0, n_frames - stride + 1, stride):
        chunk = spectrogram[start : start + stride]
        inp = tf.constant(chunk[np.newaxis, :, :], dtype=tf.float32)

        # The serve function signature uses "inputs" as key
        if hasattr(serve_fn, "__call__"):
            try:
                result = serve_fn(inputs=inp)
            except TypeError:
                result = serve_fn(inp)
        else:
            result = serve_fn(inp)

        # Result is a dict with output tensor
        if isinstance(result, dict):
            out_key = list(result.keys())[0]
            out = result[out_key]
        else:
            out = result
        pred = float(out.numpy().flatten()[0])

    return pred


def run_tflite_streaming(tflite_path, spectrogram, stride=3):
    """Run TFLite model in streaming mode."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    # Reset state variables
    try:
        interp.reset_all_variables()
    except Exception:
        pass

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    n_frames = spectrogram.shape[0]
    pred = 0.0

    input_dtype = input_details[0]["dtype"]

    for start in range(0, n_frames - stride + 1, stride):
        chunk = spectrogram[start : start + stride]
        inp = chunk[np.newaxis, :, :].astype(np.float32)

        if input_dtype == np.int8:
            quant = input_details[0].get("quantization_parameters", {})
            scale = quant.get("scales", [1.0])[0]
            zp = quant.get("zero_points", [0])[0]
            inp = np.clip(np.round(inp / scale) + zp, -128, 127).astype(np.int8)

        interp.set_tensor(input_details[0]["index"], inp)
        interp.invoke()

        raw_out = interp.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.uint8:
            quant = output_details[0].get("quantization_parameters", {})
            scale = quant.get("scales", [1 / 256])[0]
            zp = quant.get("zero_points", [0])[0]
            pred = float((raw_out.astype(np.float32) - zp) * scale)
        elif output_details[0]["dtype"] == np.float32:
            pred = float(raw_out.flatten()[0])
        else:
            pred = float(raw_out.flatten()[0])

    return pred


def main():
    checkpoint_path = "models/checkpoints/best_weights.weights.h5"
    test_dir = "data/processed/test"
    int8_tflite = "models/exported_v3/wake_word.tflite"

    # Load a divergent positive sample (known to have Keras>0.9, TFLite<0.5)
    # Load a few positive samples and find one that diverges
    print("Loading test samples...")
    np.fromfile(os.path.join(test_dir, "features.offsets"), dtype=np.int64)
    labels = np.fromfile(os.path.join(test_dir, "labels.data"), dtype=np.int32)

    # Find first 10 positive samples
    positive_indices = np.where(labels == 1)[0][:30]
    print(f"Testing {len(positive_indices)} positive samples")

    # --- Step 1: Build Python model ---
    print("\n=== Building Python StreamingExportModel ===")
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
    loaded = load_weights_from_keras3_checkpoint(model, checkpoint_path)
    print(f"  Loaded {loaded} weights")
    model.fold_batch_norms()
    print("  BN folded")

    # --- Step 2: Export to SavedModel ---
    print("\n=== Exporting to SavedModel ===")
    saved_model_dir = tempfile.mkdtemp(prefix="mww_diag_")

    try:
        export_archive = tf.keras.export.ExportArchive()
        export_archive.track(model)

        export_input_sig = [tf.TensorSpec(shape=(1, 3, 40), dtype=tf.float32, name="inputs")]

        def serve_fn(inputs):
            return model(inputs, training=False)

        export_archive.add_endpoint(
            name="serve",
            fn=serve_fn,
            input_signature=export_input_sig,
        )
        export_archive.write_out(saved_model_dir)
        print(f"  SavedModel written to {saved_model_dir}")

        # --- Step 2b: Export float32 TFLite ---
        print("\n=== Converting to Float32 TFLite ===")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.experimental_enable_resource_variables = True
        f32_tflite_bytes = converter.convert()
        f32_tflite_path = "/tmp/diag_float32.tflite"
        with open(f32_tflite_path, "wb") as f:
            f.write(f32_tflite_bytes)
        print(f"  Float32 TFLite: {len(f32_tflite_bytes) / 1024:.1f} KB")

        # --- Step 3: Run all 4 variants on each sample ---
        print("\n=== Running comparison ===")
        print(f"{'Idx':>5} {'Frames':>6} {'Python':>8} {'SavedModel':>10} {'F32 TFL':>8} {'INT8 TFL':>8} {'Py-SM':>7} {'Py-F32':>7}")
        print("-" * 75)

        py_preds = []
        sm_preds = []
        f32_preds = []
        i8_preds = []

        for j, idx in enumerate(positive_indices):
            spec, label = load_test_sample(test_dir, idx)

            # Python eager
            py_pred = run_python_streaming(model, spec, stride=3)

            # SavedModel
            sm_pred = run_savedmodel_streaming(saved_model_dir, spec, stride=3)

            # Float32 TFLite
            f32_pred = run_tflite_streaming(f32_tflite_path, spec, stride=3)

            # INT8 TFLite
            i8_pred = run_tflite_streaming(int8_tflite, spec, stride=3)

            py_preds.append(py_pred)
            sm_preds.append(sm_pred)
            f32_preds.append(f32_pred)
            i8_preds.append(i8_pred)

            py_sm_diff = abs(py_pred - sm_pred)
            py_f32_diff = abs(py_pred - f32_pred)
            flag = " ***" if py_f32_diff > 0.1 else ""

            print(f"{idx:5d} {spec.shape[0]:6d} {py_pred:8.4f} {sm_pred:10.4f} {f32_pred:8.4f} {i8_pred:8.4f} {py_sm_diff:7.4f} {py_f32_diff:7.4f}{flag}")

        # Summary
        py_a = np.array(py_preds)
        sm_a = np.array(sm_preds)
        f32_a = np.array(f32_preds)
        i8_a = np.array(i8_preds)

        print(f"\n=== Summary (n={len(positive_indices)}) ===")
        print(f"Python mean:     {py_a.mean():.4f}, recall@0.5: {(py_a > 0.5).mean():.4f}")
        print(f"SavedModel mean: {sm_a.mean():.4f}, recall@0.5: {(sm_a > 0.5).mean():.4f}")
        print(f"F32 TFLite mean: {f32_a.mean():.4f}, recall@0.5: {(f32_a > 0.5).mean():.4f}")
        print(f"INT8 TFLite mean:{i8_a.mean():.4f}, recall@0.5: {(i8_a > 0.5).mean():.4f}")

        print(f"\nMAE Python→SavedModel: {np.abs(py_a - sm_a).mean():.6f}")
        print(f"MAE Python→F32 TFLite: {np.abs(py_a - f32_a).mean():.6f}")
        print(f"MAE SavedModel→F32 TFLite: {np.abs(sm_a - f32_a).mean():.6f}")
        print(f"MAE F32→INT8 TFLite:   {np.abs(f32_a - i8_a).mean():.6f}")

        # Identify exact corruption point
        py_sm_gap = np.abs(py_a - sm_a).mean()
        sm_f32_gap = np.abs(sm_a - f32_a).mean()

        print("\n=== Corruption Attribution ===")
        print(f"Python → SavedModel:    MAE = {py_sm_gap:.6f}")
        print(f"SavedModel → F32 TFLite: MAE = {sm_f32_gap:.6f}")

        if py_sm_gap > 0.01:
            print(">>> CORRUPTION in ExportArchive → SavedModel export")
        elif sm_f32_gap > 0.01:
            print(">>> CORRUPTION in SavedModel → TFLite conversion")
        else:
            print(">>> Both steps look clean. Issue may be elsewhere.")

    finally:
        shutil.rmtree(saved_model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
