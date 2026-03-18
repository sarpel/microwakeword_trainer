#!/usr/bin/env python3
"""Benchmark: fresh interpreter per sample vs reuse."""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf


def main():
    tflite_path = "models/exported_v3/wake_word.tflite"
    test_dir = "data/processed/test"

    offsets = np.fromfile(os.path.join(test_dir, "features.offsets"), dtype=np.int64)
    lengths = np.fromfile(os.path.join(test_dir, "features.lengths"), dtype=np.int64)
    features_data = np.fromfile(os.path.join(test_dir, "features.data"), dtype=np.float32)
    labels = np.fromfile(os.path.join(test_dir, "labels.data"), dtype=np.int32)

    pos_idx = np.where(labels == 1)[0][:100]
    samples = []
    for idx in pos_idx:
        off = offsets[idx] // 4
        nf = lengths[idx] // 4
        samples.append(features_data[off : off + nf].reshape(-1, 40))

    # Read model bytes once
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    def run_one(interp, spec, stride=3):
        ind = interp.get_input_details()
        oud = interp.get_output_details()
        iq = ind[0].get("quantization_parameters", {})
        oq = oud[0].get("quantization_parameters", {})
        i_s, i_z = iq["scales"][0], iq["zero_points"][0]
        o_s, o_z = oq["scales"][0], oq["zero_points"][0]
        pred = 0.0
        for start in range(0, spec.shape[0] - stride + 1, stride):
            chunk = spec[start : start + stride][np.newaxis].astype(np.float32)
            if ind[0]["dtype"] != np.float32:
                chunk = np.clip(np.round(chunk / i_s) + i_z, -128, 127).astype(np.int8)
            interp.set_tensor(ind[0]["index"], chunk)
            interp.invoke()
            raw = interp.get_tensor(oud[0]["index"])
            if oud[0]["dtype"] != np.float32:
                pred = float((float(raw.flatten()[0]) - o_z) * o_s)
            else:
                pred = float(raw.flatten()[0])
        return pred

    # Benchmark 1: Fresh interpreter from file path
    t0 = time.time()
    for spec in samples:
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        run_one(interp, spec)
    t1 = time.time()
    print(f"Fresh interp (from file):  {t1 - t0:.2f}s for {len(samples)} samples ({(t1 - t0) / len(samples) * 1000:.1f} ms/sample)")

    # Benchmark 2: Fresh interpreter from bytes
    t0 = time.time()
    for spec in samples:
        interp = tf.lite.Interpreter(model_content=model_bytes)
        interp.allocate_tensors()
        run_one(interp, spec)
    t1 = time.time()
    print(f"Fresh interp (from bytes): {t1 - t0:.2f}s for {len(samples)} samples ({(t1 - t0) / len(samples) * 1000:.1f} ms/sample)")

    # Benchmark 3: Reuse interpreter (broken, for speed comparison only)
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    t0 = time.time()
    for spec in samples:
        interp.reset_all_variables()
        run_one(interp, spec)
    t1 = time.time()
    print(f"Reuse + reset (BROKEN):    {t1 - t0:.2f}s for {len(samples)} samples ({(t1 - t0) / len(samples) * 1000:.1f} ms/sample)")


if __name__ == "__main__":
    main()
