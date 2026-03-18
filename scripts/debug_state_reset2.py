#!/usr/bin/env python3
"""Debug state reset: test which method works for proper state zeroing."""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf


def load_samples(test_dir, mel_bins=40, n=5):
    offsets = np.fromfile(os.path.join(test_dir, "features.offsets"), dtype=np.int64)
    lengths = np.fromfile(os.path.join(test_dir, "features.lengths"), dtype=np.int64)
    labels = np.fromfile(os.path.join(test_dir, "labels.data"), dtype=np.int32)
    features_data = np.fromfile(os.path.join(test_dir, "features.data"), dtype=np.float32)
    pos_indices = np.where(labels == 1)[0][:n]
    samples = []
    for idx in pos_indices:
        off = offsets[idx] // 4
        nf = lengths[idx] // 4
        samples.append(features_data[off : off + nf].reshape(-1, mel_bins))
    return samples


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


def main():
    tflite_path = "models/exported_v3/wake_word.tflite"
    test_dir = "data/processed/test"
    samples = load_samples(test_dir, n=10)

    # Ground truth: fresh interpreter per sample
    print("=== Ground truth: Fresh interpreter per sample ===")
    gt_preds = []
    for i, spec in enumerate(samples):
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        p = run_one(interp, spec)
        gt_preds.append(p)
        print(f"  Sample {i}: {p:.6f}")

    # Find ReadVariableOp tensors
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    ind = interp.get_input_details()
    oud = interp.get_output_details()
    in_idx = {d["index"] for d in ind}
    out_idx = {d["index"] for d in oud}

    read_var = []
    resource_var = []
    for t in interp.get_tensor_details():
        idx = t["index"]
        if idx in in_idx or idx in out_idx:
            continue
        name = t.get("name", "")
        shape = tuple(t.get("shape", ()))
        dtype = t["dtype"]
        if "ReadVariableOp" in name:
            read_var.append((idx, name, shape, dtype))
        elif "stream" in name.lower() and dtype == object:
            resource_var.append((idx, name, shape, dtype))

    print(f"\nReadVariableOp tensors ({len(read_var)}):")
    for idx, name, shape, dtype in read_var:
        print(f"  [{idx}] {name} shape={shape} dtype={dtype.__name__}")
    print(f"\nResource variables ({len(resource_var)}):")
    for idx, name, shape, dtype in resource_var:
        print(f"  [{idx}] {name} shape={shape} dtype={dtype.__name__}")

    # Method 1: reset_all_variables
    print("\n=== Method 1: reset_all_variables() ===")
    for i, spec in enumerate(samples):
        interp.reset_all_variables()
        p = run_one(interp, spec)
        diff = abs(p - gt_preds[i])
        flag = " *** MISMATCH" if diff > 0.01 else ""
        print(f"  Sample {i}: {p:.6f} (gt={gt_preds[i]:.6f}, diff={diff:.6f}){flag}")

    # Method 2: set_tensor on ReadVariableOp
    print("\n=== Method 2: set_tensor on ReadVariableOp ===")
    interp2 = tf.lite.Interpreter(model_path=tflite_path)
    interp2.allocate_tensors()
    rv2 = [(idx, shape, dtype) for idx, name, shape, dtype in read_var]
    for i, spec in enumerate(samples):
        for idx, shape, dtype in rv2:
            interp2.set_tensor(idx, np.zeros(shape, dtype=dtype))
        p = run_one(interp2, spec)
        diff = abs(p - gt_preds[i])
        flag = " *** MISMATCH" if diff > 0.01 else ""
        print(f"  Sample {i}: {p:.6f} (gt={gt_preds[i]:.6f}, diff={diff:.6f}){flag}")

    # Method 3: allocate_tensors() before each sample
    print("\n=== Method 3: Re-call allocate_tensors() ===")
    interp3 = tf.lite.Interpreter(model_path=tflite_path)
    interp3.allocate_tensors()
    for i, spec in enumerate(samples):
        interp3.allocate_tensors()
        p = run_one(interp3, spec)
        diff = abs(p - gt_preds[i])
        flag = " *** MISMATCH" if diff > 0.01 else ""
        print(f"  Sample {i}: {p:.6f} (gt={gt_preds[i]:.6f}, diff={diff:.6f}){flag}")

    # Method 4: set_tensor on BOTH ReadVariableOp AND try resource vars
    print("\n=== Method 4: set_tensor on ReadVariableOp + resource vars ===")
    interp4 = tf.lite.Interpreter(model_path=tflite_path)
    interp4.allocate_tensors()
    rv4 = [(idx, shape, dtype) for idx, name, shape, dtype in read_var]
    for i, spec in enumerate(samples):
        for idx, shape, dtype in rv4:
            interp4.set_tensor(idx, np.zeros(shape, dtype=dtype))
        # Also try zeroing resource vars
        for idx, name, shape, dtype in resource_var:
            try:
                interp4.set_tensor(idx, np.zeros(shape, dtype=np.float32))
            except Exception as e:
                if i == 0:
                    print(f"  Cannot set resource var [{idx}]: {e}")
        p = run_one(interp4, spec)
        diff = abs(p - gt_preds[i])
        flag = " *** MISMATCH" if diff > 0.01 else ""
        print(f"  Sample {i}: {p:.6f} (gt={gt_preds[i]:.6f}, diff={diff:.6f}){flag}")

    # Method 5: Check what values ReadVariableOp tensors have after inference
    print("\n=== Method 5: State inspection after each sample ===")
    interp5 = tf.lite.Interpreter(model_path=tflite_path)
    interp5.allocate_tensors()

    print("Before any inference:")
    for idx, name, shape, dtype in read_var:
        val = interp5.get_tensor(idx)
        nz = np.count_nonzero(val)
        print(f"  [{idx}] nonzero={nz}/{val.size}")

    p = run_one(interp5, samples[0])
    print(f"\nAfter sample 0 (pred={p:.6f}):")
    for idx, name, shape, dtype in read_var:
        val = interp5.get_tensor(idx)
        nz = np.count_nonzero(val)
        print(f"  [{idx}] nonzero={nz}/{val.size}")

    interp5.reset_all_variables()
    print("\nAfter reset_all_variables():")
    for idx, name, shape, dtype in read_var:
        val = interp5.get_tensor(idx)
        nz = np.count_nonzero(val)
        print(f"  [{idx}] nonzero={nz}/{val.size}")

    # Manual set_tensor zeros
    for idx, name, shape, dtype in read_var:
        interp5.set_tensor(idx, np.zeros(shape, dtype=dtype))
    print("\nAfter set_tensor zeros:")
    for idx, name, shape, dtype in read_var:
        val = interp5.get_tensor(idx)
        nz = np.count_nonzero(val)
        print(f"  [{idx}] nonzero={nz}/{val.size}")

    # Now invoke once and check again
    spec = samples[0]
    ind5 = interp5.get_input_details()
    chunk = spec[0:3][np.newaxis].astype(np.float32)
    iq = ind5[0].get("quantization_parameters", {})
    chunk = np.clip(np.round(chunk / iq["scales"][0]) + iq["zero_points"][0], -128, 127).astype(np.int8)
    interp5.set_tensor(ind5[0]["index"], chunk)
    interp5.invoke()
    print("\nAfter 1 invoke post-reset:")
    for idx, name, shape, dtype in read_var:
        val = interp5.get_tensor(idx)
        nz = np.count_nonzero(val)
        print(f"  [{idx}] nonzero={nz}/{val.size}")


if __name__ == "__main__":
    main()
