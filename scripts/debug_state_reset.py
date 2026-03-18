#!/usr/bin/env python3
"""Debug why set_tensor on ReadVariableOp doesn't properly reset state.

Tests different reset methods to find the correct one.
"""

import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf


def get_all_variable_tensors(interp):
    """Get ALL tensors that could be state variables, not just ReadVariableOp."""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_indices = {d["index"] for d in input_details}
    output_indices = {d["index"] for d in output_details}

    all_tensors = []
    for tensor in interp.get_tensor_details():
        idx = tensor["index"]
        if idx in input_indices or idx in output_indices:
            continue
        name = tensor.get("name", "")
        shape = tuple(tensor.get("shape", ()))
        dtype = tensor["dtype"]
        all_tensors.append((idx, name, shape, dtype))
    return all_tensors


def dump_state(interp, label, tensor_list):
    """Dump non-zero state tensors."""
    print(f"\n--- {label} ---")
    non_zero_count = 0
    for idx, name, shape, dtype in tensor_list:
        val = interp.get_tensor(idx)
        if np.any(val != 0):
            non_zero_count += 1
            absmax = np.abs(val).max()
            nz = np.count_nonzero(val)
            total = val.size
            print(f"  [{idx:3d}] {name[:60]:60s} shape={shape}  |max|={absmax:.6f}  nz={nz}/{total}")
    if non_zero_count == 0:
        print("  (all tensors are zero)")
    return non_zero_count


def main():
    int8_path = "models/exported_v3/wake_word.tflite"
    test_dir = "data/processed/test"

    # Load a few positive samples
    offsets = np.fromfile(os.path.join(test_dir, "features.offsets"), dtype=np.int64)
    lengths = np.fromfile(os.path.join(test_dir, "features.lengths"), dtype=np.int64)
    labels = np.fromfile(os.path.join(test_dir, "labels.data"), dtype=np.int32)
    features_data = np.fromfile(os.path.join(test_dir, "features.data"), dtype=np.float32)

    pos_indices = np.where(labels == 1)[0]
    samples = []
    for idx in pos_indices[:5]:
        byte_offset = offsets[idx]
        float_offset = byte_offset // 4
        n_floats = lengths[idx] // 4
        spec = features_data[float_offset : float_offset + n_floats].reshape(-1, 40)
        samples.append(spec)

    # Load interpreter
    interp = tf.lite.Interpreter(model_path=int8_path)
    interp.allocate_tensors()

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_dtype = input_details[0]["dtype"]
    in_quant = input_details[0].get("quantization_parameters", {})
    in_scale = in_quant["scales"][0]
    in_zp = in_quant["zero_points"][0]
    out_quant = output_details[0].get("quantization_parameters", {})
    out_scale = out_quant["scales"][0]
    out_zp = out_quant["zero_points"][0]

    all_tensors = get_all_variable_tensors(interp)

    # Categorize tensors
    read_var_tensors = [(i, n, s, d) for i, n, s, d in all_tensors if "ReadVariableOp" in n]
    assign_tensors = [(i, n, s, d) for i, n, s, d in all_tensors if "AssignVariableOp" in n]
    other_tensors = [(i, n, s, d) for i, n, s, d in all_tensors if "ReadVariableOp" not in n and "AssignVariableOp" not in n]

    print(f"Total internal tensors: {len(all_tensors)}")
    print(f"  ReadVariableOp tensors: {len(read_var_tensors)}")
    print(f"  AssignVariableOp tensors: {len(assign_tensors)}")
    print(f"  Other tensors: {len(other_tensors)}")

    # Print all tensor details
    print("\n=== ALL INTERNAL TENSORS ===")
    for idx, name, shape, dtype in sorted(all_tensors, key=lambda x: x[0]):
        print(f"  [{idx:3d}] {name[:80]:80s} shape={str(shape):20s} dtype={dtype.__name__}")

    def run_inference(spec, stride=3):
        n_frames = spec.shape[0]
        pred = 0.0
        for start in range(0, n_frames - stride + 1, stride):
            chunk = spec[start : start + stride][np.newaxis, :, :].astype(np.float32)
            chunk = np.clip(np.round(chunk / in_scale) + in_zp, -128, 127).astype(np.int8)
            interp.set_tensor(input_details[0]["index"], chunk)
            interp.invoke()
            raw = interp.get_tensor(output_details[0]["index"])
            pred = float((float(raw.flatten()[0]) - out_zp) * out_scale)
        return pred

    # Test 1: Fresh state after allocate_tensors
    print("\n\n========== TEST 1: Initial state ==========")
    dump_state(interp, "After allocate_tensors", all_tensors)

    # Test 2: Run one sample
    print("\n\n========== TEST 2: After running sample 0 ==========")
    p0 = run_inference(samples[0])
    print(f"Prediction: {p0:.6f}")
    dump_state(interp, "After sample 0", read_var_tensors)

    # Test 3: Try reset_all_variables
    print("\n\n========== TEST 3: After reset_all_variables() ==========")
    interp.reset_all_variables()
    dump_state(interp, "After reset_all_variables()", read_var_tensors)

    # Test 4: Run sample again after reset
    p0_reset = run_inference(samples[0])
    print(f"Prediction after reset: {p0_reset:.6f}")
    print(f"Match original? {abs(p0 - p0_reset) < 0.001}")

    # Test 5: Run sample, then set_tensor zeros on ReadVariableOp
    print("\n\n========== TEST 5: After set_tensor zeros on ReadVariableOp ==========")
    for idx, name, shape, dtype in read_var_tensors:
        interp.set_tensor(idx, np.zeros(shape, dtype=dtype))
    dump_state(interp, "After set_tensor zeros (ReadVar only)", read_var_tensors)

    p0_setzero = run_inference(samples[0])
    print(f"Prediction after set_tensor: {p0_setzero:.6f}")
    print(f"Match original? {abs(p0 - p0_setzero) < 0.001}")

    # Test 6: Run 3 samples sequentially with different reset methods
    print("\n\n========== TEST 6: Sequential samples, comparing reset methods ==========")

    # 6a: No reset
    print("\n6a: NO reset between samples")
    interp_no = tf.lite.Interpreter(model_path=int8_path)
    interp_no.allocate_tensors()
    for i, spec in enumerate(samples[:5]):
        p = run_inference(spec)
        # This uses the SAME interp, so we need to reassign

    # Need separate interpreters for fair test
    # 6b: Fresh interpreter each time
    print("\n6b: FRESH interpreter per sample")
    for i, spec in enumerate(samples[:5]):
        fresh = tf.lite.Interpreter(model_path=int8_path)
        fresh.allocate_tensors()
        ind = fresh.get_input_details()
        oud = fresh.get_output_details()
        iq = ind[0].get("quantization_parameters", {})
        oq = oud[0].get("quantization_parameters", {})
        _is = iq["scales"][0]
        _iz = iq["zero_points"][0]
        _os = oq["scales"][0]
        _oz = oq["zero_points"][0]

        pred = 0.0
        for start in range(0, spec.shape[0] - 3 + 1, 3):
            chunk = spec[start : start + 3][np.newaxis].astype(np.float32)
            chunk = np.clip(np.round(chunk / _is) + _iz, -128, 127).astype(np.int8)
            fresh.set_tensor(ind[0]["index"], chunk)
            fresh.invoke()
            raw = fresh.get_tensor(oud[0]["index"])
            pred = float((float(raw.flatten()[0]) - _oz) * _os)
        print(f"  Sample {i}: pred={pred:.6f}")

    # 6c: Reuse interpreter + reset_all_variables
    print("\n6c: Reuse interpreter + reset_all_variables")
    interp_rv = tf.lite.Interpreter(model_path=int8_path)
    interp_rv.allocate_tensors()
    ind = interp_rv.get_input_details()
    oud = interp_rv.get_output_details()
    iq = ind[0].get("quantization_parameters", {})
    oq = oud[0].get("quantization_parameters", {})
    _is = iq["scales"][0]
    _iz = iq["zero_points"][0]
    _os = oq["scales"][0]
    _oz = oq["zero_points"][0]

    rv_tensors = [
        (t["index"], tuple(t.get("shape", ())), t["dtype"])
        for t in interp_rv.get_tensor_details()
        if "ReadVariableOp" in t.get("name", "") and t["index"] not in {d["index"] for d in ind} and t["index"] not in {d["index"] for d in oud}
    ]

    for i, spec in enumerate(samples[:5]):
        interp_rv.reset_all_variables()
        pred = 0.0
        for start in range(0, spec.shape[0] - 3 + 1, 3):
            chunk = spec[start : start + 3][np.newaxis].astype(np.float32)
            chunk = np.clip(np.round(chunk / _is) + _iz, -128, 127).astype(np.int8)
            interp_rv.set_tensor(ind[0]["index"], chunk)
            interp_rv.invoke()
            raw = interp_rv.get_tensor(oud[0]["index"])
            pred = float((float(raw.flatten()[0]) - _oz) * _os)
        print(f"  Sample {i}: pred={pred:.6f}")

    # 6d: Reuse interpreter + set_tensor zeros on ReadVariableOp
    print("\n6d: Reuse interpreter + set_tensor zeros on ReadVariableOp")
    for i, spec in enumerate(samples[:5]):
        for idx, shape, dtype in rv_tensors:
            interp_rv.set_tensor(idx, np.zeros(shape, dtype=dtype))
        pred = 0.0
        for start in range(0, spec.shape[0] - 3 + 1, 3):
            chunk = spec[start : start + 3][np.newaxis].astype(np.float32)
            chunk = np.clip(np.round(chunk / _is) + _iz, -128, 127).astype(np.int8)
            interp_rv.set_tensor(ind[0]["index"], chunk)
            interp_rv.invoke()
            raw = interp_rv.get_tensor(oud[0]["index"])
            pred = float((float(raw.flatten()[0]) - _oz) * _os)
        print(f"  Sample {i}: pred={pred:.6f}")

    # 6e: Reuse interpreter + allocate_tensors (re-invoke)
    print("\n6e: Reuse interpreter + re-call allocate_tensors()")
    for i, spec in enumerate(samples[:5]):
        interp_rv.allocate_tensors()
        pred = 0.0
        for start in range(0, spec.shape[0] - 3 + 1, 3):
            chunk = spec[start : start + 3][np.newaxis].astype(np.float32)
            chunk = np.clip(np.round(chunk / _is) + _iz, -128, 127).astype(np.int8)
            interp_rv.set_tensor(ind[0]["index"], chunk)
            interp_rv.invoke()
            raw = interp_rv.get_tensor(oud[0]["index"])
            pred = float((float(raw.flatten()[0]) - _oz) * _os)
        print(f"  Sample {i}: pred={pred:.6f}")


if __name__ == "__main__":
    main()
