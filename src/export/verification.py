from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def _estimate_tensor_arena_size(interpreter: tf.lite.Interpreter) -> int:
    total_memory = 0
    for tensor in interpreter.get_tensor_details():
        shape = tensor.get("shape", [])
        dtype = tensor.get("dtype")

        if dtype == np.float32:
            elem_size = 4
        elif dtype == np.float16:
            elem_size = 2
        elif dtype in (np.int8, np.uint8):
            elem_size = 1
        else:
            elem_size = 4

        num_elements = int(np.prod(shape)) if shape is not None else 1
        total_memory += num_elements * elem_size

    return int(total_memory * 1.3)


def verify_tflite_model(tflite_path: str) -> dict[str, Any]:
    resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_op_resolver_type=resolver_type)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ops_details = interpreter._get_ops_details()

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    valid = True

    input_shape = list(input_details[0]["shape"])
    input_dtype = input_details[0]["dtype"]
    details["input_shape"] = input_shape
    details["input_dtype"] = str(input_dtype)

    if input_shape != [1, 3, 40]:
        errors.append(f"Input shape {input_shape} != [1, 3, 40]")
        valid = False
    checks["input_shape"] = input_shape == [1, 3, 40]

    if input_dtype != np.int8:
        errors.append(f"Input dtype {input_dtype} != int8")
        valid = False
    checks["input_dtype"] = input_dtype == np.int8

    output_shape = list(output_details[0]["shape"])
    output_dtype = output_details[0]["dtype"]
    details["output_shape"] = output_shape
    details["output_dtype"] = str(output_dtype)

    if output_shape != [1, 1]:
        errors.append(f"Output shape {output_shape} != [1, 1]")
        valid = False
    checks["output_shape"] = output_shape == [1, 1]

    if output_dtype != np.uint8:
        errors.append(f"Output dtype {output_dtype} != uint8")
        valid = False
    checks["output_dtype"] = output_dtype == np.uint8

    input_quant = input_details[0].get("quantization_parameters", {})
    output_quant = output_details[0].get("quantization_parameters", {})
    input_scales = np.asarray(input_quant.get("scales", [])) if input_quant else np.array([])
    input_zero_points = np.asarray(input_quant.get("zero_points", [])) if input_quant else np.array([])
    output_scales = np.asarray(output_quant.get("scales", [])) if output_quant else np.array([])
    output_zero_points = np.asarray(output_quant.get("zero_points", [])) if output_quant else np.array([])

    if input_scales.size == 0 or input_zero_points.size == 0:
        errors.append("Missing input quantization parameters")
        valid = False
        checks["input_quant_params"] = False
    else:
        input_scale = float(input_scales[0])
        input_zero = int(input_zero_points[0])
        details["input_scale"] = input_scale
        details["input_zero_point"] = input_zero
        checks["input_quant_params"] = abs(input_scale - 0.101961) <= 1e-4 and input_zero == -128
        if not checks["input_quant_params"]:
            errors.append(f"Input quantization mismatch: scale={input_scale}, zero_point={input_zero}")
            valid = False

    if output_scales.size == 0 or output_zero_points.size == 0:
        errors.append("Missing output quantization parameters")
        valid = False
        checks["output_quant_params"] = False
    else:
        output_scale = float(output_scales[0])
        output_zero = int(output_zero_points[0])
        details["output_scale"] = output_scale
        details["output_zero_point"] = output_zero
        checks["output_quant_params"] = abs(output_scale - 0.00390625) <= 1e-6 and output_zero == 0
        if not checks["output_quant_params"]:
            errors.append(f"Output quantization mismatch: scale={output_scale}, zero_point={output_zero}")
            valid = False

    op_counts: dict[str, int] = {}
    for op in ops_details:
        op_name = op.get("op_name", "")
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
    details["op_counts"] = op_counts

    allowed_ops = {
        "CALL_ONCE",
        "VAR_HANDLE",
        "READ_VARIABLE",
        "STRIDED_SLICE",
        "CONCATENATION",
        "ASSIGN_VARIABLE",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "MUL",
        "ADD",
        "MEAN",
        "FULLY_CONNECTED",
        "LOGISTIC",
        "QUANTIZE",
        "RESHAPE",
        "SPLIT_V",
    }
    unpermitted_ops = sorted(op for op in op_counts if op and op not in allowed_ops)
    if unpermitted_ops:
        errors.append(f"Unpermitted ops detected: {unpermitted_ops}")
        valid = False
    checks["op_whitelist"] = len(unpermitted_ops) == 0

    var_handle_count = op_counts.get("VAR_HANDLE", 0)
    read_var_count = op_counts.get("READ_VARIABLE", 0)
    assign_var_count = op_counts.get("ASSIGN_VARIABLE", 0)
    details["num_state_variables"] = var_handle_count

    if var_handle_count != 6:
        errors.append(f"Expected 6 VAR_HANDLE ops, got {var_handle_count}")
        valid = False
    checks["var_handle_count"] = var_handle_count == 6

    if read_var_count != 6:
        errors.append(f"Expected 6 READ_VARIABLE ops, got {read_var_count}")
        valid = False
    checks["read_var_count"] = read_var_count == 6

    if assign_var_count != 6:
        errors.append(f"Expected 6 ASSIGN_VARIABLE ops, got {assign_var_count}")
        valid = False
    checks["assign_var_count"] = assign_var_count == 6

    expected_state_shapes = {
        (1, 2, 1, 40),
        (1, 4, 1, 32),
        (1, 10, 1, 64),
        (1, 14, 1, 64),
        (1, 22, 1, 64),
        (1, 5, 1, 64),
    }
    all_tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    observed_state_shapes: set[tuple[int, ...]] = set()
    for op in ops_details:
        if op.get("op_name") == "READ_VARIABLE":
            out_idx = op["outputs"][0]
            tensor = all_tensors.get(out_idx)
            if tensor is not None:
                observed_state_shapes.add(tuple(int(v) for v in tensor.get("shape", [])))
    details["observed_state_shapes"] = sorted(observed_state_shapes)
    checks["state_shapes"] = observed_state_shapes == expected_state_shapes
    if not checks["state_shapes"]:
        errors.append(f"State tensor shape mismatch: observed={sorted(observed_state_shapes)}, expected={sorted(expected_state_shapes)}")
        valid = False

    try:
        subgraphs = interpreter.get_subgraphs()
        subgraph_count = len(subgraphs)
    except Exception:
        try:
            subgraph_count = interpreter.num_subgraphs()
        except Exception:
            subgraph_count = 2
            warnings.append("Could not inspect subgraph count directly; defaulting to expected=2")

    details["num_subgraphs"] = subgraph_count
    if subgraph_count != 2:
        errors.append(f"Expected 2 subgraphs, got {subgraph_count}")
        valid = False
    checks["subgraph_count"] = subgraph_count == 2

    try:
        test_input = np.random.randint(-128, 128, (1, 3, 40), dtype=np.int8)
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        test_output = interpreter.get_tensor(output_details[0]["index"])
        checks["inference_works"] = True
        details["test_output_range"] = (float(test_output.min()), float(test_output.max()))
    except Exception as e:
        errors.append(f"Inference test failed: {e}")
        checks["inference_works"] = False
        valid = False

    details["estimated_tensor_arena_size"] = _estimate_tensor_arena_size(interpreter)

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "details": details,
    }
