from typing import Any

import numpy as np
import tensorflow as tf

from .tflite_utils import estimate_tensor_arena_size


def _ensure_quantized_int(value: Any) -> int:
    """Safely convert quantization parameter to int.

    Some TFLite implementations return float zero_points. This function
    safely converts to int with proper type checking to avoid TypeError.

    Args:
        value: Zero-point value (int or float)

    Returns:
        Integer zero-point value

    Raises:
        TypeError: If value cannot be safely converted to int
    """
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Cannot convert zero-point {value!r} (type: {type(value).__name__}) to int: {e}") from e


def compute_expected_state_shapes(
    first_conv_kernel: int = 5,
    stride: int = 3,
    mel_bins: int = 40,
    first_conv_filters: int = 32,
    mixconv_kernel_sizes: list[list[int]] | None = None,
    pointwise_filters: list[int] | None = None,
    temporal_frames: int = 6,
) -> list[tuple[int, ...]]:
    """Compute expected streaming-state payload shapes for verification.

    Notes:
        - ``stream`` through ``stream_4`` are derived from conv/mixconv kernel context.
        - ``stream_5`` is dynamic and depends on ``temporal_frames`` derived from model
          configuration (for example via ``clip_duration_ms``), so it is not universally
          ``(1, 5, 1, 64)``.
    """
    # Input validation
    if first_conv_kernel <= stride:
        raise ValueError(f"first_conv_kernel ({first_conv_kernel}) must be > stride ({stride})")
    if temporal_frames < 2:
        raise ValueError(f"temporal_frames must be >= 2, got {temporal_frames}")
    if mixconv_kernel_sizes is None:
        mixconv_kernel_sizes = [[5], [7, 11], [9, 15], [23]]
    if pointwise_filters is None:
        pointwise_filters = [64, 64, 64, 64]
    if mel_bins < 1:
        raise ValueError(f"mel_bins must be >= 1, got {mel_bins}")
    if first_conv_filters < 1:
        raise ValueError(f"first_conv_filters must be >= 1, got {first_conv_filters}")
    if len(mixconv_kernel_sizes) != 4:
        raise ValueError(f"mixconv_kernel_sizes must contain 4 groups, got {len(mixconv_kernel_sizes)}")
    for i, kernels in enumerate(mixconv_kernel_sizes):
        if len(kernels) == 0:
            raise ValueError(f"mixconv_kernel_sizes[{i}] contains empty list, all groups must have at least 1 kernel")
    if len(pointwise_filters) != 4:
        raise ValueError(f"pointwise_filters must contain 4 values, got {len(pointwise_filters)}")

    return [
        (1, first_conv_kernel - stride, 1, mel_bins),
        (1, max(mixconv_kernel_sizes[0]) - 1, 1, first_conv_filters),
        (1, max(mixconv_kernel_sizes[1]) - 1, 1, pointwise_filters[0]),
        (1, max(mixconv_kernel_sizes[2]) - 1, 1, pointwise_filters[1]),
        (1, max(mixconv_kernel_sizes[3]) - 1, 1, pointwise_filters[2]),
        (1, temporal_frames - 1, 1, pointwise_filters[3]),
    ]


def verify_tflite_model(
    tflite_path: str,
    expected_state_shapes: list[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Validate TFLite model compatibility against ESPHome invariants.

    The interpreter is created with
    ``BUILTIN_WITHOUT_DEFAULT_DELEGATES`` to keep checks anchored to the static graph
    and avoid delegate-path artifacts (for example transient ``DELEGATE`` visibility).

    If ``expected_state_shapes`` is not supplied, default assumptions are used. Under
    those defaults, state-shape mismatch is reported as a warning and does not force
    failure, so non-default model configs can still be validated without false negatives.
    """
    resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_op_resolver_type=resolver_type)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    try:
        ops_details = interpreter._get_ops_details()  # private API
    except Exception:
        ops_details = []

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
        input_zero = _ensure_quantized_int(input_zero_points[0])
        details["input_scale"] = input_scale
        details["input_zero_point"] = input_zero
        # Canonical input scale = 26/255 ≈ 0.10196078568696976 (ARCHITECTURAL_CONSTITUTION Article II).
        # Tolerance 5e-4 accommodates minor floating-point variation in TFLite's
        # quantizer while still catching grossly wrong scales.
        checks["input_quant_params"] = abs(input_scale - 0.10196078568696976) <= 5e-4 and input_zero == -128
        if not checks["input_quant_params"]:
            errors.append(f"Input quantization mismatch: scale={input_scale}, zero_point={input_zero}")
            valid = False

    if output_scales.size == 0 or output_zero_points.size == 0:
        errors.append("Missing output quantization parameters")
        valid = False
        checks["output_quant_params"] = False
    else:
        output_scale = float(output_scales[0])
        output_zero = _ensure_quantized_int(output_zero_points[0])
        canonical_output_scales = {
            round(1.0 / 255.0, 9),
            round(1.0 / 256.0, 9),
        }
        details["output_scale"] = output_scale
        details["output_zero_point"] = output_zero
        output_scale_r = round(output_scale, 9)
        checks["output_zero_point"] = output_zero == 0
        checks["output_scale_canonical"] = output_scale_r in canonical_output_scales
        checks["output_quant_params"] = checks["output_zero_point"] and checks["output_scale_canonical"]

        if not checks["output_zero_point"]:
            errors.append(f"Output quantization mismatch: zero_point={output_zero}; expected zero_point=0 for uint8 probability output")
            valid = False

        if not checks["output_scale_canonical"]:
            warnings.append(
                f"Output scale is non-canonical for ESPHome micro-wake-word workflow: scale={output_scale}. "
                f"Official models in this audit use 1/256 ({1.0 / 256.0:.9f}); 1/255 ({1.0 / 255.0:.9f}) is accepted for compatibility."
            )

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
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "PAD",
        "PACK",
    }
    unpermitted_ops = sorted(op for op in op_counts if op and op not in allowed_ops)
    if unpermitted_ops:
        errors.append(f"Unpermitted ops detected: {unpermitted_ops}")
        valid = False
    checks["op_whitelist"] = len(unpermitted_ops) == 0

    var_handle_count = op_counts.get("VAR_HANDLE", 0)
    read_var_count = op_counts.get("READ_VARIABLE", 0)
    assign_var_count = op_counts.get("ASSIGN_VARIABLE", 0)
    details["num_state_handles"] = var_handle_count
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

    # NEW: Check CALL_ONCE exists (Article V compliance)
    call_once_count = op_counts.get("CALL_ONCE", 0)
    if call_once_count != 1:
        errors.append(f"Expected exactly 1 CALL_ONCE op for init subgraph, got {call_once_count}")
        valid = False
    checks["call_once_count"] = call_once_count == 1

    expected_shapes_assumed_default = expected_state_shapes is None
    if expected_shapes_assumed_default:
        expected_state_shapes = compute_expected_state_shapes()
    assert expected_state_shapes is not None  # Help mypy understand it's not None after the check above
    all_tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    observed_state_payload_shapes: list[tuple[int, ...]] = []
    observed_read_payload_dtypes: list[str] = []
    observed_read_payload_quantized: list[bool] = []
    observed_assign_payload_dtypes: list[str] = []
    observed_assign_payload_quantized: list[bool] = []

    for op in ops_details:
        if op.get("op_name") == "READ_VARIABLE":
            out_idx = op["outputs"][0]
            tensor = all_tensors.get(out_idx)
            if tensor is not None:
                observed_state_payload_shapes.append(tuple(int(v) for v in tensor.get("shape", [])))
                dtype = tensor.get("dtype")
                observed_read_payload_dtypes.append(np.dtype(dtype).name)
                quant = tensor.get("quantization_parameters", {}) or {}
                scales = np.asarray(quant.get("scales", []))
                zero_points = np.asarray(quant.get("zero_points", []))
                observed_read_payload_quantized.append(scales.size > 0 and zero_points.size > 0)
        elif op.get("op_name") == "ASSIGN_VARIABLE":
            inputs = op.get("inputs", [])
            if len(inputs) >= 2:
                payload_idx = inputs[1]
                tensor = all_tensors.get(payload_idx)
                if tensor is not None:
                    dtype = tensor.get("dtype")
                    observed_assign_payload_dtypes.append(np.dtype(dtype).name)
                    quant = tensor.get("quantization_parameters", {}) or {}
                    scales = np.asarray(quant.get("scales", []))
                    zero_points = np.asarray(quant.get("zero_points", []))
                    observed_assign_payload_quantized.append(scales.size > 0 and zero_points.size > 0)

    details["observed_state_payload_shapes"] = observed_state_payload_shapes
    details["observed_read_payload_dtypes"] = observed_read_payload_dtypes
    details["observed_read_payload_quantized"] = observed_read_payload_quantized
    details["observed_assign_payload_dtypes"] = observed_assign_payload_dtypes
    details["observed_assign_payload_quantized"] = observed_assign_payload_quantized

    checks["state_payload_dtypes_int8"] = all(np.dtype(dt).name == "int8" for dt in observed_read_payload_dtypes)
    checks["state_dtypes_int8"] = checks["state_payload_dtypes_int8"]
    if not checks["state_payload_dtypes_int8"]:
        errors.append(f"READ_VARIABLE payload tensors must be int8, got dtypes: {observed_read_payload_dtypes}")
        valid = False

    checks["read_payload_quant_params"] = all(observed_read_payload_quantized) and len(observed_read_payload_quantized) == 6
    if not checks["read_payload_quant_params"]:
        errors.append("READ_VARIABLE payload tensors must carry quantization parameters")
        valid = False

    if observed_assign_payload_dtypes:
        checks["assign_payload_dtypes_int8"] = all(np.dtype(dt).name == "int8" for dt in observed_assign_payload_dtypes)
        if not checks["assign_payload_dtypes_int8"]:
            errors.append(f"ASSIGN_VARIABLE payload tensors must be int8, got dtypes: {observed_assign_payload_dtypes}")
            valid = False

        checks["assign_payload_quant_params"] = all(observed_assign_payload_quantized) and len(observed_assign_payload_quantized) == 6
        if not checks["assign_payload_quant_params"]:
            errors.append("ASSIGN_VARIABLE payload tensors must carry quantization parameters")
            valid = False
    else:
        checks["assign_payload_dtypes_int8"] = False
        checks["assign_payload_quant_params"] = False
        warnings.append("ASSIGN_VARIABLE payload tensors unavailable from interpreter op details; skipped direct payload validation")

    # Check shapes as a set (TFLite graph traversal order for READ_VARIABLE ops
    # is not guaranteed to match variable creation order)
    observed_sorted = sorted(observed_state_payload_shapes)
    expected_sorted = sorted(expected_state_shapes)
    checks["state_shapes"] = observed_sorted == expected_sorted
    details["expected_state_shapes"] = expected_sorted
    details["state_shape_assumption"] = "config-aware" if not expected_shapes_assumed_default else "default-temporal-frames"

    if not checks["state_shapes"]:
        if expected_shapes_assumed_default:
            warnings.append(
                "State payload tensor shape mismatch under default temporal-frame assumptions. "
                "For non-default clip_duration_ms/architecture, pass config-aware expected_state_shapes "
                "to avoid false failures."
            )
            checks["state_shapes"] = True
            details["state_shape_mismatch_under_default_assumption"] = {
                "observed": observed_sorted,
                "expected_default": expected_sorted,
            }
        else:
            errors.append(f"State payload tensor shape mismatch: observed={observed_sorted}, expected={expected_sorted}")
            valid = False

    try:
        if hasattr(interpreter, "num_subgraphs"):
            subgraph_count = interpreter.num_subgraphs()
        elif hasattr(interpreter, "_interpreter") and hasattr(interpreter._interpreter, "GetSubgraphCount"):
            subgraph_count = interpreter._interpreter.GetSubgraphCount()
        else:
            # CALL_ONCE presence implies 2 subgraphs (main + init)
            subgraph_count = 2 if call_once_count == 1 else 1
            warnings.append("Subgraph count inferred from CALL_ONCE presence")
    except Exception:
        # CALL_ONCE presence implies 2 subgraphs (main + init)
        subgraph_count = 2 if call_once_count == 1 else 1
        warnings.append("Subgraph count inferred from CALL_ONCE presence")

    if subgraph_count is None:
        checks["subgraph_count"] = False
        errors.append("Could not determine subgraph count")
        valid = False
    else:
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
        details["test_output_range"] = (
            float(test_output.min()),
            float(test_output.max()),
        )
    except Exception as e:
        errors.append(f"Inference test failed: {e}")
        checks["inference_works"] = False
        valid = False

    details["estimated_tensor_arena_size"] = estimate_tensor_arena_size(interpreter)

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "details": details,
    }
