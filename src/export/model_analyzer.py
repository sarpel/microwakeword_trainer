"""Model analysis and validation using ai_edge_litert (formerly TF Lite)."""

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# =============================================================================
# MODEL ARCHITECTURE ANALYSIS
# =============================================================================


def analyze_model_architecture(model_path: str) -> dict[str, Any]:
    """Analyze detailed architecture of a TFLite model using ai_edge_litert.

    Uses ai_edge_litert.Interpreter for model analysis.

    Args:
        model_path: Path to the TFLite model file

    Returns:
        Dictionary containing detailed architecture analysis:
        - layer_count: Number of layers/operators
        - operators: List of operator types used
        - input_tensors: Input tensor details
        - output_tensors: Output tensor details
        - subgraph_count: Number of subgraphs
        - has_quantization: Whether model uses quantization
        - analysis_text: Raw analysis output
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model_content = f.read()

    analysis_text = tf.lite.experimental.Analyzer.analyze(
        model_content=model_content,
        gpu_compatibility=False,
    )

    result = _parse_analysis_output(analysis_text, model_content)
    result["analysis_text"] = analysis_text
    result["model_path"] = model_path
    result["model_size_bytes"] = len(model_content)

    return result


def _parse_analysis_output(analysis_text: str, model_content: bytes) -> dict[str, Any]:
    """Parse the raw analysis output into structured data."""
    result: dict[str, Any] = {
        "layer_count": 0,
        "operators": [],
        "operator_counts": {},
        "input_tensors": [],
        "output_tensors": [],
        "subgraph_count": 1,
        "has_quantization": False,
        "total_parameters": 0,
    }

    result["has_quantization"] = "quantization" in analysis_text.lower() or "int8" in analysis_text.lower()

    subgraph_matches = re.findall(r"Subgraph#(\d+)", analysis_text)
    if subgraph_matches:
        result["subgraph_count"] = len(set(subgraph_matches))

    operator_pattern = r"Op#\d+\s+\((\w+)\)"
    operators = re.findall(operator_pattern, analysis_text)
    result["operators"] = list(set(operators))
    result["layer_count"] = len(operators)

    for op in operators:
        result["operator_counts"][op] = result["operator_counts"].get(op, 0) + 1

    tensor_pattern = r"T#\d+\s+(\w+)\s+\[(\d+(?:,\s*\d+)*)\]\s+(\w+)"
    tensors = re.findall(tensor_pattern, analysis_text)

    for tensor_type, shape, dtype in tensors[:10]:
        tensor_info = {
            "type": tensor_type,
            "shape": [int(x.strip()) for x in shape.split(",")],
            "dtype": dtype,
        }
        if tensor_type == "INPUT":
            result["input_tensors"].append(tensor_info)
        elif tensor_type == "OUTPUT":
            result["output_tensors"].append(tensor_info)

    model_size_mb = len(model_content) / (1024 * 1024)
    if result["has_quantization"]:
        result["total_parameters"] = int(model_size_mb * 4 * 1024 * 1024 / 4)
    else:
        result["total_parameters"] = int(model_size_mb * 1024 * 1024 / 4)

    return result


# =============================================================================
# MODEL QUALITY VALIDATION
# =============================================================================


def validate_model_quality(
    model_path: str,
    expected_input_shape: tuple[int, ...] | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
    expected_input_dtype: np.dtype | type[np.integer] | None = None,
    expected_output_dtype: np.dtype | type[np.integer] | None = None,
    stride: int = 3,
    mel_bins: int = 40,
) -> dict[str, Any]:
    """Validate TFLite model quality and health metrics."""
    results: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }

    if not os.path.exists(model_path):
        results["valid"] = False
        results["errors"].append(f"Model file not found: {model_path}")
        return results

    if expected_input_shape is None:
        expected_input_shape = (1, stride, mel_bins)
    if expected_output_shape is None:
        expected_output_shape = (1, 1)
    if expected_input_dtype is None:
        expected_input_dtype = np.int8
    if expected_output_dtype is None:
        expected_output_dtype = np.uint8

    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        num_subgraphs = interpreter.num_subgraphs()

        if input_details:
            inp = input_details[0]
            inp_shape = tuple(inp["shape"])
            inp_dtype = inp["dtype"]

            valid_input_shapes = [expected_input_shape, (1, 1, mel_bins)]
            if inp_shape not in valid_input_shapes:
                results["warnings"].append(f"Unexpected input shape: {inp_shape}. Expected one of {valid_input_shapes}")

            if inp_dtype != expected_input_dtype:
                results["valid"] = False
                results["errors"].append(f"Invalid input dtype: {inp_dtype}. Expected {expected_input_dtype}")
            else:
                results["info"]["input_dtype_correct"] = True

            if "quantization_parameters" in inp:
                qp = inp["quantization_parameters"]
                if qp.get("scales") is not None and qp.get("zero_points") is not None and len(qp["scales"]) > 0 and len(qp["zero_points"]) > 0:
                    results["info"]["input_quantized"] = True
                    results["info"]["input_scale"] = float(qp["scales"][0])
                    results["info"]["input_zero_point"] = int(qp["zero_points"][0])

        if output_details:
            out = output_details[0]
            out_shape = tuple(out["shape"])
            out_dtype = out["dtype"]

            if out_shape != expected_output_shape:
                results["warnings"].append(f"Unexpected output shape: {out_shape}. Expected {expected_output_shape}")

            if out_dtype != expected_output_dtype:
                results["valid"] = False
                results["errors"].append(f"Invalid output dtype: {out_dtype}. Expected {expected_output_dtype} " "(CRITICAL for ESPHome compatibility)")
            else:
                results["info"]["output_dtype_correct"] = True

            if "quantization_parameters" in out:
                qp = out["quantization_parameters"]
                if qp.get("scales") is not None and qp.get("zero_points") is not None and len(qp["scales"]) > 0 and len(qp["zero_points"]) > 0:
                    results["info"]["output_quantized"] = True
                    results["info"]["output_scale"] = float(qp["scales"][0])
                    results["info"]["output_zero_point"] = int(qp["zero_points"][0])

        if num_subgraphs != 2:
            results["warnings"].append(f"Expected 2 subgraphs for streaming model, found {num_subgraphs}")
        else:
            results["info"]["subgraph_count_correct"] = True
            results["info"]["subgraph_count"] = num_subgraphs

        tensor_details = interpreter.get_tensor_details()
        results["info"]["total_tensors"] = len(tensor_details)

        state_vars = [t for t in tensor_details if "state" in t.get("name", "").lower()]
        results["info"]["state_variables"] = len(state_vars)

        model_size_bytes = os.path.getsize(model_path)
        model_size_kb = model_size_bytes / 1024
        results["info"]["model_size_kb"] = round(model_size_kb, 2)

        if model_size_kb > 100:
            results["warnings"].append(f"Model size ({model_size_kb:.1f} KB) is larger than typical for wake word models")

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Validation failed: {str(e)}")

    return results


# =============================================================================
# MODEL COMPARISON
# =============================================================================


def compare_models(
    model_path_a: str,
    model_path_b: str,
    stride: int = 3,
    mel_bins: int = 40,
) -> dict[str, Any]:
    """Compare two model versions to identify differences."""
    analysis_a = analyze_model_architecture(model_path_a)
    analysis_b = analyze_model_architecture(model_path_b)

    validation_a = validate_model_quality(model_path_a, stride=stride, mel_bins=mel_bins)
    validation_b = validate_model_quality(model_path_b, stride=stride, mel_bins=mel_bins)

    size_a = analysis_a.get("model_size_bytes", 0)
    size_b = analysis_b.get("model_size_bytes", 0)
    size_diff = size_b - size_a
    size_diff_percent = (size_diff / size_a * 100) if size_a > 0 else 0

    layer_diff = analysis_b.get("layer_count", 0) - analysis_a.get("layer_count", 0)

    ops_a = set(analysis_a.get("operators", []))
    ops_b = set(analysis_b.get("operators", []))
    ops_added = ops_b - ops_a
    ops_removed = ops_a - ops_b

    inp_a = validation_a.get("info", {}).get("input_dtype_correct", False)
    inp_b = validation_b.get("info", {}).get("input_dtype_correct", False)
    out_a = validation_a.get("info", {}).get("output_dtype_correct", False)
    out_b = validation_b.get("info", {}).get("output_dtype_correct", False)

    compatible = inp_a == inp_b and out_a == out_b and validation_a["valid"] and validation_b["valid"]

    return {
        "model_a": {
            "path": model_path_a,
            "size_bytes": size_a,
            "layers": analysis_a.get("layer_count", 0),
            "operators": analysis_a.get("operators", []),
            "quantized": analysis_a.get("has_quantization", False),
            "validation_valid": validation_a["valid"],
        },
        "model_b": {
            "path": model_path_b,
            "size_bytes": size_b,
            "layers": analysis_b.get("layer_count", 0),
            "operators": analysis_b.get("operators", []),
            "quantized": analysis_b.get("has_quantization", False),
            "validation_valid": validation_b["valid"],
        },
        "differences": {
            "size_diff_bytes": size_diff,
            "size_diff_percent": round(size_diff_percent, 2),
            "layer_diff": layer_diff,
            "operators_added": list(ops_added),
            "operators_removed": list(ops_removed),
            "common_operators": list(ops_a & ops_b),
        },
        "compatibility": {
            "interfaces_compatible": compatible,
            "model_a_valid": validation_a["valid"],
            "model_b_valid": validation_b["valid"],
            "input_dtype_match": inp_a == inp_b,
            "output_dtype_match": out_a == out_b,
        },
    }


# =============================================================================
# PERFORMANCE ESTIMATION
# =============================================================================


def estimate_performance(
    model_path: str,
    stride: int = 3,
    mel_bins: int = 40,
) -> dict[str, Any]:
    """Estimate performance metrics for the TFLite model."""
    results: dict[str, Any] = {
        "model_size_kb": 0,
        "tensor_arena_estimate": 0,
        "estimated_latency_ms": 0,
        "estimated_memory_kb": 0,
        "optimization_recommendations": [],
    }

    if not os.path.exists(model_path):
        results["error"] = f"Model not found: {model_path}"
        return results

    try:
        model_size_bytes = os.path.getsize(model_path)
        results["model_size_kb"] = round(model_size_bytes / 1024, 2)

        analysis = analyze_model_architecture(model_path)

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()

        total_tensor_memory = 0
        for tensor in tensor_details:
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

            num_elements = max(1, np.prod([max(1, abs(d)) for d in shape]))
            total_tensor_memory += num_elements * elem_size

        tensor_arena = int(total_tensor_memory * 1.3)
        results["tensor_arena_estimate"] = tensor_arena
        results["tensor_arena_estimate_kb"] = round(tensor_arena / 1024, 2)

        total_memory = model_size_bytes + tensor_arena + 4096
        results["estimated_memory_kb"] = round(total_memory / 1024, 2)

        layer_count = analysis.get("layer_count", 0)
        operators = analysis.get("operators", [])

        base_latency = 5.0
        per_layer_latency = 0.1

        if analysis.get("has_quantization", False):
            base_latency *= 0.5
            per_layer_latency *= 0.7

        expensive_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]
        expensive_count = sum(1 for op in expensive_ops if op in operators)
        if expensive_count > 0:
            base_latency += expensive_count * 0.5

        estimated_latency = base_latency + (layer_count * per_layer_latency)
        results["estimated_latency_ms"] = round(estimated_latency, 2)

        recommendations = []

        if model_size_bytes > 50 * 1024:
            recommendations.append("Consider reducing model complexity for smaller footprint")

        if not analysis.get("has_quantization", False):
            recommendations.append("Enable INT8 quantization to reduce size and improve latency")

        if layer_count > 20:
            recommendations.append("Consider reducing number of layers for faster inference")

        if estimated_latency > 50:
            recommendations.append(f"Estimated latency ({estimated_latency:.1f}ms) may be high for real-time wake word")

        if tensor_arena > 30 * 1024:
            recommendations.append(f"Consider reducing tensor arena size ({tensor_arena // 1024}KB) if memory constrained")

        if not recommendations:
            recommendations.append("Model appears well-optimized for edge deployment")

        results["optimization_recommendations"] = recommendations

        results["metadata"] = {
            "layer_count": layer_count,
            "operator_count": len(operators),
            "operators": operators,
            "quantized": analysis.get("has_quantization", False),
            "input_shape": (1, stride, mel_bins),
        }

    except Exception as e:
        results["error"] = str(e)

    return results


# =============================================================================
# GPU COMPATIBILITY CHECK
# =============================================================================


def check_gpu_compatibility(model_path: str) -> dict[str, Any]:
    """Check if model is compatible with GPU delegation."""
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    with open(model_path, "rb") as f:
        model_content = f.read()

    analysis_text = tf.lite.experimental.Analyzer.analyze(
        model_content=model_content,
        gpu_compatibility=True,
    )

    # Improved GPU compatibility check with positive indicators and negation handling
    # Define positive indicators (must appear without negation nearby)
    positive_indicators = [
        r"GPU compatible",
        r"GPU supported",
        r"supports GPU",
        r"uses GPU",
    ]

    # Define negative/negation patterns that indicate non-compatibility
    negative_patterns = [
        r"no GPU",
        r"not supported",
        r"unsupported",
        r"failed",
        r"doesn.?t support",
        r"does not support",
        r"no support for GPU",
        r"cannot use GPU",
    ]

    # Check for positive indicators
    has_positive = False
    for pattern in positive_indicators:
        if re.search(re.escape(pattern), analysis_text, re.IGNORECASE):
            # Check if there's a negation nearby (within 50 characters)
            for neg_pattern in negative_patterns:

                # Find all occurrences of the positive pattern
                for match in re.finditer(re.escape(pattern), analysis_text, re.IGNORECASE):
                    pos_start = match.start()
                    pos_end = match.end()
                    # Check surrounding context (before and after)
                    context_start = max(0, pos_start - 50)
                    context_end = min(len(analysis_text), pos_end + 50)
                    context = analysis_text[context_start:context_end].lower()

                    # If negation found in context, this positive indicator is negated
                    if re.search(neg_pattern, context):
                        break
                else:
                    # No negation found for this positive indicator
                    has_positive = True
                    break
            if has_positive:
                break

    # Fallback: if no positive indicators, check for explicit negative mentions
    has_negative = any(
        neg in analysis_text.lower()
        for neg in [
            "no gpu",
            "not supported",
            "unsupported",
            "failed",
            "doesn't support",
        ]
    )

    gpu_compatible = has_positive and not has_negative

    issues = []

    if "not supported" in analysis_text.lower():
        issues.append("Some operations may not be supported on GPU")

    if "quantization" in analysis_text.lower():
        issues.append("Quantized models may have limited GPU delegate support")

    return {
        "gpu_compatible": gpu_compatible,
        "issues": issues,
        "analysis": analysis_text,
    }


# =============================================================================
# FULL MODEL REPORT
# =============================================================================


def generate_model_report(model_path: str, stride: int = 3, mel_bins: int = 40) -> dict[str, Any]:
    """Generate a comprehensive model analysis report."""
    import logging as _logging

    _log = _logging.getLogger(__name__)

    try:
        architecture = analyze_model_architecture(model_path)
    except Exception as e:
        _log.warning("analyze_model_architecture failed for '%s': %s", model_path, e)
        architecture = {"error": str(e), "layer_count": 0, "has_quantization": False}

    validation = validate_model_quality(model_path, stride=stride, mel_bins=mel_bins)
    performance = estimate_performance(model_path, stride=stride, mel_bins=mel_bins)
    gpu_check = check_gpu_compatibility(model_path)

    return {
        "model_path": model_path,
        "model_name": Path(model_path).name,
        "architecture": architecture,
        "validation": validation,
        "performance": performance,
        "gpu_compatibility": gpu_check,
        "summary": {
            "valid": validation.get("valid", False),
            "size_kb": performance.get("model_size_kb", 0),
            "estimated_latency_ms": performance.get("estimated_latency_ms", 0),
            "tensor_arena_kb": performance.get("tensor_arena_estimate_kb", 0),
            "quantized": architecture.get("has_quantization", False),
            "layer_count": architecture.get("layer_count", 0),
        },
    }
