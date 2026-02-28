#!/usr/bin/env python3
"""
ESPHome Compatibility Verification Script for TFLite Wake Word Models

Verifies that a TFLite model meets ESPHome micro_wake_word requirements:
- 2 subgraphs (inference + initialization)
- Input shape [1, 3, 40] with dtype int8
- Output shape [1, 1] with dtype uint8
- 6 streaming state variables (TYPE_13 tensors)
- BUILTIN operations only (no custom ops)

Usage:
    python scripts/verify_esphome.py models/wake_word.tflite
    python scripts/verify_esphome.py models/wake_word.tflite --verbose
    python scripts/verify_esphome.py models/wake_word.tflite --json

Exit codes:
    0 - Model is fully compatible
    1 - Model has compatibility errors
    2 - Verification failed (exception)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def verify_esphome_compatibility(tflite_path: str, verbose: bool = False) -> dict[str, Any]:
    """
    Verify TFLite model is compatible with ESPHome micro_wake_word.

    Based on analysis of official ESPHome microWakeWord TFLite models (okay_nabu.tflite).

    Args:
        tflite_path: Path to TFLite model file
        verbose: Print detailed information during verification

    Returns:
        Dictionary with verification results:
        {
            "compatible": bool,
            "errors": list of error messages,
            "warnings": list of warning messages,
            "details": dict with detailed model info
        }
    """
    results: dict[str, Any] = {
        "compatible": True,
        "errors": [],
        "warnings": [],
        "details": {},
    }

    input_details = None
    output_details = None

    def vprint(*args: Any, **kwargs: Any) -> None:
        if verbose:
            print(*args, file=sys.stderr, **kwargs)

    try:
        # Load model
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()

        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()
        vprint(f"Loaded: {tflite_path}")

        # Check 1: Number of subgraphs (must be 2)
        try:
            model_obj = schema.ModelT.InitFromPackedBuf(tflite_bytes, 0)
            num_subgraphs = len(model_obj.subgraphs)
            results["details"]["num_subgraphs"] = num_subgraphs

            vprint(f"  Subgraphs: {num_subgraphs}")

            if num_subgraphs != 2:
                results["compatible"] = False
                results["errors"].append(f"Expected 2 subgraphs, got {num_subgraphs}. " "ESPHome requires dual-subgraph architecture (inference + init).")
        except Exception as e:
            results["compatible"] = False
            results["errors"].append(f"Could not verify subgraphs via flatbuffers schema: {e}")
        # Check 2: Input tensor shape and dtype
        try:
            input_details = interpreter.get_input_details()
            if input_details:
                input_info = input_details[0]
                input_shape = list(input_info["shape"])
                input_dtype = input_info["dtype"]

                results["details"]["input_shape"] = input_shape
                results["details"]["input_dtype"] = str(input_dtype)

                vprint(f"  Input shape: {input_shape}, dtype: {input_dtype}")

                # Try to determine expected stride from model metadata or manifest
                expected_strides = None
                tflite_dir = Path(tflite_path).parent

                # First, try to get stride from TFLite metadata
                try:
                    model = tf.lite.Interpreter(model_path=tflite_path)
                    metadata = model.get_tensor_metadata() if hasattr(model, "get_tensor_metadata") else None
                    if metadata and len(metadata) > 0:
                        # Try to extract stride from metadata description or name
                        meta = metadata[0]
                        if "description" in meta:
                            desc = meta["description"]
                            if "stride" in desc.lower():
                                import re

                                match = re.search(r"stride[=:]\s*(\d+)", desc, re.IGNORECASE)
                                if match:
                                    expected_strides = [int(match.group(1))]
                except Exception:  # noqa: S110
                    logging.debug("Metadata not available")  # noqa: S110, continue to other methods

                # If not found in metadata, check for manifest.json in same directory
                if expected_strides is None:
                    manifest_path = tflite_dir / "manifest.json"
                    if manifest_path.exists():
                        try:
                            import json

                            with open(manifest_path) as f:
                                manifest = json.load(f)
                            # Check if stride is stored in manifest (custom field)
                            stride_val = manifest.get("stride") or manifest.get("model", {}).get("stride")
                            if stride_val:
                                expected_strides = [int(stride_val)]
                        except Exception:  # noqa: S110
                            logging.debug("Could not read manifest.json")  # noqa: S110

                # Default: only accept stride 3 per ARCHITECTURAL_CONSTITUTION
                if expected_strides is None:
                    expected_strides = [3]

                # Validate input shape against expected stride(s)
                expected_shapes = [[1, s, 40] for s in expected_strides]
                if input_shape not in expected_shapes:
                    results["compatible"] = False
                    expected_strides_str = ", ".join(map(str, expected_strides))
                    results["errors"].append(f"Expected input shape [1, {{{expected_strides_str}}}, 40], got {input_shape}. " f"Valid shapes are: {expected_shapes}")

                # Must be int8
                if input_dtype != np.int8:
                    results["compatible"] = False
                    results["errors"].append(f"Expected input dtype int8, got {input_dtype}. " "ESPHome requires INT8 input quantization.")

                # Check quantization parameters
                quant_params = input_info.get("quantization_parameters", {})
                if quant_params:
                    scales = np.asarray(quant_params.get("scales", []))
                    zero_points = np.asarray(quant_params.get("zero_points", []))
                    if scales.size > 0:
                        results["details"]["input_scale"] = float(scales[0])
                    if zero_points.size > 0:
                        results["details"]["input_zero_point"] = int(zero_points[0])
            else:
                results["compatible"] = False
                results["errors"].append("No input tensors found")

        except Exception as e:
            results["warnings"].append(f"Could not verify input: {e}")

        # Check 3: Output tensor shape and dtype
        try:
            output_details = interpreter.get_output_details()
            if output_details:
                output_info = output_details[0]
                output_shape = list(output_info["shape"])
                output_dtype = output_info["dtype"]

                results["details"]["output_shape"] = output_shape
                results["details"]["output_dtype"] = str(output_dtype)

                vprint(f"  Output shape: {output_shape}, dtype: {output_dtype}")

                # Expected: [1, 1] (batch=1, single probability)
                if output_shape != [1, 1]:
                    results["compatible"] = False
                    results["errors"].append(f"Expected output shape [1, 1], got {output_shape}. " "ESPHome requires single probability output.")

                # CRITICAL: Must be uint8, NOT int8!
                if output_dtype != np.uint8:
                    results["compatible"] = False
                    results["errors"].append(f"Expected output dtype uint8, got {output_dtype}. " "CRITICAL: ESPHome requires UINT8 output, not int8!")

                # Check quantization parameters
                quant_params = output_info.get("quantization_parameters", {})
                if quant_params:
                    scales = np.asarray(quant_params.get("scales", []))
                    zero_points = np.asarray(quant_params.get("zero_points", []))
                    if scales.size > 0:
                        results["details"]["output_scale"] = float(scales[0])
                    if zero_points.size > 0:
                        results["details"]["output_zero_point"] = int(zero_points[0])
            else:
                results["compatible"] = False
                results["errors"].append("No output tensors found")

        except Exception as e:
            results["warnings"].append(f"Could not verify output: {e}")

        # Check 4: State variables (6 streaming state tensors)
        try:
            ops = interpreter._get_ops_details()
            var_handle_count = sum(1 for op in ops if op.get("op_name") == "VAR_HANDLE")
            num_state_vars = var_handle_count
            results["details"]["num_state_variables"] = num_state_vars

            vprint(f"  State variables: {num_state_vars}")
            if verbose:
                all_tensors = interpreter.get_tensor_details()
                for op in ops:
                    if op.get("op_name") == "READ_VARIABLE":
                        out_idx = op["outputs"][0]
                        t = all_tensors[out_idx]
                        vprint(f"    - {t.get('name', 'unnamed')}: {t.get('shape', [])}")

            # Expected: 6 state variables for streaming
            if num_state_vars != 6:
                results["compatible"] = False
                results["errors"].append(f"Expected 6 state variables, got {num_state_vars}. " "ESPHome streaming models require exactly 6 state tensors for ring buffer management.")
        except Exception as e:
            results["warnings"].append(f"Could not verify state variables: {e}")
        # Check 5: Verify quantization is present
        try:
            if input_details and output_details:
                input_quant = input_details[0].get("quantization_parameters", {})
                output_quant = output_details[0].get("quantization_parameters", {})

                input_scales = np.asarray(input_quant.get("scales", [])) if input_quant else np.array([])
                if input_scales.size == 0:
                    results["compatible"] = False
                    results["errors"].append("Input quantization parameters missing. ESPHome requires INT8 quantization.")
                output_scales = np.asarray(output_quant.get("scales", [])) if output_quant else np.array([])
                if output_scales.size == 0:
                    results["compatible"] = False
                    results["errors"].append("Output quantization parameters missing. ESPHome requires UINT8 quantization.")

        except Exception as e:
            results["warnings"].append(f"Could not verify quantization: {e}")

        # Calculate tensor arena size estimate
        try:
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

            # Add 30% overhead for intermediate tensors
            arena_size = int(total_memory * 1.3)
            results["details"]["estimated_tensor_arena_size"] = arena_size

            vprint(f"  Estimated tensor arena: {arena_size} bytes")

        except Exception as e:
            results["warnings"].append(f"Could not estimate tensor arena: {e}")

    except Exception as e:
        results["compatible"] = False
        results["errors"].append(f"Failed to load or analyze model: {e}")

    return results


def print_results(results: dict[str, Any], use_json: bool = False) -> None:
    """Print verification results in human-readable or JSON format."""
    if use_json:
        print(json.dumps(results, indent=2))
        return

    # Human-readable format
    print("\n" + "=" * 60)
    print("ESPHome Compatibility Verification Results")
    print("=" * 60)

    if results["compatible"]:
        print("\n✅ Model is ESPHome compatible!")
    else:
        print("\n❌ Model has compatibility errors")

    if results["errors"]:
        print("\nERRORS:")
        for error in results["errors"]:
            print(f"  ❌ {error}")

    if results["warnings"]:
        print("\nWARNINGS:")
        for warning in results["warnings"]:
            print(f"  ⚠️  {warning}")

    # Details
    details = results.get("details", {})
    if details:
        print("\nDETAILS:")
        if "num_subgraphs" in details:
            print(f"  Subgraphs: {details['num_subgraphs']}")
        if "input_shape" in details:
            print(f"  Input: {details['input_shape']} ({details.get('input_dtype', 'unknown')})")
        if "output_shape" in details:
            print(f"  Output: {details['output_shape']} ({details.get('output_dtype', 'unknown')})")
        if "num_state_variables" in details:
            print(f"  State variables: {details['num_state_variables']}")
        if "estimated_tensor_arena_size" in details:
            size = details["estimated_tensor_arena_size"]
            print(f"  Estimated tensor arena: {size} bytes ({size / 1024:.1f} KB)")

    print("\n" + "=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify TFLite model ESPHome compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.tflite
  %(prog)s model.tflite --verbose
  %(prog)s model.tflite --json
  %(prog)s model.tflite --verbose --json > results.json
        """,
    )
    parser.add_argument("model_path", type=str, help="Path to TFLite model file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information during verification",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Validate input file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: File not found: {model_path}", file=sys.stderr)
        return 2

    if not model_path.suffix == ".tflite":
        print("Warning: File does not have .tflite extension", file=sys.stderr)

    try:
        results = verify_esphome_compatibility(str(model_path), verbose=args.verbose)
        print_results(results, use_json=args.json)

        return 0 if results["compatible"] else 1

    except Exception as e:
        print(f"Error: Verification failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
