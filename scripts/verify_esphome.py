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
import sys
from pathlib import Path
from typing import Any

from src.export.verification import verify_tflite_model


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
    shared = verify_tflite_model(tflite_path)
    details = dict(shared.get("details", {}))

    if verbose:
        print(f"Loaded: {tflite_path}", file=sys.stderr)
        if "num_subgraphs" in details:
            print(f"  Subgraphs: {details['num_subgraphs']}", file=sys.stderr)
        if "input_shape" in details and "input_dtype" in details:
            print(f"  Input shape: {details['input_shape']}, dtype: {details['input_dtype']}", file=sys.stderr)
        if "output_shape" in details and "output_dtype" in details:
            print(f"  Output shape: {details['output_shape']}, dtype: {details['output_dtype']}", file=sys.stderr)
        if "num_state_variables" in details:
            print(f"  State variables: {details['num_state_variables']}", file=sys.stderr)
        if "observed_state_shapes" in details:
            for shape in details["observed_state_shapes"]:
                print(f"    - state tensor shape: {shape}", file=sys.stderr)
        if "estimated_tensor_arena_size" in details:
            print(f"  Estimated tensor arena: {details['estimated_tensor_arena_size']} bytes", file=sys.stderr)

    return {
        "compatible": bool(shared.get("valid", False)),
        "errors": list(shared.get("errors", [])),
        "warnings": list(shared.get("warnings", [])),
        "details": details,
    }


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
