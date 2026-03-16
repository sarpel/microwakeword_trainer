#!/usr/bin/env python3
"""Streaming Equivalence Gate for TFLite Wake Word Models

Validates that the exported streaming TFLite model behaves correctly as a
stateful streaming model:

  1. Smoke test   — model runs, outputs are valid uint8 values
  2. Determinism  — identical input sequences produce identical outputs across
                    two independent interpreter instances
  3. State change — the model's internal ring-buffer state evolves over time
                    (output is NOT the same on every frame)
  4. Boundary     — model handles edge inputs (all-zeros, max-value frames)

Usage:
    python scripts/verify_streaming.py models/exported/wake_word.tflite
    python scripts/verify_streaming.py models/exported/wake_word.tflite --frames 20 --verbose
    python scripts/verify_streaming.py models/exported/wake_word.tflite --json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# TFLite interpreter helpers
# ---------------------------------------------------------------------------


def _make_interpreter(tflite_path: str):
    """Create and allocate a TFLite interpreter for the given model path."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    return interp


def _run_frame(interp, frame_int8: np.ndarray) -> int:
    """Run one [1, 3, 40] int8 frame through the streaming interpreter.

    Returns the uint8 output scalar (0–255).
    """
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.set_tensor(input_details[0]["index"], frame_int8)
    interp.invoke()
    output = interp.get_tensor(output_details[0]["index"])
    return int(output.ravel()[0])


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def _smoke_test(tflite_path: str, frames: list[np.ndarray], verbose: bool) -> dict:
    """Run all frames through a fresh interpreter and check output range."""
    interp = _make_interpreter(tflite_path)
    outputs = [_run_frame(interp, f) for f in frames]

    invalid = [v for v in outputs if v < 0 or v > 255]
    passed = len(invalid) == 0

    if verbose:
        print(f"  [smoke] outputs: {outputs}")
        if invalid:
            print(f"  [smoke] INVALID values: {invalid}")

    return {
        "passed": passed,
        "outputs": outputs,
        "invalid_count": len(invalid),
        "error": f"Got {len(invalid)} out-of-range uint8 values" if not passed else None,
    }


def _determinism_test(tflite_path: str, frames: list[np.ndarray], verbose: bool) -> dict:
    """Two independent interpreters with the same input must produce the same outputs."""
    interp_a = _make_interpreter(tflite_path)
    interp_b = _make_interpreter(tflite_path)

    outputs_a = [_run_frame(interp_a, f) for f in frames]
    outputs_b = [_run_frame(interp_b, f) for f in frames]

    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(outputs_a, outputs_b, strict=False)) if a != b]
    passed = len(mismatches) == 0

    if verbose:
        print(f"  [determinism] run A: {outputs_a}")
        print(f"  [determinism] run B: {outputs_b}")
        if mismatches:
            for i, a, b in mismatches:
                print(f"  [determinism] frame {i}: A={a} B={b}")

    return {
        "passed": passed,
        "mismatch_count": len(mismatches),
        "error": f"{len(mismatches)} frames produced different outputs" if not passed else None,
    }


def _state_change_test(tflite_path: str, frames: list[np.ndarray], verbose: bool) -> dict:
    """Outputs must not all be identical — the model must update its ring-buffer state."""
    interp = _make_interpreter(tflite_path)
    outputs = [_run_frame(interp, f) for f in frames]

    unique_outputs = len(set(outputs))
    # With random int8 input the model *should* produce different outputs across frames.
    # We allow a 1-unique-output case only if there are very few frames.
    passed = unique_outputs > 1 or len(frames) <= 2

    if verbose:
        print(f"  [state_change] outputs: {outputs}")
        print(f"  [state_change] unique output values: {unique_outputs}")

    return {
        "passed": passed,
        "unique_outputs": unique_outputs,
        "total_frames": len(frames),
        "error": "All frames produced identical output — ring buffer may not be updating" if not passed else None,
    }


def _boundary_test(tflite_path: str, verbose: bool) -> dict:
    """Feed all-zeros and all-max frames; model should not crash or produce NaN."""
    errors = []

    for label, value in [("all_zeros", 0), ("all_max", 127), ("all_min", -128)]:
        interp = _make_interpreter(tflite_path)
        frame = np.full((1, 3, 40), value, dtype=np.int8)
        try:
            out = _run_frame(interp, frame)
            if not (0 <= out <= 255):
                errors.append(f"{label}: output {out} is out of uint8 range")
            elif verbose:
                print(f"  [boundary] {label}: output = {out}")
        except Exception as exc:
            errors.append(f"{label}: crashed with {exc}")

    return {
        "passed": len(errors) == 0,
        "error": "; ".join(errors) if errors else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gate(tflite_path: str, n_frames: int = 15, seed: int = 42, verbose: bool = False) -> dict:
    """Run the full streaming equivalence gate.

    Args:
        tflite_path: Path to the TFLite model.
        n_frames: Number of random frames to test.
        seed: RNG seed for reproducibility.
        verbose: Print per-check details.

    Returns:
        Dict with keys: passed, checks (dict per check), summary.
    """
    rng = np.random.default_rng(seed)
    frames = [rng.integers(-128, 128, size=(1, 3, 40), dtype=np.int8) for _ in range(n_frames)]

    checks: dict = {}

    if verbose:
        print("\n[Streaming Gate] smoke test …")
    checks["smoke"] = _smoke_test(tflite_path, frames, verbose)

    if verbose:
        print("[Streaming Gate] determinism test …")
    checks["determinism"] = _determinism_test(tflite_path, frames, verbose)

    if verbose:
        print("[Streaming Gate] state change test …")
    checks["state_change"] = _state_change_test(tflite_path, frames, verbose)

    if verbose:
        print("[Streaming Gate] boundary test …")
    checks["boundary"] = _boundary_test(tflite_path, verbose)

    all_passed = all(c["passed"] for c in checks.values())
    failed = [name for name, c in checks.items() if not c["passed"]]

    return {
        "passed": all_passed,
        "checks": checks,
        "failed_checks": failed,
        "n_frames": n_frames,
        "model": str(tflite_path),
    }


def print_results(results: dict, use_json: bool = False) -> None:
    if use_json:
        # Remove numpy arrays from outputs before JSON serialization
        import copy

        r = copy.deepcopy(results)
        for c in r.get("checks", {}).values():
            if "outputs" in c:
                c["outputs"] = list(c["outputs"])
        print(json.dumps(r, indent=2))
        return

    print("\n" + "=" * 60)
    print("Streaming Equivalence Gate Results")
    print("=" * 60)

    status = "✅ PASSED" if results["passed"] else "❌ FAILED"
    print(f"\n{status}  ({results['n_frames']} frames tested)")

    for name, check in results["checks"].items():
        icon = "✅" if check["passed"] else "❌"
        err = f" — {check['error']}" if check.get("error") else ""
        print(f"  {icon} {name}{err}")

    if results["failed_checks"]:
        print(f"\nFailed: {', '.join(results['failed_checks'])}")

    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Streaming equivalence gate for TFLite wake word models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s models/exported/wake_word.tflite
  %(prog)s models/exported/wake_word.tflite --frames 30 --verbose
  %(prog)s models/exported/wake_word.tflite --json
        """,
    )
    parser.add_argument("model_path", type=str, help="Path to TFLite model file")
    parser.add_argument("--frames", type=int, default=15, help="Number of random frames to test (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-check details")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Error: File not found: {model_path}", file=sys.stderr)
        return 2

    if model_path.suffix != ".tflite":
        print("Warning: File does not have .tflite extension", file=sys.stderr)

    try:
        results = run_gate(str(model_path), n_frames=args.frames, seed=args.seed, verbose=args.verbose)
        print_results(results, use_json=args.json)
        return 0 if results["passed"] else 1
    except Exception as exc:
        print(f"Error: Gate failed with exception: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
