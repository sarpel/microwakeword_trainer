#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.label_guard import LABEL_HARD_NEGATIVE, LABEL_NEGATIVE, LABEL_POSITIVE, VALID_LABELS


def _ensure_evidence_dir(base: str = ".sisyphus/evidence") -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _strict_payload_shape_check(tflite_path: str, verification: dict | None = None) -> tuple[bool, str]:
    import tensorflow as tf

    observed: list[tuple[int, ...]] = []
    if verification is not None:
        details = verification.get("details", {})
        observed_raw = details.get("observed_state_payload_shapes", [])
        observed = [tuple(int(v) for v in shape) for shape in observed_raw]

    if not observed:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        details = interpreter.get_tensor_details()
        payloads = [d for d in details if "ReadVariableOp" in d.get("name", "")]
        observed = [tuple(int(v) for v in d.get("shape", [])) for d in payloads]

    if len(observed) != 6:
        return (
            False,
            f"Strict: expected 6 READ_VARIABLE payload tensors, found {len(observed)}",
        )

    observed_sorted = sorted(observed)
    canonical_first_five = sorted(
        [
            (1, 2, 1, 40),
            (1, 4, 1, 32),
            (1, 10, 1, 64),
            (1, 14, 1, 64),
            (1, 22, 1, 64),
        ]
    )

    if not all(shape in observed_sorted for shape in canonical_first_five):
        return (
            False,
            f"Strict payload shape mismatch: missing one or more canonical streaming state shapes {canonical_first_five}; observed={observed_sorted}",
        )

    dynamic_shapes = [shape for shape in observed_sorted if shape not in canonical_first_five]
    if len(dynamic_shapes) != 1:
        return (
            False,
            f"Strict payload shape mismatch: expected exactly one dynamic stream_5 shape, observed={observed_sorted}",
        )

    stream_5 = dynamic_shapes[0]
    if len(stream_5) != 4 or stream_5[0] != 1 or stream_5[2] != 1 or stream_5[3] != 64 or stream_5[1] < 1:
        return (
            False,
            f"Strict payload shape mismatch: dynamic stream_5 shape invalid: {stream_5}",
        )

    return (
        True,
        f"Strict: READ_VARIABLE payload tensor shapes are valid (dynamic stream_5 accepted: {stream_5}).",
    )


def _build_output(verification: dict, strict_result: tuple[bool, str] | None = None) -> dict:
    payload = {
        "compatible": bool(verification.get("valid", False)),
        "valid": bool(verification.get("valid", False)),
        "errors": list(verification.get("errors", [])),
        "warnings": list(verification.get("warnings", [])),
        "checks": verification.get("checks", {}),
        "details": verification.get("details", {}),
    }
    if strict_result is not None:
        strict_ok, strict_msg = strict_result
        payload["checks"] = dict(payload["checks"])
        payload["checks"]["strict_payload_shapes"] = strict_ok
        if strict_ok:
            payload["warnings"].append(strict_msg)
        else:
            payload["compatible"] = False
            payload["valid"] = False
            payload["errors"].append(strict_msg)
    return payload


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy-containing structures into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    return obj


def main():
    parser = argparse.ArgumentParser(description="ESPHome verification utility with optional strict mode.")
    parser.add_argument("tflite_path", nargs="?", help="Path to the TFLite model to verify.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict payload-shape validation for READ_VARIABLE tensors",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed checks and model details",
    )
    args = parser.parse_args()

    if not {LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_HARD_NEGATIVE}.issubset(VALID_LABELS):
        sys.stderr.write("ERROR: Label constants are inconsistent with VALID_LABELS\n")
        return 1

    evidence_dir = _ensure_evidence_dir()
    if not args.tflite_path:
        parser.print_help()
        return 1

    model_path = Path(args.tflite_path)
    if not model_path.exists():
        with open(evidence_dir / "task-9-missing-model.txt", "w") as f:
            f.write(f"Missing model: {model_path}\n")
            f.write("No TFLite file could be loaded for verification.\n")
        sys.stderr.write(f"ERROR: Model not found: {model_path}\n")
        return 1

    try:
        from src.export.tflite import get_checkpoint_metadata
        from src.export.verification import (
            compute_expected_state_shapes,
            verify_tflite_model,
        )

        config_path = os.getenv("MWW_VERIFY_CONFIG", "config/presets/standard.yaml")
        with open(config_path, "r", encoding="utf-8") as cfg_handle:
            yaml_cfg = yaml.safe_load(cfg_handle) or {}

        model_cfg = yaml_cfg.get("model", {})
        pointwise_filters_str = str(model_cfg.get("pointwise_filters", "64,64,64,64"))
        pointwise_filters = cast(list[int], [int(x.strip()) for x in pointwise_filters_str.split(",") if x.strip()])
        if not pointwise_filters:
            pointwise_filters = [64, 64, 64, 64]

        mixconv_str = str(model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"))
        mixconv_kernel_sizes = cast(list[list[int]], yaml.safe_load(f"[{mixconv_str}]") or [[5], [7, 11], [9, 15], [23]])

        checkpoint_path = os.getenv("MWW_VERIFY_CHECKPOINT", "models/checkpoints/best_weights.weights.h5")
        metadata = get_checkpoint_metadata(checkpoint_path, pointwise_filters=pointwise_filters[-1])
        temporal_frames = int(metadata.get("temporal_frames", 6))

        expected_shapes = compute_expected_state_shapes(
            first_conv_kernel=int(model_cfg.get("first_conv_kernel_size", 5)),
            stride=int(model_cfg.get("stride", 3)),
            mel_bins=int(yaml_cfg.get("hardware", {}).get("mel_bins", 40)),
            first_conv_filters=int(model_cfg.get("first_conv_filters", 32)),
            mixconv_kernel_sizes=mixconv_kernel_sizes,
            pointwise_filters=pointwise_filters,
            temporal_frames=temporal_frames,
        )

        verification = verify_tflite_model(str(model_path), expected_state_shapes=expected_shapes)
        strict_result = _strict_payload_shape_check(str(model_path), verification=verification) if args.strict else None
        payload = _build_output(verification, strict_result=strict_result)
        payload_safe = cast(dict[str, Any], _to_json_safe(payload))

        with open(evidence_dir / "task-9-verify-strict.txt", "w") as f:
            f.write(json.dumps(payload_safe, indent=2))
            f.write("\n")

        if args.json:
            print(json.dumps(payload_safe, indent=2))
        else:
            status = "OK" if payload_safe["compatible"] else "FAIL"
            print(f"Verification completed: {status}")
            for warning in payload_safe["warnings"]:
                print(f"WARN: {warning}")
            for error in payload_safe["errors"]:
                print(f"ERROR: {error}")
            if args.verbose:
                print("--- Checks ---")
                for key, value in payload_safe.get("checks", {}).items():
                    print(f"{key}: {value}")
                print("--- Details ---")
                for key, value in payload_safe.get("details", {}).items():
                    print(f"{key}: {value}")

        return 0 if payload_safe["compatible"] else 2
    except Exception as exc:
        with open(evidence_dir / "task-9-verification-error.txt", "w") as f:
            f.write("Verification error: " + str(exc) + "\n")
        sys.stderr.write("ERROR during verification: " + str(exc) + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
