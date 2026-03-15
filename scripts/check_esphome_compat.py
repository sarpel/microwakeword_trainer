#!/usr/bin/env python3
"""
ESPHome microWakeWord TFLite Compatibility Checker
===================================================
Standalone script — no project imports required. Works on any TFLite file.

All constraints are derived DIRECTLY from ESPHome source code:

  streaming_model.cpp  commit 8a5f008
    - Op resolver:   L214-L253  → ESPHOME_ALLOWED_OPS
    - Input tensor:  L65-L75    → input index=0, dtype=int8, shape=[1,stride,40]
    - Output tensor: L78-L86    → output index=0, dtype=uint8, shape=[1,1]
    - Variable arena: L48-L49   → capacity=20, arena_size=1024 bytes

  preprocessor_settings.h  commit 8a5f008
    - L14: PREPROCESSOR_FEATURE_SIZE = 40
    - FEATURE_DURATION_MS = 30

  __init__.py  commit 8a5f008
    - L305-L317: feature_step_size must be 1-30
    - L335-L340: TFLite Micro version 1.3.3~1

  TFLite flatbuffers schema version 3 (kTfLiteSchemaVersion)

Usage:
  python scripts/check_esphome_compat.py path/to/model.tflite
  python scripts/check_esphome_compat.py path/to/model.tflite --json
  python scripts/check_esphome_compat.py path/to/model.tflite --verbose
  python scripts/check_esphome_compat.py path/to/model.tflite --manifest path/to/manifest.json
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# ESPHome constraints (sourced from C++ / Python components, commit 8a5f008)
# ---------------------------------------------------------------------------

# streaming_model.cpp L214-L253
ESPHOME_ALLOWED_OPS = {
    "CALL_ONCE",
    "VAR_HANDLE",
    "RESHAPE",
    "READ_VARIABLE",
    "STRIDED_SLICE",
    "CONCATENATION",
    "ASSIGN_VARIABLE",
    "CONV_2D",
    "MUL",
    "ADD",
    "MEAN",
    "FULLY_CONNECTED",
    "LOGISTIC",
    "QUANTIZE",
    "DEPTHWISE_CONV_2D",
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
    "PAD",
    "PACK",
    "SPLIT_V",
}

# streaming_model.cpp L48-L49
ESPHOME_MAX_RESOURCE_VARIABLES = 20
ESPHOME_VARIABLE_ARENA_SIZE_BYTES = 1024

# preprocessor_settings.h L14
ESPHOME_FEATURE_SIZE = 40  # mel bins

# streaming_model.cpp L65-L75 — input[0] dtype=int8, shape=[1, stride, 40]
ESPHOME_INPUT_DTYPE = np.int8
ESPHOME_INPUT_SHAPE_FIXED_DIMS = {0: 1, 2: 40}  # dims that must be exact; dim[1]=stride is flexible

# streaming_model.cpp L78-L86 — output[0] dtype=uint8, shape=[1,1]
ESPHOME_OUTPUT_DTYPE = np.uint8
ESPHOME_OUTPUT_SHAPE = [1, 1]

# __init__.py L305-L317 — feature_step_size in manifest
ESPHOME_FEATURE_STEP_SIZE_MIN = 1
ESPHOME_FEATURE_STEP_SIZE_MAX = 30

# TFLite schema version 3
TFLITE_EXPECTED_SCHEMA_VERSION = 3

# Streaming architecture: exactly 6 state variables
# (VAR_HANDLE / READ_VARIABLE / ASSIGN_VARIABLE ops must each appear exactly 6 times)
ESPHOME_EXPECTED_STATE_VARS = 6

# Dual-subgraph requirement: main inference + CALL_ONCE init subgraph
ESPHOME_EXPECTED_SUBGRAPHS = 2


# ---------------------------------------------------------------------------
# TFLite binary header inspection (schema version without flatbuffers dep)
# ---------------------------------------------------------------------------


def _read_tflite_schema_version(model_bytes: bytes) -> int | None:
    """
    Parse TFLite flatbuffer to extract schema_version field.

    TFLite flatbuffer layout (with optional 4-byte file identifier, e.g. 'TFL3'):
      bytes 0-3: root table offset from byte 0 (little-endian uint32)
      bytes 4-7: file identifier (e.g. b'TFL3') — optional but always present in TFLite
      root table at root_offset:
        [0..3] soffset_t (signed int32): offset from root_offset BACKWARDS to vtable
                vtable_abs = root_offset - soffset
        vtable:
          [0..1] vtable_size (uint16)
          [2..3] data_size   (uint16)
          [4..5] field-0 offset from root (uint16)  ← Model.version field
      version is a uint32 at root_offset + field0_offset
    """
    try:
        if len(model_bytes) < 8:
            return None
        root_offset = struct.unpack_from("<I", model_bytes, 0)[0]
        # soffset points BACKWARDS: vtable is BEFORE root in memory
        soffset = struct.unpack_from("<i", model_bytes, root_offset)[0]
        vtable_abs = root_offset - soffset
        vtable_size = struct.unpack_from("<H", model_bytes, vtable_abs)[0]
        if vtable_size < 6:
            return None
        # field 0 (version) relative offset from root
        field0_offset = struct.unpack_from("<H", model_bytes, vtable_abs + 4)[0]
        if field0_offset == 0:
            return None
        version = struct.unpack_from("<I", model_bytes, root_offset + field0_offset)[0]
        return int(version)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------


def check_model(tflite_path: str, manifest_path: str | None = None) -> dict[str, Any]:
    """
    Run all ESPHome compatibility checks on a TFLite model file.

    Returns a dict with keys:
      compatible   bool   — True iff ALL hard requirements pass
      checks       dict   — per-check bool results
      errors       list   — hard failures (model will NOT load/work)
      warnings     list   — soft concerns (model may work but is non-canonical)
      details      dict   — raw values extracted from model
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {
            "compatible": False,
            "checks": {},
            "errors": ["TensorFlow not available — install tensorflow to run this script"],
            "warnings": [],
            "details": {},
        }

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    model_file = Path(tflite_path)
    try:
        model_bytes = model_file.read_bytes()
        details["file_size_bytes"] = len(model_bytes)
    except (FileNotFoundError, PermissionError, OSError) as err:
        details["error"] = f"Could not read {model_file}: {err}"
        raise

    # ------------------------------------------------------------------
    # CHECK 1: TFLite schema version
    # Source: kTfLiteSchemaVersion == 3
    # ------------------------------------------------------------------
    schema_version = _read_tflite_schema_version(model_bytes)
    details["tflite_schema_version"] = schema_version
    if schema_version is None:
        warnings.append("Could not parse TFLite schema version from binary header")
        checks["schema_version"] = True  # soft — don't fail on parse error
    elif schema_version != TFLITE_EXPECTED_SCHEMA_VERSION:
        errors.append(f"TFLite schema version {schema_version} != expected {TFLITE_EXPECTED_SCHEMA_VERSION}. ESPHome uses TFLite Micro 1.3.3~1 which expects schema version 3.")
        checks["schema_version"] = False
    else:
        checks["schema_version"] = True

    # ------------------------------------------------------------------
    # Load interpreter
    # ------------------------------------------------------------------
    try:
        resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=resolver_type,
        )
        interpreter.allocate_tensors()
    except Exception as exc:
        errors.append(f"Failed to load TFLite model: {exc}")
        return {
            "compatible": False,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "details": details,
        }

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    all_tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    try:
        ops_details = interpreter._get_ops_details()
    except Exception:
        ops_details = []

    # ------------------------------------------------------------------
    # CHECK 2: Input tensor
    # Source: streaming_model.cpp L65-L75
    #   input(0) dtype=kTfLiteInt8
    #   dims[0]==1, dims[2]==PREPROCESSOR_FEATURE_SIZE(40)
    #   dims[1] == stride (not hard-coded; it's model-specific)
    # ------------------------------------------------------------------
    if not input_details:
        errors.append("Model has no input tensors")
        checks["input_tensor_exists"] = False
    else:
        inp = input_details[0]
        inp_shape = list(inp["shape"])
        inp_dtype = inp["dtype"]
        details["input_shape"] = inp_shape
        details["input_dtype"] = str(np.dtype(inp_dtype))

        # dtype must be int8
        checks["input_dtype_int8"] = inp_dtype == np.int8
        if not checks["input_dtype_int8"]:
            errors.append(f"[streaming_model.cpp L65] Input dtype must be kTfLiteInt8 (int8), got {np.dtype(inp_dtype)}. ESPHome casts raw int8 features directly into input tensor.")

        # shape: [1, stride, 40]
        input_shape_ok = len(inp_shape) == 3 and inp_shape[0] == 1 and inp_shape[2] == ESPHOME_FEATURE_SIZE and inp_shape[1] >= 1
        checks["input_shape"] = input_shape_ok
        if not input_shape_ok:
            errors.append(f"[streaming_model.cpp L65-L75] Input shape {inp_shape} invalid. Expected [1, stride, {ESPHOME_FEATURE_SIZE}] where stride >= 1.")

        # input quantization
        inp_quant = inp.get("quantization_parameters", {}) or {}
        inp_scales = np.asarray(inp_quant.get("scales", []))
        inp_zps = np.asarray(inp_quant.get("zero_points", []))
        if inp_scales.size > 0 and inp_zps.size > 0:
            inp_scale_val = float(inp_scales[0])
            inp_zp_val = int(inp_zps[0])
            details["input_scale"] = inp_scale_val
            details["input_zero_point"] = inp_zp_val
            # Official scale=0.101961 (1/255 * 26), zero_point=-128 for int8 [-1, 1] range
            canonical_inp_scale = abs(inp_scale_val - 0.101961) <= 1e-4
            canonical_inp_zp = inp_zp_val == -128
            checks["input_quantization"] = canonical_inp_scale and canonical_inp_zp
            if not checks["input_quantization"]:
                warnings.append(
                    f"Input quantization non-canonical: scale={inp_scale_val:.6f} (expected ~0.101961), zero_point={inp_zp_val} (expected -128). Model may still work if preprocessing matches."
                )
        else:
            checks["input_quantization"] = False
            errors.append("Input tensor missing quantization parameters — model is not INT8 quantized")

    # ------------------------------------------------------------------
    # CHECK 3: Output tensor
    # Source: streaming_model.cpp L78-L86
    #   output(0) dtype=kTfLiteUInt8, shape=[1,1]
    # ------------------------------------------------------------------
    if not output_details:
        errors.append("Model has no output tensors")
        checks["output_tensor_exists"] = False
    else:
        out = output_details[0]
        out_shape = list(out["shape"])
        out_dtype = out["dtype"]
        details["output_shape"] = out_shape
        details["output_dtype"] = str(np.dtype(out_dtype))

        checks["output_dtype_uint8"] = out_dtype == np.uint8
        if not checks["output_dtype_uint8"]:
            errors.append(
                f"[streaming_model.cpp L78] Output dtype must be kTfLiteUInt8 (uint8), "
                f"got {np.dtype(out_dtype)}. "
                "ESPHome reads output as uint8 probability. int8 would silently produce wrong predictions."
            )

        checks["output_shape"] = out_shape == ESPHOME_OUTPUT_SHAPE
        if not checks["output_shape"]:
            errors.append(f"[streaming_model.cpp L86] Output shape {out_shape} != {ESPHOME_OUTPUT_SHAPE}. ESPHome expects exactly one probability value per inference.")

        out_quant = out.get("quantization_parameters", {}) or {}
        out_scales = np.asarray(out_quant.get("scales", []))
        out_zps = np.asarray(out_quant.get("zero_points", []))
        if out_scales.size > 0 and out_zps.size > 0:
            out_scale_val = float(out_scales[0])
            out_zp_val = int(out_zps[0])
            details["output_scale"] = out_scale_val
            details["output_zero_point"] = out_zp_val
            # zero_point must be 0 for uint8 [0, 1] probability range
            checks["output_zero_point"] = out_zp_val == 0
            if not checks["output_zero_point"]:
                errors.append(f"Output zero_point={out_zp_val}, must be 0 for uint8 probability output. Non-zero zero_point shifts the probability space, causing wrong threshold comparisons.")
            # canonical scales: 1/256 or 1/255
            canonical_out_scales = {round(1.0 / 255.0, 9), round(1.0 / 256.0, 9)}
            out_scale_rounded = round(out_scale_val, 9)
            checks["output_scale_canonical"] = out_scale_rounded in canonical_out_scales
            if not checks["output_scale_canonical"]:
                warnings.append(
                    f"Output scale {out_scale_val:.9f} is non-canonical. "
                    f"Official models use 1/256 ({1 / 256:.9f}) or 1/255 ({1 / 255:.9f}). "
                    "Model may still work but probability_cutoff interpretation may differ slightly."
                )
        else:
            checks["output_quantization"] = False
            errors.append("Output tensor missing quantization parameters — model is not INT8 quantized")

    # ------------------------------------------------------------------
    # CHECK 4: Op whitelist
    # Source: streaming_model.cpp L214-L253 (exact AddXxx() calls)
    # ------------------------------------------------------------------
    op_counts: dict[str, int] = {}
    for op in ops_details:
        name = op.get("op_name", "") or ""
        op_counts[name] = op_counts.get(name, 0) + 1
    details["op_counts"] = op_counts
    details["total_ops"] = sum(op_counts.values())

    disallowed = sorted(op for op in op_counts if op and op not in ESPHOME_ALLOWED_OPS)
    checks["op_whitelist"] = len(disallowed) == 0
    if disallowed:
        errors.append(
            f"[streaming_model.cpp L214-L253] Disallowed ops detected: {disallowed}. "
            f"ESPHome only registers these 20 ops: {sorted(ESPHOME_ALLOWED_OPS)}. "
            "Model will fail to load on device with kTfLiteError."
        )
    details["disallowed_ops"] = disallowed
    details["allowed_ops_used"] = sorted(op for op in op_counts if op in ESPHOME_ALLOWED_OPS)

    # ------------------------------------------------------------------
    # CHECK 5: State variable counts
    # Source: streaming_model.cpp — iterates MicroResourceVariables
    #   Exactly 6 VAR_HANDLE / READ_VARIABLE / ASSIGN_VARIABLE expected
    # ------------------------------------------------------------------
    var_handle_count = op_counts.get("VAR_HANDLE", 0)
    read_var_count = op_counts.get("READ_VARIABLE", 0)
    assign_var_count = op_counts.get("ASSIGN_VARIABLE", 0)
    details["var_handle_count"] = var_handle_count
    details["read_variable_count"] = read_var_count
    details["assign_variable_count"] = assign_var_count

    checks["state_var_count"] = var_handle_count == ESPHOME_EXPECTED_STATE_VARS and read_var_count == ESPHOME_EXPECTED_STATE_VARS and assign_var_count == ESPHOME_EXPECTED_STATE_VARS
    if not checks["state_var_count"]:
        errors.append(
            f"State variable op counts: VAR_HANDLE={var_handle_count}, "
            f"READ_VARIABLE={read_var_count}, ASSIGN_VARIABLE={assign_var_count}. "
            f"All three must be exactly {ESPHOME_EXPECTED_STATE_VARS} for streaming inference."
        )

    # ------------------------------------------------------------------
    # CHECK 6: Resource variable capacity
    # Source: streaming_model.cpp L48-L49
    #   MicroResourceVariables::Create(allocator, 20)  ← capacity=20
    # Your model has N state vars; N must be <= 20
    # ------------------------------------------------------------------
    checks["resource_variable_capacity"] = var_handle_count <= ESPHOME_MAX_RESOURCE_VARIABLES
    details["max_resource_variables"] = ESPHOME_MAX_RESOURCE_VARIABLES
    if not checks["resource_variable_capacity"]:
        errors.append(
            f"[streaming_model.cpp L48-L49] Model uses {var_handle_count} resource variables "
            f"but ESPHome allocates capacity for only {ESPHOME_MAX_RESOURCE_VARIABLES}. "
            "Excess variables will cause silent allocation failure."
        )

    # ------------------------------------------------------------------
    # CHECK 7: CALL_ONCE (init subgraph)
    # Required for dual-subgraph (main + init) architecture
    # ------------------------------------------------------------------
    call_once_count = op_counts.get("CALL_ONCE", 0)
    details["call_once_count"] = call_once_count
    checks["call_once_present"] = call_once_count == 1
    if call_once_count == 0:
        errors.append("Missing CALL_ONCE op — model lacks initialization subgraph. ESPHome requires dual-subgraph TFLite: subgraph 0 = inference, subgraph 1 = init.")
    elif call_once_count > 1:
        errors.append(f"Found {call_once_count} CALL_ONCE ops, expected exactly 1. Multiple init subgraphs are not supported.")

    # ------------------------------------------------------------------
    # CHECK 8: State variable shapes
    # Validate that READ_VARIABLE output shapes match the canonical
    # streaming architecture pattern. stream_5 is dynamic (depends on
    # temporal_frames), so we accept any valid 4D shape for it.
    # Canonical first 5: [1,2,1,40], [1,4,1,32], [1,10,1,64], [1,14,1,64], [1,22,1,64]
    # ------------------------------------------------------------------
    state_shapes: list[tuple[int, ...]] = []
    for op in ops_details:
        if op.get("op_name") == "READ_VARIABLE":
            out_idx = op["outputs"][0]
            t = all_tensors.get(out_idx)
            if t is not None:
                state_shapes.append(tuple(int(v) for v in t.get("shape", [])))

    details["state_variable_shapes"] = sorted(state_shapes)

    canonical_first_5 = sorted(
        [
            (1, 2, 1, 40),
            (1, 4, 1, 32),
            (1, 10, 1, 64),
            (1, 14, 1, 64),
            (1, 22, 1, 64),
        ]
    )

    if len(state_shapes) == ESPHOME_EXPECTED_STATE_VARS:
        sorted_shapes = sorted(state_shapes)
        # find stream_5 = the shape not in canonical_first_5
        stream_5_candidates = [s for s in sorted_shapes if s not in canonical_first_5]
        first_5_present = all(s in sorted_shapes for s in canonical_first_5)

        if not first_5_present:
            missing = [s for s in canonical_first_5 if s not in sorted_shapes]
            warnings.append(
                f"State buffer shapes mismatch canonical first-5. Missing: {missing}. "
                "This may indicate a non-standard architecture (different kernel sizes or filter counts). "
                "Model may still work if the architecture is self-consistent."
            )
            checks["state_shapes_canonical"] = False
        elif len(stream_5_candidates) != 1:
            warnings.append(f"Could not identify unique stream_5 buffer. Shapes not in canonical first-5: {stream_5_candidates}. This is unexpected but may not prevent operation.")
            checks["state_shapes_canonical"] = False
        else:
            stream_5 = stream_5_candidates[0]
            details["stream_5_shape"] = list(stream_5)
            stream_5_valid = len(stream_5) == 4 and stream_5[0] == 1 and stream_5[2] == 1 and stream_5[3] == 64 and stream_5[1] >= 1
            checks["state_shapes_canonical"] = stream_5_valid
            if not stream_5_valid:
                warnings.append(f"stream_5 shape {list(stream_5)} has unexpected format. Expected [1, N, 1, 64] where N >= 1.")
            else:
                temporal_frames = stream_5[1] + 1
                details["inferred_temporal_frames"] = temporal_frames
                # Compute implied clip_duration_ms assuming step=10ms, kernel=5, stride=3
                # temporal_frames = floor((total_frames - kernel + 1) / stride)
                # total_frames = temporal_frames * stride + kernel - 1
                implied_total_frames = temporal_frames * 3 + 5 - 1
                implied_clip_ms = implied_total_frames * 10
                details["inferred_clip_duration_ms_approx"] = implied_clip_ms
    else:
        checks["state_shapes_canonical"] = False  # will be caught by state_var_count check

    # ------------------------------------------------------------------
    # CHECK 9: State variable dtypes (must be int8 for quantized streaming)
    # ------------------------------------------------------------------
    state_dtypes: list[str] = []
    for op in ops_details:
        if op.get("op_name") == "READ_VARIABLE":
            out_idx = op["outputs"][0]
            t = all_tensors.get(out_idx)
            if t is not None:
                state_dtypes.append(str(np.dtype(t.get("dtype"))))

    details["state_variable_dtypes"] = state_dtypes
    checks["state_dtypes_int8"] = all(dt == "int8" for dt in state_dtypes) and len(state_dtypes) > 0
    if not checks["state_dtypes_int8"]:
        errors.append(f"State variable tensors must be int8 for quantized streaming, got: {set(state_dtypes)}. Non-int8 states indicate incomplete quantization.")

    # ------------------------------------------------------------------
    # CHECK 10: Inference smoke test
    # Source: streaming_model.cpp — Invoke() called with int8 features
    # ------------------------------------------------------------------
    if not input_details or not output_details:
        checks["inference_smoke_test"] = False
        errors.append("No input/output tensor available for inference smoke test")
    else:
        try:
            _inp_shape = list(input_details[0]["shape"]) if input_details else [1, 3, 40]
            _stride = _inp_shape[1] if len(_inp_shape) == 3 else 3
            test_input = np.zeros((1, _stride, 40), dtype=np.int8)
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            test_out = interpreter.get_tensor(output_details[0]["index"])
            checks["inference_smoke_test"] = True
            details["smoke_test_output"] = int(test_out.flat[0])
            details["smoke_test_output_dtype"] = str(test_out.dtype)
        except Exception as exc:
            checks["inference_smoke_test"] = False
            errors.append(f"Inference smoke test failed: {exc}. Model cannot be invoked.")

    # ------------------------------------------------------------------
    # CHECK 11: Manifest validation (optional, if --manifest provided)
    # Source: __init__.py L305-L317
    #   feature_step_size: 0 < value <= 30
    #   All models must share same feature_step_size
    # ------------------------------------------------------------------
    if manifest_path:
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            warnings.append(f"Manifest file not found: {manifest_path} — skipping manifest checks")
            checks["manifest"] = False
        else:
            try:
                manifest = json.loads(manifest_file.read_text())
                micro = manifest.get("micro", {})
                fss = micro.get("feature_step_size")
                details["manifest_feature_step_size"] = fss
                if fss is None:
                    warnings.append("Manifest missing 'micro.feature_step_size' field")
                    checks["manifest_feature_step_size"] = False
                else:
                    try:
                        fss_int = int(fss)
                        if not (ESPHOME_FEATURE_STEP_SIZE_MIN <= fss_int <= ESPHOME_FEATURE_STEP_SIZE_MAX):
                            errors.append(f"[__init__.py L305-L317] manifest feature_step_size={fss} out of range [{ESPHOME_FEATURE_STEP_SIZE_MIN}, {ESPHOME_FEATURE_STEP_SIZE_MAX}]")
                            checks["manifest_feature_step_size"] = False
                        else:
                            checks["manifest_feature_step_size"] = True
                    except (ValueError, TypeError):
                        errors.append(f"manifest feature_step_size='{fss}' is not a valid integer")
                        checks["manifest_feature_step_size"] = False

                # Check manifest version
                version = manifest.get("version")
                details["manifest_version"] = version
                checks["manifest_version_2"] = version == 2
                if version != 2:
                    warnings.append(f"Manifest version={version}, expected 2 (ESPHome V2 format)")

                # Check tensor_arena_size is set
                tas = micro.get("tensor_arena_size", 0)
                details["manifest_tensor_arena_size"] = tas
                if tas == 0:
                    warnings.append("manifest tensor_arena_size=0 (auto). ESPHome will attempt to auto-resolve. Set explicitly if you see arena allocation errors on device.")
                    checks["manifest_tensor_arena_size"] = True  # 0 = auto is allowed
                else:
                    checks["manifest_tensor_arena_size"] = True

            except json.JSONDecodeError as exc:
                warnings.append(f"Could not parse manifest JSON: {exc}")
                checks["manifest"] = False

    # ------------------------------------------------------------------
    # Final compatibility verdict
    # Only hard errors affect compatibility; warnings are informational
    # ------------------------------------------------------------------
    compatible = len(errors) == 0

    return {
        "compatible": compatible,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _c(text: str, color: str) -> str:
    """Wrap text in ANSI color if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def _print_human(result: dict, verbose: bool) -> None:
    compatible = result["compatible"]
    errors = result["errors"]
    warnings = result["warnings"]
    checks = result["checks"]
    details = result["details"]

    status_str = _c("COMPATIBLE ✓", _GREEN) if compatible else _c("INCOMPATIBLE ✗", _RED)
    print(f"\n{_c('ESPHome microWakeWord Compatibility', _BOLD)}: {status_str}")
    print(f"File size: {details.get('file_size_bytes', '?')} bytes")
    print(f"TFLite schema version: {details.get('tflite_schema_version', '?')}")

    if details.get("input_shape"):
        print(f"Input:  shape={details['input_shape']}  dtype={details.get('input_dtype', '?')}")
    if details.get("output_shape"):
        print(f"Output: shape={details['output_shape']}  dtype={details.get('output_dtype', '?')}")

    op_counts = details.get("op_counts", {})
    if op_counts:
        total_ops = details.get("total_ops", sum(op_counts.values()))
        print(f"Ops: {total_ops} total, {len(op_counts)} unique")

    if details.get("state_variable_shapes"):
        shapes = details["state_variable_shapes"]
        print(f"State variables: {len(shapes)}")
        if details.get("stream_5_shape"):
            print(f"  stream_5 (dynamic): {details['stream_5_shape']}")
            if details.get("inferred_temporal_frames"):
                print(f"  Inferred temporal_frames: {details['inferred_temporal_frames']} (~{details.get('inferred_clip_duration_ms_approx', '?')}ms clip)")

    if errors:
        print(f"\n{_c('ERRORS', _RED)} ({len(errors)}):")
        for e in errors:
            print(f"  {_c('✗', _RED)} {e}")

    if warnings:
        print(f"\n{_c('WARNINGS', _YELLOW)} ({len(warnings)}):")
        for w in warnings:
            print(f"  {_c('⚠', _YELLOW)} {w}")

    if not errors and not warnings:
        print(f"\n{_c('All checks passed with no warnings.', _GREEN)}")

    if verbose:
        print(f"\n{_c('── Checks ──', _BOLD)}")
        for key, val in sorted(checks.items()):
            icon = _c("✓", _GREEN) if val else _c("✗", _RED)
            print(f"  {icon} {key}")

        print(f"\n{_c('── Details ──', _BOLD)}")
        for key, val in sorted(details.items()):
            print(f"  {key}: {val}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check TFLite model compatibility with ESPHome microWakeWord (source-derived constraints).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tflite_path", help="Path to .tflite model file")
    parser.add_argument("--manifest", metavar="PATH", help="Optional path to manifest.json for additional checks")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--verbose", action="store_true", help="Show all check results and raw details")
    args = parser.parse_args()

    tflite_path = Path(args.tflite_path)
    if not tflite_path.exists():
        sys.stderr.write(f"ERROR: File not found: {tflite_path}\n")
        return 1

    result = check_model(str(tflite_path), manifest_path=args.manifest)

    # JSON-safe serialization
    def _to_json_safe(obj: Any) -> Any:
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
        return obj

    safe = _to_json_safe(result)

    if args.json:
        print(json.dumps(safe, indent=2))
    else:
        _print_human(result, verbose=args.verbose)

    return 0 if result["compatible"] else 2


if __name__ == "__main__":
    sys.exit(main())
