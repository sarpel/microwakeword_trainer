"""Model manifest generation for deployment."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Minimum tensor arena size in bytes (~22.3 KB – the minimum for hey_jarvis models)
# Minimum tensor arena size in bytes (26.0 KB - the reference size for hey_jarvis models per Article X)
DEFAULT_TENSOR_ARENA_SIZE = 26080  # Per ARCHITECTURAL_CONSTITUTION.md Article X


def generate_manifest(
    model_path: str,
    config: dict[str, Any],
    tflite_path: str | None = None,
) -> dict[str, Any]:
    """Generate ESPHome V2 manifest for wake word model.

    Args:
        model_path: Path to the TFLite model file
        config: Full configuration dictionary (with export, hardware sections)
        tflite_path: Optional explicit path to TFLite model (if different from model_path)

    Returns:
        V2 manifest dictionary for ESPHome micro_wake_word component
    """
    # Extract export configuration
    export_config = config.get("export", {})

    # Extract hardware configuration for feature_step_size
    hardware_config = config.get("hardware", {})

    # Calculate tensor arena size if TFLite model exists
    if tflite_path and Path(tflite_path).exists():
        arena_size = calculate_tensor_arena_size(tflite_path)
    else:
        # Use configured value or default
        arena_size = export_config.get("tensor_arena_size", DEFAULT_TENSOR_ARENA_SIZE)

    # Build model filename from path
    model_filename = Path(model_path).name if model_path else "wake_word.tflite"

    # Get feature step size from hardware config (window_step_ms)
    feature_step_size = hardware_config.get("window_step_ms", 10)

    # Build V2 manifest
    manifest = {
        "type": "micro",
        "wake_word": export_config.get("wake_word", "Hey Katya"),
        "author": export_config.get("author", "Sarpel GURAY"),
        "website": export_config.get("website", "https://github.com/sarpel/microwakeword_trainer"),
        "model": model_filename,
        "trained_languages": export_config.get("trained_languages", ["en"]),
        "version": 2,
        "micro": {
            "probability_cutoff": export_config.get("probability_cutoff", 0.97),
            "feature_step_size": feature_step_size,
            "sliding_window_size": export_config.get("sliding_window_size", 5),
            "tensor_arena_size": arena_size,
            "minimum_esphome_version": export_config.get("minimum_esphome_version", "2024.7.0"),
        },
    }

    return manifest


def save_manifest(manifest: dict[str, Any], output_path: str) -> str:
    """Save manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Output file path

    Returns:
        Path to saved manifest file
    """
    output_file = Path(output_path)

    # Create parent directories if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write manifest as formatted JSON
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)

    return str(output_file)


def calculate_tensor_arena_size(tflite_path: str) -> int:
    """Calculate required tensor arena size for TFLite model.

    Uses the TFLite interpreter to analyze tensor allocations and
    computes a recommended arena size with safety margin.

    Args:
        tflite_path: Path to TFLite model file

    Returns:
        Recommended tensor arena size in bytes
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        allocation = interpreter.get_tensor_details()

        total_memory = 0
        for tensor in allocation:
            shape = tensor.get("shape", [])
            dtype = tensor.get("dtype")

            # Get element size based on dtype (using numpy dtypes)
            if dtype == np.float32:
                elem_size = 4
            elif dtype == np.float16:
                elem_size = 2
            elif dtype == np.bytes_ or dtype == np.object_:
                # String size is variable; use a conservative 32-byte estimate
                elem_size = 32
            elif dtype in (np.int8, np.uint8):
                elem_size = 1
            elif dtype == np.int32:
                elem_size = 4
            elif dtype == np.int64:
                elem_size = 8
            else:
                elem_size = 4  # Default assumption

            num_elements = 1
            for dim in shape:
                if dim == -1:
                    # Dynamic dimension: use a conservative estimate of 1 and warn
                    logger.warning(
                        "Tensor '%s' has a dynamic dimension (-1). " "Arena size estimate may be too small; using 1 for this dim.",
                        tensor.get("name", "<unnamed>"),
                    )
                    d = 1
                elif dim == 0:
                    d = 1
                else:
                    d = abs(dim)
                num_elements *= d

            total_memory += num_elements * elem_size

        # Add 30% safety margin
        arena_size = int(total_memory * 1.3)

        # Ensure minimum size (26KB is the minimum for hey_jarvis models)
        arena_size = max(arena_size, DEFAULT_TENSOR_ARENA_SIZE)

        return arena_size

    except Exception as e:
        # Return default if calculation fails, but always log the failure
        logging.warning(
            f"calculate_tensor_arena_size failed for '{tflite_path}': {e}",
            exc_info=True,
        )
        return DEFAULT_TENSOR_ARENA_SIZE


def verify_esphome_compatibility(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify manifest has required fields for ESPHome compatibility.

    Args:
        manifest: Manifest dictionary to verify

    Returns:
        Dictionary with 'compatible' bool and 'errors' list
    """
    results: dict[str, Any] = {
        "compatible": True,
        "errors": [],
        "warnings": [],
    }

    # Required top-level fields
    required_fields = [
        "type",
        "wake_word",
        "author",
        "website",
        "model",
        "trained_languages",
        "version",
        "micro",
    ]

    for field in required_fields:
        if field not in manifest:
            results["compatible"] = False
            results["errors"].append(f"Missing required field: {field}")

    # Validate top-level version regardless of whether "micro" is present
    if manifest.get("version") != 2:
        results["warnings"].append("Manifest version should be 2 for ESPHome 2024.7+")

    # Required micro section fields
    if "micro" in manifest:
        micro_obj = manifest.get("micro")
        if not isinstance(micro_obj, dict):
            results["compatible"] = False
            results["errors"].append("Field 'micro' must be an object")
            return results

        micro_required = [
            "probability_cutoff",
            "feature_step_size",
            "sliding_window_size",
            "tensor_arena_size",
            "minimum_esphome_version",
        ]
        for field in micro_required:
            if field not in micro_obj:
                results["compatible"] = False
                results["errors"].append(f"Missing required micro field: {field}")

        # Validate probability_cutoff range only when the key is present
        micro = manifest.get("micro", {})
        if "probability_cutoff" in micro:
            prob_cutoff = micro["probability_cutoff"]
            if not (0.0 < prob_cutoff <= 1.0):
                results["compatible"] = False
                results["errors"].append("probability_cutoff must be between 0 and 1")

        # Validate tensor_arena_size is reasonable
        arena_size = micro.get("tensor_arena_size", 0)
        if arena_size < DEFAULT_TENSOR_ARENA_SIZE:
            results["warnings"].append(f"tensor_arena_size ({arena_size}) is below recommended minimum ({DEFAULT_TENSOR_ARENA_SIZE})")

    return results


def create_esphome_package(
    model: Any,
    config: dict[str, Any],
    output_dir: str,
    model_name: str = "wake_word",
    tflite_path: str | None = None,
    analysis_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create complete ESPHome package with manifest and model files.

    This is the main entry point for the export process, combining
    model generation with manifest creation.

    Args:
        model: Trained Keras model (not used, for API compatibility)
        config: Full configuration dictionary
        output_dir: Output directory for package files
        model_name: Name for the model (used in filenames)
        tflite_path: Optional path to TFLite model
        analysis_results: Optional ai_edge_litert analysis results to include

    Returns:
        Dictionary with paths and metadata:
        - manifest_path: Path to saved manifest JSON
        - model_path: Path to TFLite model
        - tensor_arena_size: Calculated arena size
    """
    # Determine TFLite model path
    if tflite_path is None:
        tflite_path = str(Path(output_dir) / f"{model_name}.tflite")

    # Get model filename
    model_filename = Path(tflite_path).name

    # Generate manifest
    manifest = generate_manifest(
        model_path=model_filename,
        config=config,
        tflite_path=tflite_path,
    )

    # Add ai_edge_litert analysis results to manifest if provided
    if analysis_results is not None:
        # Include key validation metrics in manifest metadata
        validation = analysis_results.get("validation_results", {})
        performance = analysis_results.get("performance_estimation", {})
        architecture = analysis_results.get("architecture_analysis", {})

        manifest["_metadata"] = {
            "model_valid": analysis_results.get("model_valid"),
            "analysis": {
                "layer_count": architecture.get("layer_count"),
                "operators": architecture.get("operators"),
                "quantized": architecture.get("has_quantization"),
                "model_size_kb": performance.get("model_size_kb"),
                "estimated_latency_ms": performance.get("estimated_latency_ms"),
                "tensor_arena_estimate_kb": performance.get("tensor_arena_estimate_kb"),
            },
            "validation": {
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
                "info": validation.get("info", {}),
            },
        }

    # Save manifest
    manifest_path = str(Path(output_dir) / "manifest.json")
    save_manifest(manifest, manifest_path)

    # Get tensor arena size from manifest
    tensor_arena_size = manifest.get("micro", {}).get("tensor_arena_size", DEFAULT_TENSOR_ARENA_SIZE)

    return {
        "manifest_path": manifest_path,
        "model_path": tflite_path,
        "model_filename": model_filename,
        "tensor_arena_size": tensor_arena_size,
        "manifest": manifest,
    }
