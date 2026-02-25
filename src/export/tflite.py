"""Export module for model conversion to TFLite."""

import os
from typing import Any, Callable, Generator, Optional

import numpy as np

import tensorflow as tf


# =============================================================================
# STREAMING MODEL CONVERSION
# =============================================================================


def convert_model_saved(
    model: tf.keras.Model,
    config: dict,
    folder: str,
    mode: str = "stream_internal_state_inference",
) -> tf.keras.Model:
    """Convert non-streaming model to streaming SavedModel.

    This wraps the model with streaming state variables for real-time inference.
    The streaming model maintains internal state between inferences to process
    audio chunks incrementally without needing full context each time.

    Args:
        model: Trained non-streaming Keras model
        config: Model configuration with stride, architecture params
        folder: Output directory for SavedModel
        mode: Streaming mode - "stream_internal_state_inference" for ESPHome

    Returns:
        Streaming Keras model

    Raises:
        ValueError: If mode is not supported
    """
    # Parse mode
    mode = mode.upper()
    if mode == "STREAM_INTERNAL_STATE_INFERENCE":
        return _convert_to_streaming_savedmodel(model, config, folder)
    else:
        raise ValueError(f"Unsupported streaming mode: {mode}")


def _convert_to_streaming_savedmodel(
    model: tf.keras.Model,
    config: dict,
    folder: str,
) -> tf.keras.Model:
    """Convert to streaming SavedModel with internal state management."""
    # Get model parameters from config
    stride = config.get("stride", 3)
    spectrogram_length = config.get("spectrogram_length", 49)
    mel_bins = config.get("mel_bins", 40)

    # Input shape: [batch, time_frames, mel_bins]
    input_shape = (spectrogram_length, mel_bins)

    # Create input tensor
    inputs = tf.keras.Input(shape=input_shape, batch_size=1, dtype=tf.float32)

    # Add channel dimension: [batch, time, 1, feature]
    net = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(inputs)

    # Get the trained model weights and apply to streaming layers
    first_conv_filters = config.get("first_conv_filters", 30)
    first_conv_kernel_size = config.get("first_conv_kernel_size", 5)

    if first_conv_filters > 0:
        # Create streaming conv layer with state
        stream_state = _create_streaming_state(shape=(1, 2, 1, mel_bins), name="stream")
        # Concatenate previous state with current input
        net = _streaming_concat(net, stream_state, stride)
        # Apply conv
        net = tf.keras.layers.Conv2D(
            first_conv_filters,
            (first_conv_kernel_size, 1),
            strides=(stride, 1),
            padding="valid",
            use_bias=False,
        )(net)
        net = tf.keras.layers.Activation("relu")(net)

    # Process through MixConv blocks with state management
    pointwise_filters = _parse_list_config(
        config.get("pointwise_filters", "60,60,60,60")
    )
    mixconv_kernel_sizes = _parse_nested_list_config(
        config.get("mixconv_kernel_sizes", "[5],[9],[13],[21]")
    )
    repeat_in_block = _parse_list_config(config.get("repeat_in_block", "1,1,1,1"))
    residual_connection = _parse_list_config(
        config.get("residual_connection", "0,0,0,0")
    )

    # Create state variables for each MixConv block
    state_shapes = [
        (1, 4, 1, 30),
        (1, 8, 1, 60),
        (1, 12, 1, 60),
        (1, 20, 1, 60),
        (1, 4, 1, 60),
    ]

    # Build MixConv blocks
    for i, (filters, kernels, repeat, use_res) in enumerate(
        zip(
            pointwise_filters,
            mixconv_kernel_sizes,
            repeat_in_block,
            residual_connection,
        )
    ):
        state_shape = state_shapes[i] if i < len(state_shapes) else state_shapes[-1]
        stream_state = _create_streaming_state(
            shape=state_shape, name=f"stream_{i + 1}"
        )

        net = _apply_mixconv_block(
            net, stream_state, kernels, filters, repeat, use_res, stride
        )

    # Classification head (from trained model)
    net = tf.keras.layers.Flatten()(net)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(net)

    # Create streaming model
    streaming_model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="streaming_model"
    )

    # Copy weights from trained model
    streaming_model.set_weights(model.get_weights())

    # Save the model
    os.makedirs(folder, exist_ok=True)
    streaming_model.save(folder)

    return streaming_model


def _create_streaming_state(shape: tuple, name: str) -> tf.Tensor:
    """Create a streaming state variable."""
    initial_state = tf.zeros(shape, dtype=tf.float32)
    return initial_state


def _streaming_concat(net: tf.Tensor, state: tf.Tensor, stride: int) -> tf.Tensor:
    """Concatenate streaming state with current input."""
    return tf.concat([state, net], axis=1)


def _apply_mixconv_block(
    net: tf.Tensor,
    state: tf.Tensor,
    kernel_sizes: list,
    filters: int,
    repeat: int,
    use_residual: int,
    stride: int,
) -> tf.Tensor:
    """Apply MixConv block with streaming state."""
    net = _streaming_concat(net, state, stride)

    outputs = []
    for k in kernel_sizes:
        if k > 1:
            conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(k, 1),
                strides=(1, 1),
                padding="same",
                use_bias=False,
            )(net)
            outputs.append(conv)

    if outputs:
        net = tf.concat(outputs, axis=-1)
    else:
        net = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
        )(net)

    net = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        use_bias=False,
    )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)

    return net


def _parse_list_config(config_str: str) -> list:
    """Parse comma-separated config string to list."""
    return [int(x.strip()) for x in config_str.split(",")]


def _parse_nested_list_config(config_str: str) -> list:
    """Parse nested list config string to list of lists."""
    result = []
    cleaned = config_str.replace("[", "").replace("]", "")
    parts = cleaned.split(",")
    for part in parts:
        try:
            result.append([int(part.strip())])
        except ValueError:
            pass
    return result if result else [[3]]


# =============================================================================
# TFLITE CONVERSION
# =============================================================================


def convert_saved_model_to_tflite(
    config: dict,
    path_to_model: str,
    output_path: str,
    representative_dataset_gen: Optional[
        Callable[[], Generator[np.ndarray, None, None]]
    ] = None,
    quantize: bool = True,
) -> bytes:
    """Convert SavedModel to TFLite format with quantization.

    Critical settings for ESPHome compatibility:
    - inference_input_type: tf.int8 (REQUIRED)
    - inference_output_type: tf.uint8 (MUST BE UINT8!)
    - _experimental_variable_quantization: True (REQUIRED for streaming state)
    """
    # Load the SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)

    # Configure optimizations
    converter.optimizations = {tf.lite.Optimize.DEFAULT}

    # CRITICAL: Required for streaming state variables
    converter._experimental_variable_quantization = True

    if quantize:
        # Use INT8 operations
        converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}

        # CRITICAL: Input must be INT8
        converter.inference_input_type = tf.int8

        # CRITICAL: Output MUST be UINT8 (NOT int8!)
        converter.inference_output_type = tf.uint8

        # Add representative dataset for calibration
        if representative_dataset_gen is not None:
            converter.representative_dataset = tf.lite.RepresentativeDataset(
                representative_dataset_gen
            )
        else:
            converter.representative_dataset = tf.lite.RepresentativeDataset(
                create_default_representative_dataset(config)
            )

    # Convert the model
    tflite_model = converter.convert()

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def create_default_representative_dataset(
    config: dict,
    num_samples: int = 500,
) -> Callable[[], Generator[np.ndarray, None, None]]:
    """Create default representative dataset generator for quantization."""
    stride = config.get("stride", 3)
    spectrogram_length = config.get("spectrogram_length", 49)
    mel_bins = config.get("mel_bins", 40)

    def representative_dataset_gen():
        np.random.seed(42)
        for _ in range(num_samples):
            sample = np.random.uniform(
                0.0, 26.0, (spectrogram_length, mel_bins)
            ).astype(np.float32)
            sample[0, 0] = 0.0
            sample[0, 1] = 26.0
            for i in range(0, spectrogram_length - stride, stride):
                yield [sample[i : i + stride, :]]

    return representative_dataset_gen


def create_representative_dataset_from_data(
    spectrograms: np.ndarray,
    config: dict,
) -> Callable[[], Generator[np.ndarray, None, None]]:
    """Create representative dataset generator from actual training data."""
    stride = config.get("stride", 3)

    def representative_dataset_gen():
        for spectrogram in spectrograms:
            spectrogram = spectrogram.astype(np.float32)
            for i in range(0, spectrogram.shape[0] - stride, stride):
                sample = spectrogram[i : i + stride, :]
                yield [sample]

    return representative_dataset_gen


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================


def export_to_tflite(
    model: tf.keras.Model,
    config: dict,
    output_dir: str,
    model_name: str = "wake_word",
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> dict:
    """Export trained model to ESPHome-compatible TFLite.

    Two-step export process:
    1. Convert to streaming SavedModel (STREAM_INTERNAL_STATE_INFERENCE mode)
    2. Convert SavedModel to TFLite with quantization
    """
    # Extract export config
    export_config = config.get("export", {})
    model_config = config.get("model", {})
    hardware_config = config.get("hardware", {})

    # Build model config for conversion
    conversion_config = {
        "stride": model_config.get("stride", 3),
        "spectrogram_length": 49,
        "mel_bins": hardware_config.get("mel_bins", 40),
        "first_conv_filters": model_config.get("first_conv_filters", 30),
        "first_conv_kernel_size": model_config.get("first_conv_kernel_size", 5),
        "pointwise_filters": model_config.get("pointwise_filters", "60,60,60,60"),
        "mixconv_kernel_sizes": model_config.get(
            "mixconv_kernel_sizes", "[5],[9],[13],[21]"
        ),
        "repeat_in_block": model_config.get("repeat_in_block", "1,1,1,1"),
        "residual_connection": model_config.get("residual_connection", "0,0,0,0"),
    }

    # Create output directories
    stream_model_dir = os.path.join(output_dir, f"{model_name}_streaming")
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")

    # Step 1: Convert to streaming SavedModel
    print(f"Converting to streaming SavedModel...")
    streaming_model = convert_model_saved(
        model=model,
        config=conversion_config,
        folder=stream_model_dir,
        mode="stream_internal_state_inference",
    )

    # Step 2: Convert to TFLite with quantization
    print(f"Converting to TFLite with quantization...")

    # Create representative dataset generator
    if representative_data is not None:
        rep_data_gen = create_representative_dataset_from_data(
            representative_data, conversion_config
        )
    else:
        rep_data_gen = None

    tflite_model = convert_saved_model_to_tflite(
        config=conversion_config,
        path_to_model=stream_model_dir,
        output_path=tflite_path,
        representative_dataset_gen=rep_data_gen,
        quantize=quantize,
    )

    # Return export metadata
    return {
        "tflite_path": tflite_path,
        "streaming_model_path": stream_model_dir,
        "model_name": model_name,
        "quantized": quantize,
    }


# =============================================================================
# TFLITE VERIFICATION
# =============================================================================


def verify_esphome_compatibility(tflite_path: str) -> dict:
    """Verify TFLite model is compatible with ESPHome micro_wake_word."""
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    results = {
        "compatible": True,
        "errors": [],
        "warnings": [],
    }

    # Check 1: Number of subgraphs (must be 2)
    subgraphs = interpreter.get_subgraphs()
    if len(subgraphs) != 2:
        results["compatible"] = False
        results["errors"].append(f"Expected 2 subgraphs, got {len(subgraphs)}")

    # Check 2: Input shape [1, stride, 40] and dtype int8
    input_details = interpreter.get_input_details()
    if input_details:
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        if list(input_shape) != [1, 3, 40]:
            results["warnings"].append(
                f"Expected input shape [1, 3, 40], got {list(input_shape)}"
            )

        if input_dtype != np.int8:
            results["compatible"] = False
            results["errors"].append(f"Expected input dtype int8, got {input_dtype}")

    # Check 3: Output shape [1, 1] and dtype uint8
    output_details = interpreter.get_output_details()
    if output_details:
        output_shape = output_details[0]["shape"]
        output_dtype = output_details[0]["dtype"]

        if list(output_shape) != [1, 1]:
            results["warnings"].append(
                f"Expected output shape [1, 1], got {list(output_shape)}"
            )

        # CRITICAL: Must be uint8, NOT int8!
        if output_dtype != np.uint8:
            results["compatible"] = False
            results["errors"].append(f"Expected output dtype uint8, got {output_dtype}")

    # Check 4: Verify quantization parameters
    if input_details:
        quant_params = input_details[0].get("quantization_parameters", {})
        if not quant_params:
            results["warnings"].append("Input quantization parameters missing")

    if output_details:
        quant_params = output_details[0].get("quantization_parameters", {})
        if not quant_params:
            results["warnings"].append("Output quantization parameters missing")

    return results


def calculate_tensor_arena_size(tflite_path: str) -> int:
    """Calculate required tensor arena size for TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    allocation = interpreter.get_tensor_details()

    total_memory = 0
    for tensor in allocation:
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

        num_elements = np.prod(shape) if shape else 1
        total_memory += num_elements * elem_size

    arena_size = int(total_memory * 1.3)
    return arena_size


# =============================================================================
# LEGACY API COMPATIBILITY
# =============================================================================


def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantize: bool = True,
) -> bytes:
    """Convert model to TFLite format (legacy API)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = {tf.lite.Optimize.DEFAULT}

    if quantize:
        converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def optimize_for_edge(model: tf.keras.Model) -> tf.keras.Model:
    """Optimize model for edge deployment."""
    return model


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


def main():
    """Main entry point for mww-export command."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Export trained model to ESPHome-compatible TFLite"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.h5 or .ckpt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/presets/standard.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/exported",
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="wake_word",
        help="Name for exported model",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization",
    )

    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required")

    # Load configuration
    try:
        from config.loader import load_preset

        config = load_preset(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Load model
    try:
        model = tf.keras.models.load_model(args.checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    # Export to TFLite
    try:
        print(f"Exporting model to {args.output}...")
        result = export_to_tflite(
            model=model,
            config=config,
            output_dir=args.output,
            model_name=args.model_name,
            quantize=not args.no_quantize,
        )

        print(f"TFLite model saved to: {result['tflite_path']}")

        # Generate manifest
        from src.export.manifest import create_esphome_package
        from src.export.manifest import create_esphome_package

        pkg = create_esphome_package(
            model=None,
            config=config,
            output_dir=args.output,
            model_name=args.model_name,
            tflite_path=result["tflite_path"],
        )

        print(f"Manifest saved to: {pkg['manifest_path']}")
        print(f"Tensor arena size: {pkg['tensor_arena_size']} bytes")

        # Verify compatibility
        verification = verify_esphome_compatibility(result["tflite_path"])
        if verification["compatible"]:
            print("Model is ESPHome compatible!")
        else:
            print("Warning: Model may not be fully ESPHome compatible:")
            for error in verification["errors"]:
                print(f"  - {error}")

    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
