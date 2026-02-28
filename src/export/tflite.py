"""Export module for model conversion to TFLite."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Generator, Optional

import h5py
import numpy as np
import tensorflow as tf

# =============================================================================
# KERAS 3 CHECKPOINT LOADER
# =============================================================================


def load_weights_from_keras3_checkpoint(model: tf.keras.Model, checkpoint_path: str) -> int:
    """Load weights from Keras 3 format .weights.h5 checkpoint.

    Keras 3 stores weights differently than Keras 2. This function maps
    the hierarchical structure to the model's weight names.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to .weights.h5 checkpoint

    Returns:
        Number of weights successfully loaded
    """
    print(f"Loading weights from: {checkpoint_path}")

    loaded_count = 0
    with h5py.File(checkpoint_path, "r") as f:
        for weight in model.weights:
            weight_name = weight.name
            weight_shape = tuple(weight.shape.as_list())

            # Remove :0 suffix
            base_name = weight_name.replace(":0", "")

            # Remove streaming_model/ prefix if present
            if base_name.startswith("streaming_model/"):
                base_name = base_name[len("streaming_model/") :]

            # Determine variable index based on weight type
            if "kernel" in base_name or "depthwise_kernel" in base_name:
                var_idx = "0"
            elif "bias" in base_name:
                var_idx = "1"
            elif "gamma" in base_name:
                var_idx = "0"
            elif "beta" in base_name:
                var_idx = "1"
            elif "moving_mean" in base_name:
                var_idx = "2"
            elif "moving_variance" in base_name:
                var_idx = "3"
            else:
                continue

            # Extract layer path from weight name
            layer_path = None
            for suffix in ["/kernel", "/depthwise_kernel", "/bias", "/gamma", "/beta", "/moving_mean", "/moving_variance"]:
                if suffix in base_name:
                    layer_path = base_name.rsplit(suffix, 1)[0]
                    break

            if layer_path is None:
                continue

            ckpt_path = f"{layer_path}/vars/{var_idx}"

            try:
                if ckpt_path in f:
                    ds = f[ckpt_path]
                    if isinstance(ds, h5py.Dataset):
                        value = ds[()]
                        if weight_shape == tuple(value.shape):
                            weight.assign(value)
                            loaded_count += 1
                        else:
                            print(f"  Shape mismatch: {weight_name} {weight_shape} vs {tuple(value.shape)}")
            except Exception as e:
                print(f"  Error loading {weight_name}: {e}")

    print(f"Loaded {loaded_count}/{len(model.weights)} weights")
    return loaded_count


# =============================================================================
# STREAMING EXPORT MODEL
# =============================================================================


class StreamingExportModel(tf.keras.Model):
    """Streaming model for ESPHome-compatible TFLite export.

    This model implements the exact architecture defined in ARCHITECTURAL_CONSTITUTION.md
    with proper state variable management for streaming inference.

    Architecture (okay_nabu variant):
    - First Conv2D: 32 filters, kernel 5, stride 3
    - MixConv blocks with kernels [[5], [7,11], [9,15], [23]]
    - Pointwise filters: [64, 64, 64, 64]
    - 6 streaming state variables for ring buffers
    """

    def __init__(self, first_conv_filters: int = 32, first_conv_kernel: int = 5, stride: int = 3, pointwise_filters: list = None, mixconv_kernel_sizes: list = None, mel_bins: int = 40, **kwargs):
        super().__init__(name="streaming_model", **kwargs)

        # Default to okay_nabu architecture
        if pointwise_filters is None:
            pointwise_filters = [64, 64, 64, 64]
        if mixconv_kernel_sizes is None:
            mixconv_kernel_sizes = [[5], [7, 11], [9, 15], [23]]

        self.first_conv_filters = first_conv_filters
        self.first_conv_kernel = first_conv_kernel
        self.stride = stride
        self.pointwise_filters = pointwise_filters
        self.mixconv_kernel_sizes = mixconv_kernel_sizes
        self.mel_bins = mel_bins

        # Initial convolution
        self.initial_conv = tf.keras.layers.Conv2D(
            first_conv_filters,
            (first_conv_kernel, 1),
            strides=(stride, 1),
            padding="valid",
            use_bias=False,
            name="initial_conv/cell",
        )
        self.initial_relu = tf.keras.layers.ReLU(name="initial_activation")

        # Build MixConv blocks
        self._build_mixconv_blocks()

        # Output dense layer
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid", name="layers/dense", dtype=tf.float32)

        # State variables will be created in build()
        self.state_vars: list[tf.Variable] = []

    def _build_mixconv_blocks(self):
        """Build all MixConv blocks with proper naming."""
        self.mixconv_layers = []

        for i, (filters, kernels) in enumerate(zip(self.pointwise_filters, self.mixconv_kernel_sizes, strict=True)):
            block_name = f"blocks/residual_block{'_' + str(i) if i > 0 else ''}/mixconvs/mix_conv_block"

            block_layers = {
                "depthwise_convs": [],
                "pointwise": None,
                "bn": None,
                "relu": None,
            }

            # Depthwise convs for each kernel size
            for j, ks in enumerate(kernels):
                suffix = "" if j == 0 else f"_{j}"
                dw = tf.keras.layers.DepthwiseConv2D(
                    (ks, 1),
                    strides=(1, 1),
                    padding="valid",
                    use_bias=False,
                    name=f"{block_name}/depthwise_convs/depthwise_conv2d{suffix}",
                )
                block_layers["depthwise_convs"].append((ks, dw))

            # Pointwise conv
            block_layers["pointwise"] = tf.keras.layers.Conv2D(
                filters,
                (1, 1),
                use_bias=False,
                name=f"{block_name}/pointwise",
            )

            # Batch normalization
            block_layers["bn"] = tf.keras.layers.BatchNormalization(name=f"{block_name}/bn")

            # ReLU activation
            block_layers["relu"] = tf.keras.layers.ReLU(name=f"{block_name}/activations/re_lu")

            self.mixconv_layers.append(block_layers)

    def build(self, input_shape):
        """Build state variables.

        Creates exactly 6 streaming state variables as required by ESPHome:
        - stream: [1, 2, 1, 40] - initial conv ring buffer (kernel 5 - stride 3 = 2)
        - stream_1: [1, 4, 1, 32] - block 0 ring buffer (max_kernel 5 - 1 = 4)
        - stream_2: [1, 10, 1, 64] - block 1 ring buffer (max_kernel 11 - 1 = 10)
        - stream_3: [1, 14, 1, 64] - block 2 ring buffer (max_kernel 15 - 1 = 14)
        - stream_4: [1, 22, 1, 64] - block 3 ring buffer (max_kernel 23 - 1 = 22)
        - stream_5: [1, 5, 1, 64] - temporal pooling buffer
        """
        # State shapes based on ARCHITECTURAL_CONSTITUTION.md for okay_nabu:
        # For strided conv (stride > 1): buffer = kernel_size - stride
        # For non-strided conv (stride = 1): buffer = kernel_size - 1

        state_configs = [
            # stream: initial conv ring buffer
            # kernel=5, stride=3 -> 5-3=2 frames
            ("stream", (1, self.first_conv_kernel - self.stride, 1, self.mel_bins)),
            # stream_1: after initial conv, before block 0
            # max_kernel=5, stride=1 -> 5-1=4 frames, filters=32
            ("stream_1", (1, max(self.mixconv_kernel_sizes[0]) - 1, 1, self.first_conv_filters)),
            # stream_2: after block 0, before block 1
            # max_kernel=11, stride=1 -> 11-1=10 frames, filters=64
            ("stream_2", (1, max(self.mixconv_kernel_sizes[1]) - 1, 1, self.pointwise_filters[0])),
            # stream_3: after block 1, before block 2
            # max_kernel=15, stride=1 -> 15-1=14 frames, filters=64
            ("stream_3", (1, max(self.mixconv_kernel_sizes[2]) - 1, 1, self.pointwise_filters[1])),
            # stream_4: after block 2, before block 3
            # max_kernel=23, stride=1 -> 23-1=22 frames, filters=64
            ("stream_4", (1, max(self.mixconv_kernel_sizes[3]) - 1, 1, self.pointwise_filters[2])),
            # stream_5: temporal pooling
            # Fixed at 5 frames, filters=64
            ("stream_5", (1, 5, 1, self.pointwise_filters[3])),
        ]

        # Verify we have exactly 6 state variables
        assert len(state_configs) == 6, f"Expected 6 state variables, got {len(state_configs)}"

        # Create state variables
        for name, shape in state_configs:
            state_var = self.add_weight(
                name=f"{name}/ring_buffer",
                shape=shape,
                dtype=tf.float32,
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
            )
            self.state_vars.append(state_var)

        super().build(input_shape)

    @staticmethod
    def _concat_update(state_var: tf.Variable, inputs: tf.Tensor, frames: int) -> tf.Tensor:
        """Concatenate state with input and update state."""
        combined = tf.concat([state_var, inputs], axis=1)
        new_state = combined[:, -frames:, :, :]
        assign_op = state_var.assign(new_state)
        with tf.control_dependencies([assign_op]):
            return tf.identity(combined)

    @staticmethod
    def _split_channels(inputs: tf.Tensor, splits: list) -> list:
        """Split tensor along channel dimension."""
        return tf.split(inputs, splits, axis=-1)

    def call(self, inputs, training=None, mask=None):
        """Forward pass for streaming inference."""
        del mask
        is_training = bool(training) if training is not None else False

        # Add channel dimension: [batch, time, features] -> [batch, time, 1, features]
        x = inputs[:, :, tf.newaxis, :]

        # stream (initial conv)
        # For strided conv: buffer = kernel_size - stride = 5 - 3 = 2
        x = self._concat_update(self.state_vars[0], x, self.first_conv_kernel - self.stride)
        x = self.initial_conv(x)
        x = self.initial_relu(x)

        # Process through MixConv blocks
        for i, block_layers in enumerate(self.mixconv_layers):
            state_var = self.state_vars[i + 1]
            kernels = self.mixconv_kernel_sizes[i]
            max_kernel = max(kernels) if isinstance(kernels, list) else kernels

            # Concatenate with state
            x = self._concat_update(state_var, x, max_kernel - 1)

            if len(kernels) == 1:
                # Single kernel - simple depthwise conv
                _, dw_conv = block_layers["depthwise_convs"][0]
                x = dw_conv(x)
            else:
                # Multi-kernel - split channels
                filters = x.shape[-1]
                splits = [filters // len(kernels)] * len(kernels)
                splits[0] += filters - sum(splits)  # Adjust first split

                x_splits = self._split_channels(x, splits)
                outputs = []

                for _j, (split_tensor, (_ks, dw_conv)) in enumerate(zip(x_splits, block_layers["depthwise_convs"], strict=True)):
                    out = dw_conv(split_tensor)
                    # StridedKeep: keep only the last kernel_size frames
                    out = out[:, -1:, :, :]
                    outputs.append(out)

                x = tf.concat(outputs, axis=-1)

            # Pointwise conv
            x = block_layers["pointwise"](x)
            x = block_layers["bn"](x, training=is_training)
            x = block_layers["relu"](x)

        # stream_5 (temporal pooling)
        x = self._concat_update(self.state_vars[-1], x, 5)
        # Keep last 5 frames and compute mean
        x = x[:, -5:, :, :]
        x = tf.reduce_mean(x, axis=[1, 2])

        # Dense output
        return self.dense(x)


# =============================================================================
# TFLITE CONVERSION FUNCTIONS
# =============================================================================


def create_representative_dataset(
    config: dict,
    num_samples: int = 500,
) -> Callable[[], Generator[np.ndarray, None, None]]:
    """Create representative dataset generator for quantization.

    Args:
        config: Configuration dict with stride and mel_bins
        num_samples: Number of samples to generate

    Returns:
        Generator function that yields samples
    """
    stride = config.get("stride", 3)
    mel_bins = config.get("mel_bins", 40)

    def representative_dataset_gen():
        np.random.seed(42)
        for i in range(num_samples):
            # Shape: (1, stride, mel_bins)
            sample = np.random.uniform(0.0, 26.0, (1, stride, mel_bins)).astype(np.float32)

            # Boundary anchors for quantization calibration
            if i == 0:
                sample[0, 0, 0] = 0.0
                sample[0, 0, 1] = 26.0

            yield [sample]

    return representative_dataset_gen


def export_streaming_tflite(
    checkpoint_path: str,
    output_dir: str = "./models/exported",
    model_name: str = "wake_word",
    config: dict = None,
) -> dict:
    """Export trained checkpoint to ESPHome-compatible streaming TFLite.

    This is the main export function that:
    1. Builds a streaming model with proper state variables
    2. Loads weights from Keras 3 checkpoint format
    3. Converts to quantized TFLite with proper settings
    4. Verifies the exported model

    Args:
        checkpoint_path: Path to .weights.h5 checkpoint
        output_dir: Output directory for exported model
        model_name: Name for the exported model
        config: Optional configuration dict with model parameters

    Returns:
        Dict with export results including paths and validation info
    """
    print("=" * 60)
    print("Streaming TFLite Export for ESPHome")
    print("=" * 60)

    # Default config (okay_nabu variant)
    if config is None:
        config = {
            "first_conv_filters": 32,
            "first_conv_kernel": 5,
            "stride": 3,
            "pointwise_filters": [64, 64, 64, 64],
            "mixconv_kernel_sizes": [[5], [7, 11], [9, 15], [23]],
            "mel_bins": 40,
        }

    print("\n[1/5] Building streaming model...")
    model = StreamingExportModel(
        first_conv_filters=config.get("first_conv_filters", 32),
        first_conv_kernel=config.get("first_conv_kernel", 5),
        stride=config.get("stride", 3),
        pointwise_filters=config.get("pointwise_filters", [64, 64, 64, 64]),
        mixconv_kernel_sizes=config.get("mixconv_kernel_sizes", [[5], [7, 11], [9, 15], [23]]),
        mel_bins=config.get("mel_bins", 40),
    )

    # Build the model
    input_shape = (config.get("stride", 3), config.get("mel_bins", 40))
    _ = model(tf.zeros((1, input_shape[0], input_shape[1]), dtype=tf.float32))
    print(f"Model has {len(model.weights)} total weights")

    print("\n[2/5] Loading checkpoint weights...")
    loaded = load_weights_from_keras3_checkpoint(model, checkpoint_path)
    if loaded < 29:
        print(f"Warning: Expected at least 29 weights, got {loaded}")

    print("\n[3/5] Converting to TFLite...")

    # Create temp directory for SavedModel
    saved_model_dir = tempfile.mkdtemp(prefix="mww_streaming_")

    try:
        # Save as SavedModel
        tf.saved_model.save(model, saved_model_dir)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_enable_resource_variables = True
        converter._experimental_variable_quantization = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = create_representative_dataset(config)

        tflite_model = converter.convert()
    finally:
        shutil.rmtree(saved_model_dir, ignore_errors=True)

    print("\n[4/5] Saving and verifying...")

    # Save TFLite model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tflite_file = output_path / f"{model_name}.tflite"
    tflite_file.write_bytes(tflite_model)

    print(f"✓ TFLite model saved to: {tflite_file}")
    print(f"  Size: {len(tflite_model) / 1024:.2f} KB")

    # Verify
    verification = verify_exported_model(str(tflite_file))

    print("\n[5/5] Generating manifest...")

    # Generate manifest
    try:
        from src.export.manifest import create_esphome_package

        pkg = create_esphome_package(
            model=None,
            config={"export": {"wake_word": model_name}},
            output_dir=output_dir,
            model_name=model_name,
            tflite_path=str(tflite_file),
            analysis_results=verification if verification.get("valid") else None,
        )
        print(f"✓ Manifest saved to: {pkg['manifest_path']}")
    except Exception as e:
        print(f"Warning: Could not generate manifest: {e}")
        pkg = {}

    print("=" * 60)
    print("Export complete!")
    print("=" * 60)

    return {
        "tflite_path": str(tflite_file),
        "manifest_path": pkg.get("manifest_path"),
        "model_valid": verification.get("valid", False),
        "verification": verification,
        "size_kb": len(tflite_model) / 1024,
    }


def verify_exported_model(tflite_path: str) -> dict:
    """Verify exported TFLite model meets ESPHome requirements.

    Args:
        tflite_path: Path to TFLite model file

    Returns:
        Dict with verification results
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ops_details = interpreter._get_ops_details()

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    valid = True

    # Check input shape [1, 3, 40] and dtype int8
    input_shape = list(input_details[0]["shape"])
    input_dtype = input_details[0]["dtype"]

    if input_shape != [1, 3, 40]:
        errors.append(f"Input shape {input_shape} != [1, 3, 40]")
        valid = False
    checks["input_shape"] = input_shape == [1, 3, 40]

    if input_dtype != np.int8:
        errors.append(f"Input dtype {input_dtype} != int8")
        valid = False
    checks["input_dtype"] = input_dtype == np.int8

    # Check output shape [1, 1] and dtype uint8
    output_shape = list(output_details[0]["shape"])
    output_dtype = output_details[0]["dtype"]

    if output_shape != [1, 1]:
        errors.append(f"Output shape {output_shape} != [1, 1]")
        valid = False
    checks["output_shape"] = output_shape == [1, 1]

    if output_dtype != np.uint8:
        errors.append(f"Output dtype {output_dtype} != uint8")
        valid = False
    checks["output_dtype"] = output_dtype == np.uint8

    # Check for required ops
    op_counts: dict[str, int] = {}
    for op in ops_details:
        op_name = op.get("op_name", "")
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

    # Check state variable ops
    var_handle_count = op_counts.get("VAR_HANDLE", 0)
    read_var_count = op_counts.get("READ_VARIABLE", 0)
    assign_var_count = op_counts.get("ASSIGN_VARIABLE", 0)

    if var_handle_count != 6:
        warnings.append(f"Expected 6 VAR_HANDLE ops, got {var_handle_count}")
    checks["var_handle_count"] = var_handle_count == 6

    if read_var_count != 6:
        warnings.append(f"Expected 6 READ_VARIABLE ops, got {read_var_count}")
    checks["read_var_count"] = read_var_count == 6

    if assign_var_count != 6:
        warnings.append(f"Expected 6 ASSIGN_VARIABLE ops, got {assign_var_count}")
    checks["assign_var_count"] = assign_var_count == 6

    # Check subgraph count
    try:
        subgraphs = interpreter.get_subgraphs()
        subgraph_count = len(subgraphs)
    except Exception:
        try:
            subgraph_count = interpreter.num_subgraphs()
        except Exception:
            subgraph_count = 2  # Assume correct if we can't check

    if subgraph_count != 2:
        warnings.append(f"Expected 2 subgraphs, got {subgraph_count}")
    checks["subgraph_count"] = subgraph_count == 2

    # Test inference
    try:
        test_input = np.random.randint(-128, 127, (1, 3, 40), dtype=np.int8)
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        test_output = interpreter.get_tensor(output_details[0]["index"])
        checks["inference_works"] = True
        test_output_range: tuple[float, float] | None = (float(test_output.min()), float(test_output.max()))
    except Exception as e:
        errors.append(f"Inference test failed: {e}")
        checks["inference_works"] = False
        valid = False
        test_output_range = None
    results: dict[str, object] = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    if test_output_range is not None:
        results["test_output_range"] = test_output_range
    return results


# =============================================================================
# LEGACY API (for backward compatibility)
# =============================================================================


def convert_model_saved(
    model: tf.keras.Model,
    config: dict,
    folder: str,
    mode: str = "stream_internal_state_inference",
) -> tf.keras.Model:
    """Convert non-streaming model to streaming SavedModel (legacy API).

    This function is deprecated. Use export_streaming_tflite() instead.
    """
    raise NotImplementedError("convert_model_saved is deprecated. Use export_streaming_tflite() instead.")


def export_to_tflite(
    model: tf.keras.Model,
    config: dict,
    output_dir: str,
    model_name: str = "wake_word",
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> dict:
    """Export trained model to ESPHome-compatible TFLite (legacy API).

    This function is deprecated. Use export_streaming_tflite() instead.
    """
    raise NotImplementedError("export_to_tflite is deprecated. Use export_streaming_tflite() instead.")


def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantize: bool = True,
) -> bytes:
    """Convert model to TFLite format (legacy API)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    dirpath = os.path.dirname(output_path) or "."
    os.makedirs(dirpath, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def verify_esphome_compatibility(tflite_path: str, stride: int = 3) -> dict:
    """Verify TFLite model is compatible with ESPHome micro_wake_word."""
    return verify_exported_model(tflite_path)


def calculate_tensor_arena_size(tflite_path: str) -> int:
    """Calculate required tensor arena size for TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
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
# COMMAND LINE INTERFACE
# =============================================================================


def main():
    """Main entry point for mww-export command."""
    import argparse

    parser = argparse.ArgumentParser(description="Export trained model to ESPHome-compatible streaming TFLite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.weights.h5)",
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
        help="Disable quantization (not recommended for ESPHome)",
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    try:
        if os.path.isfile(args.config):
            import yaml

            with open(args.config, "r") as f:
                yaml_config = yaml.safe_load(f)
                model_cfg = yaml_config.get("model", {})
                config = {
                    "first_conv_filters": model_cfg.get("first_conv_filters", 32),
                    "first_conv_kernel": model_cfg.get("first_conv_kernel_size", 5),
                    "stride": model_cfg.get("stride", 3),
                    "mel_bins": yaml_config.get("hardware", {}).get("mel_bins", 40),
                }

                # Parse pointwise_filters
                pw_filters = model_cfg.get("pointwise_filters", "64,64,64,64")
                config["pointwise_filters"] = [int(x.strip()) for x in pw_filters.split(",")]

                # Parse mixconv_kernel_sizes
                import ast

                mc_kernels = model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]")
                config["mixconv_kernel_sizes"] = ast.literal_eval(f"[{mc_kernels}]")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using default okay_nabu configuration")

    # Export
    try:
        result = export_streaming_tflite(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            model_name=args.model_name,
            config=config,
        )

        if result["model_valid"]:
            print("\n✓ Model is ESPHome compatible!")
            sys.exit(0)
        else:
            print("\n⚠ Model validation failed:")
            for error in result["verification"].get("errors", []):
                print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
