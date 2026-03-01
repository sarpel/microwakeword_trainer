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

from src.export.verification import verify_tflite_model

# =============================================================================
# KERAS 3 CHECKPOINT LOADER
# =============================================================================


def load_weights_from_keras3_checkpoint(model: tf.keras.Model, checkpoint_path: str) -> int:
    """Load weights from Keras 3 format .weights.h5 checkpoint.

    In Keras 3, weight.name is the bare variable name (e.g. "kernel", "gamma") with
    no layer-path prefix.  We therefore iterate over layers directly, using
    layer.name as the flat identifier and weight.name as the variable type.

    The checkpoint uses a hierarchical HDF5 layout (e.g. "initial_conv/cell/vars/0").
    StreamingExportModel layer names are the same paths with '/' replaced by '_'
    (e.g. "initial_conv_cell"), so a simple replace-based lookup resolves them.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to .weights.h5 checkpoint

    Returns:
        Number of weights successfully loaded
    """
    # Map Keras 3 weight names to checkpoint var index
    _VAR_IDX = {
        "kernel": "0",  # Conv2D, DepthwiseConv2D, Dense kernel
        "bias": "1",  # Dense bias
        "gamma": "0",  # BatchNorm gamma
        "beta": "1",  # BatchNorm beta
        "moving_mean": "2",  # BatchNorm moving_mean
        "moving_variance": "3",  # BatchNorm moving_variance
    }

    print(f"Loading weights from: {checkpoint_path}")

    loaded_count = 0
    with h5py.File(checkpoint_path, "r") as f:
        # Build lookup map: (flat_layer_name, var_idx_str) -> hdf5_dataset_path.
        # Checkpoint hierarchical paths (e.g. "initial_conv/cell/vars/0") are
        # converted to flat names by replacing '/' with '_'.
        ckpt_map: dict = {}

        def _collect_datasets(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and "/vars/" in name:
                parts = name.rsplit("/vars/", 1)
                ckpt_map[(parts[0].replace("/", "_"), parts[1])] = name

        f.visititems(_collect_datasets)

        consumed_keys: set[tuple[str, str]] = set()
        model_layer_names: set[str] = set()

        # Iterate over layers (not model.weights) so we have layer.name available.
        for layer in model._flatten_layers(include_self=False):
            if not layer.weights:
                continue
            flat_name = layer.name
            model_layer_names.add(flat_name)
            for weight in layer.weights:
                var_idx = _VAR_IDX.get(weight.name)
                if var_idx is None:
                    continue  # state/ring-buffer variables — not in checkpoint

                ckpt_path = ckpt_map.get((flat_name, var_idx))
                if ckpt_path is None:
                    continue

                try:
                    ds = f[ckpt_path]
                    if isinstance(ds, h5py.Dataset):
                        value = ds[()]
                        weight_shape = tuple(weight.shape.as_list())
                        if weight_shape == tuple(value.shape):
                            weight.assign(value)
                            loaded_count += 1
                            consumed_keys.add((flat_name, var_idx))
                        else:
                            print(f"  Shape mismatch: {flat_name}/{weight.name} {weight_shape} vs {tuple(value.shape)}")
                except Exception as e:
                    print(f"  Error loading {flat_name}/{weight.name}: {e}")

        unexpected_keys = [key for key in ckpt_map if key[0] in model_layer_names and key[1] in {"0", "1", "2", "3"} and key not in consumed_keys]
        if unexpected_keys:
            sample = ", ".join(f"{name}/vars/{idx}" for name, idx in unexpected_keys[:5])
            raise ValueError(
                f"Checkpoint has unsupported or unmatched layer variables for this export architecture (examples: {sample}). This usually indicates architecture drift between training and export."
            )

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

    def __init__(
        self,
        first_conv_filters: int = 32,
        first_conv_kernel: int = 5,
        stride: int = 3,
        pointwise_filters: Optional[list[int]] = None,
        mixconv_kernel_sizes: Optional[list[list[int]]] = None,
        mel_bins: int = 40,
        **kwargs,
    ):
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
            name="initial_conv_cell",
        )
        self.initial_relu = tf.keras.layers.ReLU(name="initial_activation")

        # Build MixConv blocks
        self._build_mixconv_blocks()

        # Output dense layer
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid", name="layers_dense", dtype=tf.float32)

        # State variables will be created in build()
        self.state_vars: list[tf.Variable] = []
        # BN folding state — set True after calling fold_batch_norms()
        self._bn_folded: bool = False
        self._folded_biases: list = []

    def _build_mixconv_blocks(self):
        """Build all MixConv blocks with proper naming."""
        self.mixconv_layers = []

        for i, (filters, kernels) in enumerate(zip(self.pointwise_filters, self.mixconv_kernel_sizes, strict=True)):
            block_name = f"blocks_residual_block{'_' + str(i) if i > 0 else ''}_mixconvs_mix_conv_block"

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
                    name=f"{block_name}_depthwise_convs_depthwise_conv2d{suffix}",
                )
                block_layers["depthwise_convs"].append((ks, dw))

            # Pointwise conv
            block_layers["pointwise"] = tf.keras.layers.Conv2D(
                filters,
                (1, 1),
                use_bias=False,
                name=f"{block_name}_pointwise",
            )

            # Batch normalization
            block_layers["bn"] = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn")

            # ReLU activation
            block_layers["relu"] = tf.keras.layers.ReLU(name=f"{block_name}_activations_re_lu")

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
                name=f"{name}_ring_buffer",
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

    def fold_batch_norms(self) -> None:
        """Fold BN statistics into the preceding pointwise conv kernels.

        Keras 3 wraps BN variable reads in Cast ops, which breaks the MLIR
        FreezeGlobalTensors pass used during TFLite INT8 quantization.  This
        method eliminates BN resource variables from the export graph entirely
        by absorbing gamma/beta/mean/variance into the pointwise conv weights.

        Must be called after weights are loaded from checkpoint.  After this
        the model is mathematically equivalent but BN layers are bypassed in
        call() so no ReadVariableOp is emitted for BN statistics.
        """
        if self._bn_folded:
            raise RuntimeError("fold_batch_norms() has already been applied on this model instance")

        self._folded_biases = []
        for block in self.mixconv_layers:
            bn = block["bn"]
            pw = block["pointwise"]

            mean = bn.moving_mean.numpy()  # (out_ch,)
            variance = bn.moving_variance.numpy()  # (out_ch,)
            eps = float(bn.epsilon)
            inv_std = 1.0 / np.sqrt(variance + eps)
            gamma = bn.gamma.numpy() if bn.gamma is not None else np.ones_like(mean)
            beta = bn.beta.numpy() if bn.beta is not None else np.zeros_like(mean)

            scale = gamma * inv_std  # (out_ch,)

            # Fold scale into pointwise kernel: (1, 1, in_ch, out_ch)
            kernel = pw.kernel.numpy()
            pw.kernel.assign(kernel * scale[np.newaxis, np.newaxis, np.newaxis, :])

            # Store folded bias as a tf.constant (no tf.Variable → no ReadVariableOp
            # in the traced graph)
            self._folded_biases.append(tf.constant(beta - scale * mean, dtype=tf.float32))

        self._bn_folded = True

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
                if filters is None:
                    raise ValueError("Channel dimension must be statically known for MixConv splitting")
                filters_int = int(filters)
                splits = [filters_int // len(kernels)] * len(kernels)
                splits[0] += filters_int - sum(splits)  # Adjust first split

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
            if self._bn_folded:
                # BN is folded into pointwise kernel; add the folded bias only
                x = x + tf.reshape(self._folded_biases[i], [1, 1, 1, -1])
            else:
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
) -> Callable[[], Generator[list[np.ndarray], None, None]]:
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
    config: Optional[dict] = None,
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

    # Fold BN statistics into pointwise conv weights so that no Keras BN
    # variable reads appear in the TFLite export graph (Keras 3 wraps them in
    # Cast ops that break MLIR's FreezeGlobalTensors pass).
    model.fold_batch_norms()
    print("  ✓ BatchNorm folded into pointwise convolutions")

    print("\n[3/5] Converting to TFLite...")

    # Create temp directory for SavedModel
    saved_model_dir = tempfile.mkdtemp(prefix="mww_streaming_")

    try:
        export_archive = tf.keras.export.ExportArchive()
        export_archive.track(model)

        export_input_sig = [
            tf.TensorSpec(
                shape=(1, input_shape[0], input_shape[1]),
                dtype=tf.float32,
                name="inputs",
            )
        ]

        def serve_fn(inputs: tf.Tensor) -> tf.Tensor:
            return model(inputs, training=False)

        export_archive.add_endpoint(
            name="serve",
            fn=serve_fn,
            input_signature=export_input_sig,
        )
        export_archive.write_out(saved_model_dir)

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
    return verify_tflite_model(tflite_path)


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
