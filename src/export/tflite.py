"""Export module for model conversion to TFLite."""

import json
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator, Optional

# Suppress verbose TF/XLA logs before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import h5py
import numpy as np
import tensorflow as tf

from src.export.verification import compute_expected_state_shapes, verify_tflite_model
from src.model.architecture import build_core_layers

logger = logging.getLogger(__name__)


def get_checkpoint_metadata(checkpoint_path: str, pointwise_filters: int = 64) -> dict:
    """Get checkpoint metadata from cache or scan checkpoint.

    Caches metadata to a sidecar .metadata.json file for faster subsequent access.

    Args:
        checkpoint_path: Path to .weights.h5 checkpoint
        pointwise_filters: Number of pointwise convolution filters (default 64, matches architecture)

    Returns:
        Dict with temporal_frames, dense_input_features, dense_output_features
    """
    cache_path = Path(checkpoint_path).with_suffix(".metadata.json")

    required_keys = {
        "temporal_frames",
        "dense_input_features",
        "dense_output_features",
        "pointwise_filters",
    }
    cached_extras: dict = {}

    # Check cache first
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                metadata = json.load(f)
            if isinstance(metadata, dict):
                if required_keys.issubset(metadata.keys()):
                    logger.info(f"Loaded checkpoint metadata from cache: {cache_path}")
                    return metadata
                # Keep extra fields (e.g. autotune threshold metadata) and rescan required keys.
                cached_extras = {k: v for k, v in metadata.items() if k not in required_keys}
                logger.info(f"Metadata cache missing required keys, rescanning checkpoint: {cache_path}")
            else:
                logger.info(f"Metadata cache is not an object, rescanning checkpoint: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load metadata cache, will rescan: {e}")

    # Scan checkpoint to extract metadata
    logger.info(f"Scanning checkpoint for metadata: {checkpoint_path}")
    dense_input_features = None
    dense_output_features = None

    def find_dense_kernel(name, obj):
        nonlocal dense_input_features, dense_output_features
        if name == "layers/dense/vars/0" and isinstance(obj, h5py.Dataset):
            kernel_shape = obj.shape
            dense_input_features = kernel_shape[0]
            dense_output_features = kernel_shape[1]

    with h5py.File(checkpoint_path, "r") as f:
        f.visititems(find_dense_kernel)

    if dense_input_features is None or dense_output_features is None:
        raise ValueError(f"Dense layer kernel not found in checkpoint: {checkpoint_path}")

    temporal_frames = dense_input_features // pointwise_filters

    metadata = {
        "temporal_frames": int(temporal_frames),
        "dense_input_features": int(dense_input_features),
        "dense_output_features": int(dense_output_features),
        "pointwise_filters": int(pointwise_filters),
    }
    if cached_extras:
        metadata.update(cached_extras)

    # Cache for future exports
    try:
        with open(cache_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Cached checkpoint metadata: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to cache metadata: {e}")

    return metadata


@contextmanager
def _suppress_tf_flatbuffer_warnings():
    """Suppress TensorFlow C++ warnings about ignored flatbuffer options.

    tf_tfl_flatbuffer_helpers.cc logs warnings about 'output_format'
    and 'drop_control_dependency' being ignored. These are harmless
    internal messages that clutter the export logs.

    These warnings originate from TF's C++ runtime and are written
    directly to file descriptor 2 (stderr), so Python-level warning
    filters cannot catch them. We redirect the OS-level fd instead.
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull)


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

    # Validate minimum weight count to detect incomplete checkpoint loading.
    # A model with residual connections has >=44 weights; without residuals >=29.
    # If dramatically fewer weights loaded, something went wrong.
    min_expected = 29
    if loaded_count < min_expected:
        raise ValueError(
            f"Weight loading failed: only {loaded_count} weights loaded from {checkpoint_path}. "
            f"Expected at least {min_expected}. This may indicate an architecture mismatch "
            f"between the checkpoint and the export model."
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
        residual_connections: Optional[list[int]] = None,
        mel_bins: int = 40,
        temporal_frames: int = 32,  # Added: inferred from checkpoint during export
        **kwargs,
    ):
        super().__init__(name="streaming_model", **kwargs)
        self.temporal_frames = temporal_frames  # Save for use in build() method

        # Default to okay_nabu architecture
        if pointwise_filters is None:
            pointwise_filters = [64, 64, 64, 64]
        if mixconv_kernel_sizes is None:
            mixconv_kernel_sizes = [[5], [7, 11], [9, 15], [23]]
        if residual_connections is None:
            residual_connections = [0, 1, 1, 1]
        if len(residual_connections) != len(pointwise_filters):
            raise ValueError(f"residual_connections length ({len(residual_connections)}) must match pointwise_filters length ({len(pointwise_filters)})")

        self.first_conv_filters = first_conv_filters
        self.first_conv_kernel = first_conv_kernel
        self.stride = stride
        self.pointwise_filters = pointwise_filters
        self.mixconv_kernel_sizes = mixconv_kernel_sizes
        self.residual_connections = residual_connections
        self.mel_bins = mel_bins

        core_layers = build_core_layers(
            first_conv_filters=first_conv_filters,
            first_conv_kernel_size=first_conv_kernel,
            stride=stride,
            pointwise_filters=self.pointwise_filters,
            mixconv_kernel_sizes=self.mixconv_kernel_sizes,
            repeat_in_block=[1] * len(self.pointwise_filters),
            residual_connections=self.residual_connections,
            l2_regularization=0.0,
        )

        # Initial convolution and activation (shared factory)
        self.initial_conv = core_layers["initial_conv_cell"]
        self.initial_relu = core_layers["initial_relu"]

        # Export model call() expects list[dict] with depthwise/pointwise/bn/relu/residual* keys
        self.mixconv_layers = [cfg["export_layers"] for cfg in core_layers["blocks"]]

        # Output dense layer
        self.dense = core_layers["dense"]

        # State variables will be created in build()
        self.state_vars: list[tf.Variable] = []
        # BN folding state — set True after calling fold_batch_norms()
        self._bn_folded: bool = False
        self._folded_biases: list = []
        self._residual_folded_biases: list = []

    def build(self, input_shape):
        """Build state variables.

        Creates exactly 6 streaming state variables following the official
        okay_nabu reference naming and shapes:

        - stream:   [1, 2, 1, 40] - input-side buffer before the first strided conv
        - stream_1: [1, 4, 1, 32] - block 0 depthwise-context buffer
        - stream_2: [1, 10, 1, 64] - block 1 depthwise-context buffer
        - stream_3: [1, 14, 1, 64] - block 2 depthwise-context buffer
        - stream_4: [1, 22, 1, 64] - block 3 depthwise-context buffer
        - stream_5: [1, temporal_frames - 1, 1, 64] - pre-flatten temporal buffer

        Buffer rules:
        - `stream` uses `first_conv_kernel - global_stride`
        - `stream_1`..`stream_4` use `effective_temporal_kernel - 1`
        - `stream_5` uses `temporal_frames - 1`
        """
        # State shapes based on ARCHITECTURAL_CONSTITUTION.md for okay_nabu.
        # Important distinction:
        # - stream is the only input-side buffer that uses kernel_size - stride
        # - downstream block buffers use max_kernel - 1 because the streaming
        #   depthwise context at that stage is stride-1

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
            # stream_5: pre-flatten temporal buffer (inferred from checkpoint)
            # temporal_frames - 1 = retained history before flatten/dense
            ("stream_5", (1, self.temporal_frames - 1, 1, self.pointwise_filters[3])),
        ]

        # Verify we have exactly 6 state variables
        # Create state variables
        for name, shape in state_configs:
            state_var = self.add_weight(
                name=name,
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

        Also folds residual BN into residual projection conv where applicable.

        Must be called after weights are loaded from checkpoint.  After this
        the model is mathematically equivalent but BN layers are bypassed in
        call() so no ReadVariableOp is emitted for BN statistics.
        """
        if self._bn_folded:
            raise RuntimeError("fold_batch_norms() has already been applied on this model instance")

        self._folded_biases = []
        self._residual_folded_biases = []
        for block in self.mixconv_layers:
            # Fold main path BN into pointwise conv
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

            # Fold residual BN into residual projection conv (if present)
            if block["residual_proj"] is not None:
                res_bn = block["residual_bn"]
                res_pw = block["residual_proj"]

                res_mean = res_bn.moving_mean.numpy()
                res_variance = res_bn.moving_variance.numpy()
                res_eps = float(res_bn.epsilon)
                res_inv_std = 1.0 / np.sqrt(res_variance + res_eps)
                res_gamma = res_bn.gamma.numpy() if res_bn.gamma is not None else np.ones_like(res_mean)
                res_beta = res_bn.beta.numpy() if res_bn.beta is not None else np.zeros_like(res_mean)

                res_scale = res_gamma * res_inv_std

                res_kernel = res_pw.kernel.numpy()
                res_pw.kernel.assign(res_kernel * res_scale[np.newaxis, np.newaxis, np.newaxis, :])

                self._residual_folded_biases.append(tf.constant(res_beta - res_scale * res_mean, dtype=tf.float32))
            else:
                self._residual_folded_biases.append(None)

        self._bn_folded = True

    def call(self, inputs, training=None, mask=None):
        """Forward pass for streaming inference."""
        del mask
        is_training = bool(training) if training is not None else False

        # Add channel dimension: [batch, time, features] -> [batch, time, 1, features]
        x = inputs[:, :, tf.newaxis, :]

        # stream (initial conv)
        # This is the only input-side buffer that uses kernel_size - stride.
        x = self._concat_update(self.state_vars[0], x, self.first_conv_kernel - self.stride)
        x = self.initial_conv(x)
        x = self.initial_relu(x)

        # Process through MixConv blocks
        for i, block_layers in enumerate(self.mixconv_layers):
            state_var = self.state_vars[i + 1]
            kernels = self.mixconv_kernel_sizes[i]
            max_kernel = max(kernels) if isinstance(kernels, list) else kernels

            # Compute residual BEFORE ring buffer update (on the raw 1-frame input)
            residual = None
            if block_layers["residual_proj"] is not None:
                residual = block_layers["residual_proj"](x)
                if self._bn_folded:
                    residual = residual + tf.reshape(self._residual_folded_biases[i], [1, 1, 1, -1])
                else:
                    residual = block_layers["residual_bn"](residual, training=is_training)

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

            # Add residual skip connection after activation
            if residual is not None:
                x = x + residual

        # stream_5 (pre-flatten temporal buffer — dimension inferred from checkpoint)
        # Concat new frame with ring buffer: [1, (temporal_frames-1)+1, 1, 64]
        # and save last (temporal_frames-1) frames to state for next call
        x = self._concat_update(self.state_vars[-1], x, self.temporal_frames - 1)
        # Flatten full concat: [1, temporal_frames, 1, 64] → [1, temporal_frames*64]
        x = tf.reshape(x, [1, -1])

        # Dense output
        return self.dense(x)


# =============================================================================
# TFLITE CONVERSION FUNCTIONS
# =============================================================================


def create_representative_dataset(
    config: dict,
    num_samples: int | None = None,
) -> Callable[[], Generator[list[np.ndarray], None, None]]:
    """Create representative dataset from random noise for quantization (FALLBACK ONLY).

    WARNING: This fallback uses random noise which will NOT trigger wake word
    detection. The model's output quantization range may be miscalibrated
    (capped at ~0.5 instead of full [0, 1] range). Always prefer
    ``create_representative_dataset_from_data()`` which feeds sequential chunks
    from real spectrograms including positive samples.

    Args:
        config: Configuration dict with mel_bins
        num_samples: Number of samples to generate

    Returns:
        Generator function that yields samples
    """
    mel_bins = int(config.get("mel_bins", 40))
    stride = int(config.get("stride", 3))
    export_cfg = config.get("export", {})
    raw_num_samples = num_samples if num_samples is not None else export_cfg.get("representative_dataset_size", 1000)
    if raw_num_samples is None:
        raw_num_samples = 1000
    num_samples = int(raw_num_samples)

    def representative_dataset_gen():
        # Boundary anchors for correct INT8 quantization range calibration
        anchor_min = np.zeros((1, stride, mel_bins), dtype=np.float32)  # 0.0
        anchor_max = np.full((1, stride, mel_bins), 26.0, dtype=np.float32)  # 26.0
        yield [anchor_min]
        yield [anchor_max]

        rng = np.random.RandomState(42)  # Local RNG for reproducible calibration
        for _ in range(num_samples):
            # Shape: (1, stride, mel_bins) - matches streaming model input shape
            sample = rng.uniform(0.0, 26.0, (1, stride, mel_bins)).astype(np.float32)
            yield [sample]

    return representative_dataset_gen


def create_representative_dataset_from_data(
    config: dict,
    data_dir: str,
    num_samples: int | None = None,
) -> Callable[[], Generator[list[np.ndarray], None, None]]:
    """Create representative dataset from real training features for quantization.

    Loads preprocessed spectrograms from FeatureStore and yields stride-sized
    chunks **sequentially** from each spectrogram. This is critical for correct
    INT8 quantization of streaming models — the TFLite converter runs the model
    forward through yielded samples, so state accumulates across sequential
    chunks just like during real inference.

    Includes both positive (wake word) and negative spectrograms (~30% positive)
    so the model produces both high and low confidence outputs during calibration,
    giving the quantizer the full output range to calibrate against.

    Without sequential feeding, every calibration chunk starts from zero state,
    the model never sees accumulated context, output is clamped to ~0.5, and the
    Dense layer's quantization range covers only negative logits.

    Args:
        config: Configuration dict with mel_bins, stride, export settings
        data_dir: Path to processed data directory (containing train/ subfolder)
        num_samples: Approximate number of calibration chunks to generate

    Returns:
        Generator function that yields [np.ndarray] samples for TFLite calibration
    """
    from src.data.dataset import FeatureStore

    mel_bins = int(config.get("mel_bins", 40))
    stride = int(config.get("stride", 3))
    export_cfg = config.get("export", {})
    raw_num_samples = num_samples if num_samples is not None else export_cfg.get("representative_dataset_real_size", 4000)
    if raw_num_samples is None:
        raw_num_samples = 4000
    target_chunks = int(raw_num_samples)
    store_path = Path(data_dir) / "train"

    if not store_path.exists():
        print(f"  Warning: No training data at {store_path}, falling back to random calibration")
        return create_representative_dataset(config, target_chunks)

    store = FeatureStore(store_path)
    try:
        store.open()
    except (FileNotFoundError, OSError) as e:
        print(f"  Warning: Could not open feature store: {e}, falling back to random calibration")
        return create_representative_dataset(config, target_chunks)

    try:
        n_stored = len(store)
        if n_stored == 0:
            print("  Warning: Feature store is empty, falling back to random calibration")
            return create_representative_dataset(config, target_chunks)

        # Separate positive and negative sample indices
        positive_indices = []
        negative_indices = []
        for idx in range(n_stored):
            _spec, label = store.get(idx)
            if label == 1:
                positive_indices.append(idx)
            else:
                negative_indices.append(idx)

        rng = np.random.RandomState(42)
        rng.shuffle(positive_indices)
        rng.shuffle(negative_indices)

        # Build ordered list of full spectrograms to feed sequentially.
        # Target ~30% positive for output range coverage.
        # Each spectrogram yields multiple stride-sized chunks.

        # Estimate avg chunks per spectrogram from first few samples
        sample_indices = (positive_indices[:5] if positive_indices else []) + negative_indices[:5]
        avg_chunks_per_spec = 0
        for idx in sample_indices:
            spec, _ = store.get(idx)
            if spec.ndim == 1:
                if spec.size % mel_bins != 0:
                    continue
                spec = spec.reshape(-1, mel_bins)
            avg_chunks_per_spec += spec.shape[0] // stride
        if sample_indices:
            avg_chunks_per_spec = max(1, avg_chunks_per_spec // len(sample_indices))
        else:
            avg_chunks_per_spec = 10  # Reasonable default

        # Calculate how many spectrograms we need
        target_positive_chunks = int(target_chunks * 0.3)
        target_negative_chunks = target_chunks - target_positive_chunks

        n_positive_specs = min(
            len(positive_indices),
            max(1, target_positive_chunks // avg_chunks_per_spec) if positive_indices else 0,
        )
        n_negative_specs = min(
            len(negative_indices),
            max(1, target_negative_chunks // avg_chunks_per_spec),
        )

        # Interleave: groups of negative specs followed by positive specs
        # This ensures state accumulates through negatives, then positives
        # trigger high confidence — giving calibration both extremes
        selected_positive = positive_indices[:n_positive_specs]
        selected_negative = negative_indices[:n_negative_specs]

        # Build interleaved order: chunks of negatives, then a positive
        neg_per_group = max(1, n_negative_specs // max(1, n_positive_specs))
        ordered_indices: list[int] = []
        neg_idx = 0
        pos_idx = 0

        while neg_idx < len(selected_negative) or pos_idx < len(selected_positive):
            # Add a group of negatives
            for _ in range(neg_per_group):
                if neg_idx < len(selected_negative):
                    ordered_indices.append(selected_negative[neg_idx])
                    neg_idx += 1
            # Add one positive
            if pos_idx < len(selected_positive):
                ordered_indices.append(selected_positive[pos_idx])
                pos_idx += 1

        # Remaining negatives
        while neg_idx < len(selected_negative):
            ordered_indices.append(selected_negative[neg_idx])
            neg_idx += 1

        # Extract sequential chunks from each spectrogram
        all_chunks: list[np.ndarray] = []
        n_positive_used = 0
        n_negative_used = 0

        for idx in ordered_indices:
            spec, label = store.get(idx)
            if spec.ndim == 1:
                if spec.size % mel_bins != 0:
                    continue
                spec = spec.reshape(-1, mel_bins)
            n_frames = spec.shape[0]
            if n_frames < stride:
                continue

            # Yield ALL stride-sized chunks in temporal order from this spectrogram
            for t in range(0, n_frames - stride + 1, stride):
                chunk = spec[t : t + stride].reshape(1, stride, mel_bins).astype(np.float32)
                all_chunks.append(chunk)

            if label == 1:
                n_positive_used += 1
            else:
                n_negative_used += 1

        if not all_chunks:
            print("  Warning: Could not extract any chunks, falling back to random calibration")
            return create_representative_dataset(config, target_chunks)

        positive_pct = (n_positive_used / max(1, n_positive_used + n_negative_used)) * 100
        print(
            f"  Using {len(all_chunks)} sequential calibration chunks from "
            f"{n_positive_used + n_negative_used} spectrograms "
            f"({n_positive_used} positive [{positive_pct:.0f}%], {n_negative_used} negative)"
        )

        def representative_dataset_gen():
            # Boundary anchors for correct INT8 input range calibration
            anchor_min = np.zeros((1, stride, mel_bins), dtype=np.float32)  # 0.0
            anchor_max = np.full((1, stride, mel_bins), 26.0, dtype=np.float32)  # 26.0
            yield [anchor_min]
            yield [anchor_max]

            # Yield chunks in sequential order — state accumulates across
            # chunks from the same spectrogram, allowing the model to
            # produce high-confidence outputs on positive samples
            for chunk in all_chunks:
                yield [chunk]

        return representative_dataset_gen
    finally:
        store.close()


def export_streaming_tflite(
    checkpoint_path: str,
    output_dir: str = "./models/exported",
    model_name: str = "wake_word",
    config: Optional[dict] = None,
    data_dir: Optional[str] = None,
    quantize: bool = True,
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
        data_dir: Optional path to processed data dir for real-data quantization calibration


    Returns:
        Dict with export results including paths and validation info
    """
    print("=" * 60)
    print("Streaming TFLite Export for ESPHome")
    print("=" * 60)

    # First, read checkpoint metadata to infer temporal dimensions
    print("\n[0/5] Analyzing checkpoint architecture...")
    # Assume 64 pointwise filters in last block (standard architecture)
    if config is not None:
        pw_filters_list = config.get("pointwise_filters", [64, 64, 64, 64])
        pointwise_filters = pw_filters_list[-1] if pw_filters_list else 64
    else:
        pointwise_filters = 64  # Standard architecture default

    metadata = get_checkpoint_metadata(checkpoint_path, pointwise_filters=pointwise_filters)
    temporal_frames = metadata["temporal_frames"]
    dense_input_features = metadata["dense_input_features"]
    dense_output_features = metadata["dense_output_features"]

    if dense_input_features % pointwise_filters != 0:
        raise ValueError(f"Dense input features ({dense_input_features}) is not divisible by pointwise_filters ({pointwise_filters}). Architecture mismatch?")

    print(f"  Checkpoint Dense layer: ({dense_input_features}, {dense_output_features})")
    print(f"  Inferred temporal frames: {temporal_frames} (from {dense_input_features} / {pointwise_filters})")

    # Default config (okay_nabu variant)
    if config is None:
        config = {
            "first_conv_filters": 32,
            "first_conv_kernel": 5,
            "stride": 3,
            "pointwise_filters": [64, 64, 64, 64],
            "mixconv_kernel_sizes": [[5], [7, 11], [9, 15], [23]],
            "residual_connections": [0, 1, 1, 1],
            "mel_bins": 40,
        }
    print("\n[1/5] Building streaming model...")
    model = StreamingExportModel(
        first_conv_filters=config.get("first_conv_filters", 32),
        first_conv_kernel=config.get("first_conv_kernel", 5),
        stride=config.get("stride", 3),
        pointwise_filters=config.get("pointwise_filters", [64, 64, 64, 64]),
        mixconv_kernel_sizes=config.get("mixconv_kernel_sizes", [[5], [7, 11], [9, 15], [23]]),
        residual_connections=config.get("residual_connections", [0, 1, 1, 1]),
        mel_bins=config.get("mel_bins", 40),
        temporal_frames=temporal_frames,  # Pass inferred temporal dimension from checkpoint
    )

    # Build model with stride-sized input — this is the shape ESPHome feeds per call.
    # temporal_frames is only used internally for stream_5 ring buffer size; it must
    # NOT appear in the exported input signature or ESPHome will read stride=temporal_frames
    # from input->dims->data[1] and run at the wrong cadence.
    _ = model(tf.zeros((1, config.get("stride", 3), config.get("mel_bins", 40)), dtype=tf.float32))

    print("\n[2/5] Loading checkpoint weights...")
    loaded = load_weights_from_keras3_checkpoint(model, checkpoint_path)
    if loaded < 29:
        raise RuntimeError(f"Weight loading incomplete: expected at least 29 weights (44 with residuals), got {loaded}. Checkpoint may be corrupt or incompatible: {checkpoint_path}")

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
                shape=(1, config.get("stride", 3), config.get("mel_bins", 40)),  # stride frames per call, NOT temporal_frames
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
        if quantize:
            converter.optimizations = {tf.lite.Optimize.DEFAULT}
            converter.experimental_enable_resource_variables = True
            # Quantize the payload tensors flowing through READ_VARIABLE /
            # ASSIGN_VARIABLE. VAR_HANDLE resource handles themselves are not
            # quantized payload data.
            converter._experimental_variable_quantization = True
            converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.uint8
            if data_dir:
                converter.representative_dataset = create_representative_dataset_from_data(config, data_dir)
            else:
                converter.representative_dataset = create_representative_dataset(config)
        else:
            converter.experimental_enable_resource_variables = True

        with _suppress_tf_flatbuffer_warnings():
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
    expected_shapes = compute_expected_state_shapes(
        first_conv_kernel=config.get("first_conv_kernel", 5),
        stride=config.get("stride", 3),
        mel_bins=config.get("mel_bins", 40),
        first_conv_filters=config.get("first_conv_filters", 32),
        mixconv_kernel_sizes=config.get("mixconv_kernel_sizes", [[5], [7, 11], [9, 15], [23]]),
        pointwise_filters=config.get("pointwise_filters", [64, 64, 64, 64]),
        temporal_frames=temporal_frames,
    )
    verification = verify_exported_model(str(tflite_file), expected_state_shapes=expected_shapes)

    print("\n[5/5] Generating manifest...")

    # Generate manifest
    try:
        from src.export.manifest import create_esphome_package

        manifest_config = _build_manifest_config(config, model_name, metadata)
        tuned_cutoff = manifest_config.get("export", {}).get("probability_cutoff")
        if isinstance(tuned_cutoff, float):
            print(f"  Manifest probability_cutoff: {tuned_cutoff:.4f}")

        pkg = create_esphome_package(
            model=None,
            config=manifest_config,
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


def _build_manifest_config(config: Optional[dict], model_name: str, metadata: dict) -> dict:
    """Build manifest-relevant config with optional tuned cutoff override."""
    export_cfg: dict = dict(config.get("export", {}) if isinstance(config, dict) else {})
    hardware_cfg: dict = dict(config.get("hardware", {}) if isinstance(config, dict) else {})

    tuned_cutoff = metadata.get("tuned_probability_cutoff")
    if isinstance(tuned_cutoff, (int, float)) and 0.0 < float(tuned_cutoff) <= 1.0:
        export_cfg["probability_cutoff"] = float(tuned_cutoff)

    export_cfg.setdefault("wake_word", model_name)
    # Prefer detection_threshold from evaluation section, then top-level config, then default 0.5
    if isinstance(config, dict):
        evaluation_cfg = dict(config.get("evaluation", {}))
        _detection_threshold = export_cfg.get("detection_threshold") or evaluation_cfg.get("detection_threshold") or config.get("detection_threshold", 0.5)
    else:
        _detection_threshold = 0.5
    export_cfg.setdefault("detection_threshold", _detection_threshold)

    return {
        "export": export_cfg,
        "hardware": hardware_cfg,
    }


def verify_exported_model(tflite_path: str, expected_state_shapes: list[tuple[int, ...]] | None = None) -> dict:
    """Verify exported TFLite model meets ESPHome requirements.

    Args:
        tflite_path: Path to TFLite model file

    Returns:
        Dict with verification results
    """
    return verify_tflite_model(tflite_path, expected_state_shapes=expected_state_shapes)


# =============================================================================
# =============================================================================


def convert_model_saved(
    model: tf.keras.Model,
    config: dict,
    folder: str,
    mode: str = "stream_internal_state_inference",
) -> tf.keras.Model:
    """Convert non-streaming model to streaming SavedModel (Article IX Stage 1).

    This implements the mandated Stage 1 conversion from ARCHITECTURAL_CONSTITUTION.md:
    - Takes a non-streaming trained model
    - Converts to streaming with internal state inference
    - Produces SavedModel with proper state variables

    Args:
        model: Non-streaming trained model (e.g., MixedNet in NON_STREAM mode)
        config: Configuration dict with model parameters
        folder: Output directory for streaming SavedModel
        mode: Must be "stream_internal_state_inference" (Article IX requirement)

    Returns:
        Streaming model ready for Stage 2 TFLite conversion
    """
    import os

    if mode != "stream_internal_state_inference":
        raise ValueError(f"mode must be 'stream_internal_state_inference', got {mode}")

    os.makedirs(folder, exist_ok=True)

    print("[Stage 1] Converting non-streaming model to streaming SavedModel...")
    print(f"  Input model: {model.name}")
    print(f"  Mode: {mode}")

    # Build streaming export model with state variables
    # Infer temporal_frames from training model to support non-default configs
    pointwise_filters = config.get("pointwise_filters", [64, 64, 64, 64])
    last_pw = pointwise_filters[-1] if isinstance(pointwise_filters, list) else 64
    # Dense kernel shape: (input_features, output_units). Infer temporal frames from input features step
    dense_kernel = model.output_dense.kernel
    temporal_frames = dense_kernel.shape[0] // last_pw

    streaming_model = StreamingExportModel(
        first_conv_filters=config.get("first_conv_filters", 32),
        first_conv_kernel=config.get("first_conv_kernel", 5),
        stride=config.get("stride", 3),
        pointwise_filters=config.get("pointwise_filters", [64, 64, 64, 64]),
        mixconv_kernel_sizes=config.get("mixconv_kernel_sizes", [[5], [7, 11], [9, 15], [23]]),
        residual_connections=config.get("residual_connections", [0, 1, 1, 1]),
        mel_bins=config.get("mel_bins", 40),
        temporal_frames=temporal_frames,
    )

    # Build and transfer weights
    input_shape = (config.get("stride", 3), config.get("mel_bins", 40))
    _ = streaming_model(tf.zeros((1, input_shape[0], input_shape[1]), dtype=tf.float32))

    # Transfer weights from non-streaming model to streaming model
    # This simulates the conversion process
    streaming_model.set_weights(model.get_weights())

    # Fold batch norms for clean export
    streaming_model.fold_batch_norms()

    # Export to SavedModel
    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(streaming_model)

    export_input_sig = [
        tf.TensorSpec(
            shape=(1, input_shape[0], input_shape[1]),
            dtype=tf.float32,
            name="inputs",
        )
    ]

    def serve_fn(inputs: tf.Tensor) -> tf.Tensor:
        return streaming_model(inputs, training=False)

    export_archive.add_endpoint(
        name="serve",
        fn=serve_fn,
        input_signature=export_input_sig,
    )
    export_archive.write_out(folder)

    print(f"  ✓ Streaming SavedModel saved to: {folder}")
    print(f"  ✓ Model has {len(streaming_model.state_vars)} state variables")

    return streaming_model


def verify_esphome_compatibility(tflite_path: str, stride: int = 3) -> dict:
    """Verify TFLite model is compatible with ESPHome micro_wake_word."""
    return verify_exported_model(tflite_path)


def calculate_tensor_arena_size(tflite_path: str) -> int:
    """Calculate required tensor arena size using canonical manifest logic."""
    from src.export.manifest import calculate_tensor_arena_size as _calculate_tensor_arena_size

    return _calculate_tensor_arena_size(tflite_path)


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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to processed data directory for real-data quantization calibration",
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
                export_cfg = yaml_config.get("export", {})
                config["export"] = {
                    "wake_word": export_cfg.get("wake_word", args.model_name),
                    "author": export_cfg.get("author", "Sarpel GURAY"),
                    "website": export_cfg.get("website", "https://github.com/sarpel/microwakeword-training-platform"),
                    "trained_languages": export_cfg.get("trained_languages", ["en"]),
                    "probability_cutoff": export_cfg.get("probability_cutoff", 0.97),
                    "sliding_window_size": export_cfg.get("sliding_window_size", 5),
                    "tensor_arena_size": export_cfg.get("tensor_arena_size"),
                    "minimum_esphome_version": export_cfg.get("minimum_esphome_version", "2024.7.0"),
                    "arena_size_margin": export_cfg.get("arena_size_margin", 1.3),
                }
                config["hardware"] = {
                    "window_step_ms": yaml_config.get("hardware", {}).get("window_step_ms", 10),
                }

                # Parse pointwise_filters
                pw_filters = model_cfg.get("pointwise_filters", "64,64,64,64")
                config["pointwise_filters"] = [int(x.strip()) for x in pw_filters.split(",")]

                # Parse mixconv_kernel_sizes
                import ast

                mc_kernels = model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]")
                config["mixconv_kernel_sizes"] = ast.literal_eval(f"[{mc_kernels}]")

                # Parse residual_connection
                res_conn = model_cfg.get("residual_connection", "0,1,1,1")
                config["residual_connections"] = [int(x.strip()) for x in res_conn.split(",")]
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
            data_dir=args.data_dir,
            quantize=not args.no_quantize,
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
