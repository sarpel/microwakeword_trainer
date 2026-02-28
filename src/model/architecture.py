"""Model architecture module for wake word detection - MixedNet Implementation."""

"""
MixedNet architecture based on microWakeWord:
- MixConv: Mixed depthwise convolutions with varying kernel sizes
- Streaming support with ring buffers for real-time inference
- Quantization-ready design for edge deployment
"""

import ast
import logging

import tensorflow as tf

# Import from our own streaming module
from src.model.streaming import (
    ChannelSplit,
    Modes,
    Stream,
    StridedDrop,
    StridedKeep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL PARAMETERS
# =============================================================================


def parse_model_param(text):
    """Parse model parameters from string.

    Args:
        text: String with layer parameters (e.g., '128,128' or '[5],[9]')

    Returns:
        List of parsed parameters
    """
    if not text:
        return []
    try:
        res = ast.literal_eval(text)
        if isinstance(res, tuple):
            return list(res)
        elif isinstance(res, list):
            # Handle nested lists like [[5], [9]]
            if res and isinstance(res[0], list):
                return [list(lst) if isinstance(lst, list) else lst for lst in res]
            return res
        return [res]
    except (ValueError, SyntaxError) as exc:
        logger.error("parse_model_param: failed to parse %r: %s", text, exc)
        raise ValueError(f"Cannot parse model parameter {text!r}: {exc}") from exc


def spectrogram_slices_dropped(flags):
    """Compute spectrogram slices dropped due to valid padding.

    Args:
        flags: Model parameters object or dict with:
            - first_conv_filters: int
            - first_conv_kernel_size: int
            - repeat_in_block: list
            - mixconv_kernel_sizes: list
            - stride: int

    Returns:
        int: Number of spectrogram slices dropped
    """
    spectrogram_slices_dropped = 0

    # Handle both object and dict access patterns
    if hasattr(flags, "first_conv_filters"):
        first_conv_filters = flags.first_conv_filters
    else:
        first_conv_filters = flags.get("first_conv_filters", 0)

    if hasattr(flags, "first_conv_kernel_size"):
        first_conv_kernel_size = flags.first_conv_kernel_size
    else:
        first_conv_kernel_size = flags.get("first_conv_kernel_size", 5)

    if hasattr(flags, "repeat_in_block"):
        repeat_in_block = flags.repeat_in_block
    else:
        repeat_in_block = parse_model_param(flags.get("repeat_in_block", "1,1,1,1"))

    if hasattr(flags, "mixconv_kernel_sizes"):
        mixconv_kernel_sizes = flags.mixconv_kernel_sizes
    else:
        mixconv_kernel_sizes = parse_model_param(flags.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"))

    if hasattr(flags, "stride"):
        stride = flags.stride
    else:
        stride = flags.get("stride", 1)

    if first_conv_filters > 0:
        # First conv contribution (NO stride scaling per upstream formula)
        first_conv_contribution = first_conv_kernel_size - 1
        spectrogram_slices_dropped += first_conv_contribution

    # Block contributions ARE scaled by stride (per upstream formula)
    block_contributions = 0
    for repeat, ksize in zip(repeat_in_block, mixconv_kernel_sizes, strict=False):
        # ksize can be a list like [5] or [9, 11]
        max_ksize = max(ksize) if isinstance(ksize, list) else ksize
        block_contributions += repeat * (max_ksize - 1) * stride

    spectrogram_slices_dropped += block_contributions
    return spectrogram_slices_dropped


def _split_channels(total_filters, num_groups):
    """Split channels into groups for MixConv.

    Args:
        total_filters: Total number of filters
        num_groups: Number of groups to split into

    Returns:
        List of filter counts per group
    """
    split = [total_filters // num_groups] * num_groups
    split[0] += total_filters - sum(split)
    return split


# =============================================================================
# MIXCONV BLOCK
# =============================================================================


class MixConvBlock(tf.keras.layers.Layer):
    """MixConv block with mixed depthwise convolutional kernels.

    MDConv mixes multiple kernels (e.g., 3x1, 5x1) by splitting channels
    into groups and applying different kernels to each group.

    Attributes:
        kernel_sizes: List of kernel sizes for depthwise conv
        filters: Number of output filters for pointwise conv
        mode: Inference mode (TRAINING, NON_STREAM, STREAM_INTERNAL, STREAM_EXTERNAL)
    """

    def __init__(self, kernel_sizes, filters=None, mode=Modes.NON_STREAM_INFERENCE, **kwargs):
        super().__init__(**kwargs)
        self.kernel_sizes = kernel_sizes if isinstance(kernel_sizes, list) else [kernel_sizes]
        self.filters = filters
        self.mode = mode
        # Ring buffer length is max kernel size - 1
        self.ring_buffer_length = max(self.kernel_sizes) - 1

    def build(self, input_shape):
        # Pointwise projection
        if self.filters is not None:
            self.pointwise = tf.keras.layers.Conv2D(
                self.filters,
                (1, 1),
                strides=1,
                padding="same",
                use_bias=False,
                name="pointwise",
            )
            self.bn = tf.keras.layers.BatchNormalization(name="bn")

        # Create depthwise conv layers for each kernel size
        self.depthwise_convs = []
        for i, ks in enumerate(self.kernel_sizes):
            suffix = "" if i == 0 else f"_{i}"
            self.depthwise_convs.append(
                tf.keras.layers.DepthwiseConv2D(
                    (ks, 1),
                    strides=1,
                    padding="valid",
                    use_bias=False,
                    name=f"depthwise_convs_depthwise_conv2d{suffix}",
                )
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass with MixConv logic.

        Args:
            inputs: Input tensor [batch, time, 1, channels]
            training: Training flag

        Returns:
            Output tensor after MixConv
        """
        net = inputs

        # Single kernel size - simple depthwise conv
        if len(self.kernel_sizes) == 1:
            # Causal padding for non-streaming modes
            if self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
                pad_amount = self.kernel_sizes[0] - 1
                if pad_amount > 0:
                    net = tf.pad(net, [[0, 0], [pad_amount, 0], [0, 0], [0, 0]], "constant")
            net = self.depthwise_convs[0](net)
        else:
            # Multiple kernel sizes - split channels and apply different convs
            filters = net.shape[-1]
            splits = _split_channels(filters, len(self.kernel_sizes))

            # Split channels
            x_splits = ChannelSplit(splits, axis=-1)(net)

            x_outputs = []
            for i, (x, ks) in enumerate(zip(x_splits, self.kernel_sizes, strict=False)):
                if self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
                    # Per-kernel causal padding ensures all outputs have
                    # the same time dimension (= input time dimension)
                    pad_amount = ks - 1
                    if pad_amount > 0:
                        x = tf.pad(x, [[0, 0], [pad_amount, 0], [0, 0], [0, 0]], "constant")
                else:
                    # Streaming: StridedKeep trims ring buffer for this kernel
                    x = StridedKeep(ks, mode=self.mode)(x)

                # Depthwise conv with this kernel size
                x = self.depthwise_convs[i](x)
                x_outputs.append(x)

            # Concatenate along channel dimension
            # Per-kernel padding ensures all outputs have matching time dimensions
            net = tf.keras.layers.Concatenate(axis=-1)(x_outputs)

        # Apply pointwise projection and BN
        if self.filters is not None:
            net = self.pointwise(net)
            net = self.bn(net)

        return net

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_sizes": self.kernel_sizes,
                "filters": self.filters,
                "mode": getattr(self.mode, "value", self.mode),
            }
        )
        return config


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with MixConv and optional skip connection.

    Attributes:
        filters: Number of filters in pointwise conv
        kernel_sizes: List of kernel sizes for MixConv
        repeat: Number of MixConv layers in the block
        use_residual: Whether to use residual connection
        mode: Inference mode
    """

    def __init__(
        self,
        filters,
        kernel_sizes,
        repeat=1,
        use_residual=False,
        mode=Modes.NON_STREAM_INFERENCE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes if isinstance(kernel_sizes, list) else [kernel_sizes]
        self.repeat = repeat
        self.use_residual = use_residual
        self.mode = mode

    def build(self, input_shape):
        self.mixconvs = []
        self.activations = []

        for i in range(self.repeat):
            mixconv_name = "mixconvs_mix_conv_block" if i == 0 else f"mixconvs_mix_conv_block_{i}"
            self.mixconvs.append(
                MixConvBlock(
                    kernel_sizes=self.kernel_sizes,
                    filters=self.filters,
                    mode=self.mode,
                    name=mixconv_name,
                )
            )
            self.activations.append(tf.keras.layers.ReLU(name=f"relu_{i}"))

        # Residual projection
        if self.use_residual:
            self.residual_proj = tf.keras.layers.Conv2D(
                self.filters,
                (1, 1),
                strides=1,
                padding="same",
                use_bias=False,
                name="residual_proj",
            )
            self.residual_bn = tf.keras.layers.BatchNormalization(name="residual_bn")

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass with residual connection."""
        net = inputs

        if self.use_residual:
            # Compute residual once before the mix-conv loop
            residual = self.residual_proj(inputs)
            residual = self.residual_bn(residual)
        else:
            residual = None

        for mixconv, activation in zip(self.mixconvs, self.activations, strict=False):
            net = mixconv(net, training=training)
            net = activation(net)

        # Apply residual addition once after all mix-convs
        if self.use_residual:
            assert residual is not None
            # Align time dimensions if needed
            if residual.shape[1] is not None and net.shape[1] is not None and residual.shape[1] != net.shape[1]:
                diff = residual.shape[1] - net.shape[1]
                if diff < 0:
                    raise ValueError(f"Residual has fewer time steps than net before StridedDrop (diff={diff}, residual={residual.shape[1]}, net={net.shape[1]}).")
                residual = StridedDrop(diff, mode=self.mode)(residual)
            net = net + residual

        return net

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_sizes": self.kernel_sizes,
                "repeat": self.repeat,
                "use_residual": self.use_residual,
                "mode": getattr(self.mode, "value", self.mode),
            }
        )
        return config


# =============================================================================
# MIXEDNET MODEL
# =============================================================================


class MixedNet(tf.keras.Model):
    """MixedNet model for wake word detection.

    Based on MixConv: Mixed Depthwise Convolutional Kernels
    https://arxiv.org/abs/1907.09595

    Supports streaming inference with internal state management.

    Attributes:
        first_conv_filters: Filters in initial Conv2D
        first_conv_kernel_size: Kernel size for initial conv
        stride: Stride for initial conv (3 for 30ms inference)
        pointwise_filters: List of filters for each MixConv block
        mixconv_kernel_sizes: List of kernel size lists per block
        repeat_in_block: Number of repeats per block
        residual_connections: Which blocks use residual connections
        mode: Inference mode
    """

    def __init__(
        self,
        input_shape=(100, 40),
        first_conv_filters=32,
        first_conv_kernel_size=5,
        stride=3,
        pointwise_filters=None,
        mixconv_kernel_sizes=None,
        repeat_in_block=None,
        residual_connections=None,
        dropout_rate=0.2,
        l2_regularization=0.0001,
        mode=Modes.NON_STREAM_INFERENCE,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store the provided input shape for get_config serialization
        self._input_shape_arg = input_shape

        # Default configurations
        self.first_conv_filters = first_conv_filters
        self.first_conv_kernel_size = first_conv_kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        self.mode = mode

        # Parse list parameters
        if pointwise_filters is None:
            pointwise_filters = [64, 64, 64, 64]
        if mixconv_kernel_sizes is None:
            mixconv_kernel_sizes = [[5], [7, 11], [9, 15], [23]]
        if repeat_in_block is None:
            repeat_in_block = [1, 1, 1, 1]
        if residual_connections is None:
            residual_connections = [0, 0, 0, 0]

        self.pointwise_filters = pointwise_filters
        self.mixconv_kernel_sizes = mixconv_kernel_sizes
        self.repeat_in_block = repeat_in_block
        self.residual_connections = residual_connections

        # Validate parameter lengths
        num_blocks = len(pointwise_filters)
        for name, param in [
            ("repeat_in_block", repeat_in_block),
            ("mixconv_kernel_sizes", mixconv_kernel_sizes),
            ("residual_connections", residual_connections),
        ]:
            if len(param) != num_blocks:
                raise ValueError(f"{name} length ({len(param)}) must match pointwise_filters length ({num_blocks})")

        # Input specification - accept 3D input [batch, time, features]
        self.input_spec = tf.keras.layers.InputSpec(shape=(None, *input_shape), dtype=tf.float32)  # Allow any batch size

    def build(self, input_shape):
        """Build the model layers."""
        # Input reshape: [batch, time, features] -> [batch, time, 1, features]
        # This adds channel dimension for Conv2D operations

        # Initial Conv2D with streaming wrapper
        if self.first_conv_filters > 0:
            self.initial_conv = Stream(
                cell=tf.keras.layers.Conv2D(
                    self.first_conv_filters,
                    (self.first_conv_kernel_size, 1),
                    strides=(self.stride, 1),
                    padding="valid",
                    use_bias=False,
                    kernel_regularizer=(tf.keras.regularizers.l2(self.l2_regularization) if self.l2_regularization else None),
                    name="cell",
                ),
                mode=self.mode,
                use_one_step=False,
                pad_time_dim=None,
                pad_freq_dim="valid",
                name="initial_conv_cell",
            )
            self.initial_activation = tf.keras.layers.ReLU(name="initial_relu")

        # MixConv blocks
        self.blocks = []
        for i, (filters, repeat, ksize, res) in enumerate(
            zip(
                self.pointwise_filters,
                self.repeat_in_block,
                self.mixconv_kernel_sizes,
                self.residual_connections,
                strict=False,
            ),
        ):
            block_name = "residual_block" if i == 0 else f"residual_block_{i}"
            block = ResidualBlock(
                filters=filters,
                kernel_sizes=ksize,
                repeat=repeat,
                use_residual=bool(res),
                mode=self.mode,
                name=f"blocks_{block_name}",
            )
            self.blocks.append(block)

        # Streaming for temporal pooling
        # Calculate ring_buffer_size based on input shape and stride
        # After stride 3 conv: time_dim = ceil(input_shape[0] / stride)
        # ring_buffer_size = time_dim - 1
        input_shape_list = tf.TensorShape(input_shape).as_list() if input_shape is not None else None
        if input_shape_list and input_shape_list[0] is not None:
            temporal_rb_size = max(0, (input_shape_list[0] + self.stride - 1) // self.stride - 1)
        else:
            temporal_rb_size = 0
        self.temporal_stream = Stream(
            cell=tf.keras.layers.Identity(),
            ring_buffer_size_in_time_dim=temporal_rb_size,
            use_one_step=False,
            name="temporal_stream",
        )

        # Global pooling (average or flatten)
        self.pooling = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")

        # Dropout
        if self.dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout = None

        # Output layer - must be float32 for numerical stability with mixed precision
        self.output_dense = tf.keras.layers.Dense(1, activation="sigmoid", name="layers_dense", dtype=tf.float32)

        # Flatten layer for dense (created in build to avoid re-creation on each call)
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """Forward pass.

        Args:
            inputs: Input tensor [batch, time, features]
            training: Training flag
            mask: Optional mask for streaming (alias for state)

        Returns:
            Output tensor [batch, 1] with wake word probability
        """
        net = inputs

        # Add channel dimension: [batch, time, features] -> [batch, time, 1, features]
        # Use tf.expand_dims for dynamic shape handling
        if len(net.shape) == 3:
            net = tf.expand_dims(net, axis=2)

        # Initial Conv2D with streaming
        if self.first_conv_filters > 0:
            stream_training = training if training is not None else False
            net = self.initial_conv(net, training=stream_training)
            net = self.initial_activation(net)

        # MixConv blocks
        for block in self.blocks:
            net = block(net, training=training)

        # Temporal streaming before pooling
        time_dim = net.shape[1]
        if time_dim is not None and time_dim > 1:
            # Use cached temporal_stream layer instead of creating new one
            net = self.temporal_stream(net)
        # Global pooling
        net = self.pooling(net)

        # Flatten for dense
        net = self.flatten(net)

        # Dropout
        if self.dropout is not None:
            dropout_training = training if training is not None else False
            net = self.dropout(net, training=dropout_training)

        # Output
        net = self.output_dense(net)

        return net

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self._input_shape_arg,
                "first_conv_filters": self.first_conv_filters,
                "first_conv_kernel_size": self.first_conv_kernel_size,
                "stride": self.stride,
                "pointwise_filters": self.pointwise_filters,
                "mixconv_kernel_sizes": self.mixconv_kernel_sizes,
                "repeat_in_block": self.repeat_in_block,
                "residual_connections": self.residual_connections,
                "dropout_rate": self.dropout_rate,
                "l2_regularization": self.l2_regularization,
                "mode": getattr(self.mode, "value", self.mode),
            }
        )
        return config


# =============================================================================
# MODEL FACTORY FUNCTIONS
# =============================================================================


def build_model(
    input_shape=(100, 40),
    num_classes=2,
    first_conv_filters=32,
    first_conv_kernel_size=5,
    stride=3,
    pointwise_filters="64,64,64,64",
    mixconv_kernel_sizes="[5],[7,11],[9,15],[23]",
    repeat_in_block="1,1,1,1",
    residual_connection="0,0,0,0",
    dropout_rate=0.0,
    l2_regularization=0.0,
    mode="non_stream",
    **kwargs,
):
    """Build MixedNet model with specified parameters.

    Args:
        input_shape: Input feature shape (time_steps, mel_bins)
        num_classes: Number of output classes (unused, binary classification)
        first_conv_filters: Number of filters in initial Conv2D
        first_conv_kernel_size: Kernel size for initial conv
        stride: Stride for initial conv (3 = 30ms inference)
        pointwise_filters: Comma-separated filter counts
        mixconv_kernel_sizes: Kernel size lists per block
        repeat_in_block: Number of repeats per block
        residual_connection: Which blocks use residuals (0 or 1)
        dropout_rate: Dropout rate
        l2_regularization: L2 regularization factor
        mode: Inference mode ("training", "non_stream", "stream_internal", "stream_external")

    Returns:
        MixedNet model
    """
    # Parse mode
    mode_map = {
        "training": Modes.TRAINING,
        "non_stream": Modes.NON_STREAM_INFERENCE,
        "stream_internal": Modes.STREAM_INTERNAL_STATE_INFERENCE,
        "stream_external": Modes.STREAM_EXTERNAL_STATE_INFERENCE,
    }
    if mode not in mode_map:
        logger.warning(
            "build_model: unknown mode %r, defaulting to 'non_stream'. Valid modes are: %r",
            mode,
            list(mode_map.keys()),
        )
    mode_enum = mode_map.get(mode, Modes.NON_STREAM_INFERENCE)

    # Parse list parameters
    pointwise_list = parse_model_param(pointwise_filters)
    mixconv_list = parse_model_param(mixconv_kernel_sizes)
    repeat_list = parse_model_param(repeat_in_block)
    residual_list = parse_model_param(residual_connection)

    model = MixedNet(
        input_shape=input_shape,
        first_conv_filters=first_conv_filters,
        first_conv_kernel_size=first_conv_kernel_size,
        stride=stride,
        pointwise_filters=pointwise_list,
        mixconv_kernel_sizes=mixconv_list,
        repeat_in_block=repeat_list,
        residual_connections=residual_list,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
        mode=mode_enum,
    )

    return model


# =============================================================================
# okay_nabu CONFIGURATION
# =============================================================================


def create_okay_nabu_model(input_shape=(100, 40), mode=Modes.NON_STREAM_INFERENCE):
    """Create MixedNet model for 'okay_nabu' wake word.

    Configuration with strided_keep and SPLIT_V operations.
    Uses multi-scale kernel sizes for feature extraction.

    Args:
        input_shape: Input feature shape
        mode: Inference mode

    Returns:
        MixedNet model
    """
    return MixedNet(
        input_shape=input_shape,
        first_conv_filters=32,
        first_conv_kernel_size=5,
        stride=3,
        pointwise_filters=[64, 64, 64, 64],
        mixconv_kernel_sizes=[[5], [7, 11], [9, 15], [23]],
        repeat_in_block=[1, 1, 1, 1],
        residual_connections=[0, 0, 0, 0],
        mode=mode,
    )
