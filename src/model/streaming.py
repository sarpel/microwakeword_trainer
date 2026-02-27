"""Streaming model module for real-time inference."""

"""
Streaming layers and state management for wake word detection.

Implements:
- Stream wrapper layer with ring buffer state management
- Modes enum for different inference modes
- StridedDrop and StridedKeep for streaming dimension management
- RingBuffer for efficient state storage
- ChannelSplit for MixConv channel partitioning

Tensor Shapes:
- Input: [batch, time, features] or [batch, time, 1, features]
- Streaming input: [batch, stride, features] where stride=3 for 30ms
- Output: [batch, 1] probability

State Variables (for TFLite):
- stream: Ring buffer for initial conv
- stream_1-5: Ring buffers for MixConv blocks
- Total memory: ~2.8KB-3.5KB

References:
- microWakeWord: https://github.com/OHF-Voice/micro-wake-word
- Google Research kws_streaming: https://github.com/google-research/google-research/tree/master/kws_streaming
"""

import tensorflow as tf

# =============================================================================
# MODES ENUMERATION
# =============================================================================


class Modes:
    """Definition of the mode the model is functioning in."""

    # Model is in a training state. No streaming is done.
    TRAINING = "TRAINING"

    # Below are three options for inference:

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is kept inside of the model
    STREAM_INTERNAL_STATE_INFERENCE = "STREAM_INTERNAL_STATE_INFERENCE"

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is received from outside of the model
    STREAM_EXTERNAL_STATE_INFERENCE = "STREAM_EXTERNAL_STATE_INFERENCE"

    # Model is in inference mode and its topology is the same with training
    # mode (with removed dropouts etc)
    NON_STREAM_INFERENCE = "NON_STREAM_INFERENCE"


# =============================================================================
# RING BUFFER STATE MANAGEMENT
# =============================================================================


class RingBuffer:
    """Ring buffer for streaming state management.

    Efficient circular buffer for storing temporal state in streaming inference.
    Used internally by the Stream layer for state management.

    Attributes:
        size: Maximum number of time steps to store
        dtype: Data type for the buffer
    """

    def __init__(self, size, dtype=tf.float32):
        """Initialize ring buffer.

        Args:
            size: Number of time steps to buffer
            dtype: Tensor data type
        """
        self.size = size
        self.dtype = dtype
        self.buffer = None

    def initialize(self, batch_size, feature_dims):
        """Initialize buffer with zeros.

        Args:
            batch_size: Batch size
            feature_dims: Feature dimensions (without batch)

        Returns:
            Initialized zero tensor
        """
        shape = (batch_size, *feature_dims)
        self.buffer = tf.zeros(shape, dtype=self.dtype)
        return self.buffer

    def update(self, new_data):
        """Update buffer with new data (shift and append).

        Args:
            new_data: New time step data [batch, 1, ...]
                The time axis of new_data must be exactly 1.

        Returns:
            Updated buffer [batch, size, ...]

        Raises:
            ValueError: If new_data time dimension is not 1.
        """
        # Validate that exactly 1 time step is provided
        time_dim = new_data.shape[1] if hasattr(new_data, "shape") else None
        if time_dim is not None and time_dim != 1:
            raise ValueError(
                f"RingBuffer.update expects new_data with time dim=1, got {time_dim}. "
                "Pass one step at a time."
            )

        if self.buffer is None:
            self.buffer = new_data
            return self.buffer

        # Shift buffer left and append new data.
        # Always use [:, 1:, ...] so we get a consistent (size-1) prefix,
        # even when self.size == 1 (producing an empty intermediate).
        shifted = self.buffer[:, 1:, ...]
        self.buffer = tf.concat([shifted, new_data], axis=1)
        return self.buffer


# =============================================================================
# STREAM WRAPPER LAYER
# =============================================================================


class Stream(tf.keras.layers.Layer):
    """Streaming wrapper for Keras layers.

    Wraps Keras layers for streaming inference with internal or external state.
    Manages ring buffer for temporal state in streaming mode.

    Supports multiple modes:
    1. TRAINING: Normal training mode
    2. NON_STREAM_INFERENCE: Non-streaming inference
    3. STREAM_INTERNAL_STATE_INFERENCE: Streaming with internal state
    4. STREAM_EXTERNAL_STATE_INFERENCE: Streaming with external state

    Attributes:
        cell: Keras layer to wrap
        inference_batch_size: Batch size for inference
        mode: Inference mode
        pad_time_dim: Padding in time dimension ('causal', 'same', or None)
        state_shape: Shape of the state tensor
        ring_buffer_size_in_time_dim: Size of ring buffer
        use_one_step: True for single-step inference, False for strided
        state_name_tag: Name tag for state variables
        pad_freq_dim: Padding in frequency dimension
    """

    def __init__(
        self,
        cell,
        inference_batch_size=1,
        mode=Modes.TRAINING,
        pad_time_dim=None,
        state_shape=None,
        ring_buffer_size_in_time_dim=None,
        use_one_step=True,
        state_name_tag="ExternalState",
        pad_freq_dim="valid",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if pad_freq_dim not in ["same", "valid"]:
            raise ValueError(f"Unsupported padding in frequency: `{pad_freq_dim}`")

        self.cell = cell
        self.inference_batch_size = inference_batch_size
        self.mode = mode
        self.pad_time_dim = pad_time_dim
        self.state_shape = state_shape
        self.ring_buffer_size_in_time_dim = ring_buffer_size_in_time_dim
        self.use_one_step = use_one_step
        self.state_name_tag = state_name_tag
        self.pad_freq_dim = pad_freq_dim

        self.stride = 1
        self.stride_freq = 1
        self.dilation_freq = 1
        self.kernel_size_freq = 1

        # State variables for streaming
        self.states = None
        self.input_state = None
        self.output_state = None

    def get_core_layer(self):
        """Get the core layer being wrapped."""
        core_layer = self.cell
        # Handle wrapper layers (unwrapping once is sufficient)
        if isinstance(core_layer, tf.keras.layers.Wrapper):
            core_layer = core_layer.layer
        return core_layer

    def build(self, input_shape):
        """Build the layer and state variables."""
        wrapped_cell = self.get_core_layer()

        # Handle Conv layers
        if isinstance(
            wrapped_cell,
            (
                tf.keras.layers.Conv1D,
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv1D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv1D,
                tf.keras.layers.SeparableConv2D,
            ),
        ):
            config = wrapped_cell.get_config()
            strides = config.get("strides", (1, 1))
            dilation_rate = config.get("dilation_rate", (1, 1))
            kernel_size = config.get("kernel_size", (1, 1))
            padding = config.get("padding", "valid")

            # Normalize scalar values to 2-element tuples so indexing is safe
            if isinstance(strides, int):
                strides = (strides, strides)
            if isinstance(dilation_rate, int):
                dilation_rate = (dilation_rate, dilation_rate)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            self.stride = strides[0]
            self.stride_freq = strides[1] if len(strides) > 1 else self.stride
            self.dilation_freq = (
                dilation_rate[1] if len(dilation_rate) > 1 else dilation_rate[0]
            )
            self.kernel_size_freq = (
                kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
            )

            # Calculate ring buffer size
            if self.use_one_step:
                self.ring_buffer_size_in_time_dim = (
                    (dilation_rate[0] * (kernel_size[0] - 1) + 1)
                    if isinstance(dilation_rate, (tuple, list))
                    else (dilation_rate * (kernel_size - 1) + 1)
                )
            else:
                # For strided conv
                dilation = (
                    dilation_rate[0]
                    if isinstance(dilation_rate, (tuple, list))
                    else dilation_rate
                )
                kern = (
                    kernel_size[0]
                    if isinstance(kernel_size, (tuple, list))
                    else kernel_size
                )
                stride_val = strides[0] if isinstance(strides, tuple) else strides
                # Per IMPLEMENTATION_PLAN.md: kernel_size - stride = buffer_size
                # With dilation: dilation * (kernel_size - 1) - (stride - 1)
                self.ring_buffer_size_in_time_dim = max(
                    0, dilation * (kern - 1) - stride_val + 1
                )

        # Build the wrapped cell if needed
        if isinstance(wrapped_cell, tf.keras.layers.Layer) and not wrapped_cell.built:
            # For streaming, use None for time dimension
            faked_shape = list(input_shape)
            if len(faked_shape) >= 2:
                faked_shape[1] = None  # Streaming dimension
            wrapped_cell.build(tf.TensorShape(faked_shape))

        # Calculate state shape
        if self.ring_buffer_size_in_time_dim and self.ring_buffer_size_in_time_dim > 0:
            # State shape: [batch, ring_buffer_size, ...features]
            shape_as_list = (
                list(input_shape)
                if isinstance(input_shape, (list, tuple))
                else tf.TensorShape(input_shape).as_list()
            )
            self.state_shape = [
                self.inference_batch_size,
                self.ring_buffer_size_in_time_dim,
            ] + shape_as_list[2:]

        # Create state variable for internal streaming
        if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
            if (
                self.ring_buffer_size_in_time_dim
                and self.ring_buffer_size_in_time_dim > 0
            ):
                self.states = self.add_weight(
                    name="states",
                    shape=self.state_shape,
                    trainable=False,
                    initializer=tf.zeros_initializer(),
                    dtype=tf.float32,
                )

        # Create input state for external streaming
        elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            if (
                self.ring_buffer_size_in_time_dim
                and self.ring_buffer_size_in_time_dim > 0
            ):
                self.input_state = tf.keras.layers.Input(
                    shape=self.state_shape[1:],
                    batch_size=self.inference_batch_size,
                    name=f"{self.name}/{self.state_name_tag}",
                    dtype=tf.float32,
                )

        super().build(input_shape)

    def call(self, inputs, training=None, state=None):
        """Forward pass with streaming logic.

        Args:
            inputs: Input tensor
            training: Training flag
            state: Optional runtime state tensor for external streaming.
                   If provided, overrides self.input_state.

        Returns:
            Output tensor, or (output, output_state) for external streaming
        """
        if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
            return self._streaming_internal_state(inputs)
        elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            return self._streaming_external_state(inputs, state=state)
        elif self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
            return self._non_streaming(inputs, training)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _streaming_internal_state(self, inputs):
        """Streaming with internal state management."""
        if self.use_one_step:
            # Use tf.debugging so the check works under @tf.function / tracing
            tf.debugging.assert_equal(
                tf.shape(inputs)[1],
                1,
                message="inputs time dimension must be 1 in one-step streaming mode",
            )

            # Shift buffer and add new input
            # Remove oldest: [batch, 1:buffer_size, ...]
            # Use ellipsis (...) so this works for both 3D and 4D state tensors.
            memory = self.states[:, 1 : self.ring_buffer_size_in_time_dim, ...]
            # Concatenate new input at end
            memory = tf.concat([memory, inputs], axis=1)

            # Update state
            assign_op = self.states.assign(memory)
            with tf.control_dependencies([assign_op]):
                # Apply the wrapped cell
                return self.cell(memory)
        else:
            # Strided mode
            if self.ring_buffer_size_in_time_dim:
                memory = tf.concat([self.states, inputs], axis=1)
                # Keep only the last ring_buffer_size samples
                state_update = memory[:, -self.ring_buffer_size_in_time_dim :, ...]

                assign_op = self.states.assign(state_update)
                with tf.control_dependencies([assign_op]):
                    return self.cell(memory)
            else:
                return self.cell(inputs)

    def _streaming_external_state(self, inputs, state=None):
        """Streaming with external state management.

        Args:
            inputs: Input tensor
            state: Optional runtime state tensor. If None, uses self.input_state.
        """
        input_state = state if state is not None else self.input_state
        if self.use_one_step:
            tf.debugging.assert_equal(
                tf.shape(inputs)[1],
                1,
                message="inputs time dimension must be 1 in one-step streaming mode",
            )

            # Shift buffer and add new input
            memory = input_state[:, 1 : self.ring_buffer_size_in_time_dim, ...]
            memory = tf.concat([memory, inputs], axis=1)

            output = self.cell(memory)
            self.output_state = memory
            return output, memory
        else:
            # Strided mode
            memory = tf.concat([input_state, inputs], axis=1)
            state_update = memory[:, -self.ring_buffer_size_in_time_dim :, ...]
            output = self.cell(memory)
            self.output_state = state_update
            return output, state_update

    def _non_streaming(self, inputs, training=None):
        """Non-streaming mode (training or inference)."""
        # Apply padding if specified
        if self.pad_time_dim:
            pad_total = (
                self.ring_buffer_size_in_time_dim - 1
                if self.use_one_step
                else self.ring_buffer_size_in_time_dim
            )
            if pad_total > 0:
                # Build a dynamic padding spec that matches the actual input rank
                rank = tf.rank(inputs)
                # Padding: [[0,0]] for each dim, with [pad_total, 0] or [half, half]
                # on the time dimension (axis 1).
                # We use tf.pad with a statically-built list when rank is known, else dynamic.
                static_rank = inputs.shape.rank
                n_dims = (
                    static_rank if static_rank is not None else 4
                )  # conservative default
                pad = [[0, 0]] * n_dims
                if self.pad_time_dim == "causal":
                    pad[1] = [pad_total, 0]
                elif self.pad_time_dim == "same":
                    half = pad_total // 2
                    pad[1] = [half, pad_total - half]
                inputs = tf.pad(inputs, pad, "constant")

        return self.cell(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "cell": tf.keras.layers.serialize(self.cell),
                "inference_batch_size": self.inference_batch_size,
                "mode": self.mode,
                "pad_time_dim": self.pad_time_dim,
                "state_shape": self.state_shape,
                "ring_buffer_size_in_time_dim": self.ring_buffer_size_in_time_dim,
                "use_one_step": self.use_one_step,
                "state_name_tag": self.state_name_tag,
                "pad_freq_dim": self.pad_freq_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        cell_config = config.pop("cell", None)
        if cell_config is not None:
            config["cell"] = tf.keras.layers.deserialize(cell_config)
        return cls(**config)

    def get_input_state(self):
        """Get input state for external streaming."""
        if self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            if self.input_state is None:
                raise ValueError(
                    "input_state is None. Call build() before get_input_state()."
                )
            return [self.input_state]
        raise ValueError(f"Expected external streaming mode, not {self.mode}")

    def get_output_state(self):
        """Get output state for external streaming."""
        if self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            return [self.output_state]
        raise ValueError(f"Expected external streaming mode, not {self.mode}")


# =============================================================================
# STRIDED SLICE LAYERS (for streaming dimension matching)
# =============================================================================


class StridedDrop(tf.keras.layers.Layer):
    """Drop time slices for dimension matching in streaming.

    In streaming modes (STREAM_INTERNAL / STREAM_EXTERNAL), drops the specified
    number of time slices from the beginning.
    In non-streaming modes (TRAINING, NON_STREAM_INFERENCE), passes through unchanged.

    Used for matching dimensions between residual connections and conv outputs.
    """

    def __init__(
        self, time_slices_to_drop, mode=Modes.NON_STREAM_INFERENCE, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.time_slices_to_drop = time_slices_to_drop
        self.mode = mode
        self.state_shape = []

    def call(self, inputs):
        if self.mode in (
            Modes.STREAM_INTERNAL_STATE_INFERENCE,
            Modes.STREAM_EXTERNAL_STATE_INFERENCE,
        ):
            return inputs[:, self.time_slices_to_drop :, ...]
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "time_slices_to_drop": self.time_slices_to_drop,
                "mode": self.mode,
            }
        )
        return config

    def get_input_state(self):
        return []

    def get_output_state(self):
        return []


class StridedKeep(tf.keras.layers.Layer):
    """Keep only specified time slices for streaming.

    In streaming modes (STREAM_INTERNAL / STREAM_EXTERNAL), keeps only the last
    N time slices.
    In non-streaming modes (TRAINING, NON_STREAM_INFERENCE), passes through unchanged.

    Used in MixConv to split ring buffer into branches with different kernel sizes.
    """

    def __init__(
        self, time_slices_to_keep, mode=Modes.NON_STREAM_INFERENCE, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.time_slices_to_keep = max(time_slices_to_keep, 1)
        self.mode = mode
        self.state_shape = []

    def call(self, inputs):
        if self.mode in (
            Modes.STREAM_INTERNAL_STATE_INFERENCE,
            Modes.STREAM_EXTERNAL_STATE_INFERENCE,
        ):
            # In streaming mode, keep only the last N slices
            return inputs[:, -self.time_slices_to_keep :, ...]
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "time_slices_to_keep": self.time_slices_to_keep,
                "mode": self.mode,
            }
        )
        return config

    def get_input_state(self):
        return []

    def get_output_state(self):
        return []


# =============================================================================
# CHANNEL SPLIT LAYER (for MixConv)
# =============================================================================


class ChannelSplit(tf.keras.layers.Layer):
    """Split channels into groups for MixConv.

    Splits the channel dimension into groups for applying different
    kernel sizes to different groups in MixConv.

    Attributes:
        splits: List of channel counts per group
        axis: Axis to split along (default: -1 for channel)
    """

    def __init__(self, splits, axis=-1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.splits = splits
        self.axis = axis

    def call(self, inputs):
        return tf.split(inputs, self.splits, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shapes = []
        for split in self.splits:
            new_shape = list(input_shape)
            new_shape[self.axis] = split
            output_shapes.append(tuple(new_shape))
        return output_shapes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "splits": self.splits,
                "axis": self.axis,
            }
        )
        return config


# =============================================================================
# FREQUENCY PADDING HELPER
# =============================================================================


def frequency_pad(inputs, dilation, stride, kernel_size):
    """Pad input tensor in frequency domain.

    Args:
        inputs: Input tensor [N, Time, Frequency, ...]
        dilation: Dilation in frequency dimension
        stride: Stride in frequency dimension
        kernel_size: Kernel size in frequency dimension

    Returns:
        Padded tensor
    """
    if inputs.shape.rank < 3:
        raise ValueError(
            f"input_shape.rank must be at least 3, got {inputs.shape.rank}"
        )

    kernel_size = (kernel_size - 1) * dilation + 1
    total_pad = kernel_size - stride

    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    pad = [[0, 0]] * inputs.shape.rank
    pad[2] = [pad_left, pad_right]
    return tf.pad(inputs, pad, "constant")


def frequeny_pad(inputs, dilation, stride, kernel_size):
    """Deprecated alias for frequency_pad (typo in original name).

    .. deprecated::
        Use :func:`frequency_pad` instead.
    """
    import warnings

    warnings.warn(
        "frequeny_pad is a deprecated alias; use frequency_pad instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return frequency_pad(inputs, dilation, stride, kernel_size)


# =============================================================================
# STATE VARIABLE HELPERS (for TFLite conversion)
# =============================================================================


def get_streaming_state_names(base_name="stream"):
    """Get list of state variable names for streaming model.

    Args:
        base_name: Base name for state variables

    Returns:
        List of state variable names
    """
    names = [base_name]
    for i in range(1, 6):
        names.append(f"{base_name}_{i}")
    return names


def create_state_initializer(shape, dtype=tf.float32):
    """Create initializer for state variables.

    Args:
        shape: State variable shape
        dtype: Data type

    Returns:
        TensorFlow initializer function
    """
    return lambda *args, **kwargs: tf.zeros(shape, dtype=dtype)


# =============================================================================
# STREAMING MODEL WRAPPER
# =============================================================================


class StreamingMixedNet:
    """High-level wrapper for streaming MixedNet inference.

    Manages state variables and provides convenient interface for
    streaming inference with the MixedNet model.

    Attributes:
        model: The underlying MixedNet model
        stride: Stride used during inference
        state: Dictionary of state tensors
    """

    def __init__(
        self,
        model,
        stride=3,
        inference_batch_size=1,
    ):
        """Initialize streaming model.

        Args:
            model: Trained MixedNet model
            stride: Stride used during training
            inference_batch_size: Batch size for inference
        """
        self.model = model
        self.stride = stride
        self.inference_batch_size = inference_batch_size
        self.state = {}

        # Initialize state variables based on model
        self._init_state()

    def _init_state(self):
        """Initialize state variables."""
        state_names = get_streaming_state_names()
        for name in state_names:
            self.state[name] = None

    def reset(self):
        """Reset all state variables to zero tensors."""
        state_names = get_streaming_state_names()
        for name in state_names:
            current = self.state.get(name)
            if current is not None:
                self.state[name] = tf.zeros_like(current)
            else:
                self.state[name] = None

    def predict(self, features):
        """Run inference on features.

        Args:
            features: Input features [batch, stride, 40]

        Returns:
            Detection probability [batch, 1]
        """
        # Run model inference (state is managed internally by the streaming model)
        output = self.model(features, training=False)
        return output

    def predict_clip(self, audio_samples, sample_rate=16000, step_ms=30):
        """Run inference on audio clip.

        Args:
            audio_samples: Audio samples as numpy array
            sample_rate: Sample rate (default: 16000)
            step_ms: Step size in milliseconds for streaming inference (default: 30)

        Returns:
            List of probabilities for each inference step
        """
        import numpy as np
        from ..data.features import FeatureConfig, MicroFrontend

        # Normalize audio to [-1, 1] if needed
        if audio_samples.dtype == np.int16:
            audio_samples = audio_samples.astype(np.float32) / 32767.0
        elif audio_samples.size == 0:
            return []
        elif audio_samples.max() > 1.0 or audio_samples.min() < -1.0:
            # Normalize to [-1, 1] — guard against all-zero arrays (max_val == 0)
            max_val = max(abs(audio_samples.max()), abs(audio_samples.min()))
            if max_val > 0.0:
                audio_samples = audio_samples.astype(np.float32) / max_val
            else:
                audio_samples = audio_samples.astype(np.float32)
        else:
            audio_samples = audio_samples.astype(np.float32)

        # Resample if needed (simple resampling)
        if sample_rate != 16000:
            from scipy import signal

            num_samples = int(len(audio_samples) * 16000 / sample_rate)
            audio_samples = signal.resample(audio_samples, num_samples)

        # Extract mel spectrogram
        config = FeatureConfig(sample_rate=16000, window_step_ms=10)
        frontend = MicroFrontend(config)
        spectrogram = frontend.compute_mel_spectrogram(audio_samples)

        # Convert step_ms to stride (number of frames)
        # step_ms=30ms -> stride=3 frames (at 10ms per frame)
        step_samples = int(step_ms / 10)
        if step_samples < 1:
            step_samples = 1

        # Get model stride from config (for window size)
        model_stride = getattr(self, "stride", 3)

        # Run streaming inference with configurable step size
        probabilities = []
        upper = max(1, spectrogram.shape[0] - model_stride + 1)
        for i in range(0, upper, step_samples):
            # Extract window
            window = spectrogram[i : i + model_stride, :]

            # Pad if needed
            if window.shape[0] < model_stride:
                pad_width = ((0, model_stride - window.shape[0]), (0, 0))
                window = np.pad(window, pad_width, mode="constant")

            # Add batch dimension: [1, stride, mel_bins]
            features = np.expand_dims(window, axis=0).astype(np.float32)

            # Run inference
            prob = self.predict(features)
            probabilities.append(float(prob[0, 0]))

        return probabilities
