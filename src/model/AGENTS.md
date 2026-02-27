# src/model/

MixedNet architecture for wake word detection with MixConv blocks and streaming inference support.

## Files

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `architecture.py` | 757 | Model definition and factory | MixedNet, MixConvBlock, ResidualBlock, build_model() |
| `streaming.py` | 831 | Streaming layers and state management | Stream, RingBuffer, Modes, StridedDrop, StridedKeep, ChannelSplit, StreamingMixedNet |
| `__init__.py` | 9 | Package init | |

## Architecture Patterns

**MixedNet**: tf.keras.Model subclass with configurable MixConv blocks. Input shape [batch, time, 40] mel bins. Initial Conv2D with stride 3 for 30ms inference, followed by residual blocks with mixed depthwise convolutions.

**MixConv**: MixConvBlock splits channels, applies different kernel sizes per group (e.g., [5], [9, 11]), concatenates results. Ring buffer length = max(kernel_sizes) - 1.

**Streaming**: Stream wrapper manages ring buffers. Modes enum: TRAINING, NON_STREAM_INFERENCE, STREAM_INTERNAL_STATE_INFERENCE, STREAM_EXTERNAL_STATE_INFERENCE. Internal state uses tf.Variable, external uses Input layers.

**Factory Functions**: `build_model()` parses string params (pointwise_filters, mixconv_kernel_sizes). Preset creators: `create_hey_jarvis_model()` (30/60 filters), `create_okay_nabu_model()` (32/64 filters).

## architecture.py Symbols

| Symbol | Type | Purpose |
|--------|------|---------|
| `MixConvBlock` | Class | Parallel depthwise convs with different kernel sizes per group |
| `ResidualBlock` | Class | Wraps MixConvBlock with optional skip connection |
| `MixedNet` | Class | Full model: Conv2D → N×MixConvBlock → Dense(1, sigmoid) |
| `build_model()` | Function | Factory: parses string config → builds MixedNet |
| `create_hey_jarvis_model()` | Function | Preset: 30 filters, [5],[9],[13],[21] kernels |
| `create_okay_nabu_model()` | Function | Preset: 32 filters, [5],[9],[13],[21] kernels |
| `parse_model_param()` | Function | Parse comma-separated string to list |
| `spectrogram_slices_dropped()` | Function | Calculate frames lost to striding |
| `_split_channels()` | Function | Split tensor along channel dimension for MixConv |

## streaming.py Symbols

| Symbol | Type | Purpose |
|--------|------|---------|
| `Modes` | Enum | TRAINING, NON_STREAM_INFERENCE, STREAM_INTERNAL/EXTERNAL_STATE |
| `RingBuffer` | Class | Circular buffer for temporal context (initialize, update) |
| `Stream` | Class | Wraps any Keras layer with ring buffer state management |
| `StridedDrop` | Class | Drop frames for stride alignment |
| `StridedKeep` | Class | Keep frames for stride alignment (okay_nabu variant) |
| `ChannelSplit` | Class | Split tensor along channels for MixConv groups |
| `StreamingMixedNet` | Class | Full streaming inference wrapper (predict, predict_clip, reset) |
| `frequency_pad()` | Function | Pad frequency dimension for causal convolutions |
| `get_streaming_state_names()` | Function | List state variable names for export |
| `create_state_initializer()` | Function | Create init subgraph for state zeroing |

## Anti-Patterns

- Do not use same kernel sizes in all MixConv blocks - defeats multi-scale purpose
- Do not set stride=1 - breaks 30ms real-time target (use stride=3)
- Do not forget ring_buffer_size_in_time_dim calculation - causes dimension mismatches
- Do not mix streaming modes in same model - pick internal or external, not both
- Do not skip causal padding for small time dimensions - crashes on short inputs
- Do not use `padding="same"` on time axis - produces different activations in streaming vs non-streaming

## Notes

**Streaming State Variables**: TFLite export creates 6 state vars (stream, stream_1-5). Total memory ~2.8-3.5KB. State shapes depend on kernel sizes and stride.

**ESPHome Requirements**: Input dtype int8 [1, 3, 40], output uint8 [1, 1]. Must have 2 subgraphs (main + init). Quantization required. Tensor arena size 20-30KB typical.

**Tensor Shapes**: Training shape is `(clip_duration_ms / window_step_ms, 40)` — e.g., `(100, 40)` for 1000ms clip, `(150, 40)` for 1500ms. Streaming: `[batch, 3, 40]` per step. Channel dim added internally: `[batch, time, 1, features]`.

**Ring Buffer Law**: `buffer_frames = kernel_size - stride` — inviolable identity from ARCHITECTURAL_CONSTITUTION.md.

**References**: microWakeWord (OHF-Voice), Google kws_streaming.
