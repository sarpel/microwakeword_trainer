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

**Factory Functions**: `build_model()` parses string params (pointwise_filters, mixconv_kernel_sizes). Preset creator: `create_okay_nabu_model()` (32/64 filters, [[5],[7,11],[9,15],[23]] kernels).

## architecture.py Symbols

| Symbol | Type | Purpose |
|--------|------|---------|
| `MixConvBlock` | Class | Parallel depthwise convs with different kernel sizes per group |
| `ResidualBlock` | Class | Wraps MixConvBlock with optional skip connection |
| `MixedNet` | Class | Full model: Conv2D → N×MixConvBlock → Dense(1, sigmoid) |
| `build_model()` | Function | Factory: parses string config → builds MixedNet |
| `create_okay_nabu_model()` | Function | Preset: 32 filters, [[5],[7,11],[9,15],[23]] kernels |
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
- Do not assume state variable naming without flatbuffer verification — the official `okay_nabu` flatbuffer uses `stream`, `stream_1`, …, `stream_5`; if an implementation deviates, document it explicitly as an implementation detail rather than reference truth
- Do not assume 14 ESPHome ops — there are 20 registered op resolvers (13 unique ops used by okay_nabu, 7 registered but unused: MUL, ADD, MEAN, AVERAGE_POOL_2D, MAX_POOL_2D, PAD, PACK)

### Shared Layer Factory
`build_core_layers()` creates shared layer objects for both `MixedNet` (training) and `StreamingExportModel` (export). Takes parameters like `first_conv_filters`, `pointwise_filters`, `mixconv_kernel_sizes`, `residual_connections`, etc. Returns a dict with keys: `initial_conv_cell`, `initial_relu`, `blocks`, `dense`. Stream wrappers and ring buffer states are added by each model class independently.

## Notes

**Streaming State Variables**: The verified official `okay_nabu` flatbuffer contains 6 int8 state tensors named `stream`, `stream_1`, `stream_2`, `stream_3`, `stream_4`, `stream_5`. Their shapes are [1,2,1,40], [1,4,1,32], [1,10,1,64], [1,14,1,64], [1,22,1,64], [1,5,1,64]. Total state memory is ~3.5KB.

**ESPHome Requirements**: Input dtype int8 [1, 3, 40] (scale=0.101961, zero_point=-128), output uint8 [1, 1] (scale=0.00390625, zero_point=0). Must have 2 subgraphs: Subgraph 0 (main, 95 tensors) + Subgraph 1 (initialization, 12 tensors). 13 unique op types used by okay_nabu. 20 op resolvers registered in ESPHome runtime.

**Tensor Arena**: Recommended arena = 135,873 bytes (~136KB) includes intermediate activations and overhead. Peak memory per subgraph: Subgraph 0 = 41,771 bytes, Subgraph 1 = 3,520 bytes.

**Tensor Shapes**: Training shape is `(clip_duration_ms / window_step_ms, 40)` — e.g., `(100, 40)` for 1000ms clip, `(150, 40)` for 1500ms. Streaming: `[batch, 3, 40]` per step. Channel dim added internally: `[batch, time, 1, features]`.

**Temporal Frames**: Training shape is `(clip_duration_ms / window_step_ms, 40)`. Streaming: `[batch, 3, 40]` per step. The export pipeline infers `temporal_frames = dense_input_features // 64` from checkpoint Dense kernel shape. Dense layer input = `temporal_frames × 64`.

⚠️ **Checkpoint Compatibility**: Checkpoints created before 2026-03-11 used Flatten() before Dense layer, causing shape mismatches during export. Current export expects GlobalAveragePooling2D or explicit reshape. Export script will raise `ValueError` with details if incompatible checkpoint is detected. To fix: retrain model with updated architecture or manually convert checkpoint (see MIGRATION.md).

**Ring Buffer Law**: `buffer_frames = kernel_size - stride` applies to the convolution-derived states `stream` through `stream_4`. `stream_5` is different: it is the pre-flatten temporal buffer and must be derived from the actual graph structure.

**References**: microWakeWord (OHF-Voice), Google kws_streaming.


## Related Documentation

- [Architecture Guide](../../docs/ARCHITECTURE.md) - Complete architecture documentation
- [Export Guide](../../docs/EXPORT.md) - Streaming conversion and TFLite export
- [Configuration Reference](../../docs/CONFIGURATION.md) - ModelConfig options