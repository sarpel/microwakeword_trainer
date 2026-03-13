# src/model/

MixedNet architecture for wake word detection with MixConv blocks and streaming inference support.

## Files

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `architecture.py` | 757 | Model definition and factory | `MixedNet`, `MixConvBlock`, `build_model()` |
| `streaming.py` | 831 | Streaming layers and state management | `Stream`, `RingBuffer`, `Modes` |

## Architecture Patterns

**MixedNet**: tf.keras.Model with configurable MixConv blocks. Input shape [batch, time, 40] mel bins.

**MixConvBlock**: Splits channels, applies different kernel sizes per group, concatenates results.

**Streaming**: `Stream` wrapper manages ring buffers. Modes: TRAINING, NON_STREAM_INFERENCE, STREAM_INTERNAL_STATE_INFERENCE.

## Key Symbols

| Symbol | Type | Purpose |
|--------|------|---------|
| `MixedNet` | Class | Full model: Conv2D → MixConvBlocks → Dense(1, sigmoid) |
| `MixConvBlock` | Class | Parallel depthwise convs with different kernel sizes |
| `build_model()` | Function | Factory: parses config → builds MixedNet |
| `build_core_layers()` | Function | Shared layer factory for training and export |
| `Stream` | Class | Wraps layers with ring buffer state management |
| `Modes` | Enum | Training vs inference modes |

## Anti-Patterns

- **Don't set stride=1** - Use stride=3 for 30ms inference period
- **Don't use `padding="same"` on time axis** - Causes streaming/non-streaming mismatch
- **Don't use `model.trainable_weights` for serialization** - Excludes BatchNorm moving stats
- **Don't assume 14 ESPHome ops** - 20 op resolvers registered (see ARCHITECTURAL_CONSTITUTION.md)

## Notes

- **State variables**: 6 tensors named `stream` through `stream_5` (see ARCHITECTURAL_CONSTITUTION.md for shapes)
- **ESPHome I/O**: Input int8 [1,3,40], Output uint8 [1,1]
- **Ring buffer law**: `buffer_frames = kernel_size - stride` for convolution states
- **Residual connections**: Repository default `[0,1,1,1]` produces 58 ops (vs 55 in reference)

## Related Documentation

- [Architecture Guide](../../docs/ARCHITECTURE.md)
- ARCHITECTURAL_CONSTITUTION.md - Immutable constants
