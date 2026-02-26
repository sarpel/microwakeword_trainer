# src/model/

MixedNet architecture for wake word detection with MixConv blocks and streaming inference support.

## Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| architecture.py | Model definition and factory functions | MixedNet, MixConvBlock, ResidualBlock |
| streaming.py | Streaming layers and state management | Stream, RingBuffer, Modes, StridedDrop, StridedKeep, ChannelSplit |

## Architecture Patterns

**MixedNet**: tf.keras.Model subclass with configurable MixConv blocks. Input shape [batch, time, 40] mel bins. Initial Conv2D with stride 3 for 30ms inference, followed by residual blocks with mixed depthwise convolutions.

**MixConv**: MixConvBlock splits channels, applies different kernel sizes per group (e.g., [5], [9, 11]), concatenates results. Ring buffer length = max(kernel_sizes) - 1.

**Streaming**: Stream wrapper manages ring buffers. Modes enum: TRAINING, NON_STREAM_INFERENCE, STREAM_INTERNAL_STATE_INFERENCE, STREAM_EXTERNAL_STATE_INFERENCE. Internal state uses tf.Variable, external uses Input layers.

**Factory Functions**: build_model() parses string params (pointwise_filters, mixconv_kernel_sizes). Preset configs: hey_jarvis (30/60 filters), okay_nabu (32/64 filters).

## Anti-Patterns

- Do not use same kernel sizes in all MixConv blocks - defeats multi-scale purpose
- Do not set stride=1 - breaks 30ms real-time target (use stride=3)
- Do not forget ring_buffer_size_in_time_dim calculation - causes dimension mismatches
- Do not mix streaming modes in same model - pick internal or external, not both
- Do not skip causal padding for small time dimensions - crashes on short inputs

## Notes

**Streaming State Variables**: TFLite export creates 6 state vars (stream, stream_1-5). Total memory ~2.8-3.5KB. State shapes depend on kernel sizes and stride.

**ESPHome Requirements**: Input dtype int8 [1, 3, 40], output uint8 [1, 1]. Must have 2 subgraphs (main + init). Quantization required. Tensor arena size 20-30KB typical.

**Tensor Shapes**: Training [batch, 98, 40], streaming [batch, 3, 40] per step. Channel dim added internally: [batch, time, 1, features].

**References**: microWakeWord (OHF-Voice), Google kws_streaming.
