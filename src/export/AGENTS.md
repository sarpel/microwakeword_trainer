# src/export/

**TFLite Export Pipeline** | Converts trained models to ESPHome-compatible TFLite format.

## Overview

Export module handles streaming model conversion, INT8 quantization, model analysis/verification, and ESPHome V2 manifest generation. Produces deployable artifacts for ESP32 micro_wake_word component.

## Files

| File | Lines | Purpose | Key Functions/Classes |
|------|-------|---------|----------------------|
| `tflite.py` | 780 | Streaming conversion, quantization, export | `convert_model_saved()`, `convert_saved_model_to_tflite()`, `export_streaming_tflite()`, `main()` |
| `model_analyzer.py` | 600 | Architecture analysis and validation | `analyze_model_architecture()`, `validate_model_quality()`, `compare_models()`, `generate_model_report()` |
| `manifest.py` | 330 | ESPHome V2 manifest generation | `generate_manifest()`, `save_manifest()`, `calculate_tensor_arena_size()` |
| `verification.py` | 218 | Export verification tools | Export verification functions |
| `__init__.py` | 8 | Package init | |

## Export Flow

```
Checkpoint → Streaming SavedModel → TFLite (INT8) → manifest.json
```

1. Load trained checkpoint
2. Convert to streaming SavedModel with internal state variables (`convert_model_saved()`)
3. Quantize to INT8 with representative dataset (`convert_saved_model_to_tflite()`)
4. Generate V2 manifest with tensor arena size (`generate_manifest()`)
5. Output: `.tflite` + `manifest.json`

## tflite.py Functions

| Function | Purpose |
|----------|---------|
| `convert_model_saved()` | Non-streaming → streaming SavedModel conversion |
| `_convert_to_streaming_savedmodel()` | Internal streaming conversion logic |
| `_create_streaming_state()` | Create state variables for ring buffers |
| `_streaming_concat()` | Concatenate old buffer frames with new input |
| `_apply_mixconv_block()` | Apply MixConv operations in streaming mode |
| `convert_saved_model_to_tflite()` | SavedModel → quantized TFLite conversion |
| `create_default_representative_dataset()` | Default calibration dataset (random) |
| `create_representative_dataset_from_data()` | Calibration from real training data |
| `export_streaming_tflite()` | Full export pipeline (convert + quantize + manifest) |
| `verify_esphome_compatibility()` | Quick compatibility check |
| `calculate_tensor_arena_size()` | Estimate arena memory |
| `convert_to_tflite()` | Direct conversion helper |
| `optimize_for_edge()` | Edge optimization passes |
| `main()` | CLI entry point for mww-export |

## model_analyzer.py Functions

| Function | Purpose |
|----------|---------|
| `analyze_model_architecture()` | Parse TFLite model structure (subgraphs, ops, tensors) |
| `validate_model_quality()` | Check model meets quality thresholds |
| `compare_models()` | Compare two models (size, ops, params) |
| `estimate_performance()` | Estimate inference time and memory |
| `check_gpu_compatibility()` | Verify GPU requirements |
| `generate_model_report()` | Full HTML/text report generation |

## manifest.py Functions

| Function | Purpose |
|----------|---------|
| `generate_manifest()` | Create ESPHome V2 manifest JSON |
| `save_manifest()` | Write manifest to file |
| `calculate_tensor_arena_size()` | Compute required arena size with margin |

## verification.py

Export verification tools for validating exported models against ESPHome requirements.

## Critical Settings

These settings are **required** for ESPHome compatibility:

```python
converter.inference_input_type = tf.int8      # REQUIRED
converter.inference_output_type = tf.uint8    # MUST be uint8 (not int8!)
converter._experimental_variable_quantization = True  # REQUIRED for streaming state
```

Input shape: `[1, 3, 40]` (int8) — stride=3, 40 mel bins
Output shape: `[1, 1]` (uint8 probability)

Valid streaming input shapes: (1, 3, 40) for stride=3 or (1, 1, 40) for stride=1.

Streaming model uses 2 subgraphs with 6 state variables for ring buffers.

## Temporal Frames Inference

The export pipeline dynamically infers `temporal_frames` from the checkpoint's Dense layer kernel shape:

```python
# tflite.py lines 669-690
dense_input_features, dense_output_features = dense_kernel_shape
temporal_frames = dense_input_features // pointwise_filters  # pointwise_filters = 64
# e.g., Dense kernel (2048, 1) → temporal_frames = 2048 // 64 = 32
```

**Key formula**: `Dense layer input size = temporal_frames × 64` (64 = last pointwise filter count)

This means:
- Old checkpoints (pre-Flatten fix, 2026-03-11) with Dense kernel `(64, 1)` → `temporal_frames = 1` (WRONG — these are incompatible)
- Current checkpoints with Dense kernel `(2048, 1)` → `temporal_frames = 32` (correct for okay_nabu)
- If you see a shape mismatch error during export, you likely need to retrain with the current code

## Anti-Patterns

- **Don't use int8 output** - ESPHome requires uint8 output tensors
- **Don't skip `_experimental_variable_quantization`** - Required for streaming state management
- **Don't use non-streaming models** - Must convert to streaming with `convert_model_saved()` first
- **Don't forget representative dataset** - Required for INT8 calibration; use real training data if available
- **Don't use wrong input shape** - Must be [1, stride, mel_bins] for streaming inference
- **Don't use `model.export()`** - Fails with ring buffer states; use `tf.keras.export.ExportArchive` instead
- **Don't skip model analysis** - Use `model_analyzer.py` to validate before deployment

## Notes

- Representative dataset requires minimum **500 training samples** with forced min/max boundary anchors (0.0 and 26.0)
- `model_analyzer.py` can compare models, estimate performance, and generate full reports
- DEFAULT_TENSOR_ARENA_SIZE = 22860 bytes in manifest.py (add 10% margin to measured value). Official okay_nabu recommended arena = 135,873 bytes (~136KB). Subgraph 0 uses 41,771 bytes, Subgraph 1 uses 3,520 bytes.
- Verification also available via `scripts/verify_esphome.py` (standalone, 168 lines)


## Ground Truth (from official okay_nabu analysis)

- **95 tensors** in Subgraph 0 (indices 0-94), **12 tensors** in Subgraph 1 (initialization)
- **13 unique op types**: STRIDED_SLICE(10), CONCATENATION(8), VAR_HANDLE(6), READ_VARIABLE(6), ASSIGN_VARIABLE(6), DEPTHWISE_CONV_2D(6), CONV_2D(5), RESHAPE(2), SPLIT_V(2), CALL_ONCE(1), FULLY_CONNECTED(1), LOGISTIC(1), QUANTIZE(1)
- **20 op resolvers** registered in ESPHome (MUL/ADD registered but unused in okay_nabu)
- **6 state variables**: stream_0 [1,2,1,40], stream_1 [1,4,1,32], stream_2 [1,10,1,64], stream_3 [1,14,1,64], stream_4 [1,22,1,64], stream_5 [1,5,1,64]
- **Input**: int8 [1,3,40], scale=0.101961, zero_point=-128
- **Output**: uint8 [1,1], scale=0.00390625, zero_point=0
- **Memory**: Subgraph 0 = 41,771 bytes, Subgraph 1 = 3,520 bytes, Recommended arena = 135,873 bytes
## Related Documentation

- [Export Guide](../../docs/EXPORT.md) - Complete export documentation
- [Architecture Guide](../../docs/ARCHITECTURE.md) - Model architecture details
- [Configuration Reference](../../docs/CONFIGURATION.md) - ExportConfig options
