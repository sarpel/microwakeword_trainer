# src/export/

**TFLite Export Pipeline** | Converts trained models to ESPHome-compatible TFLite format.

## Overview

Export module handles streaming model conversion, INT8 quantization, and ESPHome V2 manifest generation. Produces deployable artifacts for ESP32 micro_wake_word component.

## Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| tflite.py | Streaming conversion, quantization | `convert_model_saved()`, `convert_saved_model_to_tflite()`, `_convert_to_streaming_savedmodel()` |
| manifest.py | ESPHome V2 manifest generation | `generate_manifest()`, `save_manifest()`, `calculate_tensor_arena_size()` |

## Export Flow

```
Checkpoint → Streaming SavedModel → TFLite (INT8) → manifest.json
```

1. Load trained checkpoint
2. Convert to streaming SavedModel with internal state variables
3. Quantize to INT8 with representative dataset
4. Generate V2 manifest with tensor arena size
5. Output: `.tflite` + `manifest.json`

## Critical Settings

These settings are **required** for ESPHome compatibility:

```python
converter.inference_input_type = tf.int8      # REQUIRED
converter.inference_output_type = tf.uint8    # MUST be uint8 (not int8!)
converter._experimental_variable_quantization = True  # REQUIRED for streaming state
```

Input shape: `[1, 3, 40]` (int8) or `[1, 1, 40]` (int8)
Output shape: `[1, 1]` (uint8 probability)

Valid streaming input shapes are (1, 3, 40) for stride=3 or (1, 1, 40) for stride=1.

Streaming model uses 2 subgraphs with 6 state variables for ring buffers.

## Anti-Patterns

- **Don't use int8 output** - ESPHome requires uint8 output tensors
- **Don't skip `_experimental_variable_quantization`** - Required for streaming state management
- **Don't use non-streaming models** - Must convert to streaming with `convert_model_saved()` first
- **Don't forget representative dataset** - Required for INT8 calibration; use real training data if available
- **Don't use wrong input shape** - Must be [1, stride, mel_bins] for streaming inference
- **Don't use `model.export()`** - Fails with ring buffer states; use `tf.keras.export.ExportArchive` instead
