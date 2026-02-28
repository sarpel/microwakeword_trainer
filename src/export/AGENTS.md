# src/export/

**TFLite Export Pipeline** | Converts trained models to ESPHome-compatible TFLite format.

## Overview

Export module handles streaming model conversion, INT8 quantization, model analysis/verification, and ESPHome V2 manifest generation. Produces deployable artifacts for ESP32 micro_wake_word component.

## Files

| File | Lines | Purpose | Key Functions/Classes |
|------|-------|---------|----------------------|
| `tflite.py` | 817 | Streaming conversion, quantization, export | `convert_model_saved()`, `convert_saved_model_to_tflite()`, `export_to_tflite()`, `main()` |
| `model_analyzer.py` | 568 | Architecture analysis and validation | `analyze_model_architecture()`, `validate_model_quality()`, `compare_models()`, `generate_model_report()` |
| `manifest.py` | 327 | ESPHome V2 manifest generation | `generate_manifest()`, `save_manifest()`, `calculate_tensor_arena_size()` |
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
| `export_to_tflite()` | Full export pipeline (convert + quantize + manifest) |
| `verify_esphome_compatibility()` | Quick compatibility check |
| `calculate_tensor_arena_size()` | Estimate arena memory |
| `convert_to_tflite()` | Legacy conversion wrapper |
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
- DEFAULT_TENSOR_ARENA_SIZE = 22860 bytes in manifest.py (add 10% margin to measured value)
- Verification also available via `scripts/verify_esphome.py` (standalone, 406 lines)
