# src/export/

TFLite Export Pipeline - Converts trained models to ESPHome-compatible TFLite format.

## Files

| File | Lines | Purpose | Key Functions/Classes |
|------|-------|---------|----------------------|
| `tflite.py` | 780 | Streaming conversion, quantization | `export_streaming_tflite()`, `convert_model_saved()` |
| `model_analyzer.py` | 600 | Architecture analysis | `analyze_model_architecture()` |
| `manifest.py` | 330 | ESPHome V2 manifest | `generate_manifest()` |
| `verification.py` | 218 | Export verification | Verification functions |

## Export Flow

```
Checkpoint → Streaming SavedModel → TFLite (INT8) → manifest.json
```

## Critical Settings

```python
converter.inference_input_type = tf.int8      # REQUIRED
converter.inference_output_type = tf.uint8    # MUST be uint8
converter._experimental_variable_quantization = True  # REQUIRED
```

Input: `[1, 3, 40]` (int8) | Output: `[1, 1]` (uint8)

## Temporal Frames Inference

Export pipeline infers `temporal_frames` from Dense kernel shape:
```python
temporal_frames = dense_input_features // 64  # 64 = last pointwise filter count
```

## Anti-Patterns

- **Don't use int8 output** - ESPHome requires uint8
- **Don't skip `_experimental_variable_quantization`** - Required for streaming state
- **Don't use `model.export()`** - Use `tf.keras.export.ExportArchive`
- **Don't forget representative dataset** - 500+ samples with boundary anchors

## Ground Truth (from ARCHITECTURAL_CONSTITUTION.md)

- **94 tensors** in Subgraph 0, **12 tensors** in Subgraph 1
- **20 op resolvers** registered in ESPHome
- **6 state variables**: stream [1,2,1,40], stream_1 [1,4,1,32], stream_2 [1,10,1,64], stream_3 [1,14,1,64], stream_4 [1,22,1,64], stream_5 [1,5,1,64]

## Related Documentation

- [Export Guide](../../docs/EXPORT.md)
- ARCHITECTURAL_CONSTITUTION.md
