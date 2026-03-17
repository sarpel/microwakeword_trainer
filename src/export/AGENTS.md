# src/export/

TFLite Export Pipeline - Converts trained models to ESPHome-compatible TFLite format.

## Files

| File | Lines | Purpose | Key Functions/Classes |
|------|-------|---------|----------------------|
| `tflite.py` | 1137 | Streaming conversion, quantization | `export_streaming_tflite()`, `convert_model_saved()` |
| `model_analyzer.py` | 600 | Architecture analysis | `analyze_model_architecture()` |
| `manifest.py` | 330 | ESPHome V2 manifest | `generate_manifest()` |
| `verification.py` | 315 | Export verification | `compute_expected_state_shapes(temporal_frames, first_conv_kernel, stride, mel_bins, first_conv_filters, mixconv_kernel_sizes, pointwise_filters)` |

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
- **Don't hardcode state shapes** — use `compute_expected_state_shapes()` for config-aware validation. Models with different `clip_duration_ms` produce different `temporal_frames`, changing stream_5 shape.

## Verification Notes

- `scripts/verify_esphome.py` supports:
  - default human-readable output
  - `--verbose` for check/detail dump
  - `--json` for machine-readable CI output
- JSON payloads are sanitized for NumPy-derived values (e.g., `np.int32`, `np.float32`, `np.ndarray`, `np.dtype`) before serialization.
- Canonical compatibility invariants remain:
  - input `[1,3,40] int8`
  - output `[1,1] uint8`
  - output quantization typically `scale=0.00390625 (1/256), zero_point=0` in official models
  - required streaming state op family (`CALL_ONCE`, `VAR_HANDLE`, `READ_VARIABLE`, `ASSIGN_VARIABLE`)
- `DELEGATE` visibility is runtime/delegate-path dependent and not a required static-graph compatibility op.

## Ground Truth (from ARCHITECTURAL_CONSTITUTION.md)

- **94 tensors** in Subgraph 0, **12 tensors** in Subgraph 1
- **20 op resolvers** registered in ESPHome
- **6 state variables**: stream [1,2,1,40], stream_1 [1,4,1,32], stream_2 [1,10,1,64], stream_3 [1,14,1,64], stream_4 [1,22,1,64], stream_5 [1,5,1,64]
  - **Note**: stream_5 shape depends on `temporal_frames` (derived from `clip_duration_ms`). The shape is `(1, temporal_frames - 1, 1, pointwise_filters[3])`, NOT always `(1, 5, 1, 64)`.
- **Resolved**: All verification scripts now use config-aware state shape computation, resolving previous hardcoding issues.

## Related Documentation

- [Export Guide](../../docs/EXPORT.md)
- ARCHITECTURAL_CONSTITUTION.md
