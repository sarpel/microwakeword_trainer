# Export Documentation

## Overview

The export pipeline converts trained Keras checkpoints to ESPHome-compatible streaming TFLite models with INT8 quantization. This process transforms a trained MixedNet model into a streaming-capable TFLite model that can run efficiently on ESP32 devices via ESPHome's micro_wake_word component.

## Export Workflow

```
Keras Checkpoint → Build StreamingExportModel → Fold BatchNorm → Convert to SavedModel → TFLite Conversion + INT8 Quantization → Generate Manifest → Verification
```

### Detailed Steps

1. **Load Keras Checkpoint**: Load the trained model weights from the checkpoint file
2. **Build StreamingExportModel**: Convert the training model to a streaming variant with ring buffer state variables
3. **Fold BatchNorm**: Eliminate BatchNorm-related variable reads by folding batch normalization layers into preceding convolutions
4. **Convert to SavedModel**: Export using `tf.keras.export.ExportArchive` (NOT `model.export()`), then enable `_experimental_variable_quantization` during TFLite conversion so streaming-state payload tensors are quantized correctly.
5. **TFLite Conversion**: Convert to TFLite with INT8 quantization using representative dataset
6. **Generate Manifest**: Create ESPHome-compatible manifest file with model metadata
7. **Verification**: Validate the exported model meets ESPHome requirements

## Streaming Model Conversion

### StreamingExportModel Architecture

The streaming model uses a ring buffer approach to maintain state across inference calls, enabling real-time audio processing without storing the entire audio history.

**Key Features:**
- 6 state variables for ring buffers (~3.5KB total)
- Okay_nabu variant: 32/64 filters, [[5],[7,11],[9,15],[23]] kernels
- BatchNorm folding eliminates BatchNorm-related variable reads for better performance

### State Variables

The official `okay_nabu` reference model maintains 6 state tensors across streaming stages:

1. `stream`: [1,2,1,40] - Initial convolution buffer
2. `stream_1`: [1,4,1,32] - Block 0 buffer
3. `stream_2`: [1,10,1,64] - Block 1 buffer
4. `stream_3`: [1,14,1,64] - Block 2 buffer
5. `stream_4`: [1,22,1,64] - Block 3 buffer
6. `stream_5`: [1,5,1,64] - Pre-flatten temporal buffer

These buffers do **not** all follow the same formula:

- `stream` is the input-side buffer before the first strided convolution, so it uses `kernel - global_stride`
- `stream_1` through `stream_4` are downstream block buffers before depthwise convolutions, so they use `effective_kernel - 1`
- `stream_5` is a pre-flatten temporal buffer derived from the graph (`[1, 6, 1, 64] → [1, 5, 1, 64]` in okay_nabu)

### Conversion Process

```python
# Build streaming model from trained checkpoint
streaming_model = build_streaming_export_model(
    checkpoint_path="checkpoints/best_weights.weights.h5",
    config=model_config
)

# Fold batch normalization layers
folded_model = fold_batch_norm(streaming_model)
```

### Temporal Frames Inference

The export pipeline dynamically infers `temporal_frames` from the checkpoint Dense layer:

```python
# From tflite.py — automatic inference
dense_input_features, _ = dense_kernel_shape
temporal_frames = dense_input_features // 64  # 64 = last pointwise filter count
# Dense kernel (2048, 1) → temporal_frames = 32 (correct for okay_nabu)
# Dense kernel (64, 1) → temporal_frames = 1 (OLD checkpoint, INCOMPATIBLE)
```

**Important**: Checkpoints trained before the Flatten architecture fix (2026-03-11) have Dense kernel shape `(64, 1)` and are incompatible with current export. Must retrain with current code.

## INT8 Quantization

### Critical TFLite Settings

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = {tf.lite.Optimize.DEFAULT}
converter._experimental_variable_quantization = True  # REQUIRED for READ_VARIABLE / ASSIGN_VARIABLE payload tensors
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.uint8  # MUST be uint8 for ESPHome compatibility
```

**Important Notes:**
- `inference_output_type` must be `tf.uint8`, not `tf.int8`
- `_experimental_variable_quantization = True` is required so state payload tensors on `READ_VARIABLE` / `ASSIGN_VARIABLE` paths are emitted as int8-compatible tensors
- `VAR_HANDLE` tensors themselves are resource handles; the quantized part is the data payload flowing through variable read/write ops

### Quantization Range

The representative dataset determines the quantization range. Key boundaries:
- **Input range**: 0.0 to 26.0 (mel spectrogram values)
- **Output range**: 0-255 (probability scaled to uint8)

## Representative Dataset Creation

### Requirements

- **Minimum samples**: 500 (recommended: 2000+)
- **Input shape**: (1, 3, 40) - [batch, time, mel_bins]
- **Data type**: float32 initially, quantized to int8

### Creation Process

```python
def create_representative_dataset(data_loader, num_samples=2000):
    """Create representative dataset for quantization calibration."""

    def representative_gen():
        for features, _ in data_loader.take(num_samples):
            # Features shape: [batch, time, mel_bins]
            # Convert to required shape [1, 3, 40]
            yield [features.numpy().astype(np.float32)]

    return representative_gen

# Usage
converter.representative_dataset = create_representative_dataset(
    validation_data_loader,
    num_samples=2000
)
```

### Data Preparation

- Use validation/test data that represents real-world audio distribution
- Include both positive (wake word) and negative samples
- Ensure mel spectrogram values are in the expected range (0.0-26.0)

## Manifest Generation

### Manifest Format (V2)

```json
{
  "type": "micro",
  "wake_word": "Hey Katya",
  "author": "Sarpel GURAY",
  "website": "https://github.com/sarpel/microwakeword-training-platform",
  "model": "hey_katya.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 0,
    "minimum_esphome_version": "2024.7.0"
  }
}

### Key Parameters

- **probability_cutoff**: Threshold for wake word detection (0.0-1.0)
- **sliding_window_size**: Number of frames to average for stable detection
- **tensor_arena_size**: Memory allocation for TFLite runtime (bytes).
  - **Auto-resolve (recommended):** Set `"tensor_arena_size": 0` in your config (e.g. under `export:`) to enable automatic calculation. When `export.tensor_arena_size=0`, the exporter measures the TFLite model's tensor allocations and adds a safety margin of `arena_size_margin` (default 10%, controlled by the `export.arena_size_margin` key) to produce the final value written to the manifest.
  - **Failure behavior:** If auto-resolve fails (e.g. the TFLite interpreter cannot allocate tensors), the exporter logs an error and leaves `tensor_arena_size` as `0` in the manifest so the ESPHome runtime falls back to its own default behavior. Check the export logs for errors and re-run the export after correcting the issue.
  - **Manual override:** Set `"tensor_arena_size"` to a non-zero byte value (e.g. `22860`) to force a specific arena size regardless of the computed value. Use this when you know the exact memory budget on your target device. Related config keys: `export.tensor_arena_size` and `export.arena_size_margin`.
- **minimum_esphome_version**: Required ESPHome version for compatibility

### Calculation Methods

```python
def calculate_tensor_arena_size(model_path):
    """Calculate required tensor arena size for ESPHome."""
    # Analyze TFLite tensor allocations and add safety margin
    return measured_bytes_with_margin
def generate_manifest(model_path, config):
    """Generate ESPHome-compatible manifest file."""
    manifest = {
        "type": "micro",
        "version": 2,
        "micro": {
            "probability_cutoff": config.probability_cutoff,
            "sliding_window_size": config.sliding_window_size,
            "tensor_arena_size": calculate_tensor_arena_size(model_path),
            "minimum_esphome_version": "2024.7.0"
        }
    }
    return manifest
```

## Verification Process

### Verification Checklist

- ✅ **Input shape**: [1, 3, 40] int8 (scale=0.101961, zero_point=-128)
- ✅ **Output shape**: [1, 1] uint8 (scale=0.00390625, zero_point=0)
- ✅ **Subgraphs**: Exactly 2 — canonical project count is Subgraph 0 (main, 94 tensors) + Subgraph 1 (initialization, 12 tensors)
- ✅ **State variables**: Exactly 6 int8-quantized variables in the official reference (`stream`, `stream_1`, `stream_2`, `stream_3`, `stream_4`, `stream_5`)
- ✅ **Operations**: 13 unique op types from 20 registered ESPHome op resolvers
- ✅ **Manifest**: Valid JSON with required fields
- ✅ **Compatibility**: Passes ESPHome verification script

Note: `ai_edge_litert` may expose one extra runtime scratch tensor and report
`95` tensors for Subgraph 0. This repository does not use `ai_edge_litert` as
its canonical export/verification path, so the documentation standard here is
`94 / 12`.

### Verification Script

```python
def verify_exported_model(tflite_path, manifest_path):
    """Verify exported model meets ESPHome requirements."""

    # Load and analyze TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Check input/output shapes and types
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    assert input_details[0]['shape'] == [1, 3, 40]
    assert input_details[0]['dtype'] == np.int8
    assert output_details[0]['shape'] == [1, 1]
    assert output_details[0]['dtype'] == np.uint8

    # Check state variables
    state_vars = [tensor for tensor in interpreter.get_tensor_details()
                  if 'stream' in tensor['name']]
    assert len(state_vars) == 6

    # Verify manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest['version'] == 2
    assert 'micro' in manifest

    print("✅ All verification checks passed!")
```

## Common Export Issues and Solutions

### Issue: Shape mismatch during export (e.g., `(2624, 1) vs (2048, 1)`)
**Cause**: Old checkpoint from pre-Flatten architecture (before 2026-03-11) incompatible with current export pipeline
**Solution**: Retrain with current code. The Dense layer input size changed from 64 to `temporal_frames × 64` after the Flatten fix.

### Issue: Model fails on device with silent failure
**Cause**: Incorrect output quantization (int8 instead of uint8)
**Solution**: Ensure `converter.inference_output_type = tf.uint8`

### Issue: State variables not quantized
**Cause**: Missing `_experimental_variable_quantization` flag
**Solution**: Set `converter._experimental_variable_quantization = True`

### Issue: Insufficient representative data
**Cause**: Too few samples (< 500) or poor data distribution
**Solution**: Use 2000+ samples covering full mel spectrogram range

### Issue: Tensor arena too small
**Cause**: Incorrect calculation of memory requirements
**Solution**: Set `export.tensor_arena_size: 0` and let the export pipeline auto-resolve from the generated TFLite model

### Issue: Incompatible operations
**Cause**: Non-whitelisted ops in the model
**Solution**: Check ARCHITECTURAL_CONSTITUTION.md for allowed operations

### Issue: BatchNorm folding fails
**Cause**: Model.export() used instead of ExportArchive
**Solution**: Use `tf.keras.export.ExportArchive` for SavedModel export

## ESPHome Deployment Instructions

### 1. Copy Files to ESPHome

Place the exported files in your ESPHome configuration directory:

```
esphome_config/
├── wake_word.tflite
└── wake_word.json  # manifest file
```

### 2. Configure ESPHome YAML

```yaml
micro_wake_word:
  model: wake_word.tflite
  on_wake_word_detected:
    - logger.log: "Wake word detected!"
```

### 3. Flash to Device

```bash
esphome run your_config.yaml
```

### 4. Monitor Performance

Check ESPHome logs for:
- Model loading confirmation
- **Memory**: Set `export.tensor_arena_size: 0` to auto-resolve arena from exported model tensor allocations (+ `arena_size_margin`).
- Detection performance metrics

### 5. Tuning Thresholds

Adjust `probability_cutoff` based on your environment:
- Quiet environments: 0.95-0.98
- Noisy environments: 0.90-0.95
- Test with representative audio samples

## Performance Optimization Tips

- **Memory**: Prefer auto-resolved arena sizing (`tensor_arena_size: 0`) and validate on target hardware.
- **Latency**: Streaming model processes 3 frames (30ms) per inference
- **Accuracy**: Use sufficient representative data for quantization
- **Compatibility**: Test with target ESPHome version before deployment

## Troubleshooting

### Debug Commands

```bash
# Verify model compatibility
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Analyze model structure
python -c "import tensorflow as tf; interpreter = tf.lite.Interpreter('model.tflite'); print(interpreter.get_tensor_details())"

# Check manifest validity
python -c "import json; print(json.load(open('manifest.json')))"
```

### Common Debug Steps

1. Run verification script on exported model
2. Check ESPHome device logs for error messages
3. Validate input data range matches training
4. Ensure all state variables are properly initialized
5. Test with known wake word samples first
