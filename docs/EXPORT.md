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
3. **Fold BatchNorm**: Eliminate ReadVariableOp by folding batch normalization layers into preceding convolutions
4. **Convert to SavedModel**: Export using tf.keras.export.ExportArchive (NOT model.export())". Based on learnings: Use tf.keras.export.ExportArchive (NOT model.export()) and set converter._experimental_variable_quantization = True for TFLite export.
5. **TFLite Conversion**: Convert to TFLite with INT8 quantization using representative dataset
6. **Generate Manifest**: Create ESPHome-compatible manifest file with model metadata
7. **Verification**: Validate the exported model meets ESPHome requirements

## Streaming Model Conversion

### StreamingExportModel Architecture

The streaming model uses a ring buffer approach to maintain state across inference calls, enabling real-time audio processing without storing the entire audio history.

**Key Features:**
- 6 state variables for ring buffers (~3.5KB total)
- Okay_nabu variant: 32/64 filters, [[5],[7,11],[9,15],[23]] kernels
- BatchNorm folding eliminates ReadVariableOp for better performance

### State Variables

The model maintains 6 state variables representing ring buffers at different stages:

1. `stream`: [1,2,1,40] - Initial convolution buffer
2. `stream_1`: [1,4,1,32] - Block 0 buffer
3. `stream_2`: [1,10,1,64] - Block 1 buffer
4. `stream_3`: [1,14,1,64] - Block 2 buffer
5. `stream_4`: [1,22,1,64] - Block 3 buffer
6. `stream_5`: [1,5,1,64] - Temporal pooling buffer

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

## INT8 Quantization

### Critical TFLite Settings

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = {tf.lite.Optimize.DEFAULT}
converter._experimental_variable_quantization = True  # REQUIRED for state variables
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.uint8  # MUST be uint8 for ESPHome compatibility
```

**Important Notes:**
- `inference_output_type` must be `tf.uint8`, not `tf.int8`
- `_experimental_variable_quantization = True` is required to quantize state variables
- State variables are quantized to int8 to minimize memory usage

### Quantization Range

The representative dataset determines the quantization range. Key boundaries:
- **Input range**: 0.0 to 25.85 (mel spectrogram values)
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
- Ensure mel spectrogram values are in the expected range (0.0-25.85)

## Manifest Generation

### Manifest Format (V2)

```json
{
  "type": "micro",
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "tensor_arena_size": 26080,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

### Key Parameters

- **probability_cutoff**: Threshold for wake word detection (0.0-1.0)
- **sliding_window_size**: Number of frames to average for stable detection
- **tensor_arena_size**: Memory allocation for TFLite runtime (bytes)
- **minimum_esphome_version**: Required ESPHome version for compatibility

### Calculation Methods

```python
def calculate_tensor_arena_size(model_path):
    """Calculate required tensor arena size for ESPHome."""
    # Analyze model operations and tensor sizes
    # Return size in bytes (typically 20-30KB for MixedNet)
    return 26080

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

- ✅ **Input shape**: [1, 3, 40] int8
- ✅ **Output shape**: [1, 1] uint8
- ✅ **Subgraphs**: Exactly 2 (main graph + initialization)
- ✅ **State variables**: Exactly 6 int8-quantized variables
- ✅ **Operations**: Only whitelisted ops for ESPHome
- ✅ **Manifest**: Valid JSON with required fields
- ✅ **Compatibility**: Passes ESPHome verification script

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
**Solution**: Use the model's `calculate_tensor_arena_size()` function

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
- Memory usage (should be < 30KB arena)
- Detection performance metrics

### 5. Tuning Thresholds

Adjust `probability_cutoff` based on your environment:
- Quiet environments: 0.95-0.98
- Noisy environments: 0.90-0.95
- Test with representative audio samples

## Performance Optimization Tips

- **Memory**: Monitor tensor arena usage - target < 30KB
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
