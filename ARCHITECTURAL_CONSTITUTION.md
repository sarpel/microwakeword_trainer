# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║    microWakeWord — ARCHITECTURAL CONSTITUTION                              ║
# ║    IMMUTABLE SOURCE TRUTH — VERIFIED FROM TFLITE FLATBUFFERS + C++ SOURCE  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

> ## ⛔ SUPREME GOVERNING DOCUMENT ⛔
>
> This file is the **single, immutable source of architectural truth** for this
> project. Every constant, shape, dtype, op name, and timing value written here
> was physically extracted from the official ESPHome microWakeWord v2 TFLite
> flatbuffers (`hey_jarvis.tflite`, `okay_nabu.tflite`) and cross-verified
> against the ESPHome C++ runtime source (`micro_wake_word.cpp`).
>
> ### THE RULE IS SIMPLE:
>
> **IF ANY CODE — WHETHER IT IS A BUG FIX, NEW FEATURE, REFACTOR,
> "SMALL TWEAK", OR A "QUICK CHANGE" — CONTRADICTS EVEN ONE CONSTANT
> OR INVARIANT DEFINED IN THIS DOCUMENT, THAT CODE IS NUCLEAR WASTE.**
>
> It does not matter how clever it is. It does not matter that tests pass.
> It does not matter that it "looks right". Code that breaks these rules
> produces a model that is physically incompatible with the ESPHome runtime.
> The device will silently corrupt its state, never wake, or never stop waking.
> There will be no error message. There is no recovery. There is only deletion.
>
> **The moment a change is imagined that touches a rule below, STOP.
> Re-read this document from the top. If there is still doubt, the change is
> wrong. These rules cannot be unlocked by a config flag, a clever abstraction,
> or a senior engineer's override. They are the laws of physics for this stack.**

---

**Verification sources:**
- TFLite flatbuffer analysis: `hey_jarvis.tflite` (51.05 KB, 45 ops), `okay_nabu.tflite` (58.85 KB, 55 ops)
- ESPHome C++ runtime: `micro_wake_word.cpp` — op registration list
- Official microWakeWord training notebook (Google Colab)
- Last verified: 2025-02-25

---

## TABLE OF CONTENTS

1. [Article I — Audio Frontend Constants](#article-i--audio-frontend-constants)
2. [Article II — Model I/O Contract](#article-ii--model-io-contract)
3. [Article III — Quantization Parameters](#article-iii--quantization-parameters)
4. [Article IV — Permitted TFLite Operations](#article-iv--permitted-tflite-operations)
5. [Article V — Dual-Subgraph Structure](#article-v--dual-subgraph-structure)
6. [Article VI — Streaming State Variables](#article-vi--streaming-state-variables)
7. [Article VII — Inference Timing & Ring Buffer Math](#article-vii--inference-timing--ring-buffer-math)
8. [Article VIII — MixedNet Architecture Variants](#article-viii--mixednet-architecture-variants)
9. [Article IX — Export & Quantization Requirements](#article-ix--export--quantization-requirements)
10. [Article X — ESPHome Manifest Contract](#article-x--esphome-manifest-contract)
11. [Violation Consequence Summary](#violation-consequence-summary)

---

## Article I — Audio Frontend Constants

> **IMMUTABLE. HARDWARE-DICTATED. NON-NEGOTIABLE.**
>
> These values are burned into the ESPHome firmware. The microcontroller's
> audio pipeline produces features with exactly these parameters. Any model
> trained with different values will receive features it was never trained on
> and will produce garbage predictions — silently, at runtime, on real hardware.

| Constant | Value | Why It Cannot Change |
|---|---|---|
| `sample_rate_hz` | **16 000 Hz** | ESPHome ADC hardware clock |
| `mel_bins` | **40** | Feature tensor width; changes model input shape |
| `window_size_ms` | **30 ms** | 480 samples per FFT window; baked into pymicro-features |
| `window_step_ms` | **10 ms** | 160 samples per hop; determines temporal resolution |
| `upper_band_limit_hz` | **7 500 Hz** | Nyquist constraint for 16 kHz + margin |
| `lower_band_limit_hz` | **125 Hz** | DC rejection floor |
| `enable_pcan` | **True** | Per-Channel Amplitude Normalization; deactivating changes the feature distribution entirely |
JB|| `clip_duration_ms` | **Configurable** | Determines training input length. Common: 1000ms (1s), 1500ms, 3000ms. Does NOT affect streaming inference.

YX|### Derived Constants
PX|
KP|```
KR|samples_per_hop    = sample_rate_hz × (window_step_ms / 1000) = 160
VK|spectrogram_frames = clip_duration_ms / window_step_ms         = varies (100 for 1000ms)
QZ|input_feature_shape = [spectrogram_frames, mel_bins]           = e.g., [100, 40] for 1000ms clip
WQ|```
WR|
VQ|> **Training vs Streaming Input Shapes:**
BT|>
HM|> Training input shape depends on `clip_duration_ms`: `(clip_duration_ms / window_step_ms, 40)`
NW|> - 1000ms clip → `(100, 40)`
QJ|> - 1500ms clip → `(150, 40)`
XM|> - 3000ms clip → `(300, 40)`
XP|>
MK|> Streaming inference shape is ALWAYS `[1, 3, 40]` regardless of clip duration. The model
ZR|> processes 3 new frames (30ms of audio) per inference call on device.

```
samples_per_hop    = sample_rate_hz × (window_step_ms / 1000) = 160
spectrogram_frames = clip_duration_ms / window_step_ms         = 100
input_feature_shape = [spectrogram_frames, mel_bins]           = [100, 40]
```

---

## Article II — Model I/O Contract

> **IMMUTABLE. VERIFIED FROM TFLITE FLATBUFFERS.**
>
> These are the byte-level types and shapes the ESPHome C++ runtime passes
> into and reads out of the model. Any mismatch is a buffer overread/write.
> There is no type coercion at the edge. There is no runtime check.
> THE FIRMWARE WILL WRITE INTO ARBITRARY MEMORY AND CRASH OR CORRUPT STATE.

### Input Tensor

| Property | Value |
|---|---|
| Shape | `[1, stride, 40]` = **`[1, 3, 40]`** |
| Dtype | **`int8`** |
| Quantization scale | `0.101961` |
| Quantization zero_point | `-128` |

### Output Tensor

| Property | Value |
|---|---|
| Shape | **`[1, 1]`** |
| Dtype | **`uint8`** ← **NOT int8. NOT float32. UINT8.** |
| Quantization scale | `0.003906` |
| Quantization zero_point | `0` |

> ⛔ **THE OUTPUT DTYPE IS `uint8`. THIS IS NOT A TYPO. THIS IS NOT A DEFAULT.**
>
> ESPHome reads the output tensor as an unsigned byte and compares it against
> `probability_cutoff × 255`. If the output type is `int8`, every prediction
> will be misread. The model will appear to work in Python, pass all Python
> unit tests, produce a valid `.tflite` file — and be completely broken on
> the device. This has been verified directly from the C++ source code.

---

## Article III — Quantization Parameters

> **IMMUTABLE. CALIBRATION FROM REPRESENTATIVE DATASET MUST REPRODUCE THESE.**
>
> Quantization maps float activations to int8/uint8. Wrong scale/zero_point
> means the entire numeric range of the model is shifted. A model with
> corrupted quantization parameters will never converge to correct predictions.
> These values were measured from the official reference models.

### Input Quantization

```
float_value = (int8_value - zero_point) × scale
            = (int8_value - (-128))     × 0.101961
            = (int8_value + 128)        × 0.101961
```

Input range maps: `int8[-128, 127]` → `float[0.0, ~26.0]`

### Output Quantization

```
float_value = (uint8_value - zero_point) × scale
            = (uint8_value - 0)          × 0.003906
            = uint8_value                × 0.003906
```

Output range maps: `uint8[0, 255]` → `float[0.0, ~1.0]`

### Calibration Representative Dataset

The representative dataset **must** include forced min/max boundary samples:

```python
sample_fingerprints[0][0, 0] = 0.0   # Force minimum
sample_fingerprints[0][0, 1] = 26.0  # Force maximum
```

> Without these boundary samples, the quantizer may clip the activation range
> and produce a scale that differs from the reference. **500 training samples
> minimum** must be used for calibration.

### Required Converter Flags

```python
converter.optimizations = {tf.lite.Optimize.DEFAULT}
converter._experimental_variable_quantization = True   # MANDATORY for state vars
converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.uint8  # ← UINT8. ALWAYS. NON-NEGOTIABLE.
```

---

## Article IV — Permitted TFLite Operations

> **IMMUTABLE. EXHAUSTIVE. VERIFIED FROM FLATBUFFER OP CODES.**
>
> The ESPHome TFLite Micro runtime registers exactly 14 op resolvers. Any op
> not in this list is **not registered** and will cause a fatal error at model
> loading time on the device. There are no custom ops. All ops are BUILTIN.

### ESPHome-Registered Op Resolvers (from `micro_wake_word.cpp`)

```cpp
resolver.AddCallOnce();
resolver.AddVarHandle();
resolver.AddReadVariable();
resolver.AddStridedSlice();
resolver.AddConcatenation();
resolver.AddAssignVariable();
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddMul();
resolver.AddAdd();
resolver.AddMean();
resolver.AddFullyConnected();
resolver.AddLogistic();
resolver.AddQuantize();
```

### Op Count Verification (from TFLite flatbuffers)

| Op Name | hey_jarvis count | okay_nabu count | Purpose |
|---|---|---|---|
| `CALL_ONCE` | 1 | 1 | Invoke Subgraph 1 initialization once at startup |
| `VAR_HANDLE` | 6 | 6 | Create handles to the 6 streaming state variables |
| `READ_VARIABLE` | 6 | 6 | Read state from previous inference step |
| `ASSIGN_VARIABLE` | 6 | 6 | Write updated state for next inference step |
| `CONCATENATION` | 6 | 8 | Concatenate old buffer frames with new input frames |
| `STRIDED_SLICE` | 6 | 10 | Slice off oldest frames to update ring buffers |
| `CONV_2D` | 5 | 5 | Pointwise 1×1 convolutions |
| `DEPTHWISE_CONV_2D` | 4 | 6 | Depthwise spatial convolutions in MixConv blocks |
| `RESHAPE` | 2 | 2 | Flatten for Dense layer |
| `SPLIT_V` | 0 | 2 | Split tensor for StridedKeep (okay_nabu variant only) |
| `FULLY_CONNECTED` | 1 | 1 | Classification head |
| `LOGISTIC` | 1 | 1 | Sigmoid activation on output |
| `QUANTIZE` | 1 | 1 | Cast float32 output to uint8 |
| `MUL` | varies | varies | BatchNormalization fold or residuals |
| `ADD` | varies | varies | Residual connections / biases |

> ⛔ **ALL OPS SHOW `CustomCode: N/A` IN FLATBUFFER ANALYSIS.**
>
> This is the verified proof that there are **zero custom ops** in the model.
> Any implementation that attempts to register a custom op, use a TF Select op,
> or use any op outside this list will produce an unloadable model.
> The device will halt with an unresolved op error and never boot the wake word
> engine.

---

## Article V — Dual-Subgraph Structure

> **IMMUTABLE. BOTH SUBGRAPHS MUST BE PRESENT. SHAPES ARE EXACT.**
>
> The model contains exactly **2 subgraphs**. Subgraph 1 is the initialization
> NoOp graph that zeroes all state variables. It is invoked exactly once at
> device boot via the `CALL_ONCE` op in Subgraph 0. Both subgraphs must be
> present and intact in the exported `.tflite` file or the device will crash
> on startup.

```
Subgraph [0]: Main Inference Graph
├── Input:  [1, 3, 40]  INT8
├── CALL_ONCE  ──────────────────→  Triggers Subgraph [1] once
├── VAR_HANDLE × 6  ─────────────→  Bind handles to state vars
├── READ_VARIABLE × 6  ──────────→  Load state buffers
│     ├── stream   : [1, 2,  1, 40]   (ring buffer before first Conv2D)
│     ├── stream_1 : [1, 4,  1, 32]   (MixConv block 0)
│     ├── stream_2 : [1, 10, 1, 64]   (MixConv block 1)
│     ├── stream_3 : [1, 14, 1, 64]   (MixConv block 2)
│     ├── stream_4 : [1, 22, 1, 64]   (MixConv block 3)
│     └── stream_5 : [1, 5,  1, 64]   (Temporal mean pooling)
├── CONCATENATION  ──────────────→  [old_frames | new_input]
├── (inference ops: Conv2D, DepthwiseConv2D, FC, Logistic…)
├── STRIDED_SLICE × 6  ──────────→  Extract new ring buffer state
├── ASSIGN_VARIABLE × 6  ────────→  Write updated state back
└── Output: [1, 1]  UINT8

Subgraph [1]: Initialization Graph (invoked once, then dormant)
├── 12 ops total (hey_jarvis) / 12 ops total (okay_nabu)
├── 12 tensors
└── Zeros all 6 state variables (pseudoconst zero tensors)
```

### Subgraph Op/Tensor Counts (verified)

| | hey_jarvis | okay_nabu |
|---|---|---|
| Subgraph 0 ops | 45 | 55 |
| Subgraph 0 tensors | 71 | 95 |
| Subgraph 1 ops | 12 | 12 |
| Subgraph 1 tensors | 12 | 12 |

---

## Article VI — Streaming State Variables

> **IMMUTABLE. EXACT SHAPES. EXACT COUNT. EXACT ORDER.**
>
> There are exactly **6** streaming state variables. Not 5. Not 7. Each one
> is a ring buffer that stores the "past context" needed for the current
> inference step. Their shapes encode the temporal kernel sizes of the
> convolutional layers. Changing a kernel size changes the ring buffer shape.
> An inconsistency between the training-time kernel size and the export-time
> state variable shape produces a model that is silently wrong — it will run,
> produce predictions, and those predictions will be computed on the wrong
> temporal context.

### State Variable Shapes

| Variable | hey_jarvis shape | okay_nabu shape | Ring buffer holds |
|---|---|---|---|
| `stream` | `[1, 2, 1, 40]` | `[1, 2, 1, 40]` | 2 frames before first `Conv2D` |
| `stream_1` | `[1, 4, 1, 30]` | `[1, 4, 1, 32]` | MixConv block 0 context |
| `stream_2` | `[1, 8, 1, 60]` | `[1, 10, 1, 64]` | MixConv block 1 context |
| `stream_3` | `[1, 12, 1, 60]` | `[1, 14, 1, 64]` | MixConv block 2 context |
| `stream_4` | `[1, 20, 1, 60]` | `[1, 22, 1, 64]` | MixConv block 3 context |
| `stream_5` | `[1, 4, 1, 60]` | `[1, 5, 1, 64]` | Temporal mean pooling buffer |

### Total State Memory

| Model | State bytes |
|---|---|
| hey_jarvis | **2 840 bytes** |
| okay_nabu | **3 520 bytes** |

### Ring Buffer Law

```
buffer_frames = kernel_size - stride
```

This identity is inviolable. It governs every state variable shape. If you
change a kernel size or the global stride, every downstream state variable
shape changes. The exported model will have the wrong buffer sizes if the
streaming conversion does not re-derive them from this formula.

---

## Article VII — Inference Timing & Ring Buffer Math

> **IMMUTABLE. DRIVEN BY HARDWARE AUDIO PIPELINE TIMING.**
>
> The ESPHome audio pipeline calls the model on a fixed schedule.
> There is no mechanism to change this schedule from the model side.
> The model must consume exactly the data that arrives.

| Timing Constant | Value | Derivation |
|---|---|---|
| Feature frame period | **10 ms** | `window_step_ms` |
| New frames per inference call | **3** | `stride = 3` |
| Inference period | **30 ms** | `stride × window_step_ms = 3 × 10` |
| Samples consumed per inference | **480** | `stride × samples_per_hop = 3 × 160` |

### The Stride Constant

```
stride = 3
```

**This value appears in:**
1. Model input shape: `[1, stride, 40]` = `[1, 3, 40]`
2. All ring buffer size calculations: `buffer = kernel - stride`
3. TFLite representative dataset generation (slides by `stride` per sample)
4. ESPHome's hardware call cadence

Changing `stride` requires simultaneously changing: the input tensor shape,
all 6 state variable shapes, the representative dataset generator, and the
ESPHome YAML configuration. Changing it in one place and not the others
silently produces a misaligned model.

---

## Article VIII — MixedNet Architecture Variants

> **IMMUTABLE WITHIN EACH VARIANT. PARAMETERIZABLE ACROSS VARIANTS.**
>
> Two architecture variants exist, both verified from official models.
> These are the only two proven-compatible configurations. A custom
> configuration is permitted **only if** it follows the MixedNet structural
> rules defined here and the streaming conversion produces state variables
> whose shapes satisfy the ring buffer law in Article VII.

### Variant A — `hey_jarvis` (45 ops, simpler)

```python
first_conv_filters    = 30
first_conv_kernel_size = 5          # → stream shape [1, 2, 1, 40]  (5-3=2)
stride                = 3           # GLOBAL IMMUTABLE CONSTANT
pointwise_filters     = [60, 60, 60, 60]
mixconv_kernel_sizes  = [[5], [9], [13], [21]]
repeat_in_block       = [1, 1, 1, 1]
residual_connection   = [0, 0, 0, 0]
```

### Variant B — `okay_nabu` (55 ops, uses SPLIT_V / StridedKeep)

```python
first_conv_filters    = 32
first_conv_kernel_size = 5
stride                = 3           # GLOBAL IMMUTABLE CONSTANT
pointwise_filters     = [64, 64, 64, 64]
mixconv_kernel_sizes  = [[5], [9], [13], [21]]
repeat_in_block       = [1, 1, 1, 1]
residual_connection   = [0, 0, 0, 0]
```

### Structural Rules (apply to all variants)

1. **All temporal convolutions must be wrapped in `stream.Stream`** so that the
   streaming conversion can extract ring buffer state variables.
2. **`padding="valid"` on the time axis** — padding the time axis would produce
   different activations in non-streaming vs streaming modes, which breaks
   state consistency.
3. **`use_bias=False` on all Conv2D/DepthwiseConv2D** — biases are folded into
   BatchNormalization during export.
4. **Classification head:** exactly one `Dense(1, activation="sigmoid")` as the
   last layer. The sigmoid maps to [0,1] before being quantized to uint8.
5. **BatchNormalization** must be present after every depthwise/pointwise conv
   block. It is folded into conv weights during streaming conversion.
6. **No LSTM, GRU, attention, or recurrent layers.** The ESPHome runtime does
   not register these ops. They do not exist in the allowed op set.
7. **No custom TensorFlow ops, no `tf.py_function`, no `tf.numpy_function`.**
   TFLite Micro cannot execute any Python callback.

---

## Article IX — Export & Quantization Requirements

> **IMMUTABLE. THESE FLAGS ARE NOT OPTIONAL. NOT CONFIGURABLE. NOT DEFAULTS.**
>
> The export pipeline has two mandatory stages. Skipping either stage, or
> changing the quantization types, produces a `.tflite` file that is either
> unloadable on the device or produces completely wrong predictions.

### Stage 1: Non-Streaming → Streaming SavedModel Conversion

```python
converted_model = convert_model_saved(
    model_non_stream = model,
    config           = config,
    folder           = "stream_state_internal",
    mode             = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
)
```

This step materializes the ring buffer state variables from the `stream.Stream`
wrappers. Without it, the model has no `VAR_HANDLE`/`READ_VARIABLE`/
`ASSIGN_VARIABLE` ops and will fail ESPHome's op registration check.

### Stage 2: Streaming SavedModel → Quantized TFLite

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# These four lines are non-negotiable:
converter.optimizations                        = {tf.lite.Optimize.DEFAULT}
converter._experimental_variable_quantization  = True          # state vars MUST be quantized
converter.target_spec.supported_ops            = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
converter.inference_input_type                 = tf.int8
converter.inference_output_type                = tf.uint8       # UINT8. ALWAYS.
```

> ⛔ `converter._experimental_variable_quantization = True` is **required**.
> Without it, the 6 state variable tensors remain in float32, their
> `VAR_HANDLE` and `ASSIGN_VARIABLE` ops do not receive int8 quantization
> parameters, and the TFLite Micro int8-only kernel resolver cannot find
> matching op implementations. The model will fail to load on device.

### Representative Dataset Requirements

- Minimum **500 training samples**
- Samples sliced with `stride=3` to match runtime input cadence
- **Must include boundary anchor points** (float 0.0 and 26.0) to pin the
  quantization scale to the correct range

---

## Article X — ESPHome Manifest Contract

> **IMMUTABLE FIELDS. REQUIRED EXACT VALUES.**
>
> ESPHome reads the manifest JSON to configure its wake word engine.
> The fields below have fixed types and constraints. Wrong values produce
> either a silent misconfiguration (model never triggers) or a firmware
> compile error.

```json
{
  "type": "micro",
  "version": 2,
  "micro": {
    "feature_step_size": 10,
    "sliding_window_size": 5,
    "tensor_arena_size": 26080,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

| Field | Constraint | Consequence of violation |
|---|---|---|
| `type` | Must be `"micro"` | ESPHome ignores the file entirely |
| `version` | Must be `2` | v1 loader path is taken; state variables are not handled |
| `feature_step_size` | Must be `10` (ms) | Timing desync; model receives stale/skipped frames |
| `sliding_window_size` | Typically `5` | Affects false-positive suppression, not model correctness |
| `tensor_arena_size` | Must be ≥ actual arena | Runtime OOM crash on device |
| `minimum_esphome_version` | Must be `"2024.7.0"` | Older firmware silently uses wrong op resolver |

### Tensor Arena Sizing Rules

- `hey_jarvis` reference: **26 080 bytes**
- `okay_nabu` reference: ~28 000–30 000 bytes
- **Measure empirically** using `tflite_micro_arena_size` tool
- **Add 10% margin** to the measured value
- Underestimating causes silent memory corruption, not a clean error

---

## Violation Consequence Summary

> This table exists so that developers understand exactly what breaks —
> in production, on real hardware — when each rule is violated.
> "Works in Python" is not a definition of correctness for this project.

| Article | What you might change | What actually breaks |
|---|---|---|
| I | `mel_bins`, `window_step_ms`, `sample_rate_hz` | Input tensor shape mismatch; model receives wrong feature dimensions; predictions are garbage |
| II | Output dtype from `uint8` to `int8` | ESPHome reads signed bytes as unsigned; every prediction ≥ 128 is misinterpreted as negative; wake word never triggers |
| III | Quantization calibration dataset too small or missing boundaries | Scale/zero_point shift; dynamic range is wrong; model runs but predictions are compressed into a tiny range; effectively non-functional |
| III | Remove `_experimental_variable_quantization` | State variables stay float32; int8-only kernel resolver fails at model load; device halts |
| IV | Use any op outside the 14 registered ops | Op resolver returns `kTfLiteError` at load time; device halts; wake word engine never starts |
| V | Export without `STREAM_INTERNAL_STATE_INFERENCE` mode | No `VAR_HANDLE`/state ops in graph; model has no memory; predictions are independent per-frame; accuracy is random |
| VI | Wrong ring buffer size (kernel / stride mismatch) | Ring buffer reads from wrong temporal offset; model sees scrambled temporal context; no crash, just wrong predictions forever |
| VII | Change stride in code but not in export/manifest | Input tensor slicing misaligned with ring buffer writes; state corruption accumulates across time; model degrades after first second of audio |
| VIII | Add LSTM/GRU/attention or custom ops | Op not registered; device halts at model load |
| IX | `inference_output_type = tf.int8` | Output read as signed; probabilities inverted relative to ESPHome's uint8 comparison threshold; model either always or never triggers |
| X | `feature_step_size ≠ 10` | ESPHome feeds frames at wrong cadence; ring buffers contain wrong number of frames; model evaluates context that is 1.5× or 2× too long/short |

---

*This document is append-only with respect to verified facts.
No value may be changed without re-running flatbuffer extraction against
official ESPHome reference models and updating the verification date.
Adding unverified values is prohibited — mark any new entry as `UNVERIFIED`
and include the expected verification method.*
