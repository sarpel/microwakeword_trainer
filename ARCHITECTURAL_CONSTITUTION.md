# microWakeWord v2 — ARCHITECTURAL CONSTITUTION

## PREAMBLE: WHAT THIS DOCUMENT IS AND HOW TO USE IT

This file is the **single, authoritative source of architectural truth** for this project. Every constant, shape, dtype, op name, tensor count convention, and timing value written here was extracted and cross-verified from three independent sources:

1. **TFLite flatbuffer binary parsing** of the official `okay_nabu.tflite` reference model (60,264 bytes). Not interpreter enumeration — direct `Model.GetRootAsModel()` flatbuffer deserialization.
2. **ESPHome C++ runtime source code** (`streaming_model.cpp`, `streaming_model.h`, `micro_wake_word.cpp`) from ESPHome 2025.12.5 API documentation.
3. **OHF-Voice/micro-wake-word Python training pipeline** (`utils.py`, `model_train_eval.py`, `inference.py`, `setup.py`) from the official GitHub repository at `github.com/OHF-Voice/micro-wake-word`.

Where different interpreter implementations expose different **runtime scratch/workspace tensors**, this document states the project's **canonical counting convention** explicitly. Those runtime-only scratch tensors are not treated as architectural signals unless stated otherwise.

**Verification date:** 2026-03-13

### THE GOVERNING RULE

If any code in this project — a bugfix, new feature, refactor, "small tweak", or "quick change" — contradicts even one constant or invariant defined in this document, that code is wrong. It does not matter if tests pass. It does not matter if it "looks right" in Python. Code that breaks these rules produces a model that is physically incompatible with the ESPHome runtime. The device will silently corrupt its state, never wake, or never stop waking. There will be no error message. There is no recovery.

### AI CODING AGENT INSTRUCTIONS

You are reading this document because you are building or modifying code for this project. Follow these rules:

1. Before writing any code that touches model architecture, export pipeline, or quantization: re-read the relevant Article below.
2. If a user request or your own reasoning contradicts a rule here, the rule here wins. Ask the user to confirm before proceeding.
3. Never infer architectural values from memory or training data. Use only the values stated here.
4. The `okay_nabu.tflite` model is the v2 reference. All shapes, op counts, and structural properties are measured from this specific file.

---

## TABLE OF CONTENTS

1. [Article I — Audio Frontend Constants](#article-i--audio-frontend-constants)
2. [Article II — Model I/O Contract](#article-ii--model-io-contract)
3. [Article III — Quantization Parameters](#article-iii--quantization-parameters)
4. [Article IV — Permitted TFLite Operations](#article-iv--permitted-tflite-operations)
5. [Article V — Dual-Subgraph Structure](#article-v--dual-subgraph-structure)
6. [Article VI — Streaming State Variables](#article-vi--streaming-state-variables)
7. [Article VII — Inference Timing and Ring Buffer Math](#article-vii--inference-timing-and-ring-buffer-math)
8. [Article VIII — MixedNet Architecture](#article-viii--mixednet-architecture)
9. [Article IX — Training Pipeline and EMA Management](#article-ix--training-pipeline-and-ema-management)
10. [Article X — Export and Quantization Pipeline](#article-x--export-and-quantization-pipeline)
11. [Article XI — ESPHome Manifest Contract](#article-xi--esphome-manifest-contract)
12. [Article XII — Package Dependencies and Version Constraints](#article-xii--package-dependencies-and-version-constraints)
13. [Article XIII — v1 vs v2 Model Differences](#article-xiii--v1-vs-v2-model-differences)
14. [Violation Consequence Matrix](#violation-consequence-matrix)

---

## Article I — Audio Frontend Constants

**Status: IMMUTABLE. HARDWARE-DICTATED. NON-NEGOTIABLE.**

These values are configured in the ESPHome firmware's audio preprocessing pipeline. The microcontroller produces feature tensors with exactly these parameters. Any model trained with different values will receive features it was never trained on and will produce garbage predictions silently at runtime.

**Source:** ESPHome `micro_wake_word.cpp` lines 76-90 (setup function), `streaming_model.h` preprocessor constants.

### Runtime Feature Extraction Parameters

| Constant | Value | Why It Cannot Change |
|---|---|---|
| `sample_rate_hz` | 16,000 Hz | ESPHome ADC hardware clock; hardcoded in firmware |
| `mel_bins` (PREPROCESSOR_FEATURE_SIZE) | 40 | Defines feature tensor width; changing it changes model input shape |
| `window_size_ms` (FEATURE_DURATION_MS) | 30 ms | 480 samples per FFT window; baked into the audio frontend C code |
| `window_step_ms` (features_step_size) | 10 ms | 160 samples per hop; determines temporal resolution. This is the v2 value. |
| `upper_band_limit_hz` | 7,500 Hz | Nyquist constraint for 16 kHz sample rate with margin |
| `lower_band_limit_hz` | 125 Hz | DC rejection floor |
| `enable_pcan` | True | Per-Channel Amplitude Normalization; disabling changes the entire feature distribution |

### PCAN and Noise Reduction Parameters (compiled into C++ extension)

These values are hardcoded in `pymicro-features` (the `rhasspy/pymicro-features` package) and cannot be changed at runtime. The `MicroFrontend()` constructor accepts no arguments.

| Parameter | Value | Category |
|---|---|---|
| `pcan_strength` | 0.95 | PCAN Gain Control |
| `pcan_offset` | 80.0 | PCAN Gain Control |
| `pcan_gain_bits` | 21 | PCAN Gain Control |
| `noise_even_smoothing` | 0.025 | Noise Reduction |
| `noise_odd_smoothing` | 0.06 | Noise Reduction |
| `noise_min_signal_remaining` | 0.05 | Noise Reduction |
| `log_scale_shift` | 6 | Log Scale |

**Source:** `rhasspy/pymicro-features` source `src/micro_features.cpp::init_cfg()`.

### Training-Time Parameters (do not affect streaming inference)

`clip_duration_ms` is configurable at training time. It determines the training input length but does NOT affect the streaming inference shape, because streaming inference always processes `stride` new frames per call regardless of how long the training clips were.

Common values: 1000 ms, 1500 ms, 3000 ms.

### Derived Constants

```
samples_per_hop    = sample_rate_hz × (window_step_ms / 1000) = 160
spectrogram_frames = clip_duration_ms / window_step_ms         = varies (100 for 1000ms)
```

Training input shape depends on `clip_duration_ms`: `(clip_duration_ms / window_step_ms, 40)`. For example, a 1000ms clip produces shape `(100, 40)`.

Streaming inference shape is ALWAYS `[1, stride, 40]` = `[1, 3, 40]` regardless of clip duration. The model processes `stride` new frames (30 ms of audio) per inference call on device.

---

## Article II — Model I/O Contract

**Status: IMMUTABLE. VERIFIED FROM TFLITE FLATBUFFER BINARY.**

These are the byte-level types and shapes the ESPHome C++ runtime passes into and reads out of the model. The runtime validates these at model load time.

**Source:** ESPHome `streaming_model.cpp` lines 64-88 (input/output verification in `load_model_()`).

### Input Tensor

| Property | Value | Verification |
|---|---|---|
| Shape | `[1, 3, 40]` | `input->dims->size == 3 && input->dims->data[0] == 1 && input->dims->data[2] == PREPROCESSOR_FEATURE_SIZE` |
| Dtype | `int8` | `input->type == kTfLiteInt8` — runtime check, model fails to load if violated |
| Quantization scale | 0.10196078568696976 (≈ 26/255) | Flatbuffer tensor 0 quantization parameters |
| Quantization zero_point | -128 | Flatbuffer tensor 0 quantization parameters |

The runtime reads `stride` from `input->dims->data[1]` at load time (streaming_model.cpp line 131), so the stride value is derived from the model itself — but for v2 okay_nabu models, this is always 3.

### Output Tensor

| Property | Value | Verification |
|---|---|---|
| Shape | `[1, 1]` | `output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 1` |
| Dtype | **`uint8`** | `output->type == kTfLiteUInt8` — runtime check, model fails to load if violated |
| Quantization scale | 0.00390625 (= 1/256) | Flatbuffer tensor 93 quantization parameters |
| Quantization zero_point | 0 | Flatbuffer tensor 93 quantization parameters |

**CRITICAL: The output dtype is `uint8`, NOT `int8`, NOT `float32`.** ESPHome reads the output as `output->data.uint8[0]` (streaming_model.cpp line 151) and compares it against `probability_cutoff` which is stored as a uint8 value (0-255). If the output type is `int8`, every prediction above 128 will be misread. The model will appear to work in Python but be completely broken on device.

---

## Article III — Quantization Parameters

**Status: IMMUTABLE. CALIBRATION FROM REPRESENTATIVE DATASET MUST REPRODUCE THESE.**

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

### Calibration Representative Dataset Requirements

The official export code in `OHF-Voice/micro-wake-word/microwakeword/utils.py` (function `convert_saved_model_to_tflite`, lines 219-240) defines the representative dataset as follows:

1. Retrieve 500 training spectrograms.
2. Force boundary anchor points on the first sample: `sample_fingerprints[0][0, 0] = 0.0` (minimum) and `sample_fingerprints[0][0, 1] = 26.0` (maximum). These pin the quantization scale to the correct range.
3. Slice each spectrogram by `stride` to match runtime input cadence. Each yielded sample has shape `[stride, 40]`.

Without these boundary samples, the quantizer may select a different scale that compresses the dynamic range. Without sufficient samples (minimum 500), quantization noise increases and predictions degrade.

---

## Article IV — Permitted TFLite Operations

**Status: IMMUTABLE. EXHAUSTIVE. VERIFIED FROM ESPHOME C++ SOURCE.**

The ESPHome TFLite Micro runtime registers exactly **20 op resolvers**. Any op not in this set is not registered and will cause a fatal `kTfLiteError` at model loading time on the device. The wake word engine will never start.

**Source:** ESPHome `streaming_model.cpp` function `register_streaming_ops_()`, lines 265-308 (ESPHome 2025.12.5). Note: this function is in `streaming_model.cpp`, NOT in `micro_wake_word.cpp`. It is called from the `WakeWordModel` constructor (line 181).

### Registered Op Resolvers (exact order from source code)

```cpp
// Source: streaming_model.cpp lines 269-308
// Function: register_streaming_ops_(tflite::MicroMutableOpResolver<20>& op_resolver)

op_resolver.AddCallOnce();          // 1
op_resolver.AddVarHandle();         // 2
op_resolver.AddReshape();           // 3
op_resolver.AddReadVariable();      // 4
op_resolver.AddStridedSlice();      // 5
op_resolver.AddConcatenation();     // 6
op_resolver.AddAssignVariable();    // 7
op_resolver.AddConv2D();           // 8
op_resolver.AddMul();              // 9
op_resolver.AddAdd();              // 10
op_resolver.AddMean();             // 11
op_resolver.AddFullyConnected();   // 12
op_resolver.AddLogistic();         // 13
op_resolver.AddQuantize();         // 14
op_resolver.AddDepthwiseConv2D();  // 15
op_resolver.AddAveragePool2D();    // 16
op_resolver.AddMaxPool2D();        // 17
op_resolver.AddPad();              // 18
op_resolver.AddPack();             // 19
op_resolver.AddSplitV();          // 20
```

The resolver template parameter is `MicroMutableOpResolver<20>`, confirming exactly 20 slots. The registration order does not affect functionality — it only matters that the set is complete.

### Op Usage in okay_nabu Reference Model (flatbuffer verified)

| Op Name | Subgraph 0 Count | Subgraph 1 Count | Purpose |
|---|---|---|---|
| CALL_ONCE | 1 | 0 | Invokes Subgraph 1 initialization once at startup |
| VAR_HANDLE | 6 | 6 | Creates handles to streaming state variables |
| READ_VARIABLE | 6 | 0 | Reads state from previous inference step |
| ASSIGN_VARIABLE | 6 | 6 | Writes updated state for next inference step |
| CONCATENATION | 8 | 0 | Concatenates old buffer frames with new input frames |
| STRIDED_SLICE | 10 | 0 | Slices off oldest frames to update ring buffers |
| CONV_2D | 5 | 0 | Pointwise 1×1 convolutions |
| DEPTHWISE_CONV_2D | 6 | 0 | Depthwise spatial convolutions in MixConv blocks |
| RESHAPE | 2 | 0 | Expand dims and flatten for Dense layer |
| SPLIT_V | 2 | 0 | Splits tensor for StridedKeep dual-kernel blocks |
| FULLY_CONNECTED | 1 | 0 | Classification head (Dense layer) |
| LOGISTIC | 1 | 0 | Sigmoid activation on output |
| QUANTIZE | 1 | 0 | Re-quantizes sigmoid output from int8 encoding to uint8 |
| MUL | 0 | 0 | Registered but unused in okay_nabu (available for BatchNorm fold or residuals) |
| ADD | 0 | 0 | Registered but unused (available for residual connections) |
| MEAN | 0 | 0 | Registered but unused (available for pooling) |
| AVERAGE_POOL_2D | 0 | 0 | Registered but unused |
| MAX_POOL_2D | 0 | 0 | Registered but unused |
| PAD | 0 | 0 | Registered but unused |
| PACK | 0 | 0 | Registered but unused |

**Total: 13 unique op types used, 55 operations in Subgraph 0, 12 operations in Subgraph 1.**

This 55-op count is the **official reference flatbuffer fact** for the audited
`okay_nabu.tflite` file. The repository's default residual-enabled architecture
(`residual_connection = [0, 1, 1, 1]`) may legitimately introduce **3 `ADD` ops**
and therefore produce a **58-op** main subgraph while still remaining
ESPHome-compatible, because `ADD` is part of the registered resolver set.

All ops show `CustomCode: N/A` in flatbuffer analysis — there are zero custom ops.

`DELEGATE` visibility is runtime/delegate-path dependent and not a required architectural op in the static `.tflite` compatibility contract. Static-graph compliance must be validated against flatbuffer ops and the ESPHome-registered resolver set.

---

## Article V — Dual-Subgraph Structure

**Status: IMMUTABLE. VERIFIED FROM FLATBUFFER BINARY PARSING.**

The model contains exactly **2 subgraphs**. Both must be present and intact in the exported `.tflite` file.

**CRITICAL NOTE ON TENSOR COUNTS:** For this project, the canonical tensor counts are the counts observed on the standard TensorFlow / `tf.lite.Interpreter` path used by this repository's export, verification, and evaluation tooling. In that canonical path, okay_nabu presents **94 tensors in Subgraph 0** and **12 tensors in Subgraph 1**.

`ai_edge_litert.Interpreter` may expose an extra runtime scratch/workspace tensor (commonly the `[1, 1, 1, 200]` tensor at index 94) and therefore report **95** tensors for Subgraph 0. That extra tensor is an implementation-specific runtime detail of the `ai_edge_litert` interpreter path. It is **not** the counting convention used by this project, because this project does **not** use `ai_edge_litert` for export or canonical architecture verification.

Rule for this repository:

- If you are documenting or validating the architecture **for this project**, use **94 / 12**.
- If you inspect the same `.tflite` with `ai_edge_litert` and see **95 / 12**, do **not** treat that as an architectural mismatch by itself.
- Never infer a model bug from tensor count alone without first checking which interpreter path produced the count.

### Subgraph Metrics (project-canonical counting convention)

| Metric | Subgraph 0 (Main Inference) | Subgraph 1 (Initialization) |
|---|---|---|
| Tensors | **94** | **12** |
| Operations | **55** | **12** |
| Input tensors | [0] | [0] |
| Output tensors | [93] | [] |

### Subgraph 0: Main Inference Graph

```
Input:  [1, 3, 40]  INT8    (tensor index 0)
├── CALL_ONCE  ──────────────────→  Triggers Subgraph 1 once at first inference
├── VAR_HANDLE × 6  ─────────────→  Bind handles to 6 state variables
├── READ_VARIABLE × 6  ──────────→  Load state buffers from previous step
│     ├── stream   : [1, 2,  1, 40]   (ring buffer before first Conv2D)
│     ├── stream_1 : [1, 4,  1, 32]   (MixConv block 0)
│     ├── stream_2 : [1, 10, 1, 64]   (MixConv block 1)
│     ├── stream_3 : [1, 14, 1, 64]   (MixConv block 2)
│     ├── stream_4 : [1, 22, 1, 64]   (MixConv block 3)
│     └── stream_5 : [1, 5,  1, 64]   (Temporal flatten buffer)
├── RESHAPE  ────────────────────→  Expand input dims [1,3,40] → [1,3,1,40]
├── CONCATENATION  ──────────────→  [old_state | new_input]
├── (inference ops: Conv2D, DepthwiseConv2D, SplitV, FC, Logistic…)
├── STRIDED_SLICE  ──────────────→  Extract new ring buffer state
├── ASSIGN_VARIABLE × 6  ────────→  Write updated state back
├── LOGISTIC  ───────────────────→  Sigmoid activation → int8 internal
├── QUANTIZE  ───────────────────→  Re-quantize int8 sigmoid → uint8 output
└── Output: [1, 1]  UINT8   (tensor index 93)
```

### Subgraph 1: Initialization Graph (invoked once, then dormant)

```
├── 12 operations: 6 × VAR_HANDLE + 6 × ASSIGN_VARIABLE
├── 12 tensors: 6 resource handles (object dtype) + 6 pseudo_qconst initializers (int8)
└── Initializes all 6 state variables from embedded tfl.pseudo_qconst tensors
```

The pseudo_qconst tensors carry the initial state values (typically zeros) with the correct quantization parameters matching the corresponding state variable.

---

## Article VI — Streaming State Variables

**Status: IMMUTABLE. EXACT SHAPES. EXACT COUNT.**

There are exactly **6** streaming state variables. Each stores past temporal context needed for the current inference step. The first five (`stream` through `stream_4`) are convolution-context ring buffers. The sixth (`stream_5`) is a temporal pre-flatten buffer.

### State Variable Shapes (okay_nabu reference)

| Variable | Shape | Bytes | Ring Buffer Holds | Quantization |
|---|---|---|---|---|
| `stream` | `[1, 2, 1, 40]` | 80 | 2 frames before first Conv2D | s=0.101961, z=-128 |
| `stream_1` | `[1, 4, 1, 32]` | 128 | MixConv block 0 context | s=1.27439, z=-128 |
| `stream_2` | `[1, 10, 1, 64]` | 640 | MixConv block 1 context | s=0.0345457, z=-128 |
| `stream_3` | `[1, 14, 1, 64]` | 896 | MixConv block 2 context | s=0.0408709, z=-128 |
| `stream_4` | `[1, 22, 1, 64]` | 1,408 | MixConv block 3 context | s=0.0319873, z=-128 |
| `stream_5` | `[1, 5, 1, 64]` | 320 | Temporal flatten buffer | s=0.0262718, z=-128 |

**Total state memory: 3,472 bytes** (80 + 128 + 640 + 896 + 1408 + 320).

All state variable data payloads (the actual int8 tensors flowing through READ_VARIABLE outputs and ASSIGN_VARIABLE value inputs) are quantized with valid scale/zero_point parameters. The VAR_HANDLE tensors themselves are resource pointers (dtype `object`, shape `[]`) and are not quantized — this is correct and expected.

### Implementation in the Flatbuffer

Each state variable appears in two forms in the flatbuffer:

1. **Resource handle** (Subgraph 0): e.g., tensor 44 `stream/states`, dtype=object, shape=[]. This is the VAR_HANDLE output — a pointer, not data.
2. **Data payload** (Subgraph 0): e.g., tensor 56 `model/stream/concat/ReadVariableOp`, shape=[1,2,1,40], dtype=int8, quantized. This is the READ_VARIABLE output — the actual data flowing through the graph.
3. **Updated state** (Subgraph 0): e.g., tensor 58 `model/stream/strided_slice2`, shape=[1,2,1,40], dtype=int8, quantized. This is the ASSIGN_VARIABLE value input — the new state to store.
4. **Initialization handle** (Subgraph 1): e.g., tensor 10 `stream/states1`, dtype=object. Paired with a `tfl.pseudo_qconst` tensor that carries the initial value.

### Convolution-State Ring Buffer Law

For convolution-derived state variables (`stream` through `stream_4`):

```
buffer_frames = kernel_size - stride
```

This identity is inviolable. The buffer stores exactly the number of past frames needed so that when `stride` new frames arrive, the full kernel window is available.

For `stream_5` (the temporal buffer before flattening), the rule is different:

```
stream_5_frames = pre_flatten_temporal_frames - 1
```

In okay_nabu, the concatenated tensor before flatten is `[1, 6, 1, 64]` and the stored state is `[1, 5, 1, 64]`.

---

## Article VII — Inference Timing and Ring Buffer Math

**Status: IMMUTABLE. DRIVEN BY HARDWARE AUDIO PIPELINE TIMING.**

The ESPHome audio pipeline calls the model on a fixed schedule determined by the stride and feature step size. There is no mechanism to change this schedule from the model side.

| Timing Constant | Value | Derivation |
|---|---|---|
| Feature frame period | 10 ms | `window_step_ms` (v2 models) |
| New frames per inference call | 3 | `stride` (read from input tensor dim[1] at runtime) |
| Inference period | 30 ms | `stride × window_step_ms = 3 × 10` |
| Samples consumed per inference | 480 | `stride × samples_per_hop = 3 × 160` |

### The Stride Constant

```
stride = 3
```

This value appears in four places that must all agree:

1. **Model input shape:** `[1, stride, 40]` = `[1, 3, 40]`
2. **Ring buffer calculations:** `buffer_frames = kernel_size - stride`
3. **Representative dataset generator:** slices spectrograms by `stride` per sample
4. **ESPHome runtime:** reads stride from `input->dims->data[1]` and accumulates that many feature frames before invoking inference

The ESPHome runtime dynamically reads stride from the model's input tensor (streaming_model.cpp line 131: `uint8_t stride = this->interpreter_->input(0)->dims->data[1]`), so the model self-declares its stride. However, changing stride requires simultaneously changing: the input tensor shape, all convolution-derived state variable shapes, the representative dataset generator, and downstream temporal dimensions including `stream_5`.

---

## Article VIII — MixedNet Architecture

**Status: PARAMETERIZABLE WITHIN STRUCTURAL RULES.**

The okay_nabu model uses a MixedNet architecture with mixed depthwise convolutions, based on modified code from Google Research's "Streaming Keyword Spotting on Mobile Devices" paper.

### Repository Default MixedNet Configuration (okay_nabu-compatible, uses SPLIT_V / StridedKeep)

```python
first_conv_filters    = 32
first_conv_kernel_size = 5          # → stream shape [1, 2, 1, 40]  (5-3=2)
stride                = 3           # GLOBAL IMMUTABLE CONSTANT
pointwise_filters     = [64, 64, 64, 64]
mixconv_kernel_sizes  = [[5], [7, 11], [9, 15], [23]]
repeat_in_block       = [1, 1, 1, 1]
residual_connection   = [0, 1, 1, 1]
```

This repository's **default** MixedNet / export / training configuration enables
residual connections on blocks 2-4 (`[0, 1, 1, 1]`). This remains ESPHome-
compatible because `ADD` is registered in the ESPHome op resolver. In practice,
this means the repository-default exported variant may have **58** main-subgraph
ops instead of the official reference model's **55**.

Important distinction:

- The **official reference flatbuffer** is still the source for verified tensor
  names, quantization parameters, and op availability.
- The **repository default architecture** is `[0, 1, 1, 1]`, because that is
  what the executable training/export/config paths in this repository use.
- Therefore, residual defaults do **not** break ESPHome compatibility by
  themselves; they only change the concrete model variant being exported.

### How MixConv Kernel Sizes Map to State Variable Shapes

There are **two different contexts** in the streaming graph, and they must not be mixed up:

1. **The first input-side buffer** (`stream`) sits in front of the initial strided convolution. For that one buffer, the relevant law is:

  ```
  buffer_frames = kernel_size - global_stride
  ```

  Therefore:

  ```
  stream: first_conv_kernel=5, global_stride=3 → 5-3 = 2 → [1, 2, 1, 40]
  ```

2. **The downstream block buffers** (`stream_1` through `stream_4`) sit in front of depthwise convolutions that operate with **internal stride 1** in the already-streaming graph. For these buffers, the relevant law is:

  ```
  buffer_frames = effective_temporal_kernel - 1
  ```

For downstream **single-kernel** blocks (`[5]`, `[23]`), that gives:

```
stream_1: kernel=5,  internal_stride=1  → 5-1  = 4  → [1, 4, 1, 32]
stream_4: kernel=23, internal_stride=1  → 23-1 = 22 → [1, 22, 1, 64]
```

For downstream **dual-kernel** blocks (`[7, 11]`, `[9, 15]`), `SPLIT_V` splits the channel dimension, each half goes through its own depthwise conv with `StridedKeep`, and the buffer must preserve enough context for the **largest** temporal kernel in the pair:

```
stream_2: max(7, 11) - 1 = 10 → [1, 10, 1, 64]
stream_3: max(9, 15) - 1 = 14 → [1, 14, 1, 64]
```

**Do not apply the `kernel - global_stride` rule to `stream_1`, `stream_2`, `stream_3`, or `stream_4`.** That rule applies only to the first input-side buffer `stream`. The downstream block buffers are determined by the internal depthwise-kernel context of the streaming graph, which is stride-1 at that stage.

### Structural Rules (apply to all MixedNet variants)

1. **All temporal convolutions must be wrapped in `stream.Stream`** so streaming conversion can extract ring buffer state variables.
2. **`padding="valid"` on the time axis.** Causal padding would produce different activations in non-streaming vs streaming modes, breaking state consistency.
3. **`use_bias=False` on all Conv2D/DepthwiseConv2D.** Biases are folded into BatchNormalization during export.
4. **Classification head:** exactly one `Dense(1, activation="sigmoid")` as the last layer. The sigmoid maps to [0,1] before being quantized to uint8.
5. **BatchNormalization** must be present after every depthwise/pointwise conv block. It is folded into conv weights during the streaming conversion.
6. **No LSTM, GRU, attention, or recurrent layers.** The ESPHome runtime does not register these ops.
7. **No custom TensorFlow ops, no `tf.py_function`, no `tf.numpy_function`.** TFLite Micro cannot execute Python callbacks.

---

## Article IX — Training Pipeline and EMA Management

**Status: IMPLEMENTATION RULES FOR CHECKPOINT LOADING.**

The training pipeline uses Keras 3 with EMA (Exponential Moving Average) for smoothed weight evaluation and checkpointing. EMA is configured via `training.ema_decay` parameter.

### EMA Weight Usage in Training

When EMA is enabled (default in `max_quality.yaml` with `ema_decay: 0.999`):

1. **During training**: The optimizer maintains two weight sets:
   - **Training weights**: Current batch-updated weights for gradient updates
   - **EMA weights**: Exponentially smoothed weights for evaluation/checkpointing

2. **Before evaluation/checkpointing**: `_swap_to_ema_weights()` is called
   - Calls `optimizer.finalize_variable_values(model.trainable_variables)` to apply EMA smoothing
   - This replaces model weights with EMA-smoothed versions for validation
   - Training weights are saved internally to restore later

3. **After evaluation**: `_restore_training_weights()` is called
   - Restores original (non-EMA) training weights
   - Allows gradient updates to continue on unsmoothed weights

### Checkpoint Files and Their EMA Status

| Checkpoint File Type | When Saved | EMA Weights? | Purpose |
|---------------------|------------|------------|---------|
| `final_weights.weights.h5` | End of training (after EMA swap) | ✅ Yes | Export/inference - has smoothed EMA weights |
| `best_weights.weights.h5` | During training (best model found) | ✅ Yes | Resume training - has EMA weights |
| `checkpoint_step_NNNN.weights.h5` | Periodic checkpoints | ✅ Yes | Recovery/checkpointing - has EMA weights |

### CRITICAL RULE: Final Checkpoint Loading

**Do NOT load `best_weights.weights.h5` at end of training.** 

**Why:**
1. Training is complete - no further gradient updates needed
2. Model already contains correct weights in memory (either training or EMA from last checkpoint)
3. `final_weights.weights.h5` already has EMA-smoothed weights (preferred for inference)
4. Loading `best_weights` after EMA finalize triggers optimizer state mismatch warnings:
   - Saved checkpoint has full optimizer state (momentum + variance for each trainable variable)
   - By end of training, EMA finalize may have cleared/reset optimizer state
   - Keras warning: "Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 92 variables"

**Consequence:** Warning is informational and model weights load correctly, but it indicates unnecessary operation.

**Implementation:**
```python
# At end of training (trainer.py lines 1912-1919):
self._swap_to_ema_weights()  # Swaps in EMA weights
final_path = os.path.join(self.checkpoint_dir, "final_weights.weights.h5")
model.save_weights(final_path)  # Saves EMA-smoothed weights
self._restore_training_weights()  # Restores training weights (cleanup)

# Removed: model.load_weights(self.best_weights_path)
# This line was causing optimizer state warnings without benefit
```

### EMA and Optimizer State Interaction

Keras 3 Adam optimizer with `use_ema=True` maintains:
- **2 base variables**: iteration counter, learning rate
- **N momentum variables**: One per trainable weight (Adam first moment)
- **N variance variables**: One per trainable weight (Adam second moment)
- **Total**: `2 + 2 × num_trainable_variables`

Example with 46 trainable weights (typical MixedNet):
- **During active training**: 94 optimizer variables (2 + 2×46 = 94)
- **After `finalize_variable_values()`**: May reduce to 2 (base vars only)
- **Loading saved checkpoint**: Expecting 94 variables, but optimizer may only have 2 → warning

### Anti-Patterns

1. **Never reload `best_weights` after training completion** - Training is done; final_weights has correct EMA-smoothed weights
2. **Never load checkpoint into finalized optimizer state** - Only reload if resuming training from interruption
3. **Don't recompile optimizer after training** - Optimizer is not used after training completes
4. **Use `final_weights` for export/inference** - Already has smoothed weights that were just validated

---

## Article X — Export and Quantization Pipeline

**Status: IMMUTABLE. THESE FLAGS ARE NOT OPTIONAL.**

The export pipeline has two mandatory stages. The official implementation is in `OHF-Voice/micro-wake-word/microwakeword/utils.py`.

### Stage 1: Non-Streaming → Streaming SavedModel Conversion

```python
# Source: utils.py, function convert_model_saved()
# convert_model_saved() is the outer orchestration function.
# model_to_saved() is the lower-level helper it calls to actually materialize
# the streaming SavedModel.
converted_model = model_to_saved(
    model_non_stream = model,
    config           = config,
    mode             = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
)
```

This materializes the ring buffer state variables from the `stream.Stream` wrappers. Without it, the model has no VAR_HANDLE / READ_VARIABLE / ASSIGN_VARIABLE ops and will fail ESPHome's op registration check.

The SavedModel is exported using `tf.keras.export.ExportArchive` (NOT `model.export()` which causes quantization errors, as noted in the source code comment on line 297).

### Stage 2: Streaming SavedModel → Quantized TFLite

```python
# Source: utils.py, function convert_saved_model_to_tflite(), lines 242-265
converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)
converter.optimizations = {tf.lite.Optimize.DEFAULT}

# MANDATORY: Without this, state variable data payloads remain float32,
# causing Quantize/Dequantize ops around every ReadVariable/AssignVariable.
converter._experimental_variable_quantization = True

# These three lines produce the quantized model for device deployment:
converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.uint8       # UINT8. ALWAYS.
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
```

**CRITICAL PACKAGE NOTE:** The export uses `tf.lite.TFLiteConverter` from the standard `tensorflow` package (NOT `ai_edge_litert`). The `ai_edge_litert` package is used only for inference/testing via its `Interpreter` class. See Article XII for the full dependency map.

### What `_experimental_variable_quantization = True` Does

VAR_HANDLE tensors are resource handles (dtype `object`) — they are pointers, not data, and are never themselves quantized. The actual data payloads flowing through READ_VARIABLE outputs and ASSIGN_VARIABLE value inputs MUST be quantized to int8 with valid quantization parameters. Without `_experimental_variable_quantization = True`, these payload tensors may be emitted as float32, and the TFLite Micro int8-only kernel resolver will fail to find matching kernel implementations. The model may fail to load on device.

---

## Article XI — ESPHome Manifest Contract

**Status: IMMUTABLE. REQUIRED EXACT VALUES FOR DEVICE-CRITICAL FIELDS.**

ESPHome reads the manifest JSON to configure its wake word engine. The v2 JSON schema is documented at `esphome.io/components/micro_wake_word/`.

```json
{
  "type": "micro",
  "wake_word": "<the exact spoken phrase>",
  "author": "<author name>",
  "website": "<project URL>",
  "model": "<filename>.tflite",
  "trained_languages": ["<BCP-47 code>"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "feature_step_size": 10,
    "sliding_window_size": 5,
    "tensor_arena_size": 0,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

### Field Constraints

| Field | Constraint | Consequence of Violation |
|---|---|---|
| `type` | Must be `"micro"` | ESPHome ignores the file entirely |
| `version` | Must be `2` for v2 models | v1 loader path is taken; state variables not handled correctly |
| `wake_word` | Must be the exact spoken phrase | Wrong label shown in Home Assistant UI |
| `model` | Must match the `.tflite` filename exactly | ESPHome cannot locate the model binary |
| `trained_languages` | List of BCP-47 codes (e.g., `["en"]`) | Informational; affects ESPHome language-filter behavior |
| `probability_cutoff` | `0.0`–`1.0` float; detection threshold | Too low → false triggers; too high → missed detections. Stored internally as `uint8(value * 255)`. |
| `feature_step_size` | Must be `10` (ms) for v2 models | Timing desync; model receives stale/skipped frames |
| `sliding_window_size` | Positive integer (typically 5) | Affects false-positive suppression window length |
| `tensor_arena_size` | Must be ≥ actual arena requirement | Runtime OOM crash on device. Set to 0 for auto-resolve. |
| `minimum_esphome_version` | Must be `"2024.7.0"` for v2 models | Older firmware uses wrong op resolver or wrong feature_step_size |

### Tensor Arena Sizing

`tensor_arena_size` is model-dependent. There is no single universal constant. Setting it to `0` lets the framework auto-resolve from the exported TFLite tensor allocation. If using explicit override, always measure on the target device. Underestimating causes silent memory corruption, not a clean error.

---

## Article XII — Package Dependencies and Version Constraints

**Status: VERIFIED FROM `setup.py` AND SOURCE CODE IMPORTS.**

The official OHF-Voice/micro-wake-word project uses two separate packages for two separate purposes:

### Export Pipeline (producing .tflite files)

**Package:** `tensorflow` (≥ 2.16)
**API used:** `tf.lite.TFLiteConverter.from_saved_model()`
**Source file:** `microwakeword/utils.py`, function `convert_saved_model_to_tflite()`

The export pipeline does NOT use `ai_edge_litert` at all. The TFLiteConverter is a standard TensorFlow API. If you are only exporting models and not running inference tests, you do not need `ai_edge_litert`.

### Inference and Testing (reading .tflite files)

**Package:** `ai_edge_litert`
**API used:** `ai_edge_litert.interpreter.Interpreter`
**Source file:** `microwakeword/inference.py`, line 20

The inference/testing module uses `ai_edge_litert` for its TFLite interpreter. This is the Google-recommended replacement for `tf.lite.Interpreter` in TensorFlow 2.16+.

### Complete Dependency List (from setup.py)

```
audiomentations
audio_metadata
datasets
mmap_ninja
numpy
pymicro-features
pyyaml
tensorflow>=2.16
webrtcvad-wheels
ai-edge-litert
```

### Implications for Your Project

If your project uses `tensorflow 2.16.2` with `tf.lite.TFLiteConverter` for export but does NOT have `ai_edge_litert` installed: **your export pipeline is correct and compatible with the official pipeline.** The generated .tflite files will be identical. You only need `ai_edge_litert` if you want to run the official inference test code.

### Tensor Count Discrepancy Explained

Different interpreter implementations may enumerate different numbers of tensors for the same `.tflite` file:

| Source | SG0 Tensors | SG1 Tensors | Why |
|---|---|---|---|
| **This project's canonical convention** (`tf.lite` export / verification path) | **94** | **12** | The convention used by this repository's tooling and docs |
| `ai_edge_litert` Interpreter (some versions) | 95 | 12 | May expose a runtime scratch tensor (index 94, shape [1,1,1,200]) |
| Other interpreter/runtime variants | 94 or 95 | 12 | Implementation-dependent enumeration |

**For this repository, always use the project-canonical 94 / 12 convention unless you are explicitly documenting an `ai_edge_litert` runtime observation.** Do not treat the extra `ai_edge_litert` scratch tensor as a model-architecture difference.

---

## Article XIII — v1 vs v2 Model Differences

**Status: REFERENCE. DERIVED FROM ESPHOME MODEL REPOSITORY AND RELEASE NOTES.**

The ESPHome micro-wake-word-models repository (`github.com/esphome/micro-wake-word-models`) hosts models in two directories: `models/` (v1) and `models/v2/` (v2). This project targets v2.

| Property | v1 | v2 |
|---|---|---|
| `feature_step_size` | 20 ms | **10 ms** |
| JSON manifest `version` | 1 | **2** |
| `minimum_esphome_version` | older | **2024.7.0** |
| Inference period | 60 ms (stride 3 × 20ms) | **30 ms** (stride 3 × 10ms) |
| Temporal resolution | Lower (20ms frames) | **Higher (10ms frames)** |
| Model architecture | Same MixedNet | Same MixedNet |
| Accuracy | Lower | **Higher** (faster, more temporal detail) |

The key v2 improvement is halving the feature step size from 20ms to 10ms, which doubles the temporal resolution of the spectrogram features without changing the model architecture itself. The same stride=3 is used, so the model still processes 3 frames per inference call, but each frame represents 10ms instead of 20ms.

v2 models will NOT work on ESPHome firmware older than 2024.7.0 because older firmware hardcodes a 20ms step size.

---

## Violation Consequence Matrix

This table exists so that developers and AI agents understand exactly what breaks — on real hardware, in production — when each rule is violated. "Works in Python" is NOT a definition of correctness for this project.

| Article | What You Might Change | What Actually Breaks |
|---|---|---|
| I | `mel_bins`, `window_step_ms`, `sample_rate_hz` | Input tensor shape mismatch; model receives wrong feature dimensions; predictions are garbage |
| II | Output dtype from `uint8` to `int8` | ESPHome reads signed bytes as unsigned; every prediction ≥ 128 is misinterpreted; wake word never triggers or always triggers |
| II | Input dtype from `int8` to `float32` | `load_model_()` fails with "Streaming model tensor input is not int8"; model never loads |
| III | Calibration dataset too small or missing boundary anchors | Scale/zero_point shift; dynamic range is wrong; model runs but predictions are compressed into a tiny range; effectively non-functional |
| III | Remove `_experimental_variable_quantization` | State payload tensors may remain float32; TFLite Micro int8-only kernel resolver fails at load time; device halts |
| IV | Use any op outside the 20 registered resolvers | Op resolver returns `kTfLiteError` at load time; device halts; wake word engine never starts |
| V | Export without `STREAM_INTERNAL_STATE_INFERENCE` mode | No VAR_HANDLE/state ops in graph; model has no memory; predictions are independent per-frame; accuracy is random |
| V | Missing or corrupted Subgraph 1 | State variables not initialized at boot; undefined initial state; unpredictable behavior from first inference |
| VI | Wrong ring buffer size (kernel/stride mismatch) | Ring buffer reads from wrong temporal offset; model sees scrambled temporal context; no crash, just permanently wrong predictions |
| VII | Change stride in code but not in export/manifest | Input tensor slicing misaligned with ring buffer writes; state corruption accumulates across time; model degrades after first second of audio |
| VIII | Add LSTM/GRU/attention or custom ops | Op not registered; device halts at model load |
| IX | `inference_output_type = tf.int8` instead of `tf.uint8` | Output read as signed; probabilities inverted relative to ESPHome's uint8 comparison threshold |
| IX | Use `ai_edge_litert` converter instead of `tf.lite.TFLiteConverter` | Potentially different quantization behavior or op selection; untested path; model may work or may produce subtly different results |
| X | `feature_step_size ≠ 10` in v2 manifest | ESPHome feeds frames at wrong cadence; ring buffers contain wrong number of frames; temporal context is 1.5× or 2× too long/short |
| X | `version ≠ 2` in v2 manifest | v1 loader path is taken; state variables may not be handled correctly |
| XI | Use wrong TensorFlow version (<2.16) | `_experimental_variable_quantization` may not be available; `ExportArchive` API may differ; export may fail or produce incompatible model |

---

## APPENDIX A — Quick Reference for Common Tasks

### "I need to verify a .tflite file is v2 compliant"

Check these four things:

1. Input shape is `[1, 3, 40]` and dtype is `int8`
2. Output shape is `[1, 1]` and dtype is `uint8`
3. Model has exactly 2 subgraphs
4. All ops in both subgraphs are within the 20-op allowed set

### "I need to export a trained model"

Follow the two-stage pipeline in Article X exactly. Use `tf.lite.TFLiteConverter` from standard `tensorflow`. Do NOT use `ai_edge_litert` for export.

### "I need to count tensors"

For this repository's tooling and documentation, the canonical count for okay_nabu is **94 tensors in Subgraph 0** and **12 in Subgraph 1**.

If `ai_edge_litert` shows **95** tensors in Subgraph 0, that is expected and is usually an exposed runtime scratch tensor rather than a different model architecture.

### "I need to change the wake word"

Only training data and the manifest JSON `wake_word` field change. The model architecture, I/O contract, quantization parameters, and all timing constants remain identical.

### "I need to change the model architecture"

You may change `mixconv_kernel_sizes`, `pointwise_filters`, `repeat_in_block`, and `residual_connection` within the constraints of Article VIII. After export, verify the new model against Article IV (all ops must be in the allowed set) and Article VI (state variable shapes must follow the ring buffer law). The state variable count will change if the number of Stream-wrapped layers changes.

---

*This document is append-only with respect to verified facts. No value may be changed without re-running flatbuffer extraction against the reference model and cross-verifying against current ESPHome source code. Adding unverified values is prohibited — mark any new entry as `UNVERIFIED` and include the expected verification method. Last full verification: 2026-03-13.*
