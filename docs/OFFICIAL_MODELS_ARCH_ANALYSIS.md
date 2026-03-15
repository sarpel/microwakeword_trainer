# Official micro-wake-word Models Deep Architecture Analysis

Analyzed folder: `official_models`

Models analyzed:

- `okay_nabu.tflite`
- `alexa.tflite`
- `hey_jarvis.tflite`
- `hey_mycroft.tflite`

This report is based on **independent extraction**, not only one helper script:

1. TensorFlow Lite Interpreter scans (tensor/op/quant/state extraction)
2. ai_edge_litert Interpreter scans with equivalent resolver mode
3. TensorFlow Analyzer text capture for all 4 models
4. ai_edge_litert native analyzer wrapper capture
5. Cross-model comparison artifacts generated during this run:
   - `_official_models_quick_scan.json`
   - `_official_models_arch_signature.json`
   - `_official_models_comparison.json`
   - `_official_models_from_tf_analyzer.json`
   - `_official_models_static_vs_configurable.json`
   - `_official_models_dual_runtime_scan.json`

## Re-validation update (after removing `needalighter.py`)

- `official_models/needalighter.py` was removed and is not part of this analysis path.
- All 4 official models were re-validated directly via TensorFlow Lite + ai_edge_litert interpreter APIs and both analyzer backends.
- Re-validation confirms:
  - 2-subgraph streaming structure (confirmed via ai_edge_litert Interpreter and both analyzer text outputs)
  - stable I/O contracts (`[1,3,40] int8` input, `[1,1] uint8` output)
  - output quantization `scale=0.00390625`, `zero_point=0` across all four
  - consistent state-op envelope (`CALL_ONCE`, `VAR_HANDLE`, `READ_VARIABLE`, `ASSIGN_VARIABLE`)
- `DELEGATE` does **not** appear when both runtimes are evaluated with `BUILTIN_WITHOUT_DEFAULT_DELEGATES`; this confirms prior `DELEGATE` sightings are delegate/tooling artifacts, not a required model op in the compatibility contract.

---

## 1) What is absolutely static across all 4 official models

These are true invariants in the official set:

- **Input tensor contract**: shape `[1, 3, 40]`, dtype `int8`
- **Input quantization**: scale `0.10196078568696976`, zero-point `-128`
- **Output tensor contract**: shape `[1, 1]`, dtype `uint8`
- **Output quantization**: scale `0.00390625` (= `1/256`), zero-point `0`
- **Subgraph structure**: exactly **2 subgraphs** (main inference + init)
- **Streaming state mechanism**:
  - `CALL_ONCE` + `VAR_HANDLE` + `READ_VARIABLE` + `ASSIGN_VARIABLE`
  - **6 resource state variables** initialized in subgraph 1
- **Core op family present in all** (SG0 intersection):
  - `ASSIGN_VARIABLE`, `CALL_ONCE`, `CONCATENATION`, `CONV_2D`, `DEPTHWISE_CONV_2D`,
  - `FULLY_CONNECTED`, `LOGISTIC`, `QUANTIZE`, `READ_VARIABLE`, `RESHAPE`, `STRIDED_SLICE`, `VAR_HANDLE`

Implication: these define the true runtime compatibility contract for official-style MWW v2 models.

---

## 2) What is configurable (officially variable)

### Per-model size and complexity

- `hey_jarvis`: ~51.0 KB, 46 SG0 ops (smallest)
- `alexa`: ~54.5 KB, 49 SG0 ops
- `hey_mycroft`: ~55.9 KB, 49 SG0 ops
- `okay_nabu`: ~58.9 KB, 58 SG0 ops (largest/most complex)

### Channel width family

Two channel families exist:

- **32/64 family**: `okay_nabu`, `alexa`, `hey_mycroft`
- **30/60 family**: `hey_jarvis`

This appears in state payload shapes and conv/depthwise kernel shapes.

### Temporal state depths (stream buffers)

State windows differ by model:

- `okay_nabu`: includes `2,4,10,14,22` (+ final state depth 5)
- `alexa`: `2,4,8,12,20` (+ final state depth 4)
- `hey_jarvis`: `2,4,8,12,20` (+ final state depth 4, but channels 30/60)
- `hey_mycroft`: `2,4,8,12,20` and one deeper branch to `14` (+ final depth 14)

This is the strongest evidence that **temporal receptive field and internal stream memory depth are configurable**.

### Flattened dense input width (critical)

- `okay_nabu`: 384
- `alexa`: 320
- `hey_jarvis`: 300
- `hey_mycroft`: 960

This is a major architecture degree of freedom and confirms that fixed assumptions like “always 384” are wrong.

### Operator-level variation

- `SPLIT_V` appears only in `okay_nabu` (with extra concat/depthwise/strided-slice counts)
- Other models do not require `SPLIT_V`

This indicates multiple official architecture variants inside the same compatibility envelope.

---

## 3) New high-signal findings (important unknowns now resolved)

1. **Official set is not a single architecture instance.**
   It is a family of compatible architectures sharing strict IO/state contracts.

2. **Output quantization in official models is consistently `1/256`** (not `1/255`) in this set.

3. **“Final stream state depth” is not fixed to 5.**
   It is 5 (okay_nabu), 4 (alexa/jarvis), and 14 (hey_mycroft).

4. **Dense input width is highly variable (300/320/384/960)** and tightly coupled to temporal/channel design.

5. **`SPLIT_V` is optional in official models** (required by one model, absent in others).

6. **Two-channel-width families (30/60 vs 32/64) are both official.**

7. **`DELEGATE` is runtime/delegate-path dependent**, not a required architectural op. Under `BUILTIN_WITHOUT_DEFAULT_DELEGATES` in both TensorFlow Lite and ai_edge_litert scans, `DELEGATE` is absent while all compatibility invariants remain satisfied.

---

## 4) Static vs configurable summary for our trainer/export design

### Must remain static (hard contract)

- Input: `[1, 3, 40] int8`
- Output: `[1, 1] uint8`
- 2-subgraph pattern with state init path
- Resource-variable streaming state graph (`CALL_ONCE`, `VAR_HANDLE`, `READ_VARIABLE`, `ASSIGN_VARIABLE`)
- Core SG0 op compatibility with ESPHome runtime resolver

### Must be treated as configurable (do not hardcode)

- Intermediate channel widths (`30/60` or `32/64` etc.)
- State payload depths beyond the fixed stream stem (`2,4,...`) and especially final stream depth
- Dense input width
- Presence/absence of split/branch operators like `SPLIT_V`
- Total op counts/tensor counts/model size

---

## 5) Guardrails we should keep/enforce from this analysis

1. Never hardcode one official state-shape set in validation.
2. Never assume dense input feature width is fixed.
3. Keep output quantization check compatible with observed official scale `1/256`.
4. Keep architecture checks focused on compatibility envelope, not one reference topology.
5. Treat `SPLIT_V` as optional, but ensure resolver support remains available.

---

## 6) Per-model quick cards

- **okay_nabu**
  - 58.9 KB, SG0 ops=58
  - 32/64 channels
  - deeper split-heavy topology (`SPLIT_V` present)
  - dense flatten width 384

- **alexa**
  - 54.5 KB, SG0 ops=49
  - 32/64 channels
  - no `SPLIT_V`
  - dense flatten width 320

- **hey_jarvis**
  - 51.0 KB, SG0 ops=46
  - 30/60 channels
  - no `SPLIT_V`
  - dense flatten width 300

- **hey_mycroft**
  - 55.9 KB, SG0 ops=49
  - 32/64 channels
  - no `SPLIT_V`
  - dense flatten width 960 (largest temporal*channel head)

---

## 7) Bottom line

The official models prove that **compatibility is defined by a strict runtime interface + streaming-state mechanism**, while substantial internal architectural variation is allowed.

For alignment, we should optimize for:

- strict interface/state invariants,
- config-driven architecture generation,
- and validation that checks compatibility envelope rather than forcing one exact topology.
