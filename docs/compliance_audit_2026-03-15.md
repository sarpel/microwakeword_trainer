# microwakeword_trainer — Architectural Compliance Audit
**Date:** 2026-03-15  
**Branch:** `finishing`  
**Constitution version:** 2026-03-13  
**Test suite:** 479/479 passing

---

## Executive Summary

The codebase is **substantially compliant** with the ARCHITECTURAL_CONSTITUTION.md. All export-path invariants that produce the actual `.tflite` binary are correctly implemented. The ESPHome op-set list in the constitution matches the actual C++ source. Two noteworthy items were identified that do not represent architectural violations but are worth documenting: (1) config presets use intentionally different class weights from AGENTS.md defaults — this is a training quality parameter, not an ESPHome compatibility issue; (2) presets default to `spec_augment_backend: "tf"` rather than `"cupy"`, which silently skips GPU SpecAugment when `spec_augment_backend != "tf"` but `cupy` is not installed — however, the TF backend IS a fully functional SpecAugment implementation, not a disabled fallback.

**No constitutional violations found that would produce ESPHome-incompatible models.**

---

## Audit Scope

Four independent agents audited the following areas in parallel:

| Area | Agent | Files Audited |
|------|-------|--------------|
| Export pipeline | `explore` | `tflite.py`, `verification.py`, `manifest.py` |
| Model architecture | `explore` | `streaming.py`, `architecture.py` |
| Config & training | `explore` | All 3 preset YAMLs, `loader.py`, `trainer.py`, `mining.py` |
| ESPHome source | `librarian` | ESPHome GitHub `streaming_model.cpp`, `micro_wake_word.cpp` |

Cross-referenced against:
- `ARCHITECTURAL_CONSTITUTION.md` (primary)
- `docs/OFFICIAL_MODELS_ARCH_ANALYSIS.md`
- ESPHome `esphome/esphome` GitHub repository (live, 2025.12.5)

---

## Section 1 — Export Pipeline (tflite.py, verification.py, manifest.py)

**Verdict: FULLY COMPLIANT ✅**

| Invariant | Location | Status |
|-----------|----------|--------|
| `_experimental_variable_quantization = True` | `tflite.py:1195-1198` | ✅ Compliant |
| `inference_input_type = tf.int8` | `tflite.py:1196` | ✅ Compliant |
| `inference_output_type = tf.uint8` | `tflite.py:1197` | ✅ Compliant |
| `tf.keras.export.ExportArchive` (not `model.export()`) | `tflite.py:1166-1185` | ✅ Compliant |
| `STREAM_INTERNAL_STATE_INFERENCE` mode | `tflite.py:1166-1185` | ✅ Compliant |
| Boundary anchors 0.0 and 26.0 in `create_representative_dataset_from_data()` | `tflite.py:788-792` | ✅ Compliant |
| Boundary anchors 0.0 and 26.0 in `create_representative_dataset_from_noise()` | `tflite.py:596-600` | ✅ Compliant |
| Representative dataset ≥500 samples | `tflite.py` (1000-4000+ range) | ✅ Compliant |
| Verification: exactly 2 subgraphs | `verification.py:362-365` | ✅ Compliant |
| Verification: int8 input | `verification.py:116-119` | ✅ Compliant |
| Verification: uint8 output | `verification.py:131-134` | ✅ Compliant |
| Verification: exactly 6 state variables (VAR_HANDLE count) | `verification.py:225-228` | ✅ Compliant |
| Op whitelist (exact ESPHome 20-op set) | `verification.py:191-212` | ✅ Compliant |
| Manifest `version: 2` | `manifest.py:56, 193-195` | ✅ Compliant |
| Manifest `feature_step_size: 10` | `manifest.py:60, 229-231` | ✅ Compliant |
| Manifest `minimum_esphome_version: "2024.7.0"` | `manifest.py:62, 233-235` | ✅ Compliant |

---

## Section 2 — Model Architecture (streaming.py, architecture.py)

**Verdict: FULLY COMPLIANT ✅**

| Invariant | Location | Status |
|-----------|----------|--------|
| Ring buffer law: first conv `buffer = kernel_size - stride` (5-3=2) | `streaming.py:241-255` | ✅ Compliant |
| Ring buffer law: downstream blocks `buffer = kernel_size - 1` (e.g., 5-1=4, 11-1=10, 15-1=14, 23-1=22) | `streaming.py:241-255` | ✅ Compliant |
| State shapes match constitution: stream[1,2,1,40], stream_1[1,4,1,32], stream_2[1,10,1,64], stream_3[1,14,1,64], stream_4[1,22,1,64] | `streaming.py` ring buffer math | ✅ Compliant |
| stream_5 shape uses `temporal_frames - 1` (config-derived, not hardcoded) | `streaming.py` | ✅ Compliant |
| `STREAM_INTERNAL_STATE_INFERENCE` used in `Stream.__init__` | `streaming.py:275` | ✅ Compliant |
| `hasattr(self.model, 'layers')` guard in `StreamingMixedNet.reset()` | `streaming.py:721-727` | ✅ Compliant |
| `stride=3` on time axis only; freq axis remains stride=1 | `architecture.py:431, 176, 162, 323` | ✅ Compliant |
| No `padding='same'` on time axis; causal padding applied manually | `architecture.py:206-207, 224-225` | ✅ Compliant |
| State variable names: `'stream'`, `'stream_1'`…`'stream_5'` | `streaming.py:658-661` | ✅ Compliant |
| `build_model()` maps `"stream_internal"` → `STREAM_INTERNAL_STATE_INFERENCE` | `architecture.py:791-803` | ✅ Compliant |
| Mode propagates to all layers | `architecture.py:630, 637-639` | ✅ Compliant |

---

## Section 3 — Configuration & Training Pipeline

**Verdict: COMPLIANT with notes ✅⚠️**

### Hardware Invariants (ESPHome-critical)
| Invariant | All 3 Presets | loader.py Default | Status |
|-----------|--------------|------------------|--------|
| `mel_bins: 40` | ✅ | `HardwareConfig.mel_bins=40` | ✅ Compliant |
| `window_step_ms: 10` (feature_step_size) | ✅ | `HardwareConfig.window_step_ms=10` | ✅ Compliant |
| `stride: 3` | ✅ | `ModelConfig.stride=3` | ✅ Compliant |
| `search_eval_fraction: 0.30` | ✅ | `AutoTuningConfig.search_eval_fraction=0.30` | ✅ Compliant |

### Training Quality Parameters (not ESPHome-critical)

#### Class Weights — Note (not a constitutional violation)
The `AGENTS.md` documents default class weights of `positive=1.0, negative=20.0, hard_neg=40.0`. The presets intentionally override these:

| Preset | Phase weights (pos / neg / hard_neg) |
|--------|--------------------------------------|
| `standard.yaml` | `[10.0,20.0,30.0]` / `[1.0,1.0,1.0]` / `[1.0,1.0,1.0]` |
| `fast_test.yaml` | Same as standard |
| `max_quality.yaml` | `[3.0,4.0,6.0]` / `[2.0,4.0,6.0]` / `[3.0,5.0,7.0]` |

**Assessment:** The ARCHITECTURAL_CONSTITUTION.md does not mandate class weight values — these are training quality parameters. The presets apply per-phase progressive weighting (varying across Phase 1/2/3), which is a valid training strategy. No constitutional violation. The AGENTS.md "defaults" are reference values, not architectural invariants.

#### SpecAugment Backend — Note (not a violation)
All three presets set `spec_augment_backend: "tf"` (line 161/158/156). This routes through `spec_augment_tf.py` which is a **fully functional TensorFlow-native SpecAugment implementation** — not a disabled fallback.

When `spec_augment_backend == "tf"`, augmentation is applied in-pipeline via `tfdata_pipeline.py` (lines 209-263). When `spec_augment_backend != "tf"`, it uses the CuPy GPU path (trainer.py lines 1751-1778) with a silent-skip on RuntimeError.

**Assessment:** Presets choosing `"tf"` over `"cupy"` is a valid configuration choice. The `"tf"` path is not a degraded fallback — it runs SpecAugment inside the tf.data graph for better pipeline throughput. The README's claim that "CuPy SpecAugment has no CPU fallback" refers to the CuPy path specifically; the TF path runs independently.

**Legitimate concern:** The trainer's CuPy path silently skips SpecAugment on RuntimeError (lines 1775-1778) without failing loudly. If someone switches to `"cupy"` but CuPy isn't installed, training proceeds silently without augmentation. Consider whether this should be a hard error. Currently not a constitutional violation.

### Two-Phase Training
- **Phase 2 with hard negatives**: ✅ Compliant — Phase 2 logic correctly uses hard negatives from `AsyncHardExampleMiner`.
- **`HardExampleMiner` separation**: ✅ Compliant — Separates hard negatives by `fp_threshold`, weights applied separately in trainer.

---

## Section 4 — ESPHome Source Cross-Reference

**Verdict: CONSTITUTION IS ACCURATE ✅ (with one documentation clarification)**

### Op Set Verification

Cross-referencing against ESPHome `streaming_model.cpp` lines 269-308 (ESPHome 2025.12.5):

| Op | In ESPHome C++ | In ARCHITECTURAL_CONSTITUTION.md | In verification.py allowed_ops |
|----|---------------|-----------------------------------|-------------------------------|
| CALL_ONCE | ✅ | ✅ | ✅ |
| VAR_HANDLE | ✅ | ✅ | ✅ |
| RESHAPE | ✅ | ✅ | ✅ |
| READ_VARIABLE | ✅ | ✅ | ✅ |
| STRIDED_SLICE | ✅ | ✅ | ✅ |
| CONCATENATION | ✅ | ✅ | ✅ |
| ASSIGN_VARIABLE | ✅ | ✅ | ✅ |
| CONV_2D | ✅ | ✅ | ✅ |
| MUL | ✅ | ✅ | ✅ |
| ADD | ✅ | ✅ | ✅ |
| MEAN | ✅ | ✅ | ✅ |
| FULLY_CONNECTED | ✅ | ✅ | ✅ |
| LOGISTIC | ✅ | ✅ | ✅ |
| QUANTIZE | ✅ | ✅ | ✅ |
| DEPTHWISE_CONV_2D | ✅ | ✅ | ✅ |
| AVERAGE_POOL_2D | ✅ | ✅ | ✅ |
| MAX_POOL_2D | ✅ | ✅ | ✅ |
| PAD | ✅ | ✅ | ✅ |
| PACK | ✅ | ✅ | ✅ |
| SPLIT_V | ✅ | ✅ | ✅ |

**All 20 ops match exactly.** The constitution's Article IV is accurate against the live ESPHome source.

Note: An earlier draft of the constitution listed different ops (CAST, DEQUANTIZE, RELU, RELU_6, SHAPE, SPLIT). Those were incorrect. The current constitution has been updated with the correct set.

### Quantization Parameter Enforcement

ESPHome does **not** perform model-level scale/zero_point checks at load time. Instead it hard-codes feature→input mapping as:
```
input_value = (feature * 256) / 666 - 128
```
Where 666 ≈ 25.6 × 26.0, effectively enforcing `scale ≈ 0.1016, zp = -128`.

Output: ESPHome divides `uint8` output by `255.0f` to get probability, enforcing `scale = 1/255 ≈ 0.00392, zp = 0`.

**Assessment:** ESPHome's enforced input scale `≈ 0.1016` is close to but not identical to our quantization parameter `0.10196078568696976`. This is expected — the C++ approximation uses integer arithmetic (`/666`) while our quantization uses the precise float scale. The mismatch (`0.1016` vs `0.10196`) is ~0.04% and is inherent in the ESPHome feature pipeline, not a bug in our trainer. Official ESPHome models (okay_nabu etc.) have the same discrepancy.

### Subgraph and State Variable Constraints

- **Subgraph count**: ESPHome has **no explicit check** for exactly 2 subgraphs. The 2-subgraph structure is needed for correct `CALL_ONCE` initialization behavior, but ESPHome won't reject a different count — it just won't work correctly without it. Our verification check (enforcing 2 subgraphs) is **stricter than ESPHome** and is the correct approach.
- **State variables**: ESPHome creates `MicroResourceVariables` with hard-coded capacity of **20**. Models exceeding 20 state variables will fail to initialize. Our 6-variable models are well within this limit.

### ESP_NN Kernel Optimizations

ESPHome uses `espressif/esp-tflite-micro` with `ESP_NN` optimizations for `CONV_2D` and `DEPTHWISE_CONV_2D`. These kernels have stricter alignment requirements than standard TFLite Micro on non-ESP platforms. **This does not affect model generation** — alignment is handled by the ESP32 runtime, not by our TFLite export. No action required.

### Streaming Stride Accumulation

ESPHome's `StreamingModel` implements stride-accumulation: if the model's input tensor has a `stride > 1` second dimension, features are accumulated over multiple loop iterations before `Invoke()` is called. Our models use `stride=3`, so ESPHome calls inference every 3 feature frames (30ms). This matches our design.

### Recent ESPHome Changes (Post-2024.7.0)

- **PR #12652 (Open)**: Proposes adding `EXPAND_DIMS` op. Not yet merged; currently unsupported. **Our models do not use EXPAND_DIMS.** No action required.
- **PR #11698 (Merged)**: Added `wake_loop_threadsafe()` latency improvement. No model format impact.
- **PR #13714 (Merged)**: Trigger heap allocation refactor. No model format impact.

---

## Summary of Findings

| Category | Finding | Severity | Action Required |
|----------|---------|----------|----------------|
| Export pipeline | All converter flags, dtypes, ExportArchive, boundary anchors correct | ✅ None | None |
| Model architecture | Ring buffer law, state shapes, padding, strides, mode all correct | ✅ None | None |
| Config hardware invariants | mel_bins=40, stride=3, feature_step_size=10 consistent across all 3 presets | ✅ None | None |
| ESPHome op set | Constitution Article IV matches actual C++ source exactly | ✅ None | None |
| ESPHome quantization | ESPHome uses approximated scale (~0.1016 vs 0.10196) but this is inherent to official ESPHome design | ℹ️ Info | None — same as official models |
| ESPHome subgraph check | ESPHome has no hard subgraph count check; our stricter verification is correct | ℹ️ Info | None |
| Class weights in presets | Presets use per-phase progressive weights, different from AGENTS.md reference defaults | ℹ️ Info | None — not a constitutional constraint |
| CuPy SpecAugment silent-skip | `spec_augment_backend != "tf"` path silently skips augmentation on RuntimeError | ⚠️ Risk | Consider hard-failing if cupy backend requested but unavailable |
| `spec_augment_backend: "tf"` in presets | All presets use TF backend; CuPy path not exercised by default | ℹ️ Info | None — TF backend is fully functional |

---

## Recommendation

**The codebase is safe to export from.** Any trained model that passes `python scripts/verify_esphome.py` will be ESPHome-compatible.

One optional improvement: in `trainer.py` lines 1774-1778, change the `except RuntimeError` silent-skip to a hard `raise` when `spec_augment_backend == "cupy"`. This prevents silent training quality degradation if someone configures CuPy SpecAugment but CuPy isn't installed.
