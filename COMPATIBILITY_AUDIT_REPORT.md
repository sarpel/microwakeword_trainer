# Compatibility Audit Report

This report compares this repository against the practical ground truth used to produce and run ESPHome-compatible micro wake word models:

- OHF-Voice/micro-wake-word
- TensorFlow Lite Micro / TensorFlow Lite export behavior
- ESPHome `micro_wake_word` runtime contracts

The goal is not to score the project generally. The goal is to identify places where this repository diverges from the official path or bakes in assumptions that can break alignment.

## Ground-truth contracts that matter

These requirements are consistently supported by the external sources and by this repo's own architectural constitution:

- Input tensor must be `int8` with shape `[1, stride, 40]`; ESPHome expects `[1, 3, 40]` for the v2-style models used here.
- Output tensor must be `uint8` with shape `[1, 1]`.
- The streaming model must use resource-variable based state (`VAR_HANDLE`, `READ_VARIABLE`, `ASSIGN_VARIABLE`) and stay within ESPHome's registered op set.
- Frontend assumptions are fixed: 16 kHz, 40 bins, 30 ms window, 10 ms feature step, PCAN enabled.
- Official export flow uses `ExportArchive` + TFLite conversion + representative dataset calibration.

## Confirmed problems

### 1. `scripts/verify_esphome.py` hardcodes okay_nabu state payload shapes

**Severity:** High

`scripts/verify_esphome.py` strict mode hardcodes these six expected payload shapes:

- `(1, 2, 1, 40)`
- `(1, 4, 1, 32)`
- `(1, 10, 1, 64)`
- `(1, 14, 1, 64)`
- `(1, 22, 1, 64)`
- `(1, 5, 1, 64)`

That last shape assumes the exact okay_nabu-style temporal layout. But this repo already knows `stream_5` is config-dependent and should be derived from `temporal_frames`; `src/export/verification.py` exposes `compute_expected_state_shapes(...)` for exactly that reason.

**Why this is a real alignment problem:** a correctly exported model can be rejected by the strict verifier if `clip_duration_ms` / inferred `temporal_frames` differs from the narrow reference case.

**Evidence:**

- `scripts/verify_esphome.py`
- `src/export/verification.py`
- `src/export/AGENTS.md`
- `docs/EXPORT.md`

### 2. Output quantization verification is internally inconsistent

**Severity:** High

`src/export/verification.py` currently expects output quantization scale `1/255`.

But this repository's own audited reference material states the v2 okay_nabu reference flatbuffer uses output scale `0.00390625`, i.e. `1/256`:

- `ARCHITECTURAL_CONSTITUTION.md`
- `docs/ARCHITECTURE.md`
- `docs/KNOWLEDGE_BASE.md`

So the verifier is out of sync with the project's own reference truth.

**Why this is a real alignment problem:** the verifier can produce false failures or push future edits toward the wrong quantization expectation.

**Evidence:**

- `src/export/verification.py` expects `1.0 / 255.0`
- `ARCHITECTURAL_CONSTITUTION.md` documents `0.00390625 (= 1/256)`

### 3. Autotuner still serializes / perturbs only `trainable_weights`

**Severity:** High

The project documentation repeatedly states that weight serialization must use `model.get_weights()` / `model.set_weights()` so BatchNorm moving statistics are preserved. However `src/tuning/autotuner.py` still uses `model.trainable_weights` in multiple places:

- SWA snapshot collection
- SWA restore
- stochastic perturbation loops

`trainable_weights` excludes non-trainable BatchNorm state.

**Why this is a real alignment problem:** post-training tuning can silently drift away from the real inference model state, especially for exported streaming models where BN state correctness matters.

**Evidence:**

- `src/tuning/autotuner.py` around lines ~1800, ~1891, ~2166, ~2193
- `src/tuning/AGENTS.md`
- `docs/TROUBLESHOOTING.md`

### 4. Export can fall back to random-noise representative data instead of real training features

**Severity:** High

The official OHF exporter calibrates with real training spectrograms and explicitly injects boundary anchors. This repo supports a stronger real-data path too, but it also allows fallback to random synthetic calibration samples when `data_dir` is unavailable or unreadable.

`src/export/tflite.py` explicitly warns that this fallback can miscalibrate output range and cap the model around `~0.5` behavior.

**Why this is a real alignment problem:** this is not the official path, and it risks producing quantized models that are structurally loadable but numerically worse on device.

**Evidence:**

- OHF `microwakeword/utils.py` uses real training spectrograms
- `src/export/tflite.py::create_representative_dataset()` marks random calibration as fallback-only
- CLI supports `--data-dir` for real calibration, but the fallback still exists

## Confirmed divergences from the official export path

These are real divergences from the upstream/reference path, but I cannot prove from source alone that they always break runtime compatibility.

### 5. Local exporter sets `converter.experimental_enable_resource_variables = True`; OHF reference exporter does not

**Severity:** Medium

The OHF `microwakeword/utils.py` exporter uses:

- `Optimize.DEFAULT`
- `_experimental_variable_quantization = True`
- `inference_input_type = tf.int8`
- `inference_output_type = tf.uint8`

It does **not** set `experimental_enable_resource_variables` in the fetched reference exporter.

This repo sets that flag in both quantized and non-quantized export branches.

**Why this matters:** it is a real deviation from the official exporter and should be justified with flatbuffer-level comparison, not assumption.

**Current conclusion:** divergence confirmed, breakage not yet proven.

**Evidence:**

- OHF `microwakeword/utils.py`
- `src/export/tflite.py`

### 6. Default model presets favor a residual-enabled variant rather than upstream MixedNet defaults

**Severity:** Medium

The upstream OHF `mixednet.py` default parser uses no residuals by default. This repository's presets default to:

- `residual_connection: "0,1,1,1"`

That means the default exported model is not the plain upstream MixedNet default.

**Why this matters:** it changes graph structure/op count and means this project is not following the simplest official baseline by default.

**Current conclusion:** divergence confirmed; runtime incompatibility not proven because `ADD` is in ESPHome's registered op set.

**Evidence:**

- OHF `microwakeword/mixednet.py`
- `config/presets/standard.yaml`
- `config/presets/fast_test.yaml`
- `config/presets/max_quality.yaml`

### 6b. Evaluation logic does not model ESPHome's sum-based sliding-window detection

**Severity:** High

The latest actual `plan` task output highlighted a real risk around deployment semantics. After validating it against source, this finding holds: this repository's evaluation code scores operating thresholds directly on per-window scores, while ESPHome's runtime performs detection using a **sum over the recent probability window**:

- ESPHome detects when `sum > probability_cutoff * sliding_window_size`
- this repo's evaluation utilities sweep thresholds using direct comparisons like `y_scores >= thresh`

That means FAH/recall tuning in this repository is not modeling the exact runtime decision rule used by ESPHome.

**Why this matters:** even if the exported model is structurally compatible, the chosen thresholds and reported recall/false-alarm characteristics can be systematically misaligned with real device behavior.

**Evidence:**

- ESPHome `streaming_model.cpp` uses sum-based sliding-window detection
- `src/evaluation/metrics.py` computes threshold metrics from direct per-score comparisons rather than runtime-style sliding-window accumulation
- actual `plan` task session `ses_31758255effe4TjYWRW4EweSsG`

## Additional findings from the concurrent plan-executor task

These findings came from the latest concurrent plan/implementation session during this audit. They are **not as direct as the ESPHome export/runtime contract problems above**, but they are still important because they affect whether the platform can reliably train and validate models using the intended workflow.

### 7. Validation cloning for subclassed models was fragile enough to break evaluation shape consistency

**Severity:** Medium

The concurrent plan-executor session found that async validation had been relying on subclassed-model cloning behavior that was not preserving model construction details reliably enough. The current code now reconstructs the evaluation model with:

- `self.model.__class__.from_config(self.model.get_config())`

instead of plain cloning.

**Why this matters:** for this project, model shape construction is tightly tied to temporal configuration. If validation reconstructs the model with the wrong effective init arguments, evaluation can diverge from the trained architecture and produce incorrect results or outright shape failures.

**Evidence:**

- `src/training/trainer.py` now uses `from_config(self.model.get_config())` in the validation path
- latest concurrent plan-executor session `ses_31a790b1dffeVA5r4K59L73Vwm`

### 8. GPU SpecAugment introduced an unstable CuPy-to-Keras boundary in the training step

**Severity:** Medium

The latest concurrent plan-executor work identified that `batch_spec_augment_gpu(...)` can return a CuPy-backed array, while `train_on_batch(...)` is expecting a stable, consistent input type across steps. The current code now explicitly converts the SpecAugment output back to NumPy before handing it to Keras.

**Why this matters:** even if this is not an ESPHome runtime mismatch, it is a real training-platform correctness problem. Type instability at the training boundary can trigger retracing or hard failures during actual training runs, which means the platform is not reliably executing the intended official-style training job.

**Evidence:**

- `src/training/trainer.py` now converts post-SpecAugment fingerprints with `np.asarray(...)`
- `src/data/spec_augment_gpu.py`
- latest concurrent plan-executor session `ses_31a790b1dffeVA5r4K59L73Vwm`

### 9. The project recently had Keras/XLA training-path instability around `train_on_batch`

**Severity:** Medium

The latest concurrent plan-executor session documented a real failure mode around wrapping `train_on_batch` in an XLA/JIT-compiled path. That wrapper is not present anymore in the current repo state, which is good, but it is still worth recording in this audit because it demonstrates that the training path had been deviating into a brittle, non-official execution mode.

**Why this matters:** the target here is not just to export a compatible flatbuffer; it is to have a trustworthy training platform. If training is routed through fragile wrappers that fight current Keras behavior, then the platform can produce noisy failures or misleading results even when the export contract looks correct.

**Current conclusion:** this appears to be a **recently addressed instability**, not a still-visible source-level mismatch in the current tree.

**Evidence:**

- latest concurrent plan-executor session `ses_31a790b1dffeVA5r4K59L73Vwm`
- no current references remain to the earlier compiled `train_on_batch` helper in `src/training/trainer.py`

## Things that look okay

These major compatibility areas appear aligned with the official/runtime contracts:

- Input/output dtypes: local exporter targets `int8` input and `uint8` output.
- Frontend assumptions: 16 kHz / 40 mel bins / 30 ms / 10 ms / PCAN match ESPHome-style expectations.
- Export uses `tf.keras.export.ExportArchive`, which matches the official workaround away from `model.export()`.
- Manifest generation includes `type: micro`, `version: 2`, `feature_step_size: 10`, and `minimum_esphome_version`.
- Variable-quantized streaming state is treated as mandatory in both the reference and local exporter.

## Bottom line

This repository is **not fundamentally off-track**, but it is **not yet cleanly aligned with the official path** either.

The most important concrete problems are:

1. hardcoded strict verification state shapes,
2. wrong / inconsistent output-scale verification expectation,
3. remaining `trainable_weights` usage in autotuning,
4. non-official random calibration fallback in export,
5. evaluation metrics that do not mirror ESPHome's sum-based detection rule.

The most important unresolved divergences are:

1. `experimental_enable_resource_variables=True` in the local exporter but not in fetched OHF exporter,
2. residual-enabled local defaults instead of plain upstream MixedNet defaults.

Additional reliability findings from the concurrent plan-executor work are worth tracking too:

3. subclassed-model validation reconstruction had to be made more explicit,
4. CuPy SpecAugment output had to be normalized back to NumPy before Keras training,
5. the training path recently had XLA/`train_on_batch` instability.

If the goal is **"100% compatible model training platform"**, these items should be treated as blockers or at minimum mandatory follow-up validation points before trusting the platform as fully aligned.

## Primary evidence used

- `ARCHITECTURAL_CONSTITUTION.md`
- `scripts/verify_esphome.py`
- `src/export/tflite.py`
- `src/export/verification.py`
- `src/tuning/autotuner.py`
- `config/presets/standard.yaml`
- `config/presets/fast_test.yaml`
- `config/presets/max_quality.yaml`
- OHF-Voice `microwakeword/utils.py`
- OHF-Voice `microwakeword/mixednet.py`
- OHF-Voice `microwakeword/layers/stream.py`
- ESPHome `streaming_model.cpp`
- ESPHome `micro_wake_word.cpp`
- ESPHome `preprocessor_settings.h`
