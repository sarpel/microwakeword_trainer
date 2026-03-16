---
title: Comprehensive Codebase Audit - Bugs, Inconsistencies & ESPHome MWW Compatibility
type: audit
status: active
date: 2026-03-16
sequence: 001
origin: null
---

# Comprehensive Codebase Audit Plan

## Overview

**Objective:** Exhaustive audit of ALL source code files in the microwakeword_trainer project to identify bugs, inconsistencies, and ESPHome Micro Wake Word (MWW) model architecture incompatibilities.

**Scope:** Complete codebase (~19,685 lines Python, 50+ source files)
- 8 core modules: data, model, training, export, evaluation, tuning, tools, utils
- 35+ unit tests, 3 integration tests
- 15+ scripts and utilities

**Success Criteria:**
- [ ] All source files reviewed against architectural constitution
- [ ] All ESPHome MWW compatibility requirements verified
- [ ] All bug patterns from research checklist checked
- [ ] Documented findings with file paths, line numbers, and severity
- [ ] Prioritized remediation plan with effort estimates

---

## ESPHome MWW Compatibility Requirements

### Critical Requirements (Non-Negotiable)

| Requirement | Specification | Violation Impact |
|-------------|---------------|------------------|
| **Input Shape** | `[1, stride, 40]` where stride ∈ {1,3} | Model won't load |
| **Input Dtype** | `int8` | Quantization mismatch |
| **Output Shape** | `[1, 1]` | Dimension error |
| **Output Dtype** | `uint8` (NOT int8) | **Silent device failure** |
| **Quantization** | INT8 weights/activations mandatory | Performance issues |
| **Subgraphs** | Exactly 2 (Main: 94 tensors, Init: 12 tensors) | Load failure |
| **State Variables** | 6 variables (`stream` through `stream_5`) | State corruption |
| **TFLite Ops** | Exactly 20 registered ops (see list below) | Unsupported op error |

**Registered Operations (20 total):**
1. `Conv2D` | 2. `DepthwiseConv2D` | 3. `FullyConnected` | 4. `Add` | 5. `Mul`
6. `Mean` | 7. `Logistic` | 8. `Quantize` | 9. `AveragePool2D` | 10. `MaxPool2D`
11. `Reshape` | 12. `StridedSlice` | 13. `Concatenation` | 14. `Pad` | 15. `Pack`
16. `SplitV` | 17. `VarHandle` | 18. `ReadVariable` | 19. `AssignVariable` | 20. `CallOnce`

### Device Constraints

- **TFLite Micro Version:** esp-tflite-micro v1.3.1
- **Variable Arena Size:** 1024 bytes (fixed)
- **Tensor Arena Size:** 50KB - 150KB
- **Inference Task:** 3072 byte stack, priority 3
- **Feature Parameters:**
  - Sample rate: 16,000 Hz
  - Window size: 30ms (480 samples)
  - Window step: 10ms (160 samples)
  - Mel bins: 40
  - Frequency range: 125Hz - 7500Hz
  - PCAN: Always enabled
  - Log scale: Enabled

---

## Architectural Constitution Rules

### Immutable Hardware Constants (from ARCHITECTURAL_CONSTITUTION.md)

**Audio Preprocessing:**
- Sample rate: 16,000 Hz (must never change)
- Mel bins: 40 (matching ESPHome preprocessor)
- Window size: 30ms (480 samples at 16kHz)
- Window step: 10ms (160 samples at 16kHz)
- PCAN: Always ON in C++ backend
- Noise reduction: Fixed parameters

**Model Architecture:**
- Input shape: `[1, 3, 40]` (int8)
- Output shape: `[1, 1]` (uint8)
- 6 state variables for ring buffers: `stream` through `stream_5`
- MixedNet architecture with MixConv blocks

**Training Configuration:**
- Two-phase training (feature learning → fine-tuning)
- Class weights: positive=1.0, negative=20.0, hard_neg=40.0
- Checkpoint selection: PR-AUC warm-up → recall@target_FAH

### Critical Anti-Patterns (MUST CHECK FOR)

1. **Don't contradict ARCHITECTURAL_CONSTITUTION.md** → Silent device failure
2. **Don't use int8 output dtype** → ESPHome requires uint8
3. **Don't use `model.export()`** → Use `tf.keras.export.ExportArchive`
4. **Don't use `model.trainable_weights` for serialization** → Excludes BatchNorm moving stats. Use `model.get_weights()`/`model.set_weights()`
5. **Don't evaluate auto-tuner on same data used for FocusedSampler training** → Train-on-test contamination via `search_eval_fraction` config
6. **Don't hardcode okay_nabu state shapes in export verification** → Use `compute_expected_state_shapes()` for config-aware validation
7. **Don't reload checkpoints after EMA finalize** → "optimizer has 2 variables whereas saved has 92 variables" warning; only reload when resuming from interruption
8. **Don't feed large chunks to `pymicro_features.process_samples`** → Produces at most ONE frame per call, advances by exactly 160 samples. Feeding 480-sample chunks silently discards 320 samples per call

---

## Bug Pattern Checklist

### 1. TensorFlow Custom Training Loops

**Pattern: Missing Loss Scaling in Mixed Precision**
- **Check:** `trainer.py`, any `tf.GradientTape` usage
- **Bug:** Computing gradients directly from loss without `optimizer.get_scaled_loss()` and `optimizer.get_unscaled_gradients()`
- **Impact:** Gradient underflow to zero, model stops learning

**Pattern: Incorrect Training Mode for BatchNorm**
- **Check:** All training loops, frozen layer handling
- **Bug:** Setting `layer.trainable = False` but calling with `training=True`
- **Impact:** Moving mean/variance still update in "frozen" layers
- **Code to find:** `bn_layer(x, training=True)` after `trainable = False`

### 2. TFLite INT8 Quantization

**Pattern: Wrong Output Dtype**
- **Check:** `src/export/tflite.py`
- **Bug:** `converter.inference_output_type = tf.int8` or unset
- **Correct:** `converter.inference_output_type = tf.uint8`
- **Impact:** Silent device failure on ESPHome

**Pattern: Missing Representative Dataset**
- **Check:** TFLite conversion code
- **Bug:** No representative dataset for calibration or insufficient boundary coverage
- **Impact:** Poor quantization accuracy

**Pattern: Incorrect Scaling**
- **Check:** Feature preprocessing pipeline
- **Bug:** Wrong scaling formula for int8 quantization
- **Correct:** `input = (feature * 256) / 666 - 128`
- **Impact:** Accuracy drop on device

### 3. Streaming/Stateful Model Implementation

**Pattern: Dynamic Batch Size**
- **Check:** `src/model/streaming.py`, model building code
- **Bug:** `Input(shape=(...))` without `batch_size=1` for stateful layers
- **Impact:** State buffer allocation failure

**Pattern: State Shape Mismatch**
- **Check:** `src/export/verification.py`, streaming tests
- **Bug:** Hardcoded state shapes (especially `stream_5`) without computing from config
- **Impact:** False negatives in verification, actual incompatibility
- **Note:** `stream_5` shape depends on `temporal_frames` from `clip_duration_ms`

### 4. CuPy GPU Memory Management

**Pattern: Memory Fragmentation**
- **Check:** `src/data/spec_augment_gpu.py`
- **Bug:** Creating new CuPy arrays in training loop without reusing buffers
- **Impact:** OOM even when `nvidia-smi` shows free memory
- **Fix:** Pre-allocate masks, use in-place operations, clear pool periodically

### 5. Audio Data Pipeline

**Pattern: pymicro_features Chunking Error**
- **Check:** `src/data/features.py`
- **Bug:** Feeding chunks > 160 samples and advancing by chunk size
- **Impact:** Silently discards 2/3 of audio data
- **Correct:** Always advance `byte_idx += output.samples_read * 2` (160 samples)

**Pattern: Sample Rate Mismatch**
- **Check:** Audio loading throughout data pipeline
- **Bug:** Not resampling to 16kHz or assuming input is already 16kHz
- **Impact:** Feature extraction produces wrong frequencies

### 6. EMA (Exponential Moving Average)

**Pattern: Wrong Checkpoint Reload Timing**
- **Check:** `src/training/trainer.py`
- **Bug:** Reloading checkpoints after EMA finalization
- **Impact:** Overwrites smoothed weights with noisy weights
- **Correct:** Only reload when resuming from interruption

**Pattern: EMA Weight Serialization**
- **Check:** Any weight saving/loading code
- **Bug:** Using `model.trainable_weights` instead of `model.get_weights()`
- **Impact:** Missing BatchNorm moving stats

### 7. BatchNorm State Handling

**Pattern: Incomplete State Serialization**
- **Check:** All weight save/load operations
- **Bug:** Using `trainable_weights` (excludes moving_mean/moving_variance)
- **Correct:** Use `get_weights()`/`set_weights()` which includes ALL variables

---

## File-by-File Audit Plan

### HIGH PRIORITY (Architectural Compliance)

#### 1. src/model/architecture.py
**Lines:** ~800 | **Priority:** CRITICAL

**Audit Items:**
- [ ] MixedNet architecture matches ESPHome reference implementation
- [ ] MixConv blocks use correct kernel sizes (3, 5, 7, 9, 11)
- [ ] Input shape: `[1, 3, 40]` int8
- [ ] Output shape: `[1, 1]` uint8
- [ ] Streaming layers properly configured
- [ ] No unsupported TFLite ops in forward pass
- [ ] BatchNorm layers configured correctly
- [ ] Flatten vs GlobalAveragePooling (AUC gap fix verified)

**ESPHome Compatibility:**
- [ ] Uses only 20 registered ops
- [ ] No operations requiring dynamic shapes
- [ ] Compatible with TFLite Micro v1.3.1

---

#### 2. src/export/tflite.py
**Lines:** ~600 | **Priority:** CRITICAL

**Audit Items:**
- [ ] `converter.inference_output_type = tf.uint8` (NOT int8)
- [ ] `converter.inference_input_type = tf.int8`
- [ ] Representative dataset provided with boundary anchors
- [ ] `_experimental_variable_quantization = True` enabled
- [ ] Dual subgraph structure (94/12 tensors)
- [ ] 6 state variables properly identified
- [ ] Uses `tf.keras.export.ExportArchive` (not `model.export()`)
- [ ] INT8 quantization applied to all layers
- [ ] No quantization-aware training artifacts in inference model

**ESPHome Compatibility:**
- [ ] Output dtype is uint8
- [ ] All ops in supported list
- [ ] State shapes computed from config, not hardcoded

---

#### 3. src/training/trainer.py
**Lines:** ~1200 | **Priority:** CRITICAL

**Audit Items:**
- [ ] Two-phase training correctly implemented
- [ ] EMA weights used for evaluation, training weights for updates
- [ ] EMA finalization before final checkpoint save
- [ ] No checkpoint reload after EMA finalize (unless resuming)
- [ ] Loss scaling for mixed precision (`get_scaled_loss`/`get_unscaled_gradients`)
- [ ] Class weights correctly applied (1.0, 20.0, 40.0)
- [ ] BatchNorm training mode handling for frozen layers
- [ ] Checkpoint selection: PR-AUC warm-up → recall@target_FAH
- [ ] Uses `get_weights()`/`set_weights()` (not `trainable_weights`)

**Bug Checks:**
- [ ] No gradient underflow in mixed precision
- [ ] Proper trainable state management
- [ ] Correct EMA averaging

---

#### 4. src/tuning/autotuner.py
**Lines:** ~900 | **Priority:** HIGH

**Audit Items:**
- [ ] Search data partitioning via `search_eval_fraction` (default 0.30)
- [ ] FocusedSampler trains on search_train only (not search_eval)
- [ ] Weight serialization uses `get_weights()`/`set_weights()`
- [ ] No train-on-test contamination
- [ ] Pareto archive properly maintained
- [ ] Thompson sampling correctly implemented
- [ ] BN refresh partitioning correct

**Bug Checks:**
- [ ] No data leakage between search_train and search_eval
- [ ] Proper weight restoration after evaluation

---

### MEDIUM PRIORITY (Data Integrity)

#### 5. src/data/features.py
**Lines:** ~500 | **Priority:** HIGH

**Audit Items:**
- [ ] pymicro_features advances by 160 samples (10ms)
- [ ] No chunks larger than 160 samples fed to process_samples
- [ ] Correct byte advancement: `byte_idx += output.samples_read * 2`
- [ ] 40 mel bins output
- [ ] PCAN enabled (always ON in C++ backend)
- [ ] Sample rate: 16kHz
- [ ] Window size: 30ms, step: 10ms

**Bug Checks:**
- [ ] No silent sample discarding
- [ ] Correct frame count calculation
- [ ] Proper handling of incomplete final frames

---

#### 6. src/data/dataset.py
**Lines:** ~700 | **Priority:** MEDIUM

**Audit Items:**
- [ ] RaggedMmap storage working correctly
- [ ] Variable-length audio handled properly
- [ ] Batch collation correct
- [ ] Label encoding consistent
- [ ] Data augmentation pipeline integrated
- [ ] Negative sampling balanced

**Bug Checks:**
- [ ] No memory leaks in data loading
- [ ] Correct padding/masking for variable lengths

---

#### 7. src/data/clustering.py
**Lines:** ~600 | **Priority:** MEDIUM

**Audit Items:**
- [ ] ECAPA-TDNN embeddings computed correctly
- [ ] Speaker separation logic sound
- [ ] Transitive similarity bug prevention (line 234 comment)
- [ ] Cluster assignment correct
- [ ] Balanced speaker representation

**Bug Checks:**
- [ ] No data leakage between splits
- [ ] Proper handling of single-speaker clusters

---

#### 8. src/data/spec_augment_gpu.py
**Lines:** ~400 | **Priority:** MEDIUM

**Audit Items:**
- [ ] CuPy operations use GPU efficiently
- [ ] Memory pool management (fragmentation prevention)
- [ ] Time and frequency masking correct
- [ ] HAS_CUPY flag handling

**Bug Checks:**
- [ ] No memory leaks
- [ ] Pre-allocated buffers where possible
- [ ] Proper fallback to CPU when CuPy unavailable

---

#### 9. src/model/streaming.py
**Lines:** ~500 | **Priority:** HIGH

**Audit Items:**
- [ ] Ring buffer state management correct
- [ ] 6 state variables (`stream` through `stream_5`)
- [ ] State shapes computed from config (not hardcoded)
- [ ] Streaming inference produces correct output
- [ ] State initialization correct
- [ ] Stateful layers properly configured

**Bug Checks:**
- [ ] No state corruption between frames
- [ ] Correct state reset behavior
- [ ] Batch size fixed at 1 for stateful layers

---

#### 10. src/export/verification.py
**Lines:** ~400 | **Priority:** HIGH

**Audit Items:**
- [ ] Uses `compute_expected_state_shapes()` (not hardcoded)
- [ ] Subgraph count verification (exactly 2)
- [ ] Tensor count verification (94/12)
- [ ] State variable count verification (6)
- [ ] Input/output dtype verification
- [ ] Input/output shape verification

**Bug Checks:**
- [ ] No false negatives from hardcoded shapes
- [ ] Proper handling of different clip_duration_ms values

---

### LOW PRIORITY (Utilities & Configuration)

#### 11. src/training/mining.py
**Lines:** ~800 | **Priority:** MEDIUM

**Audit Items:**
- [ ] HardExampleMiner logic correct
- [ ] AsyncHardExampleMiner thread-safe
- [ ] Cache eviction logic bug fixed (line 283 warning)
- [ ] Negative sampling balanced
- [ ] No data races in async mining

---

#### 12. config/loader.py
**Lines:** ~600 | **Priority:** MEDIUM

**Audit Items:**
- [ ] All 14 dataclasses properly defined
- [ ] Environment variable substitution working
- [ ] Hardware section immutable enforcement
- [ ] Preset loading (fast_test, standard, max_quality)
- [ ] `search_eval_fraction` default (0.30)
- [ ] No deprecated variable names

---

#### 13. src/evaluation/metrics.py
**Lines:** ~700 | **Priority:** LOW

**Audit Items:**
- [ ] FAH calculation correct
- [ ] ROC-AUC calculation correct
- [ ] PR-AUC calculation correct
- [ ] recall@target_FAH correct
- [ ] Vectorized operations efficient

---

#### 14. src/data/augmentation.py
**Lines:** ~400 | **Priority:** LOW

**Audit Items:**
- [ ] Waveform augmentation correct
- [ ] Volume, pitch, speed perturbations sound
- [ ] No audio quality degradation
- [ ] Random seed handling

---

#### 15. src/data/preprocessing.py
**Lines:** ~300 | **Priority:** MEDIUM

**Audit Items:**
- [ ] Audio loading at 16kHz
- [ ] Resampling if needed
- [ ] Mono conversion
- [ ] Normalization correct
- [ ] Quality scoring integration

---

#### 16. Remaining Source Files

**Priority: LOW**

- [ ] src/data/ingestion.py - Audio loading/validation
- [ ] src/data/quality.py - Quality scoring
- [ ] src/export/manifest.py - ESPHome V2 manifest
- [ ] src/export/tflite_utils.py - TFLite utilities
- [ ] src/export/model_analyzer.py - Model analysis
- [ ] src/evaluation/fah_estimator.py - FAH estimation
- [ ] src/evaluation/calibration.py - Probability calibration
- [ ] src/training/augmentation.py - Training augmentation pipeline
- [ ] src/training/rich_logger.py - Training display
- [ ] src/training/profiler.py - Performance profiling
- [ ] src/utils/performance.py - GPU config, mixed precision
- [ ] src/utils/logging_config.py - Logging setup
- [ ] src/utils/terminal_logger.py - Terminal output
- [ ] src/utils/optional_deps.py - Optional dependencies
- [ ] src/utils/seed.py - Random seed management
- [ ] src/tools/cluster_analyze.py - Speaker clustering CLI
- [ ] src/tools/cluster_apply.py - Cluster application CLI
- [ ] src/pipeline.py - Main pipeline orchestration

---

### TEST FILES

#### 17. Unit Tests (35 files)
**Priority:** MEDIUM

**Check for:**
- [ ] All tests passing
- [ ] No skipped tests (except CuPy unavailable - acceptable)
- [ ] Test coverage for critical paths
- [ ] Mock objects don't mask real bugs
- [ ] Async test handling correct

**Key Test Files:**
- [ ] test_training_async_validation.py
- [ ] test_training_class_weights.py
- [ ] test_metrics_target_fah.py
- [ ] test_evaluation_metrics.py
- [ ] test_export_tflite_metadata.py
- [ ] test_export_manifest.py
- [ ] test_model_architecture_streaming.py
- [ ] test_tuning_bn_refresh_partition.py
- [ ] test_data_features.py
- [ ] test_data_augmentation.py

#### 18. Integration Tests (3 files)
**Priority:** HIGH

- [ ] test_pipeline_e2e.py - End-to-end pipeline
- [ ] test_training.py - Training integration
- [ ] test_pipeline_regression.py - Regression tests

---

### SCRIPTS

#### 19. Verification Scripts
**Priority:** HIGH

- [ ] scripts/verify_esphome.py - ESPHome compatibility
- [ ] scripts/verify_streaming.py - Streaming verification
- [ ] scripts/check_esphome_compat.py - Detailed diagnostics

#### 20. Evaluation Scripts
**Priority:** MEDIUM

- [ ] scripts/evaluate_model.py - Model evaluation
- [ ] scripts/eval_dashboard.py - Dashboard generation
- [ ] scripts/compare_models.py - Model comparison

#### 21. Utility Scripts
**Priority:** LOW

- [ ] scripts/generate_test_dataset.py
- [ ] scripts/vad_trim.py
- [ ] scripts/split_audio.py
- [ ] scripts/score_quality_full.py
- [ ] scripts/score_quality_fast.py
- [ ] scripts/phonetic_scorer.py
- [ ] scripts/count_audio_hours.py
- [ ] scripts/count_dataset.py
- [ ] scripts/cleanup_tfdata_cache.py
- [ ] scripts/audio_similarity_detector.py
- [ ] scripts/audio_analyzer.py
- [ ] scripts/ci.sh

---

## Audit Execution Plan

### Phase 1: Critical Path (Days 1-2)
**Files:** architecture.py, tflite.py, trainer.py, autotuner.py
**Focus:** ESPHome compatibility, architectural constitution compliance
**Deliverable:** Critical issues report with severity ratings

### Phase 2: Data Pipeline (Days 3-4)
**Files:** features.py, dataset.py, clustering.py, spec_augment_gpu.py, streaming.py, verification.py
**Focus:** Data integrity, feature extraction correctness
**Deliverable:** Data pipeline audit report

### Phase 3: Supporting Modules (Days 5-6)
**Files:** mining.py, metrics.py, augmentation.py, preprocessing.py, loader.py
**Focus:** Consistency, performance, code quality
**Deliverable:** Secondary modules audit report

### Phase 4: Tests & Scripts (Days 7-8)
**Files:** All test files, all scripts
**Focus:** Test coverage, script functionality, edge cases
**Deliverable:** Test and script audit report

### Phase 5: Remaining Utilities (Day 9)
**Files:** All remaining source files
**Focus:** Code quality, documentation, consistency
**Deliverable:** Final audit report

### Phase 6: Synthesis & Remediation Plan (Day 10)
**Focus:** Compile findings, prioritize fixes, estimate effort
**Deliverable:** Comprehensive audit report + remediation roadmap

---

## Severity Classification

| Severity | Definition | Examples | Response Time |
|----------|------------|----------|---------------|
| **CRITICAL** | Silent device failure or complete incompatibility | int8 output, wrong state shapes, unsupported ops | Immediate fix |
| **HIGH** | Significant accuracy degradation or training failure | Data contamination, EMA bugs, feature extraction errors | Fix within 1 week |
| **MEDIUM** | Performance issues or code quality problems | Memory fragmentation, test coverage gaps | Fix within 2 weeks |
| **LOW** | Minor issues, documentation gaps | Typos, missing comments, style inconsistencies | Fix when convenient |

---

## Documentation & References

### Internal Documentation
- **ARCHITECTURAL_CONSTITUTION.md** - Immutable architectural truth
- **AGENTS.md** - Main project knowledge base
- **docs/ARCHITECTURE.md** - MixedNet architecture details
- **docs/CONFIGURATION.md** - Config reference
- **docs/TRAINING.md** - Training guide
- **docs/EXPORT.md** - Export guide

### External References
- **ESPHome MWW Component:** https://esphome.io/components/micro_wake_word.html
- **Official Model Collection:** https://github.com/esphome/micro-wake-word-models
- **Training Framework:** https://github.com/OHF-Voice/micro-wake-word
- **ESPHome Source:** https://github.com/esphome/esphome/tree/main/esphome/components/micro_wake_word

### Module-Specific AGENTS.md
- `src/data/AGENTS.md` - Data pipeline
- `src/training/AGENTS.md` - Training loop
- `src/model/AGENTS.md` - Architecture
- `src/export/AGENTS.md` - TFLite export
- `src/evaluation/AGENTS.md` - Metrics
- `src/utils/AGENTS.md` - GPU config
- `src/tools/AGENTS.md` - CLI tools
- `src/tuning/AGENTS.md` - Auto-tuning
- `config/AGENTS.md` - Configuration
- `scripts/AGENTS.md` - Scripts

---

## Known Issues from Research

### Already Fixed (Verify Still Working)
1. **Training-Export AUC Gap** (2026-03-11) - Flatten vs GlobalAveragePooling
2. **Auto-tuner Serialization Bug** (2026-03-10) - `get_weights()`/`set_weights()`
3. **Search Data Contamination** - `search_eval_fraction` partitioning

### Potential Issues Found
1. **mining.py:283** - Warning about potential cache eviction logic bug
2. **clustering.py:234** - Transitive similarity bug prevention (verify working)
3. **test_data_augmentation.py:591** - CuPy skip conditional (acceptable)

### Recent Bug Fix Commit
- Commit `57c2a31d3` - "fix bugs across multiple modules" (details not specified)

---

## Audit Report Template

For each finding, document:

```markdown
### FINDING-XXX: [Brief Title]
**File:** `path/to/file.py:line_number`
**Severity:** CRITICAL/HIGH/MEDIUM/LOW
**Category:** ESPHome Compatibility / Bug / Inconsistency / Code Quality

**Description:**
Detailed description of the issue.

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Impact:**
Consequences of this issue.

**Remediation:**
Suggested fix with code example if applicable.

**Verification:**
How to verify the fix works.
```

---

## Success Metrics

- [ ] 100% of source files reviewed
- [ ] All CRITICAL and HIGH severity issues identified
- [ ] Complete ESPHome MWW compatibility verification
- [ ] All architectural constitution rules checked
- [ ] Documented findings with clear remediation steps
- [ ] Prioritized backlog with effort estimates
- [ ] No silent device failure risks remaining

---

## Next Steps

1. **Begin Phase 1** - Critical path audit (architecture.py, tflite.py, trainer.py, autotuner.py)
2. **Run existing tests** - Establish baseline before making changes
3. **Execute verification scripts** - Check current ESPHome compatibility
4. **Document findings** - Use audit report template for each issue
5. **Prioritize fixes** - Focus on CRITICAL and HIGH severity first

---

*Plan created: 2026-03-16*
*Estimated audit duration: 10 days*
*Estimated remediation: 2-4 weeks (depending on findings)*
