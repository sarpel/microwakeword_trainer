# Phase 1A: Code Quality Review (Raw)

## Executive Summary

The microwakeword_trainer project is a substantial ML training pipeline for microcontroller wake-word detection. The codebase demonstrates strong domain knowledge and careful attention to ESPHome deployment constraints. However, several critical and high-severity issues require attention, particularly around duplicated dead code, a God-class anti-pattern in the trainer, swallowed exceptions, and missing abstractions for repeated configuration-access patterns.

---

## 1. Critical Findings

### 1.1 Double `heapq.heapreplace` Call -- Data Corruption Bug

- **Severity:** Critical
- **File:** `src/training/mining.py`, lines 243-250
- **Description:** When the heap is full and a higher-scoring entry arrives, `heapq.heapreplace` is called twice in succession on the same `heap_entry`. The first call correctly replaces the minimum element, but the second call replaces the *new* minimum (which could be a valid top-K entry) with the exact same entry again. This silently corrupts the top-K heap: a valid hard negative is evicted and replaced by a duplicate.

```python
# Line 245: First replace -- correct
heapq.heapreplace(hard_negative_heap, heap_entry)
# Line 248: Second replace -- BUG: replaces a valid entry with a duplicate
evicted = heapq.heapreplace(hard_negative_heap, heap_entry)
```

- **Fix:** Remove the second `heapq.heapreplace` call entirely.

### 1.2 Dead Code After `return` in Dataset Cache Check

- **Severity:** Critical
- **File:** `src/data/dataset.py`, lines 990-996
- **Description:** The cache validation block in `WakeWordDataset.build()` has duplicated unreachable code after a `return self` statement. Lines 994-996 are dead code.

```python
self._load_store()
self._is_built = True
return self        # <-- Returns here
logger.info(...)   # DEAD CODE
self._load_store() # DEAD CODE
return self        # DEAD CODE
```

- **Fix:** Delete lines 994-996 entirely.

### 1.3 Duplicate Calibration Computation Overwrites Correct Binary Labels

- **Severity:** Critical
- **File:** `src/evaluation/test_evaluator.py`, lines 334-342
- **Description:** `_compute_calibration` correctly binarizes labels then computes Brier score and calibration curve. Then a stray docstring fragment appears and recomputes both using raw `y_true` (which may contain label value `2` for hard negatives), silently overwriting the correct binary values.

```python
y_true_binary = (y_true == 1).astype(np.int32)
brier = compute_brier_score(y_true_binary, y_score)     # Correct
curve = compute_calibration_curve(y_true_binary, ...)    # Correct
"""Compute calibration metrics."""                        # Stray docstring
brier = compute_brier_score(y_true, y_score)             # OVERWRITES with wrong labels
curve = compute_calibration_curve(y_true, y_score, ...)  # OVERWRITES with wrong labels
```

- **Fix:** Remove the stray docstring and duplicate computation (lines 340-342).

---

## 2. High-Severity Findings

### 2.1 God-Class: `Trainer.__init__` ~550 Lines with 80+ Instance Variables

- **Severity:** High
- **File:** `src/training/trainer.py`, lines 236-548
- **Description:** The `Trainer` class initializer extracts dozens of configuration values into instance variables, creating extremely high coupling and low cohesion (80+ instance attributes, 1500+ total lines). Nearly impossible to unit-test in isolation.
- **Fix:** Extract configuration into dedicated dataclasses (`TrainerConfig`, `EvaluationConfig`, `PerformanceConfig`, `MiningConfig`). Extract TensorBoard logging, checkpoint management, and validation into separate collaborator classes.

### 2.2 Repeated Config-Access Boilerplate in Knobs (3x duplication)

- **Severity:** High
- **File:** `src/tuning/knobs.py`, lines 48-158
- **Description:** Every knob (`LRKnob`, `WeightPerturbationKnob`, `LabelSmoothingKnob`) contains the exact same 7-line boilerplate for dict/object config resolution, duplicated 3 times verbatim (~40 lines of duplication).
- **Fix:** Extract a `_get_expert_cfg(config, key, default)` utility function on the `Knob` base class.

### 2.3 Magic Number 7 Hardcoded for Sampling Mix Arms

- **Severity:** High
- **Files:** `src/tuning/orchestrator.py` (lines 340-348), `src/tuning/knobs.py` (line 104)
- **Description:** The sampling mix policy uses a hardcoded 7-arm bandit with magic ratios. `SamplingMixKnob` hardcodes `% 7`. These must stay in sync manually.
- **Fix:** Define mix arms as a class constant; derive modulus from `len(mix_arms)`.

### 2.4 EMA Configuration Checked Three Times Redundantly

- **Severity:** High
- **File:** `src/training/trainer.py`, lines 526-534
- **Description:** The EMA configuration is checked with three separate `if ema_decay is not None:` blocks in close succession, with the third being a complete duplicate of the second.
- **Fix:** Consolidate into a single conditional block.

### 2.5 `FocusedSampler` Class is Dead Code

- **Severity:** High
- **File:** `src/tuning/knobs.py`, lines 161-179
- **Description:** `FocusedSampler` has no callers anywhere in the codebase. Its `build_batch` method returns `None` unconditionally.
- **Fix:** Remove the class entirely.

---

## 3. Medium-Severity Findings

### 3.1 `pipeline.py` Uses `sys.exit()` Instead of Raising Exceptions

- **Severity:** Medium
- **File:** `src/pipeline.py`, lines 39-42, 69-70, 161-162, 199, 214, 488
- **Description:** Pipeline module calls `sys.exit()` from helper and step functions, making pipeline steps impossible to compose programmatically or test without catching `SystemExit`.
- **Fix:** Define custom exception classes (`PipelineStepFailed`, `QualityGateFailed`, `VerificationFailed`) with `exit_code` attribute. Have `main()` translate to `sys.exit()` at the top level.

### 3.2 Triple `pass` in Exception Handler

- **Severity:** Medium
- **File:** `src/utils/performance.py`, lines 42-45
- **Description:** Three consecutive `pass` statements with duplicated comments.
- **Fix:** Reduce to a single `pass`.

### 3.3 Bare `except Exception` Swallows TF Variable Creation Failure

- **Severity:** Medium
- **File:** `src/tuning/orchestrator.py`, lines 456-461
- **Description:** Label smoothing TF variable creation failure is silently caught, disabling the feature with no warning logged.
- **Fix:** At minimum log a warning; catch specific exceptions.

### 3.4 `_run_burst` Method is 140 Lines with Deep Nesting

- **Severity:** Medium
- **File:** `src/tuning/orchestrator.py`, lines 285-421
- **Description:** Single method handles cosine LR, sampling mix arm selection, mini-batch construction, label smoothing, gradient computation, heartbeat logging, and BN freeze/unfreeze.
- **Fix:** Extract `_sample_minibatch()` and `_gradient_step()` methods.

### 3.5 Hardcoded `mel_bins=40` in `output_signature`

- **Severity:** Medium
- **File:** `src/data/tfdata_pipeline.py`, lines 224, 407, 458
- **Description:** Three pipeline creation methods hardcode `40` as mel_bins dimension instead of using a config-derived value.
- **Fix:** Replace with `self.config.get("hardware", {}).get("mel_bins", 40)`.

### 3.6 `batch_features_cache` Grows Unboundedly During Mining

- **Severity:** Medium
- **File:** `src/training/mining.py`, lines 205-253
- **Description:** Cache stores full batch feature tensors and only defers eviction, potentially consuming gigabytes of RAM for large datasets.
- **Fix:** After heap loop, evict cache entries not referenced by remaining heap entries.

### 3.7 `_load_data` in Orchestrator Uses Python Loop for All Samples

- **Severity:** Medium
- **File:** `src/tuning/orchestrator.py`, lines 120-155
- **Description:** Iterates over every sample with individual `ds[i]` calls. Very slow for 50k+ sample datasets.
- **Fix:** Use batch-loading with vectorized operations.

### 3.8 Duplicate `_suppress_stderr_fd` Context Managers

- **Severity:** Medium
- **File:** `src/export/tflite.py`, lines 20-38 and 157-177
- **Description:** Two nearly identical stderr suppression context managers.
- **Fix:** Consolidate into a single utility function.

---

## 4. Low-Severity Findings

### 4.1 Mixed Type Annotation Styles
- **Severity:** Low — Old-style `Dict`, `List`, `Optional` mixed with new-style `dict`, `list`, `X | None`
- **Fix:** Standardize on PEP 604/585 new-style annotations.

### 4.2 f-strings in Logger Calls
- **Severity:** Low — Multiple files use `logger.info(f"...")` rather than `logger.info("...", val)`, causing interpolation overhead when log level is disabled.
- **Fix:** Convert to `%`-style logging.

### 4.3 `EvaluationMetrics.__init__` Docstring After Assignment
- **Severity:** Low — `src/training/trainer.py` lines 64-67: docstring placed after first assignment, making it a string expression not a proper docstring.

### 4.4 `_pad_or_trim` Defined as Closure Inside `__init__`
- **Severity:** Low — `src/training/trainer.py` lines 290-296: makes the helper untestable in isolation.
- **Fix:** Make it a static method or module-level function.

### 4.5 `_file_content_hash` Uses SHA-1, `compute_file_hash` Uses MD5
- **Severity:** Low — `src/data/dataset.py` and `src/training/mining.py`: inconsistent with SHA-256 used elsewhere.
- **Fix:** Standardize on SHA-256.

### 4.6 `EvaluationMetrics.update()` Dict Sync in Python Loop
- **Severity:** Low — `src/training/trainer.py` lines 128-133: post-vectorization loop over 101 cutoffs partially negates optimization.
- **Fix:** Make backward-compat dict views lazy properties.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High     | 5 |
| Medium   | 8 |
| Low      | 6 |
