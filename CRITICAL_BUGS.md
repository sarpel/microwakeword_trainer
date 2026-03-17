# Critical Functional Bugs in Source Code

This document lists all confirmed functional bugs in source code files (excluding documentation, tests, configuration, and cosmetic issues).

**Total: 17 bugs** — 10 Critical, 4 High, 3 Medium

---

## Critical (Data Corruption / Crashes / Security)

### CQ-C1: Double `heapq.heapreplace` — Hard-Negative Mining Heap Corruption
- **File:** `src/training/mining.py`
- **Lines:** 245, 248
- **Bug:** Two consecutive `heapq.heapreplace` calls on the same `heap_entry`. The second call evicts a valid top-K entry and replaces it with a duplicate.
- **Impact:** Silent data corruption — every mining pass produces an incorrect top-K set.
- **Fix:** Remove the second `heapq.heapreplace(hard_negative_heap, heap_entry)` call at line 248.

---

### CQ-C2: Dead Code After `return self` — Cache Loading Skipped
- **File:** `src/data/dataset.py`
- **Lines:** 993-996
- **Bug:** Three lines (`logger.info(...)`, `self._load_store()`, `return self`) are unreachable after an earlier `return self` on line 993. The `_load_store()` side-effect is silently skipped on cache-hit path.
- **Impact:** Cache hit path never loads the feature store; `_is_built = True` not set.
- **Fix:** Delete lines 994-996.

---

### CQ-C3: Calibration Overwrites Correct Binary Labels with Raw Labels
- **File:** `src/evaluation/test_evaluator.py`
- **Lines:** 340-342
- **Bug:** After correctly binarizing labels (`y_true_binary`), a stray docstring literal appears, followed by a duplicate computation using raw `y_true` (which may contain label=2 for hard negatives). The second computation overwrites the correct Brier score and calibration curve.
- **Impact:** Wrong Brier score and calibration curve when hard negatives (label=2) are present.
- **Fix:** Remove the stray docstring and duplicate computation (lines 340-342).

---

### BP-H4: `MixConvBlock.__init__` Incomplete — AttributeError + Infinite Recursion
- **File:** `src/model/architecture.py`
- **Lines:** 151-165
- **Bug:** `self.filters = filters` is never assigned in `__init__`, causing `AttributeError` in `build()`. The `mode` setter contains `self.mode = mode` inside the setter body, causing infinite recursion.
- **Impact:** Model building crashes with `AttributeError` or infinite recursion.
- **Fix:** Complete `__init__` to assign all instance attributes. Fix setter to use `self._mode = value` throughout.

---

### SEC-H1: Pickle Deserialization — Arbitrary Code Execution
- **File:** `src/tuning/population.py`
- **Lines:** 28, 34, 86, 114
- **Bug:** `Candidate` uses `pickle.dumps()`/`pickle.loads()` for model weights serialization. If tuning state is persisted to disk and loaded from a malicious file, arbitrary code execution is possible.
- **Impact:** Remote Code Execution (RCE) vulnerability (CVSS ~7.5).
- **Fix:** Replace with `numpy.savez`/`numpy.load(allow_pickle=False)`. Never persist `weights_bytes` to disk in raw pickle form.

---

### SEC-H2: `numpy allow_pickle=True` on Cache Files from World-Writable `/tmp`
- **File:** `src/data/clustering.py`
- **Line:** 1162
- **Bug:** `np.load(cache_file, allow_pickle=True)` loads from `/tmp` (world-writable). Attacker can replace cache file with malicious numpy pickle payload.
- **Impact:** Remote Code Execution (RCE) via malicious cache file (CVSS ~7.5).
- **Fix:** Change to `allow_pickle=False`; save `model_name` field as JSON sidecar.

---

### SEC-M1: Unbounded `batch_features_cache` — Memory Exhaustion
- **File:** `src/training/mining.py`
- **Line:** 205
- **Bug:** Cache grows linearly with dataset size and is not cleared until all batches processed.
- **Impact:** OOM crash on large datasets (CWE-400).
- **Fix:** Evict non-heap-referenced entries during the mining loop.

---

### SEC-M3: `ast.literal_eval` on Unvalidated Config-Derived Strings
- **Files:**
  - `src/model/architecture.py:38`
  - `src/export/verification.py:46`
  - `scripts/debug_streaming_gap.py:42`
  - `scripts/verify_esphome.py:172`
- **Bug:** `ast.literal_eval(f"[{user_string}]")` has no format validation before parsing.
- **Impact:** Potential injection or DoS via malformed input (CWE-95).
- **Fix:** Validate against `^[\d,\[\]\s]+$` before parsing; prefer explicit integer list parsing.

---

### SEC-M4: Subprocess Args Include Unvalidated Config Values
- **File:** `src/pipeline.py`
- **Lines:** 58-60
- **Bug:** `ffprobe` call lacks `--` before filename; no path existence validation before passing to subprocess.
- **Impact:** Command injection risk if malicious filenames are processed (CWE-78).
- **Fix:** Validate paths; use `["ffprobe", ..., "--", str(file_path)]`.

---

### SEC-M6: `yaml.safe_load` on Dynamically Constructed YAML String
- **File:** `scripts/verify_esphome.py`
- **Line:** 172
- **Bug:** `yaml.safe_load(f"[{mixconv_str}]")` constructs YAML from string interpolation.
- **Impact:** Injection risk via malformed mixconv string (CWE-20).
- **Fix:** Use explicit integer list parsing instead of YAML construction.

---

## High (Silent Failures / Thread-Safety)

### CQ-M3: Bare `except Exception` Swallows TF Variable Creation
- **File:** `src/tuning/orchestrator.py`
- **Lines:** 456-461
- **Bug:** Label smoothing variable creation silently disabled with no warning if TF variable creation fails.
- **Impact:** Feature silently disabled — training proceeds without intended label smoothing.
- **Fix:** Log a warning; catch only specific exceptions (e.g., `tf.errors.ResourceExhaustedError`).

---

### AR-M5 / PERF-M5: Mutable SpecAugment Config — Thread-Safety Hazard
- **File:** `src/data/tfdata_pipeline.py`
- **Lines:** 361-370
- **Bug:** `create_training_pipeline_with_spec_augment()` temporarily mutates `self.spec_augment_config["enabled"]`.
- **Impact:** Thread-safety hazard if pipeline object is shared; non-deterministic training behavior.
- **Fix:** Pass SpecAugment config as immutable parameter; do not mutate instance state.

---

### PERF-H3: `.numpy()` Calls Inside `_run_burst` Step Loop
- **File:** `src/tuning/orchestrator.py`
- **Method:** `_run_burst`
- **Bug:** `.numpy()` calls inside the inner step loop force CPU-GPU synchronization on every step.
- **Impact:** Prevents GPU pipeline overlap; training behavior may differ from expected GPU execution.
- **Fix:** Remove `.numpy()` calls from hot path; accumulate TF tensors and convert outside the loop.

---

### BP-H2: `train_on_batch` Contradicts "No `.numpy()` in Hot Path" Docstring
- **File:** `src/training/trainer.py`
- **Line:** 1341
- **Bug:** `model.train_on_batch()` executes eagerly with Python round-trip per call, contradicting docstring claim of "no `.numpy()` calls in hot path".
- **Impact:** Forces CPU-GPU sync; cannot be XLA-compiled; eager/graph tracing issues.
- **Fix:** Override `train_step` in model subclass or use `@tf.function(reduce_retracing=True)` gradient tape loop.

---

## Medium (Edge Cases / API Issues)

### AR-L4: `num_classes=2` Parameter Unused
- **File:** `src/model/architecture.py`
- **Bug:** `build_model()` accepts `num_classes=2` but model always outputs `Dense(1)`.
- **Impact:** Wrong model output shape if called with different `num_classes` value.
- **Fix:** Honor `num_classes` parameter or remove it from signature.

---

### CQ-M1: `sys.exit()` in Library Code
- **File:** `src/pipeline.py`
- **Lines:** 39-42, 69-70, 161-162, 199, 214, 488
- **Bug:** Pipeline steps call `sys.exit()` instead of raising exceptions.
- **Impact:** Prevents programmatic composition and testing; prevents cleanup code from running.
- **Fix:** Define `PipelineStepFailed`, `QualityGateFailed` exceptions; translate to `sys.exit()` only in `main()`.

---

### CQ-M6: `batch_features_cache` Grows Unboundedly During Mining
- **File:** `src/training/mining.py`
- **Lines:** 205-253
- **Bug:** Cache defers eviction but could consume gigabytes for large datasets.
- **Impact:** Memory exhaustion on large mining runs (similar to SEC-M1).
- **Fix:** After heap loop, evict entries not referenced by remaining heap entries.

---

## Bug Count by Severity

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 10 | Data corruption, crashes, RCE vulnerabilities |
| **High** | 4 | Silent failures, thread-safety issues |
| **Medium** | 3 | Edge cases, API misuse |
| **Total** | **17** | |

---

## Priority Fix Order

### Immediate (This Week)
1. **CQ-C1** — Double heapreplace (corrupts mining results)
2. **CQ-C3** — Wrong calibration labels (wrong metrics)
3. **CQ-C2** — Dead code after return (cache broken)
4. **BP-H4** — MixConvBlock crash (model build fails)
5. **SEC-H1** — Pickle RCE (security)
6. **SEC-H2** — allow_pickle RCE (security)

### Short-term (Next Sprint)
7. **SEC-M1/CQ-M6** — Unbounded cache (OOM risk)
8. **SEC-M3/M4/M6** — Input validation gaps
9. **CQ-M3** — Silent label smoothing failure
10. **AR-M5** — Mutable config thread-safety

### Medium-term
11. **PERF-H3/BP-H2** — GPU sync issues
12. **AR-L4** — Unused num_classes parameter
13. **CQ-M1** — sys.exit() in library code

---

*Generated: 2026-03-17*
*Source: Comprehensive code review (Phases 1-4)*
