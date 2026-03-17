# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### Critical

#### CQ-C1: Double `heapq.heapreplace` Call — Data Corruption Bug
- **File:** `src/training/mining.py`, lines 243-250
- When the heap is full, `heapq.heapreplace` is called twice on the same `heap_entry`. The second call evicts a valid top-K entry and replaces it with a duplicate, silently corrupting the hard-negative mining results.
- **Fix:** Remove the second `heapq.heapreplace` call.

#### CQ-C2: Dead Code After `return self` in `WakeWordDataset.build()`
- **File:** `src/data/dataset.py`, lines 990-996
- Three lines (`logger.info(...)`, `self._load_store()`, `return self`) are unreachable after an earlier `return self`. Indicates a bad merge; `_is_built = True` is only set in the live branch.
- **Fix:** Delete lines 994-996.

#### CQ-C3: Duplicate Calibration Computation Overwrites Correct Binary Labels
- **File:** `src/evaluation/test_evaluator.py`, lines 334-342
- `_compute_calibration` correctly binarizes labels then a stray docstring string literal appears mid-function, followed by a duplicate computation using raw `y_true` (which may contain label `2` for hard negatives). The second computation silently overwrites the correct Brier score and calibration curve.
- **Fix:** Remove the stray docstring and duplicate computation (lines 340-342).

### High

#### CQ-H1: God-Class — `Trainer.__init__` ~550 Lines, 80+ Instance Variables
- **File:** `src/training/trainer.py`, lines 236-548
- The class handles training phases, evaluation metrics, TensorBoard logging, hard-negative mining, EMA, profiling, plateau LR scheduling, and async validation in a single class. Nearly impossible to unit-test in isolation.
- **Fix:** Extract `CheckpointManager`, `PhaseScheduler`, `EvaluationOrchestrator`, `TensorBoardReporter` collaborators.

#### CQ-H2: Repeated Config-Access Boilerplate in Knobs (3× duplication)
- **File:** `src/tuning/knobs.py`, lines 48-158
- `LRKnob`, `WeightPerturbationKnob`, `LabelSmoothingKnob` each contain identical 7-line dict-vs-object config resolution logic (~40 lines of duplication).
- **Fix:** Extract `_get_expert_cfg(config, key, default)` onto the `Knob` base class.

#### CQ-H3: Magic Number `7` Hardcoded for Sampling Mix Arms
- **Files:** `src/tuning/orchestrator.py` lines 340-348; `src/tuning/knobs.py` line 104
- Hardcoded 7-arm bandit with `% 7` modulus. Must stay in sync manually with no documentation.
- **Fix:** Define mix arms as a class constant; derive modulus from `len(mix_arms)`.

#### CQ-H4: EMA Configuration Checked Three Times Redundantly
- **File:** `src/training/trainer.py`, lines 526-534
- Three separate `if ema_decay is not None:` blocks in close succession; third is a complete duplicate of the second.
- **Fix:** Consolidate into a single conditional block.

#### CQ-H5: `FocusedSampler` is Dead Code
- **File:** `src/tuning/knobs.py`, lines 161-179
- No callers. `build_batch()` returns `None` unconditionally.
- **Fix:** Remove the class.

### Medium

#### CQ-M1: `pipeline.py` Uses `sys.exit()` Instead of Exceptions
- **File:** `src/pipeline.py`, lines 39-42, 69-70, 161-162, 199, 214, 488
- Prevents programmatic composition and testing of pipeline steps.
- **Fix:** Define `PipelineStepFailed`, `QualityGateFailed`, `VerificationFailed` exceptions; translate to `sys.exit()` only in `main()`.

#### CQ-M2: Triple `pass` in Exception Handler
- **File:** `src/utils/performance.py`, lines 42-45
- Three consecutive `pass` statements with duplicated comments.
- **Fix:** Reduce to a single `pass`.

#### CQ-M3: Bare `except Exception` Swallows TF Variable Creation
- **File:** `src/tuning/orchestrator.py`, lines 456-461
- Label smoothing silently disabled with no warning if TF variable creation fails.
- **Fix:** Log a warning; catch only specific exceptions.

#### CQ-M4: `_run_burst` is 140 Lines with Deep Nesting
- **File:** `src/tuning/orchestrator.py`, lines 285-421
- Handles LR schedule, batch sampling, label smoothing, gradient step, heartbeat, BN freeze/unfreeze in one method.
- **Fix:** Extract `_sample_minibatch()` and `_gradient_step()`.

#### CQ-M5: Hardcoded `mel_bins=40` in Three `output_signature` Calls
- **File:** `src/data/tfdata_pipeline.py`, lines 224, 407, 458
- **Fix:** Replace with `self.config.get("hardware", {}).get("mel_bins", 40)`.

#### CQ-M6: `batch_features_cache` Grows Unboundedly During Mining
- **File:** `src/training/mining.py`, lines 205-253
- Cache defers eviction but could consume gigabytes for large datasets.
- **Fix:** After heap loop, evict entries not referenced by remaining heap entries.

#### CQ-M7: `_load_data` Uses Python Loop for All Samples
- **File:** `src/tuning/orchestrator.py`, lines 120-155
- Individual `ds[i]` calls in a Python loop; very slow for 50k+ samples.
- **Fix:** Use vectorized batch loading.

#### CQ-M8: Duplicate stderr-Suppression Context Managers
- **File:** `src/export/tflite.py`, lines 20-38 and 157-177
- Two nearly identical context managers for redirecting FD 2.
- **Fix:** Consolidate into a single utility function.

### Low

- **CQ-L1:** Mixed old/new type annotation styles across modules (`Dict` vs `dict`, `Optional` vs `X | None`)
- **CQ-L2:** f-strings in logger calls — interpolation overhead when level is disabled
- **CQ-L3:** `EvaluationMetrics.__init__` docstring placed after first assignment (not a real docstring)
- **CQ-L4:** `_pad_or_trim` defined as closure inside `__init__` — untestable in isolation
- **CQ-L5:** `_file_content_hash` uses SHA-1; `compute_file_hash` uses MD5 — inconsistent with SHA-256 used elsewhere
- **CQ-L6:** `EvaluationMetrics.update()` syncs dict views in Python loop over 101 cutoffs, partially negating vectorization

---

## Architecture Findings

### Critical

#### AR-C1: Untyped `dict[str, Any]` Configuration Threaded Through Entire Codebase
- **Files:** `src/training/trainer.py`, `src/tuning/orchestrator.py`, `src/data/tfdata_pipeline.py`, `src/model/architecture.py`, and all consumers
- Every module re-extracts config via `.get("key", default)`, duplicating knowledge of keys and defaults. No central schema. A typo in any key or a mismatched default between modules produces a silently incorrect model.
- **Constitution risk:** `mel_bins=40` and `window_step_ms=10` are mandated constants, but each module independently defaults them via `.get()` — a config override could silently produce incompatible models.
- **Fix:** Define a typed `TrainingConfig` schema (dataclasses or Pydantic). Create a single `load_config()` that validates YAML and returns typed objects. Enforce constitution constants via `Final` annotations.

### High

#### AR-H1: Hardware Constants Computed Independently in 4+ Places
- **Files:** `src/tuning/orchestrator.py` line 104, `src/training/trainer.py` line 362, `src/data/tfdata_pipeline.py` line 90, `src/model/architecture.py` (implicitly)
- `max_time_frames = clip_duration_ms / window_step_ms` is derived independently by each consumer. A discrepancy would produce incompatible model shapes silently.
- **Fix:** Compute once in a `HardwareConstants` frozen dataclass; propagate through typed config.

#### AR-H2: ARCHITECTURAL_CONSTITUTION Enforcement is Entirely Trust-Based
- **Files:** `ARCHITECTURAL_CONSTITUTION.md` vs all code
- No runtime assertions, compile-time checks, or focused integration tests verify that exported models comply with the prescribed op set, streaming state shapes, audio frontend constants, or quantization ranges. Violations propagate silently until final `verify_esphome.py`.
- **Fix:** Add `constitution_check()` assertions in `build_model()`, at export boundaries, and in CI.

#### AR-H3: Monolithic `Trainer` Class (duplicates CQ-H1 from architecture perspective)
- **File:** `src/training/trainer.py`
- Direct dependency on 12+ internal modules; training loop, evaluation, checkpointing, TensorBoard, and mining are all entangled.

### Medium

#### AR-M1: `pipeline.py` Orchestrates via Subprocess Instead of Python API
- **File:** `src/pipeline.py`
- Each step invokes `sys.executable -m ...`, losing type safety, error detail, and adding TF re-import overhead per step. Config passes through CLI args and JSON stdout.
- **Fix:** Import and invoke `Trainer.train()`, `MicroAutoTuner.tune()`, `export_model()` directly.

#### AR-M2: Dual Config Access Patterns in Tuning (same as CQ-H2)
- `_get_cfg()` helper in orchestrator vs. inline isinstance chains in each knob.

#### AR-M3: `Knob.apply()` Parameters Are All `Any` — Candidate Attributes Are Monkey-Patched
- **File:** `src/tuning/knobs.py`, line 19
- Knobs access `candidate._sampled_lr`, `candidate._sampling_mix_arm`, etc. — private attributes set by convention, not declared in `Candidate`.
- **Fix:** Define a `CandidateState` protocol or extend `Candidate` dataclass to declare all knob-writable attributes.

#### AR-M4: `dataset.py` Conflates Three Abstraction Layers
- **File:** `src/data/dataset.py`
- `RaggedMmap` (storage), `FeatureStore` (feature management), `WakeWordDataset` (dataset access) in one file.
- **Fix:** Split into `ragged_mmap.py`, `feature_store.py`, keep `dataset.py` for `WakeWordDataset`.

#### AR-M5: Dual SpecAugment Backends Create Maintenance Risk
- **Files:** `src/data/spec_augment_gpu.py` (CuPy), `src/data/spec_augment_tf.py` (TF)
- `tfdata_pipeline.py` uses TF backend; `trainer.py` imports GPU backend. `create_training_pipeline_with_spec_augment()` mutates `self.spec_augment_config["enabled"]` (lines 361-370) — thread-safety hazard.
- **Fix:** Pass SpecAugment config as an immutable parameter; unify backend selection.

#### AR-M6: Model Construction Duplicated in Trainer and Auto-Tuner
- **Files:** `src/training/trainer.py`, `src/tuning/orchestrator.py` lines 161-190
- Both independently extract hardware/model parameters and call `build_model()`.
- **Fix:** Introduce a `ModelFactory` that centralizes model construction.

#### AR-M7: No Config Loader Module
- **File:** `src/config/__init__.py` (empty except for version string)
- YAML loading logic is inlined in `tflite.py` and presumably in the training entrypoint.
- **Fix:** Create `src/config/loader.py` with `load_config(preset_or_path, override_path)`.

#### AR-M8: Hardcoded `40` in TensorSpec (duplicates CQ-M5 from architecture perspective)

### Low

- **AR-L1:** Inconsistent lazy TF imports — `orchestrator.py` lazy, `trainer.py` eager; document or unify
- **AR-L2:** `ThresholdOptimizer.optimize()` annotated as bare `tuple` instead of `tuple[float, int, TuneMetrics]`
- **AR-L3:** `ErrorMemory` instantiated but never used (`orchestrator.py` line 450-452)
- **AR-L4:** `build_model()` `num_classes=2` parameter is unused — model always outputs `Dense(1)`
- **AR-L5:** Mixed `print()` and `logging` — `pipeline.py` and parts of `tflite.py` use `print()`

---

## Critical Issues for Phase 2 Context

The following findings are most likely to have security or performance implications:

1. **Unbounded `batch_features_cache`** (CQ-M6) — memory exhaustion risk during large mining runs
2. **Slow Python loop in `_load_data`** (CQ-M7) — CPU-bound bottleneck before every tuning run
3. **Subprocess pipeline** (AR-M1) — multiple TF re-imports add significant latency; each subprocess re-initializes GPU
4. **`_run_burst` 140-line method** (CQ-M4) — gradient computation and sampling interleaved; hard to audit for correctness
5. **Untyped config system** (AR-C1) — any config key typo silently uses a wrong default, potentially producing a model that passes unit tests but fails on-device

---

## Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Code Quality | 3 | 5 | 8 | 6 | 22 |
| Architecture | 1 | 3 | 8 | 5 | 17 |
| **Total** | **4** | **8** | **16** | **11** | **39** |
