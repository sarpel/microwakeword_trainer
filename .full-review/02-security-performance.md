# Phase 2: Security & Performance Review

## Security Findings

### High

#### SEC-H1: Unsafe Pickle Deserialization in Population Module
- **CWE:** CWE-502 — CVSS 7.5
- **File:** `src/tuning/population.py`, lines 28, 34, 86, 114
- `Candidate` uses `pickle.dumps()`/`pickle.loads()` for model weights serialization. If tuning state is ever persisted to disk (as AGENTS.md design suggests) and loaded from a malicious file, arbitrary code execution is possible.
- Compounded by `pyproject.toml:121` globally suppressing Bandit `S301` (pickle) — new pickle usage will never be flagged by linter.
- **Fix:** Replace with `numpy.savez`/`numpy.load(allow_pickle=False)`. Never persist `weights_bytes` to disk in raw pickle form.

#### SEC-H2: `numpy allow_pickle=True` on Cache Files from World-Writable `/tmp`
- **CWE:** CWE-502 — CVSS 7.5
- **File:** `src/data/clustering.py`, line 1162
- `np.load(cache_file, allow_pickle=True)` loads from `/tmp` (world-writable). Attacker can replace `emb_*.npz` cache file with malicious numpy pickle payload triggering RCE. Line 549 in same file correctly uses `allow_pickle=False`.
- **Fix:** Change to `allow_pickle=False`; save `model_name` field as JSON sidecar.

#### SEC-H3: Overly Permissive Global Ruff Security Rule Suppressions
- **CWE:** CWE-693
- **File:** `pyproject.toml`, lines 117-128
- Globally suppresses `S301` (pickle), `S603`/`S607` (subprocess), `S605`/`S606` (OS commands), `S105`-`S107` (hardcoded passwords). New code introducing these patterns is never flagged.
- **Fix:** Replace global ignores with `[tool.ruff.lint.per-file-ignores]`; use inline `# noqa: S301` only where genuinely needed.

### Medium

#### SEC-M1: Unbounded `batch_features_cache` — Memory Exhaustion (CWE-400)
- **File:** `src/training/mining.py`, line 205
- Cache grows linearly with dataset size and is not cleared until all batches processed. Potential OOM crash on large datasets.
- **Fix:** Evict non-heap-referenced entries during the mining loop.

#### SEC-M2: Temp Directories in World-Writable `/tmp` for ML Models (CWE-377, CWE-732)
- **Files:** `src/data/clustering.py` lines 124, 490, 674, 961; `src/export/tflite.py` line 1237
- SpeechBrain models and embedding caches in `/tmp` — symlink attack / model poisoning risk on shared systems.
- **Fix:** Use project-local cache directories; log warnings on cleanup failure (currently `ignore_errors=True`).

#### SEC-M3: `ast.literal_eval` on Config-Derived Strings Without Validation (CWE-95)
- **Files:** `scripts/debug_streaming_gap.py:42`, `src/model/architecture.py:38`, `src/export/verification.py:46`, `src/export/tflite.py:1633`, `scripts/verify_esphome.py:172`
- `ast.literal_eval(f"[{user_string}]")` has no format validation before parsing.
- **Fix:** Validate against `^[\d,\[\]\s]+$` before parsing; prefer explicit integer list parsing.

#### SEC-M4: Subprocess Args Include Unvalidated Config Values (CWE-78)
- **Files:** `src/pipeline.py` lines 58-60, 93-109; `scripts/count_dataset.py` lines 17-31
- List-form calls prevent shell injection, but no path existence validation before passing to subprocesses. `ffprobe` call lacks `--` before filename.
- **Fix:** Validate paths; use `["ffprobe", ..., "--", str(file_path)]`.

#### SEC-M5: Insufficient `.gitignore` Coverage (CWE-200)
- **File:** `.gitignore` (only 8 lines)
- Missing: `.env*`, `*.key`, `*.pem`, `checkpoints/`, `models/`, `dataset/`, `data/`, `logs/`, `tuning_output/`, `.sisyphus/`, `coverage_html/`, `coverage.xml`
- **Fix:** Expand `.gitignore` to cover all sensitive directories.

#### SEC-M6: `yaml.safe_load` on Dynamically Constructed YAML String (CWE-20)
- **File:** `scripts/verify_esphome.py`, line 172
- `yaml.safe_load(f"[{mixconv_str}]")` constructs YAML from string interpolation.
- **Fix:** Use explicit integer list parsing instead.

### Low
- **SEC-L1:** MD5 in `mining.py:84`, `clustering.py:509` — standardize on SHA-256
- **SEC-L2:** SHA-1 in `dataset.py:765` — standardize on SHA-256
- **SEC-L3:** `MWW_VERIFY_CONFIG`/`MWW_VERIFY_CHECKPOINT` env vars not validated against expected paths
- **SEC-L4:** `.sisyphus/evidence/` created without restrictive permissions
- **SEC-L5:** `sys.exit()` deep in library code suppresses error context and prevents cleanup

---

## Performance Findings

### Critical

#### PERF-C1: Sequential `ds[i]` Loop in `_load_data` — 60-120s Blocking Startup Cost
- **File:** `src/tuning/orchestrator.py`, lines 120-155
- Python loop calls `ds[i]` individually for all 50k+ samples before every auto-tuning run. No vectorization, no batching. Estimated 60-120s blocking startup for typical dataset sizes.
- **Fix:** Replace with vectorized bulk load using FeatureStore's batch access API; load all features as a single numpy slice.

### High

#### PERF-H1: Pre-Batched Generator Prevents tf.data Parallelism
- **File:** `src/data/tfdata_pipeline.py`
- The training data generator yields pre-batched tensors, which prevents native `tf.data.Dataset.batch()` from applying parallel map operations. Estimated 20-40% throughput loss.
- **Fix:** Refactor generator to yield individual samples; use `dataset.batch(batch_size).map(..., num_parallel_calls=tf.data.AUTOTUNE)`.

#### PERF-H2: `train_on_batch` with 8 Keras Metrics Every Step — 5-15ms Overhead
- **File:** `src/training/trainer.py`
- `model.train_on_batch()` recomputes 8 Keras metrics on every step. Estimated 5-15ms/step overhead relative to a custom `@tf.function` train step.
- **Fix:** Use a `@tf.function(jit_compile=True)` custom training step that computes only the loss and gradients, tracking metrics separately at reduced frequency.

#### PERF-H3: `.numpy()` Calls Inside `_run_burst` Step Loop Serialize GPU Pipeline
- **File:** `src/tuning/orchestrator.py`, `_run_burst` method
- `.numpy()` calls inside the inner step loop force CPU-GPU synchronization on every step, preventing GPU pipeline overlap.
- **Fix:** Remove `.numpy()` calls from the hot path; accumulate TF tensors and convert outside the loop.

#### PERF-H4: No XLA/`@tf.function` on Any Hot-Path Function
- **Files:** `src/training/trainer.py`, `src/tuning/orchestrator.py`
- No `@tf.function` or `jit_compile=True` anywhere in training or tuning hot paths. TF2 does not enable XLA by default for `train_on_batch`. Estimated 20-60% throughput gain available.
- **Fix:** `model.compile(..., jit_compile=True)` for the training step; wrap `_run_burst` inner step in `@tf.function(jit_compile=True)`.

#### PERF-H5: Mixed Precision Not Correctly Applied Through Custom Training Step
- **File:** `src/training/trainer.py`; `src/utils/performance.py`
- `configure_mixed_precision(enabled=True)` sets global Keras policy but the tf.data pipeline cast (`cast_to_fp16`) happens in the pipeline rather than being fused with the first model layer. Explicit `tf.cast(labels, tf.float32)` on every step forces unnecessary precision upgrade. Estimated 2-3× throughput loss vs. correctly-applied fp16 on Tensor Core GPUs.
- **Fix:** Apply cast inside the model graph via `Input(dtype=tf.float32)` + internal cast in the first layer — standard Keras mixed-precision pattern.

#### PERF-H6: Full Forward Pass Per Candidate Per Tuning Iteration
- **File:** `src/tuning/orchestrator.py`
- Every tuning iteration runs a full forward pass on every candidate. For non-weight knobs (LR, label smoothing), scores don't change between weight updates — forward passes are redundant.
- **Fix:** Cache forward-pass scores for non-weight knobs; only re-score candidates after weight perturbations.

### Medium

#### PERF-M1: Small Shuffle Buffer Relative to Dataset Size
- **File:** `src/data/tfdata_pipeline.py`
- Shuffle buffer may be too small relative to dataset size, reducing training set diversity and potentially affecting convergence.
- **Fix:** Use `dataset.shuffle(buffer_size=len(dataset))` for full shuffling, or at minimum `buffer_size=10*batch_size`.

#### PERF-M2: 101-Iteration Dict Sync Loop Per Validation Batch
- **File:** `src/training/trainer.py`, lines 128-133
- `EvaluationMetrics.update()` syncs backward-compatible dict views in a Python loop over 101 threshold cutoffs after every vectorized numpy update.
- **Fix:** Make dict views lazy properties computed on demand from the numpy arrays.

#### PERF-M3: `all_y_true`/`all_y_scores` Python List Growth — O(N) Extra Copy
- **File:** `src/evaluation/test_evaluator.py`
- Prediction accumulation uses Python list `.append()` then `np.array()` conversion, creating an O(N) extra copy at `compute_metrics` time.
- **Fix:** Pre-allocate numpy arrays or use a `np.empty((n_samples,))` buffer filled incrementally.

#### PERF-M4: `_apply_class_weights` Runs on CPU numpy Per Training Step
- **File:** `src/training/trainer.py`
- Class weight computation runs outside the TF graph on CPU numpy arrays on every step, requiring CPU-GPU sync.
- **Fix:** Compute class weights as TF constants at training start; use `tf.gather` inside the training step.

#### PERF-M5: `create_training_pipeline_with_spec_augment` Mutates Instance State
- **File:** `src/data/tfdata_pipeline.py`, lines 361-370
- Temporarily mutates `self.spec_augment_config["enabled"]` — thread-safety hazard if pipeline object is shared.
- **Fix:** Pass SpecAugment config as immutable parameter to pipeline creation methods.

#### PERF-M6: Full Weight Copy Per Candidate Per Tuning Iteration (~32 MB/iteration)
- **File:** `src/tuning/population.py`, `src/tuning/orchestrator.py`
- `pickle.dumps(model.get_weights())` copies all model weights per candidate per iteration. For 4 candidates × 100 iterations this is ~12.8 GB of data copied.
- **Fix:** Use memory-mapped weight snapshots or `model.save_weights()` to temp files with lazy restore.

#### PERF-M7: `ThresholdOptimizer.optimize` Re-sweeps All Thresholds Per Candidate
- **File:** `src/tuning/metrics.py`
- Full 3-pass threshold optimization runs per candidate per evaluation. ~400M comparisons total across a full tuning run.
- **Fix:** Pre-compute threshold sweep once per evaluation pass; share across candidates.

#### PERF-M8: `_compute_operating_points` Nested Threshold Sweep — 404 FAH Calls
- **File:** `src/evaluation/test_evaluator.py`
- 404 individual FAH calls per test evaluation pass.
- **Fix:** Vectorize the FAH computation; compute all thresholds in a single pass.

#### PERF-M9: Bootstrap CI — 1000-Iteration Python Loop
- **File:** `src/evaluation/test_evaluator.py`
- Bootstrap confidence intervals computed in a pure Python loop.
- **Fix:** Vectorize using numpy broadcasting: `np.random.choice(n, size=(1000, n), replace=True)` then batch-compute metrics.

#### PERF-M10: Sequential FeatureStore Loop in `_load_raw_labels`
- **File:** `src/evaluation/test_evaluator.py`
- Estimated 1-5s per test evaluation for large validation sets.
- **Fix:** Batch label loading using slice access.

#### PERF-M11: CuPy SpecAugment Forces CPU↔GPU Round Trip (2-5ms/batch)
- **File:** `src/training/trainer.py`, `src/data/spec_augment_gpu.py`
- CuPy SpecAugment path pulls data from GPU, transfers to CuPy, back to numpy, then re-uploads to TF. TF backend avoids all transfers.
- **Fix:** Deprecate the CuPy SpecAugment path for the tf.data pipeline; use TF backend exclusively.

#### PERF-M12: `setup_gpu_environment()` Called at Module Import — Side Effect
- **File:** `src/utils/performance.py`
- GPU initialization runs as a module-level side effect, making imports slow and unpredictable.
- **Fix:** Move GPU setup to explicit initialization call in entry points.

#### PERF-M13: `TrainingProfiler` Uses `cProfile` Not TF Profiler
- **File:** `src/training/profiler.py`
- `cProfile` cannot profile GPU operations; TF/GPU bottlenecks are invisible. Python overhead appears dominant even when GPU is the bottleneck.
- **Fix:** Integrate TensorFlow Profiler (`tf.profiler.experimental`) for GPU-aware profiling.

#### PERF-M14: `RaggedMmap.append` Opens 3 Files Per Call
- **File:** `src/data/dataset.py`
- Individual file opens per append operation — slow during preprocessing of large datasets.
- **Fix:** Keep file handles open for the duration of a bulk-write session; close explicitly.

### Low
- **PERF-L1:** Double GPU prefetch buffer wastes ~50 MB VRAM unnecessarily
- **PERF-L2:** `compute_roc_pr_curves` called twice per test evaluation (should reuse result from `_compute_advanced_metrics`)
- **PERF-L3:** BN freeze/unfreeze traverses full layer graph per burst call (800 total traversals) — cache BN layer references after model creation
- **PERF-L4:** `list.pop(0)` in profiler history deque is O(N) — use `collections.deque(maxlen=N)`
- **PERF-L5:** `import math` inside a per-step function

---

## Critical Issues for Phase 3 Context

The following findings directly impact testing and documentation requirements:

1. **PERF-C1** (`_load_data` sequential loop) and **PERF-H2** (`train_on_batch` overhead) — any test of the auto-tuning loop will be extremely slow without mocking `_load_data`; integration tests likely need dataset size caps
2. **SEC-H1** (pickle in population.py) — needs a dedicated security test verifying that `weights_bytes` is never written to disk
3. **SEC-H3** (global Ruff suppression) — the security linting gaps mean the test suite's linting step provides false confidence
4. **PERF-M5** (mutable SpecAugment config) — any concurrent pipeline usage in tests could produce non-deterministic results
5. **Multiple calibration/metrics bugs** (CQ-C3, PERF finding 7.3) — the test suite likely has passing tests that test the wrong behavior (raw labels instead of binary labels in calibration)

---

## Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Security | 0 | 3 | 6 | 5 | 14 |
| Performance | 1 | 6 | 14 | 5 | 26 |
| **Total** | **1** | **9** | **20** | **10** | **40** |
