# Comprehensive Code Review Report — microwakeword_trainer

**Review Date:** 2026-03-17
**Branch:** v2.0.0
**Framework:** Python 3.10+ / TensorFlow 2.16.2 / ESPHome ESP32 deployment target

---

## Executive Summary

This codebase is a well-conceived ML training pipeline with strong domain knowledge of ESPHome hardware constraints and a thorough architectural constitution. However, the codebase is **not production-ready**: it contains at least 8 confirmed runtime crashes (3 `NameError`/`AttributeError` bugs that fire on core code paths), a broken test infrastructure reporting 0.15% coverage, no CI/CD pipeline, and multiple features documented as functional that are either dead code or unimplemented stubs. The auto-tuner (`mww-autotune`) crashes at the first iteration, `_run_burst()` never performs gradient updates, and the calibration metrics computation silently overwrites correct results with wrong ones when hard-negative labels are present. Addressing the P0 issues below — particularly the runtime crashes, the `tests/` gitignore entry, and the untyped configuration system — should be the immediate priority before any further feature work.

---

## Findings by Priority

### P0 — Critical: Must Fix Immediately

**8 Critical findings across all categories**

#### [CQ-C1] Double `heapq.heapreplace` — Hard-Negative Mining Heap Corruption
- **File:** `src/training/mining.py`, lines 243-250
- `heapq.heapreplace` called twice with the same `heap_entry`. The second call evicts a valid top-K entry and inserts a duplicate. Every mining pass silently produces an incorrect top-K set. The unused `evicted` variable on line 248 is a clear indicator of a bad merge.
- **Fix:** Remove the second `heapq.heapreplace(hard_negative_heap, heap_entry)` call (line 248).

#### [CQ-C2] Dead Code After `return self` in `WakeWordDataset.build()`
- **File:** `src/data/dataset.py`, lines 990-996
- Three unreachable lines after `return self`: `logger.info(...)`, `self._load_store()`, `return self`. The `_load_store()` side-effect on the cache-hit path is silently skipped. Indicates a bad merge; `_is_built = True` is only set in the live branch.
- **Fix:** Delete lines 994-996.

#### [CQ-C3] Calibration Overwrites Correct Binary Labels with Raw Labels
- **File:** `src/evaluation/test_evaluator.py`, lines 334-342
- `_compute_calibration` correctly binarizes `y_true`, then a stray mid-function string literal (orphaned docstring) appears, followed by a duplicate computation using raw `y_true` (which contains label=2 for hard negatives). The final `brier` and `curve` are computed from wrong labels. Existing tests pass only because they use `{0,1}` labels, masking the bug entirely.
- **Fix:** Remove lines 340-342 (the stray string literal and duplicate non-binary computation).

#### [AR-C1] Untyped `dict[str, Any]` Configuration — Silent Default Divergence Risk
- **Files:** `src/training/trainer.py`, `src/tuning/orchestrator.py`, `src/data/tfdata_pipeline.py`, all consumers
- Configuration flows as raw dicts with `.get("key", default)` scattered across every module. The same key (`clip_duration_ms`, `mel_bins`, `window_step_ms`) is defaulted independently in 4+ places. A config key typo or mismatched default in one module vs. another silently produces an incompatible model. The ARCHITECTURAL_CONSTITUTION mandates `mel_bins=40` and `window_step_ms=10` as immutable hardware constants but each module defaults them independently via `.get()`.
- **Fix:** Define a typed `TrainingConfig` schema (Pydantic or frozen dataclasses). Single `load_config()` that validates YAML and returns a typed object. Enforce constitution constants via `Final` annotations.

#### [PERF-C1] Sequential `ds[i]` Loop in `_load_data` — 60-120s Blocking Startup
- **File:** `src/tuning/orchestrator.py`, lines 120-155
- Python loop calls `ds[i]` for every sample individually before every auto-tuning run. For 50k+ sample datasets, this blocks startup for 60-120 seconds on every tuning run.
- **Fix:** Replace with vectorized bulk load using FeatureStore's batch access API.

#### [DOC-C1] README Claims "Production Ready" — Contradicted by 18+ Open Critical Bugs
- **File:** `README.md`, `specs/implementation_status.md`
- `**Status**: ✅ Production Ready - All features implemented` contradicted by two bug reports at the repo root documenting 18 functional bugs (6-4 critical each). Neither bug report is linked from the README. `implementation_status.md` declares 100% completion while containing a contradictory pending-retraining note.
- **Fix:** Remove "Production Ready" banner. Add "Known Issues" section with links to bug reports.

#### [DOC-C2 / BP-C1] Undocumented `NameError` in Auto-Tuner — Crashes on First Iteration
- **File:** `src/tuning/orchestrator.py`, line 236
- `knob.apply(model, candidate, self.auto_tuning_config)` — `knob` is never defined in `tune()`'s scope. `_make_knob()` exists (lines 101-112) but is never called. Every unit test uses `dry_run=True` which returns before this line, masking the bug completely. Any production auto-tuning run raises `NameError` at iteration 1.
- **Fix:** Add `knob = self._make_knob(current_knob_name)` between lines 233-234. Document in bug report.

#### [CD-H1] `tests/` Is in `.gitignore` — Test Suite May Not Be Version-Controlled
- **File:** `.gitignore`
- The entry `tests/` excludes the entire test suite from git tracking. Contributors running `git add .` never commit test files. CI would run against a missing test suite.
- **Fix:** Remove `tests/` from `.gitignore` immediately. Verify all test files are tracked.

---

### P1 — High: Fix Before Next Release

**25 High findings across all categories**

#### Code Quality
- **[CQ-H1]** `Trainer.__init__` is ~550 lines with 80+ instance variables — nearly untestable; extract `CheckpointManager`, `PhaseScheduler`, `EvaluationOrchestrator`
- **[CQ-H2]** Knob config-access boilerplate duplicated 3× — extract `_get_expert_cfg()` base class method
- **[CQ-H3]** Magic number `7` hardcoded for sampling mix arms in orchestrator and knobs — define as class constant
- **[CQ-H4]** EMA configuration checked three times redundantly — consolidate to single block
- **[CQ-H5]** `FocusedSampler` is dead code (`build_batch()` returns `None`) — remove entirely

#### Architecture
- **[AR-H1]** `max_time_frames` formula computed independently in 4+ places — derive once in `HardwareConstants` frozen dataclass
- **[AR-H2]** ARCHITECTURAL_CONSTITUTION enforcement is entirely trust-based — add runtime assertions in `build_model()` and at export boundaries
- **[AR-H3]** Monolithic `Trainer` class (12+ dependencies, 1500+ lines) — extract collaborator classes

#### Security
- **[SEC-H1]** Pickle deserialization in `population.py` — if tuning state is persisted, arbitrary code execution risk; global Ruff `S301` suppression masks this from automated scans
- **[SEC-H2]** `np.load(allow_pickle=True)` on files from world-writable `/tmp` in `clustering.py:1162`
- **[SEC-H3]** Ruff globally suppresses `S301`, `S603/S607`, `S105-S107` — new vulnerable code is never flagged

#### Performance
- **[PERF-H1]** Pre-batched generator prevents tf.data parallel map — 20-40% throughput loss
- **[PERF-H2]** `train_on_batch` with 8 Keras metrics per step — 5-15ms/step overhead vs. custom `@tf.function` step
- **[PERF-H3]** `.numpy()` calls inside `_run_burst` step loop — serializes GPU pipeline
- **[PERF-H4]** No XLA/`@tf.function` on any hot-path function — 20-60% throughput gain available
- **[PERF-H5]** Mixed precision not correctly applied through training step — 2-3× throughput loss on Tensor Core GPUs
- **[PERF-H6]** Full forward pass per candidate per tuning iteration for non-weight knobs

#### Testing
- **[TC-H1]** No test that `weights_bytes` is never written to disk
- **[TC-H2]** `if result:` silences all assertions in 7 evaluator tests — silent pass when code is broken
- **[TC-H3]** No regression tests for any of the three known critical bugs
- **[TC-H4]** Core mining algorithm and knob `apply()` methods untested behaviorally

#### Documentation
- **[DOC-H1]** `FocusedSampler` and `ErrorMemory` documented as functional components in AGENTS.md and TROUBLESHOOTING.md — both are dead stubs
- **[DOC-H2]** Auto-tuner config schema (AGENTS.md) diverges from actual dataclass; unknown YAML keys silently dropped — undisclosed
- **[DOC-H3]** `_run_burst()` is a stub appending `0.0` — no gradient updates ever occur — undisclosed in README or AGENTS.md
- **[DOC-H4]** CuPy SpecAugment backend has a confirmed `IndexError` crash; README recommends it for "5-10x speedup" without disclosing the crash

#### Best Practices / CI
- **[BP-H1]** Dual `setup.py` + `pyproject.toml` with divergent metadata — `pip install .` installs broken package
- **[BP-H2]** `model.train_on_batch` — legacy Keras pattern, cannot be XLA-compiled
- **[BP-H3]** Deprecated `tf.data.experimental.prefetch_to_device`
- **[BP-H4]** `MixConvBlock.__init__` incomplete — `self.filters` never assigned (`AttributeError` in `build()`); `mode` setter has infinite recursion
- **[BP-H5]** Dead async validation infrastructure — `ThreadPoolExecutor` consumes a background thread but is never used
- **[CD-H2]** `.gitignore` missing all ML artifacts, checkpoints, model files, `.env` — sensitive data could be committed
- **[CD-H3]** Dependency pinning inconsistency — `scipy` differs between requirements files; no `pip-audit`; no Dependabot

---

### P2 — Medium: Plan for Next Sprint

**66 Medium findings** — key items:

- [CQ-M1] `pipeline.py` uses `sys.exit()` instead of exceptions — prevents programmatic composition
- [CQ-M4] `_run_burst` 140 lines, deep nesting — extract `_sample_minibatch()`, `_gradient_step()`
- [CQ-M6] `batch_features_cache` grows unboundedly in mining — evict non-heap entries during loop
- [AR-M1] Pipeline orchestrates via subprocess instead of Python API — multiple TF re-imports per run
- [AR-M3] `Knob.apply()` parameters all `Any`; candidate attributes are monkey-patched without interface
- [AR-M5] Dual SpecAugment backends; `create_training_pipeline_with_spec_augment` mutates instance state (thread-safety hazard)
- [SEC-M2] Temp directories in world-writable `/tmp` for ML models — model poisoning risk on shared systems
- [SEC-M5] Insufficient `.gitignore` coverage *(now elevated to P1 as CD-H2)*
- [PERF-M2] 101-iteration dict sync loop per validation batch
- [PERF-M9] Bootstrap CI uses 1000-iteration Python loop — vectorize with numpy broadcasting
- [PERF-M13] `TrainingProfiler` uses `cProfile` not TF Profiler — GPU bottlenecks are invisible
- [TC-M2] Training integration test is a placeholder (`assert True`)
- [TC-C1] Coverage report 0.15% — fix test runner configuration *(elevated to P0 as CD-C2)*
- [DOC-M1] Version numbers inconsistent across 6 documents (2.0.0 vs 2.1.0)
- [DOC-M2] Pipeline exit code semantics documented only in one module docstring
- [DOC-M5] Bug reports not cross-referenced from any primary document
- [BP-M4] Mypy leniency settings skip unannotated function bodies — misses real type bugs
- [BP-M5] Dual mypy config — `mypy.ini` silently shadows `pyproject.toml`
- [BP-M6] Dead async validation infrastructure *(also BP-H5)*
- [CD-M1] `.pre-commit-config.yaml` does not exist — `make install-dev` fails
- [CD-M2] Logging has no rotation, no structured JSON output, no `LOG_LEVEL` env var
- [CD-M3] Split virtualenv architecture undocumented and unenforceable; no `.env.example`

---

### P3 — Low: Track in Backlog

**38 Low findings** — key items:

- [CQ-L1] Mixed old/new type annotation styles (`Dict` vs `dict`, `Optional` vs `X | None`)
- [CQ-L2] f-strings in logger calls — use `%`-style or add Ruff `"G"` rule
- [CQ-L5] `_file_content_hash` uses SHA-1; `compute_file_hash` uses MD5 — standardize on SHA-256
- [SEC-L1/L2] MD5 and SHA-1 used for file hashing — inconsistent with SHA-256 elsewhere
- [PERF-L3] BN freeze/unfreeze traverses full layer graph on every burst — cache layer references
- [BP-L1] `src/config/__init__.py` empty with misleading docstring
- [BP-L3] `PrefetchGenerator` docstring example advertises deprecated `train_on_batch`
- [DOC-L1] Dataclass count inconsistent (13 vs 14 vs 15) across documents
- [DOC-L3] `specs/phase1_complete.yaml` referenced but may not exist
- [CD-L1] `line-length = 200` across all formatters — reduce to 120

---

## Findings by Category

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Code Quality | 3 | 5 | 8 | 6 | **22** |
| Architecture | 1 | 3 | 8 | 5 | **17** |
| Security | 0 | 3 | 6 | 5 | **14** |
| Performance | 1 | 6 | 14 | 5 | **26** |
| Testing | 3 | 4 | 5 | 3 | **15** |
| Documentation | 2 | 4 | 6 | 4 | **16** |
| Framework/Language | 1 | 5 | 6 | 5 | **17** |
| CI/CD & DevOps | 2 | 3 | 4 | 1 | **10** |
| **TOTAL** | **13** | **33** | **57** | **34** | **137** |

*Note: Several findings appear in multiple categories (e.g., DOC-C2 = BP-C1 = the `knob` NameError); unique issues total approximately 110.*

---

## Recommended Action Plan

### Immediate (this week)

1. **[CD-H1]** Remove `tests/` from `.gitignore` and verify all test files are git-tracked — single highest-risk line in the repo
2. **[DOC-C2/BP-C1]** Fix the `NameError` in `orchestrator.py:236` — add `knob = self._make_knob(current_knob_name)` — medium effort, unblocks any real auto-tuning use
3. **[CQ-C1]** Remove the second `heapq.heapreplace` call in `mining.py:248` — 1-line fix
4. **[CQ-C2]** Delete dead code after `return self` in `dataset.py:993-996` — 3-line deletion
5. **[CQ-C3]** Remove duplicate calibration computation lines 340-342 in `test_evaluator.py` — 3-line deletion
6. **[BP-H4]** Fix `MixConvBlock.__init__` — assign all instance attributes, fix setter recursion — small, prevents `AttributeError` crash in model building
7. **[DOC-C1]** Remove "Production Ready" from README; add "Known Issues" with bug report links

### Short-term (next sprint)

8. **[AR-C1]** Begin typed config system — define `HardwareConfig` and `TrainingConfig` dataclasses; replace the most dangerous `.get()` defaults (mel_bins, window_step_ms, clip_duration_ms) — large but highest-leverage architectural change
9. **[CD-C1]** Create `.github/workflows/ci.yml` (even 20 lines: lint + unit tests) and `.pre-commit-config.yaml` — blocks all future regressions from merging
10. **[TC-C2/TC-H3]** Add regression tests for the three critical bugs — prevents reintroduction
11. **[TC-H2]** Fix `if result:` pattern in `test_test_evaluator.py` — replace with `assert result is not None`
12. **[SEC-H1]** Replace pickle in `population.py` with `numpy.savez`/`numpy.load(allow_pickle=False)`
13. **[SEC-H2]** Fix `allow_pickle=True` in `clustering.py:1162`
14. **[BP-H1]** Delete `setup.py`; consolidate into `pyproject.toml`
15. **[DOC-H3]** Document `_run_burst()` as a stub; update README description of `mww-autotune`

### Medium-term (next 2-3 sprints)

16. **[PERF-C1]** Vectorize `_load_data` in orchestrator — replace `ds[i]` loop with bulk load — large performance impact
17. **[PERF-H2/H4]** Replace `train_on_batch` with `@tf.function(jit_compile=True)` custom train step — largest throughput lever
18. **[AR-H2]** Add `constitution_check()` assertions in `build_model()` and at export boundaries
19. **[CQ-H1/AR-H3]** Begin `Trainer` decomposition — extract `CheckpointManager` as first collaborator
20. **[CD-M1]** Create `.pre-commit-config.yaml` and `.env.example`; fix `make install-dev`
21. **[BP-M4/M5]** Delete `mypy.ini`; consolidate mypy config; enable `check_untyped_defs = true`
22. **[DOC-M1]** Standardize version numbers across all documents; add `VERSION` file

### Backlog

23. Standardize type annotations on PEP 585/604 (replace `typing` imports)
24. Replace MD5/SHA-1 with SHA-256 throughout
25. Vectorize bootstrap CI computation in test evaluator
26. Add `RotatingFileHandler` and structured JSON logging
27. Add SBOM generation (`cyclonedx-bom`) to build pipeline
28. `line-length = 200` → 120

---

## Review Metadata

- **Review date:** 2026-03-17
- **Phases completed:** Phase 1 (Code Quality + Architecture), Phase 2 (Security + Performance), Phase 3 (Testing + Documentation), Phase 4 (Framework + CI/CD)
- **Agents used:** 8 specialized review agents
- **Flags applied:** none (default mode)
- **Total findings:** ~137 (with cross-category deduplication: ~110 unique issues)
