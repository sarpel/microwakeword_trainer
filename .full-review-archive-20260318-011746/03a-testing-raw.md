# Phase 3A: Test Coverage Review (Raw)

## Executive Summary

The project has 40+ test files but **coverage.xml reports 0.15% line coverage** (18 of 11,747 lines hit, all in export helpers). The coverage run is misconfigured — the entire `src/training/`, `src/data/`, `src/evaluation/`, `src/tuning/`, and `src/pipeline.py` show zero hits. Three known critical bugs have no regression tests, and existing tests for two of those bugs actively validate incorrect behavior.

---

## Critical

### TC-C1: Coverage Report Is 0.15% — CI Coverage Gating Is Meaningless
- **File:** `coverage.xml`
- Coverage run was scoped to only `export/` subsystem or test execution didn't import source packages. Cannot trust any coverage numbers.
- **Fix:** Run `pytest --cov=src --cov=scripts tests/` with all marker groups included.

### TC-C2: `mine_from_dataset()` Completely Untested — Double-heapreplace Bug Undetected
- **File:** `tests/unit/test_training_mining_core.py`
- `test_training_mining_core.py` covers helper utilities but has **no test for `mine_from_dataset()`** — the only code path exercising the top-K heap. The double-heapreplace bug is entirely undetected.
- Also missing: `AsyncHardExampleMiner` (zero tests for thread safety, submit/flush contract), `run_top_fp_extraction()`, `consolidate_prediction_logs()`.
- **Fix:** Add heap regression test verifying top-K entries after mining with known scores.

### TC-C3: `_compute_calibration` Bug Actively Tested Against Wrong Behavior
- **File:** `tests/unit/test_test_evaluator.py`
- Existing tests use only `{0, 1}` labels so the calibration overwrite bug doesn't trigger — tests pass with broken code. No test exercises the path where `raw_labels` contains label=2 (hard negatives).
- **Fix:** Add test with `y_true = np.array([1, 0, 2, 1, 0])` and assert Brier score equals binary-label computation.

---

## High

### TC-H1: No Test that `weights_bytes` Is Never Written to Disk
- **File:** `src/tuning/population.py`, `src/tuning/dashboard.py`
- `save_artifacts()` serializes candidate dicts to JSON. No test asserts `weights_bytes` (raw pickle) is excluded from the output. If it were included, it would create a deserialization attack surface.
- **Fix:** Add test asserting `"weights_bytes"` and any pickle content are absent from `tuning_artifacts.json`.

### TC-H2: `if result:` Pattern Silences All Assertions in 7 Evaluator Tests
- **File:** `tests/unit/test_test_evaluator.py`
- Seven of nine tests wrap assertions in `if result:` — making them no-ops if `evaluate()` returns `None`. Tests silently pass when code under test is broken.
- **Fix:** Replace with unconditional `assert result is not None` then unconditional assertions.

### TC-H3: Core Mining Algorithm Untested (covers utilities only)
- **File:** `tests/unit/test_training_mining_core.py`
- Covered: `get_hard_samples`, `log_false_predictions_to_json`, file utilities. Missing: `mine_from_dataset()`, `AsyncHardExampleMiner`, `run_top_fp_extraction()`, CLI subcommands.

### TC-H4: No Regression Tests for Any Known Bug
- Three bugs with no regression test:

| Bug | Status |
|-----|--------|
| Double `heapreplace` in `mining.py:248` | No test for `mine_from_dataset()` |
| `_compute_calibration` label overwrite `test_evaluator.py:341` | Tests use only `{0,1}` labels; bug only triggers with label=2 |
| Dead code after `return self` in `dataset.py:993` | No test for cache-hit path |

---

## Medium

### TC-M1: `dataset.py` Cache-Hit Path Untested
- No test verifies `_load_store()` is called when `WakeWordDataset.build()` finds a valid cache. The dead code at lines 993-996 means this side effect is silently skipped.

### TC-M2: Training Integration Test Is a Placeholder
- **File:** `tests/integration/test_training.py`
- Contains `assert True` — provides no coverage of the actual training loop.

### TC-M3: No Dataset-Size Caps or Time Budgets in Integration Tests
- `_load_data` takes 60-120s on real data. Integration tests using real datasets have no `max_samples` cap or `assert elapsed < N` time budget.

### TC-M4: FAH Formula Correctness Not Numerically Verified
- **File:** `tests/unit/test_evaluation_metrics.py`
- FAH tests use toy 4-8 sample arrays; assertion `assert fah <= 2.0` is a very weak bound. The exact formula `FAH = FP_count / ambient_hours` is never verified against a known ground truth.

### TC-M5: Knob `apply()` Methods Untested Behaviorally
- **File:** `tests/unit/test_tuning_knobs.py`
- Tests only check class existence and `name` attribute. No test verifies what values `LRKnob.apply()`, `TemperatureKnob.apply()`, or `SamplingMixKnob.apply()` actually set on model/candidate.

### TC-M6: `_run_burst` Contract Not Verified
- **File:** `tests/unit/test_tuning_orchestrator.py`
- Core `_run_burst` is monkeypatched in all orchestrator tests. No test validates the actual gradient update step end-to-end with real TF operations or verifies the return dict contract.

---

## Low

### TC-L1: No Shared Model Fixture — Repeated TF Graph Construction
- `conftest.py` has no shared model/config fixtures. Each test recreates TF graphs from scratch, adding 100-300ms per test class. A session-scoped fixture would cut `test_test_evaluator.py` suite time by ~60%.

### TC-L2: Global `np.random.seed` in Test Methods
- **File:** `tests/unit/test_evaluation_metrics.py`
- Inline `np.random.seed(42)` bleeds global state between tests if ordering changes. Assertion `assert 0.5 <= auc <= 1.0` is incorrect — random data should use `0.0 <= auc <= 1.0`.

### TC-L3: `tf.random` Monkeypatching Unsafe Under Parallel Execution
- **File:** `tests/unit/test_tuning_orchestrator.py::test_run_burst_uses_uniform_sampling_not_shuffle`
- Monkeypatches `tf.random` at module level; count assertions would fail under `pytest-xdist`.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 4 |
| Medium | 6 |
| Low | 3 |
