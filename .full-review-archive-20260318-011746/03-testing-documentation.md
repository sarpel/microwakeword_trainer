# Phase 3: Testing & Documentation Review

## Test Coverage Findings

### Critical

#### TC-C1: Coverage Report Is 0.15% — CI Coverage Gating Is Meaningless
- **File:** `coverage.xml`
- Coverage run is misconfigured — only 18 of 11,747 lines hit, all in `export/` helpers. The entire `src/training/`, `src/data/`, `src/evaluation/`, `src/tuning/`, and `src/pipeline.py` show zero hits. Any coverage gates in CI are providing false confidence.
- **Fix:** Run `pytest --cov=src --cov=scripts tests/` with all marker groups included.

#### TC-C2: `mine_from_dataset()` Completely Untested — Double-heapreplace Bug Undetected
- **File:** `tests/unit/test_training_mining_core.py`
- `test_training_mining_core.py` covers only file utilities and helpers. **No test exists for `mine_from_dataset()`** — the only code path that exercises the top-K heap. The double-heapreplace critical bug is therefore entirely undetected by the test suite. Also missing: `AsyncHardExampleMiner` (zero tests for thread safety), `run_top_fp_extraction()`, `consolidate_prediction_logs()`, CLI subcommands.
- **Fix:** Add heap regression test verifying top-2 entries contain known highest-scoring samples.

#### TC-C3: `_compute_calibration` Bug Actively Tested Against Wrong Behavior
- **File:** `tests/unit/test_test_evaluator.py`
- Existing tests use only `{0, 1}` labels — the calibration overwrite bug only triggers when hard negatives (label=2) are present. Tests pass with broken code and provide false confidence. No test exercises the broken path.
- **Fix:** Add test with `y_true = np.array([1, 0, 2, 1, 0])` and assert Brier score equals the binary-label computation.

### High

#### TC-H1: No Test that `weights_bytes` Is Never Written to Disk
- **File:** `src/tuning/population.py`, `src/tuning/dashboard.py`
- `save_artifacts()` serializes candidates to JSON. No test asserts `weights_bytes` (raw pickle bytes) is excluded from output. If included, it creates a deserialization attack surface.
- **Fix:** Add assertion that `"weights_bytes"` is absent from `tuning_artifacts.json`.

#### TC-H2: `if result:` Pattern Silences All Assertions in 7 Evaluator Tests
- **File:** `tests/unit/test_test_evaluator.py`
- Seven of nine tests wrap all assertions in `if result:` — a no-op if `evaluate()` returns `None`. The entire test body silently passes when the code under test is broken.
- **Fix:** Replace with unconditional `assert result is not None` followed by unconditional assertions.

#### TC-H3: No Regression Tests for Any Known Bug
Three known bugs have zero regression test coverage:

| Bug | Test Status |
|-----|-------------|
| Double `heapreplace` `mining.py:248` | No test for `mine_from_dataset()` at all |
| Calibration label overwrite `test_evaluator.py:341` | Existing tests use only `{0,1}` labels; bug only triggers with label=2 |
| Dead code after `return self` `dataset.py:993` | No test for cache-hit path of `build()` |

#### TC-H4: Core Mining and Tuning Internals Untested Behaviorally
- `test_tuning_knobs.py`: Only checks class existence and `name` attribute; no test verifies what values `LRKnob.apply()`, `TemperatureKnob.apply()`, or `SamplingMixKnob.apply()` actually set.
- `test_tuning_orchestrator.py`: All orchestrator tests monkeypatch `_run_burst` away; no test validates the actual gradient update step end-to-end.

### Medium

#### TC-M1: `dataset.py` Cache-Hit Path Untested
- No test verifies `_load_store()` is called when `WakeWordDataset.build()` finds a valid cache. The dead code at lines 993-996 means this side effect is silently skipped with no test catching it.

#### TC-M2: Training Integration Test Is a Placeholder (`assert True`)
- **File:** `tests/integration/test_training.py`
- The file contains only `assert True`. Provides zero coverage of the actual training loop.

#### TC-M3: No Dataset-Size Caps or Time Budgets in Integration Tests
- `_load_data` takes 60-120s on real data. Integration tests have no `max_samples` cap or time budget assertion, risking indefinite CI hangs.

#### TC-M4: FAH Formula Correctness Not Numerically Verified
- Tests use toy 4-8 sample arrays; assertions like `assert fah <= 2.0` are very weak. The exact formula `FAH = FP_count / ambient_hours` is never verified against a known ground truth.

#### TC-M5: `_run_burst` Contract Not Verified End-to-End
- `_run_burst` is monkeypatched in all orchestrator tests. No test validates the actual gradient update step. (Note: documentation review reveals `_run_burst` always appends `0.0` and never performs real gradient updates — meaning there is nothing to test, but the stub behavior is also undisclosed.)

### Low

- **TC-L1:** No shared model fixture — repeated TF graph construction adds 100-300ms overhead per test class
- **TC-L2:** Global `np.random.seed(42)` bleeds state between tests; `assert 0.5 <= auc <= 1.0` is wrong for random data (should be `0.0 <= auc <= 1.0`)
- **TC-L3:** `tf.random` monkeypatching in `test_run_burst_uses_uniform_sampling_not_shuffle` is unsafe under `pytest-xdist` parallel execution

---

## Documentation Findings

### Critical

#### DOC-C1: README Claims "Production Ready" — Contradicted by 12+ Open Critical Bugs
- **File:** `README.md`, `specs/implementation_status.md`
- README opens with `**Status**: ✅ Production Ready - All features implemented`. Two bug reports at the repo root (`BUG_REPORT.md`, `BUG_REPORT_2026-03-16.md`) document 18 open functional bugs with 6-4 critical severity each. Neither bug report is referenced from the README.
- `implementation_status.md` simultaneously declares `Overall Completion: 100%` and contains a contradictory note at the bottom: "current model exports require retraining after architecture alignment."
- **Fix:** Remove "Production Ready" banner. Add "Known Issues" section linking to bug reports. Promote the pending-retraining note to a top-level section.

#### DOC-C2: Undocumented `NameError` in Auto-Tuner — Crashes on First Iteration
- **File:** `src/tuning/orchestrator.py`, line 236
- `knob.apply(model, candidate, self.auto_tuning_config)` references `knob` which is never defined in `tune()`'s scope. The method `_make_knob(current_knob_name)` exists but is never called. The auto-tuner will raise `NameError: name 'knob' is not defined` at the first iteration of any non-dry-run execution. **This bug does not appear in either bug report and is completely undocumented.**
- **Fix:** Add `knob = self._make_knob(current_knob_name)` between lines 233-234. Document in `BUG_REPORT.md`.

### High

#### DOC-H1: `FocusedSampler` and `ErrorMemory` Documented as Functional — Both Are Dead Code
- **Files:** `src/tuning/AGENTS.md`, `docs/TROUBLESHOOTING.md`, `docs/CONFIGURATION.md`
- `src/tuning/AGENTS.md` lists both as key components. `docs/TROUBLESHOOTING.md` lines 793-800 describes `FocusedSampler`/`ErrorMemory` architecture as if implemented ("FocusedSampler memorizes search data distribution," "ErrorMemory feedback loop"). These describe the *old* autotuner; the redesigned `MicroAutoTuner` has both as stubs.
- **Fix:** Mark as "stub/reserved" in AGENTS.md. Add note to TROUBLESHOOTING.md that this describes the previous autotuner design. Remove from CONFIGURATION.md's `search_eval_fraction` description.

#### DOC-H2: Auto-Tuner Config Schema Diverges from Implementation
- **Files:** `src/tuning/AGENTS.md`, `src/config/loader.py`
- AGENTS.md documents `micro_burst_steps` and `max_no_improve` as canonical field names. The actual `AutoTuningExpertConfig` dataclass uses `min_burst_steps`, `max_burst_steps`, `default_burst_steps`, and `patience`. The alias/normalization relationship is undocumented.
- Critical: unknown YAML keys are **silently dropped** by `loader.py` line 972 — this behavior is not documented anywhere user-facing.
- **Fix:** Update AGENTS.md Config Fields table to match the actual dataclass. Document the silent-drop behavior with a warning note in `docs/CONFIGURATION.md`.

#### DOC-H3: `_run_burst()` Is a Stub — No Gradient Updates — Undisclosed
- **File:** `src/tuning/orchestrator.py`, lines 153-187
- The burst loop body is `losses.append(0.0)` — no actual training occurs. The model weights are never updated during any tuning iteration. The README describes `mww-autotune` as "adjusts probability thresholds and hyperparameters through iterative fine-tuning" — this is inaccurate.
- **Fix:** Add docstring explicitly stating this is a stub. Update AGENTS.md and README to reflect that gradient bursts are currently placeholder.

#### DOC-H4: Dual SpecAugment Backends — Relationship and CuPy Crash Bug Undisclosed
- **Files:** `README.md`, `docs/INDEX.md`, `docs/TRAINING.md`
- README recommends CuPy SpecAugment for "5-10x speedup" without disclosing the confirmed `IndexError` crash bug (documented in both bug reports). docs/INDEX.md calls TF backend the "fallback" but the TF backend is actually the one integrated into the training pipeline.
- **Fix:** Add a documentation section comparing the two backends, stating CuPy's known crash, and which is active in the current pipeline.

### Medium

#### DOC-M1: Version Numbers Inconsistent Across 6 Documents
- README: `v2.1.0`; MASTER_GUIDE: `2.0.0`; TROUBLESHOOTING: `v2.0`; INDEX.md: `2.1.0 | Branch: consolidation`; git branch: `v2.0.0`
- **Fix:** Standardize on single version string; add `VERSION` file at repo root; update all docs to match.

#### DOC-M2: Pipeline Exit Code Semantics Invisible to Users
- Exit codes 0/1/2/3 documented only in `src/pipeline.py` module docstring. README, MASTER_GUIDE, docs/INDEX.md all omit them.
- **Fix:** Add "Exit Codes" section to README or MASTER_GUIDE; create `docs/CLI_REFERENCE.md` covering all five CLI tools.

#### DOC-M3: ARCHITECTURAL_CONSTITUTION Has No Enforcement Documentation
- The Constitution mandates immutable constants but provides no information on which are validated at load time, which have CI tests, and which rely on code review only.
- **Fix:** Add "Enforcement" appendix listing: (a) ConfigLoader-validated invariants, (b) test file coverage per invariant, (c) honor-system-only invariants.

#### DOC-M4: `SamplingMixKnob` 7-Arm Magic Number Undocumented
- The 7-arm bandit structure in `knobs.py:104` has no comment, docstring, or documentation explaining the 7 arms, what each represents, or that they are currently a no-op (each arm appends `0.0`).
- **Fix:** Add class-level comment explaining the arm structure is a placeholder from the original design.

#### DOC-M5: Bug Reports Not Cross-Referenced from Any Primary Document
- `BUG_REPORT.md` and `BUG_REPORT_2026-03-16.md` are not linked from README, MASTER_GUIDE, or docs/INDEX.md. AGENTS.md claims "all 11 CRITICAL issues from the 2026-03-16 audit have been resolved" — contradicted by the still-open bug reports.
- **Fix:** Add "Known Issues" with links to bug reports in README. Reconcile the "all resolved" claim in AGENTS.md.

#### DOC-M6: `_evaluate_candidate()` Misleadingly Named — Ignores Model
- **File:** `src/tuning/orchestrator.py`, lines 114-124
- Method receives `model` as argument but never calls it. Computes `recall = positives / total_samples` and `fah = negatives / total_samples` — partition statistics, not model-driven metrics. No docstring discloses this.
- **Fix:** Add docstring stating this computes partition statistics and does not run model inference.

### Low

- **DOC-L1:** Dataclass count stated as 13, 14, or 15 across different documents (actual count: 15)
- **DOC-L2:** `mww-mine` vs `mww-mine-hard-negatives` CLI name mismatch between docs
- **DOC-L3:** `specs/phase1_complete.yaml` referenced in INDEX.md and implementation_status.md — verify file exists
- **DOC-L4:** `docs/INDEX.md` shows `Branch: consolidation` and `Generated: 2026-03-06` — stale metadata

---

## Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Testing | 3 | 4 | 5 | 3 | 15 |
| Documentation | 2 | 4 | 6 | 4 | 16 |
| **Total** | **5** | **8** | **11** | **7** | **31** |
