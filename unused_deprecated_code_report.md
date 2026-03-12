# Unused & Deprecated Code Report

**Generated:** 2026-03-12  
**Project:** microwakeword_trainer  
**Scope:** src/, scripts/, config/ directories

---

## Executive Summary

| Category | Count | Files Affected |
|----------|-------|----------------|
| Deprecated Config Fields | 5+ | config/loader.py, src/training/trainer.py |
| Legacy Wrapper Functions | 4 | src/export/tflite.py, src/data/clustering.py, src/data/ingestion.py, src/data/dataset.py |
| Unused Imports | 15+ | src/training/__init__.py, src/data/quality.py, src/data/features.py, scripts/*.py |
| Unused Variables | 10+ | src/tuning/autotuner.py, src/data/tfdata_pipeline.py |
| Unused Functions/Classes | 7 | src/utils/performance.py, src/evaluation/*.py, src/training/performance_optimizer.py |
| Unreachable/Dead Code | 1 | src/pipeline.py |
| Obsolete Scripts | 7+ | scripts/ directory |
| Duplicate Implementations | 2 | src/training/trainer.py vs src/evaluation/metrics.py |

---

## 1. DEPRECATED CODE

### 1.1 Configuration Fields (config/loader.py)

| Field | Line | Status | Replacement |
|-------|------|--------|-------------|
| `auto_tune_on_poor_fah` | 93 | DEPRECATED | `auto_tuning.enabled` |
| `async_mining` (in PerformanceConfig) | 194 | DEPRECATED | `mining.async_mining` |
| `steps_per_iteration` | 429-437 | Legacy (ignored) | MaxQualityAutoTuner |
| `initial_lr` | 429-437 | Legacy (ignored) | MaxQualityAutoTuner |
| `lr_decay_factor` | 429-437 | Legacy (ignored) | MaxQualityAutoTuner |

**Code Snippet (Line 93):**
```python
auto_tune_on_poor_fah: bool = True  # DEPRECATED: use auto_tuning.enabled instead
```

**Code Snippet (Line 194):**
```python
async_mining: bool = True  # DEPRECATED: moved to MiningConfig (kept for backward compat)
```

### 1.2 Legacy Export Functions (src/export/tflite.py)

| Function | Lines | Status |
|----------|-------|--------|
| `_legacy_export_to_tflite_stage1_stub()` | 928-938 | Raises NotImplementedError |
| `export_to_tflite()` | 941-953 | Raises NotImplementedError |

**Impact:** These are legacy API stubs that should no longer be called.

### 1.3 Legacy Wrapper Functions

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `extract_wavlm_embeddings()` | src/data/clustering.py | 179-211 | Backward compatibility wrapper |
| Legacy data_dir handling | src/data/ingestion.py | 364-373 | Old single-directory support |
| Legacy __init__ path | src/data/dataset.py | 569-578 | Old API support |
| Legacy namelist handling | src/tools/cluster_apply.py | 51-52 | Old format support |

---

## 2. UNUSED CODE

### 2.1 Unused Imports (High Confidence - pyflakes detected)

| Import | File | Line | Recommendation |
|--------|------|------|----------------|
| `urllib.request` | src/data/quality.py | 7 | Remove |
| `pymicro_features` | src/data/features.py | 143 | Remove |
| `redefinition of unused 'AddBackgroundNoise'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'AddColorNoise'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'ApplyImpulseResponse'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'BandStopFilter'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'Gain'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'PitchShift'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'SevenBandParametricEQ'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `redefinition of unused 'TanhDistortion'` | src/training/augmentation.py | 22 | Remove duplicate import |
| `glob` | scripts/cleanup_tfdata_cache.py | 14 | Remove |
| `os` | scripts/cleanup_tfdata_cache.py | 15 | Remove |
| `ast` | scripts/analyze_legacy_code.py | 12 | Remove |
| `typing.Tuple` | scripts/analyze_legacy_code.py | 16 | Remove |

### 2.2 Unused Variables (High Confidence)

| Variable | File | Line | Context |
|----------|------|------|---------|
| `pos_w` | src/data/tfdata_pipeline.py | 235 | Assigned but never used |
| `neg_w` | src/data/tfdata_pipeline.py | 236 | Assigned but never used |
| `hn_w` | src/data/tfdata_pipeline.py | 237 | Assigned but never used |
| `best_threshold` | src/tuning/autotuner.py | 768 | Assigned but never used |
| `trainable_vars` | src/tuning/autotuner.py | 1518 | Assigned but never used |
| `stir_info` | src/tuning/autotuner.py | 2379 | Assigned but never used |
| `pre_swa_weights` | src/tuning/autotuner.py | 2415 | Assigned but never used |
| `func_start` | scripts/cleanup_legacy.py | 56 | Assigned but never used |
| `skip_until_next_key` | scripts/cleanup_deprecated_config.py | 56 | Assigned but never used |

### 2.3 Unused Functions & Classes (Medium Confidence)

| Symbol | File | Line | Type | Confidence |
|--------|------|------|------|------------|
| `check_gpu_and_cupy_available()` | src/utils/performance.py | 55 | Function | Medium |
| `setup_gpu_environment()` | src/utils/performance.py | 229 | Function | Medium |
| `format_bytes()` | src/utils/performance.py | 264 | Function | Medium |
| `PerformanceOptimizer` | src/training/performance_optimizer.py | 24 | Class | Medium |
| `FAHEstimator` | src/evaluation/fah_estimator.py | 11 | Class | Medium |
| `_compute_mcc()` | src/evaluation/test_evaluator.py | 26 | Function | High |
| `_compute_cohens_kappa()` | src/evaluation/test_evaluator.py | 35 | Function | High |
| `_compute_eer_manual()` | src/evaluation/test_evaluator.py | 47 | Function | High |
| `TestEvaluator` | src/evaluation/test_evaluator.py | 57 | Class | Medium |

### 2.4 Intentionally Public (from src/training/__init__.py)

These imports may be public API exports - verify before removal:

| Import | Line | Purpose |
|--------|------|---------|
| `EvaluationMetrics` | 11 | Public export |
| `Trainer` | 14 | Public export |
| `TrainingMetrics` | 17 | Public export |
| `main` | 20 | Public export |
| `train` | 23 | Public export |

---

## 3. DEAD/UNREACHABLE CODE

### 3.1 Duplicate Function Implementation (src/pipeline.py)

**Issue:** The `step_promote()` function has a duplicate implementation (lines 295-308) that is unreachable because the function returns at line 294.

**Lines 278-294 (ACTIVE):**
```python
def step_promote(self, config: Config) -> Config:
    """Promote model to production if quality threshold met."""
    if self.state.best_model_path and config.quality.min_quality_score <= self.state.best_quality_score:
        promote_path = config.paths.model_dir / "production"
        promote_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.state.best_model_path, promote_path / "model.tflite")
        logger.info(f"Promoted model to production: {promote_path}")
    return config
```

**Lines 295-308 (DEAD - unreachable):**
```python
def step_promote(self, config: Config) -> Config:
    # ... different implementation ...
    # This code is never executed
```

**Recommendation:** Remove lines 295-308.

---

## 4. OBSOLETE/UNUSED SCRIPTS

These scripts appear to have no references in the codebase and may be obsolete:

| Script | Purpose | Status | Recommendation |
|--------|---------|--------|----------------|
| `scripts/tidy_yaml.py` | YAML formatting utility | No references found | Review for removal |
| `scripts/ci.sh` | CI script | No references found | Review for removal |
| `scripts/install.sh` | Installation script | No references found | Review for removal |
| `scripts/cleanup_deprecated_config.py` | Config cleanup utility | No references found | Obsolete - remove |
| `scripts/cleanup_legacy.py` | Legacy cleanup script | No references found | Obsolete - remove |
| `scripts/cleanup_tfdata_cache.py` | TF data cache cleanup | No references found | Review for removal |
| `scripts/analyze_legacy_code.py` | Legacy analysis tool | No references found | Obsolete - remove |
| `scripts/personal/` | Personal scripts directory | No references found | Developer-specific - review |

---

## 5. DUPLICATE/REDUNDANT IMPLEMENTATIONS

### 5.1 FAH Calculation (Duplicate Logic)

| Location | Implementation |
|----------|----------------|
| `src/training/trainer.py` | `ambient_false_positives_per_hour` logic |
| `src/evaluation/metrics.py` | Similar ambient FP calculations |

**Recommendation:** Consolidate FAH calculation into `src/evaluation/metrics.py` and have `trainer.py` import it.

### 5.2 Quality Scoring

| Script | Purpose |
|--------|---------|
| `scripts/score_quality_fast.py` | Fast quality scoring |
| `scripts/score_quality_full.py` | Full quality scoring |

Both are referenced in docs, but the fast version may be redundant.

---

## 6. CONFIGURATION ISSUES

### 6.1 Partially Implemented TensorBoard Options

Some TensorBoard options in `config/loader.py` (in `PerformanceConfig`) are only used in `max_quality` preset:
- `tensorboard_log_activation_stats`
- `tensorboard_log_confidence_drift`

These appear to be defined but only partially implemented in the training code.

### 6.2 Duplicate Validation

Config validation logic is duplicated between:
- `config/loader.py` - `validate()` method
- `tests/unit/test_config.py` - test assertions

---

## 7. BUGS/ISSUES FOUND

### 7.1 Undefined Name (src/tools/cluster_apply.py:51)

```python
# Line 51-52
legacy_namelist = legacy  # 'legacy' is undefined!
```

**Status:** This appears to be a bug where `legacy` variable is used but not defined.

---

## 8. RECOMMENDED ACTIONS

### Immediate (Safe to Remove):
1. ✅ Remove duplicate `step_promote()` implementation (src/pipeline.py:295-308)
2. ✅ Remove unused imports (15+ instances)
3. ✅ Remove unused local variables (10+ instances)
4. ✅ Remove obsolete scripts: `cleanup_deprecated_config.py`, `cleanup_legacy.py`, `analyze_legacy_code.py`

### Review Before Removal:
5. 📝 Verify `src/training/__init__.py` exports are truly public API
6. 📝 Check if `check_gpu_and_cupy_available()`, `setup_gpu_environment()`, `format_bytes()` are used externally
7. 📝 Review `PerformanceOptimizer` and `FAHEstimator` classes - may be used via reflection or external tools
8. 📝 Verify obsolete scripts (`tidy_yaml.py`, `ci.sh`, `install.sh`, etc.) have no external dependencies

### Consolidate/Refactor:
9. 🔧 Consolidate FAH calculation logic from `trainer.py` into `metrics.py`
10. 🔧 Consider unifying quality scoring scripts (`score_quality_fast.py` and `score_quality_full.py`)

### Deprecation Cleanup:
11. 🧹 Run `python scripts/cleanup_deprecated_config.py --dry-run` to preview deprecated config removal
12. 🧹 Update code using deprecated fields (`auto_tune_on_poor_fah`, `performance.async_mining`) to use new equivalents
13. 🧹 Remove deprecated config fields after verifying no external usage

---

## 9. STATISTICS

| Metric | Count |
|--------|-------|
| Total files analyzed | 70+ |
| Files with issues | 20+ |
| Deprecated patterns | 9+ |
| Unused imports | 15+ |
| Unused variables | 10+ |
| Unused functions/classes | 7 |
| Dead code blocks | 1 |
| Obsolete scripts | 7+ |
| Duplicate implementations | 2 |

---

## 10. COMMANDS FOR CLEANUP

```bash
# Preview deprecated config removal
python scripts/cleanup_deprecated_config.py --dry-run

# Run pyflakes to see all unused code
pyflakes src/ scripts/ 2>/dev/null | grep -v "unable to detect undefined names"

# Verify no tests break before removal
pytest tests/ -v

# Check for external references to potentially obsolete scripts
grep -r "cleanup_deprecated_config\|cleanup_legacy\|analyze_legacy" . --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" --include="*.yml"
```

---

*Report generated by unused-code-cleaner analysis*
