# PR #3 — Still-Valid Proposals Report

Cross-referenced all review bot comments (Qodo, Kilo, Copilot, Gemini, CodeRabbit) against the current codebase.

## ✅ Already Fixed (No Action Needed)

| # | Issue | File | Status |
|---|-------|------|--------|
| 1 | `total_time` not defined in `log_completion` | `trainer.py` | Fixed |
| 2 | `eval_step_interval` called on undefined `global_step` | `trainer.py` | Fixed |
| 3 | `frontend.extract` used instead of `compute_mel_spectrogram` | `features.py` | Fixed |
| 4 | `self.flatten` not initialized in `MixedNet` | `architecture.py` | Fixed |
| 5 | Unnecessary `tf.expand_dims` in `MixConvBlock` | `architecture.py` | Fixed |
| 6 | Redundant `FAH` calculation in `Trainer.__init__` | `trainer.py` | Fixed |
| 7 | Duplicate `self.logger` init in `Trainer` | `trainer.py` | Fixed (mostly) |
| 8 | Deprecated `spectrogram_length` in `MixedNet` config | `trainer.py` | Fixed |
| 9 | Typo in `DEFAULT_TENSOR_ARENA_SIZE` | `manifest.py` | Fixed |
| 10 | Hardcoded `16000` sample rate check | `features.py` | Fixed |
| 11 | Duplicate `pointwise_filters` key | `max_quality.yaml` | Fixed |
| 12 | `val_data_factory` vs `val_data` generator exhaustion | `trainer.py` | Fixed (handles both) |
| 13 | `HardExampleMiner.mine_from_dataset` factory handling | `miner.py` | Fixed (handles both) |
| 14 | `*.egg-info` removed from `.gitignore` | `.gitignore:23` | ✅ Present |
| 15 | `qp.get("scales")` numpy truth-value error | `model_analyzer.py:165,186` | ✅ Fixed (explicit `is not None` + `len()`) |
| 16 | Dynamic tensor `-1` dim handling | `manifest.py:137-148` | ✅ Fixed (warns + uses `d=1`) |
| 17 | Tensor arena comment/value mismatch | `manifest.py:13-15` | ✅ Fixed (26080 bytes, comment matches) |

---

## 🔴 Critical Bugs (Still Open)

### 1. `build_model()` default `mixconv_kernel_sizes` string is unparseable
**File:** [architecture.py:581](file:///home/sarpel/mww/microwakeword_trainer/src/model/architecture.py#L581)
**Reviewers:** Qodo
**Issue:** Default `"[5],[9],[13],[21]"` is parsed by `ast.literal_eval` as a tuple of lists, BUT it works coincidentally because `parse_model_param` handles tuples. However, the `spectrogram_slices_dropped()` function (line 93) uses the same string with `parse_model_param` and it also works. **Verdict: Works but fragile** — wrapping in `"[[5],[9],[13],[21]]"` would be safer and more explicit.

### 2. `tf.lite.experimental.Analyzer` is deprecated/removed
**File:** [model_analyzer.py:41, 413](file:///home/sarpel/mww/microwakeword_trainer/src/export/model_analyzer.py#L41)
**Reviewers:** Qodo
**Issue:** `tf.lite.experimental.Analyzer.analyze()` is deprecated in TF 2.16+, removed in later versions. Will cause `AttributeError` at runtime. Should be replaced with `ai_edge_litert.Interpreter`-based analysis or wrapped in try/except.

### 3. `compute_recall_at_no_faph()` iterates low→high instead of high→low
**File:** [metrics.py:162-183](file:///home/sarpel/mww/microwakeword_trainer/src/evaluation/metrics.py#L162-L183)
**Reviewers:** Qodo
**Issue:** Iterates thresholds from 0→1 and returns at the **lowest** threshold with zero FP. Correct behavior should find the **highest** threshold with zero FP (iterate 1→0), which gives the most useful/conservative recall value.

### 4. `val_data_factory` passed as factory to `mine_from_dataset` (expects iterable)
**File:** [trainer.py:691](file:///home/sarpel/mww/microwakeword_trainer/src/training/trainer.py#L691)
**Reviewers:** CodeRabbit
**Issue:** `mine_from_dataset(self.model, val_data_factory, ...)` passes the factory function, but the miner expects an iterable to `for … in`. Should call `val_data_factory()` to produce the iterator.

---

## 🟠 Important Fixes (Still Open)

### 5. `ResidualBlock.get_config()` stores enum object for `mode`
**File:** [architecture.py:339](file:///home/sarpel/mww/microwakeword_trainer/src/model/architecture.py#L339)
**Reviewers:** Qodo
**Issue:** `self.mode` is a `Modes` enum stored directly in config dict. Keras serialization may fail. Should use `self.mode.value`. Same issue in `MixConvBlock.get_config()` (line 237) and `MixedNet.get_config()` (line 563).

### 6. `Console(force_terminal=False)` disables progress bars
**File:** [rich_logger.py:32](file:///home/sarpel/mww/microwakeword_trainer/src/training/rich_logger.py#L32)
**Reviewers:** CodeRabbit
**Issue:** `force_terminal=False` prevents terminal detection, causing progress bars to be permanently disabled. Should use `Console()` (default) or `Console(force_terminal=True)`.

### 7. `ThreadPoolExecutor.shutdown(wait=True)` in `__del__`
**File:** [augmentation.py:265](file:///home/sarpel/mww/microwakeword_trainer/src/training/augmentation.py#L265)
**Reviewers:** Copilot
**Issue:** Blocking `shutdown(wait=True)` in `__del__` can hang during GC/interpreter shutdown. Use `wait=False` or provide explicit `close()`/context-manager API.

### 8. `_resolve_path()` heuristic too broad for HuggingFace IDs
**File:** [loader.py:506](file:///home/sarpel/mww/microwakeword_trainer/config/loader.py#L506)
**Reviewers:** Copilot
**Issue:** `x/y` regex to skip HuggingFace model IDs also matches relative paths like `dataset/positive`, causing those paths to skip resolution. Needs tightening (e.g., check if path exists on disk first).

### 9. `setup_gpu_environment()` ordering not enforced
**File:** [performance.py](file:///home/sarpel/mww/microwakeword_trainer/src/utils/performance.py)
**Reviewers:** CodeRabbit
**Issue:** `configure_tensorflow_gpu()`, `configure_mixed_precision()`, and `set_threading_config()` import TF without ensuring `setup_gpu_environment()` ran first. CUDA env vars may be ineffective.

### 10. FAH input validation missing
**File:** [fah_estimator.py](file:///home/sarpel/mww/microwakeword_trainer/src/evaluation/fah_estimator.py)
**Reviewers:** CodeRabbit
**Issue:** `compute_fah_metrics` doesn't validate `y_true`/`y_scores` shape match or that `duration_hours > 0`. Shape mismatch causes silent broadcasting bugs; `duration_hours <= 0` silently returns 0.

### 11. Hard negative `max_samples` not global-aware
**File:** [hard_negatives.py](file:///home/sarpel/mww/microwakeword_trainer/src/data/hard_negatives.py)
**Reviewers:** CodeRabbit
**Issue:** `max_samples` only truncates per-round candidates, not accounting for existing files. Total can exceed the configured limit across multiple mining rounds.

### 12. Empty batch in mining → `np.concatenate` crash
**File:** [hard_negatives.py](file:///home/sarpel/mww/microwakeword_trainer/src/data/hard_negatives.py)
**Reviewers:** CodeRabbit
**Issue:** If no batches arrive, `np.concatenate(...)` raises. Should return `{"num_hard_negatives": 0}` safely.

---

## 🟡 Code Quality / Refactoring (Still Open)

### 13. `num_augmentations` validation in `augment_batch()`
**File:** [augmentation.py:227](file:///home/sarpel/mww/microwakeword_trainer/src/training/augmentation.py#L227)
**Reviewers:** CodeRabbit
**Issue:** `num_augmentations <= 0` silently produces empty/corrupted batches. Should raise `ValueError`.

### 14. `extract_wavlm_embeddings()` name vs. behavior mismatch
**File:** [clustering.py:202](file:///home/sarpel/mww/microwakeword_trainer/src/data/clustering.py#L202)
**Reviewers:** CodeRabbit
**Issue:** Function is named `extract_wavlm_embeddings` but delegates to SpeechBrain ECAPA. Should rename to `extract_ecapa_embeddings` or actually implement WavLM.

### 15. Local imports should be module-level (PEP 8)
**Files:** `augmentation.py`, `hard_negatives.py`
**Reviewers:** Gemini
**Issue:** `from src.data.ingestion import load_audio_wave` done inside methods. Move to top unless circular dep.

### 16. `compute_all_metrics()` omits calibration
**File:** [metrics.py:282-329](file:///home/sarpel/mww/microwakeword_trainer/src/evaluation/metrics.py#L282-L329)
**Reviewers:** CodeRabbit
**Issue:** Should integrate `compute_calibration_curve()`, `compute_brier_score()`, and `calibrate_probabilities()` from `calibration.py`.

### 17. `configure_tensorflow_gpu()` docstring missing `ValueError`
**File:** [performance.py](file:///home/sarpel/mww/microwakeword_trainer/src/utils/performance.py)
**Reviewers:** Copilot
**Issue:** Docstring `Raises:` only mentions `RuntimeError` but function also raises `ValueError` when both `memory_growth` and `memory_limit_mb` are set.

### 18. `rich_logger.py` docstring claims "no external dependencies"
**File:** [rich_logger.py](file:///home/sarpel/mww/microwakeword_trainer/src/training/rich_logger.py)
**Reviewers:** Copilot
**Issue:** Module depends on `rich`. Fix docstring to say "no training logic".

### 19. `-mel_bins` should be locked to 40
**File:** [features.py:44-45](file:///home/sarpel/mww/microwakeword_trainer/src/data/features.py#L44-L45)
**Reviewers:** CodeRabbit
**Issue:** `mel_bins` is configurable but the pipeline assumes 40. Consider enforcing or at least adding a runtime assert.

### 20. Sample rate check should be warning, not `ValueError`
**File:** [features.py:53](file:///home/sarpel/mww/microwakeword_trainer/src/data/features.py#L53)
**Reviewers:** CodeRabbit
**Issue:** Per docs, 16kHz validation should warn, not hard-fail.

---

## 📝 Documentation Issues (Still Open)

### 21. README: invalid `configure_tensorflow_gpu` example
**File:** [README.md:458, 720](file:///home/sarpel/mww/microwakeword_trainer/README.md#L458)
**Issue:** Shows `memory_growth=True, memory_limit_mb=8192` together, which raises `ValueError` at runtime. Show two separate examples.

### 22. README: unquoted pip version specifier
**File:** [README.md:731](file:///home/sarpel/mww/microwakeword_trainer/README.md#L731)
**Issue:** `pip install cupy-cuda12x>=13.0` — shell interprets `>=` as redirect. Should be `pip install 'cupy-cuda12x>=13.0'`.

### 23. README: "audio will be resampled" vs. strict 16kHz enforcement
**File:** README.md
**Reviewers:** Copilot
**Issue:** README says audio "will be resampled if needed", but code rejects non-16kHz. Reconcile docs with behavior.

### 24. Config loader: legacy architecture names need migration hint
**File:** [loader.py](file:///home/sarpel/mww/microwakeword_trainer/config/loader.py)
**Reviewers:** CodeRabbit
**Issue:** Validation rejects non-"mixednet" architectures with generic error. Should detect legacy names (`dnn`, `cnn`, `crnn`) and provide a helpful migration message.

### 25. Performance profiling utilities missing
**File:** [performance.py](file:///home/sarpel/mww/microwakeword_trainer/src/utils/performance.py)
**Reviewers:** CodeRabbit
**Issue:** Module is positioned as performance hub but lacks timing/profiling utilities (decorators, context managers).
