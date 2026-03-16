# Bug Report — microwakeword_trainer Full Codebase Audit

**Date:** 2026-03-16  
**Method:** Static analysis (ruff) + parallel deep code review (3 agents across all modules) + live verification  
**Scope:** All Python source in `src/`, `config/`, `scripts/`

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 3 |
| HIGH | 9 |
| MEDIUM | 4 |
| **Total** | **16** |

---

## CRITICAL

### BUG-01 — FAH-Recall Plot Always Shows Zero Recall
**File:** `src/evaluation/test_evaluator.py`  
**Lines:** 662–666  
**Verified:** ✅

`compute_fah_metrics()` returns only three keys:
```python
{"ambient_false_positives": ..., "ambient_false_positives_per_hour": ..., "ambient_duration_hours": ...}
```

The FAH-recall operating curve plot calls `.get("recall", 0)` — a key that never exists in the return dict:
```python
for thresh in thresholds:
    fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_score, threshold=thresh)
    recalls.append(fah_metrics.get("recall", 0))   # ← always 0, key doesn't exist
    fahs_list.append(fah_metrics.get("ambient_false_positives_per_hour", 0))
plt.plot(fahs_list, recalls, "b-", linewidth=2, label="Operating Curve")
```

**Impact:** `test_fah_recall.png` is a flat zero line. Any threshold/operating-point decisions based on this plot are made on silently wrong data.

**Fix:** Compute recall separately from the positive predictions before calling `compute_fah_metrics`:
```python
positive_mask = (y_true == 1)
total_positives = positive_mask.sum()
for thresh in thresholds:
    y_pred = (y_score >= thresh).astype(int)
    recall = float((y_pred[positive_mask] == 1).sum() / total_positives) if total_positives > 0 else 0.0
    recalls.append(recall)
    fah_metrics = fah_estimator.compute_fah_metrics(y_true, y_score, threshold=thresh)
    fahs_list.append(fah_metrics.get("ambient_false_positives_per_hour", 0))
```

---

### BUG-02 — Mining Consolidation Uses Modulo to Map Prediction Indices to Files
**File:** `src/training/mining.py`  
**Lines:** 1059–1070  
**Verified:** ✅

```python
for pred in epoch_preds:
    idx = pred["index"]
    if all_files:
        file_idx = idx % len(all_files)   # ← WRONG: modulo is not a valid index mapping
        file_path = all_files[file_idx]
```

`idx` is the sample index in the dataset (potentially in the thousands or millions). Mapping it via `% len(all_files)` is not equivalent to looking up the original source file — it produces pseudo-random file attribution.

**Impact:** Hard-negative mining and FP reporting identifies and potentially moves/copies entirely wrong audio files. The mined dataset is poisoned with incorrectly attributed samples. This is a silent correctness bug — it produces plausible-looking output with wrong content.

**Fix:** The prediction log must store file paths at mining time (when the dataset index→file mapping is known), not reconstruct them post-hoc from a sorted file list. If backward compatibility is needed, the dataset's `get_file_path(idx)` should be called at training time and stored in the log entry.

---

### BUG-03 — `evaluate_model.py` Config Path Argument Fails Silently
**File:** `scripts/evaluate_model.py`  
**Lines:** 1486–1488  
**Verified:** ✅ (confirmed `ValueError: Invalid preset '/path/to/config.yaml'`)

```python
config = load_full_config(args.config, args.override)
```

`load_full_config()` treats its first argument as a preset name string (validated against `VALID_PRESETS`). Passing a file path raises `ValueError: Invalid preset '/path/to/config.yaml'`. The `--config` help text and documentation advertise path-based config as supported, but the implementation only accepts preset names.

**Impact:** Any user following the documented YAML-path config workflow for evaluation gets a hard crash before evaluation runs.

**Fix:** Either update `load_full_config` to detect and handle file paths (check `Path(arg).exists()`), or update the CLI help text to clearly state only preset names are accepted.

---

## HIGH

### BUG-04 — `standard.yaml` Preset Missing `minimization_metric` and `target_minimization`
**File:** `config/presets/standard.yaml`  
**Lines:** 66–68  
**Verified:** ✅

The standard preset has a comment explaining that `minimization_metric` was "removed" but never replaced:
```yaml
maximization_metric: "average_viable_recall"
target_maximization: 1.0
# minimization_metric removed (was incorrectly set to same as maximization_metric)
```

The `TrainingConfig` dataclass (loader.py:154–155) defines:
```python
minimization_metric: str = "ambient_false_positives_per_hour"
target_minimization: float = 2.0
```

**Impact:** Standard preset silently uses dataclass defaults (`target_minimization=2.0`, `minimization_metric="ambient_false_positives_per_hour"`) rather than explicitly defined preset values. The "standard" preset is non-authoritative for the primary checkpoint selection criterion. Also note: `target_maximization` is not a recognized dataclass field — it will be silently ignored, meaning the intended maximization target has no effect.

**Fix:** Add the correct `minimization_metric` and `target_minimization` values to `standard.yaml` and verify `target_maximization` maps to the correct field name in the dataclass.

---

### BUG-05 — `fast_test.yaml` Sets `minimization_metric` to the Same Metric as `maximization_metric`
**File:** `config/presets/fast_test.yaml`  
**Lines:** 67–69  
**Verified:** ✅

```yaml
minimization_metric: "average_viable_recall"   # ← minimize recall??
target_minimization: 1.0
maximization_metric: "average_viable_recall"   # ← also maximize it
```

Both metrics point to the same field with contradictory objectives.

**Impact:** Checkpoint selection under `fast_test` tries to both minimize and maximize recall simultaneously. Depending on trainer logic priority ordering, this either selects nonsensical checkpoints or one objective silently overrides the other.

**Fix:** Set `minimization_metric: "ambient_false_positives_per_hour"` (or another appropriate metric that should be minimized) in `fast_test.yaml`.

---

### BUG-06 — Mining CLI `args.verbose` Accessed on Subcommands That Don't Define It
**File:** `src/training/mining.py`  
**Lines:** 1748  
**Verified:** ✅ (only `mine` subparser adds `--verbose`; `extract-top-fps` and `consolidate-logs` do not)

```python
_configure_mining_logging(verbose=args.verbose)
```

This runs unconditionally for all subcommands. `extract-top-fps` and `consolidate-logs` have no `--verbose` argument defined, so `args.verbose` raises `AttributeError`.

**Impact:** `mww-mining extract-top-fps` and `mww-mining consolidate-logs` crash immediately on startup with `AttributeError: Namespace object has no attribute 'verbose'`. Both subcommands are completely non-functional.

**Fix:** Add `--verbose` to all subparsers, or use `getattr(args, 'verbose', False)`.

---

### BUG-07 — Mining `generate_statistics_report` Filters on `true_label` That's Never Populated
**File:** `src/training/mining.py`  
**Lines:** 1059, 1148–1156  

Consolidated entries are built with fields `{index, score, file_path, epoch}` — `true_label` is never set. But `generate_statistics_report` filters on it:
```python
neg_hard_neg_preds = [p for p in all_predictions if p.get("true_label") in ("negative", "hard_negative")]
```

**Impact:** The statistics report silently shows zero predictions (empty list), producing a misleading report where all FP counts appear as 0. Analytics and post-training cleanup decisions based on this report are made on wrong data.

---

### BUG-08 — Trainer Class Weight Defaults Contradict Project Requirements
**File:** `src/training/trainer.py`  
**Lines:** 264–266  
**Verified:** ✅

```python
self.positive_weights  = training.get("positive_class_weight",  [5.0, 7.0, 9.0])
self.negative_weights  = training.get("negative_class_weight",  [1.5, 1.5, 1.5])
self.hard_negative_weights = training.get("hard_negative_class_weight", [3.0, 5.0, 7.0])
```

AGENTS.md specifies: `positive=1.0, negative=20.0, hard_neg=40.0`. The trainer defaults are 10-13× lower for negative and hard-negative weighting.

**Impact:** Any config that omits class weight fields (partial config, programmatic construction, future preset addition) silently trains with severely under-penalized negatives. This shifts precision/recall operating point and degrades deployed FAH performance without any error or warning.

**Fix:** Align defaults with documented values: `positive=[1.0]`, `negative=[20.0]`, `hard_negative=[40.0]` — or raise a `ValueError` when weights are missing rather than silently applying wrong defaults.

---

### BUG-09 — Async/Sync Validation Weight Snapshot Inconsistency with Hard-Negative Mining
**File:** `src/training/trainer.py`  
**Lines:** 1525–1546, 1971–1987, 1651–1669  

In the **async path**, validation uses a weight snapshot isolated from the training loop. In the **sync fallback path** (when the executor queue is full), validation uses the EMA snapshot correctly, but hard-negative mining is then invoked on `self.model` (current training weights), not the validated snapshot:

```python
# sync fallback path
self._swap_to_ema_weights()
weights_snapshot = [np.array(w, copy=True) for w in self.model.get_weights()]
val_metrics = self.validate(val_data_factory)
self._restore_training_weights()
self._handle_validation_results(..., weights_snapshot=weights_snapshot)
# Inside _handle_validation_results:
self._async_miner.start_mining(self.model, ...)  # ← uses current training weights, not snapshot
```

**Impact:** Checkpoint metrics correspond to EMA weight snapshot, but the mining that follows uses newer/different training weights. The mined hard negatives do not correspond to the evaluated model's failure modes, creating subtle training instability in high-throughput training runs where the sync fallback fires.

---

### BUG-10 — Zero Ring Buffer External State Slices Full Memory Instead of Empty
**File:** `src/model/streaming.py`  
**Lines:** 375–382  

```python
# Strided mode, _streaming_external_state
memory = tf.concat([input_state, inputs], axis=1)
state_update = memory[:, -self.ring_buffer_size_in_time_dim :, ...]
```

When `ring_buffer_size_in_time_dim == 0`, Python slice `memory[:, -0:, ...]` is equivalent to `memory[:, 0:, ...]` — the full tensor. The state update becomes the entire concatenated memory instead of an empty slice.

The internal state path guards against this correctly with `if self.ring_buffer_size_in_time_dim:` (line 342), but the external path (line 375+) has no such guard.

**Impact:** External-state streaming inference with zero-buffer layers gets incorrect state shapes, causing state bloat and potential shape contract violations downstream.

**Fix:** Mirror the internal-state guard in the external-state strided path:
```python
if self.ring_buffer_size_in_time_dim:
    state_update = memory[:, -self.ring_buffer_size_in_time_dim :, ...]
    ...
else:
    output = self.cell(inputs)
    return output, input_state  # pass state through unchanged
```

---

### BUG-11 — `test` Split in `evaluate_model.py` Returns Validation Pipeline
**File:** `scripts/evaluate_model.py`  
**Lines:** 257–283, 507–510  

```python
elif split == "test":
    return pipeline.create_validation_pipeline()  # same as val
```

**Impact:** Any evaluation code path requesting the `"test"` split silently receives validation data. Test metrics are contaminated with data that may have been used for hyperparameter selection, producing falsely optimistic reported numbers.

---

### BUG-12 — `manifest.py` Threshold Fallback Hardcoded to 0.5
**File:** `src/export/manifest.py`  
**Lines:** 124–128  

When auto-cutoff metadata is unavailable:
```python
logger.warning("... using fallback 0.5 ...")
return 0.5
```

**Impact:** If auto-cutoff detection fails (missing metadata, changed format), the exported manifest deploys with threshold 0.5 regardless of the tuned operating point. This can significantly increase false activation rate in production.

**Fix:** Fall back to `config.evaluation.default_threshold` or the export config's configured threshold rather than a hardcoded constant.

---

## MEDIUM

### BUG-13 — `autotuner.py` Gradient Step Counter Uses Planned Steps, Not Actual
**File:** `src/tuning/autotuner.py`  
**Lines:** 2688  

```python
self.total_gradient_steps += n_steps   # n_steps is planned; burst may early-stop
```

`_train_burst()` returns `burst_info["steps"]` which contains actual steps executed.

**Impact:** Budget accounting drifts when bursts early-stop. Tuner may terminate prematurely or misreport step usage.

---

### BUG-14 — `autotuner.py` Threshold Optimization Ignores Sample Weights
**File:** `src/tuning/autotuner.py`  
**Lines:** 827, 1035  

Training uses `search_*_weights` for weighted optimization, but the evaluation/threshold selection path uses unweighted recall and FAH:
```python
recall = true_positives / total_positives
fah = false_positives / val_ambient_duration_hours
```

**Impact:** Tuned operating point is systematically biased relative to the intended weighted objective. The model is optimized for one objective and evaluated against another.

---

### BUG-15 — `verify_streaming.py` Exit Codes Don't Distinguish Tool Failure from Model Failure
**File:** `scripts/verify_streaming.py`  
**Lines:** 259–267  

Gate failures and internal exceptions both return exit code `2`, same as invalid invocation.

**Impact:** CI/automation cannot distinguish "model failed quality gate" vs "verification tool crashed". Triage requires manual inspection.

---

### BUG-16 — `cluster_analyze.py` Guidance Prints Nonexistent Command Name
**File:** `src/tools/cluster_analyze.py`  
**Lines:** 378–380  

Post-analysis guidance prints:
```
python Start-Clustering.py ...
```

The actual tool is `mww-cluster-apply` / `cluster_apply.py`.

**Impact:** Users following the tool's own guidance hit `command not found`. Workflow appears broken after analysis completes.

---

## Confirmed Clean (False Positive Checks)

The following known-risky patterns were checked and are **not bugs**:

| Pattern | Status |
|---------|--------|
| `pymicro_features` chunk size in `features.py` | ✅ Correct — advances by `output.samples_read * 2` |
| EMA finalize + checkpoint reload at end of training | ✅ Correct — no post-finalize reload |
| `model.trainable_weights` for serialization | ✅ Correct — uses `get_weights()`/`set_weights()` throughout |
| Auto-tuner train/eval data contamination | ✅ Correct — `search_train`/`search_eval` split enforced |
| Export uint8 output dtype | ✅ Correct — `inference_output_type=tf.uint8` |
| `model.export()` misuse | ✅ Correct — uses `tf.keras.export.ExportArchive` |
| State shape hardcoding in export verification | ✅ Correct — uses `compute_expected_state_shapes()` |
| ROC-AUC edge cases (all-positive/all-negative) | ✅ Handled — 0.5 fallback present |
| `verify_esphome.py` NumPy JSON serialization | ✅ Correct — sanitization present |
| RaggedMmap index arithmetic | ✅ No off-by-one found |
| Dataset split contamination | ✅ Hash/path/speaker overlap checks present |
| Clustering distance metric | ✅ L2-norm + Euclidean = cosine-equivalent |
| Config dataclass field/property collision | ✅ None found |
| Async validation race condition | ✅ Weight snapshot + `_validation_lock` correct |
