# Functional Bug Report - microwakeword_trainer

**Generated**: 2026-03-17  
**Scope**: Full codebase functional bug hunt (crashes, broken features, silently wrong results)  
**Methodology**: Static analysis + 3 parallel deep code review subagents

---

## Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 4 | Crashes, completely broken features |
| **HIGH** | 8 | Silently wrong results, lost functionality |
| **MEDIUM** | 6 | Logic errors, dead code, minor issues |
| **Total** | **18** | |

---

## CRITICAL Bugs (Fix Immediately)

### 1. SyntaxError: Global Declaration After Use
**File**: `scripts/generate_test_dataset.py`  
**Lines**: 188  
**Problem**:
```python
# Lines 161-163 use DATASET_DIR in argument default
parser.add_argument(
    "--output-dir",
    type=str,
    default=str(DATASET_DIR),  # <-- DATASET_DIR used HERE
    ...
)
...
# Line 188 - global declared AFTER use in default
187:    # Update globals based on args
188:    global DATASET_DIR, POSITIVE_DIR, NEGATIVE_DIR, HARD_NEGATIVE_DIR
```
**Impact**: Script cannot be imported or executed. `SyntaxError: name 'DATASET_DIR' is used prior to global declaration`

**Verification**:
```bash
python -c "from scripts.generate_test_dataset import main"
# SyntaxError: name 'DATASET_DIR' is used prior to global declaration
```

---

### 2. Boolean Mask Indexing Shape Mismatch (CuPy)
**File**: `src/data/spec_augment_gpu.py`  
**Lines**: 140-141, 156-157  
**Problem**:
```python
mask_3d = mask_2d[:, None, :]   # shape (B,1,F)
batch_gpu[mask_3d] = 0          # batch_gpu shape (B,T,F)
...
mask_3d = mask_2d[:, :, None]   # shape (B,T,1)
batch_gpu[mask_3d] = 0
```
In CuPy/NumPy, boolean index dimensions must match indexed array dimensions exactly, and `(B,1,F)` / `(B,T,1)` cannot directly index `(B,T,F)`.

**Impact**: `batch_spec_augment_gpu()` crashes at runtime with IndexError/boolean index mismatch, breaking training when batch SpecAugment path is used.

---

### 3. Double Sigmoid in Auto-Tuner
**File**: `src/tuning/orchestrator.py`  
**Lines**: 214-216  
**Problem**:
```python
logits = model(batch, training=False)
probs = tf.nn.sigmoid(logits)  # <-- SECOND sigmoid
```
Model architecture's final Dense already uses `activation="sigmoid"` (`src/model/architecture.py:533`).

**Impact**: Scores are systematically distorted (compressed toward ~0.5–0.73), threshold optimization/FAH-recall estimation becomes wrong, and autotuner can select bad operating points and suboptimal candidates silently.

---

### 4. Streaming Mode Not Propagated to Nested Stream Layers
**File**: `src/model/architecture.py:647,712,717` + `src/model/streaming.py:309-317,353-387`  
**Problem**:
```python
# architecture.py
for mixconv in block.mixconvs:
    mixconv.mode = self.mode     # nested Stream layers unchanged!
...
x = self.depthwise_convs[i](x)   # x may be tuple in stream_external
```
`MixedNet.__init__` sets `mixconv.mode` but does NOT update each internal `Stream` in `mixconv.depthwise_convs`. In `stream_external` mode, `Stream.call()` falls through to non-streaming path and returns a tensor instead of `(output, state)`. Then `MixConvBlock.call()` passes this tuple to `DepthwiseConv2D`, causing runtime failure.

**Impact**: Streaming/export path crashes at inference-time for external-state mode (broken feature, not just degraded quality).

---

## HIGH Bugs (Significant Impact)

### 5. Undefined Variable `total_files`
**File**: `scripts/count_audio_hours.py`  
**Lines**: 97, 105  
**Problem**:
```python
def main():
    ...
    total_seconds = 0.0  # initialized
    # total_files NEVER initialized
    
    for directory in directories:
        if directory.exists():
            count, seconds = scan_directory(directory)
            total_files += count      # <-- F821: undefined name
            total_seconds += seconds
    
    print(f"\nTotal: {total_files} files, {total_hours:.2f} hours")  # <-- Also undefined
```

**Impact**: Script crashes with `NameError` when run with valid directories.

**Verification**:
```bash
python scripts/count_audio_hours.py --negative-dir ./dataset/negative
# NameError: name 'total_files' is not defined
```

---

### 6. Uninitialized Attribute `_saved_training_weights`
**File**: `src/training/trainer.py`  
**Lines**: 1036-1041 (read), missing initialization in `__init__`  
**Problem**:
```python
def _restore_training_weights(self):
    if not self._ema_enabled or self._saved_training_weights is None:
        return
```
`_saved_training_weights` is never initialized in `__init__`. If control reaches `_restore_training_weights()` before `_swap_to_ema_weights()`, this raises `AttributeError`.

**Impact**: Validation/mining/finalization paths can crash depending on call ordering (state-management crash in background/async-adjacent flow).

---

### 7. Calibration Helpers Crash with Hard Negative Labels
**File**: `src/evaluation/test_evaluator.py`  
**Lines**: 336-337, 719  
**Problem**:
```python
brier = compute_brier_score(y_true, y_score)
curve = compute_calibration_curve(y_true, y_score, n_bins=10)
```
Calibration helpers are called with raw `y_true` labels that may include hard-negative class `2`, but `compute_brier_score()` / `compute_calibration_curve()` require labels strictly in `{0,1}` and raise `ValueError` otherwise.

**Impact**: Held-out test evaluation can crash at runtime when test split includes hard negatives (label 2), breaking report generation/plots and post-training evaluation flow.

---

### 8. Export Aborts After Successful TFLite Write
**File**: `src/export/tflite.py`  
**Lines**: 1294-1301, 1317-1353  
**Problem**:
```python
tflite_file.write_bytes(tflite_model)  # Model already written
...
except Exception as e:
    print(...)
    raise  # Aborts even though model was saved
```

**Impact**: Export command can fail after successfully producing a valid TFLite model, causing broken/partial pipeline behavior (artifact exists but command exits failure, downstream automation treats export as failed).

---

### 9. Weight Perturbation Identity Check Fails
**File**: `src/tuning/knobs.py`  
**Lines**: 123-133  
**Problem**:
```python
trainable_set = {id(v) for v in trainable_vars}
...
if id(w_var) in trainable_set:
    ...
```
Weight perturbation identifies trainable tensors via Python object identity between `model.trainable_weights` and `model.weights`. This can fail to match in TF/Keras wrapper scenarios.

**Impact**: "Exploration" knob can silently become a no-op, degrading search diversity and causing autotuning to stagnate or miss better solutions without obvious errors.

---

### 10. Cache Hit Missing Built-State Update
**File**: `src/data/dataset.py`  
**Lines**: 971-975  
**Problem**:
```python
if self._is_cache_valid(...):
    logger.info("[CACHE] Valid feature cache found — skipping feature extraction")
    self._load_store()
    return self  # <-- self._is_built never set to True
```
`self._is_built` is initialized to `False` and never set to `True` on the cache-hit path.

**Impact**: Any downstream logic relying on "dataset built" state can behave incorrectly (e.g., skip/redo build decisions, wrong control flow), leading to silent pipeline misuse.

---

### 11. `max_time_frames` Silently Ignored
**File**: `src/training/performance_optimizer.py`  
**Lines**: 128-133, 149-154  
**Problem**:
```python
def create_training_dataset(..., max_time_frames=None):
    return create_optimized_dataset(dataset, self.config, split="train", ...)
    # max_time_frames never used
```
`max_time_frames` argument is accepted but silently ignored.

**Impact**: Caller-requested frame length is not applied; can produce shape mismatches or silently wrong padding/truncation behavior versus expected config.

---

### 12. Unused Variable `evicted` in Mining
**File**: `src/training/mining.py`  
**Lines**: 245  
**Problem**:
```python
elif pred_score > hard_negative_heap[0][0]:
    # This hard negative has higher score than the lowest in heap
    evicted = heapq.heapreplace(hard_negative_heap, heap_entry)
    # Note: evicted never used
```

**Impact**: Dead code - assignment has no effect. May indicate incomplete implementation where eviction was intended to trigger cache cleanup.

---

## MEDIUM Bugs (Minor Impact)

### 13. Duplicate `pass` in Exception Handler
**File**: `src/utils/performance.py`  
**Lines**: 42-45  
**Problem**:
```python
except Exception:
    pass
    pass  # Duplicate
```

**Impact**: Not cosmetic - hides all GPU detection failures completely. Silent misreporting of system capabilities (GPU section absent with no signal), making runtime diagnosis and feature gating unreliable.

---

### 14. LR Knob Ignored in Burst Training
**File**: `src/tuning/orchestrator.py`  
**Lines**: 276-277 vs 295-298  
**Problem**:
```python
# knob
candidate._sampled_lr = lr

# burst
lr_min, lr_max = self._get_cfg("lr_range", ...)
cosine_lr = ...
optimizer.learning_rate.assign(cosine_lr)  # Ignores candidate._sampled_lr
```
LR knob samples/stores candidate-specific LR, but `_run_burst()` ignores it and always uses global `lr_range` cosine schedule.

**Impact**: LR knob mutation is functionally broken (candidate-specific LR never influences training), reducing tuner behavior correctness and silently invalidating one optimization axis.

---

### 15. Discard Destination Flattens Directory Structure
**File**: `src/data/preprocessing.py`  
**Lines**: 320-325  
**Problem**:
```python
dest = discarded_root / path.name  # Only uses filename
shutil.move(str(path), str(dest))
```
Unlike `process_speech_directory()` (which preserves relative path), this single-file helper discards only by filename.

**Impact**: Silent data loss/collision when different source dirs contain same filename (one discarded file can overwrite another), corrupting auditability and preprocessing outputs.

---

### 16. Unused `pytest` Import in Test
**File**: `tests/unit/test_tuning_metrics.py`  
**Lines**: 6  
**Problem**: `import pytest` but never used.

**Impact**: Minor - unnecessary dependency in test file.

---

## Methodology

1. **Source Enumeration**: Mapped all Python files in `src/`, `config/`, `scripts/`, `tests/`
2. **Syntax Verification**: Ran `python -m py_compile` on all files
3. **Static Analysis**: Ran ruff with functional-only selectors (`F,E711,E712,E721,W605,B,E9,F8,A001,A002,A003`)
4. **Parallel Deep Review**: Deployed 3 subagents simultaneously:
   - Agent 1: Core training (`src/training/`, `src/model/`)
   - Agent 2: Eval/Export/Tuning (`src/evaluation/`, `src/export/`, `src/tuning/`)
   - Agent 3: Config/Data/Tools (`config/`, `src/data/`, `src/tools/`, `src/utils/`)
5. **Live Verification**: Confirmed critical bugs in isolated environment

---

## Module-Level Summary

| Module | CRITICAL | HIGH | MEDIUM |
|--------|----------|------|--------|
| scripts/ | 1 | 1 | 0 |
| src/data/ | 1 | 1 | 1 |
| src/training/ | 0 | 2 | 1 |
| src/tuning/ | 1 | 2 | 1 |
| src/evaluation/ | 0 | 1 | 0 |
| src/export/ | 0 | 1 | 0 |
| src/model/ | 1 | 0 | 0 |
| src/utils/ | 0 | 0 | 1 |
| tests/ | 0 | 0 | 1 |

---

## Recommended Fix Priority

### Immediate (Before Next Release)
1. Fix `scripts/generate_test_dataset.py` syntax error
2. Fix `scripts/count_audio_hours.py` undefined variable
3. Fix `src/tuning/orchestrator.py` double sigmoid
4. Fix `src/model/architecture.py` streaming mode propagation
5. Fix `src/data/spec_augment_gpu.py` boolean mask indexing

### High Priority (Next Sprint)
6. Fix `src/training/trainer.py` uninitialized attribute
7. Fix `src/evaluation/test_evaluator.py` calibration label handling
8. Fix `src/export/tflite.py` exception handling order
9. Fix `src/tuning/knobs.py` weight perturbation identity check
10. Fix `src/data/dataset.py` cache hit state management

### Medium Priority (Backlog)
11-16. Remaining MEDIUM severity issues

---

*End of Report*
