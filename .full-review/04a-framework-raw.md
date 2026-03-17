# Phase 4A: Framework & Language Best Practices (Raw)

## Critical

### BP-C1: `knob` Undefined in `orchestrator.py:236` — NameError at Runtime (duplicate of DOC-C2)
- Already captured as DOC-C2. Confirmed here as a code-level finding.

## High

### BP-H1: Dual `setup.py` + `pyproject.toml` Build System
- Both files define the same package with divergent metadata, author info, and dependency constraints.
- **Fix:** Delete `setup.py`. Merge `vad` and `quality-full` extras into `pyproject.toml`. Pin TF only in `requirements.txt`.

### BP-H2: `model.train_on_batch` Instead of `@tf.function` Train Step
- **File:** `src/training/trainer.py:1341`
- `train_on_batch` is a legacy Keras 1/2 pattern. Forces a Python-level call per step, bypasses XLA JIT tracing, cannot be traced into a `tf.Graph`.
- **Fix:** Override `train_step` in the model subclass or use a `@tf.function(reduce_retracing=True)` gradient tape loop.

### BP-H3: Deprecated `tf.data.experimental.prefetch_to_device`
- **File:** `src/data/tfdata_pipeline.py:328,430,479`
- API moved out of `experimental` in TF 2.6. Use `.prefetch(tf.data.AUTOTUNE)` with TF's automatic device placement.

### BP-H4: `MixConvBlock.__init__` Incomplete — `self.filters` Never Assigned + Infinite Recursion in `mode` Setter
- **File:** `src/model/architecture.py:151-165`
- `self.filters = filters` is never called in `__init__`, causing `AttributeError` in `build()`. The `mode` property setter contains `self.mode = mode` which recursively calls the setter (infinite recursion or `NameError`).
- **Fix:** Complete `__init__` to assign all instance attributes. Fix the setter to use `self._mode = value` throughout and remove the recursive assignment.

## Medium

### BP-M1: Deprecated `options.experimental_deterministic`
- **File:** `src/data/tfdata_pipeline.py:313,319`
- Use `options.deterministic = False` (stable since TF 2.5).

### BP-M2: Private `model._flatten_layers()` API Usage
- **Files:** `src/tuning/orchestrator.py:142,149`; `src/export/tflite.py:233`
- `_flatten_layers` is a private method removed/renamed in Keras 3.
- **Fix:** Use `model.layers` + recursive sublayer check.

### BP-M3: Mixed Old/New Type Annotation Styles
- Legacy `Dict`, `List`, `Optional`, `Union` from `typing` in ~6 files; modern `dict`, `list`, `X | None` in newer files. Project requires Python 3.10+.
- **Fix:** Replace all `typing` generics with PEP 585/604 equivalents. No `__future__` import needed for Python 3.10+.

### BP-M4: `EvaluationMetrics.__init__` Docstring Misplaced
- **File:** `src/training/trainer.py:65-67`
- Docstring appears after first attribute assignment — not attached as `__init__.__doc__`.

### BP-M5: Mypy Configured Too Leniently
- **File:** `pyproject.toml:248-257`
- `disallow_untyped_defs = false`, `check_untyped_defs = false`, `allow_untyped_calls = true` — mypy skips the body of any unannotated function, defeating the type checker on the training hot path.
- **Fix:** Enable `check_untyped_defs = true`; incrementally enable `disallow_untyped_defs = true`.

### BP-M6: Dead Async Validation Infrastructure
- **File:** `src/training/trainer.py:1544,1647-1666`
- `_schedule_validation()` always returns `False`. The `ThreadPoolExecutor` and `_validation_lock` are created but never used for actual async execution. `_compute_metrics_background` raises `RuntimeError` immediately.
- **Fix:** Remove executor, lock, and `_pending_validation` state unless async validation is re-enabled.

### BP-M7: Over-Pinning Transitive Dependencies
- **File:** `requirements.txt`
- Hard-pins transitive deps like `opentelemetry-proto==1.25.0` and `protobuf==4.25.8`. These should be managed by the resolver, not pinned directly.
- **Fix:** Use `pip-compile` / `uv lock` to separate resolved pins from declared constraints.

## Low

- **BP-L1:** f-strings in logger calls — use `%`-style for lazy evaluation; add `"G"` to Ruff `select`
- **BP-L2:** `src/config/__init__.py` empty with misleading docstring — either populate or fix docstring
- **BP-L3:** Docstring example in `PrefetchGenerator` advertises deprecated `train_on_batch`
- **BP-L4:** Ad-hoc `dict` returns where dataclasses would be safer (e.g., `_get_current_phase_settings`)
- **BP-L5:** Missed walrus operator / structural pattern matching opportunities in metric-cast patterns and mode-string → enum mapping

## Summary: 1 Critical (shared with DOC-C2), 4 High, 7 Medium, 5 Low
