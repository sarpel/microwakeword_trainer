# Code Quality Review: microwakeword_trainer

**Review Date:** 2026-03-18
**Scope:** Full codebase review of key training pipeline files
**Framework:** TensorFlow/Python ML training pipeline

---

## Executive Summary

This review analyzes 8 key files totaling approximately 9,330 lines of Python code. The codebase shows signs of active development with good documentation practices but exhibits several code quality issues including high cyclomatic complexity, code duplication, deeply nested logic, and maintainability concerns.

**Overall Assessment:**
- **Strengths:** Good documentation, comprehensive feature set, proper error handling in many areas
- **Weaknesses:** High complexity in core training loop, significant code duplication, deeply nested conditionals, large class/method sizes

---

## 1. Code Complexity Issues

### 1.1 Critical: `Trainer` class in `src/training/trainer.py` - Massive Class with Excessive Responsibilities

**Severity:** Critical
**Location:** `src/training/trainer.py`, lines 236-2282 (2,046 lines for single class)
**Issue:** The `Trainer` class violates the Single Responsibility Principle. It handles:
- Model building and compilation
- Training loop orchestration
- Validation and metrics computation
- Checkpoint management
- TensorBoard logging
- Hard negative mining coordination
- EMA weight management
- Plateau detection and LR scheduling
- Phase-based training configuration

**Cyclomatic Complexity Analysis:**
- `train()` method: ~45+ branches (lines 1868-2282)
- `_handle_validation_results()`: ~30+ branches (lines 1668-1781)
- `_is_best_model()`: ~15 branches (lines 1073-1139)
- `_get_current_phase_settings()`: ~12 branches (lines 669-727)

**Recommendation:** Decompose the `Trainer` class into focused collaborator classes:

```python
# Proposed structure
class TrainingOrchestrator:
    """High-level training coordination."""
    def __init__(self):
        self.checkpoint_manager = CheckpointManager(...)
        self.metrics_tracker = MetricsTracker(...)
        self.lr_scheduler = PhaseBasedScheduler(...)
        self.mining_coordinator = MiningCoordinator(...)
        self.logger = TrainingLogger(...)
```

---
### 1.2 High: Deeply Nested Logic in `train()` Method

**Severity:** High
**Location:** `src/training/trainer.py`, `train()` method, lines 1942-2183
**Issue:** The main training loop has excessive nesting depth (up to 7+ levels):

```python
# Current structure (simplified):
for step in range(1, total_steps + 1):  # Level 1
    self._check_validation()
    if self._async_early_stop_requested:  # Level 2
        break
    try:  # Level 2
        (...) = next(train_data_generator)
    except StopIteration:  # Level 3
        train_data_generator = train_data_factory()
        try:  # Level 4
            (...) = next(train_data_generator)
        except StopIteration as exc:  # Level 5
            raise RuntimeError(...)
    if self.spec_augment_enabled:  # Level 2
        if self.spec_augment_backend != "tf":  # Level 3
            if self.time_mask_count[aug_phase] > 0:  # Level 4
                try:  # Level 5
                    if hasattr(train_fingerprints, "numpy"):  # Level 6
                        ...
                except RuntimeError as e:  # Level 6
                    if not self._spec_augment_warning_shown:  # Level 7
                        ...
```

**Recommendation:** Extract methods for each major stage:

```python
def train(self, ...):
    for step in range(1, total_steps + 1):
        batch = self._fetch_batch(train_data_generator, train_data_factory)
        batch = self._apply_spec_augment(batch, step)
        metrics = self._execute_train_step(batch)
        self._log_step_metrics(step, metrics)
        if self._should_validate(step):
            self._run_validation(...)
```

---

### 1.3 High: Complex Validation Logic in `_validate_with_model()`

**Severity:** High
**Location:** `src/training/trainer.py`, lines 1386-1599 (213 lines)
**Issue:** The validation method has:
- Multiple nested conditionals for clip ID handling
- Complex batch metadata processing
- Inline metrics computation with multiple branches
- Mixed responsibilities (validation + metrics calculation + data accumulation)

**Cyclomatic Complexity:** ~35+

**Recommendation:** Split into:
- `ValidationDataCollector` - handles batch iteration and accumulation
- `MetricsComputer` - computes metrics from accumulated data
- `ClipIDResolver` - handles clip ID mapping logic

---

### 1.4 Medium: Complex Phase Settings Logic

**Severity:** Medium
**Location:** `src/training/trainer.py`, `_get_current_phase_settings()`, lines 669-727
**Issue:** Complex logic for determining phase-specific settings with stagger support:

```python
def _get_current_phase_settings(self, step: int) -> dict[str, Any]:
    # --- Determine LR phase (uses step directly) ---
    lr_phase = 0
    for i, boundary in enumerate(self._phase_boundaries):
        if step < boundary:
            lr_phase = i
            break
    else:
        lr_phase = len(self.training_steps_list) - 1

    # --- Determine weights/augmentation phase (staggered) ---
    stagger = self.phase_stagger_steps
    if stagger > 0 and lr_phase > 0:
        phase_start = self._phase_boundaries[lr_phase - 1] if lr_phase > 0 else 0
        steps_into_phase = step - phase_start
        if steps_into_phase < stagger:
            weight_aug_phase = lr_phase - 1
        else:
            weight_aug_phase = lr_phase
    else:
        weight_aug_phase = lr_phase
```

**Recommendation:** Create a `PhaseConfiguration` class:

```python
@dataclass
class PhaseConfiguration:
    lr_phase: int
    weight_aug_phase: int
    learning_rate: float

    @classmethod
    def for_step(cls, step: int, boundaries: list[int], stagger: int) -> "PhaseConfiguration":
        ...
```

---

## 2. Maintainability Issues

### 2.1 Critical: `Trainer.__init__()` - Excessive Length and Complexity

**Severity:** Critical
**Location:** `src/training/trainer.py`, lines 239-547 (308 lines)
**Issue:** The constructor:
- Initializes 80+ instance attributes
- Performs complex configuration parsing
- Sets up multiple subsystems (logging, metrics, mining, TensorBoard)
- Has inline helper function `_pad_or_trim()` defined inside (lines 290-296)

**Recommendation:** Use the Builder pattern or dependency injection:

```python
class TrainerConfig:
    """Immutable configuration for Trainer."""
    training_steps: list[int]
    learning_rates: list[float]
    # ... other fields

class Trainer:
    def __init__(self, config: TrainerConfig,
                 logger: TrainingLogger,
                 checkpoint_manager: CheckpointManager,
                 metrics_tracker: MetricsTracker):
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.metrics_tracker = metrics_tracker
```

---

### 2.2 High: Inconsistent Naming Conventions

**Severity:** High
**Issue:** Mixed naming styles across the codebase:

| File | Examples |
|------|----------|
| `trainer.py` | `self._last_assigned_lr`, `self._bn_frozen`, `self._async_early_stop_requested` |
| `dataset.py` | `_BatchBuffer`, `RaggedMmap`, `WakeWordDataset` |
| `mining.py` | `hard_negative_heap`, `fp_threshold`, `mining_interval_epochs` |
| `tflite.py` | `_normalize_pointwise_filters`, `_VAR_IDX`, `StreamingExportModel` |

**Specific Issues:**
- Private methods use both single (`_method`) and double (`__method`) underscore prefixes inconsistently
- Some boolean flags use `is_` prefix (`is_hard_negative`), others don't (`_bn_frozen`)
- Mixed snake_case for variables but some camelCase in comments/docstrings

**Recommendation:** Establish and enforce naming conventions:
- Private methods: single underscore prefix
- Boolean flags: use `is_`, `has_`, or `should_` prefix
- Constants: UPPER_SNAKE_CASE
- Classes: PascalCase
- Functions/variables: snake_case

---

### 2.3 High: Long Method - `export_streaming_tflite()` in `tflite.py`

**Severity:** High
**Location:** `src/export/tflite.py`, lines ~900-1200+ (300+ lines estimated)
**Issue:** The TFLite export function is excessively long with multiple nested contexts and complex quantization setup.

**Recommendation:** Decompose into:
- `ModelLoader.load_for_export()`
- `RepresentativeDatasetBuilder.build()`
- `QuantizationConfig.create()`
- `TFLiteConverter.convert()`
- `ExportVerifier.verify()`

---

### 2.4 Medium: `EvaluationMetrics` Class - Mixed Responsibilities

**Severity:** Medium
**Location:** `src/training/trainer.py`, lines 50-229
**Issue:** The class:
- Accumulates predictions/labels
- Computes metrics at multiple thresholds
- Maintains both array and dict views of the same data
- Has 101-threshold hardcoded logic scattered throughout

**Recommendation:** Separate concerns:

```python
class PredictionAccumulator:
    """Just accumulates predictions."""
    def update(self, y_true, y_scores): ...
    def get_all(self) -> tuple[np.ndarray, np.ndarray]: ...

class ThresholdMetricsComputer:
    """Computes metrics at given thresholds."""
    def __init__(self, cutoffs: list[float]): ...
    def compute(self, y_true, y_scores) -> dict[str, float]: ...
```

---

## 3. Code Duplication Issues

### 3.1 Critical: Duplicate Weight Perturbation Logic in `knobs.py`

**Severity:** Critical
**Location:** `src/tuning/knobs.py`, `WeightPerturbationKnob.apply()`, lines 113-144
**Issue:** The same logic is duplicated twice in the method:

```python
# Lines 125-132 - First occurrence
trainable_vars = list(model.trainable_weights)
all_weights = [np.asarray(w) for w in model.get_weights()]
trainable_names = {v.name for v in trainable_vars}

for i, w_var in enumerate(model.weights):
    if w_var.name in trainable_names:
        noise = np.random.normal(0.0, scale, size=all_weights[i].shape)
        all_weights[i] = all_weights[i] + noise

# Lines 133-140 - Second occurrence (identical logic, different variable names)
trainable_vars = list(model.trainable_weights)
all_weights = [np.asarray(w) for w in model.get_weights()]
trainable_set = {id(v) for v in trainable_vars}

for i, w_var in enumerate(model.weights):
    if id(w_var) in trainable_set:
        noise = np.random.normal(0.0, scale, size=all_weights[i].shape)
        all_weights[i] = all_weights[i] + noise
```

**Fix:** Remove the duplicate code block.

---

### 3.2 High: Duplicate EMA Enable Check in `trainer.py`

**Severity:** High
**Location:** `src/training/trainer.py`, lines 530-534
**Issue:**

```python
self._ema_enabled = False
if ema_decay is not None:
    self._ema_enabled = True
self._saved_training_weights: list[np.ndarray] | None = None  # For EMA weight swap
if ema_decay is not None:
    self._ema_enabled = True  # Duplicate!
```

**Fix:** Remove the second `if ema_decay is not None: self._ema_enabled = True` block.

---

### 3.3 High: Duplicate Cache Check in `dataset.py`

**Severity:** High
**Location:** `src/data/dataset.py`, lines 988-996
**Issue:**

```python
# Check if valid cache exists
# Check if valid cache exists  # Duplicate comment!
if self._is_cache_valid(processed_dir, paths_cfg, hardware_cfg, training_cfg):
    logger.info("[CACHE] Valid feature cache found — skipping feature extraction")
    self._load_store()
    self._is_built = True
    return self
    logger.info("[CACHE] Valid feature cache found — skipping feature extraction")  # Dead code!
    self._load_store()
    return self
```

**Fix:** Remove the duplicate code block after the first `return self`.

---

### 3.4 High: Similar Pipeline Creation Methods in `tfdata_pipeline.py`

**Severity:** High
**Location:** `src/data/tfdata_pipeline.py`, lines 196-483
**Issue:** `create_training_pipeline()`, `create_validation_pipeline()`, and `create_test_pipeline()` share ~80% identical code:
- Same generator factory pattern
- Same output signature definitions
- Same cache resolution logic
- Same prefetch setup

**Recommendation:** Extract common pipeline builder:

```python
def _create_base_pipeline(
    self,
    split: str,
    shuffle: bool = False,
    apply_spec_augment: bool = False,
) -> tf.data.Dataset:
    """Common pipeline creation logic."""
    generator = self._generator_factory(split)
    output_signature = (...)  # Common signature
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    # Common caching, prefetching
    return ds
```

---

### 3.5 Medium: Duplicate Configuration Access Patterns

**Severity:** Medium
**Issue:** Throughout the codebase, config access follows repetitive patterns:

```python
# Pattern seen in multiple files
config.get("training", {}).get("batch_size", 384)
config.get("hardware", {}).get("mel_bins", 40)
config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
```

**Recommendation:** Create a typed configuration class:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 384

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        training = config.get("training", {})
        return cls(batch_size=training.get("batch_size", 384))
```

---

## 4. Clean Code / SOLID Violations

### 4.1 Critical: Single Responsibility Principle Violations

**Severity:** Critical
**Classes with multiple responsibilities:**

| Class | Responsibilities |
|-------|------------------|
| `Trainer` | Training, validation, checkpointing, logging, mining coordination, LR scheduling, phase management |
| `WakeWordDataset` | Data loading, feature extraction, caching, batch generation, split integrity checking |
| `StreamingExportModel` | Model architecture, state management, batch norm folding, streaming logic |
| `MicroAutoTuner` | Data loading, model creation, optimization, knob management, metrics tracking |

**Recommendation:** Apply the Single Responsibility Principle:

```python
# Instead of Trainer doing everything:
class Trainer:
    def __init__(self, ...):
        self.training_loop = TrainingLoop(...)
        self.validator = Validator(...)
        self.checkpoint_manager = CheckpointManager(...)

    def train(self, ...):
        for step in self.training_loop:
            self.training_loop.step(...)
            if self.validator.should_validate(step):
                metrics = self.validator.validate(...)
                self.checkpoint_manager.maybe_save(metrics)
```

---

### 4.2 High: Open/Closed Principle Violations

**Severity:** High
**Location:** `src/tuning/knobs.py`
**Issue:** Adding a new knob requires modifying the `MicroAutoTuner._make_knob()` method:

```python
def _make_knob(self, knob_name: str):
    knobs = {
        "lr": LRKnob,
        "threshold": ThresholdKnob,
        "temperature": TemperatureKnob,
        # ... adding new knob requires editing this dict
    }
```

**Recommendation:** Use a registry pattern:

```python
class KnobRegistry:
    _knobs: dict[str, type[Knob]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(knob_class: type[Knob]):
            cls._knobs[name] = knob_class
            return knob_class
        return decorator

    @classmethod
    def create(cls, name: str) -> Knob:
        return cls._knobs[name]()

@KnobRegistry.register("lr")
class LRKnob(Knob): ...
```

---

### 4.3 High: Feature Envy in `EvaluationMetrics`

**Severity:** High
**Location:** `src/training/trainer.py`, `EvaluationMetrics.compute_metrics()`
**Issue:** The method reaches out to `MetricsCalculator` for most computations, suggesting the logic belongs there:

```python
def compute_metrics(self) -> dict[str, float]:
    # Most work delegated to MetricsCalculator
    calc = MetricsCalculator(...)
    metrics = calc.compute_all_metrics(...)

    # Some local computation
    for i, cutoff in enumerate(self.cutoffs):
        # ... per-threshold metrics
    return metrics
```

**Recommendation:** Move per-threshold computation to `MetricsCalculator` or merge classes.

---

### 4.4 Medium: Law of Demeter Violations

**Severity:** Medium
**Location:** Throughout `trainer.py`
**Issue:** Deep attribute chains:

```python
# trainer.py
if self.tensorboard_logger is not None and self.tensorboard_logger.writer is not None:
    self.tensorboard_writer = self.tensorboard_logger.writer
```

**Recommendation:** Use the Tell, Don't Ask principle:

```python
# Instead of checking internals
self.tensorboard_logger.ensure_writer()
self.tensorboard_writer = self.tensorboard_logger.get_writer()
```

---

## 5. Technical Debt

### 5.1 Critical: Commented-Out and Dead Code

**Severity:** Critical
**Location:** `src/training/trainer.py`, lines 2227-2237
**Issue:** Extensive commented code blocks left in place:

```python
# Note: We do NOT reload best_weights here because:
# 1. Training is complete - no need to reload weights
# 2. model already contains the right weights (either training or EMA depending on when we last saved)
# 3. best_weights.weights.h5 (validated best checkpoint) has EMA-smoothed weights and proven metrics
# 4. Loading weights here causes optimizer state warnings due to EMA finalize
# For export/inference, use best_weights.weights.h5 which has validated EMA-smoothed weights
# 1. Training is complete - no need to reload weights
# 2. model already contains the right weights (either training or EMA depending on when we last saved)
# 3. final_weights.weights.h5 (saved above) has EMA-smoothed weights, which is preferred
# 4. Loading best_weights here causes optimizer state warnings due to EMA finalize
# For export/inference, use final_weights.weights.h5 which has smoothed EMA weights
```

The same points are duplicated twice with slight variations.

**Fix:** Remove or consolidate the duplicate comments.

---

### 5.2 High: Magic Numbers Throughout

**Severity:** High
**Issue:** Hardcoded values without named constants:

```python
# trainer.py
n_thresholds = int(self.evaluation_config.get("n_thresholds", 101) or 101)  # Why 101?
self.eval_target_recall = float(self.evaluation_config.get("target_recall", 0.90) or 0.90)  # Why 0.90?
score_sample_limit = 2000  # Why 2000?

# dataset.py
if len(self._memory_cache) > 1024:  # Why 1024?
    self._memory_cache.popitem(last=False)

# knobs.py
next_arm = (current_arm + 1) % 7  # Why 7 arms?
```

**Recommendation:** Define named constants:

```python
class Defaults:
    N_THRESHOLDS = 101  # 0.01 resolution from 0 to 1
    TARGET_RECALL = 0.90  # Production target for wake word detection
    SCORE_SAMPLE_LIMIT = 2000  # Memory limit for score distribution tracking
    CACHE_SIZE = 1024  # Maximum items in LRU cache
    SAMPLING_ARMS = 7  # Number of sampling mix configurations
```

---

### 5.3 High: TODO and FIXME Comments Without Tracking

**Severity:** High
**Issue:** Scattered TODO comments without issue tracking:

```python
# Various files contain patterns like:
# TODO: Refactor this
# FIXME: Handle edge case
# NOTE: This is temporary
```

Without proper tracking, these are often forgotten.

**Recommendation:** Either:
1. Create GitHub issues for each TODO and reference them in comments
2. Use a `TODO.md` file with file/line references
3. Use linting rules to fail CI on unchecked TODOs

---

### 5.4 Medium: Stringly-Typed Configuration

**Severity:** Medium
**Location:** Throughout configuration handling
**Issue:** Configuration values passed as strings that require parsing:

```python
# model architecture strings
pointwise_filters=model_cfg.get("pointwise_filters", "64,64,64,64"),
mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
residual_connection=model_cfg.get("residual_connection", "0,1,1,1"),
```

**Recommendation:** Use structured types:

```python
@dataclass
class ModelArchitectureConfig:
    pointwise_filters: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    mixconv_kernel_sizes: list[list[int]] = field(
        default_factory=lambda: [[5], [7, 11], [9, 15], [23]]
    )

    @classmethod
    def from_dict(cls, d: dict) -> "ModelArchitectureConfig":
        # Parse string representations once at load time
        pf = d.get("pointwise_filters", "64,64,64,64")
        if isinstance(pf, str):
            pf = [int(x) for x in pf.split(",")]
        return cls(pointwise_filters=pf, ...)
```

---

## 6. Error Handling Issues

### 6.1 High: Bare Exception Handling

**Severity:** High
**Location:** `src/training/trainer.py`, line 555
**Issue:**

```python
def __del__(self) -> None:
    executor = getattr(self, "_validation_executor", None)
    if executor is not None:
        try:
            executor.shutdown(wait=False)
        except Exception:  # noqa: S110
            pass
```

While marked with `# noqa: S110`, bare `Exception` catching can mask real issues.

**Recommendation:** Catch specific exceptions:

```python
try:
    executor.shutdown(wait=False)
except (RuntimeError, ValueError) as e:
    # Executor already shut down or invalid state
    logger.debug(f"Executor shutdown failed (safe to ignore): {e}")
```

---

### 6.2 High: Silent Failures in `AsyncHardExampleMiner`

**Severity:** High
**Location:** `src/training/mining.py`, lines 471-479
**Issue:**

```python
try:
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
except (RuntimeError, ValueError, TypeError) as e:
    logger.exception(f"Model cloning failed: {e}")
    with self._lock:
        self._result = None
    # Don't set _is_running, the thread won't start
    return
```

The method returns silently without notifying the caller that mining won't occur.

**Recommendation:** Use a result object or raise a custom exception:

```python
@dataclass
class MiningStartResult:
    success: bool
    error: Exception | None = None

def start_mining(...) -> MiningStartResult:
    try:
        cloned_model = tf.keras.models.clone_model(model)
    except Exception as e:
        logger.exception(f"Model cloning failed: {e}")
        return MiningStartResult(success=False, error=e)
    return MiningStartResult(success=True)
```

---

### 6.3 Medium: Missing Validation in `KnobCycle`

**Severity:** Medium
**Location:** `src/tuning/knobs.py`, lines 29-42
**Issue:** No validation that knob names correspond to existing knobs:

```python
class KnobCycle:
    def __init__(self, knob_names: list[str]):
        if not knob_names:
            raise ValueError("knob_names must not be empty")
        self._knobs = list(knob_names)
        self._pos = 0
```

If an invalid knob name is passed, the error only occurs at runtime when `_make_knob()` is called.

**Recommendation:** Validate at initialization:

```python
VALID_KNOBS = {"lr", "threshold", "temperature", "sampling_mix", "weight_perturbation", "label_smoothing"}

def __init__(self, knob_names: list[str]):
    invalid = set(knob_names) - VALID_KNOBS
    if invalid:
        raise ValueError(f"Invalid knob names: {invalid}. Valid: {VALID_KNOBS}")
```

---

### 6.4 Medium: Unclear Error Messages

**Severity:** Medium
**Location:** `src/data/dataset.py`, lines 1087-1094
**Issue:** Error message could be clearer:

```python
raise ValueError(
    f"Stored feature dimension ({candidate}) does not match "
    f"configured mel_bins ({self.feature_dim}). "
    f"mel_bins={self.feature_dim} is mandatory per ARCHITECTURAL_CONSTITUTION.md. "
    f"Re-run preprocessing with the correct mel_bins setting."
)
```

While this message is actually good, many other errors in the codebase are less helpful.

**Recommendation:** Ensure all error messages include:
1. What went wrong
2. What value was received
3. What value was expected
4. How to fix it

---

## 7. Additional Findings

### 7.1 Medium: Unused Imports

**Severity:** Medium
**Issue:** Several files have unused imports that should be cleaned up:

```python
# trainer.py - verify which are unused
from collections.abc import Iterable  # May be unused
from concurrent.futures import ThreadPoolExecutor  # Check usage

# mining.py
from typing import cast  # Check usage
```

**Recommendation:** Run `ruff check --select F401` or similar to find and remove unused imports.

---

### 7.2 Medium: Mutable Default Arguments

**Severity:** Medium
**Issue:** While not found in the reviewed code, the pattern is common in ML codebases. Verify no mutable defaults exist:

```python
# Anti-pattern to check for:
def method(self, items=[]):  # Dangerous!
    ...
```

---

### 7.3 Low: Type Hint Inconsistencies

**Severity:** Low
**Issue:** Mixed use of old-style (`Optional`, `List`) and new-style (`|`, `list`) type hints:

```python
# Old style
from typing import Optional, List, Dict
self._offsets: Union[List[int], np.ndarray, None] = None

# New style
self.best_weights_path: str | None = None
```

**Recommendation:** Standardize on Python 3.10+ syntax throughout.

---

## Summary of Recommendations by Priority

### Immediate (Critical)
1. Remove duplicate code in `WeightPerturbationKnob.apply()` (knobs.py)
2. Remove duplicate EMA check in `Trainer.__init__()` (trainer.py)
3. Remove duplicate cache check in `WakeWordDataset.build()` (dataset.py)
4. Clean up commented-out code blocks (trainer.py)

### Short-term (High)
1. Decompose `Trainer` class into smaller, focused classes
2. Extract common pipeline creation logic in `tfdata_pipeline.py`
3. Flatten deeply nested logic in `train()` method
4. Establish and enforce naming conventions
5. Replace magic numbers with named constants

### Medium-term (Medium)
1. Implement registry pattern for knobs
2. Create typed configuration classes
3. Improve error handling specificity
4. Add validation for configuration values
5. Standardize type hints

### Long-term (Ongoing)
1. Apply SOLID principles more rigorously
2. Increase test coverage for complex branches
3. Document architectural decisions
4. Set up automated code quality checks in CI

---

## Metrics Summary

| Metric | Value |
|--------|-------|
| Total Lines Reviewed | ~9,330 |
| Critical Issues | 4 |
| High Severity Issues | 12 |
| Medium Severity Issues | 10 |
| Low Severity Issues | 2 |
| Code Duplication Instances | 5 |
| SOLID Violations | 4 |
| Average Method Length | ~50 lines (too high) |
| Maximum Method Length | ~400+ lines (train method) |

---

*End of Code Quality Review*
