# Architectural Design Review: microwakeword_trainer

**Review Date:** 2026-03-18
**Scope:** Full codebase architectural assessment
**Framework:** TensorFlow/Python ML Training Pipeline

---

## Executive Summary

The microwakeword_trainer codebase implements a wake word detection training pipeline with micro-autotuning capabilities. The architecture follows a **layered design** with clear separation between data processing, model architecture, training orchestration, tuning, and export concerns. However, several **architectural inconsistencies**, **tight coupling issues**, and **missing abstractions** were identified that impact maintainability and extensibility.

**Overall Architectural Health:** B- (Good foundation with notable technical debt)

---

## 1. Component Boundaries

### 1.1 Strengths

#### Clear Module Separation
The codebase demonstrates good high-level module organization:

```
src/
├── data/          # Data loading, augmentation, feature extraction
├── model/         # Neural network architecture and streaming layers
├── training/      # Training loop, profiling, logging
├── tuning/        # Auto-tuning orchestration and hyperparameter knobs
├── evaluation/    # Metrics computation and FAH estimation
├── export/        # TFLite conversion and verification
└── utils/         # Shared utilities
```

**Assessment:** This follows **Clean Architecture** principles with clear layer boundaries.

#### Streaming Abstraction (`src/model/streaming.py`)
The `Stream` wrapper layer effectively encapsulates streaming state management:
- Supports multiple inference modes (TRAINING, NON_STREAM_INFERENCE, STREAM_INTERNAL/EXTERNAL_STATE)
- Ring buffer management is well-encapsulated
- State variable handling for TFLite conversion is isolated

```python
# Good: Clear mode-based behavior dispatch
class Stream(tf.keras.layers.Layer):
    def call(self, inputs, training=None, state=None):
        if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
            return self._streaming_internal_state(inputs)
        elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
            return self._streaming_external_state(inputs, state=state)
        # ...
```

### 1.2 Issues

#### CRITICAL: Trainer Class God Object
**Location:** `src/training/trainer.py`, `Trainer` class (lines 236+)

**Severity:** Critical
**Architectural Impact:** High

The `Trainer` class violates the **Single Responsibility Principle**. It contains:
- Training loop orchestration
- Metric tracking and computation
- Checkpoint management
- TensorBoard logging
- Hard negative mining coordination
- Learning rate scheduling (cosine, plateau-based)
- Batch normalization freezing
- EMA (Exponential Moving Average) handling
- Async validation execution
- Phase-based training configuration

**Line Count:** ~1500+ lines (estimated from partial read)

**Recommendation:**
Decompose into focused collaborators:
```python
# Proposed structure
class Trainer:
    def __init__(self, config):
        self.loop = TrainingLoop(config)
        self.metrics = MetricsTracker(config)
        self.checkpointer = CheckpointManager(config)
        self.logger = TensorBoardLogger(config)
        self.miner = HardNegativeMiner(config)
        self.scheduler = LearningRateScheduler(config)
```

#### HIGH: Knob Configuration Duplication
**Location:** `src/tuning/knobs.py`

**Severity:** High
**Architectural Impact:** Medium

Each knob implementation duplicates config access logic:

```python
# In LRKnob.apply():
if isinstance(config, dict):
    expert_config = config.get("auto_tuning_expert", config)
    if isinstance(expert_config, dict):
        lr_min, lr_max = expert_config.get("lr_range", (1e-7, 1e-4))
    else:
        lr_min, lr_max = getattr(expert_config, "lr_range", (1e-7, 1e-4))
else:
    expert_config = getattr(config, "auto_tuning_expert", config)
    lr_min, lr_max = getattr(expert_config, "lr_range", (1e-7, 1e-4))
```

This pattern repeats in `WeightPerturbationKnob`, `LabelSmoothingKnob`, etc.

**Recommendation:**
Extract a configuration accessor utility:
```python
class ConfigAccessor:
    @staticmethod
    def get_nested(config, *keys, default=None):
        # Unified dict/object access
        ...
```

#### MEDIUM: Mixed Concerns in Data Pipeline
**Location:** `src/data/tfdata_pipeline.py`

**Severity:** Medium
**Architectural Impact:** Medium

The `OptimizedDataPipeline` class mixes:
- Cache management (lockfile cleanup, path resolution)
- Generator factory creation
- SpecAugment integration
- Mixed precision casting

**Recommendation:**
Split into `CacheManager`, `GeneratorFactory`, and `DataPipeline`.

---

## 2. Dependency Management

### 2.1 Strengths

#### Clean Import Structure
The codebase avoids circular imports through careful module organization. Key architectural modules (`streaming.py`, `architecture.py`) have minimal external dependencies.

#### Interface Segregation in Metrics
**Location:** `src/tuning/metrics.py`

The `TuneMetrics` dataclass provides a clean, immutable value object for metric passing:

```python
@dataclass
class TuneMetrics:
    fah: float = float("inf")
    recall: float = 0.0
    auc_pr: float = 0.0
    # ...

    def dominates(self, other: "TuneMetrics") -> bool:
        # Pareto dominance logic
        ...
```

### 2.2 Issues

#### CRITICAL: Implicit TensorFlow Dependencies
**Location:** Throughout codebase

**Severity:** Critical
**Architectural Impact:** High

TensorFlow is imported at module level in many files, causing:
- Slow import times
- Memory overhead even when TF functionality not used
- Test isolation difficulties

**Examples:**
- `src/training/trainer.py` lines 28-30
- `src/model/streaming.py` line 31
- `src/model/architecture.py` line 13

**Recommendation:**
Use lazy imports for heavy dependencies:
```python
# Instead of:
import tensorflow as tf

# Use:
def _get_tf():
    import tensorflow as tf
    return tf
```

#### HIGH: Orchestrator Depends on Concrete Implementations
**Location:** `src/tuning/orchestrator.py`

**Severity:** High
**Architectural Impact:** Medium

The `MicroAutoTuner` directly instantiates concrete knob classes:

```python
def _make_knob(self, knob_name: str):
    knobs = {
        "lr": LRKnob,
        "threshold": ThresholdKnob,
        # ...
    }
    return knobs[knob_name]()
```

This violates the **Dependency Inversion Principle**. Adding new knobs requires modifying the orchestrator.

**Recommendation:**
Use a registry pattern with dependency injection:
```python
class KnobRegistry:
    _knobs: dict[str, Type[Knob]] = {}

    @classmethod
    def register(cls, name: str, knob_class: Type[Knob]):
        cls._knobs[name] = knob_class

    @classmethod
    def create(cls, name: str) -> Knob:
        return cls._knobs[name]()

# In knob modules:
@KnobRegistry.register("lr")
class LRKnob(Knob): ...
```

#### MEDIUM: Dataset Module Cross-Cutting Concerns
**Location:** `src/data/dataset.py`

**Severity:** Medium
**Architectural Impact:** Medium

The `WakeWordDataset` class has dependencies on:
- `src.data.ingestion` (audio loading)
- `src.data.features` (feature extraction)
- `src.data.augmentation` (data augmentation)
- `src.data.quality` (data quality checks)

While not circular, this creates a **dense dependency graph** that makes unit testing difficult.

---

## 3. API Design

### 3.1 Strengths

#### Consistent Builder Pattern for Model Creation
**Location:** `src/model/architecture.py`

```python
def build_model(
    input_shape=(100, 40),
    num_classes=2,
    first_conv_filters=32,
    # ...
    mode="non_stream",
    **kwargs,
):
```

This provides a clean, discoverable API with sensible defaults.

#### Pareto Archive Interface
**Location:** `src/tuning/metrics.py`

The `ParetoArchive` class provides a clean, minimal API:

```python
class ParetoArchive:
    def try_add(self, metrics: TuneMetrics, candidate_id: str) -> bool
    def get_best(self, target_fah: float, target_recall: float) -> Optional[tuple]
    def get_frontier_points(self) -> list[dict]
```

### 3.2 Issues

#### HIGH: Inconsistent Return Types
**Location:** `src/tuning/orchestrator.py`, `_evaluate_candidate()`

**Severity:** High
**Architectural Impact:** Medium

```python
def _evaluate_candidate(self, model, search_eval_partition: tuple) -> TuneMetrics:
    # ... returns TuneMetrics

def _ensure_tune_metrics(self, metrics: Any) -> TuneMetrics:
    # Has to handle multiple input types due to inconsistent upstream returns
```

The need for `_ensure_tune_metrics()` indicates API inconsistency.

**Recommendation:**
Standardize on `TuneMetrics` as the universal metrics exchange format.

#### MEDIUM: Pipeline Script Lacks Abstraction
**Location:** `src/pipeline.py`

**Severity:** Medium
**Architectural Impact:** Medium

The pipeline orchestration is procedural with subprocess calls:

```python
def step_train(config: str, override: str | None) -> Path:
    cmd = [sys.executable, "-m", "src.training.trainer", "--config", config]
    _run(cmd, "Training wake word model")
    # ...
```

This makes the pipeline difficult to:
- Test in isolation
- Extend with new steps
- Run in different execution environments (local, cloud, container)

**Recommendation:**
Define a `PipelineStep` interface:
```python
from abc import ABC, abstractmethod

class PipelineStep(ABC):
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineResult:
        ...

    @abstractmethod
    def can_skip(self, context: PipelineContext) -> bool:
        ...
```

#### LOW: Config Access Patterns Inconsistent
**Location:** Throughout codebase

**Severity:** Low
**Architectural Impact:** Low

Multiple config access patterns exist:
```python
# Pattern 1: Direct dict access
lr = config["training"]["learning_rate"]

# Pattern 2: .get() with defaults
lr = config.get("training", {}).get("learning_rate", 0.001)

# Pattern 3: getattr for object configs
lr = getattr(config, "learning_rate", 0.001)

# Pattern 4: Dedicated helper
lr = _get_config_value(config, "training", "learning_rate", 0.001)
```

**Recommendation:**
Standardize on a single `Config` class with typed accessors.

---

## 4. Data Model

### 4.1 Strengths

#### RaggedMmap for Variable-Length Storage
**Location:** `src/data/dataset.py`

The `RaggedMmap` class provides efficient storage for variable-length audio features using memory-mapped files with a custom index format.

#### TypedDict for Batch Buffers
```python
class _BatchBuffer(TypedDict):
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    is_hard_neg: np.ndarray
```

### 4.2 Issues

#### HIGH: Dataset Partitioning Logic Complexity
**Location:** `src/tuning/population.py`, `partition_data()`

**Severity:** High
**Architectural Impact:** Medium

The `partition_data()` function (lines 170-322) is 150+ lines with:
- Multiple partitioning strategies (random vs. group-based)
- Nested conditional logic for small dataset handling
- Hardcoded fraction values (0.15, 0.05, 0.40, 0.30)

**Recommendation:**
Extract a `DataPartitioner` class with strategy pattern:
```python
class PartitionStrategy(ABC):
    @abstractmethod
    def partition(self, dataset, config) -> DataPartition:
        ...

class RandomPartitionStrategy(PartitionStrategy): ...
class GroupPartitionStrategy(PartitionStrategy): ...
```

#### MEDIUM: Candidate State Serialization
**Location:** `src/tuning/population.py`

**Severity:** Medium
**Architectural Impact:** Medium

The `Candidate` class uses `pickle` for weight serialization:

```python
def save_state(self, model) -> None:
    self.weights_bytes = pickle.dumps(model.get_weights())
```

Issues:
- Pickle is not version-safe across Python/TensorFlow updates
- No compression for large weight tensors
- No integrity verification

**Recommendation:**
Use a versioned serialization format (e.g., NumPy's `.npz` or Protocol Buffers).

---

## 5. Design Patterns

### 5.1 Patterns Used Well

#### Strategy Pattern: Knob Implementations
**Location:** `src/tuning/knobs.py`

```python
class Knob(ABC):
    @abstractmethod
    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        ...

class LRKnob(Knob): ...
class ThresholdKnob(Knob): ...
```

#### Template Method: Metrics Calculator
**Location:** `src/evaluation/metrics.py`

The `MetricsCalculator` provides a template for metric computation with customizable parameters.

### 5.2 Missing Patterns

#### CRITICAL: No Repository Pattern for Data Access
**Severity:** Critical
**Architectural Impact:** High

Data access is scattered across:
- `WakeWordDataset` (feature storage)
- `RaggedMmap` (low-level storage)
- `partition_data()` (data splitting)
- `OptimizedDataPipeline` (TF data pipeline)

**Recommendation:**
Introduce a `DatasetRepository` abstraction:
```python
class DatasetRepository(ABC):
    @abstractmethod
    def load(self, split: str) -> Dataset:
        ...

    @abstractmethod
    def save(self, split: str, dataset: Dataset) -> None:
        ...

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        ...
```

#### HIGH: No Factory Pattern for Model Creation
**Severity:** High
**Architectural Impact:** Medium

Model creation uses a single `build_model()` function with many conditionals. Different model variants (streaming vs. non-streaming, different sizes) would benefit from a factory.

**Recommendation:**
```python
class ModelFactory:
    _registry: dict[str, Type[tf.keras.Model]] = {}

    @classmethod
    def create(cls, model_type: str, config: dict) -> tf.keras.Model:
        return cls._registry[model_type](**config)
```

#### MEDIUM: Missing Observer Pattern for Training Events
**Severity:** Medium
**Architectural Impact:** Medium

The `Trainer` class directly calls logging and checkpointing:

```python
# In training loop
self.logger.log_metrics(metrics)
self.checkpointer.save_if_best(model, metrics)
self.tensorboard_logger.log_scalars(metrics)
```

**Recommendation:**
Use an event bus:
```python
class TrainingEventBus:
    def subscribe(self, event: str, handler: Callable): ...
    def publish(self, event: str, data: Any): ...

# Usage
event_bus.subscribe("step_complete", logger.on_step)
event_bus.subscribe("step_complete", checkpointer.on_step)
```

---

## 6. Architectural Consistency

### 6.1 Consistent Patterns

1. **Type Hints:** Modern Python type hints are used throughout
2. **Docstrings:** Google-style docstrings are consistent
3. **Logging:** Structured logging with `logging.getLogger(__name__)`
4. **Error Handling:** Graceful degradation with warnings (e.g., sklearn optional)

### 6.2 Inconsistencies

#### HIGH: Mixed Error Handling Strategies
**Location:** Throughout codebase

**Severity:** High
**Architectural Impact:** Medium

Some places raise exceptions:
```python
if n < 4:
    raise ValueError(f"Dataset too small for partitioning: need at least 4 samples, got {n}")
```

Others silently return defaults:
```python
try:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, y_scores))
except ImportError:
    return _manual_roc_auc(y_true, y_scores)
```

And others log warnings:
```python
logger.warning("Only one class present in PR-AUC labels; auc_pr is undefined")
```

**Recommendation:**
Define an error handling policy:
- Programming errors (wrong types) -> raise immediately
- Runtime recoverable issues -> log warning + use default
- External dependency missing -> raise with helpful message

#### MEDIUM: Inconsistent Naming Conventions
**Location:** Throughout codebase

**Severity:** Medium
**Architectural Impact:** Low

- `FAH` vs `fah` vs `false_alarms_per_hour`
- `AUC_PR` vs `auc_pr` vs `average_precision`
- `step` vs `iteration` vs `epoch`

**Recommendation:**
Create a project glossary and enforce via code review.

---

## 7. Specific Improvement Recommendations

### 7.1 Immediate (High Priority)

1. **Refactor Trainer Class**
   - Extract `MetricsTracker`, `CheckpointManager`, `LearningRateScheduler`
   - Target: Each class < 300 lines
   - Effort: 2-3 days

2. **Implement ConfigAccessor Utility**
   - Replace all `isinstance(config, dict)` checks
   - Provide typed accessors
   - Effort: 1 day

3. **Add Repository Pattern for Data**
   - Create `DatasetRepository` interface
   - Migrate `WakeWordDataset` to use it
   - Effort: 2-3 days

### 7.2 Short-term (Medium Priority)

4. **Implement Knob Registry**
   - Add decorator-based registration
   - Remove `_make_knob()` from orchestrator
   - Effort: 1 day

5. **Standardize Error Handling**
   - Define error handling policy
   - Add custom exception hierarchy
   - Effort: 1-2 days

6. **Extract Pipeline Abstractions**
   - Create `PipelineStep` interface
   - Refactor `pipeline.py` to use it
   - Effort: 2 days

### 7.3 Long-term (Lower Priority)

7. **Lazy Import Optimization**
   - Profile import times
   - Add lazy loading for TF and other heavy deps
   - Effort: 2-3 days

8. **Event-Driven Training**
   - Implement `TrainingEventBus`
   - Migrate logging/checkpointing to event handlers
   - Effort: 3-4 days

---

## 8. Architectural Decision Records (ADRs)

### ADR-1: Streaming State Management
**Status:** Accepted
**Context:** Need to support both internal and external state for TFLite compatibility
**Decision:** Use `Modes` enum with `Stream` wrapper layer
**Consequences:** (+) Clean abstraction, (-) Some complexity in mode propagation

### ADR-2: Pareto Archive for Multi-Objective Optimization
**Status:** Accepted
**Context:** Auto-tuning needs to balance FAH and recall
**Decision:** Use Pareto frontier with hypervolume indicator
**Consequences:** (+) Proper multi-objective handling, (-) Computational overhead

---

## 9. Summary

| Category | Grade | Notes |
|----------|-------|-------|
| Component Boundaries | B | Good high-level separation, but Trainer is a god object |
| Dependency Management | B+ | Clean imports, but tight coupling in tuning |
| API Design | B | Consistent in places, inconsistent config access |
| Data Model | B | Good use of TypedDict, but complex partitioning |
| Design Patterns | B+ | Good use of Strategy, missing Repository |
| Consistency | B- | Mixed error handling, naming inconsistencies |

**Overall:** The codebase has a solid architectural foundation with clear module boundaries and appropriate use of design patterns. The main issues are the `Trainer` god object, inconsistent configuration access patterns, and missing abstractions for data access. Addressing these would significantly improve maintainability and testability.

---

*End of Architectural Review*
