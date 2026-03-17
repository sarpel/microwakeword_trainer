# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### Critical

#### CQ-C1: Duplicate Weight Perturbation Logic in `knobs.py`
- **File:** `src/tuning/knobs.py`, `WeightPerturbationKnob.apply()`, lines 113-144
- The same 8-line logic block appears twice with only variable name differences (`trainable_names` vs `trainable_set`). Both blocks do identical weight perturbation.
- **Fix:** Remove the duplicate code block (lines 133-140).

#### CQ-C2: Duplicate EMA Enable Check in `trainer.py`
- **File:** `src/training/trainer.py`, lines 530-534
- The EMA enable check is performed twice with identical conditionals.
- **Fix:** Remove the second `if ema_decay is not None: self._ema_enabled = True` block.

#### CQ-C3: Duplicate Cache Check with Dead Code in `dataset.py`
- **File:** `src/data/dataset.py`, lines 988-996
- Duplicate comment and unreachable code after `return self`.
- **Fix:** Remove lines 994-996.

#### CQ-C4: Extensive Commented-Out Code Blocks
- **File:** `src/training/trainer.py`, lines 2227-2237
- Same points duplicated twice with slight variations.
- **Fix:** Remove or consolidate duplicate comments.

### High

#### CQ-H1: `Trainer` Class God Object — 2,046 Lines
- **File:** `src/training/trainer.py`, `Trainer` class (lines 236-2282)
- Violates Single Responsibility Principle. Handles: training loop, validation, checkpointing, TensorBoard logging, mining coordination, LR scheduling, phase management, EMA, async validation.
- **Fix:** Extract `TrainingOrchestrator`, `MetricsTracker`, `CheckpointManager`, `LearningRateScheduler`, `MiningCoordinator`, `TrainingLogger` collaborators.

#### CQ-H2: Deeply Nested Logic in `train()` Method (7+ Levels)
- **File:** `src/training/trainer.py`, `train()` method, lines 1942-2183
- Excessive nesting with try/except, if/else, and nested conditionals.
- **Fix:** Extract methods: `_fetch_batch()`, `_apply_spec_augment()`, `_execute_train_step()`, `_log_step_metrics()`, `_should_validate()`, `_run_validation()`.

#### CQ-H3: Complex Validation Logic — 213 Lines, ~35 Cyclomatic Complexity
- **File:** `src/training/trainer.py`, `_validate_with_model()`, lines 1386-1599
- Multiple nested conditionals for clip ID handling, batch metadata processing, inline metrics computation.
- **Fix:** Split into `ValidationDataCollector`, `MetricsComputer`, `ClipIDResolver`.

#### CQ-H4: `Trainer.__init__()` — 308 Lines, 80+ Instance Variables
- **File:** `src/training/trainer.py`, lines 239-547
- Complex configuration parsing, inline helper function definition (`_pad_or_trim`).
- **Fix:** Use Builder pattern or dependency injection with `TrainerConfig` dataclass.

#### CQ-H5: Inconsistent Naming Conventions
- **Files:** Throughout codebase
- Private methods use both `_method` and `__method`
- Boolean flags: some use `is_` prefix (`is_hard_negative`), others don't (`_bn_frozen`)
- **Fix:** Establish naming convention: private methods use single underscore, booleans use `is_`/`has_`/`should_` prefix.

#### CQ-H6: Long Export Function — 300+ Lines
- **File:** `src/export/tflite.py`, `export_streaming_tflite()`
- Complex quantization setup with multiple nested contexts.
- **Fix:** Decompose into `ModelLoader.load_for_export()`, `RepresentativeDatasetBuilder.build()`, `QuantizationConfig.create()`, `TFLiteConverter.convert()`, `ExportVerifier.verify()`.

#### CQ-H7: Similar Pipeline Creation Methods — 80% Code Duplication
- **File:** `src/data/tfdata_pipeline.py`, lines 196-483
- `create_training_pipeline()`, `create_validation_pipeline()`, `create_test_pipeline()` share identical patterns.
- **Fix:** Extract `_create_base_pipeline()` with common logic.

#### CQ-H8: Bare Exception Handling
- **File:** `src/training/trainer.py`, line 555; `src/training/mining.py`, lines 471-479
- Catches `Exception` without specificity, masking real issues.
- **Fix:** Catch specific exceptions; log with context.

#### CQ-H9: Silent Failures in `AsyncHardExampleMiner`
- **File:** `src/training/mining.py`, lines 471-479
- Returns silently without notifying caller that mining won't occur.
- **Fix:** Return `MiningStartResult` with success/error fields.

#### CQ-H10: Magic Numbers Throughout
- **Files:** Throughout codebase
- 101 thresholds, 0.90 target recall, 2000 score limit, 1024 cache size, 7 sampling arms — no named constants.
- **Fix:** Define `Defaults` class with named constants and documentation.

#### CQ-H11: Feature Envy in `EvaluationMetrics`
- **File:** `src/training/trainer.py`, `EvaluationMetrics.compute_metrics()`
- Delegates most work to `MetricsCalculator`, suggesting logic belongs there.
- **Fix:** Move per-threshold computation to `MetricsCalculator` or merge classes.

#### CQ-H12: Open/Closed Principle Violation in Knobs
- **File:** `src/tuning/knobs.py`
- Adding new knob requires modifying `MicroAutoTuner._make_knob()`.
- **Fix:** Use registry pattern with decorator-based registration.

### Medium

#### CQ-M1: Complex Phase Settings Logic
- **File:** `src/training/trainer.py`, `_get_current_phase_settings()`, lines 669-727
- Stagger logic with nested conditionals.
- **Fix:** Create `PhaseConfiguration` dataclass with factory method.

#### CQ-M2: `EvaluationMetrics` Mixed Responsibilities
- **File:** `src/training/trainer.py`, lines 50-229
- Accumulates predictions, computes metrics, maintains array and dict views.
- **Fix:** Separate into `PredictionAccumulator` and `ThresholdMetricsComputer`.

#### CQ-M3: Duplicate Configuration Access Patterns
- **Files:** Throughout codebase
- Repeated `config.get("training", {}).get("batch_size", 384)` pattern.
- **Fix:** Create typed `TrainingConfig` dataclass with `from_dict()` factory.

#### CQ-M4: Law of Demeter Violations
- **File:** `src/training/trainer.py`
- Deep attribute chains: `self.tensorboard_logger.writer`
- **Fix:** Use Tell, Don't Ask principle; add encapsulation methods.

#### CQ-M5: Stringly-Typed Configuration
- **Files:** Configuration handling throughout
- Model architecture uses comma-separated strings instead of structured types.
- **Fix:** Use `@dataclass` with parsing in `from_dict()` factory.

#### CQ-M6: Missing Validation in `KnobCycle`
- **File:** `src/tuning/knobs.py`, lines 29-42
- No validation that knob names correspond to existing knobs.
- **Fix:** Validate against `VALID_KNOBS` set at initialization.

#### CQ-M7: TODO/FIXME Comments Without Tracking
- **Files:** Throughout codebase
- Scattered TODOs without issue references.
- **Fix:** Create GitHub issues and reference in comments; or use `TODO.md`.

#### CQ-M8: Type Hint Inconsistencies
- **Files:** Throughout codebase
- Mixed old-style (`Optional`, `List`) and new-style (`|`, `list`) type hints.
- **Fix:** Standardize on Python 3.10+ syntax.

#### CQ-M9: Unused Imports
- **Files:** Various modules
- Several files have potentially unused imports.
- **Fix:** Run `ruff check --select F401` to identify and remove.

#### CQ-M10: Unclear Error Messages
- **Files:** Various locations
- Inconsistent error message quality.
- **Fix:** Ensure all errors include: what went wrong, received value, expected value, how to fix.

### Low

- **CQ-L1:** Mutable default arguments pattern — verify none exist
- **CQ-L2:** Inline helper function `_pad_or_trim` defined in `__init__` — untestable in isolation

---

## Architecture Findings

### Critical

#### AR-C1: `Trainer` Class God Object
- **File:** `src/training/trainer.py`, `Trainer` class
- 2,046 lines handling: training loop, metrics, checkpointing, logging, mining, LR scheduling, phase management, EMA, async validation.
- **Fix:** Decompose into focused collaborators: `TrainingLoop`, `MetricsTracker`, `CheckpointManager`, `TensorBoardLogger`, `HardNegativeMiner`, `LearningRateScheduler`.

#### AR-C2: Implicit TensorFlow Dependencies
- **Files:** Throughout codebase
- TensorFlow imported at module level, causing slow imports and test isolation issues.
- **Fix:** Use lazy imports for heavy dependencies.

#### AR-C3: No Repository Pattern for Data Access
- **Files:** `src/data/dataset.py`, `src/data/tfdata_pipeline.py`
- Data access scattered across: `WakeWordDataset`, `RaggedMmap`, `partition_data()`, `OptimizedDataPipeline`.
- **Fix:** Introduce `DatasetRepository` abstraction.

### High

#### AR-H1: Knob Configuration Duplication
- **File:** `src/tuning/knobs.py`
- Each knob duplicates the same 7-line dict-vs-object config resolution logic.
- **Fix:** Extract `ConfigAccessor` utility with `_get_nested()` method.

#### AR-H2: Orchestrator Depends on Concrete Implementations
- **File:** `src/tuning/orchestrator.py`
- `MicroAutoTuner._make_knob()` directly instantiates concrete knob classes.
- **Fix:** Use registry pattern with decorator-based registration.

#### AR-H3: Inconsistent Return Types
- **File:** `src/tuning/orchestrator.py`, `_evaluate_candidate()`
- Need for `_ensure_tune_metrics()` indicates API inconsistency.
- **Fix:** Standardize on `TuneMetrics` as universal metrics exchange format.

#### AR-H4: Dataset Partitioning Logic Complexity
- **File:** `src/tuning/population.py`, `partition_data()`
- 150+ lines with multiple strategies, nested conditionals, hardcoded fractions.
- **Fix:** Extract `DataPartitioner` class with `PartitionStrategy` pattern.

#### AR-H5: No Factory Pattern for Model Creation
- **Files:** `src/model/architecture.py`, `src/training/trainer.py`, `src/tuning/orchestrator.py`
- Model creation uses single `build_model()` function with many conditionals; duplicated in multiple places.
- **Fix:** Create `ModelFactory` with registry pattern.

#### AR-H6: Mixed Concerns in Data Pipeline
- **File:** `src/data/tfdata_pipeline.py`
- `OptimizedDataPipeline` mixes cache management, generator factory, SpecAugment integration, mixed precision casting.
- **Fix:** Split into `CacheManager`, `GeneratorFactory`, `DataPipeline`.

### Medium

#### AR-M1: Pipeline Script Lacks Abstraction
- **File:** `src/pipeline.py`
- Procedural orchestration with subprocess calls.
- **Fix:** Define `PipelineStep` interface; implement step classes.

#### AR-M2: Config Access Patterns Inconsistent
- **Files:** Throughout codebase
- Pattern 1: `config["training"]["learning_rate"]`
- Pattern 2: `config.get("training", {}).get("learning_rate", 0.001)`
- Pattern 3: `getattr(config, "learning_rate", 0.001)`
- **Fix:** Standardize on single `Config` class with typed accessors.

#### AR-M3: Dataset Module Cross-Cutting Concerns
- **File:** `src/data/dataset.py`
- `WakeWordDataset` has dependencies on ingestion, features, augmentation, quality.
- **Fix:** Apply dependency inversion; inject dependencies rather than direct imports.

#### AR-M4: Candidate State Serialization (Pickle)
- **File:** `src/tuning/population.py`
- Uses `pickle` for weight serialization — not version-safe, no compression, no integrity verification.
- **Fix:** Use NumPy's `.npz` format with `allow_pickle=False`.

#### AR-M5: Missing Observer Pattern for Training Events
- **File:** `src/training/trainer.py`
- Direct calls to logging and checkpointing in training loop.
- **Fix:** Use event bus pattern: `TrainingEventBus.subscribe("step_complete", handler)`.

#### AR-M6: Mixed Error Handling Strategies
- **Files:** Throughout codebase
- Some raise exceptions, others silently return defaults, others log warnings.
- **Fix:** Define error handling policy with custom exception hierarchy.

#### AR-M7: Inconsistent Naming Conventions
- **Files:** Throughout codebase
- `FAH` vs `fah` vs `false_alarms_per_hour`
- `AUC_PR` vs `auc_pr` vs `average_precision`
- `step` vs `iteration` vs `epoch`
- **Fix:** Create project glossary; enforce via code review.

### Low

- **AR-L1:** Lazy TF imports in some files but not others — document or unify
- **AR-L2:** `ThresholdOptimizer.optimize()` annotated as bare `tuple` instead of `tuple[float, int, TuneMetrics]`

---

## Critical Issues for Phase 2 Context

The following findings are most likely to have security or performance implications:

1. **Implicit TensorFlow Dependencies** (AR-C2) — slow imports, memory overhead, test isolation issues
2. **Pickle Serialization in Population Module** (AR-M4) — version compatibility, potential security issues
3. **No Repository Pattern** (AR-C3) — makes data access auditing difficult
4. **Mixed Error Handling** (AR-M6) — may mask error conditions affecting stability
5. **God Object Trainer** (AR-C1, CQ-H1) — complexity makes security audit difficult

---

## Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Code Quality | 4 | 12 | 10 | 2 | 28 |
| Architecture | 3 | 6 | 7 | 2 | 18 |
| **Total** | **7** | **18** | **17** | **4** | **46** |

---

## Overall Assessment

**Code Quality Grade:** C+
- Significant duplication (4 critical issues)
- God object anti-pattern in Trainer
- Deep nesting and high cyclomatic complexity
- Inconsistent naming and type hints

**Architecture Grade:** B-
- Good high-level module separation
- Well-designed streaming abstraction
- Good use of Strategy pattern for knobs
- Missing Repository pattern and Factory pattern
- Tight coupling in tuning orchestration
