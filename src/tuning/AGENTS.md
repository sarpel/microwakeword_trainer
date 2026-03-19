# Auto-Tuning Module

**Module:** `src/tuning/`
**Purpose:** Post-training auto-tuning for FAH/recall optimization using MicroAutoTuner

## OVERVIEW

MicroAutoTuner - Lightweight population-based post-training optimization to achieve target FAH/recall without full retraining. Uses cyclic coordinate descent across 6 knobs, Pareto archive with hypervolume tracking, and exploit/explore phases.

## STRUCTURE

```
src/tuning/
├── orchestrator.py   # MicroAutoTuner (main loop)
├── metrics.py        # TuneMetrics, ParetoArchive, ErrorMemory, hypervolume
├── population.py     # Candidate, Population, partition_data
├── knobs.py          # 6 knob types + KnobCycle + FocusedSampler
├── dashboard.py      # Rich console tables, JSON artifact output
└── cli.py            # mww-autotune CLI entry point
```

## Key Components

1. **MicroAutoTuner** - Main orchestration loop with population-based search
2. **TuneMetrics** - Evaluation metrics dataclass with Pareto dominance
3. **ParetoArchive** - Multi-objective non-dominated archive (NEW API: try_add(metrics, candidate_id))
4. **Candidate** - State container with get_weights/set_weights serialization
5. **Population** - Candidate pool management with exploit/explore
6. **KnobCycle** - Cyclic iterator over 6 tuning knobs
7. **FocusedSampler** - Strategy-specific batch construction
8. **ErrorMemory** - Tracks persistent false alarms and misses
9. **ThresholdOptimizer** - 3-pass threshold optimization (coarse, fine, CV)
10. **TuningDashboard** - Rich console visualization
11. **partition_data()** - Splits data into cal/search_train/search_eval/confirm/representative

## API Contract

### MicroAutoTuner (orchestrator.py)

```python
MicroAutoTuner(
    checkpoint_path: str | Path,
    config: dict,
    auto_tuning_config: Any,  # AutoTuningExpertConfig dataclass
    console=None,
    users_hard_negs_dir=None,
    dry_run: bool = False,
)

tune() -> dict
# Returns:
#   best_fah, best_recall, best_auc_pr
#   best_checkpoint (path to saved weights)
#   hypervolume_history (list[float])
#   iterations_completed (int)
#   pareto_frontier (list[dict])
```

### ParetoArchive (metrics.py)

```python
ParetoArchive(max_size=32)
  .try_add(metrics: TuneMetrics, candidate_id: str) -> bool  # True if added
  .get_best(target_fah, target_recall) -> tuple[TuneMetrics, str] | None
  .get_frontier_points() -> list[dict]  # dicts with id/fah/recall/auc_pr/threshold
  .__len__() -> int

TuneMetrics(
    fah=float("inf"),
    recall=0.0,
    auc_roc=0.0,
    auc_pr=0.0,
    ece=1.0,
    threshold=0.5,
    threshold_uint8=128,
    precision=0.0,
    f1=0.0,
)
  .dominates(other) -> bool  # Pareto dominance check
  .meets_target(target_fah, target_recall) -> bool
```

### Candidate / Population (population.py)

```python
Candidate(
    id: str,
    weights_bytes=None,      # pickle.dumps(model.get_weights())
    optimizer_state_bytes=None,
    batchnorm_state=None,
    temperature=1.0,
    metrics=None,
    knob_history=[],         # tracks applied knobs
)
  .save_state(model)        # uses model.get_weights()
  .restore_state(model)     # uses model.set_weights()

Population(model, size=4)
  .get_best() -> Candidate
  .get_worst() -> Candidate
  .exploit_explore(model, perturbation_scale=0.01)  # clones best to worst + noise

partition_data(dataset, config, expert_config) -> dict
# Keys: cal, search_train, search_eval, confirm, representative, fold_indices
```

### KnobCycle / Knobs (knobs.py)

```python
KnobCycle([
    "lr", "threshold", "temperature",
    "sampling_mix", "weight_perturbation", "label_smoothing"
])
  .current() -> str    # current knob name
  .advance() -> None   # move to next
  .position() -> int   # current index

# Knob implementations:
LRKnob              # Adjusts optimizer learning_rate via assign()
ThresholdKnob       # Creates ThresholdOptimizer instance
temperatureKnob     # Fits temperature scaling on calibration probs
SamplingMixKnob     # Rotates sampler arm
WeightPerturbationKnob  # Adds Gaussian noise to trainable weights
LabelSmoothingKnob  # Sets model._label_smoothing_var
```

### Dashboard (dashboard.py)

```python
TuningDashboard(console=None)
  .render_population_table(candidates) -> Table
  .render_pareto_table(frontier) -> Table
  .render_knob_table(current_knob, cycle_position, knob_cycle) -> Table
  .render_hypervolume_history(history) -> str

save_artifacts(
    output_dir,
    candidates,
    frontier,
    hypervolume_history,
    iteration,
    best_candidate,
)
```

## Weight Serialization

**CRITICAL**: Use `model.get_weights()` / `model.set_weights()` (NOT `trainable_weights`) to include BatchNorm moving statistics.

```python
# In Candidate.save_state():
self.weights_bytes = pickle.dumps(model.get_weights())

# In Candidate.restore_state():
model.set_weights(pickle.loads(self.weights_bytes))
```

## CLI Usage

```bash
# Basic usage
mww-autotune --checkpoint checkpoints/best.weights.h5 --config standard

# Custom targets
mww-autotune --checkpoint checkpoints/best.weights.h5 --config standard \
    --target-fah 0.2 --target-recall 0.95

# Dry run (validate config)
mww-autotune --checkpoint checkpoints/best.weights.h5 --config standard --dry-run

# Override population and burst steps
mww-autotune --checkpoint checkpoints/best.weights.h5 --config standard \
    --population-size 4 --micro-burst-steps 50
```

## Config Fields (AutoTuningExpertConfig)

| Field | Default | Description |
|-------|---------|-------------|
| population_size | 4 | Number of candidates in population |
| micro_burst_steps | 50 | Gradient steps per knob application |
| knob_cycle | ["lr", ...] | Order of knobs to cycle through |
| exploit_explore_interval | 6 | Iterations between exploit/explore phases |
| weight_perturbation_scale | 0.01 | Noise scale for weight perturbation |
| label_smoothing_range | (0.0, 0.15) | Range for label smoothing |
| lr_range | (1e-7, 1e-4) | Learning rate bounds |
| hypervolume_reference | (10.0, 0.0) | Reference point for hypervolume |
| pareto_archive_size | 32 | Max size of Pareto archive |
| search_eval_fraction | 0.30 | Fraction of search data for evaluation |

## ANTI-PATTERNS

- **Don't use `model.trainable_weights` for serialization** → Use `get_weights()`/`set_weights()` to include BatchNorm moving statistics
- **Don't call `model.compile()` during tuning** → Label smoothing uses `tf.Variable` and compile would overwrite it
- **Don't evaluate on `search_train` data** → Use `search_eval` only to prevent train-on-test contamination
- **Don't use Thompson sampling / SAM / SWA / stir levels** → Removed in redesign
- **Don't use old `AutoTuner` API directly** → `AutoTuner` is an alias for `MicroAutoTuner` in cli.py only for backward compatibility
- **Don't destroy optimizer state** → Use `optimizer.learning_rate.assign()` not re-creation
- **Don't skip confirmation phase** → Final validation on held-out confirm data required
- **Don't use fixed search threshold in confirmation** → Re-optimize threshold on confirm data

## KnobCycle Workflow

Each iteration:
1. Get current knob from cycle
2. For each candidate: restore state → apply knob → run gradient burst → evaluate → save state
3. Advance cycle position
4. Every `exploit_explore_interval` iterations: clone best candidate to worst + perturb
5. Update Pareto archive and compute hypervolume
6. Check convergence (max iterations or no improvement for `patience` iterations)

## Related Documentation

- [Configuration Reference](../../docs/CONFIGURATION.md)
- [Project AGENTS.md](../../AGENTS.md)
