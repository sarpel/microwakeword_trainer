# MicroStepAutoTuner — Definitive Implementation Plan

> **Date**: 2026-03-09
> **Status**: Design Complete — Ready for Implementation
> **Replaces**: `src/tuning/autotuner.py` (current AutoTuner — catastrophically broken)
> **Philosophy**: "I can use all the time for one single successful model than hurried half-good models."

---

## Table of Contents

1. [Problem Statement — Why The Current AutoTuner Is Broken](#1-problem-statement)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Details](#4-component-details)
5. [Strategy Arms](#5-strategy-arms)
6. [Threshold Optimization — 3-Pass System](#6-threshold-optimization)
7. [Temperature Scaling & Calibration](#7-temperature-scaling--calibration)
8. [Curriculum System](#8-curriculum-system)
9. ["Stir The Boiler" Mechanisms](#9-stir-the-boiler-mechanisms)
10. [Acceptance & Annealing](#10-acceptance--annealing)
11. [Confirmation & INT8 Verification](#11-confirmation--int8-verification)
12. [Main Loop — Complete Pseudocode](#12-main-loop--complete-pseudocode)
13. [Configuration Surface](#13-configuration-surface)
14. [Edge Cases & Failure Modes](#14-edge-cases--failure-modes)
15. [Key Differences From Current AutoTuner](#15-key-differences)
16. [Implementation Roadmap](#16-implementation-roadmap)

---

## 1. Problem Statement

### Current AutoTuner Failure Analysis

The current `AutoTuner` in `src/tuning/autotuner.py` is **completely non-functional**. Evidence from 19 iterations of a live training run:

| Metric    | Iteration 1 | Iteration 19 | Change |
|-----------|-------------|--------------|--------|
| FAH       | 169.3241    | 169.3241     | **Zero** |
| Recall    | 1.0000      | 1.0000       | **Zero** |
| Threshold | 0.0000      | 0.0000       | **Zero** |

Every single iteration produces **identical results**. ~2 hours of GPU time wasted with zero progress.

### Root Causes

**Flaw 1: Full retraining from checkpoint each iteration.**
Each iteration creates a brand-new `Trainer` + `WakeWordDataset`, loads `best_weights.weights.h5`, and trains for ~12,000 steps. This is complete retraining, not fine-tuning. The model re-converges to the same basin regardless of class weight tweaks. The optimizer state is lost between iterations (Keras warning: "Skipping variable loading for optimizer 'adam'").

**Flaw 2: No threshold optimization.**
The evaluation threshold is `0.0000` — everything is predicted positive. Recall = 1.0 trivially. FAH = 169.32 because all negatives are false alarms. The tuner never searches for a better threshold.

**Flaw 3: Class-weight knobs can't fix the fundamental problem.**
Increasing `negative_class_weight` from 20→50 won't change the model's score distribution enough when training always re-converges from the same starting point. The model needs surgical gradient updates, not full retraining with slightly different loss weights.

**Flaw 4: No calibration.**
Raw model scores are not calibrated probabilities. Threshold search on uncalibrated scores is unreliable.

**Flaw 5: No INT8 verification.**
The model is deployed as INT8 on ESP32. Improvements that don't survive quantization are worthless.

---

## 2. Design Philosophy

### Core Principles

1. **Quality over speed** — Time cost does not matter. We want one perfect model, not many mediocre ones.
2. **Branch-and-confirm search campaign** — Maintain a pool of diverse candidates on the Pareto frontier. Try different strategies from different parents. Confirm winners on held-out data.
3. **Calibration and threshold as first-class citizens** — Temperature scaling before threshold search. 3-pass robust threshold optimization. INT8-aware thresholding.
4. **Surgical gradient updates** — 500-3000 gradient steps per burst (not 12,000). Preserve optimizer state. Freeze BatchNorm.
5. **Explore then exploit** — Thompson-style bandit selects strategy arms. Start broad, narrow as the Pareto frontier matures.
6. **Never trust float-only improvements** — INT8 shadow evaluation. Confirmation set verification. Bootstrap confidence intervals.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      MaxQualityAutoTuner                             │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐               │
│  │ Campaign     │  │ Strategy     │  │ Evaluation    │               │
│  │ State        │  │ Selector     │  │ Rig           │               │
│  │              │  │ (Bandit)     │  │               │               │
│  │ • active_pool│  │ • 7 arms     │  │ • Temperature │               │
│  │ • pareto     │  │ • Thompson   │  │   scaling     │               │
│  │   archive    │  │   sampling   │  │ • 3-pass      │               │
│  │ • annealing  │  │ • Regime     │  │   threshold   │               │
│  │   temp       │  │   diagnosis  │  │ • CV folds    │               │
│  └──────┬───────┘  └──────┬───────┘  │ • Bootstrap   │               │
│         │                 │          │ • INT8 shadow  │               │
│         │                 │          └───────┬────────┘               │
│  ┌──────┴─────────────────┴──────────────────┴───────┐               │
│  │                    Main Loop                       │               │
│  │  select_parent → select_strategy → train_burst →  │               │
│  │  evaluate → calibrate → threshold_search →        │               │
│  │  accept/reject → update_frontier → stir if stuck  │               │
│  └──────────────────────────┬────────────────────────┘               │
│                             │                                        │
│  ┌──────────────────────────┴────────────────────────┐               │
│  │              Confirmation Phase                    │               │
│  │  shortlist(5) → confirmation_eval → INT8_eval →   │               │
│  │  exhaust_local_search → crown_winner              │               │
│  └───────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Partitioning (Before Campaign Starts)

The validation set is split into non-overlapping subsets, grouped by speaker/session to prevent leakage:

| Subset | Fraction | Purpose |
|--------|----------|---------|
| **Calibration** | 15% | Temperature scaling (Platt scaling / isotonic regression) |
| **Search** | 60% | 5-fold grouped CV for threshold evaluation |
| **Confirmation** | 20% | Held-out final verification (never seen during search) |
| **BN/INT8 representative** | 5% | BatchNorm stat refresh + INT8 quantization representative dataset |

---

## 4. Component Details

### 4.1 CandidateState

A candidate is a complete snapshot of a model at a specific tuning state:

```python
@dataclass
class CandidateState:
    # Model state
    weights: bytes                    # Serialized model weights
    optimizer_state: bytes            # Serialized optimizer state (CRITICAL: preserved between bursts)
    batchnorm_state: dict             # Running mean/variance for all BN layers
    swa_buffers: Optional[bytes]      # Stochastic Weight Averaging accumulated weights
    
    # Calibration & threshold
    temperature: float                # Temperature scaling parameter (default 1.0)
    threshold_float32: float          # Best operating threshold (float model)
    threshold_uint8: int              # Best operating threshold (INT8 model, 0-255)
    
    # Evaluation results (cached)
    eval_results: TuneMetrics         # Full metrics at best threshold
    eval_results_int8: TuneMetrics    # Metrics after INT8 quantization
    
    # Metadata
    sharpness_score: float            # Loss landscape sharpness (from SAM probing)
    curriculum_stage: int             # Current curriculum stage (0-3)
    strategy_arm: int                 # Which strategy produced this candidate
    parent_id: str                    # Which candidate this branched from
    iteration: int                    # When this was created
    history: list[dict]               # Full tuning history for this branch
```

### 4.2 CampaignState

The global state of the tuning campaign:

```python
@dataclass
class CampaignState:
    active_pool: list[CandidateState]    # Active candidates (max 12)
    pareto_archive: ParetoArchive        # Non-dominated archive (max 24)
    strategy_posterior: dict              # Per-arm success rates for Thompson sampling
    error_memory: ErrorMemory            # Recent false alarms and misses per candidate
    annealing_temperature: float         # Simulated annealing temp (starts 1.0, cools)
    best_confirmed_candidate: Optional[CandidateState]  # Winner after confirmation
    total_gradient_steps: int            # Total gradient steps taken across all bursts
    wall_clock_start: float              # For logging
```

### 4.3 ParetoArchive

Non-dominated set on 4 objectives:

| Objective | Direction | Weight |
|-----------|-----------|--------|
| FAH (upper CI) | Minimize ↓ | Primary |
| Recall (lower CI) | Maximize ↑ | Primary |
| INT8 degradation | Minimize ↓ | Secondary |
| AUC-PR | Maximize ↑ | Secondary |

**Tiebreakers** (when Pareto-equivalent):
1. Lower ECE (expected calibration error)
2. Higher decision boundary margin (gap between P95 of negatives and P05 of positives at threshold)
3. Lower sharpness score

**Diversity filter**: Reject candidates whose (FAH, recall) is within Euclidean distance 0.01 of any existing archive member (prevents crowding).

### 4.4 ErrorMemory

Tracks per-example error patterns across iterations:

```python
@dataclass
class ErrorMemory:
    persistent_false_alarms: dict[str, int]    # example_id → count of times it was FA
    persistent_misses: dict[str, int]           # example_id → count of times it was missed
    recent_scores: dict[str, list[float]]       # example_id → last N scores
```

This enables:
- **Curriculum**: Start with easy-to-fix errors, progress to persistent ones
- **FocusedSampler**: Build batches biased toward current errors
- **Diagnosis**: If the same examples are always wrong, it's a data/label issue, not a model issue

---

## 5. Strategy Arms

The strategy selector is a contextual Thompson-style bandit with 7 arms. Each arm defines a specific fine-tuning approach.

### Arm 1: Boundary Polish
- **Purpose**: Gentle refinement of the decision boundary
- **Steps**: 500–1000
- **LR**: Inherited from optimizer state (no reset)
- **Batch composition**: 50% near-threshold examples (scores within ±0.1 of current threshold), 50% broad replay
- **Class weights**: Default (no adjustment)
- **When**: Default arm. Good when model is close to targets.

### Arm 2: FA Suppression
- **Purpose**: Aggressively reduce false alarms
- **Steps**: 750–1500
- **LR**: 5e-6 to 2e-5
- **Batch composition**: 60% current false alarms (negatives scoring above threshold), 20% hard negatives, 20% broad replay
- **Class weights**: Pulse negative weight to 2-3× current, then cosine decay back to 1.2× by end of burst
- **Loss modification**: Add a margin penalty term: `max(0, score - (threshold - margin))` for negative examples
- **When**: FAH >> target, recall near target

### Arm 3: Recall Recovery
- **Purpose**: Improve detection of wake words without losing FA rejection
- **Steps**: 750–1500
- **LR**: 1e-5 to 5e-5
- **Batch composition**: 60% current misses (positives scoring below threshold), 20% easy positives (for stability), 20% broad replay
- **Class weights**: Boost positive weight to 2-3×, gentle increase to hard-neg weight
- **When**: Recall << target, FAH near target

### Arm 4: SAM Flatten
- **Purpose**: Escape sharp minima for more robust generalization
- **Steps**: 1500–3000
- **LR**: 3e-5
- **Technique**: Sharpness-Aware Minimization (SAM) — ascent step with `rho=0.05`, then descent step
- **SWA**: Collect weight snapshots every 100 steps; at end of burst, evaluate three variants: (a) endpoint weights, (b) best interim checkpoint, (c) SWA-averaged weights. Keep the best.
- **When**: Sharpness score is high (detected via loss landscape probing), or recent gains are fragile (improvement reverts after next arm)

### Arm 5: Cyclic Operating-Point Sweep
- **Purpose**: Explore different score distribution shapes by training toward different operating points
- **Steps**: 1500–3000
- **LR**: Cosine warm restarts (spike to 5e-5, decay to 1e-6, repeat 3 cycles)
- **Technique**: In each cycle, set a slightly different operating threshold target: `t-δ`, `t`, `t+δ` where δ=0.02. This encourages the model to learn different score distributions, one of which may dominate.
- **When**: The Pareto frontier is stagnant but the model's AUC suggests more capacity exists

### Arm 6: Macro Refine
- **Purpose**: Longer, deeper refinement from a promising parent
- **Steps**: 2000–5000
- **LR**: Cosine schedule from 3e-5 to 1e-6
- **Batch composition**: Standard proportional sampling with curriculum weighting
- **When**: A promising candidate needs more compute to fully realize its potential. Also used as a "deep exploration" arm when the frontier has been stagnant and annealing temperature is being reheated.

### Arm 7: Hardest-Only Shock
- **Purpose**: Force the model to attend to its worst errors
- **Steps**: 200–500 (short!)
- **LR**: 1e-5
- **Batch composition**: 100% hardest examples — top-K false alarms by confidence AND top-K misses by severity
- **Constraint**: Never run twice consecutively (risk of catastrophic forgetting)
- **When**: Stir mechanism level 3+, or when specific persistent errors are blocking the targets

### Arm Selection (Thompson Sampling)

```python
def select_arm(campaign: CampaignState, candidate: CandidateState) -> int:
    regime = diagnose_regime(candidate.eval_results, targets)
    
    # Regime diagnosis determines arm eligibility
    eligible_arms = get_eligible_arms(regime, candidate.history)
    
    # Thompson sampling: draw from Beta posteriors
    scores = {}
    for arm_id in eligible_arms:
        alpha = campaign.strategy_posterior[arm_id]['successes'] + 1
        beta = campaign.strategy_posterior[arm_id]['failures'] + 1
        
        # Regime bonus: arms matching current need get +0.3 bonus
        regime_bonus = get_regime_bonus(arm_id, regime)
        
        scores[arm_id] = np.random.beta(alpha, beta) + regime_bonus
    
    return max(scores, key=scores.get)

def diagnose_regime(metrics: TuneMetrics, targets: Targets) -> str:
    fah_excess = max(0, metrics.fah / targets.fah - 1)  # 0 = at target, >0 = over
    recall_deficit = max(0, targets.recall - metrics.recall)  # 0 = at target, >0 = under
    
    if fah_excess < 0.1 and recall_deficit < 0.05:
        return "near_feasible"  # Almost there
    elif fah_excess > 2 * recall_deficit:
        return "fah_dominated"   # FAH is the bottleneck
    elif recall_deficit > 2 * fah_excess:
        return "recall_dominated"  # Recall is the bottleneck
    else:
        return "balanced"  # Both need work
```

---

## 6. Threshold Optimization — 3-Pass System

After every burst evaluation, the threshold is optimized using a robust 3-pass system.

### Pass 1: Coarse Quantile Sweep (4096 thresholds)

```python
def coarse_sweep(scores: np.ndarray, labels: np.ndarray, n_thresholds=4096):
    """Sweep 4096 evenly-spaced quantile thresholds to find promising region."""
    thresholds = np.quantile(scores, np.linspace(0, 1, n_thresholds))
    
    results = []
    for t in thresholds:
        fah = compute_fah(scores, labels, t, ambient_hours)
        recall = compute_recall(scores, labels, t)
        results.append((t, fah, recall))
    
    # Find the feasible region (FAH <= target AND recall >= target)
    feasible = [(t, fah, r) for t, fah, r in results if fah <= target_fah and r >= target_recall]
    
    if feasible:
        # Best feasible: maximize recall, break ties by minimizing FAH
        best = max(feasible, key=lambda x: (x[2], -x[1]))
        region = (best[0] - 0.02, best[0] + 0.02)
    else:
        # No feasible point: minimize J = α * fah_excess + β * recall_deficit
        best = min(results, key=lambda x: scalarized_cost(x[1], x[2]))
        region = (best[0] - 0.05, best[0] + 0.05)
    
    return region, results
```

### Pass 2: Exact Unique Score Sweep (within best region)

```python
def exact_sweep(scores: np.ndarray, labels: np.ndarray, region: tuple):
    """Use every unique score value within the promising region as a threshold."""
    lo, hi = region
    unique_scores = np.unique(scores[(scores >= lo) & (scores <= hi)])
    
    results = []
    for t in unique_scores:
        fah = compute_fah(scores, labels, t, ambient_hours)
        recall = compute_recall(scores, labels, t)
        margin = compute_boundary_margin(scores, labels, t)
        results.append((t, fah, recall, margin))
    
    return results
```

### Pass 3: Robust Local Refinement (CV + worst-case regret + INT8 margin)

```python
def robust_refinement(model, search_folds, top_k_thresholds, int8_model=None):
    """Cross-validate top-K thresholds. Rank by worst-fold performance."""
    candidates = []
    
    for t in top_k_thresholds:
        fold_results = []
        for fold_data in search_folds:
            scores = model.predict(fold_data.x)
            fah = compute_fah(scores, fold_data.labels, t, fold_data.ambient_hours)
            recall = compute_recall(scores, fold_data.labels, t)
            fold_results.append((fah, recall))
        
        # Worst-case regret across folds
        worst_fah = max(fr[0] for fr in fold_results)
        worst_recall = min(fr[1] for fr in fold_results)
        mean_fah = np.mean([fr[0] for fr in fold_results])
        mean_recall = np.mean([fr[1] for fr in fold_results])
        
        # INT8 margin (if shadow model available)
        if int8_model:
            int8_scores = int8_model.predict(search_folds[0].x)
            int8_fah = compute_fah(int8_scores, search_folds[0].labels, t)
            int8_recall = compute_recall(int8_scores, search_folds[0].labels, t)
            int8_degradation = abs(mean_fah - int8_fah) + abs(mean_recall - int8_recall)
        else:
            int8_degradation = 0
        
        candidates.append({
            'threshold': t,
            'worst_fah': worst_fah,
            'worst_recall': worst_recall,
            'mean_fah': mean_fah,
            'mean_recall': mean_recall,
            'int8_degradation': int8_degradation,
            'robust_score': compute_robust_score(worst_fah, worst_recall, int8_degradation)
        })
    
    # Rank by robust_score (feasibility first, then quality)
    return sorted(candidates, key=lambda c: c['robust_score'], reverse=True)
```

---

## 7. Temperature Scaling & Calibration

Before threshold search, model scores are calibrated using temperature scaling (Platt scaling variant).

```python
def calibrate_temperature(model, calibration_data):
    """Learn a single temperature parameter T such that softmax(logits/T) is calibrated."""
    logits = model.predict(calibration_data.x, return_logits=True)
    labels = calibration_data.labels
    
    # Optimize T via L-BFGS on NLL
    from scipy.optimize import minimize_scalar
    
    def nll(T):
        scaled = sigmoid(logits / T)
        return -np.mean(labels * np.log(scaled + 1e-8) + (1 - labels) * np.log(1 - scaled + 1e-8))
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    temperature = result.x
    
    # Compute ECE (Expected Calibration Error) before and after
    ece_before = compute_ece(sigmoid(logits), labels, n_bins=20)
    ece_after = compute_ece(sigmoid(logits / temperature), labels, n_bins=20)
    
    # Constraint: temperature must be monotonic (T >= 1.0 preferred for overconfident models)
    # If ECE doesn't improve, keep T=1.0
    if ece_after >= ece_before:
        temperature = 1.0
    
    return temperature, ece_after
```

**When to recalibrate**: After every burst that is accepted into the Pareto archive. NOT after rejected bursts (waste of compute).

---

## 8. Curriculum System

Fine-tuning progresses through 4 curriculum stages, advancing as the model improves:

### Stage 0: Boundary Cleanup
- **Focus**: Examples very close to the current threshold (±0.05 in score space)
- **Batch mix**: 70% near-boundary, 30% broad replay
- **Advance when**: >50% of near-boundary errors are fixed

### Stage 1: Moderate Violators
- **Focus**: False alarms scoring 0.1-0.3 above threshold; misses scoring 0.1-0.3 below threshold
- **Batch mix**: 60% moderate violators, 40% broad replay
- **Advance when**: FAH reduced by >30% or recall improved by >10%

### Stage 2: Persistent Hard Cases
- **Focus**: Examples that have been errors for 3+ consecutive evaluations (from ErrorMemory)
- **Batch mix**: 50% persistent errors, 30% recent errors, 20% broad replay
- **Advance when**: >30% of persistent errors resolved

### Stage 3: Consolidation
- **Focus**: Maintain gains while polishing
- **Batch mix**: 20-30% focused (remaining errors), 70-80% broad replay
- **Advance when**: N/A (final stage)

**Curriculum stage is per-candidate** — different branches can be at different stages.

---

## 9. "Stir The Boiler" Mechanisms

When the Pareto frontier is stagnant (no improvement for `patience` iterations), escalating perturbation mechanisms activate.

### Level 1: SWA Collection (Stagnation count 3+)
- During the next burst, collect weight snapshots every 100 gradient steps
- At burst end, evaluate 3 variants:
  1. Endpoint weights (normal)
  2. Best interim checkpoint (by loss)
  3. SWA-averaged weights
- Keep whichever scores best on the evaluation rig
- **Rationale**: SWA finds wider optima that generalize better

### Level 2: SAM Flatten (Stagnation count 5+)
- Switch optimizer to SAM (Sharpness-Aware Minimization)
- Each gradient step becomes: ascend in weight space by `rho=0.05`, then descend from that perturbed point
- Costs 2× per step but finds flatter minima
- **When to trigger**: Also triggered when sharpness_score (from loss landscape probing) exceeds threshold

### Level 3: Loss Landscape Probing (Stagnation count 7+)
- Evaluate the model at 4-8 random perturbation directions in weight space
- Compute loss at `weights ± α * random_direction` for `α` in [0.01, 0.05, 0.1]
- **Diagnosis**:
  - Sharp valley → SAM + SWA (Level 2)
  - Flat plateau → Need stronger gradient signal (increase LR temporarily)
  - Asymmetric → The direction of improvement exists, use it as a gradient hint

### Level 4: Cyclical Operating-Point Sweep (Stagnation count 9+)
- Train toward 3 different operating points in sequence:
  1. Conservative: threshold_target = current_threshold + 0.03 (favor lower FAH)
  2. Nominal: threshold_target = current_threshold
  3. Aggressive: threshold_target = current_threshold - 0.03 (favor higher recall)
- Use cosine warm restarts between cycles
- **Rationale**: Different operating points produce different score distributions; one may break the stagnation

### Level 5: Diversify + Macro Refine (Stagnation count 12+)
- Reheat annealing temperature (allow worse candidates temporarily)
- Spawn a Macro Refine burst (Arm 6) from the most diverse Pareto archive member
- Inject 2-3 new candidates by perturbing the best candidate's weights with Gaussian noise (σ=0.001)
- **Nuclear option**: If this doesn't work, the bottleneck is data/labels/architecture, not tuning

---

## 10. Acceptance & Annealing

### Simulated Annealing Acceptance

Not all improvements are Pareto-dominating. To escape local optima, we use simulated annealing-style acceptance:

```python
def accept_candidate(new_metrics, best_metrics, annealing_temp, pareto_archive):
    # Always accept if Pareto-improving
    if pareto_archive.dominates(new_metrics):
        return True, "pareto_improving"
    
    # Compute scalarized cost (lower is better)
    J_new = scalarized_cost(new_metrics.fah, new_metrics.recall, targets)
    J_best = scalarized_cost(best_metrics.fah, best_metrics.recall, targets)
    
    delta = J_new - J_best  # positive = worse
    
    if delta <= 0:
        return True, "scalarized_improving"
    
    # Probabilistic acceptance of worse solutions (simulated annealing)
    accept_prob = math.exp(-delta / annealing_temp)
    if random.random() < accept_prob:
        return True, f"annealing_accepted (p={accept_prob:.3f})"
    
    return False, "rejected"

def update_annealing(campaign, accepted):
    if accepted:
        # Cool down (exploit more)
        campaign.annealing_temperature *= 0.95
    else:
        # Slow reheat after consecutive rejections
        campaign.consecutive_rejections += 1
        if campaign.consecutive_rejections >= 5:
            campaign.annealing_temperature = min(
                campaign.annealing_temperature * 1.3,
                1.0  # max temperature
            )
            campaign.consecutive_rejections = 0
```

### Scalarized Cost Function

```python
def scalarized_cost(fah, recall, targets):
    """Weighted violation distance. Lower is better. 0 = targets met."""
    fah_excess = max(0, fah / targets.fah - 1)       # 0 if at target, >0 if over
    recall_deficit = max(0, targets.recall - recall)   # 0 if at target, >0 if under
    
    # FAH gets higher weight because it's harder to reduce
    return 2.0 * fah_excess + 1.0 * recall_deficit
```

---

## 11. Confirmation & INT8 Verification

### INT8 Shadow Evaluation (During Search)

Every N iterations (default 5), or whenever a candidate enters the Pareto archive:

```python
def int8_shadow_eval(candidate, representative_data, search_data):
    """Convert model to INT8, evaluate, and check degradation."""
    # 1. Export to TFLite INT8
    tflite_model = export_int8(candidate.model, representative_data)
    
    # 2. Run inference on search data
    int8_scores = run_tflite_inference(tflite_model, search_data)
    
    # 3. Compute metrics at candidate's threshold (quantized to uint8)
    uint8_threshold = int(candidate.threshold_float32 * 255)
    int8_fah = compute_fah(int8_scores, search_data.labels, uint8_threshold / 255)
    int8_recall = compute_recall(int8_scores, search_data.labels, uint8_threshold / 255)
    
    # 4. Compute degradation
    degradation = {
        'fah_delta': abs(int8_fah - candidate.eval_results.fah),
        'recall_delta': abs(int8_recall - candidate.eval_results.recall),
        'score_rmse': np.sqrt(np.mean((int8_scores - candidate.eval_results.scores) ** 2))
    }
    
    return int8_fah, int8_recall, degradation
```

### Confirmation Phase (After Search Campaign)

When the search campaign concludes (targets met, max iterations, or stagnation), the top candidates undergo rigorous confirmation:

```python
def confirmation_phase(pareto_archive, confirmation_data, representative_data, targets):
    """Final verification on held-out confirmation set."""
    
    # 1. Shortlist top 5 from archive (closest to target region)
    shortlist = pareto_archive.top_k_by_target_distance(k=5, targets=targets)
    
    confirmed_winners = []
    for candidate in shortlist:
        # 2. Evaluate on frozen confirmation set with frozen temperature + threshold
        # (NO re-optimization allowed — this is a pure out-of-sample test)
        conf_scores = candidate.model.predict(confirmation_data.x)
        conf_scores_calibrated = sigmoid(logit(conf_scores) / candidate.temperature)
        
        conf_fah = compute_fah(conf_scores_calibrated, confirmation_data.labels, 
                               candidate.threshold_float32, confirmation_data.ambient_hours)
        conf_recall = compute_recall(conf_scores_calibrated, confirmation_data.labels,
                                     candidate.threshold_float32)
        
        # 3. INT8 confirmation
        int8_fah, int8_recall, _ = int8_shadow_eval(candidate, representative_data, confirmation_data)
        
        # 4. Bootstrap confidence intervals (95%)
        fah_ci, recall_ci = bootstrap_ci(conf_scores_calibrated, confirmation_data.labels,
                                          candidate.threshold_float32, n_bootstrap=1000)
        
        # 5. BOTH float32 AND INT8 must pass
        float_pass = conf_fah <= targets.fah and conf_recall >= targets.recall
        int8_pass = int8_fah <= targets.fah * 1.2 and int8_recall >= targets.recall * 0.95
        
        if float_pass and int8_pass:
            confirmed_winners.append({
                'candidate': candidate,
                'conf_fah': conf_fah, 'conf_recall': conf_recall,
                'int8_fah': int8_fah, 'int8_recall': int8_recall,
                'fah_ci': fah_ci, 'recall_ci': recall_ci
            })
    
    if confirmed_winners:
        # 6. Exhaustive local threshold sweep around each winner
        for winner in confirmed_winners:
            winner['optimized_threshold'] = exhaustive_local_sweep(
                winner['candidate'], confirmation_data, window=0.01, step=0.0001
            )
        
        # 7. Crown the best
        best = max(confirmed_winners, key=lambda w: w['conf_recall'] - 0.5 * w['conf_fah'])
        return best
    else:
        # No candidate passes confirmation — return best Pareto point with warning
        return pareto_archive.best_by_target(targets), "WARNING: No candidate passed confirmation"
```

---

## 12. Main Loop — Complete Pseudocode

```python
class MaxQualityAutoTuner:
    def tune(self, initial_checkpoint: str, config: dict) -> CandidateState:
        # ═══════════════════════════════════════════
        # PHASE 0: SETUP
        # ═══════════════════════════════════════════
        
        # Partition data
        cal_data, search_folds, confirm_data, repr_data = partition_data(
            val_dataset, cal_frac=0.15, confirm_frac=0.20, repr_frac=0.05,
            group_key='speaker_id', n_folds=5
        )
        
        # Initialize campaign
        campaign = CampaignState(
            active_pool=[], pareto_archive=ParetoArchive(max_size=24),
            strategy_posterior={arm: {'successes': 1, 'failures': 1} for arm in range(7)},
            error_memory=ErrorMemory(),
            annealing_temperature=0.5,  # Start moderate
            best_confirmed_candidate=None,
            total_gradient_steps=0
        )
        
        # Load initial model, evaluate, calibrate
        initial_model = load_model(initial_checkpoint)
        freeze_batchnorm(initial_model)  # CRITICAL: freeze BN for entire campaign
        
        initial_scores = initial_model.predict(search_folds[0].x)
        initial_temp, initial_ece = calibrate_temperature(initial_model, cal_data)
        initial_threshold = three_pass_threshold(initial_model, search_folds, initial_temp)
        initial_metrics = evaluate_at_threshold(initial_model, search_folds, initial_threshold, initial_temp)
        
        # Create initial candidate
        seed = CandidateState(
            weights=serialize_weights(initial_model),
            optimizer_state=None,  # Will be created on first burst
            batchnorm_state=capture_bn_state(initial_model),
            temperature=initial_temp,
            threshold_float32=initial_threshold,
            eval_results=initial_metrics,
            curriculum_stage=0,
            strategy_arm=-1,
            parent_id="initial",
            iteration=0,
            history=[]
        )
        
        campaign.active_pool.append(seed)
        campaign.pareto_archive.add(seed)
        
        log_initial_state(seed)
        
        # ═══════════════════════════════════════════
        # PHASE 1: SEARCH CAMPAIGN
        # ═══════════════════════════════════════════
        
        stagnation_counter = 0
        max_iterations = 50  # Generous — quality over speed
        
        for iteration in range(1, max_iterations + 1):
            # ─── Select parent candidate ───
            parent = select_parent(campaign.active_pool, campaign.pareto_archive)
            model = deserialize_model(parent.weights)
            freeze_batchnorm(model)
            
            if parent.optimizer_state:
                optimizer = deserialize_optimizer(parent.optimizer_state)
            else:
                optimizer = create_optimizer(lr=1e-5)  # Fresh optimizer for seed
            
            # ─── Select strategy arm ───
            regime = diagnose_regime(parent.eval_results, targets)
            arm_id = select_arm(campaign, parent)
            arm_config = get_arm_config(arm_id, regime, parent.curriculum_stage)
            
            # ─── Apply stir mechanisms if stagnant ───
            if stagnation_counter >= 3:
                arm_config = apply_stir(arm_config, stagnation_counter, parent)
            
            # ─── Build focused batch sampler ───
            sampler = FocusedSampler(
                error_memory=campaign.error_memory,
                curriculum_stage=parent.curriculum_stage,
                threshold=parent.threshold_float32,
                arm_config=arm_config
            )
            
            # ─── Execute training burst ───
            log_burst_start(iteration, arm_id, arm_config, parent)
            
            model, optimizer, burst_history = train_burst(
                model=model,
                optimizer=optimizer,
                sampler=sampler,
                steps=arm_config.steps,
                lr=arm_config.lr,
                class_weights=arm_config.class_weights,
                use_sam=arm_config.use_sam,
                collect_swa=arm_config.collect_swa,
                gradient_noise=arm_config.gradient_noise
            )
            
            campaign.total_gradient_steps += arm_config.steps
            
            # ─── If SWA was collected, evaluate all variants ───
            if arm_config.collect_swa and burst_history.swa_weights:
                variants = [
                    ("endpoint", model),
                    ("best_interim", burst_history.best_interim_model),
                    ("swa", apply_swa_weights(model, burst_history.swa_weights))
                ]
                model = select_best_variant(variants, search_folds, parent.temperature)
            
            # ─── Refresh BN statistics on representative data ───
            refresh_bn_stats(model, repr_data)
            
            # ─── Recalibrate temperature ───
            new_temp, new_ece = calibrate_temperature(model, cal_data)
            
            # ─── 3-pass threshold optimization ───
            new_threshold = three_pass_threshold(model, search_folds, new_temp)
            
            # ─── Full evaluation ───
            new_metrics = evaluate_at_threshold(model, search_folds, new_threshold, new_temp)
            
            # ─── Update error memory ───
            update_error_memory(campaign.error_memory, model, search_folds[0], new_threshold)
            
            # ─── INT8 shadow check (every 5 iterations or on archive entry) ───
            int8_metrics = None
            if iteration % 5 == 0:
                int8_fah, int8_recall, int8_deg = int8_shadow_eval(model, repr_data, search_folds[0])
                int8_metrics = (int8_fah, int8_recall, int8_deg)
            
            # ─── Build candidate ───
            new_candidate = CandidateState(
                weights=serialize_weights(model),
                optimizer_state=serialize_optimizer(optimizer),
                batchnorm_state=capture_bn_state(model),
                swa_buffers=burst_history.swa_weights if arm_config.collect_swa else None,
                temperature=new_temp,
                threshold_float32=new_threshold,
                eval_results=new_metrics,
                eval_results_int8=int8_metrics,
                curriculum_stage=advance_curriculum(parent.curriculum_stage, new_metrics, campaign.error_memory),
                strategy_arm=arm_id,
                parent_id=parent.id,
                iteration=iteration,
                history=parent.history + [burst_history.summary()]
            )
            
            # ─── Acceptance decision ───
            accepted, reason = accept_candidate(
                new_metrics, parent.eval_results,
                campaign.annealing_temperature, campaign.pareto_archive
            )
            
            log_burst_result(iteration, new_candidate, accepted, reason)
            
            if accepted:
                # Update pool and archive
                campaign.active_pool.append(new_candidate)
                if len(campaign.active_pool) > 12:
                    campaign.active_pool = prune_pool(campaign.active_pool, campaign.pareto_archive)
                
                archive_changed = campaign.pareto_archive.add(new_candidate)
                if archive_changed:
                    # INT8 shadow eval for new archive members
                    if int8_metrics is None:
                        int8_fah, int8_recall, _ = int8_shadow_eval(model, repr_data, search_folds[0])
                        new_candidate.eval_results_int8 = (int8_fah, int8_recall)
                
                stagnation_counter = 0
                campaign.strategy_posterior[arm_id]['successes'] += 1
            else:
                stagnation_counter += 1
                campaign.strategy_posterior[arm_id]['failures'] += 1
            
            update_annealing(campaign, accepted)
            
            # ─── Early stopping: targets met ───
            if targets_met(new_metrics, targets):
                log(f"✅ Targets met at iteration {iteration}! Proceeding to confirmation.")
                break
            
            # ─── Stagnation diagnostics ───
            if stagnation_counter >= 15:
                log("⚠️ Extended stagnation. Bottleneck may be data/labels/architecture.")
                # Still continue — user wants maximum quality
        
        # ═══════════════════════════════════════════
        # PHASE 2: CONFIRMATION
        # ═══════════════════════════════════════════
        
        log("═══ Entering Confirmation Phase ═══")
        
        winner = confirmation_phase(
            campaign.pareto_archive, confirm_data, repr_data, targets
        )
        
        if winner:
            log_winner(winner)
            save_final_checkpoint(winner)
            return winner
        else:
            log("⚠️ No candidate passed confirmation. Returning best Pareto point.")
            best = campaign.pareto_archive.best_by_target(targets)
            save_final_checkpoint(best)
            return best
```

---

## 13. Configuration Surface

### User-Facing Parameters (in YAML config)

```yaml
auto_tuning:
  # Targets (REQUIRED)
  target_fah: 0.2                    # Maximum acceptable false alarms per hour
  target_recall: 0.95                # Minimum acceptable recall
  
  # Search budget
  max_iterations: 50                 # Maximum search iterations (generous default)
  max_gradient_steps: 150000         # Total gradient step budget across all bursts
  
  # Validation rigor
  cv_folds: 5                        # Cross-validation folds for threshold search
  confirmation_fraction: 0.20        # Fraction of val data held out for confirmation
  bootstrap_samples: 1000            # Bootstrap iterations for confidence intervals
  
  # INT8 verification
  int8_shadow: true                  # Enable INT8 shadow evaluation
  int8_shadow_interval: 5            # Run INT8 check every N iterations
  require_int8_pass: true            # Winner must also pass INT8 verification
  require_confirmation: true         # Winner must pass confirmation phase
  
  # Data grouping
  group_key: "speaker_id"            # Group validation data by this key for CV
```

### Expert Parameters (in YAML, with good defaults)

```yaml
auto_tuning_expert:
  # Burst configuration
  min_burst_steps: 200               # Minimum gradient steps per burst
  max_burst_steps: 5000              # Maximum gradient steps per burst
  default_burst_steps: 500           # Default if arm doesn't specify
  
  # Learning rate
  min_lr: 1e-7
  max_lr: 1e-4
  default_lr: 1e-5
  
  # SAM
  sam_rho: 0.05                      # SAM perturbation radius
  
  # SWA
  swa_collection_interval: 100       # Collect SWA snapshot every N steps
  
  # Annealing
  initial_temperature: 0.5
  cooling_rate: 0.95
  reheat_after: 5                    # Reheat after N consecutive rejections
  reheat_factor: 1.3
  
  # Pool management
  active_pool_size: 12
  pareto_archive_size: 24
  
  # Stagnation thresholds
  stir_level_1: 3                    # SWA collection
  stir_level_2: 5                    # SAM flatten
  stir_level_3: 7                    # Loss landscape probing
  stir_level_4: 9                    # Cyclical sweep
  stir_level_5: 12                   # Diversify + macro refine
  
  # Curriculum
  curriculum_advance_threshold: 0.3  # Fraction of stage errors that must be fixed
```

### Hardcoded (Not Configurable)

- Freeze BatchNorm during all micro-bursts (safety-critical)
- Refresh BN stats on representative data before evaluation (correctness)
- Monotonic temperature scaling constraint (calibration stability)
- Use `ExportArchive` for INT8 export (ESPHome compatibility)
- uint8 threshold storage format (deployment constraint)
- Calibration set fraction = 15% (statistical requirement)
- Representative data fraction = 5% (INT8 quantization requirement)

---

## 14. Edge Cases & Failure Modes

### If the Pareto frontier can't reach the target region

**Diagnosis**: After a full sweep of all arms, no candidate achieves both FAH < target AND recall > target.

**Root cause options**:
1. **Insufficient data** — Need more diverse positive samples or hard negatives
2. **Label noise** — Some positives are mislabeled or some negatives are actually the wake word
3. **Architecture limitation** — MixedNet at 37K params may not have enough capacity
4. **Target infeasibility** — The requested FAH/recall combination may be unreachable with this data/model

**What the tuner does**: Logs a clear diagnostic with the best achievable point and the gap to targets. Does NOT pretend to succeed.

### If float improves but INT8 breaks

**Diagnosis**: Float32 metrics meet targets but INT8 metrics are significantly degraded.

**Root cause**: The model learned features that are sensitive to quantization (sharp weight distributions, narrow activation ranges).

**Solution**: Switch to quantization-aware fine-tuning (QAT) phase:
1. Insert fake-quantization nodes
2. Continue micro-step tuning with quantization noise baked into training
3. Re-evaluate on INT8 after QAT

### If stagnation persists after all stir levels

Log: "Bottleneck is likely data/labels/architecture, not tuning parameters."
Return the best Pareto point with full diagnostic report.

### If calibration worsens ECE

Keep temperature = 1.0 (identity). Log warning. This typically means the model's scores are already well-calibrated or the calibration set is too small.

---

## 15. Key Differences From Current AutoTuner

| Aspect | Current (Broken) | MaxQualityAutoTuner |
|--------|-------------------|---------------------|
| Steps per iteration | 12,000 (full retraining) | 200-5,000 (surgical burst) |
| Time per iteration | ~6 minutes | 10 sec – 3 min |
| Optimizer state | Destroyed each iteration | Preserved between bursts |
| BatchNorm | Re-computed (corrupted) | Frozen (preserved from training) |
| Threshold optimization | None (stuck at 0.0) | 3-pass robust sweep every iteration |
| Calibration | None | Temperature scaling on held-out cal set |
| INT8 verification | None | Shadow eval every 5 iterations |
| Confirmation | None | Held-out confirmation set |
| Strategy selection | 2 knobs (neg weight, hard-neg weight) | 7 strategy arms with Thompson sampling |
| Stagnation handling | ESCALATE→REVERSE→SWITCH (useless) | 5-level escalation: SWA→SAM→landscape probe→cyclic→diversify |
| Candidate management | Single best checkpoint | Pool of 12 active + archive of 24 Pareto-optimal |
| Acceptance criteria | Pareto-only | Simulated annealing (accepts some worse moves) |
| CV validation | None | 5-fold grouped cross-validation |
| Confidence intervals | None | Bootstrap CIs on FAH and recall |
| Data partitioning | Uses full val set for everything | Cal 15% / Search 60% / Confirm 20% / Repr 5% |

---

## 16. Implementation Roadmap

### Task Breakdown

| # | Task | Effort | Dependencies |
|---|------|--------|-------------|
| 1 | Data partitioning (speaker-grouped splits) | Small | None |
| 2 | CandidateState + CampaignState dataclasses | Small | None |
| 3 | ParetoArchive (4-objective, diversity filter) | Medium | #2 |
| 4 | Temperature scaling (calibrate_temperature) | Small | #1 |
| 5 | 3-pass threshold optimizer | Medium | #1, #4 |
| 6 | ErrorMemory + FocusedSampler | Medium | #1 |
| 7 | train_burst() — core micro-step training | Large | #6 |
| 8 | SAM optimizer wrapper | Medium | #7 |
| 9 | SWA collection + variant evaluation | Medium | #7 |
| 10 | Strategy arms (7 arms + configs) | Medium | #7 |
| 11 | Thompson sampling arm selector + regime diagnosis | Small | #10 |
| 12 | Stir mechanisms (5 levels) | Medium | #8, #9, #10 |
| 13 | Acceptance + simulated annealing | Small | #3 |
| 14 | INT8 shadow evaluation | Medium | #5 |
| 15 | Confirmation phase | Medium | #3, #5, #14 |
| 16 | Main loop integration | Large | All above |
| 17 | Logging + Rich tables | Small | #16 |
| 18 | CLI integration (mww-autotune) | Small | #16 |
| 19 | Unit tests | Medium | #16 |
| 20 | Integration test (end-to-end) | Medium | #19 |

### Estimated Total Effort: 3-5 days

### Parallel Implementation Opportunities

- Tasks 1-6 are independent of each other
- Tasks 7-9 can be developed together
- Tasks 10-13 can be developed together
- Task 14 is independent
- Tasks 15-18 depend on the main loop

---

## Appendix: Why The Current Threshold Is 0.0

The current auto-tuner's `_evaluate()` method uses `MetricsCalculator` which scans multiple thresholds and picks `recall_at_target_fah`. Since FAH is always >> target (169.32 >> 0.2), no threshold achieves the target FAH. The method then returns `threshold=0.0` as a fallback, which means "predict everything positive." This is the root cause of the constant FAH=169.32, Recall=1.0 results.

The fix is obvious: the threshold optimization must be **decoupled from the evaluation** and must find the **best achievable operating point**, not the one that meets impossible targets.
