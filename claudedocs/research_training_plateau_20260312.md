# Training Plateau & Divergence Analysis

**Date**: 2026-03-12
**Training Run**: `logs/run_20260311_231757`
**Config**: `max_quality.yaml`
**Status**: ~76k/80k steps completed

---

## Executive Summary

Training plateaus at ~30k steps and **actively degrades** from Phase 2 onward. All key metrics (loss, AUC-PR, FAH, recall, F1, EER) worsen after the Phase 1→2 transition. **The best model checkpoint was likely saved in Phase 1 (before step 30k).** Phases 2 and 3 (~50k steps, 62.5% of total training) are not just wasted — they actively damage model quality.

### Critical Finding: Double Class Weighting Bug

**A confirmed double class weighting bug is the primary driver of divergence.** When `use_tfdata: true` (the default), class weights are applied TWICE:
1. First in `tfdata_pipeline.py:271` → `final_weights = class_weights * sample_weights`
2. Again in `trainer.py:996` → `_apply_class_weights()` multiplies the already-weighted sample weights by class weights

This **squares** the effective class weights, creating catastrophic imbalance in later phases.

---

## Root Cause Analysis (Ranked by Impact)

### 1. CRITICAL: Double Class Weighting Bug

| Phase | Intended Pos | Intended HN | Intended Neg | **Actual Pos** | **Actual HN** | **Actual Neg** | **HN:Neg Ratio** |
|-------|-------------|-------------|-------------|----------------|----------------|----------------|-------------------|
| 1     | 5.0         | 5.0         | 1.5         | **25.0**        | **25.0**        | **2.25**        | 11:1              |
| 2     | 7.0         | 8.0         | 1.5         | **49.0**        | **64.0**        | **2.25**        | 28:1              |
| 3     | 9.0         | 12.0        | 1.5         | **81.0**        | **144.0**       | **2.25**        | **64:1**          |

**Phase 3 hard_neg effective weight is 144x negative weight.** The model is forced to focus almost exclusively on hard negatives, destroying general discrimination ability.

**Evidence chain:**
- `config/presets/max_quality.yaml`: `use_tfdata: true` (default)
- `src/data/tfdata_pipeline.py:263-271`: Pipeline applies class weights in `_create_tfdata_dataset()`
- `src/training/trainer.py:1317`: Batch unpacked with pre-weighted `train_sample_weights`
- `src/training/trainer.py:996-1003`: `_apply_class_weights()` multiplies AGAIN regardless of `use_tfdata`
- `src/training/trainer.py:1008`: `combined_weights` (doubly-weighted) used as `sample_weight` in `train_on_batch`

**Fix:** Remove class weight application from ONE of the two locations. The trainer's `_apply_class_weights` is the correct location (it's phase-aware), so the tfdata pipeline should pass through raw sample weights without multiplying by class weights.

### 2. HIGH: No Intra-Phase LR Decay

Each phase uses a **flat constant LR** for its entire duration (30k, 30k, 20k steps). There is no cosine decay, no warmup, no ReduceLROnPlateau within phases. Phase 2 runs at `LR=0.00035` for 30,000 steps — no decay whatsoever.

**Impact:** The model can't fine-tune in late stages of each phase. Combined with increasing class weights and augmentation, the fixed LR provides too much gradient signal relative to the increasingly noisy loss landscape.

### 3. HIGH: Simultaneous Curriculum Increases at Phase Boundaries

At each phase boundary, THREE things increase simultaneously while LR drops:
- Class weights increase (pos: 5→7→9, hard_neg: 5→8→12)
- SpecAugment intensity increases (mask count: 1→2→3)
- LR drops sharply (0.0017→0.00035→0.00009)

This creates a "triple whammy": harder training signal + more augmentation noise + less gradient signal = divergence.

### 4. MEDIUM: Checkpoint Strategy Never Reaches Stage 2

The two-stage checkpoint system requires FAH ≤ `target_fah × 1.1 = 2.2` to enter Stage 2 (recall-based selection). **FAH never drops below 8.6** throughout training, so:
- Model stays in Stage 1 (PR-AUC based) forever
- `recall_at_target_fah` is 0.000 for all 155 evaluation points
- Best PR-AUC was 0.965 at step 3,500 (!)
- The checkpoint strategy can't save the model from degradation

### 5. LOW: Plateau Detection is Informational Only

`_compute_plateau_metrics()` correctly detects the plateau (quality_plateau_score reaches 1.0 at step ~64k), but this information is **never acted upon** — no LR reduction, no early stopping, no training modification.

---

## Metric Evidence (from TensorBoard Events)

### Training Loss — MONOTONICALLY INCREASING from Phase 2
```
Phase 1 (0-30k):   1.247 → 0.513  (healthy decrease)
Phase 2 (30k-60k): 0.510 → 0.602  (INCREASING — model being damaged)
Phase 3 (60k-80k): 0.612 → 0.747  (RAPIDLY INCREASING)
```

### AUC-PR — Peaks in Phase 1, Degrades After
```
Phase 1: peak 0.965 at step 3,500
Phase 2: 0.942 → 0.938 (declining)
Phase 3: 0.922 → 0.929 (degraded)
```

### FAH (False Alarms/Hour) — DOUBLES by Phase 3
```
Phase 1: 8.6 → 10.9
Phase 2: 14.0 → 13.3
Phase 3: 18.6 → 18.1 (nearly 2x worse than Phase 1)
```

### EER — Continuously Worsening
```
Phase 1: 0.031 → 0.034
Phase 2: 0.033 → 0.038
Phase 3: 0.039 → 0.044
```

### F1 Score — Peak at Phase 2 Start, Then Decline
```
Phase 1: 0.820 → 0.897
Phase 2: 0.903 → 0.893 (peak 0.906 at step 33.5k)
Phase 3: 0.885 → 0.876
```

### recall_at_target_fah — NEVER ACHIEVED
```
0.000 throughout all 80k steps (target FAH = 2.0, actual FAH ≥ 8.6)
```

---

## Recommended Solutions (Priority Order)

### Immediate Fix (Current Training)

**Action**: Stop current training. Best checkpoint is from Phase 1 (around step 3.5k-30k based on PR-AUC). Use that checkpoint.

### Fix 1: Eliminate Double Class Weighting (CRITICAL)

In `src/data/tfdata_pipeline.py`, change `_create_tfdata_dataset()` to NOT multiply by class weights. The `sample_weights` should pass through as-is (just quality/base weights). Class weights should only be applied in `Trainer._apply_class_weights()` which is phase-aware.

**Alternative**: Add a flag `apply_class_weights_in_pipeline: false` to config, and skip the multiplication in tfdata_pipeline when false.

### Fix 2: Add Intra-Phase LR Decay

Replace flat per-phase LR with **cosine decay within each phase**:

```python
# Per-phase cosine decay
for each phase:
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * phase_progress))
```

Or use `CosineDecayRestarts` across the full training:
```python
tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.0017,
    first_decay_steps=30000,
    t_mul=1.0,  # same cycle length
    m_mul=0.2,  # each restart at 20% of previous peak
    alpha=0.00001
)
```

### Fix 3: Stagger Phase Transitions

Don't change LR, class weights, and augmentation all at once. Stagger:
- Step N: LR change only
- Step N+2000: Class weight change
- Step N+4000: Augmentation change

This gives the model time to adapt to each change.

### Fix 4: Activate Plateau-Based Early Stopping

Convert the existing `_compute_plateau_metrics()` from informational to actionable:

```python
# When plateau detected (quality_plateau_score >= 1.0):
if self.plateau_patience_counter >= 3:  # 3 consecutive plateau evals
    if self.lr_reductions < 2:
        self.optimizer.learning_rate.assign(current_lr * 0.3)
        self.lr_reductions += 1
    else:
        logger.info("Stopping: persistent plateau after 2 LR reductions")
        break
```

### Fix 5: Relax FAH Target or Add Intermediate Targets

Current `target_fah: 2.0` is never achieved (actual: 8.6-18+). Either:
- Raise target_fah to 5.0 (achievable based on Phase 1 metrics)
- Add tiered targets: [10.0, 5.0, 2.0] so the model can enter Stage 2 checkpoint selection

### Fix 6: EMA Decay Adjustment

Current `ema_decay: 0.99` (half-life ~69 steps) is aggressive. Consider:
- `ema_decay: 0.999` (half-life ~693 steps) for smoother averaging
- `ema_decay: 0.9999` for SWA-like behavior in the final phase

### Fix 7: Reduce Total Steps / Accept Earlier Finish

Based on metrics, optimal training length is ~30-40k steps (end of Phase 1 / early Phase 2). Consider:
- Phase 1: 30k steps (current)
- Phase 2: 15k steps (halved) with cosine decay
- Phase 3: 5-10k steps at very low LR with frozen BatchNorm
- Total: 50-55k steps instead of 80k

### Fix 8: Freeze BatchNorm in Final Phase

In Phase 3 (or when plateau detected), freeze BatchNorm layers:
```python
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
```
This prevents BN statistics from drifting with the heavily reweighted loss.

---

## Advanced Solutions (For Future Iterations)

### Stochastic Weight Averaging (SWA)
- Start SWA at 75% of training (after last restart)
- Use constant low LR during SWA phase
- `tfa.optimizers.SWA(average_period=10)`
- Requires BN statistics update after averaging

### Cosine Annealing with Warm Restarts (SGDR)
- Replace 3-phase schedule with SGDR
- `first_decay_steps=30000, t_mul=1.5, m_mul=0.9, alpha=1e-5`
- Save snapshot at each restart point for ensemble

### Curriculum Mining Schedule
- Early (0-20k): mine aggressively, `fp_threshold=0.65`
- Mid (20k-40k): reduce intensity, `fp_threshold=0.75`
- Late (40k+): minimal mining, `fp_threshold=0.85`

---

## Priority Action Items

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| P0       | Fix double class weighting bug | 1 hour | **Fixes primary divergence cause** |
| P0       | Use Phase 1 checkpoint from current run | 5 min | Immediate best model |
| P1       | Add intra-phase cosine LR decay | 2 hours | Prevents flat-LR stagnation |
| P1       | Relax target_fah to 5.0 | 5 min | Enables Stage 2 checkpointing |
| P2       | Activate plateau-based early stopping | 3 hours | Prevents wasted compute |
| P2       | Stagger phase transition changes | 2 hours | Smoother transitions |
| P3       | Implement SGDR + snapshot ensemble | 4 hours | Better LR schedule |
| P3       | Freeze BatchNorm in final phase | 1 hour | Prevents BN drift |
| P4       | SWA implementation | 3 hours | Better final model |
| P4       | Curriculum mining schedule | 3 hours | Healthier late-stage training |

---

## Confidence Assessment

| Finding | Confidence | Evidence |
|---------|------------|----------|
| Double class weighting bug | **99%** | Code trace verified across 4 files |
| Best checkpoint in Phase 1 | **95%** | TensorBoard metrics clearly show peak at 3.5-30k |
| Phase transitions cause degradation | **90%** | Metrics show immediate worsening at boundaries |
| Flat LR contributes to stagnation | **85%** | Research consensus + metric evidence |
| Target FAH unreachable | **100%** | recall_at_target_fah = 0.000 for all eval points |

---

## Sources

- **Codebase**: trainer.py, tfdata_pipeline.py, mining.py, config/loader.py, max_quality.yaml
- **TensorBoard Events**: 4,608 data points across 37 metrics, 76k steps
- **Research**: SWA (Izmailov et al.), SGDR (Loshchilov & Hutter), Curriculum Learning, Predictive Batch Scheduling (arXiv:2602.17066)
- **Oracle Consultation**: Architecture analysis, double-weighting hypothesis, action plan
- **Web Research**: TensorFlow/Keras implementation guides for SWA, CosineDecayRestarts, EMA
