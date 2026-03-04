# 🔬 8-Perspective Analysis: Wake Word Training Pipeline

## 🚨 Critical Finding (Cross-Perspective Consensus)

**The 65× FAH gap (training: 0.59 → test: 38.26) is the #1 issue.** Multiple perspectives converge on the same root causes:
- All augmentation is DISABLED → model overfits to clean training conditions
- Hard negative mining is log_only → known failure modes aren't being corrected
- FAH counting may be inconsistent between val and test (FP=17 at t=0.5 vs 110 FP in test report)
- Threshold optimized on validation may not transfer

---

## 1. Performance Profiler

**Current State**: cProfile on `train_step` only (every 100th step), no GPU profiling, no memory tracking. You're seeing Python overhead but missing the full picture — data loading, validation, GPU kernel time, and memory are all invisible.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **Add TF Profiler traces (host + GPU + tf.data)** — `tf.profiler.experimental.start/stop` over 200-500 consecutive steps → captures GPU kernel time, input stalls, op-level breakdown in TensorBoard | Turns "mystery time" into attributable hotspots | **HIGH** |
| 2 | **Profile end-to-end step timing with GPU sync points** — Measure `next(dataset_iter)`, augmentation, `train_step`, metrics/logging separately with explicit GPU sync | Reveals whether you're input-bound, Python-bound, or GPU-bound | **HIGH** |
| 3 | **Add memory + utilization telemetry** — Periodically record `tf.config.experimental.get_memory_info("GPU:0")` and correlate with step-time spikes | Explains the 14ms jitter spikes, catches fragmentation | **MEDIUM** |

---

## 2. Performance Engineer

**Current State**: ~18ms/step with repeated CPU↔GPU synchronization costs (`np.where` class weights, per-step metric dict extraction, `.numpy()` in validation). The pipeline structure is sound but leaks performance through sync points.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **Eliminate numpy class-weighting** — Replace `np.where` with `tf.where`/`tf.gather` or fold into loss function to stay on-device | ~5–15% faster steps + eliminates 14ms jitter spikes | **HIGH** |
| 2 | **Stop per-step tensor→numpy metric extraction** — Use `@tf.function` training step returning only loss; materialize full metrics every N steps | Saves ~1-2ms/step (~8-12% throughput) | **HIGH** |
| 3 | **Keep validation on-device** — Use TF-native metrics (e.g. `tf.keras.metrics.AUC(num_thresholds=101)`) instead of `.numpy()` + numpy threshold sweeps | Reduces validation wall-clock and sync stalls | **MEDIUM** |

---

## 3. Code Reviewer

**Current State**: Hot loop mixes concerns — training, weighting, metric materialization, validation conversion, per-phase control all interleaved. Several patterns are both slow and brittle.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **Make class-weighting a first-class training input** — Dataset produces `sample_weight` or loss computes it; removes numpy patch from hot loop | Big perf win + clearer semantics/testability | **HIGH** |
| 2 | **Freeze phase settings into immutable config** — Precompute phase parameters once, only update on phase change (not every step) | Small speedup, large maintainability gain | **MEDIUM** |
| 3 | **Define strict "materialization policy"** — Rule: training step stays TF-only; logging converts scalars only every N steps; eval modules own numpy conversion | Prevents sync regressions, simplifies profiling | **MEDIUM** |

---

## 4. Machine Learning Engineer

**Current State**: MixedNet architecture is near-optimal for ESP32 constraints. The biggest gains are NOT in architecture — they're in robustness (augmentation), calibration, and hard-negative handling.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **🔴 Enable SpecAugment + background noise + RIR** — All set to 0 currently; model overfits to clean features. This is likely the single largest contributor to the 65× FAH gap | Typically the largest FAH reduction at fixed recall | **HIGH** |
| 2 | **🔴 Activate hard negative mining (not log_only)** — Promote mined FPs into hard_negative set and retrain Phase2; cap per-speaker dominance to avoid overfitting to small clusters | Strong FAH improvement targeting exact failure modes | **HIGH** |
| 3 | **Optimize for operating point, not t=0.5** — Pick target FAH on validation and tune threshold in uint8 space; the t=0.5 confusion matrix doesn't align with test FAH, indicating threshold mismatch | "Free" reduction in deployed false triggers | **HIGH** |

---

## 5. AI Engineer

**Current State**: Pipeline covers data → train → export → verify → deploy, but the biggest risk is silent divergence between training-time and on-device streaming inference. Export is fragile under streaming + quantization.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **Add streaming equivalence gate** — Assert near-identical outputs between non-streaming eval and chunked ring-buffer streaming eval (including state reset, boundaries, 2-subgraph behavior) | Prevents "good in eval, fails on device" | **HIGH** |
| 2 | **Make export + verification a single versioned artifact step** — One command emits: TFLite + manifest + quant report + verification (dtype checks, subgraph count) | Higher iteration speed, fewer silent deployment failures | **HIGH** |
| 3 | **Integrate AutoTuner as default post-training step** — Tune deployed objective (FAH/recall at threshold), not just "best checkpoint" | Consistent wins in deployed FAH without arch changes | **MEDIUM** |

---

## 6. ML Engineer (Data Pipeline)

**Current State**: Pipeline has the right primitives (RaggedMmap, tf.data, speaker clustering, mining hooks) but most "domain randomization" knobs are at zero — generalization is bottlenecked by data variability, not throughput.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **🔴 Re-enable augmentation with staged schedule** — Start with mild noise/RIR, ramp SpecAugment after initial convergence; avoids early destabilization while building robustness | Major FAH reduction across rooms/devices | **HIGH** |
| 2 | **Close the data-quality loop with mined FPs** — Dedupe + label audit + curate "hard negative pack" that stays every epoch (not random sampled) | Outsized FAH improvement targeting exact breakages | **HIGH** |
| 3 | **Keep tf.data fast when enabling SpecAugment** — Isolate CuPy py_function to avoid serializing pipeline (parallel map, controlled determinism, profile GPU stalls) | Preserves ~30 min runtime while adding augmentation | **MEDIUM** |

---

## 7. Model Evaluator

**Current State**: Metric toolbox is strong (ROC/PR, FAH, calibration, TestEvaluator, AutoTuner), but the 65× FAH blow-up is a red flag. The "110 FP" vs "FP=17 at t=0.5" discrepancy suggests inconsistent counting/aggregation.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **🔴 Reconcile "what is an FP" end-to-end** — Make FAH derivation auditable: log FP_count, negative_hours, compute FAH consistently (same event definition, windowing, aggregation for val and test). Explain why confusion shows FP=17 but test shows 110 FP. | Very High — may turn "model problem" into data/metric issue | **HIGH** |
| 2 | **Introduce strict 3-way split** — Train / calibration-tuning / final test. AutoTuner uses calibration set; test is untouched until final report. | More trustworthy FAH/recall, fewer surprise regressions | **HIGH** |
| 3 | **Evaluate the deployed path** — Compare float model vs INT8 TFLite vs streaming-equivalent scoring; add sliced metrics (noise type, SNR, speaker, mic distance, hard negatives) | Finds real failure mode fast | **MEDIUM** |

---

## 8. MLOps Engineer

**Current State**: Single-developer, best-effort maturity. Config system exists but no seed control, no CI/CD, manual export. Regressions and irreproducible wins are likely.

| # | Improvement | Impact | Priority |
|---|------------|--------|----------|
| 1 | **Make every run reproducible-by-default** — Set/log seeds (Python/NumPy/TF), persist full run record (resolved config, git hash, package versions, GPU info, dataset manifests) | Cuts iteration time; prevents phantom improvements | **HIGH** |
| 2 | **Artifact-driven pipeline with gates** — One command: train → evaluate (val+tuning+test) → export INT8 → verify_esphome → emit versioned bundle. Only "promote" if targets met. | Fewer bad deployments, faster iteration | **HIGH** |
| 3 | **Add lightweight smoke checks** — Local `scripts/ci.sh` that validates config loading, runs fast_test train-eval, confirms export+verify works | Catches breakages early without heavy infra | **MEDIUM** |

---

## 📊 Priority Matrix (Cross-Perspective)

### 🔴 DO FIRST (Highest Impact, All Perspectives Agree)

| Action | Perspectives | Why |
|--------|-------------|-----|
| **Enable augmentation (SpecAugment + noise + RIR)** | ML Engineer, Data Pipeline, ML Eng | #1 cause of overfitting → 65× FAH gap |
| **Activate hard negative mining** | ML Engineer, Data Pipeline | Target exact failure modes |
| **Reconcile FAH metric counting** | Model Evaluator | Can't improve what you can't measure consistently |
| **Eliminate numpy from training hot loop** | Perf Engineer, Code Reviewer | 5-15% speedup + jitter elimination |

### 🟡 DO NEXT (High Impact, Lower Effort)

| Action | Perspectives |
|--------|-------------|
| Optimize operating threshold (not t=0.5) | ML Engineer, Model Evaluator |
| Add TF Profiler traces | Performance Profiler |
| Streaming equivalence gate | AI Engineer |
| Reproducible runs (seeds + metadata) | MLOps |
| 3-way data split | Model Evaluator |

### 🟢 DO LATER (Medium Impact, Good Practice)

| Action | Perspectives |
|--------|-------------|
| Artifact-driven export pipeline | AI Engineer, MLOps |
| Memory telemetry | Performance Profiler |
| Materialization policy for tensors | Code Reviewer |
| Smoke test CI script | MLOps |
| AutoTuner as default post-training step | AI Engineer |

---

## Implementation Status

- [x] 1.1 — Add TF Profiler traces (host + GPU + tf.data)
- [x] 1.2 — Profile end-to-end step timing with GPU sync points
- [x] 1.3 — Add memory + utilization telemetry
- [x] 2.1 — Eliminate numpy class-weighting (tf.where/tf.gather)
- [x] 2.2 — Stop per-step tensor→numpy metric extraction
- [x] 2.3 — Keep validation on-device
- [x] 3.1 — Make class-weighting a first-class training input
- [x] 3.2 — Freeze phase settings into immutable config
- [x] 3.3 — Define strict materialization policy
- [x] 4.1 — Enable SpecAugment + background noise + RIR
- [x] 4.2 — Activate hard negative mining
- [x] 4.3 — Optimize for operating point, not t=0.5
- [x] 5.1 — Add streaming equivalence gate
- [x] 5.2 — Make export + verification a single versioned artifact step
- [x] 5.3 — Integrate AutoTuner as default post-training step
- [x] 6.1 — Re-enable augmentation with staged schedule
- [x] 6.2 — Close the data-quality loop with mined FPs
- [x] 6.3 — Keep tf.data fast when enabling SpecAugment
- [x] 7.1 — Reconcile FAH metric counting end-to-end
- [x] 7.2 — Introduce strict 3-way split
- [x] 7.3 — Evaluate the deployed path
- [x] 8.1 — Make every run reproducible-by-default
- [x] 8.2 — Artifact-driven pipeline with gates
- [x] 8.3 — Add lightweight smoke checks
