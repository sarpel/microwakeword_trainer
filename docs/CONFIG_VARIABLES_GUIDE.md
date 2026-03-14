# Configuration Variables Guide (ELI5)

This guide explains all configurable variables in `max_quality.yaml` in simple terms. No ML knowledge required.

---

## Training Optimizer Parameters

### `gradient_clipnorm: 1.0`

**Real-life impact:** Imagine you're walking toward a destination (the best model). Gradient clipping prevents you from taking huge, reckless jumps that might overshoot and get lost. It's like speed limits on a highway — you'll still get there, but safely.

**On-paper impact:** Clips the L2 norm of gradients to 1.0. Prevents exploding gradients that destabilize training. A value of 0.5 was found too aggressive — it capped maximum model score at 0.87. Higher values (1.0–2.0) allow more freedom but risk instability.

---

### `ema_decay: 0.999`

**Real-life impact:** Exponential Moving Average (EMA) is like keeping a running average of your model's "thoughts" over time. Instead of reacting to every momentary mood swing, you smooth things out to find the stable, long-term personality. The decay controls how quickly you forget old memories — 0.999 means you remember ~6,900 steps back.

**On-paper impact:** EMA weights are computed as `ema_weights = decay * old_ema + (1 - decay) * current_weights`. With decay=0.999, the half-life is ~6,900 steps (how many steps for old information's influence to drop by 50%). Previous value 0.995 was too "jittery" — forgot too fast, leading to unstable checkpoints. Model evaluation and checkpointing use EMA weights; training resumes from raw weights after eval.

---

### `label_smoothing: 0.0`

**Real-life impact:** When the model is 100% sure "this IS a wake word," it might be overconfident and wrong. Label smoothing says "you're probably right, but leave a tiny 1% chance you're wrong" — like telling someone "don't get cocky, you could be mistaken."

**On-paper impact:** Softens training labels from hard 0/1 to (0.0 → smoothing/1.0 → 1.0 - smoothing). With `label_smoothing: 0.0`, it's disabled. Values like 0.1 or 0.2 improve calibration but may slightly reduce peak accuracy.

---

### `cosine_decay_alpha: 0.3`

**Real-life impact:** Learning rate (LR) is like your step size when walking toward the solution. Cosine decay gradually shrinks your steps as you get closer, but doesn't shrink them to zero — you keep taking baby steps at the end. Alpha=0.3 means your final step size is 30% of your starting step size for that phase.

**On-paper impact:** Intra-phase LR decay follows: `effective_lr = base_lr * (alpha + (1 - alpha) * 0.5 * (1 + cos(π * progress)))`. As `progress` goes from 0→1, LR decays from `base_lr` down to `base_lr * 0.3`. Prevents flat LR stagnation at phase end.

---

### `plateau_lr_factor: 0.3`

**Real-life impact:** When the model stops improving (hits a plateau), you lower the learning rate to "take smaller steps and explore more carefully." Factor=0.3 means drop to 30% of your current step size — like switching from walking to tiptoeing.

**On-paper impact:** When plateau is detected, `current_lr *= plateau_lr_factor`. With 0.3, LR reduces to 30% of its current value. Smaller factors (0.1–0.2) are more aggressive but risk getting stuck; larger factors (0.5–0.7) are gentler but may slow convergence.

---

### `plateau_patience: 3`

**Real-life impact:** Patience is how long you wait before concluding "I'm really stuck, not just having a bad moment." Patience=3 means "I'll watch for 3 consecutive evaluations with no improvement before I change strategy."

**On-paper impact:** The model tracks consecutive evaluations with no improvement (`consecutive_plateau_evals`). Only after this count exceeds `plateau_patience` does it trigger LR reduction or other plateau responses. Higher values (5–10) tolerate longer stagnation but waste training time.

---

### `plateau_max_reductions: 0`

**Real-life impact:** Limits how many times you can say "stuck, lower learning rate" before giving up and stopping training. Set to 0 means "never stop early for this reason" — let the full training steps run their course.

**On-paper impact:** After each plateau-triggered LR reduction, `reduction_count` increments. When `reduction_count >= plateau_max_reductions` (if > 0), training stops early. Set to 0 to disable (default for max quality — trust the phase schedule).

---

### `phase_stagger_steps: 2000`

**Real-life impact:** When training phases change (e.g., Phase 1 → Phase 2), you don't change EVERYTHING at once. Staggering means "change the learning rate now, but wait 2000 steps before changing class weights and augmentation." Like changing your pace first, then adjusting your gear later.

**On-paper impact:** Class weights and augmentation phase index lag behind LR phase by `phase_stagger_steps`. Implementation: `if steps_into_phase < stagger: weight_aug_phase = lr_phase - 1`. Prevents abrupt simultaneous changes that destabilize training.

---

### `freeze_bn_on_plateau: true`

**Real-life impact:** BatchNorm layers normalize data based on statistics from the current batch. Freezing them says "you've seen enough — stop updating your running averages and stick with what you know." Stabilizes training when progress stalls.

**On-paper impact:** When plateau is detected, all `BatchNormalization` layers have `trainable=False` set. This freezes their `moving_mean` and `moving_variance` statistics. Unfrozen BN can cause jitter in later training stages; freezing locks in stable normalization.

---

## Model Regularization

### `dropout_rate: 0.12`

**Real-life impact:** Dropout is like randomly telling 12% of your brain's neurons "take a break, don't think" during training. This forces the rest of your brain to learn redundantly — so even if some neurons fail later, you still know the answer. Prevents over-reliance on any single neuron.

**On-paper impact:** Applies `tf.keras.layers.Dropout(0.12)` in the classification head. Each training step, 12% of units are randomly zeroed out. Higher rates (0.2–0.5) prevent overfitting more strongly but underfit if too high. Lower rates (0.05–0.1) are gentler regularization.

---

### `l2_regularization: 0.0001`

**Real-life impact:** L2 regularization is like adding a "complexity tax" — every time a weight gets large, you pay a small penalty. This forces the model to prefer small, simple weights over large, complex ones. Think of it as "don't overthink it."

**On-paper impact:** Adds `lambda * weight^2` term to loss (lambda = 0.0001). During gradient updates, this shrinks weights slightly each step. Prevents weights from growing arbitrarily large, which improves generalization. Higher values (0.001–0.01) are more restrictive; lower values (0.00001) have minimal effect.

---

## Evaluation

### `n_thresholds: 501`

**Real-life impact:** The model outputs a probability (0–1). To decide "wake word or not," you pick a threshold. Testing 501 thresholds from 0.0→1.0 gives you a curve showing trade-offs: "low threshold = sensitive (few misses, many false alarms)" vs "high threshold = strict (many misses, few false alarms)."

**On-paper impact:** Generates `thresholds = np.linspace(0, 1, n_thresholds)` — 501 evenly-spaced values. Used to compute ROC curves, PR curves, and FAH-vs-recall curves. 501 provides finer resolution than the default 101 (step size 0.002 vs 0.01).

---

### `n_latency_runs: 100`

**Real-life impact:** Measures how long the model takes to think (inference latency). Running 100 times and taking statistics (average, std dev, percentiles) gives a reliable measure — not just a single lucky or unlucky run. Warmup runs are excluded to avoid "cold start" bias.

**On-paper impact:** Executes 100 inference passes after warmup, recording latency for each. Computes mean, std, p50, p90, p95, p99 latencies. FAH (False Alarms Per Hour) is calculated as `fp / ambient_duration_hours` — normalizes false positives to hourly rate.

---

## Auto-Tuning

### `max_iterations: 30`

**Real-life impact:** The auto-tuner tries different strategies (7 "arms" like boundary polishing, FA suppression) iteratively. `max_iterations=30` means "try at most 30 strategy bursts before stopping, even if not fully converged."

**On-paper impact:** Outer loop cap in autotuner: `for iteration in range(1, self.max_iterations + 1)`. Each iteration = one strategy arm burst + evaluation. Can also stop early via gradient budget (`max_gradient_steps`) or patience.

---

### `max_gradient_steps: 10000`

**Real-life impact:** Gradient steps are actual training updates. Even if iterations remain, you stop after 10,000 total training steps across all tuning bursts. Prevents over-tuning that destroys generalization.

**On-paper impact:** Tracks cumulative gradient steps across all bursts. Check: `if self.total_gradient_steps >= self.max_gradient_steps: break`. Each burst adds `n_steps` (controlled by `default_burst_steps`). Was 25000 in older versions — found to "destroy generalization" by over-training.

---

### `patience: 10`

**Real-life impact:** How many consecutive non-improving iterations before giving up on auto-tuning. Patience=10 means "I'll try 10 times without improvement before accepting the best result so far."

**On-paper impact:** Defined in config but **not actively used** in current autotuner code. Legacy parameter for early stopping logic that may be re-enabled in future versions.

---

### `cv_folds: 3`

**Real-life impact:** Cross-validation (CV) splits data into 3 parts (folds). Train on 2 folds, test on 1 — rotate and repeat. This prevents "getting lucky on a specific test set" and gives a more reliable measure of model quality.

**On-paper impact:** Used in Pass 3 of threshold optimization (`_cv_refine`). Creates `cv_folds` indices from the search_eval partition (NOT search_train). Each fold evaluates different thresholds, then results are aggregated. More folds (5–10) give better estimates but take longer.

---

### `confirmation_fraction: 0.50`

**Real-life impact:** After finding a promising tuned model, you hold back 50% of the data for a final "confirmation test." If the model passes this held-out test, you accept it; otherwise, reject it. Prevents overfitting to the tuning search set.

**On-paper impact:** During data splitting, `n_confirm = int(n * confirmation_fraction)`. This portion is reserved for final confirmation phase after Pareto optimization. If `require_confirmation: true`, the best candidate must perform well on this held-out set before being accepted.

---

### `search_eval_fraction: 0.35`

**Real-life impact:** The auto-tuner splits search data into two parts: 65% for training the FocusedSampler (which picks hard examples) and 35% for evaluation. Training on the same data you'd evaluate on is cheating — this prevents that.

**On-paper impact:** `n_search_eval = int(len(search_idx) * search_eval_fraction)`. The search_eval fraction is used for CV folds and candidate evaluation. Search_train fraction (`1 - search_eval_fraction`) trains the FocusedSampler. Default 0.35 means ~65/35 train/eval split.

---

### `bootstrap_samples: 2000`

**Real-life impact:** Bootstrap resampling means "run the evaluation 2000 times with random sampling of the data" to compute confidence intervals. Tells you "this 0.95 recall could really be 0.93–0.97 with 95% confidence."

**On-paper impact:** Defined in config but **not actively used** in current autotuner code. Legacy parameter for confidence interval estimation in earlier versions.

---

### `require_confirmation: true`

**Real-life impact:** When true, the best auto-tuned model must pass a final held-out test (confirmation set) before being accepted. Prevents "looking good on paper but failing in real world" due to overfitting the search data.

**On-paper impact:** After Pareto optimization, if `require_confirmation` and archive has candidates, runs `_confirmation_phase`. Confirmed result passes only if it meets thresholds on held-out confirmation data. If false, skips confirmation and accepts best Pareto candidate directly.

---

### `group_key: speaker_id`

**Real-life impact:** Speaker-aware splitting ensures the same person's voice doesn't appear in both training and test sets. Prevents the model from "memorizing voices" instead of learning the wake word itself.

**On-paper impact:** During data splitting, reads speaker IDs from `val/file_paths.json` if `group_key` is set. Groups samples by speaker, then ensures no speaker appears in both train and eval partitions. Critical for preventing data leakage in speaker-dependent tasks.

---

### `pareto_improvement_threshold: 0.003`

**Real-life impact:** Pareto optimization balances multiple goals (low FAH, high recall, high AUC-PR). A 0.3% (0.003) improvement means "only count it as progress if FAH drops by 0.3% or recall rises by 0.3%." Prevents counting meaningless tiny improvements.

**On-paper impact:** When evaluating if a new candidate improves the Pareto front, requires `abs(new_metric - old_metric) >= pareto_improvement_threshold` for at least one objective. Smaller thresholds (0.001) allow more aggressive refinement; larger thresholds (0.01) require stronger improvements.

---

### `convergence_window: 7`

**Real-life impact:** Looks at the last 7 iterations to decide "are we done improving?" If the best metric hasn't changed in 7 iterations, training likely converged. Prevents wasting time after gains plateau.

**On-paper impact:** Rolling window of iteration counts checked for convergence. If `best_improvement_iteration` is outside `current_iteration - convergence_window`, no recent improvement detected. Larger windows (10–15) tolerate longer stagnation before concluding convergence.

---

### Auto-Tuning Expert Parameters

#### Burst Control

### `min_burst_steps: 200` / `max_burst_steps: 2500` / `default_burst_steps: 500`

**Real-life impact:** A "burst" is a short fine-tuning run using one strategy. These control burst length: minimum 200 steps, maximum 2500 steps, default 500 steps. Surgical updates — small bursts prevent over-tuning. Was 25000 in older versions, which "destroyed generalization" by over-training.

**On-paper impact:** Each strategy burst runs for `n_steps` gradient steps. `n_steps = clamp(default_steps, min_burst_steps, max_burst_steps)`. Burst steps directly affect training time and fine-tuning intensity. Larger bursts (2000–5000) allow more dramatic changes but risk overfitting; smaller bursts (100–500) are conservative.

---

#### Learning Rate Control

### `min_lr: 0.0000001` / `max_lr: 0.00005` / `default_lr: 0.00001`

**Real-life impact:** Auto-tuning uses very small learning rates compared to initial training. These are the bounds: min=0.0000001, max=0.00005, default=0.00001. Small LRs are for fine-tuning — like making tiny brush strokes on a finished painting.

**On-paper impact:** Burst LR is set as `lr = clamp(parent.lr or default_lr, min_lr, max_lr)`. Values are 10–100× smaller than initial training LR (~0.001). Allows precise weight adjustments without catastrophic forgetting.

---

#### SAM (Sharpness-Aware Minimization)

### `sam_rho: 0.05`

**Real-life impact:** SAM finds "flat minima" — regions where the model's error changes slowly. Flat minima are more robust to data variations. Rho=0.05 controls how far you "nudge" weights to test flatness. Like testing if a ball is stable by poking it slightly.

**On-paper impact:** SAM perturbation radius. Before each gradient step, computes gradient at `weights + rho * grad / (norm(grad) + epsilon)`. Larger rho (0.1–0.5) increases flatness-seeking strength but slows training. Smaller rho (0.01–0.05) is gentler but less aggressive in finding flat minima.

---

#### SWA (Stochastic Weight Averaging)

### `swa_collection_interval: 100`

**Real-life impact:** SWA averages model weights collected over time, smoothing out noise. Collects a snapshot every 100 steps. Final SWA model = average of all snapshots. Like taking a time-exposure photo to blur away jitter.

**On-paper impact:** During training bursts with SWA enabled, every `swa_collection_interval` steps, weights are serialized and stored. After burst completion, SWA model = mean(all collected snapshots). More snapshots (smaller interval = 50) give better averaging but use more memory.

---

#### Simulated Annealing

### `initial_temperature: 0.5` / `cooling_rate: 0.97`

**Real-life impact:** Simulated annealing is like cooling molten metal — initially hot (high temperature), accepting many random changes (including bad ones) to explore. As it cools (temperature drops), becomes picky, accepting only improvements. Controls exploration vs exploitation trade-off.

**On-paper impact:** Temperature starts at `initial_temperature` and multiplies by `cooling_rate` each iteration: `temp *= cooling_rate`. Higher initial temp (1.0–2.0) accepts worse candidates more freely. Cooling rate 0.97 = slow cooling (3% per iteration); 0.9 = fast cooling (10% per iteration).

---

### `reheat_after: 5` / `reheat_factor: 1.3`

**Real-life impact:** If stuck (5 consecutive non-improvements), "reheat" by raising temperature 1.3×. Like saying "I'm stuck, let me be more adventurous again." Helps escape local optima.

**On-paper impact:** Tracks consecutive non-improvements. When count exceeds `reheat_after`, temperature multiplies by `reheat_factor`. Reheat allows accepting worse candidates again to jump out of stagnation. Larger `reheat_factor` (1.5–2.0) is more aggressive re-exploration.

---

#### Candidate Management

### `active_pool_size: 16`

**Real-life impact:** The autotuner maintains a pool of 16 promising candidate models simultaneously. When a new model is found, it's added to the pool if better than the worst. Keeps multiple options alive for Pareto selection.

**On-paper impact:** Maximum number of candidates kept in `active_pool`. When full, worst candidate is evicted if new candidate is better. Larger pools (32–64) preserve more diversity but slow down evaluation; smaller pools (8) are faster but may discard promising candidates too early.

---

### `pareto_archive_size: 32`

**Real-life impact:** Pareto archive stores the best models across all trade-offs (low FAH vs high recall vs high AUC-PR). Size 32 means keep at most 32 Pareto-optimal candidates. Prevents archive from growing indefinitely.

**On-paper impact:** `ParetoArchive` maintains up to `max_size` candidates. When full and new candidate dominates existing ones, dominated candidates are evicted. Larger archives (64–128) preserve more trade-off points but increase memory usage.

---

#### Stagnation Escalation (Stir)

### `stir_level_1: 3` / `stir_level_2: 5` / `stir_level_3: 7` / `stir_level_4: 9` / `stir_level_5: 12`

**Real-life impact:** "Stir" is an escape strategy when stuck. Levels escalate the response: level 1 (mild) after 3 stagnations, level 2 (moderate) after 5, etc. Up to level 5 (extreme) after 12. Like shaking the board harder each time pieces get stuck.

**On-paper impact:** `StirController.thresholds = [3, 5, 7, 9, 12]`. Each level maps to a specific escape strategy (e.g., randomize hyperparameters, switch arm, burst length mutation). Stagnation count resets after successful stir. Different thresholds allow graduated response to stuck states.

---

**Real-life impact:** Curriculum learning is like teaching — start with easy examples, then gradually introduce harder ones. Threshold=0.3 means "advance to next difficulty level only if you see at least 30% improvement." Prevents moving to harder examples too early.

**On-paper impact:** Implemented in `MaxQualityAutoTuner` for the `macro_refine` strategy arm. Controls when to advance curriculum stages (0→1→2→3) based on percentage improvement in FAH, recall, or AUC-PR metrics. Higher threshold = more conservative advancement (requires more improvement).
---

## Summary

| Section | Purpose |
|---------|---------|
| Training Optimizer | Controls gradient behavior, learning rate schedule, plateau responses |
| Model Regularization | Prevents overfitting via dropout and weight penalties |
| Evaluation | Measures model quality with threshold sweeps and latency benchmarks |
| Auto-Tuning | Post-training fine-tuning to optimize FAH and recall |
| Auto-Tuning Expert | Advanced hyperparameter control for surgical fine-tuning |

For full configuration reference, see [CONFIGURATION.md](CONFIGURATION.md).
