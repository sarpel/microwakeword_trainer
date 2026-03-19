# TensorBoard Configuration Reference

This document provides detailed explanations of all TensorBoard-related configuration options in the `performance` section of the config presets.

---

## Table of Contents

- [Core Settings](#core-settings)
- [Curve Visualization Controls](#curve-visualization-controls)
- [Visualization Types](#visualization-types)
- [Advanced Metrics](#advanced-metrics)
- [Weight and Histogram Logging](#weight-and-histogram-logging)
- [Phase 4 "Sophisticated Metrics"](#phase-4-sophisticated-metrics)
- [Quick Reference: Feature Toggle Matrix](#quick-reference-feature-toggle-matrix)
- [Usage Notes](#usage-notes)

---

## Core Settings

### `tensorboard_enabled: true`

Master on/off switch for all TensorBoard logging. When `false`, no logs are written to `tensorboard_log_dir`.

**Type:** `bool`
**Default:** `true`

---

### `tensorboard_log_dir: "./logs"`

Output directory path for TensorBoard event files. View logs by running:

```bash
tensorboard --logdir ./logs
```

**Type:** `str`
**Default:** `"./logs"`

---

## Curve Visualization Controls

### `tensorboard_log_pr_curves: true`

Enables Precision-Recall curve generation and logging. Controls both:

- Interactive PR curve (via `tf.summary.pr_curve`)
- PR curve image alongside ROC in a 2-panel figure

**Type:** `bool`
**Default:** `true`

**Impact:** Lightweight - minimal performance impact

---

### `tensorboard_log_roc_curves: true`

Enables ROC (Receiver Operating Characteristic) curve generation. Logged as an image combined with PR curve in a 2-panel figure.

**Type:** `bool`
**Default:** `true`

**Impact:** Lightweight - minimal performance impact

**Note:** Both PR and ROC curves are generated as a single `log_roc_pr_curves()` call. These flags determine which panels are drawn. Requires `tensorboard_log_images: true` to actually write the images.

---

## Visualization Types

### `tensorboard_log_histograms: false`

Enables histogram logging for score distributions. Controls:

- `log_score_histograms()` - distribution histograms for positives/negatives/hard negatives
- `log_fah_recall_curve()` - FAH (False activations per hour) vs Recall operating curve

**Type:** `bool`
**Default:** `false`

**Impact:** Light - ~5-10ms/step

---

### `tensorboard_log_images: false`

Master switch for image logging. Required for:

- ROC/PR curve plots
- Confusion matrices
- Score distribution histograms
- FAH/Recall curves

**Type:** `bool`
**Default:** `false`

**Impact:** Variable - depends on what images are enabled

**Note:** Even with specific flags enabled (e.g., `tensorboard_log_roc_curves`), images will not be written if this is `false`.

---

### `tensorboard_log_graph: false`

Logs the TensorFlow computation graph as a static trace. Shows model architecture.

**Type:** `bool`
**Default:** `false`

**Impact:** One-time cost - logs only once at initialization

**Note:** Traces a forward pass with a dummy input to capture the graph structure.

---

## Advanced Metrics

### `tensorboard_log_advanced_scalars: true`

Enables logging of computed scalar metrics during evaluation:

- **EER (Equal Error Rate)** and threshold
- **Calibration metrics** (brier_score)
- **Per-category metrics** (with `positive_`, `negative_`, `hard_negative_` prefixes)

Reduces redundant eval-time overhead by batching related metrics into a single `log_advanced_scalars()` call.

**Type:** `bool`
**Default:** `true`

**Impact:** Light - minimal overhead

---

## Weight and Histogram Logging

### `tensorboard_log_weight_histograms: false`

Enables separate logging of model weight distributions as histograms. Logged to `weights/*/` namespace.

**Different from:** `tensorboard_log_histograms` (which logs score distributions).

**Type:** `bool`
**Default:** `false`

**Impact:** Light - ~5-10ms/step

---

### `tensorboard_image_interval: 5000`

Steps between writing image summaries to disk. Higher values reduce disk I/O and TensorBoard file size.

**Type:** `int`
**Default:** `5000`

**Applied to:** Any metric that logs images (ROC/PR curves, confusion matrices, etc.)

---

### `tensorboard_histogram_interval: 5000`

Steps between writing histogram summaries. Applies to score/histogram logging when enabled.

**Type:** `int`
**Default:** `5000`

**Applied to:** Metrics under `tensorboard_log_histograms`

---

## Phase 4 "Sophisticated Metrics"

These are advanced debugging/metrics options that provide deep insights into training dynamics. They are disabled by default in `max_quality.yaml` to maintain cleaner logs and higher throughput.

### `tensorboard_log_learning_rate: false`

Logs the learning rate schedule to `train/learning_rate` scalar. Useful for analyzing LR schedule effectiveness and detecting unintended LR changes.

**Type:** `bool`
**Default:** `false`

**Impact:** Negligible - <1ms/step

---

### `tensorboard_log_gradient_norms: false`

Logs gradient statistics to detect vanishing/exploding gradient problems:

- Global gradient norm
- Per-layer gradient norms
- Gradient distribution histogram
- Mean, std, max statistics

**Type:** `bool`
**Default:** `false`

**Impact:** **Heavy** - adds ~20-40ms/step

**Use cases:**
- Debugging training instability
- Detecting dead neurons (zero gradients)
- Analyzing gradient flow through depth

---

### `tensorboard_log_activation_stats: false`

Logs per-layer activation statistics:

- Sparsity (% of zeros)
- Mean and std
- Saturation ratio (% activations near 0.95)
- Occasional activation histograms (every 5× the sophisticated_interval)

**Type:** `bool`
**Default:** `false`

**Impact:** **Heavy** - ~30-50ms/step

**Use cases:**
- Identifying dead neurons
- Detecting activation saturation
- Understanding layer-wise behavior
- Debugging batch normalization issues

---

### `tensorboard_log_confidence_drift: false`

Tracks prediction confidence changes over time using a sliding window of 100 steps:

- Mean confidence
- High confidence ratio (>0.9)
- Low confidence ratio (<0.1)
- Uncertainty metric (1 - mean(|score - 0.5|))
- Drift (change in mean confidence over recent 20 steps)

**Type:** `bool`
**Default:** `false`

**Impact:** Light - ~5-10ms/step

**Use cases:**
- Detecting training degradation
- Monitoring model calibration
- Tracking confidence distribution shifts
- Early warning for drift problems

---

### `tensorboard_log_per_class_accuracy: false`

Breaks down accuracy by class during training using a 0.5 decision threshold:

- Positive class accuracy
- Negative class accuracy
- Class balance (ratio of positives in batch)
- Predicted positive ratio

**Type:** `bool`
**Default:** `false`

**Impact:** Light - ~5ms/step

**Use cases:**
- Detecting class imbalance issues
- Monitoring classifier balance
- Debugging bias toward one class

---

### `tensorboard_sophisticated_interval: 10000`

Steps between **all** Phase 4 sophisticated metrics. Only logs when `step % interval == 0`. Higher intervals reduce logging frequency → higher throughput.

**Type:** `int`
**Default:** `2000` (in dataclass) / `10000` (in max_quality.yaml)

**Special values:**
- `0`: Disable interval-based gating (logs every step when metrics are enabled)
- Set to very high values to enable metrics but log them rarely

---

## Quick Reference: Feature Toggle Matrix

| Variable | Curve | Image | Histogram | Scalar | Heavy? | Primary Namespace |
|----------|-------|-------|-----------|--------|--------|------------------|
| `tensorboard_log_pr_curves` | ✅ PR | ✅ | ❌ | ❌ | No | `curves/pr_curve` |
| `tensorboard_log_roc_curves` | ✅ ROC | ✅ | ❌ | ❌ | No | `curves/roc_pr` |
| `tensorboard_log_histograms` | ❌ | ✅ | ✅ | ❌ | Light | `distributions/*` |
| `tensorboard_log_images` | ❌ | ✅ master | ✅ master | ❌ | Variable | `images/*` |
| `tensorboard_log_graph` | ❌ | ✅ (trace) | ❌ | ❌ | Once | `model_graph` |
| `tensorboard_log_advanced_scalars` | ❌ | ❌ | ❌ | ✅ | Light | `advanced/*`, `calibration/*`, `per_class/*` |
| `tensorboard_log_weight_histograms` | ❌ | ❌ | ✅ | ❌ | Light | `weights/*` |
| `tensorboard_log_learning_rate` | ❌ | ❌ | ❌ | ✅ | No | `train/learning_rate` |
| `tensorboard_log_gradient_norms` | ❌ | ❌ | ✅ | ✅ | **Heavy** | `gradients/*` |
| `tensorboard_log_activation_stats` | ❌ | ❌ | ✅ | ✅ | **Heavy** | `activations/*` |
| `tensorboard_log_confidence_drift` | ❌ | ❌ | ❌ | ✅ | Light | `confidence/*` |
| `tensorboard_log_per_class_accuracy` | ❌ | ❌ | ❌ | ✅ | Light | `per_class/*`, `train/*` |

---

## Usage Notes

### Image Dependencies
Most image-based logging (ROC curves, histograms, etc.) requires both:
1. The specific feature flag (e.g., `tensorboard_log_roc_curves`)
2. The master image switch: `tensorboard_log_images: true`

Enable `tensorboard_log_images` if you want any image logging at all.

---

### Frequency Control
- **Images/histograms**: Controlled by `tensorboard_image_interval` and `tensorboard_histogram_interval`
- **Sophisticated metrics**: Controlled by `tensorboard_sophisticated_interval`
- **Standard metrics**: Logged every evaluation step

---

### Performance Trade-offs

| Configuration | Throughput | Debugging Capability | Recommended For |
|---------------|-------------|-------------------|-----------------|
| All defaults (minimal) | ⚡⚡⚡ | ⚪ | Production training, long runs |
| Enable curves/images only | ⚡⚡ | 🟢 | Periodic training monitoring |
| Enable Phase 4 metrics (high interval) | ⚡ | 🟡 | Debugging specific training issues |
| Enable Phase 4 metrics (low interval) | ⚪ | 🔴 | Deep model analysis, research |

---

### Advanced Debugging Workflow

For deep debugging of training issues, enable all Phase 4 metrics with a high interval:

```yaml
performance:
  # Sophisticated metrics - enabled for debugging
  tensorboard_log_learning_rate: true
  tensorboard_log_gradient_norms: true
  tensorboard_log_activation_stats: true
  tensorboard_log_confidence_drift: true
  tensorboard_log_per_class_accuracy: true
  tensorboard_sophisticated_interval: 500  # Log every 500 steps
```

Then reduce training throughput by increasing interval as needed:

```yaml
  tensorboard_sophisticated_interval: 5000  # Less frequent, higher throughput
```

---

## Monitoring in TensorBoard

Once training is running and TensorBoard is enabled, view metrics:

```bash
tensorboard --logdir ./logs
```

Navigate to specific namespaces:

- `curves/` - ROC and PR curves
- `distributions/` - Score distributions, histograms
- `confidence/` - Confidence drift tracking
- `gradients/` - Gradient norms and statistics
- `activations/` - Per-layer activation statistics
- `advanced/` - EER, calibration, FAH metrics
- `per_class/` - Accuracy breakdown by class
- `train/` - Training-specific scalars including learning rate

---

## Common Issues

### "No curves/images showing in TensorBoard"

Check:
1. `tensorboard_enabled: true`
2. `tensorboard_log_images: true` (required for any image logging)
3. Sufficient steps have passed to hit interval thresholds
4. TensorBoard is pointing to the correct `tensorboard_log_dir`

### "Training is very slow with TensorBoard enabled"

Likely causes:
1. Phase 4 heavy metrics enabled (`gradient_norms`, `activation_stats`)
2. Intervals set too low (`tensorboard_image_interval`, `tensorboard_sophisticated_interval`)

Solutions:
1. Disable heavy metrics when not needed
2. Increase intervals to reduce logging frequency
3. Verify the bottleneck is actually TensorBoard logging, not model/ I/O

### "No sophisticated metrics appearing"

Check:
1. `tensorboard_sophisticated_interval` is not zero or extremely high
2. Metrics are individually enabled (e.g., `tensorboard_log_gradient_norms: true`)
3. You've run enough steps: `step % sophisticated_interval == 0`
