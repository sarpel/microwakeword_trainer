# src/evaluation/

Evaluation layer for wake-word quality analysis.

## Module Overview

| File | Lines | Responsibility |
|------|-------|----------------|
| `metrics.py` | 397 | Main metrics orchestration via `MetricsCalculator` + standalone functions |
| `fah_estimator.py` | 74 | FAH-specific estimation logic (`FAHEstimator`) and ambient-duration handling |
| `calibration.py` | 94 | Probability calibration analysis utilities |

## Primary Entry Point

Use `MetricsCalculator` for standardized class-based evaluation:

```python
from src.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator(y_true=y_true, y_score=y_scores)
metrics = calc.compute_all_metrics(ambient_duration_hours=ambient_hours, threshold=0.5)
```

## MetricsCalculator Methods

| Method | Purpose |
|--------|---------|
| `compute_fah_metrics()` | False activations per hour |
| `compute_roc_pr_curves()` | ROC and precision-recall curves |
| `compute_average_viable_recall()` | Average recall across viable thresholds |
| `compute_recall_at_no_faph()` | Recall at zero false activations per hour |
| `compute_all_metrics()` | All metrics in single call |
| `compute_latency()` | Detection latency analysis |
| `compute_precision_recall()` | Precision/recall at threshold |

## Standalone Functions (metrics.py)

| Function | Purpose |
|----------|---------|
| `compute_accuracy()` | Simple accuracy calculation |
| `compute_roc_auc()` | ROC AUC score |
| `_manual_roc_auc()` | Manual ROC AUC (fallback) |
| `compute_precision_recall()` | Precision/recall at threshold |
| `_synchronize_output()` | Output alignment for streaming |
| `compute_latency()` | Latency measurement |
| `compute_roc_pr_curves()` | ROC and PR curve data |
| `compute_recall_at_no_faph()` | Recall at zero FAPH |
| `compute_average_viable_recall()` | Average viable recall |
| `compute_fah_metrics()` | FAH metrics |
| `compute_all_metrics()` | All metrics (standalone wrapper) |

## FAHEstimator (`fah_estimator.py`)

```python
from src.evaluation.fah_estimator import FAHEstimator

estimator = FAHEstimator(ambient_duration_hours=24.0)
fah_metrics = estimator.compute_fah_metrics(y_true, y_score, threshold=0.5)
fah_rate = estimator.estimate_false_activations_per_hour(false_positives, duration)
```

## Calibration Section (`calibration.py`)

| Function | Purpose |
|----------|---------|
| `compute_calibration_curve()` | Reliability bins for calibration analysis |
| `compute_brier_score()` | Probabilistic error quality metric |
| `calibrate_probabilities()` | Lightweight post-hoc score calibration |

## Notes

- Keep FAH-specific fields/state (e.g., `ambient_duration_hours`) in `FAHEstimator`.
- Maintain backward compatibility via standalone wrapper functions in `metrics.py`.
- Prefer centralized ROC/AUC and PR handling through `MetricsCalculator.compute_all_metrics()`.
- Used by `src/training/trainer.py` during validation steps.
- Best model selection based on FAH (false activations per hour), then recall at thresholds.
