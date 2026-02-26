# src/evaluation/

Evaluation layer for wake-word quality analysis.

## Module Overview

| File | Responsibility |
|------|----------------|
| `metrics.py` | Main metrics orchestration via `MetricsCalculator` and compatibility wrappers |
| `fah_estimator.py` | FAH-specific estimation logic (`FAHEstimator`) and ambient-duration handling |
| `calibration.py` | Probability calibration analysis utilities |

## Primary Entry Point

Use `MetricsCalculator` for standardized class-based evaluation:

```python
from src.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator(y_true=y_true, y_score=y_scores)
metrics = calc.compute_all_metrics(ambient_duration_hours=ambient_hours, threshold=0.5)
```

## Metrics APIs (via `MetricsCalculator`)

- `compute_fah_metrics()`
- `compute_roc_pr_curves()`
- `compute_average_viable_recall()`
- `compute_recall_at_no_faph()`
- `compute_all_metrics()`
- `compute_latency()`
- `compute_precision_recall()`

## Calibration Section

`calibration.py` provides:

- `compute_calibration_curve()` for reliability bins
- `compute_brier_score()` for probabilistic error quality
- `calibrate_probabilities()` for lightweight post-hoc score calibration

## Notes

- Keep FAH-specific fields/state (e.g., `ambient_duration_hours`) in `FAHEstimator`.
- Maintain backward compatibility via wrapper functions in `metrics.py` where needed.
- Prefer centralized ROC/AUC and PR handling through `MetricsCalculator.compute_all_metrics()`.
