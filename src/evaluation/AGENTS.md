# src/evaluation/

Evaluation layer for wake-word quality analysis.

## Module Overview

| File | Lines | Responsibility |
|------|-------|----------------|
| `metrics.py` | 397 | Main metrics orchestration via `MetricsCalculator` |
| `fah_estimator.py` | 74 | FAH-specific estimation logic |
| `calibration.py` | 94 | Probability calibration analysis |

## Primary Entry Point

```python
from src.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator(y_true=y_true, y_score=y_scores)
metrics = calc.compute_all_metrics(ambient_duration_hours=hours, threshold=0.5)
```

## MetricsCalculator Methods

| Method | Purpose |
|--------|---------|
| `compute_fah_metrics()` | False activations per hour |
| `compute_roc_pr_curves()` | ROC and precision-recall curves |
| `compute_recall_at_target_fah()` | Recall at target FAH threshold |
| `compute_all_metrics()` | All metrics in single call |

## Key Metrics

- **FAH**: False Activations per Hour
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Recall@target_fah**: Recall at target false activation rate
- **Adaptive Thresholding**: Evaluation metrics (accuracy, precision, recall, F1) are re-computed at an optimized threshold derived from `target_fah` (or median fallback) for production-relevant reporting.

## Related Documentation

- [Training Guide](../../docs/TRAINING.md)
- [Configuration Reference](../../docs/CONFIGURATION.md)

## Evaluation Reporting Workflow

For end-to-end post-training assessment, use:

```bash
python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard --output-dir logs/
```

Outputs under `evaluation_artifacts/` include:
- `evaluation_report.json`
- PNG plots (ROC, PR, DET, confusion matrix, calibration, threshold/cost curves)
- `executive_report.md`
- `executive_report.html`

Optional interactive dashboard generation:

```bash
python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json
```

## Script Reliability Notes

- `scripts/evaluate_model.py` timestamp generation uses module-scope `datetime` import. A prior conditional local import caused `UnboundLocalError` in `_compute_metrics`; this is fixed as of 2026-03-17.
