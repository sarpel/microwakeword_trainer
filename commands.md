# QUICK COMMANDS

## EVALUATE

```python
python scripts/evaluate_model.py \
	--checkpoint ./models/checkpoints/best_weights.weights.h5 \
	--config max_quality \
	--split test \
	--output-dir ./logs \
	--analyze
```

```bash
python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json
```

python scripts/evaluate_model.py --model models/exported/hey_katya.tflite --config max_quality --split test --output-dir logs/ --analyze
python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json

mww-mine-hard-negatives extract-top-fps
## AUTO-TUNE

```bash
mww-autotune \
	--checkpoint ./models/checkpoints/best_weights.weights.h5 \
	--config max_quality \
	--target-fah 0.2 \
	--target-recall 0.92 \
	--max-iterations 20 \
	--output-dir ./tuning_results
```

## EVAL AFTER AUTO-TUNE

```python
python scripts/evaluate_model.py \
	--checkpoint tuning_results/checkpoints/tuned_fah0.000_rec0.820_iter1.weights.h5 \
	--config max_quality \
	--split test \
	--output-dir ./logs \
	--analyze
```

## EXPORT

```bash
mww-export \
	--checkpoint ./models/checkpoints/best_weights.weights.h5 \
	--output models/exported/ \
	--config max_quality
	--data-dir ./data/processed
```

```bash
mww-export \
	--checkpoint tuning_results/checkpoints/tuned_fah0.000_rec0.820_iter1.weights.h5 \
	--output models/exported/ \
	--config max_quality \
	--data-dir ./data/processed
```
## EVAL EXPORTED MODEL

```python
python scripts/evaluate_model.py \
	--tflite models/exported/hey_katya.tflite \
	--config max_quality \
	--split test \
	--output-dir ./logs
```
