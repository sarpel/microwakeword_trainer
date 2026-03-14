# QUICK COMMANDS

## EVALUATE

```python
python scripts/evaluate_model.py \
	--checkpoint ./models/checkpoints/final_weights.weights.h5 \
	--config max_quality \
	--split test \
	--analyze
```
mww-mine-hard-negatives extract-top-fps
## AUTO-TUNE

```bash
mww-autotune \
	--checkpoint ./models/checkpoints/final_weights.weights.h5 \
	--config max_quality \
	--target-fah 0.2 \
	--target-recall 0.95 \
	--max-iterations 20 \
	--output-dir ./tuning_results
```

## EVAL AFTER AUTO-TUNE

```python
python scripts/evaluate_model.py \
	--checkpoint ./tuning_results/checkpoints/tuned_fah0.000_rec1.000_iter3.weights.h5 \
	--config max_quality \
	--split test \
	--analyze
```

## EXPORT

```bash
mww-export \
	--checkpoint ./models/checkpoints/final_weights.weights.h5 \
	--output models/exported/ \
	--config max_quality
```

## EVAL EXPORTED MODEL

```python
python scripts/evaluate_model.py \
	--tflite models/exported/wake_word.tflite \
	--config max_quality \
	--split test
```
