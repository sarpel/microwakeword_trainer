#!/usr/bin/env bash

mww-train --config max_quality

tensorboard --logdir ./logs &

python scripts/evaluate_model.py \
    --checkpoint ./models/checkpoints/best_weights.weights.h5  \
    --config max_quality \
    --split test \
    --analyze


mww-export \
    --checkpoint ./models/checkpoints/best_weights.weights.h5  \
    --config config/presets/max_quality.yaml \
    --output ./exports \
    --model-name "hey_katya"


python scripts/evaluate_model.py \
    --tflite exports/hey_katya.tflite \
    --config max_quality \
    --split test

