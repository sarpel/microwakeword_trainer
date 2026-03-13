"""High-value core tests for mining logic used in training and post-processing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.training.mining import (
    HardExampleMiner,
    collect_false_predictions,
    copy_files_to_mined_dir,
    deduplicate_by_hash,
    filter_epochs_by_min_epoch,
    log_false_predictions_to_json,
)


def test_get_hard_samples_confidence_strategy_selects_high_score_negatives(tmp_path: Path):
    miner = HardExampleMiner(strategy="confidence", fp_threshold=0.8, output_dir=str(tmp_path))

    labels = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    predictions = np.array([0.9, 0.3, 0.95, 0.4, 0.81], dtype=np.float32)

    indices = miner.get_hard_samples(labels, predictions)

    assert indices.tolist() == [0, 4]


def test_get_hard_samples_entropy_strategy_selects_boundary_negatives(tmp_path: Path):
    miner = HardExampleMiner(strategy="entropy", output_dir=str(tmp_path))

    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    predictions = np.array([0.42, 0.51, 0.9, 0.49], dtype=np.float32)

    indices = miner.get_hard_samples(labels, predictions)

    # For entropy strategy, negatives within abs(score - 0.5) < 0.1 are selected.
    assert indices.tolist() == [0, 1]


def test_get_hard_samples_invalid_strategy_raises(tmp_path: Path):
    miner = HardExampleMiner(strategy="invalid", output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="Unknown mining strategy"):
        miner.get_hard_samples(np.array([0]), np.array([0.5], dtype=np.float32))


def test_log_false_predictions_to_json_writes_sorted_top_k_and_metadata(tmp_path: Path):
    log_file = tmp_path / "false_predictions.json"
    best_weights = tmp_path / "best.weights.h5"
    y_true = np.array([0, 0, 1, 0], dtype=np.int64)
    y_scores = np.array([0.2, 0.91, 0.95, 0.85], dtype=np.float32)
    val_paths = ["a.wav", "b.wav", "c.wav", "d.wav"]

    entry = log_false_predictions_to_json(
        epoch=3,
        y_true=y_true,
        y_scores=y_scores,
        fp_threshold=0.8,
        top_k=1,
        log_file=str(log_file),
        val_paths=val_paths,
        best_weights_path=str(best_weights),
    )

    assert entry is not None
    assert entry["epoch"] == 3
    assert entry["false_positive_count"] == 2
    assert len(entry["false_predictions"]) == 1
    assert entry["false_predictions"][0]["index"] == 1
    assert entry["false_predictions"][0]["file_path"] == "b.wav"

    data = json.loads(log_file.read_text())
    assert data["metadata"]["top_k_per_epoch"] == 1
    assert data["metadata"]["model_checkpoint"] == str(best_weights)
    assert "3" in data["epochs"]


def test_filter_collect_and_deduplicate_pipeline(tmp_path: Path):
    f1 = tmp_path / "x1.wav"
    f2 = tmp_path / "x2.wav"
    f3 = tmp_path / "x3.wav"
    f1.write_bytes(b"same")
    f2.write_bytes(b"same")
    f3.write_bytes(b"different")

    log_data = {
        "epochs": {
            "1": {"false_predictions": [{"index": 0, "score": 0.6}]},
            "10": {
                "false_predictions": [
                    {"index": 7, "score": 0.9, "file_path": str(f1)},
                    {"index": 8, "score": 0.8, "file_path": str(f2)},
                ]
            },
            "11": {
                "false_predictions": [
                    {"index": 9, "score": 0.95, "file_path": str(f3)},
                ]
            },
            "bad": {"false_predictions": [{"index": 1, "score": 0.99}]},
        }
    }

    filtered = filter_epochs_by_min_epoch(log_data, min_epoch=10)
    assert sorted(filtered.keys()) == [10, 11]

    preds = collect_false_predictions(filtered, top_k=1)
    assert len(preds) == 2
    assert preds[0]["epoch"] == 10
    assert preds[1]["epoch"] == 11

    unique, hash_map = deduplicate_by_hash(preds)
    assert len(unique) == 2
    assert len(hash_map) == 2


def test_deduplicate_by_hash_requires_path_field():
    with pytest.raises(ValueError, match="missing 'file_path' or 'path'"):
        deduplicate_by_hash([{"score": 0.9}])


def test_copy_files_to_mined_dir_dry_run_and_real_copy(tmp_path: Path):
    src = tmp_path / "sample.wav"
    src.write_bytes(b"audio")
    out_dir = tmp_path / "mined"
    predictions = [{"file_path": str(src), "score": 0.876, "epoch": 12}]

    copied_dry, skipped_dry = copy_files_to_mined_dir(predictions, out_dir, dry_run=True)
    assert copied_dry == 1
    assert skipped_dry == 0
    assert not list(out_dir.glob("*.wav"))

    copied, skipped = copy_files_to_mined_dir(predictions, out_dir, dry_run=False)
    assert copied == 1
    assert skipped == 0
    outputs = list(out_dir.glob("*.wav"))
    assert len(outputs) == 1
    assert "_e12_s0.876" in outputs[0].name
