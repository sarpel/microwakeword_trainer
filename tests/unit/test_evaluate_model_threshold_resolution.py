"""Tests for threshold resolution in scripts.evaluate_model."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_model as eval_mod


def test_load_manifest_threshold_reads_adjacent_manifest(tmp_path: Path) -> None:
    tflite_path = tmp_path / "wake_word.tflite"
    tflite_path.write_bytes(b"x")
    manifest = {
        "micro": {
            "probability_cutoff": 0.91,
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    cutoff = eval_mod._load_manifest_threshold(str(tflite_path))
    assert cutoff == 0.91


def test_resolve_eval_threshold_prefers_manifest_for_tflite(tmp_path: Path) -> None:
    tflite_path = tmp_path / "wake_word.tflite"
    tflite_path.write_bytes(b"x")
    (tmp_path / "manifest.json").write_text(json.dumps({"micro": {"probability_cutoff": 0.88}}))

    cfg = {
        "export": {"probability_cutoff": 0.75},
        "evaluation": {"default_threshold": 0.97},
    }
    threshold, source = eval_mod._resolve_eval_threshold(str(tflite_path), cfg)

    assert threshold == 0.88
    assert source == "manifest.micro.probability_cutoff"


def test_resolve_eval_threshold_fallback_order_without_manifest() -> None:
    cfg = {
        "export": {"probability_cutoff": 0.79},
        "evaluation": {"default_threshold": 0.97},
    }

    threshold_ckpt, source_ckpt = eval_mod._resolve_eval_threshold("model.weights.h5", cfg)
    assert threshold_ckpt == 0.79
    assert source_ckpt == "config.export.probability_cutoff"

    threshold_empty, source_empty = eval_mod._resolve_eval_threshold("model.weights.h5", {})
    assert threshold_empty == 0.5
    assert source_empty == "fallback=0.5"
