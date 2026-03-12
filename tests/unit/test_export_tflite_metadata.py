"""Unit tests for export metadata/cutoff propagation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.export import tflite as tflite_mod


def test_build_manifest_config_prefers_tuned_cutoff() -> None:
    config = {
        "export": {"probability_cutoff": 0.95},
        "hardware": {"window_step_ms": 10},
    }
    metadata = {"tuned_probability_cutoff": 0.8125}

    out = tflite_mod._build_manifest_config(config, model_name="hey_katya", metadata=metadata)

    assert out["export"]["probability_cutoff"] == 0.8125
    assert out["export"]["wake_word"] == "hey_katya"
    assert out["hardware"]["window_step_ms"] == 10


def test_get_checkpoint_metadata_preserves_cached_extras_when_rescanning(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "best.weights.h5"
    ckpt.write_bytes(b"x")
    sidecar = ckpt.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps({"tuned_probability_cutoff": 0.77}))

    class FakeDataset:
        def __init__(self, shape):
            self.shape = shape

    class FakeH5:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def visititems(self, callback):
            callback("layers/dense/vars/0", FakeDataset((640, 1)))

    monkeypatch.setattr(tflite_mod.h5py, "Dataset", FakeDataset)
    monkeypatch.setattr(tflite_mod.h5py, "File", lambda *_args, **_kwargs: FakeH5())

    out = tflite_mod.get_checkpoint_metadata(str(ckpt), pointwise_filters=64)

    assert out["temporal_frames"] == 10
    assert out["dense_input_features"] == 640
    assert out["dense_output_features"] == 1
    assert out["tuned_probability_cutoff"] == 0.77

    persisted = json.loads(sidecar.read_text())
    assert persisted["tuned_probability_cutoff"] == 0.77
    assert persisted["temporal_frames"] == 10
