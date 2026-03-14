"""Unit tests for export metadata/cutoff propagation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


def test_select_export_probability_cutoff_prefers_fah_at_target_recall() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.99, 0.95, 0.90, 0.10, 0.20, 0.30, 0.40, 0.80], dtype=np.float32)

    cfg = {
        "evaluation": {"target_recall": 0.90, "target_fah": 2.0, "n_thresholds": 101},
        "training": {"ambient_duration_hours": 10.0, "test_split": 0.1},
        "export": {"sliding_window_size": 1},
    }

    out = tflite_mod._select_export_probability_cutoff(y_true, y_scores, cfg)

    assert 0.0 < out["probability_cutoff"] <= 1.0
    assert 1 <= out["probability_cutoff_uint8"] <= 255
    assert out["selection_method"] == "fah_at_target_recall"
    assert out["fallback_applied"] is False


def test_select_export_probability_cutoff_falls_back_to_recall_at_target_fah() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.60, 0.55, 0.51, 0.50, 0.45, 0.40, 0.35, 0.30], dtype=np.float32)

    cfg = {
        "evaluation": {"target_recall": 1.1, "target_fah": 0.0, "n_thresholds": 101},
        "training": {"ambient_duration_hours": 10.0, "test_split": 0.1},
        "export": {"sliding_window_size": 1},
    }

    out = tflite_mod._select_export_probability_cutoff(y_true, y_scores, cfg)

    assert 0.0 < out["probability_cutoff"] <= 1.0
    assert out["selection_method"] == "recall_at_target_fah"


def test_select_export_probability_cutoff_applies_low_threshold_guardrail() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
    y_scores = np.array([0.02, 0.03, 0.04, 0.0, 0.0, 0.0], dtype=np.float32)

    cfg = {
        "evaluation": {"target_recall": 1.0, "target_fah": 0.0, "n_thresholds": 101, "default_threshold": 0.91},
        "training": {"ambient_duration_hours": 10.0, "test_split": 0.1},
        "export": {"sliding_window_size": 1, "probability_cutoff": 0.92},
    }

    out = tflite_mod._select_export_probability_cutoff(y_true, y_scores, cfg)

    assert out["raw_probability_cutoff"] <= 0.05
    assert out["fallback_applied"] is True
    assert out["probability_cutoff"] == 0.92
    assert out["selection_method"].endswith("_safety_fallback")


def test_auto_calculate_probability_cutoff_runs_tflite_inference(tmp_path: Path, monkeypatch) -> None:
    test_store = tmp_path / "test"
    test_store.mkdir(parents=True)
    (test_store / "features.data").write_bytes(b"x")

    fake_samples = [
        (np.full((6, 40), 10.0, dtype=np.float32), 1),
        (np.full((6, 40), 10.0, dtype=np.float32), 1),
        (np.full((6, 40), 10.0, dtype=np.float32), 1),
        (np.full((6, 40), 1.0, dtype=np.float32), 0),
        (np.full((6, 40), 1.0, dtype=np.float32), 0),
        (np.full((6, 40), 1.0, dtype=np.float32), 0),
    ]

    class FakeFeatureStore:
        def __init__(self, _path):
            self._samples = fake_samples

        def open(self, readonly: bool = True):
            _ = readonly

        def close(self):
            return None

        def __len__(self):
            return len(self._samples)

        def get(self, idx: int):
            return self._samples[idx]

    class FakeInterpreter:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self._input = None

        def allocate_tensors(self):
            return None

        def get_input_details(self):
            return [{"index": 0, "shape": np.array([1, 3, 40]), "dtype": np.int8, "quantization_parameters": {"scales": np.array([0.1]), "zero_points": np.array([0])}}]

        def get_output_details(self):
            return [{"index": 1, "dtype": np.uint8, "quantization_parameters": {"scales": np.array([1.0 / 255.0]), "zero_points": np.array([0])}}]

        def get_tensor_details(self):
            return [
                {"index": 0, "name": "input", "shape": np.array([1, 3, 40]), "dtype": np.int8},
                {"index": 1, "name": "output", "shape": np.array([1, 1]), "dtype": np.uint8},
                {"index": 5, "name": "ReadVariableOp_stream", "shape": np.array([1, 2, 1, 40]), "dtype": np.float32},
            ]

        def set_tensor(self, index: int, value):
            if index == 0:
                self._input = np.asarray(value)

        def invoke(self):
            return None

        def get_tensor(self, index: int):
            _ = index
            # Simulate higher score for positive sample chunks.
            mean_val = float(np.mean(self._input)) if self._input is not None else 0.0
            prob = 0.95 if mean_val > 20 else 0.05
            return np.array([[int(round(prob * 255.0))]], dtype=np.uint8)

    monkeypatch.setattr(tflite_mod, "tf", SimpleNamespace(lite=SimpleNamespace(Interpreter=FakeInterpreter)))
    monkeypatch.setattr("src.data.dataset.FeatureStore", FakeFeatureStore)

    cfg = {
        "evaluation": {"target_recall": 0.8, "target_fah": 1.0, "n_thresholds": 51},
        "training": {"ambient_duration_hours": 10.0, "test_split": 0.1},
        "export": {"sliding_window_size": 1},
    }

    out = tflite_mod._auto_calculate_probability_cutoff("fake.tflite", str(tmp_path), cfg)

    assert 0.0 < out["probability_cutoff"] <= 1.0
    assert out["selection_method"] in {
        "fah_at_target_recall",
        "recall_at_target_fah",
        "fah_at_target_recall_safety_fallback",
        "recall_at_target_fah_safety_fallback",
    }
    assert "observed_negative_chunks" in out
    assert "observed_positive_chunks" in out
