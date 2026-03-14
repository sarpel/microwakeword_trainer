"""Regression tests for BN refresh partition usage in autotuner."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tuning import autotuner as at


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "auto_tuning": {
            "target_fah": 0.5,
            "target_recall": 0.9,
            "output_dir": str(tmp_path / "out"),
            "require_confirmation": True,
        },
        "auto_tuning_expert": {},
        "training": {"batch_size": 8},
        "hardware": {},
    }


def test_confirmation_phase_refreshes_bn_using_repr_partition(tmp_path: Path, monkeypatch) -> None:
    tuner = at.AutoTuner(checkpoint_path=str(tmp_path / "ckpt.weights.h5"), config=_base_cfg(tmp_path))

    candidate = at.CandidateState(
        id="c0",
        weights_bytes=b"w",
        optimizer_state_bytes=b"o",
        batchnorm_state={},
        temperature=1.0,
        threshold_float32=0.5,
        threshold_uint8=128,
        eval_results=at.TuneMetrics(fah=0.1, recall=0.9, auc_pr=0.9, threshold=0.5, threshold_uint8=128),
    )
    tuner.archive.archive = [candidate]

    class DummyModel:
        def __init__(self):
            self.layers = []

    confirm_features = np.zeros((4, 3), dtype=np.float32)
    confirm_labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    confirm_weights = np.ones(4, dtype=np.float32)
    confirm_data = (confirm_features, confirm_labels, confirm_weights, np.arange(4))

    repr_features = np.ones((2, 3), dtype=np.float32)
    repr_data = (repr_features, np.array([0.0, 1.0], dtype=np.float32), np.ones(2, dtype=np.float32), np.arange(2))

    seen: dict[str, np.ndarray] = {}

    monkeypatch.setattr(tuner, "_deserialize_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner, "_restore_bn_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner, "_predict_scores", lambda *args, **kwargs: np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32))

    def fake_refresh(_model, feats, n_batches=50):
        _ = n_batches
        seen["features"] = np.asarray(feats)

    monkeypatch.setattr(tuner, "_refresh_bn_statistics", fake_refresh)

    def fake_optimize(y_true, y_scores, ambient_duration_hours, target_fah, target_recall):
        _ = (y_true, y_scores, ambient_duration_hours, target_fah, target_recall)
        m = at.TuneMetrics(fah=0.1, recall=0.91, auc_pr=0.9, threshold=0.5, threshold_uint8=128)
        return 0.5, 128, m

    monkeypatch.setattr(tuner.threshold_optimizer, "optimize", fake_optimize)

    best_confirmed, _ = tuner._confirmation_phase(DummyModel(), confirm_data, repr_data, ambient_hours=1.0)

    assert best_confirmed is not None
    np.testing.assert_array_equal(seen["features"], repr_features)
