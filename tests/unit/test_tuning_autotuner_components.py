"""Unit tests for lightweight components in src.tuning.autotuner."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.tuning import autotuner as at


def _make_candidate(cid: str, fah: float, recall: float, auc_pr: float, *, strategy_arm: int = 0, iteration: int = 0) -> at.CandidateState:
    return at.CandidateState(
        id=cid,
        weights_bytes=b"w",
        optimizer_state_bytes=b"o",
        batchnorm_state={},
        eval_results=at.TuneMetrics(fah=fah, recall=recall, auc_pr=auc_pr, threshold=0.5, threshold_uint8=128),
        strategy_arm=strategy_arm,
        iteration=iteration,
    )


def test_tune_metrics_dominance_meets_target_and_to_dict() -> None:
    a = at.TuneMetrics(fah=0.3, recall=0.95, auc_pr=0.91, threshold=0.4, threshold_uint8=102)
    b = at.TuneMetrics(fah=0.5, recall=0.90, auc_pr=0.88)
    assert a.dominates(b)
    assert not b.dominates(a)
    assert a.meets_target(0.5, 0.90)
    d = a.to_dict()
    assert d["fah"] == 0.3
    assert d["threshold_uint8"] == 102


def test_pareto_archive_add_dominated_and_diversity() -> None:
    archive = at.ParetoArchive(max_size=5, diversity_threshold=10.0)
    c1 = _make_candidate("c1", 0.5, 0.9, 0.8)
    c2 = _make_candidate("c2", 0.6, 0.88, 0.79)
    c3 = _make_candidate("c3", 0.45, 0.91, 0.81)

    assert archive.try_add(c1)
    # dominated and not diverse enough
    assert not archive.try_add(c2)
    assert len(archive) == 1
    assert archive.try_add(c3)
    assert len(archive) == 1  # c3 dominates c1 and replaces it
    assert archive.archive[0].id == "c3"


def test_pareto_archive_get_best_and_frontier_points() -> None:
    archive = at.ParetoArchive(max_size=10)
    archive.try_add(_make_candidate("c1", 0.4, 0.91, 0.82, strategy_arm=1, iteration=5))
    archive.try_add(_make_candidate("c2", 0.2, 0.89, 0.83, strategy_arm=2, iteration=6))

    best = archive.get_best(target_fah=0.5, target_recall=0.9)
    assert best is not None
    assert best.id == "c1"

    pts = archive.get_frontier_points()
    assert pts
    assert {"id", "fah", "recall", "auc_pr", "threshold", "threshold_uint8", "arm", "iteration"}.issubset(pts[0].keys())


def test_error_memory_update_and_queries() -> None:
    em = at.ErrorMemory(max_history=2)
    idx = np.array([0, 1, 2, 3])
    y_true = np.array([0, 1, 0, 1], dtype=float)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=float)

    for _ in range(3):
        em.update(idx, y_true, y_pred, threshold=0.5)

    assert set(em.get_persistent_fa_indices(min_count=3)) == {0, 2}
    assert set(em.get_persistent_miss_indices(min_count=3)) == {1, 3}
    assert all(len(v) <= 2 for v in em.recent_scores.values())

    near = em.get_near_boundary_indices(threshold=0.85, margin=0.1)
    assert isinstance(near, list)


def test_focused_sampler_builds_batches_for_multiple_arms() -> None:
    np.random.seed(0)
    features = np.random.randn(20, 3).astype(np.float32)
    labels = np.array([1] * 10 + [0] * 10, dtype=np.float32)
    weights = np.ones(20, dtype=np.float32)
    em = at.ErrorMemory(max_history=5)
    em.update(np.arange(20), labels, np.linspace(0.0, 1.0, 20), threshold=0.5)
    sampler = at.FocusedSampler(features, labels, weights, em)

    scores = np.linspace(0.0, 1.0, 20)
    for arm in [0, 1, 2, 3, 5, 6, 999]:
        x, y, w = sampler.build_batch(strategy_arm=arm, threshold=0.5, batch_size=8, curriculum_stage=1, recent_scores=scores)
        assert x.shape[0] == 8
        assert y.shape[0] == 8
        assert w.shape[0] == 8


def test_temperature_and_calibration_helpers() -> None:
    probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=float)

    logits = at._logit(probs)
    roundtrip = at._sigmoid(logits)
    np.testing.assert_allclose(roundtrip, probs, rtol=1e-6, atol=1e-6)

    ece = at.compute_ece(labels, probs, n_bins=4)
    assert 0.0 <= ece <= 1.0
    assert at.compute_ece(np.array([]), np.array([])) == 0.0

    temp_scores = at.apply_temperature(probs, 1.0)
    np.testing.assert_allclose(temp_scores, probs)


def test_fit_temperature_without_scipy_returns_one(monkeypatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.optimize":
            raise ImportError("no scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    t = at.fit_temperature(np.array([0.2, 0.8]), np.array([0, 1]))
    assert t == 1.0


def test_diagnose_regime_variants() -> None:
    target_fah, target_recall = 1.0, 0.9
    assert at.diagnose_regime(at.TuneMetrics(fah=1.4, recall=0.85), target_fah, target_recall) == "near_feasible"
    assert at.diagnose_regime(at.TuneMetrics(fah=1.6, recall=0.95), target_fah, target_recall) == "fah_dominated"
    assert at.diagnose_regime(at.TuneMetrics(fah=1.2, recall=0.7), target_fah, target_recall) == "recall_dominated"
    assert at.diagnose_regime(at.TuneMetrics(fah=2.0, recall=0.7), target_fah, target_recall) == "balanced"


def test_thompson_sampler_select_and_update() -> None:
    np.random.seed(1)
    s = at.ThompsonSampler(n_arms=7)
    first = s.select_arm("balanced")
    assert 0 <= first < 7
    s.last_arm = 6
    second = s.select_arm("fah_dominated")
    assert second != 6
    old_success = s.successes[first]
    s.update(first, True)
    assert s.successes[first] == old_success + 1


def test_stir_and_annealing_controllers() -> None:
    stir = at.StirController(thresholds=[1, 2, 3, 4, 5])
    assert stir.get_stir_level(0) == 0
    assert stir.get_stir_level(4) == 4
    assert stir.get_stir_level(50) == 5

    ann = at.AnnealingController(initial_temperature=1.0, cooling_rate=0.5, reheat_factor=2.0, reheat_after=2)
    better = at.TuneMetrics(fah=0.1, recall=0.95)
    worse = at.TuneMetrics(fah=2.0, recall=0.1)
    assert ann.should_accept(better, worse, target_fah=1.0, target_recall=0.9)
    temp_after_accept = ann.temperature
    assert math.isclose(temp_after_accept, 0.5)

    # Force repeated rejection path with deterministic random
    np.random.seed(123)
    ann.temperature = 1e-9
    assert not ann.should_accept(worse, better, target_fah=1.0, target_recall=0.9)
    assert not ann.should_accept(worse, better, target_fah=1.0, target_recall=0.9)
    assert ann.temperature >= 1e-9


def test_threshold_optimizer_core_paths(monkeypatch) -> None:
    opt = at.ThresholdOptimizer()
    y_true = np.array([0, 0, 1, 1, 1, 0], dtype=float)
    y_scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3], dtype=float)

    t, t_u8, m = opt.optimize(y_true, y_scores, ambient_duration_hours=1.0, target_fah=1.0, target_recall=0.5)
    assert 0.0 <= t <= 1.0
    assert 0 <= t_u8 <= 255
    assert isinstance(m, at.TuneMetrics)

    t2 = opt._optimize_threshold_cached(y_scores, y_true, target_fah=1.0, val_ambient_duration_hours=1.0)
    t3 = opt._optimize_threshold_cached(y_scores, y_true, target_fah=1.0, val_ambient_duration_hours=1.0)
    assert t2 == t3

    # Empty-path safety
    t_empty, t_empty_u8, m_empty = opt.optimize(np.array([]), np.array([]), 1.0, 1.0, 0.5)
    assert t_empty == 0.5
    assert t_empty_u8 == 128
    assert isinstance(m_empty, at.TuneMetrics)


def test_autotuner_init_and_utils(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "auto_tuning": {"target_fah": 0.3, "target_recall": 0.92, "output_dir": str(tmp_path / "out")},
        "auto_tuning_expert": {},
        "training": {},
        "hardware": {},
    }

    tuner = at.AutoTuner(checkpoint_path=str(tmp_path / "ckpt.weights.h5"), config=cfg)
    assert tuner.target_fah == 0.3
    assert tuner.target_recall == 0.92
    assert (tuner.output_dir / "checkpoints").exists()
    assert (tuner.output_dir / "logs").exists()

    class DummyModel:
        def __init__(self):
            self._w = [np.array([1.0, 2.0]), np.array([3.0])]
            self.trainable_weights = []
            self.saved_path = None

        def get_weights(self):
            return [x.copy() for x in self._w]

        def set_weights(self, w):
            self._w = [np.array(x) for x in w]

        def save_weights(self, path):
            self.saved_path = path

    model = DummyModel()
    blob = tuner._serialize_weights(model)
    model.set_weights([np.array([9.0, 9.0]), np.array([9.0])])
    tuner._deserialize_weights(model, blob)
    np.testing.assert_array_equal(model.get_weights()[0], np.array([1.0, 2.0]))

    ckpt_metrics = at.TuneMetrics(fah=0.2, recall=0.95, auc_pr=0.96, threshold=0.77, threshold_uint8=196)
    ckpt_path = Path(tuner._save_checkpoint(model, ckpt_metrics, iteration=1))
    sidecar = ckpt_path.with_suffix(".metadata.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["tuned_probability_cutoff"] == 0.77
    assert payload["tuned_probability_cutoff_uint8"] == 196


def test_confirmation_phase_uses_optimize_with_targets(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "auto_tuning": {
            "target_fah": 0.5,
            "target_recall": 0.88,
            "output_dir": str(tmp_path / "out"),
            "int8_shadow": False,
        },
        "auto_tuning_expert": {},
        "training": {},
        "hardware": {},
    }
    tuner = at.AutoTuner(checkpoint_path=str(tmp_path / "ckpt.weights.h5"), config=cfg)

    class DummyModel:
        pass

    candidate = at.CandidateState(
        id="c0",
        weights_bytes=b"w",
        optimizer_state_bytes=b"o",
        batchnorm_state={},
        temperature=1.0,
        threshold_float32=0.5,
        threshold_uint8=128,
        eval_results=at.TuneMetrics(fah=0.4, recall=0.9, auc_pr=0.9),
    )
    tuner.archive.archive = [candidate]

    called: dict[str, float] = {}

    def fake_optimize(y_true, y_scores, ambient_duration_hours, target_fah, target_recall):
        called["target_fah"] = float(target_fah)
        called["target_recall"] = float(target_recall)
        m = at.TuneMetrics(fah=0.4, recall=0.9, auc_pr=0.9, threshold=0.5, threshold_uint8=128)
        return 0.5, 128, m

    monkeypatch.setattr(tuner, "_deserialize_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner, "_restore_bn_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner, "_predict_scores", lambda *args, **kwargs: np.array([0.2, 0.9], dtype=float))
    monkeypatch.setattr(
        tuner.threshold_optimizer,
        "_compute_metrics_at_threshold",
        lambda *args, **kwargs: at.TuneMetrics(fah=0.4, recall=0.9, auc_pr=0.9, threshold=0.5, threshold_uint8=128),
    )
    monkeypatch.setattr(tuner.threshold_optimizer, "optimize", fake_optimize)

    confirm_data = (
        np.zeros((2, 3), dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.arange(2),
    )
    repr_data = (np.zeros((1, 3), dtype=np.float32),)

    best_confirmed, best_attempt_metrics = tuner._confirmation_phase(DummyModel(), confirm_data, repr_data, ambient_hours=1.0)

    assert best_confirmed is not None
    assert best_attempt_metrics is None
    assert called["target_fah"] == tuner.target_fah
    assert called["target_recall"] == tuner.target_recall
