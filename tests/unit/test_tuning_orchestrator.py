"""Unit tests for src.tuning.orchestrator (RED+GREEN TDD target)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.tuning.orchestrator import MicroAutoTuner


def _base_config() -> dict:
    return {
        "hardware": {"clip_duration_ms": 1000, "window_step_ms": 10, "mel_bins": 40},
        "model": {},
        "auto_tuning": {"output_dir": "./tuning_output"},
    }


CHECKPOINT_PATH = "checkpoints/mock.weights.h5"


def _expert_cfg(**overrides):
    data = {
        "population_size": 2,
        "micro_burst_steps": 1,
        "knob_cycle": ["lr", "threshold", "temperature"],
        "exploit_explore_interval": 2,
        "weight_perturbation_scale": 0.01,
        "label_smoothing_range": (0.0, 0.15),
        "lr_range": (1e-7, 1e-4),
        "hypervolume_reference": (10.0, 0.0),
        "pareto_archive_size": 32,
        "max_iterations": 4,
        "max_no_improve": 10,
        "output_dir": "./tuning_output",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.get_weights.return_value = [np.array([1.0], dtype=np.float32)]
    model.set_weights = MagicMock()
    model.load_weights = MagicMock()
    model._flatten_layers.return_value = []
    model.compile = MagicMock()
    return model


def _partition_payload(search_train_size: int = 6, search_eval_size: int = 2):
    features = np.zeros((8, 3, 40), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    weights = np.ones(8, dtype=np.float32)
    idx = np.arange(8)

    search_train = (
        features[:search_train_size],
        labels[:search_train_size],
        weights[:search_train_size],
        idx[:search_train_size],
    )
    search_eval = (
        features[search_train_size : search_train_size + search_eval_size],
        labels[search_train_size : search_train_size + search_eval_size],
        weights[search_train_size : search_train_size + search_eval_size],
        idx[search_train_size : search_train_size + search_eval_size],
    )
    cal = (features[:1], labels[:1], weights[:1], idx[:1])
    confirm = (features[1:2], labels[1:2], weights[1:2], idx[1:2])
    representative = (features[2:3], labels[2:3], weights[2:3], idx[2:3])

    return {
        "cal": cal,
        "search_train": search_train,
        "search_eval": search_eval,
        "confirm": confirm,
        "representative": representative,
        "fold_indices": [np.array([0], dtype=np.int64)],
    }


def test_micro_auto_tuner_init() -> None:
    tuner = MicroAutoTuner(
        checkpoint_path=CHECKPOINT_PATH,
        config=_base_config(),
        auto_tuning_config=_expert_cfg(),
    )
    assert tuner is not None
    assert hasattr(tuner, "tune")


def test_micro_auto_tuner_has_tune_method() -> None:
    tuner = MicroAutoTuner(
        checkpoint_path=CHECKPOINT_PATH,
        config=_base_config(),
        auto_tuning_config=_expert_cfg(),
    )
    assert callable(tuner.tune)


def test_dry_run_mode(monkeypatch) -> None:
    cfg = _expert_cfg(dry_run=True)
    tuner = MicroAutoTuner(
        checkpoint_path=CHECKPOINT_PATH,
        config=_base_config(),
        auto_tuning_config=cfg,
    )
    mocked_build = MagicMock()
    monkeypatch.setattr("src.tuning.orchestrator.build_model", mocked_build)

    result = tuner.tune()

    assert result == {}
    mocked_build.assert_not_called()


def test_knob_cycle_applied_per_burst(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=3, knob_cycle=["lr", "threshold", "temperature"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)
    model = _mock_model()
    seen_knobs: list[str] = []

    class _DummyKnob:
        def __init__(self, name: str):
            self._name = name

        def apply(self, model, candidate, config):
            _ = (model, candidate, config)
            seen_knobs.append(self._name)

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: _partition_payload())
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr(tuner, "_evaluate_candidate", lambda *args, **kwargs: SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128))
    monkeypatch.setattr(tuner, "_make_knob", lambda name: _DummyKnob(name))
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    tuner.tune()

    assert seen_knobs == ["lr", "threshold", "temperature"]


def test_population_exploit_explore_triggered(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=4, exploit_explore_interval=2, knob_cycle=["threshold"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)
    model = _mock_model()

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: _partition_payload())
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr(tuner, "_evaluate_candidate", lambda *args, **kwargs: SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128))
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    exploit_calls = {"count": 0}

    def _exploit(self, model):
        _ = model
        exploit_calls["count"] += 1

    monkeypatch.setattr("src.tuning.orchestrator.Population.exploit_explore", _exploit)

    tuner.tune()

    assert exploit_calls["count"] == 2


def test_hypervolume_tracked(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=3, knob_cycle=["threshold"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)
    model = _mock_model()

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: _partition_payload())
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr(tuner, "_evaluate_candidate", lambda *args, **kwargs: SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128))
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    result = tuner.tune()

    assert "hypervolume_history" in result
    assert isinstance(result["hypervolume_history"], list)


def test_search_eval_not_search_train_for_evaluation(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=1, knob_cycle=["threshold"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)
    model = _mock_model()
    partition = _partition_payload(search_train_size=6, search_eval_size=2)

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: partition)
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    def _eval(model, eval_partition):
        assert eval_partition is partition["search_eval"]
        assert len(eval_partition[1]) == 2
        return SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128)

    monkeypatch.setattr(tuner, "_evaluate_candidate", _eval)

    tuner.tune()


def test_no_model_compile_called(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=2, knob_cycle=["threshold"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)
    model = _mock_model()

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: _partition_payload())
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr(tuner, "_evaluate_candidate", lambda *args, **kwargs: SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128))
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    tuner.tune()

    model.compile.assert_not_called()


def test_no_trainable_weights_used(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=1, knob_cycle=["threshold"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)

    class GuardModel:
        def __init__(self):
            self.compile = MagicMock()
            self._w = [np.array([1.0], dtype=np.float32)]

        @property
        def trainable_weights(self):
            raise AssertionError("trainable_weights must not be accessed")

        def get_weights(self):
            return [w.copy() for w in self._w]

        def set_weights(self, w):
            self._w = [np.array(x) for x in w]

        def load_weights(self, _):
            return None

        def _flatten_layers(self, include_self=False, recursive=True):
            _ = (include_self, recursive)
            return []

    model = GuardModel()

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: _partition_payload())
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr(tuner, "_evaluate_candidate", lambda *args, **kwargs: SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128))
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    tuner.tune()


def test_convergence_detection(monkeypatch) -> None:
    cfg = _expert_cfg(population_size=1, max_iterations=20, max_no_improve=1, knob_cycle=["threshold"])
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), cfg)
    model = _mock_model()

    monkeypatch.setattr(tuner, "_create_model", lambda: model)
    monkeypatch.setattr("src.tuning.orchestrator.partition_data", lambda *args, **kwargs: _partition_payload())
    monkeypatch.setattr(tuner, "_create_optimizer", lambda: MagicMock())
    monkeypatch.setattr(tuner, "_run_burst", lambda *args, **kwargs: {"steps": 1, "final_loss": 0.0, "mean_loss": 0.0})
    monkeypatch.setattr(tuner, "_evaluate_candidate", lambda *args, **kwargs: SimpleNamespace(fah=1.0, recall=0.5, auc_pr=0.5, threshold=0.5, threshold_uint8=128))
    monkeypatch.setattr("src.tuning.orchestrator.compute_hypervolume", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("src.tuning.orchestrator.save_artifacts", lambda **kwargs: None)

    result = tuner.tune()

    assert result["iterations_completed"] < 20
    assert result["iterations_completed"] == 2
