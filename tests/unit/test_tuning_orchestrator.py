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


def test_create_model_builds_before_loading_weights(monkeypatch) -> None:
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), _expert_cfg())
    call_sequence: list[str] = []

    class BuildGuardModel:
        def __call__(self, inputs, training=False):
            _ = (inputs, training)
            call_sequence.append("call")
            return np.zeros((1, 1), dtype=np.float32)

        def load_weights(self, path):
            _ = path
            call_sequence.append("load_weights")

    model = BuildGuardModel()
    monkeypatch.setattr("src.tuning.orchestrator.build_model", lambda **kwargs: model)

    built_model = tuner._create_model()

    assert built_model is model
    assert call_sequence == ["call", "load_weights"]


def test_load_data_reshapes_flattened_features(monkeypatch) -> None:
    cfg = _base_config()
    tuner = MicroAutoTuner(CHECKPOINT_PATH, cfg, _expert_cfg())

    class FlatFeatureDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx: int):
            mel_bins = int(cfg["hardware"]["mel_bins"])
            total_frames = int(cfg["hardware"]["clip_duration_ms"] / cfg["hardware"]["window_step_ms"])
            if idx == 0:
                # Flattened [time, mel] features (50 frames).
                return np.arange(50 * mel_bins, dtype=np.float32), 1
            # Already 2D features.
            return np.ones((total_frames, mel_bins), dtype=np.float32), 0

    monkeypatch.setattr("src.tuning.orchestrator.WakeWordDataset", lambda config, split: FlatFeatureDataset(), raising=False)
    # Patch local import target used inside _load_data()
    import src.data.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "WakeWordDataset", lambda config, split: FlatFeatureDataset())

    data = tuner._load_data()

    assert data["features"].shape == (2, 100, 40)
    assert data["labels"].tolist() == [1.0, 0.0]
    assert data["weights"].tolist() == [1.0, 20.0]
    # First sample had only 50 frames -> padded to 100 frames.
    assert np.all(data["features"][0, 50:] == 0.0)


def test_load_data_rejects_non_reshapable_flattened_features(monkeypatch) -> None:
    cfg = _base_config()
    tuner = MicroAutoTuner(CHECKPOINT_PATH, cfg, _expert_cfg())

    class BadFlatFeatureDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = idx
            # 3999 is not divisible by mel_bins=40 -> should raise clear ValueError.
            return np.zeros((3999,), dtype=np.float32), 0

    import src.data.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "WakeWordDataset", lambda config, split: BadFlatFeatureDataset())

    try:
        tuner._load_data()
    except ValueError as exc:
        assert "cannot be reshaped" in str(exc)
        return
    raise AssertionError("Expected ValueError for non-reshapable flattened features")


def test_ensure_tf_seed_for_determinism(monkeypatch) -> None:
    cfg = _base_config()
    cfg["training"] = {"random_seed": 777}
    tuner = MicroAutoTuner(CHECKPOINT_PATH, cfg, _expert_cfg())

    calls: list[int] = []

    class _DummyRandom:
        @staticmethod
        def set_seed(seed: int) -> None:
            calls.append(seed)

    class _DummyTF:
        random = _DummyRandom()

    import sys

    monkeypatch.setenv("TF_DETERMINISTIC_OPS", "1")
    monkeypatch.setitem(sys.modules, "tensorflow", _DummyTF)

    tuner._ensure_tf_seed_for_determinism()
    tuner._ensure_tf_seed_for_determinism()

    assert calls == [777]


def test_run_burst_uses_uniform_sampling_not_shuffle(monkeypatch) -> None:
    cfg = _base_config()
    tuner = MicroAutoTuner(CHECKPOINT_PATH, cfg, _expert_cfg())

    class _LR:
        @staticmethod
        def assign(value):
            _ = value

    class _Optimizer:
        learning_rate = _LR()

        @staticmethod
        def apply_gradients(grads_and_vars):
            _ = list(grads_and_vars)

    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(100, 40)),
            tf.keras.layers.Dense(1),
        ]
    )
    _ = model(tf.zeros((1, 100, 40), dtype=tf.float32), training=False)

    uniform_calls = {"count": 0}
    shuffle_calls = {"count": 0}
    original_uniform = tf.random.uniform
    original_shuffle = tf.random.shuffle

    def _wrapped_uniform(*args, **kwargs):
        uniform_calls["count"] += 1
        return original_uniform(*args, **kwargs)

    def _wrapped_shuffle(*args, **kwargs):
        shuffle_calls["count"] += 1
        return original_shuffle(*args, **kwargs)

    monkeypatch.setattr(tf.random, "uniform", _wrapped_uniform)
    monkeypatch.setattr(tf.random, "shuffle", _wrapped_shuffle)

    features = np.zeros((16, 100, 40), dtype=np.float32)
    labels = np.zeros((16,), dtype=np.float32)
    weights = np.ones((16,), dtype=np.float32)

    tuner._run_burst(model, _Optimizer(), (features, labels, weights), n_steps=3)

    assert uniform_calls["count"] == 3
    assert shuffle_calls["count"] == 0


def test_evaluate_candidate_uses_model_probabilities_without_extra_sigmoid(monkeypatch) -> None:
    tuner = MicroAutoTuner(CHECKPOINT_PATH, _base_config(), _expert_cfg())

    class FixedProbModel:
        def __call__(self, batch, training=False):
            _ = training
            # Return constant probability 0.8 per sample.
            batch_size = int(batch.shape[0])
            import tensorflow as tf

            return tf.fill((batch_size, 1), tf.constant(0.8, dtype=tf.float32))

    captured = {}

    class StubThresholdOptimizer:
        def optimize(self, **kwargs):
            captured["y_scores"] = np.asarray(kwargs["y_scores"])
            from src.tuning.metrics import TuneMetrics

            return 0.5, 128, TuneMetrics(fah=1.0, recall=0.5, auc_pr=0.5)

    monkeypatch.setattr("src.tuning.orchestrator.ThresholdOptimizer", StubThresholdOptimizer)

    features = np.zeros((4, 100, 40), dtype=np.float32)
    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    weights = np.ones((4,), dtype=np.float32)

    tuner._evaluate_candidate(FixedProbModel(), (features, labels, weights))

    assert np.allclose(captured["y_scores"], 0.8)


def test_sampling_mix_knob_rotates_arm_and_logs_history() -> None:
    from src.tuning.knobs import SamplingMixKnob

    class _Candidate:
        def __init__(self):
            self._sampling_mix_arm = 0
            self.knob_history = []

    candidate = _Candidate()
    knob = SamplingMixKnob()

    knob.apply(model=None, candidate=candidate, config={})
    knob.apply(model=None, candidate=candidate, config={})

    assert candidate._sampling_mix_arm == 2
    assert candidate.knob_history == ["sampling_mix=arm1", "sampling_mix=arm2"]
