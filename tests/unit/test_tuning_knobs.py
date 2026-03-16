"""Unit tests for src.tuning.knobs."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.tuning.knobs import (
    FocusedSampler,
    KnobCycle,
    LabelSmoothingKnob,
    LRKnob,
    SamplingMixKnob,
    TemperatureKnob,
    ThresholdKnob,
    WeightPerturbationKnob,
)


def test_knob_cycle_basic() -> None:
    cycle = KnobCycle(["lr", "threshold"])
    assert cycle.current() == "lr"
    cycle.advance()
    assert cycle.current() == "threshold"


def test_knob_cycle_wraps() -> None:
    cycle = KnobCycle(["lr", "threshold"])
    cycle.advance()
    cycle.advance()
    assert cycle.current() == "lr"


def test_knob_cycle_current() -> None:
    cycle = KnobCycle(["temperature", "sampling_mix"])
    assert cycle.current() == "temperature"


def test_knob_cycle_advance() -> None:
    cycle = KnobCycle(["temperature", "sampling_mix"])
    cycle.advance()
    assert cycle.current() == "sampling_mix"


def test_knob_cycle_position() -> None:
    cycle = KnobCycle(["lr", "threshold", "temperature"])
    assert isinstance(cycle.position(), int)
    assert cycle.position() == 0
    cycle.advance()
    assert cycle.position() == 1


def test_lr_knob_exists() -> None:
    knob = LRKnob()
    assert knob.name == "lr"


def test_lr_knob_apply() -> None:
    model = MagicMock()
    model.optimizer = MagicMock()
    model.optimizer.learning_rate = MagicMock()
    model.optimizer.learning_rate.assign = MagicMock()

    candidate = MagicMock()
    candidate.knob_history = []

    config = MagicMock()
    expert = MagicMock()
    expert.lr_range = (1e-7, 1e-4)
    config.auto_tuning_expert = expert

    LRKnob().apply(model, candidate, config)

    model.optimizer.learning_rate.assign.assert_called_once()
    assert candidate.knob_history


def test_threshold_knob_exists() -> None:
    knob = ThresholdKnob()
    assert knob.name == "threshold"


def test_temperature_knob_exists() -> None:
    knob = TemperatureKnob()
    assert knob.name == "temperature"


def test_sampling_mix_knob_exists() -> None:
    knob = SamplingMixKnob()
    assert knob.name == "sampling_mix"


def test_weight_perturbation_knob_exists() -> None:
    knob = WeightPerturbationKnob()
    assert knob.name == "weight_perturbation"


def test_weight_perturbation_no_compile() -> None:
    model = MagicMock()
    model.compile = MagicMock()
    model.get_weights.return_value = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    weight_1 = MagicMock()
    weight_1.trainable = True
    weight_2 = MagicMock()
    weight_2.trainable = False
    model.weights = [weight_1, weight_2]
    model.set_weights = MagicMock()

    candidate = MagicMock()
    candidate.knob_history = []
    config = MagicMock()
    config.auto_tuning_expert = MagicMock()
    config.auto_tuning_expert.weight_perturbation_scale = 0.01

    WeightPerturbationKnob().apply(model, candidate, config)

    model.compile.assert_not_called()


def test_label_smoothing_knob_exists() -> None:
    knob = LabelSmoothingKnob()
    assert knob.name == "label_smoothing"


def test_label_smoothing_no_compile() -> None:
    model = MagicMock()
    model.compile = MagicMock()
    model._label_smoothing_var = MagicMock()
    model._label_smoothing_var.assign = MagicMock()

    candidate = MagicMock()
    candidate.knob_history = []
    config = MagicMock()
    config.auto_tuning_expert = MagicMock()
    config.auto_tuning_expert.label_smoothing_range = (0.0, 0.15)

    LabelSmoothingKnob().apply(model, candidate, config)

    model.compile.assert_not_called()
    model._label_smoothing_var.assign.assert_called_once()


def test_focused_sampler_exists() -> None:
    sampler = FocusedSampler()
    assert sampler is not None


def test_focused_sampler_build_batch() -> None:
    sampler = FocusedSampler()
    assert hasattr(sampler, "build_batch")
    assert sampler.build_batch() is None


def test_all_knobs_have_name() -> None:
    knobs = [
        LRKnob(),
        ThresholdKnob(),
        TemperatureKnob(),
        SamplingMixKnob(),
        WeightPerturbationKnob(),
        LabelSmoothingKnob(),
    ]
    for knob in knobs:
        assert isinstance(knob.name, str)
        assert knob.name


def test_all_knobs_have_describe() -> None:
    knobs = [
        LRKnob(),
        ThresholdKnob(),
        TemperatureKnob(),
        SamplingMixKnob(),
        WeightPerturbationKnob(),
        LabelSmoothingKnob(),
    ]
    for knob in knobs:
        assert callable(knob.describe)
        assert isinstance(knob.describe(), str)


def test_all_knobs_have_apply() -> None:
    knobs = [
        LRKnob(),
        ThresholdKnob(),
        TemperatureKnob(),
        SamplingMixKnob(),
        WeightPerturbationKnob(),
        LabelSmoothingKnob(),
    ]
    for knob in knobs:
        assert callable(knob.apply)
