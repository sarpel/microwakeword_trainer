"""Six knob implementations for micro-step cyclic coordinate descent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.tuning.metrics import ThresholdOptimizer, apply_temperature, fit_temperature


class Knob(ABC):
    """Base class for all tuning knobs."""

    name: str

    @abstractmethod
    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        """Apply knob mutation to model/candidate state."""

    def describe(self) -> str:
        return f"Knob({self.name})"


class KnobCycle:
    """Cyclic iterator over knob names."""

    def __init__(self, knob_names: list[str]):
        if not knob_names:
            raise ValueError("knob_names must not be empty")
        self._knobs = list(knob_names)
        self._pos = 0

    def current(self) -> str:
        return self._knobs[self._pos % len(self._knobs)]

    def advance(self) -> None:
        self._pos = (self._pos + 1) % len(self._knobs)

    def position(self) -> int:
        return self._pos


class LRKnob(Knob):
    name = "lr"

    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        expert_config = getattr(config, "auto_tuning_expert", config)
        lr_min, lr_max = getattr(expert_config, "lr_range", (1e-7, 1e-4))
        lr = float(np.exp(np.random.uniform(np.log(lr_min), np.log(lr_max))))

        if hasattr(model, "optimizer") and model.optimizer is not None:
            opt = model.optimizer
            if hasattr(opt, "learning_rate") and hasattr(opt.learning_rate, "assign"):
                opt.learning_rate.assign(lr)
            elif hasattr(opt, "lr") and hasattr(opt.lr, "assign"):
                opt.lr.assign(lr)

        if hasattr(candidate, "knob_history"):
            candidate.knob_history.append(f"lr={lr:.2e}")


class ThresholdKnob(Knob):
    name = "threshold"

    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        optimizer = ThresholdOptimizer()
        candidate.threshold_optimizer = optimizer
        if hasattr(candidate, "knob_history"):
            candidate.knob_history.append("threshold=optimized")


class TemperatureKnob(Knob):
    name = "temperature"

    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        current_temp = float(getattr(candidate, "temperature", 1.0))

        probs = getattr(candidate, "calibration_probs", None)
        labels = getattr(candidate, "calibration_labels", None)
        if probs is not None and labels is not None and len(probs) == len(labels) and len(probs) > 0:
            fitted = float(fit_temperature(np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)))
            current_temp = fitted
            if hasattr(candidate, "temperature"):
                candidate.temperature = fitted

            scores = getattr(candidate, "scores", None)
            if scores is not None:
                candidate.scores = apply_temperature(np.asarray(scores, dtype=np.float64), fitted)

        if hasattr(candidate, "knob_history"):
            candidate.knob_history.append(f"temperature={current_temp:.3f}")


class SamplingMixKnob(Knob):
    name = "sampling_mix"

    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        sampler = getattr(candidate, "sampler", None)
        if sampler is not None and hasattr(sampler, "advance_arm"):
            sampler.advance_arm()
        if hasattr(candidate, "knob_history"):
            candidate.knob_history.append("sampling_mix=rotated")


class WeightPerturbationKnob(Knob):
    name = "weight_perturbation"

    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        expert_config = getattr(config, "auto_tuning_expert", config)
        scale = float(getattr(expert_config, "weight_perturbation_scale", 0.01))

        all_weights = [np.asarray(w) for w in model.get_weights()]
        model_weights = list(getattr(model, "weights", []) or [])

        if model_weights and len(model_weights) == len(all_weights):
            for i, w_var in enumerate(model_weights):
                if bool(getattr(w_var, "trainable", False)):
                    noise = np.random.normal(0.0, scale, size=all_weights[i].shape)
                    all_weights[i] = all_weights[i] + noise
        else:
            for i in range(len(all_weights)):
                noise = np.random.normal(0.0, scale, size=all_weights[i].shape)
                all_weights[i] = all_weights[i] + noise

        model.set_weights(all_weights)
        if hasattr(candidate, "knob_history"):
            candidate.knob_history.append(f"weight_perturbation={scale}")


class LabelSmoothingKnob(Knob):
    name = "label_smoothing"

    def apply(self, model: Any, candidate: Any, config: Any) -> None:
        expert_config = getattr(config, "auto_tuning_expert", config)
        ls_min, ls_max = getattr(expert_config, "label_smoothing_range", (0.0, 0.15))
        smoothing = float(np.random.uniform(ls_min, ls_max))

        if hasattr(model, "_label_smoothing_var") and hasattr(model._label_smoothing_var, "assign"):
            model._label_smoothing_var.assign(smoothing)

        if hasattr(candidate, "knob_history"):
            candidate.knob_history.append(f"label_smoothing={smoothing:.4f}")


class FocusedSampler:
    """Adapted focused sampler interface for arm-based batch construction."""

    def __init__(self, pos_dataset: Any = None, neg_dataset: Any = None, hard_neg_dataset: Any = None, config: Any = None):
        self.pos_dataset = pos_dataset
        self.neg_dataset = neg_dataset
        self.hard_neg_dataset = hard_neg_dataset
        self.config = config
        self._arm = 0

    def build_batch(self, arm: int | None = None) -> Any:
        if arm is None:
            arm = self._arm
        _ = arm
        return None

    def advance_arm(self) -> None:
        self._arm = (self._arm + 1) % 7
