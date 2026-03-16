"""Unit tests for src.tuning.population."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from src.tuning.metrics import TuneMetrics
from src.tuning.population import Candidate, Population, partition_data


def test_candidate_construction() -> None:
    candidate = Candidate(id="c0")
    assert candidate.id == "c0"
    assert candidate.weights_bytes is None
    assert candidate.optimizer_state_bytes is None
    assert candidate.batchnorm_state is None
    assert candidate.temperature == 1.0
    assert candidate.metrics is None
    assert candidate.knob_history == []


def test_candidate_save_state() -> None:
    model = MagicMock()
    model.get_weights.return_value = [np.array([1.0, 2.0], dtype=np.float32)]

    candidate = Candidate(id="c0")
    candidate.save_state(model)

    model.get_weights.assert_called_once()
    assert candidate.weights_bytes is not None


def test_candidate_restore_state() -> None:
    model = MagicMock()
    original = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0], dtype=np.float32)]
    model.get_weights.return_value = original

    candidate = Candidate(id="c0")
    candidate.save_state(model)

    mutate = [np.array([9.0, 9.0], dtype=np.float32), np.array([9.0], dtype=np.float32)]
    model.set_weights(mutate)

    candidate.restore_state(model)

    restored = model.set_weights.call_args_list[-1][0][0]
    np.testing.assert_array_equal(restored[0], original[0])
    np.testing.assert_array_equal(restored[1], original[1])


def test_candidate_uses_get_weights() -> None:
    class ForbiddenModel:
        def __init__(self) -> None:
            self.get_weights = MagicMock(return_value=[np.array([1.0], dtype=np.float32)])

        @property
        def trainable_weights(self):
            raise AssertionError("trainable_weights must never be accessed")

    model = ForbiddenModel()
    candidate = Candidate(id="c0")
    candidate.save_state(model)

    model.get_weights.assert_called_once()
    assert candidate.weights_bytes is not None


def test_population_init() -> None:
    model = MagicMock()
    model.get_weights.return_value = [np.array([1.0], dtype=np.float32)]

    population = Population(model=model, size=4)
    assert len(population.candidates) == 4
    assert all(c.weights_bytes is not None for c in population.candidates)

    cands = [Candidate(id="a"), Candidate(id="b")]
    population2 = Population(candidates=cands)
    assert len(population2.candidates) == 2
    assert population2.candidates[0].id == "a"


def test_population_get_best() -> None:
    c1 = Candidate(id="c1", metrics=TuneMetrics(fah=0.4, recall=0.90, auc_pr=0.80))
    c2 = Candidate(id="c2", metrics=TuneMetrics(fah=0.2, recall=0.80, auc_pr=0.70))
    c3 = Candidate(id="c3", metrics=TuneMetrics(fah=0.2, recall=0.92, auc_pr=0.85))
    population = Population(candidates=[c1, c2, c3])

    best = population.get_best()
    assert best.id == "c3"


def test_population_get_worst() -> None:
    c1 = Candidate(id="c1", metrics=TuneMetrics(fah=0.1, recall=0.95, auc_pr=0.90))
    c2 = Candidate(id="c2", metrics=TuneMetrics(fah=1.2, recall=0.70, auc_pr=0.50))
    c3 = Candidate(id="c3", metrics=TuneMetrics(fah=0.8, recall=0.60, auc_pr=0.40))
    population = Population(candidates=[c1, c2, c3])

    worst = population.get_worst()
    assert worst.id == "c2"


@dataclass
class _FakeWeight:
    name: str
    trainable: bool


def test_population_exploit_explore() -> None:
    model = MagicMock()
    initial = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([2.0, 2.0], dtype=np.float32),
        np.array([3.0, 3.0], dtype=np.float32),
        np.array([4.0, 4.0], dtype=np.float32),
    ]
    model.get_weights.return_value = [w.copy() for w in initial]
    model.weights = [
        _FakeWeight("dense/kernel:0", True),
        _FakeWeight("dense/bias:0", True),
        _FakeWeight("bn/moving_mean:0", False),
        _FakeWeight("bn/moving_variance:0", False),
    ]
    model.trainable_variables = [model.weights[0], model.weights[1]]

    best = Candidate(id="best", metrics=TuneMetrics(fah=0.1, recall=0.95, auc_pr=0.9))
    best.save_state(model)
    worst = Candidate(id="worst", metrics=TuneMetrics(fah=1.5, recall=0.50, auc_pr=0.2))
    population = Population(candidates=[best, worst])

    population.exploit_explore(model, perturbation_scale=0.01)

    assert worst.weights_bytes is not None
    perturbed = model.set_weights.call_args[0][0]
    assert not np.array_equal(perturbed[0], initial[0])
    assert not np.array_equal(perturbed[1], initial[1])


def test_exploit_explore_perturbation_applies_only_trainable() -> None:
    model = MagicMock()
    baseline = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([2.0, 2.0], dtype=np.float32),
        np.array([3.0, 3.0], dtype=np.float32),  # moving_mean
        np.array([4.0, 4.0], dtype=np.float32),  # moving_variance
    ]
    model.get_weights.return_value = [w.copy() for w in baseline]
    model.weights = [
        _FakeWeight("dense/kernel:0", True),
        _FakeWeight("dense/bias:0", True),
        _FakeWeight("bn/moving_mean:0", False),
        _FakeWeight("bn/moving_variance:0", False),
    ]
    model.trainable_variables = [model.weights[0], model.weights[1]]

    best = Candidate(id="best", metrics=TuneMetrics(fah=0.1, recall=0.95, auc_pr=0.95))
    best.save_state(model)
    worst = Candidate(id="worst", metrics=TuneMetrics(fah=2.0, recall=0.2, auc_pr=0.2))
    population = Population(candidates=[best, worst])

    population.exploit_explore(model, perturbation_scale=0.02)
    out_weights = model.set_weights.call_args[0][0]

    assert not np.array_equal(out_weights[0], baseline[0])
    assert not np.array_equal(out_weights[1], baseline[1])
    np.testing.assert_array_equal(out_weights[2], baseline[2])
    np.testing.assert_array_equal(out_weights[3], baseline[3])


def _make_dataset(n: int = 120) -> dict:
    features = np.random.RandomState(0).randn(n, 3).astype(np.float32)
    labels = np.random.RandomState(1).randint(0, 2, size=n).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)
    return {"features": features, "labels": labels, "weights": weights}


def test_partition_data_basic() -> None:
    dataset = _make_dataset(200)
    config = {
        "auto_tuning": {"confirmation_fraction": 0.4, "search_eval_fraction": 0.3, "cv_folds": 3},
        "training": {"split_seed": 42},
    }

    part = partition_data(dataset, config)
    assert {"cal", "search_train", "search_eval", "confirm", "representative", "fold_indices"}.issubset(part.keys())


def test_partition_data_search_eval_fraction() -> None:
    dataset = _make_dataset(1000)
    config = {
        "auto_tuning": {"confirmation_fraction": 0.4, "search_eval_fraction": 0.3, "cv_folds": 3},
        "training": {"split_seed": 42},
    }

    part = partition_data(dataset, config)
    n_train = len(part["search_train"][1])
    n_eval = len(part["search_eval"][1])
    ratio = n_eval / (n_train + n_eval)

    assert abs(ratio - 0.30) <= 0.15


def test_partition_data_group_aware() -> None:
    n_groups = 30
    per_group = 6
    n = n_groups * per_group
    dataset = _make_dataset(n)
    speaker_ids = np.array([f"spk_{i}" for i in range(n_groups) for _ in range(per_group)], dtype=object)
    dataset["speaker_id"] = speaker_ids

    config = {
        "auto_tuning": {"confirmation_fraction": 0.35, "search_eval_fraction": 0.3, "cv_folds": 3},
        "training": {"split_seed": 42},
    }

    part = partition_data(dataset, config)
    train_idx = part["search_train"][3]
    eval_idx = part["search_eval"][3]
    train_speakers = set(speaker_ids[train_idx].tolist())
    eval_speakers = set(speaker_ids[eval_idx].tolist())
    assert train_speakers.isdisjoint(eval_speakers)


def test_partition_data_no_overlap() -> None:
    dataset = _make_dataset(240)
    config = {
        "auto_tuning": {"confirmation_fraction": 0.4, "search_eval_fraction": 0.3, "cv_folds": 3},
        "training": {"split_seed": 42},
    }

    part = partition_data(dataset, config)
    train_idx = set(part["search_train"][3].tolist())
    eval_idx = set(part["search_eval"][3].tolist())

    assert train_idx.isdisjoint(eval_idx)
