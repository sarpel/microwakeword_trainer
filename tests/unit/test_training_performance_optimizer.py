"""High-value unit tests for training performance optimizer decisions."""

from __future__ import annotations

import builtins
import types

import pytest

from src.training import performance_optimizer as perf


class _FakeDataset:
    def __init__(self):
        self.train_called_with = None
        self.val_called_with = None

    def train_generator_factory(self, max_time_frames):
        self.train_called_with = max_time_frames

        def _factory():
            return "train_gen"

        return _factory

    def val_generator_factory(self, max_time_frames):
        self.val_called_with = max_time_frames

        def _factory():
            return "val_gen"

        return _factory


def test_create_datasets_uses_legacy_generators_when_tfdata_disabled():
    config = {"performance": {"use_tfdata": False, "mixed_precision": False}}
    optimizer = perf.PerformanceOptimizer(config)
    dataset = _FakeDataset()

    train_ds, val_ds = optimizer.create_datasets(dataset, max_time_frames=123)

    assert train_ds == "train_gen"
    assert val_ds == "val_gen"
    assert dataset.train_called_with == 123
    assert dataset.val_called_with == 123


def test_is_mixed_precision_available_false_without_gpu(monkeypatch):
    monkeypatch.setattr(perf.tf.config, "list_physical_devices", lambda _: [])

    assert perf.is_mixed_precision_available() is False


def test_is_mixed_precision_available_true_for_tensor_core_gpu(monkeypatch):
    fake_gpu = types.SimpleNamespace(name="GPU:0")
    monkeypatch.setattr(perf.tf.config, "list_physical_devices", lambda _: [fake_gpu])
    monkeypatch.setattr(
        perf.tf.config.experimental,
        "get_device_details",
        lambda _gpu: {"device_name": "NVIDIA GeForce RTX 3060 Ti"},
    )

    assert perf.is_mixed_precision_available() is True


def test_get_optimal_batch_size_without_gpu_uses_config_default(monkeypatch):
    monkeypatch.setattr(perf.tf.config, "list_physical_devices", lambda _: [])
    config = {"training": {"batch_size": 96}}

    assert perf.get_optimal_batch_size(config) == 96


@pytest.mark.parametrize(
    ("bytes_total", "expected_batch"),
    [
        (18_000_000_000, 256),
        (10_000_000_000, 128),
        (6_000_000_000, 64),
        (2_000_000_000, 32),
    ],
)
def test_get_optimal_batch_size_uses_memory_tiers_from_tf_fallback(monkeypatch, bytes_total, expected_batch):
    fake_gpu = types.SimpleNamespace(name="GPU:0")
    monkeypatch.setattr(perf.tf.config, "list_physical_devices", lambda _: [fake_gpu])

    # Force pynvml import path to fail so we exercise TF fallback branch deterministically.
    real_import = builtins.__import__

    def _import_fail(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("pynvml unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _import_fail)
    monkeypatch.setattr(
        perf.tf.config.experimental,
        "get_memory_info",
        lambda _name: {"current": bytes_total},
    )

    assert perf.get_optimal_batch_size({"training": {"batch_size": 384}}) == expected_batch
