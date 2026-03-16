"""Targeted branch tests for src.utils.performance TensorFlow/GPU helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils import performance


def _install_fake_tensorflow(monkeypatch, tf_obj):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            return tf_obj
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_get_system_info_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        performance.psutil,
        "cpu_count",
        lambda logical=False: 16 if logical else 8,
    )
    monkeypatch.setattr(
        performance.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3, available=20 * 1024**3),
    )
    monkeypatch.setattr(
        performance.GPUtil,
        "getGPUs",
        lambda: (_ for _ in ()).throw(RuntimeError("gpu fail")),
    )
    info = performance.get_system_info()
    assert info["cpu_count"] == 8
    assert "gpu" not in info


def test_check_gpu_available_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        performance.GPUtil,
        "getGPUs",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert performance.check_gpu_available() is False


def test_check_gpu_and_cupy_available_import_and_runtime_paths(
    monkeypatch,
) -> None:
    monkeypatch.setattr(performance, "check_gpu_available", lambda: True)

    import builtins

    real_import = builtins.__import__

    def fake_import_missing(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("missing cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_missing)
    ok, msg = performance.check_gpu_and_cupy_available()
    assert ok is False and "CuPy not found" in msg

    def fake_import_broken(name, *args, **kwargs):
        if name == "cupy":
            raise RuntimeError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_broken)
    ok2, msg2 = performance.check_gpu_and_cupy_available()
    assert ok2 is False and "Error initializing" in msg2


def test_configure_tensorflow_gpu_paths(monkeypatch) -> None:
    class Exp:
        def __init__(self, gpus):
            self._gpus = gpus
            self.mem_growth_calls = []
            self.virtual_calls = []

        def list_physical_devices(self, kind):
            return self._gpus if kind == "GPU" else []

        def set_memory_growth(self, gpu, enabled):
            self.mem_growth_calls.append((gpu, enabled))

        class VirtualDeviceConfiguration:
            def __init__(self, memory_limit):
                self.memory_limit = memory_limit

        def set_virtual_device_configuration(self, gpu, configs):
            self.virtual_calls.append((gpu, configs))

    class DummyTF:
        def __init__(self, gpus):
            self.config = SimpleNamespace(experimental=Exp(gpus))

    no_gpu_tf = DummyTF([])
    _install_fake_tensorflow(monkeypatch, no_gpu_tf)
    with pytest.raises(RuntimeError):
        performance.configure_tensorflow_gpu()

    with_gpu = DummyTF(["GPU0", "GPU1"])
    _install_fake_tensorflow(monkeypatch, with_gpu)
    with pytest.raises(ValueError):
        performance.configure_tensorflow_gpu(memory_growth=True, memory_limit_mb=256)

    assert performance.configure_tensorflow_gpu(memory_growth=True, device_id=99) is True
    assert with_gpu.config.experimental.mem_growth_calls

    assert performance.configure_tensorflow_gpu(memory_growth=False, memory_limit_mb=512, device_id=1) is True
    assert with_gpu.config.experimental.virtual_calls


def test_configure_tensorflow_gpu_runtime_error_branch(monkeypatch) -> None:
    class Exp:
        def list_physical_devices(self, kind):
            return ["GPU0"] if kind == "GPU" else []

        def set_memory_growth(self, gpu, enabled):
            raise RuntimeError("cannot set")

    class DummyTF:
        config = SimpleNamespace(experimental=Exp())

    _install_fake_tensorflow(monkeypatch, DummyTF)
    assert performance.configure_tensorflow_gpu(memory_growth=True) is False


def test_configure_mixed_precision_paths(monkeypatch) -> None:
    calls = []

    class MP:
        @staticmethod
        def Policy(name):
            return f"policy:{name}"

        @staticmethod
        def set_global_policy(pol):
            calls.append(pol)

    class DummyTF:
        keras = SimpleNamespace(mixed_precision=MP)

    _install_fake_tensorflow(monkeypatch, DummyTF)
    performance.configure_mixed_precision(enabled=True)
    performance.configure_mixed_precision(enabled=False)
    assert calls == ["policy:mixed_float16", "policy:float32"]


def test_set_threading_config_normal_and_initialized(monkeypatch) -> None:
    class Threading:
        def __init__(self):
            self.inter = 0
            self.intra = 0
            self.raise_initialized = False

        def set_inter_op_parallelism_threads(self, n):
            if self.raise_initialized:
                raise RuntimeError("cannot be modified after initialization")
            self.inter = n

        def set_intra_op_parallelism_threads(self, n):
            if self.raise_initialized:
                raise RuntimeError("cannot be modified after initialization")
            self.intra = n

        def get_inter_op_parallelism_threads(self):
            return self.inter

        def get_intra_op_parallelism_threads(self):
            return self.intra

    th = Threading()

    class DummyTF:
        config = SimpleNamespace(threading=th)

    _install_fake_tensorflow(monkeypatch, DummyTF)
    cfg = performance.set_threading_config(num_threads=3)
    assert cfg["inter_op_parallelism"] == 3
    assert cfg["intra_op_parallelism"] == 3

    th.raise_initialized = True
    cfg2 = performance.set_threading_config(inter_op_parallelism=1, intra_op_parallelism=2)
    assert "inter_op_parallelism" in cfg2


def test_set_threading_config_unexpected_runtime_error(monkeypatch) -> None:
    class Threading:
        def set_inter_op_parallelism_threads(self, n):
            raise RuntimeError("other failure")

        def set_intra_op_parallelism_threads(self, n):
            return None

        def get_inter_op_parallelism_threads(self):
            return 0

        def get_intra_op_parallelism_threads(self):
            return 0

    class DummyTF:
        config = SimpleNamespace(threading=Threading())

    _install_fake_tensorflow(monkeypatch, DummyTF)
    with pytest.raises(RuntimeError):
        performance.set_threading_config()
