"""Unit tests for utility helper modules."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from src.utils import optional_deps, performance, seed


def test_require_optional_success_and_failure() -> None:
    math_mod = optional_deps.require_optional("math", extra="x")
    assert hasattr(math_mod, "sqrt")

    try:
        optional_deps.require_optional("definitely_missing_module_xyz", extra="quality-full", pip_name="missing-pkg")
        raise AssertionError("Expected ImportError")
    except ImportError as exc:
        msg = str(exc)
        assert "missing-pkg" in msg
        assert "microwakeword_trainer[quality-full]" in msg


def test_seed_everything_with_and_without_cupy(monkeypatch) -> None:
    called = {"tf": None, "cupy": None}

    class DummyTFRandom:
        @staticmethod
        def set_seed(v):
            called["tf"] = v

    class DummyTF:
        random = DummyTFRandom()

    class DummyCuPyRandom:
        @staticmethod
        def seed(v):
            called["cupy"] = v

    class DummyCuPy:
        random = DummyCuPyRandom()

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            return DummyTF
        if name == "cupy":
            return DummyCuPy
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    seed.seed_everything(123)
    assert called["tf"] == 123
    assert called["cupy"] == 123
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"


def test_seed_everything_without_cupy(monkeypatch) -> None:
    """Verify seed_everything works correctly when CuPy is not available."""
    called = {"tf": None}

    class DummyTFRandom:
        @staticmethod
        def set_seed(v):
            called["tf"] = v

    class DummyTF:
        random = DummyTFRandom()

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            return DummyTF
        if name == "cupy":
            raise ImportError("No module named 'cupy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    seed.seed_everything(42)
    assert called["tf"] == 42
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"


def test_performance_system_and_gpu_checks(monkeypatch) -> None:
    monkeypatch.setattr(performance.psutil, "cpu_count", lambda logical=False: 8 if logical else 4)
    monkeypatch.setattr(performance.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3))
    monkeypatch.setattr(performance.GPUtil, "getGPUs", lambda: [SimpleNamespace(name="GPU0", memoryTotal=1000, memoryFree=700, memoryUsed=300, load=0.25)])

    info = performance.get_system_info()
    assert info["cpu_count"] == 4
    assert info["gpu"]["name"] == "GPU0"
    assert performance.check_gpu_available() is True


def test_check_gpu_and_cupy_available_paths(monkeypatch) -> None:
    monkeypatch.setattr(performance, "check_gpu_available", lambda: False)
    ok, msg = performance.check_gpu_and_cupy_available()
    assert ok is False and "requires a GPU" in msg


def test_format_bytes_and_io_profiler(tmp_path: Path) -> None:
    assert performance.format_bytes(1024) == "1.00 KB"
    assert performance.format_bytes(1024**2) == "1.00 MB"

    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 100)
    prof = performance.IOProfiler()
    with prof.track_read(str(p)):
        _ = p.read_bytes()
    with prof.track_write(str(p)):
        p.write_bytes(b"x" * 128)
    rep = prof.get_report()
    assert "I/O PROFILING REPORT" in rep
    assert "Read Operations:" in rep
    assert "Write Operations:" in rep


def test_setup_gpu_environment_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    performance.setup_gpu_environment(cuda_visible_devices="0", tf_force_gpu_allow_growth=True, tf_gpu_allocator="cuda_malloc_async")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
