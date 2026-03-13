"""Unit tests for src.utils.performance_monitor."""

from __future__ import annotations

from pathlib import Path

from src.utils.performance_monitor import PerformanceMonitor


def test_track_section_and_report_and_save(tmp_path: Path) -> None:
    m = PerformanceMonitor(log_dir=str(tmp_path), enable_profiling=True)
    with m.track_section("load"):
        _ = 1 + 1
    with m.track_section("load"):
        _ = 2 + 2

    rep = m.get_report()
    assert "PERFORMANCE MONITOR REPORT" in rep
    assert "load" in rep

    out = tmp_path / "report.txt"
    m.save_report(str(out))
    assert out.exists()
    assert "PERFORMANCE MONITOR REPORT" in out.read_text()


def test_record_section_triggers_bottleneck_and_trend(tmp_path: Path) -> None:
    m = PerformanceMonitor(log_dir=str(tmp_path), enable_profiling=True)
    m._record_section("s", 10.0)  # baseline
    m._record_section("s", 30.0)  # bottleneck
    for _ in range(10):
        m._record_section("s", 10.0)
    for _ in range(10):
        m._record_section("s", 25.0)
    assert any("BOTTLENECK" in a for a in m.alerts)
    assert any("TREND" in a for a in m.alerts)


def test_monitor_memory_import_error(tmp_path: Path, monkeypatch) -> None:
    import src.utils.performance_monitor as pm_module

    m = PerformanceMonitor(log_dir=str(tmp_path), enable_profiling=True)
    # Simulate psutil not being available at module level
    monkeypatch.setattr(pm_module, "_psutil_available", False)
    out = m.monitor_memory()
    assert out == {"rss_mb": 0, "vms_mb": 0, "percent": 0}


def test_monitor_gpu_memory_no_gpu(tmp_path: Path, monkeypatch) -> None:
    import src.utils.performance as perf_module

    m = PerformanceMonitor(log_dir=str(tmp_path), enable_profiling=True)
    monkeypatch.setattr(perf_module, "check_gpu_available", lambda: False)
    out = m.monitor_gpu_memory()
    assert out == {"allocated_mb": 0, "peak_mb": 0}


def test_monitor_gpu_memory_with_gpu(tmp_path: Path, monkeypatch) -> None:
    import tensorflow as tf

    import src.utils.performance as perf_module

    m = PerformanceMonitor(log_dir=str(tmp_path), enable_profiling=True)
    monkeypatch.setattr(perf_module, "check_gpu_available", lambda: True)
    monkeypatch.setattr(
        tf.config.experimental,
        "get_memory_info",
        lambda _device: {"current": 1024 * 1024, "peak": 3 * 1024 * 1024},
    )
    out = m.monitor_gpu_memory()
    assert out["allocated_mb"] == 1.0
    assert out["peak_mb"] == 3.0


def test_disable_profiling_skips_record(tmp_path: Path) -> None:
    m = PerformanceMonitor(log_dir=str(tmp_path), enable_profiling=False)
    with m.track_section("x"):
        _ = 1
    assert m.section_history == {}
