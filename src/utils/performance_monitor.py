"""
Automatic performance bottleneck detection and monitoring system.
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Automatic bottleneck detection and performance tracking.

    Tracks execution time of code sections, detects anomalies (sudden slowdowns),
    and reports trends (gradual degradation). Integrates with training loop
    for real-time performance visibility.
    """

    def __init__(self, log_dir: str, enable_profiling: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_profiling = enable_profiling
        self.baseline_metrics = {}
        self.section_history = defaultdict(list)
        self.alerts = []
        logger.info(f"Performance monitor initialized: {log_dir}")

    @contextmanager
    def track_section(self, section: str):
        """Context manager to track a code section's execution time.

        Automatically detects if section takes unusually long compared to baseline.

        Example:
            with monitor.track_section("data_loading"):
                data = load_data()
        """
        if not self.enable_profiling:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._record_section(section, duration_ms)

    def _record_section(self, section: str, duration_ms: float):
        """Record section execution time and check for anomalies."""
        self.section_history[section].append(duration_ms)
        if len(self.section_history[section]) > 100:
            self.section_history[section].pop(0)

        if section not in self.baseline_metrics:
            self.baseline_metrics[section] = duration_ms
            logger.info(f"Baseline established for {section}: {duration_ms:.1f}ms")
            return

        # Bottleneck detection (2x slower than baseline)
        ratio = duration_ms / self.baseline_metrics[section]
        if ratio > 2.0:
            alert_msg = f"BOTTLENECK: {section} took {ratio:.1f}x longer than baseline ({duration_ms:.1f}ms vs {self.baseline_metrics[section]:.1f}ms)"
            self.alerts.append(alert_msg)
            logger.warning(alert_msg)

        # Trend detection (gradual degradation)
        if len(self.section_history[section]) >= 10:
            recent_avg = sum(self.section_history[section][-10:]) / 10
            overall_avg = sum(self.section_history[section]) / len(self.section_history[section])
            if recent_avg > 1.2 * overall_avg:
                alert_msg = f"TREND: {section} degrading ({recent_avg:.1f}ms vs {overall_avg:.1f}ms avg)"
                self.alerts.append(alert_msg)
                logger.warning(alert_msg)

    def monitor_memory(self) -> dict:
        """Track Python heap memory usage. Returns dict with rss_mb, vms_mb, percent."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": mem_info.rss / (1024**2),
                "vms_mb": mem_info.vms / (1024**2),
                "percent": process.memory_percent(),
            }
        except ImportError:
            logger.warning("psutil not available, memory monitoring disabled")
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

    def monitor_gpu_memory(self) -> dict:
        """Track GPU memory usage via TensorFlow."""
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return {"allocated_mb": 0, "peak_mb": 0}
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            return {
                "allocated_mb": mem_info["current"] / (1024**2),
                "peak_mb": mem_info["peak"] / (1024**2),
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU memory info: {e}")
            return {"allocated_mb": 0, "peak_mb": 0}

    def get_report(self) -> str:
        """Generate performance report."""
        report = ["=" * 60, "PERFORMANCE MONITOR REPORT", "=" * 60, ""]
        report.append("Section Timing:")
        for section, history in self.section_history.items():
            avg_time = sum(history) / len(history) if history else 0
            report.append(f"  {section}: {avg_time:.1f}ms avg ({len(history)} samples)")

        report.extend(["", "Baselines:"])
        for section, baseline in self.baseline_metrics.items():
            report.append(f"  {section}: {baseline:.1f}ms")

        if self.alerts:
            report.extend(["", "Alerts:"])
            for alert in self.alerts[-10:]:
                report.append(f"  {alert}")

        return "\n".join(report)

    def save_report(self, filepath: Optional[str] = None):
        """Save report to file."""
        if filepath is None:
            filepath = str(self.log_dir / "performance_report.txt")
        report = self.get_report()
        with open(filepath, "w") as f:
            f.write(report)
        logger.info(f"Performance report saved: {filepath}")
