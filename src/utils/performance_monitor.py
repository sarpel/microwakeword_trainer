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

try:
    import psutil as _psutil

    _psutil_available = True
except ImportError:
    _psutil = None
    _psutil_available = False
    logger.warning("psutil not available; memory monitoring disabled")

# Initialize GPU environment once at module import time
_gpu_env_initialized = False
try:
    from src.utils.performance import setup_gpu_environment

    setup_gpu_environment()
    _gpu_env_initialized = True
    logger.debug("GPU environment initialized at module import")
except Exception as e:
    logger.warning(f"Failed to initialize GPU environment: {e}")
    _gpu_env_initialized = False


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
        self.baseline_metrics: dict[str, float] = {}
        self.section_history: defaultdict[str, list[float]] = defaultdict(list)
        self.alerts: list[str] = []
        self._last_alert_time: dict[str, float] = {}  # section -> last alert timestamp
        self._last_alert_msg: dict[str, str] = {}  # section -> last alert message
        self._alert_cooldown_seconds = 300  # 5 minutes cooldown for same alert
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

        alpha = 0.1  # EMA smoothing factor
        if section not in self.baseline_metrics:
            self.baseline_metrics[section] = duration_ms
            logger.info(f"Baseline established for {section}: {duration_ms:.1f}ms")
        else:
            old_baseline = self.baseline_metrics[section]  # Store old baseline first

            # Bottleneck detection (2x slower than baseline) - compare against old baseline
            # Guard against zero/negative baseline to prevent ZeroDivisionError
            if old_baseline > 0:
                ratio = duration_ms / old_baseline
                if ratio > 2.0:
                    alert_msg = f"BOTTLENECK: {section} took {ratio:.1f}x longer than baseline ({duration_ms:.1f}ms vs {old_baseline:.1f}ms)"
                    # Check dedup: only alert if new message or cooldown expired
                    current_time = time.time()
                    last_time = self._last_alert_time.get(section, 0)
                    last_msg = self._last_alert_msg.get(section, "")
                    if alert_msg != last_msg or (current_time - last_time) >= self._alert_cooldown_seconds:
                        self.alerts.append(alert_msg)
                        # Bound alerts list to prevent memory growth
                        if len(self.alerts) > 200:
                            self.alerts = self.alerts[-100:]
                        logger.warning(alert_msg)
                        self._last_alert_time[section] = current_time
                        self._last_alert_msg[section] = alert_msg

            # Update EMA baseline AFTER alarm decision
            self.baseline_metrics[section] = alpha * duration_ms + (1 - alpha) * old_baseline
        # Trend detection (gradual degradation)
        history = self.section_history[section]
        if len(history) >= 20:  # Ensure both windows have enough samples
            recent = history[-10:]
            older = history[-20:-10]  # Compare two equal-sized windows
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            if recent_avg > 1.2 * older_avg:
                alert_msg = f"TREND: {section} degrading ({recent_avg:.1f}ms vs {older_avg:.1f}ms avg)"
                # Check dedup: only alert if new message or cooldown expired
                current_time = time.time()
                last_time = self._last_alert_time.get(section, 0)
                last_msg = self._last_alert_msg.get(section, "")
                if alert_msg != last_msg or (current_time - last_time) >= self._alert_cooldown_seconds:
                    self.alerts.append(alert_msg)
                    # Bound alerts list to prevent memory growth
                    if len(self.alerts) > 200:
                        self.alerts = self.alerts[-100:]
                    logger.warning(alert_msg)
                    self._last_alert_time[section] = current_time
                    self._last_alert_msg[section] = alert_msg

    def monitor_memory(self) -> dict:
        """Track Python heap memory usage. Returns dict with rss_mb, vms_mb, percent."""
        if not _psutil_available or _psutil is None:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}
        process = _psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / (1024**2),
            "vms_mb": mem_info.vms / (1024**2),
            "percent": process.memory_percent(),
        }

    def monitor_gpu_memory(self) -> dict:
        """Track GPU memory usage via TensorFlow.

        GPU environment is initialized once at module import time.
        TensorFlow import is deferred until first call to avoid overhead
        when GPU monitoring is not used.
        """
        try:
            # Import TensorFlow - environment should already be set up
            import tensorflow as tf

            from src.utils.performance import check_gpu_available

            # Use project canonical check_gpu_available
            if not check_gpu_available():
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
