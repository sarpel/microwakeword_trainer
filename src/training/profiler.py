"""Training profiler for performance monitoring and bottleneck detection using cProfile."""

import cProfile
import logging
import os
import pstats
import time
from collections.abc import Callable
from contextlib import contextmanager
from io import StringIO
from typing import Any


class TrainingProfiler:
    """Performance profiler for training pipeline using cProfile.

    Provides comprehensive profiling for training sections and steps,
    with ability to save and load profile data for analysis.

    Example:
        >>> profiler = TrainingProfiler(output_dir="./profiles")
        >>> with profiler.profile_section("data_loading"):
        ...     data = load_data()
        >>> profiler.profile_training_step(model, lambda: get_batch(), n_steps=10)
    """

    def __init__(self, output_dir: str = "./profiles"):
        """Initialize the profiler.

        Args:
            output_dir: Directory to save profile files. Created if it does not exist.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._current_profiler: cProfile.Profile | None = None
        self._current_section: str | None = None

    @contextmanager
    def profile_section(self, name: str):
        """Context manager to profile a code section.

        Enables cProfile at entry, disables at exit, saves profile to
        output_dir/name_TIMESTAMP.prof, and prints summary of top 20
        functions by cumulative time.

        Args:
            name: Name identifier for the profiled section.

        Yields:
            None - use the context manager for profiling.

        Example:
            >>> with profiler.profile_section("data_loading"):
            ...     data = load_data()
        """
        timestamp = time.time_ns()  # Nanosecond resolution prevents filename collisions
        profile_path = os.path.join(self.output_dir, f"{name}_{timestamp}.prof")

        profiler = cProfile.Profile()
        self._current_profiler = profiler
        self._current_section = name

        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            profiler.dump_stats(profile_path)

            # Print summary
            self._print_profile_summary(profiler, name)

            self._current_profiler = None
            self._current_section = None

    def profile_training_step(self, model: Any, data_fn: Callable[[], Any], n_steps: int = 10) -> str | None:
        """Profile multiple training steps.

        Runs n_steps iterations profiling the entire loop including
        data fetching and model forward/backward pass.

        Args:
            model: The model to train (should have forward/backward methods).
            data_fn: Callable that returns a batch of data.
            n_steps: Number of training steps to profile.

        Returns:
            Path to saved profile file, or None if profiling failed.
        """
        timestamp = int(time.time())
        profile_path = os.path.join(self.output_dir, f"training_step_{timestamp}.prof")

        profiler = cProfile.Profile()
        profiler.enable()

        output = None
        try:
            for _step in range(n_steps):
                # Get data batch
                batch = data_fn()

                # Forward pass (if model supports it)
                if hasattr(model, "forward"):
                    output = model.forward(batch)
                elif callable(model):
                    output = model(batch)

                # Backward pass (if model supports it)
                if hasattr(model, "backward"):
                    model.backward(output)
                elif hasattr(model, "step"):
                    model.step()
        finally:
            profiler.disable()
            profiler.dump_stats(profile_path)

            # Print summary
            self._print_profile_summary(profiler, f"training_step (n_steps={n_steps})")

        return profile_path

    @staticmethod
    def get_summary(profile_path: str, top_n: int = 20) -> str:
        """Load a saved profile file and return formatted summary.

        Args:
            profile_path: Path to the .prof profile file.
            top_n: Number of top functions to include in summary.

        Returns:
            Formatted string with profile summary.

        Example:
            >>> summary = TrainingProfiler.get_summary("./profiles/data_loading_123456.prof")
            >>> print(summary)
        """
        if not os.path.exists(profile_path):
            return f"Profile file not found: {profile_path}"

        # Capture output to StringIO
        stream = StringIO()
        ps = pstats.Stats(profile_path, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(top_n)

        return stream.getvalue()

    def _print_profile_summary(self, profiler: cProfile.Profile, name: str) -> None:
        """Print formatted summary of profiling results.

        Args:
            profiler: The cProfile.Profile instance with results.
            name: Name of the profiled section.
        """
        # Get stats and sort by cumulative time
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")

        # Print header
        print(f"\n{'=' * 60}")
        print(f"Profile Summary: {name}")
        print(f"{'=' * 60}")

        # Print top 20 functions
        stats.print_stats(20)

        # Also print call info
        print(f"\n{'=' * 60}")
        print("Top functions by cumulative time:")
        print(f"{'=' * 60}")

        # Get stats sorted by cumulative and print callers
        stats.sort_stats("cumulative")
        stats.print_callers(10)

        print(f"\nProfile saved to: {self.output_dir}/{name}_*.prof")


class TFProfiler:
    """TensorFlow Profiler wrapper for GPU kernel traces and memory analysis.

    Captures op-level GPU/CPU timing, memory allocation, and kernel traces
    viewable in TensorBoard's Profile tab.

    Example:
        >>> tf_profiler = TFProfiler(log_dir="./logs")
        >>> tf_profiler.start_trace(step=1000)
        >>> # ... run training steps ...
        >>> tf_profiler.stop_trace()
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        warmup_steps: int = 2,
        active_steps: int = 5,
    ):
        """Initialize TF Profiler.

        Args:
            log_dir: TensorBoard log directory for profile output.
            warmup_steps: Steps to skip before recording (GPU warm-up).
            active_steps: Number of steps to actually profile.
        """
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self._tracing = False
        self._logger = logging.getLogger(__name__)
        self._step_counter = 0
        self._start_step: int | None = None

    def start_trace(self, step: int | None = None) -> None:
        """Start TF Profiler trace.

        Args:
            step: Current training step (used for output directory naming).
        """
        self._start_step = step
        self._step_counter = 0
        # Don't actually start tracing until warmup_steps are done
        # The actual start is controlled by the caller checking steps
        self._tracing = False
        self._logger.info(f"TF Profiler armed at step {step} (warmup={self.warmup_steps}, active={self.active_steps})")

    def _actually_start_trace(self) -> None:
        """Internal method to actually start TF Profiler tracing."""
        try:
            import tensorflow as tf

            profile_dir = os.path.join(self.log_dir, "plugins", "profile")
            os.makedirs(profile_dir, exist_ok=True)
            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2,
                python_tracer_level=1,
                device_tracer_level=1,
            )
            tf.profiler.experimental.start(self.log_dir, options=options)
            self._tracing = True
            self._logger.info("TF Profiler actually started")
        except Exception as e:
            self._logger.warning(f"TF Profiler start failed: {e}")
            self._tracing = False

    def step(self) -> None:
        """Call this each training step to manage tracing based on warmup/active steps."""
        if self._start_step is None:
            return

        self._step_counter += 1

        # Start tracing after warmup_steps
        if not self._tracing and self._step_counter > self.warmup_steps:
            self._actually_start_trace()
            return

        # Stop tracing after warmup_steps + active_steps
        if self._tracing and self._step_counter > self.warmup_steps + self.active_steps:
            self.stop_trace()

    def stop_trace(self) -> None:
        """Stop TF Profiler trace and flush results."""
        if not self._tracing:
            return
        try:
            import tensorflow as tf

            tf.profiler.experimental.stop()
            self._tracing = False
            self._logger.info(f"TF Profiler trace saved to {self.log_dir}")
        except Exception as e:
            self._logger.warning(f"TF Profiler stop failed: {e}")
            self._tracing = False

    def is_tracing(self) -> bool:
        """Check if TF Profiler is currently tracing.

        Returns:
            True if tracing is active, False otherwise.
        """
        return self._tracing

    @contextmanager
    def trace(self, step: int | None = None):
        """Context manager for TF Profiler trace.

        Args:
            step: Current training step.

        Yields:
            None
        """
        self.start_trace(step=step)
        try:
            yield
        finally:
            self.stop_trace()

    def get_gpu_memory_info(self) -> dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dict with peak_mb and current_mb (both in MB).
        """
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return {"peak_mb": 0.0, "current_mb": 0.0}

            # TF memory info from the first visible GPU
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            return {
                "peak_mb": mem_info.get("peak", 0) / (1024 * 1024),
                "current_mb": mem_info.get("current", 0) / (1024 * 1024),
            }
        except Exception:
            return {"peak_mb": 0.0, "current_mb": 0.0}
