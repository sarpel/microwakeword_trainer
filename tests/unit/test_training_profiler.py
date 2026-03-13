"""Unit tests for training profiler module."""

import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from src.training.profiler import TrainingProfiler, TFProfiler


class TestTrainingProfiler:
    """Tests for TrainingProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = TrainingProfiler(output_dir=tmpdir)
            assert profiler.output_dir == tmpdir
            assert os.path.exists(tmpdir)

    def test_profile_section_creates_file(self):
        """Test that profile_section creates a profile file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = TrainingProfiler(output_dir=tmpdir)

            with profiler.profile_section("test_section"):
                pass  # Simulate some work

            # Check that profile file was created
            prof_files = list(Path(tmpdir).glob("test_section_*.prof"))
            assert len(prof_files) == 1

    def test_profile_section_multiple_calls(self):
        """Test profiling multiple calls to same section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = TrainingProfiler(output_dir=tmpdir)

            for _ in range(3):
                # Wait briefly to ensure different timestamps
                import time

                time.sleep(0.01)
                with profiler.profile_section("test_section"):
                    pass

            # Should create 3 separate profile files
            prof_files = list(Path(tmpdir).glob("test_section_*.prof"))
            assert len(prof_files) == 3

    def test_profile_training_step(self):
        """Test training step profiling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = TrainingProfiler(output_dir=tmpdir)

            # Mock model and data function
            mock_model = mock.Mock()
            mock_model.forward.return_value = "output"
            mock_data_fn = mock.Mock(return_value="batch")

            profile_path = profiler.profile_training_step(mock_model, mock_data_fn, n_steps=2)

            assert profile_path is not None
            assert Path(profile_path).exists()
            assert "training_step" in profile_path

    def test_profile_training_step_callable_model(self):
        """Test training step with callable model (not object)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = TrainingProfiler(output_dir=tmpdir)

            mock_model = mock.Mock()
            mock_data_fn = mock.Mock(return_value="batch")

            profile_path = profiler.profile_training_step(mock_model, mock_data_fn, n_steps=1)

            assert profile_path is not None

    def test_get_summary_existing_file(self):
        """Test getting summary from existing profile file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = TrainingProfiler(output_dir=tmpdir)

            # Create a profile file first
            with profiler.profile_section("test_section"):
                pass

            prof_files = list(Path(tmpdir).glob("test_section_*.prof"))
            summary = TrainingProfiler.get_summary(str(prof_files[0]))

            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_get_summary_nonexistent_file(self):
        """Test getting summary from non-existent file."""
        summary = TrainingProfiler.get_summary("/nonexistent/path/file.prof")
        assert "not found" in summary

    def test_output_dir_created(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new" / "nested" / "dir"
            assert not output_dir.exists()

            profiler = TrainingProfiler(output_dir=str(output_dir))
            assert output_dir.exists()


class TestTFProfiler:
    """Tests for TFProfiler class."""

    def test_initialization(self):
        """Test TFProfiler initialization."""
        profiler = TFProfiler(log_dir="/tmp/logs")
        assert profiler.log_dir == "/tmp/logs"
        assert profiler.warmup_steps == 2
        assert profiler.active_steps == 5
        assert profiler._tracing is False

    def test_initialization_custom_params(self):
        """Test TFProfiler with custom parameters."""
        profiler = TFProfiler(log_dir="/tmp/test", warmup_steps=5, active_steps=10)
        assert profiler.log_dir == "/tmp/test"
        assert profiler.warmup_steps == 5
        assert profiler.active_steps == 10

    def test_start_trace(self):
        """Test start_trace method."""
        profiler = TFProfiler()
        profiler.start_trace(step=100)

        assert profiler._start_step == 100
        assert profiler._step_counter == 0
        assert profiler._tracing is False  # Not tracing yet (warmup)

    def test_start_trace_no_step(self):
        """Test start_trace without step parameter."""
        profiler = TFProfiler()
        profiler.start_trace()

        assert profiler._start_step is None
        assert profiler._step_counter == 0

    def test_step_increments_counter(self):
        """Test that step method increments counter."""
        profiler = TFProfiler()
        profiler.start_trace(step=0)

        profiler.step()
        assert profiler._step_counter == 1

        profiler.step()
        assert profiler._step_counter == 2

    def test_step_no_start(self):
        """Test step when start_trace not called."""
        profiler = TFProfiler()

        # Should not raise error
        profiler.step()
        assert profiler._step_counter == 0  # No increment

    def test_is_tracing_property(self):
        """Test is_tracing property."""
        profiler = TFProfiler()
        assert profiler.is_tracing() is False

        # Manually set tracing state
        profiler._tracing = True
        assert profiler.is_tracing() is True

    def test_stop_trace_without_starting(self):
        """Test stop trace when not tracing."""
        profiler = TFProfiler()

        # Should not raise error
        profiler.stop_trace()
        assert profiler._tracing is False

    def test_context_manager(self):
        """Test using profiler as context manager."""
        profiler = TFProfiler()

        # Without TensorFlow, this should just pass through
        with profiler.trace(step=100):
            pass  # Simulate some work

        # Should complete without error
        assert True

    def test_get_gpu_memory_info_without_tensorflow(self):
        """Test GPU memory info when TensorFlow is not available."""
        profiler = TFProfiler()

        info = profiler.get_gpu_memory_info()
        assert info == {"peak_mb": 0.0, "current_mb": 0.0}

    def test_multiple_trace_sessions(self):
        """Test multiple start/stop trace cycles."""
        profiler = TFProfiler()

        # First session
        profiler.start_trace(step=0)
        profiler.step()
        profiler.stop_trace()

        # Second session
        profiler.start_trace(step=100)
        profiler.step()
        profiler.stop_trace()

        # Should complete without error
        assert profiler._tracing is False
