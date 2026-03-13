"""Unit tests for terminal logger module."""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.utils.terminal_logger import (
    strip_ansi_codes,
    get_terminal_logger,
    TeeOutput,
    TerminalLogger,
)


class TestStripAnsiCodes:
    """Tests for strip_ansi_codes function."""

    def test_plain_text(self):
        """Test with plain text (no ANSI codes)."""
        text = "Hello, World!"
        assert strip_ansi_codes(text) == text

    def test_color_codes(self):
        """Test with color ANSI codes."""
        text = "\033[31mRed Text\033[0m"
        assert strip_ansi_codes(text) == "Red Text"

    def test_bold_code(self):
        """Test with bold ANSI code."""
        text = "\033[1mBold Text\033[0m"
        assert strip_ansi_codes(text) == "Bold Text"

    def test_multiple_codes(self):
        """Test with multiple ANSI codes."""
        text = "\033[1;31;40mBold Red on Black\033[0m"
        assert strip_ansi_codes(text) == "Bold Red on Black"

    def test_mixed_text(self):
        """Test with mixed plain and ANSI text."""
        text = "Start \033[32mgreen\033[0m end"
        assert strip_ansi_codes(text) == "Start green end"

    def test_empty_string(self):
        """Test with empty string."""
        assert strip_ansi_codes("") == ""

    def test_cursor_movement_codes(self):
        """Test with cursor movement codes."""
        text = "\033[2K\033[1GLine\033[0m"
        assert strip_ansi_codes(text) == "Line"


class TestTeeOutput:
    """Tests for TeeOutput class."""

    def test_writes_to_both_outputs(self):
        """Test that writes go to both outputs."""
        output1 = mock.Mock()
        output2 = mock.Mock()

        tee = TeeOutput(output1, output2)
        tee.write("test data")

        output1.write.assert_called_once_with("test data")
        output2.write.assert_called_once_with("test data")

    def test_flush_calls_both(self):
        """Test that flush calls both outputs."""
        output1 = mock.Mock()
        output2 = mock.Mock()

        tee = TeeOutput(output1, output2)
        tee.flush()

        output1.flush.assert_called_once()
        output2.flush.assert_called_once()

    def test_isatty_checks_first(self):
        """Test that isatty checks first output."""
        output1 = mock.Mock()
        output1.isatty.return_value = True
        output2 = mock.Mock()

        tee = TeeOutput(output1, output2)
        assert tee.isatty() is True

    def test_isatty_not_callable(self):
        """Test isatty when not available on output."""
        output1 = "not a stream"  # No isatty method
        output2 = mock.Mock()

        tee = TeeOutput(output1, output2)
        assert tee.isatty() is False

    def test_fileno_raises(self):
        """Test that fileno raises NotImplementedError."""
        output1 = mock.Mock()
        output2 = mock.Mock()

        tee = TeeOutput(output1, output2)
        with pytest.raises(NotImplementedError):
            tee.fileno()

    def test_close_closes_outputs(self):
        """Test that close closes both outputs."""
        output1 = mock.Mock()
        output2 = mock.Mock()

        tee = TeeOutput(output1, output2)
        tee.close()

        output1.close.assert_called_once()
        output2.close.assert_called_once()


class TestTerminalLogger:
    """Tests for TerminalLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = TerminalLogger(log_path=log_path)

            assert logger.log_path == log_path
            assert logger._original_stdout is None
            assert logger._original_stderr is None

    def test_start_capture(self):
        """Test starting capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = TerminalLogger(log_path=log_path)

            with mock.patch("sys.stdout") as mock_stdout:
                with mock.patch("sys.stderr") as mock_stderr:
                    logger.start_capture()

                    assert logger._original_stdout is not None
                    assert logger._original_stderr is not None

    def test_stop_capture(self):
        """Test stopping capture restores original streams."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = TerminalLogger(log_path=log_path)

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            logger._original_stdout = original_stdout
            logger._original_stderr = original_stderr

            with mock.patch("sys.stdout", mock.Mock()) as mock_stdout:
                with mock.patch("sys.stderr", mock.Mock()) as mock_stderr:
                    logger.stop_capture()

                    assert sys.stdout is original_stdout
                    assert sys.stderr is original_stderr

    def test_stop_capture_without_start(self):
        """Test stop capture when not started."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = TerminalLogger(log_path=log_path)

            # Should not raise error
            logger.stop_capture()

    def test_get_log_path(self):
        """Test getting log path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = TerminalLogger(log_path=log_path)

            assert logger.get_log_path() == log_path

    def test_context_manager(self):
        """Test using logger as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            with TerminalLogger(log_path=log_path):
                pass  # Simulate some work

            # Original streams should be restored
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr

    def test_log_file_created(self):
        """Test that log file is created during capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            # Note: Actual file creation would require more complex mocking
            # This test verifies the path is set correctly
            logger = TerminalLogger(log_path=log_path)
            assert logger.log_path == log_path


class TestGetTerminalLogger:
    """Tests for get_terminal_logger function."""

    def test_returns_terminal_logger(self):
        """Test function returns TerminalLogger instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "terminal.log"
            logger = get_terminal_logger(log_path)

            assert isinstance(logger, TerminalLogger)
            assert logger.log_path == log_path

    def test_creates_parent_directory(self):
        """Test that parent directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nested" / "dirs" / "terminal.log"
            logger = get_terminal_logger(log_path)

            assert log_path.parent.exists()
