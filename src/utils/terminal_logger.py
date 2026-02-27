"""Terminal output capture and file logging for training.

Captures all terminal output (including Rich console output) to a log file
for later analysis and debugging.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console


class TeeOutput:
    """Captures stdout/stderr and writes to both terminal and file.

    Similar to Unix `tee` command - duplicates output to multiple destinations.
    """

    def __init__(self, *streams: TextIO) -> None:
        """Initialize with multiple output streams.

        Args:
            streams: Output streams to write to (e.g., sys.stdout, file)
        """
        self.streams = streams
        self._closed = False

    def write(self, data: str) -> int:
        """Write data to all streams."""
        if self._closed:
            return 0

        bytes_written = 0
        for stream in self.streams:
            try:
                result = stream.write(data)
                if result is not None:
                    bytes_written = max(bytes_written, result)
                stream.flush()
            except (IOError, ValueError):
                # Stream might be closed
                pass
        return bytes_written

    def flush(self) -> None:
        """Flush all streams."""
        for stream in self.streams:
            try:
                stream.flush()
            except (IOError, ValueError):
                pass

    def isatty(self) -> bool:
        """Check if the output is a terminal."""
        # Check if any stream is a terminal
        for stream in self.streams:
            try:
                if stream.isatty():
                    return True
            except (AttributeError, OSError):
                pass
        return False

    def close(self) -> None:
        """Close file streams (but not stdout/stderr)."""
        self._closed = True
        for stream in self.streams:
            if stream not in (sys.stdout, sys.stderr):
                try:
                    stream.close()
                except (IOError, ValueError):
                    pass

    def fileno(self) -> int:
        """Return file descriptor of first stream."""
        for stream in self.streams:
            try:
                return stream.fileno()
            except (AttributeError, OSError):
                pass
        raise OSError("No valid file descriptor")


class TerminalLogger:
    """Manages terminal output capture to log files.

    Creates a timestamped log file in the logs directory and captures
    all stdout/stderr output. Integrates with Rich console for formatted output.

    Usage:
        logger = TerminalLogger(log_dir="./logs")
        logger.start_capture()

        # All print() and console.print() output now goes to terminal AND file

        logger.stop_capture()  # Restore original stdout/stderr

    Attributes:
        log_file: Path to the current log file
        console: Rich Console instance that writes to both terminal and file
    """

    def __init__(
        self,
        log_dir: str | Path = "./logs",
        log_filename: str | None = None,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
    ) -> None:
        """Initialize terminal logger.

        Args:
            log_dir: Directory to save log files
            log_filename: Optional custom filename (default: terminal_YYYYMMDD_HHMMSS.log)
            capture_stdout: Whether to capture stdout
            capture_stderr: Whether to capture stderr
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"terminal_{timestamp}.log"

        self.log_file = self.log_dir / log_filename
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr

        # Save original streams
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self._file_handle: TextIO | None = None
        self._tee_stdout: TeeOutput | None = None
        self._tee_stderr: TeeOutput | None = None

        # Rich console that writes to both terminal and file
        self.console: Console | None = None
        self._is_capturing = False

    def start_capture(self) -> Path:
        """Start capturing terminal output to file.

        Returns:
            Path to the log file
        """
        if self._is_capturing:
            return self.log_file

        # Save original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Open log file
        self._file_handle = open(self.log_file, "w", encoding="utf-8")

        # Write header
        self._file_handle.write(f"Training Log Started: {datetime.now().isoformat()}\n")
        self._file_handle.write("=" * 80 + "\n\n")
        self._file_handle.flush()

        # Create tee outputs - check for None streams (can happen in some environments)
        stdout_stream = self._original_stdout if self._original_stdout is not None else sys.__stdout__
        stderr_stream = self._original_stderr if self._original_stderr is not None else sys.__stderr__

        if self.capture_stdout and stdout_stream is not None:
            self._tee_stdout = TeeOutput(stdout_stream, self._file_handle)  # type: ignore[arg-type]
            sys.stdout = self._tee_stdout  # type: ignore[assignment]

        if self.capture_stderr and stderr_stream is not None:
            self._tee_stderr = TeeOutput(stderr_stream, self._file_handle)  # type: ignore[arg-type]
            sys.stderr = self._tee_stderr  # type: ignore[assignment]

        # Create Rich console that writes to file
        # We create two consoles: one for terminal (with colors) and one for file (plain)
        self.console = Console(
            file=self._file_handle,
            force_terminal=False,
            color_system=None,  # No colors in file
            width=120,  # Wide for log files
        )

        self._is_capturing = True

        # Print startup message (goes to both terminal and file via tee)
        print(f"[TerminalLogger] Capturing output to: {self.log_file}")

        return self.log_file

    def stop_capture(self) -> None:
        """Stop capturing and restore original stdout/stderr."""
        if not self._is_capturing:
            return

        # Write footer
        if self._file_handle:
            self._file_handle.write(f"\n{'=' * 80}\n")
            self._file_handle.write(f"Training Log Ended: {datetime.now().isoformat()}\n")
            self._file_handle.flush()

        # Restore original streams
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

        # Close file handle
        if self._file_handle:
            self._file_handle.close()

        self._is_capturing = False
        print(f"[TerminalLogger] Capture stopped. Log saved to: {self.log_file}")

    def get_log_path(self) -> Path:
        """Get the path to the current log file."""
        return self.log_file

    def __enter__(self) -> "TerminalLogger":
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_capture()


def get_terminal_logger(
    log_dir: str | Path = "./logs",
    log_filename: str | None = None,
) -> TerminalLogger:
    """Get or create a terminal logger instance.

    This is a convenience function for creating a terminal logger
    with standard settings.

    Args:
        log_dir: Directory for log files
        log_filename: Optional custom filename

    Returns:
        TerminalLogger instance
    """
    return TerminalLogger(log_dir=log_dir, log_filename=log_filename)
