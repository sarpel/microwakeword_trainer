"""Centralized logging configuration with Rich formatting.

Provides unified logging setup that routes all Python logs through Rich handlers
for consistent terminal output with colors and formatting.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_rich_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
    show_time: bool = True,
    show_path: bool = True,
    console: Console | None = None,
) -> None:
    """Configure root logger to use RichHandler for all logging output.

    This replaces any existing handlers and ensures all Python logging
    (from any module) goes through Rich for consistent terminal formatting.

    Args:
        level: Logging level (default: INFO)
        log_format: Optional custom format string. If None, uses Rich's default
        show_time: Whether to show timestamp in logs
        show_path: Whether to show logger path/name
        console: Optional Rich Console instance (creates default if None)

    Example:
        >>> from src.utils.logging_config import setup_rich_logging
        >>> import logging
        >>> setup_rich_logging(level=logging.DEBUG)
        >>> logging.info("This will appear with Rich formatting")
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove all existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create Rich console if not provided
    if console is None:
        console = Console()

    # Create RichHandler with appropriate settings
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    rich_handler.setLevel(level)

    # Apply custom format if provided
    if log_format:
        formatter = logging.Formatter(log_format)
        rich_handler.setFormatter(formatter)

    root_logger.addHandler(rich_handler)


def setup_file_and_console_logging(
    log_file: str,
    level: int = logging.INFO,
    console: Console | None = None,
) -> None:
    """Configure logging to both Rich console and file.

    Sets up dual logging: RichHandler for terminal output and FileHandler
    for persistent log files.

    Args:
        log_file: Path to log file
        level: Logging level (default: INFO)
        console: Optional Rich Console instance

    Example:
        >>> setup_file_and_console_logging("./logs/training.log")
    """
    from logging import FileHandler
    from pathlib import Path

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console if not provided
    if console is None:
        console = Console()

    # RichHandler for console output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(level)
    root_logger.addHandler(rich_handler)

    # FileHandler for file output (plain text, no ANSI codes)
    file_handler = FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Convenience wrapper around logging.getLogger() that ensures
    Rich logging is properly configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
