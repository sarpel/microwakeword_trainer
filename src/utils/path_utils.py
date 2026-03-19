"""Path sanitization utilities for secure file operations.

Provides functions to safely resolve and validate file paths to prevent
path traversal attacks (CWE-22).

Usage:
    from src.utils.path_utils import resolve_path_safe, validate_path_within_dir

    # Resolve a path safely (prevents ../ traversal)
    safe_path = resolve_path_safe(user_input, base_dir=Path("/data"))

    # Validate path is within a directory
    validate_path_within_dir(file_path, allowed_dir=Path("/data"))
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def resolve_path_safe(
    user_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    allow_absolute: bool = False,
) -> Path:
    """Safely resolve a user-provided path, preventing path traversal.

    Resolves the path relative to a base directory and ensures the result
    does not escape the base directory through .. or symlinks.

    Args:
        user_path: User-provided path (can be relative or absolute)
        base_dir: Base directory to resolve relative paths from.
                  If None, uses current working directory.
        allow_absolute: Whether to allow absolute paths. If False, absolute
                       paths are resolved relative to base_dir.

    Returns:
        Resolved, absolute Path object

    Raises:
        ValueError: If the resolved path escapes the base directory
        FileNotFoundError: If the path doesn't exist (when base_dir is specified)

    Examples:
        >>> resolve_path_safe("../etc/passwd", base_dir=Path("/data"))
        ValueError: Path escapes base directory

        >>> resolve_path_safe("audio.wav", base_dir=Path("/data"))
        Path('/data/audio.wav')
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()

    user_path = Path(user_path)

    # Check if user is trying to use absolute paths
    if user_path.is_absolute():
        if not allow_absolute:
            logger.warning("Absolute path not allowed, resolving relative to base_dir: %s", user_path)
            # Treat absolute paths as relative to base_dir for safety
            user_path = Path(user_path.relative_to(user_path.anchor))
        else:
            # For absolute paths, still validate they're within base_dir
            resolved = user_path.resolve()
            try:
                resolved.relative_to(base_dir)
                return resolved
            except ValueError:
                raise ValueError(f"Absolute path {user_path} escapes base directory {base_dir}")

    # Resolve the path relative to base_dir
    resolved = (base_dir / user_path).resolve()

    # Check for path traversal: ensure resolved is within base_dir
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {user_path} resolves to {resolved} "
            f"which escapes base directory {base_dir}"
        )

    return resolved


def validate_path_within_dir(
    file_path: Union[str, Path],
    allowed_dir: Union[str, Path],
    must_exist: bool = False,
) -> Path:
    """Validate that a file path is within an allowed directory.

    Args:
        file_path: Path to validate
        allowed_dir: Directory that the file must be within
        must_exist: Whether the file must exist on disk

    Returns:
        Resolved, absolute Path object

    Raises:
        ValueError: If the path escapes the allowed directory
        FileNotFoundError: If must_exist=True and file doesn't exist

    Examples:
        >>> validate_path_within_dir("sensitive.txt", allowed_dir=Path("/data"))
        Path('/data/sensitive.txt')

        >>> validate_path_within_dir("../../../etc/passwd", allowed_dir=Path("/data"))
        ValueError: Path escapes allowed directory
    """
    allowed_dir = Path(allowed_dir).resolve()
    file_path = Path(file_path).resolve()

    try:
        file_path.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(
            f"Path {file_path} is not within allowed directory {allowed_dir}"
        )

    if must_exist and not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    return file_path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename by removing dangerous characters.

    Removes characters that could be used in path traversal or shell injection.

    Args:
        filename: User-provided filename
        max_length: Maximum allowed length (default: 255 for most filesystems)

    Returns:
        Sanitized filename safe for use in file operations

    Examples:
        >>> sanitize_filename("../../../etc/passwd")
        'etcpasswd'

        >>> sanitize_filename("audio.wav")
        'audio.wav'
    """
    # Remove path separators and dangerous characters
    dangerous_chars = ['/', '\\', '..', '\x00']
    result = filename

    for char in dangerous_chars:
        result = result.replace(char, '')

    # Remove any remaining .. sequences
    while '..' in result:
        result = result.replace('..', '')

    # Limit length
    result = result[:max_length]

    # Don't return empty string
    if not result:
        result = "unnamed"

    return result
