from __future__ import annotations

"""Optional dependency loader with helpful error messages."""

import importlib
from typing import Any


def require_optional(module_name: str, extra: str, pip_name: str | None = None) -> Any:
    """Import an optional dependency, raising ImportError with install hint if missing.

    Args:
        module_name: Python module name (e.g. 'webrtcvad', 'torch')
        extra: extras_require key (e.g. 'vad', 'quality-full')
        pip_name: pip package name if different from module_name

    Returns:
        The imported module

    Raises:
        ImportError: with pip install hint
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        pkg = pip_name or module_name
        raise ImportError(f"{module_name} is required for this feature. Install it with: pip install {pkg}\nOr: pip install microwakeword_trainer[{extra}]") from e
