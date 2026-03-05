# conftest.py - Pytest configuration and fixtures
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def dataset_path(project_root: Path) -> Path:
    """Return the dataset directory path."""
    return project_root / "dataset"


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the config directory path."""
    return project_root / "config"
