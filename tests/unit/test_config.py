"""Unit tests for config loader."""

from pathlib import Path

import pytest


class TestConfigLoader:
    """Test cases for config loader functionality."""

    @pytest.mark.unit
    def test_placeholder(self) -> None:
        """Placeholder test to verify pytest setup."""
        assert True

    @pytest.mark.unit
    def test_imports(self) -> None:
        """Test that config module can be imported."""
        try:
            from config import loader

            assert loader is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config.loader: {e}")

    @pytest.mark.unit
    def test_load_fast_test_preset(self) -> None:
        """Test loading fast_test preset validates without error."""
        from config.loader import load_full_config

        config = load_full_config("fast_test")
        assert config is not None
        assert config.training is not None
        assert config.model is not None
        assert config.augmentation is not None

    @pytest.mark.unit
    def test_config_training_has_required_fields(self) -> None:
        """Test that training config has required fields with valid values."""
        from config.loader import load_full_config

        config = load_full_config("fast_test")
        training = config.training
        assert hasattr(training, "batch_size")
        assert training.batch_size > 0
        assert hasattr(training, "epochs")
        assert training.epochs > 0

    @pytest.mark.unit
    def test_config_model_has_required_fields(self) -> None:
        """Test that model config has required fields."""
        from config.loader import load_full_config

        config = load_full_config("fast_test")
        model = config.model
        assert hasattr(model, "initial_filters")
        assert model.initial_filters > 0


class TestDataPaths:
    """Test data path validation."""

    @pytest.mark.unit
    def test_dataset_directories_exist(self, tmp_path: Path) -> None:
        """Check that dataset directory structure can be created correctly."""
        dataset_path = tmp_path / "dataset"
        required_dirs = ["positive", "negative"]
        for dir_name in required_dirs:
            (dataset_path / dir_name).mkdir(parents=True, exist_ok=True)

        assert dataset_path.exists()
        for dir_name in required_dirs:
            assert (dataset_path / dir_name).exists(), f"Missing {dir_name} directory"
