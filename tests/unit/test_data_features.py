"""Tests for data/features.py module.

Covers feature extraction, MicroFrontend, and spectrogram generation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.features import (
    FeatureConfig,
    MicroFrontend,
    SpectrogramGeneration,
    _get_cached_frontend,
    compute_mel_spectrogram,
    extract_features,
)

# =============================================================================
# FeatureConfig Tests
# =============================================================================


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeatureConfig()
        assert config.sample_rate == 16000
        assert config.window_size_ms == 30
        assert config.window_step_ms == 10
        assert config.mel_bins == 40
        assert config.num_coeffs == 10
        assert config.fft_size == 512
        assert config.low_freq == 125
        assert config.high_freq == 7500

    def test_post_init_derived_values(self):
        """Test that __post_init__ calculates derived values."""
        config = FeatureConfig()
        # 30ms at 16kHz = 480 samples
        assert config.window_size_samples == 480
        # 10ms at 16kHz = 160 samples
        assert config.window_step_samples == 160

    def test_invalid_sample_rate_raises(self):
        """Test that non-16kHz sample rate raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            FeatureConfig(sample_rate=22050)

    def test_get_frame_count_basic(self):
        """Test frame count calculation."""
        config = FeatureConfig()
        # 1 second at 16kHz with 30ms window and 10ms step
        # (16000 - 480) / 160 + 1 = 98 frames (approximately)
        frame_count = config.get_frame_count(16000)
        assert frame_count > 0

    def test_get_frame_count_short_audio(self):
        """Test frame count for very short audio."""
        config = FeatureConfig()
        frame_count = config.get_frame_count(100)  # Less than window size
        assert frame_count == 1  # Should return at least 1

    def test_validate_passes_with_defaults(self):
        """Test that default config passes validation."""
        config = FeatureConfig()
        issues = config.validate()
        assert issues == []

    def test_validate_catches_low_sample_rate(self):
        """Test validation catches sample rate < 1000."""
        # Note: This is only caught by validate(), not __post_init__
        # because __post_init__ catches non-16000 first
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 500
        config.window_size_ms = 30
        config.window_step_ms = 10
        config.mel_bins = 40
        config.low_freq = 125
        config.high_freq = 7500
        issues = config.validate()
        assert any("sample_rate must be" in issue for issue in issues)

    def test_validate_catches_zero_window_size(self):
        """Test validation catches zero window size."""
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 16000
        config.window_size_ms = 0
        config.window_step_ms = 10
        config.mel_bins = 40
        config.low_freq = 125
        config.high_freq = 7500
        issues = config.validate()
        assert any("window_size_ms must be positive" in issue for issue in issues)

    def test_validate_catches_window_step_larger_than_window(self):
        """Test validation catches window_step > window_size."""
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 16000
        config.window_size_ms = 10
        config.window_step_ms = 30
        config.mel_bins = 40
        config.low_freq = 125
        config.high_freq = 7500
        issues = config.validate()
        assert any("window_step_ms must be <=" in issue for issue in issues)

    def test_validate_catches_wrong_mel_bins(self):
        """Test validation catches mel_bins != 40."""
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 16000
        config.window_size_ms = 30
        config.window_step_ms = 10
        config.mel_bins = 20  # Wrong value
        config.low_freq = 125
        config.high_freq = 7500
        issues = config.validate()
        assert any("mel_bins must be exactly 40" in issue for issue in issues)

    def test_validate_catches_invalid_freq_range(self):
        """Test validation catches high_freq <= low_freq."""
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 16000
        config.window_size_ms = 30
        config.window_step_ms = 10
        config.mel_bins = 40
        config.low_freq = 5000
        config.high_freq = 1000  # Less than low_freq
        issues = config.validate()
        assert any("high_freq must be > low_freq" in issue for issue in issues)

    def test_validate_catches_high_freq_above_nyquist(self):
        """Test validation catches high_freq > sample_rate/2."""
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 16000
        config.window_size_ms = 30
        config.window_step_ms = 10
        config.mel_bins = 40
        config.low_freq = 125
        config.high_freq = 10000  # > 8000 (Nyquist)
        issues = config.validate()
        assert any("high_freq must be <=" in issue for issue in issues)


# =============================================================================
# MicroFrontend Tests (Mocked)
# =============================================================================


class TestMicroFrontend:
    """Tests for MicroFrontend class."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            frontend = MicroFrontend()
            assert frontend.config.sample_rate == 16000

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            config = FeatureConfig.__new__(FeatureConfig)
            config.sample_rate = 16000
            config.window_size_ms = 30
            config.window_step_ms = 10
            config.mel_bins = 40
            config.window_size_samples = 480
            frontend = MicroFrontend(config)
            assert frontend.config == config

    def test_check_pymicro_features_imports(self):
        """Test that _check_pymicro_features imports pymicro_features."""
        with patch("builtins.__import__") as mock_import:
            mock_import.return_value = MagicMock()
            config = FeatureConfig.__new__(FeatureConfig)
            config.sample_rate = 16000
            config.window_size_ms = 30
            config.window_step_ms = 10
            config.mel_bins = 40
            config.window_size_samples = 480
            frontend = MicroFrontend(config)
            frontend._check_pymicro_features()

            mock_import.assert_called()

    def test_check_pymicro_features_raises_without_pymicro(self):
        """Test that missing pymicro_features raises RuntimeError."""
        # This test verifies the error handling code path
        # We patch the import to simulate pymicro_features not being available
        config = FeatureConfig.__new__(FeatureConfig)
        config.sample_rate = 16000
        config.window_size_ms = 30
        config.window_step_ms = 10
        config.mel_bins = 40
        config.window_size_samples = 480
        frontend = MicroFrontend(config)

        # Test the import error handling by patching at the module level
        import sys

        # Remove pymicro_features from sys.modules if it exists
        modules_to_restore = {}
        if "pymicro_features" in sys.modules:
            modules_to_restore["pymicro_features"] = sys.modules.pop("pymicro_features")

        # Create a mock __import__ that raises ImportError for pymicro_features
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "pymicro_features":
                raise ImportError("No module named 'pymicro_features'")
            return original_import(name, *args, **kwargs)

        __builtins__["__import__"] = mock_import
        try:
            with pytest.raises(RuntimeError, match="pymicro-features is required"):
                frontend._check_pymicro_features()
        finally:
            __builtins__["__import__"] = original_import
            sys.modules.update(modules_to_restore)

    def test_compute_mel_spectrogram_calls_internal(self):
        """Test that compute_mel_spectrogram calls internal method."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch.object(MicroFrontend, "_compute_mel_spectrogram_pymicro") as mock_compute:
                mock_compute.return_value = np.zeros((10, 40))
                config = FeatureConfig.__new__(FeatureConfig)
                config.sample_rate = 16000
                config.window_size_ms = 30
                config.window_step_ms = 10
                config.mel_bins = 40
                config.window_size_samples = 480
                frontend = MicroFrontend(config)
                audio = np.random.randn(16000).astype(np.float32)
                result = frontend.compute_mel_spectrogram(audio)

                mock_compute.assert_called_once_with(audio)
                assert result.shape == (10, 40)

    def test_extract_alias(self):
        """Test that extract is an alias for compute_mel_spectrogram."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch.object(MicroFrontend, "_compute_mel_spectrogram_pymicro") as mock_compute:
                mock_compute.return_value = np.zeros((10, 40))
                config = FeatureConfig.__new__(FeatureConfig)
                config.sample_rate = 16000
                config.window_size_ms = 30
                config.window_step_ms = 10
                config.mel_bins = 40
                config.window_size_samples = 480
                frontend = MicroFrontend(config)
                audio = np.random.randn(16000).astype(np.float32)
                result = frontend.extract(audio)

                mock_compute.assert_called_once()


# =============================================================================
# SpectrogramGeneration Tests
# =============================================================================


class TestSpectrogramGeneration:
    """Tests for SpectrogramGeneration class."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            assert gen.config.sample_rate == 16000

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            config = FeatureConfig.__new__(FeatureConfig)
            config.sample_rate = 16000
            config.window_size_ms = 30
            config.window_step_ms = 10
            config.mel_bins = 40
            config.window_size_samples = 480
            config.window_step_samples = 160
            gen = SpectrogramGeneration(config)
            assert gen.config == config

    def test_frame_size_property(self):
        """Test frame_size property."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            assert gen.frame_size == 480  # 30ms at 16kHz

    def test_frame_step_property(self):
        """Test frame_step property."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            assert gen.frame_step == 160  # 10ms at 16kHz

    def test_num_mel_bins_property(self):
        """Test num_mel_bins property."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            assert gen.num_mel_bins == 40

    def test_slide_frames_basic(self):
        """Test sliding window frame generation."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            audio = np.arange(1000).astype(np.float32)
            frames = gen.slide_frames(audio, frame_length=480, frame_step=160)

            assert frames.shape[1] == 480  # frame_length
            # Should have multiple frames
            assert frames.shape[0] > 1

    def test_slide_frames_pads_short_audio(self):
        """Test that short audio is padded."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            audio = np.arange(100).astype(np.float32)  # Very short
            frames = gen.slide_frames(audio, frame_length=480, frame_step=160)

            # Should return at least one padded frame
            assert frames.shape[0] >= 1
            assert frames.shape[1] == 480

    def test_slide_frames_uses_config_defaults(self):
        """Test that slide_frames uses config defaults."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            audio = np.arange(1000).astype(np.float32)
            frames = gen.slide_frames(audio)  # No explicit frame params

            assert frames.shape[1] == gen.frame_size  # Uses config

    def test_generate_calls_frontend(self):
        """Test that generate calls frontend.compute_mel_spectrogram."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch.object(MicroFrontend, "compute_mel_spectrogram") as mock_compute:
                mock_compute.return_value = np.zeros((10, 40))
                gen = SpectrogramGeneration()
                audio = np.random.randn(16000).astype(np.float32)
                result = gen.generate(audio)

                mock_compute.assert_called_once_with(audio)
                assert result.shape == (10, 40)

    def test_generate_from_file(self, tmp_path):
        """Test generating spectrogram from file."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.ingestion.load_audio_wave") as mock_load:
                mock_load.return_value = np.random.randn(16000).astype(np.float32)
                with patch.object(MicroFrontend, "compute_mel_spectrogram") as mock_compute:
                    mock_compute.return_value = np.zeros((10, 40))
                    gen = SpectrogramGeneration()
                    wav_path = tmp_path / "test.wav"
                    wav_path.touch()
                    result = gen.generate_from_file(str(wav_path))

                    mock_load.assert_called_once()
                    mock_compute.assert_called_once()

    def test_generate_from_file_with_target_length(self, tmp_path):
        """Test generating spectrogram with target length."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.ingestion.load_audio_wave") as mock_load:
                mock_load.return_value = np.random.randn(20000).astype(np.float32)
                with patch.object(MicroFrontend, "compute_mel_spectrogram") as mock_compute:
                    mock_compute.return_value = np.zeros((10, 40))
                    gen = SpectrogramGeneration()
                    wav_path = tmp_path / "test.wav"
                    wav_path.touch()
                    result = gen.generate_from_file(str(wav_path), target_length=16000)

                    mock_load.assert_called_once()
                    mock_compute.assert_called_once()

    def test_generate_from_file_pads_short_audio(self, tmp_path):
        """Test generating spectrogram pads short audio."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.ingestion.load_audio_wave") as mock_load:
                mock_load.return_value = np.random.randn(8000).astype(np.float32)  # 0.5s
                with patch.object(MicroFrontend, "compute_mel_spectrogram") as mock_compute:
                    mock_compute.return_value = np.zeros((10, 40))
                    gen = SpectrogramGeneration()
                    wav_path = tmp_path / "test.wav"
                    wav_path.touch()
                    result = gen.generate_from_file(str(wav_path), target_length=16000)

                    mock_compute.assert_called_once()
                    # Should pad to target length
                    called_audio = mock_compute.call_args[0][0]
                    assert len(called_audio) == 16000

    def test_process_batch_empty(self):
        """Test processing empty batch."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            batch = np.zeros((0, 16000), dtype=np.float32)
            result = gen.process_batch(batch)

            assert result.shape == (0, 0, 40)

    def test_process_batch_single_item(self):
        """Test processing batch with single item."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch.object(MicroFrontend, "compute_mel_spectrogram") as mock_compute:
                mock_compute.return_value = np.zeros((10, 40))
                gen = SpectrogramGeneration()
                batch = np.random.randn(1, 16000).astype(np.float32)
                result = gen.process_batch(batch)

                assert result.shape[0] == 1
                assert result.shape[2] == 40

    def test_process_batch_multiple_items(self):
        """Test processing batch with multiple items."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch.object(MicroFrontend, "compute_mel_spectrogram") as mock_compute:
                mock_compute.return_value = np.zeros((10, 40))
                gen = SpectrogramGeneration()
                batch = np.random.randn(4, 16000).astype(np.float32)
                result = gen.process_batch(batch)

                assert result.shape[0] == 4
                assert mock_compute.call_count == 4

    def test_process_batch_pads_different_lengths(self):
        """Test that batch processing pads to same length."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            # Return different length spectrograms based on audio length
            call_count = [0]

            def mock_compute(audio):
                call_count[0] += 1
                if call_count[0] == 1:
                    return np.zeros((10, 40))
                else:
                    return np.zeros((15, 40))  # Different length

            with patch.object(MicroFrontend, "compute_mel_spectrogram", side_effect=mock_compute):
                gen = SpectrogramGeneration()
                batch = np.random.randn(2, 16000).astype(np.float32)
                result = gen.process_batch(batch)

                # Should pad to max length (15)
                assert result.shape[1] == 15

    def test_get_expected_output_shape(self):
        """Test getting expected output shape."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            gen = SpectrogramGeneration()
            shape = gen.get_expected_output_shape(16000)

            assert shape[1] == 40  # mel_bins
            assert shape[0] > 0  # num_frames


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGetCachedFrontend:
    """Tests for _get_cached_frontend function."""

    def test_caching_same_params(self):
        """Test that same params return cached frontend."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            frontend1 = _get_cached_frontend(16000, 30, 10, 40, 10, 512, 125, 7500)
            frontend2 = _get_cached_frontend(16000, 30, 10, 40, 10, 512, 125, 7500)
            assert frontend1 is frontend2

    def test_different_params_create_new(self):
        """Test that different params create new frontend."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            frontend1 = _get_cached_frontend(16000, 30, 10, 40, 10, 512, 125, 7500)
            frontend2 = _get_cached_frontend(
                16000,
                40,
                10,
                40,
                10,
                512,
                125,
                7500,  # Different window_size_ms
            )
            assert frontend1 is not frontend2


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.features._get_cached_frontend") as mock_get:
                mock_frontend = MagicMock()
                mock_frontend.compute_mel_spectrogram.return_value = np.zeros((10, 40))
                mock_get.return_value = mock_frontend

                audio = np.random.randn(16000).astype(np.float32)
                result = extract_features(audio)

                mock_get.assert_called_once()
                mock_frontend.compute_mel_spectrogram.assert_called_once()
                assert result.shape == (10, 40)

    def test_extract_features_custom_params(self):
        """Test feature extraction with custom params."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.features._get_cached_frontend") as mock_get:
                mock_frontend = MagicMock()
                mock_frontend.compute_mel_spectrogram.return_value = np.zeros((10, 40))
                mock_get.return_value = mock_frontend

                audio = np.random.randn(16000).astype(np.float32)
                result = extract_features(audio, sample_rate=16000, window_size_ms=40)

                # Should use cached frontend with different params
                mock_get.assert_called_once()


class TestComputeMelSpectrogram:
    """Tests for compute_mel_spectrogram function."""

    def test_compute_mel_spectrogram_basic(self):
        """Test basic mel spectrogram computation."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.features._get_cached_frontend") as mock_get:
                mock_frontend = MagicMock()
                mock_frontend.compute_mel_spectrogram.return_value = np.zeros((10, 40))
                mock_get.return_value = mock_frontend

                audio = np.random.randn(16000).astype(np.float32)
                result = compute_mel_spectrogram(audio)

                mock_get.assert_called_once()
                assert result.shape == (10, 40)

    def test_compute_mel_spectrogram_alias(self):
        """Test that compute_mel_spectrogram is an alias for extract_features."""
        with patch.object(MicroFrontend, "_check_pymicro_features"):
            with patch("src.data.features._get_cached_frontend") as mock_get:
                mock_frontend = MagicMock()
                mock_frontend.compute_mel_spectrogram.return_value = np.zeros((10, 40))
                mock_get.return_value = mock_frontend

                audio = np.random.randn(16000).astype(np.float32)
                result1 = compute_mel_spectrogram(audio)
                result2 = extract_features(audio)

                # Both should produce same result
                assert result1.shape == result2.shape
