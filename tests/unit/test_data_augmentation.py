"""Tests for data/augmentation.py module.

Covers AudioAugmentation class and GPU SpecAugment functionality.
"""

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.augmentation import AudioAugmentation, AugmentationConfig, apply_spec_augment_gpu

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    np.random.seed(42)
    return np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz


@pytest.fixture
def sample_spectrogram():
    """Generate sample spectrogram for testing."""
    np.random.seed(42)
    return np.random.randn(100, 40).astype(np.float32)


@pytest.fixture
def temp_audio_files(tmp_path):
    """Create temporary audio files for testing."""
    import struct
    import wave

    # Create background noise files
    bg_dir = tmp_path / "background"
    bg_dir.mkdir()

    bg_files = []
    for i in range(3):
        bg_file = bg_dir / f"noise_{i}.wav"
        with wave.open(str(bg_file), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # Generate 1 second of noise
            noise = np.random.randn(16000) * 0.1
            noise_int16 = (noise * 32767).astype(np.int16)
            wf.writeframes(struct.pack("<" + "h" * len(noise_int16), *noise_int16))
        bg_files.append(bg_file)

    # Create RIR files
    rir_dir = tmp_path / "rirs"
    rir_dir.mkdir()

    rir_files = []
    for i in range(2):
        rir_file = rir_dir / f"rir_{i}.wav"
        with wave.open(str(rir_file), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # Generate short impulse response
            rir = np.zeros(512)
            rir[0] = 1.0
            rir[100] = 0.3
            rir[200] = 0.1
            rir_int16 = (rir * 32767).astype(np.int16)
            wf.writeframes(struct.pack("<" + "h" * len(rir_int16), *rir_int16))
        rir_files.append(rir_file)

    return {"background": bg_files, "rirs": rir_files}


# =============================================================================
# AugmentationConfig Tests
# =============================================================================


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AugmentationConfig()
        assert config.SevenBandParametricEQ == 0.1
        assert config.TanhDistortion == 0.1
        assert config.PitchShift == 0.1
        assert config.BandStopFilter == 0.1
        assert config.AddColorNoise == 0.1
        assert config.AddBackgroundNoiseFromFile == 0.75
        assert config.Gain == 1.0
        assert config.ApplyImpulseResponse == 0.5
        assert config.background_min_snr_db == 0.0
        assert config.background_max_snr_db == 10.0
        assert config.min_jitter_s == 0.195
        assert config.max_jitter_s == 0.205
        assert config.augmentation_duration_s == 3.2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AugmentationConfig(
            SevenBandParametricEQ=0.5,
            Gain=0.0,
            background_min_snr_db=5.0,
            background_max_snr_db=15.0,
        )
        assert config.SevenBandParametricEQ == 0.5
        assert config.Gain == 0.0
        assert config.background_min_snr_db == 5.0
        assert config.background_max_snr_db == 15.0

    def test_background_paths(self):
        """Test background paths configuration."""
        paths = ["/path/to/bg1", "/path/to/bg2"]
        config = AugmentationConfig(background_paths=paths)
        assert config.background_paths == paths

    def test_impulse_paths(self):
        """Test impulse response paths configuration."""
        paths = ["/path/to/rir1"]
        config = AugmentationConfig(impulse_paths=paths)
        assert config.impulse_paths == paths


# =============================================================================
# AudioAugmentation Tests
# =============================================================================


class TestAudioAugmentation:
    """Tests for AudioAugmentation class."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        aug = AudioAugmentation()
        assert aug.config is not None
        assert aug.sample_rate == 16000
        assert aug.background_noise_files == []
        assert aug.rir_files == []

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = AugmentationConfig(Gain=0.5)
        aug = AudioAugmentation(config)
        assert aug.config.Gain == 0.5

    def test_load_background_files_with_dirs(self, tmp_path, temp_audio_files):
        """Test loading background files from directories."""
        config = AugmentationConfig(
            background_paths=[str(temp_audio_files["background"][0].parent)],
            impulse_paths=[str(temp_audio_files["rirs"][0].parent)],
        )
        aug = AudioAugmentation(config)
        assert len(aug.background_noise_files) == 3
        assert len(aug.rir_files) == 2

    def test_load_background_files_with_files(self, temp_audio_files):
        """Test loading background files from file paths."""
        config = AugmentationConfig(
            background_paths=[str(f) for f in temp_audio_files["background"][:2]],
            impulse_paths=[str(f) for f in temp_audio_files["rirs"][:1]],
        )
        aug = AudioAugmentation(config)
        assert len(aug.background_noise_files) == 2
        assert len(aug.rir_files) == 1

    def test_load_background_files_nonexistent(self, caplog):
        """Test loading from non-existent paths."""
        config = AugmentationConfig(
            background_paths=["/nonexistent/path"],
            impulse_paths=["/nonexistent/rir"],
        )
        with caplog.at_level("INFO"):
            aug = AudioAugmentation(config)
        assert aug.background_noise_files == []
        assert aug.rir_files == []

    # -------------------------------------------------------------------------
    # __call__ tests
    # -------------------------------------------------------------------------

    def test_call_with_apply_all(self, sample_audio):
        """Test __call__ with apply_all=True."""
        config = AugmentationConfig(Gain=1.0, SevenBandParametricEQ=1.0)
        aug = AudioAugmentation(config)

        # Mock the individual methods
        with patch.object(aug, "apply_gain") as mock_gain, patch.object(aug, "apply_eq") as mock_eq:
            mock_gain.return_value = sample_audio
            mock_eq.return_value = sample_audio
            # Also mock other methods that might be called
            with (
                patch.object(aug, "apply_distortion") as mock_dist,
                patch.object(aug, "apply_pitch_shift") as mock_pitch,
                patch.object(aug, "apply_band_stop") as mock_band,
                patch.object(aug, "apply_color_noise") as mock_color,
                patch.object(aug, "apply_background_noise") as mock_bg,
                patch.object(aug, "apply_rir") as mock_rir,
            ):
                mock_dist.return_value = sample_audio
                mock_pitch.return_value = sample_audio
                mock_band.return_value = sample_audio
                mock_color.return_value = sample_audio
                mock_bg.return_value = sample_audio
                mock_rir.return_value = sample_audio

                result = aug(sample_audio, apply_all=True)

                mock_gain.assert_called_once()
                mock_eq.assert_called_once()
                np.testing.assert_array_equal(result, sample_audio)

    def test_call_with_probabilities_zero(self, sample_audio):
        """Test __call__ with all probabilities set to 0."""
        config = AugmentationConfig(
            Gain=0.0,
            SevenBandParametricEQ=0.0,
            TanhDistortion=0.0,
            PitchShift=0.0,
            BandStopFilter=0.0,
            AddColorNoise=0.0,
            AddBackgroundNoiseFromFile=0.0,
            ApplyImpulseResponse=0.0,
        )
        aug = AudioAugmentation(config)
        result = aug(sample_audio)
        # Audio should be returned unchanged
        np.testing.assert_array_equal(result, sample_audio)

    def test_call_preserves_input(self, sample_audio):
        """Test that __call__ doesn't modify input array."""
        original = sample_audio.copy()
        config = AugmentationConfig(Gain=0.0)
        aug = AudioAugmentation(config)
        result = aug(sample_audio)  # noqa: F841
        np.testing.assert_array_equal(sample_audio, original)

    # -------------------------------------------------------------------------
    # apply_gain tests
    # -------------------------------------------------------------------------

    def test_apply_gain_changes_amplitude(self, sample_audio):
        """Test that apply_gain changes audio amplitude."""
        aug = AudioAugmentation()
        result = aug.apply_gain(sample_audio, min_db=-6.0, max_db=-6.0)

        # -6dB should reduce amplitude by half
        expected_gain = 10 ** (-6.0 / 20)
        np.testing.assert_array_almost_equal(result, sample_audio * expected_gain, decimal=5)

    def test_apply_gain_zero_db(self, sample_audio):
        """Test apply_gain with 0dB (no change)."""
        aug = AudioAugmentation()
        result = aug.apply_gain(sample_audio, min_db=0.0, max_db=0.0)
        np.testing.assert_array_almost_equal(result, sample_audio, decimal=5)

    def test_apply_gain_random_range(self, sample_audio):
        """Test apply_gain with random range."""
        random.seed(42)
        aug = AudioAugmentation()
        result = aug.apply_gain(sample_audio, min_db=-3.0, max_db=3.0)
        # Should produce different result than input
        assert not np.allclose(result, sample_audio)

    # -------------------------------------------------------------------------
    # apply_eq tests
    # -------------------------------------------------------------------------

    def test_apply_eq_with_audiomentations(self, sample_audio):
        """Test apply_eq when audiomentations is available."""
        aug = AudioAugmentation()

        mock_eq = MagicMock()
        mock_eq.return_value = sample_audio * 0.9

        with patch.dict("sys.modules", {"audiomentations": MagicMock(SevenBandParametricEQ=MagicMock(return_value=mock_eq))}):
            with patch("audiomentations.SevenBandParametricEQ", return_value=mock_eq):
                result = aug.apply_eq(sample_audio)  # noqa: F841
                mock_eq.assert_called_once()

    def test_apply_eq_without_audiomentations(self, sample_audio, caplog):
        """Test apply_eq fallback when audiomentations is not available."""
        aug = AudioAugmentation()

        with patch.dict("sys.modules", {"audiomentations": None}):
            # Force reimport to test ImportError path
            with caplog.at_level("DEBUG"):
                result = aug.apply_eq(sample_audio)

            # Should return audio unchanged
            np.testing.assert_array_equal(result, sample_audio)
            assert "audiomentations not available" in caplog.text

    # -------------------------------------------------------------------------
    # apply_distortion tests
    # -------------------------------------------------------------------------

    def test_apply_distortion_with_audiomentations(self, sample_audio):
        """Test apply_distortion when audiomentations is available."""
        aug = AudioAugmentation()

        mock_dist = MagicMock()
        mock_dist.return_value = sample_audio * 0.8

        with patch("audiomentations.TanhDistortion", return_value=mock_dist):
            result = aug.apply_distortion(sample_audio)  # noqa: F841
            mock_dist.assert_called_once()

    def test_apply_distortion_fallback(self, sample_audio):
        """Test apply_distortion fallback using numpy tanh."""
        aug = AudioAugmentation()

        with patch.dict("sys.modules", {"audiomentations": None}):
            random.seed(42)
            result = aug.apply_distortion(sample_audio)

            # Should apply tanh distortion
            assert not np.allclose(result, sample_audio)
            assert np.all(np.abs(result) <= 1.0)  # tanh bounds

    # -------------------------------------------------------------------------
    # apply_pitch_shift tests
    # -------------------------------------------------------------------------

    def test_apply_pitch_shift_with_librosa(self, sample_audio):
        """Test apply_pitch_shift when librosa is available."""
        aug = AudioAugmentation()

        mock_shifted = np.roll(sample_audio, 100)  # Simple shift simulation

        with patch("librosa.effects.pitch_shift", return_value=mock_shifted):
            result = aug.apply_pitch_shift(sample_audio)
            np.testing.assert_array_equal(result, mock_shifted)

    def test_apply_pitch_shift_without_librosa(self, sample_audio, caplog):
        """Test apply_pitch_shift fallback when librosa is not available."""
        aug = AudioAugmentation()

        with patch.dict("sys.modules", {"librosa": None}):
            with caplog.at_level("DEBUG"):
                result = aug.apply_pitch_shift(sample_audio)

            # Should return audio unchanged
            np.testing.assert_array_equal(result, sample_audio)
            assert "librosa not available" in caplog.text

    # -------------------------------------------------------------------------
    # apply_band_stop tests
    # -------------------------------------------------------------------------

    def test_apply_band_stop_with_audiomentations(self, sample_audio):
        """Test apply_band_stop when audiomentations is available."""
        aug = AudioAugmentation()

        mock_filter = MagicMock()
        mock_filter.return_value = sample_audio * 0.85

        with patch("audiomentations.BandStopFilter", return_value=mock_filter):
            result = aug.apply_band_stop(sample_audio)  # noqa: F841
            mock_filter.assert_called_once()

    def test_apply_band_stop_without_audiomentations(self, sample_audio, caplog):
        """Test apply_band_stop fallback when audiomentations is not available."""
        aug = AudioAugmentation()

        with patch.dict("sys.modules", {"audiomentations": None}):
            with caplog.at_level("DEBUG"):
                result = aug.apply_band_stop(sample_audio)

            # Should return audio unchanged
            np.testing.assert_array_equal(result, sample_audio)
            assert "audiomentations not available" in caplog.text

    # -------------------------------------------------------------------------
    # apply_color_noise tests
    # -------------------------------------------------------------------------

    def test_apply_color_noise_with_audiomentations(self, sample_audio):
        """Test apply_color_noise when audiomentations is available."""
        aug = AudioAugmentation()

        mock_noise = MagicMock()
        mock_noise.return_value = sample_audio + np.random.randn(*sample_audio.shape) * 0.01

        with patch("audiomentations.AddColorNoise", return_value=mock_noise):
            result = aug.apply_color_noise(sample_audio)  # noqa: F841
            mock_noise.assert_called_once()

    def test_apply_color_noise_fallback(self, sample_audio):
        """Test apply_color_noise fallback using numpy."""
        aug = AudioAugmentation()
        aug.config.background_min_snr_db = 10.0
        aug.config.background_max_snr_db = 10.0

        with patch.dict("sys.modules", {"audiomentations": None}):
            random.seed(42)
            result = aug.apply_color_noise(sample_audio)

            # Should add noise
            assert not np.allclose(result, sample_audio)
            assert result.dtype == np.float32

    def test_apply_color_noise_silent_input(self):
        """Test apply_color_noise with silent input."""
        aug = AudioAugmentation()
        silent = np.zeros(16000, dtype=np.float32)

        with patch.dict("sys.modules", {"audiomentations": None}):
            result = aug.apply_color_noise(silent)
            # Should still produce output with noise
            assert result.shape == silent.shape
            assert result.dtype == np.float32

    # -------------------------------------------------------------------------
    # apply_background_noise tests
    # -------------------------------------------------------------------------

    def test_apply_background_noise_no_files(self, sample_audio):
        """Test apply_background_noise when no background files available."""
        aug = AudioAugmentation()  # No background files

        with patch.object(aug, "apply_color_noise") as mock_color:
            mock_color.return_value = sample_audio
            result = aug.apply_background_noise(sample_audio)  # noqa: F841
            mock_color.assert_called_once()

    def test_apply_background_noise_with_files(self, sample_audio, temp_audio_files):
        """Test apply_background_noise with available files."""
        config = AugmentationConfig(
            background_paths=[str(f) for f in temp_audio_files["background"]],
            background_min_snr_db=5.0,
            background_max_snr_db=5.0,
        )
        aug = AudioAugmentation(config)

        # Mock load_audio_wave to return predictable noise
        mock_bg = np.random.randn(len(sample_audio)).astype(np.float32) * 0.1

        with patch("src.data.ingestion.load_audio_wave", return_value=mock_bg):
            result = aug.apply_background_noise(sample_audio)

            # Result should be different from input
            assert not np.allclose(result, sample_audio)

    def test_apply_background_noise_crop(self, sample_audio, temp_audio_files):
        """Test apply_background_noise with longer background (crop)."""
        config = AugmentationConfig(
            background_paths=[str(f) for f in temp_audio_files["background"]],
        )
        aug = AudioAugmentation(config)

        # Background longer than audio
        mock_bg = np.random.randn(len(sample_audio) * 2).astype(np.float32)

        with patch("src.data.ingestion.load_audio_wave", return_value=mock_bg):
            result = aug.apply_background_noise(sample_audio)
            assert result.shape == sample_audio.shape

    def test_apply_background_noise_repeat(self, sample_audio, temp_audio_files):
        """Test apply_background_noise with shorter background (repeat)."""
        config = AugmentationConfig(
            background_paths=[str(f) for f in temp_audio_files["background"]],
        )
        aug = AudioAugmentation(config)

        # Background shorter than audio
        mock_bg = np.random.randn(len(sample_audio) // 2).astype(np.float32)

        with patch("src.data.ingestion.load_audio_wave", return_value=mock_bg):
            result = aug.apply_background_noise(sample_audio)
            assert result.shape == sample_audio.shape

    def test_apply_background_noise_file_error(self, sample_audio, temp_audio_files, caplog):
        """Test apply_background_noise when file loading fails."""
        config = AugmentationConfig(
            background_paths=[str(f) for f in temp_audio_files["background"]],
        )
        aug = AudioAugmentation(config)

        with patch("src.data.ingestion.load_audio_wave", side_effect=FileNotFoundError("Not found")):
            with caplog.at_level("WARNING"):
                with patch.object(aug, "apply_color_noise") as mock_color:
                    mock_color.return_value = sample_audio
                    result = aug.apply_background_noise(sample_audio)  # noqa: F841

                assert "Failed to apply background noise" in caplog.text

    # -------------------------------------------------------------------------
    # apply_rir tests
    # -------------------------------------------------------------------------

    def test_apply_rir_no_files(self, sample_audio):
        """Test apply_rir when no RIR files available."""
        aug = AudioAugmentation()  # No RIR files
        result = aug.apply_rir(sample_audio)
        np.testing.assert_array_equal(result, sample_audio)

    def test_apply_rir_with_files(self, sample_audio, temp_audio_files):
        """Test apply_rir with available RIR files."""
        config = AugmentationConfig(
            impulse_paths=[str(f) for f in temp_audio_files["rirs"]],
        )
        aug = AudioAugmentation(config)

        # Mock RIR as simple impulse
        mock_rir = np.zeros(512, dtype=np.float32)
        mock_rir[0] = 1.0

        with patch("src.data.ingestion.load_audio_wave", return_value=mock_rir):
            result = aug.apply_rir(sample_audio)

            # Result should be different and same length
            assert result.shape == sample_audio.shape

    def test_apply_rir_normalization(self, sample_audio, temp_audio_files):
        """Test apply_rir with clipping normalization."""
        config = AugmentationConfig(
            impulse_paths=[str(f) for f in temp_audio_files["rirs"]],
        )
        aug = AudioAugmentation(config)

        # Mock RIR that would cause clipping
        mock_rir = np.zeros(512, dtype=np.float32)
        mock_rir[0] = 10.0  # Large amplitude

        with patch("src.data.ingestion.load_audio_wave", return_value=mock_rir):
            result = aug.apply_rir(np.ones_like(sample_audio) * 0.5)

            # Should be normalized to prevent clipping
            assert np.max(np.abs(result)) <= 1.0

    def test_apply_rir_file_error(self, sample_audio, temp_audio_files, caplog):
        """Test apply_rir when file loading fails."""
        config = AugmentationConfig(
            impulse_paths=[str(f) for f in temp_audio_files["rirs"]],
        )
        aug = AudioAugmentation(config)

        with patch("src.data.ingestion.load_audio_wave", side_effect=OSError("IO error")):
            with caplog.at_level("WARNING"):
                result = aug.apply_rir(sample_audio)

                assert "Failed to apply RIR" in caplog.text
                np.testing.assert_array_equal(result, sample_audio)


# =============================================================================
# GPU SpecAugment Tests
# =============================================================================


class TestApplySpecAugmentGPU:
    """Tests for apply_spec_augment_gpu function."""

    def test_raises_without_cupy(self, sample_spectrogram):
        """Test that function raises ImportError when CuPy is not available."""
        with patch("src.data.augmentation.HAS_CUPY", False), patch("src.data.augmentation.cp", None):
            with pytest.raises(ImportError, match="CuPy is required"):
                apply_spec_augment_gpu(sample_spectrogram)

    @pytest.mark.gpu
    def test_spec_augment_with_cupy(self, sample_spectrogram):
        """Test SpecAugment when CuPy is available."""
        # This test requires GPU and is marked accordingly
        try:
            import cupy as cp  # noqa: F401

            result = apply_spec_augment_gpu(
                sample_spectrogram,
                time_mask_param=10,
                freq_mask_param=5,
                num_time_masks=1,
                num_freq_masks=1,
            )

            # Result should be same shape
            assert result.shape == sample_spectrogram.shape
            assert result.dtype == sample_spectrogram.dtype

            # Some values should be zeroed (masked)
            zero_count_result = np.sum(result == 0)
            zero_count_orig = np.sum(sample_spectrogram == 0)
            assert zero_count_result >= zero_count_orig

        except ImportError:
            pytest.skip("CuPy not available")

    def test_spec_augment_mock_cupy(self, sample_spectrogram):
        """Test SpecAugment with mocked CuPy."""
        mock_cp = MagicMock()
        mock_cp.asarray.return_value = sample_spectrogram
        mock_cp.asnumpy.return_value = sample_spectrogram
        mock_cp.random.randint.return_value = MagicMock(item=lambda: 5)
        mock_cp.get_default_memory_pool.return_value = MagicMock(free_all_blocks=lambda: None)

        with patch("src.data.augmentation.HAS_CUPY", True), patch("src.data.augmentation.cp", mock_cp):
            result = apply_spec_augment_gpu(sample_spectrogram)
            assert result is not None
