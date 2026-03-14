"""Tests for data/preprocessing.py module.

Covers audio preprocessing functionality: WAV I/O, format conversion,
VAD trimming, and audio splitting.
"""

import array
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.preprocessing import (
    PreprocessResult,
    SpeechPreprocessConfig,
    SplitResult,
    _duration_ms_raw,
    _duration_ms_s16,
    _get_wav_duration_ms,
    _read_wav,
    _resample_linear_s16,
    _split_raw_frames,
    _split_raw_pcm,
    _stereo_to_mono_s16,
    _to_16khz_mono_s16,
    _vad_trim,
    _write_wav,
    find_speech_boundaries,
    process_background_directory,
    process_speech_directory,
    remove_split_originals,
    scan_and_split,
    split_file,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_wav_16khz_mono(tmp_path):
    """Create a sample mono WAV file at 16kHz."""
    wav_path = tmp_path / "test_mono.wav"
    sample_rate = 16000
    duration_sec = 1.0
    n_samples = int(sample_rate * duration_sec)

    # Generate sine wave
    import numpy as np

    t = np.linspace(0, duration_sec, n_samples)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return wav_path


@pytest.fixture
def sample_wav_stereo(tmp_path):
    """Create a sample stereo WAV file."""
    wav_path = tmp_path / "test_stereo.wav"
    sample_rate = 16000
    duration_sec = 0.5
    n_samples = int(sample_rate * duration_sec)

    import numpy as np

    t = np.linspace(0, duration_sec, n_samples)
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)

    # Interleave stereo samples
    stereo = np.empty(2 * n_samples, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    return wav_path


@pytest.fixture
def sample_wav_8khz(tmp_path):
    """Create a sample WAV file at 8kHz (needs resampling)."""
    wav_path = tmp_path / "test_8khz.wav"
    sample_rate = 8000
    duration_sec = 1.0
    n_samples = int(sample_rate * duration_sec)

    import numpy as np

    t = np.linspace(0, duration_sec, n_samples)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return wav_path


@pytest.fixture
def sample_wav_u8(tmp_path):
    """Create a sample WAV file with 8-bit samples."""
    wav_path = tmp_path / "test_u8.wav"
    sample_rate = 16000
    duration_sec = 0.5
    n_samples = int(sample_rate * duration_sec)

    import numpy as np

    t = np.linspace(0, duration_sec, n_samples)
    samples = ((np.sin(2 * np.pi * 440 * t) + 1) * 127.5).astype(np.uint8)

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return wav_path


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Create a temporary directory with multiple audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create several WAV files
    for i in range(3):
        wav_path = audio_dir / f"file_{i}.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # Different durations
            n_samples = int(16000 * (0.5 + i * 0.5))
            samples = array.array("h", [0] * n_samples)
            wf.writeframes(samples.tobytes())

    return audio_dir


# =============================================================================
# Configuration and Result Classes
# =============================================================================


class TestSpeechPreprocessConfig:
    """Tests for SpeechPreprocessConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SpeechPreprocessConfig()
        assert config.min_duration_ms == 300.0
        assert config.max_duration_ms == 2000.0
        assert config.pad_ms == 200
        assert config.vad_aggressiveness == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SpeechPreprocessConfig(
            min_duration_ms=500.0,
            max_duration_ms=3000.0,
            pad_ms=100,
            vad_aggressiveness=3,
        )
        assert config.min_duration_ms == 500.0
        assert config.max_duration_ms == 3000.0
        assert config.pad_ms == 100
        assert config.vad_aggressiveness == 3


class TestPreprocessResult:
    """Tests for PreprocessResult dataclass."""

    def test_creation(self):
        """Test PreprocessResult creation."""
        result = PreprocessResult(
            path=Path("/test/audio.wav"),
            action="trim",
            old_duration_ms=1500.0,
            new_duration_ms=1000.0,
            reason="VAD trimmed",
        )
        assert result.path == Path("/test/audio.wav")
        assert result.action == "trim"
        assert result.old_duration_ms == 1500.0
        assert result.new_duration_ms == 1000.0
        assert result.reason == "VAD trimmed"


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_creation(self):
        """Test SplitResult creation."""
        result = SplitResult(clips_written=5, clips_discarded=1)
        assert result.clips_written == 5
        assert result.clips_discarded == 1


# =============================================================================
# WAV I/O Tests
# =============================================================================


class TestReadWav:
    """Tests for _read_wav function."""

    def test_read_mono_16khz(self, sample_wav_16khz_mono):
        """Test reading mono 16kHz WAV file."""
        frames, rate, channels, width = _read_wav(sample_wav_16khz_mono)

        assert rate == 16000
        assert channels == 1
        assert width == 2
        assert len(frames) == 16000 * 2  # 1 second * 2 bytes/sample

    def test_read_stereo(self, sample_wav_stereo):
        """Test reading stereo WAV file."""
        frames, rate, channels, width = _read_wav(sample_wav_stereo)

        assert rate == 16000
        assert channels == 2
        assert width == 2

    def test_read_nonexistent(self):
        """Test reading non-existent file."""
        with pytest.raises((OSError, FileNotFoundError)):
            _read_wav(Path("/nonexistent/file.wav"))


class TestWriteWav:
    """Tests for _write_wav function."""

    def test_write_mono(self, tmp_path):
        """Test writing mono WAV file."""
        output_path = tmp_path / "output.wav"
        frames = array.array("h", [100, -100, 500, -500]).tobytes()

        _write_wav(output_path, frames, 16000, 1, 2)

        assert output_path.exists()

        # Verify by reading back
        frames_read, rate, channels, width = _read_wav(output_path)
        assert rate == 16000
        assert channels == 1
        assert width == 2
        assert frames_read == frames

    def test_write_creates_parents(self, tmp_path):
        """Test that writing creates parent directories."""
        output_path = tmp_path / "nested" / "deep" / "output.wav"
        frames = array.array("h", [100, -100]).tobytes()

        _write_wav(output_path, frames, 16000, 1, 2)

        assert output_path.exists()


class TestDurationMsRaw:
    """Tests for _duration_ms_raw function."""

    def test_mono_16bit(self):
        """Test duration calculation for mono 16-bit audio."""
        # 1 second of 16kHz mono 16-bit = 32000 bytes
        frames = b"\x00" * 32000
        duration = _duration_ms_raw(frames, 16000, 1, 2)
        assert duration == 1000.0

    def test_stereo_16bit(self):
        """Test duration calculation for stereo 16-bit audio."""
        # 1 second of 16kHz stereo 16-bit = 64000 bytes
        frames = b"\x00" * 64000
        duration = _duration_ms_raw(frames, 16000, 2, 2)
        assert duration == 1000.0


class TestDurationMsS16:
    """Tests for _duration_ms_s16 function."""

    def test_16khz_mono(self):
        """Test duration calculation for 16kHz mono s16 audio."""
        # 1 second = 32000 bytes (16000 samples * 2 bytes)
        pcm = b"\x00" * 32000
        duration = _duration_ms_s16(pcm, 16000)
        assert duration == 1000.0

    def test_default_rate(self):
        """Test duration calculation with default rate."""
        pcm = b"\x00" * 32000
        duration = _duration_ms_s16(pcm)
        assert duration == 1000.0


class TestGetWavDurationMs:
    """Tests for _get_wav_duration_ms function."""

    def test_valid_wav(self, sample_wav_16khz_mono):
        """Test getting duration of valid WAV file."""
        duration = _get_wav_duration_ms(sample_wav_16khz_mono)
        assert duration == 1000.0

    def test_nonexistent(self):
        """Test getting duration of non-existent file."""
        duration = _get_wav_duration_ms(Path("/nonexistent.wav"))
        assert duration == -1.0

    def test_invalid_file(self, tmp_path):
        """Test getting duration of invalid file."""
        invalid_file = tmp_path / "not_a_wav.txt"
        invalid_file.write_text("not audio data")
        duration = _get_wav_duration_ms(invalid_file)
        assert duration == -1.0


# =============================================================================
# Format Conversion Tests
# =============================================================================


class TestStereoToMonoS16:
    """Tests for _stereo_to_mono_s16 function."""

    def test_16bit_stereo(self):
        """Test converting 16-bit stereo to mono."""
        # Stereo samples: L0, R0, L1, R1
        stereo = array.array("h", [1000, 500, -1000, -500]).tobytes()
        mono = _stereo_to_mono_s16(stereo, 2)

        # Should average: (1000+500)/2=750, (-1000-500)/2=-750
        expected = array.array("h", [750, -750]).tobytes()
        assert mono == expected

    def test_8bit_stereo(self):
        """Test converting 8-bit stereo to mono."""
        # 8-bit stereo samples
        stereo = array.array("B", [100, 50, 200, 150]).tobytes()
        mono = _stereo_to_mono_s16(stereo, 1)

        expected = array.array("B", [75, 175]).tobytes()
        assert mono == expected


class TestResampleLinearS16:
    """Tests for _resample_linear_s16 function."""

    def test_same_rate(self):
        """Test resampling with same rate returns input."""
        pcm = array.array("h", [100, 200, 300, 400]).tobytes()
        result = _resample_linear_s16(pcm, 16000, 16000)
        assert result == pcm

    def test_upsample(self):
        """Test upsampling from 8kHz to 16kHz."""
        # 8kHz: 4 samples
        pcm = array.array("h", [0, 1000, 2000, 3000]).tobytes()
        result = _resample_linear_s16(pcm, 8000, 16000)

        # Should produce more samples
        result_array = array.array("h", result)
        assert len(result_array) == 8  # 4 * 2 = 8

    def test_downsample(self):
        """Test downsampling from 16kHz to 8kHz."""
        # 16kHz: 8 samples
        pcm = array.array("h", [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]).tobytes()
        result = _resample_linear_s16(pcm, 16000, 8000)

        # Should produce fewer samples
        result_array = array.array("h", result)
        assert len(result_array) == 4


class TestTo16khzMonoS16:
    """Tests for _to_16khz_mono_s16 function."""

    def test_already_16khz_mono_16bit(self, sample_wav_16khz_mono):
        """Test conversion of already-correct format."""
        frames, rate, channels, width = _read_wav(sample_wav_16khz_mono)
        result = _to_16khz_mono_s16(frames, rate, channels, width)

        # Should return frames unchanged
        assert result == frames

    def test_stereo_to_mono(self, sample_wav_stereo):
        """Test conversion from stereo to mono."""
        frames, rate, channels, width = _read_wav(sample_wav_stereo)
        result = _to_16khz_mono_s16(frames, rate, channels, width)

        # Should be mono
        assert len(result) == len(frames) // 2

    def test_8bit_to_16bit(self, sample_wav_u8):
        """Test conversion from 8-bit to 16-bit."""
        frames, rate, channels, width = _read_wav(sample_wav_u8)
        result = _to_16khz_mono_s16(frames, rate, channels, width)

        # Should be 16-bit (2 bytes per sample)
        assert len(result) == len(frames) * 2

    def test_resample_8khz_to_16khz(self, sample_wav_8khz):
        """Test resampling from 8kHz to 16kHz."""
        frames, rate, channels, width = _read_wav(sample_wav_8khz)
        result = _to_16khz_mono_s16(frames, rate, channels, width)

        # Should be 16kHz (approximately double the samples)
        assert len(result) > len(frames)


# =============================================================================
# Split Tests
# =============================================================================


class TestSplitRawPcm:
    """Tests for _split_raw_pcm function."""

    def test_exact_chunks(self):
        """Test splitting into exact chunks."""
        # 1 second at 16kHz mono 16-bit
        frames = b"\x00" * 32000
        chunks = _split_raw_pcm(frames, 16000, 1, 2, 500.0)  # 500ms chunks

        assert len(chunks) == 2
        assert len(chunks[0]) == 16000  # 500ms
        assert len(chunks[1]) == 16000

    def test_partial_chunk(self):
        """Test splitting with partial final chunk."""
        # 1.5 seconds
        frames = b"\x00" * 48000
        chunks = _split_raw_pcm(frames, 16000, 1, 2, 1000.0)  # 1s chunks

        assert len(chunks) == 2
        assert len(chunks[0]) == 32000  # 1s
        assert len(chunks[1]) == 16000  # 0.5s (remainder kept)


class TestSplitRawFrames:
    """Tests for _split_raw_frames function."""

    def test_minimum_duration_filter(self):
        """Test that chunks below minimum duration are filtered."""
        # 2 seconds of audio
        frames = b"\x00" * 64000
        chunks = _split_raw_frames(frames, 16000, 1, 2, target_duration_ms=500.0, min_duration_ms=400.0)

        # Should produce 4 chunks (500ms each)
        assert len(chunks) == 4

    def test_discards_short_final(self):
        """Test discarding final chunk if too short."""
        # 1.2 seconds
        frames = b"\x00" * 38400
        chunks = _split_raw_frames(frames, 16000, 1, 2, target_duration_ms=500.0, min_duration_ms=400.0)

        # 2 full chunks, final 200ms chunk discarded
        assert len(chunks) == 2


# =============================================================================
# File Processing Tests
# =============================================================================


class TestSplitFile:
    """Tests for split_file function."""

    def test_short_file_no_split(self, sample_wav_16khz_mono):
        """Test that short files are not split."""
        result = split_file(
            sample_wav_16khz_mono,
            max_duration_ms=2000.0,
            target_duration_ms=500.0,
            min_duration_ms=300.0,
        )

        assert result.clips_written == 0
        assert result.clips_discarded == 0

    def test_already_split_skipped(self, tmp_path):
        """Test that already split files are skipped."""
        # Create a long file with _part in name
        wav_path = tmp_path / "audio_part001.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 128000)  # 4 seconds

        # Create existing part files
        (tmp_path / "audio_part001_part001.wav").touch()

        result = split_file(
            wav_path,
            max_duration_ms=2000.0,
            target_duration_ms=500.0,
            min_duration_ms=300.0,
        )

        assert result.clips_written == 0
        assert result.clips_discarded == 0


class TestScanAndSplit:
    """Tests for scan_and_split function."""

    def test_empty_directory(self, tmp_path):
        """Test scanning empty directory."""
        total_long, total_written, total_discarded, total_skipped = scan_and_split(
            [tmp_path],
            max_duration_ms=2000.0,
            target_duration_ms=1000.0,
            min_duration_ms=500.0,
        )

        assert total_long == 0
        assert total_written == 0
        assert total_discarded == 0
        assert total_skipped == 0

    def test_nonexistent_directory(self):
        """Test scanning non-existent directory."""
        total_long, total_written, total_discarded, total_skipped = scan_and_split(
            [Path("/nonexistent")],
            max_duration_ms=2000.0,
            target_duration_ms=1000.0,
            min_duration_ms=500.0,
        )

        assert total_long == 0
        assert total_written == 0


class TestRemoveSplitOriginals:
    """Tests for remove_split_originals function."""

    def test_no_files_to_remove(self, tmp_path):
        """Test with no removable files."""
        removed = remove_split_originals([tmp_path], max_duration_ms=2000.0)
        assert removed == 0

    def test_removes_with_parts(self, tmp_path):
        """Test removing originals that have part files."""
        # Create long file
        wav_path = tmp_path / "long_audio.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 128000)  # 4 seconds

        # Create part files
        (tmp_path / "long_audio_part001.wav").touch()

        removed = remove_split_originals([tmp_path], max_duration_ms=2000.0)
        assert removed == 1
        assert not wav_path.exists()


# =============================================================================
# Directory Processing Tests
# =============================================================================


class TestProcessSpeechDirectory:
    """Tests for process_speech_directory function."""

    def test_empty_directory(self, tmp_path):
        """Test processing empty directory."""
        config = SpeechPreprocessConfig()
        discarded_root = tmp_path / "discarded"

        results = process_speech_directory(tmp_path, config, discarded_root, dry_run=True)

        assert results == []

    def test_skips_part_files(self, tmp_path):
        """Test that _part files are skipped."""
        # Create a file with _part in name
        wav_path = tmp_path / "audio_part001.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)

        config = SpeechPreprocessConfig()
        discarded_root = tmp_path / "discarded"

        results = process_speech_directory(tmp_path, config, discarded_root, dry_run=True)

        assert len(results) == 0  # _part files are filtered out


class TestProcessBackgroundDirectory:
    """Tests for process_background_directory function."""

    def test_empty_directory(self, tmp_path):
        """Test processing empty directory."""
        discarded_root = tmp_path / "discarded"

        results = process_background_directory(tmp_path, max_duration_ms=5000.0, discarded_root=discarded_root, dry_run=True)

        assert results == []

    def test_skips_part_files(self, tmp_path):
        """Test that _part files are skipped."""
        wav_path = tmp_path / "audio_part001.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)

        discarded_root = tmp_path / "discarded"

        results = process_background_directory(tmp_path, max_duration_ms=5000.0, discarded_root=discarded_root, dry_run=True)

        assert len(results) == 0


# =============================================================================
# VAD Tests (Mocked)
# =============================================================================


class TestFindSpeechBoundaries:
    """Tests for find_speech_boundaries function."""

    def test_no_webrtcvad(self):
        """Test behavior when webrtcvad is not available."""
        # Mock the optional deps to fail
        with patch("src.utils.optional_deps.require_optional") as mock_require:
            mock_require.side_effect = ImportError("webrtcvad not installed")

            pcm = b"\x00" * 9600  # 300ms at 16kHz 16-bit
            with pytest.raises(ImportError):
                find_speech_boundaries(pcm)

    def test_empty_pcm(self):
        """Test with empty PCM data."""
        mock_vad = MagicMock()

        with patch("src.utils.optional_deps.require_optional", return_value=mock_vad):
            result = find_speech_boundaries(b"")
            assert result is None

    def test_no_speech_detected(self):
        """Test when no speech is detected."""
        # The require_optional returns the webrtcvad module, not the class
        # We need to mock it so that webrtcvad.Vad() returns a mock instance
        mock_vad_module = MagicMock()
        mock_vad_instance = MagicMock()
        mock_vad_instance.is_speech.return_value = False
        mock_vad_module.Vad.return_value = mock_vad_instance

        with patch("src.utils.optional_deps.require_optional", return_value=mock_vad_module):
            # Create enough frames for VAD
            pcm = b"\x00" * 9600
            result = find_speech_boundaries(pcm)
            assert result is None


class TestVadTrim:
    """Tests for _vad_trim function."""

    def test_no_speech_returns_none(self):
        """Test that None is returned when no speech detected."""
        with patch("src.data.preprocessing.find_speech_boundaries") as mock_find:
            mock_find.return_value = None

            result = _vad_trim(b"test_pcm")
            assert result is None

    def test_trim_success(self):
        """Test successful trim."""
        with patch("src.data.preprocessing.find_speech_boundaries") as mock_find:
            # Return bounds that trim 10% from each end
            mock_find.return_value = (3200, 28800)  # 100ms-900ms of 1s

            pcm = b"A" * 32000  # 1 second
            result = _vad_trim(pcm)

            assert result == b"A" * 25600  # 800ms
