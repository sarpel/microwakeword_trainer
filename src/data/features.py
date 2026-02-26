"""Feature extraction module using pymicro-features.

Provides:
- MicroFrontend: Integration with pymicro-features library
- SpectrogramGeneration: Generate mel spectrograms with sliding window
- FeatureConfig: Configuration for feature extraction
"""

import functools
import logging
import math
import random
import logging
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================


@dataclass
class FeatureConfig:
    """Configuration for feature extraction.

    Attributes:
        sample_rate: Audio sample rate in Hz (default: 16000)
        window_size_ms: Window size in milliseconds (default: 30)
        window_step_ms: Step between windows in milliseconds (default: 10)
        mel_bins: Number of mel frequency bins (default: 40)
        num_coeffs: Number of coefficients to extract (default: 10)
        fft_size: FFT size for spectrogram computation (default: 512)
        low_freq: Lowest frequency in Hz (default: 0)
        high_freq: Highest frequency in Hz (default: 8000)
    """

    sample_rate: int = 16000
    window_size_ms: int = 30
    window_step_ms: int = 10
    mel_bins: int = 40
    num_coeffs: int = 10
    fft_size: int = 512
    low_freq: int = 0
    high_freq: int = 8000

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Warn on non-standard sample rate (informational only)
        if self.sample_rate != 16000:
            logger.warning(
                f"Non-standard sample rate: {self.sample_rate} Hz. "
                "Features are optimized for 16000 Hz."
            )

        # Delegate all validation to validate() - single source of truth
        issues = self.validate()
        if issues:
            raise ValueError(f"FeatureConfig validation failed: {'; '.join(issues)}")

        # Calculate derived values
        self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
        self.window_step_samples = int(self.sample_rate * self.window_step_ms / 1000)

    def get_frame_count(self, audio_length: int) -> int:
        """Calculate number of frames for given audio length.

        Args:
            audio_length: Length of audio in samples

        Returns:
            Number of frames
        """
        return max(
            1, 1 + (audio_length - self.window_size_samples) // self.window_step_samples
        )

    def validate(self) -> list:
        """Validate configuration and return list of issues.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if self.sample_rate < 1000:
            issues.append("sample_rate must be >= 1000 Hz")

        if self.window_size_ms <= 0:
            issues.append("window_size_ms must be positive")

        if self.window_step_ms <= 0:
            issues.append("window_step_ms must be positive")

        if self.window_step_ms > self.window_size_ms:
            issues.append("window_step_ms must be <= window_size_ms")

        if self.mel_bins < 1:
            issues.append("mel_bins must be >= 1")

        if self.high_freq <= self.low_freq:
            issues.append("high_freq must be > low_freq")

        if self.high_freq > self.sample_rate / 2:
            issues.append("high_freq must be <= sample_rate / 2")

        return issues


# =============================================================================
# MICROFRONTEND INTEGRATION
# =============================================================================


class MicroFrontend:
    """Interface to pymicro-features for audio feature extraction.

    Extracts mel-frequency cepstral coefficients (MFCCs) and mel
    spectrograms using the MicroFrontend algorithm.

    Note:
        Requires pymicro-features to be installed. GPU training pipeline
        requires this dependency - no CPU fallback is provided.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize MicroFrontend feature extractor.

        Args:
            config: Feature configuration (uses defaults if not provided)
        """
        self.config = config or FeatureConfig()
        self._check_pymicro_features()

    def _check_pymicro_features(self) -> None:
        """Check if pymicro-features is available.

        Raises:
            RuntimeError: If pymicro-features is not installed.
        """
        try:
            import pymicro_features  # noqa: F401

            logger.info("Using pymicro-features for feature extraction")
        except ImportError:
            raise RuntimeError(
                "pymicro-features is required for GPU training pipeline. "
                "Install: pip install pymicro-features"
            )

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio.

        Args:
            audio: Audio samples as float32 array in range [-1, 1]

        Returns:
            Mel spectrogram of shape (num_frames, mel_bins)
        """
        return self._compute_mel_spectrogram_pymicro(audio)

    def _compute_mel_spectrogram_pymicro(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram using pymicro-features.

        Uses the TFLite Micro audio frontend for ESPHome-compatible
        feature extraction (40 mel bins, 16kHz, 30ms window, 10ms step).

        Args:
            audio: Audio samples as float32 array in range [-1, 1]

        Returns:
            Mel spectrogram of shape (num_frames, mel_bins)
        """
        import pymicro_features

        # Initialize the MicroFrontend (lazy initialization for efficiency)
        if not hasattr(self, "_frontend"):
            self._frontend = pymicro_features.MicroFrontend(
                sample_rate=self.config.sample_rate,
                mel_bins=self.config.mel_bins,
                frame_length=self.config.window_size_samples,
                frame_shift=self.config.window_step_samples,
            )

        # Ensure correct dtype
        audio = audio.astype(np.float32)

        # Convert float32 [-1, 1] to 16-bit PCM
        audio_int16 = (audio * 32767.0).astype(np.int16)

        # Convert to bytes (little-endian, mono)
        audio_bytes = audio_int16.tobytes()

        # Process the entire audio - pymicro-features handles chunking internally
        # It processes 10ms frames (160 samples at 16kHz) and returns all frames
        output = self._frontend.process_samples(audio_bytes)

        # Convert to numpy array with correct shape (frames, mel_bins)
        mel_spec = np.array(output, dtype=np.float32)

        return mel_spec

    def _compute_mel_spectrogram_librosa(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram using librosa.

        Args:
            audio: Audio samples

        Returns:
            Mel spectrogram of shape (num_frames, mel_bins)
        """
        import librosa

        # Ensure correct dtype
        audio = audio.astype(np.float32)

        # Compute mel spectrogram using librosa
        # n_fft is typically 2^p where p is chosen so n_fft >= window_size_samples
        n_fft = 512  # Standard value for 16kHz audio with 30ms window
        hop_length = self.config.window_step_samples
        n_mels = self.config.mel_bins

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=self.config.sample_rate / 2,
        )

        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec.astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=self.config.sample_rate // 2,
        )

        # Transpose to get (frames, mel_bins) shape
        # librosa returns (mel_bins, frames)
        mel_spec = mel_spec.T

        # Convert to log scale (dB) - common practice for mel spectrograms
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db.astype(np.float32)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio.

        Alias for compute_mel_spectrogram.

        Args:
            audio: Audio samples

        Returns:
            Feature array of shape (num_frames, mel_bins)
        """
        return self.compute_mel_spectrogram(audio)


# =============================================================================
# SPECTROGRAM GENERATION WITH SLIDING WINDOW
# =============================================================================


class SpectrogramGeneration:
    """Generate spectrograms with sliding window for batch processing.

    This class provides utilities for generating spectrograms from audio
    with proper windowing and overlap handling.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
    ):
        """Initialize spectrogram generator.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self._frontend = MicroFrontend(self.config)

    @property
    def frame_size(self) -> int:
        """Get window size in samples."""
        return self.config.window_size_samples

    @property
    def frame_step(self) -> int:
        """Get step between frames in samples."""
        return self.config.window_step_samples

    @property
    def num_mel_bins(self) -> int:
        """Get number of mel frequency bins."""
        return self.config.mel_bins

    def slide_frames(
        self,
        audio: np.ndarray,
        frame_length: Optional[int] = None,
        frame_step: Optional[int] = None,
    ) -> np.ndarray:
        """Generate sliding window frames from audio.

        Creates overlapping frames from audio using the specified
        window size and step.

        Args:
            audio: Input audio array
            frame_length: Length of each frame in samples (default: window_size)
            frame_step: Step between frames in samples (default: window_step)

        Returns:
            Frames array of shape (num_frames, frame_length)
        """
        if frame_length is None:
            frame_length = self.config.window_size_samples
        if frame_step is None:
            frame_step = self.config.window_step_samples

        # Pad audio if necessary
        if len(audio) < frame_length:
            padding = frame_length - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")
            actual_length = len(audio)
        else:
            actual_length = len(audio)

        # Calculate number of frames using ceiling to include tail frame
        if actual_length >= frame_length:
            num_frames = int(math.ceil((actual_length - frame_length) / frame_step)) + 1
        else:
            num_frames = 1

        # Create frames
        frames = np.zeros((num_frames, frame_length), dtype=audio.dtype)

        for i in range(num_frames):
            start = i * frame_step
            end = start + frame_length
            if end <= actual_length:
                frames[i] = audio[start:end]
            else:
                # Last frame may be shorter, pad with zeros
                remaining = actual_length - start
                if remaining > 0:
                    frames[i, :remaining] = audio[start:]

        return frames

    def generate(self, audio: np.ndarray) -> np.ndarray:
        """Generate spectrogram from audio.

        Args:
            audio: Audio samples as float32 array

        Returns:
            Spectrogram of shape (num_frames, mel_bins)
        """
        return self._frontend.compute_mel_spectrogram(audio)

    def generate_from_file(
        self,
        file_path: str,
        target_length: Optional[int] = None,
    ) -> np.ndarray:
        """Generate spectrogram from audio file.

        Args:
            file_path: Path to audio file
            target_length: Optional target length in samples

        Returns:
            Spectrogram array
        """
        from .ingestion import load_audio_wave

        # Load audio with configured sample rate
        audio = load_audio_wave(file_path, target_sr=self.config.sample_rate)
        if target_length is not None:
            if len(audio) > target_length:
                # Truncate or random crop
                start = random.randint(0, len(audio) - target_length)
                audio = audio[start : start + target_length]
            elif len(audio) < target_length:
                # Pad
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

        return self.generate(audio)

    def process_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """Process a batch of audio samples.

        Args:
            audio_batch: Batch of audio samples, shape (batch, samples)

        Returns:
            Batch of spectrograms, shape (batch, frames, mel_bins)
        """
        batch_size = audio_batch.shape[0]
        if batch_size == 0:
            return np.zeros((0, 0, self.config.mel_bins), dtype=np.float32)

        results = []

        for i in range(batch_size):
            spec = self.generate(audio_batch[i])
            results.append(spec)

        if not results:
            return np.zeros((0, 0, self.config.mel_bins), dtype=np.float32)

        # Pad to same length if needed
        max_frames = max(spec.shape[0] for spec in results)

        # Use consistent dtype from results
        target_dtype = results[0].dtype
        padded = np.zeros(
            (batch_size, max_frames, self.config.mel_bins), dtype=target_dtype
        )
        for i, spec in enumerate(results):
            padded[i, : spec.shape[0], :] = spec.astype(target_dtype)

        return padded

    def get_expected_output_shape(self, audio_length: int) -> Tuple[int, int]:
        """Get expected output shape for given audio length.

        Args:
            audio_length: Length of input audio in samples

        Returns:
            Tuple of (num_frames, mel_bins)
        """
        num_frames = self.config.get_frame_count(audio_length)
        return (num_frames, self.config.mel_bins)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


@functools.lru_cache(maxsize=8)
def _get_cached_frontend(
    sample_rate: int,
    window_size_ms: int,
    window_step_ms: int,
    mel_bins: int,
    num_coeffs: int,
    fft_size: int,
    low_freq: int,
    high_freq: int,
) -> "MicroFrontend":
    """Return a cached MicroFrontend instance for the given config parameters.

    Thread-safe because Python's GIL protects lru_cache updates, and
    MicroFrontend.compute_mel_spectrogram holds no mutable state.
    """
    config = FeatureConfig(
        sample_rate=sample_rate,
        window_size_ms=window_size_ms,
        window_step_ms=window_step_ms,
        mel_bins=mel_bins,
        num_coeffs=num_coeffs,
        fft_size=fft_size,
        low_freq=low_freq,
        high_freq=high_freq,
    )
    return MicroFrontend(config)


def extract_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_size_ms: int = 30,
    window_step_ms: int = 10,
    mel_bins: int = 40,
) -> np.ndarray:
    """Extract acoustic features from audio.

    Convenience function for feature extraction.  Reuses a cached
    MicroFrontend instance so repeated calls with the same parameters
    do not incur construction overhead.

    Args:
        audio: Audio samples as float32 array
        sample_rate: Sample rate in Hz
        window_size_ms: Window size in milliseconds
        window_step_ms: Step between windows in milliseconds
        mel_bins: Number of mel frequency bins

    Returns:
        Feature array of shape (num_frames, mel_bins)
    """
    # Build a temporary config to derive default values for the un-exposed params
    config = FeatureConfig(
        sample_rate=sample_rate,
        window_size_ms=window_size_ms,
        window_step_ms=window_step_ms,
        mel_bins=mel_bins,
    )
    frontend = _get_cached_frontend(
        config.sample_rate,
        config.window_size_ms,
        config.window_step_ms,
        config.mel_bins,
        config.num_coeffs,
        config.fft_size,
        config.low_freq,
        config.high_freq,
    )
    return frontend.extract(audio)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_size_ms: int = 30,
    window_step_ms: int = 10,
    n_mels: int = 40,
) -> np.ndarray:
    """Compute mel spectrogram from audio.

    Alias for extract_features.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        window_size_ms: Window size in milliseconds
        window_step_ms: Step between windows in milliseconds
        n_mels: Number of mel bands

    Returns:
        Mel spectrogram
    """
    return extract_features(
        audio,
        sample_rate=sample_rate,
        window_size_ms=window_size_ms,
        window_step_ms=window_step_ms,
        mel_bins=n_mels,
    )
