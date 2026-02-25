"""Feature extraction module using pymicro-features.

Provides:
- MicroFrontend: Integration with pymicro-features library
- SpectrogramGeneration: Generate mel spectrograms with sliding window
- FeatureConfig: Configuration for feature extraction
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

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
        # Validate sample rate
        if self.sample_rate != 16000:
            logger.warning(
                f"Non-standard sample rate: {self.sample_rate} Hz. "
                "Features are optimized for 16000 Hz."
            )

        # Validate window parameters
        if self.window_size_ms <= 0:
            raise ValueError("window_size_ms must be positive")
        if self.window_step_ms <= 0:
            raise ValueError("window_step_ms must be positive")
        if self.window_step_ms > self.window_size_ms:
            raise ValueError("window_step_ms must be <= window_size_ms")

        # Validate mel bins
        if self.mel_bins < 1:
            raise ValueError("mel_bins must be >= 1")

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
        Requires pymicro-features to be installed. Falls back to
        scipy-based implementation if not available.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize MicroFrontend feature extractor.

        Args:
            config: Feature configuration (uses defaults if not provided)
        """
        self.config = config or FeatureConfig()
        self._use_pymicro = self._check_pymicro_features()

    def _check_pymicro_features(self) -> bool:
        """Check if pymicro-features is available.

        Returns:
            True if pymicro-features is available
        """
        try:
            import pymicro_features

            logger.info("Using pymicro-features for feature extraction")
            return True
        except ImportError:
            logger.warning(
                "pymicro-features not available, falling back to scipy implementation"
            )
            return False

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio.

        Args:
            audio: Audio samples as float32 array in range [-1, 1]

        Returns:
            Mel spectrogram of shape (num_frames, mel_bins)
        """
        if self._use_pymicro:
            return self._compute_mel_spectrogram_pymicro(audio)
        else:
            return self._compute_mel_spectrogram_scipy(audio)

    def _compute_mel_spectrogram_pymicro(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram using pymicro-features.

        Args:
            audio: Audio samples

        Returns:
            Mel spectrogram
        """
        import pymicro_features

        # Ensure correct dtype
        audio = audio.astype(np.float32)

        # pymicro-features expects int16 input
        audio_int16 = (audio * 32767).astype(np.int16)

        # Get mel spectrogram
        mel_spec = pymicro_features.mel_feature(
            audio_int16,
            self.config.sample_rate,
            self.config.window_size_samples,
            self.config.window_step_samples,
            self.config.num_coeffs,
        )

        # Shape should be (frames, coeffs)
        return mel_spec.astype(np.float32)

    def _compute_mel_spectrogram_scipy(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram using scipy (fallback).

        Args:
            audio: Audio samples

        Returns:
            Mel spectrogram
        """
        from scipy import signal
        from scipy.signal import windows

        # Ensure correct dtype and range
        audio = audio.astype(np.float32)

        # Create Hann window
        window = windows.hann(self.config.window_size_samples)

        # Compute spectrogram using STFT
        frequencies, times, spec = signal.spectrogram(
            audio,
            fs=self.config.sample_rate,
            window=window,
            nperseg=self.config.window_size_samples,
            noverlap=self.config.window_size_samples - self.config.window_step_samples,
            mode="psd",
        )

        # Convert to mel scale
        mel_spec = self._freq_to_mel(frequencies, spec)

        # Trim to desired number of bins
        if mel_spec.shape[0] > self.config.mel_bins:
            mel_spec = mel_spec[: self.config.mel_bins, :]
        elif mel_spec.shape[0] < self.config.mel_bins:
            # Pad if necessary
            padding = np.zeros(
                (self.config.mel_bins - mel_spec.shape[0], mel_spec.shape[1])
            )
            mel_spec = np.vstack([mel_spec, padding])

        return mel_spec.T  # Transpose to (frames, bins)

    def _freq_to_mel(self, frequencies: np.ndarray, spec: np.ndarray) -> np.ndarray:
        """Convert frequency spectrogram to mel scale.

        Args:
            frequencies: Frequency bins
            spec: Power spectrogram

        Returns:
            Mel-scaled spectrogram
        """
        # Create mel filterbank
        num_bins = len(frequencies)
        mel_filters = self._create_mel_filterbank(
            self.config.mel_bins,
            num_bins,
            self.config.sample_rate,
        )

        # Apply mel filterbank
        mel_spec = np.dot(mel_filters, spec)

        # Log scale (with small offset to avoid log(0))
        mel_spec = np.log(mel_spec + 1e-10)

        return mel_spec

    def _create_mel_filterbank(
        self, num_filters: int, num_bins: int, sample_rate: int
    ) -> np.ndarray:
        """Create mel filterbank matrix.

        Args:
            num_filters: Number of mel filters
            num_bins: Number of frequency bins
            sample_rate: Sample rate

        Returns:
            Mel filterbank matrix of shape (num_filters, num_bins)
        """

        # Convert frequencies to mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Calculate mel frequency points
        low_mel = hz_to_mel(self.config.low_freq)
        high_mel = hz_to_mel(min(self.config.high_freq, sample_rate / 2))
        mel_points = np.linspace(low_mel, high_mel, num_filters + 2)

        # Convert back to Hz
        hz_points = mel_to_hz(mel_points)

        # Convert to frequency bin indices
        bin_points = np.floor((num_bins + 1) * hz_points / (sample_rate / 2)).astype(
            int
        )

        # Clip to valid range
        bin_points = np.clip(bin_points, 0, num_bins)

        # Create filterbank
        filterbank = np.zeros((num_filters, num_bins))
        for i in range(1, num_filters + 1):
            left = int(bin_points[i - 1])
            center = int(bin_points[i])
            right = int(bin_points[i + 1])

            # Ensure valid ranges
            left = max(0, min(left, num_bins - 1))
            center = max(0, min(center, num_bins - 1))
            right = max(0, min(right, num_bins))

            for j in range(left, center):
                if j < num_bins and center > left:
                    filterbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if j < num_bins and right > center:
                    filterbank[i - 1, j] = (right - j) / (right - center)

        return filterbank

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
        backend: str = "auto",
    ):
        """Initialize spectrogram generator.

        Args:
            config: Feature configuration
            backend: Backend to use ('auto', 'pymicro', 'scipy')
        """
        self.config = config or FeatureConfig()
        self._backend = backend
        self._generator: Optional[MicroFrontend] = None

        if backend == "auto":
            self._frontend = MicroFrontend(self.config)
        elif backend == "pymicro":
            self._frontend = MicroFrontend(self.config)
            if not self._frontend._use_pymicro:
                raise RuntimeError("pymicro-features not available")
        elif backend == "scipy":
            # Force scipy implementation
            self._frontend = MicroFrontend(self.config)
            self._frontend._use_pymicro = False
        else:
            raise ValueError(f"Unknown backend: {backend}")

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

        # Calculate number of frames
        num_frames = max(1, 1 + (actual_length - frame_length) // frame_step)

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

        audio = load_audio_wave(file_path)

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
        results = []

        for i in range(batch_size):
            spec = self.generate(audio_batch[i])
            results.append(spec)

        # Pad to same length if needed
        max_frames = max(spec.shape[0] for spec in results)

        padded = np.zeros((batch_size, max_frames, self.config.mel_bins))
        for i, spec in enumerate(results):
            padded[i, : spec.shape[0], :] = spec

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


def extract_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_size_ms: int = 30,
    window_step_ms: int = 10,
    mel_bins: int = 40,
) -> np.ndarray:
    """Extract acoustic features from audio.

    Convenience function for feature extraction.

    Args:
        audio: Audio samples as float32 array
        sample_rate: Sample rate in Hz
        window_size_ms: Window size in milliseconds
        window_step_ms: Step between windows in milliseconds
        mel_bins: Number of mel frequency bins

    Returns:
        Feature array of shape (num_frames, mel_bins)
    """
    config = FeatureConfig(
        sample_rate=sample_rate,
        window_size_ms=window_size_ms,
        window_step_ms=window_step_ms,
        mel_bins=mel_bins,
    )

    frontend = MicroFrontend(config)
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
