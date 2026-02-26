"""Audio augmentation pipeline for wake word training.

Provides:
- AudioAugmentation: Complete augmentation pipeline
- Individual augmentations: EQ, distortion, pitch shift, etc.
- Config-driven augmentation from YAML
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation.

    All probabilities are 0.0-1.0 (0 = disabled, 1 = always apply)
    """

    # Time-domain augmentations
    SevenBandParametricEQ: float = 0.1
    TanhDistortion: float = 0.1
    PitchShift: float = 0.1
    BandStopFilter: float = 0.1
    AddColorNoise: float = 0.1
    AddBackgroundNoise: float = 0.75
    Gain: float = 1.0
    RIR: float = 0.5

    # Additional augmentations
    AddBackgroundNoiseFromFile: float = 0.0
    ApplyImpulseResponse: float = 0.0

    # Noise mixing parameters
    background_min_snr_db: float = -5.0
    background_max_snr_db: float = 10.0
    min_jitter_s: float = 0.195
    max_jitter_s: float = 0.205

    # Background sources
    impulse_paths: Optional[List[str]] = None
    background_paths: Optional[List[str]] = None
    augmentation_duration_s: float = 3.2


class AudioAugmentation:
    """Audio augmentation pipeline.

    Applies configured augmentations to audio samples during training.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self.sample_rate = 16000

        # Load background noise files
        self.background_noise_files: List[Path] = []
        self.rir_files: List[Path] = []
        self._load_background_files()

    def _load_background_files(self):
        """Load background noise and RIR file lists."""
        # Load background noise files
        if self.config.background_paths:
            for bg_path in self.config.background_paths:
                path = Path(bg_path)
                if path.exists():
                    if path.is_dir():
                        self.background_noise_files.extend(path.glob("*.wav"))
                    else:
                        self.background_noise_files.append(path)

        # Load RIR files
        if self.config.impulse_paths:
            for rir_path in self.config.impulse_paths:
                path = Path(rir_path)
                if path.exists():
                    if path.is_dir():
                        self.rir_files.extend(path.glob("*.wav"))
                    else:
                        self.rir_files.append(path)

        logger.info(
            f"Loaded {len(self.background_noise_files)} background noise files, "
            f"{len(self.rir_files)} RIR files"
        )

    def __call__(self, audio: np.ndarray, apply_all: bool = False) -> np.ndarray:
        """Apply augmentations to audio.

        Args:
            audio: Audio array (samples,)
            apply_all: If True, apply all augmentations regardless of probability

        Returns:
            Augmented audio
        """
        augmented = audio.copy()

        # Always apply gain
        if apply_all or random.random() < self.config.Gain:
            augmented = self.apply_gain(augmented)

        # Apply EQ
        if apply_all or random.random() < self.config.SevenBandParametricEQ:
            augmented = self.apply_eq(augmented)

        # Apply distortion
        if apply_all or random.random() < self.config.TanhDistortion:
            augmented = self.apply_distortion(augmented)

        # Apply pitch shift
        if apply_all or random.random() < self.config.PitchShift:
            augmented = self.apply_pitch_shift(augmented)

        # Apply band stop filter
        if apply_all or random.random() < self.config.BandStopFilter:
            augmented = self.apply_band_stop(augmented)

        # Apply color noise
        if apply_all or random.random() < self.config.AddColorNoise:
            augmented = self.apply_color_noise(augmented)

        # Apply background noise
        if apply_all or random.random() < self.config.AddBackgroundNoise:
            augmented = self.apply_background_noise(augmented)

        # Apply RIR
        if apply_all or random.random() < self.config.RIR:
            augmented = self.apply_rir(augmented)

        return augmented

    def apply_gain(
        self, audio: np.ndarray, min_db: float = -3.0, max_db: float = 3.0
    ) -> np.ndarray:
        """Apply random gain adjustment.

        Args:
            audio: Input audio
            min_db: Minimum gain in dB
            max_db: Maximum gain in dB

        Returns:
            Gain-adjusted audio
        """
        gain_db = random.uniform(min_db, max_db)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def apply_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply 7-band parametric EQ.

        Args:
            audio: Input audio

        Returns:
            EQ'd audio
        """
        try:
            import audiomentations

            augmenter = audiomentations.SevenBandParametricEQ(
                min_gain_db=-6.0, max_gain_db=6.0, p=1.0
            )
            return augmenter(samples=audio, sample_rate=self.sample_rate)
        except ImportError:
            logger.debug("audiomentations not available, skipping EQ")
            return audio

    def apply_distortion(self, audio: np.ndarray) -> np.ndarray:
        """Apply tanh distortion.

        Args:
            audio: Input audio

        Returns:
            Distorted audio
        """
        try:
            import audiomentations

            augmenter = audiomentations.TanhDistortion(
                min_distortion=0.1, max_distortion=0.5, p=1.0
            )
            return augmenter(samples=audio, sample_rate=self.sample_rate)
        except ImportError:
            # Simple tanh fallback
            drive = random.uniform(0.5, 2.0)
            return np.tanh(audio * drive)

    def apply_pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shift.

        Args:
            audio: Input audio

        Returns:
            Pitch-shifted audio
        """
        try:
            import librosa

            n_steps = random.uniform(-2, 2)
            return librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=n_steps
            )
        except ImportError:
            logger.debug("librosa not available, skipping pitch shift")
            return audio

    def apply_band_stop(self, audio: np.ndarray) -> np.ndarray:
        """Apply band stop filter.

        Args:
            audio: Input audio

        Returns:
            Filtered audio
        """
        try:
            import audiomentations

            augmenter = audiomentations.BandStopFilter(
                min_center_freq=200,
                max_center_freq=4000,
                min_bandwidth_fraction=0.5,
                max_bandwidth_fraction=1.99,
                p=1.0,
            )
            return augmenter(samples=audio, sample_rate=self.sample_rate)
        except ImportError:
            logger.debug("audiomentations not available, skipping band stop")
            return audio

    def apply_color_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply colored noise.

        Args:
            audio: Input audio

        Returns:
            Noisy audio
        """
        try:
            import audiomentations

            augmenter = audiomentations.AddColorNoise(
                min_snr_db=self.config.background_min_snr_db,
                max_snr_db=self.config.background_max_snr_db,
                p=1.0,
            )
            return augmenter(samples=audio, sample_rate=self.sample_rate)
        except ImportError:
            # Simple white noise fallback
            snr_db = random.uniform(
                self.config.background_min_snr_db, self.config.background_max_snr_db
            )
            signal_power = np.mean(audio**2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
            return audio + noise

    def apply_background_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply background noise from files.

        Args:
            audio: Input audio

        Returns:
            Noisy audio
        """
        if not self.background_noise_files:
            return self.apply_color_noise(audio)

        # Select random background file
        bg_file = random.choice(self.background_noise_files)

        try:
            from src.data.ingestion import load_audio_wave

            # Load background
            bg_audio = load_audio_wave(bg_file)

            # Random crop or repeat to match length
            target_len = len(audio)
            if len(bg_audio) < target_len:
                # Repeat to fill
                repeats = int(np.ceil(target_len / len(bg_audio)))
                bg_audio = np.tile(bg_audio, repeats)[:target_len]
            else:
                # Random crop
                start = random.randint(0, len(bg_audio) - target_len)
                bg_audio = bg_audio[start : start + target_len]

            # Mix with SNR
            snr_db = random.uniform(
                self.config.background_min_snr_db, self.config.background_max_snr_db
            )

            signal_power = np.mean(audio**2)
            noise_power = np.mean(bg_audio**2)

            if noise_power > 0:
                # Scale noise to achieve target SNR
                target_noise_power = signal_power / (10 ** (snr_db / 10))
                bg_audio = bg_audio * np.sqrt(target_noise_power / noise_power)

            return audio + bg_audio

        except Exception as e:
            logger.warning(f"Failed to apply background noise: {e}")
            return self.apply_color_noise(audio)

    def apply_rir(self, audio: np.ndarray) -> np.ndarray:
        """Apply room impulse response.

        Args:
            audio: Input audio

        Returns:
            Reverberant audio
        """
        if not self.rir_files:
            return audio

        # Select random RIR file
        rir_file = random.choice(self.rir_files)

        try:
            from src.data.ingestion import load_audio_wave

            # Load RIR
            rir = load_audio_wave(rir_file)

            # Convolve
            convolved = np.convolve(audio, rir, mode="full")[: len(audio)]

            # Normalize to prevent clipping
            max_val = np.max(np.abs(convolved))
            if max_val > 0.99:
                convolved = convolved * (0.99 / max_val)

            return convolved

        except Exception as e:
            logger.warning(f"Failed to apply RIR: {e}")
            return audio
