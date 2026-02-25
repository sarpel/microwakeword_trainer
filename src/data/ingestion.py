"""Data ingestion module for loading audio samples.

Provides:
- SampleRecord: Dataclass for audio sample metadata
- Clips: Loader with train/val/test splitting
- Audio validation utilities
- Data directory structure setup
"""

import logging
import os
import random
import struct
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class Split(Enum):
    """Data split type."""

    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class Label(Enum):
    """Sample label type."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    HARD_NEGATIVE = "hard_negative"
    BACKGROUND = "background"


# Standard audio validation constants
VALIDATION_SAMPLE_RATE = 16000
VALIDATION_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
VALIDATION_CHANNELS = 1  # Mono


# =============================================================================
# SAMPLE RECORD DATACLASS
# =============================================================================


@dataclass
class SampleRecord:
    """Record for a single audio sample.

    Attributes:
        path: Path to the audio file
        label: Label type (positive, negative, hard_negative, background)
        split: Data split (train, val, test)
        speaker_id: Optional speaker identifier for deduplication
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        weight: Sampling weight for training
    """

    path: Union[str, Path]
    label: Label
    split: Split
    speaker_id: Optional[str] = None
    duration_ms: float = 0.0
    sample_rate: int = VALIDATION_SAMPLE_RATE
    weight: float = 1.0

    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        self.path = Path(self.path)

        if not isinstance(self.label, Label):
            self.label = Label(self.label)

        if not isinstance(self.split, Split):
            self.split = Split(self.split)

        if self.sample_rate != VALIDATION_SAMPLE_RATE:
            logger.warning(
                f"Sample {self.path.name} has non-standard sample rate: "
                f"{self.sample_rate} Hz (expected {VALIDATION_SAMPLE_RATE})"
            )

    @property
    def audio_length_samples(self) -> int:
        """Get audio length in samples."""
        return int(self.duration_ms * self.sample_rate / 1000)

    @property
    def num_frames(self) -> int:
        """Get number of feature frames (at 10ms step)."""
        return int(self.duration_ms / 10)


# =============================================================================
# AUDIO VALIDATION
# =============================================================================


class AudioValidationError(Exception):
    """Raised when audio validation fails."""

    pass


def validate_audio_wave(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate audio file format and quality.

    Checks:
    - File exists
    - Is valid WAV format
    - Sample rate is 16000 Hz
    - Sample width is 16-bit (2 bytes)
    - Channels is mono (1)

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    try:
        with wave.open(str(file_path), "rb") as wf:
            # Check sample rate
            actual_rate = wf.getframerate()
            if actual_rate != VALIDATION_SAMPLE_RATE:
                return False, (
                    f"Invalid sample rate: {actual_rate} Hz "
                    f"(expected {VALIDATION_SAMPLE_RATE} Hz)"
                )

            # Check sample width (16-bit = 2 bytes)
            actual_width = wf.getsampwidth()
            if actual_width != VALIDATION_SAMPLE_WIDTH:
                return False, (
                    f"Invalid sample width: {actual_width} bytes "
                    f"(expected {VALIDATION_SAMPLE_WIDTH} bytes for 16-bit PCM)"
                )

            # Check channels (mono)
            actual_channels = wf.getnchannels()
            if actual_channels != VALIDATION_CHANNELS:
                return False, (
                    f"Invalid channels: {actual_channels} "
                    f"(expected {VALIDATION_CHANNELS} for mono)"
                )

            # Check that file has actual audio data
            nframes = wf.getnframes()
            if nframes == 0:
                return False, "Audio file contains no frames"

            # Sanity check: file shouldn't be absurdly large
            file_size = file_path.stat().st_size
            expected_max_size = (nframes * actual_width * actual_channels) + 44
            if file_size > expected_max_size * 2:
                logger.warning(
                    f"File size ({file_size}) much larger than expected "
                    f"({expected_max_size}), possible compression artifact"
                )

            return True, ""

    except wave.Error as e:
        return False, f"Invalid WAV file: {e}"
    except struct.error:
        return False, "File is not a valid WAV file"
    except Exception as e:
        return False, f"Error reading audio file: {e}"


def validate_audio(file_path: Union[str, Path]) -> bool:
    """Validate audio file format and quality.

    Args:
        file_path: Path to audio file

    Returns:
        True if valid, False otherwise

    Raises:
        AudioValidationError: If validation fails
    """
    is_valid, error_msg = validate_audio_wave(file_path)
    if not is_valid:
        raise AudioValidationError(error_msg)
    return True


def get_audio_info(file_path: Union[str, Path]) -> dict:
    """Get audio file information.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with audio metadata
    """
    file_path = Path(file_path)

    with wave.open(str(file_path), "rb") as wf:
        return {
            "path": str(file_path),
            "sample_rate": wf.getframerate(),
            "sample_width": wf.getsampwidth(),
            "channels": wf.getnchannels(),
            "frames": wf.getnframes(),
            "duration_s": wf.getnframes() / wf.getframerate(),
            "duration_ms": (wf.getnframes() / wf.getframerate()) * 1000,
        }


def load_audio_wave(
    file_path: Union[str, Path], target_sr: int = VALIDATION_SAMPLE_RATE
) -> np.ndarray:
    """Load audio file as numpy array.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (currently only 16000 supported)

    Returns:
        Audio array as float32 in range [-1, 1]
    """
    file_path = Path(file_path)

    validate_audio(file_path)

    with wave.open(str(file_path), "rb") as wf:
        # Read frames
        raw_data = wf.readframes(wf.getnframes())

        # Convert to numpy array
        audio = np.frombuffer(raw_data, dtype=np.int16)

        # Convert to float32 in range [-1, 1]
        audio = audio.astype(np.float32) / 32768.0

        return audio


# =============================================================================
# CLIPS LOADER
# =============================================================================


@dataclass
class ClipsLoaderConfig:
    """Configuration for Clips loader."""

    data_dir: Path
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    min_duration_ms: float = 500.0
    max_duration_ms: float = 3000.0
    speaker_based_split: bool = True


class Clips:
    """Loader for audio clips with train/val/test splitting.

    Loads audio files from data directory structure:
        data_dir/
            positive/      - Wake word samples
            negative/      - Background speech
            hard_negative/ - False positives
            background/   - Noise/ambient

    Attributes:
        samples: List of SampleRecord objects
        config: Loader configuration
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Optional[ClipsLoaderConfig] = None,
    ):
        """Initialize Clips loader.

        Args:
            data_dir: Root directory containing audio subdirectories
            config: Optional configuration (uses defaults if not provided)
        """
        self.data_dir = Path(data_dir)
        self.config = config or ClipsLoaderConfig(data_dir=self.data_dir)

        self.samples: List[SampleRecord] = []
        self._speaker_cache: dict = {}

        # Validate split ratios
        total_split = (
            self.config.train_split + self.config.val_split + self.config.test_split
        )
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")

        # Load samples
        self._discover_samples()

    def _discover_samples(self):
        """Discover and load all audio samples from data directory."""
        logger.info(f"Discovering samples in {self.data_dir}")

        # Mapping from directory name to label
        dir_to_label = {
            "positive": Label.POSITIVE,
            "negative": Label.NEGATIVE,
            "hard_negative": Label.HARD_NEGATIVE,
            "background": Label.BACKGROUND,
        }

        all_samples: List[SampleRecord] = []

        for dir_name, label in dir_to_label.items():
            dir_path = self.data_dir / dir_name

            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                continue

            # Find all WAV files
            wav_files = list(dir_path.glob("*.wav"))

            for wav_file in wav_files:
                try:
                    # Validate audio
                    is_valid, error_msg = validate_audio_wave(wav_file)
                    if not is_valid:
                        logger.warning(
                            f"Skipping invalid audio {wav_file}: {error_msg}"
                        )
                        continue

                    # Get audio info
                    info = get_audio_info(wav_file)

                    # Filter by duration
                    duration_ms = info["duration_ms"]
                    if duration_ms < self.config.min_duration_ms:
                        logger.debug(
                            f"Skipping too short audio {wav_file}: {duration_ms:.0f}ms"
                        )
                        continue
                    if duration_ms > self.config.max_duration_ms:
                        logger.debug(
                            f"Skipping too long audio {wav_file}: {duration_ms:.0f}ms"
                        )
                        continue

                    # Extract speaker ID from filename if present
                    # Pattern: hey_katya_YYYYMMDD_HHMMSS_vN.wav
                    speaker_id = self._extract_speaker_id(wav_file.name)

                    sample = SampleRecord(
                        path=wav_file,
                        label=label,
                        split=Split.TRAIN,  # Will be assigned during split
                        speaker_id=speaker_id,
                        duration_ms=duration_ms,
                        sample_rate=info["sample_rate"],
                    )
                    all_samples.append(sample)

                except Exception as e:
                    logger.warning(f"Error processing {wav_file}: {e}")
                    continue

        # Assign splits
        self.samples = self._assign_splits(all_samples)

        logger.info(
            f"Loaded {len(self.samples)} samples: "
            f"train={len(self.get_split(Split.TRAIN))}, "
            f"val={len(self.get_split(Split.VAL))}, "
            f"test={len(self.get_split(Split.TEST))}"
        )

    def _extract_speaker_id(self, filename: str) -> Optional[str]:
        """Extract speaker ID from filename.

        Args:
            filename: Audio filename

        Returns:
            Speaker ID if extractable, None otherwise
        """
        # Common pattern: wakeword_timestamp_variant.wav
        # Extract base name without variant
        parts = filename.replace(".wav", "").split("_")

        if len(parts) >= 2:
            # Return first two parts as speaker ID
            # e.g., "hey_katya_20260128_232017_v1.wav" -> "hey_katya"
            return "_".join(parts[:2])

        return None

    def _assign_splits(self, samples: List[SampleRecord]) -> List[SampleRecord]:
        """Assign samples to train/val/test splits.

        Args:
            samples: List of all samples

        Returns:
            Samples with split assigned
        """
        random.seed(self.config.seed)

        if self.config.speaker_based_split and any(s.speaker_id for s in samples):
            # Group by speaker
            speakers: dict = {}
            for sample in samples:
                sid = sample.speaker_id or "unknown"
                if sid not in speakers:
                    speakers[sid] = []
                speakers[sid].append(sample)

            # Shuffle speakers and assign splits
            speaker_ids = list(speakers.keys())
            random.shuffle(speaker_ids)

            n_speakers = len(speaker_ids)
            n_train = int(n_speakers * self.config.train_split)
            n_val = int(n_speakers * self.config.val_split)

            train_speakers = set(speaker_ids[:n_train])
            val_speakers = set(speaker_ids[n_train : n_train + n_val])

            result = []
            for sample in samples:
                sid = sample.speaker_id or "unknown"
                if sid in train_speakers:
                    sample.split = Split.TRAIN
                elif sid in val_speakers:
                    sample.split = Split.VAL
                else:
                    sample.split = Split.TEST
                result.append(sample)

        else:
            # Random split by sample
            shuffled = samples.copy()
            random.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * self.config.train_split)
            n_val = int(n * self.config.val_split)

            for i, sample in enumerate(shuffled):
                if i < n_train:
                    sample.split = Split.TRAIN
                elif i < n_train + n_val:
                    sample.split = Split.VAL
                else:
                    sample.split = Split.TEST

            result = shuffled

        return result

    def get_split(self, split: Split) -> List[SampleRecord]:
        """Get samples for a specific split.

        Args:
            split: Split type to retrieve

        Returns:
            List of samples in the split
        """
        return [s for s in self.samples if s.split == split]

    def get_by_label(self, label: Label) -> List[SampleRecord]:
        """Get samples by label.

        Args:
            label: Label type to filter

        Returns:
            List of samples with the label
        """
        return [s for s in self.samples if s.label == label]

    def get_split_by_label(
        self, split: Split, label: Optional[Label] = None
    ) -> List[SampleRecord]:
        """Get samples for a split, optionally filtered by label.

        Args:
            split: Split type
            label: Optional label filter

        Returns:
            List of matching samples
        """
        result = [s for s in self.samples if s.split == split]

        if label is not None:
            result = [s for s in result if s.label == label]

        return result

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleRecord:
        """Get sample by index."""
        return self.samples[idx]


# =============================================================================
# DATA DIRECTORY SETUP
# =============================================================================


def ensure_data_directory(base_dir: Union[str, Path], create: bool = True) -> dict:
    """Ensure data directory structure exists.

    Args:
        base_dir: Base directory for data
        create: If True, create directories that don't exist

    Returns:
        Dictionary mapping directory names to paths
    """
    base_dir = Path(base_dir)

    dirs = {
        "positive": base_dir / "positive",
        "negative": base_dir / "negative",
        "hard_negative": base_dir / "hard_negative",
        "background": base_dir / "background",
        "rirs": base_dir / "rirs",
    }

    if create:
        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {path}")

    return dirs


def get_data_statistics(clips: Clips) -> dict:
    """Get statistics about the loaded data.

    Args:
        clips: Clips loader instance

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_samples": len(clips),
        "by_split": {},
        "by_label": {},
        "by_split_label": {},
    }

    for split in Split:
        samples = clips.get_split(split)
        stats["by_split"][split.value] = len(samples)
        stats["by_split_label"][split.value] = {}

        for label in Label:
            label_samples = [s for s in samples if s.label == label]
            stats["by_split_label"][split.value][label.value] = len(label_samples)

    for label in Label:
        stats["by_label"][label.value] = len(clips.get_by_label(label))

    # Duration statistics
    durations = [s.duration_ms for s in clips.samples]
    stats["duration"] = {
        "min_ms": min(durations),
        "max_ms": max(durations),
        "mean_ms": np.mean(durations),
        "total_hours": sum(durations) / (1000 * 60 * 60),
    }

    return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_clips(
    data_dir: Union[str, Path],
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> Clips:
    """Convenience function to load audio clips.

    Args:
        data_dir: Root directory containing audio subdirectories
        train_split: Fraction for training set
        val_split: Fraction for validation set
        seed: Random seed for reproducibility

    Returns:
        Clips loader with samples loaded and split
    """
    config = ClipsLoaderConfig(
        data_dir=Path(data_dir),
        train_split=train_split,
        val_split=val_split,
        test_split=1.0 - train_split - val_split,
        seed=seed,
    )
    return Clips(data_dir, config)
