"""Audio augmentation pipeline with configurable augmentation strategies."""

import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

# Import audiomentations for audio augmentations
try:
    from audiomentations import (
        AddBackgroundNoise,
        AddColorNoise,
        ApplyImpulseResponse,
        BandStopFilter,
        Gain,
        PitchShift,
        SevenBandParametricEQ,
        TanhDistortion,
    )
    from audiomentations.core.utils import calculate_desired_noise_rms

    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False


logger = logging.getLogger(__name__)


class AudioAugmentationPipeline:
    """Complete audio augmentation pipeline for wake word training.

    Integrates:
    - Background noise mixing
    - Room impulse response (RIR) augmentation
    - Various audio effects (EQ, distortion, pitch shift, filtering)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        augmentation_probabilities: Optional[dict] = None,
        background_min_snr_db: int = -5,
        background_max_snr_db: int = 10,
        min_jitter_s: float = 0.195,
        max_jitter_s: float = 0.205,
        impulse_paths: Optional[List[str]] = None,
        background_paths: Optional[List[str]] = None,
        augmentation_duration_s: float = 3.2,
    ):
        """Initialize augmentation pipeline.

        Args:
            sample_rate: Audio sample rate (default: 16000)
            augmentation_probabilities: Dict of augmentation name -> probability
            background_min_snr_db: Minimum SNR for background noise mixing
            background_max_snr_db: Maximum SNR for background noise mixing
            min_jitter_s: Minimum temporal jitter
            max_jitter_s: Maximum temporal jitter
            impulse_paths: Paths to impulse response files
            background_paths: Paths to background noise files
            augmentation_duration_s: Target duration for augmentation
        """
        self.sample_rate = sample_rate
        self.probabilities = augmentation_probabilities or {}
        self.background_min_snr_db = background_min_snr_db
        self.background_max_snr_db = background_max_snr_db
        self.min_jitter_s = min_jitter_s
        self.max_jitter_s = max_jitter_s
        self.augmentation_duration_s = augmentation_duration_s

        if not HAS_AUDIOMENTATIONS:
            raise ImportError(
                "audiomentations is required. Install: pip install audiomentations"
            )

        # Initialize augmentations based on config
        self._init_augmentations(impulse_paths or [], background_paths or [])

    def _init_augmentations(
        self, impulse_paths: List[str], background_paths: List[str]
    ) -> None:
        """Initialize audiomentations based on config probabilities."""
        self.augmentations = []

        # Configure each augmentation based on probability
        if self.probabilities.get("SevenBandParametricEQ", 0) > 0:
            self.augmentations.append(
                (
                    "SevenBandParametricEQ",
                    SevenBandParametricEQ(
                        p=self.probabilities["SevenBandParametricEQ"],
                        sample_rate=self.sample_rate,
                    ),
                )
            )

        if self.probabilities.get("TanhDistortion", 0) > 0:
            self.augmentations.append(
                (
                    "TanhDistortion",
                    TanhDistortion(
                        p=self.probabilities["TanhDistortion"],
                        sample_rate=self.sample_rate,
                    ),
                )
            )

        if self.probabilities.get("PitchShift", 0) > 0:
            self.augmentations.append(
                (
                    "PitchShift",
                    PitchShift(
                        p=self.probabilities["PitchShift"],
                        sample_rate=self.sample_rate,
                        min_semitones=-2,
                        max_semitones=2,
                    ),
                )
            )

        if self.probabilities.get("BandStopFilter", 0) > 0:
            self.augmentations.append(
                (
                    "BandStopFilter",
                    BandStopFilter(
                        p=self.probabilities["BandStopFilter"],
                        sample_rate=self.sample_rate,
                        min_center_freq=100,
                        max_center_freq=5000,
                    ),
                )
            )

        if self.probabilities.get("AddColorNoise", 0) > 0:
            self.augmentations.append(
                (
                    "AddColorNoise",
                    AddColorNoise(
                        p=self.probabilities["AddColorNoise"],
                        sample_rate=self.sample_rate,
                        min_snr_db=self.background_min_snr_db,
                        max_snr_db=self.background_max_snr_db,
                    ),
                )
            )

        if self.probabilities.get("AddBackgroundNoise", 0) > 0 and background_paths:
            self.augmentations.append(
                (
                    "AddBackgroundNoise",
                    AddBackgroundNoise(
                        p=self.probabilities["AddBackgroundNoise"],
                        sample_rate=self.sample_rate,
                        min_snr_db=self.background_min_snr_db,
                        max_snr_db=self.background_max_snr_db,
                        background_paths=background_paths,
                    ),
                )
            )

        if self.probabilities.get("RIR", 0) > 0 and impulse_paths:
            self.augmentations.append(
                (
                    "ApplyImpulseResponse",
                    ApplyImpulseResponse(
                        p=self.probabilities["RIR"],
                        sample_rate=self.sample_rate,
                        ir_paths=impulse_paths,
                    ),
                )
            )

        # Always add Gain as it has minimal impact
        gain_prob = self.probabilities.get("Gain", 1.0)
        self.augmentations.append(
            (
                "Gain",
                Gain(p=gain_prob, min_gain_db=-3.0, max_gain_db=3.0),
            )
        )

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to single audio sample.

        Args:
            audio: Audio samples as numpy array

        Returns:
            Augmented audio
        """
        augmented = audio.copy()

        for name, aug in self.augmentations:
            try:
                augmented = aug(augmented, sample_rate=self.sample_rate)
            except Exception as e:
                # Keep original if augmentation fails
                logger.exception("Augmentation '%s' failed: %s", name, e)

        return augmented

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation (callable interface)."""
        return self.augment(audio)


class ParallelAugmenter:
    """Parallel audio augmentation using thread pool."""

    def __init__(
        self, num_threads: int = 32, augmentation_fn: Optional[Callable] = None
    ):
        """Initialize parallel augmenter.

        Args:
            num_threads: Number of parallel threads
            augmentation_fn: Audio augmentation function to apply
        """
        self.num_threads = num_threads
        self.augmentation_fn = augmentation_fn
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    def augment_batch(
        self, audio_samples: List[np.ndarray], num_augmentations: int = 4
    ) -> List[np.ndarray]:
        """Apply augmentation to batch in parallel.

        Args:
            audio_samples: List of audio sample arrays
            num_augmentations: Number of augmentations to apply per sample

        Returns:
            List of augmented audio samples
        """
        if self.augmentation_fn is None:
            return audio_samples

        def augment_multi(audio: np.ndarray) -> List[np.ndarray]:
            """Create independent augmented variants for one input sample."""
            variants: List[np.ndarray] = []
            for _ in range(num_augmentations):
                variants.append(self.augmentation_fn(audio.copy()))
            return variants

        # Submit all tasks to the thread pool and collect in order
        futures = [
            self.executor.submit(augment_multi, audio) for audio in audio_samples
        ]
        augmented: List[np.ndarray] = []
        for i, future in enumerate(futures):
            try:
                augmented.extend(future.result())
            except Exception as exc:
                logger.exception(
                    "Parallel augmentation failed for sample %d: %s", i, exc
                )
        return augmented

    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
