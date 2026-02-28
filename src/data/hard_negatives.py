"""Hard negative mining module for improving wake word detection.

Provides:
- HardNegativeMiningConfig: Configuration for mining
- HardNegativeMiner: Mines false positives during training
- integrate_hard_negative_mining: Helper to add mining to training loop
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.data.features import MicroFrontend


@dataclass
class HardNegativeMiningConfig:
    """Configuration for hard negative mining.

    Attributes:
        enabled: Whether to enable hard negative mining
        fp_threshold: Score threshold to consider a false positive (0-1)
        max_samples: Maximum number of hard negatives to keep
        mining_interval_epochs: Mine every N epochs
    """

    enabled: bool = True
    fp_threshold: float = 0.8
    max_samples: int = 5000
    mining_interval_epochs: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.fp_threshold <= 1.0:
            raise ValueError(f"fp_threshold must be between 0.0 and 1.0, got {self.fp_threshold}")
        if self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")
        if self.mining_interval_epochs <= 0:
            raise ValueError(f"mining_interval_epochs must be positive, got {self.mining_interval_epochs}")


class HardNegativeMiner:
    """Mines hard negatives (false positives) during training.

    Identifies samples that the model incorrectly classifies as wake words
    with high confidence, and adds them to the training set.

    Attributes:
        config: Mining configuration
        hard_negative_dir: Directory to store hard negatives
        mined_count: Total number of samples mined
    """

    def __init__(self, config: HardNegativeMiningConfig, hard_negative_dir: Path):
        """Initialize hard negative miner.

        Args:
            config: Mining configuration
            hard_negative_dir: Directory to store mined hard negatives
        """
        self.config = config
        self.hard_negative_dir = Path(hard_negative_dir)
        self.mined_count = 0
        self.feature_extractor: Optional[MicroFrontend] = None  # Lazy-initialized MicroFrontend

        # Ensure directory exists
        self.hard_negative_dir.mkdir(parents=True, exist_ok=True)

    def should_mine(self, epoch: int) -> bool:
        """Check if mining should be performed this epoch.

        Args:
            epoch: Current training epoch

        Returns:
            True if mining should be performed
        """
        if not self.config.enabled:
            return False

        return epoch > 0 and epoch % self.config.mining_interval_epochs == 0

    def mine_hard_negatives(self, model, audio_paths: List[Path], score_fn: Optional[Callable] = None) -> List[Path]:
        """Mine hard negatives from audio files.

        Args:
            model: Trained model for scoring
            audio_paths: List of audio files to check
            score_fn: Optional function to score audio (model, path) -> score

        Returns:
            List of hard negative audio paths
        """
        if not self.config.enabled:
            return []

        logger.info(f"Mining hard negatives from {len(audio_paths)} samples...")

        hard_negatives = []

        for audio_path in audio_paths:
            try:
                # Score the audio
                if score_fn:
                    score = score_fn(model, audio_path)
                else:
                    score = self._default_score_fn(model, audio_path)

                # Check if it's a false positive (high score but not wake word)
                if score >= self.config.fp_threshold:
                    hard_negatives.append((audio_path, score))

            except Exception as e:
                logger.warning(f"Failed to score {audio_path}: {e}")
                continue

        # Sort by score (highest first) and take top max_samples
        hard_negatives.sort(key=lambda x: x[1], reverse=True)
        hard_negatives = hard_negatives[: self.config.max_samples]

        # Copy to hard negative directory
        mined_paths = []
        for audio_path, score in hard_negatives:
            try:
                dest_path = self._copy_to_hard_negatives(audio_path, score)
                mined_paths.append(dest_path)
            except Exception as e:
                logger.warning(f"Failed to copy {audio_path}: {e}")

        self.mined_count += len(mined_paths)
        logger.info(f"Mined {len(mined_paths)} hard negatives (total: {self.mined_count})")

        return mined_paths

    def _default_score_fn(self, model, audio_path: Path) -> float:
        """Default scoring function using model inference.

        Args:
            model: Keras model
            audio_path: Path to audio file

        Returns:
            Model score (0-1)
        """
        from src.data.features import MicroFrontend
        from src.data.ingestion import load_audio_wave

        # Load audio
        audio = load_audio_wave(audio_path)

        # Extract features — reuse the extractor to avoid re-instantiation per call
        if self.feature_extractor is None:
            self.feature_extractor = MicroFrontend()
        features = self.feature_extractor.compute_mel_spectrogram(audio)

        # Adjust temporal dimension to match model input shape (safely)
        input_shape = getattr(model, "input_shape", None)
        if input_shape is not None and len(input_shape) > 1 and input_shape[1] is not None:
            expected_time = input_shape[1]
        else:
            # Fallback: leave features unchanged
            expected_time = features.shape[0]
        if features.shape[0] > expected_time:
            features = features[:expected_time, :]
        elif features.shape[0] < expected_time:
            padding = np.zeros(
                (expected_time - features.shape[0], features.shape[1]),
                dtype=features.dtype,
            )
            features = np.concatenate([features, padding], axis=0)

        # Add batch dimension
        features = np.expand_dims(features, axis=0)

        # Get model prediction
        prediction = np.asarray(model.predict(features, verbose=0)).ravel()
        if prediction.size == 0:
            logger.warning(f"Model returned empty prediction for {audio_path}")
            return 0.0
        score = float(prediction[0])

        return float(score)

    def _copy_to_hard_negatives(self, audio_path: Path, score: float) -> Path:
        """Copy audio file to hard negative directory.

        Args:
            audio_path: Source audio path
            score: Model score (used in filename)

        Returns:
            Destination path
        """
        import uuid

        # Create filename with score and unique suffix to prevent collisions
        base_name = audio_path.stem
        unique_suffix = uuid.uuid4().hex[:8]
        new_name = f"{base_name}_score{score:.3f}_{unique_suffix}.wav"
        dest_path = self.hard_negative_dir / new_name

        # Handle potential collision (should be rare with uuid)
        while dest_path.exists():
            unique_suffix = uuid.uuid4().hex[:8]
            new_name = f"{base_name}_score{score:.3f}_{unique_suffix}.wav"
            dest_path = self.hard_negative_dir / new_name

        # Copy file
        shutil.copy2(audio_path, dest_path)

        return dest_path

    def get_hard_negative_count(self) -> int:
        """Get current number of hard negatives in directory."""
        if not self.hard_negative_dir.exists():
            return 0
        return len(list(self.hard_negative_dir.glob("*.wav")))

    def cleanup_old_samples(self, keep_ratio: float = 0.5):
        """Remove oldest hard negatives to make room for new ones.

        Args:
            keep_ratio: Fraction of samples to keep (0-1)
        """
        if not self.hard_negative_dir.exists():
            return

        wav_files = list(self.hard_negative_dir.glob("*.wav"))

        if len(wav_files) <= self.config.max_samples:
            return

        # Sort by modification time (oldest first)
        wav_files.sort(key=lambda p: p.stat().st_mtime)

        # Calculate how many to remove
        n_keep = int(len(wav_files) * keep_ratio)
        n_remove = len(wav_files) - n_keep

        # Remove oldest files
        for wav_file in wav_files[:n_remove]:
            try:
                wav_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {wav_file}: {e}")

        logger.info(f"Cleaned up {n_remove} old hard negatives, kept {n_keep}")


def integrate_hard_negative_mining(
    trainer,
    config: HardNegativeMiningConfig,
    negative_audio_dir: Path,
    hard_negative_dir: Path,
    mining_interval_steps: int = 2500,  # Mine every N steps
):
    """Integrate hard negative mining into training loop.

    This is a helper function to add mining to an existing trainer.
    Mining is triggered at validation intervals based on steps.

    Args:
        trainer: Trainer object with train() method and validate method
        config: Mining configuration
        negative_audio_dir: Directory with negative samples to mine from
        hard_negative_dir: Directory to store mined hard negatives
        mining_interval_steps: Mine every N steps (default: 2500)
    """
    miner = HardNegativeMiner(config, hard_negative_dir)

    # Store original validate method
    original_validate = trainer.validate
    last_mining_step = 0

    def validate_with_mining(data_factory):
        """Wrapped validate method with mining at intervals."""
        nonlocal last_mining_step

        # Get current step
        current_step = getattr(trainer, "current_step", 0)

        # Mine hard negatives if it's time (based on steps)
        if current_step - last_mining_step >= mining_interval_steps:
            logger.info(f"Step {current_step}: Mining hard negatives...")

            # Get negative audio files
            negative_files = list(negative_audio_dir.rglob("*.wav"))

            if negative_files:
                # Cleanup old samples first
                miner.cleanup_old_samples()

                # Mine new hard negatives
                model = getattr(trainer, "model", None)
                if model:
                    miner.mine_hard_negatives(
                        model,
                        negative_files[:1000],  # Limit to avoid slow mining
                    )

            last_mining_step = current_step

        # Call original validate
        return original_validate(data_factory)

    # Replace validate method
    trainer.validate = validate_with_mining
    trainer.hard_negative_miner = miner

    logger.info("Hard negative mining integrated into trainer")
