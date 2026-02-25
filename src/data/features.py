"""Feature extraction module using micro-features."""

import numpy as np


def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Extract acoustic features from audio.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Feature array
    """
    pass


def compute_mfcc(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Compute MFCC features.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        MFCC features
    """
    pass
