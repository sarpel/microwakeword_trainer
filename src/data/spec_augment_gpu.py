"""
CuPy GPU-accelerated SpecAugment implementation for microwakeword_trainer.

GPU-mandatory: This module requires GPU availability and will raise RuntimeError
if GPU is not available. No CPU fallback is provided.
"""

import numpy as np

# Try to import CuPy and set up GPU availability flag
try:
    import cupy as cp

    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None  # type: ignore
    HAS_GPU = False


def spec_augment_gpu(
    spectrogram: np.ndarray,
    time_mask_max_size: int,
    time_mask_count: int,
    freq_mask_max_size: int,
    freq_mask_count: int,
) -> np.ndarray:
    """
    Apply GPU-accelerated SpecAugment to a single spectrogram.

    Args:
        spectrogram: Input mel spectrogram as numpy array with shape [time_frames, freq_bins]
        time_mask_max_size: Maximum size of time masks
        time_mask_count: Number of time masks to apply
        freq_mask_max_size: Maximum size of frequency masks
        freq_mask_count: Number of frequency masks to apply

    Returns:
        Augmented spectrogram as numpy array

    Raises:
        RuntimeError: If CuPy is not available or GPU is not accessible
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not available. Install cupy package: pip install cupy"
        )

    if not HAS_GPU:
        raise RuntimeError(
            "GPU is not available. This module requires GPU acceleration."
        )

    # Transfer to GPU
    spec_gpu = cp.asarray(spectrogram)

    # Get dimensions
    num_time_frames, num_freq_bins = spectrogram.shape

    # Apply frequency masks
    for _ in range(freq_mask_count):
        freq_mask_size = cp.random.randint(0, freq_mask_max_size + 1)
        freq_mask_start = cp.random.randint(
            0, max(1, num_freq_bins - freq_mask_size + 1)
        )
        spec_gpu[:, freq_mask_start : freq_mask_start + freq_mask_size] = 0

    # Apply time masks
    for _ in range(time_mask_count):
        time_mask_size = cp.random.randint(0, time_mask_max_size + 1)
        time_mask_start = cp.random.randint(
            0, max(1, num_time_frames - time_mask_size + 1)
        )
        spec_gpu[time_mask_start : time_mask_start + time_mask_size, :] = 0

    # Transfer back to CPU
    return cp.asnumpy(spec_gpu)


def batch_spec_augment_gpu(
    batch: np.ndarray,
    time_mask_max_size: int,
    time_mask_count: int,
    freq_mask_max_size: int,
    freq_mask_count: int,
) -> np.ndarray:
    """
    Apply GPU-accelerated SpecAugment to a batch of spectrograms.

    Applies different masks to each spectrogram in the batch using vectorized
    operations across the batch dimension.

    Args:
        batch: Input batch of mel spectrograms as numpy array with shape [batch_size, time_frames, freq_bins]
        time_mask_max_size: Maximum size of time masks
        time_mask_count: Number of time masks to apply
        freq_mask_max_size: Maximum size of frequency masks
        freq_mask_count: Number of frequency masks to apply

    Returns:
        Augmented batch as numpy array with same shape [batch_size, time_frames, freq_bins]

    Raises:
        RuntimeError: If CuPy is not available or GPU is not accessible
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not available. Install cupy package: pip install cupy"
        )

    if not HAS_GPU:
        raise RuntimeError(
            "GPU is not available. This module requires GPU acceleration."
        )

    # Transfer entire batch to GPU
    batch_gpu = cp.asarray(batch)

    # Get dimensions
    batch_size, num_time_frames, num_freq_bins = batch.shape

    # Apply frequency masks - different mask for each sample in batch
    for _ in range(freq_mask_count):
        freq_mask_sizes = cp.random.randint(0, freq_mask_max_size + 1, size=batch_size)
        freq_mask_starts = cp.random.randint(
            0, cp.maximum(1, num_freq_bins - freq_mask_sizes + 1)
        )

        # Vectorized frequency masking across batch
        for i in range(batch_size):
            mask_size = freq_mask_sizes[i]
            mask_start = freq_mask_starts[i]
            batch_gpu[i, :, mask_start : mask_start + mask_size] = 0

    # Apply time masks - different mask for each sample in batch
    for _ in range(time_mask_count):
        time_mask_sizes = cp.random.randint(0, time_mask_max_size + 1, size=batch_size)
        time_mask_starts = cp.random.randint(
            0, cp.maximum(1, num_time_frames - time_mask_sizes + 1)
        )

        # Vectorized time masking across batch
        for i in range(batch_size):
            mask_size = time_mask_sizes[i]
            mask_start = time_mask_starts[i]
            batch_gpu[i, mask_start : mask_start + mask_size, :] = 0

    # Transfer back to CPU
    return cp.asnumpy(batch_gpu)
