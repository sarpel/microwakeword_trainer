"""
CuPy GPU-accelerated SpecAugment implementation for microwakeword_trainer.

GPU-mandatory: This module requires GPU availability and will raise RuntimeError
if GPU is not available. No CPU fallback is provided.
"""

import logging
import warnings
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Suppress FutureWarning from experimental CuPy MemoryAsyncPool
# The code handles this gracefully with try/except, so we suppress the warning
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*MemoryAsyncPool.*experimental.*")

# Try to import CuPy and set up GPU availability flag
try:
    import cupy as cp

    HAS_GPU = cp.cuda.is_available()
    try:
        from cupy.cuda import MemoryAsyncPool

        cp.cuda.set_allocator(MemoryAsyncPool().malloc)
        logger.info("CuPy MemoryAsyncPool enabled for async memory allocation")
    except (ImportError, Exception) as e:
        logger.warning(f"CuPy MemoryAsyncPool not available (requires CUDA 11.2+): {e}")
except ImportError:
    cp = None
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
    if cp is None or not HAS_GPU:
        raise RuntimeError("GPU is required for spec_augment_gpu but CuPy/CUDA is not available. Please ensure a compatible GPU and CUDA installation.")

    # Transfer to GPU
    spec_gpu = cp.asarray(spectrogram)

    # Get dimensions
    num_time_frames, num_freq_bins = spectrogram.shape

    # Apply frequency masks
    for _ in range(freq_mask_count):
        freq_mask_size = cp.random.randint(0, freq_mask_max_size + 1)
        freq_mask_start = cp.random.randint(0, max(1, num_freq_bins - freq_mask_size + 1))
        spec_gpu[:, freq_mask_start : freq_mask_start + freq_mask_size] = 0

    # Apply time masks
    for _ in range(time_mask_count):
        time_mask_size = cp.random.randint(0, time_mask_max_size + 1)
        time_mask_start = cp.random.randint(0, max(1, num_time_frames - time_mask_size + 1))
        spec_gpu[time_mask_start : time_mask_start + time_mask_size, :] = 0

    # Transfer back to CPU
    spec_cpu = cast("np.ndarray[Any, Any]", cp.asnumpy(spec_gpu))
    del spec_gpu
    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory to prevent fragmentation

    return spec_cpu


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
    if cp is None or not HAS_GPU:
        raise RuntimeError("GPU is required for batch_spec_augment_gpu but CuPy/CUDA is not available. Please ensure a compatible GPU and CUDA installation.")

    # Transfer entire batch to GPU
    batch_gpu = cp.asarray(batch)

    # Get dimensions
    batch_size, num_time_frames, num_freq_bins = batch.shape

    # Apply frequency masks - fully vectorized across batch and time
    for _ in range(freq_mask_count):
        # Generate random mask sizes and starts for entire batch at once
        freq_mask_sizes = cp.random.randint(0, freq_mask_max_size + 1, size=batch_size)
        freq_max_starts = cp.maximum(1, num_freq_bins - freq_mask_sizes + 1)
        freq_mask_starts = (cp.random.random(batch_size) * freq_max_starts).astype(cp.int32)

        # Vectorized masking using advanced indexing
        # Build 2D per-sample frequency mask and broadcast across all time frames
        freq_indices = cp.arange(num_freq_bins)
        # mask shape: (batch_size, num_freq_bins)
        mask_2d = (freq_indices[None, :] >= freq_mask_starts[:, None]) & (freq_indices[None, :] < (freq_mask_starts + freq_mask_sizes)[:, None])
        # Expand to 3D: (batch_size, num_time_frames, num_freq_bins) and apply
        mask_3d = mask_2d[:, None, :]  # Broadcast across time dimension
        batch_gpu[mask_3d] = 0

    # Apply time masks - fully vectorized across batch and frequency
    for _ in range(time_mask_count):
        # Generate random mask sizes and starts for entire batch at once
        time_mask_sizes = cp.random.randint(0, time_mask_max_size + 1, size=batch_size)
        time_max_starts = cp.maximum(1, num_time_frames - time_mask_sizes + 1)
        time_mask_starts = (cp.random.random(batch_size) * time_max_starts).astype(cp.int32)

        # Vectorized time masking using advanced indexing
        # Build 2D per-sample time mask and broadcast across all frequency bins
        time_indices = cp.arange(num_time_frames)
        # mask shape: (batch_size, num_time_frames)
        mask_2d = (time_indices[None, :] >= time_mask_starts[:, None]) & (time_indices[None, :] < (time_mask_starts + time_mask_sizes)[:, None])
        # Expand to 3D: (batch_size, num_time_frames, num_freq_bins) and apply
        mask_3d = mask_2d[:, :, None]  # Broadcast across frequency dimension
        batch_gpu[mask_3d] = 0

    # Transfer back to CPU using synchronous call (async requires pinned memory)
    batch_cpu = cast("np.ndarray[Any, Any]", cp.asnumpy(batch_gpu))
    del batch_gpu
    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory to prevent fragmentation

    return batch_cpu
