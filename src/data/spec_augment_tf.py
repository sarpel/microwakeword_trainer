"""
TensorFlow-native SpecAugment implementation for microwakeword_trainer.

Pure TF implementation with no numpy dependencies.
"""

import tensorflow as tf


def spec_augment_tf(
    spectrogram: tf.Tensor,
    time_mask_max_size: int,
    time_mask_count: int,
    freq_mask_max_size: int,
    freq_mask_count: int,
    seed: int | None = None,
) -> tf.Tensor:
    """
    Apply SpecAugment to a single spectrogram using TensorFlow operations.

    Args:
        spectrogram: Input mel spectrogram with shape [time_frames, freq_bins]
        time_mask_max_size: Maximum size of time masks
        time_mask_count: Number of time masks to apply
        freq_mask_max_size: Maximum size of frequency masks
        freq_mask_count: Number of frequency masks to apply
        seed: Optional seed for random operations

    Returns:
        Augmented spectrogram with same shape and dtype as input
    """
    spec = spectrogram
    num_time_frames = tf.shape(spec)[0]
    num_freq_bins = tf.shape(spec)[1]

    # Use stateless random ops for determinism compatibility
    counter = tf.constant([0, 0], dtype=tf.int32)

    # Apply frequency masks (zero out frequency bins/columns)
    for i in range(freq_mask_count):
        seed_tensor = tf.constant([seed if seed is not None else i, i], dtype=tf.int32) if seed is not None else counter + i
        freq_mask_size = tf.random.stateless_uniform([], minval=0, maxval=freq_mask_max_size + 1, dtype=tf.int32, seed=seed_tensor)
        max_start = tf.maximum(1, num_freq_bins - freq_mask_size + 1)
        seed_tensor2 = tf.constant([seed if seed is not None else i + 1000, i], dtype=tf.int32) if seed is not None else counter + i + 1000
        freq_mask_start = tf.random.stateless_uniform([], minval=0, maxval=max_start, dtype=tf.int32, seed=seed_tensor2)

        # Create mask: 1 everywhere except masked columns which are 0
        col_indices = tf.range(num_freq_bins)
        mask_condition = tf.logical_and(col_indices >= freq_mask_start, col_indices < freq_mask_start + freq_mask_size)
        mask = tf.where(mask_condition, 0.0, 1.0)
        mask = tf.cast(mask, spec.dtype)
        mask = tf.expand_dims(mask, axis=0)  # [1, freq_bins]
        spec = spec * mask

    # Apply time masks (zero out time steps/rows)
    for i in range(time_mask_count):
        seed_tensor = tf.constant([seed if seed is not None else i + 2000, i], dtype=tf.int32) if seed is not None else counter + i + 2000
        time_mask_size = tf.random.stateless_uniform([], minval=0, maxval=time_mask_max_size + 1, dtype=tf.int32, seed=seed_tensor)
        max_start = tf.maximum(1, num_time_frames - time_mask_size + 1)
        seed_tensor2 = tf.constant([seed if seed is not None else i + 3000, i], dtype=tf.int32) if seed is not None else counter + i + 3000
        time_mask_start = tf.random.stateless_uniform([], minval=0, maxval=max_start, dtype=tf.int32, seed=seed_tensor2)

        # Create mask: 1 everywhere except masked rows which are 0
        row_indices = tf.range(num_time_frames)
        mask_condition = tf.logical_and(row_indices >= time_mask_start, row_indices < time_mask_start + time_mask_size)
        mask = tf.where(mask_condition, 0.0, 1.0)
        mask = tf.cast(mask, spec.dtype)
        mask = tf.expand_dims(mask, axis=1)  # [time_frames, 1]
        spec = spec * mask

    return spec


@tf.function
def batch_spec_augment_tf(
    batch: tf.Tensor,
    time_mask_max_size: int,
    time_mask_count: int,
    freq_mask_max_size: int,
    freq_mask_count: int,
    seed: int | None = None,
) -> tf.Tensor:
    """
    Apply SpecAugment to a batch of spectrograms using TensorFlow operations.

    Applies different masks to each spectrogram in the batch.

    Args:
        batch: Input batch of mel spectrograms with shape [batch_size, time_frames, freq_bins]
        time_mask_max_size: Maximum size of time masks
        time_mask_count: Number of time masks to apply
        freq_mask_max_size: Maximum size of frequency masks
        freq_mask_count: Number of frequency masks to apply
        seed: Optional seed for random operations

    Returns:
        Augmented batch with same shape [batch_size, time_frames, freq_bins] and dtype float32
    """
    batch_size = tf.shape(batch)[0]
    num_time_frames = tf.shape(batch)[1]
    num_freq_bins = tf.shape(batch)[2]

    result = batch

    # Use stateless random ops for determinism compatibility
    counter = tf.constant([0, 0], dtype=tf.int32)

    # Apply frequency masks - different mask for each sample in batch
    for i in range(freq_mask_count):
        # Per-sample random mask sizes (generate floats, scale to ints)
        seed1 = tf.constant([seed if seed is not None else i, i], dtype=tf.int32) if seed is not None else counter + i
        freq_mask_sizes_float = tf.random.stateless_uniform([batch_size], seed=seed1)
        freq_mask_sizes = tf.cast(freq_mask_sizes_float * tf.cast(freq_mask_max_size + 1, tf.float32), tf.int32)

        freq_mask_start_highs = tf.maximum(1, num_freq_bins - freq_mask_sizes + 1)
        seed2 = tf.constant([seed if seed is not None else i + 1000, i], dtype=tf.int32) if seed is not None else counter + i + 1000
        freq_mask_starts_float = tf.random.stateless_uniform([batch_size], seed=seed2)
        freq_mask_starts = tf.cast(freq_mask_starts_float * tf.cast(freq_mask_start_highs, tf.float32), tf.int32)

        # Build masks for each sample and apply
        # Create a mask tensor of shape [batch_size, 1, freq_bins]
        col_indices = tf.range(num_freq_bins)
        col_indices = tf.expand_dims(col_indices, axis=0)  # [1, freq_bins]
        col_indices = tf.expand_dims(col_indices, axis=0)  # [1, 1, freq_bins]
        col_indices = tf.tile(col_indices, [batch_size, 1, 1])  # [batch_size, 1, freq_bins]

        starts = tf.expand_dims(freq_mask_starts, axis=1)  # [batch_size, 1]
        starts = tf.expand_dims(starts, axis=1)  # [batch_size, 1, 1]
        sizes = tf.expand_dims(freq_mask_sizes, axis=1)  # [batch_size, 1]
        sizes = tf.expand_dims(sizes, axis=1)  # [batch_size, 1, 1]

        ends = starts + sizes

        mask_condition = tf.logical_and(col_indices >= starts, col_indices < ends)
        freq_mask = tf.where(mask_condition, 0.0, 1.0)  # [batch_size, 1, freq_bins]
        result = result * freq_mask

    # Apply time masks - different mask for each sample in batch
    for i in range(time_mask_count):
        # Per-sample random mask sizes (generate floats, scale to ints)
        seed1 = tf.constant([seed if seed is not None else i + 2000, 0], dtype=tf.int32) if seed is not None else counter + i + 2000
        time_mask_sizes_float = tf.random.stateless_uniform([batch_size], seed=seed1)
        time_mask_sizes = tf.cast(time_mask_sizes_float * tf.cast(time_mask_max_size + 1, tf.float32), tf.int32)

        time_mask_start_highs = tf.maximum(1, num_time_frames - time_mask_sizes + 1)
        seed2 = tf.constant([seed if seed is not None else i + 3000, 0], dtype=tf.int32) if seed is not None else counter + i + 3000
        time_mask_starts_float = tf.random.stateless_uniform([batch_size], seed=seed2)
        time_mask_starts = tf.cast(time_mask_starts_float * tf.cast(time_mask_start_highs, tf.float32), tf.int32)

        # Build masks for each sample and apply
        # Create a mask tensor of shape [batch_size, time_frames, 1]
        row_indices = tf.range(num_time_frames)
        row_indices = tf.expand_dims(row_indices, axis=0)  # [1, time_frames]
        row_indices = tf.expand_dims(row_indices, axis=2)  # [1, time_frames, 1]
        row_indices = tf.tile(row_indices, [batch_size, 1, 1])  # [batch_size, time_frames, 1]

        starts = tf.expand_dims(time_mask_starts, axis=1)  # [batch_size, 1]
        starts = tf.expand_dims(starts, axis=2)  # [batch_size, 1, 1]
        sizes = tf.expand_dims(time_mask_sizes, axis=1)  # [batch_size, 1]
        sizes = tf.expand_dims(sizes, axis=2)  # [batch_size, 1, 1]

        ends = starts + sizes

        mask_condition = tf.logical_and(row_indices >= starts, row_indices < ends)
        time_mask = tf.where(mask_condition, 0.0, 1.0)  # [batch_size, time_frames, 1]
        result = result * time_mask

    return result
