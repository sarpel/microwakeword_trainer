"""Optimized TensorFlow data pipeline using tf.data.Dataset.

Provides high-performance data loading with caching, prefetching, and parallel processing.
Compatible with ESPHome - only affects training speed, not model export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import tensorflow as tf

from src.data.dataset import WakeWordDataset
from src.data.spec_augment_tf import batch_spec_augment_tf

logger = logging.getLogger(__name__)


class OptimizedDataPipeline:
    """High-performance tf.data.Dataset pipeline for wake word training.

    This pipeline provides:
    - Parallel data loading (multi-threaded)
    - Automatic caching (disk and memory)
    - Prefetching (GPU never waits)
    - Vectorized operations
    - Optimal batching

    Note: This only affects training speed. The exported TFLite model
    is identical regardless of data pipeline used.

    Example:
        pipeline = OptimizedDataPipeline(dataset, config)
        train_ds = pipeline.create_training_pipeline()

        # Use with model.fit()
        model.fit(train_ds, epochs=10)
    """

    def __init__(
        self,
        dataset: WakeWordDataset,
        config: dict,
        batch_size: int | None = None,
        max_time_frames: int | None = None,
        spec_augment_config: dict | None = None,
    ):
        """Initialize optimized pipeline.

        Args:
            dataset: WakeWordDataset instance
            config: Training configuration
            batch_size: Batch size (overrides config if provided)
            max_time_frames: Max time frames for padding
        """
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size or config.get("training", {}).get("batch_size", 128)
        self.max_time_frames = max_time_frames or self._calculate_max_frames()
        self.spec_augment_config = spec_augment_config or {}

        # Performance settings
        self.autotune = tf.data.AUTOTUNE
        self.prefetch_buffer = config.get("performance", {}).get("prefetch_buffer", 2)
        self.cache_dir = config.get("performance", {}).get("tfdata_cache_dir")
        self.prefetch_to_device = config.get("performance", {}).get("tfdata_prefetch_to_device", True)
        self.prefetch_device = config.get("performance", {}).get("tfdata_prefetch_device", "/GPU:0")

    def _calculate_max_frames(self) -> int:
        """Calculate max time frames from hardware config."""
        hardware = self.config.get("hardware", {})
        clip_duration_ms = hardware.get("clip_duration_ms", 1000)
        window_step_ms = hardware.get("window_step_ms", 10)
        return int(clip_duration_ms / window_step_ms)

    def _generator_factory(
        self,
        split: str = "train",
    ) -> Callable[[], Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        """Create generator factory for tf.data.Dataset.

        Args:
            split: 'train' or 'val'

        Returns:
            Generator function that yields (features, labels, sample_weights, is_hard_neg)
        """

        def generator():
            if split == "train":
                factory = self.dataset.train_generator_factory(self.max_time_frames)
            else:
                factory = self.dataset.val_generator_factory(self.max_time_frames)

            gen = factory()
            for batch in gen:
                if not isinstance(batch, tuple) or len(batch) < 2:
                    continue
                features = batch[0]
                labels = batch[1]
                # Yield 4-tuple: (features, labels, sample_weights, is_hard_neg)
                batch_size = features.shape[0]
                sample_weights = np.ones(batch_size, dtype=np.float32)
                is_hard_neg = np.zeros(batch_size, dtype=np.bool_)
                yield features, labels, sample_weights, is_hard_neg

        return generator

    def create_training_pipeline(
        self,
        cache_dir: str | None = None,
        shuffle_buffer: int = 10000,
        shuffle_seed: int | None = None,
    ) -> tf.data.Dataset:
        """Create optimized training data pipeline.

        Pipeline stages:
        1. Generator → tf.data.Dataset
        2. Shuffle (for training)
        3. Cache (disk or memory)
        4. Batch (with padding if needed)
        5. Prefetch (to GPU)

        Args:
            cache_dir: Directory for disk cache (None = memory cache)
            shuffle_buffer: Shuffle buffer size

        Returns:
            tf.data.Dataset ready for training
        """
        # Get generator
        generator = self._generator_factory("train")

        # Create dataset from generator
        # Output signature must match generator output
        output_signature = (
            tf.TensorSpec(shape=(None, self.max_time_frames, 40), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
        )

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        # Cache first (before shuffle for proper training behavior)
        resolved_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        if resolved_cache_dir:
            cache_path = Path(resolved_cache_dir)
            if cache_path.suffix:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                cache_path.mkdir(parents=True, exist_ok=True)
            ds = ds.cache(str(cache_path))
            logger.info(f"Using disk cache: {cache_path}")
        else:
            ds = ds.cache()
            logger.info("Using memory cache")

        # Shuffle for training (after cache to reshuffle each epoch)
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed, reshuffle_each_iteration=True)

        # Apply SpecAugment if enabled (TF backend)
        if self.spec_augment_config.get("enabled", False):
            backend = self.spec_augment_config.get("backend", "cupy")
            if backend == "tf":
                time_mask_max_size = self.spec_augment_config.get("time_mask_max_size", 10)
                time_mask_count = self.spec_augment_config.get("time_mask_count", 2)
                freq_mask_max_size = self.spec_augment_config.get("freq_mask_max_size", 10)
                freq_mask_count = self.spec_augment_config.get("freq_mask_count", 2)
                seed = self.spec_augment_config.get("seed")

                def apply_spec_augment(features, labels, sample_weights, is_hard_neg):
                    augmented_features = batch_spec_augment_tf(
                        features,
                        time_mask_max_size=time_mask_max_size,
                        time_mask_count=time_mask_count,
                        freq_mask_max_size=freq_mask_max_size,
                        freq_mask_count=freq_mask_count,
                        seed=seed,
                    )
                    return augmented_features, labels, sample_weights, is_hard_neg

                ds = ds.map(apply_spec_augment, num_parallel_calls=self.autotune)
                logger.info(f"SpecAugment (TF backend) enabled: time_masks={time_mask_count}@{time_mask_max_size}, freq_masks={freq_mask_count}@{freq_mask_max_size}")

        # No extra batching (generator already yields full batches)

        # Prefetch to GPU
        ds = ds.prefetch(buffer_size=self.autotune)
        if self.prefetch_to_device:
            try:
                ds = tf.data.experimental.prefetch_to_device(self.prefetch_device)(ds)
                logger.info(f"Prefetching dataset to device: {self.prefetch_device}")
            except Exception as e:
                logger.warning(f"GPU prefetch disabled: {e}")

        return ds

    def create_validation_pipeline(
        self,
        cache_dir: str | None = None,
    ) -> tf.data.Dataset:
        """Create optimized validation data pipeline.

        Similar to training but without shuffling.

        Args:
            cache_dir: Directory for disk cache

        Returns:
            tf.data.Dataset ready for validation
        """
        generator = self._generator_factory("val")

        output_signature = (
            tf.TensorSpec(shape=(None, self.max_time_frames, 40), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
        )

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        # Cache (no shuffle for validation)
        resolved_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        if resolved_cache_dir:
            cache_path = Path(resolved_cache_dir)
            if cache_path.suffix:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                cache_path.mkdir(parents=True, exist_ok=True)
            ds = ds.cache(str(cache_path))
        else:
            ds = ds.cache()

        # No extra batching (generator already yields full batches)

        # Prefetch
        ds = ds.prefetch(buffer_size=self.autotune)
        if self.prefetch_to_device:
            try:
                ds = tf.data.experimental.prefetch_to_device(self.prefetch_device)(ds)
                logger.info(f"Prefetching dataset to device: {self.prefetch_device}")
            except Exception as e:
                logger.warning(f"GPU prefetch disabled: {e}")

        return ds

    def create_mixed_precision_pipeline(
        self,
        cache_dir: str | None = None,
    ) -> tf.data.Dataset:
        """Create pipeline optimized for mixed precision training.

        Casts inputs to float16 for faster processing on Tensor Core GPUs.

        Args:
            cache_dir: Directory for disk cache

        Returns:
            tf.data.Dataset with float16 inputs
        """
        ds = self.create_training_pipeline(cache_dir=cache_dir)

        # Cast to float16 for mixed precision (only features, keep labels in float32)
        def cast_to_fp16(features, labels):
            features = tf.cast(features, tf.float16)
            return features, labels

        ds = ds.map(cast_to_fp16, num_parallel_calls=self.autotune)

        return ds


def create_optimized_dataset(
    dataset: WakeWordDataset,
    config: dict,
    split: str = "train",
    use_mixed_precision: bool = False,
) -> tf.data.Dataset:
    """Convenience function to create optimized dataset.

    Args:
        dataset: WakeWordDataset instance
        config: Training configuration
        split: 'train' or 'val'
        use_mixed_precision: Whether to use float16

    Returns:
        Optimized tf.data.Dataset
    """
    pipeline = OptimizedDataPipeline(dataset, config)

    if use_mixed_precision and split == "train":
        return pipeline.create_mixed_precision_pipeline()
    elif split == "train":
        return pipeline.create_training_pipeline()
    else:
        return pipeline.create_validation_pipeline()


class PrefetchGenerator:
    """Legacy-compatible prefetch generator wrapper.

    Wraps tf.data.Dataset to provide generator interface
    while maintaining prefetch benefits.

    Example:
        pipeline = OptimizedDataPipeline(dataset, config)
        ds = pipeline.create_training_pipeline()

        # Use as generator
        for batch_x, batch_y in PrefetchGenerator(ds):
            model.train_on_batch(batch_x, batch_y)
    """

    def __init__(self, dataset: tf.data.Dataset):
        """Initialize with tf.data.Dataset.

        Args:
            dataset: tf.data.Dataset instance
        """
        self.dataset = dataset
        self.iterator: Iterator[Any] | None = None

    def __iter__(self):
        """Return iterator."""
        self.iterator = iter(self.dataset)
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """Get next batch.

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        if self.iterator is None:
            self.iterator = iter(self.dataset)

        features, labels = next(self.iterator)
        return features.numpy(), labels.numpy()

    def __len__(self) -> int:
        """Return dataset length.

        Raises:
            TypeError: Dataset does not have a determinable length
        """
        raise TypeError("object of type 'PrefetchGenerator' has no len()")


def benchmark_pipeline(
    dataset: WakeWordDataset,
    config: dict,
    n_batches: int = 100,
) -> dict[str, float]:
    """Benchmark data pipeline performance.

    Compares generator vs tf.data.Dataset performance.

    Args:
        dataset: WakeWordDataset instance
        config: Training configuration
        n_batches: Number of batches to test

    Returns:
        Benchmark results
    """
    import time

    results = {}

    # Benchmark original generator
    logger.info("Benchmarking original generator...")
    factory = dataset.train_generator_factory()
    gen = factory()

    start = time.time()
    for i, batch in enumerate(gen):
        if i >= n_batches:
            break
        if not isinstance(batch, tuple) or len(batch) < 2:
            continue
        x = batch[0]
        y = batch[1]
        _ = x.sum() + y.sum()
    gen_time = time.time() - start
    results["generator_time"] = gen_time
    results["generator_batches_per_sec"] = n_batches / gen_time

    # Benchmark tf.data pipeline
    logger.info("Benchmarking tf.data pipeline...")
    pipeline = OptimizedDataPipeline(dataset, config)
    ds = pipeline.create_training_pipeline()
    ds_iter = iter(ds)

    start = time.time()
    for _i in range(n_batches):
        x, y = next(ds_iter)
        # Simulate training step
        _ = tf.reduce_sum(x) + tf.reduce_sum(y)
    ds_time = time.time() - start
    results["tfdata_time"] = ds_time
    results["tfdata_batches_per_sec"] = n_batches / ds_time

    # Calculate speedup
    speedup = gen_time / ds_time
    results["speedup"] = speedup

    logger.info(f"Generator: {results['generator_batches_per_sec']:.1f} batches/sec")
    logger.info(f"tf.data: {results['tfdata_batches_per_sec']:.1f} batches/sec")
    logger.info(f"Speedup: {speedup:.2f}x")

    return results
