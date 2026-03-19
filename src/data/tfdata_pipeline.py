"""Optimized TensorFlow data pipeline using tf.data.Dataset.

Provides high-performance data loading with caching, prefetching, and parallel processing.
Compatible with ESPHome - only affects training speed, not model export.
"""

import logging
import os
import time
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
        self.batch_size = batch_size or config.get("training", {}).get("batch_size", 384)
        self.max_time_frames = max_time_frames or self._calculate_max_frames()
        self.spec_augment_config = spec_augment_config or {}

        training_cfg = config.get("training", {})
        self.training_steps = training_cfg.get("training_steps", [20000])
        # Guard against empty training_steps which would break downstream processing
        if not self.training_steps:
            raise ValueError("training_steps cannot be empty. Provide at least one training phase.")
        self.phase_boundaries: list[int] = []
        cumulative = 0
        for steps in self.training_steps:
            cumulative += int(steps)
            self.phase_boundaries.append(cumulative)

        # Performance settings
        self.autotune = tf.data.AUTOTUNE
        self.prefetch_buffer = config.get("performance", {}).get("prefetch_buffer", 8)
        self.cache_dir = config.get("performance", {}).get("tfdata_cache_dir")
        processed_dir = config.get("paths", {}).get("processed_dir", "./data/processed")
        self.default_cache_dir = str(Path(processed_dir) / "tfdata_cache")
        self.prefetch_to_device = config.get("performance", {}).get("tfdata_prefetch_to_device", True)
        self.prefetch_device = config.get("performance", {}).get("tfdata_prefetch_device", "/GPU:0")

    def _calculate_max_frames(self) -> int:
        """Calculate max time frames from hardware config."""
        hardware = self.config.get("hardware", {})
        clip_duration_ms = hardware.get("clip_duration_ms", 1000)
        window_step_ms = hardware.get("window_step_ms", 10)
        return int(clip_duration_ms / window_step_ms)

    def _resolve_cache_path(self, resolved_cache_dir: str, cache_name: str) -> Path:
        cache_root = Path(resolved_cache_dir)
        if cache_root.suffix:
            cache_root.parent.mkdir(parents=True, exist_ok=True)
        else:
            cache_root.mkdir(parents=True, exist_ok=True)

        cache_path = cache_root / cache_name
        lockfiles = list(cache_root.glob(f"{cache_path.name}_*.lockfile"))
        if lockfiles:
            # Try to clean up stale lockfiles from dead processes
            stale = []
            for lf in lockfiles:
                try:
                    # TF lockfiles aren't PID-named, so check .tempstate
                    # companion; if present, cache was never finalized
                    tempstate = lf.with_suffix(".tempstate")
                    if tempstate.exists():
                        stale.append(lf)
                    else:
                        stale.append(lf)  # orphan lockfile without data
                except OSError:
                    pass
            for lf in stale:
                try:
                    lf.unlink(missing_ok=True)
                    # Also remove any incomplete cache data
                    for leftover in cache_root.glob(f"{lf.stem}*"):
                        leftover.unlink(missing_ok=True)
                    logger.info("Removed stale cache lockfile: %s", lf)
                except OSError as e:
                    logger.warning("Could not remove stale lockfile %s: %s", lf, e)
            # Re-check if lockfiles remain after cleanup
            lockfiles = list(cache_root.glob(f"{cache_path.name}_*.lockfile"))
            if lockfiles:
                unique_name = f"{cache_name}_{os.getpid()}_{int(time.time())}"
                cache_path = cache_root / unique_name
                logger.warning(
                    "Active cache lockfile detected %s; using unique cache path %s",
                    lockfiles[0],
                    cache_path,
                )
        return cache_path

    def _effective_cache_dir(self, cache_dir: str | None) -> str | None:
        """Resolve effective cache directory with a safe default.

        Resolution order:
        1) explicit `cache_dir` argument when provided
        2) config.performance.tfdata_cache_dir
        3) default under processed_dir/tfdata_cache

        Set cache_dir to empty string to disable caching explicitly.
        """
        candidate = self.cache_dir if cache_dir is None else cache_dir
        if candidate is None:
            return self.default_cache_dir
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if not normalized:
                return None
            return normalized
        return str(candidate)

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
            elif split == "test":
                if not hasattr(self.dataset, "test_generator_factory"):
                    raise ValueError(f"Dataset {type(self.dataset).__name__} does not support 'test' split. Use 'train' or 'val'.")
                factory = self.dataset.test_generator_factory(self.max_time_frames)
            else:
                factory = self.dataset.val_generator_factory(self.max_time_frames)

            gen = factory() if callable(factory) else factory
            for batch in gen:
                if not isinstance(batch, tuple) or len(batch) < 2:
                    continue
                features = batch[0]
                labels = (np.asarray(batch[1], dtype=np.int32) == 1).astype(np.int32)  # Binarize: hard_neg(2)→0
                # Yield 4-tuple: (features, labels, sample_weights, is_hard_neg)
                batch_size = features.shape[0]
                # Use real sample_weights and is_hard_neg from underlying generator
                # (train generator yields 4-tuples with actual hard-neg flags;
                #  val/test generators yield 3-tuples — fall back to defaults)
                sample_weights = np.asarray(batch[2], dtype=np.float32) if len(batch) > 2 else np.ones(batch_size, dtype=np.float32)
                is_hard_neg = np.asarray(batch[3], dtype=np.bool_) if len(batch) > 3 else np.zeros(batch_size, dtype=np.bool_)
                yield features, labels, sample_weights, is_hard_neg

        return generator

    def create_training_pipeline(
        self,
        cache_dir: str | None = None,
        shuffle_buffer: int = 2000,
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
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
        )

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        # Cache first (before shuffle for proper training behavior)
        resolved_cache_dir = self._effective_cache_dir(cache_dir)
        if resolved_cache_dir:
            cache_path = self._resolve_cache_path(resolved_cache_dir, "tfdata_train")
            ds = ds.cache(str(cache_path))
            ds = ds.repeat()  # Replay from cache infinitely; finite generator fills cache once
            logger.info(f"Using disk cache: {cache_path}")
        else:
            logger.info("tf.data cache disabled (empty cache_dir)")
            ds = ds.repeat()  # No cache; finite generator restarts each epoch
        # Shuffle for training (after cache to reshuffle each epoch)
        training_cfg = self.config.get("training", {})
        deterministic_ops = os.environ.get("TF_DETERMINISTIC_OPS") == "1"
        if deterministic_ops and training_cfg.get("random_seed") is None:
            if shuffle_seed is None:
                shuffle_seed = int(training_cfg.get("split_seed", 42))
            tf.random.set_seed(int(shuffle_seed))
            logger.info("Determinism enabled without random_seed; using shuffle_seed=%d for TF random seed", int(shuffle_seed))
        elif shuffle_seed is None and deterministic_ops:
            shuffle_seed = int(training_cfg.get("split_seed", 42))
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed, reshuffle_each_iteration=True)

        # Apply SpecAugment if enabled (TF backend) and compute class weights in-pipeline
        if self.spec_augment_config.get("enabled", False):
            # Use TF as default for better performance (PERF-005 fix)
            # CuPy backend causes CPU-GPU memory transfers (15-25% slowdown)
            backend = self.spec_augment_config.get("backend", "tf")
            if backend == "tf":
                time_mask_max_size = self.spec_augment_config.get("time_mask_max_size", 10)
                time_mask_count = self.spec_augment_config.get("time_mask_count", 2)
                freq_mask_max_size = self.spec_augment_config.get("freq_mask_max_size", 10)
                freq_mask_count = self.spec_augment_config.get("freq_mask_count", 2)
                seed = self.spec_augment_config.get("seed")

                # Ensure per-phase lists
                if not isinstance(time_mask_max_size, (list, tuple)):
                    time_mask_max_size = [int(time_mask_max_size)]
                if not isinstance(time_mask_count, (list, tuple)):
                    time_mask_count = [int(time_mask_count)]
                if not isinstance(freq_mask_max_size, (list, tuple)):
                    freq_mask_max_size = [int(freq_mask_max_size)]
                if not isinstance(freq_mask_count, (list, tuple)):
                    freq_mask_count = [int(freq_mask_count)]

                phase_boundaries = tf.constant(self.phase_boundaries, dtype=tf.int64)
                tmask_max = tf.constant(time_mask_max_size, dtype=tf.int32)
                tmask_cnt = tf.constant(time_mask_count, dtype=tf.int32)
                fmask_max = tf.constant(freq_mask_max_size, dtype=tf.int32)
                fmask_cnt = tf.constant(freq_mask_count, dtype=tf.int32)

                counter = tf.data.Dataset.counter()
                ds = tf.data.Dataset.zip((counter, ds))

                def apply_spec_augment(step, batch):
                    features, labels, sample_weights, is_hard_neg = batch
                    # Phase index from step and boundaries
                    phase = tf.reduce_sum(tf.cast(step >= phase_boundaries, tf.int32))
                    phase = tf.minimum(phase, tf.shape(tmask_max)[0] - 1)

                    tmax = tf.gather(tmask_max, phase)
                    tcnt = tf.gather(tmask_cnt, phase)
                    fmax = tf.gather(fmask_max, phase)
                    fcnt = tf.gather(fmask_cnt, phase)

                    augmented_features = batch_spec_augment_tf(
                        features,
                        time_mask_max_size=tmax,
                        time_mask_count=tcnt,
                        freq_mask_max_size=fmax,
                        freq_mask_count=fcnt,
                        seed=seed,
                    )

                    # Class weights are applied in Trainer._apply_class_weights() (phase-aware)
                    # Do NOT apply here to avoid double weighting
                    return augmented_features, labels, sample_weights, is_hard_neg

                ds = ds.map(apply_spec_augment, num_parallel_calls=self.autotune, deterministic=False)
                logger.info(f"SpecAugment (TF backend) enabled with staged schedule: time_masks={time_mask_count}@{time_mask_max_size}, freq_masks={freq_mask_count}@{freq_mask_max_size}")
                options = tf.data.Options()
                options.experimental_deterministic = False
                ds = ds.with_options(options)
        else:
            # No SpecAugment: class weights are applied in Trainer._apply_class_weights() (phase-aware)
            # No transformation needed — pass dataset through unchanged
            options = tf.data.Options()
            options.experimental_deterministic = False
            ds = ds.with_options(options)

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

    def create_training_pipeline_with_spec_augment(
        self,
        spec_augment_config: dict,
    ) -> tf.data.Dataset:
        """Create optimized training pipeline with SpecAugment in tf.data graph.

        Integrates SpecAugment directly into the tf.data pipeline to eliminate
        CPU↔GPU data transfers that occur with the CuPy-based approach.

        Args:
            spec_augment_config: Dict with keys:
                - time_mask_max_size: int
                - time_mask_count: int
                - freq_mask_max_size: int
                - freq_mask_count: int
                - seed: Optional int

        Returns:
            tf.data.Dataset yielding (features, labels, sample_weights, is_hard_neg)
            tuples, with SpecAugment applied to features in-graph.
        """
        from src.data.spec_augment_tf import batch_spec_augment_tf

        # Temporarily disable in-pipeline SpecAugment to avoid double augmentation
        # since create_training_pipeline may already apply SpecAugment via self.spec_augment_config
        original_enabled = self.spec_augment_config.get("enabled")
        self.spec_augment_config["enabled"] = False

        # Base training dataset preserving existing cache/shuffle/prefetch behavior.
        ds = self.create_training_pipeline()

        # Restore original SpecAugment enabled state
        if original_enabled is not None:
            self.spec_augment_config["enabled"] = original_enabled
        else:
            del self.spec_augment_config["enabled"]

        def apply_spec_augment(features, labels, sample_weights, *rest):
            augmented = batch_spec_augment_tf(
                features,
                time_mask_max_size=spec_augment_config.get("time_mask_max_size", 10),
                time_mask_count=spec_augment_config.get("time_mask_count", 2),
                freq_mask_max_size=spec_augment_config.get("freq_mask_max_size", 5),
                freq_mask_count=spec_augment_config.get("freq_mask_count", 2),
                seed=spec_augment_config.get("seed"),
            )

            # Class weights are applied in Trainer._apply_class_weights() (phase-aware).
            # Keep sample_weights passthrough here to avoid double weighting.

            return (augmented, labels, sample_weights) + rest if rest else (augmented, labels, sample_weights)

        ds = ds.map(apply_spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    def create_test_pipeline(
        self,
        cache_dir: str | None = None,
    ) -> tf.data.Dataset:
        """Create optimized test data pipeline.

        Similar to validation but uses test split.

        Args:
            cache_dir: Directory for disk cache

        Returns:
            tf.data.Dataset ready for testing
        """
        generator = self._generator_factory("test")

        output_signature = (
            tf.TensorSpec(shape=(None, self.max_time_frames, 40), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
        )

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        # Cache
        resolved_cache_dir = self._effective_cache_dir(cache_dir)
        if resolved_cache_dir:
            cache_path = self._resolve_cache_path(resolved_cache_dir, "tfdata_test")
            ds = ds.cache(str(cache_path))
            logger.info(f"Using disk cache: {cache_path}")
        else:
            logger.info("tf.data cache disabled (empty cache_dir)")
        # Prefetch
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
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
        )

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        # Cache (no shuffle for validation)
        resolved_cache_dir = self._effective_cache_dir(cache_dir)
        if resolved_cache_dir:
            cache_path = self._resolve_cache_path(resolved_cache_dir, "tfdata_val")
            ds = ds.cache(str(cache_path))
            logger.info(f"Using disk cache: {cache_path}")
        else:
            logger.info("tf.data cache disabled (empty cache_dir)")
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
        def cast_to_fp16(features, labels, sample_weights, is_hard_neg):
            features = tf.cast(features, tf.float16)
            return features, labels, sample_weights, is_hard_neg

        ds = ds.map(cast_to_fp16, num_parallel_calls=self.autotune)

        return ds


def create_optimized_dataset(
    dataset: WakeWordDataset,
    config: dict,
    split: str = "train",
    use_mixed_precision: bool = False,
    max_time_frames: int | None = None,
) -> tf.data.Dataset:
    """Convenience function to create optimized dataset.

    Args:
        dataset: WakeWordDataset instance
        config: Training configuration
        split: 'train' or 'val'
        use_mixed_precision: Whether to use float16
        max_time_frames: Max time frames for padding

    Returns:
        Optimized tf.data.Dataset
    """
    pipeline = OptimizedDataPipeline(dataset, config, max_time_frames=max_time_frames)

    if use_mixed_precision and split == "train":
        return pipeline.create_mixed_precision_pipeline()
    elif split == "train":
        return pipeline.create_training_pipeline()
    elif split == "test":
        return pipeline.create_test_pipeline()
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

        features, labels, *_ = next(self.iterator)
        return features.numpy(), labels.numpy()

    def __len__(self) -> int:
        """Return dataset length.

        Raises:
            TypeError: Dataset does not have a determinable length
        """
        raise TypeError("object of type 'PrefetchGenerator' has no len()")

    def close(self) -> None:
        """Release the iterator and allow cleanup."""
        self.iterator = None

    def __del__(self):
        """Ensure iterator is released on destruction."""
        self.close()


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
    factory = dataset.train_generator_factory(None)
    gen = factory() if callable(factory) else factory

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
    shuffle_seed = int(config.get("training", {}).get("split_seed", 42))
    ds = pipeline.create_training_pipeline(shuffle_seed=shuffle_seed)
    ds_iter = iter(ds)

    start = time.time()
    for _i in range(n_batches):
        batch = next(ds_iter)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
        else:
            logger.warning(f"Skipping invalid batch at index {_i}: expected tuple/list with at least 2 elements (x, y), got {type(batch).__name__} with shape={getattr(batch, 'shape', 'N/A')}")
            continue
        # Simulate training step
        _ = tf.reduce_sum(x) + tf.reduce_sum(y)
    ds_time = time.time() - start
    del ds_iter
    results["tfdata_time"] = ds_time
    results["tfdata_batches_per_sec"] = n_batches / ds_time

    # Calculate speedup
    speedup = gen_time / ds_time
    results["speedup"] = speedup

    logger.info(f"Generator: {results['generator_batches_per_sec']:.1f} batches/sec")
    logger.info(f"tf.data: {results['tfdata_batches_per_sec']:.1f} batches/sec")
    logger.info(f"Speedup: {speedup:.2f}x")

    return results
