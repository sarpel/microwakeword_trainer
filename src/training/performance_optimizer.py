"""Performance optimization utilities for training.

Integrates mixed precision and tf.data.Dataset pipeline for maximum performance.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from rich.console import Console
from rich.table import Table

from src.data.dataset import WakeWordDataset
from src.data.tfdata_pipeline import (
    OptimizedDataPipeline,
    benchmark_pipeline,
    create_optimized_dataset,
)
from src.utils.performance import configure_mixed_precision

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """High-level performance optimizer for training.

    Combines mixed precision and tf.data.Dataset optimizations.

    Example:
        optimizer = PerformanceOptimizer(config)
        optimizer.enable_all()

        # Get optimized datasets
        train_ds, val_ds = optimizer.create_datasets(dataset)

        # Use with model
        model.fit(train_ds, validation_data=val_ds)
    """

    def __init__(
        self,
        config: dict,
        console: Console | None = None,
    ):
        """Initialize optimizer.

        Args:
            config: Training configuration
            console: Rich console for output
        """
        self.config = config
        self.console = console or Console()

        # Check config for performance settings
        perf_config = config.get("performance", {})
        self.mixed_precision_enabled = perf_config.get("mixed_precision", True)
        self.use_tfdata = perf_config.get("use_tfdata", True)

        self._is_setup = False

    def enable_all(self) -> None:
        """Enable all performance optimizations."""
        if self.mixed_precision_enabled:
            self.enable_mixed_precision()

        self._is_setup = True
        self._log_status()

    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training."""
        configure_mixed_precision(enabled=True)
        logger.info("Mixed precision enabled")

    def disable_mixed_precision(self) -> None:
        """Disable mixed precision training."""
        configure_mixed_precision(enabled=False)
        logger.info("Mixed precision disabled")

    def create_datasets(
        self,
        dataset: WakeWordDataset,
        max_time_frames: int | None = None,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create optimized training and validation datasets.

        Args:
            dataset: WakeWordDataset instance
            max_time_frames: Max time frames for padding

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if not self.use_tfdata:
            # Fall back to legacy generators
            factory = dataset.train_generator_factory(max_time_frames)
            train_gen = factory()

            val_factory = dataset.val_generator_factory(max_time_frames)
            val_gen = val_factory()

            return train_gen, val_gen

        # Use optimized tf.data pipeline
        pipeline = OptimizedDataPipeline(
            dataset,
            self.config,
            max_time_frames=max_time_frames,
        )

        train_ds = pipeline.create_training_pipeline()
        val_ds = pipeline.create_validation_pipeline()

        return train_ds, val_ds

    def create_training_dataset(
        self,
        dataset: WakeWordDataset,
        max_time_frames: int | None = None,
    ) -> tf.data.Dataset:
        """Create optimized training dataset.

        Args:
            dataset: WakeWordDataset instance
            max_time_frames: Max time frames for padding

        Returns:
            Optimized tf.data.Dataset
        """
        return create_optimized_dataset(
            dataset,
            self.config,
            split="train",
            use_mixed_precision=self.mixed_precision_enabled,
        )

    def create_validation_dataset(
        self,
        dataset: WakeWordDataset,
        max_time_frames: int | None = None,
    ) -> tf.data.Dataset:
        """Create optimized validation dataset.

        Args:
            dataset: WakeWordDataset instance
            max_time_frames: Max time frames for padding

        Returns:
            Optimized tf.data.Dataset
        """
        return create_optimized_dataset(
            dataset,
            self.config,
            split="val",
            use_mixed_precision=False,  # No mixed precision for validation
        )

    def benchmark(self, dataset: WakeWordDataset, n_batches: int = 100) -> dict[str, float]:
        """Benchmark data pipeline performance.

        Args:
            dataset: WakeWordDataset instance
            n_batches: Number of batches to test

        Returns:
            Benchmark results
        """
        self.console.print("[cyan]Benchmarking data pipeline...[/]")
        results = benchmark_pipeline(dataset, self.config, n_batches)

        # Display results
        table = Table(title="Pipeline Benchmark Results")
        table.add_column("Method", style="bold")
        table.add_column("Batches/sec", justify="right")
        table.add_column("Time (100 batches)", justify="right")

        table.add_row(
            "Legacy Generator",
            f"{results['generator_batches_per_sec']:.1f}",
            f"{results['generator_time']:.2f}s",
        )
        table.add_row(
            "tf.data.Dataset",
            f"{results['tfdata_batches_per_sec']:.1f}",
            f"{results['tfdata_time']:.2f}s",
        )
        table.add_row(
            "Speedup",
            f"[green]{results['speedup']:.2f}x[/]",
            "",
        )

        self.console.print(table)

        return results

    def _log_status(self) -> None:
        """Log current optimization status."""
        table = Table(title="Performance Optimizations")
        table.add_column("Feature", style="bold")
        table.add_column("Status", justify="center")

        mp_status = "[green]Enabled[/]" if self.mixed_precision_enabled else "[dim]Disabled[/]"
        tfdata_status = "[green]Enabled[/]" if self.use_tfdata else "[dim]Disabled[/]"

        table.add_row("Mixed Precision", mp_status)
        table.add_row("tf.data.Dataset", tfdata_status)

        self.console.print(table)


def setup_performance_optimizations(
    config: dict,
    console: Console | None = None,
) -> PerformanceOptimizer:
    """Convenience function to setup all performance optimizations.

    Args:
        config: Training configuration
        console: Rich console for output

    Returns:
        Configured PerformanceOptimizer
    """
    optimizer = PerformanceOptimizer(config, console)
    optimizer.enable_all()
    return optimizer


def is_mixed_precision_available() -> bool:
    """Check if mixed precision is available on this system.

    Returns:
        True if GPU with Tensor Cores is available
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False

    # Check for Tensor Cores (Volta, Turing, Ampere, Ada)
    # These architectures benefit most from mixed precision
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "")

    tensor_core_gpus = [
        "V100",
        "T4",
        "RTX 20",
        "RTX 30",
        "RTX 40",
        "A100",
        "A10",
        "H100",
        "A6000",
        "A4000",
    ]

    return any(name in gpu_name for name in tensor_core_gpus)


def get_optimal_batch_size(config: dict) -> int:
    """Get optimal batch size based on GPU memory.

    Args:
        config: Training configuration

    Returns:
        Optimal batch size
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return config.get("training", {}).get("batch_size", 128)

    try:
        # Get GPU memory
        gpu_memory = tf.config.experimental.get_memory_info(gpus[0].name)["current"]

        # Estimate optimal batch size based on available memory
        # This is a rough heuristic
        if gpu_memory > 16_000_000_000:  # > 16GB
            return 256
        elif gpu_memory > 8_000_000_000:  # > 8GB
            return 128
        elif gpu_memory > 4_000_000_000:  # > 4GB
            return 64
        else:
            return 32
    except Exception:
        return config.get("training", {}).get("batch_size", 128)
