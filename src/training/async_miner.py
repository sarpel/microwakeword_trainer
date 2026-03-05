"""Async hard example mining module for non-blocking training."""

import threading
from typing import Any

import tensorflow as tf

from .miner import HardExampleMiner


class AsyncHardExampleMiner:
    """Async wrapper for hard example mining.

    Runs hard negative mining in a background thread to avoid blocking
    the training loop. Model is cloned before passing to the thread
    to prevent thread-safety issues.
    """

    def __init__(
        self,
        strategy: str = "confidence",
        fp_threshold: float = 0.8,
        max_samples: int = 5000,
        mining_interval_epochs: int = 5,
        output_dir: str = "./data/raw/hard_negative",
    ):
        """Initialize async miner.

        Args:
            strategy: Mining strategy ("confidence" or "entropy")
            fp_threshold: Prediction threshold for hard negative detection
            max_samples: Maximum number of hard negatives to collect
            mining_interval_epochs: Epochs between mining operations
            output_dir: Directory to save mined hard negatives
        """
        self._miner = HardExampleMiner(
            strategy=strategy,
            fp_threshold=fp_threshold,
            max_samples=max_samples,
            mining_interval_epochs=mining_interval_epochs,
            output_dir=output_dir,
        )
        self._thread: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._is_running = False

    def _mining_worker(
        self,
        model: tf.keras.Model,
        data_generator: Any,
        epoch: int,
    ) -> None:
        """Worker function that runs in the background thread.

        Args:
            model: Cloned model for mining
            data_generator: Generator yielding (features, labels, weights) tuples
            epoch: Current training epoch
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            result = self._miner.mine_from_dataset(model, data_generator, epoch)
        except Exception as e:
            logger.exception(f"Mining failed at epoch {epoch}: {e}")
            result = None
        finally:
            with self._lock:
                self._result = result
                self._is_running = False

    def start_mining(
        self,
        model: tf.keras.Model,
        data_generator: Any,
        epoch: int,
    ) -> None:
        """Start mining in a background thread.

        Args:
            model: Model to use for mining (will be cloned)
            data_generator: Generator yielding (features, labels, weights) tuples
            epoch: Current training epoch
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check if already running before cloning
        with self._lock:
            if self._is_running:
                raise RuntimeError("Mining is already in progress")
            self._result = None

        # Clone model to avoid sharing training model with thread
        # Do this outside the lock to prevent blocking training
        try:
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.set_weights(model.get_weights())
        except Exception as e:
            logger.exception(f"Model cloning failed: {e}")
            with self._lock:
                self._result = None
            # Don't set _is_running, the thread won't start
            return

        # Claim lock after successful cloning, then start thread
        with self._lock:
            if self._is_running:
                # Someone else started mining while we were cloning
                raise RuntimeError("Mining is already in progress (started during model cloning)")
            self._is_running = True

        # Start background thread
        try:
            self._thread = threading.Thread(
                target=self._mining_worker,
                args=(cloned_model, data_generator, epoch),
                daemon=False,
            )
            self._thread.start()
        except Exception as e:
            with self._lock:
                self._is_running = False
            logger.exception(f"Failed to start mining thread: {e}")
            raise

    def is_mining(self) -> bool:
        """Check if mining is currently running.

        Returns:
            True if mining thread is active, False otherwise
        """
        with self._lock:
            return self._is_running

    def get_result(self) -> dict[str, Any] | None:
        """Get the mining result if available.

        Returns:
            Mining result dictionary or None if not complete
        """
        with self._lock:
            return self._result

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for mining thread to complete.

        Args:
            timeout: Maximum time to wait in seconds, or None for indefinite

        Returns:
            True if thread completed, False if timeout occurred
        """
        with self._lock:
            thread = self._thread
        if thread is None:
            return True

        thread.join(timeout=timeout)
        return not thread.is_alive()
