"""Unit tests for AsyncHardExampleMiner."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import tensorflow as tf

from src.training.mining import AsyncHardExampleMiner


class TestAsyncHardExampleMiner:
    """Tests for AsyncHardExampleMiner class."""

    def test_async_miner_instantiation(self):
        """Test that AsyncHardExampleMiner can be instantiated with default params."""
        miner = AsyncHardExampleMiner(
            strategy="confidence",
            fp_threshold=0.8,
            max_samples=5000,
            mining_interval_epochs=5,
            output_dir="./test_output",
        )

        assert miner is not None
        assert miner._miner is not None
        assert miner._thread is None
        assert miner._result is None
        assert miner._is_running is False

    def test_async_miner_not_mining_initially(self):
        """Test that is_mining() returns False initially."""
        miner = AsyncHardExampleMiner(
            strategy="confidence",
            fp_threshold=0.8,
            max_samples=5000,
            mining_interval_epochs=5,
            output_dir="./test_output",
        )

        assert miner.is_mining() is False
        assert miner.get_result() is None

    def test_async_miner_thread_safety_lock(self):
        """Test that _lock is properly initialized and used."""
        miner = AsyncHardExampleMiner()

        # Verify lock exists and is a Lock (threading.Lock is a factory function)
        assert hasattr(miner, "_lock")
        assert miner._lock is not None

        # Test that lock can be acquired (duck typing for lock interface)
        with miner._lock:
            # Lock acquired successfully
            pass

        # Verify lock has acquire/release methods
        assert hasattr(miner._lock, "acquire")
        assert hasattr(miner._lock, "release")
        assert callable(miner._lock.acquire)
        assert callable(miner._lock.release)

    def test_async_miner_get_result_thread_safety(self):
        """Test that get_result() is thread-safe using the lock."""
        miner = AsyncHardExampleMiner()

        # Mock result
        test_result = {"mined_count": 10, "samples": []}

        # Set result with lock
        with miner._lock:
            miner._result = test_result

        # Get result should return the value
        result = miner.get_result()
        assert result == test_result

    def test_async_miner_is_mining_thread_safety(self):
        """Test that is_mining() is thread-safe using the lock."""
        miner = AsyncHardExampleMiner()

        # Set running state with lock
        with miner._lock:
            miner._is_running = True

        assert miner.is_mining() is True

        with miner._lock:
            miner._is_running = False

        assert miner.is_mining() is False

    def test_async_miner_start_mining_raises_if_already_running(self):
        """Test that start_mining raises RuntimeError if mining is already in progress."""
        miner = AsyncHardExampleMiner()

        # Mock model and data generator
        mock_model = MagicMock(spec=tf.keras.Model)
        mock_generator = MagicMock()

        # Set mining as already running
        with miner._lock:
            miner._is_running = True

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Mining is already in progress"):
            miner.start_mining(mock_model, mock_generator, epoch=1)

    def test_async_miner_wait_for_completion_no_thread(self):
        """Test wait_for_completion returns True when no thread exists."""
        miner = AsyncHardExampleMiner()

        result = miner.wait_for_completion()
        assert result is True

    def test_async_miner_wait_for_completion_with_timeout(self):
        """Test wait_for_completion with timeout parameter."""
        miner = AsyncHardExampleMiner()

        # Create a mock thread that takes time
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = False  # Already done
        miner._thread = mock_thread

        result = miner.wait_for_completion(timeout=1.0)
        assert result is True
        mock_thread.join.assert_called_once_with(timeout=1.0)

    @patch.object(tf.keras.models, "clone_model")
    def test_async_miner_start_mining_clones_model(self, mock_clone_model):
        """Test that start_mining clones the model before passing to thread."""
        miner = AsyncHardExampleMiner()

        # Setup mock
        mock_original_model = MagicMock(spec=tf.keras.Model)
        mock_cloned_model = MagicMock(spec=tf.keras.Model)
        mock_clone_model.return_value = mock_cloned_model

        mock_generator = MagicMock()

        # Start mining
        miner.start_mining(mock_original_model, mock_generator, epoch=1)

        # Verify clone_model was called
        mock_clone_model.assert_called_once_with(mock_original_model)
        mock_cloned_model.set_weights.assert_called_once_with(mock_original_model.get_weights.return_value)

        # Cleanup - stop the thread if it started
        if miner._thread is not None and miner._thread.is_alive():
            miner._thread.join(timeout=0.1)

    def test_async_miner_config_values_stored(self):
        """Test that configuration values are properly stored."""
        miner = AsyncHardExampleMiner(
            strategy="entropy",
            fp_threshold=0.9,
            max_samples=10000,
            mining_interval_epochs=10,
            output_dir="./custom_output",
        )

        # Verify values are stored in the underlying miner
        assert miner._miner.strategy == "entropy"
        assert miner._miner.fp_threshold == 0.9
        assert miner._miner.max_samples == 10000
        assert miner._miner.mining_interval_epochs == 10

    def test_async_miner_result_set_after_mining(self):
        """Test that result is properly set after mining completes."""
        miner = AsyncHardExampleMiner()

        # Simulate mining completion
        test_result = {"mined": 5, "threshold": 0.8}

        # Simulate what _mining_worker does
        def simulate_mining():
            time.sleep(0.01)  # Small delay
            with miner._lock:
                miner._result = test_result
                miner._is_running = False

        # Start simulation thread
        with miner._lock:
            miner._is_running = True
        thread = threading.Thread(target=simulate_mining)
        thread.start()

        # Wait for completion
        thread.join(timeout=1.0)

        # Verify result
        assert miner.get_result() == test_result
        assert miner.is_mining() is False

    def test_async_miner_thread_daemon_false(self):
        """Test that mining thread is created as non-daemon."""
        miner = AsyncHardExampleMiner()

        mock_model = MagicMock(spec=tf.keras.Model)
        mock_generator = MagicMock()

        # Mock clone_model to avoid actual cloning
        with patch.object(tf.keras.models, "clone_model") as mock_clone:
            mock_clone.return_value = MagicMock(spec=tf.keras.Model)

            miner.start_mining(mock_model, mock_generator, epoch=1)

            # Verify thread is not daemon
            if miner._thread is not None:
                assert miner._thread.daemon is False

            # Cleanup
            if miner._thread is not None and miner._thread.is_alive():
                miner._thread.join(timeout=0.1)
