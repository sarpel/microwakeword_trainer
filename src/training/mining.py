"""Unified mining module for hard negative mining, false prediction logging, and extraction.

Consolidates all mining-related functionality into a single module:
- HardExampleMiner: In-training heap-based hard negative mining on feature-level data
- AsyncHardExampleMiner: Thread-safe async wrapper for non-blocking mining
- log_false_predictions_to_json(): Epoch-level false prediction logging
- run_top_fp_extraction(): Post-training top-N% false positive extraction
- consolidate_prediction_logs(): Merge per-epoch logs with file path mapping
- mine_from_prediction_logs(): Post-training CLI mining from JSON logs
- CLI main() with subcommands: mine, extract-top-fps, consolidate-logs

Usage:
    # As a library (from trainer.py):
    from src.training.mining import (
        AsyncHardExampleMiner,
        HardExampleMiner,
        log_false_predictions_to_json,
        run_top_fp_extraction,
    )

    # As CLI:
    mww-mine-hard-negatives mine --prediction-log logs/false_predictions.json
    mww-mine-hard-negatives extract-top-fps --config standard
    mww-mine-hard-negatives consolidate-logs --config standard
"""

__all__ = [
    "HardExampleMiner",
    "AsyncHardExampleMiner",
    "log_false_predictions_to_json",
    "run_top_fp_extraction",
    "consolidate_prediction_logs",
    "mine_from_prediction_logs",
    "compute_file_hash",
    "main",
]

import argparse
import gc
import hashlib
import heapq
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import tensorflow as tf

from src.training.rich_logger import RichTrainingLogger
from src.utils.logging_config import setup_rich_logging

logger = logging.getLogger(__name__)
rich_logger = RichTrainingLogger()


def _configure_mining_logging(verbose: bool = False) -> None:
    """Configure Rich logging for mining CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_rich_logging(level=level, show_time=True, show_path=True)


# =============================================================================
# Utilities
# =============================================================================


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for deduplication.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of file hash
    """
    hash_md5 = hashlib.md5()  # noqa: S324
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# =============================================================================
# HardExampleMiner — in-training heap-based mining on feature-level data
# =============================================================================


class HardExampleMiner:
    """Mines hard negative examples during training.

    Identifies false positives (hard negatives) and maintains a collection
    for inclusion in subsequent training iterations.
    """

    def __init__(
        self,
        strategy: str = "confidence",
        fp_threshold: float = 0.8,
        max_samples: int = 5000,
        mining_interval_epochs: int = 5,
        output_dir: str = "./data/raw/hard_negative",
    ):
        """Initialize the hard example miner.

        Args:
            strategy: Mining strategy ("confidence" or "entropy")
            fp_threshold: Prediction threshold for hard negative detection
            max_samples: Maximum number of hard negatives to collect
            mining_interval_epochs: Epochs between mining operations
            output_dir: Directory to save mined hard negatives
        """
        self.strategy = strategy
        self.fp_threshold = fp_threshold
        self.max_samples = max_samples
        self.mining_interval_epochs = mining_interval_epochs
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Storage for hard negatives
        self.hard_negatives: list[dict] = []
        self.mining_history: list[dict] = []
        self._max_history_size = 100  # Limit mining history to prevent unbounded growth

    def get_hard_samples(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Identify hard negative samples from labels and predictions.

        Hard negatives are negative samples (label 0) that the model
        predicts as positive with high confidence.

        Args:
            labels: Ground truth labels array
            predictions: Model predictions array

        Returns:
            Array of indices of hard negative samples
        """
        # Get negative samples (label == 0)
        negative_mask = labels == 0
        negative_indices = np.where(negative_mask)[0]

        if len(negative_indices) == 0:
            return np.array([], dtype=np.int64)

        # Get predictions for negative samples only
        negative_predictions = predictions[negative_indices]

        # Find false positives: negative samples predicted as positive
        if self.strategy == "confidence":
            # High confidence false positives
            false_positive_mask = negative_predictions >= self.fp_threshold
            hard_indices = negative_indices[false_positive_mask]
        elif self.strategy == "entropy":
            # High uncertainty samples (close to threshold)
            uncertainty = np.abs(negative_predictions - 0.5)
            hard_mask = uncertainty < 0.1  # Within 0.1 of decision boundary
            hard_indices = negative_indices[hard_mask]
        else:
            error_msg = f"Unknown mining strategy: {self.strategy}"
            raise ValueError(error_msg)

        return hard_indices

    def mine_from_dataset(
        self,
        model: tf.keras.Model,
        data_generator: Any,
        epoch: int,
    ) -> dict[str, Any]:
        """Mine hard negatives from a data generator.

        Memory-efficient implementation that streams through data and only
        keeps hard negative samples, not the entire dataset.

        Args:
            model: Trained model
            data_generator: Generator yielding (features, labels, weights) tuples
            epoch: Current training epoch

        Returns:
            Mining result dictionary with keys: epoch, num_hard_negatives,
            indices (global dataset indices), avg_prediction.
        """
        # Use a heap to track top-K hard negatives by prediction score
        # Each entry: (pred_score, global_index, batch_id, local_index, label, prediction)
        # We store batch_id + local_index instead of feature copies to reduce memory ~80MB
        hard_negative_heap: list[tuple[float, int, int, int, int, float]] = []
        global_offset = 0
        batch_counter = 0
        # Cache batch features keyed by batch_counter; only entries still referenced
        # in the heap need to be retained (cleaned up after heap is finalized)
        batch_features_cache: dict[int, np.ndarray] = {}

        # Handle both generator factories and direct generators
        if callable(data_generator):
            gen = data_generator()
        else:
            gen = data_generator

        for features, labels, _ in cast(Any, gen):
            predictions = model(features, training=False)
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                scores = predictions[:, 1].numpy()
            else:
                scores = tf.reshape(predictions, [-1]).numpy()

            # Get hard indices local to this batch
            local_hard_indices = self.get_hard_samples(labels, scores)

            if len(local_hard_indices) > 0:
                # Store batch features reference (only if we have hard samples from this batch)
                batch_features_cache[batch_counter] = features

            # Add hard negatives to heap
            for local_idx in local_hard_indices:
                global_idx = global_offset + int(local_idx)
                pred_score = float(scores[local_idx])
                # Store (pred_score, global_idx, batch_id, local_idx, label, prediction)
                heap_entry = (
                    pred_score,
                    global_idx,
                    batch_counter,
                    int(local_idx),
                    int(labels[local_idx]),
                    pred_score,
                )

                if len(hard_negative_heap) < self.max_samples:
                    heapq.heappush(hard_negative_heap, heap_entry)
                elif pred_score > hard_negative_heap[0][0]:
                    # This hard negative has higher score than the lowest in heap
                    evicted = heapq.heapreplace(hard_negative_heap, heap_entry)
                    # Note: batch cache eviction deferred until after materialization
                    # to prevent dropping valid top-K entries whose batches were prematurely freed

            global_offset += len(features)
            batch_counter += 1

        # Check if we found any hard negatives
        if not hard_negative_heap:
            batch_features_cache.clear()
            mining_result = {
                "epoch": epoch,
                "num_hard_negatives": 0,
                "indices": [],
                "avg_prediction": 0.0,
            }
            self.mining_history.append(mining_result)
            if len(self.mining_history) > self._max_history_size:
                self.mining_history = self.mining_history[-self._max_history_size :]
            return mining_result

        # Extract results from heap
        # Sort by prediction score (descending) - heap stores positive scores
        sorted_hard = sorted(hard_negative_heap, key=lambda x: x[0], reverse=True)

        selected_indices = [int(entry[1]) for entry in sorted_hard]
        avg_prediction = float(np.mean([entry[5] for entry in sorted_hard]))

        # Store mining result
        mining_result = {
            "epoch": epoch,
            "num_hard_negatives": len(sorted_hard),
            "indices": selected_indices,
            "avg_prediction": avg_prediction,
        }

        self.mining_history.append(mining_result)
        # Limit history size to prevent unbounded memory growth
        if len(self.mining_history) > self._max_history_size:
            self.mining_history = self.mining_history[-self._max_history_size :]

        # Create hard negative records - fetch features from batch cache using batch_id + local_idx
        hard_negative_records = [
            {
                "global_index": int(entry[1]),
                "feature": batch_features_cache[entry[2]][entry[3]].copy(),
                "label": entry[4],
                "prediction": entry[5],
            }
            for entry in sorted_hard
        ]

        # Free the batch cache now that records have been materialized
        batch_features_cache.clear()

        self.hard_negatives.extend(hard_negative_records)
        if len(self.hard_negatives) > self.max_samples:
            self.hard_negatives = self.hard_negatives[-self.max_samples :]

        return mining_result

    def save_hard_negatives(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        filepath: str | None = None,
    ) -> str | None:
        """Save hard negatives to disk for later use.

        Args:
            features: Feature array
            labels: Label array
            predictions: Model predictions
            filepath: Optional custom filepath

        Returns:
            Path to saved file, or None if no hard samples were found.
        """
        # Get hard sample indices
        hard_indices = self.get_hard_samples(labels, predictions)

        if len(hard_indices) == 0:
            return None

        if filepath is None:
            filepath = os.path.join(
                self.output_dir,
                f"hard_negatives_step_{len(self.mining_history)}.npz",
            )

        np.savez(
            filepath,
            features=features[hard_indices],
            labels=labels[hard_indices],
            predictions=predictions[hard_indices],
            indices=hard_indices,
        )

        return filepath

    def load_hard_negatives(
        self,
        filepath: str,
    ) -> dict | None:
        """Load hard negatives from disk.

        Args:
            filepath: Path to saved hard negatives file

        Returns:
            Dict with features, labels, predictions, indices or None
        """
        if not os.path.exists(filepath):
            return None

        data = np.load(filepath)
        return {
            "features": data["features"],
            "labels": data["labels"],
            "predictions": data["predictions"],
            "indices": data["indices"],
        }

    def get_all_hard_negatives(self) -> list[dict]:
        """Get all collected hard negatives across mining iterations.

        Returns:
            List of hard negative records
        """
        return self.hard_negatives


# =============================================================================
# AsyncHardExampleMiner — thread-safe async wrapper
# =============================================================================


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
        result: dict[str, Any] | None = None
        try:
            result = self._miner.mine_from_dataset(model, data_generator, epoch)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(f"Mining failed at epoch {epoch}: {e}")
            result = None
        finally:
            # Clean up model to free memory
            try:
                del model
                gc.collect()
            except (NameError, UnboundLocalError):
                pass
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
        except (RuntimeError, ValueError, TypeError) as e:
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
        except RuntimeError as e:
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

    def __del__(self):
        """Ensure mining thread is cleaned up on destruction."""
        try:
            self.wait_for_completion(timeout=1.0)
        except (RuntimeError, AttributeError):
            pass  # Ignore errors during cleanup


# =============================================================================
# log_false_predictions_to_json — called from trainer per-epoch
# =============================================================================


def log_false_predictions_to_json(
    epoch: int,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fp_threshold: float = 0.8,
    top_k: int = 100,
    log_file: str = "logs/false_predictions.json",
    val_paths: list[str] | None = None,
    best_weights_path: str | None = None,
    logger: Any | None = None,
) -> dict[str, Any] | None:
    """Log false positive predictions to a JSON file for post-training mining.

    Identifies negative samples predicted above fp_threshold, takes the top_k
    by score, and appends the epoch entry to a cumulative JSON log file.

    Args:
        epoch: Current training epoch number.
        y_true: Ground truth labels (0=negative, 1=positive).
        y_scores: Model prediction scores.
        fp_threshold: Minimum score to consider a false positive.
        top_k: Maximum number of false positives to log per epoch.
        log_file: Path to the JSON log file (created if missing).
        val_paths: Optional list of file paths aligned with y_true/y_scores.
        best_weights_path: Path to current best model weights.
        logger: Optional logger instance (RichTrainingLogger or logging.Logger).

    Returns:
        The epoch entry dict, or None if no false positives found.
    """
    _log = logger or logging.getLogger(__name__)

    # Find false positives: negative samples (label 0) with score >= threshold
    neg_mask = y_true == 0
    neg_scores = y_scores[neg_mask]
    neg_indices = np.where(neg_mask)[0]

    fp_mask = neg_scores >= fp_threshold
    fp_scores = neg_scores[fp_mask]
    fp_indices = neg_indices[fp_mask]

    if len(fp_scores) == 0:
        return None

    # Sort descending, take top_k
    top_order = np.argsort(fp_scores)[::-1][:top_k]
    top_scores = fp_scores[top_order]
    top_indices = fp_indices[top_order]

    # Build per-prediction entries
    false_predictions = []
    for idx, score in zip(top_indices, top_scores, strict=True):
        entry: dict[str, Any] = {
            "index": int(idx),
            "score": float(score),
        }
        if val_paths is not None and int(idx) < len(val_paths):
            entry["file_path"] = val_paths[int(idx)]
        else:
            entry["file_path"] = None
        false_predictions.append(entry)

    epoch_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch": int(epoch),
        "fp_threshold": float(fp_threshold),
        "total_val_samples": int(len(y_true)),
        "false_positive_count": int(len(fp_scores)),
        "false_predictions": false_predictions,
    }

    # Read existing log or create new
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            log_data = {"epochs": {}, "metadata": {}}
    else:
        log_data = {"epochs": {}, "metadata": {}}

    # Store metadata on first write
    if not log_data.get("metadata"):
        log_data["metadata"] = {
            "fp_threshold": float(fp_threshold),
            "top_k_per_epoch": int(top_k),
            "model_checkpoint": (str(best_weights_path) if best_weights_path else None),
        }

    log_data["epochs"][str(epoch)] = epoch_entry

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    msg = f"Logged {len(false_predictions)} false predictions for epoch {epoch} to {log_file}"
    if hasattr(_log, "info"):
        _log.info(msg)
    else:
        _log.log_info(msg)

    return epoch_entry


# =============================================================================
# Top FP extraction — post-training model inference on hard_neg audio files
# =============================================================================


def _resolve_checkpoint(config: dict[str, Any], checkpoint_path: str | None) -> str:
    """Find the best model checkpoint.

    Args:
        config: Full config dict.
        checkpoint_path: Explicit path (used if exists).

    Returns:
        Resolved checkpoint path.

    Raises:
        FileNotFoundError: If no checkpoint found.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path

    # Try standard checkpoint locations
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
    candidates = [
        os.path.join(checkpoint_dir, "best_weights.weights.h5"),
        os.path.join(checkpoint_dir, "best_model.weights.h5"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Glob for any .weights.h5 file
    import glob

    pattern = os.path.join(checkpoint_dir, "*.weights.h5")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No model checkpoint found. Tried: {candidates} and {pattern}. Use --checkpoint to specify a path.")


def _build_model(config: dict[str, Any], input_shape: tuple[int, ...]):
    """Build the model architecture for inference (matches Trainer._build_model).

    Note: model is NOT compiled here because run_top_fp_extraction only does
    forward-pass inference — no optimizer needed. Compiling would create a fresh
    Adam with 2 variables, causing a spurious mismatch warning when load_weights()
    tries to restore the 92-variable optimizer from a fully-trained checkpoint.

    Args:
        config: Full config dict.
        input_shape: Model input shape (time_frames, mel_bins).

    Returns:
        Uncompiled Keras model (weights only, no optimizer).
    """
    from src.model.architecture import build_model

    model_cfg = config.get("model", {})

    model = build_model(
        input_shape=input_shape,
        first_conv_filters=model_cfg.get("first_conv_filters", 32),
        first_conv_kernel_size=model_cfg.get("first_conv_kernel_size", 5),
        stride=model_cfg.get("stride", 3),
        pointwise_filters=model_cfg.get("pointwise_filters", "64,64,64,64"),
        mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
        repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
        residual_connection=model_cfg.get("residual_connection", "0,1,1,1"),
        dropout_rate=model_cfg.get("dropout_rate", 0.08),
        l2_regularization=model_cfg.get("l2_regularization", 0.00003),
    )

    return model


def _pad_or_truncate(features: np.ndarray, max_time_frames: int) -> np.ndarray:
    """Pad or truncate feature array to fixed length.

    Args:
        features: 2D array (time_frames, mel_bins).
        max_time_frames: Target number of time frames.

    Returns:
        Array of shape (max_time_frames, mel_bins).
    """
    if features.shape[0] >= max_time_frames:
        return features[:max_time_frames]
    else:
        padding = np.zeros(
            (max_time_frames - features.shape[0], features.shape[1]),
            dtype=features.dtype,
        )
        return np.concatenate([features, padding], axis=0)


def run_top_fp_extraction(
    config: dict[str, Any],
    checkpoint_path: str | None = None,
    top_percent_override: float | None = None,
    threshold_override: float | None = None,
    log_file_override: str | None = None,
) -> dict[str, Any]:
    """Scan hard_negative files, run inference, and log top-N% false positives.

    Called automatically at end of training or via CLI.

    Args:
        config: Full training config dict.
        checkpoint_path: Path to model weights (.weights.h5). If None,
            auto-detects from checkpoints/ directory.
        top_percent_override: Override ``mining.top_fp_percent``.
        threshold_override: Override ``mining.extraction_confidence_threshold``.
        log_file_override: Override ``mining.extraction_log_file``.

    Returns:
        Dict with extraction results (also written to log file).
    """
    from src.data.features import FeatureConfig, MicroFrontend
    from src.data.ingestion import load_audio_wave

    # ── Config ────────────────────────────────────────────────────────────
    mining_cfg = config.get("mining", {})
    top_percent = mining_cfg.get("top_fp_percent", 5.0) if top_percent_override is None else top_percent_override
    confidence_threshold = mining_cfg.get("extraction_confidence_threshold", 0.8) if threshold_override is None else threshold_override
    log_file = log_file_override or mining_cfg.get("extraction_log_file", "logs/top_fp_extraction.json")
    batch_size = mining_cfg.get("extraction_batch_size", 128)

    hard_neg_dir = config.get("paths", {}).get("hard_negative_dir", "dataset/hard_negative")
    hard_neg_path = Path(hard_neg_dir)

    hardware = config.get("hardware", {})
    sample_rate = hardware.get("sample_rate_hz", 16000)
    mel_bins = hardware.get("mel_bins", 40)
    window_size_ms = hardware.get("window_size_ms", 30)
    window_step_ms = hardware.get("window_step_ms", 10)
    clip_duration_ms = hardware.get("clip_duration_ms", 1000)
    max_time_frames = int(clip_duration_ms / window_step_ms)

    # ── Model ─────────────────────────────────────────────────────────────
    checkpoint_path = _resolve_checkpoint(config, checkpoint_path)
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    model = _build_model(config, input_shape=(max_time_frames, mel_bins))
    # Build model by calling it on dummy data before loading weights
    _ = model(
        tf.zeros((1, max_time_frames, mel_bins), dtype=tf.float32),
        training=False,
    )
    model.load_weights(checkpoint_path)

    # ── Feature extractor ─────────────────────────────────────────────────
    feature_config = FeatureConfig(
        sample_rate=sample_rate,
        mel_bins=mel_bins,
        window_size_ms=window_size_ms,
        window_step_ms=window_step_ms,
    )
    frontend = MicroFrontend(feature_config)

    # ── Scan hard_negative directory ──────────────────────────────────────
    logger.info(f"Scanning hard_negative directory: {hard_neg_path}")
    audio_files = sorted(hard_neg_path.rglob("*.wav"))
    if not audio_files:
        logger.warning(f"No .wav files found in {hard_neg_path}")
        return {"total_hard_neg_files": 0, "files": []}

    logger.info(f"Found {len(audio_files)} audio files")

    # ── Batched inference ─────────────────────────────────────────────────
    all_scores: list[tuple[str, float]] = []  # (relative_path, score)

    batch_features: list[np.ndarray] = []
    batch_paths: list[str] = []

    for i, audio_file in enumerate(audio_files):
        try:
            audio = load_audio_wave(audio_file, target_sr=sample_rate)
            features = frontend.compute_mel_spectrogram(audio)
            features = _pad_or_truncate(features, max_time_frames)
            batch_features.append(features)
            batch_paths.append(str(audio_file))
        except (OSError, ValueError, RuntimeError, TypeError) as e:
            logger.warning(f"Failed to process {audio_file}: {e}")
            continue

        # Run inference when batch is full or at the end
        if len(batch_features) >= batch_size or i == len(audio_files) - 1:
            if batch_features:
                batch_arr = np.array(batch_features, dtype=np.float32)
                predictions = model(batch_arr, training=False)

                # Handle both binary [N] and multi-class [N, C] outputs
                if predictions.ndim == 2 and predictions.shape[1] > 1:
                    scores = predictions[:, 1].numpy()
                else:
                    scores = tf.reshape(predictions, [-1]).numpy()

                for path, score in zip(batch_paths, scores, strict=True):
                    all_scores.append((path, float(score)))

                batch_features.clear()
                batch_paths.clear()

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(audio_files)} files...")

    logger.info(f"Inference complete. Processed {len(all_scores)} files.")

    # ── Count true false positives (score >= threshold) for reporting ────
    false_positives = [(p, s) for p, s in all_scores if s >= confidence_threshold]
    logger.info(f"Found {len(false_positives)} false positives (score >= {confidence_threshold}) out of {len(all_scores)} files")

    # ── Take top N% of confirmed FPs, but always include all FPs when the set is small ─
    # Using only 5% of a small FP set (e.g. 35) collapses to 1 via int(35*0.05)=1.
    # Fix: compute a ceiling from 5% of ALL files; take the minimum of that ceiling
    # and the full FP count so we always log all confirmed FPs for small datasets,
    # while throttling to the most challenging ones when FPs number in the thousands.
    false_positives.sort(key=lambda x: x[1], reverse=True)
    ceiling = max(1, int(len(all_scores) * top_percent / 100.0))
    top_count = min(len(false_positives), ceiling)
    top_fps = false_positives[:top_count]

    logger.info(f"Selected {len(top_fps)} false positives (top {top_percent}% ceiling = {ceiling} of {len(all_scores)} total files)")

    # ── Build result ──────────────────────────────────────────────────────
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_checkpoint": str(checkpoint_path),
        "confidence_threshold": confidence_threshold,
        "top_percent": top_percent,
        "total_hard_neg_files": len(all_scores),
        "total_false_positives": len(false_positives),
        "top_fp_count": len(top_fps),
        "files": [{"path": p, "score": s} for p, s in top_fps],
    }

    # ── Write log file ────────────────────────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results written to {log_path}")
    return result


def move_extracted_files(
    config: dict[str, Any],
    log_file_override: str | None = None,
    dry_run: bool = False,
) -> None:
    """Read the JSON log and move files to the output directory.

    Args:
        config: Full config dict.
        log_file_override: Override path to the JSON log.
        dry_run: Print what would be moved without actually moving.
    """
    mining_cfg = config.get("mining", {})
    log_file = log_file_override or mining_cfg.get("extraction_log_file", "logs/top_fp_extraction.json")
    output_dir = mining_cfg.get("extraction_output_dir", "dataset/top5fps")
    hard_neg_dir = config.get("paths", {}).get("hard_negative_dir", "dataset/hard_negative")

    log_path = Path(log_file)
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}")
        print("Run extraction first (during training or standalone without --move-now).")
        sys.exit(1)

    with open(log_path) as f:
        data = json.load(f)

    files = data.get("files", [])
    if not files:
        print("No files to move.")
        return

    output_path = Path(output_dir)
    hard_neg_path = Path(hard_neg_dir)

    moved_count = 0
    skipped_count = 0
    errors: list[str] = []

    print(f"\n{'DRY RUN - ' if dry_run else ''}Moving {len(files)} files to {output_path}/")
    print(f"  Source: {hard_neg_path}")
    print(f"  Checkpoint: {data.get('model_checkpoint', 'unknown')}")
    print(f"  Threshold: {data.get('confidence_threshold', '?')}")
    print(f"  Top %: {data.get('top_percent', '?')}")
    print(f"  Extraction time: {data.get('timestamp', '?')}")
    print()

    for entry in files:
        src = Path(entry["path"])
        score = entry["score"]

        if not src.exists():
            skipped_count += 1
            print(f"  SKIP (not found): {src}")
            continue

        # Preserve speaker subdirectory structure
        try:
            rel = src.relative_to(hard_neg_path)
        except ValueError:
            # File is not under hard_neg_dir — use just filename
            rel = Path(src.name)

        dst = output_path / rel

        if dry_run:
            print(f"  WOULD MOVE: {src} → {dst}  (score={score:.4f})")
            moved_count += 1
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved_count += 1
                print(f"  MOVED: {src} → {dst}  (score={score:.4f})")
            except (OSError, shutil.Error) as e:
                errors.append(f"{src}: {e}")
                print(f"  ERROR: {src}: {e}")

    # Update log with move timestamp
    if not dry_run and moved_count > 0:
        data["moved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data["moved_count"] = moved_count
        data["move_errors"] = errors
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)

    action = "Would move" if dry_run else "Moved"
    print(f"\nSummary: {action} {moved_count} files, skipped {skipped_count}")
    if errors:
        print(f"  Errors: {len(errors)}")


# =============================================================================
# Consolidate prediction logs — merge per-epoch logs with file path mapping
# =============================================================================


def load_all_epoch_logs(log_dir: Path) -> dict[int, dict]:
    """Load all epoch-specific false prediction logs.

    Args:
        log_dir: Directory containing epoch_*_false_predictions.json files.

    Returns:
        Dict of epoch_number -> epoch_data.
    """
    epoch_logs: dict[int, dict] = {}
    for log_file in sorted(log_dir.glob("epoch_*_false_predictions.json")):
        try:
            # Parse epoch number from filename like "epoch_0010_false_predictions.json"
            parts = log_file.stem.split("_")
            epoch_num = int(parts[1])
        except (IndexError, ValueError):
            continue

        try:
            with open(log_file, "r") as f:
                epoch_logs[epoch_num] = json.load(f)
            logger.info(f"Loaded epoch {epoch_num}: {epoch_logs[epoch_num]['false_positive_count']} false positives")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {log_file}: {e}")

    if not epoch_logs:
        logger.warning(f"No epoch logs found in {log_dir}")

    return epoch_logs


def get_all_negative_hard_negative_files(
    config_dict: dict[str, Any],
) -> list[str]:
    """Get all file paths from negative and hard_negative directories.

    Args:
        config_dict: Full config as dict.

    Returns:
        Sorted list of .wav file paths.
    """
    paths_cfg = config_dict.get("paths", {})
    neg_dir = Path(paths_cfg.get("negative_dir", "./dataset/negative"))
    hard_neg_dir = Path(paths_cfg.get("hard_negative_dir", "./dataset/hard_negative"))

    all_files: list[str] = []
    for directory in [neg_dir, hard_neg_dir]:
        if directory.exists():
            files = [str(f) for f in sorted(directory.rglob("*.wav"))]
            all_files.extend(files)
            logger.info(f"Found {len(files)} files in {directory}")

    return all_files


def consolidate_prediction_logs(
    epoch_logs: dict[int, dict],
    all_files: list[str],
) -> dict[str, Any]:
    """Consolidate all epoch logs into single structure with file paths.

    Maps prediction indices to file paths via modulo mapping, counts per-file
    false positives, and builds a format compatible with the mining tool.

    Args:
        epoch_logs: Dict of epoch_number -> epoch_data.
        all_files: List of all negative/hard_negative file paths.

    Returns:
        Consolidated dict with epochs, metadata, stats.
    """
    all_false_positives: list[dict[str, Any]] = []
    fp_counter: Counter[str] = Counter()
    epoch_summary: dict[int, dict[str, Any]] = {}

    for epoch, epoch_data in sorted(epoch_logs.items()):
        epoch_preds = epoch_data["false_predictions"]

        # Map indices to file paths using recorded file_path from prediction; fall back to index hint only if unavailable
        for pred in epoch_preds:
            idx = pred["index"]
            if pred.get("file_path"):
                file_path = pred["file_path"]
            elif all_files:
                logger.warning(
                    "File path not recorded at write time; file attribution for index %d is unknown",
                    idx,
                )
                file_path = f"unknown_file_index_{idx}"
            else:
                file_path = f"unknown_file_index_{idx}"

            fp_counter[file_path] += 1

            pred_with_path = pred.copy()
            pred_with_path["file_path"] = file_path
            pred_with_path["epoch"] = epoch
            pred_with_path["true_label"] = pred.get("true_label", "negative")
            all_false_positives.append(pred_with_path)

        epoch_summary[epoch] = {
            "false_positive_count": epoch_data["false_positive_count"],
            "fp_threshold": epoch_data["fp_threshold"],
            "timestamp": epoch_data["timestamp"],
        }

    logger.info(f"Consolidated {len(all_false_positives)} false positive predictions across {len(epoch_logs)} epochs")

    # Build epochs dict in format: {"10": {...}, "20": {...}}
    epochs_dict: dict[str, dict[str, Any]] = {}
    for epoch_num, summary_data in epoch_summary.items():
        epochs_dict[str(epoch_num)] = {
            "timestamp": summary_data["timestamp"],
            "epoch": epoch_num,
            "fp_threshold": summary_data["fp_threshold"],
            "total_val_samples": 0,  # Not available in our data
            "false_positive_count": summary_data["false_positive_count"],
            "false_predictions": [p for p in all_false_positives if p.get("epoch") == epoch_num],
        }

    return {
        "epochs": epochs_dict,
        "metadata": {
            "created_at": (all_false_positives[0].get("timestamp", "") if all_false_positives else ""),
            "model_checkpoint": "consolidated",
            "fp_threshold": (epoch_logs[min(epoch_logs.keys())]["fp_threshold"] if epoch_logs else 0.8),
        },
        "total_epochs": len(epoch_logs),
        "total_false_positives": len(all_false_positives),
        "unique_files_false_predicted": len(fp_counter),
        "per_file_stats": dict(fp_counter.most_common()),  # Already sorted by frequency
    }


def write_consolidated_log(output_path: Path, consolidated: dict[str, Any]) -> None:
    """Write consolidated false predictions to JSON file.

    Args:
        output_path: Output file path.
        consolidated: Consolidated data dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    logger.info(f"Wrote consolidated log to {output_path}")


def generate_statistics_report(
    consolidated: dict[str, Any],
    output_path: Path,
) -> Counter[str]:
    """Generate human-readable statistics report.

    Args:
        consolidated: Consolidated prediction data.
        output_path: Path to write the text report.

    Returns:
        Counter of file_path -> false prediction count.
    """
    # Extract all predictions from epochs dict
    epochs_dict = consolidated.get("epochs", {})
    all_predictions: list[dict[str, Any]] = []
    for epoch_data in epochs_dict.values():
        epoch_preds = epoch_data.get("false_predictions", [])
        all_predictions.extend(epoch_preds)

    # Get predictions from negative/hard_neg only
    neg_hard_neg_preds = [p for p in all_predictions if p.get("true_label") in ("negative", "hard_negative")]

    # Count per file
    fp_counter: Counter[str] = Counter()
    for p in neg_hard_neg_preds:
        file_path = p.get("file_path")
        if file_path and not file_path.startswith("unknown"):
            fp_counter[file_path] += 1

    # Update consolidated with all predictions for other uses
    consolidated["false_predictions"] = all_predictions

    report_lines = [
        "=" * 80,
        "FALSE PREDICTION STATISTICS REPORT",
        "=" * 80,
        f"Total epochs: {consolidated['total_epochs']}",
        f"Total false predictions (all splits): {consolidated['total_false_positives']}",
        f"Unique files with false predictions: {consolidated['unique_files_false_predicted']}",
        f"False predictions from negative/hard_negative splits: {len(neg_hard_neg_preds)}",
        "",
        "PER-FILE FALSE PREDICTION COUNT (Top 50)",
        "-" * 80,
        f"{'Count':>8} | {'File Path'}",
        "-" * 80,
    ]

    for file_path, count in fp_counter.most_common(50):
        short_path = file_path if len(file_path) <= 70 else "..." + file_path[-67:]
        report_lines.append(f"{count:>8} | {short_path}")

    report_lines.extend(
        [
            "",
            "TOP 10 MOST FALSE-PREDICTED FILES",
            "-" * 80,
        ]
    )

    for i, (file_path, count) in enumerate(fp_counter.most_common(10), 1):
        report_lines.append(f"{i}. {count} times: {file_path}")

    report_lines.extend(
        [
            "",
            "=" * 80,
        ]
    )

    report = "\n".join(report_lines)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Wrote statistics report to {output_path}")
    logger.info(report)

    return fp_counter


def move_top_false_predicted_files(
    fp_counter: Counter[str],
    dest_dir: Path,
    top_n: int = 5,
    dry_run: bool = False,
) -> list[Path]:
    """Move top N most false-predicted files to destination directory.

    Args:
        fp_counter: Counter of file_path -> count.
        dest_dir: Destination directory.
        top_n: Number of top files to move.
        dry_run: Simulate without copying.

    Returns:
        List of destination file paths.
    """
    dest_files: list[Path] = []

    logger.info(f"Moving top {top_n} most false-predicted files to {dest_dir}")

    for i, (file_path, count) in enumerate(fp_counter.most_common(top_n), 1):
        if not file_path:
            continue

        src = Path(file_path)
        if not src.exists():
            logger.warning(f"File not found: {src}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        # Prefix with count for easy identification
        dest_name = f"{count:03d}_false_{src.name}"
        dest = dest_dir / dest_name

        if dry_run:
            logger.info(f"[DRY RUN] Would copy: {src} -> {dest}")
        else:
            logger.info(f"[{i}/{top_n}] Copying ({count} false predictions): {src.name}")
            logger.info(f"  Source: {src}")
            logger.info(f"  Destination: {dest}")
            shutil.copy2(src, dest)
            dest_files.append(dest)

    return dest_files


# =============================================================================
# Post-training mining from prediction logs (mine_hard_negatives CLI)
# =============================================================================


def load_prediction_log(log_path: Path) -> dict[str, Any]:
    """Load the false predictions JSON log file.

    Args:
        log_path: Path to the JSON log file.

    Returns:
        Dictionary containing the log data.

    Raises:
        FileNotFoundError: If log file does not exist.
        json.JSONDecodeError: If log file is invalid JSON.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Prediction log not found: {log_path}")

    with open(log_path, "r") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Prediction log root must be a JSON object: {log_path}")
    return cast(dict[str, Any], loaded)


def filter_epochs_by_min_epoch(log_data: dict[str, Any], min_epoch: int) -> dict[int, dict]:
    """Filter epochs to only include those >= min_epoch.

    Args:
        log_data: The loaded log data.
        min_epoch: Minimum epoch number to include.

    Returns:
        Dictionary of epoch_number -> epoch_data.
    """
    filtered = {}
    epochs = log_data.get("epochs", {})

    for epoch_str, epoch_data in epochs.items():
        try:
            epoch_num = int(epoch_str)
            if epoch_num >= min_epoch:
                filtered[epoch_num] = epoch_data
        except ValueError:
            logger.warning(f"Skipping invalid epoch key '{epoch_str}' (not an integer)")
            continue

    return filtered


def collect_false_predictions(
    filtered_epochs: dict[int, dict],
    top_k: int,
) -> list[dict]:
    """Collect false predictions from filtered epochs.

    Args:
        filtered_epochs: Dictionary of epoch_number -> epoch_data.
        top_k: Maximum predictions per epoch.

    Returns:
        List of false prediction entries with epoch, index, score.
    """
    all_predictions = []

    for epoch_num, epoch_data in sorted(filtered_epochs.items()):
        predictions = epoch_data.get("false_predictions", [])

        # Sort by score descending and take top_k
        sorted_preds = sorted(predictions, key=lambda x: x.get("score", 0), reverse=True)
        top_predictions = sorted_preds[:top_k]

        for pred in top_predictions:
            new_pred = dict(pred)
            new_pred["epoch"] = epoch_num
            all_predictions.append(new_pred)

    return all_predictions


def deduplicate_by_hash(
    predictions: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    """Deduplicate predictions by file hash.

    Args:
        predictions: List of prediction entries.

    Returns:
        Tuple of (unique_predictions, hash_to_path mapping).

    Raises:
        ValueError: If predictions lack file_path/path fields.
    """
    seen_hashes: dict[str, str] = {}
    unique_predictions = []

    for pred in predictions:
        # Check if prediction contains path information
        file_path_str = pred.get("file_path") or pred.get("path")
        if not file_path_str:
            raise ValueError("Prediction log missing 'file_path' or 'path' field. Regenerate logs with path information to enable deduplication.")

        file_path = Path(file_path_str)

        if not file_path.exists():
            continue

        file_hash = compute_file_hash(file_path)

        if file_hash not in seen_hashes:
            seen_hashes[file_hash] = str(file_path)
            pred["file_path"] = str(file_path)
            pred["file_hash"] = file_hash
            unique_predictions.append(pred)

    return unique_predictions, seen_hashes


def copy_files_to_mined_dir(
    predictions: list[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Copy false positive files to the mined directory.

    Args:
        predictions: List of prediction entries with file_path.
        output_dir: Destination directory.
        dry_run: If True, do not actually copy files.

    Returns:
        Tuple of (copied_count, skipped_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for pred in predictions:
        file_path = Path(pred.get("file_path", ""))
        if not file_path.exists():
            skipped += 1
            continue

        # Generate destination filename with score
        score = pred.get("score", 0.0)
        epoch = pred.get("epoch", 0)
        dest_name = f"{file_path.stem}_e{epoch}_s{score:.3f}{file_path.suffix}"
        dest_path = output_dir / dest_name

        if dry_run:
            logger.info(f"[DRY RUN] Would copy: {file_path} -> {dest_path}")
            copied += 1
        else:
            try:
                shutil.copy2(file_path, dest_path)
                logger.debug(f"Copied: {file_path} -> {dest_path}")
                copied += 1
            except (OSError, shutil.Error) as e:
                logger.warning(f"Failed to copy {file_path}: {e}")
                skipped += 1

    return copied, skipped


def generate_mining_summary(
    total_epochs: int,
    filtered_epochs: int,
    total_predictions: int,
    unique_predictions: int,
    copied: int,
    skipped: int,
    output_dir: Path,
) -> None:
    """Generate a summary table of the mining operation.

    Uses Rich for formatted console output.

    Args:
        total_epochs: Total epochs in log.
        filtered_epochs: Epochs after filtering.
        total_predictions: Total false predictions found.
        unique_predictions: Unique predictions after deduplication.
        copied: Number of files copied.
        skipped: Number of files skipped.
        output_dir: Output directory path.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="Hard Negative Mining Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Epochs in Log", str(total_epochs))
    table.add_row("Epochs After Filtering", str(filtered_epochs))
    table.add_row("Total False Predictions", str(total_predictions))
    table.add_row("Unique Predictions", str(unique_predictions))
    table.add_row("Files Copied", str(copied))
    table.add_row("Files Skipped", str(skipped))
    table.add_row("Output Directory", str(output_dir))

    console.print()
    console.print(table)


def mine_from_prediction_logs(
    prediction_log: Path,
    output_dir: Path,
    min_epoch: int = 10,
    top_k: int = 100,
    deduplicate: bool = False,
    dry_run: bool = False,
) -> int:
    """Post-training mining: read prediction log and copy files.

    Args:
        prediction_log: Path to false_predictions.json.
        output_dir: Destination for mined files.
        min_epoch: Minimum epoch to consider.
        top_k: Top K files per epoch.
        deduplicate: Enable hash-based deduplication.
        dry_run: Show what would be copied without copying.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    from rich.console import Console

    console = Console()

    try:
        # Load the prediction log
        console.print(f"[bold blue]Loading prediction log: {prediction_log}[/bold blue]")
        log_data = load_prediction_log(prediction_log)

        metadata = log_data.get("metadata", {})
        total_epochs = len(log_data.get("epochs", {}))

        console.print(f"Found {total_epochs} epochs in log")
        console.print(f"FP Threshold: {metadata.get('fp_threshold', 'unknown')}")

        # Filter epochs
        filtered_epochs = filter_epochs_by_min_epoch(log_data, min_epoch)
        console.print(f"[bold blue]Filtering epochs >= {min_epoch}[/bold blue]")
        console.print(f"Epochs after filtering: {len(filtered_epochs)}")

        if not filtered_epochs:
            console.print("[yellow]No epochs match the filter criteria. Exiting.[/yellow]")
            return 0

        # Collect false predictions
        console.print(f"[bold blue]Collecting top-{top_k} false predictions per epoch...[/bold blue]")
        predictions = collect_false_predictions(filtered_epochs, top_k)
        console.print(f"Total predictions collected: {len(predictions)}")

        if not predictions:
            console.print("[yellow]No false predictions found. Exiting.[/yellow]")
            return 0

        # Deduplicate if requested
        if deduplicate:
            console.print("[bold blue]Deduplicating predictions by file hash...[/bold blue]")
            try:
                unique_predictions, _ = deduplicate_by_hash(predictions)
                console.print(f"[green]Deduplicated {len(predictions)} -> {len(unique_predictions)} predictions[/green]")
            except ValueError as e:
                console.print(f"[red]Deduplication failed: {e}[/red]")
                console.print("[yellow]Continuing without deduplication[/yellow]")
                unique_predictions = predictions
        else:
            unique_predictions = predictions

        # Copy files
        console.print(f"[bold blue]{'Simulating' if dry_run else 'Copying'} files to {output_dir}...[/bold blue]")
        copied, skipped = copy_files_to_mined_dir(unique_predictions, output_dir, dry_run)

        # Generate summary
        generate_mining_summary(
            total_epochs=total_epochs,
            filtered_epochs=len(filtered_epochs),
            total_predictions=len(predictions),
            unique_predictions=len(unique_predictions),
            copied=copied,
            skipped=skipped,
            output_dir=output_dir,
        )

        if dry_run:
            console.print("\n[bold yellow]This was a dry run. No files were actually copied.[/bold yellow]")
            console.print("Run without --dry-run to copy files.")

        return 0

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return 1
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error: Invalid JSON in prediction log: {e}[/bold red]")
        return 1
    except (OSError, ValueError, RuntimeError, TypeError) as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        logger.exception("Full traceback:")
        return 1


# =============================================================================
# CLI entry point with subcommands
# =============================================================================


def main():
    """CLI entry point for mww-mine-hard-negatives.

    Subcommands:
        mine              Post-training mining from prediction logs
        extract-top-fps   Extract top N% false positives via model inference
        consolidate-logs  Consolidate per-epoch logs with file path mapping
    """
    parser = argparse.ArgumentParser(
        prog="mww-mine-hard-negatives",
        description="Unified hard negative mining and false prediction tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mine hard negatives from prediction log
  mww-mine-hard-negatives mine --prediction-log logs/false_predictions.json

  # Dry run
  mww-mine-hard-negatives mine --prediction-log logs/false_predictions.json --dry-run

  # Extract top false positives
  mww-mine-hard-negatives extract-top-fps --config standard

  # Move extracted files
  mww-mine-hard-negatives extract-top-fps --config standard --move-now

  # Consolidate per-epoch logs
  mww-mine-hard-negatives consolidate-logs --config standard
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── mine subcommand ───────────────────────────────────────────────────
    mine_parser = subparsers.add_parser(
        "mine",
        help="Post-training hard negative mining from prediction logs",
    )
    mine_parser.add_argument(
        "--prediction-log",
        type=Path,
        required=True,
        help="Path to the false_predictions.json log file",
    )
    mine_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dataset/hard_negative/mined"),
        help="Output directory for mined hard negatives (default: ./dataset/hard_negative/mined)",
    )
    mine_parser.add_argument(
        "--min-epoch",
        type=int,
        default=10,
        help="Minimum epoch to consider (default: 10)",
    )
    mine_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top K files per epoch (default: 100)",
    )
    mine_parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Enable hash-based deduplication",
    )
    mine_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying",
    )
    mine_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # ── extract-top-fps subcommand ────────────────────────────────────────
    extract_parser = subparsers.add_parser(
        "extract-top-fps",
        help="Extract top N%% most confident false positives via model inference",
    )
    extract_parser.add_argument(
        "--move-now",
        action="store_true",
        help="Move files listed in the JSON log to the output directory",
    )
    extract_parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name or path to config YAML (default: standard)",
    )
    extract_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (default: auto-detect from checkpoints/)",
    )
    extract_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    extract_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Override JSON log file path",
    )
    extract_parser.add_argument(
        "--top-percent",
        type=float,
        default=None,
        help="Override top percent (default from config)",
    )
    extract_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold (default from config)",
    )

    # ── consolidate-logs subcommand ───────────────────────────────────────
    consolidate_parser = subparsers.add_parser(
        "consolidate-logs",
        help="Consolidate per-epoch logs, add file paths, generate statistics",
    )
    consolidate_parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name",
    )
    consolidate_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory containing epoch logs (default: logs/)",
    )
    consolidate_parser.add_argument(
        "--output",
        type=str,
        default="logs/false_predictions.json",
        help="Output consolidated false predictions file",
    )
    consolidate_parser.add_argument(
        "--stats-output",
        type=str,
        default="logs/false_predictions_stats.txt",
        help="Output statistics report file",
    )
    consolidate_parser.add_argument(
        "--move-to",
        type=str,
        default="dataset/most_false_predicted",
        help="Destination directory for top false-predicted files",
    )
    consolidate_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top files to move (default: 5)",
    )
    consolidate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate moves without copying files",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Setup Rich logging for all project logs
    _configure_mining_logging(verbose=getattr(args, "verbose", False))

    # ── mine ──────────────────────────────────────────────────────────────
    if args.command == "mine":
        return mine_from_prediction_logs(
            prediction_log=args.prediction_log,
            output_dir=args.output_dir,
            min_epoch=args.min_epoch,
            top_k=args.top_k,
            deduplicate=args.deduplicate,
            dry_run=args.dry_run,
        )

    # ── extract-top-fps ──────────────────────────────────────────────────
    elif args.command == "extract-top-fps":
        import dataclasses

        from config.loader import load_full_config

        config_obj = load_full_config(args.config)
        config = dataclasses.asdict(config_obj)

        if args.move_now:
            move_extracted_files(
                config,
                log_file_override=args.log_file,
                dry_run=args.dry_run,
            )
        else:
            if args.dry_run:
                logger.info("DRY RUN — skipping extraction (would scan and report)")
                confidence_threshold = float(args.threshold) if args.threshold is not None else 0.0
                top_percent = float(args.top_percent) if args.top_percent is not None else 0.0
                result = {
                    "total_hard_neg_files": 0,
                    "confidence_threshold": confidence_threshold,
                    "total_false_positives": 0,
                    "top_percent": top_percent,
                    "top_fp_count": 0,
                    "files": [],
                }
            else:
                result = run_top_fp_extraction(
                    config,
                    checkpoint_path=args.checkpoint,
                    top_percent_override=args.top_percent,
                    threshold_override=args.threshold,
                    log_file_override=args.log_file,
                )

            if args.dry_run:
                print("\nDRY RUN Results:")
            else:
                print("\nExtraction Results:")

            print(f"  Total hard_neg files scanned: {result['total_hard_neg_files']}")
            print(f"  False positives (score >= {result['confidence_threshold']}): {result['total_false_positives']}")
            print(f"  Top {result['top_percent']}% selected: {result['top_fp_count']}")

            if result["files"]:
                print("\n  Top 10 most confident false positives:")
                for entry in result["files"][:10]:
                    print(f"    {entry['score']:.4f}  {entry['path']}")

            if not args.dry_run:
                mining_cfg = config.get("mining", {})
                log_file = args.log_file or mining_cfg.get("extraction_log_file", "logs/top_fp_extraction.json")
                print(f"\n  Results saved to: {log_file}")
                print("  To move files, run: mww-mine-hard-negatives extract-top-fps --move-now")

        return 0

    # ── consolidate-logs ─────────────────────────────────────────────────
    elif args.command == "consolidate-logs":
        import dataclasses

        from config.loader import load_full_config

        logger.info(f"Loading config: {args.config}")
        full_config = load_full_config(args.config)
        config_dict = dataclasses.asdict(full_config)

        log_dir = Path(args.log_dir or "logs")

        # Step 1: Load all epoch logs
        logger.info("=" * 80)
        logger.info("STEP 1: Loading epoch logs")
        logger.info("=" * 80)
        epoch_logs = load_all_epoch_logs(log_dir)
        if not epoch_logs:
            logger.error("No epoch logs found. Exiting.")
            return 1

        # Step 2: Get list of all negative/hard_neg files
        logger.info("=" * 80)
        logger.info("STEP 2: Getting all negative/hard_negative file paths")
        logger.info("=" * 80)
        all_files = get_all_negative_hard_negative_files(config_dict)

        # Step 3: Consolidate predictions
        logger.info("=" * 80)
        logger.info("STEP 3: Consolidating predictions")
        logger.info("=" * 80)
        consolidated = consolidate_prediction_logs(epoch_logs, all_files)

        # Step 4: Write consolidated log
        logger.info("=" * 80)
        logger.info("STEP 4: Writing consolidated log file")
        logger.info("=" * 80)
        output_path = Path(args.output)
        write_consolidated_log(output_path, consolidated)

        # Step 5: Generate statistics report
        logger.info("=" * 80)
        logger.info("STEP 5: Generating statistics report")
        logger.info("=" * 80)
        stats_output = Path(args.stats_output)
        fp_counter = generate_statistics_report(consolidated, stats_output)

        # Step 6: Move top false-predicted files
        logger.info("=" * 80)
        logger.info("STEP 6: Moving top false-predicted files")
        logger.info("=" * 80)
        dest_dir = Path(args.move_to)
        dest_files = move_top_false_predicted_files(
            fp_counter,
            dest_dir,
            top_n=args.top_n,
            dry_run=args.dry_run,
        )

        if not args.dry_run and dest_files:
            logger.info(f"Successfully copied {len(dest_files)} files to {dest_dir}")
        elif args.dry_run:
            logger.info(f"[DRY RUN COMPLETED] Would have copied {len(dest_files) if dest_files else 0} files")

        logger.info("=" * 80)
        logger.info("DONE!")
        logger.info("=" * 80)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
