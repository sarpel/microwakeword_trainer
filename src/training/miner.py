"""Hard negative mining for wake word training."""

from __future__ import annotations

import heapq
import logging
import os
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


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
        # Each entry: (negative_score, global_index, feature, label, prediction)
        # negative_score is -prediction so that highest predictions are popped first
        hard_negative_heap: list[tuple[float, int, np.ndarray, int, float]] = []
        global_offset = 0

        # Handle both generator factories and direct generators
        if callable(data_generator):
            gen = data_generator()
        else:
            gen = data_generator

        for features, labels, _ in gen:
            predictions = model(features, training=False).numpy()

            # Get hard indices local to this batch
            local_hard_indices = self.get_hard_samples(labels, predictions)

            # Add hard negatives to heap
            for local_idx in local_hard_indices:
                global_idx = global_offset + int(local_idx)
                pred_score = float(predictions[local_idx])
                # Use negative score for max-heap behavior with min-heap
                heap_entry = (-pred_score, global_idx, features[local_idx].copy(), int(labels[local_idx]), pred_score)

                if len(hard_negative_heap) < self.max_samples:
                    heapq.heappush(hard_negative_heap, heap_entry)
                elif -pred_score > hard_negative_heap[0][0]:
                    # This hard negative has higher score than the lowest in heap
                    heapq.heapreplace(hard_negative_heap, heap_entry)

            global_offset += len(features)

        # Check if we found any hard negatives
        if not hard_negative_heap:
            mining_result = {
                "epoch": epoch,
                "num_hard_negatives": 0,
                "indices": [],
                "avg_prediction": 0.0,
            }
            self.mining_history.append(mining_result)
            # Limit history size to prevent memory leak
            if len(self.mining_history) > self._max_history_size:
                self.mining_history = self.mining_history[-self._max_history_size :]
            return mining_result

        # Extract results from heap
        # Sort by prediction score (descending) - heap is sorted by negative score
        sorted_hard = sorted(hard_negative_heap, key=lambda x: x[0])

        selected_indices = [int(entry[1]) for entry in sorted_hard]
        avg_prediction = float(np.mean([entry[4] for entry in sorted_hard]))

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

        # Create hard negative records
        hard_negative_records = [
            {
                "global_index": int(entry[1]),
                "feature": entry[2],
                "label": entry[3],
                "prediction": entry[4],
            }
            for entry in sorted_hard
        ]

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
            filepath = os.path.join(self.output_dir, f"hard_negatives_step_{len(self.mining_history)}.npz")

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
