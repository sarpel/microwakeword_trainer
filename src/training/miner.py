"""Hard example mining module for training improved wake word detection."""

import os

import numpy as np


def mine_hard_examples(
    features: np.ndarray,
    labels: np.ndarray,
    model,
    n_samples: int = 1000,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine hard examples for training.

    Identifies negative samples that the model incorrectly predicts as positive
    (false positives) - these are the most valuable for improving the model.

    Args:
        features: Feature array [n_samples, feature_dim]
        labels: Label array [n_samples] (0=negative, 1=positive)
        model: Trained model with predict method
        n_samples: Number of hard samples to mine
        threshold: Classification threshold for FP detection

    Returns:
        Tuple of (hard_features, hard_labels) for mined samples
    """
    predictions = model.predict(features, verbose=0)
    labels = np.asarray(labels).reshape(-1)
    predictions = model.predict(features, verbose=0)

    # Handle both single and batch prediction formats
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()

    # Find false positives: negative samples predicted as positive
    negative_mask = labels == 0
    fp_predictions = predictions[negative_mask]
    fp_indices = np.where(negative_mask)[0]

    # Filter to those above threshold
    hard_mask = fp_predictions > threshold
    hard_indices = fp_indices[hard_mask]
    hard_scores = fp_predictions[hard_mask]

    # Sort by confidence (most confident false positives first)
    sort_order = np.argsort(-hard_scores)
    hard_indices = hard_indices[sort_order]

    # Limit to n_samples
    if len(hard_indices) > n_samples:
        hard_indices = hard_indices[:n_samples]

    return features[hard_indices], labels[hard_indices]


class HardExampleMiner:
    """Hard example mining for improved training.

    Implements iterative hard negative mining to progressively improve
    the model's ability to distinguish false positives.
    """

    def __init__(
        self,
        strategy: str = "confidence",
        fp_threshold: float = 0.8,
        max_samples: int = 5000,
        mining_interval_epochs: int = 5,
        output_dir: str = "./data/raw/hard_negative",
    ):
        """Initialize miner.

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

        # Storage for hard negatives
        self.hard_negatives: list[dict] = []
        self.mining_history: list[dict] = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def get_hard_samples(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Get hard samples based on predictions.

        Identifies:
        - False positives (FP): negative samples predicted as positive
        - Hard negatives: negative samples with high prediction scores

        Args:
            features: Feature array [n_samples, feature_dim]
            labels: Label array [n_samples]
            predictions: Model predictions [n_samples]

        Returns:
            Indices of hard samples
        """
        # Flatten labels and predictions if needed
        labels = np.asarray(labels).reshape(-1)
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        if self.strategy == "confidence":
            # Hard negatives: negative samples with high prediction scores
            negative_mask = labels == 0
            negative_predictions = predictions[negative_mask]
            negative_indices = np.where(negative_mask)[0]

            # Get indices sorted by prediction confidence (descending)
            hard_order = np.argsort(-negative_predictions)
            hard_indices = negative_indices[hard_order]

            # Filter to those above threshold
            valid_mask = negative_predictions[hard_order] > self.fp_threshold
            hard_indices = hard_indices[valid_mask]

        elif self.strategy == "entropy":
            # Entropy-based: samples where model is most uncertain
            # Low entropy = high confidence (both positive and negative)
            epsilon = 1e-10
            entropy = -(predictions * np.log(predictions + epsilon) + (1 - predictions) * np.log(1 - predictions + epsilon))

            # Get negative samples with highest entropy (most uncertain)
            negative_mask = labels == 0
            negative_entropy = entropy[negative_mask]
            negative_indices = np.where(negative_mask)[0]

            # Sort by entropy (most uncertain first)
            hard_order = np.argsort(-negative_entropy)
            hard_indices = negative_indices[hard_order]

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Limit to max_samples
        if len(hard_indices) > self.max_samples:
            hard_indices = hard_indices[: self.max_samples]

        return hard_indices

    def mine_from_dataset(
        self,
        model,
        data_generator,
        epoch: int,
    ) -> dict:
        """Mine hard negatives from a data generator.

        Args:
            model: Trained model
            data_generator: Generator yielding (features, labels, weights) tuples
            epoch: Current training epoch

        Returns:
            Mining result dictionary with keys: epoch, num_hard_negatives,
            indices (global dataset indices), avg_prediction.
        """
        all_hard_global_indices = []
        all_predictions = []
        all_labels = []
        all_features = []

        # Collect predictions across dataset, tracking global index offset
        global_offset = 0
        for features, labels, _ in data_generator:
            predictions = model.predict(features, verbose=0)
            all_features.append(features)
            all_predictions.append(predictions)
            all_labels.append(labels)

            # Get hard indices local to this batch, then convert to global
            local_hard_indices = self.get_hard_samples(features, labels, predictions)
            global_hard_indices = local_hard_indices + global_offset
            all_hard_global_indices.extend(global_hard_indices.tolist())

            global_offset += len(features)

        # Combine all predictions
        all_features = np.concatenate(all_features)
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        # Deduplicate indices while preserving original (hardness-ranked) order
        unique_indices = list(dict.fromkeys(all_hard_global_indices))
        selected_indices = unique_indices[: self.max_samples]

        # Store mining result
        mining_result = {
            "epoch": epoch,
            "num_hard_negatives": len(unique_indices),
            "indices": selected_indices,
            "avg_prediction": (float(np.mean(all_predictions[unique_indices])) if unique_indices else 0.0),
        }

        self.mining_history.append(mining_result)
        hard_negative_records = [
            {
                "global_index": int(idx),
                "feature": all_features[idx],
                "label": int(all_labels[idx]),
                "prediction": float(all_predictions[idx]),
            }
            for idx in selected_indices
        ]
        self.hard_negatives.extend(hard_negative_records)

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
        hard_indices = self.get_hard_samples(features, labels, predictions)

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
