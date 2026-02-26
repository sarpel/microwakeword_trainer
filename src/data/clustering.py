"""Clustering module for speaker diversity and leakage auditing.

Provides:
- Speaker clustering using WavLM embeddings
- Similarity threshold-based clustering
- Leakage audit for train/test separation
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Optional dependencies for embeddings and clustering
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    from transformers import WavLMModel, WavLMProcessor  # type: ignore[attr-defined]

    HAS_TRANSFORMERS = True
except ImportError:
    torch = None  # type: ignore
    WavLMModel = None  # type: ignore
    WavLMProcessor = None  # type: ignore
    HAS_TRANSFORMERS = False

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore
    HAS_LIBROSA = False


logger = logging.getLogger(__name__)
MAX_DENSE_SIMILARITY_BYTES = 1_000_000_000  # ~1GB safety limit


def extract_wavlm_embeddings(
    audio_paths: List[str],
    model_name: str = "microsoft/wavlm-base-plus",
    device: Optional[str] = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract speaker embeddings using WavLM.

    Args:
        audio_paths: List of paths to audio files
        model_name: WavLM model name
        device: Device to run model on ("cuda" or "cpu"). If None, auto-detect.
        batch_size: Batch size for processing

    Returns:
        Array of embeddings [n_samples, embedding_dim]
    """
    if not HAS_TRANSFORMERS or not HAS_SKLEARN:
        raise ImportError(
            "transformers and sklearn are required for speaker clustering. "
            "Install: pip install transformers torch scikit-learn"
        )
    if not HAS_LIBROSA:
        raise ImportError(
            "librosa is required to load audio for speaker clustering. "
            "Install: pip install librosa"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    processor = WavLMProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
    model = WavLMModel.from_pretrained(model_name)
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()

    embeddings = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        batch_embeddings = []

        for audio_path in batch_paths:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)

            # Process audio
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean-pool temporal hidden states to obtain speaker embedding
                embedding = (
                    outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                )
                batch_embeddings.append(embedding)

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def cluster_samples(
    features: np.ndarray,
    n_clusters: int = 100,
    similarity_threshold: float = 0.72,
    method: str = "agglomerative",
) -> np.ndarray:
    """Cluster feature samples for diversity.

    Args:
        features: Feature array [n_samples, feature_dim]
        n_clusters: Target number of clusters (may differ based on threshold)
        similarity_threshold: Cosine similarity threshold for clustering
        method: Clustering method ("agglomerative" or "threshold")

    Returns:
        Cluster labels [n_samples]
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for clustering")

    if method == "agglomerative":
        # Use AgglomerativeClustering with cosine affinity
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(features)

    elif method == "threshold":
        # Similarity-based clustering with threshold
        labels = _similarity_clustering(features, similarity_threshold)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


def _similarity_clustering(
    features: np.ndarray,
    threshold: float = 0.72,
) -> np.ndarray:
    """Similarity-based clustering with threshold.

    Clusters samples that have cosine similarity above threshold.

    Args:
        features: Feature array
        threshold: Similarity threshold for same cluster

    Returns:
        Cluster labels
    """
    n_samples = len(features)
    labels = np.full(n_samples, -1, dtype=int)
    current_label = 0

    n = int(features.shape[0])
    estimated_bytes = n * n * 8
    use_dense_similarity = estimated_bytes <= MAX_DENSE_SIMILARITY_BYTES

    if not use_dense_similarity:
        logger.warning(
            "Skipping dense cosine similarity matrix: n=%d would require ~%.2f GB. "
            "Using memory-safe row-wise pairwise computation.",
            n,
            estimated_bytes / (1024**3),
        )
        similarity = None
    else:
        similarity = cosine_similarity(features)

    # Find connected components based on threshold
    visited = set()

    for i in range(n_samples):
        if i in visited:
            continue

        # BFS to find all connected samples
        queue = deque([i])
        labels[i] = current_label
        visited.add(i)

        while queue:
            node = queue.popleft()

            # Find all similar samples not yet visited
            if similarity is not None:
                neighbors = np.where(similarity[node] >= threshold)[0]
            else:
                # Compute one row at a time to avoid O(n^2) memory blow-up
                sim_row = cosine_similarity(features[node : node + 1], features)[0]
                neighbors = np.where(sim_row >= threshold)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    labels[neighbor] = current_label
                    queue.append(neighbor)

        current_label += 1

    return labels


def select_diverse_samples(
    features: np.ndarray,
    labels: np.ndarray,
    samples_per_cluster: int = 10,
    selection_strategy: str = "random",
) -> np.ndarray:
    """Select diverse samples from clusters.

    Args:
        features: Feature array [n_samples, feature_dim]
        labels: Cluster labels [n_samples]
        samples_per_cluster: Number of samples to select per cluster
        selection_strategy: How to select ("random", "centroid", "boundary")

    Returns:
        Indices of selected samples
    """
    unique_labels = np.unique(labels)
    selected_indices = []

    for label in unique_labels:
        # Get indices for this cluster
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features[cluster_mask]

        if len(cluster_indices) == 0:
            continue

        n_select = min(samples_per_cluster, len(cluster_indices))

        if selection_strategy == "random":
            # Random selection
            chosen = np.random.choice(cluster_indices, n_select, replace=False)

        elif selection_strategy == "centroid":
            # Select samples closest to centroid
            centroid = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            sorted_indices = cluster_indices[np.argsort(distances)]
            chosen = sorted_indices[:n_select]

        elif selection_strategy == "boundary":
            # Select samples most different from each other
            chosen = _select_boundary_samples(
                cluster_indices, cluster_features, n_select
            )

        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        selected_indices.extend(chosen.tolist())

    return np.array(selected_indices)


def _select_boundary_samples(
    indices: np.ndarray,
    features: np.ndarray,
    n_select: int,
) -> np.ndarray:
    """Select boundary samples (most different from each other)."""
    if len(indices) <= n_select:
        return indices

    # Greedy selection: iteratively add most different sample
    selected = [indices[0]]
    remaining = list(indices[1:])

    while len(selected) < n_select and remaining:
        # Find sample most different from current selections
        selected_features = features[[i in selected for i in indices]]

        max_min_dist = -1
        best_idx = remaining[0]

        for idx in remaining:
            feat = features[indices == idx]
            min_dist = np.min(np.linalg.norm(selected_features - feat, axis=1))

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(selected)


def audit_leakage(
    train_features: np.ndarray,
    test_features: np.ndarray,
    similarity_threshold: float = 0.9,
) -> Dict[str, Any]:
    """Audit for speaker leakage between train and test sets.

    Checks if any test speakers also appear in training set.

    Args:
        train_features: Training set features
        test_features: Test set features
        similarity_threshold: Threshold to consider same speaker

    Returns:
        Audit results with leakage statistics
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for leakage audit")

    # Compute similarity between train and test
    similarity = cosine_similarity(test_features, train_features)

    # Find maximum similarity for each test sample
    max_sim_per_test = similarity.max(axis=1)

    # Count potential leakage
    leakage_mask = max_sim_per_test >= similarity_threshold
    n_leaked = np.sum(leakage_mask)

    # Get indices of leaked samples
    leaked_test_indices = np.where(leakage_mask)[0]
    matched_train_indices = similarity[leaked_test_indices].argmax(axis=1)

    return {
        "has_leakage": n_leaked > 0,
        "num_leaked_samples": int(n_leaked),
        "total_test_samples": len(test_features),
        "leakage_percentage": (
            float(n_leaked / len(test_features) * 100)
            if len(test_features) > 0
            else 0.0
        ),
        "leaked_test_indices": leaked_test_indices.tolist(),
        "matched_train_indices": matched_train_indices.tolist(),
        "max_similarities": max_sim_per_test.tolist(),
    }


def cluster_by_speaker(
    audio_paths: List[str],
    embedding_model: str = "microsoft/wavlm-base-plus",
    similarity_threshold: float = 0.72,
    method: Optional[str] = None,
    leakage_threshold: float = 0.9,
    leakage_audit_enabled: bool = True,
    train_test_split: Optional[Tuple[List[str], List[str]]] = None,
) -> Dict[str, Any]:
    """Complete speaker clustering pipeline.

    Args:
        audio_paths: List of audio file paths
        embedding_model: WavLM model to use
        similarity_threshold: Clustering threshold
        method: Clustering method ("agglomerative" or "threshold").
            If None, defaults to "threshold" when similarity_threshold is provided.
        leakage_threshold: Similarity threshold for leakage audit.
        leakage_audit_enabled: Whether to audit for train/test leakage
        train_test_split: Optional (train_paths, test_paths) tuple

    Returns:
        Dict with cluster labels and audit results
    """
    # Extract embeddings
    embeddings = extract_wavlm_embeddings(audio_paths, embedding_model)

    # Cluster
    selected_method = method or "threshold"
    labels = cluster_samples(
        embeddings,
        similarity_threshold=similarity_threshold,
        method=selected_method,
    )

    result = {
        "audio_paths": audio_paths,
        "embeddings": embeddings,
        "cluster_labels": labels,
        "n_clusters": len(np.unique(labels)),
    }

    # Leakage audit if requested
    if leakage_audit_enabled and train_test_split is not None:
        train_paths, test_paths = train_test_split

        path_to_idx = {p: i for i, p in enumerate(audio_paths)}

        # Validate that all train/test paths exist in audio_paths
        all_split_paths = set(train_paths) | set(test_paths)
        missing_paths = all_split_paths - set(path_to_idx.keys())
        if missing_paths:
            raise ValueError(
                f"Found {len(missing_paths)} paths in train/test split that are missing "
                f"from audio_paths. First few missing: {list(missing_paths)[:5]}"
            )

        train_indices = [path_to_idx[p] for p in train_paths]
        test_indices = [path_to_idx[p] for p in test_paths]

        train_embeddings = embeddings[train_indices]
        test_embeddings = embeddings[test_indices]

        leakage = audit_leakage(
            train_embeddings,
            test_embeddings,
            similarity_threshold=leakage_threshold,
        )

        result["leakage_audit"] = leakage

    return result


# =============================================================================
# SPEAKER CLUSTERING CLASS (CONFIG-DRIVEN)
# =============================================================================



@dataclass
class SpeakerClusteringConfig:
    """Configuration for speaker clustering."""

    enabled: bool = True
    method: str = "wavlm_ecapa"
    embedding_model: str = "microsoft/wavlm-base-plus"
    similarity_threshold: float = 0.72
    leakage_audit_enabled: bool = True


class SpeakerClustering:
    """Speaker clustering using WavLM embeddings."""

    def __init__(self, config=None):
        self.config = config or SpeakerClusteringConfig()

    def cluster_samples(self, audio_paths):
        """Cluster audio samples by speaker."""
        if not self.config.enabled:
            return {p: 0 for p in audio_paths}

        embeddings = extract_wavlm_embeddings(
            audio_paths, model_name=self.config.embedding_model
        )

        labels = cluster_samples(
            embeddings,
            similarity_threshold=self.config.similarity_threshold,
            method="agglomerative",
        )

        return {p: int(label) for p, label in zip(audio_paths, labels)}

    def audit_leakage(self, train_paths, test_paths):
        """Audit for speaker leakage between train and test."""
        if not self.config.leakage_audit_enabled:
            return {"audited": False}

        all_paths = train_paths + test_paths
        embeddings = extract_wavlm_embeddings(
            all_paths, model_name=self.config.embedding_model
        )

        train_emb = embeddings[: len(train_paths)]
        test_emb = embeddings[len(train_paths) :]

        return audit_leakage(
            train_emb, test_emb, similarity_threshold=self.config.similarity_threshold
        )
