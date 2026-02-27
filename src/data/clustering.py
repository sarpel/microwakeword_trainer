"""Clustering module for speaker diversity and leakage auditing.

Provides:
- Speaker clustering using SpeechBrain ECAPA-TDNN embeddings
- Similarity threshold-based clustering
- Leakage audit for train/test separation

Note on embedding model choice: This module uses SpeechBrain ECAPA-TDNN embeddings
(via `extract_speaker_embeddings`) as the primary speaker embedding method. Although
some guidelines suggest WavLM, ECAPA-TDNN was chosen because it achieves state-of-the-art
speaker verification performance with lower inference overhead and does not require
the Hugging Face `transformers` library. A `extract_wavlm_embeddings` compatibility
wrapper is provided but internally delegates to ECAPA-TDNN.
"""

import logging
import os
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
    import torchaudio

    # SpeechBrain for speaker embeddings (ECAPA-TDNN)
    from speechbrain.pretrained import EncoderClassifier

    HAS_SPEECHBRAIN = True
except ImportError:
    torch = None
    torchaudio = None
    EncoderClassifier = None  # type: ignore
    HAS_SPEECHBRAIN = False

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore
    HAS_LIBROSA = False

logger = logging.getLogger(__name__)


def extract_speaker_embeddings(
    audio_paths: List[str],
    model_name: str = "speechbrain/ecapa-tdnn-voxceleb",
    device: Optional[str] = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN.

    Args:
        audio_paths: List of paths to audio files
        model_name: SpeechBrain model name (default: ecapa-tdnn-voxceleb)
        device: Device to run model on ("cuda" or "cpu"). If None, auto-detect.
        batch_size: Batch size for processing (must be >= 1)

    Returns:
        Array of embeddings [n_samples, embedding_dim]

    Raises:
        ValueError: If batch_size < 1
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not HAS_SPEECHBRAIN or not HAS_SKLEARN:
        raise ImportError(
            "speechbrain and sklearn are required for speaker clustering. "
            "Install: pip install speechbrain torch scikit-learn"
        )
    if not HAS_LIBROSA:
        raise ImportError(
            "librosa is required to load audio for speaker clustering. "
            "Install: pip install librosa"
        )

    import tempfile

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sanitize model name for use in path
    sanitized_name = model_name.replace("/", "_")
    savedir = os.path.join(
        tempfile.gettempdir(), f"speechbrain_embeddings_{sanitized_name}"
    )

    # Load SpeechBrain encoder classifier for speaker embeddings
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        savedir=savedir,
        run_opts={"device": device},
    )

    embeddings = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        batch_embeddings = []

        for audio_path in batch_paths:
            # Load audio using librosa (resample to 16kHz if needed)
            audio, _ = librosa.load(audio_path, sr=16000)

            # SpeechBrain expects (batch, waveform) or (waveform,)
            # The encoder expects torch tensor with shape [1, samples]
            audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

            # Extract embeddings using the encoder method
            # encoder_classifier encodes audio and returns embeddings
            with torch.no_grad():
                embedding = classifier.encode_batch(audio_tensor)
                # Shape: [1, 1, embedding_dim] -> squeeze to [embedding_dim]
                embedding = embedding.squeeze().cpu().numpy()
                batch_embeddings.append(embedding)

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def extract_wavlm_embeddings(
    audio_paths: List[str],
    model_name: str = "microsoft/wavlm-base-plus",
    device: Optional[str] = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract speaker embeddings using WavLM (legacy wrapper).

    This function is kept for backward compatibility.
    Use extract_speaker_embeddings() for SpeechBrain ECAPA-TDNN.

    Args:
        audio_paths: List of paths to audio files
        model_name: WavLM model name (ignored, kept for compatibility)
        device: Device to run model on ("cuda" or "cpu"). If None, auto-detect.
        batch_size: Batch size for processing

    Returns:
        Array of embeddings [n_samples, embedding_dim]
    """
    # Redirect to the new SpeechBrain implementation
    return extract_speaker_embeddings(
        audio_paths,
        model_name="speechbrain/ecapa-tdnn-voxceleb",
        device=device,
        batch_size=batch_size,
    )


def cluster_samples(
    features: np.ndarray,
    n_clusters: Optional[int] = None,
    similarity_threshold: float = 0.72,
) -> np.ndarray:
    """Cluster feature samples using hierarchical agglomerative clustering.

    Two modes of operation:
    1. Explicit n_clusters: Uses Ward linkage on L2-normalized features for
       balanced cluster sizes. Best when you know the approximate speaker count.
    2. Similarity threshold only: Uses complete linkage with distance threshold,
       guaranteeing ALL pairs in a cluster meet the similarity threshold.
       No transitive similarity chains (A~B, B~C forcing A~C).

    Args:
        features: Feature array [n_samples, feature_dim]
        n_clusters: Target number of clusters (overrides threshold if set)
        similarity_threshold: Cosine similarity threshold (0-1). Higher = stricter.

    Returns:
        Cluster labels [n_samples]
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for clustering")

    # Handle empty input
    if features.size == 0 or len(features) == 0:
        return np.array([], dtype=int)

    # Single sample: trivial clustering
    if len(features) == 1:
        return np.array([0])

    if n_clusters is not None:
        # Mode 1: Explicit cluster count - use Ward linkage for balanced clusters
        effective_n_clusters = min(n_clusters, len(features))
        if effective_n_clusters < 1:
            effective_n_clusters = 1

        # Ward linkage requires Euclidean metric, so we L2-normalize features first
        # This makes Euclidean distance on normalized features equivalent to
        # cosine distance on original features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_features = features / norms

        clustering = AgglomerativeClustering(
            n_clusters=effective_n_clusters,
            metric="euclidean",
            linkage="ward",
        )
        labels = clustering.fit_predict(normalized_features)

    else:
        # Mode 2: Similarity threshold - use complete linkage with distance threshold
        # Complete linkage guarantees: for ALL pairs (i,j) in cluster, distance <= threshold
        # This prevents the transitive similarity bug (A~B, B~C forcing A~C)

        # Convert similarity threshold to distance threshold
        # cosine_distance = 1 - cosine_similarity
        distance_threshold = 1.0 - similarity_threshold

        # L2-normalize for cosine distance computation via Euclidean
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_features = features / norms

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="euclidean",
            linkage="complete",
        )
        labels = clustering.fit_predict(normalized_features)

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

    # Operate on positions within the indices array to avoid global vs local confusion
    # selected_positions / remaining_positions are positions in range(len(indices))
    selected_positions = [0]
    remaining_positions = list(range(1, len(indices)))

    while len(selected_positions) < n_select and remaining_positions:
        # Compute selected feature vectors using positions into features array
        selected_features = features[selected_positions]

        max_min_dist = -1.0
        best_pos = remaining_positions[0]

        for pos in remaining_positions:
            feat = features[pos]
            min_dist = float(np.min(np.linalg.norm(selected_features - feat, axis=1)))

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_pos = pos

        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)

    # Return the original global indices corresponding to the selected positions
    return indices[selected_positions]


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

    # Handle empty inputs
    if test_features.size == 0:
        return {
            "has_leakage": False,
            "num_leaked_samples": 0,
            "total_test_samples": 0,
            "leakage_percentage": 0.0,
            "leaked_test_indices": [],
            "matched_train_indices": [],
            "max_similarities": [],
        }

    if train_features.size == 0:
        return {
            "has_leakage": False,
            "num_leaked_samples": 0,
            "total_test_samples": len(test_features),
            "leakage_percentage": 0.0,
            "leaked_test_indices": [],
            "matched_train_indices": [],
            "max_similarities": [0.0] * len(test_features),
        }

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
    embedding_model: str = "speechbrain/ecapa-tdnn-voxceleb",
    n_clusters: Optional[int] = None,
    similarity_threshold: float = 0.72,
    leakage_threshold: float = 0.9,
    leakage_audit_enabled: bool = True,
    train_test_split: Optional[Tuple[List[str], List[str]]] = None,
) -> Dict[str, Any]:
    """Complete speaker clustering pipeline.

    Args:
        audio_paths: List of audio file paths
        embedding_model: SpeechBrain model to use (default: ecapa-tdnn-voxceleb)
        n_clusters: Explicit number of clusters (overrides threshold if set)
        similarity_threshold: Cosine similarity threshold for clustering
        leakage_threshold: Similarity threshold for leakage audit.
        leakage_audit_enabled: Whether to audit for train/test leakage
        train_test_split: Optional (train_paths, test_paths) tuple

    Returns:
        Dict with cluster labels and audit results
    """
    # Extract embeddings using SpeechBrain ECAPA-TDNN
    embeddings = extract_speaker_embeddings(audio_paths, model_name=embedding_model)

    # Cluster
    labels = cluster_samples(
        embeddings,
        n_clusters=n_clusters,
        similarity_threshold=similarity_threshold,
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
    """Configuration for speaker clustering using SpeechBrain ECAPA-TDNN."""

    enabled: bool = True
    method: str = "agglomerative"
    embedding_model: str = "speechbrain/ecapa-tdnn-voxceleb"
    similarity_threshold: float = 0.72
    n_clusters: Optional[int] = None  # Explicit cluster count (overrides threshold)
    leakage_audit_enabled: bool = True
    leakage_similarity_threshold: float = (
        0.9  # Stricter threshold for leakage detection
    )

class SpeakerClustering:
    """Speaker clustering using SpeechBrain ECAPA-TDNN embeddings."""

    def __init__(self, config=None):
        self.config = config or SpeakerClusteringConfig()

    def cluster_samples(self, audio_paths):
        """Cluster audio samples by speaker."""
        if not self.config.enabled:
            return {p: 0 for p in audio_paths}

        embeddings = extract_speaker_embeddings(
            audio_paths, model_name=self.config.embedding_model
        )

        labels = cluster_samples(
            embeddings,
            n_clusters=self.config.n_clusters,
            similarity_threshold=self.config.similarity_threshold,
        )

        return {p: int(label) for p, label in zip(audio_paths, labels, strict=True)}

    def audit_leakage(self, train_paths, test_paths):
        """Audit for speaker leakage between train and test."""
        if not self.config.leakage_audit_enabled:
            return {"audited": False}

        all_paths = train_paths + test_paths
        embeddings = extract_speaker_embeddings(
            all_paths, model_name=self.config.embedding_model
        )

        train_emb = embeddings[: len(train_paths)]
        test_emb = embeddings[len(train_paths) :]

        # Use stricter leakage threshold, fallback to 0.9 if not set
        leakage_threshold = getattr(self.config, "leakage_similarity_threshold", 0.9)

        return audit_leakage(
            train_emb, test_emb, similarity_threshold=leakage_threshold
        )
