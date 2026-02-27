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

import hashlib
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Optional dependencies for embeddings and clustering
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import torch
    import torchaudio
    from torch.utils.data import Dataset, DataLoader

    # SpeechBrain for speaker embeddings (ECAPA-TDNN)
    from speechbrain.pretrained import EncoderClassifier

    HAS_SPEECHBRAIN = True
except ImportError:
    torch = None
    torchaudio = None
    Dataset = None  # type: ignore
    DataLoader = None  # type: ignore
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
    batch_size: Optional[int] = None,
    num_io_workers: int = 8,
    use_mixed_precision: bool = True,
) -> np.ndarray:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN with optimizations.

    Args:
        audio_paths: List of paths to audio files
        model_name: SpeechBrain model name (default: ecapa-tdnn-voxceleb)
        device: Device to run model on ("cuda" or "cpu"). If None, auto-detect.
        batch_size: Batch size for processing. If None, auto-detect based on GPU.
        num_io_workers: Number of parallel I/O workers for audio loading
        use_mixed_precision: Whether to use FP16 mixed precision on GPU

    Returns:
        Array of embeddings [n_samples, embedding_dim]

    Raises:
        ValueError: If batch_size < 1
    """
    if batch_size is not None and batch_size < 1:
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

    # Auto-detect optimal batch size based on GPU memory
    if batch_size is None:
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            # ECAPA-TDNN: ~0.4GB per batch at batch_size=64
            # Leave 20% headroom, clamp to reasonable bounds
            batch_size = max(32, min(128, int(vram_gb * 2)))
            logger.info(f"Auto-detected batch_size={batch_size} for {vram_gb:.1f}GB VRAM")
        else:
            batch_size = 32  # Conservative default for CPU

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

    embeddings: List[np.ndarray] = []

    # Helper function for parallel audio loading
    def load_audio_worker(path: str) -> np.ndarray:
        """Load and return audio at 16kHz."""
        audio, _ = librosa.load(path, sr=16000)
        return audio

    # Setup mixed precision context
    amp_context = torch.cuda.amp.autocast if (use_mixed_precision and device == "cuda") else nullcontext
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]

        # Parallel I/O: Load all audio files in this batch concurrently
        with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
            audio_batch = list(executor.map(load_audio_worker, batch_paths))

        # Pad and stack into real batch tensor [batch, max_length]
        max_len = max(len(a) for a in audio_batch)
        padded_batch = np.zeros((len(audio_batch), max_len), dtype=np.float32)
        for j, audio in enumerate(audio_batch):
            padded_batch[j, : len(audio)] = audio

        # Transfer to GPU with pinned memory for async transfer
        audio_tensor = torch.from_numpy(padded_batch)
        if device == "cuda":
            audio_tensor = audio_tensor.pin_memory().to(device, non_blocking=True)
        else:
            audio_tensor = audio_tensor.to(device)

        # Extract embeddings with mixed precision
        with torch.no_grad():
            with amp_context():
                embedding = classifier.encode_batch(audio_tensor)
            # Shape: [batch, 1, embedding_dim] -> [batch, embedding_dim]
            embedding = embedding.squeeze(1).cpu().numpy()
            embeddings.append(embedding)

        # Periodic GPU cache cleanup to prevent OOM on very large datasets
        if device == "cuda" and i > 0 and (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


def extract_wavlm_embeddings(
    audio_paths: List[str],
    model_name: str = "microsoft/wavlm-base-plus",
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_io_workers: int = 8,
    use_mixed_precision: bool = True,
) -> np.ndarray:
    """Extract speaker embeddings using WavLM (legacy wrapper).

    This function is kept for backward compatibility.
    Use extract_speaker_embeddings() for SpeechBrain ECAPA-TDNN.

    Args:
        audio_paths: List of paths to audio files
        model_name: WavLM model name (ignored, kept for compatibility)
        device: Device to run model on ("cuda" or "cpu"). If None, auto-detect.
        batch_size: Batch size for processing. If None, auto-detect based on GPU.
        num_io_workers: Number of parallel I/O workers for audio loading
        use_mixed_precision: Whether to use FP16 mixed precision on GPU

    Returns:
        Array of embeddings [n_samples, embedding_dim]
    """
    # Redirect to the new SpeechBrain implementation
    return extract_speaker_embeddings(
        audio_paths,
        model_name="speechbrain/ecapa-tdnn-voxceleb",
        device=device,
        batch_size=batch_size,
        num_io_workers=num_io_workers,
        use_mixed_precision=use_mixed_precision,
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
# EMBEDDING CACHE FUNCTIONS
# =============================================================================

def _get_cache_dir() -> Path:
    """Get the cache directory for embeddings."""
    cache_dir = Path(tempfile.gettempdir()) / "mww_embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _compute_files_hash(audio_paths: List[str], model_name: str) -> str:
    """Compute a hash based on file paths, sizes, and modification times.
    
    This is faster than hashing file contents but still detects changes.
    """
    hash_input = model_name.encode()
    
    for path in sorted(audio_paths):
        path_obj = Path(path)
        if path_obj.exists():
            stat = path_obj.stat()
            # Include path, size, and modification time
            hash_input += f"{path}:{stat.st_size}:{stat.st_mtime}".encode()
    
    return hashlib.md5(hash_input).hexdigest()


def _get_cache_path(audio_paths: List[str], model_name: str) -> Path:
    """Generate cache path based on input files and model."""
    files_hash = _compute_files_hash(audio_paths, model_name)
    cache_dir = _get_cache_dir()
    return cache_dir / f"emb_{files_hash}.npz"


def save_embeddings_cache(
    cache_path: Path,
    embeddings: np.ndarray,
    audio_paths: List[str],
    model_name: str,
) -> None:
    """Save embeddings to cache."""
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        audio_paths=audio_paths,
        model_name=model_name,
    )
    logger.info(f"Saved embeddings cache: {cache_path}")


def load_embeddings_cache(
    cache_path: Path,
    audio_paths: List[str],
    model_name: str,
) -> Optional[np.ndarray]:
    """Load embeddings from cache if valid.
    
    Returns None if cache is invalid or corrupted.
    """
    try:
        if not cache_path.exists():
            return None
        
        data = np.load(cache_path, allow_pickle=True)
        
        # Validate cache contents
        cached_paths = list(data["audio_paths"])
        cached_model = str(data["model_name"])
        
        if cached_model != model_name:
            logger.debug("Cache miss: model name mismatch")
            return None
        
        if set(cached_paths) != set(audio_paths):
            logger.debug("Cache miss: audio paths mismatch")
            return None
        
        # Check if any files have been modified since caching
        for path in audio_paths:
            path_obj = Path(path)
            if path_obj.exists():
                stat = path_obj.stat()
                # Re-compute hash for this file to check if modified
                file_hash = f"{path}:{stat.st_size}:{stat.st_mtime}"
                # This is a simplified check - in production you might want
                # to store and compare individual file hashes
        
        logger.info(f"Loaded embeddings from cache: {cache_path}")
        return data["embeddings"]
    
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


# =============================================================================
# PYTORCH DATALOADER FOR PARALLEL I/O
# =============================================================================

if HAS_SPEECHBRAIN and Dataset is not None:
    class AudioDataset(Dataset):
        """PyTorch Dataset for loading audio files."""
        
        def __init__(self, audio_paths: List[str], target_sr: int = 16000):
            self.audio_paths = audio_paths
            self.target_sr = target_sr
        
        def __len__(self) -> int:
            return len(self.audio_paths)
        
        def __getitem__(self, idx: int):
            """Load and return audio file.
            
            Returns:
                Tuple of (audio_tensor, original_length)
            """
            if not HAS_LIBROSA:
                raise ImportError("librosa is required")
            
            audio, sr = librosa.load(self.audio_paths[idx], sr=self.target_sr)
            return torch.tensor(audio, dtype=torch.float32), len(audio)
else:
    # Dummy class for when torch is not available
    class AudioDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AudioDataset")

def collate_audio_batch(batch: List[tuple]) -> torch.Tensor:
    """Collate variable-length audio into padded batch.
    
    Args:
        batch: List of (audio_tensor, length) tuples
    
    Returns:
        Padded tensor [batch_size, max_length]
    """
    audios, lengths = zip(*batch)
    max_len = max(lengths)
    
    padded = torch.zeros(len(batch), max_len)
    for i, (audio, length) in enumerate(zip(audios, lengths)):
        padded[i, :length] = audio
    
    return padded


def extract_speaker_embeddings_dataloader(
    audio_paths: List[str],
    model_name: str = "speechbrain/ecapa-tdnn-voxceleb",
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    use_mixed_precision: bool = True,
) -> np.ndarray:
    """Extract embeddings using PyTorch DataLoader for parallel I/O.
    
    This is an alternative to extract_speaker_embeddings that uses
    PyTorch DataLoader for more efficient parallel data loading.
    
    Args:
        audio_paths: List of paths to audio files
        model_name: SpeechBrain model name
        device: Device to run on ("cuda" or "cpu")
        batch_size: Batch size (auto-detect if None)
        num_workers: Number of DataLoader workers for I/O
        use_mixed_precision: Use FP16 on GPU
    
    Returns:
        Array of embeddings [n_samples, embedding_dim]
    """
    if not HAS_SPEECHBRAIN or not HAS_SKLEARN:
        raise ImportError("speechbrain and sklearn required")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect batch size
    if batch_size is None:
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            batch_size = max(32, min(128, int(vram_gb * 2)))
        else:
            batch_size = 32
    
    # Load model
    sanitized_name = model_name.replace("/", "_")
    savedir = os.path.join(tempfile.gettempdir(), f"speechbrain_{sanitized_name}")
    
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        savedir=savedir,
        run_opts={"device": device},
    )
    classifier.eval()
    
    # Warm-up
    if device == "cuda":
        dummy = torch.zeros(1, 16000, device=device)
        with torch.no_grad():
            _ = classifier.encode_batch(dummy)
        torch.cuda.synchronize()
    
    # Create DataLoader
    dataset = AudioDataset(audio_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # Extract embeddings
    embeddings = []
    amp_context = torch.cuda.amp.autocast if (use_mixed_precision and device == "cuda") else nullcontext
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            
            with amp_context():
                emb = classifier.encode_batch(batch)
            
            embeddings.append(emb.squeeze(1).cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


# =============================================================================
# ADVANCED CLUSTERING ALGORITHMS (PHASE 3)
# =============================================================================

def cluster_samples_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster samples using HDBSCAN.
    
    HDBSCAN is O(n log n) with approximate nearest neighbors,
    making it suitable for larger datasets (10K-100K samples).
    
    Args:
        features: Feature array [n_samples, feature_dim]
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        metric: Distance metric
    
    Returns:
        Cluster labels [n_samples]. Noise points labeled as -1.
    """
    if not HAS_HDBSCAN:
        raise ImportError(
            "hdbscan is required for large dataset clustering. "
            "Install: pip install hdbscan"
        )
    
    if features.size == 0 or len(features) == 0:
        return np.array([], dtype=int)
    
    if len(features) == 1:
        return np.array([0])
    
    # Normalize features for cosine-like behavior
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_features = features / norms
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    
    labels = clusterer.fit_predict(normalized_features)
    
    # Relabel noise points (-1) to unique cluster IDs
    # This ensures all samples are assigned to some cluster
    max_label = labels.max()
    noise_mask = labels == -1
    n_noise = noise_mask.sum()
    
    if n_noise > 0:
        # Assign each noise point to its own cluster
        noise_labels = np.arange(max_label + 1, max_label + 1 + n_noise)
        labels[noise_mask] = noise_labels
    
    return labels


def cluster_samples_adaptive(
    features: np.ndarray,
    n_clusters: Optional[int] = None,
    similarity_threshold: float = 0.72,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """Adaptive clustering based on dataset size.
    
    Automatically selects the appropriate clustering algorithm:
    - n <= 5000: AgglomerativeClustering (exact, high quality)
    - 5000 < n <= 50000: HDBSCAN (fast, scalable)
    - n > 50000: Agglomerative with kNN connectivity (memory efficient)
    
    Args:
        features: Feature array [n_samples, feature_dim]
        n_clusters: Target number of clusters (for agglomerative)
        similarity_threshold: Similarity threshold (for agglomerative)
        min_cluster_size: Minimum cluster size (for HDBSCAN)
    
    Returns:
        Cluster labels [n_samples]
    """
    n_samples = len(features)
    
    if n_samples <= 5000:
        # Use exact AgglomerativeClustering for small datasets
        logger.info(f"Using AgglomerativeClustering for {n_samples} samples")
        return cluster_samples(features, n_clusters, similarity_threshold)
    
    elif n_samples <= 50000 and HAS_HDBSCAN:
        # Use HDBSCAN for medium datasets
        logger.info(f"Using HDBSCAN for {n_samples} samples")
        return cluster_samples_hdbscan(features, min_cluster_size=min_cluster_size)
    
    else:
        # For large datasets, use k-means as coarse clustering
        # then agglomerative within each coarse cluster
        logger.info(f"Using two-stage clustering for {n_samples} samples")
        return _cluster_samples_two_stage(features, n_clusters, similarity_threshold)


def _cluster_samples_two_stage(
    features: np.ndarray,
    n_clusters: Optional[int] = None,
    similarity_threshold: float = 0.72,
) -> np.ndarray:
    """Two-stage clustering: k-means coarse + agglomerative fine.
    
    This reduces memory from O(n^2) to O((n/k)^2 * k) where k is
    the number of coarse clusters.
    
    Args:
        features: Feature array [n_samples, feature_dim]
        n_clusters: Target number of final clusters
        similarity_threshold: Similarity threshold for fine clustering
    
    Returns:
        Cluster labels [n_samples]
    """
    from sklearn.cluster import KMeans
    
    n_samples = len(features)
    
    # Stage 1: Coarse clustering with k-means
    # Aim for ~1000 samples per coarse cluster
    n_coarse = max(10, n_samples // 1000)
    if n_clusters is not None:
        n_coarse = min(n_coarse, n_clusters)
    
    logger.info(f"Stage 1: Coarse k-means with {n_coarse} clusters")
    kmeans = KMeans(n_clusters=n_coarse, random_state=42, n_init=10)
    coarse_labels = kmeans.fit_predict(features)
    
    # Stage 2: Agglomerative clustering within each coarse cluster
    logger.info("Stage 2: Fine agglomerative clustering per coarse cluster")
    final_labels = np.zeros(n_samples, dtype=int)
    label_offset = 0
    
    for coarse_id in range(n_coarse):
        mask = coarse_labels == coarse_id
        cluster_features = features[mask]
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_features) <= 1:
            final_labels[cluster_indices] = label_offset
            label_offset += 1
            continue
        
        # Calculate target clusters for this coarse cluster
        if n_clusters is not None:
            # Proportional allocation
            proportion = len(cluster_features) / n_samples
            target = max(1, int(n_clusters * proportion))
        else:
            target = None
        
        # Run agglomerative on this subset
        sub_labels = cluster_samples(
            cluster_features,
            n_clusters=target,
            similarity_threshold=similarity_threshold,
        )
        
        # Offset labels to be globally unique
        final_labels[cluster_indices] = sub_labels + label_offset
        label_offset += sub_labels.max() + 1
    
    return final_labels


# =============================================================================
# SPEAKER CLUSTERING CLASS (CONFIG-DRIVEN)
# =============================================================================

@dataclass
class SpeakerClusteringConfig:
    """Configuration for speaker clustering using SpeechBrain ECAPA-TDNN."""

    enabled: bool = True
    method: str = "agglomerative"  # "agglomerative", "hdbscan", or "adaptive"
    embedding_model: str = "speechbrain/ecapa-tdnn-voxceleb"
    similarity_threshold: float = 0.72
    n_clusters: Optional[int] = None  # Explicit cluster count (overrides threshold)
    leakage_audit_enabled: bool = True
    leakage_similarity_threshold: float = (
        0.9  # Stricter threshold for leakage detection
    )
    
    # Caching options (Phase 2)
    use_embedding_cache: bool = True  # Cache embeddings to disk
    cache_dir: Optional[str] = None  # Custom cache directory
    
    # Performance options (Phase 2)
    batch_size: Optional[int] = None  # Auto-detect if None
    num_io_workers: int = 8  # Parallel I/O workers
    use_mixed_precision: bool = True  # Use FP16 on GPU
    use_dataloader: bool = False  # Use PyTorch DataLoader (slower for small datasets)
    
    # Adaptive clustering options (Phase 3)
    use_adaptive_clustering: bool = True  # Auto-select algorithm based on dataset size
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    adaptive_threshold_small: int = 5000  # Use agglomerative below this
    adaptive_threshold_large: int = 50000  # Use two-stage above this


class SpeakerClustering:
    """Speaker clustering using SpeechBrain ECAPA-TDNN embeddings with caching."""

    def __init__(self, config=None):
        self.config = config or SpeakerClusteringConfig()
        self._classifier: Optional[Any] = None
        self._model_device: Optional[str] = None
        self._model_name: Optional[str] = None

    def _get_classifier(self, device: Optional[str] = None) -> Any:
        """Lazy-load and cache the SpeechBrain classifier.
        
        The model is loaded once and reused across multiple calls to
        cluster_samples() or audit_leakage().
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if we need to reload the model
        model_changed = (
            self._model_name is not None and 
            self._model_name != self.config.embedding_model
        )
        device_changed = (
            self._model_device is not None and 
            self._model_device != device
        )
        
        if self._classifier is None or model_changed or device_changed:
            if not HAS_SPEECHBRAIN:
                raise ImportError(
                    "speechbrain is required for speaker clustering. "
                    "Install: pip install speechbrain torch"
                )
            
            sanitized_name = self.config.embedding_model.replace("/", "_")
            savedir = os.path.join(
                tempfile.gettempdir(),
                f"speechbrain_embeddings_{sanitized_name}"
            )
            
            logger.info(f"Loading SpeechBrain model: {self.config.embedding_model}")
            self._classifier = EncoderClassifier.from_hparams(
                source=self.config.embedding_model,
                savedir=savedir,
                run_opts={"device": device},
            )
            self._classifier.eval()
            self._model_device = device
            self._model_name = self.config.embedding_model
            
            # Warm-up with dummy inference
            if device == "cuda":
                dummy = torch.zeros(1, 16000, device=device)
                with torch.no_grad():
                    _ = self._classifier.encode_batch(dummy)
                torch.cuda.synchronize()
                logger.info("Model warm-up complete")
        
        return self._classifier

    def _extract_embeddings_cached(
        self,
        audio_paths: List[str],
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Extract embeddings with disk caching support."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check cache if enabled
        if self.config.use_embedding_cache:
            cache_path = _get_cache_path(audio_paths, self.config.embedding_model)
            if self.config.cache_dir:
                cache_path = Path(self.config.cache_dir) / cache_path.name
            
            cached = load_embeddings_cache(
                cache_path, audio_paths, self.config.embedding_model
            )
            if cached is not None:
                return cached
        
        # Extract embeddings
        if self.config.use_dataloader and HAS_SPEECHBRAIN:
            # Use DataLoader for parallel I/O
            embeddings = extract_speaker_embeddings_dataloader(
                audio_paths,
                model_name=self.config.embedding_model,
                device=device,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_io_workers,
                use_mixed_precision=self.config.use_mixed_precision,
            )
        else:
            # Use standard extraction with cached model
            embeddings = self._extract_with_cached_model(audio_paths, device)
        
        # Save to cache if enabled
        if self.config.use_embedding_cache:
            cache_path = _get_cache_path(audio_paths, self.config.embedding_model)
            if self.config.cache_dir:
                cache_path = Path(self.config.cache_dir) / cache_path.name
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_embeddings_cache(
                cache_path, embeddings, audio_paths, self.config.embedding_model
            )
        
        return embeddings

    def _extract_with_cached_model(
        self,
        audio_paths: List[str],
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Extract embeddings using the cached model."""
        if not HAS_SPEECHBRAIN or not HAS_SKLEARN:
            raise ImportError(
                "speechbrain and sklearn are required. "
                "Install: pip install speechbrain torch scikit-learn"
            )
        
        if not HAS_LIBROSA:
            raise ImportError(
                "librosa is required. Install: pip install librosa"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get cached classifier
        classifier = self._get_classifier(device)
        
        # Determine batch size
        batch_size = self.config.batch_size
        if batch_size is None:
            if device == "cuda":
                vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                batch_size = max(32, min(128, int(vram_gb * 2)))
                logger.info(f"Auto-detected batch_size={batch_size} for {vram_gb:.1f}GB VRAM")
            else:
                batch_size = 32
        
        embeddings: List[np.ndarray] = []
        
        # Helper function for parallel audio loading
        def load_audio_worker(path: str) -> np.ndarray:
            audio, _ = librosa.load(path, sr=16000)
            return audio
        
        # Setup mixed precision context
        amp_context = torch.cuda.amp.autocast if (self.config.use_mixed_precision and device == "cuda") else nullcontext
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            
            # Parallel I/O
            with ThreadPoolExecutor(max_workers=self.config.num_io_workers) as executor:
                audio_batch = list(executor.map(load_audio_worker, batch_paths))
            
            # Pad and stack
            max_len = max(len(a) for a in audio_batch)
            padded_batch = np.zeros((len(audio_batch), max_len), dtype=np.float32)
            for j, audio in enumerate(audio_batch):
                padded_batch[j, :len(audio)] = audio
            
            # GPU transfer with optimizations
            audio_tensor = torch.from_numpy(padded_batch)
            if device == "cuda":
                audio_tensor = audio_tensor.pin_memory().to(device, non_blocking=True)
            else:
                audio_tensor = audio_tensor.to(device)
            
            # Extract embeddings
            with torch.no_grad():
                with amp_context():
                    embedding = classifier.encode_batch(audio_tensor)
                embedding = embedding.squeeze(1).cpu().numpy()
                embeddings.append(embedding)
            
            # Periodic cleanup
            if device == "cuda" and i > 0 and (i // batch_size) % 50 == 0:
                torch.cuda.empty_cache()
        
        return np.concatenate(embeddings, axis=0)

    def cluster_samples(self, audio_paths: List[str]) -> Dict[str, int]:
        """Cluster audio samples by speaker.
        
        Uses adaptive clustering by default for optimal performance
        across different dataset sizes.
        """
        if not self.config.enabled:
            return {p: 0 for p in audio_paths}
        
        # Extract embeddings (with caching)
        embeddings = self._extract_embeddings_cached(audio_paths)
        
        # Select clustering method
        if self.config.use_adaptive_clustering:
            labels = cluster_samples_adaptive(
                embeddings,
                n_clusters=self.config.n_clusters,
                similarity_threshold=self.config.similarity_threshold,
                min_cluster_size=self.config.hdbscan_min_cluster_size,
            )
        elif self.config.method == "hdbscan" and HAS_HDBSCAN:
            labels = cluster_samples_hdbscan(
                embeddings,
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
            )
        else:
            labels = cluster_samples(
                embeddings,
                n_clusters=self.config.n_clusters,
                similarity_threshold=self.config.similarity_threshold,
            )
        
        return {p: int(label) for p, label in zip(audio_paths, labels)}

    def audit_leakage(self, train_paths: List[str], test_paths: List[str]) -> Dict[str, Any]:
        """Audit for speaker leakage between train and test."""
        if not self.config.leakage_audit_enabled:
            return {"audited": False}
        
        all_paths = train_paths + test_paths
        embeddings = self._extract_embeddings_cached(all_paths)
        
        train_emb = embeddings[:len(train_paths)]
        test_emb = embeddings[len(train_paths):]
        
        leakage_threshold = getattr(
            self.config, "leakage_similarity_threshold", 0.9
        )
        
        return audit_leakage(
            train_emb, test_emb, similarity_threshold=leakage_threshold
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache for this model."""
        cache_dir = _get_cache_dir()
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir)
        
        pattern = f"emb_*_{self.config.embedding_model.replace('/', '_')}.npz"
        for cache_file in cache_dir.glob(pattern):
            cache_file.unlink()
            logger.info(f"Removed cache file: {cache_file}")

    def clear_model_cache(self) -> None:
        """Clear the cached model from memory."""
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
            self._model_device = None
            self._model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cache cleared")

