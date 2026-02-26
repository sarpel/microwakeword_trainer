"""Dataset module for managing training data with efficient mmap storage.

Provides:
- RaggedMmap: Variable-length array storage using memory-mapped files
- WakeWordDataset: PyTorch-compatible dataset for training
- FeatureStore: Manage features on disk with mmap access
"""

import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# RAGGED MMAP STORAGE
# =============================================================================


@dataclass
class RaggedMmapConfig:
    """Configuration for RaggedMmap storage."""

    base_dir: Path
    name: str
    dtype: "np.dtype[Any]" = np.dtype(np.float32)
    create: bool = True


class RaggedMmap:
    """Memory-mapped storage for variable-length arrays.

    This class provides efficient storage and retrieval of variable-length
    arrays (ragged tensors) using memory-mapped files. Each array can have
    a different length, and they are stored contiguously with an index.

    Storage format:
        - data.bin: Raw array data (flattened)
        - offsets.bin: Starting offset for each array (int64)
        - lengths.bin: Length of each array (int64)

    Attributes:
        base_dir: Base directory for storage files
        name: Name of the storage
        dtype: Data type for arrays
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        name: str = "ragged",
        dtype: "np.dtype[Any]" = np.dtype(np.float32),
        create: bool = True,
    ):
        """Initialize RaggedMmap storage.

        Args:
            base_dir: Base directory for storage files
            name: Name for this storage
            dtype: NumPy data type for arrays
            create: If True, create directory if it doesn't exist
        """
        self.base_dir = Path(base_dir)
        self.name = name
        self.dtype = np.dtype(dtype)
        self._data_file: Optional[str] = None
        self._offsets_file: Optional[str] = None
        self._lengths_file: Optional[str] = None
        self._data: Optional[np.memmap] = None
        self._offsets: Union[List[int], np.ndarray, None] = None
        self._lengths: Union[List[int], np.ndarray, None] = None
        self._num_arrays: int = 0
        self._total_bytes: int = 0

        if create:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        self._init_files()

    def _init_files(self):
        """Initialize storage files."""
        self._data_file = str(self.base_dir / f"{self.name}.data")
        self._offsets_file = str(self.base_dir / f"{self.name}.offsets")
        self._lengths_file = str(self.base_dir / f"{self.name}.lengths")

    @property
    def num_arrays(self) -> int:
        """Get number of stored arrays."""
        return self._num_arrays

    @property
    def total_elements(self) -> int:
        """Get total number of elements across all arrays."""
        return self._total_bytes

    def _load_index(self):
        """Load index files (offsets and lengths)."""
        if os.path.exists(self._offsets_file) and os.path.exists(self._lengths_file):
            # Read binary offset data
            with open(self._offsets_file, "rb") as f:
                offset_data = f.read()
            num_offsets = len(offset_data) // 8  # 8 bytes per int64
            self._offsets = np.frombuffer(
                offset_data, dtype=np.int64, count=num_offsets
            )

            # Read binary lengths data
            with open(self._lengths_file, "rb") as f:
                length_data = f.read()
            self._lengths = np.frombuffer(length_data, dtype=np.int64)

            self._num_arrays = len(self._lengths)
            self._total_bytes = int(self._lengths.sum())

            # Validate offsets and lengths have the same length
            if len(self._offsets) != len(self._lengths):
                raise ValueError(
                    f"Offsets ({len(self._offsets)}) and lengths ({len(self._lengths)}) mismatch"
                )
            self._total_bytes = int(self._lengths.sum())

    def open(self, mode: str = "r"):
        """Open the storage for reading or writing.

        Args:
            mode: 'r' for read, 'w' for write
        """
        if mode == "r":
            self._load_index()
            if os.path.exists(self._data_file):
                self._data = np.memmap(
                    self._data_file,
                    dtype=self.dtype,
                    mode="r",
                )
        elif mode == "w":
            # Truncate existing files before writing to prevent data corruption
            for f in [self._data_file, self._offsets_file, self._lengths_file]:
                if os.path.exists(f):
                    os.truncate(f, 0)

            # Initialize in-memory index as mutable lists for write mode
            self._offsets = []
            self._lengths = []
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def close(self):
        """Close the storage and flush any pending writes."""
        if self._data is not None:
            self._data.flush()
            del self._data
            self._data = None

    def append(self, arrays: List[np.ndarray]):
        """Append arrays to storage.

        Args:
            arrays: List of arrays to append
        """
        # Convert to numpy arrays
        arrays = [np.asarray(arr, dtype=self.dtype) for arr in arrays]

        # Get lengths (in bytes for offset calculation)
        # Get lengths (in bytes for offset calculation)
        lengths = [arr.nbytes for arr in arrays]
        lengths = [arr.nbytes for arr in arrays]

        # Calculate offsets
        if self._num_arrays > 0:
            last_offset = int(self._offsets[-1]) + int(self._lengths[-1])
        else:
            last_offset = 0

        offsets = [last_offset + sum(lengths[:i]) for i in range(len(arrays))]

        # Append to data file
        with open(self._data_file, "ab") as f:
            for arr in arrays:
                f.write(arr.tobytes())

        # Append to index files
        with open(self._offsets_file, "ab") as f:
            for offset in offsets:
                f.write(struct.pack("q", offset))

        with open(self._lengths_file, "ab") as f:
            for length in lengths:
                f.write(struct.pack("q", length))

        # Update in-memory index so subsequent append() calls can compute offsets
        if isinstance(self._offsets, list):
            self._offsets.extend(offsets)
        else:
            self._offsets = (
                list(self._offsets) + list(offsets)
                if self._offsets is not None
                else list(offsets)
            )
        if isinstance(self._lengths, list):
            self._lengths.extend(lengths)
        else:
            self._lengths = (
                list(self._lengths) + list(lengths)
                if self._lengths is not None
                else list(lengths)
            )

        # Update metadata
        self._num_arrays += len(arrays)
        self._total_bytes += sum(lengths)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get array by index.

        Args:
            idx: Array index

        Returns:
            Array data
        """
        if self._data is None or self._offsets is None or self._lengths is None:
            self.open("r")

        offset = int(self._offsets[idx])
        length = int(self._lengths[idx])

        # Convert byte offsets/lengths to element counts
        itemsize = self._data.itemsize
        elem_offset = offset // itemsize
        elem_length = length // itemsize

        return np.array(self._data[elem_offset : elem_offset + elem_length])

    def __len__(self) -> int:
        """Get number of stored arrays."""
        return self._num_arrays

    @staticmethod
    def create_from_arrays(
        arrays: List[np.ndarray],
        base_dir: Union[str, Path],
        name: str = "ragged",
        dtype: "np.dtype[Any]" = np.dtype(np.float32),
    ) -> "RaggedMmap":
        """Create RaggedMmap from list of arrays.

        Args:
            arrays: List of arrays to store
            base_dir: Base directory for storage
            name: Name for storage
            dtype: Data type

        Returns:
            RaggedMmap instance
        """
        storage = RaggedMmap(base_dir, name, dtype, create=True)

        # Write arrays
        storage.open("w")
        storage.append(arrays)
        storage.close()

        return storage


# =============================================================================
# FEATURE STORE
# =============================================================================


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""

    base_dir: Path
    features_name: str = "features"
    labels_name: str = "labels"
    metadata_name: str = "metadata"
    dtype: "np.dtype[Any]" = np.dtype(np.float32)


class FeatureStore:
    """Store and retrieve extracted features.

    Manages storage of feature arrays with associated metadata,
    providing efficient mmap-based access for training.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        config: Optional[FeatureStoreConfig] = None,
    ):
        """Initialize feature store.

        Args:
            base_dir: Base directory for storage
            config: Feature store configuration
        """
        self.base_dir = Path(base_dir)
        self.config = config or FeatureStoreConfig(base_dir=self.base_dir)

        self.features: Optional[RaggedMmap] = None
        self.labels: Optional[RaggedMmap] = None
        self.metadata: Dict = {}

        # Ensure directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        num_samples: int,
        feature_dim: int,
    ):
        """Initialize empty feature store.

        Args:
            num_samples: Expected number of samples
            feature_dim: Dimension of feature vectors
        """
        # Create ragged mmap stores
        self.features = RaggedMmap(
            self.base_dir,
            self.config.features_name,
            self.config.dtype,
            create=True,
        )

        self.labels = RaggedMmap(
            self.base_dir,
            self.config.labels_name,
            np.dtype(np.int32),  # Labels as integers
            create=True,
        )

        self.metadata = {
            "num_samples": 0,
            "expected_samples": num_samples,
            "feature_dim": feature_dim,
        }
    def add(self, feature: np.ndarray, label: int):
        """Add a single feature-label pair.

        Args:
            feature: Feature array
            label: Label (0 or 1)
        """
        if self.features is None:
            raise RuntimeError("Feature store not initialized")

        self.features.append([feature])
        self.labels.append([np.array([label], dtype=np.int32)])
        self.metadata["num_samples"] += 1

    def add_batch(self, features: List[np.ndarray], labels: List[int]):
        """Add a batch of feature-label pairs.

        Args:
            features: List of feature arrays
            labels: List of labels

        Raises:
            ValueError: If features and labels have different lengths or are empty
        """
        if not features:
            raise ValueError("features must be non-empty")
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length, "
                f"got {len(features)} vs {len(labels)}"
            )

        if self.features is None:
            # Initialize with first sample to get feature dim
            self.initialize(len(features), features[0].shape[-1])

        self.features.append(features)
        self.labels.append([np.array([label], dtype=np.int32) for label in labels])
        self.metadata["num_samples"] += len(features)

    def get(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get feature and label by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (feature, label)
        """
        if self.features is None or self.labels is None:
            raise RuntimeError("Feature store not loaded")

        return self.features[idx], int(self.labels[idx][0])

    def __len__(self) -> int:
        """Get number of stored samples."""
        if self.features is None:
            return 0
        return len(self.features)

    def open(self):
        """Open store for reading."""
        self.features = RaggedMmap(
            self.base_dir,
            self.config.features_name,
            self.config.dtype,
            create=False,
        )
        self.features.open("r")

        self.labels = RaggedMmap(
            self.base_dir,
            self.config.labels_name,
            np.int32,
            create=False,
        )
        self.labels.open("r")

    def close(self):
        """Close store and flush writes."""
        if self.features is not None:
            self.features.close()
        if self.labels is not None:
            self.labels.close()


# =============================================================================
# WAKE WORD DATASET (PyTorch-compatible)
# =============================================================================


class WakeWordDataset:
    """PyTorch-compatible dataset for wake word training.

    Provides efficient access to stored features using mmap.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        batch_size: int = 32,
        feature_dim: int = 40,
    ):
        """Initialize dataset.

        Args:
            config: Configuration dictionary with paths and hardware settings.
                   When provided, the build() method can be called to process
                   raw audio files into features.
            data_path: Path to processed data directory (legacy compatibility)
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for training
            feature_dim: Dimension of feature vectors
        """
        # Handle config-based initialization
        if config is not None:
            self._config = config
            paths_cfg = config.get("paths", {})
            hardware_cfg = config.get("hardware", {})
            training_cfg = config.get("training", {})

            self.data_path = Path(paths_cfg.get("processed_dir", "./data/processed"))
            self.batch_size = training_cfg.get("batch_size", batch_size)
            self.feature_dim = hardware_cfg.get("mel_bins", feature_dim)
            self.split = split
            self._is_built = False
        else:
            # Legacy initialization
            self._config = None
            self.data_path = Path(data_path) if data_path else Path("./data/processed")
            self.split = split
            self.batch_size = batch_size
            self.feature_dim = feature_dim
            self._is_built = False

        # Try to load from store
        self.feature_store: Optional[FeatureStore] = None
        self._load_store()
    def _load_store(self):
        """Load feature store if available."""
        store_path = self.data_path / self.split
        if store_path.exists():
            self.feature_store = FeatureStore(store_path)
            try:
                self.feature_store.open()
            except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                logger.warning(f"Could not open feature store at {store_path}: {e}")
                self.feature_store = None

    def __len__(self) -> int:
        """Return dataset length."""
        if self.feature_store is not None:
            return len(self.feature_store)
        return 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get item by index.

        Args:
            idx: Item index

        Returns:
            Tuple of (features, label)
        """
        if self.feature_store is None:
            raise RuntimeError("Feature store not available")

        return self.feature_store.get(idx)

    def build(self, config: Optional[Dict[str, Any]] = None) -> "WakeWordDataset":
        """Build the dataset from raw audio files.

        Args:
            config: Configuration dictionary with paths and hardware settings.
                   If not provided, uses config from __init__.

        Returns:
            self for method chaining
        """
        # Use provided config or fall back to stored config
        cfg = config if config is not None else self._config
        if cfg is None:
            raise ValueError(
                "No config provided. Pass config to __init__ or build(config)"
            )

        # Extract paths from config
        paths_cfg = cfg.get("paths", {})
        positive_dir = paths_cfg.get("positive_dir")
        negative_dir = paths_cfg.get("negative_dir")
        hard_negative_dir = paths_cfg.get("hard_negative_dir")
        processed_dir = paths_cfg.get("processed_dir", "./data/processed")

        # Extract hardware config for feature extraction
        hardware_cfg = cfg.get("hardware", {})
        sample_rate = hardware_cfg.get("sample_rate_hz", 16000)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        window_size_ms = hardware_cfg.get("window_size_ms", 30)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)

        # Create feature config
        from src.data.features import FeatureConfig, MicroFrontend
        from src.data.ingestion import Clips, ClipsLoaderConfig, Split, Label

        feature_config = FeatureConfig(
            sample_rate=sample_rate,
            mel_bins=mel_bins,
            window_size_ms=window_size_ms,
            window_step_ms=window_step_ms,
        )

        # Initialize feature extractor
        frontend = MicroFrontend(feature_config)

        # Ensure processed directories exist
        dirs = ensure_processed_directory(processed_dir)

        # Load clips using ClipsLoaderConfig
        logger.info("Loading audio clips from dataset directories...")

        clips_config = ClipsLoaderConfig(
            positive_dir=Path(positive_dir) if positive_dir else None,
            negative_dir=Path(negative_dir) if negative_dir else None,
            hard_negative_dir=Path(hard_negative_dir) if hard_negative_dir else None,
            train_split=0.8,
            val_split=0.2,
            test_split=0.0,
            seed=42,
        )

        clips = Clips(config=clips_config)

        # Get train and val samples
        train_samples = clips.get_split(Split.TRAIN)
        val_samples = clips.get_split(Split.VAL)

        logger.info(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples")

        if not train_samples:
            raise RuntimeError("No training samples found. Please check your dataset directories.")

        # Extract features and store for training
        logger.info(f"Extracting features for {len(train_samples)} training clips...")
        train_store = FeatureStore(dirs["train"])
        train_store.initialize(len(train_samples), feature_dim=mel_bins)

        for sample in train_samples:
            try:
                # Load audio
                import wave

                with wave.open(str(sample.path), 'rb') as wf:
                    n_frames = wf.getnframes()
                    audio_data = wf.readframes(n_frames)
                    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0

                # Extract features
                features = frontend.compute_mel_spectrogram(audio)

                # Determine label (0 for negative, 1 for positive/hard_negative)
                label = 1 if sample.label == Label.POSITIVE else 0

                # Add to store
                train_store.add(features, label)

            except Exception as e:
                logger.warning(f"Failed to process {sample.path}: {e}")
                continue

        train_store.close()
        logger.info(f"Processed {len(train_store)} training samples")

        # Extract features for validation
        if val_samples:
            logger.info(f"Extracting features for {len(val_samples)} validation clips...")
            val_store = FeatureStore(dirs["val"])
            val_store.initialize(len(val_samples), feature_dim=mel_bins)

            for sample in val_samples:
                try:
                    with wave.open(str(sample.path), 'rb') as wf:
                        n_frames = wf.getnframes()
                        audio_data = wf.readframes(n_frames)
                        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0

                    features = frontend.compute_mel_spectrogram(audio)
                    label = 1 if sample.label == Label.POSITIVE else 0
                    val_store.add(features, label)

                except Exception as e:
                    logger.warning(f"Failed to process {sample.path}: {e}")
                    continue

            val_store.close()
            logger.info(f"Processed {len(val_store)} validation samples")

        # Reload the train store for training
        self._load_store()

        return self
    def _pad_or_truncate(self, features: np.ndarray, max_time_frames: int) -> np.ndarray:
        """Pad or truncate features to fixed time length."""
        # Handle potentially flattened array from RaggedMmap
        if features.ndim == 1:
            total_elements = features.shape[0]
            time_frames = total_elements // self.feature_dim
            if time_frames * self.feature_dim != total_elements:
                raise ValueError(f"Cannot reshape flattened array")
            features = features.reshape(time_frames, self.feature_dim)

        current_length = features.shape[0]
        if current_length > max_time_frames:
            return features[:max_time_frames, :]
        elif current_length < max_time_frames:
            padding = np.zeros((max_time_frames - current_length, self.feature_dim), dtype=features.dtype)
            return np.vstack([features, padding])
        return features

    def train_generator_factory(self, max_time_frames: int = 49):
        """Create a factory for infinite training data generator."""
        def factory():
            num_samples = len(self)
            if num_samples == 0:
                return iter([])
            indices = list(range(num_samples))
            rng = np.random.RandomState(42)
            while True:
                epoch_indices = rng.permutation(indices).tolist()
                batch_features = []
                batch_labels = []
                for idx in epoch_indices:
                    try:
                        feature, label = self[idx]
                    except (RuntimeError, IndexError):
                        continue
                    fixed_feature = self._pad_or_truncate(feature, max_time_frames)
                    batch_features.append(fixed_feature)
                    batch_labels.append(label)
                    if len(batch_features) >= self.batch_size:
                        fingerprints = np.array(batch_features, dtype=np.float32)
                        ground_truth = np.array(batch_labels, dtype=np.int32)
                        sample_weights = np.ones(len(batch_labels), dtype=np.float32)
                        yield (fingerprints, ground_truth, sample_weights)
                        batch_features = []
                        batch_labels = []
                if batch_features:
                    fingerprints = np.array(batch_features, dtype=np.float32)
                    ground_truth = np.array(batch_labels, dtype=np.int32)
                    sample_weights = np.ones(len(batch_labels), dtype=np.float32)
                    yield (fingerprints, ground_truth, sample_weights)
        return factory

    def val_generator_factory(self, max_time_frames: int = 49):
        """Create a factory for finite validation data generator."""
        def factory():
            num_samples = len(self)
            if num_samples == 0:
                return iter([])
            indices = list(range(num_samples))
            batch_features = []
            batch_labels = []
            for idx in indices:
                try:
                    feature, label = self[idx]
                except (RuntimeError, IndexError):
                    continue
                fixed_feature = self._pad_or_truncate(feature, max_time_frames)
                batch_features.append(fixed_feature)
                batch_labels.append(label)
                if len(batch_features) >= self.batch_size:
                    fingerprints = np.array(batch_features, dtype=np.float32)
                    ground_truth = np.array(batch_labels, dtype=np.int32)
                    sample_weights = np.ones(len(batch_labels), dtype=np.float32)
                    yield (fingerprints, ground_truth, sample_weights)
                    batch_features = []
                    batch_labels = []
            if batch_features:
                fingerprints = np.array(batch_features, dtype=np.float32)
                ground_truth = np.array(batch_labels, dtype=np.int32)
                sample_weights = np.ones(len(batch_labels), dtype=np.float32)
                yield (fingerprints, ground_truth, sample_weights)
        return factory



# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================


def ensure_processed_directory(
    base_dir: Union[str, Path],
    create: bool = True,
) -> dict:
    """Ensure processed data directory structure exists.

    Args:
        base_dir: Base directory for processed data
        create: If True, create directories that don't exist

    Returns:
        Dictionary mapping directory names to paths
    """
    base_dir = Path(base_dir)

    dirs = {
        "train": base_dir / "train",
        "val": base_dir / "val",
        "test": base_dir / "test",
    }

    if create:
        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {path}")

    return dirs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_feature_store(
    base_dir: Union[str, Path],
    name: str = "features",
) -> FeatureStore:
    """Create a new feature store.

    Args:
        base_dir: Base directory for storage
        name: Name for the store

    Returns:
        FeatureStore instance
    """
    return FeatureStore(Path(base_dir) / name)


def load_dataset(
    data_path: Union[str, Path],
    split: str = "train",
    batch_size: int = 32,
) -> WakeWordDataset:
    """Convenience function to load wake word dataset.

    Args:
        data_path: Path to processed data directory
        split: Data split to load
        batch_size: Batch size

    Returns:
        WakeWordDataset instance
    """
    return WakeWordDataset(data_path=data_path, split=split, batch_size=batch_size)
