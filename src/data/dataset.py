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
from typing import Dict, List, Optional, Tuple, Union

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
    dtype: np.dtype = np.float32
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
        dtype: np.dtype = np.float32,
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
        self._offsets: Optional[np.ndarray] = None
        self._lengths: Optional[np.ndarray] = None
        self._num_arrays: int = 0
        self._total_elements: int = 0

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
        return self._total_elements

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
            self._total_elements = int(self._lengths.sum())

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

        # Get lengths
        lengths = [len(arr) for arr in arrays]

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
        self._total_elements += sum(lengths)

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

        return np.array(self._data[offset : offset + length])

    def __len__(self) -> int:
        """Get number of stored arrays."""
        return self._num_arrays

    @staticmethod
    def create_from_arrays(
        arrays: List[np.ndarray],
        base_dir: Union[str, Path],
        name: str = "ragged",
        dtype: np.dtype = np.float32,
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
    dtype: np.dtype = np.float32


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
            np.int32,  # Labels as integers
            create=True,
        )

        self.metadata = {
            "num_samples": 0,
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
        """
        if self.features is None:
            # Initialize with first sample to get feature dim
            self.initialize(len(features), features[0].shape[-1])

        self.features.append(features)
        self.labels.append([np.array([l], dtype=np.int32) for l in labels])
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
        data_path: Union[str, Path],
        split: str = "train",
        batch_size: int = 32,
        feature_dim: int = 40,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to processed data directory
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for training
            feature_dim: Dimension of feature vectors
        """
        self.data_path = Path(data_path)
        self.split = split
        self.batch_size = batch_size
        self.feature_dim = feature_dim

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
    return WakeWordDataset(data_path, split, batch_size)
