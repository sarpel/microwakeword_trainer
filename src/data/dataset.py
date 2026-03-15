"""Dataset module for managing training data with efficient mmap storage.

Provides:
- RaggedMmap: Variable-length array storage using memory-mapped files
- WakeWordDataset: PyTorch-compatible dataset for training
- FeatureStore: Manage features on disk with mmap access
"""

import logging
import os
import struct
from collections import OrderedDict
import json
from datetime import datetime
from hashlib import sha1, sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.data.ingestion import load_audio_wave

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
        dtype: Optional["np.dtype[Any]"] = None,
        create: bool = True,
    ):
        """Initialize RaggedMmap storage.

        Args:
            base_dir: Base directory for storage files
            name: Name for this storage
            dtype: NumPy data type for arrays
            create: If True, create directory if it does not exist
        """
        self.base_dir = Path(base_dir)
        self.name = name
        self.dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
        self._data_file: Optional[str] = None
        self._offsets_file: Optional[str] = None
        self._lengths_file: Optional[str] = None
        self._data: Optional[np.ndarray] = None
        self._offsets: Union[List[int], np.ndarray, None] = None
        self._lengths: Union[List[int], np.ndarray, None] = None
        self._num_arrays: int = 0
        self._total_bytes: int = 0
        self._memory_cache: OrderedDict[int, np.ndarray] | None = None

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
    def total_bytes(self) -> int:
        """Get total number of bytes across all arrays."""
        return self._total_bytes

    def _load_index(self):
        """Load index files (offsets and lengths)."""
        assert self._offsets_file is not None, "offsets_file not initialized"
        assert self._lengths_file is not None, "lengths_file not initialized"
        if os.path.exists(self._offsets_file) and os.path.exists(self._lengths_file):
            # Read binary offset data
            with open(self._offsets_file, "rb") as f:
                offset_data = f.read()
            num_offsets = len(offset_data) // 8  # 8 bytes per int64
            self._offsets = np.frombuffer(offset_data, dtype=np.int64, count=num_offsets)

            # Read binary lengths data
            with open(self._lengths_file, "rb") as f:
                length_data = f.read()
            self._lengths = np.frombuffer(length_data, dtype=np.int64)

            self._num_arrays = len(self._lengths)
            self._total_bytes = int(self._lengths.sum())

            # Validate offsets and lengths have the same length
            if len(self._offsets) != len(self._lengths):
                raise ValueError(f"Offsets ({len(self._offsets)}) and lengths ({len(self._lengths)}) mismatch")
            self._total_bytes = int(self._lengths.sum())

    def open(self, mode: str = "r", disable_mmap: bool = False):
        """Open the storage for reading or writing.

        Args:
            mode: 'r' for read, 'w' for write
        """
        if mode == "r":
            self._load_index()
            assert self._data_file is not None, "data_file not initialized"
            if os.path.exists(self._data_file):
                if disable_mmap:
                    with open(self._data_file, "rb") as handle:
                        data = handle.read()
                    self._data = np.frombuffer(data, dtype=self.dtype)
                    self._memory_cache = OrderedDict()
                else:
                    self._data = np.memmap(
                        self._data_file,
                        dtype=self.dtype,
                        mode="r",
                    )
                    data_size = os.path.getsize(self._data_file)
                    if data_size < self._total_bytes:
                        raise RuntimeError(f"RaggedMmap data file is smaller than expected: {data_size} bytes < {self._total_bytes} bytes")
        elif mode == "w":
            # Truncate existing files before writing to prevent data corruption
            for f in [self._data_file, self._offsets_file, self._lengths_file]:
                assert f is not None, "file path not initialized"
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
            flush_method = getattr(self._data, "flush", None)
            if callable(flush_method):
                try:
                    flush_method()
                except ValueError:
                    # Memmap already closed/invalid in some read-only scenarios
                    pass
            try:
                mmap_obj = getattr(self._data, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
            except AttributeError:
                pass
            del self._data
            self._data = None
        self._offsets = None
        self._lengths = None
        self._memory_cache = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def append(self, arrays: List[np.ndarray]):
        """Append arrays to storage.

        Args:
            arrays: List of arrays to append
        """
        # Convert to numpy arrays
        arrays = [np.asarray(arr, dtype=self.dtype) for arr in arrays]

        # Get lengths (in bytes for offset calculation)
        lengths = [arr.nbytes for arr in arrays]

        # Calculate offsets
        if self._num_arrays > 0:
            assert self._offsets is not None, "offsets not loaded"
            assert self._lengths is not None, "lengths not loaded"
            last_offset = int(self._offsets[-1]) + int(self._lengths[-1])
        else:
            last_offset = 0

        offsets = [last_offset + sum(lengths[:i]) for i in range(len(arrays))]

        # Append to data file
        assert self._data_file is not None, "data_file not initialized"
        with open(self._data_file, "ab") as f:
            for arr in arrays:
                f.write(arr.tobytes())

        # Append to index files
        assert self._offsets_file is not None, "offsets_file not initialized"
        with open(self._offsets_file, "ab") as f:
            for offset in offsets:
                f.write(struct.pack("q", offset))

        assert self._lengths_file is not None, "lengths_file not initialized"
        with open(self._lengths_file, "ab") as f:
            for length in lengths:
                f.write(struct.pack("q", length))

        # Update in-memory index so subsequent append() calls can compute offsets
        if isinstance(self._offsets, list):
            self._offsets.extend(offsets)
        else:
            self._offsets = list(self._offsets) + list(offsets) if self._offsets is not None else list(offsets)
        if isinstance(self._lengths, list):
            self._lengths.extend(lengths)
        else:
            self._lengths = list(self._lengths) + list(lengths) if self._lengths is not None else list(lengths)

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

        assert self._offsets is not None, "offsets not loaded"
        assert self._lengths is not None, "lengths not loaded"
        assert self._data is not None, "data not loaded"

        offset = int(self._offsets[idx])
        length = int(self._lengths[idx])
        if offset < 0 or length < 0:
            raise RuntimeError("RaggedMmap offset/length must be non-negative")

        # Convert byte offsets/lengths to element counts
        itemsize = self._data.itemsize
        elem_offset = offset // itemsize
        elem_length = length // itemsize
        if elem_offset + elem_length > self._data.shape[0]:
            raise RuntimeError("RaggedMmap index out of bounds; data file may be corrupted")

        if self._memory_cache is not None:
            if idx in self._memory_cache:
                self._memory_cache.move_to_end(idx)
                return self._memory_cache[idx]
            array = np.array(self._data[elem_offset : elem_offset + elem_length])
            self._memory_cache[idx] = array
            if len(self._memory_cache) > 1024:
                self._memory_cache.popitem(last=False)
            return array
        array = np.array(self._data[elem_offset : elem_offset + elem_length])
        return array

    def clear_cache(self):
        """Clear memory cache to release memory between epochs."""
        if self._memory_cache is not None:
            cache_size = len(self._memory_cache)
            self._memory_cache.clear()
            logger.debug(f"Cleared RaggedMmap cache ({cache_size} items)")

    def get_batch(self, indices: List[int]) -> List[np.ndarray]:
        """Get multiple arrays by indices with optimized sequential access.

        This method sorts indices for sequential memory access, which is more
        efficient than random access when using memory-mapped files. The returned
        arrays are in the same order as the requested indices.

        Args:
            indices: List of array indices to retrieve

        Returns:
            List of arrays in the order of the input indices
        """
        if not indices:
            return []

        # Sort indices for sequential read efficiency
        sorted_indices = sorted(enumerate(indices), key=lambda x: x[1])
        result: List[Optional[np.ndarray]] = [None] * len(indices)

        for orig_idx, idx in sorted_indices:
            result[orig_idx] = self[idx]

        # Filter out None values (shouldn't happen, but type-safe)
        return [arr for arr in result if arr is not None]

    def __len__(self) -> int:
        """Get number of stored arrays."""
        return self._num_arrays

    @staticmethod
    def create_from_arrays(
        arrays: List[np.ndarray],
        base_dir: Union[str, Path],
        name: str = "ragged",
        dtype: Optional[np.dtype] = None,
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

        # Open stores in write mode so append() works immediately
        self.features.open("w")
        self.labels.open("w")

    def add(self, feature: np.ndarray, label: int):
        """Add a single feature-label pair.

        Args:
            feature: Feature array
            label: Label (0=negative, 1=positive, 2=hard_negative)
        """
        if self.features is None:
            raise RuntimeError("Feature store not initialized")
        if self.labels is None:
            raise RuntimeError("Labels store not initialized")

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
            raise ValueError(f"features and labels must have the same length, got {len(features)} vs {len(labels)}")

        if self.features is None:
            # Initialize with first sample to get feature dim
            self.initialize(len(features), features[0].shape[-1])

        assert self.features is not None, "initialize() must set self.features"
        assert self.labels is not None, "initialize() must set self.labels"
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

    def open(self, readonly: bool = True, disable_mmap: bool = False):
        """Open store for reading."""
        self.features = RaggedMmap(
            self.base_dir,
            self.config.features_name,
            self.config.dtype,
            create=False,
        )
        self.features.open("r", disable_mmap=disable_mmap)

        self.labels = RaggedMmap(
            self.base_dir,
            self.config.labels_name,
            np.dtype(np.int32),
            create=False,
        )
        self.labels.open("r", disable_mmap=disable_mmap)

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
        config: Dict[str, Any],
        split: str = "train",
        batch_size: int = 32,
        feature_dim: int = 40,
    ):
        """Initialize dataset.

        Args:
            config: Configuration dictionary with paths and hardware settings.
                   When provided, the build() method can be called to process
                   raw audio files into features.
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for training
            feature_dim: Dimension of feature vectors
        """
        self._config: Dict[str, Any] = config
        paths_cfg = config.get("paths", {})
        hardware_cfg = config.get("hardware", {})
        training_cfg = config.get("training", {})

        self.data_path = Path(paths_cfg.get("processed_dir", "./data/processed"))
        self.batch_size = training_cfg.get("batch_size", batch_size)
        self.feature_dim = hardware_cfg.get("mel_bins", feature_dim)
        self.split = self._normalize_split_name(split)
        self._is_built = False

        # Derive max_time_frames from hardware config
        clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)
        if window_step_ms <= 0:
            raise ValueError(f"window_step_ms must be positive, got {window_step_ms}")
        self.max_time_frames = int(clip_duration_ms / window_step_ms)

        # Try to load from store
        self.feature_store: Optional[FeatureStore] = None
        self._load_store()

        # Pre-allocate batch buffers to avoid repeated allocation
        max_batch_size = self.batch_size
        self._batch_buffer = {
            "features": np.empty((max_batch_size, self.max_time_frames, self.feature_dim), dtype=np.float32),
            "labels": np.empty(max_batch_size, dtype=np.int64),
            "weights": np.empty(max_batch_size, dtype=np.float32),
            "is_hard_neg": np.empty(max_batch_size, dtype=np.bool_),
        }
        self._batch_idx = 0
        logger.info(f"Batch buffer pre-allocated: {max_batch_size}x{self.max_time_frames}x{self.feature_dim}")

    @staticmethod
    def _normalize_split_name(split: str) -> str:
        normalized = split.strip().lower()
        if normalized in {"validation", "val"}:
            return "val"
        if normalized in {"train", "test"}:
            return normalized
        raise ValueError(f"Unsupported split '{split}'. Expected one of: train, val, validation, test")

    def _load_store(self):
        """Load feature store if available."""
        self.split = self._normalize_split_name(self.split)
        store_path = self.data_path / self.split
        if store_path.exists():
            self.feature_store = FeatureStore(store_path)
            try:
                readonly = True
                disable_mmap = False
                if self._config is not None:
                    readonly = bool(self._config.get("performance", {}).get("mmap_readonly", True))
                    disable_mmap = bool(self._config.get("performance", {}).get("disable_mmap", False))
                self.feature_store.open(readonly=readonly, disable_mmap=disable_mmap)
            except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                logger.warning(f"Could not open feature store at {store_path}: {e}")
                self.feature_store = None

    def __len__(self) -> int:
        """Return dataset length."""
        if self.feature_store is not None:
            return len(self.feature_store)
        return 0

    def get_label_distribution(self, split: str = "train") -> dict[int, int]:
        """Count samples per label class in a split.

        Labels: 0=negative, 1=positive, 2=hard_negative.

        Args:
            split: Data split name ('train', 'val', 'test').

        Returns:
            Dict mapping label int to sample count.
        """
        split_name = self._normalize_split_name(split)
        store_path = self.data_path / split_name
        if not store_path.exists():
            return {}
        store = FeatureStore(store_path)
        try:
            readonly = True
            disable_mmap = False
            if self._config is not None:
                readonly = bool(self._config.get("performance", {}).get("mmap_readonly", True))
                disable_mmap = bool(self._config.get("performance", {}).get("disable_mmap", False))
            store.open(readonly=readonly, disable_mmap=disable_mmap)
            counts: dict[int, int] = {}
            for idx in range(len(store)):
                _, label = store.get(idx)
                counts[label] = counts.get(label, 0) + 1
            store.close()
            return counts
        except (FileNotFoundError, PermissionError, OSError, IOError) as e:
            logger.warning(f"Could not read label distribution for {split_name}: {e}")
            return {}

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

    @staticmethod
    def _label_to_int(label: Any) -> int:
        from src.data.ingestion import Label

        if label == Label.POSITIVE:
            return 1
        if label == Label.HARD_NEGATIVE:
            return 2
        return 0

    @staticmethod
    def _resolved_sample_paths(samples: List[Any]) -> set[str]:
        return {str(Path(sample.path).resolve()) for sample in samples}

    @staticmethod
    def _normalized_speaker_ids(samples: List[Any]) -> set[str]:
        speaker_ids: set[str] = set()
        for sample in samples:
            sid = sample.speaker_id
            if sid is None:
                continue
            sid_norm = str(sid).strip().lower()
            if not sid_norm or sid_norm in {"unknown", "none", "null"}:
                continue
            speaker_ids.add(sid_norm)
        return speaker_ids

    def _assert_split_integrity(self, train_samples: List[Any], val_samples: List[Any], test_samples: List[Any], cfg: Dict[str, Any]) -> None:
        split_to_samples = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

        split_paths = {name: self._resolved_sample_paths(samples) for name, samples in split_to_samples.items()}

        for split_name, paths in split_paths.items():
            if len(paths) != len(split_to_samples[split_name]):
                raise RuntimeError(f"Duplicate clip paths detected within split '{split_name}'. This can cause silent leakage and inflated metrics.")

        for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_paths[left_name].intersection(split_paths[right_name])
            if overlap:
                examples = sorted(overlap)[:5]
                raise RuntimeError(f"Data leakage detected: {len(overlap)} overlapping file(s) between {left_name} and {right_name}. Examples: {examples}")

        speaker_cfg = cfg.get("speaker_clustering", {})
        if speaker_cfg.get("leakage_audit_enabled", False):
            split_speakers = {name: self._normalized_speaker_ids(samples) for name, samples in split_to_samples.items()}
            for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = split_speakers[left_name].intersection(split_speakers[right_name])
                if overlap:
                    examples = sorted(overlap)[:10]
                    raise RuntimeError(f"Speaker leakage detected: {len(overlap)} overlapping speaker_id(s) between {left_name} and {right_name}. Examples: {examples}")

        check_hashes = cfg.get("training", {}).get("strict_content_hash_leakage_check", True)
        if check_hashes:
            split_hashes = {
                "train": self._sample_content_hashes(train_samples),
                "val": self._sample_content_hashes(val_samples),
                "test": self._sample_content_hashes(test_samples),
            }
            for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = split_hashes[left_name].intersection(split_hashes[right_name])
                if overlap:
                    raise RuntimeError(f"Content leakage detected: {len(overlap)} duplicate-audio hash(es) between {left_name} and {right_name}.")

    @staticmethod
    def _file_content_hash(file_path: Path) -> str:
        digest = sha1()
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _sample_content_hashes(self, samples: List[Any]) -> set[str]:
        return {self._file_content_hash(Path(sample.path)) for sample in samples}

    # =========================================================================
    # CACHE MANAGEMENT - Skip expensive feature extraction when valid cache exists
    # =========================================================================

    def _compute_file_list_hash(self, paths_cfg: dict) -> str:
        """Compute hash of all WAV files by path, mtime, and size (fast, no reads)."""
        entries: list[str] = []
        for dir_key in ("positive_dir", "negative_dir", "hard_negative_dir"):
            dir_path = paths_cfg.get(dir_key)
            if not dir_path:
                continue
            dir_path = Path(dir_path)
            if not dir_path.exists():
                continue
            for wav in sorted(dir_path.rglob("*.wav")):
                try:
                    stat = wav.stat()
                    entries.append(f"{wav}|{stat.st_mtime_ns}|{stat.st_size}")
                except OSError:
                    continue
        return sha256("\n".join(entries).encode()).hexdigest()

    def _compute_hardware_hash(self, hardware_cfg: dict) -> str:
        """Compute hash of hardware/feature extraction config."""
        hardware_key = {
            "sample_rate_hz": hardware_cfg.get("sample_rate_hz", 16000),
            "mel_bins": hardware_cfg.get("mel_bins", 40),
            "window_size_ms": hardware_cfg.get("window_size_ms", 30),
            "window_step_ms": hardware_cfg.get("window_step_ms", 10),
            "clip_duration_ms": hardware_cfg.get("clip_duration_ms", 1000),
        }
        return sha256(json.dumps(hardware_key, sort_keys=True).encode()).hexdigest()

    def _compute_split_hash(self, training_cfg: dict) -> str:
        """Compute hash of split configuration."""
        split_key = {
            "train_split": float(training_cfg.get("train_split", 0.8)),
            "val_split": float(training_cfg.get("val_split", 0.1)),
            "test_split": float(training_cfg.get("test_split", 0.1)),
            "split_seed": int(training_cfg.get("split_seed", 42)),
            "speaker_based_split": bool(training_cfg.get("speaker_based_split", False)),
        }
        return sha256(json.dumps(split_key, sort_keys=True).encode()).hexdigest()

    def _is_cache_valid(self, processed_dir: str, paths_cfg: dict, hardware_cfg: dict, training_cfg: dict) -> bool:
        """Check if cached features are still valid.

        Returns True if:
        - Manifest file exists
        - All hashes match (file list, hardware config, split config)
        - All RaggedMmap files exist for train/val/test
        """
        manifest_path = Path(processed_dir) / "cache_manifest.json"
        if not manifest_path.exists():
            return False

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        # Verify version
        if manifest.get("version") != 1:
            return False

        # Compute current hashes
        current_file_hash = self._compute_file_list_hash(paths_cfg)
        current_hardware_hash = self._compute_hardware_hash(hardware_cfg)
        current_split_hash = self._compute_split_hash(training_cfg)

        # Compare hashes
        if manifest.get("file_list_hash") != current_file_hash:
            return False
        if manifest.get("hardware_hash") != current_hardware_hash:
            return False
        if manifest.get("split_hash") != current_split_hash:
            return False

        # Verify RaggedMmap files exist for all splits
        for split_name in ("train", "val", "test"):
            split_dir = Path(processed_dir) / split_name
            for name in ("features", "labels"):
                data_file = split_dir / f"{name}.data"
                offsets_file = split_dir / f"{name}.offsets"
                lengths_file = split_dir / f"{name}.lengths"
                if not data_file.exists() or not offsets_file.exists() or not lengths_file.exists():
                    return False

        return True

    def _write_cache_manifest(
        self,
        processed_dir: str,
        paths_cfg: dict,
        hardware_cfg: dict,
        training_cfg: dict,
        splits: dict,
    ) -> None:
        """Write cache manifest after successful feature extraction."""
        manifest = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "file_list_hash": self._compute_file_list_hash(paths_cfg),
            "hardware_hash": self._compute_hardware_hash(hardware_cfg),
            "split_hash": self._compute_split_hash(training_cfg),
            "splits": {name: {"count": count, "dir": name} for name, count in splits.items()},
        }
        manifest_path = Path(processed_dir) / "cache_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _extract_features_to_store(self, samples: List[Any], store_dir: Path, frontend: Any, sample_rate: int, split_name: str, mel_bins: int) -> int:
        logger.info(f"Extracting features for {len(samples)} {split_name} clips...")
        store = FeatureStore(store_dir)
        store.initialize(len(samples), feature_dim=mel_bins)

        processed_paths: list[str] = []
        for sample in samples:
            try:
                audio = load_audio_wave(sample.path, target_sr=sample_rate)
                features = frontend.compute_mel_spectrogram(audio)
                store.add(features, self._label_to_int(sample.label))
                processed_paths.append(str(sample.path))
            except Exception as e:
                logger.warning(f"Failed to process {sample.path}: {e}")
                continue

        count = len(store)
        store.close()

        # Persist ordered file paths for index-to-filepath mapping
        paths_file = store_dir / "file_paths.json"
        with open(paths_file, "w") as f:
            json.dump(processed_paths, f)
        logger.info(f"Saved {len(processed_paths)} file paths to {paths_file}")

        logger.info(f"Processed {count} {split_name} samples")
        return count

    def get_split_file_paths(self, split_name: str) -> list[str] | None:
        """Get ordered file paths for a dataset split.

        Returns the list of file paths in the same order as samples in the
        FeatureStore. Useful for mapping prediction indices to file paths.

        Args:
            split_name: Split name ('val', 'train', 'test').

        Returns:
            Ordered list of file path strings, or None if not available.
        """
        split_name = self._normalize_split_name(split_name)
        paths_file = Path(self.data_path) / split_name / "file_paths.json"
        if not paths_file.exists():
            logger.warning(f"File paths not found: {paths_file}. Rebuild dataset to enable file path tracking.")
            return None

        try:
            with open(paths_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load file paths from {paths_file}: {e}")
            return None

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
            raise ValueError("No config provided. Pass config to __init__ or build(config)")

        # Extract paths from config
        paths_cfg = cfg.get("paths", {})
        positive_dir = paths_cfg.get("positive_dir")
        negative_dir = paths_cfg.get("negative_dir")
        hard_negative_dir = paths_cfg.get("hard_negative_dir")
        hn_cfg = cfg.get("mining", {})
        mined_subdir = hn_cfg.get("mined_subdirectory", "mined")
        include_mined = bool(hn_cfg.get("enable_post_training_mining", True))
        processed_dir = paths_cfg.get("processed_dir", "./data/processed")

        # Extract hardware config for feature extraction
        hardware_cfg = cfg.get("hardware", {})
        sample_rate = hardware_cfg.get("sample_rate_hz", 16000)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        window_size_ms = hardware_cfg.get("window_size_ms", 30)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)

        # Extract training config for splits
        training_cfg = cfg.get("training", {})
        train_split = float(training_cfg.get("train_split", 0.8))
        val_split = float(training_cfg.get("val_split", 0.1))
        test_split = float(training_cfg.get("test_split", 0.1))

        # Ensure processed directories exist
        dirs = ensure_processed_directory(processed_dir)

        # Check if valid cache exists
        if self._is_cache_valid(processed_dir, paths_cfg, hardware_cfg, training_cfg):
            logger.info("[CACHE] Valid feature cache found — skipping feature extraction")
            self._load_store()
            return self

        logger.info("[CACHE] No valid cache found — performing full feature extraction")

        # Create feature config
        from src.data.features import FeatureConfig, MicroFrontend
        from src.data.ingestion import Clips, ClipsLoaderConfig, Split

        feature_config = FeatureConfig(
            sample_rate=sample_rate,
            mel_bins=mel_bins,
            window_size_ms=window_size_ms,
            window_step_ms=window_step_ms,
        )

        # Initialize feature extractor
        frontend = MicroFrontend(feature_config)

        # Load clips using ClipsLoaderConfig
        logger.info("Loading audio clips from dataset directories...")

        split_seed = int(training_cfg.get("split_seed", 42))
        split_sum = train_split + val_split + test_split
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"training split ratios must sum to 1.0, got {split_sum:.6f}")

        # Merge mined hard negatives into hard_negative_dir when enabled
        hard_negative_path = Path(hard_negative_dir) if hard_negative_dir else None
        if include_mined and hard_negative_path is not None:
            mined_path = hard_negative_path / mined_subdir
            if mined_path.exists():
                # Ensure mined directory is included in hard negative discovery
                logger.info(f"Including mined hard negatives from {mined_path}")
        clips_config = ClipsLoaderConfig(
            positive_dir=Path(positive_dir) if positive_dir else None,
            negative_dir=Path(negative_dir) if negative_dir else None,
            hard_negative_dir=hard_negative_path if hard_negative_path else None,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=split_seed,
        )

        clips = Clips(config=clips_config)

        train_samples = clips.get_split(Split.TRAIN)
        val_samples = clips.get_split(Split.VAL)
        test_samples = clips.get_split(Split.TEST)

        self._assert_split_integrity(train_samples, val_samples, test_samples, cfg)

        logger.info(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples, {len(test_samples)} test samples")

        if not train_samples:
            raise RuntimeError("No training samples found. Please check your dataset directories.")
        if not test_samples:
            raise RuntimeError("No held-out test samples found. A strict test split is required for leakage-safe evaluation.")

        train_count = self._extract_features_to_store(train_samples, dirs["train"], frontend, sample_rate, "training", mel_bins)
        val_count = 0
        test_count = 0
        if val_samples:
            val_count = self._extract_features_to_store(val_samples, dirs["val"], frontend, sample_rate, "validation", mel_bins)
        if test_samples:
            test_count = self._extract_features_to_store(test_samples, dirs["test"], frontend, sample_rate, "test", mel_bins)

        # Write cache manifest for subsequent runs
        self._write_cache_manifest(
            processed_dir,
            paths_cfg,
            hardware_cfg,
            training_cfg,
            splits={"train": train_count, "val": val_count, "test": test_count},
        )

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
                # Try to infer the actual stored feature_dim for a helpful error message.
                # mel_bins=40 is mandatory (ARCHITECTURAL_CONSTITUTION.md).
                # Common wrong value is 48; check a few candidates to give a clear hint.
                for candidate in (48, 32, 64, 80):
                    if total_elements % candidate == 0:
                        raise ValueError(
                            f"Stored feature dimension ({candidate}) does not match "
                            f"configured mel_bins ({self.feature_dim}). "
                            f"mel_bins={self.feature_dim} is mandatory per ARCHITECTURAL_CONSTITUTION.md. "
                            f"Re-run preprocessing with the correct mel_bins setting."
                        )
                raise ValueError(f"Cannot reshape flattened array of {total_elements} elements using feature_dim={self.feature_dim} (mel_bins={self.feature_dim} is mandatory). Re-run preprocessing.")
            features = features.reshape(time_frames, self.feature_dim)

        # Defensive check: 2-D features must match the configured feature_dim (mel_bins).
        # Stale preprocessed data (e.g. generated with mel_bins=48) will be caught here
        # instead of silently producing wrong shapes and cryptic broadcast errors later.
        if features.ndim == 2 and features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Stored feature_dim={features.shape[1]} does not match "
                f"configured mel_bins={self.feature_dim}. "
                f"mel_bins={self.feature_dim} is mandatory per ARCHITECTURAL_CONSTITUTION.md. "
                f"Re-run preprocessing with the correct mel_bins setting."
            )

        current_length = features.shape[0]
        if current_length > max_time_frames:
            return features[:max_time_frames, :]
        elif current_length < max_time_frames:
            padding = np.zeros(
                (max_time_frames - current_length, self.feature_dim),
                dtype=features.dtype,
            )
            return np.vstack([features, padding])
        return features

    def _add_to_batch(self, feature: np.ndarray, label: int, weight: float = 1.0, is_hard_neg: bool = False) -> bool:
        """Add sample to pre-allocated buffer. Returns True if added, False if buffer full."""
        if self._batch_idx >= len(self._batch_buffer["features"]):
            return False
        self._batch_buffer["features"][self._batch_idx] = feature
        self._batch_buffer["labels"][self._batch_idx] = label
        self._batch_buffer["weights"][self._batch_idx] = weight
        self._batch_buffer["is_hard_neg"][self._batch_idx] = is_hard_neg
        self._batch_idx += 1
        return True

    def _flush_batch(self):
        """Return current batch contents and reset buffer index."""
        if self._batch_idx == 0:
            return None
        result = (
            self._batch_buffer["features"][: self._batch_idx].copy(),
            self._batch_buffer["labels"][: self._batch_idx].copy(),
            self._batch_buffer["weights"][: self._batch_idx].copy(),
            self._batch_buffer["is_hard_neg"][: self._batch_idx].copy(),
        )
        self._batch_idx = 0
        return result

    def _ensure_batch_buffer(self, max_time_frames: int) -> None:
        """Ensure pre-allocated batch buffer shape matches current generation settings."""
        features_buffer = self._batch_buffer["features"]
        expected_shape = (self.batch_size, max_time_frames, self.feature_dim)
        if features_buffer.shape != expected_shape:
            self._batch_buffer = {
                "features": np.empty(expected_shape, dtype=np.float32),
                "labels": np.empty(self.batch_size, dtype=np.int64),
                "weights": np.empty(self.batch_size, dtype=np.float32),
                "is_hard_neg": np.empty(self.batch_size, dtype=np.bool_),
            }
            logger.info(f"Batch buffer resized: {self.batch_size}x{max_time_frames}x{self.feature_dim}")
        self._batch_idx = 0

    def _iter_split_batches(
        self,
        split: str,
        max_time_frames: int,
        *,
        infinite: bool,
        shuffle: bool,
        include_hard_negative_flag: bool,
    ):
        batch_buffer = {
            "features": np.empty((self.batch_size, max_time_frames, self.feature_dim), dtype=np.float32),
            "labels": np.empty(self.batch_size, dtype=np.int64),
            "weights": np.empty(self.batch_size, dtype=np.float32),
            "is_hard_neg": np.empty(self.batch_size, dtype=np.bool_),
        }
        batch_idx = 0
        split_name = self._normalize_split_name(split)
        store_path = self.data_path / split_name
        if not store_path.exists():
            logger.warning(f"Split store not found at {store_path}")
            return

        store = FeatureStore(store_path)
        try:
            readonly = True
            disable_mmap = False
            if self._config is not None:
                readonly = bool(self._config.get("performance", {}).get("mmap_readonly", True))
                disable_mmap = bool(self._config.get("performance", {}).get("disable_mmap", False))
            store.open(readonly=readonly, disable_mmap=disable_mmap)
            num_samples = len(store)
            if num_samples == 0:
                return

            indices = list(range(num_samples))
            rng = np.random.RandomState(42)

            while True:
                epoch_indices = rng.permutation(indices).tolist() if shuffle else indices
                batch_idx = 0

                for idx in epoch_indices:
                    try:
                        feature, label = store.get(idx)
                    except (RuntimeError, IndexError):
                        continue

                    fixed_feature = self._pad_or_truncate(feature, max_time_frames)
                    if batch_idx < len(batch_buffer["features"]):
                        batch_buffer["features"][batch_idx] = fixed_feature
                        batch_buffer["labels"][batch_idx] = label & 1
                        batch_buffer["weights"][batch_idx] = 1.0
                        batch_buffer["is_hard_neg"][batch_idx] = label == 2
                        batch_idx += 1
                        added = True
                    else:
                        added = False
                    if not added:
                        # Flush and yield the full batch until sample can be added
                        max_retries = 2  # Prevent infinite loop if buffer is misconfigured
                        retry_count = 0
                        while True:
                            if batch_idx != 0:
                                batch = (
                                    batch_buffer["features"][:batch_idx].copy(),
                                    batch_buffer["labels"][:batch_idx].copy(),
                                    batch_buffer["weights"][:batch_idx].copy(),
                                    batch_buffer["is_hard_neg"][:batch_idx].copy(),
                                )
                                batch_idx = 0
                                fingerprints, ground_truth, sample_weights, is_hard_neg = batch
                                if include_hard_negative_flag:
                                    yield (
                                        fingerprints,
                                        ground_truth.astype(np.int32, copy=False),
                                        sample_weights,
                                        is_hard_neg,
                                    )
                                else:
                                    yield (
                                        fingerprints,
                                        ground_truth.astype(np.int32, copy=False),
                                        sample_weights,
                                    )
                                retry_count = 0  # Reset counter after successful flush
                            else:
                                retry_count += 1
                                if retry_count > max_retries:
                                    raise RuntimeError(f"Failed to add sample after {max_retries} retries. Check batch_size ({self.batch_size}) is positive.")
                            # Retry adding the sample after flushing
                            if batch_idx < len(batch_buffer["features"]):
                                batch_buffer["features"][batch_idx] = fixed_feature
                                batch_buffer["labels"][batch_idx] = label & 1
                                batch_buffer["weights"][batch_idx] = 1.0
                                batch_buffer["is_hard_neg"][batch_idx] = label == 2
                                batch_idx += 1
                                added = True
                            else:
                                added = False
                            if added:
                                break

                if batch_idx != 0:
                    batch = (
                        batch_buffer["features"][:batch_idx].copy(),
                        batch_buffer["labels"][:batch_idx].copy(),
                        batch_buffer["weights"][:batch_idx].copy(),
                        batch_buffer["is_hard_neg"][:batch_idx].copy(),
                    )
                    batch_idx = 0
                    fingerprints, ground_truth, sample_weights, is_hard_neg = batch
                    if include_hard_negative_flag:
                        yield (
                            fingerprints,
                            ground_truth.astype(np.int32, copy=False),
                            sample_weights,
                            is_hard_neg,
                        )
                    else:
                        yield (
                            fingerprints,
                            ground_truth.astype(np.int32, copy=False),
                            sample_weights,
                        )

                if not infinite:
                    break
        finally:
            try:
                store.close()
            except (ValueError, RuntimeError) as e:
                message = str(e).lower()
                # Whitelist specific benign cases, log all others
                benign_phrases = [
                    "mmap closed",
                    "invalid file descriptor",
                    "closed file",
                    "i/o operation on closed",
                ]
                is_benign = any(phrase in message for phrase in benign_phrases)
                if not is_benign:
                    import logging

                    logging.warning(f"Error closing store: {e}")

    def train_generator_factory(self, max_time_frames: Optional[int] = None):
        if max_time_frames is None:
            max_time_frames = self.max_time_frames

        def factory():
            yield from self._iter_split_batches(
                split="train",
                max_time_frames=max_time_frames,
                infinite=True,
                shuffle=True,
                include_hard_negative_flag=True,
            )

        return factory

    def val_generator_factory(self, max_time_frames: Optional[int] = None):
        if max_time_frames is None:
            max_time_frames = self.max_time_frames

        def factory():
            yield from self._iter_split_batches(
                split="val",
                max_time_frames=max_time_frames,
                infinite=False,
                shuffle=False,
                include_hard_negative_flag=False,
            )

        return factory

    def test_generator_factory(self, max_time_frames: Optional[int] = None):
        if max_time_frames is None:
            max_time_frames = self.max_time_frames

        def factory():
            yield from self._iter_split_batches(
                split="test",
                max_time_frames=max_time_frames,
                infinite=False,
                shuffle=False,
                include_hard_negative_flag=False,
            )

        return factory

    def train_mining_generator_factory(self, max_time_frames: Optional[int] = None):
        if max_time_frames is None:
            max_time_frames = self.max_time_frames

        def factory():
            yield from self._iter_split_batches(
                split="train",
                max_time_frames=max_time_frames,
                infinite=False,
                shuffle=False,
                include_hard_negative_flag=False,
            )

        return factory

    def close(self) -> None:
        """Close the dataset and release resources."""
        if self.feature_store is not None:
            self.feature_store.close()
            self.feature_store = None

    def on_epoch_end(self):
        """Called at end of each epoch to release cached data."""
        stores = [self.feature_store]
        for store in stores:
            if store is None:
                continue
            for ragged_attr in ("features", "labels"):
                ragged_store = getattr(store, ragged_attr, None)
                clear_cache = getattr(ragged_store, "clear_cache", None)
                if callable(clear_cache):
                    clear_cache()
        logger.debug("Cleared dataset caches at end of epoch")

    def __enter__(self) -> "WakeWordDataset":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()


# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================


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
        create: If True, create directories that do not exist

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
        for _name, path in dirs.items():
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
    config: Dict[str, Any],
    split: str = "train",
) -> WakeWordDataset:
    """Convenience function to load wake word dataset.

    Args:
        config: Configuration dictionary with paths/hardware/training sections
        split: Data split to load

    Returns:
        WakeWordDataset instance
    """
    return WakeWordDataset(config=config, split=split)
