"""Dataset module for managing training data with efficient mmap storage.

Provides:
- RaggedMmap: Variable-length array storage using memory-mapped files
- WakeWordDataset: PyTorch-compatible dataset for training
- FeatureStore: Manage features on disk with mmap access
"""

import logging
import os
import struct
from hashlib import sha1
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
            create: If True, create directory if it doesn't exist
        """
        self.base_dir = Path(base_dir)
        self.name = name
        self.dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
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

    def open(self, mode: str = "r"):
        """Open the storage for reading or writing.

        Args:
            mode: 'r' for read, 'w' for write
        """
        if mode == "r":
            self._load_index()
            assert self._data_file is not None, "data_file not initialized"
            if os.path.exists(self._data_file):
                self._data = np.memmap(
                    self._data_file,
                    dtype=self.dtype,
                    mode="r",
                )
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
        dtype: Optional["np.dtype[Any]"] = None,
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
            np.dtype(np.int32),
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
        self._config: Optional[Dict[str, Any]] = None
        if config is not None:
            self._config = config
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
            self.max_time_frames = int(clip_duration_ms / window_step_ms)
        else:
            # Legacy initialization
            # Legacy initialization
            self.data_path = Path(data_path) if data_path else Path("./data/processed")
            self.split = self._normalize_split_name(split)
            self.batch_size = batch_size
            self.feature_dim = feature_dim
            self._is_built = False
            self.max_time_frames = 100  # Default: 1000ms / 10ms

        # Try to load from store
        self.feature_store: Optional[FeatureStore] = None
        self._load_store()

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

    def _extract_features_to_store(self, samples: List[Any], store_dir: Path, frontend: Any, sample_rate: int, split_name: str, mel_bins: int) -> int:
        logger.info(f"Extracting features for {len(samples)} {split_name} clips...")
        store = FeatureStore(store_dir)
        store.initialize(len(samples), feature_dim=mel_bins)

        for sample in samples:
            try:
                audio = load_audio_wave(sample.path, target_sr=sample_rate)
                features = frontend.compute_mel_spectrogram(audio)
                store.add(features, self._label_to_int(sample.label))
            except Exception as e:
                logger.warning(f"Failed to process {sample.path}: {e}")
                continue

        count = len(store)
        store.close()
        logger.info(f"Processed {count} {split_name} samples")
        return count

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
        processed_dir = paths_cfg.get("processed_dir", "./data/processed")

        # Extract hardware config for feature extraction
        hardware_cfg = cfg.get("hardware", {})
        sample_rate = hardware_cfg.get("sample_rate_hz", 16000)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        window_size_ms = hardware_cfg.get("window_size_ms", 30)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)

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

        # Ensure processed directories exist
        dirs = ensure_processed_directory(processed_dir)

        # Load clips using ClipsLoaderConfig
        logger.info("Loading audio clips from dataset directories...")

        training_cfg = cfg.get("training", {})
        train_split = float(training_cfg.get("train_split", 0.8))
        val_split = float(training_cfg.get("val_split", 0.1))
        test_split = float(training_cfg.get("test_split", 0.1))
        split_seed = int(training_cfg.get("split_seed", 42))
        split_sum = train_split + val_split + test_split
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"training split ratios must sum to 1.0, got {split_sum:.6f}")

        clips_config = ClipsLoaderConfig(
            positive_dir=Path(positive_dir) if positive_dir else None,
            negative_dir=Path(negative_dir) if negative_dir else None,
            hard_negative_dir=Path(hard_negative_dir) if hard_negative_dir else None,
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

        self._extract_features_to_store(train_samples, dirs["train"], frontend, sample_rate, "training", mel_bins)
        if val_samples:
            self._extract_features_to_store(val_samples, dirs["val"], frontend, sample_rate, "validation", mel_bins)
        if test_samples:
            self._extract_features_to_store(test_samples, dirs["test"], frontend, sample_rate, "test", mel_bins)

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
                raise ValueError("Cannot reshape flattened array")
            features = features.reshape(time_frames, self.feature_dim)

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

    def _iter_split_batches(
        self,
        split: str,
        max_time_frames: int,
        *,
        infinite: bool,
        shuffle: bool,
        include_hard_negative_flag: bool,
    ):
        split_name = self._normalize_split_name(split)
        store_path = self.data_path / split_name
        if not store_path.exists():
            logger.warning(f"Split store not found at {store_path}")
            return

        store = FeatureStore(store_path)
        try:
            store.open()
            num_samples = len(store)
            if num_samples == 0:
                return

            indices = list(range(num_samples))
            rng = np.random.RandomState(42)

            while True:
                epoch_indices = rng.permutation(indices).tolist() if shuffle else indices
                batch_features = []
                batch_labels = []
                batch_is_hard_neg = []

                for idx in epoch_indices:
                    try:
                        feature, label = store.get(idx)
                    except (RuntimeError, IndexError):
                        continue

                    fixed_feature = self._pad_or_truncate(feature, max_time_frames)
                    batch_features.append(fixed_feature)
                    batch_labels.append(label & 1)
                    if include_hard_negative_flag:
                        batch_is_hard_neg.append(label == 2)

                    if len(batch_features) >= self.batch_size:
                        fingerprints = np.array(batch_features, dtype=np.float32)
                        ground_truth = np.array(batch_labels, dtype=np.int32)
                        sample_weights = np.ones(len(batch_labels), dtype=np.float32)
                        if include_hard_negative_flag:
                            is_hard_neg = np.array(batch_is_hard_neg, dtype=np.bool_)
                            yield (fingerprints, ground_truth, sample_weights, is_hard_neg)
                        else:
                            yield (fingerprints, ground_truth, sample_weights)
                        batch_features = []
                        batch_labels = []
                        batch_is_hard_neg = []

                if batch_features:
                    fingerprints = np.array(batch_features, dtype=np.float32)
                    ground_truth = np.array(batch_labels, dtype=np.int32)
                    sample_weights = np.ones(len(batch_labels), dtype=np.float32)
                    if include_hard_negative_flag:
                        is_hard_neg = np.array(batch_is_hard_neg, dtype=np.bool_)
                        yield (fingerprints, ground_truth, sample_weights, is_hard_neg)
                    else:
                        yield (fingerprints, ground_truth, sample_weights)

                if not infinite:
                    break
        finally:
            store.close()

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
