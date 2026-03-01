"""
YAML configuration loader with validation for microwakeword_trainer v2.0

Provides:
- ConfigLoader: Main loader class with load, load_preset, merge, validate methods
- Dataclasses for type-safe config access
- Environment variable substitution
- Path resolution
- Configuration validation
"""

import dataclasses
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# DATACLASS CONFIGURATION STRUCTURES
# =============================================================================


@dataclass
class HardwareConfig:
    """Hardware/audio processing parameters (typically immutable)."""

    sample_rate_hz: int = 16000
    mel_bins: int = 40
    window_size_ms: int = 30
    window_step_ms: int = 10
    clip_duration_ms: int = 1000


@dataclass
class PathsConfig:
    """Directory and file paths for data and outputs."""

    positive_dir: str = "./data/raw/positive"
    negative_dir: str = "./data/raw/negative"
    hard_negative_dir: str = "./data/raw/hard_negative"
    background_dir: str = "./data/raw/background"
    rir_dir: str = "./data/raw/rirs"
    processed_dir: str = "./data/processed"
    checkpoint_dir: str = "./checkpoints"
    export_dir: str = "./models/exported"


@dataclass
class TrainingConfig:
    """Training parameters and schedule."""

    training_steps: List[int] = field(default_factory=lambda: [20000, 10000])
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.0001])
    batch_size: int = 128
    eval_step_interval: int = 500
    # Class weights
    positive_class_weight: List[float] = field(default_factory=lambda: [1.0, 1.0])
    negative_class_weight: List[float] = field(default_factory=lambda: [20.0, 20.0])
    hard_negative_class_weight: List[float] = field(default_factory=lambda: [40.0, 40.0])  # Higher weight for false positives
    # SpecAugment parameters
    time_mask_max_size: List[int] = field(default_factory=lambda: [0, 0])
    time_mask_count: List[int] = field(default_factory=lambda: [0, 0])
    freq_mask_max_size: List[int] = field(default_factory=lambda: [0, 0])
    freq_mask_count: List[int] = field(default_factory=lambda: [0, 0])
    # Checkpoint selection
    minimization_metric: str = "ambient_false_positives_per_hour"
    target_minimization: float = 0.5
    maximization_metric: str = "average_viable_recall"
    steps_per_epoch: int = 1000
    ambient_duration_hours: float = 11.3
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42
    strict_content_hash_leakage_check: bool = True


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    architecture: str = "mixednet"
    first_conv_filters: int = 30
    first_conv_kernel_size: int = 5
    stride: int = 3
    pointwise_filters: str = "60,60,60,60"
    mixconv_kernel_sizes: str = "[5],[9],[13],[21]"
    repeat_in_block: str = "1,1,1,1"
    residual_connection: str = "0,0,0,0"
    dropout_rate: float = 0.0
    l2_regularization: float = 0.0


@dataclass
class AugmentationConfig:
    """Audio augmentation parameters."""

    # Time-domain augmentations — probabilities
    SevenBandParametricEQ: float = 0.1
    TanhDistortion: float = 0.1
    PitchShift: float = 0.1
    BandStopFilter: float = 0.1
    AddColorNoise: float = 0.1
    AddBackgroundNoiseFromFile: float = 0.75
    Gain: float = 1.0
    ApplyImpulseResponse: float = 0.5
    # Noise mixing parameters
    background_min_snr_db: float = -5.0
    background_max_snr_db: float = 10.0
    min_jitter_s: float = 0.195
    max_jitter_s: float = 0.205
    # Augmentation magnitude ranges
    eq_min_gain_db: float = -6.0
    eq_max_gain_db: float = 6.0
    distortion_min: float = 0.1
    distortion_max: float = 0.5
    pitch_shift_min_semitones: float = -2.0
    pitch_shift_max_semitones: float = 2.0
    band_stop_min_center_freq: float = 100.0
    band_stop_max_center_freq: float = 5000.0
    band_stop_min_bandwidth_fraction: float = 0.5
    band_stop_max_bandwidth_fraction: float = 1.99
    gain_min_db: float = -3.0
    gain_max_db: float = 3.0
    color_noise_min_snr_db: float = -5.0
    color_noise_max_snr_db: float = 10.0
    # Background sources
    impulse_paths: List[str] = field(default_factory=lambda: ["mit_rirs"])
    background_paths: List[str] = field(default_factory=lambda: ["fma_16k", "audioset_16k"])
    augmentation_duration_s: float = 3.2


@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""

    gpu_only: bool = False
    mixed_precision: bool = True
    num_workers: int = 16
    num_threads_per_worker: int = 2
    prefetch_factor: int = 8
    pin_memory: bool = True
    max_memory_gb: int = 60
    inter_op_parallelism: int = 16
    intra_op_parallelism: int = 16
    # Profiling
    enable_profiling: bool = True
    profile_every_n_steps: int = 100
    profile_output_dir: str = "./profiles"
    # TensorBoard
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = "./logs"


@dataclass
class SpeakerClusteringConfig:
    """Speaker clustering configuration with performance optimizations."""

    enabled: bool = True
    method: str = "adaptive"  # agglomerative, hdbscan, or adaptive
    embedding_model: str = "speechbrain/ecapa-tdnn-voxceleb"
    similarity_threshold: float = 0.72
    n_clusters: Optional[int] = None
    leakage_audit_enabled: bool = True
    leakage_similarity_threshold: float = 0.9

    # Performance optimizations (Phase 2)
    use_embedding_cache: bool = True
    cache_dir: Optional[str] = None  # Custom cache directory
    batch_size: Optional[int] = None  # None = auto-detect from GPU
    num_io_workers: int = 8
    use_mixed_precision: bool = True
    use_dataloader: bool = False

    # Adaptive clustering (Phase 3)
    use_adaptive_clustering: bool = True
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3


@dataclass
class HardNegativeMiningConfig:
    """Hard negative mining configuration."""

    enabled: bool = True
    fp_threshold: float = 0.8
    max_samples: int = 5000
    mining_interval_epochs: int = 5


@dataclass
class ExportConfig:
    """Model export settings."""

    wake_word: str = "Hey Katya"
    author: str = "Your Name"
    website: str = "https://your-repo.com"
    trained_languages: List[str] = field(default_factory=lambda: ["en"])
    quantize: bool = True
    inference_input_type: str = "int8"
    inference_output_type: str = "uint8"
    probability_cutoff: float = 0.97
    sliding_window_size: int = 5
    tensor_arena_size: int = 26080
    minimum_esphome_version: str = "2024.7.0"


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline parameters (VAD, splitting, duration filtering)."""

    # Duration filtering applied by vad_trim_audio.py
    min_duration_ms: float = 300.0
    max_duration_ms: float = 2000.0
    discarded_dir: str = "./discarded"

    # VAD trim parameters
    vad_aggressiveness: int = 2  # webrtcvad aggressiveness (0-3)
    vad_pad_ms: int = 200  # silence padding to keep around speech (ms)
    vad_frame_ms: int = 30  # VAD frame size (must be 10, 20, or 30 ms)

    # Background audio splitting (split_long_audio.py)
    split_max_chunk_ms: float = 2000.0
    split_min_chunk_ms: float = 500.0
    split_target_chunk_ms: float = 2000.0


@dataclass
class QualityConfig:
    """Audio quality scoring thresholds for dataset curation."""

    # Clipping detection
    clip_threshold: float = 0.001  # fraction of samples at or beyond clip level
    max_clip_ratio: float = 0.01  # maximum allowed clipping ratio (0.0-1.0)

    # Score-based filtering
    discard_bottom_pct: float = 5.0  # discard lowest N% of files by WQI score
    min_wqi: float = 0.0  # absolute minimum WQI threshold (0.0-1.0)
    discarded_quality_dir: str = "./discarded/quality"

    # WADA-SNR threshold
    min_snr_db: float = -10.0  # minimum acceptable SNR in dB

    # Silero VAD speech threshold (used by score_audio_quality_full.py)
    vad_speech_threshold: float = 0.3  # minimum fraction of frames containing speech

    # DNSMOS thresholds (score_audio_quality_full.py)
    dnsmos_min_ovrl: float = 0.0  # minimum DNSMOS OVRL score (0.0-5.0)
    dnsmos_min_sig: float = 0.0  # minimum DNSMOS SIG score (0.0-5.0)

    # DNSMOS model cache directory
    dnsmos_cache_dir: str = "~/.cache/dnsmos"


@dataclass
class FullConfig:
    """Complete configuration container."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    speaker_clustering: SpeakerClusteringConfig = field(default_factory=SpeakerClusteringConfig)
    hard_negative_mining: HardNegativeMiningConfig = field(default_factory=HardNegativeMiningConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================


class ConfigLoader:
    """
    Main configuration loader class.

    Provides:
    - load(path): Load config from YAML file
    - load_preset(name): Load preset config
    - merge(base, override): Merge two configs
    - validate(config): Validate configuration
    """

    # Valid preset names
    VALID_PRESETS = {"fast_test", "standard", "max_quality", "test", "standart", "high_quality"}
    PRESET_ALIASES = {
        "test": "fast_test",
        "standart": "standard",
        "high_quality": "max_quality",
    }

    # Config section mapping to dataclass
    SECTION_CLASSES = {
        "hardware": HardwareConfig,
        "paths": PathsConfig,
        "training": TrainingConfig,
        "model": ModelConfig,
        "augmentation": AugmentationConfig,
        "performance": PerformanceConfig,
        "speaker_clustering": SpeakerClusteringConfig,
        "hard_negative_mining": HardNegativeMiningConfig,
        "export": ExportConfig,
        "preprocessing": PreprocessingConfig,
        "quality": QualityConfig,
    }

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize ConfigLoader.

        Args:
            base_dir: Base directory for resolving relative paths.
                     Defaults to current working directory.
        """
        self.base_dir = base_dir or Path.cwd()
        self.presets_dir = Path(__file__).parent / "presets"
        self._config_dir: Optional[Path] = None

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Dictionary with configuration data

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Resolve config file directory for relative path resolution
        self._config_dir = path.parent

        with open(path, "r") as f:
            # First pass: load raw YAML
            raw_config = yaml.safe_load(f)

        # Process environment variables and path resolution
        config = self._process_config(raw_config)

        return config

    def load_preset(self, name: str) -> Dict[str, Any]:
        """
        Load a preset configuration by name.

        Args:
            name: Preset name (fast_test, standard, max_quality)

        Returns:
            Dictionary with preset configuration

        Raises:
            ValueError: If preset name is invalid
            FileNotFoundError: If preset file doesn't exist
        """
        canonical_name = self.PRESET_ALIASES.get(name, name)

        if canonical_name not in {"fast_test", "standard", "max_quality"}:
            raise ValueError(f"Invalid preset '{name}'. Valid presets: {', '.join(sorted(self.VALID_PRESETS))}")

        preset_path = self.presets_dir / f"{canonical_name}.yaml"
        return self.load(preset_path)

    def merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with override taking precedence.

        Nested dictionaries are merged recursively.
        Lists are replaced (not merged).

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = self._deep_copy_dict(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self.merge(result[key], value)
            else:
                # Replace value
                result[key] = value

        return result

    def load_and_merge(self, preset_name: str, override_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load preset and optionally merge with override config.

        Args:
            preset_name: Name of preset to load
            override_path: Optional path to override YAML

        Returns:
            Merged configuration dictionary
        """
        base_config = self.load_preset(preset_name)

        if override_path is not None:
            override_config = self.load(override_path)
            return self.merge(base_config, override_config)

        return base_config

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of issues.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check required sections
        for section in self.SECTION_CLASSES:
            if section not in config:
                issues.append(f"Missing required section: '{section}'")

        # Validate hardware section
        if "hardware" in config:
            hw = config["hardware"]
            if hw.get("sample_rate_hz", 0) < 1000:
                issues.append("hardware.sample_rate_hz must be >= 1000")
            if hw.get("mel_bins", 0) < 10:
                issues.append("hardware.mel_bins must be >= 10")
            if hw.get("window_size_ms", 0) <= 0:
                issues.append("hardware.window_size_ms must be > 0")
            if hw.get("window_step_ms", 0) <= 0:
                issues.append("hardware.window_step_ms must be > 0")

        # Validate training section
        if "training" in config:
            tr = config["training"]
            if not isinstance(tr.get("training_steps", []), list):
                issues.append("training.training_steps must be a list")
            if not isinstance(tr.get("learning_rates", []), list):
                issues.append("training.learning_rates must be a list")
            if len(tr.get("training_steps", [])) != len(tr.get("learning_rates", [])):
                issues.append("training.training_steps and learning_rates must have same length")
            if tr.get("batch_size", 0) <= 0:
                issues.append("training.batch_size must be > 0")
            train_split = tr.get("train_split", 0.0)
            val_split = tr.get("val_split", 0.0)
            test_split = tr.get("test_split", 0.0)
            split_sum = train_split + val_split + test_split
            if abs(split_sum - 1.0) > 1e-6:
                issues.append("training.train_split + training.val_split + training.test_split must equal 1.0")

        # Validate model section
        if "model" in config:
            md = config["model"]
            valid_architectures = ["mixednet"]
            if md.get("architecture") not in valid_architectures:
                issues.append(f"model.architecture must be one of: {valid_architectures}")

        # Validate performance section
        if "performance" in config:
            perf = config["performance"]
            if perf.get("num_workers", 0) < 0:
                issues.append("performance.num_workers must be >= 0")
            if perf.get("max_memory_gb", 0) <= 0:
                issues.append("performance.max_memory_gb must be > 0")

        return issues

    def to_dataclass(self, config: Dict[str, Any]) -> "FullConfig":
        """
        Convert dictionary config to FullConfig dataclass.

        Args:
            config: Configuration dictionary

        Returns:
            FullConfig instance with nested dataclasses
        """
        result: Dict[str, Any] = {}

        for section_name, section_class in self.SECTION_CLASSES.items():
            if section_name in config:
                section_data = config[section_name]
                # Filter to only fields the dataclass accepts
                valid_fields = {f.name for f in dataclasses.fields(section_class)}
                filtered = {k: v for k, v in section_data.items() if k in valid_fields}
                if len(filtered) < len(section_data):
                    unknown = set(section_data) - valid_fields
                    logger.warning(f"Config section '{section_name}' has unknown fields (ignored): {unknown}")
                result[section_name] = section_class(**filtered)
        return FullConfig(**result)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration: env vars, path resolution, defaults."""
        if not isinstance(config, dict):
            return config  # type: ignore[unreachable]

        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                # Environment variable substitution
                value = self._substitute_env_vars(value)
                # Path resolution
                value = self._resolve_path(value)
            elif isinstance(value, dict):
                value = self._process_config(value)
            elif isinstance(value, list):
                value = self._process_list(value)

            result[key] = value

        return result

    def _process_list(self, items: List[Any]) -> List[Any]:
        """Process list items for env vars and paths."""
        result = []
        for item in items:
            if isinstance(item, str):
                item = self._substitute_env_vars(item)
                item = self._resolve_path(item)
            elif isinstance(item, dict):
                item = self._process_config(item)
            elif isinstance(item, list):
                item = self._process_list(item)
            result.append(item)
        return result

    def _substitute_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in string.

        Supports ${VAR} and ${VAR:-default} syntax.

        Args:
            value: String potentially containing env var references

        Returns:
            String with substituted values
        """
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)

            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                # Leave unresolved if no default
                return match.group(0)

        return re.sub(pattern, replacer, value)

    def _resolve_path(self, value: str) -> str:
        """
        Resolve relative paths.

        - Paths starting with ``./`` resolve against the **current working
          directory** (project root), because they represent data/output paths.
        - Paths starting with ``../`` resolve against the **config file
          directory**, because they reference sibling/parent configs.
        - Bare relative paths with slashes (e.g. ``config/data.yaml``) resolve
          against CWD if they look like file paths.

        Args:
            value: String potentially containing a path

        Returns:
            Resolved path string
        """
        if not isinstance(value, str):
            return value  # type: ignore[unreachable]

        # Skip URLs and absolute paths
        if value.startswith(("http://", "https://", "file://", "/")):
            return value

        # Skip HuggingFace-style model IDs (e.g. "microsoft/wavlm-base-plus", "org/model.v2")
        # Treat as HF ID if it contains '/' and the last segment doesn't end with a known file extension
        _file_extensions = (
            ".json",
            ".yaml",
            ".yml",
            ".py",
            ".txt",
            ".cfg",
            ".ini",
            ".toml",
        )
        if "/" in value and not value.split("/")[-1].endswith(_file_extensions):
            return value

        # ./paths → resolve against CWD (project root)
        if value.startswith("./"):
            resolved = (Path.cwd() / value).resolve()
            return str(resolved)

        # ../paths → resolve against config file directory
        if value.startswith("../") and self._config_dir is not None:
            resolved = (self._config_dir / value).resolve()
            return str(resolved)

        # Bare relative paths with slashes → resolve against CWD
        if self._config_dir is not None and not os.path.isabs(value) and "/" in value:
            parts = value.split("/")
            has_extension = "." in parts[-1]
            has_multiple_segments = len(parts) > 2
            if has_extension or has_multiple_segments:
                resolved = (Path.cwd() / value).resolve()
                return str(resolved)

        return value

    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of dictionary."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_dict(value)
            elif isinstance(value, list):
                result[key] = list(value)  # type: ignore[assignment]  # shallow copy for lists
            else:
                result[key] = value
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_loader: Optional[ConfigLoader] = None


def get_default_loader() -> ConfigLoader:
    """Get or create default ConfigLoader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Convenience function using default loader.
    """
    return get_default_loader().load(path)


def load_preset(name: str) -> Dict[str, Any]:
    """
    Load a preset configuration by name.

    Convenience function using default loader.

    Args:
        name: Preset name (fast_test, standard, max_quality, test, standart, high_quality)
    """
    return get_default_loader().load_preset(name)


def load_full_config(preset_name: str = "standard", override_path: Optional[Union[str, Path]] = None) -> FullConfig:
    """
    Load complete configuration as dataclass.

    Args:
        preset_name: Name of preset to load (default: standard)
        override_path: Optional path to override YAML

    Returns:
        FullConfig dataclass instance
    """
    loader = get_default_loader()
    config_dict = loader.load_and_merge(preset_name, override_path)
    return loader.to_dataclass(config_dict)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage
    import json

    # Load a preset
    config = load_preset("standard")
    print("Loaded standard preset:")
    print(json.dumps(config, indent=2))

    # Validate
    loader = get_default_loader()
    issues = loader.validate(config)
    if issues:
        print(f"\nValidation issues: {issues}")
    else:
        print("\nConfiguration is valid!")

    # Get dataclass
    full_config = load_full_config("fast_test")
    print(f"\nBatch size: {full_config.training.batch_size}")
    print(f"Model architecture: {full_config.model.architecture}")
