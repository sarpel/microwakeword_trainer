"""
YAML configuration loader with validation for microwakeword_trainer v2.0

Provides:
- ConfigLoader: Main loader class with load, load_preset, merge, validate methods
- Dataclasses for type-safe config access
- Environment variable substitution
- Path resolution
- Configuration validation
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    architecture: str = "mixednet"
    first_conv_filters: int = 30
    first_conv_kernel_size: int = 5
    stride: int = 3
    spectrogram_length: int = 49  # Number of time frames in input spectrogram
    pointwise_filters: str = "60,60,60,60"
    mixconv_kernel_sizes: str = "[5],[9],[13],[21]"
    repeat_in_block: str = "1,1,1,1"
    residual_connection: str = "0,0,0,0"
    dropout_rate: float = 0.0
    l2_regularization: float = 0.0


@dataclass
class FeatureSetConfig:
    """Single feature set configuration."""

    features_dir: str
    sampling_weight: float = 1.0
    penalty_weight: float = 1.0
    truth: bool = False
    truncation_strategy: str = "random"
    type: str = "mmap"


@dataclass
class FeaturesConfig:
    """Feature sets configuration (list of FeatureSetConfig)."""

    features: List[FeatureSetConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> "FeaturesConfig":
        """Create FeaturesConfig from list of dicts."""
        features = [FeatureSetConfig(**f) for f in data]
        return cls(features=features)


@dataclass
class AugmentationConfig:
    """Audio augmentation parameters."""

    # Time-domain augmentations
    SevenBandParametricEQ: float = 0.1
    TanhDistortion: float = 0.1
    PitchShift: float = 0.1
    BandStopFilter: float = 0.1
    AddColorNoise: float = 0.1
    AddBackgroundNoise: float = 0.75
    Gain: float = 1.0
    RIR: float = 0.5
    # Additional augmentations for max quality
    AddBackgroundNoiseFromFile: float = 0.0
    ApplyImpulseResponse: float = 0.0
    # Noise mixing parameters
    background_min_snr_db: int = -5
    background_max_snr_db: int = 10
    min_jitter_s: float = 0.195
    max_jitter_s: float = 0.205
    # Background sources
    impulse_paths: List[str] = field(default_factory=lambda: ["mit_rirs"])
    background_paths: List[str] = field(
        default_factory=lambda: ["fma_16k", "audioset_16k"]
    )
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
    """Speaker clustering configuration."""

    enabled: bool = True
    method: str = "wavlm_ecapa"
    embedding_model: str = "microsoft/wavlm-base-plus"
    similarity_threshold: float = 0.72
    leakage_audit_enabled: bool = True


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
class FullConfig:
    """Complete configuration container."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    speaker_clustering: SpeakerClusteringConfig = field(
        default_factory=SpeakerClusteringConfig
    )
    hard_negative_mining: HardNegativeMiningConfig = field(
        default_factory=HardNegativeMiningConfig
    )
    export: ExportConfig = field(default_factory=ExportConfig)


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
    VALID_PRESETS = {"fast_test", "standard", "max_quality"}

    # Config section mapping to dataclass
    SECTION_CLASSES = {
        "hardware": HardwareConfig,
        "paths": PathsConfig,
        "training": TrainingConfig,
        "model": ModelConfig,
        "features": FeaturesConfig,
        "augmentation": AugmentationConfig,
        "performance": PerformanceConfig,
        "speaker_clustering": SpeakerClusteringConfig,
        "hard_negative_mining": HardNegativeMiningConfig,
        "export": ExportConfig,
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
        if name not in self.VALID_PRESETS:
            raise ValueError(
                f"Invalid preset '{name}'. "
                f"Valid presets: {', '.join(sorted(self.VALID_PRESETS))}"
            )

        preset_path = self.presets_dir / f"{name}.yaml"
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
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dicts
                result[key] = self.merge(result[key], value)
            else:
                # Replace value
                result[key] = value

        return result

    def load_and_merge(
        self, preset_name: str, override_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
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
                issues.append(
                    "training.training_steps and learning_rates must have same length"
                )
            if tr.get("batch_size", 0) <= 0:
                issues.append("training.batch_size must be > 0")

        # Validate model section
        if "model" in config:
            md = config["model"]
            valid_architectures = ["mixednet", "dnn", "cnn", "crnn"]
            if md.get("architecture") not in valid_architectures:
                issues.append(
                    f"model.architecture must be one of: {valid_architectures}"
                )

        # Validate performance section
        if "performance" in config:
            perf = config["performance"]
            if perf.get("num_workers", 0) < 0:
                issues.append("performance.num_workers must be >= 0")
            if perf.get("max_memory_gb", 0) <= 0:
                issues.append("performance.max_memory_gb must be > 0")

        return issues

    def to_dataclass(self, config: Dict[str, Any]) -> FullConfig:
        """
        Convert dictionary config to FullConfig dataclass.

        Args:
            config: Configuration dictionary

        Returns:
            FullConfig instance with nested dataclasses
        """
        result = {}

        for section_name, section_class in self.SECTION_CLASSES.items():
            if section_name in config:
                section_data = config[section_name]

                # Handle features specially (list of FeatureSetConfig)
                if section_name == "features":
                    result[section_name] = FeaturesConfig.from_dict(section_data)
                else:
                    result[section_name] = section_class(**section_data)

        return FullConfig(**result)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration: env vars, path resolution, defaults."""
        if not isinstance(config, dict):
            return config

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
        Resolve relative paths relative to config file location.
        
        Only resolves paths that start with ../ (parent directory references).
        Paths starting with ./ are left as-is to allow user to define
        their own base directory.
        
        Args:
            value: String potentially containing a path
            
        Returns:
            Resolved path string
        """
        # Only resolve obvious path patterns
        if not isinstance(value, str):
            return value
        
        # Only resolve ../ (parent directory) references
        # Leave ./ and relative paths as-is for user flexibility
        if value.startswith('../') and self._config_dir is not None:
            resolved = (self._config_dir / value).resolve()
            return str(resolved)
        
        return value
        """
        Resolve relative paths relative to config file location.
        
        Only resolves paths that:
        - Start with ./ or ../ (explicit relative paths)
        - Contain path separators but don't look like identifiers
        
        Args:
            value: String potentially containing a path
            
        Returns:
            Resolved path string
        """
        # Only resolve obvious path patterns
        if not isinstance(value, str):
            return value
        
        # Check if it looks like a path - must start with ./ or ../
        # or be an absolute path
        is_relative_path = value.startswith('./') or value.startswith('../')
        
        if is_relative_path and self._config_dir is not None:
            # Resolve relative to config file directory
            resolved = (self._config_dir / value).resolve()
            return str(resolved)
        
        return value
        """
        Resolve relative paths relative to config file location.
        
        Only resolves paths that look like file/directory paths
        (contain path separators like / or are clearly relative paths like ./ or ../).
        
        Args:
            value: String potentially containing a path
            
        Returns:
            Resolved path string
        """
        # Only resolve obvious path patterns
        if not isinstance(value, str):
            return value
        
        # Check if it looks like a path - must contain path separator
        # or start with ./ or ../
        is_path = (
            value.startswith('./') or 
            value.startswith('../') or 
            ('/' in value and not value.startswith('$'))
        )
        
        if is_path and self._config_dir is not None:
            # Resolve relative to config file directory
            resolved = (self._config_dir / value).resolve()
            return str(resolved)
        
        return value
        """
        Resolve relative paths relative to config file location.

        Only resolves paths that look like file/directory paths
        (start with ./ or ../ or are relative without leading /).

        Args:
            value: String potentially containing a path

        Returns:
            Resolved path string
        """
        # Only resolve obvious path patterns
        if not isinstance(value, str):
            return value

        # Check if it looks like a path
        path_patterns = [
            r"^\.\./",  # starts with ../
            r"^\./",  # starts with ./
            r"^[^/]",  # starts with non-/ (relative path)
        ]

        is_path = any(re.match(p, value) for p in path_patterns)

        if is_path and self._config_dir is not None:
            # Resolve relative to config file directory
            resolved = (self._config_dir / value).resolve()
            return str(resolved)

        return value

    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of dictionary."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_dict(value)
            elif isinstance(value, list):
                result[key] = list(value)  # Shallow copy for lists
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
        name: Preset name (fast_test, standard, max_quality)
    """
    return get_default_loader().load_preset(name)


def load_full_config(
    preset_name: str = "standard", override_path: Optional[Union[str, Path]] = None
) -> FullConfig:
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
