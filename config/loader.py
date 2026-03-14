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
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def scale_learning_rates(learning_rates: List[float], from_batch: int, to_batch: int, scaling_method: str = "sqrt") -> List[float]:
    """
    Scale learning rates when changing batch size.

    Args:
        learning_rates: Base learning rates for `from_batch`
        from_batch: Original batch size
        to_batch: New batch size
        scaling_method: "sqrt" (default) or "linear"

    Returns:
        Scaled learning rates for `to_batch`

    Raises:
        ValueError: Unknown scaling method
    """
    if scaling_method == "sqrt":
        scale_factor = math.sqrt(to_batch / from_batch)
    elif scaling_method == "linear":
        scale_factor = to_batch / from_batch
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    scaled = [lr * scale_factor for lr in learning_rates]
    logger.info(f"LR scaling ({scaling_method}): {learning_rates} × {scale_factor:.4f} → {scaled} (batch {from_batch} → {to_batch})")
    return scaled


def validate_lr_scaling(learning_rates: List[float], batch_size: int, base_batch_size: int = 128, base_learning_rates: List[float] | None = None, tolerance: float = 0.05) -> bool:
    """
    Check if learning rates match expected sqrt scaling from base.

    Args:
        learning_rates: Learning rates to validate
        batch_size: Current batch size
        base_batch_size: Reference batch size (default 128)
        base_learning_rates: Reference learning rates (default for batch=128)
        tolerance: Relative tolerance for deviation (default 5%)

    Returns:
        True if within tolerance of expected values.

    Logs:
        Warning if values are manually set but deviate from expected.
    """
    if base_learning_rates is None:
        base_learning_rates = [0.001, 0.0002, 0.00005]
    expected = scale_learning_rates(base_learning_rates, base_batch_size, batch_size)
    expected = scale_learning_rates(base_learning_rates, base_batch_size, batch_size)

    if len(learning_rates) != len(expected):
        logger.warning(f"LR array length mismatch: {len(learning_rates)} vs expected {len(expected)} for batch_size={batch_size}")
        return False

    for actual, exp in zip(learning_rates, expected, strict=False):
        deviation = abs(actual - exp) / exp
        if deviation > tolerance:
            logger.warning(f"LR value {actual:.6f} deviates from expected sqrt-scaled value {exp:.6f} by {deviation * 100:.1f}% for batch_size={batch_size} (tolerance: {tolerance * 100:.0f}%)")
            return False

    logger.info(f"LR validation passed: {learning_rates} matches sqrt-scaled expected values within {tolerance * 100:.0f}% tolerance")
    return True


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

    def __post_init__(self) -> None:
        if self.sample_rate_hz < 1000:
            raise ValueError("hardware.sample_rate_hz must be >= 1000")


@dataclass
class PathsConfig:
    """Directory and file paths for data and outputs."""

    positive_dir: str = "${DATASET_DIR:-./dataset}/positive"
    negative_dir: str = "${DATASET_DIR:-./dataset}/negative"
    hard_negative_dir: str = "${DATASET_DIR:-./dataset}/hard_negative"
    background_dir: str = "${DATASET_DIR:-./dataset}/background"
    rir_dir: str = "${DATASET_DIR:-./dataset}/rirs"
    processed_dir: str = "${DATA_DIR:-./data}/processed"
    checkpoint_dir: str = "${CHECKPOINT_DIR:-./models/checkpoints}"
    export_dir: str = "${MODEL_EXPORT_DIR:-./models/exported}"


@dataclass
class TrainingConfig:
    """Training parameters and schedule."""

    training_steps: List[int] = field(default_factory=lambda: [40000, 25000, 15000])
    # Learning rate auto-scaling: if learning_rates is None, auto-scale from base_learning_rates
    # Reference point for scaling (default 128)
    base_batch_size: int = 128
    base_learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.0002, 0.00005])
    # Explicit values override auto-scaling
    learning_rates: Optional[List[float]] = None
    batch_size: int = 384
    # Scaling method: "sqrt" (default) or "linear"
    # Scaling method: "sqrt" (default) or "linear"
    auto_lr_scale_method: str = "sqrt"
    eval_step_interval: int = 500
    eval_basic_step_interval: int = 500
    materialize_metrics_interval: int = 1000
    eval_advanced_step_interval: int = 2000
    eval_confusion_matrix_interval: int = 5000
    eval_checkpoints_interval: int = 1000
    eval_log_every_step: bool = True
    # Class weights
    positive_class_weight: List[float] = field(default_factory=lambda: [5.0, 7.0, 9.0])
    negative_class_weight: List[float] = field(default_factory=lambda: [1.5, 1.5, 1.5])
    hard_negative_class_weight: List[float] = field(default_factory=lambda: [3.0, 5.0, 7.0])
    # SpecAugment parameters
    time_mask_max_size: List[int] = field(default_factory=lambda: [1, 2, 3])
    time_mask_count: List[int] = field(default_factory=lambda: [1, 1, 1])
    freq_mask_max_size: List[int] = field(default_factory=lambda: [1, 2, 3])
    freq_mask_count: List[int] = field(default_factory=lambda: [1, 1, 1])
    # Checkpoint selection
    minimization_metric: str = "ambient_false_positives_per_hour"
    target_minimization: float = 2.0
    maximization_metric: str = "average_viable_recall"
    steps_per_epoch: int = 1000
    ambient_duration_hours: float = 42.02
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42
    strict_content_hash_leakage_check: bool = True
    random_seed: Optional[int] = 42
    auto_tune_on_poor_fah: bool = True  # DEPRECATED: use auto_tuning.enabled instead
    # Optimizer and loss parameters (NEW)
    optimizer: str = "adam"  # Optimizer type (currently only "adam" supported)
    label_smoothing: float = 0.01  # Label smoothing for BinaryCrossentropy (0.0 = disabled)
    gradient_clipnorm: Optional[float] = 2.0  # Gradient clipping (None = disabled)
    ema_decay: Optional[float] = 0.999  # EMA decay rate (None = disabled)
    # Intra-phase cosine LR decay
    cosine_decay_alpha: float = 0.0  # Min LR fraction within each phase (0.0 = decay to 0, 1.0 = no decay/flat LR)
    # Plateau-based LR reduction and early stopping
    plateau_lr_factor: float = 0.3  # Factor to multiply LR on plateau (e.g., 0.3 = reduce to 30%)
    plateau_patience: int = 3  # Number of consecutive plateau evaluations before LR reduction
    plateau_max_reductions: int = 2  # Max LR reductions before early stopping (0 = disabled)
    # Phase transition staggering
    phase_stagger_steps: int = 2000  # Steps to delay class weights and augmentation after LR transition
    # BatchNorm freezing
    freeze_bn_on_plateau: bool = True  # Freeze BatchNorm layers when plateau detected

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"training.label_smoothing must be in range [0.0, 1.0), got {self.label_smoothing}")
        if not 0.0 <= self.cosine_decay_alpha <= 1.0:
            raise ValueError(f"training.cosine_decay_alpha must be in [0.0, 1.0], got {self.cosine_decay_alpha}")
        if not 0.0 < self.plateau_lr_factor <= 1.0:
            raise ValueError(f"training.plateau_lr_factor must be in (0.0, 1.0], got {self.plateau_lr_factor}")
        if not isinstance(self.plateau_patience, int) or self.plateau_patience <= 0:
            raise ValueError(f"training.plateau_patience must be an integer > 0, got {self.plateau_patience}")
        if not isinstance(self.plateau_max_reductions, int) or self.plateau_max_reductions < 0:
            raise ValueError(f"training.plateau_max_reductions must be an integer >= 0, got {self.plateau_max_reductions}")
        if not isinstance(self.phase_stagger_steps, int) or self.phase_stagger_steps < 0:
            raise ValueError(f"training.phase_stagger_steps must be an integer >= 0, got {self.phase_stagger_steps}")
        if not isinstance(self.phase_stagger_steps, int) or self.phase_stagger_steps < 0:
            raise ValueError(f"training.phase_stagger_steps must be an integer >= 0, got {self.phase_stagger_steps}")
        # Auto-scale learning rates if not explicitly set
        if self.learning_rates is None:
            self.learning_rates = scale_learning_rates(self.base_learning_rates, from_batch=self.base_batch_size, to_batch=self.batch_size, scaling_method=self.auto_lr_scale_method)
        if len(self.training_steps) != len(self.learning_rates):
            raise ValueError("training.training_steps and learning_rates must have same length")
            raise ValueError("training.training_steps and learning_rates must have same length")


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    architecture: str = "mixednet"
    first_conv_filters: int = 32
    first_conv_kernel_size: int = 5
    stride: int = 3
    pointwise_filters: str = "64,64,64,64"
    mixconv_kernel_sizes: str = "[5],[7,11],[9,15],[23]"
    repeat_in_block: str = "1,1,1,1"
    residual_connection: str = "0,1,1,1"
    dropout_rate: float = 0.08
    l2_regularization: float = 0.00003

    def __post_init__(self) -> None:
        if self.architecture != "mixednet":
            raise ValueError("model.architecture must be 'mixednet'")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"model.dropout_rate must be in range [0.0, 1.0], got {self.dropout_rate}")


@dataclass
class AugmentationConfig:
    """Audio augmentation parameters."""

    # Time-domain augmentations — probabilities
    SevenBandParametricEQ: float = 0.20
    TanhDistortion: float = 0.08
    PitchShift: float = 0.10
    BandStopFilter: float = 0.1
    AddColorNoise: float = 0.20
    AddBackgroundNoiseFromFile: float = 0.30
    Gain: float = 1.0
    ApplyImpulseResponse: float = 0.30
    # Noise mixing parameters
    background_min_snr_db: float = -5.0
    background_max_snr_db: float = 15.0
    min_jitter_s: float = 0.15
    max_jitter_s: float = 0.20
    # Augmentation magnitude ranges
    eq_min_gain_db: float = -6.0
    eq_max_gain_db: float = 6.0
    distortion_min: float = 0.05
    distortion_max: float = 0.2
    pitch_shift_min_semitones: float = -1.2
    pitch_shift_max_semitones: float = 1.2
    band_stop_min_center_freq: float = 125.0
    band_stop_max_center_freq: float = 7500.0
    band_stop_min_bandwidth_fraction: float = 0.5
    band_stop_max_bandwidth_fraction: float = 1.99
    gain_min_db: float = -5.0
    gain_max_db: float = 5.0
    color_noise_min_snr_db: float = -10.0
    color_noise_max_snr_db: float = 20.0
    # Background sources
    impulse_paths: List[str] = field(default_factory=lambda: ["${DATASET_DIR:-./dataset}/rirs"])
    background_paths: List[str] = field(default_factory=lambda: ["${DATASET_DIR:-./dataset}/background"])
    augmentation_duration_s: float = 3.2


@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""

    gpu_only: bool = True
    spec_augment_backend: str = "tf"
    async_mining: bool = True  # DEPRECATED: moved to MiningConfig (kept for backward compat)
    mixed_precision: bool = True
    num_workers: int = 8
    num_threads_per_worker: int = 2
    prefetch_factor: int = 12
    pin_memory: bool = True
    max_memory_gb: int = 40
    inter_op_parallelism: int = 16
    intra_op_parallelism: int = 16
    # Profiling
    enable_profiling: bool = False
    profile_every_n_steps: int = 1000
    profile_output_dir: str = "./profiles"
    tf_profile_start_step: int = 0  # Step to start TF Profiler GPU trace (0 = disabled)
    gpu_memory_log_interval: int = 1000  # Log GPU memory every N steps (0 = disabled)
    # TensorBoard
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = "./logs"
    tensorboard_log_histograms: bool = False
    tensorboard_log_images: bool = False
    tensorboard_log_pr_curves: bool = True
    tensorboard_log_graph: bool = True
    tensorboard_log_advanced_scalars: bool = True
    tensorboard_log_weight_histograms: bool = False
    tensorboard_image_interval: int = 5000
    tensorboard_histogram_interval: int = 5000
    # New sophisticated TensorBoard metrics (Phase 4)
    tensorboard_log_learning_rate: bool = True  # Track LR schedule
    tensorboard_log_gradient_norms: bool = False  # Gradient norm histograms (expensive)
    tensorboard_log_activation_stats: bool = False  # Per-layer activation statistics
    tensorboard_log_confidence_drift: bool = True  # Track prediction confidence over time
    tensorboard_log_per_class_accuracy: bool = True  # Accuracy breakdown by class
    tensorboard_sophisticated_interval: int = 2000  # Interval for sophisticated metrics
    # Data pipeline (NEW)
    prefetch_buffer: int = 8  # Buffer size for tf.data pipeline
    use_tfdata: bool = True  # Use tf.data pipeline instead of Python generators
    tfdata_cache_dir: Optional[str] = None  # Cache directory (None = memory cache)
    mmap_readonly: bool = True  # Open feature store mmap as read-only
    tfdata_prefetch_to_device: bool = True  # Use GPU prefetch when available
    tfdata_prefetch_device: str = "/GPU:0"  # Device string for prefetch_to_device
    benchmark_pipeline: bool = False  # Benchmark generator vs tf.data pipeline
    log_throughput: bool = False  # Log data/step throughput to detect bottlenecks
    log_throughput_interval: int = 1000  # Steps between throughput logs
    disable_mmap: bool = False  # Disable feature-store mmap (use on SIGBUS)


@dataclass
class SpeakerClusteringConfig:
    """Speaker clustering configuration with performance optimizations."""

    enabled: bool = True
    method: str = "adaptive"  # agglomerative, hdbscan, or adaptive
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    similarity_threshold: float = 0.68
    n_clusters: Optional[int] = None
    leakage_audit_enabled: bool = True
    leakage_similarity_threshold: float = 0.9

    # Performance optimizations (Phase 2)
    use_embedding_cache: bool = True
    cache_dir: Optional[str] = None  # Custom cache directory
    batch_size: Optional[int] = None  # None = auto-detect from GPU
    num_io_workers: int = 8
    use_mixed_precision: bool = True
    use_dataloader: bool = True

    # Adaptive clustering (Phase 3)
    use_adaptive_clustering: bool = True
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    adaptive_threshold_small: int = 5000  # Use agglomerative below this
    adaptive_threshold_large: int = 50000  # Use two-stage above this


@dataclass
class MiningConfig:
    """Unified configuration for all mining, false prediction extraction, and logging.

    Consolidates the former HardNegativeMiningConfig, TopFPExtractionConfig,
    and PerformanceConfig.async_mining into a single section.
    """

    # General
    enabled: bool = True
    async_mining: bool = True

    # In-training mining
    fp_threshold: float = 0.65
    max_samples: int = 5000
    mining_interval_epochs: int = 1
    collection_mode: str = "log_only"  # "log_only" | "mine_immediately"

    # Logging
    log_predictions: bool = True
    log_file: str = "logs/false_predictions.json"
    top_k_per_epoch: int = 150

    # Post-training mining
    enable_post_training_mining: bool = True
    mined_subdirectory: str = "mined"
    min_epochs_before_mining: int = 10
    deduplicate_by_hash: bool = True

    # Top false positive extraction (formerly TopFPExtractionConfig)
    extract_top_fps: bool = True
    top_fp_percent: float = 5.0  # Top N% of false positives to extract
    extraction_confidence_threshold: float = 0.8  # Min score to consider as FP
    extraction_output_dir: str = "${DATASET_DIR:-./dataset}/top5fps"  # Destination for extracted files
    extraction_log_file: str = "logs/top_fp_extraction.json"  # JSON log path
    run_extraction_at_training_end: bool = True  # Auto-run at end of training
    extraction_batch_size: int = 128  # Batch size for inference scan

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.fp_threshold <= 1.0:
            raise ValueError(f"fp_threshold must be between 0.0 and 1.0, got {self.fp_threshold}")
        if self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")
        if self.mining_interval_epochs <= 0:
            raise ValueError(f"mining_interval_epochs must be positive, got {self.mining_interval_epochs}")
        if not isinstance(self.min_epochs_before_mining, int) or self.min_epochs_before_mining <= 0:
            raise ValueError(f"min_epochs_before_mining must be a positive integer, got {self.min_epochs_before_mining}")
        if not isinstance(self.top_k_per_epoch, int) or self.top_k_per_epoch <= 0:
            raise ValueError(f"top_k_per_epoch must be a positive integer, got {self.top_k_per_epoch}")
        if self.collection_mode not in ("log_only", "mine_immediately"):
            raise ValueError(f"collection_mode must be 'log_only' or 'mine_immediately', got {self.collection_mode}")
        if not 0.0 < self.top_fp_percent <= 100.0:
            raise ValueError(f"top_fp_percent must be between 0 and 100, got {self.top_fp_percent}")
        if not 0.0 <= self.extraction_confidence_threshold <= 1.0:
            raise ValueError(f"extraction_confidence_threshold must be between 0.0 and 1.0, got {self.extraction_confidence_threshold}")
        if self.extraction_batch_size <= 0:
            raise ValueError(f"extraction_batch_size must be positive, got {self.extraction_batch_size}")


@dataclass
class ExportConfig:
    """Model export settings."""

    wake_word: str = "Hey Katya"
    author: str = "Sarpel GURAY"
    website: str = "https://github.com/sarpel/microwakeword_trainer"
    trained_languages: List[str] = field(default_factory=lambda: ["en"])
    quantize: bool = True
    inference_input_type: str = "int8"
    inference_output_type: str = "uint8"
    probability_cutoff: float = 0.97
    sliding_window_size: int = 5
    tensor_arena_size: int = 0  # 0 means auto-calculate from exported TFLite
    minimum_esphome_version: str = "2024.7.0"
    # TFLite calibration (NEW)
    representative_dataset_size: int = 1000  # Number of samples for random calibration
    representative_dataset_real_size: int = 4000  # Number of samples for real-data calibration
    max_samples_for_cutoff_calc: int = 5000  # Max test samples for export auto-cutoff inference
    arena_size_margin: float = 1.3  # Multiplier for tensor arena size (1.3 = 30% margin)

    def __post_init__(self):
        """Validate export configuration."""
        if self.tensor_arena_size < 0:
            raise ValueError(f"export.tensor_arena_size must be >= 0, got {self.tensor_arena_size}")
        if self.arena_size_margin <= 0.0:
            raise ValueError(f"export.arena_size_margin must be > 0.0, got {self.arena_size_margin}")


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
class EvaluationConfig:
    """Evaluation and metrics configuration."""

    default_threshold: float = 0.97  # Default probability threshold for metrics (legacy name)
    n_thresholds: int = 101  # Number of thresholds for ROC/PR curves
    max_fah: float = 10.0  # Maximum FAH for average viable recall calculation
    target_fah: float = 2.0  # Target FAH for recall@FAH metrics
    target_recall: float = 0.90  # Target recall for FAH@recall metrics
    gain_window_steps: int = 1000  # Step window for gain metrics
    plateau_window_evals: int = 5  # Rolling evals for plateau detection
    plateau_min_delta: float = 0.001  # Minimum improvement to consider progress
    plateau_slope_eps: float = 0.0001  # Slope epsilon for plateau detection
    warmup_runs: int = 10  # Warmup runs for latency measurement
    n_latency_runs: int = 100  # Number of runs for latency measurement


@dataclass
class AutoTuningConfig:
    """Auto-tuning configuration for post-training fine-tuning."""

    # Activation
    enabled: bool = True

    # Targets
    target_fah: float = 2.0
    target_recall: float = 0.90

    # Iteration control
    max_iterations: int = 50
    max_gradient_steps: int = 250_000
    patience: int = 15

    # Legacy fields (kept for backward compat, ignored by MaxQualityAutoTuner)
    steps_per_iteration: int = 8000
    initial_lr: float = 0.00005
    lr_decay_factor: float = 0.7
    min_lr: float = 1e-5
    positive_weight_range: List[float] = field(default_factory=lambda: [0.5, 3.0])
    negative_weight_range: List[float] = field(default_factory=lambda: [10.0, 50.0])
    hard_negative_weight_range: List[float] = field(default_factory=lambda: [20.0, 100.0])

    # Cross-validation & confirmation
    cv_folds: int = 3
    confirmation_fraction: float = 0.40
    # Fraction of search partition reserved for evaluation (rest for training)
    search_eval_fraction: float = 0.30
    bootstrap_samples: int = 2000

    require_confirmation: bool = True

    # Grouping
    group_key: str = "speaker_id"

    # Pareto / convergence (legacy)
    pareto_improvement_threshold: float = 0.003
    convergence_window: int = 7

    # Output
    output_dir: str = "./tuning_output"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.target_recall <= 1.0:
            raise ValueError(f"AutoTuningConfig: target_recall must be between 0.0 and 1.0, got {self.target_recall}")
        if not (0.0 < self.lr_decay_factor < 1.0):
            raise ValueError(f"AutoTuningConfig: lr_decay_factor must be >0.0 and <1.0, got {self.lr_decay_factor}")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError(f"AutoTuningConfig: max_iterations must be a positive integer, got {self.max_iterations}")
        if not isinstance(self.patience, int) or self.patience <= 0:
            raise ValueError(f"AutoTuningConfig: patience must be a positive integer, got {self.patience}")
        if not isinstance(self.steps_per_iteration, int) or self.steps_per_iteration <= 0:
            raise ValueError(f"AutoTuningConfig: steps_per_iteration must be a positive integer, got {self.steps_per_iteration}")
        if not isinstance(self.convergence_window, int) or self.convergence_window <= 0:
            raise ValueError(f"AutoTuningConfig: convergence_window must be a positive integer, got {self.convergence_window}")
        if self.min_lr >= self.initial_lr:
            raise ValueError(f"AutoTuningConfig: min_lr must be < initial_lr, got min_lr={self.min_lr} >= initial_lr={self.initial_lr}")
        for range_name, range_val in (
            ("positive_weight_range", self.positive_weight_range),
            ("negative_weight_range", self.negative_weight_range),
            ("hard_negative_weight_range", self.hard_negative_weight_range),
        ):
            if len(range_val) != 2:
                raise ValueError(f"AutoTuningConfig: {range_name} must contain exactly two numbers, got {range_val}")
            if range_val[0] >= range_val[1]:
                raise ValueError(f"AutoTuningConfig: {range_name}[0] must be < {range_name}[1], got {range_val}")
        if not isinstance(self.cv_folds, int) or self.cv_folds < 2:
            raise ValueError(f"AutoTuningConfig: cv_folds must be >= 2, got {self.cv_folds}")
        if not 0.0 < self.confirmation_fraction < 1.0:
            raise ValueError(f"AutoTuningConfig: confirmation_fraction must be between 0 and 1, got {self.confirmation_fraction}")


@dataclass
class AutoTuningExpertConfig:
    """Expert-level auto-tuning parameters. Most users should use defaults."""

    # Burst step bounds
    min_burst_steps: int = 200
    max_burst_steps: int = 25000
    default_burst_steps: int = 750

    # Learning rate bounds
    min_lr: float = 1e-7
    max_lr: float = 1e-4
    default_lr: float = 1e-5

    # SAM / SWA
    sam_rho: float = 0.05
    swa_collection_interval: int = 100

    # Simulated annealing
    initial_temperature: float = 0.5
    cooling_rate: float = 0.97
    reheat_after: int = 5
    reheat_factor: float = 1.3

    # Pool / archive sizes
    active_pool_size: int = 16
    pareto_archive_size: int = 32

    # Stir level thresholds (stagnation counts)
    stir_level_1: int = 3
    stir_level_2: int = 5
    stir_level_3: int = 7
    stir_level_4: int = 9
    stir_level_5: int = 12

    # Curriculum
    curriculum_advance_threshold: float = 0.3

    def __post_init__(self):
        """Validate expert configuration."""
        if self.min_burst_steps >= self.max_burst_steps:
            raise ValueError("AutoTuningExpertConfig: min_burst_steps must be < max_burst_steps")
        if self.min_lr >= self.max_lr:
            raise ValueError("AutoTuningExpertConfig: min_lr must be < max_lr")
        if not 0.0 < self.sam_rho < 1.0:
            raise ValueError("AutoTuningExpertConfig: sam_rho must be between 0 and 1")
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError("AutoTuningExpertConfig: cooling_rate must be between 0 and 1")


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
    mining: MiningConfig = field(default_factory=MiningConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    auto_tuning: AutoTuningConfig = field(default_factory=AutoTuningConfig)
    auto_tuning_expert: AutoTuningExpertConfig = field(default_factory=AutoTuningExpertConfig)


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
        "mining": MiningConfig,
        "export": ExportConfig,
        "preprocessing": PreprocessingConfig,
        "quality": QualityConfig,
        "evaluation": EvaluationConfig,
        "auto_tuning": AutoTuningConfig,
        "auto_tuning_expert": AutoTuningExpertConfig,
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
            # ARCHITECTURAL_CONSTITUTION enforcement - IMMUTABLE VALUES
            if hw.get("sample_rate_hz") != 16000:
                issues.append("hardware.sample_rate_hz must be 16000 (ARCHITECTURAL_CONSTITUTION)")
            if hw.get("mel_bins") != 40:
                issues.append("hardware.mel_bins must be 40 (ARCHITECTURAL_CONSTITUTION)")
            if hw.get("window_step_ms") != 10:
                issues.append("hardware.window_step_ms must be 10 (ARCHITECTURAL_CONSTITUTION)")
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

        # Cross-section feasibility: label smoothing vs deployment threshold
        if "training" in config and "export" in config:
            ls = config["training"].get("label_smoothing", 0.0)
            threshold = config["export"].get("probability_cutoff", 0.97)
            eval_threshold = config.get("evaluation", {}).get("default_threshold", threshold)
            deploy_threshold = max(threshold, eval_threshold)

            if ls > 0:
                # With smoothing, target is capped below 1.0: 1.0 - 0.5*ls
                smoothed_target = 1.0 - 0.5 * ls
                headroom = smoothed_target - deploy_threshold
                min_headroom = 0.05  # 5% minimum for smoothed targets
            else:
                # Without smoothing, model can output 0-1.0, but 1.0 is theoretical max
                # Allow tighter headroom (2%) since smoothed_target=1.0 is not a hard cap
                smoothed_target = 1.0
                headroom = smoothed_target - deploy_threshold
                min_headroom = 0.02  # 2% minimum for unsmoothed

            if headroom < min_headroom:
                issues.append(
                    f"Mathematical infeasibility: smoothed_target={smoothed_target:.3f}, "
                    f"deployment threshold={deploy_threshold:.3f} (headroom={headroom:.3f} < {min_headroom:.0%}). "
                    f"The model has insufficient headroom to reliably cross threshold. "
                    f"Either reduce label_smoothing or lower the threshold."
                )
        # Validate model section
        if "model" in config:
            md = config["model"]
            valid_architectures = ["mixednet"]
            if md.get("architecture") not in valid_architectures:
                issues.append(f"model.architecture must be one of: {valid_architectures}")
            # ARCHITECTURAL_CONSTITUTION enforcement - stride must be 3
            if md.get("stride") != 3:
                issues.append("model.stride must be 3 (ARCHITECTURAL_CONSTITUTION)")
        # Validate performance section
        if "performance" in config:
            perf = config["performance"]
            if perf.get("num_workers", 0) < 0:
                issues.append("performance.num_workers must be >= 0")
            if perf.get("max_memory_gb", 0) <= 0:
                issues.append("performance.max_memory_gb must be > 0")

        # Validate export section
        if "export" in config:
            exp = config["export"]
            # ARCHITECTURAL_CONSTITUTION enforcement - output type must be uint8
            if exp.get("inference_output_type") != "uint8":
                issues.append("export.inference_output_type must be 'uint8' (ARCHITECTURAL_CONSTITUTION)")

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
        full_config = FullConfig(**result)

        # Post-process: substitute env vars and resolve paths in ALL string
        # fields, including dataclass defaults that were never in the YAML.
        # Without this, defaults like "${CHECKPOINT_DIR:-./models/checkpoints}"
        # pass through as literal strings.
        self._process_dataclass_defaults(full_config)

        return full_config

    def _process_dataclass_defaults(self, full_config: "FullConfig") -> None:
        """Substitute env vars and resolve paths in all string fields of FullConfig sections.

        Mutates the dataclass in-place. This catches default values that contain
        ``${VAR:-default}`` patterns which were never processed because the
        corresponding YAML section was absent or incomplete.
        """
        env_pattern = re.compile(r"\$\{[^}:]+(?::-[^}]*)?\}")

        for section_name in self.SECTION_CLASSES:
            section = getattr(full_config, section_name, None)
            if section is None:
                continue
            for f in dataclasses.fields(section):
                value = getattr(section, f.name)
                if isinstance(value, str) and env_pattern.search(value):
                    processed = self._substitute_env_vars(value)
                    processed = self._resolve_path(processed)
                    object.__setattr__(section, f.name, processed)

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

        def _deep_copy_value(value: Any) -> Any:
            if isinstance(value, dict):
                return self._deep_copy_dict(value)
            if isinstance(value, list):
                return [_deep_copy_value(item) for item in value]
            return value

        result = {}
        for key, value in d.items():
            result[key] = _deep_copy_value(value)
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
