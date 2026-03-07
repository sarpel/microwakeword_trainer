# microwakeword_trainer — Project Index

> GPU-accelerated wake word training framework for ESPHome.
> Version 2.0.0 | Branch: `consolidation`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Entry Points & CLI Tools](#entry-points--cli-tools)
4. [Module Reference](#module-reference)
   - [config/](#config)
   - [src/model/](#srcmodel)
   - [src/training/](#srctraining)
   - [src/data/](#srcdata)
   - [src/evaluation/](#srcevaluation)
   - [src/export/](#srcexport)
   - [src/tools/](#srctools)
   - [src/utils/](#srcutils)
   - [src/pipeline.py](#srcpipelinepy)
5. [Configuration System](#configuration-system)
6. [Key Data Flows](#key-data-flows)
7. [Tests](#tests)
8. [Documentation Map](#documentation-map)

---

## Project Overview

Trains compact **MixedNet** models for on-device wake word detection and exports them as INT8-quantized TFLite files compatible with the ESPHome `micro_wake_word` component. Two separate virtual environments are required:

| Env | Purpose | Key Deps |
|-----|---------|----------|
| `mww-tf` | Training, export, inference | TensorFlow, CuPy, tflite |
| `mww-torch` | Speaker clustering | PyTorch, SpeechBrain, ECAPA-TDNN |

---

## Repository Layout

```
microwakeword_trainer/
├── config/                   # Configuration system
│   ├── loader.py             # ConfigLoader + 13 dataclasses
│   ├── presets/
│   │   ├── fast_test.yaml    # ~1 hr, basic accuracy
│   │   ├── standard.yaml     # ~8 hr, production
│   │   └── max_quality.yaml  # ~24 hr, best accuracy
│   └── AGENTS.md
├── src/
│   ├── pipeline.py           # Orchestrator: train -> autotune -> export -> verify -> evaluate
│   ├── model/                # MixedNet + streaming layers
│   ├── training/             # Trainer, augmentation, hard-example mining, TensorBoard
│   ├── data/                 # Dataset I/O, features, preprocessing, augmentation, clustering
│   ├── evaluation/           # Metrics, FAH estimator, calibration, test evaluator
│   ├── export/               # TFLite export, verification, manifest, model analysis
│   ├── tools/                # CLI wrappers (cluster-analyze, cluster-apply, hard-negative mining)
│   └── utils/                # GPU config, performance helpers, logging, optional deps
├── tests/
│   ├── unit/                 # 5 unit test modules
│   └── integration/          # Training integration test
├── scripts/                  # Standalone utility scripts
├── dataset/                  # User-provided audio (positive/negative/hard_negative/background/rirs)
├── checkpoints/              # Saved model weights (*.weights.h5)
├── models/                   # Exported TFLite + manifests
├── docs/                     # Architecture, training, export, config guides
└── specs/                    # Implementation status reports
```

---

## Entry Points & CLI Tools

All registered in `setup.py` / `pyproject.toml`:

| Command | Module | Purpose |
|---------|--------|---------|
| `mww-train` | `src.training.trainer:main` | Full training pipeline |
| `mww-export` | `src.export.tflite:main` | Export checkpoint -> TFLite |
| `mww-autotune` | `src.pipeline:main` (step_autotune) | Post-training FAH/recall tuning |
| `mww-cluster-analyze` | `src.tools.cluster_analyze:main` | Speaker clustering analysis |
| `mww-cluster-apply` | `src.tools.cluster_apply:main` | Apply cluster file organization |
| `mww-mine` | `src.tools.mine_hard_negatives:main` | Mine hard negatives from logs |

Pipeline stages invoked by `src/pipeline.py`:

```
step_train -> step_autotune -> step_export -> step_verify_esphome
           -> step_verify_streaming -> step_evaluate -> step_gate -> step_promote
```

---

## Module Reference

### `config/`

**`config/loader.py`** — Configuration loading and validation.

| Symbol | Kind | Description |
|--------|------|-------------|
| `HardwareConfig` | dataclass | Sample rate, mel bins, window sizes, clip duration |
| `PathsConfig` | dataclass | dataset_dir, checkpoints_dir, output_dir, logs_dir |
| `TrainingConfig` | dataclass | Steps, learning rates, batch size, 2-phase settings |
| `ModelConfig` | dataclass | Architecture params: first_conv_filters, stride, mixconv kernels |
| `AugmentationConfig` | dataclass | SpecAugment, noise mix, audiomentations settings |
| `PerformanceConfig` | dataclass | Mixed precision, threading, prefetch, GPU settings |
| `SpeakerClusteringConfig` | dataclass | ECAPA-TDNN settings, threshold, n_clusters |
| `HardNegativeMiningConfig` | dataclass | Mining frequency, ratio, min_epoch |
| `ExportConfig` | dataclass | wake_word name, author, quantize, arena size, cutoff |
| `PreprocessingConfig` | dataclass | VAD settings, resample, split parameters |
| `QualityConfig` | dataclass | SNR thresholds, clipping detection |
| `EvaluationConfig` | dataclass | FAH targets, recall targets, evaluation intervals |
| `FullConfig` | dataclass | Aggregates all configs above |
| `ConfigLoader` | class | `load()`, `merge()`, `load_and_merge()`, `validate()`, env-var substitution |
| `load_full_config(preset, override)` | function | Primary public API for loading configs |

---

### `src/model/`

**`src/model/architecture.py`** — MixedNet model definition (Keras/TensorFlow).

| Symbol | Kind | Description |
|--------|------|-------------|
| `MixConvBlock` | class | Parallel depthwise convolutions with different kernel sizes |
| `ResidualBlock` | class | Residual wrapper around MixConvBlock |
| `MixedNet` | class | Full model: InitConv -> N×ResidualBlock -> sigmoid head |
| `build_model(config)` | function | Instantiates MixedNet from `ModelConfig` |
| `create_okay_nabu_model(config)` | function | Alias for official model variant |
| `parse_model_param(s)` | function | Parses `"[5],[9],[13]"` kernel strings |
| `spectrogram_slices_dropped(config)` | function | Computes frames consumed by strided ops |

**`src/model/streaming.py`** — Streaming inference layers for TFLite export.

| Symbol | Kind | Description |
|--------|------|-------------|
| `StreamingMixedNet` | class | Wraps MixedNet with ring-buffer state for real-time inference |
| `RingBuffer` | class | Fixed-size circular buffer layer |
| `Stream` | class | Stateful streaming wrapper for conv layers |
| `StridedDrop` / `StridedKeep` | class | Frame-stride helpers |
| `ChannelSplit` | class | Splits channels for MixConv in streaming mode |
| `Modes` | class | Enum: `TRAINING`, `INFERENCE`, `STREAM_INTERNAL_STATE_INFERENCE` |
| `get_streaming_state_names(model)` | function | Returns list of state variable names |
| `create_state_initializer(model)` | function | Generates the TFLite init subgraph |

---

### `src/training/`

**`src/training/trainer.py`** — Main training loop.

| Symbol | Kind | Description |
|--------|------|-------------|
| `Trainer` | class | Orchestrates 2-phase training, checkpointing, TensorBoard |
| `EvaluationMetrics` | dataclass | Per-epoch accuracy, FAH, recall |
| `TrainingMetrics` | namedtuple | Batch-level loss/accuracy |
| `train(config)` | function | High-level entry: build model -> train -> save best |
| `main()` | function | CLI entry point (`mww-train`) |

**`src/training/async_miner.py`** — Background hard-negative mining.

| Symbol | Kind | Description |
|--------|------|-------------|
| `AsyncHardExampleMiner` | class | Runs mining in a separate thread; feeds results back to training queue |

**`src/training/miner.py`** — Synchronous hard-negative miner.

| Symbol | Kind | Description |
|--------|------|-------------|
| `HardExampleMiner` | class | Scores dataset with current model, returns hardest negatives |

**`src/training/augmentation.py`** — Training-time audio augmentation.

| Symbol | Kind | Description |
|--------|------|-------------|
| `AudioAugmentationPipeline` | class | Chains audiomentations transforms for training |
| `ParallelAugmenter` | class | Thread-pool wrapper for batch augmentation |

**`src/training/tensorboard_logger.py`**

| Symbol | Kind | Description |
|--------|------|-------------|
| `TensorBoardLogger` | class | Logs scalars, histograms, PR curves to TensorBoard |

**`src/training/performance_optimizer.py`**, **`src/training/profiler.py`**, **`src/training/rich_logger.py`** — Performance and display utilities.

---

### `src/data/`

**`src/data/dataset.py`** — Core dataset abstraction.

| Symbol | Kind | Description |
|--------|------|-------------|
| `FeatureStore` / `FeatureStoreConfig` | class | Memory-mapped feature storage for fast epoch iteration |
| `RaggedMmap` / `RaggedMmapConfig` | class | Ragged array backed by memory-mapped file |
| `WakeWordDataset` | class | Loads positive/negative/hard_negative splits, applies oversampling |
| `create_feature_store(config)` | function | Builds or loads feature store from raw audio |
| `load_dataset(config)` | function | Returns `WakeWordDataset` ready for training |

**`src/data/features.py`** — Mel spectrogram feature extraction.

| Symbol | Kind | Description |
|--------|------|-------------|
| `MicroFrontend` | class | TF-based micro frontend matching ESPHome's feature extractor |
| `SpectrogramGeneration` | class | Batch spectrogram generation pipeline |
| `FeatureConfig` | dataclass | mel_bins, window_size, hop_size, sample_rate |
| `extract_features(audio, config)` | function | Audio -> mel spectrogram frames |
| `compute_mel_spectrogram(audio, config)` | function | Single-sample spectrogram |

**`src/data/augmentation.py`** — Data augmentation (GPU + CPU).

| Symbol | Kind | Description |
|--------|------|-------------|
| `AudioAugmentation` | class | Applies noise mixing, RIR convolution, gain |
| `AugmentationConfig` | dataclass | Per-augmentation probabilities and levels |
| `apply_spec_augment_gpu(spec, config)` | function | CuPy-based SpecAugment (time/freq masking) |

**`src/data/spec_augment_gpu.py`** / **`src/data/spec_augment_tf.py`** — SpecAugment implementations (GPU and TF fallback).

**`src/data/tfdata_pipeline.py`** — `tf.data` pipeline.

| Symbol | Kind | Description |
|--------|------|-------------|
| `OptimizedDataPipeline` | class | Prefetch, shuffle, batching pipeline |
| `PrefetchGenerator` | class | Background data generator |
| `create_optimized_dataset(config)` | function | Returns `tf.data.Dataset` for training |

**`src/data/preprocessing.py`** — Audio preprocessing utilities.

| Symbol | Kind | Description |
|--------|------|-------------|
| `SpeechPreprocessConfig` | dataclass | VAD, resampling, split parameters |
| `find_speech_boundaries(audio, sr)` | function | VAD-based start/end frame detection |
| `trim_speech_file(path, config)` | function | Trim silence from single WAV |
| `split_background_file(path, config)` | function | Splits long background audio into clips |
| `process_speech_directory(dir, config)` | function | Batch-process a directory of speech files |
| `scan_and_split(dir, config)` | function | Discover + split all audio in a directory tree |

**`src/data/ingestion.py`** — Data ingestion helpers.
**`src/data/hard_negatives.py`** — Hard negative sample management.
**`src/data/clustering.py`** — Speaker clustering (ECAPA-TDNN embeddings).
**`src/data/quality.py`** — Audio quality scoring and filtering.

---

### `src/evaluation/`

**`src/evaluation/metrics.py`** — Vectorized evaluation metrics.

| Symbol | Kind | Description |
|--------|------|-------------|
| `MetricsCalculator` | class | Aggregates predictions; computes accuracy, ROC-AUC, PR |
| `compute_accuracy(y_true, y_pred)` | function | Threshold-based accuracy |
| `compute_roc_auc(y_true, scores)` | function | ROC-AUC (with manual fallback if sklearn absent) |
| `compute_recall_at_no_faph(...)` | function | Recall at 0 false activations per hour |
| `compute_recall_at_target_fah(...)` | function | Recall at specified FAH budget |
| `compute_fah_at_target_recall(...)` | function | FAH at specified recall target |
| `compute_average_viable_recall(...)` | function | Mean recall across FAH=[0,1] range |
| `compute_all_metrics(...)` | function | Master function returning full metrics dict |

**`src/evaluation/test_evaluator.py`**

| Symbol | Kind | Description |
|--------|------|-------------|
| `TestEvaluator` | class | Runs model on held-out test set; computes MCC, Cohen's Kappa, EER |

**`src/evaluation/fah_estimator.py`**

| Symbol | Kind | Description |
|--------|------|-------------|
| `FAHEstimator` | class | Estimates false-activation-per-hour rate from streaming simulation |

**`src/evaluation/calibration.py`**

| Symbol | Kind | Description |
|--------|------|-------------|
| `compute_calibration_curve(...)` | function | Reliability diagram data |
| `compute_brier_score(...)` | function | Brier score for probability calibration |
| `calibrate_probabilities(...)` | function | Platt scaling / isotonic calibration |

---

### `src/export/`

**`src/export/tflite.py`** — Primary export pipeline.

| Symbol | Kind | Description |
|--------|------|-------------|
| `StreamingExportModel` | class | Wraps streaming model for TFLite conversion |
| `export_streaming_tflite(model, config)` | function | Converts streaming model to 2-subgraph TFLite |
| `create_representative_dataset(...)` | function | Calibration data generator for INT8 quantization |
| `verify_exported_model(path, config)` | function | Validates subgraphs, dtypes, shapes |
| `verify_esphome_compatibility(path)` | function | Strict ESPHome format check |
| `export_to_tflite(checkpoint, config)` | function | Full export: load weights -> stream -> quantize -> verify |
| `calculate_tensor_arena_size(path)` | function | Estimates tensor arena for manifest |
| `main()` | function | CLI entry point (`mww-export`) |

**`src/export/manifest.py`** — ESPHome manifest generation.

| Symbol | Kind | Description |
|--------|------|-------------|
| `generate_manifest(config, tflite_path)` | function | Produces `manifest.json` dict |
| `save_manifest(manifest, output_dir)` | function | Writes manifest.json to disk |
| `create_esphome_package(config, output_dir)` | function | Bundles .tflite + manifest |

**`src/export/model_analyzer.py`** — Model introspection utilities.
**`src/export/verification.py`** — Additional verification helpers.

---

### `src/tools/`

**`src/tools/cluster_analyze.py`** — Speaker clustering CLI.

| Symbol | Kind | Description |
|--------|------|-------------|
| `discover_audio_files(directory)` | function | Recursively finds WAV files |
| `analyze_clusters(files, config)` | function | Runs ECAPA-TDNN embedding + clustering |
| `save_namelist_json(clusters, path)` | function | Saves `file->speaker_id` mapping |
| `save_cluster_report(clusters, path)` | function | Saves human-readable report |
| `main()` | function | CLI entry (`mww-cluster-analyze`) |

**`src/tools/cluster_apply.py`** — Apply/undo file organization from cluster results.
CLI: `mww-cluster-apply [--namelist|--namelist-dir|--undo] [--dry-run]`

**`src/tools/mine_hard_negatives.py`** — Post-training hard-negative extraction.

| Symbol | Kind | Description |
|--------|------|-------------|
| `load_prediction_log(path)` | function | Parses training prediction log |
| `filter_epochs_by_min_epoch(log, n)` | function | Filters to mature model predictions |
| `collect_false_predictions(log, thresh)` | function | Selects false positives above threshold |
| `deduplicate_by_hash(files)` | function | Removes duplicate audio by content hash |
| `copy_files_to_mined_dir(files, dest)` | function | Copies mined files to hard_negative/ |
| `main()` | function | CLI entry (`mww-mine`) |

---

### `src/utils/`

**`src/utils/performance.py`** — GPU and threading setup.

| Symbol | Kind | Description |
|--------|------|-------------|
| `configure_tensorflow_gpu(memory_growth, memory_limit_mb)` | function | GPU memory configuration |
| `configure_mixed_precision(enabled)` | function | FP16 mixed precision toggle |
| `set_threading_config(inter_op, intra_op)` | function | TF threading parallelism |
| `setup_gpu_environment(config)` | function | Applies all GPU settings from `PerformanceConfig` |
| `check_gpu_available()` | function | Returns bool; logs GPU info |
| `check_gpu_and_cupy_available()` | function | Additional CuPy check |
| `get_system_info()` | function | Dict of CUDA version, GPU name, RAM |
| `format_bytes(n)` | function | Human-readable byte size string |

**`src/utils/terminal_logger.py`** — Rich console terminal logging.
**`src/utils/seed.py`** — Reproducibility seed setter.
**`src/utils/optional_deps.py`** — Graceful handling of optional imports.

---

### `src/pipeline.py`

Top-level orchestrator. Functions:

| Function | Description |
|----------|-------------|
| `step_train(config)` | Calls `train(config)` |
| `step_autotune(config, checkpoint)` | Iterative threshold/hyperparameter tuning |
| `step_export(config, checkpoint)` | Converts best checkpoint -> TFLite |
| `step_verify_esphome(tflite_path)` | Runs ESPHome compatibility checks |
| `step_verify_streaming(tflite_path)` | Smoke-tests streaming subgraph |
| `step_evaluate(config, tflite_path)` | Full test-set evaluation |
| `step_gate(metrics, config)` | Pass/fail gate on FAH/recall targets |
| `step_promote(src, dest)` | Copies to `models/` with manifest |
| `main()` | Parses args; dispatches pipeline steps |

---

## Configuration System

Configs are loaded in **layered priority** (later wins):

```
preset YAML  ->  override YAML  ->  environment variables
```

```python
from config.loader import load_full_config

# Load standard preset with custom overrides
config = load_full_config("standard", "my_config.yaml")

# Access sections
config.training.batch_size
config.model.first_conv_filters
config.export.wake_word
```

Environment variable substitution in YAML:
```yaml
paths:
  dataset_dir: ${DATASET_DIR:/home/user/dataset}
```

**13 config dataclasses** in `config/loader.py`:
`HardwareConfig`, `PathsConfig`, `TrainingConfig`, `ModelConfig`, `AugmentationConfig`,
`PerformanceConfig`, `SpeakerClusteringConfig`, `HardNegativeMiningConfig`, `ExportConfig`,
`PreprocessingConfig`, `QualityConfig`, `EvaluationConfig`, `FullConfig`

---

## Key Data Flows

### Training Flow
```
dataset/ audio files
  -> preprocessing.py  (VAD trim, 16kHz mono, split)
  -> features.py       (MicroFrontend -> 40-bin mel spectrogram)
  -> dataset.py        (FeatureStore mmap, WakeWordDataset)
  -> tfdata_pipeline.py (tf.data, prefetch, shuffle)
  -> augmentation.py + spec_augment_gpu.py (SpecAugment, noise mix, RIR)
  -> trainer.py        (2-phase training, hard-negative mining)
  -> checkpoints/*.weights.h5
```

### Export Flow
```
checkpoints/best_weights.weights.h5
  -> architecture.py   (MixedNet)
  -> streaming.py      (StreamingMixedNet + ring buffers)
  -> tflite.py         (INT8 quantization, 2-subgraph TFLite)
  -> manifest.py       (ESPHome manifest.json)
  -> models/exported/{name}.tflite + manifest.json
```

### Speaker Clustering Flow (PyTorch env)
```
dataset/positive/ audio
  -> cluster_analyze.py  (ECAPA-TDNN embeddings -> speaker clusters)
  -> cluster_output/{dataset}_namelist.json
  -> cluster_apply.py    (reorganize files into speaker subdirs)
```

---

## Tests

```
tests/
├── conftest.py                       # Shared fixtures
├── unit/
│   ├── test_async_miner.py           # AsyncHardExampleMiner
│   ├── test_config.py                # ConfigLoader & all dataclasses
│   ├── test_test_evaluator.py        # TestEvaluator
│   ├── test_vectorized_metrics.py    # MetricsCalculator functions
│   └── test_spec_augment_tf.py       # TF SpecAugment
└── integration/
    └── test_training.py              # End-to-end training smoke test
```

Run tests:
```bash
# Unit tests (fast, no GPU required)
pytest tests/unit/ -v

# Integration tests (GPU required)
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov=config tests/
```

---

## Documentation Map

| File | Contents |
|------|----------|
| `README.md` | User guide, quickstart, full workflow |
| `MASTER_GUIDE.md` | Extended operational guide |
| `ARCHITECTURAL_CONSTITUTION.md` | Immutable architectural decisions |
| `docs/ARCHITECTURE.md` | MixedNet architecture details |
| `docs/CONFIGURATION.md` | Complete config reference |
| `docs/TRAINING.md` | Training workflow and optimization |
| `docs/EXPORT.md` | TFLite export and ESPHome deployment |
| `specs/implementation_status.md` | Per-component implementation status |
| `AGENTS.md` | AI agent guidelines |
| `src/*/AGENTS.md` | Per-module agent guidelines |

---

*Generated 2026-03-06*
