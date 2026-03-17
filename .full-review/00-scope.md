# Review Scope

## Target

Comprehensive review of the microwakeword_trainer codebase - a TensorFlow-based wake word training pipeline with auto-tuning capabilities.

## Files

### Core Source (54 Python files)
**Training Pipeline:**
- `src/training/trainer.py` - Main training loop, checkpointing, evaluation
- `src/training/mining.py` - Hard-negative mining implementation
- `src/training/profiler.py` - Training performance profiling
- `src/training/tensorboard_logger.py` - TensorBoard integration
- `src/training/rich_logger.py` - Rich terminal logging
- `src/training/augmentation.py` - Training data augmentation
- `src/training/performance_optimizer.py` - Performance optimizations

**Auto-Tuning:**
- `src/tuning/orchestrator.py` - MicroAutoTuner main orchestration
- `src/tuning/knobs.py` - Hyperparameter knobs (LR, weight perturbation, label smoothing)
- `src/tuning/population.py` - Candidate population management
- `src/tuning/metrics.py` - Threshold optimization
- `src/tuning/dashboard.py` - Tuning dashboard
- `src/tuning/cli.py` - Tuning CLI interface

**Data Pipeline:**
- `src/data/dataset.py` - WakeWordDataset, RaggedMmap, FeatureStore
- `src/data/tfdata_pipeline.py` - TensorFlow data pipeline
- `src/data/clustering.py` - Audio clustering with embeddings
- `src/data/ingestion.py` - Data ingestion utilities
- `src/data/features.py` - Feature extraction
- `src/data/preprocessing.py` - Audio preprocessing
- `src/data/augmentation.py` - Data augmentation
- `src/data/quality.py` - Data quality checks
- `src/data/spec_augment_tf.py` - TensorFlow SpecAugment
- `src/data/spec_augment_gpu.py` - CuPy SpecAugment (GPU)

**Model:**
- `src/model/architecture.py` - CNN model architecture
- `src/model/streaming.py` - Streaming inference support

**Evaluation:**
- `src/evaluation/test_evaluator.py` - Test evaluation logic
- `src/evaluation/metrics.py` - Metric calculations
- `src/evaluation/fah_estimator.py` - False Accepts per Hour
- `src/evaluation/calibration.py` - Model calibration

**Export:**
- `src/export/tflite.py` - TFLite model export
- `src/export/tflite_utils.py` - TFLite utilities
- `src/export/verification.py` - Model verification
- `src/export/model_analyzer.py` - Model analysis
- `src/export/manifest.py` - Export manifest

**Orchestration:**
- `src/pipeline.py` - End-to-end pipeline orchestration

**Utilities:**
- `src/utils/performance.py` - GPU/performance setup
- `src/utils/logging_config.py` - Logging configuration
- `src/utils/terminal_logger.py` - Terminal logging
- `src/utils/seed.py` - Random seed management
- `src/utils/optional_deps.py` - Optional dependency handling

**Tools:**
- `src/tools/cluster_apply.py` - Cluster operations
- `src/tools/cluster_analyze.py` - Cluster analysis
- `src/tools/help_panel.py` - Help display

## Flags

- Security Focus: No
- Performance Critical: Yes (ML training pipeline)
- Strict Mode: No
- Framework: TensorFlow 2.x, Python 3.10+

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
