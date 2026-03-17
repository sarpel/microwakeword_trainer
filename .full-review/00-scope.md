# Review Scope

## Target

Full codebase review of `microwakeword_trainer` — a Python/TensorFlow ML training pipeline for microcontroller wake-word detection models. The project trains TFLite models deployable on ESPHome/ESP32 microcontrollers.

## Files

### Core Source (src/)
- src/pipeline.py
- src/data/dataset.py, augmentation.py, clustering.py, features.py, ingestion.py, preprocessing.py, quality.py, spec_augment_gpu.py, spec_augment_tf.py, tfdata_pipeline.py
- src/evaluation/calibration.py, fah_estimator.py, metrics.py, test_evaluator.py
- src/export/manifest.py, model_analyzer.py, tflite.py, tflite_utils.py, verification.py
- src/model/architecture.py, streaming.py
- src/training/augmentation.py, mining.py, performance_optimizer.py, profiler.py, rich_logger.py, tensorboard_logger.py, trainer.py
- src/tuning/__init__.py, cli.py, dashboard.py, knobs.py, metrics.py, orchestrator.py, population.py
- src/utils/logging_config.py, optional_deps.py, performance.py, performance_monitor.py, seed.py, terminal_logger.py
- src/tools/cluster_analyze.py, cluster_apply.py, help_panel.py

### Scripts (scripts/)
- evaluate_model.py, generate_test_dataset.py, verify_esphome.py, count_audio_hours.py (and ~15 others)

### Tests (tests/)
- ~45 test files across unit/ and integration/

### Config & Docs
- pyproject.toml, requirements*.txt, setup.py
- ARCHITECTURAL_CONSTITUTION.md, MASTER_GUIDE.md, README.md, AGENTS.md
- specs/, docs/

## Flags

- Security Focus: no
- Performance Critical: no
- Strict Mode: no
- Framework: TensorFlow/Python ML training pipeline

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
