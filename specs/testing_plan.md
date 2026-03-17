# Testing Plan for microwakeword_trainer

**Last Updated**: 2026-03-16 (Post-Audit Test Expansion)
**Last Updated**: 2026-03-16 (Post-Audit Test Expansion)
**Project Version**: 2.1.0

## Overview

This document outlines the comprehensive testing strategy for microwakeword_trainer, covering unit tests, integration tests, manual testing procedures, and performance validation. The testing framework ensures code quality, correctness, and production readiness.

## Testing Philosophy

- **Test-Driven Development**: Write tests before implementation where possible
- **Continuous Testing**: Run tests on every commit via CI/CD
- **Coverage Targets**: Aim for >80% code coverage
- **Manual Validation**: Complement automated tests with manual verification
- **Real-World Testing**: Validate on actual ESPHome devices

---

## Test Suite Structure

```
tests/
├── conftest.py                       # Shared fixtures and configuration
├── unit/                             # Unit tests (fast, isolated)
│   ├── test_async_miner.py           # AsyncHardExampleMiner
│   ├── test_config.py                # ConfigLoader and dataclasses
│   ├── test_test_evaluator.py        # TestEvaluator
│   ├── test_vectorized_metrics.py    # MetricsCalculator
│   ├── test_spec_augment_tf.py       # TF SpecAugment
│   ├── test_tuning_autotuner_components.py  # Auto-tuner components (8 tests)
│   ├── test_export_op_whitelist.py   # ESPHome op registration (20 ops) (NEW)
│   └── test_scripts_exit_codes.py     # Verify script exit code contract (NEW)
└── integration/                       # Integration tests (slow, end-to-end)
    └── test_training.py              # Full training pipeline
```

---

## 1. Unit Tests

### 1.1 Async Hard Negative Miner (`test_async_miner.py`)

**Purpose**: Test background hard negative mining functionality

**Test Cases**:
- ✅ Test miner initialization with valid config
- ✅ Test queue population with false predictions
- ✅ Test graceful shutdown on training completion
- ✅ Test thread safety with concurrent access
- ✅ Test error handling for invalid predictions

**Fixtures**:
- Mock dataset with labels
- Mock model with predictions
- Test configuration

**Expected Coverage**:
- `src/training/async_miner.py`: >80%

**Run Command**:
```bash
pytest tests/unit/test_async_miner.py -v
```

---

### 1.2 Configuration Loader (`test_config.py`)

**Purpose**: Test configuration loading, validation, and merging

**Test Cases**:
- ✅ Test preset loading (fast_test, standard, max_quality)
- ✅ Test environment variable substitution
- ✅ Test override merging (preset + override)
- ✅ Test validation rules (sample rate, stride, dtypes)
- ✅ Test invalid configuration handling
- ✅ Test dataclass field constraints

**Fixtures**:
- Valid YAML files for all presets
- Override configurations
- Invalid configurations for error testing

**Expected Coverage**:
- `config/loader.py`: >90%

**Run Command**:
```bash
pytest tests/unit/test_config.py -v
```

**Edge Cases**:
- Missing required fields
- Invalid data types
- Out-of-range values
- Invalid paths
- Conflicting overrides

---

### 1.3 Test Evaluator (`test_test_evaluator.py`)

**Purpose**: Test comprehensive test set evaluation

**Test Cases**:
- ✅ Test initialization with model and dataset
- ✅ Test FAH calculation with varying durations
- ✅ Test recall at target FAH
- ✅ Test ROC-AUC computation
- ✅ Test PR-AUC computation
- ✅ Test MCC and Cohen's Kappa
- ✅ Test EER calculation
- ✅ Test latency measurement

**Fixtures**:
- Mock model with predictions
- Mock dataset with ground truth
- Test configuration

**Expected Coverage**:
- `src/evaluation/test_evaluator.py`: >85%

**Run Command**:
```bash
pytest tests/unit/test_test_evaluator.py -v
```

**Validation**:
- Compare metrics with manual calculations
- Verify FAH formula correctness
- Check threshold behavior at boundaries

---

### 1.4 Vectorized Metrics (`test_vectorized_metrics.py`)

**Purpose**: Test vectorized metrics calculation

**Test Cases**:
- ✅ Test accuracy calculation with threshold
- ✅ Test ROC-AUC with binary predictions
- ✅ Test PR-AUC with precision/recall
- ✅ Test recall at specific FAH
- ✅ Test FAH at specific recall
- ✅ Test average viable recall
- ✅ Test Brier score calculation
- ✅ Test vectorization speedup

**Fixtures**:
- Prediction arrays (float)
- Label arrays (int)
- Threshold arrays

**Expected Coverage**:
- `src/evaluation/metrics.py`: >80%

**Run Command**:
```bash
pytest tests/unit/test_vectorized_metrics.py -v
```

**Performance**:
- Measure vectorization speedup vs. loop-based calculation
- Verify NumPy vectorization is effective

---

### 1.5 SpecAugment TF (`test_spec_augment_tf.py`)

**Purpose**: Test TensorFlow-based SpecAugment

**Test Cases**:
- ✅ Test time masking with random parameters
- ✅ Test frequency masking with random parameters
- ✅ Test combined time+frequency masking
- ✅ Test mask size limits
- ✅ Test mask count limits
- ✅ Test invalid mask sizes (should fail or clip)

**Fixtures**:
- Mock spectrograms [batch, time, mel_bins]
- SpecAugment configuration

**Expected Coverage**:
- `src/data/spec_augment_tf.py`: >75%

**Run Command**:
```bash
pytest tests/unit/test_spec_augment_tf.py -v
```

**Note**: GPU SpecAugment (`spec_augment_gpu.py`) is tested manually due to CuPy requirements

---

### 1.6 Auto-Tuner Components (`test_tuning_autotuner_components.py`)

**Purpose**: Test auto-tuner search data partitioning and threshold re-optimization

**Test Cases** (8 total):

**TestPartitionSearchSplit**:
- `test_basic_split_ratio` — verifies 70/30 split respects search_eval_fraction
- `test_group_aware_splitting` — ensures no group leaks across train/eval
- `test_no_overlap` — confirms zero index overlap between search_train and search_eval

**TestErrorMemoryOffset**:
- `test_offset_indices_match_eval` — verifies ErrorMemory indices map to search_eval
- `test_offset_preserves_ordering` — confirms relative ordering preserved after offset

**TestConfirmationReoptimize**:
- `test_reoptimize_threshold_differs` — confirms threshold re-optimized on confirm data differs from search threshold
- `test_reoptimize_uses_confirm_data` — verifies confirm data (not search data) is used
- `test_reoptimize_improves_metric` — confirms re-optimized threshold improves metric on confirm data

**Fixtures**:
- Mock search dataset with group labels
- Mock config with search_eval_fraction

**Expected Coverage**:
- `src/tuning/autotuner.py`: >85% for `_partition_data()` and confirmation phase

**Run Command**:
```bash
pytest tests/unit/test_tuning_autotuner_components.py -v
```

---

## 2. Integration Tests

### 2.1 Training Pipeline (`test_training.py`)

**Purpose**: Test end-to-end training workflow

**Test Cases**:
- ✅ Test full training with minimal dataset
- ✅ Test checkpoint saving and loading
- ✅ Test TensorBoard logging
- ✅ Test hard negative mining integration
- ✅ Test two-phase training completion
- ✅ Test model export readiness

**Fixtures**:
- Minimal synthetic dataset (100 samples)
- Test configuration (fast_test preset)
- Temporary directories for outputs

**Expected Duration**: 10-15 minutes

**Expected Coverage**:
- `src/training/trainer.py`: >70%
- Integration across multiple modules

**Run Command**:
```bash
pytest tests/integration/test_training.py -v
```

**Validation**:
- Verify checkpoint files exist
- Verify TensorBoard logs created
- Verify metrics are logged
- Verify best model is saved

---

## 3. Manual Testing Procedures

### 3.1 ESPHome Compatibility Testing

**Purpose**: Verify exported models work on real ESPHome devices

**Procedure**:
1. Train model with standard preset
2. Export model with `mww-export`
3. Run `scripts/verify_esphome.py` on exported `.tflite` (both human-readable and JSON modes)
4. Flash model to ESP32 device
5. Test wake word detection with various audio samples
6. Verify probability threshold behavior

**Devices**:
- ESP32-S3-BOX3
- M5Stack Atom Echo
- Custom ESP32 + INMP441

**Test Samples**:
- Positive samples (wake word)
- Negative samples (background speech)
- Hard negatives (similar words)
- Edge cases (quiet, noisy)

**Validation Criteria**:
- ✅ Model loads on device without errors
- ✅ Wake word triggers on positive samples
- ✅ No false triggers on negative samples
- ✅ Probability cutoff works as expected
- ✅ Memory usage within tensor arena

**Recommended local verification commands**:
```bash
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
python scripts/verify_esphome.py models/exported/wake_word.tflite --json
```

---

### 3.2 Dataset Quality Testing

**Purpose**: Validate dataset preparation and quality

**Procedure**:
1. Run speaker clustering analysis
2. Review cluster reports for correctness
3. Apply cluster organization
4. Run quality scoring on all samples
5. Remove low-quality samples
6. Verify dataset balance

**Tools**:
- `mww-cluster-analyze`
- `mww-cluster-apply`
- `python scripts/score_quality_fast.py`
- `python scripts/score_quality_full.py`

**Validation Criteria**:
- ✅ Speakers grouped correctly (no cross-speaker clusters)
- ✅ No train/test leakage
- ✅ Quality scores above threshold
- ✅ Balanced class distribution (positive:negative = 1:10+)
- ✅ No corrupted or invalid audio files

---

### 3.3 Performance Benchmarking

**Purpose**: Measure training and inference performance

**Procedure**:
1. Run training with profiling enabled
2. Measure time per epoch
3. Record GPU memory usage
4. Benchmark SpecAugment (CPU vs GPU)
5. Measure export time
6. Test inference latency on device

**Metrics to Capture**:
- Steps per second
- GPU memory usage
- CPU utilization
- SpecAugment speedup
- Export time
- Inference latency on device

**Target Benchmarks**:
- Training: ~8 hours for standard preset
- SpecAugment: 5-10x faster than CPU
- Export: <5 minutes for standard model
- Inference: <30ms per inference call

---

### 3.4 Auto-Tuning Validation

**Purpose**: Test auto-tuning effectiveness

**Procedure**:
1. Train baseline model
2. Export baseline TFLite
3. Run `mww-autotune` with baseline
4. Evaluate tuned model on test set
5. Compare FAH/recall metrics

**Validation Criteria**:
- ✅ Auto-tune completes without errors
- ✅ Tuned model meets target FAH (<0.5)
- ✅ Tuned model maintains recall (>0.90)
- ✅ Improvement over baseline (FAH reduction)
- ✅ User-defined hard negatives used (if provided)

### 3.5 Architecture Alignment Verification

**Purpose**: Verify that model architecture matches official okay_nabu TFLite model to eliminate AUC gap between training and export.

**Background**:
- Identified issue: architectural mismatch between training and export caused ~15% AUC gap
- Training and export now use consistent temporal pooling approach

**Procedural Test Cases**:
- Verify training model uses `Flatten` layer for temporal pooling (not `GlobalAveragePooling2D`)
- Verify export model uses `tf.reshape` (not `tf.reduce_mean`) for temporal pooling
- Verify Dense layer receives correct input shape (temporal_rb_size_plus_1 * last_pointwise_filters)
- Run numerical comparison script `scripts/debug_streaming_gap.py` after any architecture change
- Verify BatchNorm folding is numerically perfect (mean abs diff < 1e-6)

**Test Execution**:
```bash
# Build training model and verify architecture
python3 -c "from src.model.architecture import build_model; print(build_model((100, 40, 1)).summary())"

# Build export model and verify architecture
python3 scripts/debug_streaming_gap.py

# Check for stale GlobalAveragePooling references
grep -r "GlobalAveragePooling" src/model/architecture.py src/export/tflite.py
```

**Additional Critical Validation Steps**:
```bash
# 1. Verify ESPHome compatibility using verify_esphome_compatibility() from src/export/tflite.py
python3 -c "from src.export.tflite import verify_esphome_compatibility; print(verify_esphome_compatibility('models/exported/hey_katya.tflite'))"

# 2. Run analyze_model() from src/export/model_analyzer.py before deployment
python3 -c "from src.export.model_analyzer import analyze_model; import json; print(json.dumps(analyze_model('models/exported/hey_katya.tflite'), indent=2))"

# 3. Validate TFLite input dimensions (must be 3D per VALID_TFLITE_INPUT_SHAPES; reject 4D shapes)
python3 -c "
import tensorflow as tf
import numpy as np
interp = tf.lite.Interpreter('models/exported/hey_katya.tflite')
interp.allocate_tensors()
input_details = interp.get_input_details()
main_input = input_details[0]
assert len(main_input['shape']) == 3, f'Expected 3D input, got {main_input[\"shape\"]}'
print('Input dims OK:', main_input['shape'])
"

# 4. Assert TFLite output dtype is uint8 (kTfLiteUInt8), NOT int8
python3 -c "
import tensorflow as tf
import numpy as np
interp = tf.lite.Interpreter('models/exported/hey_katya.tflite')
interp.allocate_tensors()
output = interp.get_output_details()[0]
assert output['dtype'] == np.uint8, f'Expected uint8 output, got {output[\"dtype\"]}'
print('Output dtype OK: uint8')
"
```

> **⚠️ Review Requirement**: Before changing any code in `src/model/` or `src/export/`, open and read `docs/ARCHITECTURE.md` to ensure changes remain compliant with the reference architecture. Failing any of the above checks (`verify_esphome_compatibility`, `analyze_model`, input dims == 3D, output dtype == uint8) must produce a clear error message referencing the specific function names (`verify_esphome_compatibility`, `analyze_model`, `VALID_TFLITE_INPUT_SHAPES`).

**Expected Outcomes**:
- Training model uses `Flatten` layer (confirm in architecture summary)
- Export model uses `tf.reshape` for temporal pooling (confirm in script output)
- Dense input shape matches between training and export
- No `GlobalAveragePooling` references in codebase
- BN folding diff < 1e-6 (numerical precision maintained)
- All state variables present in export model
- `verify_esphome_compatibility()` passes without errors
- `analyze_model()` shows valid=True, 6 streaming state tensors
- Input dims size == 3 (3D shape, not 4D)
- Output dtype == uint8

---

## 4. Regression Testing

### 4.1 Pre-Commit Checks

**Purpose**: Catch regressions before commit

**Checks**:
- ✅ Run all unit tests (`pytest tests/unit/`)
- ✅ Run linting (if applicable)
- ✅ Check documentation consistency
- ✅ Verify no debug prints in production code
- ✅ Check for TODO/FIXME comments

**Automation**:
```bash
#!/bin/bash
# Pre-commit hook
pytest tests/unit/ -q --tb=no
if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi
```

---

### 4.2 Post-Merge Validation

**Purpose**: Validate merged code in integration

**Checks**:
- ✅ Run all tests (`pytest tests/`)
- ✅ Run integration tests with GPU
- ✅ Manual smoke test with small dataset
- ✅ Verify documentation builds
- ✅ Check for breaking changes

**Procedure**:
1. Create feature branch
2. Implement changes with tests
3. Run unit tests locally
4. Push and create PR
5. CI runs all tests
6. Manual review and merge
7. Post-merge validation

---

## 5. Performance Testing

### 5.1 Load Testing

**Purpose**: Test system limits and scaling

**Tests**:
- Large dataset (100,000+ samples)
- Long training (24+ hours)
- Maximum batch size (VRAM limit)
- Maximum augmentation intensity

**Metrics**:
- Training time vs dataset size
- Memory usage vs batch size
- GPU utilization
- Stability under load

**Validation**:
- ✅ No OOM errors
- ✅ Stable training (no crashes)
- ✅ Reasonable performance degradation

---

### 5.2 Stress Testing

**Purpose**: Identify edge cases and failure modes

**Tests**:
- Invalid audio files (corrupted, wrong format)
- Extreme audio durations (very short, very long)
- Invalid configurations (all invalid values)
- Missing datasets (empty directories)
- Concurrent training runs

**Expected Behavior**:
- Graceful error messages
- No silent failures
- Helpful error recovery suggestions
- Clean resource cleanup on failure

---

## 6. Continuous Testing

### 6.1 CI/CD Integration

**Purpose**: Automated testing on every commit

**CI Pipeline**:
```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt pytest pytest-cov
      - name: Run unit tests
        run: pytest tests/unit/ --cov=src --cov=config
      - name: Run integration tests
        run: pytest tests/integration/
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

**Frequency**: Every push and pull request

**Coverage Tracking**: Codecov or similar service

---

### 6.2 Nightly Builds

**Purpose**: Long-running tests and stability validation

**Schedule**: Daily (00:00 UTC)

**Tests**:
- Full training with standard preset
- Auto-tuning on multiple models
- Export verification on multiple devices
- Performance regression testing

**Alerting**:
- Email on test failures
- Slack notification on critical failures
- Jira ticket creation for bugs

---

## 7. Test Data Management

### 7.1 Test Datasets

**Minimal Dataset** (for fast testing):
- 100 positive samples
- 500 negative samples
- 50 hard negatives
- 100 background samples
- Total: ~750 samples

**Standard Dataset** (for integration testing):
- 1,000 positive samples
- 5,000 negative samples
- 500 hard negatives
- 1,000 background samples
- Total: ~7,500 samples

**Synthetic Dataset** (for unit tests):
- Generated with `scripts/generate_test_dataset.py`
- Controlled properties (duration, SNR, etc.)
- Deterministic for reproducible tests

### 7.2 Test Data Isolation

**Requirements**:
- Separate test data directory
- No overlap with training data
- Clean state for each test run
- Automatic cleanup after tests

**Implementation**:
```python
# conftest.py
@pytest.fixture
def test_data_dir(tmp_path):
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    # Copy test data to temp directory
    yield data_dir
    # Cleanup after test
    shutil.rmtree(data_dir)
```

---

## 8. Coverage Goals

### 8.1 Target Coverage

| Module | Target | Current |
|--------|--------|----------|
| `config/loader.py` | >90% | ✅ Achieved |
| `src/training/trainer.py` | >70% | ✅ Achieved |
| `src/data/dataset.py` | >75% | 🔄 In Progress |
| `src/evaluation/metrics.py` | >80% | ✅ Achieved |
| `src/export/tflite.py` | >70% | 🔄 In Progress |

**Overall Target**: >80% code coverage

### 8.2 Coverage Tracking

**Commands**:
```bash
# Generate coverage report
pytest --cov=src --cov=config --cov-report=html tests/

# Check coverage in terminal
pytest --cov=src --cov=config tests/ --cov-report=term-missing

# Upload to coverage service
pytest --cov=src tests/ --cov-report=xml
codecov -f coverage.xml
```

**Review**: Weekly coverage reviews to identify gaps

---

## 9. Test Maintenance

### 9.1 Test Hygiene

**Guidelines**:
- Keep tests independent (no test-to-test dependencies)
- Use descriptive test names
- Add docstrings explaining test purpose
- Update tests when implementation changes
- Remove obsolete tests
- Refactor common test logic into fixtures

### 9.2 Test Performance

**Goals**:
- Unit tests: <5 seconds per module
- Integration tests: <15 minutes total
- CI pipeline: <10 minutes total

**Monitoring**:
- Track test execution time
- Identify slow tests
- Optimize or mark as slow

---

## 10. Bug Reporting and Validation

### 10.1 Bug Report Template

```
**Description**: Clear description of issue
**Steps to Reproduce**:
1. Step 1
2. Step 2
3. ...
**Expected Behavior**: What should happen
**Actual Behavior**: What actually happens
**Environment**:
- Python version
- TensorFlow version
- CUDA version
- GPU model
- Configuration preset
**Logs**: Relevant error messages or stack traces
**Test Case**: Specific test that fails (if applicable)
```

### 10.2 Validation Checklist

Before marking issue as resolved, verify:
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] Documentation updated
- [ ] Regression tests passed
- [ ] Edge cases covered

---

## Summary

This testing plan provides comprehensive coverage for microwakeword_trainer:

**Unit Tests**: 8 test modules, 475+ tests passing
**Integration Tests**: Full training pipeline validation
**Manual Testing**: ESPHome compatibility, performance, auto-tuning
**Continuous Testing**: CI/CD pipeline, nightly builds
**Coverage Goals**: >80% code coverage

**Next Steps**:
1. Implement additional unit tests for uncovered modules
2. Add more integration tests for edge cases
3. Set up CI/CD pipeline with automated testing
4. Establish nightly builds for stability validation
5. Track and improve coverage over time

---

**Testing Philosophy Summary**:
- Quality over quantity (meaningful tests > many trivial tests)
- Automation over manual (CI/CD > ad-hoc testing)
- Prevention over reaction (catch bugs early > fix bugs later)
- Real-world validation (device testing > simulation only)
