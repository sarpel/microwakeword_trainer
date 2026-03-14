# Scripts - Standalone Utilities

**Directory:** `scripts/`
**Purpose:** Standalone tools for dataset management, audio analysis, and model evaluation

## OVERVIEW
Collection of utility scripts for dataset preparation, audio quality analysis, and post-training evaluation. These are standalone tools invoked directly via Python, not part of the main training pipeline.

**Note**: Mining-related scripts (`extract_top5_fps.py`, `consolidate_predictions.py`, `consolidate_false_predictions.py`) have been consolidated into `src/training/mining.py`. Use `mww-mine-hard-negatives` CLI instead.

## STRUCTURE
```
scripts/
├── audio_analyzer.py                 # Audio file analysis tool (387 lines)
├── audio_similarity_detector.py      # Duplicate/similar audio detection (924 lines)
├── check_esphome_compat.py           # ESPHome compatibility checker with detailed diagnostics (684 lines)
├── cleanup_tfdata_cache.py           # Clean up TF.data cache tempstate files (115 lines)
├── compare_models.py                 # Compare two models side-by-side (362 lines)
├── count_audio_hours.py              # Count total audio hours in datasets (109 lines)
├── count_dataset.py                  # Dataset sample counter (114 lines)
├── debug_streaming_gap.py            # Debug streaming vs training model gap (219 lines)
├── eval_dashboard.py                 # Interactive HTML dashboard from evaluation_report.json
├── evaluate_model.py                 # Advanced post-training model evaluation + report artifacts
├── generate_test_dataset.py          # Synthetic dataset generator (190 lines)
├── phonetic_scorer.py                # Phonetic similarity scoring for hard negatives (683 lines)
├── score_quality_fast.py             # Fast audio quality scoring (72 lines)
├── score_quality_full.py             # Full audio quality scoring (82 lines)
├── split_audio.py                    # Audio splitting utility (59 lines)
├── tidy_yaml.py                      # Clean up and normalize YAML config files (246 lines)
├── vad_trim.py                       # VAD-based audio trimming (168 lines)
├── verify_esphome.py                 # TFLite ESPHome compatibility checker (168 lines)
└── verify_streaming.py               # Verify streaming TFLite model correctness (271 lines)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| ESPHome verification | `verify_esphome.py` | Check TFLite model compatibility |
| ESPHome detailed check | `check_esphome_compat.py` | Source-derived constraint validation, supports --json |
| Synthetic data | `generate_test_dataset.py` | Create test datasets quickly |
| Audio analysis | `audio_analyzer.py` | Analyze audio properties |
| Duplicate detection | `audio_similarity_detector.py` | Find similar/duplicate audio files |
| Dataset counting | `count_dataset.py` | Count samples per dataset |
| Audio duration | `count_audio_hours.py` | Calculate total ambient duration hours for config |
| Quality scoring | `score_quality_fast.py` / `score_quality_full.py` | Assess audio quality |
| Audio splitting | `split_audio.py` | Split long audio into clips |
| VAD trimming | `vad_trim.py` | Trim silence from audio |
| YAML cleanup | `tidy_yaml.py` | Normalize YAML config formatting |
| Cache cleanup | `cleanup_tfdata_cache.py` | Clean up TF.data cache tempstate files |
| Model comparison | `compare_models.py` | Side-by-side FAH/recall comparison of two models |
| Streaming debug | `debug_streaming_gap.py` | Diagnose streaming vs training model gap |
| Streaming verify | `verify_streaming.py` | Validate streaming TFLite model correctness |
| Phonetic scoring | `phonetic_scorer.py` | Score hard negatives by phonetic similarity |
| Model evaluation | `evaluate_model.py` | Comprehensive evaluation with report artifacts |
| Eval dashboard | `eval_dashboard.py` | Build interactive HTML dashboard from report |

## CONVENTIONS

### Script Organization
- **Standalone**: Each script is self-contained with its own main() function
- **CLI arguments**: Use argparse for command-line interface
- **Exit codes**: Use sys.exit() with non-zero codes for errors
- **Error handling**: Try-except blocks with user-friendly error messages

### Audio File Handling
- **Formats**: Primarily WAV, 16-bit PCM, 16kHz mono
- **Resampling**: Scripts handle resampling if sample rate differs
- **Validation**: Check file format, duration, channels before processing
- **Output**: Print progress updates, save results to file or stdout

### Quality Assessment
- **Fast scoring**: `score_quality_fast.py` uses lightweight metrics
- **Full scoring**: `score_quality_full.py` uses comprehensive metrics
- **Metrics**: SNR, clipping detection, spectral analysis, WQI score
- **Thresholds**: Recommended quality thresholds for dataset inclusion

### Exit Codes
All scripts follow this standard exit code contract:
- **0** = Success (operation completed, all checks passed)
- **2** = Validation/compatibility failure (e.g., model incompatible, file not found)
- **1** = Runtime/internal error (exception, unexpected failure)

This allows CI/CD pipelines to distinguish between "expected failure" (2) and "unexpected crash" (1).

## ANTI-PATTERNS (THIS DIRECTORY)

- **Don't modify dataset in place** - Always verify before batch operations
- **Don't assume correct audio format** - Validate WAV format, sample rate, channels
- **Don't skip error handling** - All scripts must handle file not found, invalid format
- **Don't silence output** - Print progress, especially for long-running operations
- **Don't overwrite without backup** - Always check before overwriting files
- **Don't try to import scripts as a module** - scripts/ has no __init__.py (intentional). Use subprocess or run directly.
  - Exception: `trainer.py` line 1614 wraps `from scripts.extract_top5_fps import run_extraction` in try/except (graceful degradation)
- **Don't assume uniform exit codes** - Some scripts use non-standard exit codes (e.g., `check_esphome_compat.py` returns 4 for incompatibility). Check each script's documentation.
## NOTES

### Critical Tools

**verify_esphome.py**
- Validates TFLite model for ESPHome compatibility
- Checks input/output shapes and dtypes
- Verifies operation registration (only ESPHome-registered ops allowed)
- **Usage**: `python scripts/verify_esphome.py models/exported/wake_word.tflite`

**audio_similarity_detector.py**
- Detects duplicate or highly similar audio files
- Uses MFCC feature extraction and cosine similarity
- Useful for removing redundant samples from dataset
- **Usage**: `python scripts/audio_similarity_detector.py /path/to/dataset/`

**generate_test_dataset.py**
- Creates synthetic test dataset for quick testing
- Generates positive, negative, hard_negative samples
- Useful for CI/CD and quick iteration
- **Usage**: `python scripts/generate_test_dataset.py --output-dir test_data/`

**check_esphome_compat.py**
- Detailed ESPHome compatibility checker with source-derived constraints
- Validates ops, tensor shapes, state variables, quantization
- Supports --json for CI/CD and --verbose for diagnostics
- **Usage**: `python scripts/check_esphome_compat.py models/exported/wake_word.tflite --verbose`

**verify_streaming.py**
- Validates streaming TFLite model correctness
- Tests determinism, state changes, boundary conditions
- **Usage**: `python scripts/verify_streaming.py models/exported/wake_word.tflite`

**phonetic_scorer.py**
- Scores hard negatives by phonetic similarity to wake word
- Uses IPA conversion + articulatory feature distance
- Supports moving high-risk files to quarantine
- **Usage**: `python scripts/phonetic_scorer.py score dataset/hard_negative/ --wake-word "Hey Katya"`

**compare_models.py**
- Compares two models side-by-side on same test data
- Supports .weights.h5 and .tflite formats
- Shows delta metrics and operating points
- **Usage**: `python scripts/compare_models.py model_a.tflite model_b.tflite --config standard`
### Dataset Preparation Workflow
1. **Count audio hours**: `python scripts/count_audio_hours.py --config standard`
2. **Count samples**: `python scripts/count_dataset.py dataset/`
3. **Analyze audio**: `python scripts/audio_analyzer.py dataset/positive/speaker_001/rec_001.wav`
4. **Score quality**: `python scripts/score_quality_fast.py dataset/positive/speaker_001/rec_001.wav`
5. **Detect duplicates**: `python scripts/audio_similarity_detector.py dataset/positive/`
6. **Phonetic scoring**: `python scripts/phonetic_scorer.py score dataset/hard_negative/ --wake-word "Hey Katya"`
7. **Trim silence**: `python scripts/vad_trim.py long_recording.wav --output-dir trimmed/`
8. **Split audio**: `python scripts/split_audio.py long_recording.wav --output-dir clips/`
9. **Tidy YAML configs**: `python scripts/tidy_yaml.py --config-dir config/ --apply`

### Model Evaluation Workflow
1. **Export model**: `mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/`
2. **Verify compatibility**: `python scripts/verify_esphome.py models/exported/wake_word.tflite`
3. **Detailed ESPHome check**: `python scripts/check_esphome_compat.py models/exported/wake_word.tflite --verbose`
4. **Verify streaming**: `python scripts/verify_streaming.py models/exported/wake_word.tflite`
5. **Evaluate**: `python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard --output-dir logs/`
6. **Interactive dashboard (optional)**: `python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json`
7. **Compare models (optional)**: `python scripts/compare_models.py model_a.tflite model_b.tflite --config standard`

### Post-Training Debugging Workflow
1. **Debug streaming gap**: `python scripts/debug_streaming_gap.py --checkpoint checkpoints/best_weights.weights.h5 --config standard`
2. **Cleanup TF cache**: `python scripts/cleanup_tfdata_cache.py --delete`

`evaluate_model.py` now writes a full artifact bundle in `evaluation_artifacts/` including:
- `evaluation_report.json` (machine-readable metrics)
- PNG plots (confusion matrix, ROC, PR, DET, calibration, threshold/cost curves)
- `executive_report.md` and `executive_report.html`

### Integration with Main Pipeline
- **Audio validation**: Used by `src/data/ingestion.py` for dataset loading
- **Quality scoring**: Integrated into `src/data/quality.py`
- **Verification**: Used by `src/export/verification.py`
- **Synthetic data**: Used by test suite in `tests/integration/`

### Dependencies
- **Audio processing**: scipy, librosa, soundfile
- **Feature extraction**: pymicro-features
- **Machine Learning**: numpy, tensorflow (for evaluation)
- **VAD**: webrtcvad (optional, for vad_trim.py)


(End of file - total 170 lines)