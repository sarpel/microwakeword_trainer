# Scripts - Standalone Utilities

**Directory:** `scripts/`
**Purpose:** Standalone tools for dataset management, audio analysis, and model evaluation

## OVERVIEW
Collection of utility scripts for dataset preparation, audio quality analysis, and post-training evaluation. These are standalone tools invoked directly via Python, not part of the main training pipeline.

**Note**: Mining-related scripts (`extract_top5_fps.py`, `consolidate_predictions.py`, `consolidate_false_predictions.py`) have been consolidated into `src/training/mining.py`. Use `mww-mine-hard-negatives` CLI instead.

## STRUCTURE
```
scripts/
├── verify_esphome.py              # TFLite ESPHome compatibility checker (168 lines)
├── generate_test_dataset.py         # Synthetic dataset generator (190 lines)
├── evaluate_model.py               # Advanced post-training model evaluation + report artifacts
├── eval_dashboard.py               # Interactive HTML dashboard from evaluation_report.json
├── audio_analyzer.py              # Audio file analysis tool (387 lines)
├── audio_similarity_detector.py     # Duplicate/similar audio detection (924 lines)
├── count_dataset.py               # Dataset sample counter (114 lines)
├── score_quality_fast.py          # Fast audio quality scoring (72 lines)
├── score_quality_full.py          # Full audio quality scoring (82 lines)
├── split_audio.py                # Audio splitting utility (59 lines)
└── vad_trim.py                   # VAD-based audio trimming (168 lines)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| ESPHome verification | `verify_esphome.py` | Check TFLite model compatibility |
| Synthetic data | `generate_test_dataset.py` | Create test datasets quickly |
| Audio analysis | `audio_analyzer.py` | Analyze audio properties |
| Duplicate detection | `audio_similarity_detector.py` | Find similar/duplicate audio files |
| Dataset counting | `count_dataset.py` | Count samples per dataset |
| Quality scoring | `score_quality_fast.py` / `score_quality_full.py` | Assess audio quality |
| Audio splitting | `split_audio.py` | Split long audio into clips |
| VAD trimming | `vad_trim.py` | Trim silence from audio |

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

## ANTI-PATTERNS (THIS DIRECTORY)

- **Don't modify dataset in place** - Always verify before batch operations
- **Don't assume correct audio format** - Validate WAV format, sample rate, channels
- **Don't skip error handling** - All scripts must handle file not found, invalid format
- **Don't silence output** - Print progress, especially for long-running operations
- **Don't overwrite without backup** - Always check before overwriting files

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

### Dataset Preparation Workflow
1. **Count samples**: `python scripts/count_dataset.py dataset/`
2. **Analyze audio**: `python scripts/audio_analyzer.py dataset/positive/speaker_001/rec_001.wav`
3. **Score quality**: `python scripts/score_quality_fast.py dataset/positive/speaker_001/rec_001.wav`
4. **Detect duplicates**: `python scripts/audio_similarity_detector.py dataset/positive/`
5. **Trim silence**: `python scripts/vad_trim.py long_recording.wav --output-dir trimmed/`
6. **Split audio**: `python scripts/split_audio.py long_recording.wav --output-dir clips/`

### Model Evaluation Workflow
1. **Export model**: `mww-export --checkpoint checkpoints/best_weights.weights.h5 --output models/exported/`
2. **Verify compatibility**: `python scripts/verify_esphome.py models/exported/wake_word.tflite`
3. **Evaluate**: `python scripts/evaluate_model.py --model models/exported/wake_word.tflite --config standard --output-dir logs/`
4. **Interactive dashboard (optional)**: `python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json`

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

### Anti-Patterns
- **Don't try to import scripts as a module** - scripts/ has no __init__.py (intentional). Use subprocess or run directly.
  - Exception: `trainer.py` line 1614 wraps `from scripts.extract_top5_fps import run_extraction` in try/except (graceful degradation)
- **Don't modify dataset in place** - Always copy or move with backup
- **Don't overwrite without backup** - Use `--dry-run` first, then execute
- **Don't add __init__.py to scripts/** - Scripts are standalone utilities, not a module
