# Tech Stack: microwakeword_trainer

## Core Languages
- **Python (3.10/3.11):** Chosen for its extensive ecosystem of deep learning and audio processing libraries.

## Machine Learning & Deep Learning
- **TensorFlow (2.1x+):** Used for the main training loop, model architecture implementation (Keras), and model export (TFLite).
- **PyTorch (2.x):** Specifically used for speaker clustering and embedding extraction (ECAPA-TDNN) via SpeechBrain.
- **TensorFlow Lite Micro:** The target format for deployment on ESP32 microcontrollers.

## GPU Acceleration
- **CUDA (12.x):** Required for high-performance training on NVIDIA GPUs.
- **CuPy:** Used to implement GPU-accelerated audio augmentation (SpecAugment), bypassing the limitations of CPU-bound processing.

## Audio Processing
- **libsndfile:** For robust WAV file reading and writing.
- **FFmpeg:** For audio format conversion and preprocessing.

## Tooling & Utilities
- **ESPHome:** The target platform for deployment, ensuring that all models are compatible with its `micro_wake_word` component.
- **Rich:** For enhanced CLI output and progress reporting.
- **Pytest:** Used for unit and integration testing of the training pipeline.
- **Ruff/Mypy:** For static analysis and linting.

## Target Platform
- **ESP32-S3:** Primary target device due to its vector instructions (AI instructions) that significantly accelerate TFLite INT8 inference.
- **ESP32/ESP32-C3:** Supported as secondary targets.
