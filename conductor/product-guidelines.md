# Product Guidelines: microwakeword_trainer

## UX and CLI Principles
- **Clarity:** CLI tools should provide clear, actionable feedback. Use progress bars (via `rich` or `tqdm`) for long-running processes like training or data processing.
- **Fail-Fast:** Validate hardware (GPU/CUDA/CuPy) and configuration immediately before starting resource-intensive operations.
- **Automation:** Minimize manual steps. Quantization, manifest generation, and compatibility checks should be part of the default export flow.
- **Predictability:** Use standard, descriptive names for CLI commands (e.g., `mww-train`, `mww-export`, `mww-verify`).

## Documentation and Prose Style
- **Educational:** Explain *why* certain steps are necessary (e.g., speaker clustering to prevent data leakage).
- **Tutorial-First:** Documentation should prioritize the "Getting Started" experience, followed by deep dives into configuration.
- **Tone:** Professional, technical, yet encouraging for developers of all skill levels.
- **Formatting:** Use consistent Markdown with clear tables for configuration options and code blocks for command examples.

## Code and Architectural Principles
- **GPU-First:** The framework is built for GPU acceleration. Ensure that CPU fallbacks are clearly labeled (if available) or that hardware requirements are strictly enforced.
- **Modular Pipeline:** Maintain a clear separation between data ingestion, model architecture, training logic, and export/verification.
- **ESPHome Alignment:** The final output (TFLite models and manifests) must strictly adhere to the standards defined by the ESPHome `micro_wake_word` component.

## Error Handling and Validation
- **Hardware Checks:** Provide detailed error messages when CUDA or CuPy is missing, including links to installation guides.
- **Dataset Validation:** Automatically check for common dataset issues (e.g., empty directories, incorrect audio formats) before training begins.
- **Compatibility Verification:** The `mww-verify` script is mandatory for all exported models to ensure they function on the target ESP32 hardware.
