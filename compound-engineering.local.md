---
review_agents: [kieran-python-reviewer, code-simplicity-reviewer, security-sentinel, performance-oracle]
plan_review_agents: [kieran-python-reviewer, code-simplicity-reviewer]
---

# Review Context

This is a GPU-accelerated wake word training framework for ESPHome. Key context for reviewers:

- **TensorFlow + CuPy pipeline**: Must run on GPU. No CPU fallback for SpecAugment.
- **Output dtype**: Must be uint8, not int8 — ESPHome will silently fail with int8.
- **Export**: Use `tf.keras.export.ExportArchive`, never `model.export()`.
- **Serialization**: Use `model.get_weights()`/`model.set_weights()`, not `model.trainable_weights`.
- **Two venvs**: TF (main) and PyTorch (clustering) — never mix.
- **Auto-tuner data split**: FocusedSampler trains on search_train (70%), evaluates on search_eval (30%) via `search_eval_fraction` config. Train-on-test contamination is a critical bug here.
- **State shapes**: stream_5 shape depends on temporal_frames derived from clip_duration_ms — do not hardcode.
- **ARCHITECTURAL_CONSTITUTION.md** is the supreme source of truth for all architectural constants.
