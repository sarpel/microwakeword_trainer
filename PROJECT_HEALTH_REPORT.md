# Project Health Report: microwakeword_trainer v2.0

## Executive Summary

**Score: 0/100** - Critical Failure Risk

This project suffers from fundamental disconnects between documentation and implementation. Six critical issues render core functionality broken or misleading, including mathematically incorrect metrics, fake architecture selection, and broken export compatibility. The codebase cannot reliably train or deploy wake word models as advertised. Immediate intervention required before any production use.

---

## Critical Issues (P0 - Fix Immediately)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **Evaluation modules are DEAD CODE** - metrics.py, calibration.py, fah_estimator.py exist but are never imported | `src/evaluation/*.py` | All claimed evaluation features are disconnected; trainer uses internal TrainingMetrics class instead |
| 2 | **FAH calculation is MATHEMATICALLY WRONG** - Uses true negative count as ambient duration hours | `src/training/trainer.py:387,436` | Checkpoints selected based on garbage metric; formula is `fp / max(metrics.get("tn", 1), 1) * 1000` |
| 3 | **ai-edge-litert is NOT USED** - Listed in requirements but never imported | `requirements.txt` | False dependency claim; export uses tf.lite.TFLiteConverter instead |
| 4 | **2-Subgraph format NOT created** - README claims ESPHome-compatible models but creates single-subgraph | `src/export/tflite.py` | Models will FAIL ESPHome verification; no TYPE_13 state tensors for streaming |
| 5 | **Architecture selection is FAKE** - Config accepts "dnn", "cnn", "crnn" but ONLY MixedNet implemented | `src/model/architecture.py` | Config architecture value is completely ignored |
| 6 | **Config parameters NOT passed to model** - first_conv_filters, pointwise_filters, mixconv_kernel_sizes ignored | `src/training/trainer.py:308` | Hardcoded defaults used; config has no effect on model architecture |

---

## Major Issues (P1 - Fix Soon)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **Dead code in features.py** - Unreachable librosa fallback after return statement | `src/data/features.py` | Alternative audio processing path never executes |
| 2 | **Speaker clustering uses wrong model** - SpeechBrain ECAPA used, not WavLM as claimed | `src/data/clustering.py` | Leakage audit accuracy compromised |
| 3 | **verify_esphome.py hardcoded shape** - Only accepts [1, 3, 40], rejects stride=1 models | `src/export/verify_esphome.py` | Blocks valid model configurations from verification |
| 4 | **Hardcoded state shapes in export** - State tensor shapes fixed at [1, 96] | `src/export/tflite.py` | Won't adapt to different model configurations |
| 5 | **Stream recreation bug** - Stream layers created on every forward pass | `src/model/streaming.py` | Performance degradation; unnecessary overhead |

---

## What's Actually Working

| Component | Status | Notes |
|-----------|--------|-------|
| Config loader | Complete | All 9 dataclasses, env substitution, preset merging |
| MixedNet architecture | Functional | Proper MixConv blocks, residual connections |
| Data pipeline | Working | augmentation.py, dataset.py, features.py, clustering.py, hard_negatives.py |
| Training loop | Complete | Checkpointing, step-based training, mixed precision |
| Entry points | Working | mww-train and mww-export registered in setup.py |
| Performance utils | Working | GPU config, mixed precision, threading |
| Preset configs | Valid | fast_test.yaml, standard.yaml, max_quality.yaml all parse correctly |

---

## What's Claimed But Broken

| Claim | Reality | Evidence |
|-------|---------|----------|
| "ESPHome-compatible 2-subgraph TFLite models" | Single-subgraph with regular tf.Variable | No TYPE_13 state tensors in export |
| "Supports dnn, cnn, crnn architectures" | Only MixedNet implemented | build_model() ignores architecture parameter |
| "FAH (False Accepts per Hour) metric" | Calculates nonsense | Uses TN count instead of ambient hours |
| "ai-edge-litert for TFLite export" | Uses deprecated tf.lite.TFLiteConverter | Package listed but never imported |
| "WavLM-based speaker clustering" | Uses SpeechBrain ECAPA | Wrong model loaded for embeddings |
| "Configurable model dimensions via config" | Hardcoded defaults | Config values not passed to build_model() |
| "Comprehensive evaluation framework" | Dead code | metrics.py, calibration.py, fah_estimator.py never imported |

---

## Module-by-Module Assessment

### src/config/loader.py - HEALTHY
Status: Complete

All 9 dataclasses implemented with proper validation. Environment variable substitution works. Preset merging functional. 625 lines of solid config infrastructure.

### src/model/architecture.py - DYSFUNCTIONAL
Status: Misleading

MixedNet implementation is solid, but architecture selection is fake. Config accepts "dnn", "cnn", "crnn" but build_model() ignores this parameter entirely. Only MixedNet ever instantiated.

### src/training/trainer.py - BROKEN
Status: Critical

Training loop works but FAH calculation is mathematically wrong. Checkpoints saved based on garbage metric. Config parameters not passed to model (hardcoded defaults used).

### src/export/tflite.py - BROKEN
Status: Critical

Creates single-subgraph models instead of required 2-subgraph format. No TYPE_13 state tensors. Hardcoded state shapes. Models will fail ESPHome verification despite passing local script.

### src/data/augmentation.py - HEALTHY
Status: Complete

8 augmentation types properly implemented. GPU-accelerated SpecAugment via CuPy. All config parameters connected.

### src/data/dataset.py - HEALTHY
Status: Complete

RaggedMmap storage for efficient audio loading. Proper batching and shuffling. Config integration working.

### src/data/features.py - WORKING WITH DEAD CODE
Status: Partial

Primary feature extraction functional. Dead code exists: unreachable librosa fallback after return statement.

### src/data/clustering.py - MISLEADING
Status: Partial

Speaker clustering works but uses wrong model. Claims WavLM, implements SpeechBrain ECAPA. Leakage audit accuracy affected.

### src/data/hard_negatives.py - HEALTHY
Status: Complete

False positive detection and auto-mining functional. Config parameters properly connected.

### src/evaluation/metrics.py - DEAD CODE
Status: Unused

Never imported by trainer. trainer.py uses internal TrainingMetrics class instead.

### src/evaluation/calibration.py - DEAD CODE
Status: Unused

No references found in codebase.

### src/evaluation/fah_estimator.py - DEAD CODE
Status: Unused

Mathematically wrong FAH calculation in trainer.py instead.

### src/export/verify_esphome.py - BROKEN
Status: Major Issue

Hardcoded input shape [1, 3, 40] only. Rejects valid stride=1 models. Won't catch real ESPHome compatibility issues.

### src/utils/performance.py - HEALTHY
Status: Complete

GPU configuration, mixed precision setup, threading utilities all functional.

---

## Recommendations by Priority

### P0 (Critical - Fix Immediately)

1. **Fix FAH calculation** in trainer.py
   - Replace `fp / max(metrics.get("tn", 1), 1) * 1000` with proper formula
   - Need actual ambient audio duration in hours as denominator
   - Current checkpoints may be suboptimal

2. **Implement 2-subgraph TFLite export**
   - Add TYPE_13 state tensors for ESPHome streaming
   - Separate reset and streaming subgraphs
   - Reference: TensorFlow streaming examples

3. **Connect config parameters to model**
   - Pass first_conv_filters, pointwise_filters, mixconv_kernel_sizes to build_model()
   - Remove hardcoded defaults in architecture.py

4. **Fix architecture selection or remove options**
   - Either implement dnn/cnn/crnn or remove from config schema
   - Current behavior is misleading

5. **Remove or use ai-edge-litert**
   - Either migrate TFLiteConverter to ai-edge-litert
   - Or remove from requirements.txt

6. **Connect evaluation modules or delete them**
   - Either import metrics.py/calibration.py/fah_estimator.py in trainer
   - Or delete dead code to reduce confusion

### P1 (Major - Fix Soon)

1. **Fix verify_esphome.py hardcoded shapes**
   - Read model config to determine valid input shapes
   - Support stride=1 models properly

2. **Fix speaker clustering model selection**
   - Either implement actual WavLM support
   - Or update documentation to reflect ECAPA usage

3. **Remove dead code from features.py**
   - Delete unreachable librosa fallback
   - Clean up return statement placement

4. **Fix hardcoded state shapes in export**
   - Infer from model configuration instead of magic numbers

5. **Fix stream recreation bug**
   - Create stream layers once, reuse on forward passes
   - Cache and reset properly

### P2 (Minor - Nice to Have)

1. Add proper unit tests for evaluation metrics
2. Add integration test for ESPHome export
3. Document actual vs claimed features
4. Add architecture selection tests
5. Add config parameter passthrough validation

---

## Conclusion

The microwakeword_trainer project has solid infrastructure (config, data pipeline, training loop) but critical failures in core functionality. The disconnect between documentation and implementation is severe enough to cause production failures. **Do not use for production wake word training until P0 issues are resolved.**

The most dangerous issues are:
- **Wrong FAH metric** causing poor checkpoint selection
- **Broken ESPHome export** causing deployment failures
- **Fake architecture selection** misleading users

Fixing these requires changes to core training and export logic. The good news: the infrastructure is sound once these connections are properly made.
