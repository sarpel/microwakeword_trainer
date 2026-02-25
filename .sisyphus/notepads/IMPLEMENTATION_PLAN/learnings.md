# Atlas Orchestration Session - ESPHome microWakeWord v2.0

**Session ID:** ses_36a929762ffe6iefO0if4kd8Bi  
**Started:** 2026-02-25T15:31:41.388Z  
**Plan:** IMPLEMENTATION_PLAN_v2.0_Performance.md

---

## Current Status

**Base Repository:** `/home/sarpel/mww/mww-official-repo/` - Official microWakeWord codebase exists

**Existing Implementation:**
- Audio processing: clips.py, augmentation.py, audio_utils.py, spectrograms.py
- Model architecture: mixednet.py, inception.py
- Training: train.py, model_train_eval.py, data.py
- Streaming layers: stream.py, delay.py, modes.py
- Utilities: utils.py, test.py, inference.py

**v2.0 Enhancements Required:**
1. GPU-accelerated SpecAugment (CuPy)
2. 32-thread parallel audio augmentation
3. Performance profiling infrastructure
4. TensorBoard integration enhancements
5. Configuration schema improvements
6. Performance monitoring utilities

---

## Task Groups

### Group 1: Project Structure & Configuration (Sequential)
- [ ] Create new microwakeword_trainer package structure
- [ ] Implement YAML configuration loader with validation
- [ ] Create performance configuration module

### Group 2: GPU-Accelerated Components (Can parallelize after Group 1)
- [ ] Implement CuPy SpecAugment (GPU-mandatory)
- [ ] Add parallel audio augmentation (32 threads)
- [ ] Configure TensorFlow GPU settings

### Group 3: Profiling & Monitoring (Can parallelize after Group 1)
- [ ] Implement TrainingProfiler with cProfile
- [ ] Add performance timing decorators
- [ ] Create GPU/CPU monitoring utilities

### Group 4: Integration & Testing (Depends on Groups 2-3)
- [ ] Integrate performance modules into training pipeline
- [ ] Add TensorBoard performance metrics
- [ ] Create ESPHome compatibility verification

---

## Learnings & Decisions

**From Plan Analysis:**
- GPU-only policy for SpecAugment (no CPU fallback)
- 32 threads for parallel processing
- Input: [1, 3, 40] INT8, Output: [1, 1] UINT8
- 2 subgraphs required (inference + init)
- 6 state variables with VAR_HANDLE pattern

**Architecture Decisions:**
- Keep existing mww-official-repo as base
- Create new microwakeword_trainer/ package for v2.0 enhancements
- Use import hooks to extend existing functionality
