# ESPHome microWakeWord Training Pipeline - Task Checklist

## Phase 1 — Data Ingestion & Validation
- [ ] Create audio ingestion module (SampleRecord dataclass) - USE dataset/ folder
- [ ] Implement audio validation (16kHz, 16-bit PCM, mono) - USE dataset/ folder
- [ ] Create Clips loader with train/val/test splitting - USE dataset/ folder
- [ ] Implement data directory structure setup - USE dataset/ folder

## Phase 2 — Feature Extraction Pipeline
- [ ] Integrate pymicro-features MicroFrontend
- [ ] Create SpectrogramGeneration class with slide_frames
- [ ] Implement RaggedMmap storage for efficient loading
- [ ] Add feature parameter validation

## Phase 3 — Training Loop (Step-Based)
- [ ] Implement step-based training loop (not epoch-based)
- [ ] Add class weighting (positive/negative weights)
- [ ] Implement two-priority checkpoint selection
- [ ] Add evaluation every N steps

## Phase 4 — MixedNet Model Architecture
- [ ] Implement MixedNet streaming architecture
- [ ] Add StridedSlice and Concatenation for streaming
- [ ] Implement MixConv blocks with ring buffers
- [ ] Add VAR_HANDLE/READ/ASSIGN for state variables

## Phase 5 — Dataset Engineering
- [ ] Implement FeatureSet configuration with sampling weights
- [ ] Add truncation strategies (random, truncate_start, truncate_end, split)
- [ ] Implement FAH estimation for ambient data
- [ ] Create data balancing utilities

## Phase 6 — Augmentation Pipeline
- [ ] Integrate audiomentations for time-domain augmentation
- [ ] Add background noise mixing (AddBackgroundNoise)
- [ ] Implement RIR (room impulse response) augmentation
- [ ] Configure augmentation probabilities

## Phase 7 — Hard Negative Mining
- [ ] Implement hard negative detection (FP > threshold)
- [ ] Add hard negative storage and reloading
- [ ] Implement iterative mining loop
- [ ] Add hard negative sampling weight adjustment

## Phase 8 — Export & TFLite Conversion
- [ ] Implement streaming SavedModel conversion
- [ ] Add TFLite quantization (int8 input, uint8 output)
- [ ] Implement representative dataset generation
- [ ] Add _experimental_variable_quantization support

## Phase 9 — ESPHome Manifest Generation
- [ ] Implement V2 manifest JSON generation
- [ ] Add tensor_arena_size calculation
- [ ] Implement probability_cutoff configuration
- [ ] Add ESPHome version compatibility checking

## Phase 10 — Comprehensive Metrics Suite
- [ ] Implement wake word metrics (recall, precision, F1)
- [ ] Add ROC/PR curve computation
- [ ] Implement FAH metrics (ambient_false_positives_per_hour)
- [ ] Add average_viable_recall calculation

## Phase 11 — Performance Optimization (GPU-First)
- [x] Implement CuPy GPU-accelerated SpecAugment
- [ ] Add parallel audio augmentation (32 threads)
- [ ] Configure TensorFlow GPU memory growth
- [ ] Add mixed precision training support

## Phase 12 — Profiling & Monitoring
- [x] Implement TrainingProfiler with cProfile
- [ ] Add TensorBoard logging integration
- [ ] Implement step timing tracking
- [ ] Add performance debugging utilities

## Project Structure & Configuration
- [x] Create project directory structure
- [x] Implement YAML configuration loader
- [x] Add requirements.txt with all dependencies
- [x] Create preset configs (fast_test, standard, max_quality)

## Documentation
- [ ] Add inline code documentation
- [ ] Create README with usage examples
- [ ] Add ESPHome compatibility verification script
- [ ] Document GPU setup and performance tuning
