# Implementation Plan: Standardize Training Pipeline and Implement Core Unit Tests

## Phase 1: Research and Component Analysis
- [x] Task: Audit existing `src/training/` and `src/data/` modules for testability
    - [x] Identify key functions and classes for unit testing
    - [x] Document the current state of test coverage
- [x] Task: Define the standard CLI argument parsing and error handling pattern
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Research and Component Analysis' (Protocol in workflow.md)

## Phase 2: Implementation of Unit Tests for Core Modules
- [x] Task: Write Unit Tests for `src/data/` (Ingestion, Augmentation)
    - [x] Implement tests for WAV file loading and preprocessing
    - [x] Implement tests for GPU-accelerated SpecAugment (where mockable or GPU-available)
- [x] Task: Write Unit Tests for `src/model/` (Architecture, MixedNet)
    - [x] Implement tests for model creation and input/output shapes
- [~] Task: Write Unit Tests for `src/training/` (Trainer, Miner)
    - [ ] Implement tests for the main training loop and checkpointing
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation of Unit Tests for Core Modules' (Protocol in workflow.md)

## Phase 3: Pipeline Standardization
- [ ] Task: Standardize CLI Argument Parsing
    - [ ] Refactor `mww-train` and `mww-export` to use a consistent argument parser
    - [ ] Implement robust error messages for missing hardware (GPU/CUDA)
- [ ] Task: Implement Centralized Logging and Progress Reporting
    - [ ] Standardize the use of `rich` for CLI output across all tools
- [ ] Task: Final Verification and Pipeline Dry-Run
    - [ ] Execute a full `mww-train --dry-run` to ensure all components are correctly integrated
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Pipeline Standardization' (Protocol in workflow.md)
