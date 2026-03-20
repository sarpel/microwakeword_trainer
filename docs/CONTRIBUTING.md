# Contributing Guide

Guide for contributing to microwakeword_trainer, including development setup, testing procedures, and code style enforcement.

<!-- AUTO-GENERATED: Generated from Makefile, pyproject.toml, and project conventions -->
<!-- Last updated: 2026-03-20 -->

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or 3.11
- **GPU**: CUDA-capable NVIDIA GPU (for training and testing)
- **CUDA**: Version 12.x (required for CuPy compatibility)
- **OS**: Linux (recommended), macOS (limited GPU support)
- **RAM**: 16GB+ recommended for standard training

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg

# Verify CUDA installation
nvidia-smi
```

---

## Development Environment Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/mww/microwakeword_trainer.git
cd microwakeword_trainer
```

### Step 2: Create Virtual Environment

⚠️ **Two separate environments required**: TensorFlow (main) and PyTorch (clustering).

**Environment 1: TensorFlow (Main Development)**
```bash
python3.11 -m venv ~/.venvs/mww-dev-tf
source ~/.venvs/mww-dev-tf/bin/activate
```

**Environment 2: PyTorch (Speaker Clustering)**
```bash
python3.11 -m venv ~/.venvs/mww-dev-torch
source ~/.venvs/mww-dev-torch/bin/activate
```

### Step 3: Install Development Dependencies

**TensorFlow Environment:**
```bash
cd microwakeword_trainer
make install-dev
```

This installs:
- Production dependencies (TensorFlow, CuPy, audiomentations, etc.)
- Development tools (pytest, ruff, black, mypy, pre-commit)
- Project in editable mode (`pip install -e .`)
- Pre-commit hooks

**PyTorch Environment (if working on clustering):**
```bash
cd microwakeword_trainer
pip install -r requirements-torch.txt
```

### Step 4: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10 or 3.11

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run pre-commit self-check
pre-commit run --all-files
```

---

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow project conventions:
- Use existing code style and patterns
- Add type hints for new functions (optional but recommended)
- Write tests for new functionality
- Update documentation if needed

### 3. Run Checks Locally

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run tests
make test

# Run all checks at once
make check
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"  # or "fix: resolve issue"
```

**Commit Message Convention:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `refactor:` Code refactoring (no functional change)
- `test:` Test addition or update
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/my-feature-name
# Create PR on GitHub
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_async_miner.py       # AsyncHardExampleMiner tests
│   ├── test_config.py            # ConfigLoader & dataclasses tests
│   ├── test_test_evaluator.py    # TestEvaluator tests
│   ├── test_vectorized_metrics.py # MetricsCalculator tests
│   └── test_spec_augment_tf.py   # TF SpecAugment tests
└── integration/
    └── test_training.py          # End-to-end training smoke test
```

### Running Tests

```bash
# All tests
make test

# Unit tests only (fast, no GPU required)
make test-unit

# Integration tests only (GPU required)
make test-integration

# Parallel execution (faster)
make test-parallel

# Fast tests only (excludes slow and gpu markers)
make test-fast

# Tests with coverage
make coverage
```

### Writing Tests

Follow these conventions:

**Unit Tests:**
- Test single functions or classes in isolation
- Use pytest fixtures for common setup
- Mock external dependencies (file I/O, GPU operations)
- Fast execution (< 1 second per test)

```python
import pytest
from config.loader import ConfigLoader

def test_load_config_success(tmp_path):
    # Arrange
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model:\n  first_conv_filters: 8")

    # Act
    config = ConfigLoader.load(str(config_file))

    # Assert
    assert config.model.first_conv_filters == 8
```

**Integration Tests:**
- Test multiple components together
- Use real data (small synthetic dataset)
- Require GPU for training operations
- Mark as `@pytest.mark.integration`

```python
import pytest

@pytest.mark.integration
@pytest.mark.gpu
def test_full_training_pipeline(config_preset="fast_test"):
    # Run actual training with small dataset
    from src.training.trainer import train
    result = train(config_preset)

    assert result.best_checkpoint is not None
```

### Test Markers

```python
@pytest.mark.slow          # Takes > 10 seconds
@pytest.mark.gpu           # Requires GPU
@pytest.mark.integration    # Integration test
@pytest.mark.unit          # Unit test (default)
```

**Run specific markers:**
```bash
# Only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Only GPU tests
pytest -m gpu
```

---

## Code Style

### Linting

The project uses **Ruff** for linting (replaces flake8, pylint, isort).

```bash
# Run linter
make lint

# Auto-fix issues
make lint-fix
```

**Ruff Rules:**
- Pyflakes (F): Catches undefined names, unused imports
- Pycodestyle functional rules (E9, E71, E72): Syntax and runtime errors
- isort (I): Import organization
- Flake8-bugbear (B): Common bugs
- Flake8-comprehensions (C4): List comprehension performance issues
- Flake8-bandit (S): Security issues

**Line Length:** 200 characters (permissive, focus on bugs not cosmetics)

### Formatting

The project uses **Ruff format** (replaces Black).

```bash
# Format all files
make format

# Check formatting without changes
make format-check
```

**Formatting Rules:**
- 4-space indentation
- Double quotes for strings
- Trailing commas in multi-line lists/dicts
- No magic trailing comma removal

### Type Checking

The project uses **MyPy** for type checking.

```bash
make type-check
```

**Type Checking Policy:**
- Relaxed mode: `disallow_untyped_defs = false`
- Type hints optional but recommended for public APIs
- Bug-finding enabled: `warn_return_any`, `warn_no_return`, `strict_optional`

**External Libraries:** Most external libs are ignored in mypy (missing stubs).

### Pre-commit Hooks

Pre-commit hooks enforce code quality automatically.

```bash
# Install hooks
make pre-commit

# Run all hooks manually
pre-commit run --all-files
```

**Hooks configured:**
- Trailing whitespace
- End of file fixer
- Check YAML/JSON syntax
- Ruff formatting
- Ruff linting
- Type checking (optional)

---

## Project Structure

```
microwakeword_trainer/
├── config/                   # Configuration system
│   ├── loader.py             # ConfigLoader + dataclasses
│   ├── presets/              # YAML presets
│   └── AGENTS.md
├── src/
│   ├── pipeline.py           # Top-level orchestrator
│   ├── model/               # MixedNet architecture
│   ├── training/            # Training loop, augmentation, mining
│   ├── data/                # Dataset, features, preprocessing
│   ├── evaluation/          # Metrics, FAH estimator
│   ├── export/              # TFLite export, verification
│   ├── tools/               # CLI wrappers
│   ├── tuning/              # Auto-tuning
│   └── utils/              # GPU, logging, seed
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── scripts/                 # Standalone utilities
├── docs/                   # Documentation
└── specs/                  # Implementation specs
```

---

## Documentation

### Code Documentation

- **Docstrings**: Google-style docstrings for public APIs
- **Comments**: Explain "why", not "what"
- **AGENTS.md**: Per-module AI agent guidelines in `src/*/AGENTS.md`

```python
def train(config: FullConfig) -> TrainingResult:
    """Train wake word model with two-phase training.

    Args:
        config: Full configuration loaded from YAML preset.

    Returns:
        TrainingResult containing best checkpoint path and metrics.

    Raises:
        ValueError: If dataset is empty or GPU not available.
    """
```

### Updating Documentation

- **Keep docs/INDEX.md updated** with new modules/commands
- **Update CONFIGURATION.md** when adding new config fields
- **Add to COMMANDS.md** when creating new CLI commands
- **Update KNOWN_ISSUES.md** in TROUBLESHOOTING.md when fixing bugs

---

## Two Virtual Environments

### TensorFlow Environment (mww-dev-tf)
- **Purpose**: Training, export, inference, data processing
- **Key Deps**: TensorFlow, CuPy, audiomentations, pymicro-features
- **Used for**: 90% of development work

### PyTorch Environment (mww-dev-torch)
- **Purpose**: Speaker clustering only
- **Key Deps**: PyTorch, SpeechBrain, ECAPA-TDNN
- **Used for**: Clustering-related changes only

**Switch between environments:**
```bash
# Activate TensorFlow env
source ~/.venvs/mww-dev-tf/bin/activate

# Activate PyTorch env
source ~/.venvs/mww-dev-torch/bin/activate
```

---

## Common Issues

### Import Errors After Install

```bash
# Solution: Install in editable mode
pip install -e .
```

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Check TensorFlow GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Solution: Install correct TensorFlow version with CUDA support
pip install "tensorflow[and-cuda]>=2.16,<2.17"
```

### CuPy Installation Fails

```bash
# Solution: Install CuPy matching CUDA version
pip install "cupy-cuda12x>=13.0"
```

### Pre-commit Hooks Fail

```bash
# Solution: Update hooks
pre-commit autoupdate

# Or skip for one commit (not recommended)
git commit --no-verify
```

---

## Getting Help

- **Documentation**: [docs/INDEX.md](INDEX.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: Report bugs via GitHub Issues
- **AGENTS.md**: See [AGENTS.md](../AGENTS.md) for AI agent guidelines

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
