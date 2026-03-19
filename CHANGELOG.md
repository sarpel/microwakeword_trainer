# Changelog

All notable changes to the microwakeword_trainer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-19

### Security Fixes

- **SEC-001**: Fixed pickle RCE vulnerability in tuning module (CVSS 9.8)
  - Replaced `pickle.dumps/pickle.loads()` with `np.savez/np.load(allow_pickle=False)`
- **SEC-002**: Fixed unsafe cache loading with `allow_pickle=True` (CVSS 9.1)
- **SEC-003**: Re-enabled S301 security rule in pyproject.toml
- **SEC-005**: Added path traversal validation to all CLI tools (CWE-22)
  - Created `src/utils/path_utils.py` with `resolve_path_safe()` and `validate_path_within_dir()`
  - Integrated validation in: `tflite.py`, `trainer.py`, `mining.py`, `tuning/cli.py`, `cluster_analyze.py`, `cluster_apply.py`

### Dependency Updates

- **DEP-001**: Fixed CVE-2026-32274 in black package
  - Updated from `black>=26.0` to `black>=26.3.1`

### Performance Optimizations

- **PERF-001**: Converted ~18 print statements to structured logging (5-15% throughput improvement)
  - `src/data/quality.py`: 8 conversions
  - `src/export/tflite.py`: 10 conversions
- **PERF-002**: Verified memory-aware cache eviction in RaggedMmap
- **PERF-003**: Added parallel audio loading with ThreadPoolExecutor (20-40% faster)
  - Modified `src/data/ingestion.py` for concurrent I/O
- **PERF-004**: Implemented frontend caching (10-20% overhead reduction)
  - Modified `src/data/features.py` to use cached MicroFrontend
- **PERF-005**: Changed default SpecAugment backend from CuPy to TF (15-25% faster)
  - Modified `src/data/tfdata_pipeline.py`
- **PERF-006**: Verified reservoir sampling in validation metrics

### API Updates

- **TF 2.16+ Compatibility**: Fixed deprecated `TensorShape.as_list()` API
  - Replaced 6 occurrences with `list()` in `architecture.py`, `streaming.py`, `tflite.py`

### Infrastructure

- Added CI/CD pipeline with security scanning
  - `.github/workflows/security.yml`: Bandit, Safety, TruffleHog, Gitleaks, Ruff S-rules
  - `.github/workflows/ci.yml`: Lint, test, build validation
- Added security documentation (`docs/SECURITY.md`)
- Added performance tuning guide (`docs/PERFORMANCE.md`)

### Code Quality

- Created `src/utils/path_utils.py` for secure path operations
- Improved error handling with proper logging
- Enhanced type safety with modern Python 3.11+ features

### Breaking Changes

- **Ruff security rules**: S301 (pickle) now enforced - code using pickle must migrate to safe alternatives
- **Default SpecAugment backend**: Changed from "cupy" to "tf" for better performance

### Migration Guide

#### For Pickle Usage

If you were using `pickle` directly (not recommended):

```python
# OLD (VULNERABLE)
import pickle
data = pickle.load(file)

# NEW (SECURE)
import numpy as np
data = np.load(file, allow_pickle=False)
```

#### For Path Operations

```python
# OLD (VULNERABLE)
path = Path(user_input)

# NEW (SECURE)
from src.utils.path_utils import resolve_path_safe
path = resolve_path_safe(user_input, base_dir=Path("/data"))
```

---

## [1.x.x] - Previous Releases

### Changes in 1.x releases were not tracked in this changelog.

### Key Milestones

- v1.0.0: Initial release with wake word detection training
- v1.5.0: Added hard negative mining
- v1.8.0: Added streaming model export
- v1.9.0: Added auto-tuning capabilities
- v2.0.0: Major security and performance improvements

---

## Contributors

- Sarpel GURY (@tsarpel15) - Main developer

## License

MIT License - see LICENSE file for details
