# Phase 4: Best Practices & Standards

## Framework & Language Findings

### Critical

#### BP-C1: `knob` Undefined in `orchestrator.py:236` — NameError at Runtime
*(Also captured as DOC-C2; confirmed as a code-level finding)*
- **File:** `src/tuning/orchestrator.py`, line 236
- `knob.apply(model, candidate, self.auto_tuning_config)` references `knob` which is never assigned in `tune()`'s scope. `_make_knob()` exists but is never called. All unit tests gate on `dry_run=True` which returns early, masking the bug. Any real auto-tuning run raises `NameError` on the first iteration.
- **Fix:** Add `knob = self._make_knob(current_knob_name)` between lines 233-234.

### High

#### BP-H1: Dual `setup.py` + `pyproject.toml` Build System with Divergent Metadata
- Both files define the package with different authors, emails, dependency constraints, and extras.
- `setup.py` `install_requires` includes TensorFlow, CuPy, numba. `pyproject.toml` only lists `requests>=2.32`. `pip install .` using `pyproject.toml` installs a broken package missing all ML dependencies.
- `setup.py` `find_packages()` includes `tests/`; `pyproject.toml` scopes to `["src*", "config*"]` — different installed packages.
- **Fix:** Delete `setup.py`. Consolidate all metadata into `pyproject.toml`. Accept `requirements.txt` as the runtime dependency source of truth.

#### BP-H2: `model.train_on_batch` — Legacy Keras 1/2 Pattern
- **File:** `src/training/trainer.py:1341`
- Forces a Python-level call per step, bypasses XLA JIT tracing, cannot be graph-traced. The docstring says "no `.numpy()` calls in hot path" but `train_on_batch` executes eagerly with a Python round-trip per call, contradicting this intent.
- **Fix:** Override `train_step` in the model subclass or use a `@tf.function(reduce_retracing=True)` gradient tape loop.

#### BP-H3: Deprecated `tf.data.experimental.prefetch_to_device`
- **File:** `src/data/tfdata_pipeline.py:328,430,479`
- Graduated out of `experimental` in TF 2.6. Use `.prefetch(tf.data.AUTOTUNE)` with TF automatic device placement.

#### BP-H4: `MixConvBlock.__init__` Incomplete — `AttributeError` + Infinite Recursion
- **File:** `src/model/architecture.py:151-165`
- `self.filters = filters` never assigned in `__init__` — `AttributeError` in `build()` at the `if self.filters is not None` check. The `mode` setter contains `self.mode = mode` inside the setter body, causing infinite recursion (or `NameError` — `mode` is the `__init__` parameter, not in setter scope).
- **Fix:** Complete `__init__` to assign all instance attributes. Fix setter to use `self._mode = value` throughout.

#### BP-H5: Dead Async Validation Infrastructure
- **File:** `src/training/trainer.py:1544,1647-1666`
- `_schedule_validation()` unconditionally returns `False`. The `ThreadPoolExecutor` and `_validation_lock` are created at `__init__` but never used. `_compute_metrics_background` raises `RuntimeError` immediately. Consumes a background thread for the entire lifetime of `Trainer`.
- **Fix:** Remove executor, lock, and `_pending_validation` state until async validation is re-implemented.

### Medium

#### BP-M1: Deprecated `options.experimental_deterministic`
- **File:** `src/data/tfdata_pipeline.py:313,319`
- Use `options.deterministic = False` (stable since TF 2.5).

#### BP-M2: Private `model._flatten_layers()` API Removed in Keras 3
- **Files:** `src/tuning/orchestrator.py:142,149`; `src/export/tflite.py:233`
- **Fix:** Use `model.layers` + recursive sublayer check via public API.

#### BP-M3: Mixed Old/New Type Annotation Styles
- Legacy `Dict`, `List`, `Optional`, `Union` from `typing` in ~6 files; project requires Python 3.10+.
- **Fix:** Replace all `typing` generics with PEP 585/604 equivalents.

#### BP-M4: Mypy Configured Too Leniently — Skips Unannotated Function Bodies
- **File:** `pyproject.toml:248-257`
- `check_untyped_defs = false`, `disallow_untyped_defs = false`, `allow_untyped_calls = true` — mypy skips the body of the entire training hot path.
- **Fix:** Enable `check_untyped_defs = true`; incrementally enable `disallow_untyped_defs = true`.

#### BP-M5: Dual Mypy Config — `mypy.ini` Silently Shadows `pyproject.toml`
- `mypy.ini` takes precedence over `pyproject.toml [tool.mypy]`. The 6 extra module overrides in `pyproject.toml` are silently ignored, causing `Missing stub` errors for those modules.
- **Fix:** Delete `mypy.ini`; consolidate all mypy config into `pyproject.toml`.

#### BP-M6: Over-Pinning Transitive Dependencies
- `requirements.txt` pins transitive deps like `opentelemetry-proto==1.25.0`. `scipy` differs between `requirements.txt` (1.15.3) and `requirements-torch.txt` (1.17.1) with no reconciliation.
- **Fix:** Use `pip-compile` / `uv lock` to manage pins; declare only direct dependencies in `pyproject.toml`.

### Low

- **BP-L1:** f-strings in logger calls — add `"G"` to Ruff `select` to enforce `%`-style lazy evaluation
- **BP-L2:** `src/config/__init__.py` empty with misleading "provides configuration management" docstring
- **BP-L3:** `PrefetchGenerator` docstring example hard-codes deprecated `train_on_batch`
- **BP-L4:** Ad-hoc `dict` returns where dataclasses would be safer (e.g., `_get_current_phase_settings`)
- **BP-L5:** Missed walrus operator / structural pattern matching for repeated metric-cast patterns and mode-string → enum mapping

---

## CI/CD & DevOps Findings

### Critical

#### CD-C1: No CI/CD Pipeline Exists
- No `.github/workflows/`, no `.gitlab-ci.yml`, no CircleCI, no Dockerfile. The only automation is a Makefile run manually.
- `make pre-commit` and `make install-dev` both invoke `pre-commit install` / `pre-commit run --all-files` but `.pre-commit-config.yaml` **does not exist** — these targets will error out.
- **Operational risk:** Every regression, broken import, security violation, and lint error reaches the main branch undetected.
- **Fix:** (1) Create `.pre-commit-config.yaml` with ruff, mypy, trailing-whitespace hooks. (2) Create `.github/workflows/ci.yml` with lint + type-check + unit test jobs. (3) Add branch protection requiring CI to pass.

#### CD-C2: Coverage Gate Nonfunctional — `fail_under = 0`
- **File:** `pyproject.toml`
- Coverage is enforced at 0% (`fail_under = 0`), meaning coverage failures never block CI. The coverage report (0.15% line rate) is generated on every test run but provides no signal.
- **Fix:** Set `fail_under = 5` as a floor immediately. Fix import failures that prevent test collection (likely missing `cupy`/GPU dependencies — add `importorskip` guards). Raise threshold incrementally.

### High

#### CD-H1: `tests/` is in `.gitignore` — Test Suite May Not Be Version-Controlled
- **File:** `.gitignore`
- The entry `tests/` in `.gitignore` excludes the entire test suite from git tracking. If this was accidental, test files may not be committed on all developer machines. If intentional, the entire test infrastructure is outside version control.
- **Operational risk:** Contributors may run `git add .` and never commit test files. CI would run against a missing test suite.
- **Fix:** Remove `tests/` from `.gitignore` immediately. Run `git status` to verify all test files are tracked.

#### CD-H2: `.gitignore` Missing All ML Artifact and Sensitive Paths
- Current `.gitignore`: only 7 lines covering `*.pyc`, `__pycache__/`, `manifest.json`, and two personal files.
- Missing: `.env*`, `*.key`, `*.pem`, `checkpoints/`, `models/`, `data/`, `dataset/`, `logs/`, `tuning_output/`, `coverage_html/`, `coverage.xml`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `*.h5`, `*.tflite`, `*.pb`, `*.ckpt`, `*.pth`
- `tuning_output/` is already untracked and accumulating; model checkpoints could be accidentally committed.

#### CD-H3: Dependency Pinning Inconsistency — `pip install .` Installs Broken Package
- `pyproject.toml [project.dependencies]` lists only `requests>=2.32`. `pip install .` installs a package with no ML dependencies. `setup.py install_requires` has TF, CuPy, NumPy. `setup.py` `find_packages()` includes `tests/`; `pyproject.toml` doesn't.
- No `pip-audit`, no Dependabot, no supply chain security. `scipy` differs between `requirements.txt` and `requirements-torch.txt`.
- **Fix:** Delete `setup.py`. Add `pip-audit -r requirements.txt` to Makefile `check` target.

### Medium

#### CD-M1: Makefile Minor Issues
- `install-dev` depends on `install` then `pip install -e .` (uses different deps than `requirements.txt`).
- `clean` target doesn't remove `tuning_output/`, `logs/`, `coverage.xml`.
- No `check-env` target to validate Python version and GPU availability before expensive operations.
- **Fix:** Add `check-env` target; extend `clean`; document that `install` must run in a fresh virtualenv.

#### CD-M2: Logging Infrastructure Lacks Rotation and Structured Output
- **File:** `src/utils/logging_config.py`
- `FileHandler` with `mode="a"` grows unboundedly. No log rotation. No structured JSON logging for aggregation systems. No `LOG_LEVEL` env var respected. `get_logger()` returns unconfigured logger if `setup_rich_logging()` hasn't been called.
- **Fix:** Replace with `RotatingFileHandler(maxBytes=50MB, backupCount=5)`. Add `LOG_LEVEL` env var check. Add `logging.NullHandler()` to root logger at module level.

#### CD-M3: Split Virtualenv Architecture Undocumented and Unenforceable
- Two separate venvs required (TF + PyTorch); no Makefile targets to create/validate/switch them. No `tox.ini`, no `.env.example`, no env var documentation.
- **Fix:** Add `make create-tf-env` / `make create-torch-env` targets. Create `.env.example`. Add `check-env` guard detecting TF+PyTorch co-installation.

#### CD-M4: No SBOM, No Supply Chain Security
- No `pip-audit`, no `--require-hashes` in requirements, no Sigstore signing. `setup.py` has placeholder `url = "https://github.com/mww/microwakeword_trainer"` — supply chain confusion risk if package is published to PyPI.
- **Fix:** Add `pip-audit` to CI. Correct repository URL in all packaging metadata.

### Low

- **CD-L1:** `line-length = 200` across Ruff, Black, isort — reduce to 120 for code review readability

---

## Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Framework/Language | 1 | 5 | 6 | 5 | 17 |
| CI/CD & DevOps | 2 | 3 | 4 | 1 | 10 |
| **Total** | **3** | **8** | **10** | **6** | **27** |
