# Phase 2A: Security Audit (Raw)

## Executive Summary

Generally sound security practices for a local ML training tool. YAML loading consistently uses `yaml.safe_load()`, subprocess calls use list-form arguments (no `shell=True`), no hardcoded credentials. Key concerns: **unsafe pickle deserialization** in tuning module, **`allow_pickle=True`** on disk-loaded cache files, **overly permissive Bandit/Ruff suppression list**, and **unbounded memory allocation** during mining.

---

## High Findings

### SEC-H1: Unsafe Pickle Deserialization in Population Module
- **CWE:** CWE-502 — CVSS 7.5
- **File:** `src/tuning/population.py`, lines 28, 34, 86, 114
- `Candidate` uses `pickle.dumps()`/`pickle.loads()` for model weights. If state is ever persisted to disk (as AGENTS.md suggests) and loaded from a malicious file, arbitrary code execution is possible.
- `pyproject.toml` line 121 globally suppresses Bandit rule `S301` (pickle), masking this in automated scans.
- **Fix:** Replace with `numpy.savez`/`numpy.load(allow_pickle=False)`. Never persist `weights_bytes` to disk in raw pickle form.

### SEC-H2: `numpy allow_pickle=True` on Cache Files from Disk
- **CWE:** CWE-502 — CVSS 7.5
- **File:** `src/data/clustering.py`, line 1162
- `np.load(cache_file, allow_pickle=True)` loads from `/tmp` (world-writable). An attacker can replace an `emb_*.npz` file with a malicious numpy pickle payload.
- Line 549 in the same file correctly uses `allow_pickle=False`.
- **Fix:** Change to `allow_pickle=False`; save `model_name` field as JSON sidecar.

### SEC-H3: Overly Permissive Global Ruff Security Rule Suppressions
- **CWE:** CWE-693 — CVSS 6.5
- **File:** `pyproject.toml`, lines 117-128
- Globally suppresses: `S301` (pickle), `S603`/`S607` (subprocess), `S605`/`S606` (OS commands), `S105`/`S106`/`S107` (hardcoded passwords). New code introducing these patterns will never be flagged.
- **Fix:** Replace global ignores with per-file ignores using `[tool.ruff.lint.per-file-ignores]`. Use inline `# noqa: S301` at specific lines that genuinely need suppression.

---

## Medium Findings

### SEC-M1: Unbounded `batch_features_cache` — Memory Exhaustion (CWE-400)
- **File:** `src/training/mining.py`, line 205
- Cache grows linearly with dataset size; not cleared until after all batches processed.
- **Fix:** Evict entries during the mining loop when no longer referenced by heap.

### SEC-M2: Temp Directories in World-Writable `/tmp` for ML Models
- **CWE:** CWE-377, CWE-732
- **Files:** `src/data/clustering.py` lines 124, 490, 674, 961; `src/export/tflite.py` line 1237
- SpeechBrain models and embedding caches stored in `/tmp` — symlink attacks and model poisoning risk on shared systems.
- **Fix:** Use project-local cache directories. Log warnings on cleanup failure (currently uses `ignore_errors=True`).

### SEC-M3: `ast.literal_eval` on Config-Derived Strings (CWE-95)
- **Files:** `scripts/debug_streaming_gap.py:42`, `src/model/architecture.py:38`, `src/export/verification.py:46`, `src/export/tflite.py:1633`, `scripts/verify_esphome.py:172`
- Multiple files wrap user-controllable config strings in `ast.literal_eval(f"[{user_string}]")`. Safe from code injection but no format validation.
- **Fix:** Validate against regex `^[\d,\[\]\s]+$` before parsing; prefer explicit integer list parsing.

### SEC-M4: Subprocess Args Include User-Controllable Config Values (CWE-78)
- **Files:** `src/pipeline.py` lines 58-60, 93-109; `scripts/count_dataset.py` lines 17-31
- List-form calls prevent shell injection, but no path validation before passing to subprocesses.
- **Fix:** Validate paths exist and are files. Use `--` before filename in ffprobe calls.

### SEC-M5: Insufficient `.gitignore` Coverage (CWE-200)
- **File:** `.gitignore` (only 8 lines)
- Missing: `.env*`, `*.key`, `*.pem`, `checkpoints/`, `models/`, `dataset/`, `data/`, `logs/`, `tuning_output/`, `.sisyphus/`, `coverage_html/`, `coverage.xml`
- **Fix:** Expand `.gitignore` to cover all sensitive directories.

### SEC-M6: `yaml.safe_load` on Dynamically Constructed YAML String (CWE-20)
- **File:** `scripts/verify_esphome.py`, line 172
- `yaml.safe_load(f"[{mixconv_str}]")` constructs YAML from config-derived string interpolation.
- **Fix:** Use explicit integer list parsing instead.

---

## Low Findings

- **SEC-L1:** MD5 used in `src/training/mining.py:84` and `src/data/clustering.py:509` — standardize on SHA-256
- **SEC-L2:** SHA-1 used in `src/data/dataset.py:765` — standardize on SHA-256
- **SEC-L3:** `MWW_VERIFY_CONFIG`/`MWW_VERIFY_CHECKPOINT` env vars not validated against expected paths
- **SEC-L4:** `.sisyphus/evidence/` created without restrictive permissions (0o700)
- **SEC-L5:** `sys.exit()` deep in library code suppresses error context and prevents cleanup

---

## Positive Security Observations
- All YAML uses `yaml.safe_load()` — no `yaml.load()` found
- All subprocess calls use list-form args (no `shell=True`)
- `ingestion.py` performs thorough WAV file validation
- No hardcoded credentials found
- `numpy allow_pickle=False` used correctly in `clustering.py:549`
- `IOProfiler` has `_MAX_OPS = 500` cap
- `tflite.py` uses try/finally for temp dir cleanup

---

## Summary: 0 Critical, 3 High, 6 Medium, 5 Low
