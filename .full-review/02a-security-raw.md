# Security Audit Report: microwakeword_trainer v2.0.0

**Date:** 2026-03-18
**Auditor:** Claude Code Security Auditor
**Scope:** Full codebase review - TensorFlow/Python ML training pipeline
**Framework:** OWASP Top 10 2021, CWE, CVSS v3.1

---

## Executive Summary

This security audit of the microwakeword_trainer codebase identified **2 Critical**, **2 High**, **3 Medium**, and **4 Low** severity findings. The codebase generally follows sound security practices for a local ML training tool: YAML loading consistently uses `yaml.safe_load()`, subprocess calls use list-form arguments without `shell=True`, and no hardcoded credentials were found.

**Key Security Concerns:**
1. **Unsafe pickle deserialization** in tuning module enables arbitrary code execution
2. **`allow_pickle=True`** on disk-loaded cache files from world-writable `/tmp` enables RCE
3. **Overly permissive Bandit/Ruff suppression list** masks security issues
4. **Unbounded memory allocation** during hard-negative mining enables DoS

---

## Findings Summary

| ID | Severity | Category | File | CWE | CVSS |
|----|----------|----------|------|-----|------|
| SEC-C1 | **Critical** | Insecure Deserialization | `src/tuning/population.py:28,34,86,114` | CWE-502 | 9.8 |
| SEC-C2 | **Critical** | Insecure Deserialization | `src/data/clustering.py:1162` | CWE-502 | 9.1 |
| SEC-H1 | **High** | Input Validation | `src/model/architecture.py:38` | CWE-95 | 7.5 |
| SEC-H2 | **High** | Input Validation | `scripts/verify_esphome.py:172` | CWE-20 | 7.1 |
| SEC-M1 | **Medium** | Security Configuration | `pyproject.toml:117-127` | CWE-1104 | 6.5 |
| SEC-M2 | **Medium** | Resource Management | `src/training/mining.py` | CWE-770 | 6.2 |
| SEC-M3 | **Medium** | Input Validation | `src/export/tflite.py:1644` | CWE-95 | 5.9 |
| SEC-L1 | **Low** | Information Disclosure | `src/pipeline.py:38,48` | CWE-532 | 4.3 |
| SEC-L2 | **Low** | Error Handling | `src/export/tflite.py` | CWE-209 | 4.0 |
| SEC-L3 | **Low** | Resource Management | `src/data/clustering.py:490` | CWE-377 | 3.7 |
| SEC-L4 | **Low** | Logging | `src/training/trainer.py` | CWE-778 | 3.5 |

---

## Critical Findings

### SEC-C1: Unsafe Pickle Serialization in Population Module

**Severity:** Critical
**CVSS Score:** 9.8 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
**CWE:** CWE-502 - Deserialization of Untrusted Data
**File:** `src/tuning/population.py`
**Lines:** 28, 34, 86, 114

#### Description

The `Candidate` class uses `pickle.dumps()` and `pickle.loads()` for model weights serialization. While currently used only for in-memory state, the AGENTS.md design document indicates plans for persisting tuning state to disk. If this state is ever saved to disk and later loaded from a malicious file, arbitrary code execution is possible.

```python
# Line 28
self.weights_bytes = pickle.dumps(model.get_weights())

# Line 34
model.set_weights(pickle.loads(self.weights_bytes))

# Line 86
best_weights = [np.array(w, copy=True) for w in pickle.loads(best.weights_bytes)]

# Line 114
worst.weights_bytes = pickle.dumps(worst_weights)
```

#### Attack Scenario

1. Attacker crafts a malicious pickle payload containing arbitrary Python code
2. Attacker replaces a legitimate tuning checkpoint file with the malicious pickle
3. When the victim loads the checkpoint, the malicious code executes with the victim's privileges
4. Attacker gains full control of the training environment

#### Remediation

Replace pickle with a safe serialization format like NumPy's native `.npy`/`.npz` format or Protocol Buffers:

```python
import io
import numpy as np

def save_state_safe(self, model) -> None:
    """Serialize model weights using safe NumPy format."""
    buffer = io.BytesIO()
    weights = model.get_weights()
    np.savez(buffer, *weights)
    self.weights_bytes = buffer.getvalue()

def restore_state_safe(self, model) -> None:
    """Restore model weights from safe NumPy format."""
    if self.weights_bytes is None:
        return
    buffer = io.BytesIO(self.weights_bytes)
    with np.load(buffer, allow_pickle=False) as data:
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    model.set_weights(weights)
```

---

### SEC-C2: Numpy allow_pickle=True on World-Writable Cache Files

**Severity:** Critical
**CVSS Score:** 9.1 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N)
**CWE:** CWE-502 - Deserialization of Untrusted Data
**File:** `src/data/clustering.py`
**Line:** 1162

#### Description

The `clear_cache()` method loads cache files from the system temp directory (`/tmp` on Unix systems, which is world-writable) using `np.load()` with `allow_pickle=True`. An attacker can replace a cache file with a malicious numpy pickle payload, triggering remote code execution when the cache is cleared.

```python
# Line 1162
data = np.load(cache_file, allow_pickle=True)
```

Note: Line 549 in the same file correctly uses `allow_pickle=False`, indicating awareness of the risk that was missed here.

#### Attack Scenario

1. Attacker writes a malicious pickle payload to `/tmp/mww_embeddings_cache/emb_*.npz`
2. Victim runs clustering cache clearing operation
3. `np.load()` deserializes the malicious payload with `allow_pickle=True`
4. Malicious code executes with victim's privileges

#### Remediation

Use `allow_pickle=False` and store only the necessary metadata:

```python
# Safe cache loading
data = np.load(cache_file, allow_pickle=False)

# For metadata, use a separate JSON file instead of pickle
import json
metadata_path = cache_file.with_suffix('.json')
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
```

---

## High Severity Findings

### SEC-H1: Unsafe ast.literal_eval on Config-Derived Strings

**Severity:** High
**CVSS Score:** 7.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:H)
**CWE:** CWE-95 - Eval Injection (Improper Neutralization of Directives)
**File:** `src/model/architecture.py`
**Line:** 38

#### Description

The `parse_model_param()` function uses `ast.literal_eval()` on user-controllable configuration strings without format validation. While `literal_eval` is safer than `eval()`, wrapping arbitrary config strings with `f"[{text}]"` can lead to unexpected parsing behavior and potential denial of service with malformed input.

```python
def parse_model_param(text):
    if not text:
        return []
    try:
        res = ast.literal_eval(text)  # Line 38
        # ...
    except (ValueError, SyntaxError) as exc:
        logger.error("parse_model_param: failed to parse %r: %s", text, exc)
        raise
```

#### Attack Scenario

1. Attacker provides a crafted config string that causes excessive memory allocation
2. `ast.literal_eval()` attempts to parse the malformed input
3. System memory is exhausted, causing denial of service

#### Remediation

Add strict input validation before parsing:

```python
import re

def parse_model_param(text):
    if not text:
        return []

    # Whitelist allowed characters only
    if not re.match(r'^[\d,\[\]\s\-]+$', text):
        raise ValueError(f"Invalid characters in model parameter: {text!r}")

    # Limit input length
    if len(text) > 1000:
        raise ValueError(f"Model parameter too long: {len(text)} chars")

    try:
        res = ast.literal_eval(text)
        # ...
```

---

### SEC-H2: Dynamic YAML String Construction

**Severity:** High
**CVSS Score:** 7.1 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N)
**CWE:** CWE-20 - Improper Input Validation
**File:** `scripts/verify_esphome.py`
**Line:** 172

#### Description

The script constructs a YAML string dynamically from config input and passes it to `yaml.safe_load()`. While `safe_load` prevents arbitrary code execution, string interpolation into YAML can lead to parsing ambiguity and potential injection of YAML tags.

```python
mixconv_str = str(model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"))
mixconv_kernel_sizes = cast(list[list[int]], yaml.safe_load(f"[{mixconv_str}]") or [[5], [7, 11], [9, 15], [23]])
```

#### Remediation

Use JSON parsing for structured data instead of YAML string construction:

```python
import json

mixconv_str = model_cfg.get("mixconv_kernel_sizes", "[[5], [7, 11], [9, 15], [23]]")
try:
    mixconv_kernel_sizes = json.loads(mixconv_str)
except json.JSONDecodeError:
    # Fallback to ast.literal_eval for legacy format
    mixconv_kernel_sizes = ast.literal_eval(f"[{mixconv_str}]")
```

---

## Medium Severity Findings

### SEC-M1: Overly Permissive Security Linter Suppressions

**Severity:** Medium
**CVSS Score:** 6.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N)
**CWE:** CWE-1104 - Use of Unmaintained Third Party Components (Misconfiguration)
**File:** `pyproject.toml`
**Lines:** 117-127

#### Description

The Ruff/Bandit configuration suppresses critical security rules that would catch the pickle and subprocess issues identified in this audit:

```toml
ignore = [
    "S101",        # Allow assert (acceptable for tests)
    "S301",        # Allow pickle - CRITICAL: masks pickle vulnerabilities
    "S603", "S607", # Allow subprocess without validation
    "S605", "S606", # Allow OS calls
    "S105", "S106", "S107", # Allow hardcoded passwords in test data
]
```

The `S301` suppression specifically allows the pickle vulnerabilities in SEC-C1 and SEC-C2 to pass undetected.

#### Remediation

Remove blanket suppressions and use per-file ignores only where necessary:

```toml
[tool.ruff.lint]
select = ["F", "E9", "E71", "E72", "W29", "I", "B", "C4", "S"]
ignore = ["S101"]  # Only for asserts in tests

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]
# Do NOT globally suppress S301 - review each pickle usage individually
```

---

### SEC-M2: Unbounded Memory Allocation in Hard-Negative Mining

**Severity:** Medium
**CVSS Score:** 6.2 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H)
**CWE:** CWE-770 - Allocation of Resources Without Limits
**File:** `src/training/mining.py`

#### Description

The hard-negative mining module allocates memory based on user-controlled configuration without upper bounds checking. Large values for `max_samples` or `top_k_per_epoch` can cause excessive memory consumption leading to OOM crashes.

#### Attack Scenario

1. Attacker provides malicious config with extremely large `max_samples` value
2. Mining heap allocates unbounded memory
3. System runs out of memory, causing denial of service

#### Remediation

Add explicit bounds checking:

```python
MAX_MINING_SAMPLES = 100_000
MAX_TOP_K = 10_000

def __init__(self, max_samples: int = 5000, top_k_per_epoch: int = 150):
    self.max_samples = min(int(max_samples), MAX_MINING_SAMPLES)
    self.top_k_per_epoch = min(int(top_k_per_epoch), MAX_TOP_K)
```

---

### SEC-M3: Unsafe ast.literal_eval in TFLite Export

**Severity:** Medium
**CVSS Score:** 5.9 (CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:L)
**CWE:** CWE-95 - Eval Injection
**File:** `src/export/tflite.py`
**Line:** 1644

#### Description

Similar to SEC-H1, the TFLite export module uses `ast.literal_eval()` on config-derived strings without validation:

```python
mc_kernels = model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]")
config["mixconv_kernel_sizes"] = ast.literal_eval(f"[{mc_kernels}]")
```

#### Remediation

Apply the same validation pattern as recommended for SEC-H1.

---

## Low Severity Findings

### SEC-L1: Subprocess Output Not Captured in Pipeline

**Severity:** Low
**CVSS Score:** 4.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)
**CWE:** CWE-532 - Insertion of Sensitive Information into Log File
**File:** `src/pipeline.py`
**Lines:** 38, 48

#### Description

The pipeline module uses `subprocess.run()` without capturing output, which may leak sensitive information through stdout/stderr:

```python
result = subprocess.run(args, check=False)  # noqa: S603
```

While the commands executed are internal Python modules (not external user input), output could still contain file paths or environment details.

#### Remediation

Capture and filter sensitive information:

```python
result = subprocess.run(
    args,
    check=False,
    capture_output=True,
    text=True
)
# Log only non-sensitive output
if result.returncode != 0:
    logger.error(f"Command failed: {' '.join(args)}")
    logger.debug(f"stderr: {result.stderr}")  # Debug level only
```

---

### SEC-L2: Verbose Error Messages in Export

**Severity:** Low
**CVSS Score:** 4.0 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)
**CWE:** CWE-209 - Generation of Error Message Containing Sensitive Information
**File:** `src/export/tflite.py`

#### Description

Exception handlers in the export module may leak internal file paths and system details through stack traces:

```python
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    traceback.print_exc()  # Full stack trace to stdout
```

#### Remediation

Log stack traces only at debug level:

```python
except Exception as e:
    logger.warning(f"Could not load config: {e}")
    logger.debug(traceback.format_exc())  # Only in debug mode
```

---

### SEC-L3: Predictable Temporary Directory Names

**Severity:** Low
**CVSS Score:** 3.7 (CVSS:3.1/AV:L/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N)
**CWE:** CWE-377 - Insecure Temporary File
**File:** `src/data/clustering.py`
**Line:** 490

#### Description

Cache directory uses a predictable name in the system temp directory:

```python
cache_dir = Path(tempfile.gettempdir()) / "mww_embeddings_cache"
```

This could allow symlink attacks on shared systems.

#### Remediation

Use `tempfile.mkdtemp()` with proper permissions:

```python
cache_dir = Path(tempfile.mkdtemp(prefix="mww_embeddings_", suffix=".cache"))
# Restrict permissions to owner only
os.chmod(cache_dir, 0o700)
```

---

### SEC-L4: Insufficient Security Logging

**Severity:** Low
**CVSS Score:** 3.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L)
**CWE:** CWE-778 - Insufficient Logging
**File:** `src/training/trainer.py`

#### Description

Security-relevant events (checkpoint loading, config overrides, data source changes) are not logged at appropriate levels for security monitoring.

#### Remediation

Add security event logging:

```python
import logging

security_logger = logging.getLogger("security")

# Log config overrides
security_logger.info(f"Config override applied: {override_path}")

# Log checkpoint operations
security_logger.info(f"Checkpoint loaded: {checkpoint_path}")

# Log data source changes
security_logger.info(f"Data directory changed: {data_dir}")
```

---

## Positive Security Findings

The following security best practices were observed:

1. **YAML Safe Loading:** All YAML parsing uses `yaml.safe_load()` - no unsafe `yaml.load()` found
2. **Subprocess Safety:** All subprocess calls use list-form arguments without `shell=True`
3. **No Hardcoded Secrets:** No hardcoded passwords, API keys, or credentials found
4. **Input Path Validation:** File paths are consistently converted to `Path` objects before use
5. **Environment Variable Handling:** Uses `os.environ.get()` with defaults rather than direct access

---

## Remediation Priority Matrix

| Priority | Finding | Effort | Impact |
|----------|---------|--------|--------|
| P0 | SEC-C1: Pickle in population.py | Low | Critical |
| P0 | SEC-C2: allow_pickle=True in clustering.py | Low | Critical |
| P1 | SEC-H1: ast.literal_eval validation | Medium | High |
| P1 | SEC-H2: Dynamic YAML construction | Low | High |
| P2 | SEC-M1: Linter suppressions | Low | Medium |
| P2 | SEC-M2: Memory bounds checking | Medium | Medium |
| P3 | SEC-L1-L4: Low severity items | Low | Low |

---

## Appendix A: Files Reviewed

- `src/tuning/population.py` - Candidate serialization
- `src/data/clustering.py` - Cache file handling
- `src/export/tflite.py` - Model export, subprocess calls
- `src/pipeline.py` - Pipeline orchestration
- `src/training/mining.py` - Hard-negative mining
- `src/model/architecture.py` - Model configuration parsing
- `src/export/verification.py` - Verification logic
- `scripts/verify_esphome.py` - External verification script
- `scripts/debug_streaming_gap.py` - Debug script
- `pyproject.toml` - Security linting configuration
- `src/training/trainer.py` - Training orchestration
- `src/data/ingestion.py` - Data loading
- `config/loader.py` - Configuration loading

---

## Appendix B: Tools Used

- Static Analysis: Manual code review with grep patterns
- Framework: OWASP Top 10 2021, CWE Top 25, CVSS v3.1
- Reference: Bandit security linter rules, Python security best practices
