# Security Guidelines

This document outlines security best practices and known security considerations for the microwakeword_trainer project.

## Security Status

**Overall Security Grade: B+**

All Critical and High severity vulnerabilities from the March 2026 security audit have been addressed.

## Addressed Vulnerabilities

### ✅ Fixed (March 2026)

| CVE | Issue | Fix | Severity |
|-----|-------|-----|----------|
| CVE-2026-32274 | Black 26.1.0 arbitrary file write | Updated to 26.3.1 | 5.3 MEDIUM |
| SEC-001 | Pickle RCE in population.py | Replaced with `np.savez` | 9.8 CRITICAL |
| SEC-002 | Unsafe cache loading | `allow_pickle=False` | 9.1 CRITICAL |
| SEC-003 | Security rules suppressed | Re-enabled S301 | 8.6 HIGH |
| SEC-005 | Path traversal | Path validation added | 7.1 HIGH |

### Secure Development Practices

#### File Operations (CWE-22)

All CLI tools now use path validation via `resolve_path_safe()`:

```python
from src.utils.path_utils import resolve_path_safe

# Validates paths and prevents ../ traversal
safe_path = resolve_path_safe(user_input, base_dir=Path("/data"))
```

**Protected tools:**
- `mww-export` - Checkpoint, config, data-dir, output paths
- `mww-train` - Config, override paths
- `mww-autotune` - Checkpoint, config, output-dir
- `mww-mine-hard-negatives` - All path arguments
- `mww-cluster-*` - Cluster analysis/apply tools

#### Serialization Security

**Safe practices:**
- ✅ Uses `np.savez`/`np.load(allow_pickle=False)` for model weights
- ✅ Uses `yaml.safe_load()` for YAML configuration
- ✅ Uses list-based subprocess arguments (no `shell=True`)

**Prohibited patterns:**
- ❌ `pickle.dumps/pickle.loads()` - RCE vulnerability
- ❌ `yaml.load()` / `yaml.unsafe_load()` - Arbitrary code execution
- ❌ `subprocess.run(shell=True)` - Shell injection

#### Automated Security Scanning

**CI/CD pipeline includes:**
- **Bandit** - Python security linter
- **Safety** - Dependency vulnerability scanner
- **TruffleHog** - Secret scanner
- **Gitleaks** - Additional secret scanner
- **Ruff S-rules** - Security-focused linting

## Security Configuration

### Environment Variables

```bash
# Disable TF verbose logging (security best practice)
export TF_CPP_MIN_LOG_LEVEL=3
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"

# Python security
export PYTHONHASHSEED=random
export PYTHONNOUSERSITE=1
```

### Dependency Management

**Security scans run on:**
- Every pull request
- Every push to main/v* branches
- Manual workflow dispatch

**Automatic blocking:**
- PRs with Critical findings require approval
- High severity issues trigger warnings

## Threat Model

### Attack Surfaces

| Surface | Risk | Mitigation |
|---------|------|------------|
| Config files | Malicious YAML | `yaml.safe_load()` + schema validation |
| Checkpoint files | Pickle RCE | `np.savez` + `allow_pickle=False` |
| CLI arguments | Path traversal | `resolve_path_safe()` validation |
| Dependencies | Known CVEs | Automated scanning + PR blocking |
| Dataset loading | File traversal | Path sanitization in ingestion.py |

### Assumptions

1. **Trusted environment:** Training data is assumed to be from trusted sources
2. **Local execution:** No network exposure during training
3. **Config trust:** YAML configs are read from trusted locations

## Reporting Security Issues

To report a security vulnerability:

1. **Do NOT open a public issue**
2. Email details to: tsarpel15@gmail.com
3. Include:
   - Vulnerability description
   - Proof of concept (if applicable)
   - Suggested fix
   - Your contact information

**Response time:** Within 48 hours for Critical/High severity issues

## Security Checklist

Before deploying to production:

- [ ] Run `pip install -e .[dev]` to ensure all dependencies
- [ ] Run `bandit -r src/` to check for security issues
- [ ] Run `safety check` to verify dependencies
- [ ] Review `docs/SECURITY.md` for current guidelines
- [ ] Verify no hardcoded secrets in code
- [ ] Check that CI/CD security scans pass
- [ ] Ensure model is exported with INT8 quantization
- [ ] Verify TFLite model with `python scripts/verify_esphome.py`

## Version History

| Date | Change |
|------|--------|
| 2026-03-19 | Added path validation, CI/CD security scanning |
| 2026-03-18 | Fixed pickle RCE vulnerabilities |
| 2026-03-18 | Fixed unsafe cache loading |

---

**Last Updated:** March 19, 2026
**Next Review:** After next dependency update
