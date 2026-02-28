# src/tools/

**Speaker Clustering CLI Tools** | Dry-run analysis and file organization by speaker identity.

## Overview

Two-command pipeline for ML-based speaker clustering. Run `mww-cluster-analyze` first (read-only, dry-run) to inspect cluster assignments, then `mww-cluster-apply` to physically organize files into per-speaker subdirectories. Uses the PyTorch environment (ECAPA-TDNN via SpeechBrain).

## Files

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `cluster_analyze.py` | 375 | Speaker cluster dry-run analysis | `discover_audio_files()`, `analyze_clusters()`, `print_cluster_report()`, `save_namelist_json()`, `save_cluster_report()`, `main()` |
| `cluster_apply.py` | 364 | File organization by speaker clusters | `load_namelist()`, `discover_namelists()`, `plan_moves()`, `preview_moves()`, `execute_moves()`, `undo_moves()`, `process_namelist()`, `main()` |
| `__init__.py` | — | Package init | |

## Entry Points

```python
# From setup.py console_scripts
mww-cluster-analyze = src.tools.cluster_analyze:main
mww-cluster-apply   = src.tools.cluster_apply:main
```

---

## mww-cluster-analyze

**Read-only dry-run.** Loads audio files, runs ECAPA-TDNN speaker embedding + clustering, prints a report, and saves two output files per dataset. No files are moved.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | **required** | Config preset name (`standard`, `fast_test`, `max_quality`) or path to YAML file |
| `--override` | string | None | Optional override YAML file on top of preset |
| `--dataset` | string | `positive` | Dataset(s) to cluster: `positive`, `negative`, `hard_negative`, `all` |
| `--n-clusters` | int | None | Explicit cluster count (overrides threshold). Use when you know approximate speaker count. Recommended for short wake word clips where threshold-based clustering over-fragments. |
| `--threshold` | float | from config | Override similarity threshold (default comes from `SpeakerClusteringConfig.similarity_threshold`) |
| `--output-dir` | string | `./cluster_output` | Directory for output files |
| `--max-files` | int | None | Limit number of files to process (for testing) |

### Output Files (per dataset)

```
cluster_output/
├── {dataset}_namelist.json         # file path → speaker_XXXX mapping (input for mww-cluster-apply)
└── {dataset}_cluster_report.txt    # Human-readable summary report
```

### Example Usage

```bash
# Cluster positive dataset (default)
mww-cluster-analyze --config standard

# Cluster all datasets at once
mww-cluster-analyze --config standard --dataset all

# Explicit speaker count (recommended for short clips)
mww-cluster-analyze --config standard --n-clusters 200

# Custom threshold
mww-cluster-analyze --config standard --threshold 0.65

# Limit files for quick testing
mww-cluster-analyze --config standard --max-files 100
```

---

## mww-cluster-apply

**Mutates files.** Reads namelist JSON(s) produced by `mww-cluster-analyze` and moves audio files into per-speaker subdirectories. Saves a backup manifest automatically before any moves. Supports undo via `--undo`.

### CLI Arguments

One of `--namelist`, `--namelist-dir`, or `--undo` is **required** (mutually exclusive):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--namelist` | string | — | Path to a single `*_namelist.json` file from `mww-cluster-analyze` |
| `--namelist-dir` | string | — | Directory containing `*_namelist.json` files (processes all) |
| `--undo` | string | — | Path to a backup manifest JSON to reverse a previous organization |
| `--output-dir` | string | `./cluster_output` | Directory for backup manifests |
| `--dry-run` | flag | off | Preview planned moves without moving any files |

### Backup Manifest

Before executing any moves, a backup manifest is saved to:
```
cluster_output/{dataset}_backup_manifest.json
```
Pass this to `--undo` to reverse the operation.

### Example Usage

```bash
# Preview first (recommended)
mww-cluster-apply --namelist cluster_output/positive_namelist.json --dry-run

# Organize a single dataset
mww-cluster-apply --namelist cluster_output/positive_namelist.json

# Organize all datasets at once
mww-cluster-apply --namelist-dir cluster_output

# Undo a previous organization
mww-cluster-apply --undo cluster_output/positive_backup_manifest.json
```

---

## Typical Workflow

```bash
# 1. Switch to PyTorch environment
mww-torch

# 2. Analyze clusters (dry-run, no files moved)
mww-cluster-analyze --config standard --dataset all --n-clusters 200

# 3. Review output
cat cluster_output/positive_cluster_report.txt

# 4. Preview file moves
mww-cluster-apply --namelist-dir cluster_output --dry-run

# 5. Execute
mww-cluster-apply --namelist-dir cluster_output

# 6. Undo if something looks wrong
mww-cluster-apply --undo cluster_output/positive_backup_manifest.json
```

---

## Critical Constraints

- **PyTorch environment required** — SpeechBrain ECAPA-TDNN runs in `mww-torch`, not `mww-tf`
- **Hugging Face login required** — Run `huggingface-cli login` once to accept model terms
- **No CPU fallback** — Embedding extraction benefits strongly from GPU; slow on CPU
- **`--namelist`, `--namelist-dir`, `--undo` are mutually exclusive** — pass exactly one
- **`mww-cluster-apply` moves files physically** — always use `--dry-run` first on production data
- **Backup manifests are per-dataset** — `positive_backup_manifest.json`, not a single combined file
- **Already-organized files are skipped** — `cluster_apply` detects `speaker_*` parent dirs and skips them

## Anti-Patterns

- **Don't run `mww-cluster-apply` without reviewing the report first** — check `*_cluster_report.txt` to verify speaker assignments make sense
- **Don't delete backup manifests** — they are the only way to undo file moves
- **Don't run in TF environment** — SpeechBrain requires PyTorch; wrong environment gives import errors
- **Don't mix `--n-clusters` and `--threshold` without intention** — `--n-clusters` overrides threshold; only one takes effect

## Notes

- `mww-cluster-analyze` is safe to run multiple times — it never moves files
- `cluster_apply` skips files that are not found or already organized into `speaker_*` subdirectories
- The `--dataset all` flag runs clustering on `positive`, `negative`, and `hard_negative` sequentially
- `SpeakerClustering` (from `src.data.clustering`) handles embedding extraction and cluster assignment — these tools are CLI wrappers
