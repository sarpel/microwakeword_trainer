# src/tools/ - Speaker Clustering CLI Tools

Speaker clustering pipeline: dry-run analysis and file organization by speaker identity.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `cluster_analyze.py` | 375 | Dry-run cluster analysis |
| `cluster_apply.py` | 364 | File organization by clusters |

## Entry Points

```python
mww-cluster-analyze = src.tools.cluster_analyze:main
mww-cluster-apply   = src.tools.cluster_apply:main
```

## mww-cluster-analyze

**Read-only dry-run.** Analyzes audio files, clusters by speaker, saves report.

```bash
mww-cluster-analyze --config standard --dataset all --n-clusters 200
```

Output: `cluster_output/{dataset}_namelist.json` + `{dataset}_cluster_report.txt`

## mww-cluster-apply

**Mutates files.** Organizes audio into per-speaker subdirectories.

```bash
mww-cluster-apply --namelist-dir cluster_output --dry-run  # Preview first
mww-cluster-apply --namelist-dir cluster_output            # Execute
mww-cluster-apply --undo cluster_output/positive_backup_manifest.json
```

## Critical Constraints

- **PyTorch environment required** - SpeechBrain ECAPA-TDNN needs `mww-torch`
- **Hugging Face login required** - Run `huggingface-cli login` once
- **Always use --dry-run first** - Preview before executing moves

## Anti-Patterns

- **Don't run apply without reviewing report** - Check `*_cluster_report.txt` first
- **Don't delete backup manifests** - Only way to undo file moves
- **Don't run in TF environment** - Requires PyTorch

## Related Documentation

- [Training Guide](../../docs/TRAINING.md)
