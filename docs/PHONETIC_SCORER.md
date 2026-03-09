# Phonetic Scorer

Identifies hard negatives that are phonetically similar to your wake word. Files scoring above risk thresholds can be quarantined to prevent them from confusing the model during training.

## Prerequisites

```bash
pip install epitran panphon jellyfish
```

> **Note on English (`eng-Latn`)**: English requires CMU Flite's `lex_lookup` binary. Install with `sudo apt install flite`, or use `tur-Latn` (default) which works without extra system dependencies and handles "Katya" well.

## Quick Start

### 1. Score hard negatives

```bash
python scripts/phonetic_scorer.py score dataset/hard_negative/ \
    --output logs/phonetic_scores.json \
    --verbose
```

This scans all `.wav` files recursively, extracts text from filenames, converts to IPA, and computes phonetic similarity against the wake word variants.

### 2. Review the report

Open `logs/phonetic_scores.json` to inspect scored files. Each entry includes:
- `filename`: path to the WAV file
- `extracted_text`: text parsed from filename
- `ipa`: IPA transcription
- `max_similarity`: highest similarity score across all wake word variants
- `risk_level`: `HIGH` (≥0.82), `MEDIUM` (≥0.72), or `LOW` (<0.72)

### 3. Move flagged files

```bash
# Preview what would be moved (dry run)
python scripts/phonetic_scorer.py move \
    --report logs/phonetic_scores.json \
    --risk-levels HIGH,MEDIUM \
    --dry-run

# Execute the move
python scripts/phonetic_scorer.py move \
    --report logs/phonetic_scores.json \
    --risk-levels HIGH,MEDIUM
```

Files are moved to `./data/quarantined_hard_negatives/` by default. Use `--quarantine-dir` to change.

## Command Reference

### `score`

```
python scripts/phonetic_scorer.py score <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `path` | *(required)* | Directory or single file to score |
| `--wake-word` | `"Hey Katya"` | Wake word phrase |
| `--language` | `tur-Latn` | Epitran language code or alias |
| `--output` | `logs/phonetic_scores.json` | JSON report output path |
| `--recursive` | `True` | Scan subdirectories |
| `--min-score` | *(none)* | Only show files above this score |
| `--verbose` | `False` | Show all results, not just flagged |

### `move`

```
python scripts/phonetic_scorer.py move [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--report` | *(required)* | Path to JSON score report |
| `--quarantine-dir` | `./data/quarantined_hard_negatives/` | Destination for moved files |
| `--risk-levels` | `HIGH,MEDIUM` | Comma-separated risk levels to move |
| `--dry-run` | `False` | Preview without moving |

## Language Codes

The `--language` flag accepts either epitran codes directly or short aliases:

| Alias | Epitran Code | Notes |
|-------|-------------|-------|
| `tr` | `tur-Latn` | **Default.** No extra deps. Good for "Katya". |
| `en` | `eng-Latn` | Requires `lex_lookup` from flite |
| `de` | `deu-Latn` | |
| `fr` | `fra-Latn` | |
| `es` | `spa-Latn` | |
| `it` | `ita-Latn` | |
| `nl` | `nld-Latn` | |
| `pt` | `por-Latn` | |

## How Scoring Works

1. **Text extraction**: Filename is parsed (e.g. `hey_katya_loud.wav` → `"hey katya loud"`)
2. **IPA conversion**: Text is transliterated to IPA via epitran (e.g. `"hej katja"`)
3. **Similarity**: Panphon articulatory feature edit distance against wake word IPA variants. Falls back to jellyfish Jaro-Winkler if panphon fails.
4. **Risk classification**: HIGH ≥ 0.82, MEDIUM ≥ 0.72, LOW < 0.72

## Typical Workflow

```bash
# Score everything
python scripts/phonetic_scorer.py score dataset/hard_negative/ \
    --output logs/phonetic_scores.json --verbose

# Check what would be quarantined
python scripts/phonetic_scorer.py move \
    --report logs/phonetic_scores.json --dry-run

# Quarantine high-risk files
python scripts/phonetic_scorer.py move \
    --report logs/phonetic_scores.json --risk-levels HIGH

# Re-train without the confusing hard negatives
mww-train --config config/presets/max_quality.yaml
```
