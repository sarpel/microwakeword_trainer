#!/usr/bin/env python3
"""Phonetic similarity scorer for wake word hard negatives.

This script scores hard-negative audio filenames by phonetic similarity to a wake
word (default: "Hey Katya") using IPA conversion + articulatory feature distance.

It helps you identify risky hard negatives that are too phonetically close to the
wake word and optionally move them into a quarantine directory.

Usage examples:

  # Score an entire hard-negative directory
  python scripts/phonetic_scorer.py score dataset/hard_negative

  # Score recursively with a custom wake word and save report
  python scripts/phonetic_scorer.py score dataset/hard_negative \
      --wake-word "Hey Katya" \
      --output logs/phonetic_scores.json

  # Preview moving HIGH and MEDIUM risk files from report
  python scripts/phonetic_scorer.py move \
      --report logs/phonetic_scores.json \
      --risk-levels HIGH,MEDIUM \
      --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

DEFAULT_WAKE_WORD = "Hey Katya"
WAKE_WORD_PRONUNCIATIONS: dict[str, list[str]] = {"Hey Katya": ["heɪ kɑtjə", "heɪ kætjə", "heɪ kɑːtjɑ", "heɪ kɑtjɑ"]}
RISK_HIGH = 0.82
RISK_MEDIUM = 0.72
DEFAULT_QUARANTINE_DIR = Path("./data/quarantined_hard_negatives")
DEFAULT_OUTPUT_PATH = Path("logs/phonetic_scores.json")
TOOL_VERSION = "1.1.0"
DEFAULT_LANGUAGE = "tur-Latn"

# Mapping from common short codes to epitran language codes
LANGUAGE_ALIASES: dict[str, str] = {
    "en": "eng-Latn",
    "eng": "eng-Latn",
    "english": "eng-Latn",
    "tr": "tur-Latn",
    "tur": "tur-Latn",
    "turkish": "tur-Latn",
    "de": "deu-Latn",
    "deu": "deu-Latn",
    "german": "deu-Latn",
    "fr": "fra-Latn",
    "fra": "fra-Latn",
    "french": "fra-Latn",
    "es": "spa-Latn",
    "spa": "spa-Latn",
    "spanish": "spa-Latn",
    "it": "ita-Latn",
    "ita": "ita-Latn",
    "italian": "ita-Latn",
    "nl": "nld-Latn",
    "nld": "nld-Latn",
    "dutch": "nld-Latn",
    "pt": "por-Latn",
    "por": "por-Latn",
    "portuguese": "por-Latn",
}

# Languages that require external tools (flite lex_lookup)
LANGUAGES_REQUIRING_FLITE = {"eng-Latn"}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


@dataclass
class PhoneticScore:
    path: str
    filename: str
    text: str
    ipa: str
    score: float
    risk: str
    best_match_ipa: str


def _install_instructions(missing: list[str]) -> str:
    missing_display = ", ".join(sorted(missing))
    return f"Missing optional dependency/dependencies: {missing_display}\n\nInstall phonetic dependencies with one of:\n  pip install '.[phonetic]'\n  pip install epitran panphon jellyfish"


def _lazy_import(module_name: str):
    """Import a module by name, supporting dotted submodules (e.g. 'panphon.distance')."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _resolve_language(language: str) -> str:
    """Resolve a language shorthand (e.g. 'en') to an epitran code (e.g. 'eng-Latn')."""
    normalized = language.strip().lower()
    return LANGUAGE_ALIASES.get(normalized, language)


def _check_flite_for_english(language: str) -> None:
    """Validate epitran can handle the requested language. eng-Latn needs lex_lookup."""
    if language not in LANGUAGES_REQUIRING_FLITE:
        return
    # lex_lookup is NOT in the Ubuntu 'flite' package — it must be compiled from source.
    # Test if epitran can actually produce output for this language.
    epitran_mod = _lazy_import("epitran")
    if epitran_mod is None:
        return  # will be caught later by missing-dependency check
    try:
        epi = epitran_mod.Epitran(language)
        epi.transliterate("test")
    except Exception:
        raise RuntimeError(
            f"Language '{language}' requires the 'lex_lookup' binary (part of CMU Flite).\n"
            f"The Ubuntu 'flite' package does NOT include it — you must compile flite from source:\n"
            f"  git clone https://github.com/festvox/flite && cd flite && ./configure && make\n"
            f"  gcc -o bin/lex_lookup testsuite/lex_lookup_main.c -I include -L build/x86_64-linux-gnu/lib -lflite_cmulex -lflite_cmu_us_kal -lflite -lasound -lm\n"
            f"  sudo cp bin/lex_lookup /usr/local/bin/\n"
            f"Or use a language that works out of the box, e.g. --language tur-Latn (recommended for 'Katya')\n"
            f"Supported aliases: {', '.join(sorted(LANGUAGE_ALIASES.keys()))}"
        )


def check_dependencies(console: Console | None = None) -> bool:
    """Verify scoring dependencies are available (epitran + panphon)."""
    console = console or Console()
    missing: list[str] = []

    if _lazy_import("epitran") is None:
        missing.append("epitran")
    if _lazy_import("panphon") is None:
        missing.append("panphon")

    if missing:
        console.print(
            Panel(
                _install_instructions(missing),
                title="Dependency Error",
                border_style="red",
            )
        )
        return False

    return True


def filename_to_text(filename: str) -> str:
    """Convert filename to normalized text for phonetic analysis.

    Examples:
      hey_katia_sample_003.wav -> hey katia
      okay-google-v2.wav       -> okay google
      cat_meow_recording.wav   -> cat meow
      hey_cut_hair_20260128_215718.wav -> hey cut hair
    """
    stem = Path(filename).stem
    stem = re.sub(r"[-_]v?\d+$", "", stem)
    stem = re.sub(
        r"[-_]?(sample|recording|clip|audio|rec|take|noise|ambient)[-_]?\d*$",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    # Strip date-timestamp patterns (YYYYMMDD, HHMMSS) that pollute phonetic analysis
    stem = re.sub(r"[-_]?\d{8}[-_]?\d{4,6}", "", stem)
    stem = re.sub(r"[-_]?\d{8}$", "", stem)
    # Strip isolated hash/id-like tokens (e.g. d38z5, odq5z, 2eiww, 21m00)
    # Must contain both letters and digits to avoid eating real words like 'kayla'
    stem = re.sub(r"[-_](?=[a-z0-9]{4,6}[-_])(?=[^-_]*\d)(?=[^-_]*[a-z])[a-z0-9]{4,6}(?=[-_])", "", stem)
    # Strip trailing standalone numbers (segment indices like _2, _36, _66)
    stem = re.sub(r"[-_]\d{1,3}$", "", stem)
    text = re.sub(r"[-_]+", " ", stem).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text or "unknown"


def text_to_ipa(text: str, language: str = DEFAULT_LANGUAGE, epi: Any | None = None) -> str:
    """Convert text phrase to IPA using epitran."""
    if not text.strip():
        return ""

    if epi is None:
        epitran = _lazy_import("epitran")
        if epitran is None:
            raise RuntimeError(_install_instructions(["epitran"]))
        language = _resolve_language(language)
        _check_flite_for_english(language)
        epi = epitran.Epitran(language)

    ipa = epi.transliterate(text)
    ipa = re.sub(r"\s+", " ", ipa).strip()
    return ipa


def compute_phonetic_similarity(ipa1: str, ipa2: str, distance_obj: Any | None = None) -> float:
    """Compute IPA similarity in [0, 1].

    Primary: panphon feature edit distance converted to similarity.
    Fallback: jellyfish Jaro-Winkler similarity.
    """
    if not ipa1 or not ipa2:
        return 0.0

    panphon_distance = _lazy_import("panphon.distance")
    if panphon_distance is not None:
        try:
            if distance_obj is None:
                distance_obj = panphon_distance.Distance()
            distance = distance_obj.feature_edit_distance_div_maxlen(ipa1, ipa2)
            similarity = 1.0 - float(distance)
            return max(0.0, min(1.0, similarity))
        except Exception:
            distance_obj = None
    jellyfish = _lazy_import("jellyfish")
    if jellyfish is not None:
        try:
            return max(0.0, min(1.0, float(jellyfish.jaro_winkler_similarity(ipa1, ipa2))))
        except Exception:
            return 0.0

    return 0.0


def score_against_wake_word(
    text_ipa: str,
    wake_word_variants: list[str],
    distance_obj: Any | None = None,
) -> tuple[float, str]:
    """Score IPA against wake word IPA variants, returning max score and match."""
    best_score = 0.0
    best_match = ""

    for variant in wake_word_variants:
        score = compute_phonetic_similarity(text_ipa, variant, distance_obj=distance_obj)
        if score > best_score:
            best_score = score
            best_match = variant

    return best_score, best_match


def classify_risk(score: float) -> str:
    """Classify score as HIGH / MEDIUM / LOW."""
    if score >= RISK_HIGH:
        return "HIGH"
    if score >= RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def score_file(
    filepath: Path,
    wake_word_variants: list[str],
    epi: Any,
    language: str = DEFAULT_LANGUAGE,
    distance_obj: Any | None = None,
) -> PhoneticScore:
    """Score a single audio file path by filename-derived text."""
    text = filename_to_text(filepath.name)
    ipa = text_to_ipa(text, language=language, epi=epi)
    score, best_match = score_against_wake_word(ipa, wake_word_variants, distance_obj=distance_obj)

    return PhoneticScore(
        path=str(filepath),
        filename=filepath.name,
        text=text,
        ipa=ipa,
        score=round(score, 4),
        risk=classify_risk(score),
        best_match_ipa=best_match,
    )


def _iter_audio_files(dirpath: Path, recursive: bool = True):
    iterator = dirpath.rglob("*") if recursive else dirpath.glob("*")
    for p in iterator:
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            yield p


def score_directory(
    dirpath: Path,
    wake_word_variants: list[str],
    recursive: bool = True,
    language: str = DEFAULT_LANGUAGE,
) -> list[PhoneticScore]:
    """Score all audio files in directory (or one file)."""
    epitran = _lazy_import("epitran")
    panphon_distance = _lazy_import("panphon.distance")
    if epitran is None or panphon_distance is None:
        missing = []
        if epitran is None:
            missing.append("epitran")
        if panphon_distance is None:
            missing.append("panphon")
        raise RuntimeError(_install_instructions(missing))

    language = _resolve_language(language)
    _check_flite_for_english(language)
    epi = epitran.Epitran(language)
    distance_obj = panphon_distance.Distance()

    files: list[Path]
    if dirpath.is_file():
        files = [dirpath]
    elif dirpath.is_dir():
        files = sorted(_iter_audio_files(dirpath, recursive=recursive))
    else:
        return []

    results: list[PhoneticScore] = []
    for filepath in files:
        try:
            results.append(
                score_file(
                    filepath=filepath,
                    wake_word_variants=wake_word_variants,
                    epi=epi,
                    language=language,
                    distance_obj=distance_obj,
                )
            )
        except Exception as e:
            logger.warning(f"Error scoring file {filepath}: {e}")
            results.append(
                PhoneticScore(
                    path=str(filepath),
                    filename=filepath.name,
                    text=filename_to_text(filepath.name),
                    ipa="",
                    score=0.0,
                    risk="LOW",
                    best_match_ipa="",
                )
            )

    return sorted(results, key=lambda r: r.score, reverse=True)


def write_json_report(
    results: list[PhoneticScore],
    output_path: Path,
    wake_word: str,
    variants: list[str],
    scan_root: str,
) -> None:
    """Write JSON report in mining-style structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    high_count = sum(1 for r in results if r.risk == "HIGH")
    medium_count = sum(1 for r in results if r.risk == "MEDIUM")
    low_count = sum(1 for r in results if r.risk == "LOW")
    unique_texts = len({r.text for r in results})

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": "phonetic_scorer",
        "version": TOOL_VERSION,
        "wake_word": wake_word,
        "wake_word_ipa_variants": variants,
        "thresholds": {"HIGH": RISK_HIGH, "MEDIUM": RISK_MEDIUM},
        "scan_root": scan_root,
        "summary": {
            "total_files_scanned": len(results),
            "unique_texts": unique_texts,
            "high_risk_count": high_count,
            "medium_risk_count": medium_count,
            "low_risk_count": low_count,
        },
        "results": [asdict(result) for result in results],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _risk_label(risk: str) -> str:
    if risk == "HIGH":
        return "🔴 HIGH"
    if risk == "MEDIUM":
        return "🟡 MED"
    return "🟢 LOW"


def print_rich_summary(
    results: list[PhoneticScore],
    console: Console,
    min_score: float = 0.0,
    verbose: bool = False,
) -> None:
    """Print grouped summary table (grouped by unique text)."""
    if not results:
        console.print(Panel("No files matched for scoring.", title="Phonetic Similarity", border_style="yellow"))
        return

    grouped: dict[str, list[PhoneticScore]] = {}
    for row in results:
        grouped.setdefault(row.text, []).append(row)

    aggregate_rows: list[tuple[str, str, float, str, int]] = []
    for text, rows in grouped.items():
        best = max(rows, key=lambda r: r.score)
        if best.score < min_score:
            continue
        if not verbose and best.risk == "LOW":
            continue
        aggregate_rows.append((text, best.ipa, best.score, best.risk, len(rows)))

    aggregate_rows.sort(key=lambda x: x[2], reverse=True)

    if not aggregate_rows:
        console.print(Panel("No rows matched display filters.", title="Phonetic Similarity", border_style="yellow"))
        return

    table = Table(title="Phonetic Similarity Scores", show_header=True, header_style="bold magenta")
    table.add_column("Text", style="cyan", max_width=28, overflow="fold")
    table.add_column("IPA", style="white", max_width=28, overflow="fold")
    table.add_column("Score", justify="right", min_width=5)
    table.add_column("Risk", justify="center", min_width=8)
    table.add_column("Files", justify="right", min_width=5)

    for text, ipa, score, risk, count in aggregate_rows:
        table.add_row(text, ipa or "-", f"{score:.2f}", _risk_label(risk), str(count))

    total = len(results)
    high = sum(1 for r in results if r.risk == "HIGH")
    medium = sum(1 for r in results if r.risk == "MEDIUM")
    low = sum(1 for r in results if r.risk == "LOW")

    console.print(table)
    console.print(
        Panel(
            f"Total files: {total}\nHigh risk: {high}\nMedium risk: {medium}\nLow risk: {low}",
            title="Summary",
            border_style="blue",
        )
    )


def _parse_risk_levels(raw: str) -> list[str]:
    levels = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"HIGH", "MEDIUM", "LOW"}
    invalid = [x for x in levels if x not in valid]
    if invalid:
        raise ValueError(f"Invalid risk levels: {', '.join(invalid)}")
    return levels or ["HIGH"]


def move_flagged_files(
    report_path: Path,
    quarantine_dir: Path,
    risk_levels: list[str],
    dry_run: bool,
    console: Console | None = None,
) -> int:
    """Move flagged files from a prior report while preserving relative paths."""
    console = console or Console()

    if not report_path.exists():
        console.print(Panel(f"Report file not found: {report_path}", title="Move Error", border_style="red"))
        return 1

    try:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(Panel(f"Invalid JSON report: {exc}", title="Move Error", border_style="red"))
        return 1

    rows = report.get("results", [])
    if not rows:
        console.print(Panel("Report contains no results to move.", title="Move", border_style="yellow"))
        return 0

    scan_root_raw = report.get("scan_root")
    scan_root = Path(scan_root_raw).resolve() if scan_root_raw else None

    candidates = [row for row in rows if str(row.get("risk", "")).upper() in set(risk_levels)]
    if not candidates:
        console.print(Panel("No rows matched selected risk levels.", title="Move", border_style="yellow"))
        return 0

    action = "WOULD MOVE" if dry_run else "MOVED"
    preview = Table(title="Flagged File Moves", show_header=True, header_style="bold cyan")
    preview.add_column("Risk", style="magenta")
    preview.add_column("Source", overflow="fold")
    preview.add_column("Destination", overflow="fold")

    moved_count = 0
    skipped_count = 0
    error_count = 0

    for row in candidates:
        src = Path(str(row.get("path", "")))
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()

        if not src.exists():
            skipped_count += 1
            continue

        try:
            if scan_root is not None:
                rel = src.relative_to(scan_root)
            else:
                rel = Path(src.name)
        except ValueError:
            rel = Path(src.name)

        dst = quarantine_dir / rel

        if dry_run:
            moved_count += 1
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved_count += 1
            except Exception:
                error_count += 1
                continue

        if preview.row_count < 30:
            preview.add_row(str(row.get("risk", "")), str(src), str(dst))

    if preview.row_count:
        console.print(preview)

    console.print(
        Panel(
            f"{action}: {moved_count}\nSkipped (missing): {skipped_count}\nErrors: {error_count}\nRisk levels: {', '.join(risk_levels)}\nQuarantine dir: {quarantine_dir}",
            title="Move Summary",
            border_style="blue",
        )
    )

    if not dry_run and moved_count > 0:
        report["moved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["moved_count"] = moved_count
        report["moved_risk_levels"] = risk_levels
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

    return 0


def _get_wake_word_ipa_variants(wake_word: str, language: str, console: Console) -> list[str]:
    if wake_word in WAKE_WORD_PRONUNCIATIONS:
        return WAKE_WORD_PRONUNCIATIONS[wake_word]

    if not check_dependencies(console):
        raise RuntimeError("Missing required dependencies for IPA generation.")

    epitran = _lazy_import("epitran")
    assert epitran is not None
    language = _resolve_language(language)
    _check_flite_for_english(language)
    epi = epitran.Epitran(language)
    ipa = text_to_ipa(wake_word, language=language, epi=epi)
    return [ipa] if ipa else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phonetic similarity scorer for wake word hard negatives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Commands:\n  score       Score hard negative files by phonetic similarity\n  move        Move flagged files from a previous score report\n"),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    score_parser = subparsers.add_parser("score", help="Score files by phonetic similarity")
    score_parser.add_argument("path", type=Path, help="Directory or file to score")
    score_parser.add_argument("--wake-word", type=str, default=DEFAULT_WAKE_WORD, help='Wake word phrase (default: "Hey Katya")')
    score_parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="epitran language code or alias (default: tur-Latn). Aliases: en, tr, de, fr, es, it, nl, pt")
    score_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="JSON report output path")
    score_parser.add_argument("--recursive", action="store_true", default=False, help="Scan directories recursively (default: False)")
    score_parser.add_argument("--min-score", type=float, default=0.0, help="Only show results above this score")
    score_parser.add_argument("--verbose", action="store_true", help="Show all results, not just flagged")

    move_parser = subparsers.add_parser("move", help="Move flagged files from a score report")
    move_parser.add_argument("--report", type=Path, required=True, help="Path to JSON report from score command")
    move_parser.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE_DIR, help="Where to move files")
    move_parser.add_argument(
        "--risk-levels",
        type=str,
        default="HIGH",
        help="Comma-separated risk levels to move (HIGH,MEDIUM,LOW)",
    )
    move_parser.add_argument("--dry-run", action="store_true", help="Preview moves without executing")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "score":
        if not check_dependencies(console):
            return 1

        target_path: Path = args.path
        if not target_path.exists():
            console.print(Panel(f"Path not found: {target_path}", title="Score Error", border_style="red"))
            return 1

        try:
            wake_variants = _get_wake_word_ipa_variants(args.wake_word, args.language, console)
            if not wake_variants:
                console.print(Panel("Could not build IPA variants for wake word.", title="Score Error", border_style="red"))
                return 1

            results = score_directory(
                dirpath=target_path,
                wake_word_variants=wake_variants,
                recursive=args.recursive,
                language=args.language,
            )

            scan_root = str(target_path.resolve() if target_path.is_dir() else target_path.resolve().parent)
            write_json_report(results, args.output, args.wake_word, wake_variants, scan_root=scan_root)
            print_rich_summary(results, console, min_score=args.min_score, verbose=args.verbose)

            console.print(
                Panel(
                    f"Report saved to: {args.output}\nWake word: {args.wake_word}\nIPA variants: {', '.join(wake_variants)}",
                    title="Score Complete",
                    border_style="green",
                )
            )
            return 0
        except Exception as exc:
            console.print(Panel(str(exc), title="Score Error", border_style="red"))
            return 1

    if args.command == "move":
        try:
            risk_levels = _parse_risk_levels(args.risk_levels)
        except ValueError as exc:
            console.print(Panel(str(exc), title="Move Error", border_style="red"))
            return 1

        return move_flagged_files(
            report_path=args.report,
            quarantine_dir=args.quarantine_dir,
            risk_levels=risk_levels,
            dry_run=args.dry_run,
            console=console,
        )

    console.print(Panel(f"Unknown command: {args.command}", title="Error", border_style="red"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
