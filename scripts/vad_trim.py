#!/usr/bin/env python3
"""Thin CLI wrapper — see src/data/preprocessing for logic."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocessing import (
    SpeechPreprocessConfig,
    process_background_directory,
    process_speech_directory,
)


def _print_summary(
    speech_results: list,
    bg_results: list,
    dry_run: bool,
    min_dur: float,
    max_dur: float,
    discarded_root: Path,
) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""

    s_kept = [r for r in speech_results if r.action == "keep"]
    s_trimmed = [r for r in speech_results if r.action == "trim"]
    s_split = [r for r in speech_results if r.action == "split"]
    s_discarded = [r for r in speech_results if r.action == "discard"]
    s_skipped = [r for r in speech_results if r.action == "skip"]
    s_out_clips = len(s_kept) + len(s_trimmed) + sum(int(r.reason) for r in s_split if r.reason.isdigit())

    b_kept = [r for r in bg_results if r.action == "keep"]
    b_split = [r for r in bg_results if r.action == "split"]
    b_skipped = [r for r in bg_results if r.action == "skip"]
    b_out_clips = len(b_kept) + sum(int(r.reason) for r in b_split if r.reason.isdigit())

    total_in = len(speech_results) + len(bg_results)
    total_out = s_out_clips + b_out_clips
    total_disc = len(s_discarded)

    print()
    print("=" * 62)
    print(f"{prefix}Summary  (range: {min_dur:.0f}–{max_dur:.0f}ms)")
    print("=" * 62)

    if speech_results:
        print("\n  Speech dirs (VAD trim + discard if out-of-range):")
        print(f"    input files:                  {len(speech_results):>8,}")
        print(f"    kept     (no trim needed):    {len(s_kept):>8,}")
        print(f"    trimmed  (fits in range):     {len(s_trimmed):>8,}")
        print(f"    discarded → {discarded_root}/:  {len(s_discarded):>8,}")
        print(f"    skipped  (errors / _parts):   {len(s_skipped):>8,}")

    if bg_results:
        print("\n  Background dirs (split-only, no VAD):")
        print(f"    input files:                  {len(bg_results):>8,}")
        print(f"    kept (≤{max_dur:.0f}ms):              {len(b_kept):>8,}")
        print(f"    split (>{max_dur:.0f}ms):             {len(b_split):>8,}")
        print(f"    skipped:                      {len(b_skipped):>8,}")

    print("\n  ── Grand total ────────────────────────────────────────────")
    print(f"    input files:     {total_in:>8,}")
    print(f"    output clips:    {total_out:>8,}  (Δ={total_out - total_in:+,})")
    print(f"    moved to {discarded_root}/:  {total_disc:>8,}")
    if dry_run:
        print("\n  ⚠  DRY-RUN: No files were modified.")
        print("  Re-run with --apply to apply changes.")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAD-trim speech files and split background files for wake word training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--speech-dirs",
        nargs="+",
        default=[],
        metavar="DIR",
        help="Speech dirs to VAD-trim (negative/, hard_negative/, positive/). Files outside range are moved to --discarded-dir.",
    )
    parser.add_argument(
        "--bg-dirs",
        nargs="+",
        default=[],
        metavar="DIR",
        help="Background dirs to split-only, no VAD (background/). Files >max-duration are split into max-duration chunks in-place.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run (no files touched).")
    parser.add_argument("--min-duration", type=float, default=300.0, metavar="MS", help="Discard speech clips shorter than this after VAD trim (default: 300ms)")
    parser.add_argument("--max-duration", type=float, default=2000.0, metavar="MS", help="Discard speech clips longer than this after VAD trim; split background at this length (default: 2000ms)")
    parser.add_argument("--pad-ms", type=int, default=200, metavar="MS", help="Silence padding around detected speech region (default: 200ms)")
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3], metavar="N", help="webrtcvad aggressiveness 0-3 (default: 2)")
    parser.add_argument("--discarded-dir", type=str, default="discarded", metavar="DIR", help="Root dir for moved out-of-range speech files (default: discarded/)")
    parser.add_argument("--verbose", action="store_true", help="Print a line for every file including kept/trimmed")

    args = parser.parse_args()
    if not args.speech_dirs and not args.bg_dirs:
        parser.error("Provide at least one of --speech-dirs or --bg-dirs")

    discarded_root = Path(args.discarded_dir)
    if args.apply:
        print("=" * 62)
        print("  ⚠  WARNING: --apply is set.")
        print("  Speech files outside range will be MOVED to discarded/.")
        print("  Background files >max-duration will be SPLIT in-place")
        print("  and their originals DELETED.")
        print("  Make sure you have a backup / git history first.")
        print("=" * 62)
        answer = input("  Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return
    else:
        print("DRY-RUN mode — no files will be touched. Use --apply to apply changes.")

    config = SpeechPreprocessConfig(
        min_duration_ms=args.min_duration,
        max_duration_ms=args.max_duration,
        pad_ms=args.pad_ms,
        vad_aggressiveness=args.aggressiveness,
    )

    dry_run = not args.apply
    speech_results: list = []
    bg_results: list = []

    for raw_dir in args.speech_dirs:
        d = Path(raw_dir)
        if not d.exists():
            print(f"[WARN] Not found, skipping: {d}")
            continue
        speech_results.extend(process_speech_directory(d, config, discarded_root, dry_run))

    for raw_dir in args.bg_dirs:
        d = Path(raw_dir)
        if not d.exists():
            print(f"[WARN] Not found, skipping: {d}")
            continue
        bg_results.extend(process_background_directory(d, args.max_duration, discarded_root, dry_run))

    _print_summary(speech_results, bg_results, dry_run, args.min_duration, args.max_duration, discarded_root)


if __name__ == "__main__":
    main()
