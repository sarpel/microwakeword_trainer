#!/usr/bin/env python3
"""Thin CLI wrapper — see src/data/quality for logic."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.quality import QualityScoreConfig, apply_discard, print_summary, score_directory, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast audio quality scoring (WADA-SNR + clipping). No new deps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dirs", nargs="+", required=True, metavar="DIR", help="Directories to score (e.g. dataset/negative dataset/positive)")
    parser.add_argument("--apply", action="store_true", help="Move discarded files. Default is dry-run.")
    parser.add_argument("--discard-bottom", type=float, default=5.0, metavar="PCT", help="Discard bottom N%% by WQI within each dir (default: 5.0)")
    parser.add_argument("--min-wqi", type=float, default=0.0, metavar="SCORE", help="Absolute WQI floor — discard any file below this (default: 0.0 = off)")
    parser.add_argument("--clip-threshold", type=float, default=0.001, metavar="RATIO", help="Clipping ratio hard gate (default: 0.001 = 0.1%%)")
    parser.add_argument("--discarded-dir", type=str, default="discarded/quality", metavar="DIR", help="Root dir for discarded files (default: discarded/quality)")
    parser.add_argument("--csv", type=str, default="", metavar="FILE", help="Write scores CSV to this path (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print per-file scores")

    args = parser.parse_args()
    discarded_root = Path(args.discarded_dir)
    dirs = [Path(d) for d in args.dirs]

    if args.apply:
        print("=" * 64)
        print("  ⚠  WARNING: --apply is set.")
        print("  Files in the bottom percentile / above clip threshold")
        print("  will be MOVED to discarded/quality/.")
        print("  Make sure you have a backup / git history first.")
        print("=" * 64)
        answer = input("  Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return
    else:
        print("DRY-RUN mode — no files moved. Use --apply to apply.")

    config = QualityScoreConfig(
        clip_threshold=args.clip_threshold,
        discard_bottom_pct=args.discard_bottom,
        min_wqi=args.min_wqi,
        discarded_dir=discarded_root,
        verbose=args.verbose,
    )

    all_results = []
    for d in dirs:
        if not d.exists():
            print(f"[WARN] Not found, skipping: {d}")
            continue
        all_results.extend(score_directory(d, config, mode="fast"))

    if args.csv:
        write_csv(all_results, Path(args.csv))
        print(f"\nCSV written: {args.csv}  ({len(all_results):,} rows)")

    dry_run = not args.apply
    print_summary(all_results, config, dry_run=dry_run, mode="fast")

    if not dry_run:
        moved = apply_discard(all_results, discarded_root, dry_run=False)
        print(f"\nMoved {moved:,} files → {discarded_root}/")


if __name__ == "__main__":
    main()
