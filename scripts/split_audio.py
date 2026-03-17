#!/usr/bin/env python3
"""Thin CLI wrapper — see src/data/preprocessing for logic."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocessing import remove_split_originals, scan_and_split

# Default configuration
DEFAULT_TARGET_DURATION = 2000.0
DEFAULT_MAX_DURATION = 3000.0
DEFAULT_MIN_DURATION = 500.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split long audio files into shorter clips for wake word training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more dataset directories to scan recursively",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        metavar="MS",
        help=f"Split files longer than this many milliseconds (default: {DEFAULT_MAX_DURATION})",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=DEFAULT_TARGET_DURATION,
        metavar="MS",
        help=f"Target clip length in milliseconds (default: {DEFAULT_TARGET_DURATION})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        metavar="MS",
        help=f"Discard remainder clips shorter than this (default: {DEFAULT_MIN_DURATION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be split without writing any files",
    )
    parser.add_argument(
        "--remove-originals",
        action="store_true",
        help="Remove original files that have already been split (run this after verifying the output clips)",
    )

    args = parser.parse_args()
    dirs = [Path(d) for d in args.dir]

    if args.remove_originals:
        print("Removing original files that were already split...")
        remove_split_originals(dirs, args.max_duration)
        return

    total_long, total_written, total_discarded, total_skipped = scan_and_split(
        directories=dirs,
        max_duration_ms=args.max_duration,
        target_duration_ms=args.target_duration,
        min_duration_ms=args.min_duration,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Long files found (>{args.max_duration:.0f}ms): {total_long}")
    if args.dry_run:
        print(f"  Clips that would be written:  {total_written}")
    else:
        print(f"  Clips written:                {total_written}")
    print(f"  Files skipped (unreadable or already split): {total_skipped}")
    print(f"  Files that produced no clips (all too short): {total_discarded}")
    if not args.dry_run and total_written > 0:
        print("\n  NOTE: Original files were NOT deleted.")
        print("  After verifying the splits look correct, you may remove")
        print("  the originals with:")
        if args.target_duration != DEFAULT_TARGET_DURATION or args.max_duration != DEFAULT_MAX_DURATION:
            max_dur_arg = f"--max-duration {args.max_duration}" if args.max_duration != DEFAULT_MAX_DURATION else ""
            target_dur_arg = f"--target-duration {args.target_duration}" if args.target_duration != DEFAULT_TARGET_DURATION else ""
            args_str = f"{max_dur_arg} {target_dur_arg}".strip()
            print(f"\n    python scripts/split_audio.py --dir <DIR> --remove-originals {args_str}")
        else:
            print("\n    python scripts/split_audio.py --dir <DIR> --remove-originals")
    print("=" * 60)


if __name__ == "__main__":
    main()
