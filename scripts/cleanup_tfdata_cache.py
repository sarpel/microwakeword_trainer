#!/usr/bin/env python3
"""Clean up TensorFlow tf.data cache tempstate files safely.

TensorFlow's tf.data.Dataset.cache() creates large tempstate files during training.
These should be cleaned up automatically but may persist if training crashes.

Usage:
    python scripts/cleanup_tfdata_cache.py                    # Dry run (shows what would be deleted)
    python scripts/cleanup_tfdata_cache.py --delete           # Actually delete files
    python scripts/cleanup_tfdata_cache.py --path ./cache     # Custom cache directory
"""

import argparse
import sys
from pathlib import Path


def find_tempstate_files(cache_dir: str) -> list[Path]:
    """Find all tf.data tempstate files in cache directory."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    # Look for tempstate files
    patterns = [
        "*.tempstate*",
        "*.lockfile",
    ]

    files: list[Path] = []
    for pattern in patterns:
        files.extend(cache_path.glob(pattern))

    # Also check for incomplete data shard files (data-*.tempstate*)
    for ext in [".data", ".index"]:
        files.extend(cache_path.glob(f"*{ext}.tempstate*"))

    return sorted(set(files))


def get_file_size_gb(path: Path) -> float:
    """Get file size in GB."""
    return path.stat().st_size / (1024**3)


def main():
    parser = argparse.ArgumentParser(description="Clean up TensorFlow tf.data cache tempstate files")
    parser.add_argument(
        "--path",
        type=str,
        default="./cache/tfdata",
        help="Path to tf.data cache directory (default: ./cache/tfdata)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the files (default: dry run)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also delete completed cache files (not just tempstate)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.path)

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        # Try to find cache files elsewhere
        alt_paths = ["./data/processed", ".", "./cache"]
        for alt in alt_paths:
            alt_path = Path(alt)
            if alt_path.exists():
                temp_files = list(alt_path.glob("**/*.tempstate*"))
                if temp_files:
                    print(f"\nFound tempstate files in: {alt}")
                    print("Run with: --path {}".format(alt))
                    break
        return

    # Find tempstate files
    temp_files = find_tempstate_files(str(cache_dir))

    if not temp_files:
        print(f"No tempstate files found in: {cache_dir}")
        return

    # Calculate total size
    total_gb = sum(get_file_size_gb(f) for f in temp_files)

    print(f"Found {len(temp_files)} tempstate file(s) in: {cache_dir}")
    print(f"Total size: {total_gb:.2f} GB\n")

    for f in temp_files:
        size_gb = get_file_size_gb(f)
        print(f"  {f.name} ({size_gb:.2f} GB)")

    if args.delete:
        print(f"\nDeleting {len(temp_files)} file(s)...")
        deleted = 0
        for f in temp_files:
            try:
                f.unlink()
                deleted += 1
            except OSError as e:
                print(f"  Error deleting {f}: {e}")
        print(f"Deleted {deleted} file(s), freed {total_gb:.2f} GB")
    else:
        print("\nDry run - no files deleted.")
        print(f"To actually delete, run: python {sys.argv[0]} --path {cache_dir} --delete")


if __name__ == "__main__":
    main()
