#!/usr/bin/env python3
"""
Count total audio duration in your dataset directories.

Usage:
  python count_audio_hours.py --config config/presets/standard.yaml
  python count_audio_hours.py --negative-dir ./dataset/negative --background-dir ./dataset/background --hard-negative-dir ./dataset/hard_negative

This calculates the correct value for `ambient_duration_hours` in your config.
"""

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile not installed. Install with: pip install soundfile")
    sys.exit(1)


def get_audio_duration(filepath: Path) -> float:
    """Get duration of audio file in seconds."""
    try:
        info = sf.info(str(filepath))
        return float(info.duration)
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return 0.0


def scan_directory(directory: Path, extensions: tuple = (".wav", ".mp3", ".flac", ".ogg")) -> tuple:
    """Scan directory for audio files and return (count, total_seconds)."""
    if not directory.exists():
        return 0, 0.0

    total_seconds = 0.0
    file_count = 0

    for ext in extensions:
        for filepath in directory.rglob(f"*{ext}"):
            duration = get_audio_duration(filepath)
            if duration > 0:
                total_seconds += duration
                file_count += 1

    return file_count, total_seconds


def main():
    parser = argparse.ArgumentParser(description="Count total audio hours in dataset")
    parser.add_argument("--config", help="Path to config file (to read directories)")
    parser.add_argument("--negative-dir", help="Path to negative samples directory")
    parser.add_argument("--background-dir", help="Path to background noise directory")
    parser.add_argument("--hard-negative-dir", help="Path to hard negatives directory")
    args = parser.parse_args()

    directories = []

    if args.config:
        try:
            import yaml

            with open(args.config) as f:
                config = yaml.safe_load(f)
            paths = config.get("paths", {})
            if "negative_dir" in paths:
                directories.append(Path(paths["negative_dir"]))
            if "background_dir" in paths:
                directories.append(Path(paths["background_dir"]))
            if "hard_negative_dir" in paths:
                directories.append(Path(paths["hard_negative_dir"]))
        except Exception as e:
            print(f"Error reading config: {e}")
            sys.exit(1)

    if args.negative_dir:
        directories.append(Path(args.negative_dir))
    if args.background_dir:
        directories.append(Path(args.background_dir))
    if args.hard_negative_dir:
        directories.append(Path(args.hard_negative_dir))

    if not directories:
        print("Error: No directories specified. Use --config or specify directories directly.")
        sys.exit(1)

    total_files = 0
    total_seconds = 0.0

    print("\nScanning directories...")
    for directory in directories:
        if directory.exists():
            count, seconds = scan_directory(directory)
            total_files += count
            total_seconds += seconds
            hours = seconds / 3600
            print(f"  {directory}: {count} files, {hours:.2f} hours")
        else:
            print(f"  {directory}: does not exist, skipping")

    total_hours = total_seconds / 3600
    print(f"\nTotal: {total_files} files, {total_hours:.2f} hours")
    print(f"\nRecommended ambient_duration_hours: {total_hours:.1f}")


if __name__ == "__main__":
    main()
