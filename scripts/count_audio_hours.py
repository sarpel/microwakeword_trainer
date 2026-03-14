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
        return info.duration
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
    parser.add_argument("--config", type=str, help="Path to config file (to read paths from)")
    parser.add_argument("--negative-dir", type=str, help="Path to negative audio directory")
    parser.add_argument("--background-dir", type=str, help="Path to background audio directory")
    parser.add_argument("--hard-negative-dir", type=str, help="Path to hard negative audio directory")
    parser.add_argument("--update-config", type=str, help="Update the specified config file with correct ambient_duration_hours")

    args = parser.parse_args()

    # Load paths from config if provided
    if args.config:
        try:
            import yaml

            with open(args.config) as f:
                config = yaml.safe_load(f)

            paths = config.get("paths", {})
            negative_dir = Path(paths.get("negative_dir", "./dataset/negative"))
            background_dir = Path(paths.get("background_dir", "./dataset/background"))
            hard_negative_dir = Path(paths.get("hard_negative_dir", "./dataset/hard_negative"))
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        negative_dir = Path(args.negative_dir) if args.negative_dir else None
        background_dir = Path(args.background_dir) if args.background_dir else None
        hard_negative_dir = Path(args.hard_negative_dir) if args.hard_negative_dir else None

    print("=" * 70)
    print("AUDIO DURATION CALCULATOR")
    print("=" * 70)
    print()

    total_hours = 0.0

    # Count negative directory
    if negative_dir and negative_dir.exists():
        print(f"Scanning: {negative_dir}")
        count, seconds = scan_directory(negative_dir)
        hours = seconds / 3600
        total_hours += hours
        print(f"  Files: {count}")
        print(f"  Duration: {seconds:.1f}s = {hours:.2f} hours")
        print()
    elif args.negative_dir:
        print(f"Warning: Directory not found: {negative_dir}")
        print()

    # Count background directory
    if background_dir and background_dir.exists():
        print(f"Scanning: {background_dir}")
        count, seconds = scan_directory(background_dir)
        hours = seconds / 3600
        total_hours += hours
        print(f"  Files: {count}")
        print(f"  Duration: {seconds:.1f}s = {hours:.2f} hours")
        print()
    elif args.background_dir:
        print(f"Warning: Directory not found: {background_dir}")
        print()

    # Count hard negative directory
    if hard_negative_dir and hard_negative_dir.exists():
        print(f"Scanning: {hard_negative_dir}")
        count, seconds = scan_directory(hard_negative_dir)
        hours = seconds / 3600
        total_hours += hours
        print(f"  Files: {count}")
        print(f"  Duration: {seconds:.1f}s = {hours:.2f} hours")

        print()

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal ambient audio (negative + background + hard_negative): {total_hours:.2f} hours")
    print()
    print("RECOMMENDATION:")
    print(f"  Set ambient_duration_hours to: {total_hours:.2f}")
    print()
    print("In your config YAML, add:")
    print("  training:")
    print(f"    ambient_duration_hours: {total_hours:.2f}")
    print()

    # Update config file if requested
    if args.update_config:
        try:
            import yaml

            with open(args.update_config) as f:
                config = yaml.safe_load(f)

            if "training" not in config:
                config["training"] = {}

            old_value = config["training"].get("ambient_duration_hours", "NOT SET")
            config["training"]["ambient_duration_hours"] = round(total_hours, 2)

            with open(args.update_config, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"✅ Updated {args.update_config}")
            print(f"   Changed ambient_duration_hours from {old_value} to {total_hours:.2f}")
            print()
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
