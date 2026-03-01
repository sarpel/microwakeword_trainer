#!/usr/bin/env python3
"""Count and analyze dataset statistics."""

import concurrent.futures
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(file_path)], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def count_category(category_name: str, category_path: Path, max_workers: int = None) -> tuple:
    """Count files and calculate durations in a category folder using parallel processing."""
    if not category_path.exists():
        return 0, 0.0

    # Collect all audio files first
    audio_files = []
    for root, _, files in os.walk(category_path):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac")):
                audio_files.append(Path(root) / file)

    if not audio_files:
        return 0, 0.0

    # Use all available cores if not specified
    if max_workers is None:
        max_workers = os.cpu_count() or 8

    # Process files in parallel
    file_count = len(audio_files)
    total_duration = 0.0

    print(f"  Processing {file_count:,} files with {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(get_audio_duration, str(f)): f for f in audio_files}

        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            duration = future.result()
            total_duration += duration
            completed += 1

            # Progress indicator
            if completed % 500 == 0 or completed == file_count:
                percentage = (completed / file_count) * 100
                print(f"    {category_name}: {completed:,}/{file_count:,} ({percentage:.1f}%)")

    return file_count, total_duration


def main():
    """Main analysis function."""
    dataset_root = Path("dataset")

    # Category definitions
    categories = {"Positive": "positive", "Negative": "negative", "Hard Negative": "hard_negative", "Background": "background", "RIRs": "rirs"}

    results = defaultdict(lambda: {"count": 0, "duration": 0.0})

    # Analyze each category
    print("Analyzing dataset...")
    for display_name, folder_name in categories.items():
        print(f"\nProcessing {display_name}...")
        category_path = dataset_root / folder_name
        count, duration = count_category(display_name, category_path)
        results[display_name] = {"count": count, "duration": duration}

    # Calculate totals
    total_count = sum(r["count"] for r in results.values())
    total_duration = sum(r["duration"] for r in results.values())
    results["TOTAL"] = {"count": total_count, "duration": total_duration}

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Category':<20} {'File Count':<15} {'Total Duration (Min)':<22} {'Avg Duration (Sec)':<18}")
    print("=" * 80)

    # Order: TOTAL first, then the rest
    order = ["TOTAL"] + [k for k in results.keys() if k != "TOTAL"]

    for category in order:
        data = results[category]
        count = data["count"]
        duration_min = data["duration"] / 60
        avg_duration = data["duration"] / count if count > 0 else 0.0

        # Use bold for TOTAL
        if category == "TOTAL":
            # ANSI escape code: \033[1m = bold, \033[0m = reset
            category_display = f"\033[1m{category}\033[0m"
        else:
            category_display = category

        print(f"{category_display:<20} {count:>14,} {duration_min:>20.2f} {avg_duration:>18.2f}")

    print("=" * 80)
    print(f"\nAnalysis complete! Total files: {total_count:,}")


if __name__ == "__main__":
    main()
