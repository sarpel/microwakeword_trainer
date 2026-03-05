"""Post-training hard negative mining CLI tool.

This tool reads the false predictions JSON log created during training
and copies the identified false positive files to the mined subdirectory
for post-training fine-tuning.

Usage:
    mww-mine-hard-negatives --prediction-log logs/false_predictions.json
    mww-mine-hard-negatives --prediction-log logs/false_predictions.json --min-epoch 15 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for deduplication.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of file hash
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_prediction_log(log_path: Path) -> dict[str, Any]:
    """Load the false predictions JSON log file.

    Args:
        log_path: Path to the JSON log file

    Returns:
        Dictionary containing the log data

    Raises:
        FileNotFoundError: If log file doesn't exist
        json.JSONDecodeError: If log file is invalid JSON
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Prediction log not found: {log_path}")

    with open(log_path, "r") as f:
        return json.load(f)


def filter_epochs_by_min_epoch(log_data: dict[str, Any], min_epoch: int) -> dict[int, dict]:
    """Filter epochs to only include those >= min_epoch.

    Args:
        log_data: The loaded log data
        min_epoch: Minimum epoch number to include

    Returns:
        Dictionary of epoch_number -> epoch_data
    """
    filtered = {}
    epochs = log_data.get("epochs", {})

    for epoch_str, epoch_data in epochs.items():
        try:
            epoch_num = int(epoch_str)
            if epoch_num >= min_epoch:
                filtered[epoch_num] = epoch_data
        except ValueError:
            console.print(f"[yellow]Warning: Skipping invalid epoch key '{epoch_str}' (not an integer)[/yellow]")
            continue

    return filtered


def collect_false_predictions(
    filtered_epochs: dict[int, dict],
    top_k: int,
) -> list[dict]:
    """Collect false predictions from filtered epochs.

    Args:
        filtered_epochs: Dictionary of epoch_number -> epoch_data
        top_k: Maximum predictions per epoch

    Returns:
        List of false prediction entries with epoch, index, score
    """
    all_predictions = []

    for epoch_num, epoch_data in sorted(filtered_epochs.items()):
        predictions = epoch_data.get("false_predictions", [])

        # Sort by score descending and take top_k
        sorted_preds = sorted(predictions, key=lambda x: x.get("score", 0), reverse=True)
        top_predictions = sorted_preds[:top_k]

        for pred in top_predictions:
            pred["epoch"] = epoch_num
            all_predictions.append(pred)

    return all_predictions


def deduplicate_by_hash(
    predictions: list[dict],
    dataset_dir: Path = Path("."),
) -> tuple[list[dict], dict[str, str]]:
    """Deduplicate predictions by file hash.

    Args:
        predictions: List of prediction entries
        dataset_dir: Base directory for dataset (to resolve relative paths, unused currently)

    Returns:
        Tuple of (unique_predictions, hash_to_path mapping)
    """
    seen_hashes: dict[str, str] = {}
    unique_predictions = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing file hashes...", total=len(predictions))

        for pred in predictions:
            # Check if prediction contains path information
            file_path_str = pred.get("file_path") or pred.get("path")
            if not file_path_str:
                console.print("[red]Error: Deduplication requires file_path or path in prediction log[/red]")
                console.print("[yellow]Regenerate logs with path information to enable deduplication[/yellow]")
                raise ValueError("Prediction log missing 'file_path' or 'path' field")

            file_path = Path(file_path_str)

            if not file_path.exists():
                progress.advance(task)
                continue

            file_hash = compute_file_hash(file_path)

            if file_hash not in seen_hashes:
                seen_hashes[file_hash] = str(file_path)
                pred["file_path"] = str(file_path)
                pred["file_hash"] = file_hash
                unique_predictions.append(pred)

            progress.advance(task)

    return unique_predictions, seen_hashes


def copy_files_to_mined_dir(
    predictions: list[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Copy false positive files to the mined directory.

    Args:
        predictions: List of prediction entries with file_path
        output_dir: Destination directory
        dry_run: If True, don't actually copy files

    Returns:
        Tuple of (copied_count, skipped_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for pred in predictions:
        file_path = Path(pred.get("file_path", ""))
        if not file_path.exists():
            skipped += 1
            continue

        # Generate destination filename with score
        score = pred.get("score", 0.0)
        epoch = pred.get("epoch", 0)
        dest_name = f"{file_path.stem}_e{epoch}_s{score:.3f}{file_path.suffix}"
        dest_path = output_dir / dest_name

        if dry_run:
            logger.info(f"[DRY RUN] Would copy: {file_path} -> {dest_path}")
            copied += 1
        else:
            try:
                shutil.copy2(file_path, dest_path)
                logger.debug(f"Copied: {file_path} -> {dest_path}")
                copied += 1
            except Exception as e:
                logger.warning(f"Failed to copy {file_path}: {e}")
                skipped += 1

    return copied, skipped


def generate_summary(
    total_epochs: int,
    filtered_epochs: int,
    total_predictions: int,
    unique_predictions: int,
    copied: int,
    skipped: int,
    output_dir: Path,
) -> None:
    """Generate a summary table of the mining operation.

    Args:
        total_epochs: Total epochs in log
        filtered_epochs: Epochs after filtering
        total_predictions: Total false predictions found
        unique_predictions: Unique predictions after deduplication
        copied: Number of files copied
        skipped: Number of files skipped
        output_dir: Output directory path
    """
    table = Table(title="Hard Negative Mining Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Epochs in Log", str(total_epochs))
    table.add_row("Epochs After Filtering", str(filtered_epochs))
    table.add_row("Total False Predictions", str(total_predictions))
    table.add_row("Unique Predictions", str(unique_predictions))
    table.add_row("Files Copied", str(copied))
    table.add_row("Files Skipped", str(skipped))
    table.add_row("Output Directory", str(output_dir))

    console.print()
    console.print(table)


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Post-training hard negative mining tool. Reads false predictions log and copies files for fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  mww-mine-hard-negatives --prediction-log logs/false_predictions.json

  # Dry run to see what would be copied
  mww-mine-hard-negatives --prediction-log logs/false_predictions.json --dry-run

  # Only use epochs >= 15 with deduplication
  mww-mine-hard-negatives --prediction-log logs/false_predictions.json --min-epoch 15 --deduplicate
        """,
    )

    parser.add_argument(
        "--prediction-log",
        type=Path,
        required=True,
        help="Path to the false_predictions.json log file created during training",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (optional, for verification)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dataset/hard_negative/mined"),
        help="Output directory for mined hard negatives (default: ./dataset/hard_negative/mined)",
    )

    parser.add_argument(
        "--min-epoch",
        type=int,
        default=10,
        help="Minimum epoch to consider (default: 10)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top K files per epoch (default: 100)",
    )

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Enable hash-based deduplication",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load the prediction log
        console.print(f"[bold blue]Loading prediction log: {args.prediction_log}[/bold blue]")
        log_data = load_prediction_log(args.prediction_log)

        metadata = log_data.get("metadata", {})
        total_epochs = len(log_data.get("epochs", {}))

        console.print(f"Found {total_epochs} epochs in log")
        console.print(f"FP Threshold: {metadata.get('fp_threshold', 'unknown')}")

        # Filter epochs
        filtered_epochs = filter_epochs_by_min_epoch(log_data, args.min_epoch)
        console.print(f"[bold blue]Filtering epochs >= {args.min_epoch}[/bold blue]")
        console.print(f"Epochs after filtering: {len(filtered_epochs)}")

        if not filtered_epochs:
            console.print("[yellow]No epochs match the filter criteria. Exiting.[/yellow]")
            return 0

        # Collect false predictions
        console.print(f"[bold blue]Collecting top-{args.top_k} false predictions per epoch...[/bold blue]")
        predictions = collect_false_predictions(filtered_epochs, args.top_k)
        console.print(f"Total predictions collected: {len(predictions)}")

        if not predictions:
            console.print("[yellow]No false predictions found. Exiting.[/yellow]")
            return 0

        # Deduplicate if requested
        if args.deduplicate:
            console.print("[bold blue]Deduplicating predictions by file hash...[/bold blue]")
            try:
                unique_predictions, _ = deduplicate_by_hash(predictions)
                console.print(f"[green]Deduplicated {len(predictions)} -> {len(unique_predictions)} predictions[/green]")
            except ValueError as e:
                console.print(f"[red]Deduplication failed: {e}[/red]")
                console.print("[yellow]Continuing without deduplication[/yellow]")
                unique_predictions = predictions
        else:
            unique_predictions = predictions

        # Copy files
        console.print(f"[bold blue]{'Simulating' if args.dry_run else 'Copying'} files to {args.output_dir}...[/bold blue]")
        copied, skipped = copy_files_to_mined_dir(unique_predictions, args.output_dir, args.dry_run)

        # Generate summary
        generate_summary(
            total_epochs=total_epochs,
            filtered_epochs=len(filtered_epochs),
            total_predictions=len(predictions),
            unique_predictions=len(unique_predictions),
            copied=copied,
            skipped=skipped,
            output_dir=args.output_dir,
        )

        if args.dry_run:
            console.print("\n[bold yellow]This was a dry run. No files were actually copied.[/bold yellow]")
            console.print("Run without --dry-run to copy files.")

        return 0

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return 1
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error: Invalid JSON in prediction log: {e}[/bold red]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
