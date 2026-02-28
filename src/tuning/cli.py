"""CLI entry point for mww-autotune command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.tuning.autotuner import AutoTuner


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for mww-autotune."""
    parser = argparse.ArgumentParser(
        prog="mww-autotune",
        description="""
Auto-tuning system for wake word model fine-tuning.

Iteratively improves model quality to achieve target metrics:
- FAH (False Activations per Hour) < 0.3
- Recall > 0.92

Uses hard negative mining and micro-config adjustments.
Time is not a constraint - only quality matters.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint to fine-tune",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        help="Config preset name or path to config file (default: standard)",
    )

    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override config file path",
    )

    parser.add_argument(
        "--target-fah",
        type=float,
        default=0.3,
        help="Target FAH value (default: 0.3)",
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.92,
        help="Target recall value (default: 0.92)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum tuning iterations (default: 100)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tuning",
        help="Output directory for tuned checkpoints (default: ./tuning)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Iterations without improvement before strategy switch (default: 10)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running tuning",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    console = Console()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        console.print(f"[red]Error: Checkpoint not found: {args.checkpoint}[/]")
        return False

    # Validate target values
    if args.target_fah <= 0:
        console.print("[red]Error: Target FAH must be positive[/]")
        return False

    if not 0 < args.target_recall <= 1:
        console.print("[red]Error: Target recall must be between 0 and 1[/]")
        return False

    if args.max_iterations < 1:
        console.print("[red]Error: Max iterations must be at least 1[/]")
        return False

    return True


def print_config_summary(args: argparse.Namespace, config: dict) -> None:
    """Print configuration summary."""
    console = Console()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Checkpoint", args.checkpoint)
    table.add_row("Config", args.config)
    table.add_row("Target FAH", f"{args.target_fah}")
    table.add_row("Target Recall", f"{args.target_recall}")
    table.add_row("Max Iterations", f"{args.max_iterations}")
    table.add_row("Output Directory", args.output_dir)
    table.add_row("Patience", f"{args.patience}")

    # Add training config summary
    training = config.get("training", {})
    if training:
        table.add_row("Batch Size", str(training.get("batch_size", "N/A")))
        table.add_row("Learning Rates", str(training.get("learning_rates", "N/A")))

    panel = Panel(
        table,
        title="🔧 Auto-Tune Configuration",
        border_style="blue",
        expand=False,
    )
    console.print(panel)


def main() -> int:
    """Main entry point for mww-autotune command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    console = Console()

    # Validate arguments
    if not validate_args(args):
        return 1

    # Load config
    try:
        from config.loader import load_full_config

        config = load_full_config(args.config, args.override)
        import dataclasses

        config_dict = dataclasses.asdict(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/]")
        return 1

    # Print summary
    print_config_summary(args, config_dict)

    # Dry run
    if args.dry_run:
        console.print("\n[yellow]Dry run mode - configuration validated successfully[/]")
        return 0

    # Confirm
    console.print("\n[cyan]Starting auto-tuning...[/]")
    console.print("[dim]This may take a long time. Quality is prioritized over speed.[/]")
    console.print()

    # Run auto-tuning
    try:
        tuner = AutoTuner(
            checkpoint_path=args.checkpoint,
            config=config_dict,
            output_dir=args.output_dir,
            target_fah=args.target_fah,
            target_recall=args.target_recall,
            max_iterations=args.max_iterations,
            console=console,
        )

        # Update patience in target
        tuner.target.patience = args.patience

        result = tuner.tune()

        # Print final results
        console.print("\n" + "=" * 80)
        console.print("[bold]TUNING COMPLETE[/]")
        console.print("=" * 80)

        result_table = Table(title="Final Results")
        result_table.add_column("Metric", style="bold")
        result_table.add_column("Value")

        result_table.add_row("Best FAH", f"{result['best_fah']:.4f}")
        result_table.add_row("Best Recall", f"{result['best_recall']:.4f}")
        result_table.add_row("Iterations", str(result["iterations"]))
        result_table.add_row("Target Met", "✅ Yes" if result["target_met"] else "❌ No")
        result_table.add_row("Best Checkpoint", str(result["best_checkpoint"] or "N/A"))

        console.print(result_table)

        return 0 if result["target_met"] else 0  # Success even if target not met

    except KeyboardInterrupt:
        console.print("\n[yellow]Auto-tuning interrupted by user[/]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error during auto-tuning: {e}[/]")
        if args.verbose:
            import traceback

            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
