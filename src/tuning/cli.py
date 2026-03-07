"""CLI entry point for mww-autotune command."""

from __future__ import annotations

import os

# Suppress verbose TF/XLA logs before importing tensorflow (via autotuner)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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

Iteratively improves model quality to achieve target metrics using:
- Multi-phase optimization (FAH reduction → balanced → recall → polish)
- Adaptive knob selection with impact memory
- Pareto frontier tracking (never regress)
- Multi-threshold evaluation (not hardcoded 0.5)

Reads targets from config auto_tuning section, overridable via CLI args.
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
        default=None,
        help="Target FAH value (overrides config auto_tuning.target_fah)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Target recall value (overrides config auto_tuning.target_recall)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum tuning iterations (overrides config auto_tuning.max_iterations)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tuned checkpoints (overrides config auto_tuning.output_dir)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Iterations without improvement before strategy escalation (overrides config auto_tuning.patience)",
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

    # Validate target values (only check if explicitly provided)
    if args.target_fah is not None and args.target_fah <= 0:
        console.print("[red]Error: Target FAH must be positive[/]")
        return False

    if args.target_recall is not None and not 0 < args.target_recall <= 1:
        console.print("[red]Error: Target recall must be between 0 and 1[/]")
        return False

    if args.max_iterations is not None and args.max_iterations < 1:
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

    # Show auto_tuning config values (with CLI overrides applied)
    at = config.get("auto_tuning", {})
    table.add_row("Target FAH", f"< {at.get('target_fah', 0.3)}")
    table.add_row("Target Recall", f"> {at.get('target_recall', 0.92)}")
    table.add_row("Max Iterations", str(at.get("max_iterations", 100)))
    table.add_row("Patience", str(at.get("patience", 10)))
    table.add_row("Steps/Iteration", str(at.get("steps_per_iteration", 5000)))
    table.add_row("Output Dir", str(at.get("output_dir", "./tuning_output")))

    panel = Panel(
        table,
        title="\U0001f527 Auto-Tune Configuration",
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

    # Apply CLI overrides to auto_tuning section
    at = config_dict.setdefault("auto_tuning", {})
    if args.target_fah is not None:
        at["target_fah"] = args.target_fah
    if args.target_recall is not None:
        at["target_recall"] = args.target_recall
    if args.max_iterations is not None:
        at["max_iterations"] = args.max_iterations
    if args.output_dir is not None:
        at["output_dir"] = args.output_dir
    if args.patience is not None:
        at["patience"] = args.patience

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
            auto_tuning_config=at,
            console=console,
        )

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
        result_table.add_row("Target Met", "\u2705 Yes" if result["target_met"] else "\u274c No")
        result_table.add_row("Best Checkpoint", str(result["best_checkpoint"] or "N/A"))
        result_table.add_row("Pareto Points", str(len(result.get("pareto_frontier", []))))

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
