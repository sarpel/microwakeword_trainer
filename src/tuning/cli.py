"""CLI entry point for mww-autotune command."""

import os

# Suppress verbose TF/XLA logs before importing tensorflow (via autotuner)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.tuning.autotuner import AutoTuner
from src.utils.logging_config import setup_rich_logging


def _configure_logging(verbose: bool = False) -> None:
    """Configure Rich logging for auto-tuning."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_rich_logging(level=level, show_time=True, show_path=True)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for mww-autotune."""
    parser = argparse.ArgumentParser(
        prog="mww-autotune",
        description="""
MaxQualityAutoTuner — post-training auto-tuning for wake word models.

Iteratively improves model quality to achieve target FAH/recall using:
  - 7 surgical strategy arms with Thompson sampling
  - 3-pass threshold optimization (quantile → exact → CV refinement)
  - Temperature scaling for probability calibration
  - Pareto archive (never regress on any objective)
  - INT8 shadow evaluation (optional)
  - Simulated annealing acceptance criterion
  - Confirmation phase for final candidate validation

Quality is prioritized over speed. Time cost does not matter.
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
        "--users-hard-negs",
        type=str,
        default=None,
        help="Path to user's custom hard negative audio files (overrides config paths.hard_negative_dir during tuning)",
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

    parser.add_argument(
        "--max-gradient-steps",
        type=int,
        default=None,
        help="Max gradient steps per burst (overrides config auto_tuning.max_gradient_steps)",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Cross-validation folds for threshold refinement (overrides config auto_tuning.cv_folds)",
    )

    parser.add_argument(
        "--no-confirmation",
        action="store_true",
        help="Skip confirmation phase (overrides config auto_tuning.require_confirmation)",
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
    table.add_row("Target FAH", f"< {at.get('target_fah', 2.0)}")
    table.add_row("Target Recall", f"> {at.get('target_recall', 0.90)}")
    table.add_row("Max Iterations", str(at.get("max_iterations", 50)))
    table.add_row("Max Gradient Steps", str(at.get("max_gradient_steps", 250_000)))
    table.add_row("CV Folds", str(at.get("cv_folds", 3)))
    table.add_row("Confirmation", str(at.get("require_confirmation", True)))
    table.add_row("Output Dir", str(at.get("output_dir", "./tuning_output")))

    panel = Panel(
        table,
        title="Auto-Tune Configuration",
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

    # Configure Rich logging for all project logs
    _configure_logging(verbose=args.verbose)

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
    if args.max_gradient_steps is not None:
        at["max_gradient_steps"] = args.max_gradient_steps
    if args.cv_folds is not None:
        at["cv_folds"] = args.cv_folds
    if args.no_confirmation:
        at["require_confirmation"] = False

    # Print summary
    print_config_summary(args, config_dict)

    # Dry run
    if args.dry_run:
        console.print("\n[yellow]Dry run mode - configuration validated successfully[/]")
        return 0

    # Confirm
    console.print("\n[cyan]Starting auto-tuning...[/]")
    console.print("[dim]Quality is prioritized over speed. Time cost does not matter.[/]")
    console.print()

    # Run auto-tuning
    try:
        tuner = AutoTuner(
            checkpoint_path=args.checkpoint,
            config=config_dict,
            auto_tuning_config=at,
            console=console,
            users_hard_negs_dir=args.users_hard_negs,
        )
        results = tuner.tune()

        # Display final results summary
        if results:
            console.print()
            from rich.panel import Panel
            from rich.rule import Rule

            console.print(Rule("[bold]Final Results[/bold]", style="cyan"))

            # Extract final metrics
            best_recall = results.get("best_recall", 0.0)
            best_fah = results.get("best_fah", 0.0)
            threshold = results.get("recommended_probability_cutoff", 0.5)
            checkpoint = results.get("best_checkpoint", "N/A")

            console.print(
                Panel(
                    f"[bold]Best Metrics:[/bold]\n"
                    f"  Detection: {best_recall * 100:.1f}%\n"
                    f"  False Alarms/hr: {best_fah:.2f}\n"
                    f"  Probability Cutoff: {threshold:.4f}\n"
                    f"  Best Checkpoint: {checkpoint}",
                    title="Configuration",
                    border_style="cyan",
                )
            )

        return 0  # Success even if target not met

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
