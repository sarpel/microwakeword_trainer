"""Rich terminal UI for wake word model training.

Provides progress bars, metric tables, confusion matrices, and formatted logging.
"""

from typing import Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.table import Table
from rich.rule import Rule


class RichTrainingLogger:
    """Beautiful Rich terminal output for the wake word training pipeline.

    Handles all terminal display: progress bars, metric tables, confusion matrices,
    phase transitions, checkpoint logging, and completion summaries. Pure display
    module with no training logic or external dependencies.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        if console is None:
            self.console = Console(force_terminal=False)
        else:
            self.console = console

    def log_header(self, config: dict, total_steps: int) -> None:
        """Display a training configuration summary panel."""
        training = config["training"]
        steps = training["training_steps"]
        lrs = training["learning_rates"]

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")

        # Training phases
        for i, (s, lr) in enumerate(zip(steps, lrs)):
            table.add_row(
                f"Phase {i + 1}",
                f"{s:,} steps @ LR {lr:.6f}",
            )

        # Class weights (per-phase lists)
        pos_w = training.get("positive_class_weight", [1.0])
        neg_w = training.get("negative_class_weight", [20.0])
        hard_neg_w = training.get("hard_negative_class_weight", [40.0])
        table.add_row(
            "Class Weights",
            f"pos={pos_w}  neg={neg_w}  hard_neg={hard_neg_w}",
        )

        # Core training params
        table.add_row("Batch Size", str(training.get("batch_size", "N/A")))
        table.add_row(
            "Eval Interval",
            f"every {training.get('eval_step_interval', 'N/A')} steps",
        )
        table.add_row("Total Steps", f"{total_steps:,}")

        # Mixed precision
        perf = config.get("performance", {})
        mp = perf.get("mixed_precision", False)
        table.add_row(
            "Mixed Precision",
            "[green]Enabled[/green]" if mp else "[dim]Disabled[/dim]",
        )

        # SpecAugment (keys live under training, not augmentation)
        time_mask_sizes = training.get("time_mask_max_size", [0])
        time_mask_counts = training.get("time_mask_count", [0])
        freq_mask_sizes = training.get("freq_mask_max_size", [0])
        freq_mask_counts = training.get("freq_mask_count", [0])
        has_spec_aug = any(
            x > 0
            for x in time_mask_sizes
            + time_mask_counts
            + freq_mask_sizes
            + freq_mask_counts
        )
        if has_spec_aug:
            table.add_row(
                "SpecAugment",
                f"time_mask={time_mask_sizes} freq_mask={freq_mask_sizes}",
            )

        # Hard negative mining
        mining = config.get("hard_negative_mining", {})
        mining_enabled = mining.get("enabled", False)
        table.add_row(
            "Hard Neg Mining",
            "[green]Enabled[/green]" if mining_enabled else "[dim]Disabled[/dim]",
        )

        panel = Panel(
            table,
            title="🎯 Wake Word Training",
            border_style="blue",
            expand=False,
        )
        self.console.print(panel)

    def create_progress(self, total_steps: int) -> Tuple[Progress, TaskID]:
        """Create a Rich Progress bar with custom columns.

        Returns (progress, task_id). The caller manages the context manager.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[metrics]}"),
            console=self.console,
            disable=not self.console.is_terminal,
        )
        task_id = progress.add_task("Phase 1", total=total_steps, metrics="")
        return progress, task_id

    def update_step(
        self,
        progress: Progress,
        task_id: TaskID,
        step: int,
        metrics: dict,
        phase_info: dict,
    ) -> None:
        """Update the progress bar with current step metrics."""
        loss = metrics.get("loss", 0)
        accuracy = metrics.get("accuracy", 0)
        lr = phase_info["learning_rate"]
        phase = phase_info["phase"]

        progress.update(
            task_id,
            completed=step,
            description=f"Phase {phase + 1}",
            metrics=f"loss={loss:.4f} acc={accuracy:.4f} lr={lr:.6f}",
        )

    def log_validation_results(
        self, metrics: dict, step: int, total_steps: int
    ) -> None:
        """Display a validation results table with all available metrics."""
        table = Table(
            title=f"📊 Validation Results — Step {step}/{total_steps}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        # Ordered metric definitions: (key, label, format, style)
        metric_defs = [
            ("accuracy", "Accuracy", ".4f", None),
            ("precision", "Precision", ".4f", None),
            ("recall", "Recall", ".4f", None),
            ("f1_score", "F1 Score", ".4f", "bold green"),
            ("auc_roc", "AUC-ROC", ".4f", None),
            ("auc_pr", "AUC-PR", ".4f", None),
        ]

        for key, label, fmt, style in metric_defs:
            if key not in metrics:
                continue
            value = metrics[key]
            if value is None:
                continue
            formatted = f"{value:{fmt}}"
            if style:
                table.add_row(label, f"[{style}]{formatted}[/{style}]")
            else:
                table.add_row(label, formatted)

        # Ambient false positives per hour — color-coded
        if "ambient_false_positives_per_hour" in metrics:
            faph = metrics["ambient_false_positives_per_hour"]
            if faph <= 0.5:
                color = "green"
            elif faph <= 2.0:
                color = "yellow"
            else:
                color = "red"
            table.add_row(
                "Ambient FA/Hour",
                f"[{color}]{faph:.2f}[/{color}]",
            )

        # Additional operational metrics
        optional_metrics = [
            ("recall_at_no_faph", "Recall @ No FAPH", ".4f"),
            ("threshold_for_no_faph", "Threshold for No FAPH", ".4f"),
            ("average_viable_recall", "Avg Viable Recall", ".4f"),
        ]
        for key, label, fmt in optional_metrics:
            if key in metrics and metrics[key] is not None:
                table.add_row(label, f"{metrics[key]:{fmt}}")

        self.console.print(table)

    def log_confusion_matrix(self, tp: int, fp: int, tn: int, fn: int) -> None:
        """Display a confusion matrix table at threshold=0.5."""
        table = Table(
            title="Confusion Matrix (threshold=0.5)",
            show_header=True,
            header_style="bold",
        )
        table.add_column("", style="bold")
        table.add_column("Predicted Positive", justify="right")
        table.add_column("Predicted Negative", justify="right")

        table.add_row(
            "Actual Positive",
            f"[green]{tp:,}[/green]",
            f"[red]{fn:,}[/red]",
        )
        table.add_row(
            "Actual Negative",
            f"[red]{fp:,}[/red]",
            f"[green]{tn:,}[/green]",
        )

        total = tp + fp + tn + fn
        table.add_section()
        table.add_row("Total", "", f"[bold]{total:,}[/bold]")

        self.console.print(table)

    def log_phase_transition(
        self,
        phase: int,
        total_phases: int,
        lr: float,
        pos_weight: float,
        neg_weight: float,
    ) -> None:
        """Display a phase transition rule."""
        title = (
            f"Phase {phase + 1}/{total_phases} — "
            f"LR: {lr:.6f} | Weights: pos={pos_weight:.1f} neg={neg_weight:.1f}"
        )
        self.console.print(Rule(title=title, style="bold cyan"))

    def log_checkpoint(self, reason: str, is_best: bool, path: str = "") -> None:
        """Log a checkpoint save event."""
        if is_best:
            self.console.print(f"[bold green]✅ BEST MODEL[/] {reason}")
            if path:
                self.console.print(f"   [dim]→ {path}[/dim]")
        else:
            self.console.print(f"[dim]💾 Checkpoint:[/] {reason}")

    def log_completion(
        self,
        total_time: float,
        best_path: str,
        best_fah: float,
        best_recall: float,
    ) -> None:
        """Display the training completion summary panel."""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Total Time", time_str)
        table.add_row("Best FA/Hour", f"{best_fah:.2f}")
        table.add_row("Best Recall", f"{best_recall:.4f}")
        table.add_row("Best Weights", best_path)

        panel = Panel(
            table,
            title="🏁 Training Complete",
            border_style="green",
            expand=False,
        )
        self.console.print(panel)

    def log_mining(self, message: str, count: Optional[int] = None) -> None:
        """Log a hard negative mining event."""
        msg = f"[yellow]⛏️  Mining:[/] {message}"
        if count is not None:
            msg += f" — Found {count} hard negatives"
        self.console.print(msg)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.console.print(f"[bold yellow]⚠️  Warning:[/] {message}")

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.console.print(f"[blue]ℹ️  {message}[/]")
