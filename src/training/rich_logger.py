"""Rich terminal UI for wake word model training.

Provides progress bars, metric tables, confusion matrices, and formatted logging.
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table


class RichTrainingLogger:
    """Beautiful Rich terminal output for the wake word training pipeline.

    Handles all terminal display: progress bars, metric tables, confusion matrices,
    phase transitions, checkpoint logging, and completion summaries. Pure display
    module with no training logic.
    """

    def __init__(self, console: Console | None = None) -> None:
        if console is None:
            self.console = Console()
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
        for i, (s, lr) in enumerate(zip(steps, lrs, strict=True)):
            table.add_row(
                f"Phase {i + 1}",
                f"{s:,} steps @ LR {lr:.6f}",
            )

        # Class weights (per-phase lists)
        pos_w = training.get("positive_class_weight", [5.0, 7.0, 9.0])
        neg_w = training.get("negative_class_weight", [1.5, 1.5, 1.5])
        hard_neg_w = training.get("hard_negative_class_weight", [3.0, 5.0, 7.0])
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
        basic_interval = training.get("eval_basic_step_interval", None)
        advanced_interval = training.get("eval_advanced_step_interval", None)
        if basic_interval:
            table.add_row("Basic Eval Interval", f"every {basic_interval} steps")
        if advanced_interval:
            table.add_row("Advanced Eval Interval", f"every {advanced_interval} steps")
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
        has_spec_aug = any(x > 0 for x in time_mask_sizes + time_mask_counts + freq_mask_sizes + freq_mask_counts)
        if has_spec_aug:
            table.add_row(
                "SpecAugment",
                f"time_mask={time_mask_sizes} freq_mask={freq_mask_sizes}",
            )

        # Hard negative mining
        mining = config.get("mining", {})
        mining_enabled = mining.get("enabled", False)
        table.add_row(
            "Hard Neg Mining",
            ("[green]Enabled[/green]" if mining_enabled else "[dim]Disabled[/dim]"),
        )

        panel = Panel(
            table,
            title="🎯 Wake Word Training",
            border_style="blue",
            expand=False,
        )
        self.console.print(panel)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format seconds as H:MM:SS string."""
        total_seconds = max(0, int(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def _compute_eta(self, progress: Progress, task_id: TaskID, completed: int) -> str:
        """Compute ETA from elapsed wall-clock time and completed steps.

        Rich's TimeRemainingColumn needs a speed estimate and typically requires
        at least two progress samples, so it can show unknown ETA (`-:--:--`) on
        the first update. This computes ETA from elapsed/completed so we can show
        an estimate from the first visible training update.
        """
        task = progress.tasks[task_id]
        total = task.total
        elapsed = task.elapsed
        if total is None or elapsed is None or elapsed <= 0 or completed <= 0:
            return "--:--:--"
        remaining = float(total) - float(completed)
        if remaining <= 0:
            return "0:00:00"
        speed = float(completed) / elapsed
        if speed <= 0:
            return "--:--:--"
        eta_seconds = remaining / speed
        return self._format_eta(eta_seconds)

    def create_progress(self, total_steps: int) -> tuple[Progress, TaskID]:
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
            TextColumn("{task.fields[eta]}"),
            TextColumn("•"),
            TextColumn("{task.fields[metrics]}"),
            console=self.console,
            disable=not self.console.is_terminal,
        )
        task_id = progress.add_task("Phase 1", total=total_steps, eta="--:--:--", metrics="")
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
        lr = phase_info.get("learning_rate", 0.0)
        phase = phase_info.get("phase", 0)
        epoch = phase_info.get("epoch", 0)

        progress.update(
            task_id,
            completed=step,
            description=f"Phase {phase + 1}",
            eta=self._compute_eta(progress, task_id, step),
            metrics=f"epoch={epoch} loss={loss:.4f} acc={accuracy:.4f} lr={lr:.6f}",
        )

    def log_validation_results(self, metrics: dict, step: int, total_steps: int) -> None:
        """Display a validation results table with all available metrics."""
        from rich import box

        table = Table(
            title=f"📊 Validation Results — Step {step}/{total_steps}",
            show_header=True,
            header_style="bold magenta",
            box=box.ASCII,
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
            ("quality_score", "Quality Score", ".4f", "bold cyan"),
            ("recall_at_target_fah", "Recall @ Target FAH", ".4f", None),
            ("fah_at_target_recall", "FAH @ Target Recall", ".4f", None),
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
            ("threshold_for_target_fah", "Threshold for Target FAH", ".4f"),
            (
                "threshold_for_target_recall",
                "Threshold for Target Recall",
                ".4f",
            ),
            ("quality_plateau_score", "Quality Plateau", ".0f"),
            ("quality_plateau_slope", "Quality Slope", ".6f"),
            ("quality_plateau_gap", "Quality Gap", ".6f"),
            ("gain_f1_score_per_1k_steps", "F1 Gain / 1k steps", ".4f"),
            (
                "gain_average_viable_recall_per_1k_steps",
                "Avg Recall Gain / 1k steps",
                ".4f",
            ),
            (
                "gain_recall_at_no_faph_per_1k_steps",
                "Recall@NoFAPH Gain / 1k steps",
                ".4f",
            ),
            ("gain_auc_pr_per_1k_steps", "AUC-PR Gain / 1k steps", ".4f"),
            (
                "gain_recall_at_target_fah_per_1k_steps",
                "Recall@TargetFAH Gain / 1k steps",
                ".4f",
            ),
            (
                "gain_fah_at_target_recall_per_1k_steps",
                "FAH@TargetRecall Gain / 1k steps",
                ".4f",
            ),
            ("val_positive_count", "Val Positives", ".0f"),
            ("val_negative_count", "Val Negatives", ".0f"),
            ("val_total_count", "Val Total", ".0f"),
            ("score_min", "Score Min", ".4f"),
            ("score_p05", "Score P05", ".4f"),
            ("score_p50", "Score P50", ".4f"),
            ("score_p95", "Score P95", ".4f"),
            ("score_max", "Score Max", ".4f"),
            ("score_sample_count", "Score Sample Count", ".0f"),
        ]
        for key, label, fmt in optional_metrics:
            if key in metrics and metrics[key] is not None:
                table.add_row(label, f"{metrics[key]:{fmt}}")

        self.console.print(table)

    def log_confusion_matrix(self, tp: int, fp: int, tn: int, fn: int, threshold: float = 0.5) -> None:
        """Display a confusion matrix table."""
        from rich import box

        table = Table(
            title=f"Confusion Matrix (threshold={threshold:.2f})",
            show_header=True,
            header_style="bold",
            box=box.ASCII,
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

    def log_per_class_analysis(self, tp: int, fp: int, tn: int, fn: int, threshold: float) -> None:
        """Display per-class error rates at the given threshold."""
        pos_total = tp + fn
        neg_total = tn + fp
        pos_recall = tp / pos_total if pos_total > 0 else 0.0
        neg_rejection = tn / neg_total if neg_total > 0 else 0.0
        pos_miss_rate = fn / pos_total if pos_total > 0 else 0.0
        neg_fa_rate = fp / neg_total if neg_total > 0 else 0.0

        from rich import box

        table = Table(
            title=f"Per-Class Analysis (threshold={threshold:.2f})",
            show_header=True,
            header_style="bold",
            box=box.ASCII,
        )
        table.add_column("Class", style="bold")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        table.add_row("Positive", "Recall (hit rate)", f"[green]{pos_recall:.4f}[/green]")
        table.add_row("Positive", "Miss rate", f"[red]{pos_miss_rate:.4f}[/red]")
        table.add_row("Positive", "Samples", f"{pos_total:,}")
        table.add_section()
        table.add_row("Negative", "Rejection rate", f"[green]{neg_rejection:.4f}[/green]")
        table.add_row("Negative", "False alarm rate", f"[red]{neg_fa_rate:.4f}[/red]")
        table.add_row("Negative", "Samples", f"{neg_total:,}")

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
        title = f"Phase {phase + 1}/{total_phases} — LR: {lr:.6f} | Weights: pos={pos_weight:.1f} neg={neg_weight:.1f}"
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

        from rich import box

        table = Table(show_header=False, box=box.ASCII, padding=(0, 2))
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

    def log_mining(self, message: str, count: int | None = None) -> None:
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

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.console.print(f"[bold red]❌ Error:[/] {message}")

    def log_next_steps(self, best_path: str, config_preset: str) -> None:
        """Display a 'What\'s Next?' panel with a clean, heavily-iconed, compartmentalized layout."""

        P = best_path
        C = config_preset

        def _cmd(raw: str) -> str:
            """Cyan command; substituted checkpoint/config values highlighted in bold yellow."""
            colored = raw
            if P and P in colored:
                colored = colored.replace(P, f"[/cyan][bold yellow]{P}[/bold yellow][cyan]")
            config_token = f"--config {C}"
            if C and config_token in colored:
                colored = colored.replace(
                    config_token,
                    f"--config [/cyan][bold yellow]{C}[/bold yellow][cyan]",
                )
            return f"[cyan]{colored}[/cyan]"

        def _section(title: str, style: str, entries: list[tuple[str, str, str | None]]) -> Panel:
            t = Table(
                show_header=False,
                box=None,
                padding=(0, 2, 0, 2),
                expand=False,
                show_edge=False,
            )
            t.add_column("Content", justify="left", overflow="fold")

            for i, (cmd_raw, desc, args) in enumerate(entries):
                cmd_str = _cmd(cmd_raw)
                desc_str = f"  [bold white]{desc}[/bold white]"

                row_content = f"✨ {desc_str}\n  💻 {cmd_str}"
                if args:
                    args_str = f"    [dim]↳[/dim] [bright_black]{args}[/bright_black]"
                    row_content += f"\n{args_str}"

                # Add an extra newline for all except the last item
                if i < len(entries) - 1:
                    row_content += "\n"

                t.add_row(row_content)

            return Panel(
                t,
                title=f"[{style}]{title}[/{style}]",
                border_style=style,
                padding=(1, 2),
                expand=False,
            )

        panels: list = []

        # ── Improve Model Quality ──────────────────────────────────────
        panels.append(
            _section(
                "🔨  Improve Model Quality",
                "bold yellow",
                [
                    (
                        f"mww-mine-hard-negatives extract-top-fps --config {C}",
                        "Run model over negative set, extract highest-confidence false positives",
                        "--checkpoint PATH(auto-detect) · --top-percent FLOAT · --threshold FLOAT · --move-now(move files immediately) · --dry-run",
                    ),
                    (
                        "mww-mine-hard-negatives mine --prediction-log logs/false_predictions.json",
                        "Copy top mined FPs → dataset/hard_negative/mined/",
                        "--output-dir PATH(./dataset/hard_negative/mined) · --min-epoch INT(10) · --top-k INT(100) · --deduplicate · --dry-run · --verbose",
                    ),
                    (
                        f"mww-mine-hard-negatives consolidate-logs --config {C}",
                        "Aggregate per-epoch FP logs → consolidated JSON + stats report",
                        "--log-dir PATH(logs/) · --output PATH(logs/false_predictions.json) · --top-n INT(5) · --move-to PATH · --dry-run",
                    ),
                    (
                        f"mww-autotune --checkpoint {P} --config {C}",
                        "Tune FAH/recall without full retraining (~5-10 min) — Pareto archive + simulated annealing",
                        "--target-fah FLOAT · --target-recall FLOAT · --output-dir PATH · --users-hard-negs DIR · --max-iterations INT · --patience INT · --cv-folds INT · --no-confirmation",
                    ),
                ],
            )
        )

        # ── Evaluate Checkpoint ───────────────────────────────────────────
        panels.append(
            _section(
                "📊  Evaluate Checkpoint",
                "bold magenta",
                [
                    (
                        f"python scripts/evaluate_model.py --model {P} --config {C} --output-dir logs/",
                        "Full eval: ROC, PR, DET, calibration, confusion matrix + executive HTML report",
                        "--split train|val|test(test) · --analyze(quality warnings) · --no-plots(JSON-only) · --bootstrap-iterations INT(400) · --fp-cost FLOAT(20) · --fn-cost FLOAT(1) · --json",
                    ),
                    (
                        "python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json",
                        "Build interactive HTML dashboard from evaluation report",
                        "--output PATH  (default: <report_dir>/interactive_dashboard.html)",
                    ),
                    (
                        f"python scripts/compare_models.py <other_model> {P} --config {C}",
                        "Side-by-side FAH/recall delta  (accepts .weights.h5 or .tflite for both args)",
                        "--json(stdout) · --output PATH(save JSON) · --override YAML",
                    ),
                ],
            )
        )

        # ── Export to TFLite ───────────────────────────────────────────────
        panels.append(
            _section(
                "📦  Export to TFLite",
                "bold blue",
                [
                    (
                        f"mww-export --checkpoint {P} --output models/exported/",
                        "Convert to ESPHome streaming TFLite with INT8 quantization and dual subgraphs",
                        "--model-name NAME(wake_word) · --config PATH(max_quality.yaml) · --data-dir PATH(real-data INT8 calibration — better quantization accuracy)",
                    ),
                ],
            )
        )

        # ── Verify & Evaluate TFLite ───────────────────────────────────────
        panels.append(
            _section(
                "✅  Verify & Evaluate TFLite",
                "bold green",
                [
                    (
                        "python scripts/verify_esphome.py models/exported/wake_word.tflite",
                        "ESPHome compatibility: shapes, dtypes, op set  —  exit 0=pass · 2=fail · 1=error",
                        "--verbose · --strict(READ_VARIABLE payload-shape validation) · --json(CI/CD machine-readable)",
                    ),
                    (
                        "python scripts/check_esphome_compat.py models/exported/wake_word.tflite",
                        "Deep architecture + streaming state variable analysis  —  exit 0=pass · 4=incompatible · 1=error",
                        "--verbose · --manifest PATH(cross-validate with manifest.json) · --json",
                    ),
                    (
                        "python scripts/verify_streaming.py models/exported/wake_word.tflite",
                        "Streaming inference correctness: determinism, state mutation per frame, boundary conditions",
                        "--frames INT(15) · --seed INT(42) · --verbose · --json",
                    ),
                    (
                        f"python scripts/evaluate_model.py --tflite models/exported/wake_word.tflite --config {C} --output-dir logs/",
                        "Evaluate exported TFLite model metrics  (same args as checkpoint evaluation above)",
                        None,
                    ),
                    (
                        f"python scripts/compare_models.py {P} models/exported/wake_word.tflite --config {C}",
                        "Compare checkpoint vs TFLite — detect quantization drift",
                        None,
                    ),
                ],
            )
        )

        self.console.print()

        # Display as a clean list of separated panels, with a top header
        self.console.print("[bold]🚀 What's Next? (Post-Training Actions)[/bold]\n")
        for panel in panels:
            self.console.print(panel)
