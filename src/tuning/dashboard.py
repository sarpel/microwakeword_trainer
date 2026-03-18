"""Dashboard module for auto-tuning visualization.

Provides:
- TuningDashboard: Rich-based dashboard for displaying tuning progress
- save_artifacts: Save tuning artifacts to JSON files
"""

from __future__ import annotations

import json
import os
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table


class TuningDashboard:
    """Dashboard for displaying auto-tuning progress and results."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize dashboard.

        Args:
            console: Optional Rich console instance. Creates default if None.
        """
        self.console = console if console is not None else Console()

    def render_population_table(self, candidates: list[dict]) -> Table:
        """Render a table displaying the current candidate population.

        Args:
            candidates: List of candidate dicts with keys:
                - id: str
                - metrics: object with fah, recall, auc_pr attrs
                - is_best: bool
                - knob: str
                - iteration: int

        Returns:
            Rich Table with candidate information
        """
        table = Table(title="Population")
        table.add_column("ID", style="bold")
        table.add_column("FAH", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("AUC-PR", justify="right")
        table.add_column("Knob")
        table.add_column("Iteration", justify="right")
        table.add_column("Status")

        for candidate in candidates:
            metrics = candidate.get("metrics", None)
            fah = getattr(metrics, "fah", float("nan")) if metrics else float("nan")
            recall = getattr(metrics, "recall", float("nan")) if metrics else float("nan")
            auc_pr = getattr(metrics, "auc_pr", float("nan")) if metrics else float("nan")

            status = "★ BEST" if candidate.get("is_best", False) else ""

            table.add_row(
                str(candidate.get("id", "")),
                f"{fah:.4f}" if isinstance(fah, (int, float)) and not isinstance(fah, bool) else str(fah),
                f"{recall:.4f}" if isinstance(recall, (int, float)) and not isinstance(recall, bool) else str(recall),
                f"{auc_pr:.4f}" if isinstance(auc_pr, (int, float)) and not isinstance(auc_pr, bool) else str(auc_pr),
                str(candidate.get("knob", "")),
                str(candidate.get("iteration", "")),
                status,
            )

        self.console.print(table)
        return table

    def render_pareto_table(self, frontier: list[dict]) -> Table:
        """Render a table displaying the Pareto frontier.

        Args:
            frontier: List of dicts with keys:
                - id: str
                - fah: float
                - recall: float
                - auc_pr: float

        Returns:
            Rich Table with Pareto frontier points
        """
        table = Table(title="Pareto Frontier")
        table.add_column("ID", style="bold")
        table.add_column("FAH", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("AUC-PR", justify="right")

        for point in frontier:
            table.add_row(
                str(point.get("id", "")),
                f"{point.get('fah', float('nan')):.4f}",
                f"{point.get('recall', float('nan')):.4f}",
                f"{point.get('auc_pr', float('nan')):.4f}",
            )

        self.console.print(table)
        return table

    def render_knob_table(self, current_knob: str, cycle_position: int, knob_cycle: list[str]) -> Table:
        """Render a table displaying the current knob cycle position.

        Args:
            current_knob: The currently active knob name
            cycle_position: Current position in the cycle (0-indexed)
            knob_cycle: List of knob names in the cycle

        Returns:
            Rich Table showing knob cycle with current position marked
        """
        table = Table(title="Knob Cycle")
        table.add_column("Position", justify="right")
        table.add_column("Knob")
        table.add_column("Status")

        for i, knob in enumerate(knob_cycle):
            status = "● ACTIVE" if knob == current_knob else ""
            table.add_row(
                str(i),
                knob,
                status,
            )

        self.console.print(table)
        return table

    def render_hypervolume_history(self, history: list[float]) -> Any:
        """Render the hypervolume history.

        Args:
            history: List of hypervolume values over iterations

        Returns:
            A renderable object (string representation of history)
        """
        if not history:
            output = "No hypervolume history"
            self.console.print(output)
            return output

        # Return a simple string representation
        lines = ["Hypervolume History:"]
        for i, hv in enumerate(history):
            lines.append(f"  Iteration {i}: {hv:.6f}")

        output = "\n".join(lines)
        self.console.print(output)
        return output

    def render_iteration_summary(
        self,
        iteration: int,
        max_iterations: int,
        hypervolume: float,
        best_hv: float,
        no_improve_count: int,
        patience: int,
        best_metrics: dict | None = None,
    ) -> None:
        """Render and print a compact per-iteration tuning summary line."""
        patience_ratio = (no_improve_count / patience) if patience > 0 else 0.0
        if patience_ratio > 0.75:
            patience_style = "bold red"
        elif patience_ratio > 0.50:
            patience_style = "bold yellow"
        else:
            patience_style = "green"

        best_segment = ""
        if best_metrics:
            best_fah = best_metrics.get("fah", None)
            best_recall = best_metrics.get("recall", None)
            if isinstance(best_fah, (int, float)) and isinstance(best_recall, (int, float)):
                best_segment = f" │ Best: FAH={best_fah:.2f} Recall={best_recall:.2f}"

        summary = f"[bold]Iter {iteration}/{max_iterations}[/] │ HV: {hypervolume:.6f} │ Best HV: {best_hv:.6f} │ Patience: [{patience_style}]{no_improve_count}/{patience}[/]{best_segment}"
        self.console.print(summary)

    def render_confirmation_results(
        self,
        metrics: Any,
        target_fah: float,
        target_recall: float,
    ) -> None:
        """Render and print confirmation metrics with target pass/fail status."""
        fah = float(getattr(metrics, "fah", float("nan")))
        recall = float(getattr(metrics, "recall", float("nan")))
        auc_pr = float(getattr(metrics, "auc_pr", float("nan")))
        threshold = float(getattr(metrics, "threshold", float("nan")))

        fah_pass = fah <= target_fah
        recall_pass = recall >= target_recall
        overall_pass = fah_pass and recall_pass

        table = Table(
            title="Confirmation Results",
            show_header=True,
            header_style="bold magenta",
            box=box.ASCII,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")

        table.add_row(
            "FAH",
            f"{fah:.4f}",
            f"≤ {target_fah:.4f}",
            "[green]✓ PASS[/]" if fah_pass else "[red]✗ FAIL[/]",
        )
        table.add_row(
            "Recall",
            f"{recall:.4f}",
            f"≥ {target_recall:.4f}",
            "[green]✓ PASS[/]" if recall_pass else "[red]✗ FAIL[/]",
        )
        table.add_row("AUC-PR", f"{auc_pr:.4f}", "[dim]n/a[/]", "[dim]—[/]")
        table.add_row("Threshold", f"{threshold:.4f}", "[dim]n/a[/]", "[dim]—[/]")
        table.add_section()
        table.add_row(
            "Overall",
            "",
            "",
            "[bold green]✓ TARGETS MET[/]" if overall_pass else "[bold red]✗ TARGETS NOT MET[/]",
        )

        self.console.print(table)

    def render_tuning_header(
        self,
        max_iterations: int,
        population_size: int,
        burst_steps: int,
        knob_cycle: list[str],
        patience: int,
    ) -> None:
        """Render and print a panel summarizing tuning configuration."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")

        table.add_row("Max Iterations", f"{max_iterations:,}")
        table.add_row("Population Size", str(population_size))
        table.add_row("Burst Steps", str(burst_steps))
        table.add_row("Knob Cycle", " → ".join(knob_cycle) if knob_cycle else "[dim]None[/]")
        table.add_row("Patience", str(patience))

        panel = Panel(
            table,
            title="🎛️ Micro Auto-Tuning",
            border_style="blue",
            expand=False,
        )
        self.console.print(panel)
        self.console.print(Rule(style="dim"))


def _serialize_candidate(candidate: dict) -> dict:
    """Serialize a candidate for JSON output."""
    result = {
        "id": candidate.get("id", ""),
        "is_best": candidate.get("is_best", False),
        "knob": candidate.get("knob", ""),
        "iteration": candidate.get("iteration", 0),
    }

    metrics = candidate.get("metrics", None)
    if metrics:
        result["metrics"] = {
            "fah": getattr(metrics, "fah", None),
            "recall": getattr(metrics, "recall", None),
            "auc_pr": getattr(metrics, "auc_pr", None),
            "auc_roc": getattr(metrics, "auc_roc", None),
            "ece": getattr(metrics, "ece", None),
            "threshold": getattr(metrics, "threshold", None),
            "threshold_uint8": getattr(metrics, "threshold_uint8", None),
            "precision": getattr(metrics, "precision", None),
            "f1": getattr(metrics, "f1", None),
        }

    return result


def save_artifacts(
    output_dir: str,
    candidates: list[dict],
    frontier: list[dict],
    hypervolume_history: list[float],
    iteration: int,
    best_candidate: dict | None,
) -> None:
    """Save tuning artifacts to JSON files.

    Args:
        output_dir: Directory to write artifacts to
        candidates: List of candidate dictionaries
        frontier: List of Pareto frontier point dictionaries
        hypervolume_history: List of hypervolume values
        iteration: Current iteration number
        best_candidate: The best candidate dict or None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build the artifacts dictionary
    artifacts = {
        "iteration": iteration,
        "candidates": [_serialize_candidate(c) for c in candidates],
        "frontier": frontier,
        "hypervolume_history": hypervolume_history,
    }

    if best_candidate:
        artifacts["best_candidate"] = _serialize_candidate(best_candidate)

    # Write to JSON file
    output_path = os.path.join(output_dir, "tuning_artifacts.json")
    with open(output_path, "w") as f:
        json.dump(artifacts, f, indent=2)
