"""Dashboard module for auto-tuning visualization.

Provides:
- TuningDashboard: Rich-based dashboard for displaying tuning progress
- save_artifacts: Save tuning artifacts to JSON files
"""

from __future__ import annotations

import json
import os
from typing import Any

from rich.console import Console
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

        return table

    def render_hypervolume_history(self, history: list[float]) -> Any:
        """Render the hypervolume history.

        Args:
            history: List of hypervolume values over iterations

        Returns:
            A renderable object (string representation of history)
        """
        if not history:
            return "No hypervolume history"

        # Return a simple string representation
        lines = ["Hypervolume History:"]
        for i, hv in enumerate(history):
            lines.append(f"  Iteration {i}: {hv:.6f}")

        return "\n".join(lines)


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
