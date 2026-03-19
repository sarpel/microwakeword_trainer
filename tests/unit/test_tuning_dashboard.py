"""Unit tests for src.tuning.dashboard — RED phase (all tests must FAIL with ImportError)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

from src.tuning.dashboard import TuningDashboard, save_artifacts

# These imports will ALL FAIL (ImportError) until dashboard.py is created — that's the RED phase.


# ---------------------------------------------------------------------------
# Minimal fake data classes for testing — no dependency on metrics.py
# ---------------------------------------------------------------------------


@dataclass
class FakeTuneMetrics:
    fah: float = 1.0
    recall: float = 0.9
    auc_pr: float = 0.95
    auc_roc: float = 0.98
    ece: float = 0.05
    threshold: float = 0.5
    threshold_uint8: int = 128
    precision: float = 0.9
    f1: float = 0.9


def _make_candidate(cid: str, fah: float, recall: float, is_best: bool = False) -> dict:
    return {
        "id": cid,
        "metrics": FakeTuneMetrics(fah=fah, recall=recall),
        "is_best": is_best,
        "knob": "lr",
        "iteration": 1,
    }


# ---------------------------------------------------------------------------
# TuningDashboard construction
# ---------------------------------------------------------------------------


def test_dashboard_instantiation() -> None:
    dash = TuningDashboard()
    assert dash is not None


def test_dashboard_instantiation_with_console() -> None:
    from rich.console import Console

    console = Console(record=True)
    dash = TuningDashboard(console=console)
    assert dash is not None


# ---------------------------------------------------------------------------
# render_population_table
# ---------------------------------------------------------------------------


def test_render_population_table_returns_rich_table() -> None:
    from rich.table import Table

    dash = TuningDashboard()
    candidates = [_make_candidate("c1", 0.5, 0.90), _make_candidate("c2", 0.3, 0.95)]
    table = dash.render_population_table(candidates)
    assert isinstance(table, Table)


def test_render_population_table_has_required_columns() -> None:
    dash = TuningDashboard()
    candidates = [_make_candidate("c1", 0.5, 0.90)]
    table = dash.render_population_table(candidates)
    column_names = [str(col.header) for col in table.columns]
    for required in ("ID", "FAH", "Recall"):
        assert any(required.lower() in name.lower() for name in column_names), f"Column '{required}' missing from population table. Got: {column_names}"


def test_render_population_table_has_rows() -> None:
    dash = TuningDashboard()
    candidates = [_make_candidate("c1", 0.5, 0.90), _make_candidate("c2", 0.3, 0.95)]
    table = dash.render_population_table(candidates)
    assert table.row_count == 2


def test_render_population_table_empty_input() -> None:
    from rich.table import Table

    dash = TuningDashboard()
    table = dash.render_population_table([])
    assert isinstance(table, Table)
    assert table.row_count == 0


# ---------------------------------------------------------------------------
# render_pareto_table
# ---------------------------------------------------------------------------


def test_render_pareto_table_returns_rich_table() -> None:
    from rich.table import Table

    dash = TuningDashboard()
    frontier = [
        {"fah": 0.3, "recall": 0.95, "auc_pr": 0.98, "id": "c1"},
        {"fah": 0.5, "recall": 0.90, "auc_pr": 0.94, "id": "c2"},
    ]
    table = dash.render_pareto_table(frontier)
    assert isinstance(table, Table)


def test_render_pareto_table_has_fah_recall_columns() -> None:
    dash = TuningDashboard()
    frontier = [{"fah": 0.3, "recall": 0.95, "auc_pr": 0.98, "id": "c1"}]
    table = dash.render_pareto_table(frontier)
    column_names = [str(col.header) for col in table.columns]
    assert any("fah" in name.lower() for name in column_names)
    assert any("recall" in name.lower() for name in column_names)


# ---------------------------------------------------------------------------
# render_knob_table
# ---------------------------------------------------------------------------


def test_render_knob_table_returns_rich_table() -> None:
    from rich.table import Table

    dash = TuningDashboard()
    knob_cycle = ["lr", "threshold", "temperature", "sampling_mix", "weight_perturbation", "label_smoothing"]
    table = dash.render_knob_table(current_knob="threshold", cycle_position=1, knob_cycle=knob_cycle)
    assert isinstance(table, Table)


def test_render_knob_table_marks_active_knob() -> None:
    from rich.console import Console

    console = Console(record=True)
    dash = TuningDashboard(console=console)
    knob_cycle = ["lr", "threshold"]
    dash.render_knob_table(current_knob="lr", cycle_position=0, knob_cycle=knob_cycle)
    # Should not raise — verification is that rendering succeeds


# ---------------------------------------------------------------------------
# render_hypervolume_history
# ---------------------------------------------------------------------------


def test_render_hypervolume_history_returns_renderable() -> None:
    dash = TuningDashboard()
    history = [0.0, 0.5, 1.2, 1.8, 2.1]
    result = dash.render_hypervolume_history(history)
    assert result is not None  # Some renderable object (string, Panel, or Text)


def test_render_hypervolume_history_empty() -> None:
    dash = TuningDashboard()
    result = dash.render_hypervolume_history([])
    assert result is not None


# ---------------------------------------------------------------------------
# save_artifacts
# ---------------------------------------------------------------------------


def test_save_artifacts_creates_json_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        candidates = [_make_candidate("c1", 0.5, 0.90)]
        frontier = [{"fah": 0.5, "recall": 0.90, "auc_pr": 0.94, "id": "c1"}]
        save_artifacts(
            output_dir=tmpdir,
            candidates=candidates,
            frontier=frontier,
            hypervolume_history=[0.0, 1.2],
            iteration=5,
            best_candidate=candidates[0],
        )
        files = os.listdir(tmpdir)
        json_files = [f for f in files if f.endswith(".json")]
        assert len(json_files) >= 1, f"Expected at least one JSON file, got: {files}"


def test_save_artifacts_json_is_valid() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        candidates = [_make_candidate("c1", 0.5, 0.90)]
        frontier = [{"fah": 0.5, "recall": 0.90, "auc_pr": 0.94, "id": "c1"}]
        save_artifacts(
            output_dir=tmpdir,
            candidates=candidates,
            frontier=frontier,
            hypervolume_history=[0.0, 1.2],
            iteration=5,
            best_candidate=candidates[0],
        )
        json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        path = os.path.join(tmpdir, json_files[0])
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
