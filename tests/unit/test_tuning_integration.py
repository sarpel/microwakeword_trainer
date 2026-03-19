"""Integration tests for MicroAutoTuner module interactions and dry-run behavior."""

from __future__ import annotations


def test_dry_run_exits_cleanly() -> None:
    """mww-autotune --config fast_test --dry-run should exit 0 without Traceback."""
    import subprocess
    import sys
    from pathlib import Path

    missing_checkpoint = Path.cwd() / ".nonexistent_checkpoint.h5"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tuning.cli",
            "--config",
            "fast_test",
            "--checkpoint",
            str(missing_checkpoint),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, f"Exit code {result.returncode}. stderr: {result.stderr}"
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_memory_stability_100_cycles() -> None:
    """100 mock burst cycles should use <5% RSS growth."""
    import os

    import psutil

    from src.tuning.knobs import KnobCycle
    from src.tuning.metrics import ParetoArchive, TuneMetrics

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss

    for i in range(100):
        archive = ParetoArchive(max_size=32)
        cycle = KnobCycle(["lr", "threshold", "temperature", "sampling_mix", "weight_perturbation", "label_smoothing"])
        for _ in range(6):
            cycle.advance()
        m = TuneMetrics(fah=float(i % 3), recall=0.9, auc_roc=0.95, auc_pr=0.92)
        archive.try_add(m, f"cand_{i}")

    rss_after = process.memory_info().rss
    growth = (rss_after - rss_before) / max(rss_before, 1) * 100
    assert growth < 5, f"Memory leak: {growth:.1f}% growth (before={rss_before}, after={rss_after})"


def test_module_interaction() -> None:
    """All 6 modules can be imported and interact correctly."""
    from src.tuning import MicroAutoTuner as PublicAPI
    from src.tuning.dashboard import TuningDashboard, save_artifacts
    from src.tuning.knobs import (
        FocusedSampler,
        KnobCycle,
        LabelSmoothingKnob,
        LRKnob,
        SamplingMixKnob,
        TemperatureKnob,
        ThresholdKnob,
        WeightPerturbationKnob,
    )
    from src.tuning.metrics import ErrorMemory, ParetoArchive, TuneMetrics, compute_hypervolume
    from src.tuning.orchestrator import MicroAutoTuner
    from src.tuning.population import Candidate, Population, partition_data

    # Imports are part of integration contract and must resolve.
    assert PublicAPI is MicroAutoTuner
    assert callable(save_artifacts)
    assert callable(compute_hypervolume)
    assert all(
        cls is not None
        for cls in (
            Candidate,
            Population,
            partition_data,
            LRKnob,
            ThresholdKnob,
            TemperatureKnob,
            SamplingMixKnob,
            WeightPerturbationKnob,
            LabelSmoothingKnob,
            FocusedSampler,
        )
    )

    # KnobCycle cycles all 6 knobs
    cycle = KnobCycle(["lr", "threshold", "temperature", "sampling_mix", "weight_perturbation", "label_smoothing"])
    knob_names = []
    for _ in range(6):
        knob_names.append(cycle.current())
        cycle.advance()
    assert len(set(knob_names)) == 6, f"Expected 6 unique knobs, got: {knob_names}"

    # ParetoArchive with metrics
    archive = ParetoArchive(max_size=32)
    m1 = TuneMetrics(fah=2.0, recall=0.85, auc_roc=0.9, auc_pr=0.91)
    m2 = TuneMetrics(fah=1.0, recall=0.90, auc_roc=0.92, auc_pr=0.88)
    assert archive.try_add(m1, "c1")
    assert archive.try_add(m2, "c2")
    assert len(archive) == 2

    # Dashboard renders without error
    dashboard = TuningDashboard(console=None)
    table = dashboard.render_pareto_table([{"id": "c1", "fah": 2.0, "recall": 0.85}])
    assert table is not None

    # ErrorMemory exposes update API
    err = ErrorMemory(max_history=100)
    assert hasattr(err, "update")
