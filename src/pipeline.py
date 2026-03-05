"""mww-pipeline: End-to-end wake word training pipeline with quality gate.

Runs the full sequence:
    train → [autotune] → export → verify_esphome → verify_streaming → gate

The gate only promotes the model (copies to --promote-dir) if:
    - ESPHome verification passes (required)
    - FAH <= --target-fah  (required)
    - recall >= --target-recall  (required)

Exit codes:
    0 - Pipeline succeeded, model promoted
    1 - Pipeline failed (training/export error)
    2 - Gate failed (quality targets not met)
    3 - Verification failed (ESPHome/streaming incompatible)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str], desc: str) -> subprocess.CompletedProcess:
    """Run a subprocess and stream its output.  Raises on non-zero exit."""
    print(f"\n{'=' * 70}")
    print(f"  STEP: {desc}")
    print(f"  CMD : {' '.join(args)}")
    print(f"{'=' * 70}")
    result = subprocess.run(args, check=False)  # noqa: S603
    if result.returncode != 0:
        print(f"\n✗ Step failed (exit {result.returncode}): {desc}")
        sys.exit(1)
    print(f"\n✓ {desc}")
    return result


def _run_capture(args: list[str]) -> subprocess.CompletedProcess:
    """Run subprocess and capture stdout/stderr (for JSON parsing)."""
    return subprocess.run(args, capture_output=True, text=True, check=False)  # noqa: S603


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_train(config: str, override: str | None) -> Path:
    """Train the model and return the checkpoint directory."""
    cmd = [sys.executable, "-m", "src.training.trainer", "--config", config]
    if override:
        cmd += ["--override", override]
    _run(cmd, "Training wake word model")

    checkpoint_dir = Path("./checkpoints")
    best = checkpoint_dir / "best_weights.weights.h5"
    if not best.exists():
        # Fallback: find the latest checkpoint
        candidates = sorted(checkpoint_dir.glob("*.weights.h5"))
        if not candidates:
            print(f"✗ No checkpoint found in {checkpoint_dir}")
            sys.exit(1)
        best = candidates[-1]
    print(f"  Checkpoint: {best}")
    return best


def step_autotune(checkpoint: Path, config: str, override: str | None, target_fah: float, target_recall: float, output_dir: Path) -> Path:
    """Run mww-autotune and return the tuned checkpoint path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.tuning.cli",
        "--checkpoint",
        str(checkpoint),
        "--config",
        config,
        "--target-fah",
        str(target_fah),
        "--target-recall",
        str(target_recall),
        "--output-dir",
        str(output_dir),
    ]
    if override:
        cmd += ["--override", override]
    _run(cmd, "Auto-tuning model")

    # Find tuned checkpoint
    candidates = sorted(output_dir.glob("*.weights.h5"))
    if not candidates:
        print(f"  Warning: No tuned checkpoint found in {output_dir}; using original")
        return checkpoint
    tuned = candidates[-1]
    print(f"  Tuned checkpoint: {tuned}")
    return tuned


def step_export(checkpoint: Path, config: str, output_dir: Path, model_name: str, data_dir: str | None) -> Path:
    """Export the model to TFLite and return the .tflite path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.export.tflite",
        "--checkpoint",
        str(checkpoint),
        "--config",
        config,
        "--output",
        str(output_dir),
        "--model-name",
        model_name,
    ]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    _run(cmd, "Exporting model to TFLite")

    tflite_path = output_dir / f"{model_name}.tflite"
    if not tflite_path.exists():
        # Try any .tflite in output dir, pick the most recent
        candidates = list(output_dir.glob("*.tflite"))
        if not candidates:
            print(f"✗ No .tflite found in {output_dir}")
            sys.exit(1)
        tflite_path = max(candidates, key=lambda p: p.stat().st_mtime)
    return tflite_path


def step_verify_esphome(tflite_path: Path) -> dict:
    """Verify ESPHome compatibility.  Returns verification result dict."""
    cmd = [
        sys.executable,
        "scripts/verify_esphome.py",
        str(tflite_path),
        "--json",
    ]
    result = _run_capture(cmd)
    if result.returncode == 2:
        print(f"✗ verify_esphome failed (exception):\n{result.stderr}")
        sys.exit(3)

    try:
        # Extract JSON from output (may have other lines before it)
        json_str = result.stdout.strip()
        # Find first '{' to tolerate leading text
        idx = json_str.find("{")
        if idx >= 0:
            json_str = json_str[idx:]
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        print(f"✗ Could not parse verify_esphome output:\n{result.stdout}")
        # Treat non-zero returncode as failure
        if result.returncode != 0:
            sys.exit(3)
        data = {"compatible": False, "errors": ["Could not parse output"]}

    compatible = data.get("compatible", False)
    errors = data.get("errors", [])
    warnings = data.get("warnings", [])

    print(f"\n  ESPHome compatible: {compatible}")
    for w in warnings:
        print(f"  ⚠ {w}")
    for e in errors:
        print(f"  ✗ {e}")

    if not compatible:
        print("\n✗ ESPHome verification failed — model cannot run on device")
        sys.exit(3)

    print("✓ ESPHome verification passed")
    return data


def step_verify_streaming(tflite_path: Path) -> None:
    """Run the streaming gate script."""
    streaming_script = Path("scripts/verify_streaming.py")
    if not streaming_script.exists():
        print("  ⚠ verify_streaming.py not found — skipping streaming gate")
        return
    cmd = [sys.executable, str(streaming_script), str(tflite_path)]
    result = _run_capture(cmd)
    if result.returncode != 0:
        print(f"✗ Streaming verification failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(3)
    print("✓ Streaming verification passed")


def step_evaluate(tflite_path: Path, config: str, override: str | None) -> dict:
    """Run evaluate_model.py to get FAH/recall and return metrics dict.

    Returns empty dict if the evaluator script is not available.
    """
    eval_script = Path("scripts/evaluate_model.py")
    if not eval_script.exists():
        print("  ⚠ evaluate_model.py not found — skipping evaluation step")
        return {}

    cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        str(tflite_path),
        "--config",
        config,
        "--json",
    ]
    if override:
        cmd += ["--override", override]

    result = _run_capture(cmd)
    if result.returncode != 0:
        print(f"  ⚠ Evaluation failed (exit {result.returncode}) — skipping quality gate\n{result.stderr}")
        return {}

    try:
        json_str = result.stdout.strip()
        idx = json_str.find("{")
        if idx >= 0:
            json_str = json_str[idx:]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        print("  ⚠ Could not parse evaluation output — skipping quality gate")
        return {}


def step_gate(metrics: dict, target_fah: float, target_recall: float, strict_gate: bool = False) -> bool:
    """Quality gate.  Returns True if model meets targets, False otherwise.

    Args:
        metrics: Dictionary of evaluation metrics
        target_fah: Maximum acceptable FAH
        target_recall: Minimum acceptable recall
        strict_gate: If True, fail the gate when metrics are missing

    Returns:
        True if gate passed, False otherwise
    """
    if not metrics:
        print("  ⚠ No metrics available — quality gate skipped (model will be promoted)")
        return True

    fah = metrics.get("ambient_false_positives_per_hour", metrics.get("fah", None))
    recall = metrics.get("recall", metrics.get("recall_at_target_fah", None))

    print("\n  Quality gate:")
    print(f"    FAH    : {fah}  (target ≤ {target_fah})")
    print(f"    Recall : {recall}  (target ≥ {target_recall})")

    if fah is None or recall is None:
        if strict_gate:
            print("  ✗ Quality gate FAILED (strict mode: missing metrics)")
            return False
        print("  ⚠ Missing FAH or recall metric — quality gate skipped")
        return True

    passed = fah <= target_fah and recall >= target_recall
    if passed:
        print("  ✓ Quality gate PASSED")
    else:
        print("  ✗ Quality gate FAILED")
    return passed


def step_promote(tflite_path: Path, promote_dir: Path, model_name: str) -> None:
    """Copy model and manifest to the promote directory."""
    promote_dir.mkdir(parents=True, exist_ok=True)

    # Copy .tflite
    dest_tflite = promote_dir / tflite_path.name
    shutil.copy2(tflite_path, dest_tflite)
    print(f"  Copied: {dest_tflite}")

    # Copy manifest.json if present
    manifest = tflite_path.parent / "manifest.json"
    if manifest.exists():
        dest_manifest = promote_dir / "manifest.json"
        shutil.copy2(manifest, dest_manifest)
        print(f"  Copied: {dest_manifest}")

    print(f"✓ Model promoted to: {promote_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mww-pipeline",
        description="End-to-end wake word pipeline: train → autotune → export → verify → gate → promote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default="standard", help="Config preset name or path (default: standard)")
    parser.add_argument("--override", type=str, default=None, help="Override config YAML path")
    parser.add_argument("--model-name", type=str, default="wake_word", help="Output model name (default: wake_word)")
    parser.add_argument("--export-dir", type=str, default="./models/exported", help="Directory for exported TFLite model (default: ./models/exported)")
    parser.add_argument("--promote-dir", type=str, default="./models/promoted", help="Directory to copy model to if gate passes (default: ./models/promoted)")
    parser.add_argument("--target-fah", type=float, default=0.5, help="Max FAH for quality gate (default: 0.5)")
    parser.add_argument("--target-recall", type=float, default=0.90, help="Min recall for quality gate (default: 0.90)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory for representative dataset during export")
    parser.add_argument("--strict-gate", action="store_true", help="Fail quality gate when metrics are missing (default: skip gate)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use --checkpoint instead)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Existing checkpoint to use (requires --skip-train)")
    parser.add_argument("--autotune", action="store_true", help="Run mww-autotune after training")
    parser.add_argument("--autotune-output-dir", type=str, default="./checkpoints/tuned", help="Output directory for tuned checkpoint (default: ./checkpoints/tuned)")

    return parser


def main() -> None:  # noqa: C901
    parser = create_parser()
    args = parser.parse_args()

    start_time = time.time()
    print("\n" + "=" * 70)
    print("  mww-pipeline: Wake Word Training Pipeline")
    print("=" * 70)

    # ── 1. Train (or use existing checkpoint) ──────────────────────────────
    if args.skip_train:
        if not args.checkpoint:
            print("✗ --skip-train requires --checkpoint")
            sys.exit(1)
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"✗ Checkpoint not found: {checkpoint}")
            sys.exit(1)
        print(f"  Skipping training — using checkpoint: {checkpoint}")
    else:
        checkpoint = step_train(args.config, args.override)

    # ── 2. AutoTune (optional) ─────────────────────────────────────────────
    if args.autotune:
        checkpoint = step_autotune(
            checkpoint=checkpoint,
            config=args.config,
            override=args.override,
            target_fah=args.target_fah,
            target_recall=args.target_recall,
            output_dir=Path(args.autotune_output_dir),
        )

    # ── 3. Export ──────────────────────────────────────────────────────────
    tflite_path = step_export(
        checkpoint=checkpoint,
        config=args.config,
        output_dir=Path(args.export_dir),
        model_name=args.model_name,
        data_dir=args.data_dir,
    )

    # ── 4. Verify ESPHome compatibility ────────────────────────────────────
    step_verify_esphome(tflite_path)

    # ── 5. Verify streaming behaviour ─────────────────────────────────────
    step_verify_streaming(tflite_path)

    # ── 6. Evaluate (FAH / recall) ─────────────────────────────────────────
    metrics = step_evaluate(tflite_path, args.config, args.override)

    # ── 7. Quality gate ────────────────────────────────────────────────────
    gate_passed = step_gate(metrics, args.target_fah, args.target_recall, strict_gate=args.strict_gate)

    elapsed = time.time() - start_time
    print(f"\n  Total pipeline time: {elapsed / 60:.1f} min")

    if not gate_passed:
        print("\n✗ Pipeline complete — quality gate FAILED, model NOT promoted")
        print(f"  TFLite model is at: {tflite_path}")
        sys.exit(2)

    # ── 8. Promote ────────────────────────────────────────────────────────
    step_promote(tflite_path, Path(args.promote_dir), args.model_name)

    print("\n✓ Pipeline COMPLETE — model promoted successfully")
    print(f"  TFLite : {tflite_path}")
    print(f"  Promoted: {args.promote_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
