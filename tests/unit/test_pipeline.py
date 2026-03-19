"""Unit tests for src.pipeline orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.pipeline as pipeline

# Platform-agnostic path type for tests
FakePath = type(Path())


def test_run_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    res = pipeline._run(["python", "-V"], "desc")
    out = capsys.readouterr().out
    assert res.returncode == 0
    assert "STEP: desc" in out
    assert "✓ desc" in out


def test_run_failure_exits(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )
    with pytest.raises(SystemExit) as exc:
        pipeline._run(["x"], "bad")
    assert exc.value.code == 1


def test_run_capture_calls_subprocess(monkeypatch) -> None:
    called = {}

    def fake_run(args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    res = pipeline._run_capture(["cmd", "--json"])
    assert res.returncode == 0
    assert called["kwargs"]["capture_output"] is True
    assert called["kwargs"]["text"] is True


def test_step_train_uses_best_checkpoint_when_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    best = cp / "best_weights.weights.h5"
    best.write_bytes(b"x")

    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    out = pipeline.step_train("standard", None)
    assert out == Path("checkpoints") / "best_weights.weights.h5"


def test_step_train_fallbacks_and_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    # fallback candidate
    c1 = cp / "z.weights.h5"
    c1.write_bytes(b"x")

    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    assert pipeline.step_train("standard", "ovr") == Path("checkpoints") / "z.weights.h5"

    c1.unlink()
    with pytest.raises(SystemExit) as exc:
        pipeline.step_train("standard", None)
    assert exc.value.code == 1


def test_step_autotune_returns_latest_or_original(tmp_path: Path, monkeypatch) -> None:
    chk = tmp_path / "orig.weights.h5"
    chk.write_bytes(b"x")
    out_dir = tmp_path / "tuned"

    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    # no candidates => original
    out = pipeline.step_autotune(chk, "standard", None, 0.5, 0.9, out_dir)
    assert out == chk

    a = out_dir / "a.weights.h5"
    b = out_dir / "b.h5"
    nested = out_dir / "checkpoints" / "c.weights.h5"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"c")
    out2 = pipeline.step_autotune(chk, "standard", None, 0.5, 0.9, out_dir)
    assert out2 in {a, b, nested}


def test_step_export_finds_named_and_fallback(tmp_path: Path, monkeypatch) -> None:
    chk = tmp_path / "orig.weights.h5"
    chk.write_bytes(b"x")
    out_dir = tmp_path / "exp"
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    named = out_dir / "wake_word.tflite"
    out_dir.mkdir(parents=True)
    named.write_bytes(b"x")
    assert pipeline.step_export(chk, "standard", out_dir, "wake_word", None) == named

    named.unlink()
    alt = out_dir / "alt.tflite"
    alt.write_bytes(b"y")
    assert pipeline.step_export(chk, "standard", out_dir, "wake_word", "data") == alt

    alt.unlink()
    with pytest.raises(SystemExit) as exc:
        pipeline.step_export(chk, "standard", out_dir, "wake_word", None)
    assert exc.value.code == 1


def test_step_verify_esphome_paths(monkeypatch, tmp_path: Path) -> None:
    tfl = tmp_path / "m.tflite"
    tfl.write_bytes(b"x")

    # exception path
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stdout="", stderr="e"),
    )
    with pytest.raises(SystemExit) as exc:
        pipeline.step_verify_esphome(tfl)
    assert exc.value.code == 3

    # parse with leading noise
    payload = {"compatible": True, "errors": [], "warnings": ["w"]}
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="noise\n" + json.dumps(payload), stderr=""),
    )
    out = pipeline.step_verify_esphome(tfl)
    assert out["compatible"] is True

    # bad json + nonzero => exit
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="not-json", stderr=""),
    )
    with pytest.raises(SystemExit) as exc2:
        pipeline.step_verify_esphome(tfl)
    assert exc2.value.code == 3

    # compatible false => exit
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"compatible": False, "errors": ["x"], "warnings": []}),
            stderr="",
        ),
    )
    with pytest.raises(SystemExit) as exc3:
        pipeline.step_verify_esphome(tfl)
    assert exc3.value.code == 3


def test_step_verify_streaming_paths(monkeypatch, tmp_path: Path) -> None:
    tfl = tmp_path / "m.tflite"
    tfl.write_bytes(b"x")

    # missing script => skip
    monkeypatch.setattr(
        pipeline,
        "Path",
        lambda p: (Path(tmp_path / "missing.py") if p == "scripts/verify_streaming.py" else Path(p)),
    )
    pipeline.step_verify_streaming(tfl)


def test_step_verify_streaming_failure_and_success(monkeypatch, tmp_path: Path) -> None:
    tfl = tmp_path / "m.tflite"
    tfl.write_bytes(b"x")
    script = tmp_path / "verify_streaming.py"
    script.write_text("#x")

    def fake_path(p: str):
        if p == "scripts/verify_streaming.py":
            return script
        return Path(p)

    monkeypatch.setattr(pipeline, "Path", fake_path)

    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="o", stderr="e"),
    )
    with pytest.raises(SystemExit) as exc:
        pipeline.step_verify_streaming(tfl)
    assert exc.value.code == 3

    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    pipeline.step_verify_streaming(tfl)


def test_step_evaluate_paths(monkeypatch, tmp_path: Path) -> None:
    tfl = tmp_path / "m.tflite"
    tfl.write_bytes(b"x")

    # missing script
    monkeypatch.setattr(
        pipeline,
        "Path",
        lambda p: (Path(tmp_path / "missing.py") if p == "scripts/evaluate_model.py" else Path(p)),
    )
    assert pipeline.step_evaluate(tfl, "standard", None) == {}


def test_step_evaluate_parse_and_failure(monkeypatch, tmp_path: Path) -> None:
    tfl = tmp_path / "m.tflite"
    tfl.write_bytes(b"x")
    script = tmp_path / "evaluate_model.py"
    script.write_text("#x")

    def fake_path(p: str):
        if p == "scripts/evaluate_model.py":
            return script
        return Path(p)

    monkeypatch.setattr(pipeline, "Path", fake_path)
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout='prefix\n{"fah": 0.1}', stderr=""),
    )
    out = pipeline.step_evaluate(tfl, "standard", "ovr")
    assert out["fah"] == 0.1

    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )
    assert pipeline.step_evaluate(tfl, "standard", None) == {}

    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )
    assert pipeline.step_evaluate(tfl, "standard", None) == {}


def test_step_gate_variants() -> None:
    assert pipeline.step_gate({}, 0.5, 0.9, strict_gate=False) is True
    assert pipeline.step_gate({}, 0.5, 0.9, strict_gate=True) is False
    assert pipeline.step_gate({"fah": 0.4, "recall": 0.91}, 0.5, 0.9) is True
    assert (
        pipeline.step_gate(
            {
                "ambient_false_positives_per_hour": 0.6,
                "recall_at_target_fah": 0.8,
            },
            0.5,
            0.9,
        )
        is False
    )
    assert pipeline.step_gate({"fah": 0.1}, 0.5, 0.9) is False


def test_step_promote_copies_files(tmp_path: Path) -> None:
    src = tmp_path / "x" / "m.tflite"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"1")
    manifest = src.parent / "manifest.json"
    manifest.write_text("{}")

    dst = tmp_path / "promoted"
    pipeline.step_promote(src, dst, "wake")
    assert (dst / "wake.tflite").exists()
    assert (dst / "wake_manifest.json").exists()


def test_create_parser_defaults() -> None:
    parser = pipeline.create_parser()
    args = parser.parse_args([])
    assert args.config == "standard"
    assert args.target_fah == 0.5
    assert args.target_recall == 0.9


def test_main_skip_train_requires_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "create_parser",
        lambda: SimpleNamespace(parse_args=lambda: SimpleNamespace(skip_train=True, checkpoint=None)),
    )
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 1
