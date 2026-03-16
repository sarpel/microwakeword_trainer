"""Unit tests for src.tuning.cli."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.tuning.cli as cli


def test_create_parser_requires_checkpoint() -> None:
    parser = cli.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_validate_args_failures_and_success(tmp_path: Path) -> None:
    missing = argparse.Namespace(
        checkpoint=str(tmp_path / "missing"),
        target_fah=None,
        target_recall=None,
        max_iterations=None,
    )
    assert cli.validate_args(missing) is False

    ckpt = tmp_path / "ok.weights.h5"
    ckpt.write_bytes(b"x")

    bad_fah = argparse.Namespace(
        checkpoint=str(ckpt),
        target_fah=0.0,
        target_recall=None,
        max_iterations=None,
    )
    assert cli.validate_args(bad_fah) is False

    bad_recall = argparse.Namespace(
        checkpoint=str(ckpt),
        target_fah=None,
        target_recall=1.2,
        max_iterations=None,
    )
    assert cli.validate_args(bad_recall) is False

    bad_iters = argparse.Namespace(
        checkpoint=str(ckpt),
        target_fah=None,
        target_recall=None,
        max_iterations=0,
    )
    assert cli.validate_args(bad_iters) is False

    good = argparse.Namespace(
        checkpoint=str(ckpt),
        target_fah=0.5,
        target_recall=0.9,
        max_iterations=5,
    )
    assert cli.validate_args(good) is True


def test_print_config_summary(monkeypatch) -> None:
    printed = {}

    class DummyConsole:
        def print(self, obj=None, *args, **kwargs):
            printed["obj"] = obj

    monkeypatch.setattr(cli, "Console", DummyConsole)

    args = argparse.Namespace(checkpoint="c", config="standard")
    cfg = {
        "auto_tuning": {
            "target_fah": 0.2,
            "target_recall": 0.95,
            "max_iterations": 9,
            "max_gradient_steps": 77,
            "cv_folds": 4,
            "int8_shadow": False,
            "require_confirmation": False,
            "output_dir": "out",
        }
    }
    cli.print_config_summary(args, cfg)
    assert "obj" in printed


@dataclasses.dataclass
class _DummyCfg:
    auto_tuning: dict


def _base_args(tmp_path: Path) -> SimpleNamespace:
    ckpt = tmp_path / "base.weights.h5"
    ckpt.write_bytes(b"x")
    return SimpleNamespace(
        checkpoint=str(ckpt),
        config="standard",
        override=None,
        target_fah=None,
        target_recall=None,
        max_iterations=None,
        output_dir=None,
        patience=None,
        users_hard_negs=None,
        dry_run=False,
        verbose=False,
        max_gradient_steps=None,
        cv_folds=None,
        no_int8_shadow=False,
        no_confirmation=False,
    )


def test_main_returns_1_when_validate_fails(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    monkeypatch.setattr(cli, "create_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "validate_args", lambda _a: False)
    assert cli.main() == 1


def test_main_config_load_error(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    monkeypatch.setattr(cli, "create_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "validate_args", lambda _a: True)

    import config.loader

    monkeypatch.setattr(
        config.loader,
        "load_full_config",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad cfg")),
    )
    assert cli.main() == 1


def test_main_dry_run(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.dry_run = True
    monkeypatch.setattr(cli, "create_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "validate_args", lambda _a: True)

    import config.loader

    monkeypatch.setattr(
        config.loader,
        "load_full_config",
        lambda *_a, **_k: _DummyCfg(auto_tuning={}),
    )
    assert cli.main() == 0


def test_main_success_and_overrides(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.target_fah = 0.25
    args.target_recall = 0.93
    args.max_iterations = 11
    args.output_dir = str(tmp_path / "out")
    args.patience = 7
    args.max_gradient_steps = 123
    args.cv_folds = 2
    args.no_confirmation = True

    monkeypatch.setattr(cli, "create_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "validate_args", lambda _a: True)

    import config.loader

    monkeypatch.setattr(
        config.loader,
        "load_full_config",
        lambda *_a, **_k: _DummyCfg(auto_tuning={}),
    )
    monkeypatch.setattr(cli, "print_config_summary", lambda *_a, **_k: None)

    captured = {}

    class DummyTuner:
        def __init__(
            self,
            checkpoint_path,
            config,
            auto_tuning_config,
            console,
            users_hard_negs_dir,
        ):
            captured["checkpoint_path"] = checkpoint_path
            captured["config"] = config
            captured["auto_tuning_config"] = auto_tuning_config
            captured["users_hard_negs_dir"] = users_hard_negs_dir

        def tune(self):
            return {
                "best_fah": 0.2,
                "best_recall": 0.95,
                "iterations": 3,
                "target_met": True,
                "best_checkpoint": "x.weights.h5",
                "pareto_frontier": [{"id": "a"}],
            }

    monkeypatch.setattr(cli, "AutoTuner", DummyTuner)
    assert cli.main() == 0
    at = captured["auto_tuning_config"]
    assert at["target_fah"] == 0.25
    assert at["target_recall"] == 0.93
    assert at["max_iterations"] == 11
    assert at["output_dir"] == str(tmp_path / "out")
    assert at["patience"] == 7
    assert at["max_gradient_steps"] == 123
    assert at["cv_folds"] == 2
    assert at["require_confirmation"] is False


def test_main_handles_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    monkeypatch.setattr(cli, "create_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "validate_args", lambda _a: True)

    import config.loader

    monkeypatch.setattr(
        config.loader,
        "load_full_config",
        lambda *_a, **_k: _DummyCfg(auto_tuning={}),
    )
    monkeypatch.setattr(cli, "print_config_summary", lambda *_a, **_k: None)

    class InterruptTuner:
        def __init__(self, *args, **kwargs):
            pass

        def tune(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "AutoTuner", InterruptTuner)
    assert cli.main() == 130


def test_main_handles_exception_verbose(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.verbose = True
    monkeypatch.setattr(cli, "create_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "validate_args", lambda _a: True)

    import config.loader

    monkeypatch.setattr(
        config.loader,
        "load_full_config",
        lambda *_a, **_k: _DummyCfg(auto_tuning={}),
    )
    monkeypatch.setattr(cli, "print_config_summary", lambda *_a, **_k: None)

    class ExplodingTuner:
        def __init__(self, *args, **kwargs):
            pass

        def tune(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "AutoTuner", ExplodingTuner)
    assert cli.main() == 1
