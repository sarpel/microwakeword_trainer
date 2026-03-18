"""Unit tests for src.export.manifest."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.export import manifest as manifest_mod


def test_generate_manifest_uses_defaults_when_no_tflite(
    tmp_path: Path,
) -> None:
    cfg: dict[str, Any] = {"export": {}, "hardware": {}}
    out = manifest_mod.generate_manifest(
        model_path="wake_word.tflite",
        config=cfg,
        tflite_path=str(tmp_path / "missing.tflite"),
    )

    assert out["type"] == "micro"
    assert out["version"] == 2
    assert out["model"] == "wake_word.tflite"
    assert out["website"] == "https://github.com/sarpel/microwakeword_trainer"
    assert out["micro"]["feature_step_size"] == 10
    assert out["micro"]["tensor_arena_size"] == manifest_mod.DEFAULT_TENSOR_ARENA_SIZE


def test_generate_manifest_uses_calculated_arena_when_tflite_exists(tmp_path: Path, monkeypatch) -> None:
    tflite = tmp_path / "m.tflite"
    tflite.write_bytes(b"x")

    monkeypatch.setattr(
        manifest_mod,
        "calculate_tensor_arena_size",
        lambda *_args, **_kwargs: 54321,
    )
    cfg = {
        "export": {"arena_size_margin": 1.5, "wake_word": "Hi"},
        "hardware": {"window_step_ms": 10},
    }
    out = manifest_mod.generate_manifest(model_path=str(tflite), config=cfg, tflite_path=str(tflite))

    assert out["wake_word"] == "Hi"
    assert out["micro"]["feature_step_size"] == 10
    assert out["micro"]["tensor_arena_size"] == 54321


def test_resolve_tensor_arena_size_prefers_explicit_override(
    tmp_path: Path,
) -> None:
    tflite = tmp_path / "x.tflite"
    tflite.write_bytes(b"1")

    size = manifest_mod.resolve_tensor_arena_size(
        tflite_path=str(tflite),
        export_config={"tensor_arena_size": 42424, "arena_size_margin": 9.9},
    )

    assert size == 42424


def test_resolve_tensor_arena_size_zero_means_auto(tmp_path: Path, monkeypatch) -> None:
    tflite = tmp_path / "x.tflite"
    tflite.write_bytes(b"1")
    monkeypatch.setattr(
        manifest_mod,
        "calculate_tensor_arena_size",
        lambda *_args, **_kwargs: 33333,
    )

    size = manifest_mod.resolve_tensor_arena_size(
        tflite_path=str(tflite),
        export_config={"tensor_arena_size": 0, "arena_size_margin": 1.3},
    )

    assert size == 33333


def test_save_manifest_creates_parent_and_writes_json(tmp_path: Path) -> None:
    manifest = {"a": 1, "micro": {"b": 2}}
    out_path = tmp_path / "nested" / "manifest.json"

    saved = manifest_mod.save_manifest(manifest, str(out_path))
    assert saved == str(out_path)
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == manifest


def test_calculate_tensor_arena_size_happy_path(monkeypatch, tmp_path: Path) -> None:
    class FakeInterpreter:
        def __init__(self, model_path: str):
            self.model_path = model_path

        def allocate_tensors(self) -> None:
            return None

        def get_tensor_details(self):
            return [
                {"shape": [1, 10], "dtype": np.float32, "name": "f32"},
                {"shape": [2, 0], "dtype": np.int8, "name": "i8_zero"},
                {"shape": [1, -1], "dtype": np.uint8, "name": "u8_dynamic"},
                {"shape": [1, 5], "dtype": np.bytes_, "name": "bytes"},
            ]

    fake_tf = SimpleNamespace(lite=SimpleNamespace(Interpreter=FakeInterpreter))
    monkeypatch.setattr(manifest_mod, "tf", fake_tf)

    tflite = tmp_path / "x.tflite"
    tflite.write_bytes(b"1")
    size = manifest_mod.calculate_tensor_arena_size(str(tflite), margin=1.1)
    assert size >= manifest_mod.DEFAULT_TENSOR_ARENA_SIZE


def test_calculate_tensor_arena_size_fallback_on_exception(
    monkeypatch,
) -> None:
    class ExplodingInterpreter:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    fake_tf = SimpleNamespace(lite=SimpleNamespace(Interpreter=ExplodingInterpreter))
    monkeypatch.setattr(manifest_mod, "tf", fake_tf)

    size = manifest_mod.calculate_tensor_arena_size("does-not-matter.tflite")
    assert size == manifest_mod.DEFAULT_TENSOR_ARENA_SIZE


def test_verify_esphome_compatibility_success_with_warning() -> None:
    data = {
        "type": "micro",
        "wake_word": "hi",
        "author": "a",
        "website": "w",
        "model": "m.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": 0.95,
            "feature_step_size": 10,
            "sliding_window_size": 5,
            "tensor_arena_size": 1024,
            "minimum_esphome_version": "2024.7.0",
        },
    }

    result = manifest_mod.verify_esphome_compatibility(data)
    assert result["compatible"] is True
    assert result["errors"] == []
    assert result["warnings"]


def test_verify_esphome_compatibility_rejects_invalid_micro_and_fields() -> None:
    bad = {"type": "wrong", "version": 1, "micro": "not-a-dict"}
    result = manifest_mod.verify_esphome_compatibility(bad)
    assert result["compatible"] is False
    assert any("type" in err for err in result["errors"])
    assert any("version" in err for err in result["errors"])
    assert any("must be an object" in err for err in result["errors"])


def test_create_esphome_package_includes_metadata_and_saves_manifest(tmp_path: Path, monkeypatch) -> None:
    saved: dict[str, Any] = {}

    def fake_save(manifest: dict, output_path: str) -> str:
        saved["manifest"] = manifest
        saved["path"] = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(manifest))
        return output_path

    monkeypatch.setattr(manifest_mod, "save_manifest", fake_save)
    monkeypatch.setattr(
        manifest_mod,
        "generate_manifest",
        lambda **_kwargs: {"micro": {"tensor_arena_size": 777}},
    )

    res = manifest_mod.create_esphome_package(
        model=None,
        config={"export": {}, "hardware": {}},
        output_dir=str(tmp_path),
        model_name="hello",
        tflite_path=str(tmp_path / "hello.tflite"),
        analysis_results={
            "model_valid": True,
            "validation_results": {
                "errors": [],
                "warnings": [],
                "info": {"k": "v"},
            },
            "performance_estimation": {
                "model_size_kb": 1.2,
                "estimated_latency_ms": 3.4,
                "tensor_arena_estimate_kb": 50.0,
            },
            "architecture_analysis": {
                "layer_count": 10,
                "operators": ["CONV_2D"],
                "has_quantization": True,
            },
        },
    )

    assert res["model_filename"] == "hello.tflite"
    assert res["tensor_arena_size"] == 777
    assert "_metadata" not in saved["manifest"]
    assert Path(saved["path"]).exists()
