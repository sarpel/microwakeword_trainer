"""Architectural fitness tests to prevent pipeline regressions.

These checks are static/pattern-based guardrails over source text/AST.
They intentionally avoid fragile line-number assertions and runtime behavior.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.model.architecture import MixedNet

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ARCH_PATH = SRC_ROOT / "model" / "architecture.py"
TFLITE_PATH = SRC_ROOT / "export" / "tflite.py"
AUTOTUNER_PATH = SRC_ROOT / "tuning" / "autotuner.py"
LOADER_PATH = REPO_ROOT / "config" / "loader.py"
PRESETS_DIR = REPO_ROOT / "config" / "presets"


def _python_files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.py") if p.is_file())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_token_absent_in_python_files(root: Path, token: str) -> None:
    offenders: list[str] = []
    for py_file in _python_files_under(root):
        if token in _read(py_file):
            offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert not offenders, f"Found forbidden token {token!r} in: {offenders}"


@pytest.mark.integration
def test_no_legacy_convert_to_tflite() -> None:
    """Guardrail 1: no legacy convert_to_tflite references in src Python code."""
    _assert_token_absent_in_python_files(SRC_ROOT, "convert_to_tflite")


@pytest.mark.integration
def test_no_model_export_usage() -> None:
    """Guardrail 2: forbid model.export( usage; ExportArchive is required."""
    _assert_token_absent_in_python_files(SRC_ROOT, "model.export(")


@pytest.mark.integration
def test_no_trainable_weights_in_export_serialization() -> None:
    """Guardrail 3: export code must not use trainable_weights serialization."""
    _assert_token_absent_in_python_files(SRC_ROOT / "export", "trainable_weights")


@pytest.mark.integration
def test_uint8_output_dtype() -> None:
    """Guardrail 4: TFLite export output dtype must remain uint8 (not int8)."""
    content = _read(TFLITE_PATH)
    assert "converter.inference_output_type = tf.uint8" in content
    assert "converter.inference_output_type = tf.int8" not in content


@pytest.mark.integration
def test_mixednet_default_residual_connections() -> None:
    """Guardrail 5: MixedNet default residual connections are fixed."""
    tree = ast.parse(_read(ARCH_PATH))

    init_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MixedNet":
            init_fn = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
            break

    assert init_fn is not None, "MixedNet.__init__ not found"

    # Accept either direct default in signature OR fallback assignment in body.
    sig = inspect.signature(MixedNet.__init__)
    sig_default = sig.parameters["residual_connections"].default
    if sig_default == [0, 1, 1, 1]:
        return

    found_fallback = False
    for node in ast.walk(init_fn):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "residual_connections"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Is)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            ):
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "residual_connections"
                        and isinstance(stmt.value, ast.List)
                    ):
                        literal = [elt.value for elt in stmt.value.elts if isinstance(elt, ast.Constant)]
                        if literal == [0, 1, 1, 1]:
                            found_fallback = True
    assert found_fallback, "MixedNet residual_connections must default/fallback to [0,1,1,1]"


@pytest.mark.integration
def test_create_okay_nabu_model_residual_connections_literal() -> None:
    """Guardrail 6: create_okay_nabu_model must pass [0,1,1,1] to MixedNet."""
    tree = ast.parse(_read(ARCH_PATH))

    target_call: ast.Call | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "create_okay_nabu_model":
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "MixedNet":
                    target_call = child
                    break

    assert target_call is not None, "Could not find MixedNet(...) call in create_okay_nabu_model"
    residual_kw = next((kw for kw in target_call.keywords if kw.arg == "residual_connections"), None)
    assert residual_kw is not None, "Missing residual_connections keyword in create_okay_nabu_model"

    value = residual_kw.value
    assert isinstance(value, ast.List), "residual_connections must be a literal list"
    literal = [elt.value for elt in value.elts if isinstance(elt, ast.Constant)]
    assert literal == [0, 1, 1, 1]


@pytest.mark.integration
def test_no_int8_shadow_artifacts_in_tuning() -> None:
    """Guardrail 7: remove legacy INT8 shadow evaluation artifacts from tuning."""
    content = _read(AUTOTUNER_PATH)
    for forbidden in ("_evaluate_int8", "int8_shadow", "eval_results_int8"):
        assert forbidden not in content


@pytest.mark.integration
def test_build_core_layers_used_by_mixednet_and_streaming_export_model() -> None:
    """Guardrail 8: shared build_core_layers factory must be used by both models."""
    arch_tree = ast.parse(_read(ARCH_PATH))
    tflite_tree = ast.parse(_read(TFLITE_PATH))

    def _class_calls_build_core_layers(tree: ast.AST, class_name: str) -> bool:
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "build_core_layers":
                        return True
        return False

    assert _class_calls_build_core_layers(arch_tree, "MixedNet")
    assert _class_calls_build_core_layers(tflite_tree, "StreamingExportModel")


@pytest.mark.integration
def test_no_enable_pcan_in_loader_and_presets() -> None:
    """Additional guardrail: no Python-configurable PCAN flag should exist."""
    assert "enable_pcan" not in _read(LOADER_PATH)
    for preset in sorted(PRESETS_DIR.glob("*.yaml")):
        assert "enable_pcan" not in _read(preset), f"Found enable_pcan in {preset.name}"
