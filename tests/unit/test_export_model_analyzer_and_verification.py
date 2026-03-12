"""Unit tests for export model analyzer and verification modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.export import model_analyzer, verification


def test_parse_analysis_output_extracts_core_fields() -> None:
    text = """
Subgraph#0
Op#0 (CONV_2D)
Op#1 (DEPTHWISE_CONV_2D)
T#0 INPUT [1, 3, 40] int8
T#1 OUTPUT [1, 1] uint8
quantization: int8
"""
    out = model_analyzer._parse_analysis_output(text, b"abcd")
    assert out["subgraph_count"] == 1
    assert out["layer_count"] == 2
    assert "CONV_2D" in out["operators"]
    assert out["input_tensors"][0]["shape"] == [1, 3, 40]
    assert out["has_quantization"] is True


def test_build_interpreter_analysis_fallback(monkeypatch) -> None:
    class DummyInterpreter:
        def __init__(self, model_content):
            self.model_content = model_content

        def allocate_tensors(self):
            return None

        def get_input_details(self):
            return [{"shape": [1, 3, 40], "dtype": np.int8, "index": 0, "quantization_parameters": {"scales": [0.1], "zero_points": [-128]}}]

        def get_output_details(self):
            return [{"shape": [1, 1], "dtype": np.uint8, "index": 9, "quantization_parameters": {"scales": [0.0039], "zero_points": [0]}}]

    monkeypatch.setattr(model_analyzer, "Interpreter", DummyInterpreter)
    txt = model_analyzer._build_interpreter_analysis(b"x")
    assert "INPUT" in txt
    assert "OUTPUT" in txt


def test_analyze_model_architecture_with_analyzer_fallback(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "m.tflite"
    p.write_bytes(b"abc")

    class DummyAnalyzer:
        @staticmethod
        def analyze(**kwargs):
            raise AttributeError("removed")

    monkeypatch.setattr(model_analyzer.tf.lite.experimental, "Analyzer", DummyAnalyzer)
    monkeypatch.setattr(model_analyzer, "_build_interpreter_analysis", lambda _b: "T#0 INPUT [1, 3, 40] int8\nT#1 OUTPUT [1, 1] uint8")
    out = model_analyzer.analyze_model_architecture(str(p))
    assert out["model_path"] == str(p)
    assert out["model_size_bytes"] == 3
    assert out["input_tensors"]


def test_validate_model_quality_invalid_and_valid_paths(tmp_path: Path, monkeypatch) -> None:
    missing = model_analyzer.validate_model_quality(str(tmp_path / "missing.tflite"))
    assert missing["valid"] is False

    p = tmp_path / "ok.tflite"
    p.write_bytes(b"x")

    class DummyInterpreter:
        def __init__(self, model_path):
            self.model_path = model_path

        def allocate_tensors(self):
            return None

        def get_input_details(self):
            return [{"shape": np.array([1, 3, 40]), "dtype": np.int8, "quantization_parameters": {"scales": np.array([0.101961], dtype=np.float32), "zero_points": np.array([-128], dtype=np.int32)}}]

        def get_output_details(self):
            return [{"shape": np.array([1, 1]), "dtype": np.uint8, "quantization_parameters": {"scales": np.array([0.00390625], dtype=np.float32), "zero_points": np.array([0], dtype=np.int32)}}]

        def get_tensor_details(self):
            return [
                {"name": "state_0"},
                {"name": "state_1"},
                {"name": "state_2"},
                {"name": "state_3"},
                {"name": "state_4"},
                {"name": "state_5"},
                {"name": "x"},
            ]

        def num_subgraphs(self):
            return 2

    monkeypatch.setattr(model_analyzer, "Interpreter", DummyInterpreter)
    out = model_analyzer.validate_model_quality(str(p))
    assert out["valid"] is True
    assert out["info"]["input_dtype_correct"] is True
    assert out["info"]["output_dtype_correct"] is True


def test_compare_estimate_gpu_and_report(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.tflite"
    b = tmp_path / "b.tflite"
    a.write_bytes(b"a" * 200)
    b.write_bytes(b"b" * 400)

    monkeypatch.setattr(
        model_analyzer, "analyze_model_architecture", lambda path: {"model_size_bytes": 200 if path == str(a) else 400, "layer_count": 5, "operators": ["CONV_2D"], "has_quantization": True}
    )
    monkeypatch.setattr(model_analyzer, "validate_model_quality", lambda *args, **kwargs: {"valid": True, "info": {"input_dtype_correct": True, "output_dtype_correct": True}})
    cmp_out = model_analyzer.compare_models(str(a), str(b))
    assert cmp_out["differences"]["size_diff_bytes"] == 200

    class DummyInterpreter:
        def __init__(self, model_path):
            self.model_path = model_path

        def allocate_tensors(self):
            return None

        def get_tensor_details(self):
            return [{"shape": [1, 3, 40], "dtype": np.int8}, {"shape": [1, 1], "dtype": np.uint8}]

    monkeypatch.setattr(model_analyzer, "Interpreter", DummyInterpreter)
    perf = model_analyzer.estimate_performance(str(a))
    assert perf["model_size_kb"] > 0
    assert perf["tensor_arena_estimate"] > 0

    class DummyAnalyzer:
        @staticmethod
        def analyze(**kwargs):
            return "GPU compatible and supports GPU"

    monkeypatch.setattr(model_analyzer.tf.lite.experimental, "Analyzer", DummyAnalyzer)
    gpu = model_analyzer.check_gpu_compatibility(str(a))
    assert gpu["gpu_compatibility_checked"] is True

    report = model_analyzer.generate_model_report(str(a))
    assert report["summary"]["size_kb"] >= 0


def test_verification_estimate_and_verify(monkeypatch) -> None:
    class DummyInterpreter:
        def __init__(self, model_path=None, experimental_op_resolver_type=None):
            self.model_path = model_path

        def allocate_tensors(self):
            return None

        def get_input_details(self):
            return [
                {
                    "shape": np.array([1, 3, 40]),
                    "dtype": np.int8,
                    "index": 0,
                    "quantization_parameters": {"scales": np.array([0.101961], dtype=np.float32), "zero_points": np.array([-128], dtype=np.int32)},
                }
            ]

        def get_output_details(self):
            return [
                {
                    "shape": np.array([1, 1]),
                    "dtype": np.uint8,
                    "index": 1,
                    "quantization_parameters": {"scales": np.array([0.00390625], dtype=np.float32), "zero_points": np.array([0], dtype=np.int32)},
                }
            ]

        def _get_ops_details(self):
            return [
                {"op_name": "CALL_ONCE", "outputs": [10]},
                {"op_name": "VAR_HANDLE", "outputs": [11]},
                {"op_name": "VAR_HANDLE", "outputs": [12]},
                {"op_name": "VAR_HANDLE", "outputs": [13]},
                {"op_name": "VAR_HANDLE", "outputs": [14]},
                {"op_name": "VAR_HANDLE", "outputs": [15]},
                {"op_name": "VAR_HANDLE", "outputs": [16]},
                {"op_name": "READ_VARIABLE", "outputs": [20]},
                {"op_name": "READ_VARIABLE", "outputs": [21]},
                {"op_name": "READ_VARIABLE", "outputs": [22]},
                {"op_name": "READ_VARIABLE", "outputs": [23]},
                {"op_name": "READ_VARIABLE", "outputs": [24]},
                {"op_name": "READ_VARIABLE", "outputs": [25]},
                {"op_name": "ASSIGN_VARIABLE", "inputs": [11, 20], "outputs": [0]},
                {"op_name": "ASSIGN_VARIABLE", "inputs": [12, 21], "outputs": [0]},
                {"op_name": "ASSIGN_VARIABLE", "inputs": [13, 22], "outputs": [0]},
                {"op_name": "ASSIGN_VARIABLE", "inputs": [14, 23], "outputs": [0]},
                {"op_name": "ASSIGN_VARIABLE", "inputs": [15, 24], "outputs": [0]},
                {"op_name": "ASSIGN_VARIABLE", "inputs": [16, 25], "outputs": [0]},
                {"op_name": "CONV_2D", "outputs": [0]},
            ]

        def get_tensor_details(self):
            details = []
            expected = [
                (20, (1, 2, 1, 40)),
                (21, (1, 4, 1, 32)),
                (22, (1, 10, 1, 64)),
                (23, (1, 14, 1, 64)),
                (24, (1, 22, 1, 64)),
                (25, (1, 5, 1, 64)),
            ]
            for idx, shp in expected:
                details.append(
                    {
                        "index": idx,
                        "shape": np.array(shp),
                        "dtype": np.int8,
                        "name": f"state_{idx}",
                        "quantization_parameters": {
                            "scales": np.array([0.1], dtype=np.float32),
                            "zero_points": np.array([-128], dtype=np.int32),
                        },
                    }
                )
            details += [
                {"index": 0, "shape": np.array([1, 3, 40]), "dtype": np.int8, "name": "input"},
                {"index": 1, "shape": np.array([1, 1]), "dtype": np.uint8, "name": "output"},
            ]
            return details

        def num_subgraphs(self):
            return 2

        def set_tensor(self, *_args, **_kwargs):
            return None

        def invoke(self):
            return None

        def get_tensor(self, _index):
            return np.array([[128]], dtype=np.uint8)

    fake_tf = SimpleNamespace(
        lite=SimpleNamespace(
            experimental=SimpleNamespace(OpResolverType=SimpleNamespace(BUILTIN_WITHOUT_DEFAULT_DELEGATES="x")),
            Interpreter=DummyInterpreter,
        )
    )
    monkeypatch.setattr(verification, "tf", fake_tf)
    res = verification.verify_tflite_model("fake.tflite")
    assert res["checks"]["input_shape"] is True
    assert res["checks"]["output_dtype"] is True
    assert res["checks"]["state_shapes"] is True
    assert res["checks"]["state_payload_dtypes_int8"] is True
    assert res["checks"]["read_payload_quant_params"] is True
    assert res["checks"]["assign_payload_dtypes_int8"] is True
    assert res["checks"]["assign_payload_quant_params"] is True
