"""Analyze official MWW TFLite models for ESPHome compatibility comparison.

Uses standard TFLite Interpreter API + flatbuffer parsing for subgraph analysis.
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def get_subgraph_count_from_flatbuffer(model_data: bytes) -> int:
    """Parse TFLite flatbuffer to get subgraph count."""
    try:
        from tensorflow.lite.python import schema_py_generated as schema

        model = schema.ModelT.InitFromPackedBuf(model_data, 0)
        return len(model.subgraphs) if model.subgraphs else 0
    except Exception:
        pass
    try:
        from tflite.Model import Model

        buf = bytearray(model_data)
        model = Model.GetRootAs(buf, 0)
        return model.SubgraphsLength()
    except Exception:
        pass
    return -1


def analyze_tflite_model(model_path: str) -> dict:
    """Deep analysis of a TFLite model."""
    with open(model_path, "rb") as f:
        model_data = f.read()

    result = {
        "file": os.path.basename(model_path),
        "file_size_bytes": len(model_data),
        "num_subgraphs": get_subgraph_count_from_flatbuffer(model_data),
    }

    interpreter = tf.lite.Interpreter(model_content=model_data)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()

    result["main_subgraph"] = {"num_tensors": len(tensor_details), "num_inputs": len(input_details), "num_outputs": len(output_details)}

    result["inputs"] = []
    for inp in input_details:
        info = {"name": inp["name"], "shape": inp["shape"].tolist(), "dtype": inp["dtype"].__name__ if hasattr(inp["dtype"], "__name__") else str(inp["dtype"]), "index": int(inp["index"])}
        qp = inp.get("quantization_parameters", {})
        if qp and len(qp.get("scales", [])) > 0:
            info["quant_scale"] = float(qp["scales"][0])
            info["quant_zero_point"] = int(qp["zero_points"][0])
        result["inputs"].append(info)

    result["outputs"] = []
    for out in output_details:
        info = {"name": out["name"], "shape": out["shape"].tolist(), "dtype": out["dtype"].__name__ if hasattr(out["dtype"], "__name__") else str(out["dtype"]), "index": int(out["index"])}
        qp = out.get("quantization_parameters", {})
        if qp and len(qp.get("scales", [])) > 0:
            info["quant_scale"] = float(qp["scales"][0])
            info["quant_zero_point"] = int(qp["zero_points"][0])
        result["outputs"].append(info)

    state_tensors = []
    for td in tensor_details:
        name = td["name"].lower()
        if any(kw in name for kw in ["stream", "state", "assignvariable", "readvariable", "variable"]):
            info = {"name": td["name"], "shape": td["shape"].tolist(), "dtype": td["dtype"].__name__ if hasattr(td["dtype"], "__name__") else str(td["dtype"]), "index": int(td["index"])}
            qp = td.get("quantization_parameters", {})
            if qp and len(qp.get("scales", [])) > 0:
                info["quant_scale"] = float(qp["scales"][0])
                info["quant_zero_point"] = int(qp["zero_points"][0])
            state_tensors.append(info)
    result["state_tensors"] = state_tensors

    seen_shapes = set()
    unique_states = []
    for st in state_tensors:
        shape_key = tuple(st["shape"])
        if shape_key not in seen_shapes:
            seen_shapes.add(shape_key)
            unique_states.append(st)
    result["unique_state_shapes"] = unique_states

    try:
        op_details = interpreter._get_ops_details()
        unique_ops = sorted(set(op.get("op_name", "unknown") for op in op_details))
        result["unique_ops"] = unique_ops
        result["num_unique_ops"] = len(unique_ops)
        result["total_ops"] = len(op_details)
    except Exception as e:
        result["ops_error"] = str(e)

    return result


def print_model_analysis(result: dict):
    indent = "  "
    print(f"{indent}File size: {result['file_size_bytes']:,} bytes")
    print(f"{indent}Subgraphs: {result['num_subgraphs']}")
    print(f"{indent}Main subgraph tensors: {result['main_subgraph']['num_tensors']}")
    print(f"\n{indent}INPUTS ({len(result['inputs'])}):")
    for inp in result["inputs"]:
        print(f"{indent}  [{inp['index']}] {inp['name']}: shape={inp['shape']} dtype={inp['dtype']}")
        if "quant_scale" in inp:
            print(f"{indent}      quant: scale={inp['quant_scale']:.15f}, zp={inp['quant_zero_point']}")
    print(f"\n{indent}OUTPUTS ({len(result['outputs'])}):")
    for out in result["outputs"]:
        print(f"{indent}  [{out['index']}] {out['name']}: shape={out['shape']} dtype={out['dtype']}")
        if "quant_scale" in out:
            print(f"{indent}      quant: scale={out['quant_scale']:.15f}, zp={out['quant_zero_point']}")
    print(f"\n{indent}STATE VARIABLES (unique shapes: {len(result['unique_state_shapes'])}):")
    for st in result["unique_state_shapes"]:
        print(f"{indent}  {st['name']}: shape={st['shape']} dtype={st['dtype']}")
        if "quant_scale" in st:
            print(f"{indent}      quant: scale={st['quant_scale']:.10f}, zp={st['quant_zero_point']}")
    if "unique_ops" in result:
        print(f"\n{indent}OPS: {result['total_ops']} total, {result['num_unique_ops']} unique")
        print(f"{indent}  {', '.join(result['unique_ops'])}")


def compare_models(official_results, our_results):
    print("\n" + "=" * 80)
    print("DETAILED COMPATIBILITY COMPARISON")
    print("=" * 80)

    ref_name = "okay_nabu.tflite"
    if ref_name not in official_results:
        ref_name = list(official_results.keys())[0]
    ref = official_results[ref_name]

    esphome_ops = {
        "CALL_ONCE",
        "VAR_HANDLE",
        "READ_VARIABLE",
        "ASSIGN_VARIABLE",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "CONCATENATION",
        "RESHAPE",
        "STRIDED_SLICE",
        "LOGISTIC",
        "QUANTIZE",
        "SPLIT_V",
        "ADD",
        "SOFTMAX",
        "PAD",
        "PADV2",
        "MEAN",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
    }

    for our_name, our in our_results.items():
        print(f"\n{'─' * 60}")
        print(f"Comparing: {our_name} vs {ref_name} (official reference)")
        print(f"{'─' * 60}")

        ref_inp = ref["inputs"][0] if ref["inputs"] else {}
        our_inp = our["inputs"][0] if our["inputs"] else {}
        ref_out = ref["outputs"][0] if ref["outputs"] else {}
        our_out = our["outputs"][0] if our["outputs"] else {}

        checks = [
            ("Input shape", ref_inp.get("shape") == our_inp.get("shape"), f"ours={our_inp.get('shape')} ref={ref_inp.get('shape')}"),
            ("Input dtype", ref_inp.get("dtype") == our_inp.get("dtype"), f"ours={our_inp.get('dtype')} ref={ref_inp.get('dtype')}"),
            ("Output shape", ref_out.get("shape") == our_out.get("shape"), f"ours={our_out.get('shape')} ref={ref_out.get('shape')}"),
            ("Output dtype", ref_out.get("dtype") == our_out.get("dtype"), f"ours={our_out.get('dtype')} ref={ref_out.get('dtype')}"),
        ]

        # Input quant
        if ref_inp.get("quant_scale") and our_inp.get("quant_scale"):
            ok = abs(ref_inp["quant_scale"] - our_inp["quant_scale"]) < 0.001
            checks.append(("Input quant scale", ok, f"ours={our_inp['quant_scale']:.10f} ref={ref_inp['quant_scale']:.10f}"))
        if ref_inp.get("quant_zero_point") is not None and our_inp.get("quant_zero_point") is not None:
            ok = ref_inp["quant_zero_point"] == our_inp["quant_zero_point"]
            checks.append(("Input quant zp", ok, f"ours={our_inp['quant_zero_point']} ref={ref_inp['quant_zero_point']}"))

        # Output quant
        if ref_out.get("quant_scale") and our_out.get("quant_scale"):
            ok = abs(ref_out["quant_scale"] - our_out["quant_scale"]) < 0.0001
            checks.append(("Output quant scale", ok, f"ours={our_out['quant_scale']:.10f} ref={ref_out['quant_scale']:.10f}"))
        if ref_out.get("quant_zero_point") is not None and our_out.get("quant_zero_point") is not None:
            ok = ref_out["quant_zero_point"] == our_out["quant_zero_point"]
            checks.append(("Output quant zp", ok, f"ours={our_out['quant_zero_point']} ref={ref_out['quant_zero_point']}"))

        for name, ok, detail in checks:
            status = "✓" if ok else "✗"
            print(f"  {status} {name}: {detail}")

        # State variables
        print("\n  State variables:")
        print(f"    Reference ({ref_name}):")
        for st in ref["unique_state_shapes"]:
            print(f"      {st['name']}: {st['shape']} ({st['dtype']})")
        print(f"    Ours ({our_name}):")
        for st in our["unique_state_shapes"]:
            print(f"      {st['name']}: {st['shape']} ({st['dtype']})")

        # Ops
        ref_ops = set(ref.get("unique_ops", []))
        our_ops = set(our.get("unique_ops", []))
        extra = our_ops - ref_ops
        missing = ref_ops - our_ops
        unsupported = our_ops - esphome_ops

        print("\n  Op compatibility:")
        print(f"    Common: {len(our_ops & ref_ops)}")
        if extra:
            print(f"    Extra ops (not in ref): {extra}")
        if missing:
            print(f"    Missing ops (in ref not us): {missing}")
        if unsupported:
            print(f"    ✗ CRITICAL: Ops NOT in ESPHome resolver: {unsupported}")
        else:
            print("    ✓ All our ops are in ESPHome's 20-op resolver set")

        # Verdict
        fails = [c for c in checks if not c[1]]
        if fails or unsupported:
            print("\n  ⚠ COMPATIBILITY ISSUES:")
            for name, _, detail in fails:
                print(f"    ✗ {name}: {detail}")
            if unsupported:
                print(f"    ✗ Unsupported ops: {unsupported}")
        else:
            print("\n  ✓ ALL CHECKS PASSED — ESPHome compatible")


def main():
    official_dir = "official_mww_models"
    our_dir = "models/exported"

    print("=" * 80)
    print("MICROWAKEWORD TFLITE MODEL COMPATIBILITY ANALYSIS")
    print("=" * 80)

    official_results = {}
    for fname in sorted(os.listdir(official_dir)):
        if not fname.endswith(".tflite") or "vad" in fname.lower():
            continue
        fpath = os.path.join(official_dir, fname)
        print(f"\n{'─' * 60}")
        print(f"OFFICIAL: {fname}")
        print(f"{'─' * 60}")
        result = analyze_tflite_model(fpath)
        official_results[fname] = result
        print_model_analysis(result)

    our_results = {}
    if os.path.isdir(our_dir):
        for fname in sorted(os.listdir(our_dir)):
            if not fname.endswith(".tflite"):
                continue
            fpath = os.path.join(our_dir, fname)
            print(f"\n{'─' * 60}")
            print(f"OUR MODEL: {fname}")
            print(f"{'─' * 60}")
            result = analyze_tflite_model(fpath)
            our_results[fname] = result
            print_model_analysis(result)

    if our_results and official_results:
        compare_models(official_results, our_results)

    # Cross-model summary
    print(f"\n{'=' * 80}")
    print("ALL OFFICIAL MODELS SUMMARY")
    print(f"{'=' * 80}")
    for name, r in official_results.items():
        inp = r["inputs"][0] if r["inputs"] else {}
        out = r["outputs"][0] if r["outputs"] else {}
        print(
            f"  {name}: in={inp.get('shape')}/{inp.get('dtype')} out={out.get('shape')}/{out.get('dtype')} ops={r.get('total_ops')}/{r.get('num_unique_ops')} tensors={r['main_subgraph']['num_tensors']}"
        )

    with open("official_model_analysis.json", "w") as f:
        json.dump({"official": official_results, "ours": our_results}, f, indent=2, default=str)
    print("\nFull JSON: official_model_analysis.json")


if __name__ == "__main__":
    main()
