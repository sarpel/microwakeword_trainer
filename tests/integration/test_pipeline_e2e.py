"""End-to-end integration test for build → export → verification pipeline."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import tensorflow as tf

from src.export.tflite import export_streaming_tflite, verify_esphome_compatibility
from src.model.architecture import build_model


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_build_save_export_verify(tmp_path: Path) -> None:
    """Validate model build, checkpoint save, TFLite export, and ESPHome verification."""
    start = time.monotonic()

    input_shape = (15, 40)
    model = build_model(
        input_shape=input_shape,
        first_conv_filters=32,
        first_conv_kernel_size=5,
        stride=3,
        pointwise_filters="64,64,64,64",
        mixconv_kernel_sizes="[5],[7,11],[9,15],[23]",
        repeat_in_block="1,1,1,1",
        residual_connection="0,1,1,1",
        dropout_rate=0.0,
        l2_regularization=0.0,
        mode="non_stream",
    )

    _ = model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)

    checkpoint_path = tmp_path / "checkpoint.weights.h5"
    model.save_weights(str(checkpoint_path))
    assert checkpoint_path.exists()

    export_dir = tmp_path / "exported"
    export_config = {
        "first_conv_filters": 32,
        "first_conv_kernel": 5,
        "stride": 3,
        "pointwise_filters": [64, 64, 64, 64],
        "mixconv_kernel_sizes": [[5], [7, 11], [9, 15], [23]],
        "residual_connections": [0, 1, 1, 1],
        "mel_bins": 40,
        "export": {"representative_dataset_size": 8},
    }

    result = export_streaming_tflite(
        checkpoint_path=str(checkpoint_path),
        output_dir=str(export_dir),
        model_name="pipeline_e2e",
        config=export_config,
        quantize=True,
    )

    tflite_path = Path(result["tflite_path"])
    assert tflite_path.exists(), "Expected exported TFLite file to exist"

    verify_result = verify_esphome_compatibility(str(tflite_path))
    assert "valid" in verify_result and "checks" in verify_result

    verify_script = Path(__file__).resolve().parents[2] / "scripts" / "verify_esphome.py"
    proc = subprocess.run(
        [sys.executable, str(verify_script), str(tflite_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"verify_esphome.py failed: {proc.stderr}\n{proc.stdout}"

    assert (time.monotonic() - start) < 60.0
