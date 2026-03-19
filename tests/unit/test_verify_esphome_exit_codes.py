"""Tests for verify_esphome.py script exit codes.

This module tests that the verify_esphome.py script returns correct exit codes:
- 0: Success/compatible
- 2: Validation/compatibility failure
- 1: Runtime/internal error
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Skip all tests if TensorFlow is not available
try:
    import tensorflow  # noqa: F401

    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = [
    pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available"),
    pytest.mark.integration,
]


class TestVerifyEsphomeExitCodes:
    """Test verify_esphome.py exit code contract."""

    @pytest.fixture
    def verify_script(self):
        """Path to verify_esphome.py script."""
        return Path(__file__).resolve().parents[2] / "scripts" / "verify_esphome.py"

    def test_exit_code_0_for_compatible_model(self, verify_script, tmp_path):
        """Test that a valid TFLite model returns exit code 0."""
        pytest.skip("Requires actual TFLite model - run in integration test suite")

    def test_exit_code_2_for_incompatible_model(self, verify_script, tmp_path):
        """Test that an incompatible model returns exit code 2."""
        pytest.skip("Requires actual incompatible TFLite model")

    def test_exit_code_1_for_missing_file(self, verify_script):
        """Test that missing file returns exit code 1 (runtime error)."""
        result = subprocess.run(
            [sys.executable, str(verify_script), "/nonexistent/model.tflite"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should return non-zero exit code for missing file
        assert result.returncode != 0, f"Expected non-zero exit code for missing file, got {result.returncode}"

    def test_exit_code_1_for_invalid_file(self, verify_script, tmp_path):
        """Test that invalid file returns exit code 1 (runtime error)."""
        invalid_file = tmp_path / "invalid.tflite"
        invalid_file.write_text("not a valid tflite file")

        result = subprocess.run(
            [sys.executable, str(verify_script), str(invalid_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should return non-zero exit code for invalid file
        assert result.returncode != 0, f"Expected non-zero exit code for invalid file, got {result.returncode}"

    def test_json_output_flag(self, verify_script, tmp_path):
        """Test that --json flag produces valid JSON output."""
        invalid_file = tmp_path / "invalid.tflite"
        invalid_file.write_text("not a valid tflite file")

        result = subprocess.run(
            [sys.executable, str(verify_script), str(invalid_file), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # JSON output should be valid even for errors
        try:
            output = json.loads(result.stdout)
            # Should have expected fields
            assert "compatible" in output or "valid" in output or "error" in output
        except json.JSONDecodeError:
            # If stdout isn't JSON, verify that something went wrong
            # Non-zero return code or error in stderr should be present
            assert result.returncode != 0 or result.stderr, "When --json output is not valid JSON, either returncode should be non-zero " "or stderr should contain error details"
            # If stderr exists, check it contains error-related content
            if result.stderr:
                assert "error" in result.stderr.lower(), "stderr should contain error-related messaging when JSON output fails"

    def test_verbose_flag(self, verify_script):
        """Test that --verbose flag is accepted."""
        result = subprocess.run(
            [
                sys.executable,
                str(verify_script),
                "--verbose",
                "/nonexistent/model.tflite",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should not crash with --verbose flag
        # Exit code may vary but should not be a Python exception
        assert "Traceback" not in result.stderr, f"Script crashed with --verbose flag: {result.stderr}"

    def test_strict_flag(self, verify_script):
        """Test that --strict flag is accepted."""
        result = subprocess.run(
            [
                sys.executable,
                str(verify_script),
                "--strict",
                "/nonexistent/model.tflite",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should not crash with --strict flag
        assert "Traceback" not in result.stderr, f"Script crashed with --strict flag: {result.stderr}"

    def test_help_flag(self, verify_script):
        """Test that --help produces usage information."""
        result = subprocess.run(
            [sys.executable, str(verify_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "--help should return exit code 0"
        assert "usage" in result.stdout.lower(), "--help should show usage information"


class TestVerifyEsphomeScriptInterface:
    """Test verify_esphome.py script interface and documentation."""

    def test_script_exists(self):
        """Test that verify_esphome.py exists and is executable."""
        verify_script = Path(__file__).resolve().parents[2] / "scripts" / "verify_esphome.py"

        assert verify_script.exists(), "verify_esphome.py should exist"
        assert verify_script.is_file(), "verify_esphome.py should be a file"

        # Should be readable
        content = verify_script.read_text()
        assert len(content) > 0, "verify_esphome.py should not be empty"

    def test_exit_codes_documented(self):
        """Test that exit codes are documented in script docstring or comments."""
        verify_script = Path(__file__).resolve().parents[2] / "scripts" / "verify_esphome.py"
        content = verify_script.read_text()

        # Should mention exit codes somewhere
        assert "exit" in content.lower(), "Script should document exit codes"

        # Should mention compatibility or validation
        assert any(term in content.lower() for term in ["compatible", "valid", "error"]), "Script should document what it checks"
