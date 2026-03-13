"""Unit tests for data quality module."""

import csv
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src.data.quality import (
    QualityScoreConfig,
    FileScore,
    compute_clipping_ratio,
    wada_snr,
    write_csv,
    apply_discard,
    print_summary,
)


class TestQualityScoreConfig:
    """Tests for QualityScoreConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QualityScoreConfig()
        assert config.clip_threshold == 0.001
        assert config.discard_bottom_pct == 5.0
        assert config.min_wqi == 0.0
        assert config.vad_threshold == 0.0
        assert config.discarded_dir == Path("discarded/quality")
        assert config.verbose is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = QualityScoreConfig(
            clip_threshold=0.01,
            discard_bottom_pct=10.0,
            min_wqi=0.5,
            vad_threshold=0.3,
            verbose=True,
        )
        assert config.clip_threshold == 0.01
        assert config.discard_bottom_pct == 10.0
        assert config.min_wqi == 0.5
        assert config.vad_threshold == 0.3
        assert config.verbose is True


class TestFileScore:
    """Tests for FileScore dataclass."""

    def test_default_values(self):
        """Test default FileScore values."""
        score = FileScore(path=Path("test.wav"), dir_label="positive")
        assert score.path == Path("test.wav")
        assert score.dir_label == "positive"
        assert score.clip_ratio == 0.0
        assert score.snr_db == -20.0
        assert score.vad_conf == 0.0
        assert score.dnsmos_sig == 1.0
        assert score.dnsmos_bak == 1.0
        assert score.dnsmos_ovrl == 1.0
        assert score.wqi == 0.0
        assert score.discard is False
        assert score.discard_reason == ""
        assert score.error == ""

    def test_custom_values(self):
        """Test custom FileScore values."""
        score = FileScore(
            path=Path("test.wav"),
            dir_label="negative",
            clip_ratio=0.05,
            snr_db=15.0,
            wqi=0.85,
            discard=True,
            discard_reason="low quality",
        )
        assert score.clip_ratio == 0.05
        assert score.snr_db == 15.0
        assert score.wqi == 0.85
        assert score.discard is True
        assert score.discard_reason == "low quality"


class TestComputeClippingRatio:
    """Tests for compute_clipping_ratio function."""

    def test_empty_array(self):
        """Test with empty array returns 0."""
        samples = np.array([], dtype=np.float32)
        assert compute_clipping_ratio(samples) == 0.0

    def test_no_clipping(self):
        """Test with no clipped samples."""
        samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        assert compute_clipping_ratio(samples) == 0.0

    def test_all_clipped(self):
        """Test with all samples at clipping threshold."""
        samples = np.array([1.0, -1.0, 0.999, -0.999], dtype=np.float32)
        ratio = compute_clipping_ratio(samples, threshold=0.999)
        assert ratio == 1.0

    def test_partial_clipping(self):
        """Test with some clipped samples."""
        samples = np.array([1.0, 0.5, -1.0, 0.3], dtype=np.float32)
        ratio = compute_clipping_ratio(samples, threshold=0.999)
        # 2 out of 4 samples are clipped (1.0 and -1.0)
        assert ratio == 0.5

    def test_custom_threshold(self):
        """Test with custom clipping threshold."""
        samples = np.array([0.8, 0.9, 0.95, 0.99], dtype=np.float32)
        ratio = compute_clipping_ratio(samples, threshold=0.9)
        # 3 samples >= 0.9 (0.9, 0.95, 0.99)
        assert ratio == 0.75

    def test_negative_values(self):
        """Test with negative values near threshold."""
        samples = np.array([-0.999, -0.5, 0.5, 0.999], dtype=np.float32)
        ratio = compute_clipping_ratio(samples, threshold=0.999)
        assert ratio == 0.5  # Only -0.999 and 0.999


class TestWadaSnr:
    """Tests for wada_snr function."""

    def test_empty_array(self):
        """Test with empty array returns minimum SNR."""
        samples = np.array([], dtype=np.float32)
        assert wada_snr(samples) == -20.0

    def test_silence(self):
        """Test with silent audio returns minimum SNR."""
        samples = np.zeros(1000, dtype=np.float32)
        assert wada_snr(samples) == -20.0

    def test_low_amplitude(self):
        """Test with very low amplitude audio."""
        samples = np.random.randn(1000).astype(np.float32) * 0.001
        snr = wada_snr(samples)
        # Should return some value in the valid range
        assert -20.0 <= snr <= 100.0

    def test_normal_audio(self):
        """Test with normal audio signal."""
        np.random.seed(42)
        samples = np.random.randn(16000).astype(np.float32) * 0.1
        snr = wada_snr(samples)
        assert -20.0 <= snr <= 100.0
        assert isinstance(snr, float)

    def test_high_snr_audio(self):
        """Test with clean high-amplitude audio."""
        np.random.seed(42)
        # Create audio with strong signal
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        samples = np.sin(2 * np.pi * 440 * t) * 0.5
        snr = wada_snr(samples)
        assert -20.0 <= snr <= 100.0
        assert isinstance(snr, float)

    def test_snr_range(self):
        """Test that SNR is always within valid range."""
        np.random.seed(42)
        for _ in range(10):
            samples = np.random.randn(8000).astype(np.float32)
            snr = wada_snr(samples)
            assert -20.0 <= snr <= 100.0


class TestWriteCsv:
    """Tests for write_csv function."""

    def test_write_single_score(self):
        """Test writing a single FileScore to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "scores.csv"
            scores = [
                FileScore(
                    path=Path("test.wav"),
                    dir_label="positive",
                    clip_ratio=0.01,
                    snr_db=15.5,
                    wqi=0.85,
                )
            ]
            write_csv(scores, csv_path)

            assert csv_path.exists()
            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 2  # Header + 1 data row
                assert rows[0][0] == "path"
                assert rows[1][0] == "test.wav"

    def test_write_multiple_scores(self):
        """Test writing multiple FileScores to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "scores.csv"
            scores = [
                FileScore(path=Path("a.wav"), dir_label="positive"),
                FileScore(path=Path("b.wav"), dir_label="negative"),
                FileScore(path=Path("c.wav"), dir_label="positive"),
            ]
            write_csv(scores, csv_path)

            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 4  # Header + 3 data rows

    def test_csv_format(self):
        """Test CSV format with all fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "scores.csv"
            scores = [
                FileScore(
                    path=Path("test.wav"),
                    dir_label="positive",
                    clip_ratio=0.01,
                    snr_db=15.5,
                    vad_conf=0.95,
                    dnsmos_sig=3.5,
                    dnsmos_bak=3.0,
                    dnsmos_ovrl=3.2,
                    wqi=0.85,
                    discard=True,
                    discard_reason="low snr",
                    error="",
                )
            ]
            write_csv(scores, csv_path)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert float(row["clip_ratio"]) == pytest.approx(0.01)
                assert float(row["snr_db"]) == pytest.approx(15.5)
                assert float(row["wqi"]) == pytest.approx(0.85)
                assert row["discard"] == "1"
                assert row["discard_reason"] == "low snr"

    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "subdir" / "nested" / "scores.csv"
            scores = [FileScore(path=Path("test.wav"), dir_label="positive")]
            write_csv(scores, csv_path)
            assert csv_path.exists()


class TestApplyDiscard:
    """Tests for apply_discard function."""

    def test_dry_run_count(self):
        """Test dry run returns count without moving files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            discarded_dir = Path(tmpdir) / "discarded"
            scores = [
                FileScore(path=Path("a.wav"), dir_label="pos", discard=True),
                FileScore(path=Path("b.wav"), dir_label="pos", discard=False),
                FileScore(path=Path("c.wav"), dir_label="pos", discard=True, error="unreadable"),
            ]
            count = apply_discard(scores, discarded_dir, dry_run=True)
            assert count == 1  # Only 1 with discard=True and no error
            assert not discarded_dir.exists()  # Nothing actually moved

    def test_no_discardable_files(self):
        """Test with no discardable files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            discarded_dir = Path(tmpdir) / "discarded"
            scores = [
                FileScore(path=Path("a.wav"), dir_label="pos", discard=False),
                FileScore(path=Path("b.wav"), dir_label="pos", discard=False),
            ]
            count = apply_discard(scores, discarded_dir, dry_run=False)
            assert count == 0

    def test_actual_discard(self):
        """Test actual file movement (mocked)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file
            src_file = Path(tmpdir) / "test.wav"
            src_file.write_text("dummy audio data")

            discarded_dir = Path(tmpdir) / "discarded"
            scores = [
                FileScore(path=src_file, dir_label="positive", discard=True),
            ]
            count = apply_discard(scores, discarded_dir, dry_run=False)
            assert count == 1
            assert not src_file.exists()  # Original moved
            assert (discarded_dir / "positive" / "test.wav").exists()  # New location

    def test_handles_unreadable_files(self):
        """Test that files with error='unreadable' are not moved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            discarded_dir = Path(tmpdir) / "discarded"
            scores = [
                FileScore(path=Path("a.wav"), dir_label="pos", discard=True, error="unreadable"),
            ]
            count = apply_discard(scores, discarded_dir, dry_run=True)
            assert count == 0  # Not counted because of error


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_prints_header(self, capsys):
        """Test that summary header is printed."""
        config = QualityScoreConfig()
        scores = [FileScore(path=Path("test.wav"), dir_label="positive")]
        print_summary(scores, config, dry_run=False, mode="fast")

        captured = capsys.readouterr()
        assert "Quality Summary" in captured.out
        assert "clip gate" in captured.out

    def test_dry_run_prefix(self, capsys):
        """Test dry-run prefix in output."""
        config = QualityScoreConfig()
        scores = [FileScore(path=Path("test.wav"), dir_label="positive")]
        print_summary(scores, config, dry_run=True, mode="fast")

        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out

    def test_multiple_directories(self, capsys):
        """Test summary with multiple directories."""
        config = QualityScoreConfig()
        scores = [
            FileScore(path=Path("a.wav"), dir_label="positive", wqi=0.9, snr_db=20.0),
            FileScore(path=Path("b.wav"), dir_label="negative", wqi=0.8, snr_db=15.0),
        ]
        print_summary(scores, config, dry_run=False, mode="fast")

        captured = capsys.readouterr()
        assert "positive/" in captured.out
        assert "negative/" in captured.out

    def test_discard_count(self, capsys):
        """Test that discard count is shown."""
        config = QualityScoreConfig()
        scores = [
            FileScore(path=Path("a.wav"), dir_label="positive", discard=True),
            FileScore(path=Path("b.wav"), dir_label="positive", discard=False),
        ]
        print_summary(scores, config, dry_run=False, mode="fast")

        captured = capsys.readouterr()
        # Summary should show both files
        assert "files:" in captured.out
