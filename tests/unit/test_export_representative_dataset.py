"""Unit tests for representative dataset generators used in TFLite quantization."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.export.tflite import (
    create_representative_dataset,
    create_representative_dataset_from_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {"mel_bins": 40, "stride": 3}


class FakeFeatureStore:
    """Minimal FeatureStore mock that returns canned spectrograms with labels."""

    def __init__(self, specs_and_labels: list[tuple[np.ndarray, int]]):
        self._data = specs_and_labels

    def open(self, **_kwargs):
        pass

    def close(self):
        pass

    def get(self, idx: int):
        spec, label = self._data[idx]
        return spec, label

    def __len__(self):
        return len(self._data)


def _make_spectrogram(n_frames: int, mel_bins: int = 40, value: float = 10.0) -> np.ndarray:
    """Create a simple spectrogram [n_frames, mel_bins] filled with `value`."""
    return np.full((n_frames, mel_bins), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# create_representative_dataset (random fallback)
# ---------------------------------------------------------------------------


class TestCreateRepresentativeDataset:
    def test_yields_boundary_anchors_first(self):
        gen_fn = create_representative_dataset(DEFAULT_CONFIG, num_samples=5)
        samples = list(gen_fn())
        # First two are boundary anchors
        assert np.allclose(samples[0][0], 0.0)  # min anchor
        assert np.allclose(samples[1][0], 26.0)  # max anchor

    def test_correct_sample_count(self):
        gen_fn = create_representative_dataset(DEFAULT_CONFIG, num_samples=10)
        samples = list(gen_fn())
        # 2 anchors + 10 samples
        assert len(samples) == 12

    def test_sample_shape(self):
        gen_fn = create_representative_dataset(DEFAULT_CONFIG, num_samples=3)
        samples = list(gen_fn())
        for s in samples:
            assert len(s) == 1  # list of one array
            assert s[0].shape == (1, 3, 40)
            assert s[0].dtype == np.float32

    def test_reproducible(self):
        gen_fn = create_representative_dataset(DEFAULT_CONFIG, num_samples=5)
        samples_a = [s[0].copy() for s in gen_fn()]
        samples_b = [s[0].copy() for s in gen_fn()]
        for a, b in zip(samples_a, samples_b, strict=False):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# create_representative_dataset_from_data (sequential chunks)
# ---------------------------------------------------------------------------


class TestCreateRepresentativeDatasetFromData:
    def _make_store_data(
        self,
        n_positive: int = 5,
        n_negative: int = 15,
        frames_per_spec: int = 30,
        mel_bins: int = 40,
    ) -> list[tuple[np.ndarray, int]]:
        """Create a list of (spectrogram, label) pairs."""
        data = []
        for _i in range(n_positive):
            # Positive samples with higher values (simulates wake word features)
            data.append((_make_spectrogram(frames_per_spec, mel_bins, value=20.0), 1))
        for _i in range(n_negative):
            # Negative samples with lower values
            data.append((_make_spectrogram(frames_per_spec, mel_bins, value=5.0), 0))
        return data

    @patch("src.data.dataset.FeatureStore")
    def test_yields_boundary_anchors_first(self, mock_fs_cls, tmp_path):
        store_data = self._make_store_data(n_positive=2, n_negative=5)
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=50)
        samples = list(gen_fn())

        assert len(samples) >= 3  # At least anchors + 1 chunk
        assert np.allclose(samples[0][0], 0.0)  # min anchor
        assert np.allclose(samples[1][0], 26.0)  # max anchor

    @patch("src.data.dataset.FeatureStore")
    def test_chunks_are_sequential_within_spectrogram(self, mock_fs_cls, tmp_path):
        """Verify chunks from a single spectrogram are yielded in temporal order."""
        mel_bins = 40
        stride = 3
        n_frames = 12  # 4 chunks of stride 3

        # Create a spectrogram where each frame has a distinct value
        spec = np.arange(n_frames * mel_bins, dtype=np.float32).reshape(n_frames, mel_bins)
        store_data = [(spec, 1)]  # Single positive sample
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(
            {"mel_bins": mel_bins, "stride": stride},
            str(tmp_path),
            num_samples=100,
        )
        samples = list(gen_fn())

        # Skip 2 anchors, remaining should be sequential chunks
        data_samples = samples[2:]
        assert len(data_samples) == 4  # 12 frames / stride 3 = 4 chunks

        for i, s in enumerate(data_samples):
            expected_start = i * stride
            expected = spec[expected_start : expected_start + stride].reshape(1, stride, mel_bins)
            np.testing.assert_array_almost_equal(s[0], expected)

    @patch("src.data.dataset.FeatureStore")
    def test_includes_positive_and_negative(self, mock_fs_cls, tmp_path):
        """Both positive and negative spectrograms contribute chunks."""
        store_data = self._make_store_data(n_positive=3, n_negative=10, frames_per_spec=12)
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=200)
        samples = list(gen_fn())

        data_samples = samples[2:]  # Skip anchors
        # With 13 specs * 4 chunks each = 52 chunks
        assert len(data_samples) > 0

        # Verify we see both positive (value=20.0) and negative (value=5.0) chunks
        values_seen = set()
        for s in data_samples:
            # Check the dominant value of the chunk
            mean_val = s[0].mean()
            if mean_val > 15.0:
                values_seen.add("positive")
            elif mean_val < 10.0:
                values_seen.add("negative")

        assert "positive" in values_seen, "No positive sample chunks found"
        assert "negative" in values_seen, "No negative sample chunks found"

    @patch("src.data.dataset.FeatureStore")
    def test_correct_chunk_shape(self, mock_fs_cls, tmp_path):
        store_data = self._make_store_data(n_positive=1, n_negative=3, frames_per_spec=9)
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=50)
        for s in gen_fn():
            assert len(s) == 1
            assert s[0].shape == (1, 3, 40)
            assert s[0].dtype == np.float32

    @patch("src.data.dataset.FeatureStore")
    def test_handles_flat_spectrogram(self, mock_fs_cls, tmp_path):
        """Handles spectrograms stored as flat 1D arrays."""
        mel_bins = 40
        n_frames = 9
        flat_spec = np.full(n_frames * mel_bins, 15.0, dtype=np.float32)
        store_data = [(flat_spec, 0)]
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=50)
        samples = list(gen_fn())

        data_samples = samples[2:]
        assert len(data_samples) == 3  # 9 frames / stride 3 = 3 chunks

    def test_fallback_when_no_data_dir(self, tmp_path):
        """Falls back to random dataset when train/ doesn't exist."""
        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=10)
        samples = list(gen_fn())
        # Should be random fallback: 2 anchors + 10 samples
        assert len(samples) == 12

    @patch("src.data.dataset.FeatureStore")
    def test_handles_all_negative_gracefully(self, mock_fs_cls, tmp_path):
        """Works even with zero positive samples (still yields sequential chunks)."""
        store_data = self._make_store_data(n_positive=0, n_negative=5, frames_per_spec=9)
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=50)
        samples = list(gen_fn())

        data_samples = samples[2:]
        # 5 specs * 3 chunks = 15 chunks
        assert len(data_samples) == 15

    @patch("src.data.dataset.FeatureStore")
    def test_skips_too_short_spectrograms(self, mock_fs_cls, tmp_path):
        """Spectrograms shorter than stride are skipped."""
        mel_bins = 40
        store_data = [
            (
                _make_spectrogram(2, mel_bins, 10.0),
                0,
            ),  # Too short (2 < stride 3)
            (_make_spectrogram(6, mel_bins, 10.0), 0),  # OK: 2 chunks
        ]
        fake_store = FakeFeatureStore(store_data)
        mock_fs_cls.return_value = fake_store

        train_dir = tmp_path / "train"
        train_dir.mkdir()

        gen_fn = create_representative_dataset_from_data(DEFAULT_CONFIG, str(tmp_path), num_samples=50)
        samples = list(gen_fn())

        data_samples = samples[2:]
        assert len(data_samples) == 2  # Only from the 6-frame spec
