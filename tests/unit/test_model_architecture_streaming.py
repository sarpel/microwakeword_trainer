"""Unit tests for model architecture and streaming helpers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import tensorflow as tf

from src.model import architecture as arch
from src.model import streaming as sm


def test_parse_model_param_and_helpers() -> None:
    assert arch.parse_model_param("") == []
    assert arch.parse_model_param("(1,2)") == [1, 2]
    assert arch.parse_model_param("[1,2]") == [1, 2]
    assert arch.parse_model_param("[[5],[9]]") == [[5], [9]]
    with pytest.raises(ValueError):
        arch.parse_model_param("not-valid")

    class Flags:
        first_conv_filters = 32
        first_conv_kernel_size = 5
        repeat_in_block = [1, 1]
        mixconv_kernel_sizes = [[5], [7]]
        stride = 3

    assert arch.spectrogram_slices_dropped(Flags()) > 0
    assert arch.spectrogram_slices_dropped({"first_conv_filters": 32, "first_conv_kernel_size": 5, "repeat_in_block": "1,1", "mixconv_kernel_sizes": "[5],[7]", "stride": 3}) > 0
    assert arch._split_channels(10, 3) == [4, 3, 3]


def test_mixconv_and_residual_block_forward() -> None:
    x = tf.ones((1, 8, 1, 8), dtype=tf.float32)
    block = arch.MixConvBlock(kernel_sizes=[5], filters=8, mode=sm.Modes.NON_STREAM_INFERENCE)
    y = block(x, training=False)
    assert y.shape[-1] == 8

    multi = arch.MixConvBlock(kernel_sizes=[3, 5], filters=8, mode=sm.Modes.NON_STREAM_INFERENCE)
    y2 = multi(x, training=False)
    assert y2.shape[-1] == 8

    rb = arch.ResidualBlock(filters=8, kernel_sizes=[5], repeat=1, use_residual=True, mode=sm.Modes.NON_STREAM_INFERENCE)
    y3 = rb(x, training=False)
    assert y3.shape[-1] == 8


def test_mixednet_build_and_factory_functions() -> None:
    model = arch.MixedNet(
        input_shape=(12, 40),
        first_conv_filters=8,
        first_conv_kernel_size=3,
        stride=3,
        pointwise_filters=[8, 8],
        mixconv_kernel_sizes=[[3], [5]],
        repeat_in_block=[1, 1],
        residual_connections=[0, 1],
        mode=sm.Modes.NON_STREAM_INFERENCE,
    )
    out = model(tf.zeros((1, 12, 40), dtype=tf.float32), training=False)
    assert tuple(out.shape) == (1, 1)
    cfg = model.get_config()
    assert cfg["stride"] == 3

    built = arch.build_model(input_shape=(12, 40), pointwise_filters="8,8", mixconv_kernel_sizes="[3],[5]", repeat_in_block="1,1", residual_connection="0,1", mode="unknown-mode")
    out2 = built(tf.zeros((1, 12, 40), dtype=tf.float32), training=False)
    assert tuple(out2.shape) == (1, 1)

    ok = arch.create_okay_nabu_model(input_shape=(12, 40), mode=sm.Modes.NON_STREAM_INFERENCE)
    out3 = ok(tf.zeros((1, 12, 40), dtype=tf.float32), training=False)
    assert tuple(out3.shape) == (1, 1)


def test_streaming_layers_and_helpers() -> None:
    rb = sm.RingBuffer(size=2)
    init = rb.initialize(batch_size=1, feature_dims=(2, 1, 4))
    assert tuple(init.shape) == (1, 2, 1, 4)
    u = rb.update(tf.ones((1, 1, 1, 4), dtype=tf.float32))
    assert tuple(u.shape) == (1, 2, 1, 4)
    with pytest.raises(ValueError):
        rb.update(tf.ones((1, 2, 1, 4), dtype=tf.float32))

    s = sm.Stream(cell=tf.keras.layers.Conv2D(4, (3, 1), padding="valid", use_bias=False), mode=sm.Modes.NON_STREAM_INFERENCE, pad_time_dim="causal", use_one_step=False)
    y = s(tf.zeros((1, 5, 1, 4), dtype=tf.float32), training=False)
    assert y.shape[-1] == 4

    s_same = sm.Stream(cell=tf.keras.layers.Conv2D(4, (3, 1), padding="valid", use_bias=False), mode=sm.Modes.TRAINING, pad_time_dim="same", use_one_step=False)
    _ = s_same(tf.zeros((1, 5, 1, 4), dtype=tf.float32), training=True)

    s_bad = sm.Stream(
        cell=tf.keras.layers.Identity(),
        mode=sm.Modes.NON_STREAM_INFERENCE,
        pad_time_dim="same",
        ring_buffer_size_in_time_dim=2,
        use_one_step=False,
    )
    with pytest.raises(ValueError):
        _ = s_bad(tf.zeros((1, 5, 1, 4), dtype=tf.float32), training=False)

    d = sm.StridedDrop(2, mode=sm.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    k = sm.StridedKeep(2, mode=sm.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    x = tf.zeros((1, 5, 1, 4), dtype=tf.float32)
    assert d(x).shape[1] == 3
    assert k(x).shape[1] == 2

    # Startup streaming context: keep should left-pad causally when input
    # has fewer frames than requested.
    k_pad = sm.StridedKeep(4, mode=sm.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    x_short = tf.ones((1, 1, 1, 2), dtype=tf.float32)
    kept = k_pad(x_short)
    assert kept.shape[1] == 4
    np.testing.assert_allclose(kept.numpy()[:, :3, :, :], 0.0)
    np.testing.assert_allclose(kept.numpy()[:, -1:, :, :], 1.0)

    # MixConv streaming path should remain one-step even with mixed kernels
    # when only a short startup context is available.
    mix_stream = arch.MixConvBlock(kernel_sizes=[3, 5], filters=8, mode=sm.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    y_stream = mix_stream(tf.ones((1, 1, 1, 8), dtype=tf.float32), training=False)
    assert y_stream.shape[1] == 1
    assert y_stream.shape[-1] == 8

    split = sm.ChannelSplit([2, 2], axis=-1)
    parts = split(tf.zeros((1, 3, 1, 4), dtype=tf.float32))
    assert len(parts) == 2

    with pytest.raises(ValueError):
        _ = sm.frequency_pad(tf.zeros((1, 2), dtype=tf.float32), dilation=1, stride=1, kernel_size=3)
    padded = sm.frequency_pad(tf.zeros((1, 3, 10, 1), dtype=tf.float32), dilation=1, stride=1, kernel_size=3)
    assert int(tf.shape(padded)[2]) >= 10

    assert sm.get_streaming_state_names() == ["stream", "stream_1", "stream_2", "stream_3", "stream_4", "stream_5"]
    init_fn = sm.create_state_initializer((1, 2, 1, 4))
    assert tuple(init_fn().shape) == (1, 2, 1, 4)


def test_streaming_mixednet_wrapper_predict_clip(monkeypatch) -> None:
    class DummyModel:
        def __call__(self, features, training=False):
            batch = features.shape[0]
            return tf.ones((batch, 1), dtype=tf.float32) * 0.5

    wrapper = sm.StreamingMixedNet(model=DummyModel(), stride=3)
    assert wrapper.predict(tf.zeros((1, 3, 40), dtype=tf.float32)).shape == (1, 1)
    wrapper.reset()  # state warnings are acceptable

    fake_mod = types.ModuleType("src.data.features")

    class FakeFeatureConfig:
        def __init__(self, sample_rate=16000, window_step_ms=10):
            self.sample_rate = sample_rate
            self.window_step_ms = window_step_ms

    class FakeMicroFrontend:
        def __init__(self, config):
            self.config = config

        def compute_mel_spectrogram(self, audio):
            # Return a minimal valid spectrogram: [num_frames, mel_bins]
            return np.zeros((3, 40), dtype=np.float32)

    fake_mod.__dict__["FeatureConfig"] = FakeFeatureConfig
    fake_mod.__dict__["MicroFrontend"] = FakeMicroFrontend
    sys.modules["src.data.features"] = fake_mod

    probs = wrapper.predict_clip(np.zeros((16000,), dtype=np.float32), sample_rate=16000, step_ms=30)
    assert probs
    assert all(0.0 <= p <= 1.0 for p in probs)

    probs_empty = wrapper.predict_clip(np.array([], dtype=np.float32), sample_rate=16000, step_ms=30)
    assert probs_empty == []
