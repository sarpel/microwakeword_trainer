"""Unit tests for async validation model building.

Regression test: eval model built via from_config() + forward pass must accept
variable batch sizes and correctly transfer weights from the training model.

MixedNet is a subclassed Keras model with no custom build() method, so
model.build() creates 0 weights. Weight creation requires a concrete forward
pass through the model.
"""

import numpy as np
import tensorflow as tf

from src.model.architecture import MixedNet


def _make_model(input_shape=(100, 40)) -> MixedNet:
    """Create a small MixedNet for testing."""
    return MixedNet(
        input_shape=input_shape,
        first_conv_filters=32,
        first_conv_kernel_size=5,
        stride=3,
        pointwise_filters=[64],
        mixconv_kernel_sizes=[[5]],
        repeat_in_block=[1],
        residual_connections=[0],
        dropout_rate=0.0,
    )


class TestAsyncValidationModelBuild:
    """Tests that eval model cloning for async validation handles dynamic batches."""

    def test_forward_pass_creates_weights_and_handles_variable_batches(
        self,
    ) -> None:
        """A concrete forward pass creates weights and still allows variable batch sizes.

        This is the approach used in _compute_metrics_background: from_config()
        + forward pass with batch=1 to create weights, then set_weights().
        The model must handle different batch sizes after this.
        """
        input_shape = (100, 40)
        model = _make_model(input_shape)
        # Forward pass creates weights on the original model
        _ = model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)

        # Simulate _compute_metrics_background: from_config + forward pass + set_weights
        eval_model = model.__class__.from_config(model.get_config())
        _ = eval_model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
        eval_model.set_weights(model.get_weights())

        # Must handle batch=1
        out1 = eval_model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
        assert out1.shape == (1, 1), f"Expected (1, 1), got {out1.shape}"

        # Must handle batch=4
        out4 = eval_model(tf.zeros((4, *input_shape), dtype=tf.float32), training=False)
        assert out4.shape == (4, 1), f"Expected (4, 1), got {out4.shape}"

        # Must handle batch=256 (typical validation batch)
        out256 = eval_model(tf.zeros((256, *input_shape), dtype=tf.float32), training=False)
        assert out256.shape == (
            256,
            1,
        ), f"Expected (256, 1), got {out256.shape}"

    def test_build_creates_no_weights(self) -> None:
        """model.build() on subclassed MixedNet creates 0 weights.

        MixedNet has no custom build() method — all layers are created in
        __init__(). A forward pass is required to trigger Keras lazy weight
        creation. This test documents why _compute_metrics_background uses
        a forward pass rather than build().
        """
        input_shape = (100, 40)
        model = _make_model(input_shape)
        model.build((None, *input_shape))

        expected_weights_message = (
            f"Expected 0 weights after build(), got {len(model.get_weights())}. If MixedNet now has a custom build() method, update _compute_metrics_background to use build() instead of forward pass."
        )
        assert len(model.get_weights()) == 0, expected_weights_message

    def test_weights_transfer_correctly_after_forward_pass(self) -> None:
        """Weights set via set_weights() after forward pass must produce same outputs."""
        input_shape = (100, 40)
        model = _make_model(input_shape)
        # Create weights via forward pass
        _ = model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)

        tf.random.set_seed(42)
        test_input = tf.random.uniform((2, *input_shape), dtype=tf.float32)
        original_output = model(test_input, training=False)

        # Clone via from_config + forward pass + set_weights
        eval_model = model.__class__.from_config(model.get_config())
        _ = eval_model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
        eval_model.set_weights(model.get_weights())

        cloned_output = eval_model(test_input, training=False)
        np.testing.assert_allclose(
            original_output.numpy(),
            cloned_output.numpy(),
            atol=1e-5,
            err_msg="Cloned model outputs must match original after weight transfer",
        )
