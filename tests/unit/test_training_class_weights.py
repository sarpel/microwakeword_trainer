"""Unit tests for training class-weight application logic."""

from typing import Any, cast

import numpy as np
import pytest
import tensorflow as tf

from src.training.trainer import Trainer


def test_apply_class_weights_accepts_tensor_inputs() -> None:
    """Regression: tf.Tensor batch inputs should not raise End-of-sequence errors."""
    trainer = object.__new__(Trainer)
    y_true = tf.constant([1, 0, 0], dtype=tf.int32)
    sample_weights = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
    is_hard_negative = tf.constant([False, True, False], dtype=tf.bool)

    combined = trainer._apply_class_weights(
        y_true=cast(Any, y_true),
        sample_weights=cast(Any, sample_weights),
        positive_weight=1.0,
        negative_weight=20.0,
        hard_negative_weight=40.0,
        is_hard_negative=cast(Any, is_hard_negative),
    )

    combined_tensor = cast(Any, combined)
    np.testing.assert_allclose(combined_tensor.numpy(), np.array([1.0, 40.0, 20.0], dtype=np.float32))


def test_apply_class_weights_rejects_mismatched_lengths() -> None:
    """Class-weight path should fail fast when label/weight lengths diverge."""
    trainer = object.__new__(Trainer)
    y_true = tf.constant([1, 0, 1], dtype=tf.int32)
    sample_weights = tf.constant([1.0, 1.0], dtype=tf.float32)

    with pytest.raises(
        tf.errors.InvalidArgumentError,
        match="sample_weights size must match labels size",
    ):
        trainer._apply_class_weights(
            y_true=cast(Any, y_true),
            sample_weights=cast(Any, sample_weights),
            positive_weight=1.0,
            negative_weight=20.0,
            hard_negative_weight=40.0,
            is_hard_negative=None,
        )


def test_apply_class_weights_accepts_tfdata_iterator_outputs() -> None:
    """Regression: batches coming directly from tf.data iteration are supported."""
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant([1, 0, 0], dtype=tf.int32),
            tf.constant([1.0, 1.0, 1.0], dtype=tf.float32),
            tf.constant([False, True, False], dtype=tf.bool),
        )
    ).batch(3)

    trainer = object.__new__(Trainer)
    y_true_batch, sample_weights_batch, is_hard_negative_batch = next(iter(ds))
    combined = trainer._apply_class_weights(
        y_true=cast(Any, y_true_batch),
        sample_weights=cast(Any, sample_weights_batch),
        positive_weight=1.0,
        negative_weight=20.0,
        hard_negative_weight=40.0,
        is_hard_negative=cast(Any, is_hard_negative_batch),
    )

    combined_tensor = cast(Any, combined)
    np.testing.assert_allclose(combined_tensor.numpy(), np.array([1.0, 40.0, 20.0], dtype=np.float32))
