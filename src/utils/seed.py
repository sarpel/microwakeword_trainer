"""Centralized RNG seeding for reproducibility."""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set random seeds across all frameworks for reproducibility.

    Seeds Python random, NumPy, TensorFlow, and optionally CuPy.
    Also enables TF deterministic ops.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    import tensorflow as tf

    tf.random.set_seed(seed)

    # Seed CuPy if available
    try:
        import cupy

        cupy.random.seed(seed)
    except ImportError:
        pass

    logger.info("Seeded all RNGs with seed=%d", seed)
