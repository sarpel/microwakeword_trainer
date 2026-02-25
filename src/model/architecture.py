"""Model architecture module for wake word detection."""

import tensorflow as tf


def build_model(input_shape: tuple, num_classes: int = 2) -> tf.keras.Model:
    """Build wake word detection model.

    Args:
        input_shape: Input feature shape
        num_classes: Number of output classes

    Returns:
        Keras model
    """
    pass


class WakeWordModel(tf.keras.Model):
    """Wake word detection model."""

    def __init__(self, config: dict):
        """Initialize model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

    def call(self, inputs, training=None):
        """Forward pass."""
        pass
