"""Export module for model conversion."""

import tensorflow as tf


def convert_to_tflite(
    model: tf.keras.Model, output_path: str, quantize: bool = True
) -> bytes:
    """Convert model to TFLite format.

    Args:
        model: Keras model
        output_path: Output file path
        quantize: Whether to apply quantization

    Returns:
        TFLite model bytes
    """
    pass


def optimize_for_edge(model: tf.keras.Model) -> tf.keras.Model:
    """Optimize model for edge deployment.

    Args:
        model: Keras model

    Returns:
        Optimized model
    """
    pass
