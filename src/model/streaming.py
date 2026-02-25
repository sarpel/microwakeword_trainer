"""Streaming model module for real-time inference."""

import tensorflow as tf


class StreamingModel:
    """Streaming wake word model for real-time inference."""

    def __init__(self, model_path: str, window_size: int = 1600):
        """Initialize streaming model.

        Args:
            model_path: Path to saved model
            window_size: Audio window size
        """
        self.window_size = window_size
        self.model = None

    def process_chunk(self, audio_chunk: bytes) -> float:
        """Process audio chunk.

        Args:
            audio_chunk: Raw audio bytes

        Returns:
            Detection probability
        """
        pass

    def reset(self):
        """Reset internal state."""
        pass
