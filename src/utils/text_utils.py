"""Text utility functions for the microwakeword trainer."""

import re


def to_snake_case(name: str) -> str:
    """Convert a display name to snake_case for filenames.

    Examples:
        "Hey Katya" -> "hey_katya"
        "Okay Nabu" -> "okay_nabu"
        "Hey Jarvis" -> "hey_jarvis"

    Args:
        name: Display name (e.g., "Hey Katya")

    Returns:
        Snake case filename (e.g., "hey_katya")
    """
    # Convert to lowercase and replace non-alphanumeric with underscores
    snake = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    # Remove leading/trailing underscores
    snake = snake.strip("_")
    # Collapse multiple underscores
    snake = re.sub(r"_+", "_", snake)
    return snake
