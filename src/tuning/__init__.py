"""Auto-tuning module for wake word models."""

from .autotuner import AutoTuner


def autotune(checkpoint_path: str, config: "FullConfig", output_dir: str, target_fah: float, target_recall: float, max_iterations: int):
    """Convenience function to run auto-tuning.

    Args:
        checkpoint_path: Path to the model checkpoint to tune.
        config: FullConfig object containing tuning parameters.
        output_dir: Directory for tuning outputs.
        target_fah: Target false alarms per hour.
        target_recall: Target recall score.
        max_iterations: Maximum tuning iterations.

    Returns:
        Auto-tuning results.
    """
    tuner = AutoTuner(checkpoint_path=checkpoint_path, config=config, output_dir=output_dir)
    return tuner.run(target_fah=target_fah, target_recall=target_recall, max_iterations=max_iterations)


__all__ = ["AutoTuner", "autotune"]
