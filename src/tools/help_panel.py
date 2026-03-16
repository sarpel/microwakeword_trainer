"""mww-help — show post-training command reference panel."""

import argparse
from pathlib import Path


def _autodetect_checkpoint(
    checkpoint_dir: str = "./models/checkpoints",
) -> str:
    """Return the best available checkpoint path, or a placeholder string."""
    base = Path(checkpoint_dir)
    for name in ("best_weights.weights.h5", "final_weights.weights.h5"):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    # Try any .weights.h5 file
    candidates = sorted(base.glob("*.weights.h5"))
    if candidates:
        return str(candidates[-1])
    return "models/checkpoints/best_weights.weights.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the post-training command reference panel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with no arguments to auto-detect checkpoint and use the 'max_quality' config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to substitute into commands (default: auto-detect from models/checkpoints/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="max_quality",
        help="Config preset name or path to substitute into commands (default: max_quality)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./models/checkpoints",
        help="Directory to search for checkpoints when --checkpoint is not given (default: ./models/checkpoints)",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or _autodetect_checkpoint(args.checkpoint_dir)

    from src.training.rich_logger import RichTrainingLogger

    logger = RichTrainingLogger()
    logger.log_next_steps(checkpoint, args.config)


if __name__ == "__main__":
    main()
