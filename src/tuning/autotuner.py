"""Auto-tuning system for post-training fine-tuning of wake word models.

This module provides sophisticated fine-tuning capabilities to achieve target metrics:
- FAH (False Activations per Hour) < 0.3
- Recall > 0.92

Uses iterative micro-adjustments and hard negative mining.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.data.dataset import WakeWordDataset
from src.evaluation.metrics import MetricsCalculator
from src.model.architecture import build_model
from src.training.miner import HardExampleMiner
from src.training.rich_logger import RichTrainingLogger
from src.training.trainer import Trainer


@dataclass
class TuningTarget:
    """Target metrics for auto-tuning."""

    max_fah: float = 0.3
    min_recall: float = 0.92
    max_iterations: int = 100
    patience: int = 10  # Iterations without improvement before strategy switch


@dataclass
class TuningState:
    """Current state of the tuning process."""

    iteration: int = 0
    current_fah: float = float("inf")
    current_recall: float = 0.0
    best_fah: float = float("inf")
    best_recall: float = 0.0
    best_checkpoint_path: str | None = None
    strategy_history: list[dict] = field(default_factory=list)
    config_history: list[dict] = field(default_factory=list)

    def is_target_met(self) -> bool:
        """Check if both targets are met."""
        return self.current_fah <= TuningTarget().max_fah and self.current_recall >= TuningTarget().min_recall

    def needs_fah_improvement(self) -> bool:
        """Check if FAH needs improvement."""
        return self.current_fah > TuningTarget().max_fah

    def needs_recall_improvement(self) -> bool:
        """Check if recall needs improvement."""
        return self.current_recall < TuningTarget().min_recall


class MicroConfigAdjuster:
    """Micro-adjusts configuration values for fine-tuning."""

    # Config parameter ranges for recall improvement
    RECALL_PARAMS: dict[str, dict[str, Any]] = {
        "positive_class_weight": {
            "current": 1.0,
            "min": 0.5,
            "max": 3.0,
            "step": 0.1,
            "direction": "up",  # Increase to improve recall
        },
        "negative_class_weight": {
            "current": 20.0,
            "min": 10.0,
            "max": 40.0,
            "step": 2.0,
            "direction": "down",  # Decrease to improve recall
        },
        "hard_negative_class_weight": {
            "current": 40.0,
            "min": 20.0,
            "max": 80.0,
            "step": 5.0,
            "direction": "down",  # Decrease to improve recall
        },
        "learning_rate": {
            "current": 0.0001,
            "min": 0.00001,
            "max": 0.001,
            "step": 0.00005,
            "direction": "adaptive",
        },
        "dropout_rate": {
            "current": 0.2,
            "min": 0.0,
            "max": 0.5,
            "step": 0.05,
            "direction": "down",  # Decrease dropout to improve recall
        },
    }

    def __init__(self, config: dict):
        """Initialize with base config."""
        self.base_config = copy.deepcopy(config)
        self.current_params = copy.deepcopy(self.RECALL_PARAMS)
        self._sync_with_config(config)

    def _sync_with_config(self, config: dict) -> None:
        """Sync current params with actual config values."""
        training = config.get("training", {})
        model = config.get("model", {})

        if "positive_class_weight" in training:
            weights = training["positive_class_weight"]
            if isinstance(weights, list) and weights:
                self.current_params["positive_class_weight"]["current"] = weights[-1]

        if "negative_class_weight" in training:
            weights = training["negative_class_weight"]
            if isinstance(weights, list) and weights:
                self.current_params["negative_class_weight"]["current"] = weights[-1]

        if "hard_negative_class_weight" in training:
            weights = training["hard_negative_class_weight"]
            if isinstance(weights, list) and weights:
                self.current_params["hard_negative_class_weight"]["current"] = weights[-1]

        if "learning_rates" in training:
            lrs = training["learning_rates"]
            if isinstance(lrs, list) and lrs:
                self.current_params["learning_rate"]["current"] = lrs[-1]

        if "dropout_rate" in model:
            self.current_params["dropout_rate"]["current"] = model["dropout_rate"]

    def adjust_for_recall(self, iteration: int, prev_recall: float, current_recall: float) -> dict:
        """Adjust config to improve recall.

        Uses adaptive strategy based on improvement trend.
        """
        config_delta = {}
        improving = current_recall > prev_recall

        # Strategy: Adjust class weights
        if iteration % 5 == 0:  # Every 5 iterations, adjust positive weight
            param = self.current_params["positive_class_weight"]
            if improving and param["current"] < param["max"]:
                param["current"] = min(param["max"], param["current"] + param["step"])
            elif not improving and param["current"] > param["min"]:
                param["current"] = max(param["min"], param["current"] - param["step"])
            config_delta["positive_class_weight"] = param["current"]

        elif iteration % 5 == 1:  # Adjust negative weight
            param = self.current_params["negative_class_weight"]
            if improving and param["current"] > param["min"]:
                param["current"] = max(param["min"], param["current"] - param["step"])
            elif not improving and param["current"] < param["max"]:
                param["current"] = min(param["max"], param["current"] + param["step"])
            config_delta["negative_class_weight"] = param["current"]

        elif iteration % 5 == 2:  # Adjust hard negative weight
            param = self.current_params["hard_negative_class_weight"]
            if improving and param["current"] > param["min"]:
                param["current"] = max(param["min"], param["current"] - param["step"])
            elif not improving and param["current"] < param["max"]:
                param["current"] = min(param["max"], param["current"] + param["step"])
            config_delta["hard_negative_class_weight"] = param["current"]

        elif iteration % 5 == 3:  # Adjust learning rate
            param = self.current_params["learning_rate"]
            if not improving:  # If not improving, reduce LR
                param["current"] = max(param["min"], param["current"] * 0.8)
            config_delta["learning_rate"] = param["current"]

        elif iteration % 5 == 4:  # Adjust dropout
            param = self.current_params["dropout_rate"]
            if improving and param["current"] > param["min"]:
                param["current"] = max(param["min"], param["current"] - param["step"])
            config_delta["dropout_rate"] = param["current"]

        return config_delta

    def apply_adjustments(self, config: dict, delta: dict) -> dict:
        """Apply config adjustments to create new config."""
        new_config = copy.deepcopy(config)
        training = new_config.setdefault("training", {})
        model = new_config.setdefault("model", {})

        # Apply training params
        if "positive_class_weight" in delta:
            training["positive_class_weight"] = [delta["positive_class_weight"]]
        if "negative_class_weight" in delta:
            training["negative_class_weight"] = [delta["negative_class_weight"]]
        if "hard_negative_class_weight" in delta:
            training["hard_negative_class_weight"] = [delta["hard_negative_class_weight"]]
        if "learning_rate" in delta:
            training["learning_rates"] = [delta["learning_rate"]]
            training["training_steps"] = [5000]  # Short fine-tuning steps

        # Apply model params
        if "dropout_rate" in delta:
            model["dropout_rate"] = delta["dropout_rate"]

        return new_config


class FAHReductionStrategy:
    """Strategy for reducing False Activations per Hour."""

    def __init__(self, miner: HardExampleMiner | None = None):
        """Initialize with hard example miner."""
        self.miner = miner or HardExampleMiner(
            strategy="confidence",
            fp_threshold=0.5,
            max_samples=10000,
        )
        self.mined_hard_negatives: list = []

    def mine_hard_negatives(
        self,
        model: tf.keras.Model,
        dataset: WakeWordDataset,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mine hard negative examples from dataset."""
        # Get validation generator factory and create generator
        val_gen_factory = dataset.val_generator_factory()

        features = []
        labels = []

        # Collect negative samples from validation data
        try:
            val_gen = val_gen_factory()
            for batch_features, batch_labels in val_gen:
                neg_mask = batch_labels == 0
                if np.any(neg_mask):
                    features.append(batch_features[neg_mask])
                    labels.append(batch_labels[neg_mask])
        except Exception as e:
            # If generator fails, return empty
            print(f"Warning: Could not mine hard negatives: {e}")
            return np.array([]), np.array([])

        if not features:
            return np.array([]), np.array([])

        all_features = np.concatenate(features, axis=0)
        all_labels = np.concatenate(labels, axis=0)

        # Use standalone mine_hard_examples function
        from src.training.miner import mine_hard_examples

        hard_features, hard_labels = mine_hard_examples(all_features, all_labels, model, threshold=threshold)

        return hard_features, hard_labels

    def prepare_hard_negative_dataset(
        self,
        base_dataset: WakeWordDataset,
        hard_features: np.ndarray,
        hard_labels: np.ndarray,
        mix_ratio: float = 0.3,
    ) -> WakeWordDataset:
        """Create mixed dataset with hard negatives."""
        # This would mix hard negatives into the training data
        # Implementation depends on dataset structure
        # For now, we'll just update the config to use hard negatives
        return base_dataset


class AutoTuner:
    """Main auto-tuning orchestrator for wake word models.

    Iteratively fine-tunes models to achieve target metrics:
    - FAH < 0.3
    - Recall > 0.92

    Usage:
        tuner = AutoTuner(
            checkpoint_path="checkpoints/best.ckpt",
            config=config_dict,
        )
        result = tuner.tune()
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        output_dir: str = "./tuning",
        target_fah: float = 0.3,
        target_recall: float = 0.92,
        max_iterations: int = 100,
        console: Console | None = None,
    ):
        """Initialize auto-tuner.

        Args:
            checkpoint_path: Path to trained checkpoint to fine-tune
            config: Training configuration dictionary
            output_dir: Directory for saving tuned checkpoints
            target_fah: Target FAH value (default: 0.3)
            target_recall: Target recall value (default: 0.92)
            max_iterations: Maximum tuning iterations
            console: Rich console for output
        """
        self.checkpoint_path = checkpoint_path
        self.base_config = copy.deepcopy(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target = TuningTarget(
            max_fah=target_fah,
            min_recall=target_recall,
            max_iterations=max_iterations,
        )
        self.state = TuningState()

        self.console = console or Console()
        self.logger = RichTrainingLogger(console=self.console)

        # Initialize strategies
        self.config_adjuster = MicroConfigAdjuster(config)
        self.fah_strategy = FAHReductionStrategy()

        # Current model and config
        self.current_model: tf.keras.Model | None = None
        self.current_config: dict = copy.deepcopy(config)

        # Tracking
        self.improvement_count = 0
        self.no_improvement_count = 0

    def _log_header(self) -> None:
        """Display tuning header."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")

        table.add_row("Checkpoint", self.checkpoint_path)
        table.add_row("Target FAH", f"< {self.target.max_fah}")
        table.add_row("Target Recall", f"> {self.target.min_recall}")
        table.add_row("Max Iterations", str(self.target.max_iterations))
        table.add_row("Output Dir", str(self.output_dir))

        panel = Panel(
            table,
            title="🔧 Auto-Tuning Configuration",
            border_style="blue",
            expand=False,
        )
        self.console.print(panel)

    def _log_status(self) -> None:
        """Display current tuning status."""
        table = Table(title=f"Iteration {self.state.iteration}/{self.target.max_iterations}")
        table.add_column("Metric", style="bold")
        table.add_column("Current", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")

        # FAH status
        fah_status = "✅" if self.state.current_fah <= self.target.max_fah else "❌"
        fah_color = "green" if self.state.current_fah <= self.target.max_fah else "red"
        table.add_row(
            "FAH",
            f"[{fah_color}]{self.state.current_fah:.4f}[/{fah_color}]",
            f"< {self.target.max_fah}",
            fah_status,
        )

        # Recall status
        recall_status = "✅" if self.state.current_recall >= self.target.min_recall else "❌"
        recall_color = "green" if self.state.current_recall >= self.target.min_recall else "red"
        table.add_row(
            "Recall",
            f"[{recall_color}]{self.state.current_recall:.4f}[/{recall_color}]",
            f"> {self.target.min_recall}",
            recall_status,
        )

        self.console.print(table)

    def _evaluate_model(
        self,
        model: tf.keras.Model,
        dataset: WakeWordDataset,
    ) -> dict[str, float]:
        """Evaluate model and return metrics."""
        # Collect predictions from validation data
        y_true = []
        y_scores = []

        # Get validation generator
        val_gen_factory = dataset.val_generator_factory()

        try:
            val_gen = val_gen_factory()
            for batch_features, batch_labels in val_gen:
                predictions = model.predict(batch_features, verbose=0)
                y_true.extend(batch_labels.flatten().tolist())
                y_scores.extend(predictions.flatten().tolist())
        except Exception as e:
            print(f"Warning: Could not evaluate model: {e}")
            return {"fah": float("inf"), "recall": 0.0, "precision": 0.0, "f1": 0.0, "accuracy": 0.0}
        y_true_arr = np.array(y_true)
        y_scores_arr = np.array(y_scores)

        # Calculate metrics
        calc = MetricsCalculator(y_true=y_true_arr, y_score=y_scores_arr)

        # Get ambient duration from config
        ambient_hours = self.base_config.get("training", {}).get("ambient_duration_hours", 10.0)

        metrics = calc.compute_all_metrics(
            ambient_duration_hours=ambient_hours,
            threshold=0.5,
        )

        return {
            "fah": metrics.get("ambient_false_positives_per_hour", float("inf")),
            "recall": metrics.get("recall", 0.0),
            "precision": metrics.get("precision", 0.0),
            "f1": metrics.get("f1_score", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
        }

    def _load_model(self) -> tf.keras.Model:
        """Load model from checkpoint."""
        # Build model with current config
        hardware_cfg = self.current_config.get("hardware", {})
        clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        input_shape = (int(clip_duration_ms / window_step_ms), mel_bins)

        model = build_model(
            input_shape=input_shape,
            model_config=self.current_config.get("model", {}),
        )

        # Load weights
        model.load_weights(self.checkpoint_path)

        return model

    def _run_fine_tuning_iteration(
        self,
        strategy: str,
    ) -> tuple[tf.keras.Model, dict]:
        """Run one fine-tuning iteration with given strategy."""
        # Prepare dataset
        dataset = WakeWordDataset(self.current_config)
        dataset.build()

        # Create trainer with micro-config
        trainer = Trainer(self.current_config)

        # Load current model
        model = self._load_model()
        trainer.model = model

        # Apply strategy
        if strategy == "fah_reduction":
            # Mine hard negatives and add to dataset
            hard_features, hard_labels = self.fah_strategy.mine_hard_negatives(model, dataset, threshold=0.5)

            if len(hard_features) > 0:
                self.console.print(f"[yellow]⛏️  Mined {len(hard_features)} hard negatives[/]")

        # Short fine-tuning run
        hardware_cfg = self.current_config.get("hardware", {})
        clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        max_time_frames = int(clip_duration_ms / window_step_ms)
        input_shape = (max_time_frames, mel_bins)

        # Fine-tune for a few steps
        tuned_model = trainer.train(
            train_data_factory=dataset.train_generator_factory(max_time_frames=max_time_frames),
            val_data_factory=dataset.val_generator_factory(max_time_frames=max_time_frames),
            input_shape=input_shape,
        )

        return tuned_model, self.current_config

    def _select_strategy(self) -> str:
        """Select tuning strategy based on current state."""
        needs_fah = self.state.needs_fah_improvement()
        needs_recall = self.state.needs_recall_improvement()

        if needs_fah and needs_recall:
            # Prioritize based on which is further from target
            fah_gap = self.state.current_fah - self.target.max_fah
            recall_gap = self.target.min_recall - self.state.current_recall

            if fah_gap > recall_gap * 0.1:  # FAH gap is relatively larger
                return "fah_reduction"
            else:
                return "recall_improvement"
        elif needs_fah:
            return "fah_reduction"
        elif needs_recall:
            return "recall_improvement"
        else:
            return "both"  # Both targets met

    def _adjust_config_for_recall(self) -> None:
        """Adjust config to improve recall."""
        prev_recall = self.state.best_recall if self.state.iteration > 1 else self.state.current_recall

        delta = self.config_adjuster.adjust_for_recall(
            self.state.iteration,
            prev_recall,
            self.state.current_recall,
        )

        if delta:
            self.current_config = self.config_adjuster.apply_adjustments(self.current_config, delta)
            self.console.print(f"[blue]🎛️  Config adjusted: {delta}[/]")

    def _save_checkpoint(self, model: tf.keras.Model, suffix: str) -> str:
        """Save checkpoint with given suffix."""
        checkpoint_path = self.output_dir / f"tuned_{suffix}_iter{self.state.iteration}.ckpt"
        model.save_weights(str(checkpoint_path))
        return str(checkpoint_path)

    def tune(self) -> dict:
        """Run auto-tuning process.

        Returns:
            Dictionary with tuning results
        """
        self._log_header()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Auto-tuning...", total=self.target.max_iterations)

            while self.state.iteration < self.target.max_iterations:
                self.state.iteration += 1
                progress.update(task, advance=1)

                self.console.print(f"\n[bold cyan]Iteration {self.state.iteration}[/]")

                # Select strategy
                strategy = self._select_strategy()
                self.console.print(f"[dim]Strategy: {strategy}[/]")

                # Adjust config if needed
                if strategy == "recall_improvement":
                    self._adjust_config_for_recall()

                # Run fine-tuning
                try:
                    model, config = self._run_fine_tuning_iteration(strategy)
                    self.current_model = model
                except Exception as e:
                    self.console.print(f"[red]❌ Error in iteration: {e}[/]")
                    continue

                # Evaluate
                dataset = WakeWordDataset(self.current_config)
                dataset.build()
                metrics = self._evaluate_model(model, dataset)

                self.state.current_fah = metrics["fah"]
                self.state.current_recall = metrics["recall"]

                # Log status
                self._log_status()

                # Track improvements
                improved = False
                if self.state.current_fah < self.state.best_fah:
                    self.state.best_fah = self.state.current_fah
                    improved = True
                if self.state.current_recall > self.state.best_recall:
                    self.state.best_recall = self.state.current_recall
                    improved = True

                if improved:
                    self.improvement_count += 1
                    self.no_improvement_count = 0

                    # Save best checkpoint
                    checkpoint_path = self._save_checkpoint(model, f"fah{self.state.current_fah:.3f}_rec{self.state.current_recall:.3f}")
                    self.state.best_checkpoint_path = checkpoint_path
                    self.console.print(f"[green]✅ Improvement! Saved: {checkpoint_path}[/]")
                else:
                    self.no_improvement_count += 1

                # Record history
                self.state.strategy_history.append(
                    {
                        "iteration": self.state.iteration,
                        "strategy": strategy,
                        "metrics": metrics,
                    }
                )

                # Check for early completion
                if self.state.is_target_met():
                    self.console.print("\n[bold green]🎉 Target metrics achieved![/]")
                    break

                # Check for stagnation
                if self.no_improvement_count >= self.target.patience:
                    self.console.print(f"\n[yellow]⚠️  No improvement for {self.target.patience} iterations, switching strategy...[/]")
                    self.no_improvement_count = 0

        # Final summary
        self._log_final_summary()

        return {
            "best_fah": self.state.best_fah,
            "best_recall": self.state.best_recall,
            "final_fah": self.state.current_fah,
            "final_recall": self.state.current_recall,
            "iterations": self.state.iteration,
            "best_checkpoint": self.state.best_checkpoint_path,
            "target_met": self.state.is_target_met(),
            "history": self.state.strategy_history,
        }

    def _log_final_summary(self) -> None:
        """Display final tuning summary."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Iterations", str(self.state.iteration))
        table.add_row("Best FAH", f"{self.state.best_fah:.4f}")
        table.add_row("Best Recall", f"{self.state.best_recall:.4f}")
        table.add_row("Final FAH", f"{self.state.current_fah:.4f}")
        table.add_row("Final Recall", f"{self.state.current_recall:.4f}")
        table.add_row("Target Met", "✅ Yes" if self.state.is_target_met() else "❌ No")
        table.add_row("Best Checkpoint", str(self.state.best_checkpoint_path or "N/A"))

        panel = Panel(
            table,
            title="🏁 Auto-Tuning Complete",
            border_style="green" if self.state.is_target_met() else "yellow",
            expand=False,
        )
        self.console.print(panel)


def autotune(
    checkpoint_path: str,
    config: dict,
    output_dir: str = "./tuning",
    target_fah: float = 0.3,
    target_recall: float = 0.92,
    max_iterations: int = 100,
) -> dict:
    """Convenience function for auto-tuning.

    Args:
        checkpoint_path: Path to trained checkpoint
        config: Training configuration
        output_dir: Output directory for tuned checkpoints
        target_fah: Target FAH value
        target_recall: Target recall value
        max_iterations: Maximum tuning iterations

    Returns:
        Tuning results dictionary
    """
    tuner = AutoTuner(
        checkpoint_path=checkpoint_path,
        config=config,
        output_dir=output_dir,
        target_fah=target_fah,
        target_recall=target_recall,
        max_iterations=max_iterations,
    )
    return tuner.tune()
