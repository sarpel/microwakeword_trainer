from __future__ import annotations

import copy
import enum
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.dataset import WakeWordDataset
from src.evaluation.metrics import MetricsCalculator
from src.model.architecture import build_model
from src.training.miner import HardExampleMiner
from src.training.trainer import Trainer


@dataclass
class ParetoPoint:
    fah: float
    recall: float
    checkpoint_path: str
    config_snapshot: dict = field(default_factory=dict)
    iteration: int = 0


class ParetoFrontier:
    """Track non-dominated (FAH, recall) solutions."""

    def __init__(self):
        self.points: list[ParetoPoint] = []

    def add(self, fah, recall, checkpoint_path, config_snapshot, iteration) -> bool:
        """Add point. Returns True if it's Pareto-improving (not dominated by any existing)."""
        if self.is_dominated(fah, recall):
            return False

        new_point = ParetoPoint(
            fah=fah,
            recall=recall,
            checkpoint_path=checkpoint_path,
            config_snapshot=config_snapshot,
            iteration=iteration,
        )

        filtered = []
        for point in self.points:
            if fah <= point.fah and recall >= point.recall:
                continue
            filtered.append(point)

        filtered.append(new_point)
        self.points = filtered
        return True

    def is_dominated(self, fah, recall) -> bool:
        """Check if (fah, recall) is dominated by any frontier point."""
        for point in self.points:
            if point.fah <= fah and point.recall >= recall:
                return True
        return False

    def best_by_target(self, target_fah, target_recall) -> ParetoPoint | None:
        """Find point closest to meeting both targets. Use weighted distance."""
        if not self.points:
            return None

        best_point = None
        best_score = float("inf")
        for point in self.points:
            fah_gap = max(0.0, point.fah - target_fah)
            recall_gap = max(0.0, target_recall - point.recall)
            score = (fah_gap * 2.0) + recall_gap
            if score < best_score:
                best_score = score
                best_point = point
        return best_point

    @property
    def frontier(self) -> list[ParetoPoint]:
        """Return frontier sorted by FAH ascending."""
        return sorted(self.points, key=lambda p: p.fah)


@dataclass
class KnobRecord:
    param_name: str
    delta_value: float
    fah_before: float
    fah_after: float
    recall_before: float
    recall_after: float
    iteration: int


class ImpactMemory:
    """Track parameter adjustment effectiveness."""

    def __init__(self):
        self.records: list[KnobRecord] = []
        self._last_tried: dict[str, int] = {}

    def record(self, param_name, delta_value, fah_before, fah_after, recall_before, recall_after, iteration):
        """Record an adjustment and its effect."""
        self.records.append(
            KnobRecord(
                param_name=param_name,
                delta_value=delta_value,
                fah_before=fah_before,
                fah_after=fah_after,
                recall_before=recall_before,
                recall_after=recall_after,
                iteration=iteration,
            )
        )
        self._last_tried[param_name] = iteration

    def get_impact_scores(self) -> dict[str, float]:
        """Return impact score per knob. Higher = more effective.
        Score = mean(fah_improvement_normalized + recall_improvement_normalized) for each knob.
        Unexplored knobs get 0.5 (encourage exploration)."""
        if not self.records:
            return {}

        by_param: dict[str, list[KnobRecord]] = {}
        for record in self.records:
            by_param.setdefault(record.param_name, []).append(record)

        scores: dict[str, float] = {}
        for param_name, entries in by_param.items():
            effects = []
            for entry in entries:
                fah_improvement = entry.fah_before - entry.fah_after
                recall_improvement = entry.recall_after - entry.recall_before
                fah_norm = np.tanh(fah_improvement)
                recall_norm = np.tanh(recall_improvement)
                effects.append(fah_norm + recall_norm)
            scores[param_name] = float(np.mean(effects)) if effects else 0.0

        return scores

    def best_knob(self) -> str | None:
        """Highest impact parameter name."""
        scores = self.get_impact_scores()
        if not scores:
            return None
        best_name = None
        best_score = float("-inf")
        for name, score in scores.items():
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def last_was_harmful(self, param_name: str) -> bool:
        """True if last adjustment to this param made things worse (both FAH AND recall)."""
        for record in reversed(self.records):
            if record.param_name == param_name:
                return record.fah_after > record.fah_before and record.recall_after < record.recall_before
        return False

    def iterations_since_last_tried(self, param_name: str, current_iteration: int) -> int:
        """How many iterations since this knob was last adjusted."""
        if param_name not in self._last_tried:
            return current_iteration
        return max(0, current_iteration - self._last_tried[param_name])


class TrendAnalyzer:
    """Sliding window trend analysis over metrics history."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: list[tuple[int, float, float]] = []

    def add(self, iteration: int, fah: float, recall: float):
        self.history.append((iteration, fah, recall))

    def _window(self) -> list[tuple[int, float, float]]:
        if len(self.history) <= self.window_size:
            return self.history
        return self.history[-self.window_size :]

    def fah_trend(self) -> float:
        """Linear regression slope of FAH. Negative = improving."""
        window = self._window()
        if len(window) < 2:
            return 0.0
        x = np.array([p[0] for p in window], dtype=np.float32)
        y = np.array([p[1] for p in window], dtype=np.float32)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def recall_trend(self) -> float:
        """Linear regression slope of recall. Positive = improving."""
        window = self._window()
        if len(window) < 2:
            return 0.0
        x = np.array([p[0] for p in window], dtype=np.float32)
        y = np.array([p[2] for p in window], dtype=np.float32)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def is_stagnant(self) -> bool:
        """True if no meaningful change in last window_size iterations.
        Meaningful = abs(slope) > 0.001 for either metric."""
        return abs(self.fah_trend()) <= 0.001 and abs(self.recall_trend()) <= 0.001

    def momentum_direction(self) -> str:
        """Return 'improving', 'worsening', or 'flat'."""
        fah_slope = self.fah_trend()
        recall_slope = self.recall_trend()
        improving = fah_slope < -0.001 or recall_slope > 0.001
        worsening = fah_slope > 0.001 and recall_slope < -0.001
        if improving and not worsening:
            return "improving"
        if worsening:
            return "worsening"
        return "flat"


class Phase(enum.Enum):
    AGGRESSIVE_FAH = "aggressive_fah"
    BALANCED = "balanced"
    PRECISION_RECALL = "precision_recall"
    POLISH = "polish"


@dataclass
class PhaseParams:
    lr_multiplier: float
    step_multiplier: float
    preferred_knobs: list[str]


class PhaseController:
    """Determine optimization phase based on metric gaps."""

    PHASE_PARAMS = {
        Phase.AGGRESSIVE_FAH: PhaseParams(
            lr_multiplier=1.0,
            step_multiplier=1.5,
            preferred_knobs=["negative_class_weight", "hard_negative_class_weight"],
        ),
        Phase.BALANCED: PhaseParams(
            lr_multiplier=0.8,
            step_multiplier=1.0,
            preferred_knobs=[],
        ),
        Phase.PRECISION_RECALL: PhaseParams(
            lr_multiplier=0.5,
            step_multiplier=1.0,
            preferred_knobs=["positive_class_weight", "dropout_rate"],
        ),
        Phase.POLISH: PhaseParams(
            lr_multiplier=0.3,
            step_multiplier=0.5,
            preferred_knobs=["learning_rate"],
        ),
    }

    def determine_phase(self, current_fah, current_recall, target_fah, target_recall) -> Phase:
        if current_fah > target_fah * 2.0:
            return Phase.AGGRESSIVE_FAH
        if current_fah > target_fah and current_recall < target_recall:
            return Phase.BALANCED
        if current_fah <= target_fah and current_recall < target_recall:
            return Phase.PRECISION_RECALL
        return Phase.POLISH

    def get_phase_params(self, phase: Phase) -> PhaseParams:
        return self.PHASE_PARAMS[phase]


class AdaptiveKnobController:
    """Replaces rigid MicroConfigAdjuster. Uses impact memory for knob selection."""

    ALL_KNOBS = {
        "positive_class_weight": {"min": 0.5, "max": 3.0, "step": 0.2, "default_dir": 1},
        "negative_class_weight": {"min": 10.0, "max": 50.0, "step": 3.0, "default_dir": 1},
        "hard_negative_class_weight": {"min": 20.0, "max": 100.0, "step": 5.0, "default_dir": 1},
        "learning_rate": {"min": 1e-6, "max": 0.001, "step": 0.00002, "default_dir": -1},
        "dropout_rate": {"min": 0.0, "max": 0.5, "step": 0.05, "default_dir": -1},
    }

    def __init__(self, config: dict, auto_tuning_config: dict):
        self.current_values: dict[str, float] = {}
        self._sync_from_config(config)
        at = auto_tuning_config
        self.ALL_KNOBS["positive_class_weight"]["min"] = at.get("positive_weight_range", [0.5, 3.0])[0]
        self.ALL_KNOBS["positive_class_weight"]["max"] = at.get("positive_weight_range", [0.5, 3.0])[1]
        self.ALL_KNOBS["negative_class_weight"]["min"] = at.get("negative_weight_range", [10.0, 50.0])[0]
        self.ALL_KNOBS["negative_class_weight"]["max"] = at.get("negative_weight_range", [10.0, 50.0])[1]
        self.ALL_KNOBS["hard_negative_class_weight"]["min"] = at.get("hard_negative_weight_range", [20.0, 100.0])[0]
        self.ALL_KNOBS["hard_negative_class_weight"]["max"] = at.get("hard_negative_weight_range", [20.0, 100.0])[1]
        self._last_knob: str | None = None
        self._last_direction: int = 0
        self._escalation_factor: float = 1.0

    def _sync_from_config(self, config: dict):
        """Read current values from config dict."""
        training = config.get("training", {})
        model = config.get("model", {})
        pos_w = training.get("positive_class_weight", [1.0])
        self.current_values["positive_class_weight"] = pos_w[-1] if isinstance(pos_w, list) else pos_w
        neg_w = training.get("negative_class_weight", [20.0])
        self.current_values["negative_class_weight"] = neg_w[-1] if isinstance(neg_w, list) else neg_w
        hn_w = training.get("hard_negative_class_weight", [40.0])
        self.current_values["hard_negative_class_weight"] = hn_w[-1] if isinstance(hn_w, list) else hn_w
        lrs = training.get("learning_rates", [0.0001])
        self.current_values["learning_rate"] = lrs[-1] if isinstance(lrs, list) else lrs
        self.current_values["dropout_rate"] = model.get("dropout_rate", 0.2)

    def select_knob(
        self,
        phase: Phase,
        impact_memory: ImpactMemory,
        trend: TrendAnalyzer,
        current_iteration: int,
    ) -> tuple[str, int, float]:
        """Select best knob to adjust. Returns (knob_name, direction, step_size)."""
        phase_params = PhaseController.PHASE_PARAMS[phase]

        scores: dict[str, float] = {}
        impact_scores = impact_memory.get_impact_scores()
        for knob_name in self.ALL_KNOBS:
            phase_bonus = 2.0 if knob_name in phase_params.preferred_knobs else 0.0
            impact = impact_scores.get(knob_name, 0.5)
            recency = impact_memory.iterations_since_last_tried(knob_name, current_iteration)
            scores[knob_name] = phase_bonus + impact + min(recency * 0.1, 1.0)

        best_knob = "positive_class_weight"
        best_score = float("-inf")
        for name, score in scores.items():
            if score > best_score:
                best_score = score
                best_knob = name

        knob_info = self.ALL_KNOBS[best_knob]
        direction = knob_info["default_dir"]
        if phase == Phase.AGGRESSIVE_FAH and best_knob in (
            "negative_class_weight",
            "hard_negative_class_weight",
        ):
            direction = 1
        elif phase == Phase.PRECISION_RECALL and best_knob == "positive_class_weight":
            direction = 1

        if impact_memory.last_was_harmful(best_knob):
            direction = -direction

        step = knob_info["step"] * self._escalation_factor

        self._last_knob = best_knob
        self._last_direction = direction

        return (best_knob, direction, step)

    def apply_adjustment(self, config: dict, knob_name: str, direction: int, step: float) -> dict:
        """Apply knob adjustment to config dict. Returns new config."""
        new_config = copy.deepcopy(config)
        training = new_config.setdefault("training", {})
        model = new_config.setdefault("model", {})

        knob_info = self.ALL_KNOBS[knob_name]
        current = self.current_values[knob_name]
        new_value = current + direction * step
        new_value = max(knob_info["min"], min(knob_info["max"], new_value))
        self.current_values[knob_name] = new_value

        if knob_name == "positive_class_weight":
            training["positive_class_weight"] = [new_value]
        elif knob_name == "negative_class_weight":
            training["negative_class_weight"] = [new_value]
        elif knob_name == "hard_negative_class_weight":
            training["hard_negative_class_weight"] = [new_value]
        elif knob_name == "learning_rate":
            training["learning_rates"] = [new_value]
        elif knob_name == "dropout_rate":
            model["dropout_rate"] = new_value

        return new_config

    def escalate(self):
        """Increase step sizes by 1.5x for stuck situations."""
        self._escalation_factor *= 1.5

    def reset_escalation(self):
        self._escalation_factor = 1.0

    def reverse_last(self):
        """Reverse the last direction."""
        self._last_direction = -self._last_direction


class StagnationAction(enum.Enum):
    ESCALATE = "escalate"
    REVERSE = "reverse"
    SWITCH = "switch"
    RESTART = "restart"


@dataclass
class TuneMetrics:
    fah: float
    recall: float
    precision: float
    f1: float
    threshold: float
    recall_at_target_fah: float
    fah_at_target_recall: float
    avg_viable_recall: float


class AutoTuner:
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        auto_tuning_config: dict | None = None,
        console: Console | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.base_config = copy.deepcopy(config)
        self.current_config = copy.deepcopy(config)
        self.console = console or Console()

        at = auto_tuning_config or config.get("auto_tuning", {})
        self.target_fah = at.get("target_fah", 0.3)
        self.target_recall = at.get("target_recall", 0.92)
        self.max_iterations = at.get("max_iterations", 100)
        self.patience = at.get("patience", 10)
        self.steps_per_iteration = at.get("steps_per_iteration", 5000)
        self.initial_lr = at.get("initial_lr", 0.0001)
        self.lr_decay_factor = at.get("lr_decay_factor", 0.7)
        self.min_lr = at.get("min_lr", 1e-6)
        self.pareto_threshold = at.get("pareto_improvement_threshold", 0.005)
        self.convergence_window = at.get("convergence_window", 5)
        self.output_dir = Path(at.get("output_dir", "./tuning_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_lr = self.initial_lr
        self.iteration = 0
        self.no_improvement_count = 0
        self.stagnation_count = 0
        self.best_checkpoint = checkpoint_path

        self.pareto = ParetoFrontier()
        self.impact = ImpactMemory()
        self.trend = TrendAnalyzer(window_size=self.convergence_window)
        self.phase_ctrl = PhaseController()
        self.knob_ctrl = AdaptiveKnobController(config, at)
        self.hard_miner = HardExampleMiner(strategy="confidence", fp_threshold=0.8, max_samples=10000)

        self._setup_logging()

    def _setup_logging(self):
        """Setup file + console logging."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.file_logger = logging.getLogger(f"autotune_{id(self)}")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.handlers.clear()
        fh = logging.FileHandler(log_dir / "autotune.log", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.file_logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.file_logger.addHandler(sh)

    def _evaluate(self, model: tf.keras.Model, dataset: WakeWordDataset) -> TuneMetrics:
        """Evaluate model with multi-threshold scanning. NOT hardcoded threshold=0.5."""
        start_time = time.perf_counter()
        y_true_list = []
        y_scores_list = []

        val_gen_factory = dataset.val_generator_factory()
        try:
            val_gen = val_gen_factory()
            for batch in val_gen:
                if isinstance(batch, tuple):
                    batch_features = batch[0]
                    batch_labels = batch[1]
                else:
                    batch_features = batch.get("features")
                    batch_labels = batch.get("labels")
                predictions = model.predict(batch_features, verbose=0)
                y_true_list.extend(batch_labels.flatten().tolist())
                y_scores_list.extend(predictions.flatten().tolist())
        except Exception as e:
            self.file_logger.error(f"Evaluation failed: {e}")
            return TuneMetrics(
                fah=float("inf"),
                recall=0.0,
                precision=0.0,
                f1=0.0,
                threshold=0.0,
                recall_at_target_fah=0.0,
                fah_at_target_recall=float("inf"),
                avg_viable_recall=0.0,
            )

        y_true = np.array(y_true_list)
        y_scores = np.array(y_scores_list)
        calc = MetricsCalculator(y_true=y_true, y_score=y_scores)

        ambient_hours = float(self.base_config.get("training", {}).get("ambient_duration_hours", 0.0))

        recall_at_fah, thresh_fah, actual_fah = calc.compute_recall_at_target_fah(ambient_hours, self.target_fah)
        fah_at_recall, thresh_recall, actual_recall = calc.compute_fah_at_target_recall(ambient_hours, self.target_recall)
        avg_viable = calc.compute_average_viable_recall(ambient_hours, max_fah=self.target_fah * 2)

        if actual_fah <= self.target_fah:
            best_threshold = thresh_recall
        else:
            best_threshold = thresh_fah

        full_metrics = calc.compute_all_metrics(ambient_hours, threshold=best_threshold)

        metrics = TuneMetrics(
            fah=full_metrics.get("ambient_false_positives_per_hour", float("inf")),
            recall=full_metrics.get("recall", 0.0),
            precision=full_metrics.get("precision", 0.0),
            f1=full_metrics.get("f1_score", 0.0),
            threshold=best_threshold,
            recall_at_target_fah=recall_at_fah,
            fah_at_target_recall=fah_at_recall,
            avg_viable_recall=avg_viable,
        )
        elapsed = time.perf_counter() - start_time
        self.file_logger.info(f"Evaluation completed in {elapsed:.2f}s")
        return metrics

    def _load_model(self) -> tf.keras.Model:
        hardware_cfg = self.current_config.get("hardware", {})
        clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        input_shape = (int(clip_duration_ms / window_step_ms), mel_bins)
        model = build_model(input_shape=input_shape, model_config=self.current_config.get("model", {}))
        _ = model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
        model.load_weights(self.best_checkpoint)
        return model

    def _apply_iteration_config(self, config: dict, phase: Phase) -> dict:
        """Override config for a fine-tuning iteration (steps, LR, based on phase)."""
        config = copy.deepcopy(config)
        training = config.setdefault("training", {})
        phase_params = self.phase_ctrl.get_phase_params(phase)
        steps = int(self.steps_per_iteration * phase_params.step_multiplier)
        training["training_steps"] = [steps]
        effective_lr = max(self.current_lr * phase_params.lr_multiplier, self.min_lr)
        training["learning_rates"] = [effective_lr]
        return config

    def _run_fine_tuning_iteration(self, config: dict) -> tf.keras.Model:
        """Run one fine-tuning iteration. Returns tuned model."""
        dataset = WakeWordDataset(config)
        dataset.build()

        hardware_cfg = config.get("hardware", {})
        clip_duration_ms = hardware_cfg.get("clip_duration_ms", 1000)
        window_step_ms = hardware_cfg.get("window_step_ms", 10)
        mel_bins = hardware_cfg.get("mel_bins", 40)
        max_time_frames = int(clip_duration_ms / window_step_ms)
        input_shape = (max_time_frames, mel_bins)

        trainer = Trainer(config)
        model = trainer.train(
            train_data_factory=dataset.train_generator_factory(max_time_frames=max_time_frames),
            val_data_factory=dataset.val_generator_factory(max_time_frames=max_time_frames),
            input_shape=input_shape,
            weights_path=self.best_checkpoint,
        )
        dataset.close()
        return model

    def _save_checkpoint(self, model: tf.keras.Model, metrics: TuneMetrics) -> str:
        path = self.output_dir / f"tuned_fah{metrics.fah:.3f}_rec{metrics.recall:.3f}_iter{self.iteration}.weights.h5"
        model.save_weights(str(path))
        return str(path)

    def _handle_stagnation(self) -> StagnationAction:
        self.stagnation_count += 1
        if self.stagnation_count == 1:
            return StagnationAction.ESCALATE
        if self.stagnation_count == 2:
            return StagnationAction.REVERSE
        if self.stagnation_count == 3:
            return StagnationAction.SWITCH
        self.stagnation_count = 0
        self.current_lr = max(self.current_lr * self.lr_decay_factor, self.min_lr)
        return StagnationAction.RESTART

    def _log_header(self):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")
        table.add_row("Checkpoint", self.checkpoint_path)
        table.add_row("Target FAH", f"< {self.target_fah}")
        table.add_row("Target Recall", f"> {self.target_recall}")
        table.add_row("Max Iterations", str(self.max_iterations))
        table.add_row("Steps/Iteration", str(self.steps_per_iteration))
        table.add_row("Output Dir", str(self.output_dir))
        panel = Panel(table, title="Auto-Tuning Configuration", border_style="blue", expand=False)
        self.console.print(panel)

    def _log_iteration_status(self, metrics: TuneMetrics, phase: Phase, knob: str, direction: int):
        table = Table(title=f"Iteration {self.iteration}/{self.max_iterations} — Phase: {phase.value}")
        table.add_column("Metric", style="bold")
        table.add_column("Current", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")

        fah_ok = metrics.fah <= self.target_fah
        recall_ok = metrics.recall >= self.target_recall
        table.add_row(
            "FAH",
            f"[{'green' if fah_ok else 'red'}]{metrics.fah:.4f}[/]",
            f"< {self.target_fah}",
            "✓" if fah_ok else "✗",
        )
        table.add_row(
            "Recall",
            f"[{'green' if recall_ok else 'red'}]{metrics.recall:.4f}[/]",
            f"> {self.target_recall}",
            "✓" if recall_ok else "✗",
        )
        table.add_row("Threshold", f"{metrics.threshold:.4f}", "-", "-")
        table.add_row("AVR", f"{metrics.avg_viable_recall:.4f}", "-", "-")
        table.add_row("Knob", f"{knob} ({'↑' if direction > 0 else '↓'})", "-", "-")
        self.console.print(table)

    def _log_final_summary(self, result: dict[str, Any]):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        table.add_row("Iterations", str(result["iterations"]))
        table.add_row("Best FAH", f"{result['best_fah']:.4f}")
        table.add_row("Best Recall", f"{result['best_recall']:.4f}")
        table.add_row("Target Met", "✅ Yes" if result["target_met"] else "❌ No")
        table.add_row("Pareto Points", str(len(self.pareto.points)))
        table.add_row("Best Checkpoint", str(result["best_checkpoint"] or "N/A"))
        border = "green" if result["target_met"] else "yellow"
        panel = Panel(table, title="Auto-Tuning Complete", border_style=border, expand=False)
        self.console.print(panel)

    def tune(self) -> dict:
        """Main tuning loop with sophisticated multi-phase optimization."""
        self._log_header()
        self.file_logger.info(f"Starting auto-tuning: target_fah={self.target_fah}, target_recall={self.target_recall}")

        self.console.print("[cyan]Evaluating initial model...[/]")
        model = self._load_model()
        dataset = WakeWordDataset(self.base_config)
        dataset.build()
        initial_metrics = self._evaluate(model, dataset)
        dataset.close()
        del model
        tf.keras.backend.clear_session()

        self.file_logger.info(f"Initial: FAH={initial_metrics.fah:.4f}, Recall={initial_metrics.recall:.4f}, Threshold={initial_metrics.threshold:.4f}")

        self.pareto.add(
            initial_metrics.fah,
            initial_metrics.recall,
            self.checkpoint_path,
            copy.deepcopy(self.base_config),
            0,
        )
        self.trend.add(0, initial_metrics.fah, initial_metrics.recall)

        prev_fah = initial_metrics.fah
        prev_recall = initial_metrics.recall

        if initial_metrics.fah <= self.target_fah and initial_metrics.recall >= self.target_recall:
            self.console.print("[bold green]Targets already met! No tuning needed.[/]")
            result = {
                "best_fah": initial_metrics.fah,
                "best_recall": initial_metrics.recall,
                "final_fah": initial_metrics.fah,
                "final_recall": initial_metrics.recall,
                "iterations": 0,
                "best_checkpoint": self.checkpoint_path,
                "target_met": True,
                "pareto_frontier": [(p.fah, p.recall) for p in self.pareto.points],
            }
            self._log_final_summary(result)
            return result

        last_knob = None
        last_direction = 0
        last_step = 0.0

        while self.iteration < self.max_iterations:
            self.iteration += 1

            phase = self.phase_ctrl.determine_phase(prev_fah, prev_recall, self.target_fah, self.target_recall)
            self.file_logger.info(f"=== Iteration {self.iteration} | Phase: {phase.value} ===")

            momentum = self.trend.momentum_direction()

            if momentum == "improving" and last_knob is not None:
                knob, direction, step = last_knob, last_direction, last_step
                self.file_logger.info(f"Momentum: continuing {knob} {'↑' if direction > 0 else '↓'}")
            else:
                knob, direction, step = self.knob_ctrl.select_knob(phase, self.impact, self.trend, self.iteration)
                self.file_logger.info(f"Selected knob: {knob} {'↑' if direction > 0 else '↓'} step={step:.6f}")

            self.current_config = self.knob_ctrl.apply_adjustment(self.current_config, knob, direction, step)
            iter_config = self._apply_iteration_config(self.current_config, phase)

            try:
                model = self._run_fine_tuning_iteration(iter_config)
            except Exception as e:
                self.file_logger.error(f"Training failed: {e}")
                self.console.print(f"[red]Training error: {e}[/]")
                continue

            dataset = WakeWordDataset(self.base_config)
            dataset.build()
            metrics = self._evaluate(model, dataset)
            dataset.close()

            self.impact.record(
                knob,
                direction * step,
                prev_fah,
                metrics.fah,
                prev_recall,
                metrics.recall,
                self.iteration,
            )
            self.trend.add(self.iteration, metrics.fah, metrics.recall)

            self._log_iteration_status(metrics, phase, knob, direction)
            self.file_logger.info(f"Result: FAH={metrics.fah:.4f}, Recall={metrics.recall:.4f}, Threshold={metrics.threshold:.4f}")

            is_pareto_improving = self.pareto.add(
                metrics.fah,
                metrics.recall,
                "",
                copy.deepcopy(self.current_config),
                self.iteration,
            )

            if is_pareto_improving:
                checkpoint_path = self._save_checkpoint(model, metrics)
                self.pareto.points[-1].checkpoint_path = checkpoint_path
                self.best_checkpoint = checkpoint_path
                self.no_improvement_count = 0
                self.stagnation_count = 0
                self.knob_ctrl.reset_escalation()
                last_knob, last_direction, last_step = knob, direction, step
                self.console.print(f"[green]✓ Pareto improvement! Saved: {checkpoint_path}[/]")
                self.file_logger.info(f"PARETO IMPROVEMENT: {checkpoint_path}")
            else:
                self.no_improvement_count += 1
                last_knob = None
                self.file_logger.info(f"No Pareto improvement ({self.no_improvement_count}/{self.patience})")

            del model
            tf.keras.backend.clear_session()

            prev_fah = metrics.fah
            prev_recall = metrics.recall

            if metrics.fah <= self.target_fah and metrics.recall >= self.target_recall:
                self.console.print("[bold green]🎯 Target metrics achieved![/]")
                self.file_logger.info("TARGET MET!")
                break

            if self.no_improvement_count >= self.patience:
                action = self._handle_stagnation()
                self.file_logger.info(f"Stagnation action: {action.value}")
                self.console.print(f"[yellow]Stagnation detected → {action.value}[/]")

                if action == StagnationAction.ESCALATE:
                    self.knob_ctrl.escalate()
                elif action == StagnationAction.REVERSE:
                    self.knob_ctrl.reverse_last()
                elif action == StagnationAction.SWITCH:
                    pass
                elif action == StagnationAction.RESTART:
                    best_point = self.pareto.best_by_target(self.target_fah, self.target_recall)
                    if best_point:
                        self.best_checkpoint = best_point.checkpoint_path
                        self.current_config = copy.deepcopy(best_point.config_snapshot)
                        self.knob_ctrl._sync_from_config(self.current_config)
                        self.console.print(f"[yellow]Restarting from best Pareto point (FAH={best_point.fah:.4f}, Recall={best_point.recall:.4f})[/]")

                self.no_improvement_count = 0

        best_point = self.pareto.best_by_target(self.target_fah, self.target_recall)
        result = {
            "best_fah": best_point.fah if best_point else prev_fah,
            "best_recall": best_point.recall if best_point else prev_recall,
            "final_fah": prev_fah,
            "final_recall": prev_recall,
            "iterations": self.iteration,
            "best_checkpoint": best_point.checkpoint_path if best_point else self.best_checkpoint,
            "target_met": prev_fah <= self.target_fah and prev_recall >= self.target_recall,
            "pareto_frontier": [(p.fah, p.recall) for p in self.pareto.points],
        }
        self._log_final_summary(result)
        return result


def autotune(
    checkpoint_path: str,
    config: dict,
    output_dir: str = "./tuning_output",
    target_fah: float = 0.3,
    target_recall: float = 0.92,
    max_iterations: int = 100,
) -> dict:
    """Convenience function for auto-tuning."""
    at_config = config.get("auto_tuning", {})
    at_config.setdefault("target_fah", target_fah)
    at_config.setdefault("target_recall", target_recall)
    at_config.setdefault("max_iterations", max_iterations)
    at_config.setdefault("output_dir", output_dir)
    tuner = AutoTuner(
        checkpoint_path=checkpoint_path,
        config=config,
        auto_tuning_config=at_config,
    )
    return tuner.tune()
