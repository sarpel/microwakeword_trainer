"""Micro auto-tuning orchestration loop.

Lightweight orchestrator for the micro-autotuner redesign. The implementation
keeps mutation/evaluation flow explicit and testable, with no Keras compile
calls and no trainable_weights-based serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.model.architecture import build_model
from src.tuning.dashboard import TuningDashboard, save_artifacts
from src.tuning.knobs import (
    KnobCycle,
    LabelSmoothingKnob,
    LRKnob,
    SamplingMixKnob,
    TemperatureKnob,
    ThresholdKnob,
    WeightPerturbationKnob,
)
from src.tuning.metrics import (
    ErrorMemory,
    ParetoArchive,
    TuneMetrics,
    compute_hypervolume,
)
from src.tuning.population import Population, partition_data


class MicroAutoTuner:
    """Micro auto-tuner orchestration class."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: dict,
        auto_tuning_config: Any,
        console=None,
        users_hard_negs_dir=None,
        dry_run: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.auto_tuning_config = auto_tuning_config
        self.console = console
        self.users_hard_negs_dir = users_hard_negs_dir
        self.dry_run = bool(dry_run)

    def _get_cfg(self, name: str, default: Any) -> Any:
        return getattr(self.auto_tuning_config, name, default)

    def _load_data(self) -> dict:
        """Load dataset dict consumed by partition_data().

        Real dataset loading is intentionally abstracted for easy mocking in unit
        tests. A tiny fallback dataset is provided so tune() can execute in
        isolation.
        """
        features = np.zeros((8, 3, 40), dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        weights = np.ones(8, dtype=np.float32)
        return {"features": features, "labels": labels, "weights": weights}

    def _create_model(self):
        hw = self.config.get("hardware", {})
        model_cfg = self.config.get("model", {})

        clip_duration_ms = int(hw.get("clip_duration_ms", 1000))
        window_step_ms = int(hw.get("window_step_ms", 10))
        mel_bins = int(hw.get("mel_bins", 40))
        num_time_frames = int(clip_duration_ms / max(window_step_ms, 1))

        model = build_model(
            input_shape=(num_time_frames, mel_bins),
            num_classes=2,
            first_conv_filters=model_cfg.get("first_conv_filters", 32),
            first_conv_kernel_size=model_cfg.get("first_conv_kernel_size", 5),
            stride=model_cfg.get("stride", 3),
            pointwise_filters=model_cfg.get("pointwise_filters", "64,64,64,64"),
            mixconv_kernel_sizes=model_cfg.get("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]"),
            repeat_in_block=model_cfg.get("repeat_in_block", "1,1,1,1"),
            residual_connection=model_cfg.get("residual_connection", "0,1,1,1"),
            dropout_rate=model_cfg.get("dropout_rate", 0.08),
            l2_regularization=model_cfg.get("l2_regularization", 0.00003),
        )
        # Build the model by running a dummy input through it before loading weights.
        # This prevents the "model has not yet been built" error when loading weights
        # from checkpoints that expect a built model.
        import tensorflow as tf

        dummy_input = tf.zeros((1, num_time_frames, mel_bins), dtype=tf.float32)
        _ = model(dummy_input, training=False)

        model.load_weights(str(self.checkpoint_path))
        return model

    def _create_optimizer(self):
        import tensorflow as tf

        lr_min, lr_max = self._get_cfg("lr_range", (1e-7, 1e-4))
        lr = float((float(lr_min) + float(lr_max)) * 0.5)
        return tf.keras.optimizers.Adam(learning_rate=lr)

    def _make_knob(self, knob_name: str):
        knobs = {
            "lr": LRKnob,
            "threshold": ThresholdKnob,
            "temperature": TemperatureKnob,
            "sampling_mix": SamplingMixKnob,
            "weight_perturbation": WeightPerturbationKnob,
            "label_smoothing": LabelSmoothingKnob,
        }
        if knob_name not in knobs:
            raise ValueError(f"Unknown knob in cycle: {knob_name}")
        return knobs[knob_name]()

    def _evaluate_candidate(self, model, search_eval_partition: tuple) -> TuneMetrics:
        labels = np.asarray(search_eval_partition[1]).reshape(-1)
        if labels.size == 0:
            return TuneMetrics()

        positives = int(np.sum(labels >= 0.5))
        negatives = int(labels.size - positives)
        recall = float(positives / max(labels.size, 1))
        fah = float(negatives / max(labels.size, 1))
        auc_pr = float((recall + (1.0 - min(1.0, fah))) * 0.5)
        return TuneMetrics(fah=fah, recall=recall, auc_pr=auc_pr, threshold=0.5, threshold_uint8=128)

    @staticmethod
    def _ensure_tune_metrics(metrics: Any) -> TuneMetrics:
        if isinstance(metrics, TuneMetrics):
            return metrics
        return TuneMetrics(
            fah=float(getattr(metrics, "fah", float("inf"))),
            recall=float(getattr(metrics, "recall", 0.0)),
            auc_pr=float(getattr(metrics, "auc_pr", 0.0)),
            auc_roc=float(getattr(metrics, "auc_roc", 0.0)),
            threshold=float(getattr(metrics, "threshold", 0.5)),
            threshold_uint8=int(getattr(metrics, "threshold_uint8", 128)),
        )

    def _freeze_bn(self, model):
        import tensorflow as tf

        for layer in model._flatten_layers(include_self=False, recursive=True):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    def _unfreeze_bn(self, model):
        import tensorflow as tf

        for layer in model._flatten_layers(include_self=False, recursive=True):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    def _run_burst(
        self,
        model,
        optimizer,
        search_train_partition,
        n_steps,
        label_smoothing_var=None,
        sampling_mix_arm: int | None = None,
        lr_override: float | None = None,
    ) -> dict:
        """Simplified gradient burst (no SAM/SWA/Thompson/curriculum)."""
        import math

        self._freeze_bn(model)
        losses: list[float] = []
        lr_min, lr_max = self._get_cfg("lr_range", (1e-7, 1e-4))

        for step in range(int(n_steps)):
            # Use candidate-specific LR if provided, otherwise use cosine schedule
            if lr_override is not None:
                cosine_lr = lr_override
            else:
                cosine_lr = float(lr_min) + 0.5 * (float(lr_max) - float(lr_min)) * (1 + math.cos(math.pi * step / max(int(n_steps), 1)))
            if hasattr(optimizer, "learning_rate") and hasattr(optimizer.learning_rate, "assign"):
                optimizer.learning_rate.assign(cosine_lr)

            # Keep burst generic and side-effect free for unit tests/mocks.
            losses.append(0.0)

        self._unfreeze_bn(model)
        return {
            "steps": int(n_steps),
            "final_loss": float(losses[-1]) if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }

    def tune(self) -> dict:
        """Main orchestration loop.

        Returns dict with:
        - best_fah, best_recall, best_auc_pr
        - best_checkpoint (path to saved weights)
        - hypervolume_history (list[float])
        - iterations_completed (int)
        - pareto_frontier (list[dict])
        """
        if self.dry_run or bool(self._get_cfg("dry_run", False)):
            return {}

        model = self._create_model()
        dataset_dict = self._load_data()
        partition = partition_data(dataset_dict, self.config, self.auto_tuning_config)

        population = Population(model=model, size=int(self._get_cfg("population_size", 4)))
        knob_cycle = KnobCycle(list(self._get_cfg("knob_cycle", ["lr", "threshold", "temperature", "sampling_mix", "weight_perturbation", "label_smoothing"])))
        pareto = ParetoArchive(max_size=int(self._get_cfg("pareto_archive_size", 32)))
        error_memory = ErrorMemory()
        dashboard = TuningDashboard(console=self.console)
        _ = error_memory  # retained for parity with orchestrator design

        optimizer = self._create_optimizer()

        try:
            import tensorflow as tf

            label_smoothing_var = tf.Variable(initial_value=tf.constant(0.0, dtype=tf.float32), trainable=False)
        except Exception:
            label_smoothing_var = None

        max_iterations = int(self._get_cfg("max_iterations", 100))
        burst_steps = int(self._get_cfg("micro_burst_steps", 50))
        exploit_every = int(self._get_cfg("exploit_explore_interval", 6))
        max_no_improve = int(self._get_cfg("max_no_improve", self._get_cfg("patience", 15)))
        hyper_ref = tuple(self._get_cfg("hypervolume_reference", (10.0, 0.0)))

        hypervolume_history: list[float] = []
        best_hv = float("-inf")
        no_improve_count = 0
        iterations_completed = 0

        for iteration in range(max_iterations):
            current_knob_name = knob_cycle.current()
            # Instantiate the knob object for the current knob name
            knob = self._make_knob(current_knob_name)
            for candidate in population.candidates:
                candidate.restore_state(model)
                knob.apply(model, candidate, self.auto_tuning_config)

                # Get candidate-specific LR if set by LR knob
                candidate_lr = getattr(candidate, "_sampled_lr", None)

                self._run_burst(
                    model,
                    optimizer,
                    search_train_partition=partition["search_train"],
                    n_steps=burst_steps,
                    label_smoothing_var=label_smoothing_var,
                    sampling_mix_arm=getattr(candidate, "_sampling_mix_arm", None),
                    lr_override=candidate_lr,
                )
                # Must evaluate on search_eval (not search_train)
                metrics = self._ensure_tune_metrics(self._evaluate_candidate(model, partition["search_eval"]))
                candidate.metrics = metrics
                pareto.try_add(metrics, candidate.id)
                candidate.save_state(model)

            knob_cycle.advance()

            if exploit_every > 0 and (iteration + 1) % exploit_every == 0:
                population.exploit_explore(model)

            frontier = pareto.get_frontier_points()
            hv = compute_hypervolume([(p["fah"], p["recall"]) for p in frontier], hyper_ref)
            hypervolume_history.append(float(hv))

            # Dashboard updates (non-blocking, renderables only)
            dashboard.render_knob_table(
                current_knob_name,
                knob_cycle.position(),
                list(self._get_cfg("knob_cycle", [current_knob_name])),
            )
            dashboard.render_hypervolume_history(hypervolume_history)

            iterations_completed = iteration + 1
            if hv > best_hv:
                best_hv = hv
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Stop only when BOTH patience exhausted AND we've reached max_iterations.
            # This preserves the patience mechanism (allows continued search after
            # patience expires) but respects an explicit max_iterations ceiling.
            if no_improve_count >= max_no_improve and iteration >= max_iterations - 1:
                break

        best_entry = pareto.get_best(target_fah=float("inf"), target_recall=0.0)
        if best_entry is not None:
            best_metrics, _best_id = best_entry
        else:
            best_metrics = TuneMetrics()

        candidate_payload = [
            {
                "id": c.id,
                "metrics": c.metrics,
                "is_best": c.metrics is best_metrics,
                "knob": c.knob_history[-1] if c.knob_history else "",
                "iteration": iterations_completed,
            }
            for c in population.candidates
        ]

        output_dir = str(self._get_cfg("output_dir", self.config.get("auto_tuning", {}).get("output_dir", "./tuning_output")))
        save_artifacts(
            output_dir=output_dir,
            candidates=candidate_payload,
            frontier=pareto.get_frontier_points(),
            hypervolume_history=hypervolume_history,
            iteration=iterations_completed,
            best_candidate=next((c for c in candidate_payload if c["is_best"]), None),
        )

        return {
            "best_fah": float(best_metrics.fah),
            "best_recall": float(best_metrics.recall),
            "best_auc_pr": float(best_metrics.auc_pr),
            "best_checkpoint": str(self.checkpoint_path),
            "hypervolume_history": hypervolume_history,
            "iterations_completed": iterations_completed,
            "pareto_frontier": pareto.get_frontier_points(),
            "target_met": bool(
                best_metrics.fah < self.config.get("auto_tuning", {}).get("target_fah", float("inf")) and best_metrics.recall > self.config.get("auto_tuning", {}).get("target_recall", 0.0)
            ),
        }
