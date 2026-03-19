"""Micro auto-tuning orchestration loop.

Lightweight orchestrator for the micro-autotuner redesign. The implementation
keeps mutation/evaluation flow explicit and testable, with no Keras compile
calls and no trainable_weights-based serialization.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

from src.data.dataset import FeatureStore
from src.model.architecture import build_model
from src.tuning.dashboard import TuningDashboard, save_artifacts
from src.tuning.knobs import (
    Knob,
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
    ThresholdOptimizer,
    TuneMetrics,
    compute_hypervolume,
)
from src.tuning.population import Population, partition_data

logger = logging.getLogger(__name__)


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
        """Load dataset dict consumed by partition_data()."""
        paths_cfg = self.config.get("paths", {})
        hardware_cfg = self.config.get("hardware", {})
        processed_dir = Path(paths_cfg.get("processed_dir", "./data/processed"))

        clip_duration_ms = int(hardware_cfg.get("clip_duration_ms", 1000))
        window_step_ms = int(hardware_cfg.get("window_step_ms", 10))
        max_time_frames = int(clip_duration_ms / max(window_step_ms, 1))

        loaded_features: list[np.ndarray] = []
        loaded_labels: list[float] = []
        loaded_weights: list[float] = []
        pos_count = 0
        neg_count = 0
        hard_neg_count = 0

        def _pad_or_truncate(feature_2d: np.ndarray) -> np.ndarray:
            current_frames = int(feature_2d.shape[0])
            if current_frames > max_time_frames:
                return feature_2d[:max_time_frames, :]
            if current_frames < max_time_frames:
                pad = np.zeros((max_time_frames - current_frames, 40), dtype=np.float32)
                return np.vstack([feature_2d, pad])
            return feature_2d

        def _load_store_path(store_path: Path) -> None:
            nonlocal pos_count, neg_count, hard_neg_count
            if not store_path.exists():
                return

            store = FeatureStore(store_path)
            try:
                store.open(readonly=True)
                for idx in range(len(store)):
                    flat_feature, label_int = store.get(idx)
                    flat_feature = np.asarray(flat_feature, dtype=np.float32).reshape(-1)
                    if flat_feature.size == 0:
                        continue
                    if flat_feature.size % 40 != 0:
                        continue

                    feature_2d = flat_feature.reshape(-1, 40)
                    feature_fixed = _pad_or_truncate(feature_2d)

                    if int(label_int) == 1:
                        label_bin = 1.0
                        sample_weight = 1.0
                        pos_count += 1
                    elif int(label_int) == 2:
                        label_bin = 0.0
                        sample_weight = 2.0
                        hard_neg_count += 1
                    else:
                        label_bin = 0.0
                        sample_weight = 1.0
                        neg_count += 1

                    loaded_features.append(feature_fixed.astype(np.float32, copy=False))
                    loaded_labels.append(float(label_bin))
                    loaded_weights.append(float(sample_weight))
            finally:
                store.close()

        def _load_all_stores() -> None:
            for split_name in ("train", "val"):
                _load_store_path(processed_dir / split_name)

            if self.users_hard_negs_dir is not None:
                _load_store_path(Path(self.users_hard_negs_dir))

        if self.console:
            with self.console.status("Loading tuning data..."):
                _load_all_stores()
        else:
            _load_all_stores()

        if loaded_features:
            features = np.stack(loaded_features, axis=0).astype(np.float32, copy=False)
            labels = np.asarray(loaded_labels, dtype=np.float32)
            weights = np.asarray(loaded_weights, dtype=np.float32)
        else:
            features = np.zeros((0, max_time_frames, 40), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.float32)
            weights = np.zeros((0,), dtype=np.float32)

        logger.info(f"Loaded {len(features)} samples ({pos_count} positive, {neg_count} negative, {hard_neg_count} hard_neg)")
        if self.console:
            self.console.print(f"[green]✓[/] Loaded {len(features)} samples ([bold]{pos_count}[/] positive, [bold]{neg_count}[/] negative, [bold]{hard_neg_count}[/] hard_neg)")
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

    def _make_knob(self, knob_name: str) -> Knob:
        knobs: dict[str, type[Knob]] = {
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

    def _evaluate_candidate(self, model, search_eval_partition: tuple, fold_indices: list | None = None) -> TuneMetrics:
        import tensorflow as tf

        features = np.asarray(search_eval_partition[0])
        labels_np = np.asarray(search_eval_partition[1]).reshape(-1).astype(np.float32, copy=False)
        if features.size == 0 or labels_np.size == 0:
            return TuneMetrics()

        batch_size = 512
        preds: list[np.ndarray] = []
        n_samples = int(features.shape[0])
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_features = tf.cast(features[start:end], tf.float32)
            batch_pred = model(batch_features, training=False)
            preds.append(np.asarray(batch_pred).reshape(-1))

        predictions = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)

        clip_duration_ms = float(self.config.get("hardware", {}).get("clip_duration_ms", 1000))
        negatives = int(np.sum(labels_np < 0.5))
        ambient_hours = float(negatives * clip_duration_ms / (3600.0 * 1000.0))

        optimizer = ThresholdOptimizer()
        target_fah = float(self.config.get("auto_tuning", {}).get("target_fah", 0.3))
        target_recall = float(self.config.get("auto_tuning", {}).get("target_recall", 0.92))
        _threshold_f32, _threshold_u8, metrics = optimizer.optimize(
            y_true=labels_np,
            y_scores=predictions,
            ambient_duration_hours=ambient_hours,
            target_fah=target_fah,
            target_recall=target_recall,
            cv_folds=int(self.config.get("auto_tuning", {}).get("cv_folds", 5)),
            fold_indices=fold_indices,
        )
        return metrics

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
        lr_override: float | None = None,
        rng_seed: int = 42,
    ) -> dict:
        """Simplified gradient burst (no SAM/SWA/Thompson/curriculum)."""
        import math

        import tensorflow as tf

        self._freeze_bn(model)
        losses: list[float] = []
        lr_min, lr_max = self._get_cfg("lr_range", (1e-7, 1e-4))
        train_features = np.asarray(search_train_partition[0])
        train_labels = np.asarray(search_train_partition[1]).reshape(-1)
        train_weights = np.asarray(search_train_partition[2]).reshape(-1)

        if train_features.shape[0] == 0:
            self._unfreeze_bn(model)
            return {
                "steps": int(n_steps),
                "final_loss": 0.0,
                "mean_loss": 0.0,
            }

        batch_size = 64
        rng = np.random.RandomState(rng_seed)
        indices = np.arange(train_features.shape[0], dtype=np.int64)
        rng.shuffle(indices)
        cursor = 0

        def _next_batch_indices() -> np.ndarray:
            nonlocal cursor, indices
            if cursor >= len(indices):
                indices = np.arange(train_features.shape[0], dtype=np.int64)
                rng.shuffle(indices)
                cursor = 0
            end = min(cursor + batch_size, len(indices))
            batch_idx = indices[cursor:end]
            cursor = end
            return batch_idx

        try:
            for step in range(int(n_steps)):
                # Use candidate-specific LR if provided, otherwise use cosine schedule
                if lr_override is not None:
                    cosine_lr = lr_override
                else:
                    cosine_lr = float(lr_min) + 0.5 * (float(lr_max) - float(lr_min)) * (1 + math.cos(math.pi * step / max(int(n_steps), 1)))
                if hasattr(optimizer, "learning_rate") and hasattr(optimizer.learning_rate, "assign"):
                    optimizer.learning_rate.assign(cosine_lr)

                batch_idx = _next_batch_indices()
                batch_features_np = train_features[batch_idx]
                batch_labels_np = train_labels[batch_idx]
                batch_weights_np = train_weights[batch_idx]

                batch_features = tf.cast(batch_features_np, tf.float32)
                batch_labels = tf.cast(batch_labels_np, tf.float32)
                batch_weights = tf.cast(batch_weights_np, tf.float32)

                with tf.GradientTape() as tape:
                    predictions = model(batch_features, training=True)
                    predictions = tf.squeeze(predictions, axis=-1)

                    smoothed_labels = batch_labels
                    if label_smoothing_var is not None:
                        eps = tf.cast(label_smoothing_var, tf.float32)
                        smoothed_labels = batch_labels * (1.0 - eps) + 0.5 * eps

                    bce = tf.keras.losses.binary_crossentropy(smoothed_labels, predictions, from_logits=False)
                    weighted_loss = bce * batch_weights
                    loss = tf.reduce_mean(weighted_loss)

                grads = tape.gradient(loss, model.trainable_variables)
                grad_var_pairs = [(g, v) for g, v in zip(grads, model.trainable_variables, strict=False) if g is not None]
                if grad_var_pairs:
                    optimizer.apply_gradients(grad_var_pairs)
                losses.append(float(loss.numpy()))
        finally:
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

        # Create label smoothing variable for TensorFlow optimization
        try:
            import tensorflow as tf

            label_smoothing_var = tf.Variable(initial_value=tf.constant(0.0, dtype=tf.float32), trainable=False)
        except (ImportError, AttributeError):
            # TensorFlow not available or version issue
            logger.warning("TensorFlow not available for label smoothing; feature disabled")
            label_smoothing_var = None
        except Exception as e:
            # Specific TF errors (e.g., memory allocation, device initialization)
            logger.warning(f"Failed to create TensorFlow label smoothing variable: {e}")
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

        progress_ctx = (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TextColumn("{task.fields[metrics]}"),
                console=self.console,
                disable=(not self.console.is_terminal) if self.console else True,
            )
            if self.console
            else nullcontext(None)
        )

        if self.console:
            self.console.print(Rule(title="Auto-Tuning", style="bold cyan"))

        with progress_ctx as progress:
            task_id = None
            if progress is not None:
                task_id = progress.add_task("Iteration 0/0", total=max_iterations, metrics="starting")

            for iteration in range(max_iterations):
                current_knob_name = knob_cycle.current()
                # Instantiate the knob object for the current knob name
                knob = self._make_knob(current_knob_name)

                if progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        description=f"Iteration {iteration + 1}/{max_iterations} • knob={current_knob_name}",
                        metrics=f"candidates=0/{len(population.candidates)}",
                    )

                for candidate_idx, candidate in enumerate(population.candidates, start=1):
                    if progress is not None and task_id is not None:
                        progress.update(
                            task_id,
                            description=(f"Iteration {iteration + 1}/{max_iterations} • candidate {candidate_idx}/{len(population.candidates)}"),
                            metrics=f"knob={current_knob_name}",
                        )

                    candidate.restore_state(model)
                    knob.apply(model, candidate, self.auto_tuning_config)

                    # Bug 3 fix: Sync label_smoothing_var from candidate._label_smoothing
                    if label_smoothing_var is not None:
                        ls_val = float(getattr(candidate, "_label_smoothing", 0.0))
                        label_smoothing_var.assign(ls_val)

                    # Get candidate-specific LR if set by LR knob
                    candidate_lr = getattr(candidate, "_sampled_lr", None)

                    burst_result = self._run_burst(
                        model,
                        optimizer,
                        search_train_partition=partition["search_train"],
                        n_steps=burst_steps,
                        label_smoothing_var=label_smoothing_var,
                        lr_override=candidate_lr,
                        rng_seed=42 + iteration * 100 + candidate_idx,
                    )

                    if progress is not None and task_id is not None:
                        progress.update(
                            task_id,
                            description=(f"Iteration {iteration + 1}/{max_iterations} • candidate {candidate_idx}/{len(population.candidates)}"),
                            metrics=f"knob={current_knob_name}",
                        )
                        progress.update(
                            task_id,
                            description=(
                                f"Iteration {iteration + 1}/{max_iterations} • candidate {candidate_idx}/{len(population.candidates)} • loss={float(burst_result.get('final_loss', 0.0)):.4f}"
                            ),
                            metrics=f"knob={current_knob_name}",
                        )

                    # Must evaluate on search_eval (not search_train)
                    metrics = self._ensure_tune_metrics(self._evaluate_candidate(model, partition["search_eval"], fold_indices=partition.get("fold_indices")))
                    candidate.metrics = metrics
                    pareto.try_add(metrics, candidate.id)
                    candidate.save_state(model)

                knob_cycle.advance()

                if exploit_every > 0 and (iteration + 1) % exploit_every == 0:
                    population.exploit_explore(model)

                frontier = pareto.get_frontier_points()
                hv = compute_hypervolume([(p["fah"], p["recall"]) for p in frontier], hyper_ref)
                hypervolume_history.append(float(hv))

                # Dashboard updates — methods now print internally
                if self.console:
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

                iter_best_entry = pareto.get_best(
                    target_fah=float(self.config.get("auto_tuning", {}).get("target_fah", 0.3)), target_recall=float(self.config.get("auto_tuning", {}).get("target_recall", 0.92))
                )
                if iter_best_entry is not None:
                    iter_best_metrics, _ = iter_best_entry
                else:
                    iter_best_metrics = TuneMetrics()

                if self.console:
                    self.console.print(
                        "[cyan]Iteration {}/{}[/] • hv=[bold]{:.6f}[/] • patience={}/{} • best(fah={:.6f}, recall={:.6f}, auc_pr={:.6f})".format(
                            iteration + 1,
                            max_iterations,
                            float(hv),
                            int(no_improve_count),
                            int(max_no_improve),
                            float(iter_best_metrics.fah),
                            float(iter_best_metrics.recall),
                            float(iter_best_metrics.auc_pr),
                        )
                    )

                if progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        completed=iteration + 1,
                        description=f"Iteration {iteration + 1}/{max_iterations} • knob={current_knob_name}",
                        metrics=f"hv={float(hv):.6f} patience={int(no_improve_count)}/{int(max_no_improve)}",
                    )

                # Early stopping: stop when patience is exhausted
                if no_improve_count >= max_no_improve:
                    break

        best_entry = pareto.get_best(target_fah=float(self.config.get("auto_tuning", {}).get("target_fah", 0.3)), target_recall=float(self.config.get("auto_tuning", {}).get("target_recall", 0.92)))
        if best_entry is not None:
            best_metrics, _best_id = best_entry
        else:
            best_metrics = TuneMetrics()

        best_candidate = None
        for c in population.candidates:
            if c.metrics is best_metrics:
                best_candidate = c
                break

        if best_candidate is not None:
            best_candidate.restore_state(model)

        if self.console:
            self.console.print(Rule(title="Confirmation Phase", style="bold yellow"))
        confirm_metrics = self._ensure_tune_metrics(self._evaluate_candidate(model, partition["confirm"]))
        logger.info(
            "Confirmation metrics: fah=%.6f recall=%.6f auc_pr=%.6f threshold=%.6f (u8=%d)",
            float(confirm_metrics.fah),
            float(confirm_metrics.recall),
            float(confirm_metrics.auc_pr),
            float(confirm_metrics.threshold),
            int(confirm_metrics.threshold_uint8),
        )
        if self.console:
            self.console.print(
                "[yellow]confirm[/] fah=[bold]{:.6f}[/] recall=[bold]{:.6f}[/] auc_pr=[bold]{:.6f}[/] threshold=[bold]{:.6f}[/] (u8=[bold]{}[/])".format(
                    float(confirm_metrics.fah),
                    float(confirm_metrics.recall),
                    float(confirm_metrics.auc_pr),
                    float(confirm_metrics.threshold),
                    int(confirm_metrics.threshold_uint8),
                )
            )

        output_dir = Path(self._get_cfg("output_dir", self.config.get("auto_tuning", {}).get("output_dir", "./tuning_output")))
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = output_dir / "tuned_weights.weights.h5"
        model.save_weights(str(best_path))
        if self.console:
            self.console.print(f"[green]✓[/] Saved tuned weights → {best_path}")

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

        artifacts_output_dir: str = self._get_cfg("output_dir", self.config.get("auto_tuning", {}).get("output_dir", "./tuning_output"))
        save_artifacts(
            output_dir=artifacts_output_dir,
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
            "best_checkpoint": str(best_path),
            "hypervolume_history": hypervolume_history,
            "iterations_completed": iterations_completed,
            "pareto_frontier": pareto.get_frontier_points(),
            "confirm_fah": float(confirm_metrics.fah),
            "confirm_recall": float(confirm_metrics.recall),
            "confirm_auc_pr": float(confirm_metrics.auc_pr),
            "confirm_threshold": float(confirm_metrics.threshold),
            "confirm_threshold_uint8": int(confirm_metrics.threshold_uint8),
            "target_met": bool(
                confirm_metrics.fah < self.config.get("auto_tuning", {}).get("target_fah", float("inf")) and confirm_metrics.recall > self.config.get("auto_tuning", {}).get("target_recall", 0.0)
            ),
        }
