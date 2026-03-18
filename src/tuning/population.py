"""Population primitives for micro-step auto-tuning."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

from src.tuning.metrics import TuneMetrics


@dataclass
class Candidate:
    """State container for one tuning candidate."""

    id: str
    weights_bytes: Optional[bytes] = None
    optimizer_state_bytes: Optional[bytes] = None
    batchnorm_state: Optional[dict] = None
    temperature: float = 1.0
    metrics: Optional[TuneMetrics] = None
    knob_history: list[str] = field(default_factory=list)

    def save_state(self, model) -> None:
        """Serialize all model weights via get_weights() using numpy (pickle-free)."""
        weights = model.get_weights()
        buffer = io.BytesIO()
        np.savez(buffer, *weights)
        self.weights_bytes = buffer.getvalue()

    def restore_state(self, model) -> None:
        """Restore serialized model weights via set_weights() (pickle-free)."""
        if self.weights_bytes is None:
            return
        buffer = io.BytesIO(self.weights_bytes)
        with np.load(buffer, allow_pickle=False) as data:
            weights = [data[f"arr_{i}"] for i in range(len(data.files))]
        model.set_weights(weights)


class Population:
    """Manages a candidate pool for micro-step tuning."""

    def __init__(self, model=None, size: int = 4, candidates: Optional[list[Candidate]] = None):
        if candidates is not None:
            self.candidates = list(candidates)
            return

        self.candidates = [Candidate(id=f"candidate_{i}") for i in range(size)]
        if model is not None:
            for candidate in self.candidates:
                candidate.save_state(model)

    @staticmethod
    def _metric_key_for_best(candidate: Candidate) -> tuple[float, float, float]:
        m = candidate.metrics
        if m is None:
            return (float("inf"), float("inf"), float("inf"))
        return (float(m.fah), -float(m.recall), -float(m.auc_pr))

    @staticmethod
    def _metric_key_for_worst(candidate: Candidate) -> tuple[float, float, float]:
        m = candidate.metrics
        if m is None:
            return (float("inf"), float("inf"), float("inf"))
        return (-float(m.fah), float(m.recall), float(m.auc_pr))

    def get_best(self) -> Candidate:
        """Return best candidate: low FAH, then high recall, then high AUC-PR."""
        if not self.candidates:
            raise ValueError("Population has no candidates")
        return min(self.candidates, key=self._metric_key_for_best)

    def get_worst(self) -> Candidate:
        """Return worst candidate by inverse ordering of get_best()."""
        if not self.candidates:
            raise ValueError("Population has no candidates")
        return min(self.candidates, key=self._metric_key_for_worst)

    def exploit_explore(self, model, perturbation_scale: float = 0.01) -> None:
        """Clone best weights to worst candidate and perturb trainable-only tensors."""
        best = self.get_best()
        worst = self.get_worst()

        if best.weights_bytes is None:
            best.save_state(model)
        if best.weights_bytes is None:
            raise ValueError("Best candidate has no serialized weights")

        best_buffer = io.BytesIO(best.weights_bytes)
        with np.load(best_buffer, allow_pickle=False) as data:
            best_weights = [np.array(data[f"arr_{i}"], copy=True) for i in range(len(data.files))]
        worst_weights = [np.array(w, copy=True) for w in best_weights]

        # Build trainable weight index set, preferring explicit trainable flags.
        trainable_indices: set[int] = set()
        model_weights = list(getattr(model, "weights", []) or [])
        if model_weights and len(model_weights) == len(worst_weights):
            trainable_indices = {i for i, w in enumerate(model_weights) if bool(getattr(w, "trainable", False))}

        # Fallback: map trainable_variables by name onto weights by name.
        if not trainable_indices and model_weights:
            weight_index_by_name: dict[str, int] = {str(getattr(w, "name", f"w_{i}")): i for i, w in enumerate(model_weights)}
            for tv in list(getattr(model, "trainable_variables", []) or []):
                idx = weight_index_by_name.get(str(getattr(tv, "name", "")))
                if idx is not None:
                    trainable_indices.add(idx)

        # Last-resort fallback for simple mocks: perturb first N tensors.
        if not trainable_indices:
            n_trainable = len(list(getattr(model, "trainable_variables", []) or []))
            trainable_indices = set(range(min(n_trainable, len(worst_weights))))

        rng = np.random.RandomState(42)
        for idx in sorted(trainable_indices):
            weight = np.asarray(worst_weights[idx])
            noise = rng.normal(loc=0.0, scale=perturbation_scale, size=weight.shape).astype(weight.dtype, copy=False)
            worst_weights[idx] = weight + noise

        worst_buffer = io.BytesIO()
        np.savez(worst_buffer, *worst_weights)
        worst.weights_bytes = worst_buffer.getvalue()
        model.set_weights(worst_weights)


def _get_config_value(config: Any, section: str, key: str, default: Any) -> Any:
    if config is None:
        return default

    # Dataclass/object access
    if hasattr(config, section):
        section_obj = getattr(config, section)
        if hasattr(section_obj, key):
            return getattr(section_obj, key)

    # Dict access
    if isinstance(config, dict):
        section_obj = config.get(section, {})
        if isinstance(section_obj, dict):
            return section_obj.get(key, default)

    return default


def _unpack_dataset(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if isinstance(dataset, dict):
        features = np.asarray(dataset["features"])
        labels = np.asarray(dataset["labels"])
        weights = np.asarray(dataset.get("weights", np.ones(len(labels), dtype=np.float32)))
        group_ids = dataset.get("speaker_id", dataset.get("group", dataset.get("group_ids")))
        return features, labels, weights, None if group_ids is None else np.asarray(group_ids)

    if isinstance(dataset, tuple):
        if len(dataset) == 3:
            features, labels, weights = dataset
            return np.asarray(features), np.asarray(labels), np.asarray(weights), None
        if len(dataset) >= 4:
            features, labels, weights, group_ids = dataset[:4]
            return np.asarray(features), np.asarray(labels), np.asarray(weights), None if group_ids is None else np.asarray(group_ids)

    # Generic object with attributes
    obj = cast(Any, dataset)
    features = np.asarray(obj.features)
    labels = np.asarray(obj.labels)
    weights_attr = obj.weights if hasattr(obj, "weights") else np.ones(len(labels), dtype=np.float32)
    weights = np.asarray(weights_attr)
    if hasattr(obj, "speaker_id"):
        group_ids = obj.speaker_id
    elif hasattr(obj, "group"):
        group_ids = obj.group
    elif hasattr(obj, "group_ids"):
        group_ids = obj.group_ids
    else:
        group_ids = None
    return features, labels, weights, None if group_ids is None else np.asarray(group_ids)


def partition_data(dataset, config, expert_config=None):
    """Partition data into calibration/search_train/search_eval/confirm/representative."""
    _ = expert_config
    features, labels, weights, group_ids = _unpack_dataset(dataset)
    n = len(labels)
    if n < 4:
        raise ValueError(f"Dataset too small for partitioning: need at least 4 samples, got {n}")

    confirmation_fraction = float(_get_config_value(config, "auto_tuning", "confirmation_fraction", 0.40))
    search_eval_fraction = float(_get_config_value(config, "auto_tuning", "search_eval_fraction", 0.30))
    cv_folds = int(_get_config_value(config, "auto_tuning", "cv_folds", 3))
    seed = int(_get_config_value(config, "training", "split_seed", 42))

    n_cal = max(1, int(n * 0.15))
    n_repr = max(1, int(n * 0.05))
    n_confirm = max(1, int(n * confirmation_fraction))
    n_search = n - n_cal - n_repr - n_confirm

    while n_search < 1 and n_confirm > 1:
        n_confirm -= 1
        n_search = n - n_cal - n_repr - n_confirm
    while n_search < 1 and n_repr > 1:
        n_repr -= 1
        n_search = n - n_cal - n_repr - n_confirm
    while n_search < 1 and n_cal > 1:
        n_cal -= 1
        n_search = n - n_cal - n_repr - n_confirm
    if n_search < 1:
        raise ValueError(f"Dataset too small to partition: n={n}, need at least 4 samples for all partitions")

    rng = np.random.RandomState(seed)
    indices = np.arange(n)

    use_group_partition = False
    cal_idx = np.array([], dtype=np.int64)
    search_idx = np.array([], dtype=np.int64)
    confirm_idx = np.array([], dtype=np.int64)
    representative_idx = np.array([], dtype=np.int64)
    if group_ids is not None and len(group_ids) == n:
        non_unknown = group_ids[group_ids != "unknown"]
        use_group_partition = len(np.unique(non_unknown)) >= 2

    if use_group_partition:
        assert group_ids is not None
        target_sizes = {"cal": n_cal, "search": n_search, "confirm": n_confirm, "representative": n_repr}
        bins: dict[str, list[int]] = {k: [] for k in target_sizes}
        group_to_indices: dict[str, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            group_to_indices.setdefault(str(gid), []).append(int(idx))

        groups = list(group_to_indices.keys())
        rng.shuffle(groups)
        groups.sort(key=lambda g: len(group_to_indices[g]), reverse=True)

        for gid in groups:
            gidx = group_to_indices[gid]
            candidate_bins: list[tuple[float, int, str]] = []
            for b in target_sizes:
                target = max(target_sizes[b], 1)
                ratio = len(bins[b]) / target
                candidate_bins.append((ratio, len(bins[b]), b))
            candidate_bins.sort()
            chosen = candidate_bins[0][2]
            bins[chosen].extend(gidx)

        cal_idx = np.array(bins["cal"], dtype=np.int64)
        search_idx = np.array(bins["search"], dtype=np.int64)
        confirm_idx = np.array(bins["confirm"], dtype=np.int64)
        representative_idx = np.array(bins["representative"], dtype=np.int64)

        if min(len(cal_idx), len(search_idx), len(confirm_idx), len(representative_idx)) == 0:
            use_group_partition = False

    if not use_group_partition:
        indices = rng.permutation(indices)
        cal_idx = indices[:n_cal]
        search_idx = indices[n_cal : n_cal + n_search]
        confirm_idx = indices[n_cal + n_search : n_cal + n_search + n_confirm]
        representative_idx = indices[n_cal + n_search + n_confirm :]

    if len(search_idx) < 2:
        raise ValueError(f"Need at least 2 search samples for train/eval split, got {len(search_idx)}")

    n_search_eval = max(1, min(len(search_idx) - 1, int(round(len(search_idx) * search_eval_fraction))))
    n_search_train = len(search_idx) - n_search_eval

    if use_group_partition and group_ids is not None:
        search_group_to_indices: dict[str, list[int]] = {}
        for idx_val in search_idx:
            search_group_to_indices.setdefault(str(group_ids[idx_val]), []).append(int(idx_val))

        search_groups = list(search_group_to_indices.keys())
        rng.shuffle(search_groups)
        search_groups.sort(key=lambda g: len(search_group_to_indices[g]), reverse=True)

        bins = {"train": [], "eval": []}
        targets = {"train": n_search_train, "eval": n_search_eval}

        for gid in search_groups:
            gidx = search_group_to_indices[gid]
            train_ratio = len(bins["train"]) / max(targets["train"], 1)
            eval_ratio = len(bins["eval"]) / max(targets["eval"], 1)
            if train_ratio <= eval_ratio:
                bins["train"].extend(gidx)
            else:
                bins["eval"].extend(gidx)

        search_train_idx = np.array(bins["train"], dtype=np.int64)
        search_eval_idx = np.array(bins["eval"], dtype=np.int64)
        if len(search_train_idx) == 0 or len(search_eval_idx) == 0:
            perm = rng.permutation(len(search_idx))
            search_train_idx = search_idx[perm[:n_search_train]]
            search_eval_idx = search_idx[perm[n_search_train:]]
    else:
        perm = rng.permutation(len(search_idx))
        search_train_idx = search_idx[perm[:n_search_train]]
        search_eval_idx = search_idx[perm[n_search_train:]]

    fold_indices = []
    fold_size = len(search_eval_idx) // max(cv_folds, 1)
    for i in range(cv_folds):
        start = i * fold_size
        end = start + fold_size if i < cv_folds - 1 else len(search_eval_idx)
        fold_indices.append(np.arange(start, end))

    return {
        "cal": (features[cal_idx], labels[cal_idx], weights[cal_idx], cal_idx),
        "search_train": (
            features[search_train_idx],
            labels[search_train_idx],
            weights[search_train_idx],
            search_train_idx,
        ),
        "search_eval": (
            features[search_eval_idx],
            labels[search_eval_idx],
            weights[search_eval_idx],
            search_eval_idx,
        ),
        "confirm": (
            features[confirm_idx],
            labels[confirm_idx],
            weights[confirm_idx],
            confirm_idx,
        ),
        "representative": (
            features[representative_idx],
            labels[representative_idx],
            weights[representative_idx],
            representative_idx,
        ),
        "fold_indices": fold_indices,
    }
