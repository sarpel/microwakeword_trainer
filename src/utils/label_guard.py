"""Centralized label validation and manipulation utilities.

Prevents the ``y_true == 0`` bug by providing canonical functions for:
- Label constants
- Label validation
- Safe binarization (all non-positive -> 0)
- Distribution checking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
LABEL_HARD_NEGATIVE = 2
VALID_LABELS = {LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_HARD_NEGATIVE}


@dataclass
class LabelValidationResult:
    """Result object for label validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def is_negative(labels: Any) -> np.ndarray | tf.Tensor:
    """Return mask for all negative samples (everything except positive label)."""
    if tf.is_tensor(labels):
        return tf.not_equal(labels, LABEL_POSITIVE)
    labels_np = np.asarray(labels)
    return labels_np != LABEL_POSITIVE


def is_positive(labels: Any) -> np.ndarray | tf.Tensor:
    """Return mask for positive samples (label == 1)."""
    if tf.is_tensor(labels):
        return tf.equal(labels, LABEL_POSITIVE)
    labels_np = np.asarray(labels)
    return labels_np == LABEL_POSITIVE


def binarize_labels(labels: Any) -> np.ndarray | tf.Tensor:
    """Map tri-state labels to binary labels for BCE: {0,2}->0 and {1}->1."""
    if tf.is_tensor(labels):
        return tf.cast(is_positive(labels), labels.dtype)
    labels_np = np.asarray(labels)
    return (labels_np == LABEL_POSITIVE).astype(labels_np.dtype)


def _to_numpy(labels: Any) -> np.ndarray:
    if tf.is_tensor(labels):
        return labels.numpy()
    return np.asarray(labels)


def validate_labels(labels: Any, context: str = "") -> LabelValidationResult:
    """Validate labels and return result with errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    context_prefix = f"[{context}] " if context else ""

    labels_np = _to_numpy(labels).reshape(-1)

    unique_values = set(np.unique(labels_np).tolist())
    invalid_values = sorted(value for value in unique_values if value not in VALID_LABELS)
    if invalid_values:
        errors.append(f"{context_prefix}Invalid label values found: {invalid_values}. Valid labels: {sorted(VALID_LABELS)}")

    positive_count = int(np.sum(labels_np == LABEL_POSITIVE))
    negative_count = int(np.sum(labels_np == LABEL_NEGATIVE))
    hard_negative_count = int(np.sum(labels_np == LABEL_HARD_NEGATIVE))
    all_negative_count = negative_count + hard_negative_count

    if positive_count == 0:
        errors.append(f"{context_prefix}No positive samples found (label={LABEL_POSITIVE}).")

    if all_negative_count == 0:
        warnings.append(f"{context_prefix}No negative samples found (labels={LABEL_NEGATIVE}/{LABEL_HARD_NEGATIVE}).")

    if all_negative_count > 0:
        hard_negative_ratio = hard_negative_count / all_negative_count
        if hard_negative_ratio < 0.01:
            warnings.append(f"{context_prefix}Hard negative ratio is unusually low: {hard_negative_ratio:.2%} (<1%).")
        elif hard_negative_ratio > 0.80:
            warnings.append(f"{context_prefix}Hard negative ratio is unusually high: {hard_negative_ratio:.2%} (>80%).")

    return LabelValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def check_label_distribution(labels: Any, context: str = "") -> dict[str, float | int]:
    """Return label distribution stats and log a concise summary."""
    labels_np = _to_numpy(labels).reshape(-1)
    total = int(labels_np.size)

    negative_count = int(np.sum(labels_np == LABEL_NEGATIVE))
    positive_count = int(np.sum(labels_np == LABEL_POSITIVE))
    hard_negative_count = int(np.sum(labels_np == LABEL_HARD_NEGATIVE))
    all_negative_count = negative_count + hard_negative_count

    def pct(count: int) -> float:
        if total == 0:
            return 0.0
        return 100.0 * count / total

    distribution: dict[str, float | int] = {
        "total": total,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "hard_negative_count": hard_negative_count,
        "all_negative_count": all_negative_count,
        "negative_pct": pct(negative_count),
        "positive_pct": pct(positive_count),
        "hard_negative_pct": pct(hard_negative_count),
        "all_negative_pct": pct(all_negative_count),
    }

    context_prefix = f"[{context}] " if context else ""
    logger.info(
        "%sLabel distribution: total=%d, neg=%d (%.2f%%), pos=%d (%.2f%%), hard_neg=%d (%.2f%%)",
        context_prefix,
        total,
        negative_count,
        distribution["negative_pct"],
        positive_count,
        distribution["positive_pct"],
        hard_negative_count,
        distribution["hard_negative_pct"],
    )

    return distribution


def assert_labels_valid(labels: Any, context: str = "") -> None:
    """Raise ValueError if labels are invalid for pipeline entry points."""
    result = validate_labels(labels, context=context)
    if not result.is_valid:
        joined = "; ".join(result.errors)
        raise ValueError(f"Label validation failed: {joined}")
