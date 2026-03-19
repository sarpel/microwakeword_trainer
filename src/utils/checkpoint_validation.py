"""Checkpoint validation utilities for ensuring compatibility and consistency.

This module provides validation functions for:
1. Checkpoint compatibility - verifying checkpoint matches model architecture
2. EMA weight state consistency - ensuring EMA weights are valid and consistent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass
class CheckpointValidationResult:
    """Result of checkpoint validation.

    Attributes:
        is_valid: Whether the checkpoint passed all validation checks
        errors: List of validation error messages
        warnings: List of validation warning messages
        checkpoint_info: Dictionary with checkpoint metadata
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checkpoint_info: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "CheckpointValidationResult") -> "CheckpointValidationResult":
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.checkpoint_info.update(other.checkpoint_info)
        self.is_valid = self.is_valid and other.is_valid
        return self


@dataclass
class EMWeightValidationResult:
    """Result of EMA weight validation.

    Attributes:
        is_valid: Whether EMA weights passed validation
        errors: List of validation error messages
        warnings: List of validation warning messages
        ema_stats: Dictionary with EMA statistics
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ema_stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


def validate_checkpoint_compatibility(
    model: tf.keras.Model,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> CheckpointValidationResult:
    """Validate that a checkpoint is compatible with the current model architecture.

    This function checks:
    1. Checkpoint file exists and is readable
    2. Weight shapes match the model architecture
    3. Layer names and structure are compatible
    4. No missing or extra weights

    Args:
        model: The model to validate against
        checkpoint_path: Path to the checkpoint file
        strict: If True, fail on any mismatch. If False, allow partial compatibility.

    Returns:
        CheckpointValidationResult with validation status and details
    """
    result = CheckpointValidationResult()
    checkpoint_path = Path(checkpoint_path)

    # Check file exists
    if not checkpoint_path.exists():
        result.add_error(f"Checkpoint file not found: {checkpoint_path}")
        return result

    result.checkpoint_info["checkpoint_path"] = str(checkpoint_path)
    result.checkpoint_info["checkpoint_exists"] = True

    # Try to load checkpoint metadata without loading full weights
    try:
        # Create a temporary model copy to test loading
        current_weights = model.get_weights()
        result.checkpoint_info["current_weight_count"] = len(current_weights)
        result.checkpoint_info["current_model_layers"] = len(model.layers)

        # Try to load weights into a fresh model structure
        # This will raise an error if shapes don't match
        try:
            # Use load_weights with skip_mismatch in non-strict mode
            if strict:
                model.load_weights(str(checkpoint_path))
            else:
                model.load_weights(str(checkpoint_path), skip_mismatch=True)

            # Verify shapes match
            loaded_weights = model.get_weights()
            result.checkpoint_info["loaded_weight_count"] = len(loaded_weights)

            # Check weight count consistency
            if len(loaded_weights) != len(current_weights):
                if strict:
                    result.add_error(f"Weight count mismatch: checkpoint has {len(loaded_weights)} weights, model expects {len(current_weights)}")
                else:
                    result.add_warning(f"Weight count mismatch: checkpoint has {len(loaded_weights)} weights, model expects {len(current_weights)}")

            # Check individual weight shapes
            shape_mismatches = []
            for i, (current, loaded) in enumerate(zip(current_weights, loaded_weights, strict=False)):
                if current.shape != loaded.shape:
                    shape_mismatches.append(f"Weight {i}: expected {current.shape}, got {loaded.shape}")

            if shape_mismatches:
                mismatch_msg = "; ".join(shape_mismatches[:5])  # Limit to first 5
                if len(shape_mismatches) > 5:
                    mismatch_msg += f"; and {len(shape_mismatches) - 5} more"

                if strict:
                    result.add_error(f"Shape mismatches: {mismatch_msg}")
                else:
                    result.add_warning(f"Shape mismatches: {mismatch_msg}")

            # Check for NaN or Inf values in weights
            nan_inf_issues = []
            for i, w in enumerate(loaded_weights):
                if np.isnan(w).any():
                    nan_inf_issues.append(f"Weight {i}: contains NaN values")
                if np.isinf(w).any():
                    nan_inf_issues.append(f"Weight {i}: contains Inf values")

            if nan_inf_issues:
                issues_msg = "; ".join(nan_inf_issues[:5])
                if len(nan_inf_issues) > 5:
                    issues_msg += f"; and {len(nan_inf_issues) - 5} more"
                result.add_error(f"Invalid weight values: {issues_msg}")

            # Restore original weights
            model.set_weights(current_weights)

        except (ValueError, tf.errors.InvalidArgumentError) as e:
            error_msg = str(e)
            if "shape" in error_msg.lower():
                result.add_error(f"Shape incompatibility: {error_msg}")
            elif "variable" in error_msg.lower():
                result.add_error(f"Variable mismatch: {error_msg}")
            else:
                result.add_error(f"Failed to load checkpoint: {error_msg}")

    except Exception as e:
        result.add_error(f"Unexpected error during checkpoint validation: {e}")

    return result


def validate_ema_weight_consistency(
    model: tf.keras.Model,
    saved_training_weights: Sequence[np.ndarray] | None,
    ema_enabled: bool,
) -> EMWeightValidationResult:
    """Validate EMA weight state consistency.

    This function checks:
    1. EMA is properly configured in optimizer
    2. Training weights exist and are valid
    3. Training weights match model architecture
    4. Weight swap can be performed correctly

    Args:
        model: The model to validate
        saved_training_weights: Previously saved training weights (or None)
        ema_enabled: Whether EMA is enabled

    Returns:
        EMWeightValidationResult with validation status and details
    """
    result = EMWeightValidationResult()

    # If EMA not enabled, nothing to validate
    if not ema_enabled:
        result.ema_stats["ema_enabled"] = False
        return result

    result.ema_stats["ema_enabled"] = True

    # Check model exists
    if model is None:
        result.add_error("Model is None - cannot validate EMA weights")
        return result

    # Check optimizer exists
    optimizer = model.optimizer
    if optimizer is None:
        result.add_error("Model optimizer is not set - EMA requires optimizer")
        return result

    # Check if optimizer has EMA support
    if not hasattr(optimizer, "use_ema"):
        result.add_error("Optimizer does not support EMA (missing use_ema attribute)")
        return result

    result.ema_stats["optimizer_has_ema"] = True
    result.ema_stats["optimizer_ema_enabled"] = bool(getattr(optimizer, "use_ema", False))

    # Check if we have saved training weights
    if saved_training_weights is None:
        # This is okay if we haven't swapped yet
        result.ema_stats["training_weights_saved"] = False
        result.add_warning("No saved training weights - EMA swap has not been performed yet")
        return result

    result.ema_stats["training_weights_saved"] = True
    result.ema_stats["saved_weight_count"] = len(saved_training_weights)

    # Validate saved weights
    current_weights = model.get_weights()
    result.ema_stats["current_weight_count"] = len(current_weights)

    # Check weight count consistency
    if len(saved_training_weights) != len(current_weights):
        result.add_error(f"Training weight count mismatch: saved {len(saved_training_weights)} weights, model has {len(current_weights)}")
        return result

    # Check individual weight shapes
    shape_mismatches = []
    for i, (current, saved) in enumerate(zip(current_weights, saved_training_weights, strict=False)):
        if current.shape != saved.shape:
            shape_mismatches.append(f"Weight {i}: model has {current.shape}, saved is {saved.shape}")

    if shape_mismatches:
        mismatch_msg = "; ".join(shape_mismatches[:5])
        if len(shape_mismatches) > 5:
            mismatch_msg += f"; and {len(shape_mismatches) - 5} more"
        result.add_error(f"Training weight shape mismatches: {mismatch_msg}")

    # Check for NaN or Inf in saved weights
    nan_inf_issues = []
    for i, w in enumerate(saved_training_weights):
        if np.isnan(w).any():
            nan_inf_issues.append(f"Saved weight {i}: contains NaN values")
        if np.isinf(w).any():
            nan_inf_issues.append(f"Saved weight {i}: contains Inf values")

    if nan_inf_issues:
        issues_msg = "; ".join(nan_inf_issues[:5])
        if len(nan_inf_issues) > 5:
            issues_msg += f"; and {len(nan_inf_issues) - 5} more"
        result.add_error(f"Invalid saved weight values: {issues_msg}")

    # Check that saved weights differ from current (they should if EMA is active)
    if result.is_valid:
        weight_differences = []
        for _i, (current, saved) in enumerate(zip(current_weights, saved_training_weights, strict=False)):
            diff = np.abs(current - saved).mean()
            weight_differences.append(diff)

        avg_diff = np.mean(weight_differences)
        result.ema_stats["average_weight_difference"] = float(avg_diff)
        result.ema_stats["max_weight_difference"] = float(np.max(weight_differences))

        if avg_diff < 1e-10:
            result.add_warning("Training weights are nearly identical to current weights - EMA may not be active or weights were not swapped")

    return result


def validate_checkpoint_before_loading(
    model: tf.keras.Model,
    checkpoint_path: str | Path,
) -> tuple[bool, str]:
    """Quick validation before attempting to load a checkpoint.

    Returns:
        Tuple of (is_valid, error_message)
    """
    result = validate_checkpoint_compatibility(model, checkpoint_path, strict=True)

    if result.is_valid:
        return True, ""

    error_msg = "Checkpoint validation failed:\n" + "\n".join(f"  - {error}" for error in result.errors)
    return False, error_msg


def validate_ema_state_before_swap(
    model: tf.keras.Model,
    saved_training_weights: Sequence[np.ndarray] | None,
    ema_enabled: bool,
) -> tuple[bool, str]:
    """Quick validation before EMA weight swap.

    Returns:
        Tuple of (is_valid, error_message)
    """
    result = validate_ema_weight_consistency(model, saved_training_weights, ema_enabled)

    if result.is_valid:
        return True, ""

    error_msg = "EMA state validation failed:\n" + "\n".join(f"  - {error}" for error in result.errors)
    return False, error_msg


def log_validation_result(
    result: CheckpointValidationResult | EMWeightValidationResult,
    logger_instance: logging.Logger | None = None,
    prefix: str = "",
) -> None:
    """Log validation results at appropriate levels.

    Args:
        result: Validation result to log
        logger_instance: Logger to use (defaults to module logger)
        prefix: Prefix string for log messages
    """
    log = logger_instance or logger

    if prefix:
        prefix = f"{prefix}: "

    # Log info/stats
    if hasattr(result, "checkpoint_info") and result.checkpoint_info:
        for key, value in result.checkpoint_info.items():
            log.debug(f"{prefix}Checkpoint info - {key}: {value}")

    if hasattr(result, "ema_stats") and result.ema_stats:
        for key, value in result.ema_stats.items():
            log.debug(f"{prefix}EMA stats - {key}: {value}")
    # Log warnings
    for warning in result.warnings:
        log.warning(f"{prefix}{warning}")

    # Log errors
    for error in result.errors:
        log.error(f"{prefix}{error}")

    # Log final status
    if result.is_valid:
        if result.warnings:
            log.info(f"{prefix}Validation passed with warnings")
        else:
            log.debug(f"{prefix}Validation passed")
    else:
        log.error(f"{prefix}Validation failed with {len(result.errors)} error(s)")
