"""
Shared utilities for TFLite model analysis.
"""

import numpy as np
import tensorflow as tf


def estimate_tensor_arena_size(interpreter: tf.lite.Interpreter, margin: float = 1.3) -> int:
    """Estimate TFLite tensor arena size from model structure.

    This function computes the total tensor memory and applies a margin to estimate
    the arena size needed by the TFLite interpreter.

    Args:
        interpreter: TFLite interpreter instance
        margin: Safety margin multiplier (default 1.3 = 30% overhead)

    Returns:
        Estimated arena size in bytes

    Note:
        This matches the logic from verification.py's _estimate_tensor_arena_size()
        and is shared with manifest.py for consistent estimates.
    """
    total_memory = 0
    for tensor in interpreter.get_tensor_details():
        shape = tensor.get("shape", [])
        dtype = tensor.get("dtype")

        # Get element size based on dtype (using numpy dtypes)
        if dtype == np.float32:
            elem_size = 4
        elif dtype == np.float16:
            elem_size = 2
        elif dtype == np.bytes_ or dtype == np.object_:
            # String size is variable; use a conservative 32-byte estimate
            elem_size = 32
        elif dtype in (np.int8, np.uint8):
            elem_size = 1
        elif dtype == np.int32:
            elem_size = 4
        elif dtype == np.int64:
            elem_size = 8
        else:
            elem_size = 4  # Default assumption

        num_elements = 1
        for dim in shape:
            if dim == -1:
                # Dynamic dimension: use a conservative estimate of 1 and warn
                # Note: Warning is logged by caller if needed
                d = 1
            elif dim == 0:
                d = 1
            else:
                d = abs(dim)
            num_elements *= d

        total_memory += num_elements * elem_size

    return int(total_memory * margin)
