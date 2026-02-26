"""
Performance utilities for microwakeword_trainer v2.0
"""

import os
import psutil
import GPUtil
from typing import Any, Dict, Optional

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================


def get_system_info() -> Dict[str, Any]:
    """Get system resource information."""
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()

    info = {
        "cpu_count": cpu_count,
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
    }

    # GPU info
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            info["gpu"] = {
                "name": gpus[0].name,
                "memory_total_mb": gpus[0].memoryTotal,
                "memory_free_mb": gpus[0].memoryFree,
                "memory_used_mb": gpus[0].memoryUsed,
                "load_percent": gpus[0].load * 100,
            }
    except Exception:
        pass

    return info


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        gpus = GPUtil.getGPUs()
        return len(gpus) > 0
    except Exception:
        return False


# =============================================================================
# TENSORFLOW GPU CONFIGURATION
# =============================================================================


def configure_tensorflow_gpu(
    memory_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    device_id: Optional[int] = None,
) -> bool:
    """Configure TensorFlow GPU settings for optimal training performance.

    Args:
        memory_growth: Enable memory growth to avoid allocating all GPU memory
        memory_limit_mb: Optional memory limit in MB
        device_id: Specific GPU device ID to use

    Returns:
        True if GPU was configured successfully

    Raises:
        RuntimeError: If no GPU is available
    """
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError(
            "No GPU available. This project requires a CUDA-capable GPU for training."
        )

    if memory_growth and memory_limit_mb:
        raise ValueError(
            "memory_growth and memory_limit_mb are mutually exclusive in TensorFlow GPU config. "
            "Please provide only one."
        )

    try:
        if device_id is not None and 0 <= device_id < len(gpus):
            target_gpu = gpus[device_id]
        else:
            if device_id is not None:
                print(
                    f"Warning: device_id={device_id} out of range for {len(gpus)} GPUs. "
                    "Falling back to GPU 0."
                )
            target_gpu = gpus[0]

        if memory_growth:
            tf.config.experimental.set_memory_growth(target_gpu, True)

        if memory_limit_mb:
            tf.config.experimental.set_virtual_device_configuration(
                target_gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )
                ],
            )

        # Log configuration
        print(f"GPU configured: {target_gpu}")
        print(f"  Memory growth: {memory_growth}")
        if memory_limit_mb:
            print(f"  Memory limit: {memory_limit_mb} MB")

        return True

    except (RuntimeError, IndexError) as e:
        print(f"Error configuring GPU: {e}")
        return False


def configure_mixed_precision(enabled: bool = True) -> None:
    """Configure mixed precision (FP16) training for faster computation.

    Mixed precision uses FP16 for most operations while keeping FP32 for
    critical parts. This provides up to 2-3x speedup on modern GPUs with
    minimal accuracy impact.

    Args:
        enabled: Whether to enable mixed precision

    Note:
        Only effective on GPUs with Tensor Cores (Volta, Turing, Ampere, Ada)
    """
    import tensorflow as tf

    if enabled:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled (FP16)")
    else:
        policy = tf.keras.mixed_precision.Policy("float32")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision disabled (FP32)")


# =============================================================================
# CPU THREADING CONFIGURATION
# =============================================================================


def set_threading_config(
    inter_op_parallelism: int = 0,
    intra_op_parallelism: int = 0,
    num_threads: Optional[int] = None,
) -> Dict[str, int]:
    """Configure TensorFlow threading for CPU operations.

    Args:
        inter_op_parallelism: Threads for independent operations (0=auto)
        intra_op_parallelism: Threads for single operation parallelism (0=auto)
        num_threads: Override both settings with a single value

    Returns:
        Dictionary with applied configuration

    Note:
        These settings primarily affect data loading and preprocessing.
        Training itself runs on GPU.
    """
    import tensorflow as tf

    if num_threads is not None:
        inter_op_parallelism = num_threads
        intra_op_parallelism = num_threads

    tf.config.threading.set_inter_op_parallelism_threads(inter_op_parallelism)
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_parallelism)

    config = {
        "inter_op_parallelism": tf.config.threading.get_inter_op_parallelism_threads(),
        "intra_op_parallelism": tf.config.threading.get_intra_op_parallelism_threads(),
    }

    print(
        f"Threading configured: inter={config['inter_op_parallelism']}, "
        f"intra={config['intra_op_parallelism']}"
    )

    return config


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def setup_gpu_environment(
    cuda_visible_devices: Optional[str] = None,
    tf_force_gpu_allow_growth: bool = True,
    tf_gpu_allocator: str = "cuda_malloc_async",
) -> None:
    """Set environment variables for GPU operation.

    Call this BEFORE importing TensorFlow for settings to take effect.

    Args:
        cuda_visible_devices: Comma-separated GPU IDs (e.g., "0" or "0,1")
        tf_force_gpu_allow_growth: Set TF_FORCE_GPU_ALLOW_GROWTH
        tf_gpu_allocator: GPU memory allocator (cuda_malloc_async recommended)

    Example:
        >>> setup_gpu_environment(cuda_visible_devices="0")
        >>> import tensorflow as tf  # Must import AFTER setup
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices}")

    if tf_force_gpu_allow_growth:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        print("TF_FORCE_GPU_ALLOW_GROWTH set to true")

    os.environ["TF_GPU_ALLOCATOR"] = tf_gpu_allocator
    print(f"TF_GPU_ALLOCATOR set to: {tf_gpu_allocator}")


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    value = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value = value / 1024.0
    return f"{value:.2f} PB"
