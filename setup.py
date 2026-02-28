"""microwakeword_trainer - GPU-Accelerated Wake Word Training Framework"""

from setuptools import find_packages, setup

setup(
    name="microwakeword_trainer",
    version="2.0.0",
    description="GPU-accelerated wake word model training pipeline",
    author="MWW Team",
    author_email="team@mww.dev",
    url="https://github.com/mww/microwakeword_trainer",
    packages=find_packages(),
    package_data={
        "microwakeword_trainer": ["presets/**/*.yaml", "presets/**/*.yml"],
    },
    install_requires=[
        # Core
        "tensorflow>=2.16",
        "ai-edge-litert",
        "pymicro-features>=0.1",
        "numpy>=1.26",
        "scipy",
        "pyyaml",
        "mmap_ninja",
        "audiomentations",
        "audio_metadata",
        "webrtcvad-wheels",
        "absl-py",
        # Performance (v2.0)
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "cupy-cuda12x>=13.0; sys_platform == 'linux'",
        "numba>=0.58",
    ],
    extras_require={
        "vad": ["webrtcvad-wheels"],
        "quality-full": ["torch>=2.0", "onnxruntime>=1.16"],
        "optional": [
            "speechbrain>=1.0.0",
            "transformers>=4.40.0",
            "scikit-learn>=1.4.0",
            "optuna>=3.6.0",
            "matplotlib",
            "seaborn",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "mww-train=src.training.trainer:main",
            "mww-export=src.export.tflite:main",
            "mww-autotune=src.tuning.cli:main",
            "mww-cluster-analyze=src.tools.cluster_analyze:main",
            "mww-cluster-apply=src.tools.cluster_apply:main",
        ],
    },
)
