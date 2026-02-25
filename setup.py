"""microwakeword_trainer - GPU-Accelerated Wake Word Training Framework"""

from setuptools import setup, find_packages

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
        "python>=3.10,<3.13",
        "tensorflow>=2.16",
        "ai-edge-litert",
        "pymicro-features>=0.1",
        "numpy>=1.26",
        "scipy",
        "pyyaml",
        "mmap_ninja",
        "datasets>=2.14",
        "audiomentations",
        "audio_metadata",
        "webrtcvad-wheels",
        "absl-py",
        # Performance (v2.0)
        "cupy-cuda12x>=13.0",
        "pyarrow>=15.0",
        "numba>=0.58",
    ],
    extras_require={
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
        ],
    },
)
