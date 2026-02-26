#!/usr/bin/env python3
"""
Generate synthetic test dataset for microwakeword training pipeline.
Creates minimal synthetic WAV files for testing purposes.
"""

import wave
import numpy as np
from pathlib import Path

# Configuration
SAMPLE_RATE = 16000  # 16 kHz
DURATION = 1.0  # 1 second
NUM_CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # 16-bit (2 bytes)

# Output directories
DATASET_DIR = Path("/home/sarpel/mww/microwakeword_trainer/dataset")
POSITIVE_DIR = DATASET_DIR / "positive" / "speaker_001"
NEGATIVE_DIR = DATASET_DIR / "negative" / "speech"


def generate_sine_wave(
    frequency: float, duration: float, sample_rate: int
) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Apply fade in/out to avoid clicks
    fade_len = int(sample_rate * 0.05)
    envelope = np.ones_like(t)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    return np.sin(2 * np.pi * frequency * t) * envelope * 0.7


def generate_noise(duration: float, sample_rate: int) -> np.ndarray:
    """Generate random noise."""
    samples = np.random.randn(int(duration * sample_rate))
    # Apply fade in/out
    fade_len = int(sample_rate * 0.05)
    envelope = np.ones_like(samples)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    return envelope * samples * 0.3


def save_wav_file(filepath: Path, samples: np.ndarray, sample_rate: int):
    """Save samples as 16-bit PCM WAV file."""
    # Convert to 16-bit integers
    samples_int = np.clip(samples * 32767, -32768, 32767).astype(np.int16)

    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(NUM_CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples_int.tobytes())


def create_positive_samples():
    """Create synthetic positive (wake word) samples."""
    print(f"Creating positive samples in {POSITIVE_DIR}")
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Different frequencies to simulate different "wake words"
    frequencies = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]

    for i, freq in enumerate(frequencies, 1):
        samples = generate_sine_wave(freq, DURATION, SAMPLE_RATE)
        filepath = POSITIVE_DIR / f"sample_{i:03d}.wav"
        save_wav_file(filepath, samples, SAMPLE_RATE)
        print(f"  Created {filepath.name}")

    print(f"  Total: {len(frequencies)} positive samples\n")


def create_negative_samples():
    """Create synthetic negative (non-wake word) samples."""
    print(f"Creating negative samples in {NEGATIVE_DIR}")
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Different frequencies for negative samples (lower frequency range)
    frequencies = [100, 150, 200, 250, 300, 350, 180, 220, 280, 320]

    # Generate 50 samples
    for i in range(50):
        # Mix of sine waves and noise
        if i % 3 == 0:
            # Pure noise
            samples = generate_noise(DURATION, SAMPLE_RATE)
        else:
            # Sine wave at different frequency
            freq = frequencies[i % len(frequencies)]
            samples = generate_sine_wave(freq, DURATION, SAMPLE_RATE)

        filepath = NEGATIVE_DIR / f"sample_{i + 1:03d}.wav"
        save_wav_file(filepath, samples, SAMPLE_RATE)

        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1} negative samples...")

    print("  Total: 50 negative samples\n")


def verify_wav_file(filepath: Path) -> dict:
    """Verify WAV file properties."""
    try:
        with wave.open(str(filepath), "r") as wav:
            return {
                "channels": wav.getnchannels(),
                "sample_rate": wav.getframerate(),
                "sample_width": wav.getsampwidth(),
                "n_frames": wav.getnframes(),
                "duration": wav.getnframes() / wav.getframerate(),
            }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("Synthetic Test Dataset Generator")
    print("=" * 60)
    print("Configuration:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Duration: {DURATION} seconds")
    print(f"  Channels: {NUM_CHANNELS} (mono)")
    print(f"  Bit Depth: {SAMPLE_WIDTH * 8}-bit")
    print()

    # Create samples
    create_positive_samples()
    create_negative_samples()

    # Verify a few files
    print("Verification (sample files):")
    pos_sample = list(POSITIVE_DIR.glob("*.wav"))[0]
    neg_sample = list(NEGATIVE_DIR.glob("*.wav"))[0]

    pos_info = verify_wav_file(pos_sample)
    neg_info = verify_wav_file(neg_sample)

    print(f"  Positive: {pos_sample.name}")
    print(f"    Channels: {pos_info['channels']}, Rate: {pos_info['sample_rate']} Hz")
    print(f"    Duration: {pos_info['duration']:.2f}s, Frames: {pos_info['n_frames']}")

    print(f"  Negative: {neg_sample.name}")
    print(f"    Channels: {neg_info['channels']}, Rate: {neg_info['sample_rate']} Hz")
    print(f"    Duration: {neg_info['duration']:.2f}s, Frames: {neg_info['n_frames']}")

    # Count files
    pos_count = len(list(POSITIVE_DIR.glob("*.wav")))
    neg_count = len(list(NEGATIVE_DIR.glob("*.wav")))

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Positive samples: {pos_count} (in {POSITIVE_DIR})")
    print(f"  Negative samples: {neg_count} (in {NEGATIVE_DIR})")
    print("=" * 60)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in POSITIVE_DIR.glob("*.wav"))
    total_size += sum(f.stat().st_size for f in NEGATIVE_DIR.glob("*.wav"))
    print(f"  Total size: {total_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
