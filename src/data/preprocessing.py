from __future__ import annotations

"""Audio preprocessing library extracted from standalone scripts.

Contains logic from scripts/vad_trim_audio.py and scripts/split_long_audio.py.
"""

import array
import shutil
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Result/config types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SpeechPreprocessConfig:
    min_duration_ms: float = 300.0
    max_duration_ms: float = 2000.0
    pad_ms: int = 200
    vad_aggressiveness: int = 2


@dataclass(slots=True)
class PreprocessResult:
    path: Path
    action: str
    old_duration_ms: float
    new_duration_ms: float
    reason: str = ""


@dataclass(slots=True)
class SplitResult:
    clips_written: int
    clips_discarded: int


# ──────────────────────────────────────────────────────────────────────────────
# Audio I/O helpers (stdlib-only)
# ──────────────────────────────────────────────────────────────────────────────


def _read_wav(path: Path) -> tuple[bytes, int, int, int]:
    """Return (raw_pcm_bytes, sample_rate_hz, n_channels, sample_width_bytes)."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    return frames, rate, channels, width


def _write_wav(path: Path, frames: bytes, rate: int, channels: int, width: int) -> None:
    """Write raw PCM bytes to a WAV file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(frames)


def _duration_ms_raw(frames: bytes, rate: int, channels: int, width: int) -> float:
    """Duration of raw PCM bytes in milliseconds."""
    return len(frames) / (rate * channels * width) * 1000.0


def _duration_ms_s16(pcm_s16: bytes, rate: int = 16000) -> float:
    """Duration of 16-bit mono PCM in milliseconds."""
    return len(pcm_s16) / 2 / rate * 1000.0


def _get_wav_duration_ms(path: Path) -> float:
    """Return duration of a WAV file in milliseconds. Returns -1 on error."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return (frames / rate) * 1000.0
    except Exception:
        return -1.0


# ──────────────────────────────────────────────────────────────────────────────
# PCM conversion helpers
# ──────────────────────────────────────────────────────────────────────────────


def _stereo_to_mono_s16(frames: bytes, width: int) -> bytes:
    """Mix stereo PCM down to mono int16."""
    if width == 2:
        s = array.array("h", frames)
        mono = array.array("h", [(s[i] + s[i + 1]) // 2 for i in range(0, len(s), 2)])
        return mono.tobytes()
    if width == 1:
        u = array.array("B", frames)
        mono_u = array.array("B", [(u[i] + u[i + 1]) // 2 for i in range(0, len(u), 2)])
        return mono_u.tobytes()

    out = bytearray()
    for i in range(0, len(frames), width * 2):
        out.extend(frames[i : i + width])
    return bytes(out)


def _resample_linear_s16(frames: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample int16 mono PCM via linear interpolation (stdlib-only)."""
    if src_rate == dst_rate:
        return frames
    samples = array.array("h", frames)
    n_src = len(samples)
    n_dst = int(n_src * dst_rate / src_rate)
    if n_dst == 0:
        return b""

    out = array.array("h", [0] * n_dst)
    ratio = (n_src - 1) / max(n_dst - 1, 1)
    for i in range(n_dst):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, n_src - 1)
        frac = pos - lo
        out[i] = int(samples[lo] * (1.0 - frac) + samples[hi] * frac)
    return out.tobytes()


def _to_16khz_mono_s16(frames: bytes, rate: int, channels: int, width: int) -> bytes:
    """Convert arbitrary WAV PCM to 16kHz mono signed-16-bit PCM."""
    # Step 1: to int16
    if width == 1:
        u8 = array.array("B", frames)
        frames = array.array("h", [(b - 128) * 256 for b in u8]).tobytes()
        width = 2
    elif width == 2:
        pass
    elif width == 3:
        out = bytearray()
        for i in range(0, len(frames), 3):
            b0, b1, b2 = frames[i], frames[i + 1], frames[i + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val >= (1 << 23):
                val -= 1 << 24
            out.extend(struct.pack("<h", max(-32768, min(32767, val >> 8))))
        frames = bytes(out)
        width = 2
    elif width == 4:
        s32 = array.array("i", frames)
        frames = array.array("h", [max(-32768, min(32767, s >> 16)) for s in s32]).tobytes()
        width = 2

    # Step 2: to mono
    if channels == 2:
        frames = _stereo_to_mono_s16(frames, width)
    elif channels > 2:
        out = bytearray()
        for i in range(0, len(frames), width * channels):
            out.extend(frames[i : i + width])
        frames = bytes(out)

    # Step 3: to 16kHz
    if rate != 16000:
        frames = _resample_linear_s16(frames, rate, 16000)

    return frames


# ──────────────────────────────────────────────────────────────────────────────
# VAD trimming
# ──────────────────────────────────────────────────────────────────────────────

_VAD_FRAME_MS = 30
_VAD_RATE = 16000
_VAD_FRAME_BYTES = _VAD_RATE * _VAD_FRAME_MS // 1000 * 2


def find_speech_boundaries(
    pcm_s16: bytes,
    aggressiveness: int = 2,
    pad_ms: int = 200,
) -> tuple[int, int] | None:
    """Find speech boundaries using webrtcvad.

    Args:
        pcm_s16: Signed 16-bit mono PCM at 16kHz
        aggressiveness: webrtcvad mode 0-3 (higher = more aggressive)
        pad_ms: Silence padding to keep around speech region

    Returns:
        (start_byte, end_byte) in pcm_s16, or None if no speech found.
    """
    from src.utils.optional_deps import require_optional

    webrtcvad = require_optional("webrtcvad", extra="vad")
    vad = webrtcvad.Vad(aggressiveness)
    n_frames = len(pcm_s16) // _VAD_FRAME_BYTES
    if n_frames == 0:
        return None

    speech: list[bool] = []
    for i in range(n_frames):
        frame = pcm_s16[i * _VAD_FRAME_BYTES : (i + 1) * _VAD_FRAME_BYTES]
        try:
            speech.append(vad.is_speech(frame, _VAD_RATE))
        except Exception:
            speech.append(False)

    first = next((i for i, s in enumerate(speech) if s), None)
    last = next((i for i, s in reversed(list(enumerate(speech))) if s), None)
    if first is None or last is None:
        return None

    pad_frames = pad_ms // _VAD_FRAME_MS
    start = max(0, first - pad_frames)
    end = min(n_frames, last + pad_frames + 1)
    return start * _VAD_FRAME_BYTES, end * _VAD_FRAME_BYTES


def _vad_trim(
    pcm_s16: bytes,
    aggressiveness: int = 2,
    pad_ms: int = 200,
) -> bytes | None:
    """Trim leading/trailing silence using webrtcvad."""
    bounds = find_speech_boundaries(pcm_s16, aggressiveness=aggressiveness, pad_ms=pad_ms)
    if bounds is None:
        return None
    start, end = bounds
    trimmed = pcm_s16[start:end]
    return trimmed if trimmed else None


# ──────────────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────────────


def _split_raw_pcm(
    frames: bytes,
    rate: int,
    channels: int,
    width: int,
    chunk_ms: float,
) -> list[bytes]:
    """Split raw PCM into fixed-length chunks. Keeps all remainder (no min length)."""
    bytes_per_sample = width * channels
    bytes_per_chunk = int(rate * chunk_ms / 1000) * bytes_per_sample
    chunks: list[bytes] = []
    offset = 0
    while offset < len(frames):
        chunk = frames[offset : offset + bytes_per_chunk]
        if chunk:
            chunks.append(chunk)
        offset += bytes_per_chunk
    return chunks


def _split_raw_frames(
    raw_frames: bytes,
    sample_rate: int,
    n_channels: int,
    sample_width: int,
    target_duration_ms: float,
    min_duration_ms: float,
) -> list[bytes]:
    """Split raw PCM bytes into fixed-length chunks.

    Returns only chunks with at least min_duration_ms.
    """
    bytes_per_sample = sample_width * n_channels
    samples_per_clip = int(sample_rate * target_duration_ms / 1000)
    bytes_per_clip = samples_per_clip * bytes_per_sample
    min_samples = int(sample_rate * min_duration_ms / 1000)
    min_bytes = min_samples * bytes_per_sample

    clips: list[bytes] = []
    offset = 0
    while offset + min_bytes <= len(raw_frames):
        end = offset + bytes_per_clip
        chunk = raw_frames[offset:end]
        if len(chunk) >= min_bytes:
            clips.append(chunk)
        offset += bytes_per_clip
    return clips


# ──────────────────────────────────────────────────────────────────────────────
# Required public API
# ──────────────────────────────────────────────────────────────────────────────


def trim_speech_file(path: Path, config: SpeechPreprocessConfig, discarded_root: Path) -> PreprocessResult:
    """VAD-trim a speech file. Move to discarded/ if outside configured range."""
    if "_part" in path.stem:
        return PreprocessResult(path=path, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason="already a split part")

    try:
        raw, rate, channels, width = _read_wav(path)
    except Exception as exc:
        return PreprocessResult(path=path, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason=f"read error: {exc}")

    old_ms = _duration_ms_raw(raw, rate, channels, width)
    try:
        pcm = _to_16khz_mono_s16(raw, rate, channels, width)
    except Exception as exc:
        return PreprocessResult(path=path, action="skip", old_duration_ms=old_ms, new_duration_ms=old_ms, reason=f"conversion error: {exc}")

    trimmed = _vad_trim(pcm, aggressiveness=config.vad_aggressiveness, pad_ms=config.pad_ms)
    if trimmed is None:
        new_ms = 0.0
        reason = "no speech detected"
    else:
        new_ms = _duration_ms_s16(trimmed)
        reason = ""

    in_range = (trimmed is not None) and (config.min_duration_ms <= new_ms <= config.max_duration_ms)
    if not in_range:
        dest = discarded_root / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        if not reason:
            reason = f"out of range after trim ({new_ms:.0f}ms)"
        return PreprocessResult(path=path, action="discard", old_duration_ms=old_ms, new_duration_ms=new_ms, reason=reason)

    if trimmed is None:
        return PreprocessResult(path=path, action="discard", old_duration_ms=old_ms, new_duration_ms=new_ms, reason=reason)

    _write_wav(path, trimmed, 16000, 1, 2)
    action = "trim" if new_ms < old_ms * 0.99 else "keep"
    return PreprocessResult(path=path, action=action, old_duration_ms=old_ms, new_duration_ms=new_ms)


def split_background_file(path: Path, max_duration_ms: float, discarded_root: Path) -> PreprocessResult:
    """Background file split-only path: split files above max duration in-place."""
    del discarded_root
    if "_part" in path.stem:
        return PreprocessResult(path=path, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason="already a split part")

    try:
        raw, rate, channels, width = _read_wav(path)
    except Exception as exc:
        return PreprocessResult(path=path, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason=f"read error: {exc}")

    old_ms = _duration_ms_raw(raw, rate, channels, width)
    if old_ms <= max_duration_ms:
        return PreprocessResult(path=path, action="keep", old_duration_ms=old_ms, new_duration_ms=old_ms)

    chunks = _split_raw_pcm(raw, rate, channels, width, max_duration_ms)
    if not chunks:
        return PreprocessResult(path=path, action="skip", old_duration_ms=old_ms, new_duration_ms=old_ms, reason="split produced no chunks")

    stem = path.stem
    for i, chunk in enumerate(chunks, start=1):
        out_path = path.parent / f"{stem}_part{i:03d}.wav"
        _write_wav(out_path, chunk, rate, channels, width)
    path.unlink()
    return PreprocessResult(path=path, action="split", old_duration_ms=old_ms, new_duration_ms=old_ms, reason=str(len(chunks)))


def process_speech_directory(
    root: Path,
    config: SpeechPreprocessConfig,
    discarded_root: Path,
    dry_run: bool,
) -> list[PreprocessResult]:
    """Process speech directory recursively with VAD trim + discard policy."""
    files = sorted(f for f in root.rglob("*.wav") if "_part" not in f.stem)
    results: list[PreprocessResult] = []
    for f in files:
        if dry_run:
            try:
                raw, rate, channels, width = _read_wav(f)
            except Exception as exc:
                results.append(PreprocessResult(path=f, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason=f"read error: {exc}"))
                continue
            old_ms = _duration_ms_raw(raw, rate, channels, width)
            try:
                pcm = _to_16khz_mono_s16(raw, rate, channels, width)
                trimmed = _vad_trim(pcm, aggressiveness=config.vad_aggressiveness, pad_ms=config.pad_ms)
            except Exception as exc:
                results.append(PreprocessResult(path=f, action="skip", old_duration_ms=old_ms, new_duration_ms=old_ms, reason=f"conversion error: {exc}"))
                continue

            if trimmed is None:
                results.append(PreprocessResult(path=f, action="discard", old_duration_ms=old_ms, new_duration_ms=0.0, reason="no speech detected"))
                continue

            new_ms = _duration_ms_s16(trimmed)
            in_range = config.min_duration_ms <= new_ms <= config.max_duration_ms
            if not in_range:
                results.append(PreprocessResult(path=f, action="discard", old_duration_ms=old_ms, new_duration_ms=new_ms, reason=f"out of range after trim ({new_ms:.0f}ms)"))
                continue
            action = "trim" if new_ms < old_ms * 0.99 else "keep"
            results.append(PreprocessResult(path=f, action=action, old_duration_ms=old_ms, new_duration_ms=new_ms))
            continue

        try:
            raw, rate, channels, width = _read_wav(f)
        except Exception as exc:
            results.append(PreprocessResult(path=f, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason=f"read error: {exc}"))
            continue

        old_ms = _duration_ms_raw(raw, rate, channels, width)
        try:
            pcm = _to_16khz_mono_s16(raw, rate, channels, width)
            trimmed = _vad_trim(pcm, aggressiveness=config.vad_aggressiveness, pad_ms=config.pad_ms)
        except Exception as exc:
            results.append(PreprocessResult(path=f, action="skip", old_duration_ms=old_ms, new_duration_ms=old_ms, reason=f"conversion error: {exc}"))
            continue

        if trimmed is None:
            new_ms = 0.0
            reason = "no speech detected"
        else:
            new_ms = _duration_ms_s16(trimmed)
            reason = ""

        in_range = (trimmed is not None) and (config.min_duration_ms <= new_ms <= config.max_duration_ms)
        if not in_range:
            rel = f.relative_to(root)
            dest = discarded_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            if not reason:
                reason = f"out of range after trim ({new_ms:.0f}ms)"
            results.append(PreprocessResult(path=f, action="discard", old_duration_ms=old_ms, new_duration_ms=new_ms, reason=reason))
            continue

        if trimmed is None:
            results.append(PreprocessResult(path=f, action="discard", old_duration_ms=old_ms, new_duration_ms=new_ms, reason=reason))
            continue

        _write_wav(f, trimmed, 16000, 1, 2)
        action = "trim" if new_ms < old_ms * 0.99 else "keep"
        results.append(PreprocessResult(path=f, action=action, old_duration_ms=old_ms, new_duration_ms=new_ms))
    return results


def process_background_directory(
    root: Path,
    max_duration_ms: float,
    discarded_root: Path,
    dry_run: bool,
) -> list[PreprocessResult]:
    """Process background directory recursively with split-only behavior."""
    del discarded_root
    files = sorted(f for f in root.rglob("*.wav") if "_part" not in f.stem)
    results: list[PreprocessResult] = []
    for f in files:
        if dry_run:
            try:
                raw, rate, channels, width = _read_wav(f)
            except Exception as exc:
                results.append(PreprocessResult(path=f, action="skip", old_duration_ms=0.0, new_duration_ms=0.0, reason=f"read error: {exc}"))
                continue
            old_ms = _duration_ms_raw(raw, rate, channels, width)
            if old_ms <= max_duration_ms:
                results.append(PreprocessResult(path=f, action="keep", old_duration_ms=old_ms, new_duration_ms=old_ms))
            else:
                chunks = _split_raw_pcm(raw, rate, channels, width, max_duration_ms)
                if not chunks:
                    results.append(PreprocessResult(path=f, action="skip", old_duration_ms=old_ms, new_duration_ms=old_ms, reason="split produced no chunks"))
                else:
                    results.append(PreprocessResult(path=f, action="split", old_duration_ms=old_ms, new_duration_ms=old_ms, reason=str(len(chunks))))
            continue
        results.append(split_background_file(f, max_duration_ms=max_duration_ms, discarded_root=Path(".")))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Additional extracted split_long_audio.py logic
# ──────────────────────────────────────────────────────────────────────────────


def split_file(
    src: Path,
    max_duration_ms: float,
    target_duration_ms: float,
    min_duration_ms: float,
    dry_run: bool = False,
) -> SplitResult:
    """Split a single WAV file if it exceeds max_duration_ms.

    Source file is never modified; split clips are created alongside source.
    """
    duration_ms = _get_wav_duration_ms(src)
    if duration_ms < 0 or duration_ms <= max_duration_ms:
        return SplitResult(clips_written=0, clips_discarded=0)

    stem = src.stem
    existing = list(src.parent.glob(f"{stem}_part*.wav"))
    if existing:
        return SplitResult(clips_written=0, clips_discarded=0)

    raw_frames, sample_rate, n_channels, sample_width = _read_wav(src)
    chunks = _split_raw_frames(
        raw_frames,
        sample_rate,
        n_channels,
        sample_width,
        target_duration_ms,
        min_duration_ms,
    )

    written = 0
    for i, chunk in enumerate(chunks, start=1):
        out_path = src.parent / f"{stem}_part{i:03d}.wav"
        if not dry_run:
            _write_wav(out_path, chunk, sample_rate, n_channels, sample_width)
        written += 1

    return SplitResult(clips_written=written, clips_discarded=(1 if len(chunks) == 0 else 0))


def scan_and_split(
    directories: list[Path],
    max_duration_ms: float,
    target_duration_ms: float,
    min_duration_ms: float,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    """Recursively scan directories and split long audio files.

    Returns tuple: (total_long, total_written, total_discarded, total_skipped)
    """
    total_long = 0
    total_written = 0
    total_discarded = 0
    total_skipped = 0

    for root in directories:
        if not root.exists():
            print(f"[WARN] Directory not found, skipping: {root}")
            continue

        wav_files = sorted(root.rglob("*.wav"))
        print(f"\nScanning {root}  ({len(wav_files)} WAV files)")

        for wav in wav_files:
            if "_part" in wav.stem:
                continue

            duration_ms = _get_wav_duration_ms(wav)
            if duration_ms < 0:
                print(f"  [SKIP] Could not read: {wav.relative_to(root)}")
                total_skipped += 1
                continue

            if duration_ms <= max_duration_ms:
                continue

            total_long += 1
            stem = wav.stem
            existing = list(wav.parent.glob(f"{stem}_part*.wav"))
            if existing:
                print(f"  [SKIP] Already split ({len(existing)} parts): {wav.relative_to(root)}")
                total_skipped += 1
                continue

            action = "[DRY]" if dry_run else "[SPLIT]"
            print(f"  {action} {wav.relative_to(root)} ({duration_ms:.0f}ms → clips of {target_duration_ms:.0f}ms)")

            res = split_file(
                wav,
                max_duration_ms=max_duration_ms,
                target_duration_ms=target_duration_ms,
                min_duration_ms=min_duration_ms,
                dry_run=dry_run,
            )
            total_written += res.clips_written
            total_discarded += res.clips_discarded

    return total_long, total_written, total_discarded, total_skipped


def remove_split_originals(directories: list[Path], max_duration_ms: float) -> int:
    """Remove original files that have already been split."""
    removed = 0
    for root in directories:
        if not root.exists():
            continue
        for wav in sorted(root.rglob("*.wav")):
            if "_part" in wav.stem:
                continue
            duration_ms = _get_wav_duration_ms(wav)
            if duration_ms <= max_duration_ms:
                continue
            parts = list(wav.parent.glob(f"{wav.stem}_part*.wav"))
            if parts:
                print(f"  [REMOVE] {wav.relative_to(root)} ({len(parts)} parts kept)")
                wav.unlink()
                removed += 1
    print(f"\nRemoved {removed} original files.")
    return removed
