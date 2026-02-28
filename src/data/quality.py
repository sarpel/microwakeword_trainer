from __future__ import annotations

"""Audio quality scoring library (fast + full modes)."""

import csv
import os
import shutil
import threading
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.optional_deps import require_optional


@dataclass(slots=True)
class QualityScoreConfig:
    clip_threshold: float = 0.001
    discard_bottom_pct: float = 5.0
    min_wqi: float = 0.0
    vad_threshold: float = 0.0
    discarded_dir: Path = Path("discarded/quality")
    dnsmos_cache: Path = Path.home() / ".cache" / "dnsmos"
    verbose: bool = False


@dataclass(slots=True)
class FileScore:
    path: Path
    dir_label: str
    clip_ratio: float = 0.0
    snr_db: float = -20.0
    vad_conf: float = 0.0
    dnsmos_sig: float = 1.0
    dnsmos_bak: float = 1.0
    dnsmos_ovrl: float = 1.0
    wqi: float = 0.0
    discard: bool = False
    discard_reason: str = ""
    error: str = ""


# DNSMOS model download
# Source: Microsoft DNS-Challenge (ICASSP 2021)
# https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
_DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
_DNSMOS_SAMPLE_RATE = 16000
# DNSMOS model input length — read dynamically from the ONNX graph at runtime
# (the sig_bak_ovr.onnx model expects exactly 144160 samples, not 9*16000=144000)


def _ensure_dnsmos_model(cache_dir: Path) -> Path:
    """Download DNSMOS ONNX model if not already cached. Returns path to .onnx."""
    model_path = cache_dir / "DNSMOS" / "sig_bak_ovr.onnx"
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading DNSMOS model → {model_path} ...")
    urllib.request.urlretrieve(_DNSMOS_URL, str(model_path))
    print(f"  DNSMOS model downloaded ({model_path.stat().st_size / 1024:.0f} KB)")
    return model_path


def _read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV file → float32 array in [-1, 1] + sample rate."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
    except Exception:
        return np.array([], dtype=np.float32), 0

    if len(raw) == 0:
        return np.array([], dtype=np.float32), rate

    if width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 3:
        n = len(raw) // 3
        padded = bytearray(n * 4)
        for i in range(n):
            padded[i * 4 + 1] = raw[i * 3]
            padded[i * 4 + 2] = raw[i * 3 + 1]
            padded[i * 4 + 3] = raw[i * 3 + 2]
        samples = np.frombuffer(bytes(padded), dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return np.array([], dtype=np.float32), rate

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples, rate


def _resample_to_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Linear-interpolation resample to 16kHz (torchaudio if available)."""
    if src_rate == _DNSMOS_SAMPLE_RATE:
        return samples
    try:
        torchaudio = require_optional("torchaudio", extra="quality-full")
        torch = require_optional("torch", extra="quality-full")
        t = torch.tensor(samples.astype(np.float32)).unsqueeze(0)
        t = torchaudio.functional.resample(t, src_rate, _DNSMOS_SAMPLE_RATE)
        return t.squeeze(0).numpy()
    except ImportError:
        pass

    n_src = len(samples)
    n_dst = int(n_src * _DNSMOS_SAMPLE_RATE / src_rate)
    if n_dst == 0:
        return np.array([], dtype=np.float32)
    idx = np.linspace(0, n_src - 1, n_dst)
    lo = idx.astype(int)
    hi = np.minimum(lo + 1, n_src - 1)
    frac = idx - lo
    return (samples[lo] * (1.0 - frac) + samples[hi] * frac).astype(np.float32)


def compute_clipping_ratio(samples: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at or above clipping threshold in absolute value."""
    if len(samples) == 0:
        return 0.0
    clipped = np.sum(np.abs(samples) >= threshold)
    return float(clipped) / len(samples)


_WADA_DB_VALS = np.arange(-20, 101, dtype=np.float64)
_WADA_G_VALS = np.array(
    [
        0.40974774,
        0.40986926,
        0.40998566,
        0.40969089,
        0.40986186,
        0.40999006,
        0.41027138,
        0.41052627,
        0.41101024,
        0.41143264,
        0.41231718,
        0.41337272,
        0.41526426,
        0.41781920,
        0.42077252,
        0.42452799,
        0.42918886,
        0.43510373,
        0.44234195,
        0.45161485,
        0.46221153,
        0.47491647,
        0.48883809,
        0.50509236,
        0.52353709,
        0.54372088,
        0.56532427,
        0.58847532,
        0.61346212,
        0.63954496,
        0.66750818,
        0.69583724,
        0.72454762,
        0.75414799,
        0.78323148,
        0.81240985,
        0.84219775,
        0.87166406,
        0.90030504,
        0.92880418,
        0.95655449,
        0.98353490,
        1.01047155,
        1.03620950,
        1.06136425,
        1.08579312,
        1.10948190,
        1.13277995,
        1.15472826,
        1.17627308,
        1.19703503,
        1.21671694,
        1.23535898,
        1.25364313,
        1.27103891,
        1.28718029,
        1.30302865,
        1.31839527,
        1.33294817,
        1.34700935,
        1.36057270,
        1.37345513,
        1.38577122,
        1.39733504,
        1.40856397,
        1.41959619,
        1.42983624,
        1.43958467,
        1.44902176,
        1.45804831,
        1.46669568,
        1.47486938,
        1.48269965,
        1.49034339,
        1.49748214,
        1.50435106,
        1.51076426,
        1.51698915,
        1.52290970,
        1.52857800,
        1.53389835,
        1.53912110,
        1.54390650,
        1.54858517,
        1.55310776,
        1.55744391,
        1.56164927,
        1.56566348,
        1.56938671,
        1.57307767,
        1.57654764,
        1.57980083,
        1.58304129,
        1.58602496,
        1.58880681,
        1.59162477,
        1.59419690,
        1.59693155,
        1.59944600,
        1.60185011,
        1.60408668,
        1.60627134,
        1.60826199,
        1.61004547,
        1.61192472,
        1.61369656,
        1.61534074,
        1.61688905,
        1.61838916,
        1.61985374,
        1.62135878,
        1.62268119,
        1.62390423,
        1.62513143,
        1.62632463,
        1.62740270,
        1.62842767,
        1.62945532,
        1.63033070,
        1.63128026,
        1.63204102,
    ],
    dtype=np.float64,
)


def wada_snr(samples: np.ndarray) -> float:
    """Estimate SNR in dB using Waveform Amplitude Distribution Analysis."""
    eps = 1e-10
    wav = samples.astype(np.float64)

    peak = np.abs(wav).max() if len(wav) else 0.0
    if peak < eps:
        return -20.0

    wav = wav / peak
    abs_wav = np.abs(wav)
    abs_wav = np.where(abs_wav < eps, eps, abs_wav)

    v1 = float(abs_wav.mean())
    v2 = float(np.log(abs_wav).mean())
    v3 = np.log(max(v1, eps)) - v2

    candidates = np.where(_WADA_G_VALS < v3)[0]
    if len(candidates) == 0:
        return float(_WADA_DB_VALS[0])
    idx = int(candidates.max())
    if idx >= len(_WADA_DB_VALS) - 1:
        return float(_WADA_DB_VALS[-1])

    t = (v3 - _WADA_G_VALS[idx]) / (_WADA_G_VALS[idx + 1] - _WADA_G_VALS[idx] + eps)
    snr_db = float(_WADA_DB_VALS[idx]) + t * float(_WADA_DB_VALS[idx + 1] - _WADA_DB_VALS[idx])
    return float(np.clip(snr_db, -20.0, 100.0))


def _load_dnsmos_session(model_path: Path) -> Any:
    ort = require_optional("onnxruntime", extra="quality-full")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    sess_options.inter_op_num_threads = 1
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )


def score_dnsmos(samples_16k: np.ndarray, session: Any) -> tuple[float, float, float]:
    """Run DNSMOS on a 16kHz float32 mono clip.

    Returns (SIG, BAK, OVRL) on [1, 5] MOS scale.
    DNSMOS was trained on 9-second clips; we tile/trim to that length.
    """
    # Derive required input length from the ONNX model itself (avoids hardcoding)
    target_len: int = session.get_inputs()[0].shape[1]
    audio = samples_16k.astype(np.float32)
    if len(audio) == 0:
        return 1.0, 1.0, 1.0
    if len(audio) < target_len:
        repeats = (target_len // len(audio)) + 1
        audio = np.tile(audio, repeats)
    audio = audio[:target_len]

    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak * 0.99

    input_tensor = audio.reshape(1, -1)
    out = session.run(None, {"input_1": input_tensor})[0][0]
    sig = float(np.clip(out[0], 1.0, 5.0))
    bak = float(np.clip(out[1], 1.0, 5.0))
    ovrl = float(np.clip(out[2], 1.0, 5.0))
    return sig, bak, ovrl


def _load_silero_vad() -> tuple[Any, Any]:
    """Load Silero VAD model via torch.hub. Returns (model, utils)."""
    torch = require_optional("torch", extra="quality-full")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    return model, utils


def score_silero_vad(samples_16k: np.ndarray, model: Any) -> float:
    """Run Silero VAD on 16kHz float32 mono. Returns mean speech probability [0,1]."""
    torch = require_optional("torch", extra="quality-full")
    if len(samples_16k) < 512:
        return 0.0
    frame_size = 512
    n_frames = len(samples_16k) // frame_size
    if n_frames == 0:
        return 0.0
    # Batch all frames into one tensor — avoids per-frame Python loop
    frames = torch.tensor(
        samples_16k[: n_frames * frame_size].astype(np.float32)
    ).reshape(n_frames, frame_size)
    with torch.no_grad():
        probs = model(frames, 16000)  # shape: [n_frames]
    return float(probs.mean().item())


def _wqi_fast(snr_db: float) -> float:
    return float(min(1.0, max(0.0, snr_db / 30.0)))


def _compute_wqi(dnsmos_ovrl: float, vad_conf: float, snr_db: float) -> float:
    dnsmos_norm = (dnsmos_ovrl - 1.0) / 4.0
    snr_norm = min(1.0, max(0.0, snr_db / 30.0))
    wqi = 0.50 * dnsmos_norm + 0.30 * vad_conf + 0.20 * snr_norm
    return float(np.clip(wqi, 0.0, 1.0))


def score_file_fast(path: Path, config: QualityScoreConfig) -> FileScore:
    """Score file with clipping + WADA-SNR only."""
    samples, _ = _read_wav_float32(path)
    dir_label = path.parent.name
    if len(samples) == 0:
        return FileScore(
            path=path,
            dir_label=dir_label,
            wqi=0.0,
            discard=True,
            discard_reason="read error or empty",
            error="unreadable",
        )

    clip = compute_clipping_ratio(samples)
    snr = wada_snr(samples)
    wqi = _wqi_fast(snr)

    discard = False
    reason = ""
    if clip > config.clip_threshold:
        discard = True
        reason = f"clipping {clip * 100:.2f}% > {config.clip_threshold * 100:.2f}%"

    return FileScore(
        path=path,
        dir_label=dir_label,
        clip_ratio=clip,
        snr_db=snr,
        wqi=wqi,
        discard=discard,
        discard_reason=reason,
    )


def score_file_full(path: Path, config: QualityScoreConfig) -> FileScore:
    """Score file with DNSMOS + Silero VAD + WADA-SNR + clipping."""
    model_path = _ensure_dnsmos_model(config.dnsmos_cache)
    dnsmos_session = _load_dnsmos_session(model_path)
    vad_model, _ = _load_silero_vad()
    return _score_file_full_loaded(path, config, dnsmos_session, vad_model)


def _score_file_full_loaded(path: Path, config: QualityScoreConfig, dnsmos_session: Any, vad_model: Any) -> FileScore:
    samples, rate = _read_wav_float32(path)
    dir_label = path.parent.name
    if len(samples) == 0:
        return FileScore(
            path=path,
            dir_label=dir_label,
            discard=True,
            discard_reason="read error or empty",
            error="unreadable",
        )

    samples_16k = _resample_to_16k(samples, rate) if rate != _DNSMOS_SAMPLE_RATE else samples
    clip = compute_clipping_ratio(samples)
    snr = wada_snr(samples)
    vad_conf = score_silero_vad(samples_16k, vad_model)
    sig, bak, ovrl = score_dnsmos(samples_16k, dnsmos_session)
    wqi = _compute_wqi(ovrl, vad_conf, snr)

    discard = False
    reason = ""
    if clip > config.clip_threshold:
        discard = True
        reason = f"clipping {clip * 100:.2f}% > {config.clip_threshold * 100:.2f}%"
    elif config.vad_threshold > 0.0 and vad_conf < config.vad_threshold:
        discard = True
        reason = f"vad_conf {vad_conf:.3f} < {config.vad_threshold:.3f}"

    return FileScore(
        path=path,
        dir_label=dir_label,
        clip_ratio=clip,
        snr_db=snr,
        vad_conf=vad_conf,
        dnsmos_sig=sig,
        dnsmos_bak=bak,
        dnsmos_ovrl=ovrl,
        wqi=wqi,
        discard=discard,
        discard_reason=reason,
    )


def _apply_percentile_gate(results: list[FileScore], discard_bottom_pct: float, min_wqi: float) -> None:
    if discard_bottom_pct <= 0.0 and min_wqi <= 0.0:
        return

    by_dir: dict[str, list[FileScore]] = {}
    for r in results:
        by_dir.setdefault(r.dir_label, []).append(r)

    for label, group in by_dir.items():
        eligible = [r for r in group if not r.discard]
        if not eligible:
            continue
        wqi_values = np.array([r.wqi for r in eligible])

        if discard_bottom_pct > 0.0:
            cutoff = float(np.percentile(wqi_values, discard_bottom_pct))
            for r in eligible:
                if r.wqi <= cutoff and not r.discard:
                    r.discard = True
                    r.discard_reason = f"bottom {discard_bottom_pct:.1f}% of {label} (wqi={r.wqi:.3f} ≤ {cutoff:.3f})"

        if min_wqi > 0.0:
            for r in eligible:
                if r.wqi < min_wqi and not r.discard:
                    r.discard = True
                    r.discard_reason = f"wqi {r.wqi:.3f} < min-wqi {min_wqi:.3f}"


def score_directory(root: Path, config: QualityScoreConfig, mode: str, num_workers: int = 0) -> list[FileScore]:
    """Score all WAV files under root in mode='fast' or mode='full'."""
    files = sorted(f for f in root.rglob("*.wav"))
    n = len(files)
    print(f"\nScoring: {root}  ({n:,} files)")

    if mode not in {"fast", "full"}:
        raise ValueError("mode must be 'fast' or 'full'")

    results: list[FileScore] = []
    if mode == "fast":
        nw = num_workers or min(32, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(score_file_fast, f, config): f for f in files}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                r.dir_label = root.name
                results.append(r)
                done += 1
                if config.verbose:
                    print(f"  {r.path.relative_to(root)}  clip={r.clip_ratio * 100:.3f}%  snr={r.snr_db:.1f}dB  wqi={r.wqi:.3f}" + (f"  [GATE: {r.discard_reason}]" if r.discard else ""))
                elif done % 5000 == 0:
                    print(f"  ... {done:,}/{n:,} scored")
    else:
        model_path = _ensure_dnsmos_model(config.dnsmos_cache)
        nw = num_workers or min(16, os.cpu_count() or 4)
        print("  Loading DNSMOS + Silero VAD (per-thread, lazy) ...")

        # Thread-local storage: each thread gets its own model instances
        _tls: threading.local = threading.local()

        def _score_full(f: Path) -> FileScore:
            if not hasattr(_tls, "dnsmos"):
                _tls.dnsmos = _load_dnsmos_session(model_path)
                _tls.vad, _ = _load_silero_vad()
            r = _score_file_full_loaded(f, config, _tls.dnsmos, _tls.vad)
            r.dir_label = root.name
            return r

        done = 0
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_score_full, f): f for f in files}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                done += 1
                if config.verbose:
                    print(
                        f"  {r.path.relative_to(root)}  clip={r.clip_ratio * 100:.3f}%  snr={r.snr_db:.1f}dB  "
                        f"vad={r.vad_conf:.3f}  dnsmos={r.dnsmos_ovrl:.2f}  wqi={r.wqi:.3f}" + (f"  [GATE: {r.discard_reason}]" if r.discard else "")
                    )
                elif done % 1000 == 0:
                    print(f"  ... {done:,}/{n:,} scored")
        print("  DNSMOS + Silero VAD ✓")

    _apply_percentile_gate(results, config.discard_bottom_pct, config.min_wqi)
    return results


def write_csv(results: list[FileScore], csv_path: Path) -> None:
    """Write CSV with full schema used by fast/full scripts."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "dir_label",
                "clip_ratio",
                "snr_db",
                "vad_conf",
                "dnsmos_sig",
                "dnsmos_bak",
                "dnsmos_ovrl",
                "wqi",
                "discard",
                "discard_reason",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    str(r.path),
                    r.dir_label,
                    f"{r.clip_ratio:.6f}",
                    f"{r.snr_db:.2f}",
                    f"{r.vad_conf:.4f}",
                    f"{r.dnsmos_sig:.3f}",
                    f"{r.dnsmos_bak:.3f}",
                    f"{r.dnsmos_ovrl:.3f}",
                    f"{r.wqi:.4f}",
                    "1" if r.discard else "0",
                    r.discard_reason,
                    r.error,
                ]
            )


def apply_discard(scores: list[FileScore], discarded_dir: Path, dry_run: bool) -> int:
    """Move files marked for discard into discarded_dir. Returns count moved."""
    moved = 0
    if dry_run:
        return sum(1 for r in scores if r.discard and r.error != "unreadable")
    for r in scores:
        if not r.discard or r.error == "unreadable":
            continue
        dest = discarded_dir / r.dir_label / r.path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(r.path), str(dest))
        moved += 1
    return moved


def print_summary(scores: list[FileScore], config: QualityScoreConfig, dry_run: bool, mode: str) -> None:
    """Print summary matching original script behavior."""
    prefix = "[DRY-RUN] " if dry_run else ""
    width = 64 if mode == "fast" else 68
    print()
    print("=" * width)
    print(f"{prefix}Quality Summary  (clip gate: {config.clip_threshold * 100:.2f}%,  bottom {config.discard_bottom_pct:.1f}% per dir)")
    print("=" * width)

    by_dir: dict[str, list[FileScore]] = {}
    for r in scores:
        by_dir.setdefault(r.dir_label, []).append(r)
    total_discard = sum(1 for r in scores if r.discard)

    for label, group in sorted(by_dir.items()):
        to_discard = [r for r in group if r.discard]
        valid = [r for r in group if not r.error]
        print(f"\n  {label}/")
        print(f"    files:      {len(group):>8,}")
        if valid:
            wqis = [r.wqi for r in valid]
            snrs = [r.snr_db for r in valid]
            print(f"    wqi:        min={min(wqis):.3f}  med={float(np.median(wqis)):.3f}  mean={float(np.mean(wqis)):.3f}  max={max(wqis):.3f}")
            print(f"    snr_db:     min={min(snrs):.1f}  med={float(np.median(snrs)):.1f}  mean={float(np.mean(snrs)):.1f}  max={max(snrs):.1f}")
            if mode == "full":
                vads = [r.vad_conf for r in valid]
                ovrl = [r.dnsmos_ovrl for r in valid]
                print(f"    vad_conf:   min={min(vads):.3f}  med={float(np.median(vads)):.3f}  mean={float(np.mean(vads)):.3f}  max={max(vads):.3f}")
                print(f"    dnsmos:     min={min(ovrl):.2f}  med={float(np.median(ovrl)):.2f}  mean={float(np.mean(ovrl)):.2f}  max={max(ovrl):.2f}")

        clips_gated = sum(1 for r in to_discard if "clipping" in r.discard_reason)
        pct_gated = sum(1 for r in to_discard if "bottom" in r.discard_reason or "wqi" in r.discard_reason)
        vad_gated = sum(1 for r in to_discard if "vad_conf" in r.discard_reason)
        if mode == "full":
            print(f"    discard:    {len(to_discard):>8,}  (clip: {clips_gated}, vad_gate: {vad_gated}, percentile/floor: {pct_gated})")
        else:
            print(f"    discard:    {len(to_discard):>8,}  (clipping gate: {clips_gated}, percentile/floor: {pct_gated})")

    print("\n  ── Grand total ──────────────────────────────────────────────")
    print(f"    total files:    {len(scores):>8,}")
    pct = total_discard / max(len(scores), 1) * 100.0
    print(f"    to discard:     {total_discard:>8,}  ({pct:.1f}%)")
    print(f"    moved to:       {config.discarded_dir}/")
    if dry_run:
        print()
        print("  ⚠  DRY-RUN: No files were modified.")
        print("  Re-run with --apply to move files.")
    print("=" * width)
