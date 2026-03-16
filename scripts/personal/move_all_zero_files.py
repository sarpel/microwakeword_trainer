"""
Dataset Garbage Collector - Cache Poison Yapan Dosyaları Ayıklar

CONCEPT: Training engine'i bozan dosyaları temizler.
WHY: NaN loss, crash, veya sessiz gradient sorunlarını önler.

NOT: Süre, sample rate, channel sayısı gibi format kontrolleri YAPILMAZ.
     Bunlar runtime'da handle edilir. Sadece BOZUK veri ayıklanır.

Usage:
    python move_all_zero_files.py              # Normal çalıştır
    python move_all_zero_files.py --dry-run    # Sadece listele, taşıma
"""

import logging
import os
import shutil
import sys
import uuid
import warnings

import numpy as np
from scipy.io import wavfile

from src.utils.logging_config import get_logger

# --- CONFIGURATION ---
SOURCE_DIR = r"/home/sarpel/microwakeword-training-platform/data"
DEST_DIR = r"/home/sarpel/microwakeword-training-platform/all_zeroes"
LOG_FILE = r"/home/sarpel/microwakeword-training-platform/logs/dataset_cleaning.log"

# Runtime flags
DRY_RUN = "--dry-run" in sys.argv

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = get_logger(__name__)


def is_garbage(filepath: str) -> tuple[bool, str]:
    """
    Dosyanın training engine'i bozup bozmayacağını kontrol eder.

    SADECE CACHE POISON yapan dosyaları yakalar:
    - Açılamayan/bozuk dosyalar
    - Tamamen sıfır içerik
    - NaN/Inf değerler
    - Constant array (tüm değerler aynı)

    YAPILMAYAN kontroller (runtime'da handle edilir):
    - Süre kontrolü (RIR, background farklı olabilir)
    - Sample rate (resample edilir)
    - Channel sayısı (downmix yapılır)
    - Mel bin sayısı (farklı formatlar olabilir)

    Args:
        filepath: Kontrol edilecek dosya yolu

    Returns:
        tuple[bool, str]: (is_garbage, reason) - Bozuksa True ve sebep
    """
    filename = os.path.basename(filepath).lower()

    # ═══════════════════════════════════════════════════════════════════════
    # WAV CONTROLS - Sadece okunamayan veya matematiksel olarak bozuk
    # ═══════════════════════════════════════════════════════════════════════
    if filename.endswith(".wav"):
        # A) MAGIC BYTES - RIFF/WAVE header kontrolü
        try:
            with open(filepath, "rb") as f:
                header = f.read(12)
                if not header.startswith(b"RIFF") or b"WAVE" not in header:
                    return True, "SAHTE_WAV_HEADER"
        except Exception as e:
            return True, f"OKUNAMIYOR: {e}"

        # B) CONTENT ANALYSIS
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sr, data = wavfile.read(filepath)

                # Kritik scipy uyarıları - SADECE gerçekten veri kaybına yol açanlar
                # NOT: "Chunk not understood" ZARARSIZ - sadece metadata atlıyor
                for warning in w:
                    msg = str(warning.message).lower()
                    # "corrupt" = dosya gerçekten bozuk
                    # "truncated" = dosya kesilmiş, eksik veri
                    # "chunk" = genellikle zararsız metadata, ATLA
                    if "corrupt" in msg or "truncated" in msg:
                        return True, f"BOZUK_WAV_UYARI: {warning.message}"

                # Boş veri = kesin garbage
                if data.size == 0:
                    return True, "BOS_DOSYA"

                # Stereo ise mono'ya çevir (analiz için)
                if data.ndim > 1:
                    data = data[:, 0]

                # ───────────────────────────────────────────────────────────
                # NaN / Inf KONTROLÜ - Training'i çökertir
                # ───────────────────────────────────────────────────────────
                if np.issubdtype(data.dtype, np.floating):
                    if np.isnan(data).any():
                        return True, "NaN_ICERIYOR"
                    if np.isinf(data).any():
                        return True, "Inf_ICERIYOR"

                # ───────────────────────────────────────────────────────────
                # TAMAMEN SIFIR - Gradient = 0, öğrenme yok
                # ───────────────────────────────────────────────────────────
                if not np.any(data):
                    return True, "TAMAMEN_SIFIR"

                # ───────────────────────────────────────────────────────────
                # CONSTANT ARRAY - Tüm sample'lar aynı değer
                # ───────────────────────────────────────────────────────────
                if np.all(data == data.flat[0]):
                    return True, "SABIT_DEGER"

        except Exception as e:
            return True, f"BOZUK_WAV: {e}"

        return False, "OK"

    # ═══════════════════════════════════════════════════════════════════════
    # NPY CONTROLS - Feature arrays
    # ═══════════════════════════════════════════════════════════════════════
    elif filename.endswith(".npy"):
        # A) MAGIC BYTES
        try:
            with open(filepath, "rb") as f:
                if f.read(6) != b"\x93NUMPY":
                    return True, "SAHTE_NPY_HEADER"
        except Exception as e:
            return True, f"OKUNAMIYOR: {e}"

        # B) CONTENT ANALYSIS
        try:
            data = np.load(filepath, allow_pickle=False)

            # Numpy array değilse
            if not isinstance(data, np.ndarray):
                return True, "ARRAY_DEGIL"

            # Boş array
            if data.size == 0:
                return True, "BOS_DOSYA"

            # ───────────────────────────────────────────────────────────────
            # NaN / Inf KONTROLÜ - Loss = NaN yapar
            # ───────────────────────────────────────────────────────────────
            if np.issubdtype(data.dtype, np.floating):
                if np.isnan(data).any():
                    return True, "NaN_ICERIYOR"
                if np.isinf(data).any():
                    return True, "Inf_ICERIYOR"

            # ───────────────────────────────────────────────────────────────
            # TAMAMEN SIFIR - Zero gradient
            # ───────────────────────────────────────────────────────────────
            if not np.any(data):
                return True, "TAMAMEN_SIFIR"

            # ───────────────────────────────────────────────────────────────
            # CONSTANT ARRAY - Tüm değerler aynı = no information
            # ───────────────────────────────────────────────────────────────
            if np.all(data == data.flat[0]):
                return True, "SABIT_DEGER"

        except Exception as e:
            return True, f"BOZUK_NPY: {e}"

        return False, "OK"

    # Desteklenmeyen format - dokunma
    return False, "IGNORED"


def clean_dataset() -> dict[str, int]:
    """
    Dataset'i tarar ve bozuk dosyaları hedef klasöre taşır.

    Returns:
        dict[str, int]: Sebep başına taşınan dosya sayısı
    """
    os.makedirs(DEST_DIR, exist_ok=True)

    mode_str = "DRY-RUN (taşıma yapılmayacak)" if DRY_RUN else "LIVE"
    logger.info("=" * 70)
    logger.info(f"WAKEWORD DATASET CLEANER - {mode_str}")
    logger.info("=" * 70)
    logger.info(f"Kaynak : {SOURCE_DIR}")
    logger.info(f"Hedef  : {DEST_DIR}")
    logger.info("-" * 70)

    # İstatistikler
    stats: dict[str, int] = {}
    total_files = 0
    garbage_files = 0

    # Önce dosya sayısını al (progress için)
    all_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith((".wav", ".npy")):
                all_files.append(os.path.join(root, file))

    total_to_scan = len(all_files)
    logger.info(f"Taranacak dosya sayısı: {total_to_scan}")
    logger.info("-" * 70)

    # Ana tarama döngüsü
    for idx, full_path in enumerate(all_files, 1):
        total_files += 1

        # Progress (her 1000 dosyada bir)
        if idx % 1000 == 0:
            logger.info(f"İlerleme: {idx}/{total_to_scan} ({idx / total_to_scan * 100:.1f}%)")

        is_bad, reason = is_garbage(full_path)

        if is_bad:
            garbage_files += 1
            stats[reason] = stats.get(reason, 0) + 1

            # Dosya adı kısalt (log okunabilirliği için)
            short_path = os.path.relpath(full_path, SOURCE_DIR)
            logger.info(f"[{reason}] {short_path}")

            if not DRY_RUN:
                filename = os.path.basename(full_path)
                dest_path = os.path.join(DEST_DIR, filename)

                # Çakışma varsa UUID ekle
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(DEST_DIR, f"{name}_{uuid.uuid4().hex[:6]}{ext}")

                try:
                    shutil.move(full_path, dest_path)
                except Exception as e:
                    logger.error(f"TAŞIMA HATASI: {full_path} -> {e}")

    # Final rapor
    logger.info("=" * 70)
    logger.info("SONUÇ RAPORU")
    logger.info("=" * 70)
    logger.info(f"Toplam taranan  : {total_files}")
    logger.info(f"Bozuk bulunan   : {garbage_files}")
    logger.info(f"Temiz kalan     : {total_files - garbage_files}")
    logger.info(f"Bozuk oranı     : {garbage_files / max(total_files, 1) * 100:.2f}%")
    logger.info("-" * 70)

    if stats:
        logger.info("SEBEP DAĞILIMI:")
        for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")
    else:
        logger.info("Bozuk dosya bulunamadı!")

    if DRY_RUN:
        logger.info("-" * 70)
        logger.info("DRY-RUN modunda çalıştırıldı. Hiçbir dosya taşınmadı.")
        logger.info("Gerçek taşıma için: python move_all_zero_files.py")

    logger.info("=" * 70)

    return stats


if __name__ == "__main__":
    try:
        stats = clean_dataset()
        # Exit code: 0 başarılı, 1 bozuk bulundu
        sys.exit(0 if not stats else 0)
    except KeyboardInterrupt:
        logger.info("\nKullanıcı tarafından iptal edildi.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Beklenmeyen hata: {e}")
        sys.exit(1)
