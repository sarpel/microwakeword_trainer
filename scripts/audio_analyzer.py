#!/usr/bin/env python3
"""
Wakeword Dataset Audio Analyzer
=============================

Purpose: Bu script wakeword dataset dizinlerindeki tüm ses dosyalarını analiz eder
ve dosya sayısı, toplam boyut ve süre bilgilerini terminale yazdırır.

Logic Flow:
1. Belirtilen dizinlerde recursive olarak ses dosyalarını arar
2. Her dosya için boyut ve süre bilgilerini toplar
3. Desteklenen formatları otomatik olarak tespit eder
4. Detaylı istatistikleri formatlı şekilde gösterir

Edge Cases Handled:
- Bozuk/okunamayan ses dosyaları: hata yakalama ile atlanır
- Desteklenmeyen formatlar: bilgi verilir ama analiz devam eder
- Büyük dosyalar: efficient memory kullanımı için metadata-only okuma

Learning Note: librosa kullanarak ses dosyalarının metadata bilgilerini
verimli şekilde okuyoruz, dosyanın tamamını memory'ye yüklemeden.
"""

import argparse
import glob
import os
import sys
import time
from typing import Dict, List

# Ses dosyası işleme için gerekli kütüphaneler
try:
    import librosa
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("UYARI: librosa ve soundfile kütüphaneleri bulunamadı.")
    print("Sadece dosya boyutu analizi yapılacak, süre hesaplanamayacak.")
    print("Kurulum için: pip install librosa soundfile")
    print()


def format_size(size_bytes: float) -> str:
    """
    Purpose: Byte cinsinden boyutu human-readable formata çevirir

    Logic Flow:
    1. Byte değerini uygun birime böler (KB, MB, GB, TB)
    2. Formatlanmış string döndürür

    Args:
        size_bytes: Byte cinsinden dosya boyutu

    Returns:
        Formatlanmış boyut string'i (örn: "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Purpose: Saniye cinsinden süreyi okunabilir formata çevirir

    Logic Flow:
    1. Toplam saniyeyi saat, dakika, saniyeye böler
    2. Uygun formatı seçer (sadece saniye, dakika:saniye, saat:dakika:saniye)

    Args:
        seconds: Saniye cinsinden süre

    Returns:
        Formatlanmış süre string'i
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_audio_duration(file_path: str) -> float:
    """
    Purpose: Ses dosyasının süresini saniye cinsinden döndürür

    Logic Flow:
    1. Önce soundfile ile hızlı metadata okuma dener
    2. Başarısız olursa librosa ile dener
    3. Her iki yöntem de başarısız olursa 0 döndürür

    Args:
        file_path: Analiz edilecek ses dosyasının yolu

    Returns:
        Süre (saniye) veya hata durumunda 0
    """
    if not AUDIO_LIBS_AVAILABLE:
        return 0

    try:
        # Önce soundfile ile hızlı metadata okuma
        info = sf.info(file_path)
        return info.duration
    except Exception:
        try:
            # soundfile başarısız olursa librosa dene
            duration = librosa.get_duration(path=file_path)
            return duration
        except Exception:
            # Her iki yöntem de başarısız olursa
            return 0


def find_audio_files(directory: str, extensions: List[str]) -> List[str]:
    """
    Purpose: Belirtilen dizinde recursive olarak ses dosyalarını bulur

    Logic Flow:
    1. Desteklenen uzantılar için glob pattern oluşturur
    2. Recursive arama yapar (**/* pattern ile)
    3. Bulunan dosyaları liste olarak döndürür

    Args:
        directory: Aranacak ana dizin
        extensions: Aranacak dosya uzantıları listesi

    Returns:
        Bulunan ses dosyalarının yolları listesi
    """
    audio_files = []

    for ext in extensions:
        # Hem büyük hem küçük harf için arama yap
        pattern = os.path.join(directory, "**", f"*.{ext.lower()}")
        audio_files.extend(glob.glob(pattern, recursive=True))

        pattern = os.path.join(directory, "**", f"*.{ext.upper()}")
        audio_files.extend(glob.glob(pattern, recursive=True))

    # Duplikatları kaldır ve sırala
    return sorted(set(audio_files))


def analyze_directory(directory: str, show_details: bool = False) -> Dict:
    """
    Purpose: Belirtilen dizindeki tüm ses dosyalarını analiz eder

    Logic Flow:
    1. Desteklenen ses formatlarını tanımlar
    2. Dizinde recursive arama yapar
    3. Her dosya için boyut ve süre bilgilerini toplar
    4. İstatistikleri hesaplar ve döndürür

    Args:
        directory: Analiz edilecek dizin
        show_details: Detaylı dosya listesi gösterilsin mi

    Returns:
        Analiz sonuçlarını içeren dictionary
    """
    # Desteklenen ses dosyası formatları
    audio_extensions = ["wav", "mp3", "flac", "ogg", "aac", "m4a", "wma"]

    print(f"🔍 Dizin analiz ediliyor: {directory}")
    print(f"📁 Desteklenen formatlar: {', '.join(audio_extensions)}")
    print("⏳ Lütfen bekleyin...")
    print()

    # Ses dosyalarını bul
    audio_files = find_audio_files(directory, audio_extensions)

    if not audio_files:
        return {"directory": directory, "total_files": 0, "total_size": 0, "total_duration": 0, "files": []}

    total_size = 0.0
    total_duration = 0.0
    analyzed_files = []
    errors = []

    print(f"📊 {len(audio_files)} dosya bulundu, analiz başlıyor...")

    # Progress bar için
    for i, file_path in enumerate(audio_files, 1):
        try:
            # Dosya boyutunu al
            file_size = os.path.getsize(file_path)
            total_size += file_size

            # Ses süresini al (eğer kütüphaneler mevcutsa)
            duration = get_audio_duration(file_path)
            total_duration += duration

            # Dosya bilgilerini kaydet
            file_info = {"path": file_path, "name": os.path.basename(file_path), "size": file_size, "duration": duration, "extension": os.path.splitext(file_path)[1].lower()}
            analyzed_files.append(file_info)

            # Progress göster (her 50 dosyada bir)
            if i % 50 == 0 or i == len(audio_files):
                percent = (i / len(audio_files)) * 100
                print(f"⚡ İlerleme: {i}/{len(audio_files)} (%{percent:.1f})")

        except Exception as e:
            error_msg = f"Hata - {file_path}: {str(e)}"
            errors.append(error_msg)
            if show_details:
                print(f"❌ {error_msg}")

    print("✅ Analiz tamamlandı!")
    print()

    if errors:
        print(f"⚠️  {len(errors)} dosya analiz edilemedi.")
        if show_details:
            for error in errors[:5]:  # İlk 5 hatayı göster
                print(f"   {error}")
            if len(errors) > 5:
                print(f"   ... ve {len(errors) - 5} hata daha")
        print()

    return {"directory": directory, "total_files": len(analyzed_files), "total_size": total_size, "total_duration": total_duration, "files": analyzed_files, "errors": errors}


def print_statistics(analysis_result: Dict, show_file_list: bool = False):
    """
    Purpose: Analiz sonuçlarını formatlı şekilde terminale yazdırır

    Logic Flow:
    1. Başlık ve genel istatistikleri yazdırır
    2. Format bazında breakdown gösterir
    3. İsteğe bağlı dosya listesi detayını gösterir

    Args:
        analysis_result: analyze_directory'den dönen sonuç dictionary'si
        show_file_list: Detaylı dosya listesi gösterilsin mi
    """
    result = analysis_result

    print("=" * 70)
    print("🎵 WAKEWORD DATASET ANALİZ SONUÇLARI 🎵")
    print("=" * 70)
    print()

    print(f"📂 Dizin: {result['directory']}")
    print(f"📁 Toplam dosya sayısı: {result['total_files']:,}")
    print(f"💾 Toplam boyut: {format_size(result['total_size'])}")

    if AUDIO_LIBS_AVAILABLE and result["total_duration"] > 0:
        print(f"⏱️  Toplam süre: {format_duration(result['total_duration'])}")
        avg_duration = result["total_duration"] / result["total_files"] if result["total_files"] > 0 else 0
        print(f"📊 Ortalama dosya süresi: {format_duration(avg_duration)}")
    else:
        print("⏱️  Toplam süre: Hesaplanamadı (librosa/soundfile gerekli)")

    print()

    # Format bazında breakdown
    if result["files"]:
        format_stats = {}
        for file_info in result["files"]:
            ext = file_info["extension"]
            if ext not in format_stats:
                format_stats[ext] = {"count": 0, "size": 0, "duration": 0}

            format_stats[ext]["count"] += 1
            format_stats[ext]["size"] += file_info["size"]
            format_stats[ext]["duration"] += file_info["duration"]

        print("📋 FORMAT BAZINDA DETAYLAR:")
        print("-" * 50)
        for ext, stats in sorted(format_stats.items()):
            print(f"{ext.upper():<6} | Dosya: {stats['count']:>6,} | Boyut: {format_size(stats['size']):>10} | Süre: {format_duration(stats['duration']):>12}")
        print()

    # Detaylı dosya listesi (isteğe bağlı)
    if show_file_list and result["files"]:
        print("📄 DETAYLI DOSYA LİSTESİ:")
        print("-" * 70)
        for file_info in result["files"][:20]:  # İlk 20 dosyayı göster
            rel_path = os.path.relpath(file_info["path"], result["directory"])
            duration_str = format_duration(file_info["duration"]) if file_info["duration"] > 0 else "N/A"
            print(f"{rel_path:<50} | {format_size(file_info['size']):>8} | {duration_str:>8}")

        if len(result["files"]) > 20:
            print(f"... ve {len(result['files']) - 20} dosya daha")
        print()


def main():
    """
    Purpose: Ana program entry point'i - komut satırı argümanlarını işler

    Logic Flow:
    1. Argparse ile komut satırı parametrelerini parse eder
    2. Belirtilen dizin(ler)i analiz eder
    3. Sonuçları yazdırır
    """
    parser = argparse.ArgumentParser(
        description="Wakeword dataset ses dosyalarını analiz eder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python audio_analyzer.py "D:/MCP/Precise-Community-Data"
  python audio_analyzer.py -d "D:/Dataset" --details
  python audio_analyzer.py -m "D:/Data1" "D:/Data2" "D:/Data3"
        """,
    )

    parser.add_argument("directories", nargs="*", default=["D:/MCP/Precise-Community-Data"], help="Analiz edilecek dizin(ler) (varsayılan: D:/MCP/Precise-Community-Data)")

    parser.add_argument("-d", "--details", action="store_true", help="Detaylı dosya listesi ve hata mesajlarını göster")

    parser.add_argument("-m", "--multiple", action="store_true", help="Birden fazla dizini ayrı ayrı analiz et")

    args = parser.parse_args()

    # Kütüphane kontrolü
    if not AUDIO_LIBS_AVAILABLE:
        print("⚠️ UYARI: Ses analizi kütüphaneleri eksik!")
        print("Sadece dosya sayısı ve boyut bilgileri gösterilecek.")
        print("Süre hesaplama için: pip install librosa soundfile")
        print()

    total_results = {"total_files": 0, "total_size": 0, "total_duration": 0, "directories_analyzed": 0}

    # Başlangıç zamanı
    start_time = time.time()

    for directory in args.directories:
        if not os.path.exists(directory):
            print(f"❌ Hata: Dizin bulunamadı: {directory}")
            continue

        print(f"\n🚀 Analiz başlıyor: {directory}")
        print("=" * 70)

        # Dizini analiz et
        result = analyze_directory(directory, args.details)

        # Sonuçları yazdır
        print_statistics(result, args.details)

        # Toplam istatistikleri güncelle
        total_results["total_files"] += result["total_files"]
        total_results["total_size"] += result["total_size"]
        total_results["total_duration"] += result["total_duration"]
        total_results["directories_analyzed"] += 1

        if args.multiple and len(args.directories) > 1:
            print("\n" + "=" * 70 + "\n")

    # Birden fazla dizin analiz edildiyse özet göster
    if total_results["directories_analyzed"] > 1:
        print("🌟 GENEL ÖZET 🌟")
        print("=" * 70)
        print(f"📂 Analiz edilen dizin sayısı: {total_results['directories_analyzed']}")
        print(f"📁 Toplam dosya sayısı: {total_results['total_files']:,}")
        print(f"💾 Toplam boyut: {format_size(total_results['total_size'])}")
        if AUDIO_LIBS_AVAILABLE and total_results["total_duration"] > 0:
            print(f"⏱️  Toplam süre: {format_duration(total_results['total_duration'])}")
        print()

    # Çalışma süresi
    elapsed_time = time.time() - start_time
    print(f"⚡ Analiz süresi: {format_duration(elapsed_time)}")
    print("✨ Analiz tamamlandı!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Kullanıcı tarafından iptal edildi.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        sys.exit(1)
