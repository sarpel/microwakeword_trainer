import argparse
import re
import sys


def main():
    parser = argparse.ArgumentParser(description="Bir Python dosyasında 'box=None' ifadelerini arar.")
    parser.add_argument("file_path", nargs="?", default="src/training/rich_logger.py", help="Taranacak dosya yolu (varsayılan: src/training/rich_logger.py)")
    args = parser.parse_args()

    try:
        with open(args.file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Hata: '{args.file_path}' dosyası bulunamadı.", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Hata: Dosya okunamadı: {e}", file=sys.stderr)
        sys.exit(1)

    matches = list(re.finditer(r"box=None", content))
    print(f"'{args.file_path}' dosyasında box=None araması:")
    print(f"Toplam {len(matches)} eşleşme bulundu.")
    for m in matches:
        print(f"  Pozisyon: {m.start()}")

    sys.exit(0)


if __name__ == "__main__":
    main()
