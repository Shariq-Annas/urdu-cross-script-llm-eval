"""Download and cache the four benchmark and parallel corpus datasets."""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DIRECT_FILES: dict[str, str] = {
    "belebele/urd_Arab.jsonl": (
        "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/urd_Arab.jsonl"
    ),
    "belebele/eng_Latn.jsonl": (
        "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/eng_Latn.jsonl"
    ),
    "urdu_idioms/urdu_idiom_sentences.xlsx": (
        "https://raw.githubusercontent.com/Mousumi44/Urdu-Idioms/main/data/urdu_idiom_sentences.xlsx"
    ),
    "urdu_idioms/roman_idiom_sentences.xlsx": (
        "https://raw.githubusercontent.com/Mousumi44/Urdu-Idioms/main/data/roman_idiom_sentences.xlsx"
    ),
    "dakshina/ur.parquet": (
        "https://huggingface.co/datasets/Anvesh-Lankala/"
        "Copy_Dakshina_Google_research_dataset/resolve/main/data/ur-00000-of-00001.parquet"
    ),
    "roman_urdu_parl/small_validation_set.csv": (
        "https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split/"
        "resolve/main/small_validation_set.csv"
    ),
    "roman_urdu_parl/small_test_set.csv": (
        "https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split/"
        "resolve/main/small_test_set.csv"
    ),
    "roman_urdu_parl/test_set.csv": (
        "https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split/"
        "resolve/main/test_set.csv"
    ),
}

ROMAN_URDU_PARL_TRAIN_URL = (
    "https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split/resolve/main/train_set.csv"
)


def download_file(url: str, dest: Path, force: bool = False) -> None:
    """Download a URL to dest atomically if missing or force=True."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dest.relative_to(RAW_DIR)} ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1024 * 256):
            out.write(chunk)
    tmp.replace(dest)
    print(f"[ok]   {dest.relative_to(RAW_DIR)} ({dest.stat().st_size:,} bytes)")


def stream_roman_urdu_parl_train(
    dest: Path, max_rows: int = 200_000, force: bool = False
) -> None:
    """Stream the first max_rows rows of Roman-Urdu-Parl train_set.csv without downloading 1.13GB."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dest.relative_to(RAW_DIR)} ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    req = urllib.request.Request(ROMAN_URDU_PARL_TRAIN_URL, headers={"User-Agent": "Mozilla/5.0"})
    written = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text_stream)
        with tmp.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            header = next(reader, None)
            if header:
                writer.writerow(header)
            for row in reader:
                if len(row) >= 2:
                    writer.writerow(row[:2])
                    written += 1
                    if written >= max_rows:
                        break
    tmp.replace(dest)
    print(f"[ok]   {dest.relative_to(RAW_DIR)} ({written:,} rows, {dest.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for cross-script Urdu evaluation.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument(
        "--parl-train-rows",
        type=int,
        default=200_000,
        help="Max rows to stream from Roman-Urdu-Parl train_set.csv.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path, url in DIRECT_FILES.items():
        download_file(url, RAW_DIR / rel_path, force=args.force)

    if args.parl_train_rows > 0:
        stream_roman_urdu_parl_train(
            RAW_DIR / "roman_urdu_parl" / f"train_sample_{args.parl_train_rows}.csv",
            max_rows=args.parl_train_rows,
            force=args.force,
        )


if __name__ == "__main__":
    main()
