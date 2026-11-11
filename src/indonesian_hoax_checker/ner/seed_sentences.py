# Pilih kalimat untuk seed set anotasi CLAIM/TIME (docs/decisions/0007):
#   300 kalimat "Beredar ... yang mengeklaim ..." dari Komdigi (satu per artikel, acak dari seluruh tanggal)
#   100 kalimat pertama berita benar dari Mafindo-2024 (split train saja, supaya tidak menyentuh test task 2)
# 100 kalimat pertama (75 Komdigi + 25 Mafindo) ditandai batch "calibration" untuk hitung kappa.
#
# Butuh CSV Komdigi dari Kaggle (login sendiri):
#   kaggle datasets download ireddragonicy/indonesian-hoax-news-dataset -p data/raw/refuted_claims --unzip
# Jalankan:
#   python -m indonesian_hoax_checker.ner.seed_sentences

import argparse
import csv
import random
import re
import sys
from pathlib import Path

from indonesian_hoax_checker.ner.annotation import write_jsonl
from indonesian_hoax_checker.preprocessing.tokenizer import tokenize

# Akhir kalimat: . ! ? (boleh diikuti tanda kutip) lalu huruf kapital, dengan atau tanpa spasi
SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"'”’)]?\s*(?=[A-Z\"“])")
PREFIX = re.compile(r"^\s*(penjelasan|faktanya|klarifikasi)\s*:\s*", re.IGNORECASE)
MIN_TOKENS, MAX_TOKENS = 6, 60
csv.field_size_limit(sys.maxsize)


def _quotes_open(text: str) -> bool:
    return text.count("“") != text.count("”") or text.count('"') % 2 == 1


# Pecahan yang tanda kutipnya belum tertutup digabung dengan pecahan berikutnya,
# supaya kalimat seperti: narasi “HEBOH!! Pemerintah ...” tidak terpotong di tengah kutipan
def split_text(text: str) -> list[str]:
    sentences, current = [], ""
    for piece in SENTENCE_BREAK.split(text or ""):
        current = f"{current} {piece}".strip() if current else piece.strip()
        if current and not _quotes_open(current):
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)
    return sentences


def good_length(sentence: str) -> bool:
    return MIN_TOKENS <= len(tokenize(sentence)) <= MAX_TOKENS


# Kalimat "Beredar ..." pertama dari tiap artikel Komdigi
def komdigi_sentences(csv_path) -> list[dict]:
    rows = []
    # utf-8-sig: CSV Kaggle diawali BOM, tanpa ini kolom pertama terbaca "\ufeffid"
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for article in csv.DictReader(f):
            for sentence in split_text(article.get("body_text", "")):
                sentence = PREFIX.sub("", sentence)
                if "beredar" in sentence.lower() and good_length(sentence):
                    rows.append({
                        "id": f"komdigi-{article['id']}",
                        "text": sentence,
                        "source": "komdigi",
                        "origin_id": str(article["id"]),
                        "published_at": (article.get("published_at") or "")[:10] or None,
                    })
                    break
    return rows


# Kalimat pertama artikel valid (is_hoax = 0); articles = baris dataset dengan title/content/is_hoax
def mafindo_sentences(articles) -> list[dict]:
    rows = []
    for i, article in enumerate(articles):
        if str(article["is_hoax"]) != "0":
            continue
        sentences = split_text(article["content"])
        if sentences and good_length(sentences[0]):
            rows.append({
                "id": f"mafindo-train-{i}",
                "text": sentences[0],
                "source": "mafindo",
                "origin_id": f"train-{i}",
                "published_at": None,
            })
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    seen, kept = set(), []
    for row in rows:
        key = row["text"].lower()
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def select(komdigi, mafindo, n_komdigi=300, n_mafindo=100, n_calibration=100, seed=42) -> list[dict]:
    rng = random.Random(seed)
    komdigi = rng.sample(dedup(komdigi), min(n_komdigi, len(komdigi)))
    mafindo = rng.sample(dedup(mafindo), min(n_mafindo, len(mafindo)))

    # Batch kalibrasi mengikuti proporsi sumber (300:100 -> 75:25)
    total = len(komdigi) + len(mafindo)
    n_cal_k = round(n_calibration * len(komdigi) / total) if total else 0
    n_cal_m = n_calibration - n_cal_k
    calibration = komdigi[:n_cal_k] + mafindo[:n_cal_m]
    main = komdigi[n_cal_k:] + mafindo[n_cal_m:]
    rng.shuffle(calibration)
    rng.shuffle(main)
    return [{**r, "batch": "calibration"} for r in calibration] + [{**r, "batch": "main"} for r in main]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilih kalimat seed set untuk anotasi.")
    parser.add_argument("--komdigi-csv", default="data/raw/refuted_claims/komdigi_hoaks.csv")
    parser.add_argument("--mafindo-cache", default="data/raw/classifier/mafindo2024")
    parser.add_argument("--output", default="data/annotations/ner/seed_sentences.jsonl")
    parser.add_argument("--n-komdigi", type=int, default=300)
    parser.add_argument("--n-mafindo", type=int, default=100)
    parser.add_argument("--n-calibration", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.komdigi_csv).exists():
        sys.exit(f"{args.komdigi_csv} belum ada. Unduh dulu dari Kaggle (lihat komentar di atas file ini).")

    from datasets import load_dataset

    mafindo_train = load_dataset("umn-cyber/mafindo-2024", split="train", cache_dir=args.mafindo_cache)
    komdigi = komdigi_sentences(args.komdigi_csv)
    mafindo = mafindo_sentences(mafindo_train)
    print(f"kandidat: Komdigi {len(komdigi)}, Mafindo {len(mafindo)}")

    rows = select(komdigi, mafindo, args.n_komdigi, args.n_mafindo, args.n_calibration, args.seed)
    write_jsonl(args.output, rows)
    print(f"{len(rows)} kalimat ditulis ke {args.output}")


if __name__ == "__main__":
    main()
