# Pilih 1.500 kalimat seed set CLAIM/TIME dan tetapkan split-nya (docs/decisions/0007):
#   Komdigi = kalimat "Beredar ... yang mengeklaim ..." pertama tiap artikel
#   Mafindo = kalimat pertama berita benar Mafindo-2024 (split train saja, supaya tidak menyentuh test task 2)
#
#   split  dipilih  isi                                         label
#   train  1.545    845 Komdigi sebelum 1 Sep 2026 + 700 Mafindo   otomatis saja
#   dev       50     28 Komdigi sebelum 1 Sep 2026 +  22 Mafindo   otomatis, lalu dikoreksi manual
#   test     100     60 Komdigi sejak 1 Sep 2026  +  40 Mafindo   otomatis, lalu dikoreksi manual
# Train dipilih sedikit lebih banyak dari target 1.350 karena kalimat train yang tidak mendapat CLAIM
# dari aturan dibuang saat build (sekitar 88% kalimat mendapat CLAIM). Total terpakai sekitar 1.500.
# Kalimat Komdigi periode test (sejak 1 Sep 2026) hanya boleh masuk test; sisanya tidak dipakai.
#
# Butuh CSV Komdigi dari Kaggle (login sendiri):
#   kaggle datasets download ireddragonicy/indonesian-hoax-news-dataset -p data/raw/refuted_claims --unzip
# Jalankan:
#   python -m indonesian_hoax_checker.ner.seed_sentences

import argparse
import csv
from collections import Counter
import random
import re
import sys
from pathlib import Path

from indonesian_hoax_checker.ner.annotation import TEST_PERIOD_START, write_jsonl
from indonesian_hoax_checker.preprocessing.tokenizer import tokenize

# Akhir kalimat: . ! ? (boleh diikuti tanda kutip) lalu huruf kapital, dengan atau tanpa spasi
SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"'”’)]?\s*(?=[A-Z\"“])")
# Awalan artikel Komdigi, mis. "Penjelasan:" atau "KATEGORI: HOAKS Penjelasan :"
PREFIX = re.compile(r"^\s*(kategori\s*:\s*\w+\s*)?(penjelasan|faktanya|klarifikasi)\s*:\s*", re.IGNORECASE)
TITLE_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")  # "[HOAKS] ..." / "[Hoaks] ..."
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
                        # Judul tanpa "[HOAKS]" = ringkasan klaim, dipakai pelabel CLAIM otomatis
                        "title": TITLE_PREFIX.sub("", article.get("title", "")).strip(),
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
                "title": (article.get("title") or "").strip(),
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


# Jumlah kalimat per split dan sumber
COMPOSITION = {
    "train": {"komdigi": 845, "mafindo": 700},
    "dev": {"komdigi": 28, "mafindo": 22},
    "test": {"komdigi": 60, "mafindo": 40},
}


def in_test_period(row: dict) -> bool:
    return (row.get("published_at") or "") >= TEST_PERIOD_START


def _take(pool: list[dict], n: int, what: str) -> list[dict]:
    if len(pool) < n:
        print(f"peringatan: {what} hanya {len(pool)} kalimat, diminta {n}")
    taken = pool[:n]
    del pool[:n]
    return taken


def select(komdigi, mafindo, composition=COMPOSITION, seed=42) -> list[dict]:
    rng = random.Random(seed)
    komdigi, mafindo = dedup(komdigi), dedup(mafindo)
    for pool in (komdigi, mafindo):
        rng.shuffle(pool)
    komdigi_test = [r for r in komdigi if in_test_period(r)]
    komdigi_rest = [r for r in komdigi if not in_test_period(r)]

    rows = []
    for split in ("test", "dev", "train"):
        k_pool = komdigi_test if split == "test" else komdigi_rest
        rows += [{**r, "split": split} for r in _take(k_pool, composition[split]["komdigi"], f"Komdigi {split}")]
        rows += [{**r, "split": split} for r in _take(mafindo, composition[split]["mafindo"], f"Mafindo {split}")]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilih kalimat seed set untuk anotasi.")
    parser.add_argument("--komdigi-csv", default="data/raw/refuted_claims/komdigi_hoaks.csv")
    parser.add_argument("--mafindo-cache", default="data/raw/classifier/mafindo2024")
    parser.add_argument("--output", default="data/annotations/ner/seed_sentences.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.komdigi_csv).exists():
        sys.exit(f"{args.komdigi_csv} belum ada. Unduh dulu dari Kaggle (lihat komentar di atas file ini).")

    from datasets import load_dataset

    mafindo_train = load_dataset("umn-cyber/mafindo-2024", split="train", cache_dir=args.mafindo_cache)
    komdigi = komdigi_sentences(args.komdigi_csv)
    mafindo = mafindo_sentences(mafindo_train)
    print(f"kandidat: Komdigi {len(komdigi)}, Mafindo {len(mafindo)}")

    rows = select(komdigi, mafindo, seed=args.seed)
    write_jsonl(args.output, rows)
    counts = Counter((r["split"], r["source"]) for r in rows)
    print(f"{len(rows)} kalimat ditulis ke {args.output}: " + ", ".join(f"{s}/{src} {n}" for (s, src), n in sorted(counts.items())))


if __name__ == "__main__":
    main()
