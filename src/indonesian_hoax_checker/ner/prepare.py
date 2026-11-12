# Langkah 2 task NER: unduh NERGrit dan NERP, petakan label, dedup, lalu simpan sebagai CoNLL.
# Jalankan: python -m indonesian_hoax_checker.ner.prepare
#
# Hasil di data/processed/ner/:
#   nergrit_{train,dev,test}.conll     data utama (cased)
#   *_test_informal.conll              versi informal sintetis dari test (RQ1)
#   nerp_lower_{train,dev,test}.conll  hanya untuk varian lowercase RQ1
#   stats.json                         jumlah kalimat dan span
# test = file valid IndoNLU (label test asli disembunyikan), dev = 10% train untuk early stopping.

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

from indonesian_hoax_checker.ner.dataset import Sentence, dedup, read_conll, write_conll
from indonesian_hoax_checker.ner.labels import NERGRIT_MAPPING, NERP_MAPPING, get_spans, map_tag, to_iob2
from indonesian_hoax_checker.preprocessing.augment import TEST_PROBS, augment_corpus

SOURCES = {"nergrit": ("nergrit", NERGRIT_MAPPING), "nerp": ("nerp_lower", NERP_MAPPING)}


def download(base_url: str, filename: str, raw_dir: Path) -> Path:
    target = raw_dir / filename
    if not target.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{base_url}/{filename}", target)
    return target


def convert(sentences: list[Sentence], mapping: dict[str, str]) -> list[Sentence]:
    return [(tokens, to_iob2([map_tag(t, mapping) for t in tags])) for tokens, tags in sentences]


def split_dev(sentences: list[Sentence], dev_ratio: float, seed: int) -> tuple[list[Sentence], list[Sentence]]:
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    n_dev = round(len(sentences) * dev_ratio)
    dev_idx = set(order[:n_dev])
    train = [s for i, s in enumerate(sentences) if i not in dev_idx]
    dev = [s for i, s in enumerate(sentences) if i in dev_idx]
    return train, dev


def describe(sentences: list[Sentence]) -> dict:
    spans = Counter(kind for _, tags in sentences for kind, _, _ in get_spans(tags))
    return {"sentences": len(sentences), "spans": dict(sorted(spans.items()))}


def prepare(config_path: str, out_dir: str, dev_ratio: float, seed: int) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())["ner"]
    stats = {}
    for source, (out_name, mapping) in SOURCES.items():
        info = config[source]
        raw_dir = Path(info["raw_dir"])
        train_raw = read_conll(download(info["base_url"], config["files"]["train"], raw_dir))
        valid_raw = read_conll(download(info["base_url"], config["files"]["valid"], raw_dir))

        # Dedup di dalam train, lalu buang kalimat valid yang juga ada di train (kebocoran)
        train = dedup(convert(train_raw, mapping))
        test = dedup(convert(valid_raw, mapping), exclude=train)
        train, dev = split_dev(train, dev_ratio, seed)

        # Set uji informal sintetis (RQ1): versi informal dari test, seed tetap supaya sama untuk semua run
        test_informal = augment_corpus(test, seed=2026, probs=TEST_PROBS)
        for split, data in (("train", train), ("dev", dev), ("test", test), ("test_informal", test_informal)):
            write_conll(Path(out_dir) / f"{out_name}_{split}.conll", data)
        stats[out_name] = {
            "raw": {"train": len(train_raw), "valid": len(valid_raw)},
            "removed": {
                "train_duplicates": len(train_raw) - len(train) - len(dev),
                "valid_duplicates_or_in_train": len(valid_raw) - len(test),
            },
            "train": describe(train),
            "dev": describe(dev),
            "test": describe(test),
        }

    out_path = Path(out_dir) / "stats.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan data NER (NERGrit, NERP).")
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument("--out-dir", default="data/processed/ner")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = prepare(args.config, args.out_dir, args.dev_ratio, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
