# Error analysis NER dari file prediksi hasil ner.train (token, label gold, label prediksi).
# Tiap entitas gold dikelompokkan:
#   benar            batas dan tipe sama persis
#   salah_tipe       batas sama, tipe beda
#   salah_batas      tipe sama, batas beda (terpotong/kelebihan)
#   salah_keduanya   tumpang tindih, batas dan tipe beda
#   terlewat         tidak ada prediksi yang tumpang tindih
# Entitas prediksi yang tidak menyentuh entitas gold mana pun = salah_deteksi (false positive).
#
# Jalankan:
#   python -m indonesian_hoax_checker.ner.error_analysis outputs/predictions/ner_indobert_seed42.conll

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from indonesian_hoax_checker.ner.labels import get_spans

CATEGORIES = ["benar", "salah_tipe", "salah_batas", "salah_keduanya", "terlewat"]


def read_predictions(path) -> list[tuple[list[str], list[str], list[str]]]:
    sentences, rows = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            rows.append(line.split("\t"))
        elif rows:
            tokens, gold, pred = (list(col) for col in zip(*rows))
            sentences.append((tokens, gold, pred))
            rows = []
    return sentences


def overlaps(a, b) -> bool:
    return a[1] < b[2] and b[1] < a[2]


def classify(gold_span, pred_spans) -> tuple[str, tuple | None]:
    touching = [p for p in pred_spans if overlaps(gold_span, p)]
    if not touching:
        return "terlewat", None
    for p in touching:
        if p == gold_span:
            return "benar", p
    for p in touching:
        if p[1:] == gold_span[1:]:
            return "salah_tipe", p
    for p in touching:
        if p[0] == gold_span[0]:
            return "salah_batas", p
    return "salah_keduanya", touching[0]


def analyze(sentences, n_examples: int = 5) -> dict:
    counts = defaultdict(Counter)        # tipe gold -> kategori -> jumlah
    confusion = defaultdict(Counter)     # tipe gold -> tipe prediksi (untuk salah_tipe)
    spurious = Counter()                 # tipe prediksi yang salah deteksi
    examples = defaultdict(list)

    for tokens, gold, pred in sentences:
        gold_spans, pred_spans = get_spans(gold), get_spans(pred)
        for g in gold_spans:
            category, p = classify(g, pred_spans)
            counts[g[0]][category] += 1
            if category == "salah_tipe":
                confusion[g[0]][p[0]] += 1
            if category != "benar" and len(examples[category]) < n_examples:
                examples[category].append({
                    "kalimat": " ".join(tokens),
                    "gold": f"{' '.join(tokens[g[1]:g[2]])} [{g[0]}]",
                    "prediksi": f"{' '.join(tokens[p[1]:p[2]])} [{p[0]}]" if p else "-",
                })
        for p in pred_spans:
            if not any(overlaps(p, g) for g in gold_spans):
                spurious[p[0]] += 1
                if len(examples["salah_deteksi"]) < n_examples:
                    examples["salah_deteksi"].append({
                        "kalimat": " ".join(tokens), "gold": "-",
                        "prediksi": f"{' '.join(tokens[p[1]:p[2]])} [{p[0]}]",
                    })

    return {
        "per_tipe": {t: {c: counts[t][c] for c in CATEGORIES} for t in sorted(counts)},
        "salah_deteksi": dict(spurious),
        "salah_tipe_menjadi": {t: dict(c) for t, c in confusion.items()},
        "contoh": dict(examples),
    }


def to_markdown(report: dict) -> str:
    lines = ["| Tipe | " + " | ".join(CATEGORIES) + " | salah_deteksi |", "|---" * (len(CATEGORIES) + 2) + "|"]
    for kind, row in report["per_tipe"].items():
        cells = [str(row[c]) for c in CATEGORIES] + [str(report["salah_deteksi"].get(kind, 0))]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    for category, items in report["contoh"].items():
        lines.append(f"\n**{category}**")
        lines += [f"- {e['kalimat']}\n  - gold: {e['gold']}; prediksi: {e['prediksi']}" for e in items]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis NER dari file prediksi.")
    parser.add_argument("predictions", nargs="+", help="File outputs/predictions/*.conll")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    for path in args.predictions:
        report = analyze(read_predictions(path), args.examples)
        out = Path("outputs/metrics") / f"{Path(path).stem}_errors.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"## {Path(path).stem}\n\n{to_markdown(report)}\n")


if __name__ == "__main__":
    main()
