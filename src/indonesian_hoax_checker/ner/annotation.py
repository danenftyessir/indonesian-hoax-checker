# Alat bantu anotasi seed set CLAIM/TIME dengan Label Studio.
#
# Alur:
#   1. tasks  : seed_sentences.jsonl -> file impor Label Studio, sudah berisi pra-anotasi
#               (TIME dari aturan, PER/ORG/LOC dari model baseline bila --checkpoint diisi)
#   2. kappa  : hasil ekspor Label Studio (JSON) -> Cohen's kappa per label antar annotator
#   3. build  : ekspor -> data/annotations/ner/seed_adjudicated.jsonl (dilacak Git)
#   4. split  : seed_adjudicated.jsonl -> data/processed/ner/seed_{train,dev,test}.conll
#
# Contoh:
#   python -m indonesian_hoax_checker.ner.annotation tasks
#   python -m indonesian_hoax_checker.ner.annotation kappa --export export.json
#   python -m indonesian_hoax_checker.ner.annotation build --export export.json --adjudicator jonathan@email
#   python -m indonesian_hoax_checker.ner.annotation split

import argparse
import itertools
import json
import random
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from indonesian_hoax_checker.ner.dataset import write_conll
from indonesian_hoax_checker.ner.labels import TYPES, get_spans
from indonesian_hoax_checker.ner.time_rules import add_time_tags
from indonesian_hoax_checker.preprocessing.tokenizer import tokenize

ANNOTATION_DIR = Path("data/annotations/ner")
TEST_PERIOD_START = "2026-09-01"  # docs/decisions/0008


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ---------- 1. Tugas Label Studio ----------

def label_studio_result(text, tokens, tags):
    results = []
    for kind, first, last in get_spans(tags):
        start, end = tokens[first][1], tokens[last - 1][2]
        results.append({
            "from_name": "label", "to_name": "text", "type": "labels",
            "value": {"start": start, "end": end, "text": text[start:end], "labels": [kind]},
        })
    return results


# predictor (opsional) = NERPredictor baseline untuk pra-anotasi PER/ORG/LOC
def make_tasks(sentences: list[dict], predictor=None) -> list[dict]:
    tasks = []
    for row in sentences:
        tokens = tokenize(row["text"])
        words = [t for t, _, _ in tokens]
        tags = predictor.tag([words])[0] if predictor and words else ["O"] * len(words)
        tags = add_time_tags(words, tags)
        tasks.append({
            "data": row,
            "predictions": [{"model_version": "pra-anotasi", "result": label_studio_result(row["text"], tokens, tags)}],
        })
    return tasks


# ---------- 2-3. Baca ekspor, kappa, adjudikasi ----------

def annotator_name(annotation) -> str:
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        return str(who.get("email") or who.get("id"))
    return str(who)


# Ubah span karakter hasil anotasi menjadi label IOB2 per token.
# Token ikut span bila tumpang tindih dengan span.
def spans_to_tags(tokens, results) -> list[str]:
    tags = ["O"] * len(tokens)
    spans = sorted(
        (r["value"]["start"], r["value"]["end"], r["value"]["labels"][0])
        for r in results if r.get("type") == "labels"
    )
    for start, end, kind in spans:
        inside = [i for i, (_, s, e) in enumerate(tokens) if s < end and e > start]
        for n, i in enumerate(inside):
            tags[i] = ("B-" if n == 0 else "I-") + kind
    return tags


# Hasil: {sentence_id: {"row": data, "tokens": [...], "by": {annotator: tags}}}
def read_export(path) -> dict:
    items = {}
    for task in json.loads(Path(path).read_text(encoding="utf-8")):
        row = task["data"]
        tokens = tokenize(row["text"])
        by = {}
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            by[annotator_name(annotation)] = spans_to_tags(tokens, annotation["result"])
        items[row["id"]] = {"row": row, "tokens": [t for t, _, _ in tokens], "by": by}
    return items


def _types(tags):
    return [t[2:] if t != "O" else "O" for t in tags]


# Kappa per label (token masuk label itu atau tidak) dan keseluruhan, rata-rata semua pasangan annotator
def kappa_report(items: dict) -> dict:
    annotators = sorted({a for item in items.values() for a in item["by"]})
    report = {"annotators": annotators, "pairs": {}}
    for a, b in itertools.combinations(annotators, 2):
        shared = [item for item in items.values() if a in item["by"] and b in item["by"]]
        ta = [t for item in shared for t in _types(item["by"][a])]
        tb = [t for item in shared for t in _types(item["by"][b])]
        if not ta:
            continue
        pair = {"sentences": len(shared), "overall": cohen_kappa_score(ta, tb)}
        for kind in TYPES:
            ya, yb = [t == kind for t in ta], [t == kind for t in tb]
            pair[kind] = cohen_kappa_score(ya, yb) if any(ya) or any(yb) else None
        report["pairs"][f"{a} vs {b}"] = pair

    pairs = list(report["pairs"].values())
    report["mean"] = {}
    for key in ["overall"] + TYPES:
        values = [p[key] for p in pairs if p.get(key) is not None]
        report["mean"][key] = sum(values) / len(values) if values else None
    return report


# Versi final tiap kalimat: anotasi adjudikator bila ada, atau anotasi yang sama persis dari semua annotator
def adjudicate(items: dict, adjudicator: str | None = None) -> tuple[list[dict], list[str]]:
    final, unresolved = [], []
    for sid, item in items.items():
        by = item["by"]
        if not by:
            continue
        if adjudicator and adjudicator in by:
            tags = by[adjudicator]
        elif len({tuple(t) for t in by.values()}) == 1:
            tags = next(iter(by.values()))
        else:
            unresolved.append(sid)
            continue
        final.append({**item["row"], "tokens": item["tokens"], "tags": tags})
    return final, unresolved


# ---------- 4. Split seed set ----------

# Kalimat Komdigi dari periode test (sejak 1 Sep 2026) wajib masuk test (docs/decisions/0007).
# Kalimat informal buatan tim (source "informal") tidak masuk di sini; dipisah jadi set uji sendiri.
def split_seed(rows: list[dict], n_dev=50, n_test=50, seed=42):
    rows = [r for r in rows if r.get("source") != "informal"]

    def in_test_period(r):
        return r.get("source") == "komdigi" and (r.get("published_at") or "") >= TEST_PERIOD_START

    forced = [r for r in rows if in_test_period(r)]
    rest = [r for r in rows if not in_test_period(r)]
    random.Random(seed).shuffle(rest)
    n_fill = max(0, n_test - len(forced))
    test = forced + rest[:n_fill]
    dev = rest[n_fill:n_fill + n_dev]
    train = rest[n_fill + n_dev:]
    return train, dev, test


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Alat bantu anotasi seed set NER.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tasks", help="Buat file impor Label Studio dengan pra-anotasi")
    p.add_argument("--input", nargs="+", default=[str(ANNOTATION_DIR / "seed_sentences.jsonl")],
                   help="Satu atau lebih file kalimat (mis. seed_sentences.jsonl informal_sentences.jsonl)")
    p.add_argument("--batch", help="Hanya kalimat dengan batch ini, mis. calibration atau main")
    p.add_argument("--output", default=str(ANNOTATION_DIR / "label_studio_tasks.json"))
    p.add_argument("--checkpoint", help="Checkpoint NER baseline untuk pra-anotasi PER/ORG/LOC")

    p = sub.add_parser("kappa", help="Hitung Cohen's kappa dari ekspor Label Studio")
    p.add_argument("--export", required=True)

    p = sub.add_parser("build", help="Gabungkan anotasi menjadi seed_adjudicated.jsonl")
    p.add_argument("--export", required=True)
    p.add_argument("--adjudicator", help="Nama/email annotator yang anotasinya dipakai saat berbeda")
    p.add_argument("--output", default=str(ANNOTATION_DIR / "seed_adjudicated.jsonl"))

    p = sub.add_parser("split", help="Bagi seed set menjadi train/dev/test CoNLL")
    p.add_argument("--input", default=str(ANNOTATION_DIR / "seed_adjudicated.jsonl"))
    p.add_argument("--out-dir", default="data/processed/ner")

    args = parser.parse_args()

    if args.command == "tasks":
        predictor = None
        if args.checkpoint:
            from indonesian_hoax_checker.ner.predict import NERPredictor

            predictor = NERPredictor(args.checkpoint)
        sentences = [row for path in args.input for row in read_jsonl(path)]
        if args.batch:
            sentences = [row for row in sentences if row.get("batch") == args.batch]
        tasks = make_tasks(sentences, predictor)
        Path(args.output).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(tasks)} tugas ditulis ke {args.output}")

    elif args.command == "kappa":
        report = kappa_report(read_export(args.export))
        print(json.dumps(report, indent=2))
        claim = report["mean"].get("CLAIM")
        if claim is not None:
            print(f"Kappa CLAIM rata-rata {claim:.3f}: " + ("lanjut anotasi" if claim >= 0.6 else "revisi pedoman (< 0,6)"))

    elif args.command == "build":
        final, unresolved = adjudicate(read_export(args.export), args.adjudicator)
        write_jsonl(args.output, final)
        print(f"{len(final)} kalimat final ditulis ke {args.output}")
        if unresolved:
            print(f"{len(unresolved)} kalimat masih berbeda dan perlu adjudikasi: {unresolved}")

    elif args.command == "split":
        rows = read_jsonl(args.input)
        train, dev, test = split_seed(rows)
        informal = [r for r in rows if r.get("source") == "informal"]
        splits = [("train", train), ("dev", dev), ("test", test)] + ([("informal_test", informal)] if informal else [])
        for name, part in splits:
            write_conll(Path(args.out_dir) / f"seed_{name}.conll", [(r["tokens"], r["tags"]) for r in part])
        print(", ".join(f"seed {name} {len(part)}" for name, part in splits))


if __name__ == "__main__":
    main()
