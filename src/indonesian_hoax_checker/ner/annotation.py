# Alat bantu seed set CLAIM/TIME: label otomatis, koreksi manual di Label Studio, lalu data training.
#
# Alur (docs/annotation_guidelines.md):
#   1. auto   : seed_sentences.jsonl -> seed_auto.jsonl, label otomatis untuk SEMUA kalimat
#               (PER/ORG/LOC dari model NER tahap 1, TIME dan CLAIM dari aturan)
#   2. tasks  : seed_auto.jsonl -> file impor Label Studio, hanya split dev dan test (yang dikoreksi)
#   3. report : ekspor Label Studio -> seberapa sering label otomatis dikoreksi (kualitas label otomatis)
#   4. build  : ekspor + seed_auto.jsonl -> seed_adjudicated.jsonl
#               (dev/test = hasil koreksi manusia, train = label otomatis)
#   5. split  : seed_adjudicated.jsonl -> data/processed/ner/seed_{train,dev,test}.conll
#
# Contoh:
#   python -m indonesian_hoax_checker.ner.annotation auto --checkpoint artifacts/checkpoints/ner/ner_indobert_seed42
#   python -m indonesian_hoax_checker.ner.annotation tasks
#   python -m indonesian_hoax_checker.ner.annotation report --export data/annotations/ner/export_x.json
#   python -m indonesian_hoax_checker.ner.annotation build --export data/annotations/ner/export_x.json
#   python -m indonesian_hoax_checker.ner.annotation split

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from indonesian_hoax_checker.ner.claim_rules import add_claim_tag
from indonesian_hoax_checker.ner.dataset import write_conll
from indonesian_hoax_checker.ner.labels import TYPES, get_spans
from indonesian_hoax_checker.ner.metrics import entity_scores
from indonesian_hoax_checker.ner.time_rules import add_time_tags
from indonesian_hoax_checker.preprocessing.tokenizer import tokenize

ANNOTATION_DIR = Path("data/annotations/ner")
TEST_PERIOD_START = "2026-09-01"  # docs/decisions/0008
VERIFIED_SPLITS = ("dev", "test", "informal_test")  # wajib dikoreksi manusia
SPLITS = ("train", "dev", "test", "informal_test")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ---------- Dua layer: entitas dan CLAIM ----------
# Label gabungan (`tags`) memberi CLAIM prioritas, sehingga PER/ORG/LOC/TIME di dalam span CLAIM hilang.
# Karena itu tiap kalimat juga menyimpan `entity_tags` (tanpa CLAIM, entitas lengkap) dan `claim_tags`
# (hanya CLAIM), untuk melatih model entitas dan model CLAIM secara terpisah.

def claim_layer(tags: list[str]) -> list[str]:
    return [t if t.endswith("CLAIM") else "O" for t in tags]


def entity_layer(tags: list[str]) -> list[str]:
    return ["O" if t.endswith("CLAIM") else t for t in tags]


def merge_layers(entity_tags: list[str], claim_tags: list[str]) -> list[str]:
    return [c if c != "O" else e for e, c in zip(entity_tags, claim_tags)]


def layers(row: dict) -> tuple[list[str], list[str]]:
    if "entity_tags" in row and "claim_tags" in row:
        return row["entity_tags"], row["claim_tags"]
    return entity_layer(row["tags"]), claim_layer(row["tags"])


# Tandai frasa pada satu layer; span lama yang bersinggungan dihapus utuh. label=None -> hapus saja.
def set_span(tokens: list[str], tags: list[str], phrase: str, label: str | None, nth: int = 0) -> None:
    words = phrase.split()
    hits = [i for i in range(len(tokens) - len(words) + 1) if tokens[i:i + len(words)] == words]
    if len(hits) <= nth:
        raise ValueError(f"frasa tidak ditemukan: {phrase!r} di {' '.join(tokens)}")
    start, end = hits[nth], hits[nth] + len(words)
    for kind, a, b in get_spans(tags):
        if a < end and start < b:
            tags[a:b] = ["O"] * (b - a)
    if label:
        tags[start] = "B-" + label
        for i in range(start + 1, end):
            tags[i] = "I-" + label


# patch = {"claim": "frasa" | null, "entities": [["PER", "frasa"] atau ["PER", "frasa", n], ...],
#          "remove": ["frasa", ...]}; hanya kunci yang ada yang dijalankan. Hasil: layer baru + tags gabungan
def apply_patch(row: dict, patch: dict) -> dict:
    tokens = row["tokens"]
    entity_tags, claim_tags = (list(t) for t in layers(row))
    for phrase in patch.get("remove", []):
        set_span(tokens, entity_tags, phrase, None)
    for item in patch.get("entities", []):
        set_span(tokens, entity_tags, item[1], item[0], item[2] if len(item) > 2 else 0)
    if "claim" in patch:
        claim_tags = ["O"] * len(tokens)
        if patch["claim"]:
            set_span(tokens, claim_tags, patch["claim"], "CLAIM")
    return {**row, "entity_tags": entity_tags, "claim_tags": claim_tags, "tags": merge_layers(entity_tags, claim_tags)}


# ---------- 1. Label otomatis ----------

# predictor (opsional) = NERPredictor tahap 1 untuk PER/ORG/LOC; tanpa itu hanya TIME dan CLAIM
def auto_label(rows: list[dict], predictor=None) -> list[dict]:
    words = [[t for t, _, _ in tokenize(r["text"])] for r in rows]
    if predictor:
        nonempty = [w for w in words if w]
        predicted = iter(predictor.tag(nonempty))
        tags = [next(predicted) if w else [] for w in words]
    else:
        tags = [["O"] * len(w) for w in words]
    out = []
    for row, w, t in zip(rows, words, tags):
        entity_tags = add_time_tags(w, t)  # PER/ORG/LOC/TIME tanpa CLAIM
        merged = add_claim_tag(w, entity_tags, row.get("title"))
        out.append({**row, "tokens": w, "tags": merged, "entity_tags": entity_tags, "claim_tags": claim_layer(merged)})
    return out


# ---------- 2. Tugas Label Studio ----------

def label_studio_result(text, tokens, tags):
    results = []
    for kind, first, last in get_spans(tags):
        start, end = tokens[first][1], tokens[last - 1][2]
        results.append({
            "from_name": "label", "to_name": "text", "type": "labels",
            "value": {"start": start, "end": end, "text": text[start:end], "labels": [kind]},
        })
    return results


# Label otomatis tampil sebagai pra-anotasi; salinannya disimpan di data.auto_tags untuk laporan koreksi
def make_tasks(auto_rows: list[dict]) -> list[dict]:
    tasks = []
    for row in auto_rows:
        data = {k: v for k, v in row.items() if k not in ("tokens", "tags")}
        data["auto_tags"] = row["tags"]
        result = label_studio_result(row["text"], tokenize(row["text"]), row["tags"])
        tasks.append({"data": data, "predictions": [{"model_version": "otomatis", "result": result}]})
    return tasks


# ---------- 3-4. Baca ekspor, laporan koreksi, gabungkan ----------

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


# Baca satu atau beberapa file ekspor. Anotasi digabung per kalimat (data.id).
# Bila nama annotator sama di file berbeda, nama file ditambahkan supaya tidak tertimpa.
# Hasil: {sentence_id: {"row": data, "tokens": [...], "by": {annotator: tags}}}
def read_export(paths) -> dict:
    paths = [paths] if isinstance(paths, (str, Path)) else paths
    items, source_of = {}, {}
    for path in paths:
        for task in json.loads(Path(path).read_text(encoding="utf-8")):
            row = task["data"]
            tokens = tokenize(row["text"])
            item = items.setdefault(row["id"], {"row": row, "tokens": [t for t, _, _ in tokens], "by": {}})
            for annotation in task.get("annotations", []):
                if annotation.get("was_cancelled"):
                    continue
                name = annotator_name(annotation)
                if source_of.get(name, path) != path:
                    name = f"{name} ({Path(path).stem})"
                source_of.setdefault(name, path)
                item["by"][name] = spans_to_tags(tokens, annotation["result"])
    return items


def _types(tags):
    return [t[2:] if t != "O" else "O" for t in tags]


# Kappa antar annotator (hanya bila ada lebih dari satu annotator)
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


# Versi final tiap kalimat: anotasi adjudikator bila ada, atau anotasi yang sama persis dari semua annotator.
# Dengan satu annotator, anotasinya langsung menjadi versi final.
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
        row = {k: v for k, v in item["row"].items() if k != "auto_tags"}
        final.append({**row, "tokens": item["tokens"], "tags": tags})
    return final, unresolved


# Seberapa akurat label otomatis dibanding hasil koreksi manusia (manusia = kunci jawaban)
def correction_report(final: list[dict], auto_by_id: dict) -> dict:
    pairs = [(r["tags"], auto_by_id[r["id"]]["tags"]) for r in final if r["id"] in auto_by_id]
    if not pairs:
        return {"sentences": 0}
    human, auto = [h for h, _ in pairs], [a for _, a in pairs]
    return {
        "sentences": len(pairs),
        "unchanged_sentences": sum(h == a for h, a in pairs) / len(pairs),
        "auto_vs_human": entity_scores(human, auto),
    }


# Dev/test/informal_test = hasil koreksi manusia; train = label otomatis (silver).
# - Kalimat dev/test yang belum dikoreksi TIDAK diisi label otomatis: data evaluasi harus diperiksa manusia.
# - Kalimat train tanpa CLAIM dari aturan dibuang: hampir semua kalimat seed memuat klaim, jadi label O
#   di kalimat itu kemungkinan besar salah dan akan mengajari model untuk tidak menandai klaim.
# accept_auto=True: dev/test yang tidak ada di ekspor dianggap sudah diperiksa manusia dan dipakai
# apa adanya (annotator memeriksa semuanya dan tidak menemukan yang perlu diubah).
def build(final: list[dict], auto_rows: list[dict], accept_auto: bool = False) -> tuple[list[dict], list[str], int]:
    human_ids = {r["id"] for r in final}
    rows = [{**r, "quality": "human"} for r in final]
    missing, dropped = [], 0
    for row in auto_rows:
        if row["id"] in human_ids:
            continue
        if row.get("split") in VERIFIED_SPLITS:
            if accept_auto:
                rows.append({**row, "quality": "human"})
            else:
                missing.append(row["id"])
        elif "B-CLAIM" not in row["tags"]:
            dropped += 1
        else:
            rows.append({**row, "quality": "silver"})
    return rows, missing, dropped


# ---------- 5. Split ----------

def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    parts = {name: [] for name in SPLITS}
    for row in rows:
        split = "informal_test" if row.get("source") == "informal" else row.get("split")
        if split in parts:
            parts[split].append(row)
    return parts


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Alat bantu seed set NER (label otomatis + koreksi).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("auto", help="Beri label otomatis ke semua kalimat")
    p.add_argument("--input", nargs="+", default=[str(ANNOTATION_DIR / "seed_sentences.jsonl")],
                   help="Satu atau lebih file kalimat (mis. seed_sentences.jsonl informal_sentences.jsonl)")
    p.add_argument("--output", default=str(ANNOTATION_DIR / "seed_auto.jsonl"))
    p.add_argument("--checkpoint", help="Checkpoint NER tahap 1 untuk PER/ORG/LOC")

    p = sub.add_parser("tasks", help="Buat file impor Label Studio untuk kalimat yang harus dikoreksi")
    p.add_argument("--input", default=str(ANNOTATION_DIR / "seed_auto.jsonl"))
    p.add_argument("--split", nargs="+", default=list(VERIFIED_SPLITS), help="Split yang diimpor")
    p.add_argument("--output", default=str(ANNOTATION_DIR / "label_studio_tasks.json"))

    p = sub.add_parser("report", help="Laporan koreksi: kualitas label otomatis dibanding hasil koreksi")
    p.add_argument("--export", required=True, nargs="+", help="Satu atau beberapa file ekspor")
    p.add_argument("--auto", default=str(ANNOTATION_DIR / "seed_auto.jsonl"))

    p = sub.add_parser("kappa", help="Cohen's kappa antar annotator (bila lebih dari satu)")
    p.add_argument("--export", required=True, nargs="+", help="Satu atau beberapa file ekspor")

    p = sub.add_parser("build", help="Gabungkan koreksi dan label otomatis menjadi seed_adjudicated.jsonl")
    p.add_argument("--export", nargs="+", default=[], help="Satu atau beberapa file ekspor (boleh kosong dengan --accept-auto)")
    p.add_argument("--accept-auto", action="store_true",
                   help="Dev/test yang tidak ada di ekspor dipakai dengan label otomatis (sudah diperiksa, tanpa koreksi)")
    p.add_argument("--auto", default=str(ANNOTATION_DIR / "seed_auto.jsonl"))
    p.add_argument("--adjudicator", help="Nama/email annotator yang dipakai saat anotasi berbeda")
    p.add_argument("--output", default=str(ANNOTATION_DIR / "seed_adjudicated.jsonl"))

    p = sub.add_parser("patch", help="Terapkan koreksi per kalimat (review_patches.json) ke seed_adjudicated.jsonl")
    p.add_argument("--patches", default=str(ANNOTATION_DIR / "review_patches.json"))
    p.add_argument("--input", default=str(ANNOTATION_DIR / "seed_adjudicated.jsonl"))
    p.add_argument("--output", default=str(ANNOTATION_DIR / "seed_adjudicated.jsonl"))

    p = sub.add_parser("split", help="Tulis seed set ke CoNLL per split")
    p.add_argument("--input", default=str(ANNOTATION_DIR / "seed_adjudicated.jsonl"))
    p.add_argument("--out-dir", default="data/processed/ner")

    args = parser.parse_args()

    if args.command == "auto":
        predictor = None
        if args.checkpoint:
            from indonesian_hoax_checker.ner.predict import NERPredictor

            predictor = NERPredictor(args.checkpoint)
        rows = auto_label([row for path in args.input for row in read_jsonl(path)], predictor)
        write_jsonl(args.output, rows)
        spans = Counter(kind for r in rows for kind, _, _ in get_spans(r["tags"]))
        print(f"{len(rows)} kalimat berlabel otomatis ditulis ke {args.output}; span: {dict(spans)}")

    elif args.command == "tasks":
        rows = [r for r in read_jsonl(args.input) if r.get("split") in args.split or
                ("informal_test" in args.split and r.get("source") == "informal")]
        tasks = make_tasks(rows)
        Path(args.output).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(tasks)} tugas ditulis ke {args.output}")

    elif args.command == "report":
        final, _ = adjudicate(read_export(args.export))
        report = correction_report(final, {r["id"]: r for r in read_jsonl(args.auto)})
        print(json.dumps(report, indent=2))

    elif args.command == "kappa":
        report = kappa_report(read_export(args.export))
        print(json.dumps(report, indent=2))

    elif args.command == "build":
        if not args.export and not args.accept_auto:
            parser.error("build butuh --export, atau --accept-auto bila semua dev/test sudah diperiksa tanpa koreksi")
        final, unresolved = adjudicate(read_export(args.export), args.adjudicator) if args.export else ([], [])
        rows, missing, dropped = build(final, read_jsonl(args.auto), args.accept_auto)
        write_jsonl(args.output, rows)
        counts = Counter((r.get("split"), r["quality"]) for r in rows)
        print(f"{len(rows)} kalimat ditulis ke {args.output}: " + ", ".join(f"{s}/{q} {n}" for (s, q), n in sorted(counts.items(), key=str)))
        print(f"{dropped} kalimat train dibuang karena aturan tidak menemukan CLAIM")
        if unresolved:
            print(f"{len(unresolved)} kalimat anotasinya berbeda antar annotator: {unresolved}")
        if missing:
            print(f"PERINGATAN: {len(missing)} kalimat dev/test belum dikoreksi dan tidak dipakai: {missing[:10]} ...")

    elif args.command == "patch":
        patches = json.loads(Path(args.patches).read_text(encoding="utf-8"))
        rows = read_jsonl(args.input)
        ids = {r["id"] for r in rows}
        unknown = [i for i in patches if i not in ids]
        if unknown:
            parser.error(f"id tidak ada di {args.input}: {unknown}")
        rows = [{**apply_patch(r, patches[r["id"]]), "quality": "reviewed"} if r["id"] in patches else r for r in rows]
        write_jsonl(args.output, rows)
        print(f"{len(patches)} kalimat dikoreksi, hasil di {args.output}")

    elif args.command == "split":
        parts = split_rows(read_jsonl(args.input))
        for name, part in parts.items():
            if part or name != "informal_test":
                out = Path(args.out_dir)
                write_conll(out / f"seed_{name}.conll", [(r["tokens"], r["tags"]) for r in part])
                write_conll(out / f"seed_entity_{name}.conll", [(r["tokens"], layers(r)[0]) for r in part])
                write_conll(out / f"seed_claim_{name}.conll", [(r["tokens"], layers(r)[1]) for r in part])
        print(", ".join(f"seed {name} {len(part)}" for name, part in parts.items()))


if __name__ == "__main__":
    main()
