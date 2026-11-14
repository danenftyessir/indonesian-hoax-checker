# Pasca-proses span CLAIM prediksi: pangkas ujung yang menurut pedoman anotasi tidak termasuk CLAIM.
# Error analysis (docs/experiment_protocol.md) menunjukkan model menemukan letak klaim dengan baik,
# tetapi sering memperpanjang span ke keterangan waktu/tempat dan obrolan di akhir kalimat.
#
# Aturan (docs/annotation_guidelines.md, Bagian A):
#   time         TIME tidak masuk CLAIM: span dipotong di token TIME pertama (dari layer entitas),
#                kata depan waktu di ujung ("mulai", "pada", "sblm") ikut dibuang
#   punct        span berhenti sebelum "?" atau "!" (sisanya biasanya obrolan: "? ? hati2 ya bun")
#   attribution  sumber di ujung tidak masuk: "... kata bmkg", "... menurut Kominfo"
#   loc          keterangan tempat "di <LOC>" di ujung tidak masuk (contoh 4 pedoman: "di Bali")
#   subject      subjek bukan entitas tidak masuk (contoh 5 pedoman): span mulai dari predikat pertama
#                ("garam merk tertentu dicampur plastik" -> "dicampur plastik"), paling jauh 4 token
#
# Layer entitas bisa dilengkapi TIME dari time_rules.py (fill_time), karena model entitas sering
# melewatkan TIME informal ("mggu dpn", "skrg").
#
# Hasil pemilihan di dev ada di docs/experiment_protocol.md.
# Bandingkan kombinasi aturan di dev (dipakai untuk memilih) lalu test dan uji informal:
#   python -m indonesian_hoax_checker.ner.claim_trim

import argparse
from pathlib import Path

from indonesian_hoax_checker.ner.claim_rules import is_predicate
from indonesian_hoax_checker.ner.dataset import read_conll
from indonesian_hoax_checker.ner.labels import get_spans
from indonesian_hoax_checker.ner.time_rules import add_time_tags

RULES = ("time", "punct", "attribution", "loc", "subject")
# Dipakai NERPredictor: dev tidak turun, test naik. attribution tidak berpengaruh; loc dan subject menurunkan dev.
# fill_time tidak dipakai default: dev sama, TIME test formal sedikit turun (lihat docs/experiment_protocol.md)
DEFAULT_RULES = ("time", "punct")
TIME_PREPOSITIONS = {"mulai", "pada", "sejak", "sebelum", "sblm", "sampai", "smp", "hingga", "per", "selama",
                     "slama", "tanggal", "tgl", "tahun", "thn", "bulan", "hari", "hr", "jd", "menjadi", "di"}
ATTRIBUTION = {"kata", "menurut", "info", "sumber"}
STOP_PUNCT = {"?", "!"}
EDGE_PUNCT = {",", ".", ";", ":", "-", "?", "!"}


def strip_end(tokens: list[str], start: int, end: int, words: set[str]) -> int:
    while end > start and tokens[end - 1].lower() in words:
        end -= 1
    return end


def trim_span(tokens: list[str], entity_tags: list[str], start: int, end: int, rules=RULES) -> tuple[int, int]:
    # Mengembalikan span CLAIM baru [start, end)
    if "subject" in rules and not is_predicate(tokens[start]):
        for j in range(start + 1, min(start + 5, end)):
            if is_predicate(tokens[j]):
                start = j
                break
    if "punct" in rules:
        for i in range(start + 1, end):
            if tokens[i] in STOP_PUNCT:
                end = i
                break
    if "attribution" in rules:
        for i in range(start + 1, end):
            if tokens[i].lower() in ATTRIBUTION and any(t != "O" for t in entity_tags[i + 1:end + 2]):
                end = i
                break
    if "time" in rules:
        for i in range(start + 1, end):
            if entity_tags[i].endswith("TIME"):
                end = strip_end(tokens, start + 1, i, TIME_PREPOSITIONS)
                break
        else:
            # TIME tepat setelah span: kata depannya ("sblm", "mulai") tidak ikut CLAIM
            if end < len(tokens) and entity_tags[end].endswith("TIME"):
                end = strip_end(tokens, start + 1, end, TIME_PREPOSITIONS)
    if "loc" in rules:
        spans = [(a, b) for kind, a, b in get_spans(entity_tags) if kind == "LOC" and start < a and b == end]
        if spans and tokens[spans[0][0] - 1].lower() == "di" and spans[0][0] - 1 > start:
            end = spans[0][0] - 1
    return start, strip_end(tokens, start + 1, end, EDGE_PUNCT)


def trim_claims(tokens: list[str], entity_tags: list[str], claim_tags: list[str], rules=RULES) -> list[str]:
    out = ["O"] * len(tokens)
    for _, start, end in get_spans(claim_tags):
        start, end = trim_span(tokens, entity_tags, start, end, rules)
        out[start] = "B-CLAIM"
        for i in range(start + 1, end):
            out[i] = "I-CLAIM"
    return out


def f1(gold: list[list[str]], pred: list[list[str]]) -> float:
    g = [set(get_spans(t)) for t in gold]
    p = [set(get_spans(t)) for t in pred]
    tp = sum(len(a & b) for a, b in zip(g, p))
    n_g, n_p = sum(map(len, g)), sum(map(len, p))
    return 2 * tp / (n_g + n_p) if n_g + n_p else 0.0


def fill_time(tokens: list[str], entity_tags: list[str]) -> list[str]:
    # TIME dari aturan hanya di token yang masih O (entitas model tidak ditimpa)
    return add_time_tags(tokens, entity_tags)


def main() -> None:
    from indonesian_hoax_checker.ner.predict import NERPredictor

    parser = argparse.ArgumentParser(description="Nilai fill_time dan aturan pangkas CLAIM per kombinasi.")
    parser.add_argument("--entity-checkpoint", default="artifacts/checkpoints/ner/ner_indobert_entity_seed42")
    parser.add_argument("--claim-checkpoint", default="artifacts/checkpoints/ner/ner_indobert_claim_seed42")
    parser.add_argument("--data-dir", default="data/processed/ner")
    args = parser.parse_args()

    entity_model = NERPredictor(args.entity_checkpoint)
    claim_model = NERPredictor(args.claim_checkpoint, trim_claim=False)
    uji = ["dev", "test", "informal_test", "test_sim_informal"]
    data = {}
    for name in uji:
        claim_gold = read_conll(Path(args.data_dir) / f"seed_claim_{name}.conll")
        entity_gold = read_conll(Path(args.data_dir) / f"seed_entity_{name}.conll")
        words = [t for t, _ in claim_gold]
        ent = entity_model.tag(words)
        data[name] = (words, [g for _, g in claim_gold], [g for _, g in entity_gold], ent,
                      [fill_time(w, e) for w, e in zip(words, ent)], claim_model.tag(words))

    def time_only(tags):
        return [[t if t.endswith("TIME") else "O" for t in s] for s in tags]

    print("Layer entitas (F1 semua tipe / F1 TIME):")
    print("| entitas | " + " | ".join(uji) + " |")
    print("|---" * (len(uji) + 1) + "|")
    for label, k in (("model", 3), ("model + fill_time", 4)):
        cells = [f"{f1(d[2], d[k]):.3f} / {f1(time_only(d[2]), time_only(d[k])):.3f}" for d in data.values()]
        print(f"| {label} | " + " | ".join(cells) + " |")

    combos = [(), ("time",), ("time", "punct"), ("subject",), ("time", "punct", "subject"), ("loc",)]
    print("\nCLAIM (F1 strict):")
    print("| aturan | entitas | " + " | ".join(uji) + " |")
    print("|---" * (len(uji) + 2) + "|")
    for rules in combos:
        for label, k in (("model", 3), ("+ fill_time", 4)):
            if not rules and k == 4:
                continue
            cells = []
            for d in data.values():
                pred = [trim_claims(w, e, c, rules) for w, e, c in zip(d[0], d[k], d[5])]
                cells.append(f"{f1(d[1], pred):.3f}")
            print(f"| {'+'.join(rules) or 'tanpa'} | {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
