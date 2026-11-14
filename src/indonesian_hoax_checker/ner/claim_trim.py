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
#
# Hasil pemilihan di dev ada di docs/experiment_protocol.md.
# Bandingkan kombinasi aturan di dev (dipakai untuk memilih) lalu test dan uji informal:
#   python -m indonesian_hoax_checker.ner.claim_trim

import argparse
from pathlib import Path

from indonesian_hoax_checker.ner.dataset import read_conll
from indonesian_hoax_checker.ner.labels import get_spans

RULES = ("time", "punct", "attribution", "loc")
# Dipakai NERPredictor: dev tidak turun, test naik. attribution tidak berpengaruh, loc menurunkan dev
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


def trim_span(tokens: list[str], entity_tags: list[str], start: int, end: int, rules=RULES) -> int:
    # Mengembalikan ujung baru span CLAIM [start, end)
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
    if "loc" in rules:
        spans = [(a, b) for kind, a, b in get_spans(entity_tags) if kind == "LOC" and start < a and b == end]
        if spans and tokens[spans[0][0] - 1].lower() == "di" and spans[0][0] - 1 > start:
            end = spans[0][0] - 1
    return strip_end(tokens, start + 1, end, EDGE_PUNCT)


def trim_claims(tokens: list[str], entity_tags: list[str], claim_tags: list[str], rules=RULES) -> list[str]:
    out = ["O"] * len(tokens)
    for _, start, end in get_spans(claim_tags):
        end = trim_span(tokens, entity_tags, start, end, rules)
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


def main() -> None:
    from indonesian_hoax_checker.ner.predict import NERPredictor

    parser = argparse.ArgumentParser(description="Nilai aturan pangkas CLAIM per kombinasi aturan.")
    parser.add_argument("--entity-checkpoint", default="artifacts/checkpoints/ner/ner_indobert_entity_seed42")
    parser.add_argument("--claim-checkpoint", default="artifacts/checkpoints/ner/ner_indobert_claim_seed42")
    parser.add_argument("--data-dir", default="data/processed/ner")
    args = parser.parse_args()

    entity_model = NERPredictor(args.entity_checkpoint)
    claim_model = NERPredictor(args.claim_checkpoint)
    combos = [(), ("time",), ("punct",), ("attribution",), ("loc",), ("time", "punct"),
              ("time", "punct", "attribution"), RULES]
    uji = ["dev", "test", "informal_test", "test_sim_informal"]
    print("| aturan | " + " | ".join(uji) + " |")
    print("|---" * (len(uji) + 1) + "|")
    data = {}
    for name in uji:
        sentences = read_conll(Path(args.data_dir) / f"seed_claim_{name}.conll")
        words = [t for t, _ in sentences]
        data[name] = ([g for _, g in sentences], words, entity_model.tag(words), claim_model.tag(words))
    for rules in combos:
        scores = []
        for name in uji:
            gold, words, ent, clm = data[name]
            pred = [trim_claims(w, e, c, rules) for w, e, c in zip(words, ent, clm)]
            scores.append(f"{f1(gold, pred):.3f}")
        print(f"| {'+'.join(rules) or 'tanpa'} | " + " | ".join(scores) + " |")


if __name__ == "__main__":
    main()
