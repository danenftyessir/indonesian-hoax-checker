# Uji dampak NER ke retrieval: apakah kueri dari core_claim (hasil NER) menemukan klaim terbantah
# lebih baik daripada memakai seluruh teks? Retrieval = BM25 milik Danendra, tanpa filter entitas.
#
# Kueri uji = kalimat klaim dari artikel Komdigi ("Beredar sebuah unggahan ... yang mengeklaim bahwa ..."),
# hanya dari split test (artikel sejak 1 Sep 2026, tidak dipakai melatih NER). Indeks hanya berisi judul
# klaim (fields: [claim]), jadi kalimat kueri tidak identik dengan dokumen yang dicari.
# Emas = klaim asal beserta anggota klasternya (near-duplicate).
#
# Pembanding kedua: 100 kueri `original` (judul apa adanya) dari retrieval_queries.jsonl. Itu uji yang
# mudah karena kueri sama dengan judul yang diindeks; kueri parafrasa belum diisi (status draft).
#
# Jalankan:
#   python -m indonesian_hoax_checker.ner.query_impact

import argparse
import json
from pathlib import Path

from indonesian_hoax_checker.pipeline.stages import NERStage
from indonesian_hoax_checker.retrieval import metrics as M
from indonesian_hoax_checker.retrieval.corpus import read_jsonl
from indonesian_hoax_checker.retrieval.search import build_retriever, load_config


# Aturan kueri pipeline (sama dengan retrieval_query di pipeline/orchestrator.py): core_claim ditambah
# entitas PER/ORG/LOC yang belum ada di dalamnya; bila core_claim kosong, teks penuh
def entity_query(text: str, res: dict) -> str:
    core = res["core_claim"].strip()
    if not core:
        return text
    extra: list[str] = []
    for e in res["entities"]:
        if e["label"] in ("PER", "ORG", "LOC") and e["text"] not in core and e["text"] not in extra:
            extra.append(e["text"])
    return " ".join([core, *extra])


# Dua kueri dari hasil NER: core_claim saja (kosong -> teks asli) dan core_claim + entitas
def claim_queries(ner: NERStage, texts: list[str]) -> tuple[list[str], list[str], int]:
    claim_only, with_entities, empty = [], [], 0
    for text, res in zip(texts, ner.run_batch(texts)):
        claim_only.append(res["core_claim"] or text)
        with_entities.append(entity_query(text, res))
        empty += not res["core_claim"]
    return claim_only, with_entities, empty


def score(retriever, ids: list[str], texts: list[str], gold: dict[str, set[str]]) -> dict:
    run = {qid: retriever.ranked_ids(text, top_k=10) for qid, text in zip(ids, texts)}
    return M.evaluate_rankings(run, gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dampak core_claim NER pada retrieval BM25.")
    parser.add_argument("--queries", default="data/annotations/claim_verification/retrieval_queries.jsonl")
    parser.add_argument("--config", default="configs/models/retrieval_bm25.yaml")
    parser.add_argument("--claims", default=None, help="claims.jsonl; default dari config")
    parser.add_argument("--output", default="outputs/metrics/ner_query_impact.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.claims:
        config["claims_path"] = args.claims
    config["fields"] = ["claim"]  # indeks judul saja
    retriever = build_retriever(config)
    ner = NERStage.from_config()
    claims = retriever.claims
    result = {"retrieval": "bm25, indeks judul klaim saja, tanpa filter entitas"}

    # 1. Kalimat klaim Komdigi (split test); emas = klaim asal + klaster near-duplicate-nya
    by_cluster: dict[str, set[str]] = {}
    for c in claims:
        by_cluster.setdefault(c["cluster_id"], set()).add(c["claim_id"])
    test = [c for c in claims if c.get("split") == "test" and (c.get("claim_sentence") or "").strip()]
    ids = [c["claim_id"] for c in test]
    texts = [c["claim_sentence"] for c in test]
    gold = {c["claim_id"]: by_cluster[c["cluster_id"]] for c in test}
    core, core_ent, empty = claim_queries(ner, texts)
    result["kalimat_komdigi_test"] = {
        "n_queries": len(test),
        "core_claim_kosong": empty,
        "teks_penuh": score(retriever, ids, texts, gold),
        "core_claim": score(retriever, ids, core, gold),
        "core_claim_plus_entitas": score(retriever, ids, core_ent, gold),
    }
    result["kalimat_komdigi_test"]["contoh"] = [
        {"kueri": texts[i], "core_claim": core[i], "core_claim_plus_entitas": core_ent[i], "judul": test[i]["claim"]}
        for i in range(0, len(test), max(1, len(test) // 5))
    ][:5]

    # 2. Judul apa adanya (uji mudah)
    originals = [q for q in read_jsonl(args.queries) if q.get("type") == "original" and q.get("gold_claim_ids")]
    o_ids = [q["query_id"] for q in originals]
    o_texts = [q["query"] for q in originals]
    o_gold = {q["query_id"]: set(q["gold_claim_ids"]) for q in originals}
    o_core, o_core_ent, o_empty = claim_queries(ner, o_texts)
    result["judul_original"] = {
        "n_queries": len(originals),
        "core_claim_kosong": o_empty,
        "teks_penuh": score(retriever, o_ids, o_texts, o_gold),
        "core_claim": score(retriever, o_ids, o_core, o_gold),
        "core_claim_plus_entitas": score(retriever, o_ids, o_core_ent, o_gold),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    keep = ("success@1", "success@5", "success@10", "mrr")
    for name in ("kalimat_komdigi_test", "judul_original"):
        r = result[name]
        print(f"\n== {name}: n={r['n_queries']}, core_claim kosong={r['core_claim_kosong']}")
        for cond in ("teks_penuh", "core_claim", "core_claim_plus_entitas"):
            print(f"  {cond:24s}", {k: round(r[cond][k], 3) for k in keep})
    for ex in result["kalimat_komdigi_test"]["contoh"]:
        print("\n- kueri:", ex["kueri"][:150]); print("  + entitas :", ex["core_claim_plus_entitas"][:140]); print("  judul     :", ex["judul"])


if __name__ == "__main__":
    main()
