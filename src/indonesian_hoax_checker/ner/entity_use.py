# Uji cara memakai entitas NER di retrieval, dengan kueri tetap TEKS PENUH (hasil query_impact:
# menulis ulang kueri dari core_claim lebih buruk). Retrieval = BM25 milik Danendra, indeks judul klaim.
#
#   polos         teks penuh, tanpa entitas (pembanding)
#   filter_keras  hanya klaim yang berbagi >= 1 entitas dengan kueri (EntityCandidateFilter Danendra)
#   boost         skor BM25 (dibagi skor maksimum) + beta bila berbagi entitas; beta dipilih di split dev
#   teks_entitas  teks penuh + teks entitas PER/ORG/LOC ditambahkan di akhir kueri
#
# Entitas klaim di basis = NER atas judul klaim; entitas kueri = NER atas kalimat kueri.
# Kueri = kalimat klaim artikel Komdigi: dev untuk memilih beta, test untuk angka akhir.
#
# Jalankan:
#   python -m indonesian_hoax_checker.ner.entity_use --claims data/processed/refuted_claims/claims.jsonl

import argparse
import json
from pathlib import Path

import numpy as np

from indonesian_hoax_checker.pipeline.stages import NERStage
from indonesian_hoax_checker.retrieval import metrics as M
from indonesian_hoax_checker.retrieval.candidate_filter import EntityCandidateFilter, entity_keys
from indonesian_hoax_checker.retrieval.search import build_retriever, load_config

BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def top_ids(retriever, scores: np.ndarray, k: int = 10) -> list[str]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [retriever.claims[int(i)]["claim_id"] for i in order if np.isfinite(scores[i])]


def boosted(scores: np.ndarray, shares: np.ndarray, beta: float) -> np.ndarray:
    top = scores.max()
    base = scores / top if top > 0 else scores
    return base + beta * shares


def evaluate(retriever, queries: list[dict], score_fn) -> dict:
    run = {q["id"]: top_ids(retriever, score_fn(q)) for q in queries}
    gold = {q["id"]: q["gold"] for q in queries}
    return M.evaluate_rankings(run, gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cara memakai entitas NER di retrieval BM25.")
    parser.add_argument("--config", default="configs/models/retrieval_bm25.yaml")
    parser.add_argument("--claims", default=None)
    parser.add_argument("--output", default="outputs/metrics/ner_entity_use.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.claims:
        config["claims_path"] = args.claims
    config["fields"] = ["claim"]
    retriever = build_retriever(config)
    claims = retriever.claims
    ner = NERStage.from_config()

    # Entitas basis klaim (judul) dan kunci pencocokannya
    corpus_entities = [r["entities"] for r in ner.run_batch([c["claim"] for c in claims])]
    flt = EntityCandidateFilter(corpus_entities)

    by_cluster: dict[str, set[str]] = {}
    for c in claims:
        by_cluster.setdefault(c["cluster_id"], set()).add(c["claim_id"])

    def build_queries(split: str) -> list[dict]:
        rows = [c for c in claims if c.get("split") == split and (c.get("claim_sentence") or "").strip()]
        results = ner.run_batch([c["claim_sentence"] for c in rows])
        out = []
        for c, res in zip(rows, results):
            keys = entity_keys(res["entities"])
            shares = np.array([float(bool(keys & dk)) for dk in flt.corpus_keys])
            ent_text = " ".join(dict.fromkeys(e["text"] for e in res["entities"] if e["label"] in ("PER", "ORG", "LOC")))
            out.append({
                "id": c["claim_id"], "text": c["claim_sentence"], "gold": by_cluster[c["cluster_id"]],
                "entities": res["entities"], "shares": shares, "with_entities": f'{c["claim_sentence"]} {ent_text}'.strip(),
                "has_entities": bool(keys),
            })
        return out

    dev, test = build_queries("dev"), build_queries("test")

    def plain(q):
        return retriever.scorer.score(q["text"])

    def hard(q):
        return flt.apply(plain(q), q["entities"])

    def expanded(q):
        return retriever.scorer.score(q["with_entities"])

    # Pilih beta di dev berdasarkan MRR, lalu laporkan di test
    dev_by_beta = {b: evaluate(retriever, dev, lambda q, b=b: boosted(plain(q), q["shares"], b)) for b in BETAS}
    best_beta = max(BETAS, key=lambda b: dev_by_beta[b]["mrr"])

    def soft(q):
        return boosted(plain(q), q["shares"], best_beta)

    # Diagnostik: pada berapa kueri test klaim emas ikut lolos filter keras? (kueri tanpa entitas tidak disaring)
    id_index = {c["claim_id"]: i for i, c in enumerate(claims)}
    with_ents = [q for q in test if q["has_entities"]]
    kept = sum(any(flt.mask(q["entities"])[id_index[g]] for g in q["gold"]) for q in with_ents)
    shared_any = sum(any(q["shares"][id_index[g]] for g in q["gold"]) for q in with_ents)

    result = {
        "retrieval": "bm25, indeks judul klaim saja, kueri = teks penuh",
        "diagnostik_filter": {
            "kueri_test_dengan_entitas": len(with_ents),
            "emas_berbagi_entitas_dengan_kueri": shared_any,
            "emas_lolos_filter": kept,
        },
        "n_dev": len(dev), "n_test": len(test),
        "kueri_tanpa_entitas_test": sum(not q["has_entities"] for q in test),
        "beta_dipilih_di_dev": best_beta,
        "dev_mrr_per_beta": {str(b): round(v["mrr"], 4) for b, v in dev_by_beta.items()},
        "test": {
            "polos": evaluate(retriever, test, plain),
            "filter_keras": evaluate(retriever, test, hard),
            "boost": evaluate(retriever, test, soft),
            "teks_entitas": evaluate(retriever, test, expanded),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    keep = ("success@1", "success@5", "success@10", "mrr")
    print(f"dev={len(dev)} test={len(test)}; kueri test tanpa entitas: {result['kueri_tanpa_entitas_test']}")
    print("diagnostik filter:", result["diagnostik_filter"])
    print("MRR dev per beta:", result["dev_mrr_per_beta"], "-> beta", best_beta)
    for name, r in result["test"].items():
        print(f"  {name:13s}", {k: round(r[k], 3) for k in keep})


if __name__ == "__main__":
    main()
