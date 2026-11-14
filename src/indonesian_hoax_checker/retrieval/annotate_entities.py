# Prediksi entitas NER untuk basis klaim dan kueri retrieval (ablasi filter entitas RQ3).
# Keputusan 0005: entitas untuk task 3 adalah PREDIKSI NER, bukan gold. Pemilik: Danendra.
#
#   python -m indonesian_hoax_checker.retrieval.annotate_entities \
#       --checkpoint artifacts/checkpoints/ner/ner_indobert_entity_seed42 \
#       --claim-checkpoint artifacts/checkpoints/ner/ner_indobert_claim_seed42 \
#       --claims data/processed/refuted_claims/claims.jsonl \
#       --claims-out data/processed/refuted_claims/claims_entities.jsonl \
#       --queries data/annotations/claim_verification/retrieval_queries.jsonl
#
# Checkpoint: ekstrak ner_checkpoints_seed42.zip (Drive tim hasil_training/ner_*) ke
# artifacts/checkpoints/ner/. Layer entitas dan CLAIM terpisah (docs/decisions/0012); tanpa
# --claim-checkpoint, core_claim tidak diprediksi.
# Keluaran claims_entities.jsonl: {claim_id, entities, core_claim, ner_checkpoint}; dibaca
# retrieval/search.py (load_corpus_entities). Kueri: field `entities` diisi di tempat, field lain utuh.

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from indonesian_hoax_checker.common.schemas import NERResult
from indonesian_hoax_checker.retrieval.corpus import read_jsonl, write_jsonl

PredictBatch = Callable[[Sequence[str]], list[NERResult]]


def annotate_rows(
    rows: list[dict[str, Any]],
    text_key: str,
    predict_batch: PredictBatch,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Isi `entities` (dan `core_claim`) tiap baris dari teks di `text_key`. Baris dengan teks
    kosong (mis. parafrasa yang belum ditulis) mendapat entities=[] tanpa dipanggil ke model."""
    todo = [i for i, r in enumerate(rows) if (r.get(text_key) or "").strip()]
    for start in range(0, len(todo), batch_size):
        idx = todo[start : start + batch_size]
        results = predict_batch([rows[i][text_key] for i in idx])
        if len(results) != len(idx):
            raise RuntimeError(f"predict_batch mengembalikan {len(results)} hasil untuk {len(idx)} teks")
        for i, res in zip(idx, results):
            rows[i]["entities"] = res["entities"]
            rows[i]["core_claim"] = res["core_claim"]
    for i, r in enumerate(rows):
        if i not in set(todo):
            r["entities"] = []
            r["core_claim"] = ""
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    for r in rows:
        for e in r.get("entities", []):
            labels[e["label"]] = labels.get(e["label"], 0) + 1
    n = len(rows)
    with_narrowing = sum(
        1 for r in rows if any(e["label"] in ("PER", "ORG", "LOC") for e in r.get("entities", []))
    )
    return {
        "n": n,
        "with_any_entity": sum(1 for r in rows if r.get("entities")),
        # proporsi baris yang bisa dipersempit filter (punya PER/ORG/LOC); kalau rendah, filter RQ3 jarang aktif
        "with_per_org_loc": with_narrowing,
        "with_core_claim": sum(1 for r in rows if r.get("core_claim")),
        "label_counts": dict(sorted(labels.items())),
    }


def _load_predictor(checkpoint: str, device: str, claim_checkpoint: str | None = None) -> PredictBatch:
    # impor di sini: ner.predict menarik torch/transformers/seqeval, tidak perlu untuk tes unit
    from indonesian_hoax_checker.pipeline.stages import NERStage

    stage = NERStage(checkpoint, device, claim_checkpoint)
    return stage.run_batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prediksi entitas NER untuk basis klaim dan kueri.")
    p.add_argument("--checkpoint", required=True, help="folder checkpoint NER (IndoBERT atau BiLSTM-CRF)")
    p.add_argument("--claim-checkpoint", default=None,
                   help="folder checkpoint NER layer CLAIM (untuk core_claim); kosong = core_claim tidak diisi")
    p.add_argument("--device", default="auto")
    p.add_argument("--claims", default="data/processed/refuted_claims/claims.jsonl")
    p.add_argument("--claims-out", default="data/processed/refuted_claims/claims_entities.jsonl")
    p.add_argument("--claim-text", default="claim", help="field klaim yang diprediksi (claim / claim_sentence)")
    p.add_argument("--queries", default=None, help="retrieval_queries.jsonl; field entities diisi di tempat")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    predict = _load_predictor(args.checkpoint, args.device, args.claim_checkpoint)
    ckpt = Path(args.checkpoint).name
    if args.claim_checkpoint:
        ckpt += "+" + Path(args.claim_checkpoint).name
    report: dict[str, Any] = {"checkpoint": ckpt}

    claims = list(read_jsonl(args.claims))
    out_rows = [{"claim_id": c["claim_id"], args.claim_text: c.get(args.claim_text, "")} for c in claims]
    annotate_rows(out_rows, args.claim_text, predict, args.batch_size)
    for r in out_rows:
        r.pop(args.claim_text)
        r["ner_checkpoint"] = ckpt
    write_jsonl(out_rows, args.claims_out)
    report["claims"] = {"path": args.claims_out, **summarize(out_rows)}

    if args.queries:
        qrows = list(read_jsonl(args.queries))
        annotate_rows(qrows, "query", predict, args.batch_size)
        for r in qrows:
            r["entities_checkpoint"] = ckpt
        write_jsonl(qrows, args.queries)
        report["queries"] = {"path": args.queries, **summarize(qrows)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
