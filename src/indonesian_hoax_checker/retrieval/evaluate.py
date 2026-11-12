# Evaluasi RQ3: jalankan satu konfigurasi retriever pada file kueri dan tulis metrik + prediksi.
#
#   python -m indonesian_hoax_checker.retrieval.evaluate configs/models/retrieval_bm25.yaml \
#       --queries data/annotations/claim_verification/retrieval_queries.jsonl \
#       --run-name rq3_bm25_nofilter
#
# Format kueri: docs/data_dictionary.md "Kueri retrieval" (query_id, query, type, gold_claim_ids),
# opsional `entities` (daftar Entity hasil NER untuk kueri) dan `split`.

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from indonesian_hoax_checker import __version__
from indonesian_hoax_checker.retrieval import metrics as M
from indonesian_hoax_checker.retrieval.corpus import read_jsonl, write_jsonl
from indonesian_hoax_checker.retrieval.search import build_retriever, load_config


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - tidak ada git bukan kegagalan evaluasi
        return "unknown"


def run_evaluation(
    config: dict[str, Any],
    queries: list[dict[str, Any]],
    top_k: int = 10,
    reference_date: str | None = None,
    rejection_threshold: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever = build_retriever(config)
    retriever.top_k = max(top_k, retriever.top_k)
    claim_dates = {c["claim_id"]: c.get("published_at", "") for c in retriever.claims}

    run: dict[str, list[str]] = {}
    top_scores: dict[str, float | None] = {}
    predictions = []
    for q in queries:
        ranked = retriever.rank(q["query"], q.get("entities"), top_k=top_k)
        ids = [retriever.claims[i]["claim_id"] for i, _ in ranked]
        run[q["query_id"]] = ids
        top_scores[q["query_id"]] = ranked[0][1] if ranked else None
        predictions.append(
            {
                "query_id": q["query_id"],
                "type": q.get("type"),
                "gold_claim_ids": q.get("gold_claim_ids", []),
                "ranked_claim_ids": ids,
                "scores": [round(s, 4) for _, s in ranked],
            }
        )

    gold = {q["query_id"]: set(q.get("gold_claim_ids") or []) for q in queries if q.get("gold_claim_ids")}
    is_no_match = {q["query_id"]: not q.get("gold_claim_ids") for q in queries}
    qtypes = {q["query_id"]: q.get("type", "unknown") for q in queries}

    result: dict[str, Any] = {
        "overall": M.evaluate_rankings(run, gold),
        "by_type": M.by_query_type(run, gold, qtypes),
    }
    if any(is_no_match.values()):
        thr = rejection_threshold
        if thr is None:
            thr = M.best_rejection_threshold(top_scores, is_no_match)["threshold"]
        result["no_match"] = M.no_match_rejection(top_scores, is_no_match, float(thr))
    if reference_date:
        result["by_claim_age"] = M.recall_by_claim_age(run, gold, claim_dates, reference_date)
    return result, predictions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluasi retrieval RQ3.")
    p.add_argument("config", help="configs/models/retrieval_bm25.yaml atau retrieval_embeddings.yaml")
    p.add_argument("--queries", required=True)
    p.add_argument("--run-name", required=True, help="mis. rq3_bm25_nofilter")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--reference-date", default="2026-11-18", help="tanggal pembekuan, untuk usia klaim")
    p.add_argument("--rejection-threshold", type=float, default=None)
    p.add_argument("--entity-filter", action="store_true", help="paksa filter entitas aktif (ablasi RQ3)")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--predictions-dir", default="outputs/predictions")
    args = p.parse_args(argv)

    config = load_config(args.config)
    if args.entity_filter:
        config["entity_filter"] = True
    queries = list(read_jsonl(args.queries))
    result, predictions = run_evaluation(
        config, queries, args.top_k, args.reference_date, args.rejection_threshold
    )
    result["meta"] = {
        "run_name": args.run_name,
        "config": args.config,
        "config_resolved": config,
        "queries": args.queries,
        "n_queries": len(queries),
        "git_commit": _git_commit(),
        "package_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics_path = Path(args.metrics_dir) / f"{args.run_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(predictions, Path(args.predictions_dir) / f"{args.run_name}.jsonl")
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
