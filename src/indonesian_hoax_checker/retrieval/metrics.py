# Metrik retrieval RQ3 (docs/experiment_protocol.md, task 3a): success@k, MRR, Recall@k,
# recall terhadap usia klaim, dan ketepatan menolak kueri tanpa padanan (no_match).
# Protokol mengikuti SemEval-2025 Task 7 (previously fact-checked claim retrieval).

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

DEFAULT_KS = (1, 5, 10)


def _first_hit_rank(ranked: Sequence[str], gold: set[str]) -> int | None:
    for r, cid in enumerate(ranked, start=1):
        if cid in gold:
            return r
    return None


def success_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if any(cid in gold for cid in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    r = _first_hit_rank(ranked, gold)
    return 0.0 if r is None else 1.0 / r


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def evaluate_rankings(
    run: dict[str, Sequence[str]],
    gold: dict[str, set[str]],
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, float]:
    """run: query_id -> ranked claim_ids; gold: query_id -> gold claim_ids (hanya kueri berpadanan).
    Rata-rata atas kueri di `gold`; kueri tanpa baris di `run` dihitung gagal."""
    ks = list(ks)
    if not gold:
        return {}
    acc = {f"success@{k}": 0.0 for k in ks} | {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    for qid, g in gold.items():
        ranked = list(run.get(qid, []))
        for k in ks:
            acc[f"success@{k}"] += success_at_k(ranked, g, k)
            acc[f"recall@{k}"] += recall_at_k(ranked, g, k)
        acc["mrr"] += reciprocal_rank(ranked, g)
    n = len(gold)
    out = {m: round(v / n, 4) for m, v in acc.items()}
    out["n_queries"] = n
    return out


def no_match_rejection(
    top_scores: dict[str, float | None], is_no_match: dict[str, bool], threshold: float
) -> dict[str, float]:
    """Sistem 'menolak' bila skor teratas < threshold atau tidak ada kandidat.
    rejection_rate_no_match = porsi kueri no_match yang ditolak (ingin tinggi);
    false_rejection_rate = porsi kueri berpadanan yang ikut ditolak (ingin rendah)."""
    nm_total = nm_rej = m_total = m_rej = 0
    for qid, flag in is_no_match.items():
        s = top_scores.get(qid)
        rejected = s is None or s < threshold
        if flag:
            nm_total += 1
            nm_rej += rejected
        else:
            m_total += 1
            m_rej += rejected
    return {
        "threshold": threshold,
        "rejection_rate_no_match": round(nm_rej / nm_total, 4) if nm_total else float("nan"),
        "false_rejection_rate": round(m_rej / m_total, 4) if m_total else float("nan"),
        "n_no_match": nm_total,
        "n_match": m_total,
    }


def best_rejection_threshold(
    top_scores: dict[str, float | None], is_no_match: dict[str, bool], n_grid: int = 50
) -> dict[str, float]:
    """Pilih ambang pada dev yang memaksimalkan (rejection_rate_no_match - false_rejection_rate)."""
    vals = [s for s in top_scores.values() if s is not None]
    if not vals:
        return {"threshold": 0.0}
    best = None
    for t in np.linspace(min(vals), max(vals), n_grid):
        r = no_match_rejection(top_scores, is_no_match, float(t))
        gain = r["rejection_rate_no_match"] - r["false_rejection_rate"]
        if best is None or gain > best[0]:
            best = (gain, r)
    return best[1]


def _months_between(older: datetime, newer: datetime) -> int:
    return max(0, (newer.year - older.year) * 12 + newer.month - older.month)


def recall_by_claim_age(
    run: dict[str, Sequence[str]],
    gold: dict[str, set[str]],
    claim_dates: dict[str, str],
    reference_date: str,
    k: int = 10,
    buckets: Sequence[int] = (3, 6, 12),
) -> dict[str, dict[str, float]]:
    """success@k dikelompokkan menurut usia (bulan) klaim gold tertua relatif ke reference_date
    (tanggal pembekuan test set). Bucket: <3, 3-6, 6-12, >=12 bulan (default)."""
    ref = datetime.fromisoformat(reference_date).replace(tzinfo=timezone.utc)
    edges = list(buckets)
    labels = [f"<{edges[0]}m"] + [f"{a}-{b}m" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}m"]
    agg: dict[str, list[float]] = {lab: [] for lab in labels}
    for qid, g in gold.items():
        ages = []
        for cid in g:
            d = claim_dates.get(cid)
            if d:
                dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ages.append(_months_between(dt, ref))
        if not ages:
            continue
        age = max(ages)
        idx = sum(age >= e for e in edges)
        agg[labels[idx]].append(success_at_k(list(run.get(qid, [])), g, k))
    return {
        lab: {"n": len(v), f"success@{k}": round(float(np.mean(v)), 4) if v else float("nan")}
        for lab, v in agg.items()
    }


def by_query_type(
    run: dict[str, Sequence[str]],
    gold: dict[str, set[str]],
    query_types: dict[str, str],
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, dict[str, Any]]:
    """Metrik terpisah per `type` kueri (original / paraphrase)."""
    out: dict[str, dict[str, Any]] = {}
    for t in sorted(set(query_types.values())):
        sub = {q: g for q, g in gold.items() if query_types.get(q) == t}
        if sub:
            out[t] = evaluate_rankings(run, sub, ks)
    return out
