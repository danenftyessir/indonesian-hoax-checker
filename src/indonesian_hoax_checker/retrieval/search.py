# Pencarian klaim terbantah: gabungan korpus + scorer (BM25 / embedding) + filter entitas,
# mengembalikan RetrievalResult sesuai common/schemas.py. Dipakai pipeline/stages.py dan evaluasi.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from indonesian_hoax_checker.common.schemas import Entity, RetrievalCandidate, RetrievalResult
from indonesian_hoax_checker.retrieval.bm25 import BM25
from indonesian_hoax_checker.retrieval.candidate_filter import EntityCandidateFilter
from indonesian_hoax_checker.retrieval.corpus import load_claims, read_jsonl

ScorerName = Literal["bm25", "embedding"]

# Field klaim yang diindeks. `claim` (judul) wajib; `claim_sentence` menambah konteks.
DEFAULT_FIELDS = ("claim", "claim_sentence")


def document_text(claim: dict[str, Any], fields: Sequence[str] = DEFAULT_FIELDS) -> str:
    return " ".join(str(claim.get(f) or "") for f in fields).strip()


class Retriever:
    def __init__(
        self,
        claims: list[dict[str, Any]],
        scorer: ScorerName = "bm25",
        fields: Sequence[str] = DEFAULT_FIELDS,
        top_k: int = 10,
        min_score: float | None = None,
        entity_filter: EntityCandidateFilter | None = None,
        bm25_params: dict[str, float] | None = None,
        embedding_model: str | None = None,
        embedding_cache_dir: str | Path | None = "artifacts/retrieval_indexes",
    ) -> None:
        self.claims = claims
        self.scorer_name: ScorerName = scorer
        self.fields = tuple(fields)
        self.top_k = top_k
        self.min_score = min_score
        self.entity_filter = entity_filter
        docs = [document_text(c, self.fields) for c in claims]
        if scorer == "bm25":
            self.scorer = BM25(**(bm25_params or {})).fit(docs)
        elif scorer == "embedding":
            from indonesian_hoax_checker.retrieval.embeddings import DEFAULT_MODEL, EmbeddingScorer

            self.scorer = EmbeddingScorer(
                embedding_model or DEFAULT_MODEL, cache_dir=embedding_cache_dir
            ).fit(docs)
        else:
            raise ValueError(f"scorer tidak dikenal: {scorer}")

    def scores(self, query: str, entities: Sequence[Entity] | None = None) -> np.ndarray:
        s = self.scorer.score(query)
        if self.entity_filter is not None and entities is not None:
            s = self.entity_filter.apply(s, entities)
        return s

    def rank(
        self, query: str, entities: Sequence[Entity] | None = None, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """[(indeks klaim, skor)] terurut menurun; dokumen yang tersaring (-inf) tidak ikut."""
        k = top_k or self.top_k
        s = self.scores(query, entities)
        if s.size == 0:
            return []
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order if np.isfinite(s[i])]

    def ranked_ids(self, query: str, entities: Sequence[Entity] | None = None, top_k: int | None = None) -> list[str]:
        return [self.claims[i]["claim_id"] for i, _ in self.rank(query, entities, top_k)]

    def retrieve(self, query: str, entities: Sequence[Entity] | None = None) -> RetrievalResult:
        cands: list[RetrievalCandidate] = []
        for i, score in self.rank(query, entities):
            if self.min_score is not None and score < self.min_score:
                break
            c = self.claims[i]
            cands.append(
                {
                    "claim_id": c["claim_id"],
                    "claim": c["claim"],
                    "clarification": c.get("clarification", ""),
                    "url": c.get("url", ""),
                    "published_at": c.get("published_at", ""),
                    "score": score,
                }
            )
        first = cands[0] if cands else None
        return {
            "matched_claim": first["claim"] if first else None,
            "clarification": first["clarification"] if first else None,
            "similarity_score": first["score"] if first else None,
            "candidates": cands,
            "scorer": self.scorer_name,
        }


def load_corpus_entities(path: str | Path, claims: list[dict[str, Any]]) -> list[list[Entity]]:
    """claims_entities.jsonl -> daftar entitas sejajar dengan `claims` (kosong bila tidak ada)."""
    by_id = {r["claim_id"]: r.get("entities", []) for r in read_jsonl(path)}
    return [by_id.get(c["claim_id"], []) for c in claims]


def build_retriever(config: dict[str, Any], claims: list[dict[str, Any]] | None = None) -> Retriever:
    """Bangun Retriever dari dict config (lihat configs/models/retrieval_*.yaml)."""
    claims = claims if claims is not None else load_claims(config["claims_path"])
    ent_filter = None
    if config.get("entity_filter"):
        ents_path = config.get("corpus_entities_path")
        if ents_path and Path(ents_path).exists():
            corpus_entities = load_corpus_entities(ents_path, claims)
        else:
            from indonesian_hoax_checker.retrieval.candidate_filter import heuristic_entities

            corpus_entities = [heuristic_entities(c["claim"]) for c in claims]
        ent_filter = EntityCandidateFilter(corpus_entities)
    return Retriever(
        claims,
        scorer=config.get("scorer", "bm25"),
        fields=config.get("fields", DEFAULT_FIELDS),
        top_k=int(config.get("top_k", 10)),
        min_score=config.get("min_score"),
        entity_filter=ent_filter,
        bm25_params=config.get("bm25"),
        embedding_model=config.get("model_name"),
        embedding_cache_dir=config.get("cache_dir", "artifacts/retrieval_indexes"),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
