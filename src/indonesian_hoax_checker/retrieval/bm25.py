# BM25 (Okapi) baseline retrieval, task 3a. Implementasi sendiri supaya tidak bergantung pada
# rank-bm25 dan supaya skor per dokumen bisa diambil sebagai array (dipakai candidate_filter).

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Huruf kecil, token alfanumerik; tanda baca dibuang. Tanpa stemming (bandingkan di RQ3)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: np.ndarray = np.zeros(0)
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, dict[int, int]] = {}
        self.n_docs = 0

    def fit(self, docs: list[str]) -> BM25:
        tokenized = [tokenize(d) for d in docs]
        self.n_docs = len(tokenized)
        self.doc_len = np.array([len(t) for t in tokenized], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        self.postings = {}
        for i, toks in enumerate(tokenized):
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, {})[i] = tf
            df.update(set(toks))
        # IDF Okapi dengan pembatas bawah 0 (hindari skor negatif untuk term yang sangat umum)
        self.idf = {
            t: max(0.0, math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)) for t, n in df.items()
        }
        return self

    def score(self, query: str) -> np.ndarray:
        """Skor BM25 untuk setiap dokumen (array sepanjang n_docs)."""
        scores = np.zeros(self.n_docs)
        if self.n_docs == 0:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_idx, tf in posting.items():
                scores[doc_idx] += idf * tf * (self.k1 + 1) / (tf + norm[doc_idx])
        return scores

    # Simpan/muat indeks (artifacts/retrieval_indexes/) supaya demo tidak mengindeks ulang.
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len.tolist(),
            "avg_len": self.avg_len,
            "idf": self.idf,
            "postings": {t: list(p.items()) for t, p in self.postings.items()},
            "n_docs": self.n_docs,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BM25:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(payload["k1"], payload["b"])
        obj.doc_len = np.array(payload["doc_len"], dtype=float)
        obj.avg_len = payload["avg_len"]
        obj.idf = payload["idf"]
        obj.postings = {t: {int(i): tf for i, tf in p} for t, p in payload["postings"].items()}
        obj.n_docs = payload["n_docs"]
        return obj
