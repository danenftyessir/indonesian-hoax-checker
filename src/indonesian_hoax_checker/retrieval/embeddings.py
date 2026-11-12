# Retrieval dengan sentence embedding siap pakai + cosine (model kuat task 3a).
# Kandidat utama: intfloat/multilingual-e5-base (configs/models/retrieval_embeddings.yaml).
# sentence-transformers diimpor malas supaya BM25, tes, dan --help tetap cepat.

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


class EmbeddingScorer:
    """Encode korpus sekali (cache .npy di artifacts/retrieval_indexes/), lalu cosine per kueri.
    Model e5 butuh prefiks 'query: ' / 'passage: '; model lain tidak."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = "artifacts/retrieval_indexes",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.doc_embeddings: np.ndarray = np.zeros((0, 0))

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # impor malas

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        if _is_e5(self.model_name):
            texts = [f"{kind}: {t}" for t in texts]
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(emb, dtype=np.float32)

    def _cache_path(self, docs: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        for d in docs:
            h.update(d.encode("utf-8"))
            h.update(b"\x00")
        return self.cache_dir / f"emb_{self.model_name.replace('/', '__')}_{h.hexdigest()[:16]}.npy"

    def fit(self, docs: list[str]) -> EmbeddingScorer:
        cache = self._cache_path(docs)
        if cache is not None and cache.exists():
            self.doc_embeddings = np.load(cache)
            return self
        self.doc_embeddings = self._encode(docs, "passage")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.doc_embeddings)
        return self

    def score(self, query: str) -> np.ndarray:
        """Cosine kueri terhadap semua dokumen (embedding sudah dinormalisasi)."""
        if self.doc_embeddings.size == 0:
            return np.zeros(0)
        q = self._encode([query], "query")[0]
        return self.doc_embeddings @ q

    def score_many(self, queries: list[str]) -> np.ndarray:
        if self.doc_embeddings.size == 0:
            return np.zeros((len(queries), 0))
        q = self._encode(queries, "query")
        return q @ self.doc_embeddings.T
