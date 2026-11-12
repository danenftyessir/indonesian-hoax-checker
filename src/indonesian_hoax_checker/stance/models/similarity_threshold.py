# Baseline stance RQ4: ambang similarity saja. Tidak pernah mengeluarkan `refutes`, karena
# similarity tidak melihat arah. Inilah yang ingin dibuktikan RQ4: baseline ini gagal memisahkan
# hoaks yang diulang dari koreksinya.

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from indonesian_hoax_checker.common.schemas import StanceLabel

# token 1 huruf pun ikut (default sklearn butuh >= 2 karakter)
_TOKEN_PATTERN = r"(?u)\b\w+\b"


def tfidf_cosine(pairs: Sequence[tuple[str, str]]) -> np.ndarray:
    """Cosine TF-IDF kata (1–2-gram) tiap pasangan, dilatih pada teks pasangan itu sendiri."""
    if not pairs:
        return np.zeros(0)
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=_TOKEN_PATTERN)
    texts = [p for p, _ in pairs] + [h for _, h in pairs]
    try:
        x = vec.fit_transform(texts)
    except ValueError:  # semua teks kosong -> tidak ada kosakata
        return np.zeros(len(pairs))
    n = len(pairs)
    a, b = x[:n], x[n:]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


class SimilarityThresholdStance:
    """similarity >= threshold -> supports, selain itu unrelated.
    `similarity_fn` menerima daftar (premise, hypothesis) dan mengembalikan array skor; default
    TF-IDF cosine, bisa diganti cosine embedding (EmbeddingScorer) untuk varian yang lebih kuat."""

    def __init__(
        self,
        threshold: float = 0.5,
        similarity_fn: Callable[[Sequence[tuple[str, str]]], np.ndarray] = tfidf_cosine,
    ) -> None:
        self.threshold = threshold
        self.similarity_fn = similarity_fn

    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[StanceLabel, float]]:
        sims = self.similarity_fn(pairs)
        out: list[tuple[StanceLabel, float]] = []
        for s in sims:
            s = float(s)
            if s >= self.threshold:
                out.append(("supports", s))
            else:
                out.append(("unrelated", 1.0 - s))
        return out

    def predict(self, premise: str, hypothesis: str) -> tuple[StanceLabel, float]:
        return self.predict_batch([(premise, hypothesis)])[0]

    def fit_threshold(
        self, pairs: Sequence[tuple[str, str]], labels: Sequence[str], n_grid: int = 50
    ) -> float:
        """Pilih ambang pada dev yang memaksimalkan macro-F1 (refutes tetap tak terjangkau)."""
        from indonesian_hoax_checker.stance.metrics import macro_f1

        sims = self.similarity_fn(pairs)
        best_t, best_f = self.threshold, -1.0
        for t in np.linspace(float(sims.min()), float(sims.max()), n_grid):
            pred = ["supports" if s >= t else "unrelated" for s in sims]
            f = macro_f1(labels, pred)
            if f > best_f:
                best_t, best_f = float(t), f
        self.threshold = best_t
        return best_t
