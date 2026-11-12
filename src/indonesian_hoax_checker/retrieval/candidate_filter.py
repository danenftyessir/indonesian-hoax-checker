# Penyempitan kandidat berbasis entitas (RQ3): hanya klaim di basis yang berbagi >= 1 entitas
# dengan kueri yang dipertimbangkan. Bila tidak ada yang berbagi, kembalikan semua (fallback)
# supaya recall tidak jatuh ke nol karena NER meleset.
#
# Entitas basis klaim berasal dari prediksi NER (0005): file data/processed/refuted_claims/
# claims_entities.jsonl, satu baris {"claim_id": ..., "entities": [Entity, ...]}. Selama NER
# belum ada, `heuristic_entities` memberi tebakan kasar dari huruf kapital; hanya untuk uji coba.

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from indonesian_hoax_checker.common.schemas import Entity

# Entitas yang dipakai untuk narrowing. TIME dan CLAIM tidak dipakai: terlalu umum (TIME) atau
# justru inti yang dicari secara leksikal/semantik (CLAIM).
NARROWING_LABELS = ("PER", "ORG", "LOC")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_CAP_RE = re.compile(r"\b[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*")
_STOP_CAP = {
    "beredar", "video", "foto", "unggahan", "informasi", "narasi", "hoaks", "klarifikasi",
    "faktanya", "pesan", "tautan", "link", "kabar", "berita",
}


def normalize_entity(text: str) -> str:
    """'Kementerian Kesehatan RI' -> 'kementerian kesehatan ri'; hilangkan tanda baca."""
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def entity_keys(entities: Iterable[Entity | dict], labels: Iterable[str] = NARROWING_LABELS) -> set[str]:
    """Kunci pencocokan: entitas ternormalisasi plus tiap token-nya yang > 3 huruf, supaya
    'Kemenkes' dan 'Kementerian Kesehatan' masih bisa beririsan lewat token (kasar, sengaja)."""
    labels = set(labels)
    keys: set[str] = set()
    for e in entities:
        if e.get("label") not in labels:
            continue
        norm = normalize_entity(e["text"])
        if not norm:
            continue
        keys.add(norm)
        keys.update(t for t in norm.split() if len(t) > 3)
    return keys


def heuristic_entities(text: str) -> list[Entity]:
    """Stub: rangkaian kata berawalan kapital (bukan di awal kalimat) dianggap ORG.
    Hanya untuk uji coba sebelum model NER baseline tersedia."""
    out: list[Entity] = []
    for m in _CAP_RE.finditer(text):
        if m.start() == 0 or text[m.start() - 2 : m.start()].strip() in {".", "!", "?"}:
            continue
        if m.group(0).lower() in _STOP_CAP:
            continue
        out.append({"text": m.group(0), "label": "ORG", "start": m.start(), "end": m.end()})
    return out


class EntityCandidateFilter:
    def __init__(self, corpus_entities: list[list[Entity]]) -> None:
        self.corpus_keys = [entity_keys(ents) for ents in corpus_entities]

    def mask(self, query_entities: Iterable[Entity]) -> np.ndarray:
        """Boolean array: True untuk dokumen yang berbagi >= 1 kunci entitas dengan kueri.
        Semua True bila kueri tanpa entitas atau tidak ada dokumen yang beririsan."""
        qkeys = entity_keys(query_entities)
        n = len(self.corpus_keys)
        if not qkeys:
            return np.ones(n, dtype=bool)
        m = np.array([bool(qkeys & dkeys) for dkeys in self.corpus_keys], dtype=bool)
        if not m.any():
            return np.ones(n, dtype=bool)
        return m

    def apply(self, scores: np.ndarray, query_entities: Iterable[Entity]) -> np.ndarray:
        """Skor dokumen di luar mask dibuat -inf supaya tersingkir dari peringkat."""
        m = self.mask(query_entities)
        out = scores.astype(float).copy()
        out[~m] = -np.inf
        return out
