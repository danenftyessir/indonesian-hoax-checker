# Pembungkus tiap tahap. Saat ini semuanya mengembalikan nilai netral dengan bentuk sesuai
# common/schemas.py supaya pipeline bisa dijalankan tanpa model yang sebenarnya

from __future__ import annotations

import re

from indonesian_hoax_checker.common.schemas import (
    ClassificationResult,
    NERResult,
    RetrievalResult,
    StanceResult,
)

STUB_STAGES: set[str] = {"ner", "classify", "retrieve", "stance"}


def run_ner(text: str) -> NERResult:
    # Stub: tanpa entitas; klaim inti = kalimat pertama
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return {"entities": [], "core_claim": first}


def run_classify(text: str, ner: NERResult | None = None) -> ClassificationResult:
    # Stub: 0.5 = tidak tahu. Skor asli harus terkalibrasi (docs/experiment_protocol.md)
    return {"label": "valid", "confidence": 0.5}


def run_retrieve(text: str, ner: NERResult) -> RetrievalResult:
    # Stub: tidak ada klaim terbantah yang cocok
    return {"matched_claim": None, "clarification": None, "similarity_score": None}


def run_stance(text: str, retrieval: RetrievalResult) -> StanceResult | None:
    # premise = klaim terbantah, hypothesis = klaim input (docs/decisions/0002)
    # Tanpa klaim cocok tidak ada pasangan yang bisa dinilai
    if retrieval["matched_claim"] is None:
        return None
    return {"label": "unrelated", "confidence": 0.5}
