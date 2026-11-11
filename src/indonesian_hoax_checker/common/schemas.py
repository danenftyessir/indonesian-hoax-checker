# Format data yang dikirim antar komponen (NER, klasifikasi, retrieval, stance).
# Aturan lengkap: docs/decisions/0011-keputusan-teknis-tambahan.md

from typing import Literal, TypedDict


# Satu entitas hasil NER; start/end = posisi karakter di teks input
class Entity(TypedDict):
    text: str
    label: str  # PER, ORG, LOC, TIME, atau CLAIM
    start: int
    end: int


# core_claim = span CLAIM pertama di input, string kosong bila tidak ada
class NERResult(TypedDict):
    entities: list[Entity]
    core_claim: str


# confidence = skor 0-1 dari classifier
class ClassificationResult(TypedDict):
    label: Literal["hoax", "valid"]
    confidence: float


# Tahap retrieval mengembalikan list[RetrievalResult] (top-k, skor tertinggi dulu);
# list kosong berarti tidak ada klaim terbantah yang mirip
class RetrievalResult(TypedDict):
    matched_claim: str | None
    clarification: str | None
    similarity_score: float | None


class StanceResult(TypedDict):
    label: Literal["supports", "refutes", "unrelated"]
    confidence: float
