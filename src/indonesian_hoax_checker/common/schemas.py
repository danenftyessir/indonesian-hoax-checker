# Format data yang dikirim antar komponen (NER, klasifikasi, retrieval, stance).
# Masih draf: disepakati bertiga, Danendra yang meninjau.

from typing import Literal, TypedDict


# Satu entitas hasil NER; start/end = posisi karakter di teks input
class Entity(TypedDict):
    text: str
    label: str  # PER, ORG, LOC, TIME, atau CLAIM
    start: int
    end: int


class NERResult(TypedDict):
    entities: list[Entity]
    core_claim: str


# confidence = skor 0-1 dari classifier
class ClassificationResult(TypedDict):
    label: Literal["hoax", "valid"]
    confidence: float


# Bernilai None kalau tidak ada klaim terbantah yang mirip
class RetrievalResult(TypedDict):
    matched_claim: str | None
    clarification: str | None
    similarity_score: float | None


class StanceResult(TypedDict):
    label: Literal["supports", "refutes", "unrelated"]
    confidence: float
