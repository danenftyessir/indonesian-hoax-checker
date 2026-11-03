"""Data contracts exchanged between components (NER, classification, retrieval, stance).

Initial draft. Adjust once the dataset structure and annotation definitions are final.
"""

from typing import Literal, TypedDict


class Entity(TypedDict):
    text: str
    label: str
    start: int
    end: int


class NERResult(TypedDict):
    entities: list[Entity]
    core_claim: str


class ClassificationResult(TypedDict):
    label: Literal["hoax", "valid"]
    confidence: float


class RetrievalResult(TypedDict):
    matched_claim: str | None
    clarification: str | None
    similarity_score: float | None


class StanceResult(TypedDict):
    label: Literal["supports", "refutes", "unrelated"]
    confidence: float
