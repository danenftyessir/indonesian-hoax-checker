# Penggabungan hasil klasifikasi, retrieval, dan stance menjadi indikasi kredibilitas
# threshold dipilih di fase 4 (RQ5)
# Skor retrieval tidak dibatasi (BM25) atau -1..1 (cosine), jadi similarity_score tidak dibandingkan antar-scorer
# Confidence stance masih softmax mentah (belum dikalibrasi); dianggap setara skor mentah sampai kalibrasi

from __future__ import annotations

from typing import TypedDict

from indonesian_hoax_checker.common.schemas import (
    ClassificationResult,
    RetrievalResult,
    StanceResult,
)

REVIEW_THRESHOLD = 0.6  


class Verdict(TypedDict):
    indication: str  # rendah / kemungkinan benar / hanya dari classifier / tinjauan manusia
    explanation: str
    needs_review: bool


def combine(
    classification: ClassificationResult,
    retrieval: RetrievalResult,
    stance: StanceResult | None,
    classifier_note: str | None = None,
) -> Verdict:
    label, conf = classification["label"], classification["confidence"]

    # Bukti dari basis klaim terbantah didahulukan utk menentukan arti skor classifier
    if retrieval["matched_claim"] is not None and stance is not None:
        if stance["label"] == "supports":
            return {
                "indication": "kredibilitas rendah",
                "explanation": "Input mengulang klaim yang sudah dibantah; lihat bantahan resmi",
                "needs_review": False,
            }
        if stance["label"] == "refutes":
            return {
                "indication": "kemungkinan benar",
                "explanation": "Input menyangkal klaim yang sudah dibantah",
                "needs_review": False,
            }

    if conf < REVIEW_THRESHOLD:
        return {
            "indication": "perlu tinjauan manusia",
            "explanation": classifier_note or f"Keyakinan classifier ({conf:.2f}) di bawah ambang {REVIEW_THRESHOLD}",
            "needs_review": True,
        }
    return {
        "indication": "terindikasi hoaks" if label == "hoax" else "tidak terindikasi hoaks",
        "explanation": "Hanya dari classifier (skor belum dikalibrasi); tidak ada klaim terbantah yang cocok, sampaikan hati-hati",
        "needs_review": False,
    }
