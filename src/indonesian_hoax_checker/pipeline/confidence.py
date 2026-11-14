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
    indication: str  # kredibilitas rendah / hanya dari classifier / tinjauan manusia
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
            # Bukan bukti kebenaran: di data/demo_cases.txt (14 Nov 2026) 5 dari 8 `refutes` keliru, yaitu klaim
            # lain bertopik sama (diskon 80% vs 50%), satir, pesan yang justru mengulang hoaks, dan teks Inggris.
            # Maka tidak dijadikan verdict otomatis; bantahan resmi ditampilkan untuk dicek manusia.
            return {
                "indication": "perlu tinjauan manusia",
                "explanation": "Input tampak menyangkal klaim yang sudah dibantah; cocokkan dengan bantahan resmi "
                               "sebelum menyimpulkan",
                "needs_review": True,
            }

    if conf < REVIEW_THRESHOLD:
        return {
            "indication": "perlu tinjauan manusia",
            "explanation": classifier_note or f"Keyakinan classifier ({conf:.2f}) di bawah ambang {REVIEW_THRESHOLD}",
            "needs_review": True,
        }
    # Klaim mirip bisa saja ditemukan retrieval tetapi stance menilainya tidak berkaitan
    evidence = ("klaim terbantah yang mirip dinilai tidak berkaitan" if retrieval["matched_claim"] is not None
                else "tidak ada klaim terbantah yang cocok")
    return {
        "indication": "terindikasi hoaks" if label == "hoax" else "tidak terindikasi hoaks",
        "explanation": f"Hanya dari classifier (skor belum dikalibrasi); {evidence}, sampaikan hati-hati",
        "needs_review": False,
    }
