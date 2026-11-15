import pytest

from indonesian_hoax_checker.pipeline.confidence import combine

MATCH = {"matched_claim": "Air nanas panas sembuhkan kanker", "clarification": "Hoaks", "similarity_score": 0.9,
         "candidates": [], "scorer": "embedding"}
NO_MATCH = {"matched_claim": None, "clarification": None, "similarity_score": None, "candidates": [],
            "scorer": "embedding"}


def test_supports_low_credibility():
    v = combine({"label": "valid", "confidence": 0.5}, MATCH, {"label": "supports", "confidence": 0.9, "claim_id": "1"})
    assert v["indication"] == "kredibilitas rendah" and not v["needs_review"]


@pytest.mark.parametrize("cls", [{"label": "valid", "confidence": 0.95}, {"label": "hoax", "confidence": 0.95}])
def test_refutes_goes_to_review(cls):
    v = combine(cls, MATCH, {"label": "refutes", "confidence": 0.99, "claim_id": "1"})
    assert v["indication"] == "perlu tinjauan manusia" and v["needs_review"]


def test_unrelated_falls_back_to_classifier():
    v = combine({"label": "hoax", "confidence": 0.9}, MATCH, {"label": "unrelated", "confidence": 0.9, "claim_id": "1"})
    assert v["indication"] == "terindikasi hoaks"


def test_no_match_low_classifier_confidence_needs_review():
    v = combine({"label": "valid", "confidence": 0.5}, NO_MATCH, None, "teks pendek")
    assert v["needs_review"] and v["explanation"] == "teks pendek"


def test_explanation_reflects_retrieval_evidence():
    cls = {"label": "hoax", "confidence": 0.9}
    with_match = combine(cls, MATCH, {"label": "unrelated", "confidence": 0.9, "claim_id": "1"})
    assert "dinilai tidak berkaitan" in with_match["explanation"]
    assert "tidak ada klaim terbantah yang cocok" in combine(cls, NO_MATCH, None)["explanation"]


def test_strong_match_without_stance_decision_goes_to_review():
    unrelated = {"label": "unrelated", "confidence": 0.9, "claim_id": "1"}
    confident = {"label": "valid", "confidence": 0.99}
    assert combine(confident, MATCH, unrelated)["indication"] == "tidak terindikasi hoaks"
    held = combine(confident, MATCH, unrelated, None, False, True)
    assert held["needs_review"] and "sangat mirip" in held["explanation"]
    assert combine(confident, NO_MATCH, None, None, False, True)["indication"] == "tidak terindikasi hoaks"
    supports = {"label": "supports", "confidence": 0.9, "claim_id": "1"}
    assert combine(confident, MATCH, supports, None, False, True)["indication"] == "kredibilitas rendah"
