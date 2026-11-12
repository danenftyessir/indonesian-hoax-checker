import json

import pytest

from indonesian_hoax_checker.retrieval.corpus import write_jsonl
from indonesian_hoax_checker.stance import dataset as D
from indonesian_hoax_checker.stance import labels as L
from indonesian_hoax_checker.stance import metrics as M
from indonesian_hoax_checker.stance.models.similarity_threshold import SimilarityThresholdStance
from indonesian_hoax_checker.stance.predict import evaluate_file, load_model, predict_against_candidates, predict_stance


def test_labels_consistent_with_config():
    cfg = L.load_from_config()
    assert tuple(cfg["labels"]) == L.STANCE_LABELS
    assert L.to_stance("e") == "supports" and L.to_stance("contradiction") == "refutes"
    assert L.to_stance("NEUTRAL") == "unrelated" and L.to_stance("refutes") == "refutes"
    with pytest.raises(ValueError):
        L.to_stance("maybe")


def test_has_negation():
    assert D.has_negation("Bantuan MBG tidak dihentikan")
    assert D.has_negation("gak ada penghentian")
    assert not D.has_negation("Bantuan MBG dihentikan")


def test_convert_indonli_and_brin(tmp_path):
    indonli = tmp_path / "train.jsonl"
    write_jsonl([
        {"pair_id": 1, "premise": "P satu.", "hypothesis": "H tidak satu.", "label": "c", "annotator_type": "lay"},
        {"pair_id": 2, "premise": "P dua.", "hypothesis": "H dua.", "label": "entailment"},
    ], indonli)
    rows = D.convert_indonli(indonli, "train")
    assert [r["label"] for r in rows] == ["refutes", "supports"]
    assert rows[0]["has_negation"] and rows[0]["pair_id"] == "indonli-train-1"

    import pandas as pd

    brin = tmp_path / "brin.csv"
    pd.DataFrame([
        {"ID": "7", "Title": "[SALAH] Vaksin mengandung chip", "Fact": "Tidak ada chip dalam vaksin menurut BPOM.",
         "Classification": "Fabricated", "Label": "1"},
        {"ID": "8", "Title": "Berita benar", "Fact": "x", "Classification": "", "Label": "0"},
    ]).to_csv(brin, index=False)
    b = D.convert_brin(brin)
    assert len(b) == 1 and b[0]["premise"] == "Vaksin mengandung chip" and b[0]["label"] == "refutes"
    assert b[0]["source"] == "brin_fact" and b[0]["hoax_type"] == "Fabricated"


def test_convert_team_requires_two_annotators(tmp_path):
    f = tmp_path / "team.jsonl"
    write_jsonl([{"pair_id": "t1", "premise": "a", "hypothesis": "b", "label": "supports", "annotators": ["x"]}], f)
    with pytest.raises(ValueError):
        D.convert_team(f)
    write_jsonl([{"pair_id": "t1", "premise": "a", "hypothesis": "b", "label": "supports", "annotators": ["x", "y"]},
                 {"pair_id": "t2", "premise": "a", "hypothesis": "b", "label": "", "annotators": []}], f)
    assert len(D.convert_team(f)) == 1


def test_similarity_baseline_never_predicts_refutes():
    m = SimilarityThresholdStance(threshold=0.3)
    pairs = [
        ("Pemerintah menghentikan bantuan MBG bulan depan", "Pemerintah menghentikan bantuan MBG bulan depan"),
        ("Pemerintah menghentikan bantuan MBG bulan depan", "Bantuan MBG tidak dihentikan pemerintah bulan depan"),
        ("Pemerintah menghentikan bantuan MBG bulan depan", "Harga cabai turun di pasar tradisional"),
    ]
    preds = m.predict_batch(pairs)
    assert preds[0][0] == "supports" and preds[2][0] == "unrelated"
    assert all(lab != "refutes" for lab, _ in preds)
    t = m.fit_threshold(pairs, ["supports", "refutes", "unrelated"])
    assert 0 <= t <= 1


def test_metrics_direction_swap_and_negation_split():
    y = ["supports", "refutes", "unrelated", "refutes"]
    p = ["refutes", "refutes", "unrelated", "supports"]
    r = M.evaluate(y, p, has_negation=[False, True, False, True])
    assert r["confusion"]["supports_as_refutes"] == 1 and r["confusion"]["refutes_as_supports"] == 1
    assert r["confusion"]["direction_swap_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["with_negation"]["n"] == 2 and r["without_negation"]["n"] == 2
    assert 0 < r["macro_f1"] < 1


def test_predict_stance_schema():
    m = load_model("similarity", threshold=0.3)
    cand = {"claim_id": "1", "claim": "Pemerintah menghentikan bantuan MBG", "clarification": "",
            "url": "", "published_at": "", "score": 1.0}
    res = predict_stance(m, "Pemerintah menghentikan bantuan MBG", cand)
    assert res == {"label": "supports", "confidence": pytest.approx(1.0), "claim_id": "1"}
    assert predict_stance(m, "apa pun", None) == {"label": "unrelated", "confidence": 1.0, "claim_id": None}
    assert len(predict_against_candidates(m, "x", [cand, cand])) == 2


def test_evaluate_file_roundtrip(tmp_path):
    pairs = [
        {"pair_id": "1", "premise": "a b c", "hypothesis": "a b c", "label": "supports"},
        {"pair_id": "2", "premise": "a b c", "hypothesis": "x y z", "label": "unrelated"},
    ]
    result, rows = evaluate_file(SimilarityThresholdStance(0.5), pairs)
    assert result["macro_f1"] > 0 and rows[0]["pred"] == "supports"
    assert json.dumps(result)  # serialisable
