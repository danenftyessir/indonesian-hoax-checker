import pytest

from indonesian_hoax_checker.pipeline.stages import NERStage


def test_ner_stage_stub():
    stage = NERStage(checkpoint=None)
    assert stage.is_stub
    assert stage("apa saja") == {"entities": [], "core_claim": ""}


def test_ner_stage_from_config(tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("ner:\n  checkpoint: null\n")
    assert NERStage.from_config(str(cfg)).is_stub
    cfg.write_text("ner:\n  checkpoint: artifacts/x\n  device: cpu\n")
    stage = NERStage.from_config(str(cfg))
    assert (stage.checkpoint, stage.device) == ("artifacts/x", "cpu")


def test_ner_stage_missing_checkpoint_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="hasil_training"):
        NERStage(str(tmp_path / "tidak_ada"))("teks")


def test_ner_stage_uses_predictor(tmp_path):
    from indonesian_hoax_checker.ner.labels import LABELS
    from indonesian_hoax_checker.ner.models import bilstm_crf

    words, chars = bilstm_crf.build_vocabs([["Joko", "ke", "Bandung"]])
    args = {"word_dim": 8, "char_dim": 4, "char_filters": 6, "hidden": 10, "dropout": 0.0}
    bilstm_crf.save(tmp_path, bilstm_crf.BiLSTMCRF(len(words), len(chars), len(LABELS), **args), words, chars,
                    {"model_args": args})
    stage = NERStage(str(tmp_path), device="cpu")
    results = stage.run_batch(["Joko ke Bandung.", "Halo"])
    assert len(results) == 2 and set(results[0]) == {"entities", "core_claim"}


# --- retrieval dan stance dari configs/pipeline.yaml ---

import json

from indonesian_hoax_checker.pipeline import stages


@pytest.fixture
def pipeline_cfg(tmp_path, monkeypatch):
    """Config pipeline sementara + reset cache model di stages."""
    claims = tmp_path / "claims.jsonl"
    rows = [
        {"claim_id": "1", "claim": "Air nanas panas sembuhkan kanker", "clarification": "Tidak benar",
         "url": "u1", "published_at": "2026-09-01"},
        {"claim_id": "2", "claim": "Pertamina batasi pembelian BBM di SPBU", "clarification": "Hoaks",
         "url": "u2", "published_at": "2026-09-02"},
    ]
    claims.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    model_cfg = tmp_path / "retrieval.yaml"
    model_cfg.write_text(f"scorer: bm25\nclaims_path: {claims.as_posix()}\nfields: [claim]\n", encoding="utf-8")
    cfg = tmp_path / "pipeline.yaml"

    def write(retrieval: str, stance: str) -> None:
        cfg.write_text(f"retrieval:\n{retrieval}\nstance:\n{stance}\n", encoding="utf-8")

    write(f"  config: {model_cfg.as_posix()}\n  min_score: 0.5", "  model: similarity\n  threshold: 0.3")
    monkeypatch.setattr(stages, "PIPELINE_CONFIG", str(cfg))
    for name, value in [("_retriever", None), ("_retriever_failed", False),
                        ("_stance_model", None), ("_stance_failed", False)]:
        monkeypatch.setattr(stages, name, value)
    monkeypatch.setattr(stages, "STUB_STAGES", {"ner", "classify", "retrieve", "stance"})
    return write


def test_run_retrieve_uses_config(pipeline_cfg):
    res = stages.run_retrieve("benarkah air nanas panas bisa sembuhkan kanker", None)
    assert res["matched_claim"] == "Air nanas panas sembuhkan kanker"
    assert res["candidates"][0]["claim_id"] == "1" and res["scorer"] == "bm25"
    assert "retrieve" not in stages.STUB_STAGES


def test_run_retrieve_min_score_rejects(pipeline_cfg):
    res = stages.run_retrieve("taman baca kelurahan", None)
    assert res["matched_claim"] is None and res["candidates"] == []


def test_run_retrieve_stub_without_config(pipeline_cfg, tmp_path):
    pipeline_cfg("  config: null", "  model: null")
    res = stages.run_retrieve("apa saja", None)
    assert res["matched_claim"] is None and res["candidates"] == []
    assert "retrieve" in stages.STUB_STAGES


def test_run_stance_uses_top_candidate(pipeline_cfg):
    retrieval = stages.run_retrieve("air nanas panas sembuhkan kanker", None)
    st = stages.run_stance("air nanas panas sembuhkan kanker", retrieval)
    assert st["label"] == "supports" and st["claim_id"] == "1"
    assert "stance" not in stages.STUB_STAGES


def test_run_stance_none_without_match(pipeline_cfg):
    assert stages.run_stance("x", {"matched_claim": None, "clarification": None, "similarity_score": None,
                                   "candidates": [], "scorer": "bm25"}) is None


def test_run_stance_missing_local_checkpoint_is_stub(pipeline_cfg, tmp_path):
    pipeline_cfg("  config: null", f"  model: {(tmp_path / 'tidak_ada').as_posix()}")
    retrieval = {"matched_claim": "k", "clarification": "c", "similarity_score": 1.0, "scorer": "bm25",
                 "candidates": [{"claim_id": "1", "claim": "k", "clarification": "c", "url": "", "published_at": "",
                                 "score": 1.0}]}
    st = stages.run_stance("k", retrieval)
    assert st["label"] == "unrelated" and "stance" in stages.STUB_STAGES


def test_combine_directions():
    from indonesian_hoax_checker.stance.predict import combine_directions as cd

    assert cd(("unrelated", 0.9), ("supports", 0.8)) == ("supports", 0.8)
    assert cd(("supports", 0.7), ("supports", 0.9)) == ("supports", 0.9)
    assert cd(("refutes", 0.6), ("unrelated", 0.9)) == ("refutes", 0.6)
    assert cd(("supports", 0.9), ("refutes", 0.9)) == ("unrelated", 0.5)
    assert cd(("unrelated", 0.9), ("unrelated", 0.7)) == ("unrelated", 0.7)


def test_run_stance_bidirectional(pipeline_cfg, monkeypatch):
    class OneWay:  # supports hanya bila premise lebih panjang dari hypothesis
        def predict_batch(self, pairs):
            return [("supports", 0.9) if len(p) > len(h) else ("unrelated", 0.8) for p, h in pairs]

    retrieval = stages.run_retrieve("air nanas panas sembuhkan kanker", None)
    long_input = "katanya air nanas panas bisa sembuhkan kanker, sebarkan ke keluarga"
    monkeypatch.setattr(stages, "_stance_model", OneWay())
    assert stages.run_stance(long_input, retrieval)["label"] == "unrelated"
    cfg = stages._pipeline_section("retrieval")["config"]
    pipeline_cfg(f"  config: {cfg}\n  min_score: 0.5", "  model: similarity\n  bidirectional: true")
    st = stages.run_stance(long_input, retrieval)
    assert st["label"] == "supports" and st["claim_id"] == "1"


def test_stance_skip_is_not_stub_when_model_available(pipeline_cfg):
    # retrieval nyata tanpa padanan + model stance tersedia -> "dilewati" bukan stub, tanpa memuat model
    retrieval = stages.run_retrieve("taman baca kelurahan", None)
    assert stages.run_stance("taman baca kelurahan", retrieval) is None
    assert "stance" not in stages.STUB_STAGES and stages._stance_model is None


def test_stance_skip_stays_stub_without_model(pipeline_cfg):
    cfg = stages._pipeline_section("retrieval")["config"]
    pipeline_cfg(f"  config: {cfg}\n  min_score: 0.5", "  model: null")
    retrieval = stages.run_retrieve("taman baca kelurahan", None)
    assert stages.run_stance("taman baca kelurahan", retrieval) is None
    assert "stance" in stages.STUB_STAGES
