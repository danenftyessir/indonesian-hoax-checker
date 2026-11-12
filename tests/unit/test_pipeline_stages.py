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
