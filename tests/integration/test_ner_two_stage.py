# Uji alur training dua tahap BiLSTM-CRF dengan data mini di CPU (beberapa detik).

from indonesian_hoax_checker.ner.dataset import write_conll
from indonesian_hoax_checker.ner.predict import NERPredictor
from indonesian_hoax_checker.ner.train import run

STAGE1 = [(["Joko", "ke", "Bandung"], ["B-PER", "O", "B-LOC"]), (["Ani", "di", "Kemenkes"], ["B-PER", "O", "B-ORG"])]
STAGE2 = [(["Pemerintah", "menghentikan", "bansos", "besok"], ["B-ORG", "B-CLAIM", "I-CLAIM", "B-TIME"])] * 3


def test_two_stage_bilstm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, data in (("s1", STAGE1), ("s2", STAGE2)):
        for split in ("train", "dev", "test"):
            write_conll(f"{name}_{split}.conll", data)

    common = {"model": "bilstm_crf", "lr": 0.01, "batch_size": 2, "epochs": 2, "patience": 2, "device": "cpu"}
    args = {"word_dim": 8, "char_dim": 4, "char_filters": 6, "hidden": 10, "dropout": 0.0}
    run({**common, "name": "stage1", "model_args": args,
         "data": {s: f"s1_{s}.conll" for s in ("train", "dev", "test")}}, seed=1)
    result = run({**common, "name": "stage2", "init_from": "artifacts/checkpoints/ner/stage1_seed{seed}",
                  "data": {s: f"s2_{s}.conll" for s in ("train", "dev", "test")},
                  "extra_eval": {"stage1_test": "s1_test.conll"}}, seed=1)

    assert "stage1_test" in result["extra"]
    predictor = NERPredictor("artifacts/checkpoints/ner/stage2_seed1", device="cpu")
    assert set(predictor.predict("Pemerintah menghentikan bansos besok")) == {"entities", "core_claim"}


def test_rq1_options_run(tmp_path, monkeypatch):
    # normalize + augment + daftar file train + extra_eval yang filenya tidak ada
    monkeypatch.chdir(tmp_path)
    for split in ("train", "dev", "test"):
        write_conll(f"s1_{split}.conll", STAGE1)
    cfg = {"model": "bilstm_crf", "name": "rq1", "lr": 0.01, "batch_size": 2, "epochs": 1, "patience": 1,
           "device": "cpu", "normalize": True, "augment": {"copies": 1},
           "model_args": {"word_dim": 8, "char_dim": 4, "char_filters": 6, "hidden": 10, "dropout": 0.0},
           "data": {"train": ["s1_train.conll", "s1_dev.conll"], "dev": "s1_dev.conll", "test": "s1_test.conll"},
           "extra_eval": {"informal": "s1_test.conll", "missing": "tidak_ada.conll"}}
    result = run(cfg, seed=3)
    assert set(result["extra"]) == {"informal"}
    assert NERPredictor("artifacts/checkpoints/ner/rq1_seed3", device="cpu").normalize is True
