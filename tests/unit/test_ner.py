from pathlib import Path

import yaml

from indonesian_hoax_checker.ner.dataset import dedup, read_conll, write_conll
from indonesian_hoax_checker.ner.labels import (
    NERGRIT_MAPPING,
    NERP_MAPPING,
    TYPES,
    get_spans,
    map_tag,
    to_iob2,
)


def test_labels_match_config():
    config = yaml.safe_load(Path("configs/labels.yaml").read_text())["ner"]
    assert TYPES == config["types"]
    assert NERGRIT_MAPPING == config["nergrit_mapping"]
    assert NERP_MAPPING == {k: v for k, v in config["nerp_mapping"].items() if v != "O"}


def test_map_tag():
    assert map_tag("B-PERSON", NERGRIT_MAPPING) == "B-PER"
    assert map_tag("I-PLACE", NERGRIT_MAPPING) == "I-LOC"
    assert map_tag("B-EVT", NERP_MAPPING) == "O"
    assert map_tag("O", NERP_MAPPING) == "O"


def test_to_iob2_fixes_orphan_inside_tag():
    assert to_iob2(["O", "I-PER", "I-PER", "I-LOC"]) == ["O", "B-PER", "I-PER", "B-LOC"]


def test_get_spans():
    tags = ["B-ORG", "B-CLAIM", "I-CLAIM", "O", "B-TIME", "I-TIME"]
    assert get_spans(tags) == [("ORG", 0, 1), ("CLAIM", 1, 3), ("TIME", 4, 6)]


def test_dedup_keeps_first_and_excludes():
    a = (["x", "y"], ["O", "O"])
    a_other_tags = (["x", "y"], ["B-PER", "O"])
    b = (["z"], ["O"])
    assert dedup([a, a_other_tags, b]) == [a, b]
    assert dedup([a, b], exclude=[b]) == [a]


def test_conll_roundtrip(tmp_path):
    sentences = [(["Pemerintah", "menghentikan"], ["B-ORG", "B-CLAIM"]), (["besok"], ["B-TIME"])]
    path = tmp_path / "x.conll"
    write_conll(path, sentences)
    assert read_conll(path) == sentences


def test_entity_scores_strict():
    from indonesian_hoax_checker.ner.metrics import entity_scores

    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    pred = [["B-PER", "O", "O", "B-LOC"]]  # PER terpotong = salah, LOC benar
    scores = entity_scores(gold, pred)
    assert scores["precision"] == 0.5
    assert scores["per_type"]["LOC"]["f1"] == 1.0
    assert scores["per_type"]["PER"]["f1"] == 0.0


def test_bilstm_crf_forward():
    import torch

    from indonesian_hoax_checker.ner.labels import LABELS
    from indonesian_hoax_checker.ner.models import bilstm_crf

    sentences = [["Joko", "ke", "Bandung"], ["Halo"]]
    words, chars = bilstm_crf.build_vocabs(sentences)
    model = bilstm_crf.BiLSTMCRF(len(words), len(chars), len(LABELS), hidden=20, word_dim=10, char_dim=5, char_filters=6)
    word_ids, char_ids, mask = bilstm_crf.encode_batch(sentences, words, chars)
    loss = model.loss(word_ids, char_ids, mask, torch.zeros_like(word_ids))
    assert loss.item() > 0
    assert [len(p) for p in model.decode(word_ids, char_ids, mask)] == [3, 1]
