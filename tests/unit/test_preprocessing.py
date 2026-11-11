import random

from indonesian_hoax_checker.preprocessing.augment import augment_corpus, informalize
from indonesian_hoax_checker.preprocessing.normalize import normalize
from indonesian_hoax_checker.preprocessing.slang import FORMAL_TO_SLANG, SLANG_TO_FORMAL


def test_normalize_keeps_length_and_fixes_slang():
    tokens = "Yg bsk sabarrr , Joko di DKI".split()
    assert normalize(tokens) == ["Yang", "besok", "sabar", ",", "Joko", "di", "DKI"]


def test_normalize_lowercases_shouting():
    assert normalize("PEMERINTAH AKAN MEMUTUS INTERNET".split()) == ["pemerintah", "akan", "memutus", "internet"]


def test_slang_dictionary_has_no_ambiguous_entries():
    assert "dr" not in SLANG_TO_FORMAL and "km" not in SLANG_TO_FORMAL
    assert FORMAL_TO_SLANG["yang"] == "yg"


def test_informalize_keeps_tags_aligned():
    tokens = "Presiden Joko Widodo yang hadir di Bali .".split()
    tags = ["O", "B-PER", "I-PER", "O", "O", "O", "B-LOC", "O"]
    probs = {"lower": 1.0, "slang": 1.0, "typo": 0.0, "elongate": 0.0, "exclaim": 1.0}
    out_tokens, out_tags = informalize(tokens, tags, random.Random(0), probs)
    assert out_tokens == ["presiden", "joko", "widodo", "yg", "hadir", "di", "bali", "!", "!", "!"]
    assert out_tags == tags[:-1] + ["O", "O", "O"]


def test_augment_corpus_is_deterministic():
    data = [("Joko yang hadir di Bandung untuk rapat".split(), ["B-PER"] + ["O"] * 6)]
    assert augment_corpus(data, copies=2, seed=1) == augment_corpus(data, copies=2, seed=1)
    assert len(augment_corpus(data, copies=2, seed=1)) == 2
