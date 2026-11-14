from indonesian_hoax_checker.ner.claim_trim import DEFAULT_RULES, trim_claims


def claim(tokens, tags):
    return " ".join(t for t, g in zip(tokens, tags) if g != "O")


def test_time_inside_claim_is_cut_with_preposition():
    tokens = "pemerintah hentikan bantuan mulai bulan depan".split()
    entity = ["O", "O", "O", "O", "B-TIME", "I-TIME"]
    claim_tags = ["O", "B-CLAIM", "I-CLAIM", "I-CLAIM", "I-CLAIM", "I-CLAIM"]
    assert claim(tokens, trim_claims(tokens, entity, claim_tags, DEFAULT_RULES)) == "hentikan bantuan"


def test_question_mark_ends_claim():
    tokens = "vaksin bikin mandul ? ? hati2 ya".split()
    entity = ["O"] * len(tokens)
    claim_tags = ["B-CLAIM"] + ["I-CLAIM"] * (len(tokens) - 1)
    assert claim(tokens, trim_claims(tokens, entity, claim_tags, DEFAULT_RULES)) == "vaksin bikin mandul"


def test_claim_without_trailing_extras_is_unchanged():
    tokens = "Harga Pertalite naik menjadi Rp13 ribu".split()
    entity = ["O"] * len(tokens)
    claim_tags = ["O", "O", "B-CLAIM", "I-CLAIM", "I-CLAIM", "I-CLAIM"]
    assert trim_claims(tokens, entity, claim_tags, DEFAULT_RULES) == claim_tags


def test_claim_keeps_at_least_first_token():
    tokens = "mulai besok".split()
    entity = ["O", "B-TIME"]
    claim_tags = ["B-CLAIM", "I-CLAIM"]
    assert claim(tokens, trim_claims(tokens, entity, claim_tags, DEFAULT_RULES)) == "mulai"
