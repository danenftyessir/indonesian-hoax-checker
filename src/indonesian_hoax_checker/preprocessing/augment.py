# Augmentasi teks informal untuk RQ1: membuat versi "media sosial" dari kalimat berlabel.
# Label ikut diperbarui, jadi hasilnya langsung bisa dipakai untuk training/evaluasi.
#   - huruf kecil semua (sering di pesan WhatsApp)
#   - kata baku -> singkatan ("yang" -> "yg")
#   - typo: tukar, hapus, atau gandakan satu huruf
#   - huruf terakhir dipanjangkan ("banget" -> "bangettt")
#   - tanda titik akhir diganti "!!!" (token tambahan berlabel O)
#
# Buat set uji informal dari file CoNLL:
#   python -m indonesian_hoax_checker.preprocessing.augment --input a.conll --output b.conll

import argparse
import random

from indonesian_hoax_checker.ner.dataset import Sentence, read_conll, write_conll
from indonesian_hoax_checker.preprocessing.slang import FORMAL_TO_SLANG

DEFAULT_PROBS = {"lower": 0.5, "slang": 0.5, "typo": 0.05, "elongate": 0.03, "exclaim": 0.3}


def typo(token: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(token) - 1)
    kind = rng.choice(["swap", "delete", "double"])
    if kind == "swap":
        return token[:i] + token[i + 1] + token[i] + token[i + 2:]
    if kind == "delete":
        return token[:i] + token[i + 1:]
    return token[:i] + token[i] + token[i:]


def informalize(tokens: list[str], tags: list[str], rng: random.Random, probs: dict = DEFAULT_PROBS) -> Sentence:
    out_tokens = []
    for token in tokens:
        slang = FORMAL_TO_SLANG.get(token.lower())
        if slang and rng.random() < probs["slang"]:
            token = slang
        elif token.isalpha() and len(token) > 3 and rng.random() < probs["typo"]:
            token = typo(token, rng)
        elif token.isalpha() and rng.random() < probs["elongate"]:
            token = token + token[-1] * 2
        out_tokens.append(token)

    if rng.random() < probs["lower"]:
        out_tokens = [t.lower() for t in out_tokens]

    out_tags = list(tags)
    if out_tokens and out_tokens[-1] == "." and rng.random() < probs["exclaim"]:
        out_tokens = out_tokens[:-1] + ["!", "!", "!"]
        out_tags = out_tags[:-1] + ["O", "O", "O"]
    return out_tokens, out_tags


def augment_corpus(sentences: list[Sentence], copies: int = 1, seed: int = 42, probs: dict = DEFAULT_PROBS) -> list[Sentence]:
    rng = random.Random(seed)
    return [informalize(tokens, tags, rng, probs) for _ in range(copies) for tokens, tags in sentences]


# Set uji informal: semua kalimat diubah, dengan peluang lebih tinggi supaya bedanya jelas
TEST_PROBS = {"lower": 0.7, "slang": 0.8, "typo": 0.08, "elongate": 0.05, "exclaim": 0.5}


def main() -> None:
    parser = argparse.ArgumentParser(description="Buat versi informal dari file CoNLL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    sentences = augment_corpus(read_conll(args.input), seed=args.seed, probs=TEST_PROBS)
    write_conll(args.output, sentences)
    print(f"{len(sentences)} kalimat informal ditulis ke {args.output}")


if __name__ == "__main__":
    main()
