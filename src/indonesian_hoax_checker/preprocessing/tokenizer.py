# Tokenizer sederhana untuk teks mentah: memecah kata dan tanda baca, sambil mencatat posisi karakter.
# Angka dan kata bersambung tetap satu token, mis. "Rp13.000", "e-KTP", "13,5".

import re

TOKEN_PATTERN = re.compile(r"\w+(?:[.,'’\-]\w+)*|[^\w\s]")
SENTENCE_END = {".", "!", "?"}

# Satu token = (teks, posisi_awal, posisi_akhir) di teks asli
Token = tuple[str, int, int]


def tokenize(text: str) -> list[Token]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


# Pecah token menjadi kalimat di tanda . ! ? (deretan "!!!" tetap satu kalimat);
# kalimat panjang dipotong per max_tokens
def split_sentences(tokens: list[Token], max_tokens: int = 128) -> list[list[Token]]:
    sentences, current = [], []
    for i, token in enumerate(tokens):
        current.append(token)
        next_is_end = i + 1 < len(tokens) and tokens[i + 1][0] in SENTENCE_END
        if (token[0] in SENTENCE_END and not next_is_end) or len(current) >= max_tokens:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences
