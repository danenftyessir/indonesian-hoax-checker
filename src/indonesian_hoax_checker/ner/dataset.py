# Baca/tulis file CoNLL (token<TAB>label, kalimat dipisah baris kosong) dan dedup kalimat.

from pathlib import Path

# Satu kalimat = (daftar token, daftar label)
Sentence = tuple[list[str], list[str]]


def read_conll(path: str | Path) -> list[Sentence]:
    sentences = []
    tokens, tags = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
            continue
        token, tag = line.rsplit("\t", 1)
        tokens.append(token)
        tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def write_conll(path: str | Path, sentences: list[Sentence]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = ["\n".join(f"{tok}\t{tag}" for tok, tag in zip(tokens, tags)) for tokens, tags in sentences]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# Buang kalimat dengan urutan token yang sama; yang pertama muncul dipertahankan
def dedup(sentences: list[Sentence], exclude: list[Sentence] = ()) -> list[Sentence]:
    seen = {tuple(tokens) for tokens, _ in exclude}
    kept = []
    for tokens, tags in sentences:
        key = tuple(tokens)
        if key not in seen:
            seen.add(key)
            kept.append((tokens, tags))
    return kept
