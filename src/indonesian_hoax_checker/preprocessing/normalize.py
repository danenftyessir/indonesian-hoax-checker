# Normalisasi teks informal untuk RQ1. Satu token masuk, satu token keluar,
# jadi label NER tidak perlu diselaraskan ulang.
#   - singkatan -> baku ("bsk" -> "besok"), huruf kapital awal dipertahankan
#   - huruf berulang dipendekkan ("bangeeet" -> "banget", "!!!" tetap token terpisah)
#   - kalimat yang hampir semua HURUF BESAR diubah ke huruf kecil

import re

from indonesian_hoax_checker.preprocessing.slang import SLANG_TO_FORMAL

ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")


def normalize_token(token: str) -> str:
    token = ELONGATED.sub(r"\1", token)
    formal = SLANG_TO_FORMAL.get(token.lower())
    if formal:
        return formal.capitalize() if token[:1].isupper() else formal
    return token


def is_shouting(tokens: list[str]) -> bool:
    words = [t for t in tokens if any(c.isalpha() for c in t)]
    upper = [t for t in words if len(t) > 1 and t.isupper()]
    return len(words) >= 3 and len(upper) / len(words) > 0.6


def normalize(tokens: list[str]) -> list[str]:
    if is_shouting(tokens):
        tokens = [t.lower() for t in tokens]
    return [normalize_token(t) for t in tokens]
