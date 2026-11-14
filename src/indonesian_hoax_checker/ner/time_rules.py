# Penanda TIME berbasis aturan: pra-anotasi seed set (annotator tinggal mengoreksi) dan pengisi TIME
# yang terlewat model entitas saat prediksi (NERPredictor, terutama singkatan informal "mggu dpn", "skrg").
# Aturan mengikuti docs/annotation_guidelines.md: kata depan (pada, mulai, sejak) tidak masuk,
# durasi "selama tiga hari" masuk.

import re

MONTHS = {
    "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus", "september",
    "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "jun", "jul", "agu", "agt", "ags", "sep", "sept", "okt", "nov", "des",
}
DAYS = {"senin", "selasa", "rabu", "kamis", "jumat", "jum'at", "sabtu", "minggu", "ahad"}
UNITS = {"detik", "menit", "jam", "hari", "minggu", "pekan", "bulan", "tahun", "abad",
         "hr", "mggu", "mgg", "bln", "thn", "taun"}  # baris kedua: singkatan informal
RELATIVE = {"depan", "lalu", "ini", "kemarin", "mendatang", "sebelumnya", "dpn", "kmrn", "kemaren"}
SINGLE = {"besok", "bsk", "kemarin", "kmrn", "kemaren", "lusa", "semalam", "skrg", "sekarang"}
NUMBER_WORDS = {
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh",
    "sebelas", "beberapa", "puluhan", "ratusan",
}
ZONES = {"wib", "wita", "wit"}


def _is_number(tok):
    return bool(re.fullmatch(r"\d+([.,:]\d+)?", tok))


def _is_day_number(tok):
    return tok.isdigit() and 1 <= int(tok) <= 31


def _is_year(tok):
    return bool(re.fullmatch(r"(19|20)\d\d", tok))


def _is_amount(tok):
    return _is_number(tok) or tok in NUMBER_WORDS


# Tiap pola = daftar syarat per token; syarat diakhiri "?" berarti opsional
PATTERNS = [
    # Senin, 1 Desember 2026 / 1 Des 2026 / 1 Desember
    [lambda t: t in DAYS, "?", lambda t: t == ",", "?", _is_day_number, lambda t: t in MONTHS, _is_year, "?"],
    # Desember 2026
    [lambda t: t in MONTHS, _is_year],
    # 12/11/2026 atau 12-11-2026 (tokenizer memecah "/" jadi token sendiri)
    [_is_day_number, lambda t: t in {"/", "-"}, lambda t: t.isdigit(), lambda t: t in {"/", "-"}, _is_year],
    # tahun 2026, thn 2027
    [lambda t: t in {"tahun", "thn", "taun"}, _is_year],
    # akhir tahun ini, awal bulan depan, akhir Desember 2026
    [lambda t: t in {"akhir", "awal", "pertengahan"}, lambda t: t in UNITS or t in MONTHS,
     lambda t: t in RELATIVE or _is_year(t), "?"],
    # bulan depan, tahun lalu, minggu ini
    [lambda t: t in UNITS, lambda t: t in RELATIVE],
    # selama tiga hari, 2 minggu
    [lambda t: t == "selama", "?", _is_amount, lambda t: t in UNITS],
    # pukul 10.00 WIB
    [lambda t: t in {"pukul", "jam"}, _is_number, lambda t: t in ZONES, "?"],
    # hari ini, pagi ini, malam ini
    [lambda t: t in {"hari", "hr", "pagi", "siang", "sore", "malam"}, lambda t: t == "ini"],
    # besok, kemarin, lusa
    [lambda t: t in SINGLE],
    # nama hari saja
    [lambda t: t in DAYS],
]


# Panjang kecocokan pola mulai posisi i (0 = tidak cocok)
def _match(pattern, tokens, i):
    j, k = i, 0
    while k < len(pattern):
        cond = pattern[k]
        optional = k + 1 < len(pattern) and pattern[k + 1] == "?"
        if j < len(tokens) and cond(tokens[j]):
            j += 1
        elif not optional:
            return 0
        k += 2 if optional else 1
    return j - i


# Cari span TIME sebagai (token_awal, token_akhir_eksklusif); ambil kecocokan terpanjang
def time_spans(tokens: list[str]) -> list[tuple[int, int]]:
    lower = [t.lower() for t in tokens]
    spans, i = [], 0
    while i < len(lower):
        # Nomor peraturan/pasal bukan waktu, mis. "No 28 Tahun 2024"
        if i > 0 and lower[i - 1] in {"no", "nomor", "nomer", "pasal", "ayat", "uu", "pp"}:
            j = i
            while j < len(lower) and (lower[j].isdigit() or lower[j] in {"tahun", "/", "-", "."}):
                j += 1
            i = max(j, i + 1)
            continue
        length = max(_match(p, lower, i) for p in PATTERNS)
        if length:
            spans.append((i, i + length))
            i += length
        else:
            i += 1
    return spans


# Tambahkan label TIME hanya pada token yang masih O
def add_time_tags(tokens: list[str], tags: list[str]) -> list[str]:
    tags = list(tags)
    for start, end in time_spans(tokens):
        if all(t == "O" for t in tags[start:end]):
            tags[start] = "B-TIME"
            for j in range(start + 1, end):
                tags[j] = "I-TIME"
    return tags
