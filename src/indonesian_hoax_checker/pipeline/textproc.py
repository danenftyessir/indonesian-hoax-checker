# Pengolahan teks input sebelum masuk tahap pipeline: pembersihan, pemecahan kalimat, penanda
# bantahan, dan pembuangan bingkai laporan. Dipakai orchestrator.py dan stages.py.

from __future__ import annotations

import re

MIN_SEGMENT_WORDS = 3
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Penanda bahwa teks MEMBAHAS atau MENYANGKAL sebuah klaim (artikel cek fakta, "itu hoaks").
# Lebih sempit dari DEBUNK_RE di classification/data_audit/mendeley_audit.py: di sini satu
# kecocokan sudah menahan verdict otomatis, jadi kata umum ("menegaskan", "klarifikasi") dibuang.
_DEBUNK = re.compile(
    r"\bhoaks?\b|\bhoax\b|tidak benar|bukan fakta|faktanya|\bbohong\b|disinformasi|cek fakta"
    r"|\bkeliru\b|\[salah\]|\bdibantah\b|\bmembantah\b|jangan percaya"
    r"|(?:tidak|belum|tak) terbukti|(?:tidak|belum) ada bukti",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Satu baris: karakter kontrol dibuang, deretan spasi/tab/baris baru dijadikan satu spasi.
    Spasi di awal/akhir dan baris baru mengubah keluaran model stance, jadi dibersihkan sekali di sini."""
    return " ".join(_CONTROL.sub(" ", text).split())


def split_segments(text: str) -> list[str]:
    """Kalimat-kalimat input (dipisah baris baru dan tanda . ! ?), tiap kalimat sudah dibersihkan.
    Potongan di bawah MIN_SEGMENT_WORDS kata dibuang. Dipakai sebagai kueri retrieval tambahan dan
    sebagai hypothesis stance, supaya klaim yang terselip di teks multi-kalimat tetap ditemukan."""
    out: list[str] = []
    for line in re.split(r"[\r\n]+", _CONTROL.sub(" ", text)):
        for part in _SENTENCE_END.split(line.strip()):
            part = " ".join(part.split())
            if len(part.split()) >= MIN_SEGMENT_WORDS and part not in out:
                out.append(part)
    return out


def has_debunk_marker(text: str) -> bool:
    return bool(_DEBUNK.search(text))


# Bingkai laporan di awal kalimat: "beredar kabar bahwa X", "katanya X", "ada info di WhatsApp kalau X".
# Model stance menilai "beredar kabar bahwa X" tidak sama dengan X, dan bingkainya menurunkan skor
# retrieval, jadi X (subjeknya utuh) dijadikan kueri tambahan. Sengaja TIDAK membuang atribusi
# "<tokoh> menyebut/mengatakan ...": siapa yang berkata adalah bagian dari banyak klaim Komdigi.
_MEDIUM = r"(?:kabar|info|informasi|pesan(?:\s+berantai)?|unggahan|narasi|isu|berita|postingan|klaim|video|foto)"
_PLACE = r"(?:\s+(?:di|pada|lewat|melalui)\s+(?:media\s+sosial|medsos|whatsapp|wa|facebook|tiktok|instagram|twitter|telegram|grup\s+\w+))?"
_REPORT_FRAME = re.compile(
    r"^(?:(?:selain\s+itu|lalu|kemudian|terus)\s*,?\s+)?"
    r"(?:"
    r"(?:(?:telah|sudah|sedang|kini|lagi)\s+)?(?:beredar|tersebar|viral|ramai|muncul|ada)\s+(?:sebuah\s+)?" + _MEDIUM + _PLACE +
    # wajib ada penghubung ("yang menyebut", "bahwa", "kalau", ":"), supaya "ada kabar baik untuk ..." tidak ikut terpotong
    r"(?:\s+yang\s+(?:menyebut(?:kan)?|mengeklaim|mengklaim|menyatakan|mengatakan|menarasikan|berisi)(?:\s+(?:bahwa|kalau))?"
    r"|\s+(?:bahwa|kalau|jika)|\s*:)\s*,?\s+"
    r"|(?:katanya|kabarnya|konon(?:nya)?|infonya|denger-denger|dengar-dengar)\s*,?\s+"
    r")",
    re.IGNORECASE,
)
MIN_FRAMELESS_WORDS = 4


def strip_report_frame(sentence: str) -> str | None:
    """Isi kalimat tanpa bingkai laporannya, atau None bila tidak ada bingkai / sisanya terlalu pendek."""
    rest = _REPORT_FRAME.sub("", sentence, count=1).strip()
    if rest == sentence.strip() or len(rest.split()) < MIN_FRAMELESS_WORDS:
        return None
    return rest
