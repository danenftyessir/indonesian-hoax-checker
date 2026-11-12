# Pelabel CLAIM otomatis berbasis aturan, untuk label awal seed set (lalu dikoreksi manusia).
# CLAIM = frasa predikat-objek yang bisa dicek, maksimal satu per kalimat, tanpa subjek berlabel
# entitas dan tanpa TIME di ujung (docs/annotation_guidelines.md).
#
# Petunjuk dipakai berurutan (penanda kuat -> judul -> penanda lemah -> struktur); yang pertama berhasil dipakai:
#   1. Judul artikel, hanya untuk kalimat "Beredar ..." (Komdigi). Judul Komdigi ("[HOAKS] TNI Bakal
#      Berikan ...") adalah ringkasan klaimnya. Kata judul dicocokkan dengan kalimat (termasuk bentuk
#      berimbuhan: "berikan" ~ "memberikan"); bagian kalimat yang paling rapat kecocokannya menjadi CLAIM.
#      Judul berita biasa (Mafindo) berupa parafrase, jadi tidak dipakai.
#   2. Kata penanda klaim: "mengeklaim (bahwa) X", "diklaim X", "dengan narasi X", ...
#   3. Struktur kalimat (kalimat berita): kata kerja utama (awalan me-/ber-/di-/ter-, atau kata bantu
#      akan/telah/tidak/...) dianggap awal predikat; CLAIM dari predikat sampai koma pertama.
# Tanpa POS tagger, jadi hasilnya tetap perkiraan dan selalu diukur dengan annotation report.

import re

STRONG_MARKERS = {
    "mengeklaim", "mengklaim", "diklaim", "menyebut", "menyebutkan", "menyatakan", "mengatakan",
    "menyampaikan", "menginformasikan",
}
WEAK_MARKERS = {"berisi", "narasi", "menampilkan", "memperlihatkan"}
SKIP_AFTER_MARKER = {"bahwa", "adanya", "narasi", "sebagai", "terkait", "mengenai", "tentang", "soal",
                     ",", ":", "“", '"', "'"}
TITLES = {
    "presiden", "wakil", "menteri", "gubernur", "bupati", "wali", "kota", "ketua", "kapolri", "panglima",
    "jenderal", "direktur", "kepala", "sekretaris", "menko", "jaksa", "juru", "bicara", "mantan", "calon",
    "pj", "plt", "pelaksana", "tugas", "umum", "pimpinan", "capres", "cawapres", "caleg", "cagub",
    "cawagub", "nomor", "urut", "kapolda", "kapolres", "kapolsek", "wakapolri", "kombes", "akbp",
    "ketum", "waketum", "sekjen", "kabid", "kadiv", "humas", "dirut", "anggota", "politikus", "pengamat",
}
# Kata bantu predikat: "akan memberikan", "telah dihapus", "tidak benar", "resmi dibuka"
AUX = {
    "akan", "bakal", "telah", "sudah", "sedang", "tengah", "tidak", "tak", "belum", "resmi", "segera",
    "bisa", "dapat", "mampu", "harus", "wajib", "mulai", "kembali", "masih", "baru", "pernah", "ingin",
    "hendak", "siap", "berhasil", "turut",
}
# Kata berawalan me-/ber-/di-/ter- yang bukan predikat utama
NOT_PREDICATE = {
    "bernama", "berinisial", "berusia", "berasal", "bersama", "berupa", "berbagai", "beberapa", "berbeda",
    "bertajuk", "berjudul", "beredar", "berbasis", "berkaitan", "berdasarkan", "bersangkutan", "tersebut",
    "terkait", "terhadap", "termasuk", "terutama", "terakhir", "terbaru", "tertentu", "terdekat", "dimana",
    "diri", "dirinya", "dinas", "direktur", "dirut", "dirjen", "direksi", "digital", "dini", "dia", "dian",
    "menurut", "mengenai", "melalui", "meski", "meskipun", "merah", "media", "menteri", "menko", "menkes",
    "menkeu", "mendagri", "menag", "mensos", "mendikdasmen", "menpan", "metro", "merek", "medan", "mei",
    "terdapat", "berita", "beserta", "berikut", "berikutnya", "beragam", "terbesar", "tertinggi",
}
PREDICATE_PREFIX = re.compile(r"^(memper|diper|meng|meny|mem|men|me|ber|di|ter)[a-z]{3,}$")
# Ekor atribusi berita: ", kata X", ", ujar X" tidak masuk klaim
ATTRIBUTION = {"kata", "ujar", "ungkap", "jelas", "tutur", "sebut", "imbuh", "tegas", "terang", "katanya"}
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "dalam", "ini", "itu", "atau", "akan",
    "bakal", "telah", "sudah", "tidak", "bisa", "para", "oleh", "hoaks", "disinformasi", "misinformasi",
    "klarifikasi", "beredar", "video", "foto", "unggahan", "gambar", "narasi", "akun", "media", "sosial",
    "facebook", "tiktok", "instagram", "whatsapp", "twitter", "youtube", "threads", "soal", "jadi", "ada",
    "mengeklaim", "mengklaim", "diklaim", "kata", "secara", "agar", "bagi", "kepada", "sejak", "hingga",
}
TRAILING = {".", ",", ";", ":", "!", "?", '"', "”", "’", ")", "'"}
TIME_PREPOSITIONS = {"pada", "sejak", "mulai", "pukul"}
BAD_START = {"atau", "dan", "yang", "dari", "di", "ke", "pada", "untuk", "bersama", "serta", "dengan"}
MIN_TOKENS, MAX_TOKENS = 2, 45


# ---------- Pencocokan kata judul ----------

# Bentuk dasar kasar: lepas awalan/akhiran umum, termasuk peluluhan me- (menyebut -> sebut, memakai -> pakai)
def stems(word: str) -> set[str]:
    w = word.lower()
    out = {w}
    for prefix, restore in (("memper", ""), ("diper", ""), ("meng", "k"), ("meny", "s"), ("mem", "p"),
                            ("men", "t"), ("meng", ""), ("mem", ""), ("men", ""), ("me", ""), ("ber", ""),
                            ("di", ""), ("ter", ""), ("ke", ""), ("se", "")):
        if w.startswith(prefix) and len(w) - len(prefix) >= 4:
            out.add(restore + w[len(prefix):])
    for s in list(out):
        for suffix in ("kannya", "nya", "kan", "an", "lah", "i"):
            if s.endswith(suffix) and len(s) - len(suffix) >= 4:
                out.add(s[: -len(suffix)])
    return out


def _same_word(a: set[str], b: set[str]) -> bool:
    if a & b:
        return True
    return any(len(x) >= 5 and len(y) >= 5 and (x.startswith(y) or y.startswith(x)) for x in a for y in b)


def _is_content(word: str) -> bool:
    return any(c.isalnum() for c in word) and len(word) >= 3 and word.lower() not in STOPWORDS


# Bagian kalimat (mulai posisi lo) yang paling rapat cocok dengan kata-kata judul -> (awal, akhir) atau None
def align_title(tokens: list[str], title: str, lo: int = 0) -> tuple[int, int] | None:
    title_words = [stems(w) for w in re.findall(r"[\w'-]+", title or "") if _is_content(w)]
    if len(title_words) < 2:
        return None
    hits = [i for i in range(lo, len(tokens)) if _is_content(tokens[i])
            and any(_same_word(stems(tokens[i]), t) for t in title_words)]
    if len(hits) < 2:
        return None
    # Kelompokkan kecocokan yang berdekatan (jarak <= 6 token); ambil kelompok terbanyak, seri -> yang paling depan
    clusters, current = [], [hits[0]]
    for i in hits[1:]:
        if i - current[-1] <= 6:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)
    best = max(clusters, key=lambda c: (len(c), -c[0]))
    if len(best) < 2 or len(best) / len(title_words) < 0.3:
        return None
    return best[0], best[-1] + 1


# ---------- Bantuan umum ----------

def _skip_entity(tags, i):
    if i < len(tags) and tags[i].startswith("B-") and tags[i][2:] in {"PER", "ORG", "LOC"}:
        kind = tags[i][2:]
        i += 1
        while i < len(tags) and tags[i] == f"I-{kind}":
            i += 1
    return i


def _skip_parenthetical(tokens, i):
    if i < len(tokens) and tokens[i] == "(":
        j = i + 1
        while j < len(tokens) and tokens[j] != ")" and j - i <= 6:
            j += 1
        if j < len(tokens) and tokens[j] == ")":
            return j + 1
    return i


# Lewati subjek di posisi start: jabatan, entitas, kata berhuruf kapital, keterangan dalam kurung.
# Mengembalikan (posisi sesudah subjek, apakah subjek memuat entitas/jabatan).
def _skip_subject(tokens, tags, lower, start):
    j, has_subject = start, False
    while j < len(lower):
        if tags[j].startswith("B-") and tags[j][2:] in {"PER", "ORG", "LOC"}:
            j, has_subject = _skip_entity(tags, j), True
        elif lower[j] in TITLES or (lower[j].isdigit() and j > 0 and lower[j - 1] in {"urut", "nomor"}):
            j, has_subject = j + 1, True
        elif tokens[j] == "(" and _skip_parenthetical(tokens, j) > j:
            j = _skip_parenthetical(tokens, j)
        elif tokens[j][:1].isupper() and tags[j] == "O":
            j += 1
        elif (lower[j] in {"dan", "atau", "serta", ","} and j > start and j + 1 < len(lower)
              and (tokens[j + 1][:1].isupper() or tags[j + 1].startswith("B-"))):
            j += 1  # subjek majemuk: "DPR RI dan Polri", "Prabowo, Gibran"
        else:
            break
    return j, has_subject


def is_predicate(word: str) -> bool:
    w = word.lower()
    return w in AUX or (w.isalpha() and len(w) >= 5 and bool(PREDICATE_PREFIX.match(w)) and w not in NOT_PREDICATE)


# Rapikan batas: tanda baca di depan, kurung yang belum ditutup, tanda baca dan TIME di belakang
def _finish(tokens, tags, lower, start, end):
    while start < end and lower[start] in {",", ":", "“", '"', "'", "-"}:
        start += 1
    if start >= end or lower[start] in BAD_START:
        return None
    if end < len(tokens) and tokens[start:end].count("(") > tokens[start:end].count(")") and tokens[end] == ")":
        end += 1
    while end > start and tokens[end - 1] in TRAILING:
        if tokens[end - 1] == ")" and tokens[start:end - 1].count("(") > tokens[start:end - 1].count(")"):
            break
        end -= 1
    while end > start and tags[end - 1].endswith("TIME"):
        i = end - 1
        while i > start and tags[i].startswith("I-"):
            i -= 1
        end = i
        while end > start and (tokens[end - 1] in TRAILING or lower[end - 1] in TIME_PREPOSITIONS):
            end -= 1
    if not MIN_TOKENS <= end - start <= MAX_TOKENS:
        return None
    return start, end


# Akhir klausa utama: sebelum ", kata X" / ", ujar X", atau akhir kalimat
def _clause_end(lower, start):
    for i in range(start + 1, len(lower) - 1):
        if lower[i] == "," and lower[i + 1] in ATTRIBUTION:
            return i
    return len(lower)


# ---------- Tiga petunjuk ----------

def _marker_start(lower):
    marker = next((i for i, t in enumerate(lower) if t in STRONG_MARKERS), None)
    kind = "strong"
    if marker is None:
        marker = next((i for i, t in enumerate(lower) if t in WEAK_MARKERS), None)
        kind = "weak"
    if marker is None:
        bahwa = next((i for i, t in enumerate(lower) if t == "bahwa"), None)
        if bahwa is not None:
            return bahwa + 1, "strong"
        return None, None
    start = marker + 1
    while start < len(lower) and lower[start] in SKIP_AFTER_MARKER:
        start += 1
    return start, kind


def _from_title(tokens, tags, lower, title, marker_start):
    lo = marker_start if marker_start is not None else (1 if lower and lower[0] == "beredar" else 0)
    aligned = align_title(tokens, title, lo)
    if aligned is None:
        return None
    start, end = aligned
    # Kata predikat di antara penanda dan kecocokan pertama ikut masuk ("akan memberikan" ~ judul "Bakal Berikan")
    if marker_start is not None and marker_start < start <= marker_start + 8:
        start = marker_start
    # Subjek di awal (sering ikut cocok dengan judul) dilewati
    start = _skip_subject(tokens, tags, lower, start)[0]
    # Kata bantu predikat tepat sebelum awal ikut masuk
    while start > lo and lower[start - 1] in AUX:
        start -= 1
    if start >= end:
        return None
    if not (tokens[start][:1].islower() or tokens[start][:1].isdigit()):
        return None
    # Kata judul terakhir belum tentu akhir frasa ("... penyebab" -> "... penyebab bencana di Sumatra"):
    # perpanjang sampai batas klausa terdekat bila tidak terlalu jauh
    boundary = next((k for k in range(end, len(lower)) if lower[k] in {",", ".", ";", "!", "?"}), len(lower))
    if boundary - end <= 6:
        end = boundary
    return _finish(tokens, tags, lower, start, end)


def _from_marker(tokens, tags, lower, start, kind):
    j, has_subject = _skip_subject(tokens, tags, lower, start)
    if has_subject:
        start = j
    elif kind != "strong":
        return None
    if start >= len(lower):
        return None
    if has_subject and not (tokens[start][:1].islower() or tokens[start][:1].isdigit()):
        return None
    return _finish(tokens, tags, lower, start, _clause_end(lower, start))


def _from_predicate(tokens, tags, lower):
    if lower and lower[0] == "beredar":
        return None  # bingkai "Beredar unggahan ... yang ..." penuh kata kerja yang bukan klaim
    depth = 0
    for i, w in enumerate(lower):
        depth += (w == "(") - (w == ")")
        if depth or not is_predicate(w) or tags[i] != "O":
            continue
        if i > 0 and lower[i - 1] == "yang":
            continue  # anak kalimat "yang ...", bukan predikat utama
        if i > 12:
            return None  # subjek terlalu panjang, kemungkinan salah tebak
        # Berhenti di koma pertama (sesudah minimal 3 token): sisa kalimat berita biasanya keterangan
        # tempat/waktu/atribusi, bukan inti klaim
        end = next((k for k in range(i + 3, len(lower)) if lower[k] == ","), _clause_end(lower, i))
        return _finish(tokens, tags, lower, i, end)
    return None


# Cari span CLAIM sebagai (token_awal, token_akhir_eksklusif), atau None
def claim_span(tokens: list[str], tags: list[str], title: str | None = None) -> tuple[int, int] | None:
    lower = [t.lower() for t in tokens]
    if not lower:
        return None
    marker_start, kind = _marker_start(lower)
    if marker_start is not None and kind == "strong":
        span = _from_marker(tokens, tags, lower, marker_start, kind)
        if span:
            return span
    if title and lower[0] == "beredar":
        span = _from_title(tokens, tags, lower, title, marker_start)
        if span:
            return span
    if marker_start is not None and kind == "weak":
        span = _from_marker(tokens, tags, lower, marker_start, kind)
        if span:
            return span
    return _from_predicate(tokens, tags, lower)


# Tambahkan CLAIM ke label yang sudah ada; token di dalam span ditimpa menjadi CLAIM
def add_claim_tag(tokens: list[str], tags: list[str], title: str | None = None) -> list[str]:
    span = claim_span(tokens, tags, title)
    out = list(tags)
    if span is None:
        return out
    start, end = span
    for i in range(start, end):
        out[i] = ("B-" if i == start else "I-") + "CLAIM"
    # Entitas yang terpotong di batas akhir CLAIM dirapikan supaya IOB2 tetap valid
    if end < len(out) and out[end].startswith("I-"):
        out[end] = "B-" + out[end][2:]
    return out
