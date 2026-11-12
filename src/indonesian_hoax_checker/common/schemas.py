# Format data yang dikirim antar komponen (NER, klasifikasi, retrieval, stance).
# Status: draf yang sudah ditinjau Danendra (11 Nov 2026); disahkan bertiga di rapat 13 Nov.
# Aturan lengkap: docs/decisions/0011-keputusan-teknis-tambahan.md. Catatan: 0011 §1 menulis
# `list[RetrievalResult]`, sedangkan di sini top-k ada di `RetrievalResult.candidates`; samakan sebelum
# pipeline/confidence.py ditulis.
#
# Hasil tinjauan (lihat docs/architecture.md "Kontrak data"):
# 1. Retrieval mengembalikan top-k, bukan satu. Ditambah `RetrievalCandidate` dan
#    `RetrievalResult.candidates`; `matched_claim`/`clarification`/`similarity_score`
#    tetap ada (= kandidat pertama) supaya kode lama dan laporan tidak berubah.
# 2. Nama field entitas tetap `label` (bukan `type`), sama dengan kolom label CoNLL.
# 3. `core_claim` tetap satu string: untuk input multi-kalimat NER memilih CLAIM
#    pertama (atau terpanjang) dan sisanya tetap tersedia di `entities`. Retrieval
#    memakai `core_claim` kalau tidak kosong, kalau kosong memakai teks input.

from typing import Literal, TypedDict

EntityLabel = Literal["PER", "ORG", "LOC", "TIME", "CLAIM"]
StanceLabel = Literal["supports", "refutes", "unrelated"]


# Satu entitas hasil NER; start/end = posisi karakter di teks input (end eksklusif)
class Entity(TypedDict):
    text: str
    label: EntityLabel
    start: int
    end: int


# core_claim = span CLAIM pertama di input, string kosong bila tidak ada
class NERResult(TypedDict):
    entities: list[Entity]
    core_claim: str  # "" bila tidak ada span CLAIM


# confidence = skor 0-1 dari classifier
class ClassificationResult(TypedDict):
    label: Literal["hoax", "valid"]
    confidence: float


# Satu kandidat klaim terbantah dari basis Komdigi
class RetrievalCandidate(TypedDict):
    claim_id: str
    claim: str
    clarification: str
    url: str
    published_at: str  # ISO 8601
    score: float  # BM25 (tidak dibatasi) atau cosine (-1..1); lihat `scorer`


# `candidates` terurut dari skor tertinggi, panjang <= top_k, kosong bila tidak ada yang
# lolos ambang. Tiga field pertama = kandidat pertama atau None (kompatibel dengan draf lama).
class RetrievalResult(TypedDict):
    matched_claim: str | None
    clarification: str | None
    similarity_score: float | None
    candidates: list[RetrievalCandidate]
    scorer: Literal["bm25", "embedding"]


# Stance klaim input (hypothesis) terhadap klaim terbantah (premise), lihat configs/labels.yaml
class StanceResult(TypedDict):
    label: StanceLabel
    confidence: float
    claim_id: str | None  # klaim terbantah yang dinilai; None bila tidak ada kandidat
