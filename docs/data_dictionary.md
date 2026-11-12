# Data Dictionary

Format file olahan di `data/processed/` dan anotasi di `data/annotations/`. Nama kolom ini **usulan**; ubah di sini dulu sebelum mengubah kode. Semua file teks UTF-8.

## Basis klaim terbantah — `data/processed/refuted_claims/claims.jsonl`

Satu baris per artikel Komdigi.

| Field | Tipe | Keterangan |
|---|---|---|
| `claim_id` | str | `id` asli Kaggle |
| `claim` | str | Judul tanpa prefiks `[HOAKS]` |
| `claim_sentence` | str \| null | Kalimat "Beredar ... yang mengeklaim ..." pertama |
| `clarification` | str | `body_text` bersih (tanpa HTML) |
| `url` | str | Tautan artikel Komdigi |
| `published_at` | str | Tanggal ISO 8601 |
| `tags` | list[str] | Dari kolom `tags`/`topics`, huruf kecil |
| `title_tags` | list[str] | Prefiks judul yang dibuang, mis. `["HOAKS"]`, `["DISINFORMASI"]` |
| `fact_sentence` | str \| null | Kalimat bantahan pertama ("Faktanya, ..."); draf hypothesis `refutes` |
| `cluster_id` | int | Klaster near-duplicate ([0008](decisions/0008-near-duplicate-dan-pembekuan-test-set.md)) |
| `split` | str | `train` / `dev` / `test`; dev = klaster bertanggal terbaru 1 Jul–31 Agu 2026 (usulan) |

Dibuat oleh `retrieval/corpus.py` (tanpa `cluster_id`/`split`, ke `data/interim/refuted_claims/claims_raw.jsonl`) lalu `evaluation/splits.py`. File pendamping di `data/interim/refuted_claims/`: `near_duplicate_pairs.csv` (semua pasangan ≥ ambang) dan `near_duplicate_review.csv` (50 pasangan ≥ ambang + 20 di bawah ambang, kolom `is_duplicate` diisi manual).

## NER — `data/processed/ner/{nergrit,nerp_lower,seed}_{train,valid,test}.conll`

Format CoNLL: satu token per baris, `token<TAB>label`, kalimat dipisah baris kosong. Label IOB2 dengan tipe PER, ORG, LOC, TIME, CLAIM ([label_mapping.md](label_mapping.md)).

```
Pemerintah	B-ORG
menghentikan	B-CLAIM
...
bulan	B-TIME
depan	I-TIME
```

## Klasifikasi — `data/processed/classifier/*.jsonl`

| Field | Tipe | Keterangan |
|---|---|---|
| `id` | str | ID unik (prefiks sumber, mis. `mafindo-123`) |
| `text` | str | Judul + isi, atau satu kalimat untuk klaim pendek |
| `label` | str | `hoax` / `valid` |
| `source` | str | `mafindo2024`, `mendeley`, `short_claims`, `komdigi_neutral` |
| `split` | str | `train` / `dev` / `test` |

## Klaim pendek — `data/short_claims.jsonl`

| Field | Tipe | Keterangan |
|---|---|---|
| `id` | str | ID unik |
| `text` | str | Satu kalimat klaim |
| `label` | str | `hoax` (judul Komdigi) / `valid` (kalimat pertama berita Tempo/Kompas) |
| `origin_id` | str | ID artikel asal |

## Stance — `data/processed/stance/*.jsonl` dan `data/annotations/claim_verification/`

| Field | Tipe | Keterangan |
|---|---|---|
| `pair_id` | str | ID unik |
| `premise` | str | Klaim yang sudah dibantah |
| `hypothesis` | str | Klaim input |
| `label` | str | `supports` / `refutes` / `unrelated` |
| `source` | str | `indonli`, `brin_fact`, `brin_paraphrase`, `team` |
| `annotators` | list[str] | Hanya untuk anotasi tim |
| `has_negation` | bool | Hypothesis memuat tidak/bukan/belum/tak/nggak/gak/enggak/tanpa; untuk kontrol bias negasi |
| `hoax_type` | str \| null | Hanya `brin_fact`: kolom `Classification` BRIN |

Template anotasi tim (kolom tambahan `premise_claim_id`, `status`, `instruction`): lihat `data/annotations/claim_verification/README.md`.

## Kueri retrieval — `data/annotations/claim_verification/retrieval_queries.jsonl`

| Field | Tipe | Keterangan |
|---|---|---|
| `query_id` | str | ID unik |
| `query` | str | Teks kueri |
| `type` | str | `original`, `paraphrase`, `no_match` |
| `gold_claim_ids` | list[str] | Kosong untuk `no_match` |

## Anotasi NER seed — `data/annotations/ner/`

Hasil ekspor alat anotasi, satu file per annotator, lalu file gabungan `seed_adjudicated.conll` setelah perbedaan diselesaikan. Format sama dengan file CoNLL NER di atas.

## Split — `data/splits/`

| File | Isi |
|---|---|
| `manifest.json` | Nama file, jumlah baris, SHA-256, tanggal pembekuan, versi Kaggle, statistik klaster dan split |
| `refuted_claims_split.csv` | `claim_id, cluster_id, split, published_at` untuk seluruh basis klaim (dilacak Git, supaya split bisa direkonstruksi tanpa data mentah) |

## Keluaran eksperimen

| Folder | Isi |
|---|---|
| `outputs/predictions/` | Prediksi per run (`{rq}_{model}_{seed}.jsonl`) |
| `outputs/metrics/` | Metrik per run (JSON) |
| `outputs/logs/` | Log pelatihan |
| `reports/tables/`, `reports/figures/` | Tabel dan gambar final untuk laporan |
