# Data Dictionary

Format file olahan di `data/processed/` dan anotasi di `data/annotations/`. Format ini menjadi acuan kode; bila perlu diubah, ubah di sini dulu. Semua file teks UTF-8.

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
| `tags` | list[str] | Dari kolom `tags`/`topics` |
| `cluster_id` | int | Klaster near-duplicate ([0008](decisions/0008-near-duplicate-dan-pembekuan-test-set.md)) |
| `split` | str | `train` / `dev` / `test` |

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
| `manifest.json` | Nama file, jumlah baris, SHA-256, tanggal pembekuan, versi Kaggle |

## Keluaran eksperimen

| Folder | Isi |
|---|---|
| `outputs/predictions/` | Prediksi per run (`{rq}_{model}_{seed}.jsonl`) |
| `outputs/metrics/` | Metrik per run (JSON) |
| `outputs/logs/` | Log pelatihan |
| `reports/tables/`, `reports/figures/` | Tabel dan gambar final untuk laporan |
