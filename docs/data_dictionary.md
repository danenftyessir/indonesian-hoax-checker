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

File `*_test_informal.conll` adalah versi informal sintetis dari test (huruf kecil, singkatan, typo, "!!!"), dibuat `ner.prepare` dengan seed tetap untuk RQ1.

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

## Anotasi NER seed — `data/annotations/ner/` (dilacak Git)

Dibuat dan diolah dengan `python -m indonesian_hoax_checker.ner.annotation` (lihat [annotation_guidelines.md](annotation_guidelines.md)).

**`seed_sentences.jsonl`** — kalimat yang akan dianotasi, satu baris per kalimat:

| Field | Tipe | Keterangan |
|---|---|---|
| `id` | str | ID unik, mis. `komdigi-1234-1` |
| `text` | str | Satu kalimat |
| `source` | str | `komdigi` atau `mafindo` |
| `origin_id` | str | ID artikel asal |
| `published_at` | str \| null | Tanggal artikel (ISO 8601); menentukan apakah kalimat wajib masuk test |
| `batch` | str | `calibration` (100 kalimat untuk kappa), `main`, atau `informal_test` |

**`informal_sentences.jsonl`** — 30 kalimat uji informal buatan tim, format sama dengan `seed_sentences.jsonl` dengan `source: informal`. Tidak masuk train/dev/test seed; menjadi `seed_informal_test.conll`.

**`label_studio_tasks.json`** — file impor Label Studio, berisi pra-anotasi (TIME dari aturan, PER/ORG/LOC dari model baseline).

**`export_*.json`** — ekspor JSON dari Label Studio (semua annotator dalam satu file).

**`seed_adjudicated.jsonl`** — versi final per kalimat: semua field `seed_sentences.jsonl` ditambah `tokens` (list[str]) dan `tags` (list[str], IOB2). Dari file ini dibuat `data/processed/ner/seed_{train,dev,test}.conll`.

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
