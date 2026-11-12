# Anotasi task 3: pasangan stance dan kueri retrieval

Pemilik: Danendra. Target (keputusan 0007): **300 pasangan stance** (100 per label) diverifikasi dua orang, dan **300 kueri retrieval** (100 judul asli, 100 parafrasa, 100 tanpa padanan). Pedoman label: `docs/annotation_guidelines.md` Bagian B.

File di folder ini dilacak Git. Jangan menjalankan generator dengan `--overwrite` setelah anotasi dimulai.

## Cara membuat template

```
python -m indonesian_hoax_checker.data.claim_verification_templates data/processed/refuted_claims/claims.jsonl --valid-claims data/no_match_queries.jsonl --prefer-short-no-match
```

Kueri `no_match` saat ini (12 Nov 2026) = 100 terpendek dari 144 kalimat pembuka berita Mafindo kiriman Dzaky (`data/no_match_queries.jsonl`, `source_id` = `sc150..sc299`). `--prefer-short-no-match` dipakai karena kalimat itu jauh lebih panjang dari judul Komdigi (median 17 vs 8 kata) dan skor BM25 naik dengan panjang kueri. Pemeriksaan BM25 top-1: maksimum 29,8, sedangkan judul asli terhadap dirinya sendiri minimum 16,6; lima skor tertinggi sudah dibaca dan hanya kebetulan leksikal (nama tokoh/lembaga), bukan padanan.

Entitas kueri dan basis klaim (ablasi filter RQ3) diisi oleh skrip, dari checkpoint NER tim (keputusan 0005):

```
python -m indonesian_hoax_checker.retrieval.annotate_entities --checkpoint artifacts/checkpoints/ner/ner_indobert_seed42 --queries data/annotations/claim_verification/retrieval_queries.jsonl
```

Semua premise dan gold diambil dari split `test` (artikel Komdigi sejak 1 Sep 2026), satu klaim per klaster near-duplicate. Untuk set tuning jalankan lagi dengan `--split dev --out-dir data/annotations/claim_verification/dev`.

## `stance_pairs.jsonl`

| Field | Diisi oleh | Keterangan |
|---|---|---|
| `pair_id` | otomatis | `team-sup-###`, `team-ref-###`, `team-unr-###` |
| `premise`, `premise_claim_id`, `premise_url` | otomatis | Klaim terbantah (judul Komdigi tanpa prefiks) |
| `hypothesis` | **manusia** | supports: tulis parafrasa. refutes: draf dari kalimat "Faktanya", ubah menjadi kalimat yang menyangkal klaim. unrelated: otomatis dari klaim lain, cukup diperiksa |
| `label` | manusia | Label target sudah terisi; ganti bila menurut annotator labelnya lain |
| `annotators` | manusia | Dua kode anggota, mis. `["danendra", "jonathan"]` |
| `status` | manusia | `draft` → `annotated` (satu orang) → `verified` (orang kedua setuju) |

Aturan singkat: nilai arah hypothesis terhadap premise, bukan kebenarannya. Ubah angka/tanggal/entitas yang mengubah isi klaim → `unrelated`, kecuali jelas menyangkal.

Setelah semua `verified`, konversi ke format latih/uji:

```
python -m indonesian_hoax_checker.stance.dataset team data/annotations/claim_verification/stance_pairs.jsonl data/processed/stance/team_test.jsonl
```

## `retrieval_queries.jsonl`

| Field | Diisi oleh | Keterangan |
|---|---|---|
| `query_id`, `type` | otomatis | `original` / `paraphrase` / `no_match` |
| `query` | otomatis untuk original dan no_match (bila `--valid-claims` diberikan); **manusia** untuk paraphrase | Parafrasa boleh bergaya pesan berantai (singkatan, huruf kapital berantakan) |
| `gold_claim_ids` | otomatis | Seluruh anggota klaster near-duplicate; kosong untuk no_match |
| `entities` | opsional | Daftar Entity hasil NER untuk ablasi filter entitas (RQ3); diisi skrip, bukan manual |

Kueri `no_match` wajib dicek: jalankan BM25 top-10 dan pastikan memang tidak ada padanan di basis.

## Cara mengisi tanpa alat khusus

File JSONL satu baris per item; edit dengan VS Code (ekstensi "JSON Lines" membantu) atau konversi ke CSV sementara dengan pandas. Jangan ubah `pair_id`/`query_id`.
