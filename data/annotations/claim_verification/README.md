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

## Set dev (`dev/retrieval_queries.jsonl`)

Dibuat 14 Nov 2026 untuk memilih ambang tolak retrieval tanpa menyentuh set test: 44 judul asli, 44 parafrasa, 44 `no_match` (sisa `data/no_match_queries.jsonl` yang tidak dipakai set test). Tidak ada `gold_claim_ids` atau `source_id` yang sama dengan set test.

```
python -m indonesian_hoax_checker.data.claim_verification_templates data/processed/refuted_claims/claims.jsonl --split dev --out-dir data/annotations/claim_verification/dev --valid-claims data/no_match_queries.jsonl --exclude-no-match-from data/annotations/claim_verification/retrieval_queries.jsonl --n-per-label 44
```

`dev/stance_pairs.jsonl` yang ikut dihasilkan perintah ini dihapus: model stance dipakai zero-shot, jadi belum ada yang dituning di dev.

## Draf LLM (14 Nov 2026): wajib diperiksa manusia

Supaya anotasi tidak dimulai dari kolom kosong, draf berikut ditulis LLM (Claude) dan ditandai field `draft_source`. Semua barisnya **masih `status: draft` dan `annotators: []`**:

| File | Yang didraf | Jumlah |
|---|---|---|
| `stance_pairs.jsonl` | `hypothesis` untuk `supports` (parafrasa judul) | 100 |
| `stance_pairs.jsonl` | `hypothesis` untuk `refutes` (sangkalan berdiri sendiri; menggantikan draf "Faktanya, klaim tersebut tidak benar" yang tidak menyebut isi klaim. Kalimat aslinya tetap ada di `fact_sentence` pada `claims.jsonl`) | 100 |
| `retrieval_queries.jsonl` | `query` untuk `paraphrase` | 100 |
| `dev/retrieval_queries.jsonl` | `query` untuk `paraphrase` | 44 |

Draf ditulis hanya dari judul dan `claim_sentence`, tanpa menambah fakta. Cara memakainya: baca tiap baris, perbaiki bila artinya bergeser dari klaim, lalu isi `annotators` dan naikkan `status` (`annotated`, lalu `verified` oleh orang kedua). `draft_source` boleh dibiarkan sebagai catatan asal.

Pengaman: `stance.dataset team` menolak baris tanpa dua annotator, dan notebook task 3 hanya memakai pasangan `verified`. Evaluasi retrieval tetap menilai kueri draf, tetapi mencatat jumlahnya di `n_draft_queries`; angka dengan `n_draft_queries > 0` bukan hasil anotasi final dan harus disebut begitu di laporan.

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
