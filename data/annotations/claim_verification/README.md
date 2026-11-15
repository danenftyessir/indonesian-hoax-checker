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

## Draf LLM (14-15 Nov 2026): wajib diverifikasi manusia

Supaya anotasi tidak dimulai dari kolom kosong, isi berikut ditulis LLM (Claude) dan ditandai field `draft_source`. Statusnya **`annotated` dengan `annotators: ["llm:claude"]`**: artinya satu lintasan anotator (LLM) sudah selesai dan **belum ada manusia yang memverifikasi**. Tidak ada baris yang berstatus `verified`.

| File | Yang dikerjakan LLM | Jumlah |
|---|---|---|
| `stance_pairs.jsonl` | `hypothesis` untuk `supports` (parafrasa judul) | 100 |
| `stance_pairs.jsonl` | `hypothesis` untuk `refutes` (sangkalan berdiri sendiri; menggantikan draf "Faktanya, klaim tersebut tidak benar" yang tidak menyebut isi klaim. Kalimat aslinya tetap ada di `fact_sentence` pada `claims.jsonl`) | 100 |
| `stance_pairs.jsonl` | pasangan `unrelated` (dibuat otomatis oleh generator) dibaca satu per satu: semuanya klaim berbeda, tidak ada parafrasa atau sangkalan premis | 100 |
| `retrieval_queries.jsonl` | `query` untuk `paraphrase` | 100 |
| `dev/retrieval_queries.jsonl` | `query` untuk `paraphrase` | 44 |

Draf ditulis hanya dari judul dan `claim_sentence`, tanpa menambah fakta.

Kontrol bias negasi: draf `refutes` pertama hampir semuanya memakai "tidak/bukan" (91% vs 5% pada `supports`), sehingga label bisa ditebak dari kata negasi. 42 di antaranya ditulis ulang ke bentuk tanpa kata negasi ("... adalah hoaks", "... palsu", "... kabar bohong"); sekarang 49% `refutes` memuat kata negasi. Saat memverifikasi, pertahankan campuran ini dan tambahkan beberapa `supports` yang memuat negasi bila klaimnya memang bernada negatif.

Cara memverifikasi: baca tiap baris, perbaiki bila artinya bergeser dari klaim, lalu tambahkan kode anggota ke `annotators` dan ubah `status` menjadi `verified`. Untuk pasangan stance, set uji final butuh **dua** annotator manusia; `llm:claude` tidak dihitung sebagai salah satunya, jadi hapus dari `annotators` saat dua anggota sudah memeriksa (`draft_source` tetap mencatat asalnya).

Pengaman:

- `stance.dataset team` menolak baris dengan kurang dari dua annotator, dan notebook task 3 hanya memakai pasangan `verified`.
- Untuk analisis sebelum verifikasi selesai: `python -m indonesian_hoax_checker.stance.dataset team data/annotations/claim_verification/stance_pairs.jsonl data/processed/stance/team_draft.jsonl --include-unverified` menghasilkan pasangan ber-`source: team_draft`. Hasil evaluasi di file itu indikatif dan harus disebut "draf LLM belum diverifikasi" bila dikutip.
- Evaluasi retrieval tetap menilai kueri `draft`/`annotated`, tetapi mencatat jumlahnya di `n_draft_queries`; angka dengan `n_draft_queries > 0` bukan hasil anotasi final.
