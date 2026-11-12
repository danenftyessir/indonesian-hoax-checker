# Rencana Pengerjaan: Indonesian Hoax Checker

Disusun 2026-11-11 dari dokumen *Resolusi Diskusi dan Audit Dataset Indonesian Hoax Checker* (Drive tim), *Bahan Diskusi Tugas Besar*, laporan desain BEAM, slide Progress NLP, dan kondisi repo ini.

Semua keputusan di [docs/decisions/](decisions/README.md) berstatus **Final**; rencana ini dijalankan tanpa rapat tambahan.

Tanggal pasti: pembekuan test set **18 Nov 2026**. Jadwal presentasi minggu 14–15 belum dipastikan.

## Kondisi repo saat ini

Fase 0 selesai (dependensi, entry point CLI, label, keputusan, docs). Task NER sudah punya persiapan data dan kode baseline (`ner/`); komponen lain masih kosong.

## Fase 0 — Fondasi repo (selesai)

- Isi `pyproject.toml`: dependensi, entry point, dan `__main__.py` untuk `python -m indonesian_hoax_checker`.
- Isi `requirements/*.txt` dan `.env.example`. Kunci model `indobenchmark/indobert-base-p1` (NER dan klasifikasi).
- Tulis `docs/decisions/` satu file per keputusan dari lembar 9 keputusan, status Final.
- Perbaiki `configs/labels.yaml`. Arah stance saat ini terbalik terhadap keputusan baru: premise = klaim yang dibantah, hypothesis = klaim input, `entailment` → supports. Tulis di `docs/label_mapping.md`.

## Fase 1 — Data (jalur kritis, sampai pembekuan 18 Nov)

**Dzaky**
- Unduh Mafindo-2024 dan Mendeley. Hitung duplikat judul dan sebaran label. Catat di `docs/dataset_documentation.md`.

**Danendra**
- Unduh Kaggle Komdigi, catat nomor versi. Ekstrak klaim dari judul `[HOAKS]` dan kalimat "Beredar ...".
- Kelompokkan near-duplicate: TF-IDF char 3–5-gram, ambang 0,8. Periksa manual 50 pasangan.
- Minta akses `nlp-brin-id/fakenews-mafindo`; cek dataverse BRIN dari jaringan kampus.

**Jonathan**
- Siapkan NERGrit dan NERP. Pemetaan: NERGrit PERSON→PER, ORGANISATION→ORG, PLACE→LOC; NERP PPL→PER, PLC→LOC, IND/EVT/FNB→O. Buang 323 duplikat NERP. NERP hanya untuk eksperimen lowercase (RQ1).
- Tulis pedoman anotasi CLAIM (span) dan TIME, satu halaman dengan 10 contoh (termasuk negasi dan klaim ganda).

**Bersama**
- Seed set ±1.500 kalimat CLAIM/TIME: label otomatis (model + aturan), lalu Jonathan mengoreksi dev 50 + test 100 di Label Studio ([0007](decisions/0007-seed-set-dan-annotator.md)).
- Anotasi 300 pasangan stance (100 supports, 100 refutes, 100 unrelated), diverifikasi dua orang.
- Buat `data/short_claims.jsonl` (300 klaim pendek: 150 judul Komdigi tanpa prefiks, 150 kalimat pertama berita Tempo/Kompas). Pemilik Dzaky, diverifikasi Danendra.
- Bekukan split dan simpan hash di `data/splits/`. Test set = artikel Komdigi sejak 1 Sep 2026. Unduh ulang Kaggle tepat sebelum pembekuan.

Modul terkait: `data/`, `preprocessing/`, `evaluation/splits.py`.

## Fase 2 — Baseline mandiri (paralel dengan Fase 1)

| Task | Baseline | Catatan |
|---|---|---|
| NER (Jonathan) | BiLSTM-CRF dan IndoBERT-p1 pada NERGrit cased | Agar task 2 dan 3 segera punya entitas prediksi |
| Klasifikasi (Dzaky) | TF-IDF + LR pada Mafindo-2024 | Laporkan F1 in-domain dan lintas-sumber (Mendeley 600, 300 klaim pendek, 200 klaim Komdigi netral) berdampingan |
| Retrieval (Danendra) | BM25 pada 3.043 klaim Komdigi | success@10, MRR, recall terhadap usia klaim |
| Stance (Danendra) | Ambang similarity saja | |

## Fase 3 — Model kuat dan kerangka sistem

- **NER:** IndoBERT-p1 dengan label CLAIM dan TIME. RQ1: normalisasi dan augmentasi teks informal.
- **Klasifikasi:** IndoBERT-p1 dengan kalibrasi skor. RQ2: dengan dan tanpa fitur entitas.
- **Retrieval:** BM25 vs embedding siap pakai (mis. multilingual-e5). Danendra memilih setelah uji cepat pada 100 kueri pertama. RQ3: narrowing berbasis entitas.
- **Stance:** pra-latih di IndoNLI train, lanjut ke pasangan BRIN (Title vs Fact) dan parafrasanya. Evaluasi wajib pada `test_expert` dan 300 pasangan buatan sendiri (RQ4).
- **CLI dan pipeline (Dzaky):** stub per tahap, `--verbose`, mode per task, mode interaktif. `pipeline/` menjalankan klasifikasi dan retrieval paralel, digabung di `confidence.py`.

## Fase 4 — Integrasi dan evaluasi akhir

- Varian pipeline A–D (RQ5), tiga seed per model, ablasi RQ1–RQ5.
- Error analysis, pilih ambang keyakinan, siapkan `data/demo_cases.txt` (8 kategori kasus).
- Isi `docs/architecture.md`, README, laporan. Sesuaikan laporan dan diagram BEAM dengan perubahan klasifikasi dan retrieval yang kini paralel.
- Simpan model di `artifacts/`; uji waktu load di laptop demo.

## Model yang di-fine-tune

| Task | Model | Base | Data |
|---|---|---|---|
| NER | IndoBERT token classification | `indobenchmark/indobert-base-p1` | NERGrit train + seed set CLAIM/TIME (varian RQ1: + NERP dedup lowercase, augmentasi) |
| Klasifikasi | IndoBERT sequence classifier | `indobenchmark/indobert-base-p1` | Mafindo-2024 train (varian RQ2: + fitur entitas) |
| Stance | Model NLI | `indobenchmark/indobert-base-p1` | IndoNLI train, lalu pasangan BRIN Title vs Fact + parafrasa |

Tidak di-fine-tune: BiLSTM-CRF dan TF-IDF + LR (dilatih dari nol), BM25 (hanya indeks), model embedding retrieval (siap pakai), baseline stance (ambang similarity).

## Status pengerjaan bagian Danendra (11 Nov 2026)

Kode siap dan teruji (`tests/unit/test_retrieval.py`, `tests/unit/test_stance.py`). Basis Komdigi (Kaggle v396, 11 Nov 2026) sudah diproses: 3.048 klaim, 63 pasangan near-duplicate, split train/dev/test = 2.398/299/351.

| Bagian | Modul | Status |
|---|---|---|
| Ekstraksi klaim Komdigi (`[HOAKS]`, "Beredar ...", "Faktanya ...") | `retrieval/corpus.py` | selesai, sudah dijalankan pada data asli |
| Near-duplicate TF-IDF char 3–5-gram ≥ 0,8 + split waktu + manifest hash + sampel 50 pasangan | `evaluation/splits.py` | selesai, sudah dijalankan; `dev_start` 1 Jul 2026 masih usulan, 70 pasangan `near_duplicate_review.csv` belum diperiksa manual |
| BM25, embedding e5, filter entitas, metrik RQ3, skrip evaluasi | `retrieval/` | selesai; model embedding dipilih setelah uji 100 kueri |
| Stance: konversi IndoNLI/BRIN/tim, baseline ambang similarity, NLI wrapper, metrik RQ4, skrip latih | `stance/` | selesai; latih butuh GPU |
| Template 300 pasangan stance + 300 kueri | `data/claim_verification_templates.py` | selesai; isi manual setelah split beku |
| Tinjauan `schemas.py` | `common/schemas.py` | selesai; **bentuk top-k berbeda dengan 0011 §1** (lihat di bawah), perlu diselaraskan sebelum 13 Nov |
| Akses BRIN gated, verifikasi `short_claims.jsonl` | — | manual, belum (unduh Kaggle sudah) |
| Sumber 100 kueri `no_match` | `data/no_match_queries.jsonl` (Dzaky, 144 kandidat) | diterima 12 Nov; sudah dicek BM25 top-10, belum dimasukkan ke `retrieval_queries.jsonl` |

Catatan kontrak retrieval: 0011 §1 menetapkan tahap retrieval mengembalikan `list[RetrievalResult]`. Implementasi saat ini mengembalikan satu `RetrievalResult` berisi `candidates` (top-k) dengan tiga field lama berisi kandidat pertama. Isinya setara; yang perlu diputuskan adalah bentuk yang dipakai `pipeline/confidence.py`.

## Keputusan yang sudah ditetapkan

Hal yang sebelumnya terbuka sudah ditetapkan di [0011](decisions/0011-keputusan-teknis-tambahan.md): kontrak `schemas.py`, parafrasa manual, base model stance IndoBERT-p1, fine-tuning IndoBERT di Colab, dan tanpa permintaan izin ke penulis mana pun.

Yang tetap dikerjakan manual oleh pemilik task:

1. **Akses data:** login Kaggle untuk basis Komdigi; klik persetujuan gated HF untuk dataset BRIN.
2. **Tanggal presentasi** belum diumumkan, jadi tenggat Fase 3–4 diisi setelah jadwal keluar.
