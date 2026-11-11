# Rencana Pengerjaan: Indonesian Hoax Checker

Disusun 2026-11-11 dari dokumen *Resolusi Diskusi dan Audit Dataset Indonesian Hoax Checker* (Drive tim), *Bahan Diskusi Tugas Besar*, laporan desain BEAM, slide Progress NLP, dan kondisi repo ini.

Status: **rencana, belum dikerjakan.** Semua keputusan di lembar keputusan masih berstatus "Usulan" sampai disahkan di rapat 13 Nov 2026.

Tanggal pasti: rapat **13 Nov 2026**, pembekuan test set **18 Nov 2026**. Jadwal presentasi minggu 14–15 belum dipastikan.

## Kondisi repo saat ini

Repo masih kerangka. Semua `.py`, `docs/*.md`, `configs/*.yaml` (kecuali `labels.yaml`), dan semua test kosong. Yang berisi hanya `common/schemas.py` (draf kontrak) dan `configs/labels.yaml`. `pyproject.toml` belum punya dependensi maupun entry point CLI.

## Fase 0 — Fondasi repo (sebelum rapat 13 Nov)

- Isi `pyproject.toml`: dependensi, entry point, dan `__main__.py` untuk `python -m indonesian_hoax_checker`.
- Isi `requirements/*.txt` dan `.env.example`. Kunci model `indobenchmark/indobert-base-p1` (NER dan klasifikasi).
- Tulis `docs/decisions/` satu file per keputusan dari lembar 9 keputusan, status tetap "Usulan".
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
- Hubungi penulis Cahyo dkk. (2025) soal izin memakai anotasi NER Komdigi.

**Bersama**
- Anotasi 100 kalimat kalibrasi, hitung Cohen's kappa per label. Revisi pedoman bila kappa CLAIM < 0,6.
- Anotasi seed set 400 kalimat (300 "Beredar ..." Komdigi + 100 berita benar), dua annotator per kalimat.
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
- Danendra meninjau `schemas.py`.

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
| Stance | Model NLI | **belum ditentukan** | IndoNLI train, lalu pasangan BRIN Title vs Fact + parafrasa |

Tidak di-fine-tune: BiLSTM-CRF dan TF-IDF + LR (dilatih dari nol), BM25 (hanya indeks), model embedding retrieval (siap pakai), baseline stance (ambang similarity).

## Hal yang perlu dicek / diputuskan

1. **`schemas.py`** tidak persis "tidak berubah bentuk". Retrieval mengembalikan `top_k` klaim sedangkan `RetrievalResult` hanya satu. `Entity` memakai `label` sedangkan resolusi menulis `Entity.type`. `core_claim: str` perlu dicek untuk artikel multi-kalimat (aturannya satu CLAIM per kalimat).
2. **Sumber parafrasa** untuk pasangan stance "supports" belum ditentukan (manual atau generasi model).
3. **Base model stance** belum dikunci (`configs/models/stance_nli.yaml` kosong): IndoBERT-p1 atau model NLI multibahasa.
4. **Compute** (Colab/Kaggle GPU) belum diputuskan. Tiga seed per model plus varian RQ berarti cukup banyak run GPU.
5. **Akses data manual:** Kaggle butuh login; HF Mafindo BRIN butuh persetujuan akses.
6. **Tanggal presentasi** belum pasti, jadi tenggat Fase 3–4 belum bisa diisi.
