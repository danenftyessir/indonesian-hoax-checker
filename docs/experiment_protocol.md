# Experiment Protocol

Aturan bersama supaya semua eksperimen bisa dibandingkan. Config tiap RQ ada di `configs/experiments/rq*.yaml`.

## Aturan umum

- **Satu test set beku.** Semua ablasi dinilai pada data uji yang sama. Pembekuan: **18 Nov 2026**, hash di `data/splits/manifest.json`. Test set tidak boleh dipakai untuk tuning.
- **Split waktu untuk data Komdigi:** latih dan tuning pada klarifikasi lama, uji pada artikel sejak **1 Sep 2026**.
- **Near-duplicate satu split:** judul Komdigi dengan cosine TF-IDF char 3–5-gram ≥ 0,8 dikelompokkan; seluruh klaster masuk split yang sama ([0008](decisions/0008-near-duplicate-dan-pembekuan-test-set.md)).
- **Tiga seed per model** (mis. 13, 42, 2026); laporkan rata-rata dan simpangan baku.
- **Entitas prediksi**, bukan gold, untuk evaluasi akhir task 2 dan 3 ([0005](decisions/0005-entitas-untuk-task-2-dan-3.md)).
- **Kasus gagal tidak disembunyikan**; masuk error analysis.

## Task 1 — NER (RQ1)

| | |
|---|---|
| Baseline | BiLSTM-CRF |
| Model kuat | IndoBERT-base-p1 token classification |
| Data | NERGrit (cased) + seed set; NERP dedup hanya varian lowercase |
| Metrik | Entity-level F1 (seqeval, strict) keseluruhan dan per tipe; F1 CLAIM; formal vs informal |
| Ablasi | Dengan/tanpa normalisasi teks; dengan/tanpa augmentasi informal (huruf kecil, typo, singkatan, tanda baca berlebih); varian huruf kecil + NERP |
| Varian | `configs/experiments/rq1_ner.yaml`: base, norm, aug, norm_aug, lower_nerp, untuk BiLSTM-CRF dan IndoBERT; dijalankan `notebooks/02_ner/rq1_ner_colab.ipynb` |
| Uji informal | `nergrit_test_informal` (sintetis, seed 2026) sekarang; `seed_informal_test` (30 kalimat buatan tim) setelah dianotasi |
| Error analysis | `python -m indonesian_hoax_checker.ner.error_analysis <prediksi>`: salah tipe, salah batas, terlewat, salah deteksi per tipe |
| Target wajar | F1 NERGrit sekitar 70 (IndoNLU melaporkan 69,87 untuk IndoBERT-base); CLAIM dan TIME diperkirakan lebih rendah karena data kecil |

## Task 2 — Klasifikasi (RQ2)

| | |
|---|---|
| Baseline | TF-IDF + regresi logistik |
| Model kuat | IndoBERT-base-p1 sequence classification |
| Data | Mafindo-2024 train/test (stratified, setelah dedup judul) |
| Metrik | Macro-F1; precision dan recall kelas `hoax`; kalibrasi skor (ECE, reliability diagram) |
| Ablasi | Dengan/tanpa fitur entitas NER |
| Protokol Mendeley | Acak 5-fold, 5-fold terkelompok per klaster near-duplicate, leave-one-topic-out. Split acak dan terkelompok diulang (rata-rata dan sd); lihat `classification/baseline.py`. Label = mayoritas penilai |
| Uji wajib | F1 in-domain **dan** lintas-sumber (Mendeley 600, 300 klaim pendek, 200 klaim Komdigi netral) berdampingan |

Selisih besar antara in-domain dan lintas-sumber = bukti konfound sumber ([0001](decisions/0001-sumber-kelas-valid.md)); dilaporkan, bukan disembunyikan.

## Task 3a — Retrieval (RQ3)

| | |
|---|---|
| Baseline | BM25 |
| Model kuat | Sentence embedding siap pakai + cosine (kandidat: multilingual-e5); dipilih setelah uji 100 kueri pertama |
| Data | Basis 3.043 klaim Komdigi; 300 kueri (100 judul asli, 100 parafrasa manual, 100 tanpa padanan) |
| Metrik | success@10, MRR, Recall@k; recall terhadap usia klaim; ketepatan menolak kueri tanpa padanan |
| Ablasi | Dengan/tanpa penyempitan kandidat berbasis entitas |

Protokol mengikuti SemEval-2025 Task 7 (previously fact-checked claim retrieval).

## Task 3b — Stance (RQ4)

| | |
|---|---|
| Baseline | Ambang similarity saja |
| Model kuat | Model NLI, pra-latih IndoNLI lalu pasangan BRIN Title vs Fact |
| Data uji | IndoNLI val dan **test_expert**; 300 pasangan beranotasi tim |
| Metrik | Macro-F1 supports/refutes/unrelated; confusion supports vs refutes |
| Ablasi | Similarity saja vs similarity + stance |
| Kontrol | Bias negasi IndoNLI: laporkan hasil terpisah untuk hypothesis dengan dan tanpa "tidak/bukan/belum" |

## End-to-end (RQ5)

| Varian | Isi |
|---|---|
| A | Classifier saja |
| B | A + fitur entitas |
| C | B + bukti retrieval dan stance |
| D | C + ambang keyakinan (kasus di bawah ambang diserahkan ke manusia) |

Metrik: FN rate (hoaks lolos) dan FP rate per kelas dan per kategori (`tags` Komdigi), **beban tinjauan** (porsi kasus yang ditahan) vs akurasi sisanya. Hasil yang baik: lebih sedikit hoaks lolos pada beban tinjauan yang wajar.

## Error analysis

Kategori minimal (juga dipakai untuk `data/demo_cases.txt`):

1. Klaim hoaks yang ada di basis Komdigi
2. Klaim sama dengan kata berbeda
3. Klaim yang menyangkal hoaks
4. Klaim tanpa padanan di basis
5. Teks informal (singkatan, kapital berantakan, typo)
6. Perubahan kecil: tanggal/angka berbeda, negasi
7. Berita benar yang mirip hoaks (satir, kutipan politik)
8. Input ekstrem: sangat pendek, beberapa paragraf, kosong, campur bahasa

## Pencatatan

- Prediksi: `outputs/predictions/{rq}_{model}_{seed}.jsonl`
- Metrik: `outputs/metrics/{rq}_{model}_{seed}.json`
- Tabel dan gambar final: `reports/tables/`, `reports/figures/`
- Catat versi model, commit Git, dan versi dataset di setiap file metrik.
