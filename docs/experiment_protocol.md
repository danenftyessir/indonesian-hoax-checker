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
| Varian | `configs/experiments/rq1_ner.yaml`, dua layer tahap 2 (IndoBERT-p1, dari model NERGrit): CLAIM `base`, `norm`, `aug` (+1 salinan informal), `aug3` (+3); entitas `base`, `norm`, `aug`; 3 seed; dijalankan `notebooks/02_ner/rq1_ner_colab.ipynb` |
| Uji informal | `seed_informal_test` (30 kalimat tulisan tangan, dikoreksi), `seed_sim_informal_test` (seed test versi medsos sintetis: huruf kecil, singkatan, typo, "!!!"), dan seed test formal sebagai pembanding. Model CLAIM pembanding: F1 formal 0,630, informal 0,101 |
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

## Hasil awal: dampak NER ke retrieval (2026-11-13)

Skrip: `python -m indonesian_hoax_checker.ner.query_impact`. Retrieval BM25 (Danendra) tanpa filter entitas, indeks hanya judul klaim. NER = IndoBERT entitas + CLAIM seed 42 ([0012](decisions/0012-layer-entitas-dan-claim-terpisah.md)).

Kueri = kalimat klaim artikel Komdigi split test (351 kalimat, tidak dipakai melatih NER); emas = klaim asal + klaster near-duplicate-nya.

| Kueri | success@1 | success@5 | success@10 | MRR |
|---|---|---|---|---|
| Teks penuh | **0,841** | 0,909 | 0,943 | 0,872 |
| `core_claim` saja | 0,735 | 0,838 | 0,877 | 0,780 |
| `core_claim` + entitas PER/ORG/LOC (aturan pipeline) | 0,786 | 0,900 | 0,932 | 0,835 |

Pada 100 kueri `original` (judul apa adanya; mudah karena sama dengan judul yang diindeks): teks penuh 1,000, `core_claim` + entitas 0,970, `core_claim` saja 0,890.

Kesimpulan awal: untuk BM25, kueri dari NER **tidak lebih baik** daripada teks penuh. `core_claim` membuang subjek (nama tokoh dan lembaga), padahal itu kata kunci pencocokan judul; menambahkan entitas memulihkan sebagian besar selisihnya. Kata pengantar ("Beredar ... di media sosial") ternyata tidak mengganggu BM25 karena bobot kata itu rendah.

Batasan: kueri berupa kalimat Komdigi yang baku, bukan pesan berantai yang berisik; kueri parafrasa (`retrieval_queries.jsonl`) masih draf kosong; hanya BM25 (embedding belum diuji). Ulangi setelah parafrasa terisi dan dengan scorer embedding.

### Cara memakai entitas di retrieval (kueri = teks penuh)

Skrip: `python -m indonesian_hoax_checker.ner.entity_use`. Kueri dan emas sama seperti di atas; entitas klaim basis = NER atas judul klaim; `beta` boost dipilih di split dev (299 kueri, beta 0,1), hasil di test (351 kueri).

| Cara | success@1 | success@5 | success@10 | MRR |
|---|---|---|---|---|
| Polos (teks penuh) | 0,841 | 0,909 | 0,943 | 0,872 |
| Filter keras entitas | 0,746 | 0,818 | 0,841 | 0,776 |
| Boost skor bila berbagi entitas | 0,838 | 0,920 | 0,943 | 0,872 |
| Teks penuh + entitas di akhir kueri | 0,778 | 0,920 | 0,940 | 0,836 |

Diagnostik filter: dari 291 kueri test yang punya entitas, klaim emas tetap lolos filter hanya pada 243 (83%). Pada 48 kueri entitas kueri tidak beririsan dengan entitas judul emas (judul pendek sering tidak memuat nama yang sama, atau NER memberi bentuk berbeda), sehingga klaim emas tersingkir. 60 kueri test tidak punya entitas sama sekali.

Kesimpulan awal untuk BM25 + kueri kalimat Komdigi: entitas NER **tidak menaikkan** retrieval. Filter keras menurunkan success@1 sembilan poin, boost tidak berpengaruh berarti, dan menambah entitas ke kueri menurunkan success@1. Pembanding polos sudah kuat (0,841), jadi ruang perbaikannya kecil. Yang masih perlu diuji: scorer embedding, kueri parafrasa dan pesan berisik, serta filter dengan pencocokan entitas yang lebih longgar (alias, singkatan).
