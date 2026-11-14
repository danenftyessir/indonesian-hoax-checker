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
| Data | Basis 3.048 klaim Komdigi (versi 396); 300 kueri (100 judul asli, 100 parafrasa manual, 100 tanpa padanan) |
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

## Hasil RQ1: normalisasi dan augmentasi (2026-11-13)

Notebook `rq1_ner_colab.ipynb`, 7 varian × 3 seed (13, 42, 2026), IndoBERT-p1 dilanjutkan dari model NERGrit; entity-level F1 strict, rata-rata ± simpangan baku. Hasil mentah ada di Drive `hasil_training/rq1_ner_20261113-*`.

Layer CLAIM:

| Varian | Seed test formal | Informal tulisan tangan (30) | Informal sintetis (100) |
|---|---|---|---|
| base | 0,630 ± 0,024 | 0,101 ± 0,021 | 0,348 ± 0,015 |
| norm | 0,629 ± 0,017 | 0,089 ± 0,005 | 0,351 ± 0,018 |
| aug (+1 salinan) | 0,647 ± 0,029 | 0,091 ± 0,003 | 0,624 ± 0,029 |
| aug3 (+3 salinan) | **0,654 ± 0,034** | 0,106 ± 0,052 | **0,638 ± 0,053** |

Layer entitas (PER/ORG/LOC/TIME):

| Varian | Seed test formal | Informal tulisan tangan (30) | Informal sintetis (100) | NERGrit test |
|---|---|---|---|---|
| base | **0,803 ± 0,012** | 0,616 ± 0,087 | 0,717 ± 0,019 | 0,705 ± 0,025 |
| norm | 0,804 ± 0,007 | 0,600 ± 0,093 | 0,726 ± 0,024 | 0,712 ± 0,027 |
| aug | 0,792 ± 0,007 | **0,630 ± 0,072** | **0,770 ± 0,010** | **0,720 ± 0,036** |

Temuan:

1. **Augmentasi memperbaiki teks informal sintetis** (CLAIM 0,35 → 0,62–0,64; entitas 0,72 → 0,77) tanpa merusak teks formal (CLAIM formal naik sedikit; entitas formal turun 0,011, sekitar satu simpangan baku). Selisih F1 formal antarvarian CLAIM masih di dalam simpangan baku.
2. **Normalisasi tidak membantu**: selisih dengan `base` ada di dalam simpangan baku pada semua uji. Normalisasi berbasis kamus tidak mengubah kata yang membuat model salah.
3. **CLAIM pada kalimat informal tulisan tangan tetap sekitar 0,10 pada semua varian.** Penyebab utama kemungkinan ketidakcocokan panjang span, bukan ejaan: span CLAIM di 30 kalimat tulisan tangan rata-rata 3,6 token (median 3, kalimat 12,6 token), sedangkan di seed set 12,6 token (median 11, kalimat 25,8 token). Pesan pendek diberi span minimal ("diliburin", "bikin mandul", "gak berlaku lg"), model yang dilatih pada span panjang predikat-objek memperluas batas, dan F1 strict menghitung batas salah sebagai salah penuh. Prediksi uji informal belum tersimpan pada run ini (checkpoint dihapus); `ner.train` kini menyimpan prediksi tiap `extra_eval` agar bisa dianalisis, dan hipotesis ini dicek pada run berikutnya (lihat F1 longgar/tumpang tindih di `error_analysis`).
4. Entitas pada informal tulisan tangan naik 0,014 dengan augmentasi, tetapi simpangan baku 0,07–0,09 (n = 30), jadi belum meyakinkan.

Batasan: uji informal sintetis dibuat dengan generator yang sama dengan augmentasi latih (probabilitas dan seed berbeda), jadi kenaikannya sebagian bersifat dalam-distribusi dan tidak membuktikan perbaikan pada pesan nyata. Hanya uji tulisan tangan 30 kalimat yang mewakili teks nyata, dan di sana CLAIM tidak membaik. Pemilihan model memakai F1 dev (rata-rata 3 seed), bukan test:

| Varian | CLAIM dev | Entitas dev |
|---|---|---|
| base | 0,752 ± 0,019 | **0,778 ± 0,003** |
| norm | 0,775 ± 0,015 | 0,782 ± 0,007 |
| aug | 0,761 ± 0,024 | 0,776 ± 0,001 |
| aug3 | **0,802 ± 0,013** | — |

Keputusan: `ner_indobert_claim` memakai `augment` 3 salinan (dev naik 0,05, konsisten dengan test). `ner_indobert_entity` tetap tanpa augmentasi: dev tidak berbeda, dan kenaikannya hanya terlihat di uji sintetis yang sejenis dengan augmentasi latih. Di antara 3 seed `aug3`, seed 42 punya dev tertinggi (0,816), jadi model terpasang tetap seed 42. Checkpoint varian RQ1 sudah dihapus, jadi model CLAIM dilatih ulang lewat `train_ner_colab.ipynb` bagian 5c (cukup layer CLAIM, seed 42); namanya sama (`ner_indobert_claim_seed42`), jadi `pipeline.yaml` tidak berubah.

### Model CLAIM terpasang (2026-11-14)

`ner_indobert_claim_seed42` dilatih ulang dengan augmentasi 3 salinan (`train_ner_colab.ipynb`, `STAGE2 = ["ner_indobert_claim"]`, Drive `hasil_training/ner_20261114-*`). Hasilnya sama persis dengan run RQ1 `aug3` seed 42 (dev 0,816, test 0,618), jadi training ini reprodusibel.

Uji tambahan kini menyimpan prediksi, sehingga dugaan batas span bisa dicek. F1 strict dibandingkan dengan F1 tumpang tindih (span prediksi dihitung benar bila beririsan dengan span gold):

| Uji | F1 strict | F1 tumpang tindih | Benar | Salah batas | Terlewat | Salah deteksi |
|---|---|---|---|---|---|---|
| Seed test formal (99 span) | 0,618 | 0,933 | 64 | 30 | 5 | 9 |
| Informal sintetis (99) | 0,577 | 0,950 | 62 | 33 | 4 | 7 |
| Informal tulisan tangan (28) | 0,164 | 0,870 | 5 | 21 | 2 | 6 |

Model menemukan letak klaim dengan baik di ketiga uji (26 dari 28 klaim tulisan tangan tersentuh). Hampir semua kesalahan adalah batas span. Pada 21 salah batas tulisan tangan:

- **Prediksi lebih panjang (16):** model ikut mengambil keterangan tempat/waktu atau obrolan di akhir ("diliburin seminggu krn polusi", "bakal guncang padang mggu dpn kata bmkg", "... ? ? hati2 ya bun"), atau memotong TIME di tengah ("sblm", "tgl", "jd"). Ini kesalahan model.
- **Subjek bukan entitas ikut dalam prediksi (4 dari 16 di atas):** "vaksin booster bikin mandul", "kuota internet bakal dihapus ...", "garam merk tertentu dicampur plastik ...", "rs boleh nolak pasien bpjs ...". Pedoman (contoh 5) mengeluarkan subjek bukan entitas dari CLAIM, jadi label gold sudah benar dan ini kesalahan model. Seed set train/test formal tidak selalu mengikuti contoh 5 (mis. "seorang kakek penjual keripik ditendang oleh ..."), sehingga model belajar menyertakan subjek.
- **Prediksi lebih pendek atau bergeser (5):** "bagi2 saldo", "tanpa tes", "naik 2x lipat", "gak tilang lg di jalan", "cair hari" (gold "900rb cair").

### Perbaikan label informal dan aturan pangkas CLAIM (2026-11-14)

**Label 30 kalimat tulisan tangan.** Semua 30 kalimat diperiksa ulang terhadap pedoman (bukan hanya kalimat yang salah diprediksi). Ditemukan kesalahan konversi: label informal ditulis sebagai satu layer gabungan, sehingga CLAIM terpotong tepat sebelum entitas (0 dari 30 CLAIM berisi entitas, sedangkan seed test 62 dari 100), padahal sejak [0012](decisions/0012-layer-entitas-dan-claim-terpisah.md) entitas di dalam klaim tetap bagian CLAIM. Koreksi di `review_patches.json` (8 kalimat):

| id | Sebelum | Sesudah | Alasan |
|---|---|---|---|
| informal-009 | padamin listrik | padamin listrik se-jawa bali | objek terpotong entitas |
| informal-011 | bakal guncang | bakal guncang padang | objek terpotong entitas |
| informal-026 | boleh nolak pasien | boleh nolak pasien bpjs | objek terpotong entitas |
| informal-027 | bkl tenggelamin | bkl tenggelamin jakarta utara | objek terpotong entitas |
| informal-020 | 900rb cair | cair | span mulai dari predikat (aturan 1, contoh 5) |
| informal-013 | umumin pajak motor naik 2x lipat | naik 2x lipat | kata pelapor tidak masuk, seperti "bilang"/"nyatain" di kalimat 007, 019, 026 |
| informal-014 | gak tilang lg di jalan , semua pke kamera etle | gak tilang lg | satu CLAIM per kalimat (aturan 2); keterangan tempat tidak masuk (contoh 4) |
| informal-010 | tanpa tes , cukup transfer biaya admin 250rb | tanpa tes | satu CLAIM per kalimat; permintaan tidak masuk (contoh 6) |

Catatan bias: koreksi ini dibuat setelah melihat prediksi model. Dua koreksi (013, 010) kini sama persis dengan prediksi, empat koreksi objek justru menjauhkan gold dari prediksi. F1 strict model CLAIM pada uji tulisan tangan: 0,164 → 0,230 karena label saja.

**Aturan pangkas (`ner/claim_trim.py`).** Pasca-proses span CLAIM memakai entitas prediksi model entitas, diturunkan dari pedoman, lalu dipilih di dev (model CLAIM dan entitas seed 42; F1 strict CLAIM):

| Aturan | Dev | Test | Tulisan tangan | Sintetis |
|---|---|---|---|---|
| tanpa | 0,816 | 0,618 | 0,230 | 0,577 |
| time (potong di TIME + kata depan waktu) | 0,816 | 0,638 | 0,361 | 0,595 |
| punct (berhenti di "?"/"!") | 0,816 | 0,618 | 0,230 | 0,577 |
| attribution ("kata"/"menurut" + entitas) | 0,816 | 0,618 | 0,230 | 0,577 |
| loc ("di <LOC>" di ujung) | 0,796 | 0,580 | 0,295 | 0,540 |
| **time + punct (dipakai)** | **0,816** | **0,638** | **0,361** | **0,595** |
| semua | 0,796 | 0,599 | 0,426 | 0,558 |

Tidak ada aturan yang menaikkan dev (span dev formal jarang memuat TIME), jadi pemilihannya: aturan dari pedoman yang tidak menurunkan dev. `loc` ditolak karena menurunkan dev dan test formal walaupun menaikkan tulisan tangan. `punct` tidak mengubah F1 tetapi membuang obrolan dari `core_claim` ("... ? ? hati2 ya bun"). `NERPredictor` menjalankan `time + punct` secara default bila memakai `claim_checkpoint` (`trim_claim=False` untuk mematikan).

Batasan: uji tulisan tangan sudah dipakai untuk error analysis, jadi angkanya bukan lagi uji yang benar-benar terpisah; angka test formal dan dev yang menjadi dasar keputusan. Kesalahan yang tersisa di tulisan tangan: model entitas melewatkan TIME informal ("mggu dpn", "skrg", "per bulan depan"), subjek bukan entitas ikut CLAIM, dan atribusi tanpa entitas ("kata bmkg").
