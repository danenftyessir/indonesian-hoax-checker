# Dataset Documentation

Sumber angka: *Resolusi Diskusi dan Audit Dataset* (11 Nov 2026). Angka NER dan IndoNLI dihitung dari file mentah; angka Kaggle dan Mendeley dari halaman dataset. Angka bertanda **(cek ulang)** harus diverifikasi setelah tim mengunduh sendiri.

Dataset tidak di-commit (`data/raw/`, `data/interim/`, `data/processed/` ada di `.gitignore`). Anotasi sendiri di `data/annotations/` dilacak Git.

## Ringkasan per task

| Task | Latih | Validasi / uji in-domain | Uji buatan sendiri |
|---|---|---|---|
| 1 NER | NERGrit train (1.672, cased) + seed ±1.350 (label otomatis); NERP dedup (6.397) hanya varian lowercase | NERGrit valid (209), seed dev 50 | Seed test 100 (dikoreksi manual) + 30 kalimat informal |
| 2 Klasifikasi | Mafindo-2024 train (23.641) | Mafindo-2024 test (5.911) | Mendeley 600 (label mayoritas penilai: 230 hoaks / 370 valid), 300 klaim pendek, 200 klaim Komdigi netral |
| 3a Retrieval | — (BM25 / embedding siap pakai) | Basis Komdigi 3.048 klaim | 300 kueri: 100 judul asli, 100 parafrasa, 100 tanpa padanan |
| 3b Stance | IndoNLI train (10.330) + pasangan BRIN | IndoNLI val (2.197), test_expert (2.984) | 300 pasangan beranotasi tim |

## Dataset utama

### Indonesian Hoax News Dataset (Kaggle, ireddragonicy) — basis klaim terbantah

- **URL:** https://www.kaggle.com/datasets/ireddragonicy/indonesian-hoax-news-dataset
- **Lisensi:** Apache 2.0. Diperbarui harian dari API resmi Komdigi.
- **Ukuran:** 3.048 record (versi 396, diunduh 11 Nov 2026), artikel 27 Des 2024 – 9 Nov 2026. File `komdigi_hoaks.csv`, 10,07 MB (10.073.039 byte). Jumlah baris dan rentang tanggal dicek ulang dari salinan di Drive tim (3.048 baris, tanggal terawal 2024-12-27, terakhir 2026-11-09).
- **Kolom:** `id`, `url`, `title`, `slug`, `published_at`, `view_count`, `excerpt`, `body_html`, `body_text`, `main_image_url`, `category`, `tags`, `topics`.
- **Catatan:**
  - Angka "15.000 record" di deskripsi tidak berlaku lagi (sisa portal lama kominfo.go.id).
  - Isi adalah **klarifikasi**, bukan teks hoaks. Klaim harus diekstraksi dari judul `[HOAKS] ...` dan kalimat "Beredar ... yang mengeklaim ...".
  - `excerpt` kosong 61%; kategori tunggal "Klarifikasi Hoaks".
  - `tags`/`topics` dipakai untuk hasil per kategori (kesehatan, politik, penipuan tautan). Dari 50 judul terbaru: 9 penipuan "Tautan ...", sekitar 10 foto/video/audio rekayasa AI.
  - Butuh login Kaggle untuk mengunduh.
- **Unduhan tercatat:** versi **396**, diunduh **11 Nov 2026** lewat browser (`archive.zip`, berisi `komdigi_hoaks.csv` 10,07 MB dan `komdigi_hoaks.json`). Hasil `retrieval/corpus.py`: **3.048** record, artikel 27 Des 2024 – 9 Nov 2026, semua judul berprefiks `[HOAKS]`, 0 id ganda, 5 judul persis sama; `claim_sentence` terisi 3.048, `fact_sentence` 2.948 (97%); tag terbanyak `hoaks`, `hoaks hari ini`, `penipuan` (229), `pejabat publik` (74), `kesehatan` (40). Near-duplicate ≥ 0,8: 63 pasangan, 48 klaster ganda, klaster terbesar 4. Split: train 2.398 / dev 299 / test 351. **Unduh ulang dan jalankan lagi saat pembekuan 18 Nov.**

### NERGrit (IndoNLU) — NER utama

- **URL:** https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nergrit_ner-grit/train_preprocess.txt
- **Lisensi:** MIT.
- **Ukuran:** train 1.672 / valid 209 kalimat; 5.321 span (PLACE 2.688, PERSON 1.618, ORGANISATION 1.015); rata-rata 33,6 token, maks 216.
- **Format:** IOB, cased (hanya 9 kalimat huruf kecil semua). Tidak ada duplikat.
- **Catatan:** domain Wikipedia dan berita formal; tidak ada TIME dan CLAIM.

### NERP (IndoNLU) — NER varian lowercase

- **URL:** https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/nerp_ner-prosa/train_preprocess.txt
- **Lisensi:** MIT.
- **Ukuran:** train 6.720 / valid 840 kalimat; 13.853 span (PPL 4.845, PLC 4.379, IND 3.065, EVT 1.022, FNB 542); rata-rata 23,3 token.
- **Catatan:** seluruhnya huruf kecil (6.717 dari 6.720). 323 kalimat duplikat persis di train, 7 di valid; dibuang. Kartu dataset HF salah menyebut label PER/LOC. Nol kalimat tumpang tindih dengan NERGrit.

### Mafindo-2024 (`umn-cyber/mafindo-2024`) — klasifikasi utama

- **URL:** https://huggingface.co/datasets/umn-cyber/mafindo-2024
- **Lisensi:** MIT.
- **Ukuran:** 29.552 artikel (train 23.641, test 5.911); kolom `title`, `content`, `is_hoax`. **(cek ulang)**
- **Catatan:** berita benar dari Tempo, Kompas, dll.; hoaks dari CekFakta/turnbackhoax. Kelas hoaks adalah artikel cek fakta, bukan teks hoaks asli → risiko konfound sumber ([0001](decisions/0001-sumber-kelas-valid.md)). Deskripsi menyebut >15k tapi file 29,6k baris. Tidak ada tanggal.

**Hasil audit shortcut** (`classification/data_audit/mafindo_audit.py`, dijalankan di Colab pada 11 Nov 2026; train 23.641 / test 5.911, proporsi hoaks 0,473 / 0,486):

| Fitur | Kelas valid | Kelas hoaks |
|---|---|---|
| Nama media di isi (detikcom, kompas.com, tempo.co, antara, cnn indonesia) | 42,6% | 3,6% |
| Judul berakhir tanda kutip | 0,8% | 26,7% |
| Isi kurang dari 30 kata | 1,0% | 43,0% |
| Judul huruf kapital semua | 0,2% | 8,4% |
| Hashtag | 0,5% | 6,7% |
| URL | 1,2% | 5,6% |

- Judul duplikat di train: 141. Judul yang muncul di train dan test: 74 (sekitar 1,3% test).
- Macro-F1 pada test: hanya bendera artefak 0,822; TF-IDF isi 0,984; TF-IDF isi dengan nama media, "[SALAH)", URL dan hashtag dibuang 0,984.
- Pembersihan artefak tidak mengubah skor, jadi sinyalnya tersebar di kosakata dan gaya penulisan. Skor 0,984 mengukur pembedaan sumber/gaya, bukan kebenaran klaim, dan tidak boleh dibaca sebagai kemampuan mendeteksi hoaks. Selalu laporkan berdampingan dengan hasil lintas-sumber ([0001](decisions/0001-sumber-kelas-valid.md)).
- Angka tunggal tanpa CI atau seed; belum ada pengecekan manual baris.

### Indonesian Hoax News Detection Dataset (Mendeley, Rahutomo dkk.) — uji lintas-domain

- **URL:** https://data.mendeley.com/datasets/p3hfgr5j3m
- **Lisensi:** CC BY 4.0.
- **Versi dipakai:** 600 berita (`600 news with valid hoax label.csv`) dan lembar pelabelan `600 news labelling process.xlsx` (URL + 3 penilai per artikel, sheet KW1–KW12). Paket unduhan juga memuat versi 250 berita; tidak dipakai.
- **Cara pengumpulan:** 12 topik × 50 artikel, dikumpulkan dari pencarian Google atas 12 klaim viral. Tahun terbaca dari URL hanya untuk 247 dari 600 artikel (238 di antaranya 2014–2018); sisanya tidak terbaca.
- **Skrip:** `classification/data_audit/mendeley_audit.py`, `classification/baseline.py`. Manifest tanpa teks artikel: `data/manifests/mendeley_manifest.csv` (join ke CSV lokal lewat `row_id` atau `text_sha1`).
- **Catatan:** terlalu kecil dan tua untuk fine-tuning; hanya dipakai sebagai set uji lintas-domain.

**Hasil audit** (dijalankan ulang 11 Nov 2026, `python classification/data_audit/mendeley_audit.py --data-dir data/raw/mendeley`):

| Temuan | Nilai |
|---|---|
| Label mayoritas penilai | 370 valid / 230 hoaks. CSV bawaan: 372 / 228 |
| Baris CSV berbeda dari mayoritas | 10, dan 9 di antaranya bertentangan dengan suara bulat 3–0 |
| Kesepakatan penilai | Fleiss kappa 0,585; 70,3% suara bulat |
| URL tercantum dua kali | 31 |
| Artikel dengan near-duplicate (cosine TF-IDF ≥ 0,8, satu topik) | 282 artikel, 417 klaster |
| Proporsi hoaks menurut domain | 16,1% untuk media arus utama, 47,4% untuk domain lain (heuristik kasar) |
| Penanda bantahan (heuristik kata) | 69,2% pada kelas valid, 27,8% pada kelas hoaks |

Label yang dipakai adalah **mayoritas penilai**, bukan kolom `tagging` di CSV.

Kelas `valid` memuat artikel yang membantah hoaks, bukan hanya berita netral. Data Mafindo-2024 sebaliknya memberi label hoaks pada teks bertema hoaks, jadi transfer Mafindo → Mendeley berpotensi bentrok semantik label. Hasil lintas-sumber harus dilaporkan juga pada irisan tanpa artikel bantahan.

**Baseline TF-IDF + regresi logistik** (macro-F1, out-of-fold; `python classification/baseline.py --data-dir data/raw/mendeley`, sklearn 1.6.1):

| Protokol | Macro-F1 | Catatan |
|---|---|---|
| Mayoritas (semua valid) | 0,381 | CI bootstrap [0,366; 0,396] |
| Acak 5-fold | 0,734, sd 0,010 | 20 ulangan, rentang [0,712; 0,756] |
| 5-fold terkelompok per klaster near-duplicate | 0,704, sd 0,012 | 20 ulangan, rentang [0,681; 0,730] |
| Leave-one-topic-out | 0,636 | deterministik; CI bootstrap [0,599; 0,677] |
| Nama domain saja, acak | 0,575, sd 0,012 | 20 ulangan |
| Nama domain saja, leave-one-topic-out | 0,561 | CI bootstrap [0,521; 0,605] |

Cara membaca:
- Kebocoran terbesar adalah **topik yang sama** (0,734 → 0,636 saat topik ditahan). Near-duplicate berpengaruh kecil (0,734 → 0,704).
- Split terkelompok dilaporkan sebagai **rata-rata dan sebaran atas ulangan**. Satu kali jalan `GroupKFold` berubah sekitar ±0,02 hanya karena urutan klaster yang berukuran sama; ini bukan sinyal. Skrip audit sekali jalan memberi 0,709 (pada sklearn 1.6.1), sedangkan catatan awal tim memberi 0,716 (versi pustaka tidak diketahui); keduanya di dalam rentang ulangan (0,681–0,730).
- Nama domain saja sudah melampaui tebakan mayoritas (0,56–0,58): sebagian sinyal adalah sumber, bukan isi.
- CI bootstrap menyatakan ketidakpastian sampel (600 item) dan dihitung dari satu ulangan; sd menyatakan variasi split. Keduanya dilaporkan terpisah.

**Uji cross-dataset Mafindo & Mendeley** (`classification/cross_dataset.py`, dijalankan di Colab, sklearn 1.6.1; TF-IDF kata min_df=3 + Logistic Regression seimbang, hanya `content`, threshold 0,5; satu kali jalan, CI bootstrap atas item uji):

| Uji | Macro-F1 [CI 95%] | AUROC | Hoaks P / R | Hoaks terprediksi / sebenarnya |
|---|---|---|---|---|
| Referensi in-domain: Mafindo → Mafindo test (5.911) | 0,984 [0,981; 0,987] | 0,997 | 0,98 / 0,99 | 0,49 / 0,49 |
| A. Mafindo → Mendeley 600 | 0,569 [0,529; 0,612] | 0,597 | 0,50 / 0,36 | 0,28 / 0,38 |
| B. A, irisan tanpa penanda bantahan (n=280) | 0,493 [0,435; 0,549] | 0,493 | 0,61 / 0,41 | 0,40 / 0,59 |
| B. A, irisan dengan penanda bantahan (n=320) | 0,551 [0,496; 0,610] | 0,569 | 0,29 / 0,25 | 0,17 / 0,20 |
| D. A, artefak sumber dibuang di kedua sisi | 0,572 [0,533; 0,614] | 0,596 | 0,50 / 0,38 | 0,29 / 0,38 |
| C. Mendeley 600 → Mafindo test | 0,464 [0,452; 0,475] | 0,561 | 0,50 / 0,86 | 0,83 / 0,49 |

Cara membaca:
- Skor in-domain 0,984 turun ke 0,569 saat model Mafindo diuji pada Mendeley; AUROC 0,597 hanya sedikit di atas tebakan acak. Skor in-domain mengukur pembedaan sumber/gaya, bukan deteksi hoaks.
- Membuang artefak sumber (D) hampir tidak mengubah hasil (0,572 vs 0,569), konsisten dengan temuan in-domain.
- Dugaan bentrok semantik label (artikel bantahan di Mendeley berlabel valid) **tidak terbukti menjadi penyebab utama**: pada irisan tanpa penanda bantahan AUROC justru 0,493 (CI 0,423–0,560, mencakup 0,5). Penanda bantahan adalah heuristik kata yang kasar dan irisannya kecil, jadi ini bukan bukti bahwa bentrok tidak ada.
- Arah sebaliknya (C) gagal: dilatih pada 600 artikel, model menebak hoaks untuk 83% artikel Mafindo.
- Per topik (n≈50, bising), proporsi hoaks terprediksi berkisar 0,00–0,74 sementara proporsi sebenarnya 0,20–0,54. Ini mengarah ke prediksi yang mengikuti topik/kosakata, tetapi belum diuji lebih lanjut.
- Pembanding dalam Mendeley sendiri: TF-IDF + LR dengan leave-one-topic-out mencapai 0,636, di atas 0,569 dari model Mafindo.
- Batas: satu kali jalan, ambang tetap 0,5, hanya `content`. Mendeley (artikel 2014–2018 dari banyak situs) dan Mafindo (2024, kelas hoaks berisi artikel cek fakta) berbeda domain dan waktu.

**Fine-tuning IndoBERT-p1 pada Mafindo-2024** (`classification/finetune_indobert.py`, Colab T4, 11–12 Nov 2026; 3 seed 13/42/2026, 3 epoch, batch 32, lr 2e-5, panjang maks 256 token, hanya `content`, 10% train sebagai validasi untuk memilih epoch, ambang 0,5; rata-rata ± sd antar seed; angka disalin dari keluaran notebook Colab):

| Tes | TF-IDF + LR (satu kali jalan) | IndoBERT-p1 (3 seed) |
|---|---|---|
| Mafindo test, macro-F1 | 0,984 [0,981; 0,987] | 0,987 ± 0,001 |
| Mafindo test, AUROC | 0,997 | 0,998 ± 0,000 |
| Mendeley 600, macro-F1 | 0,569 [0,529; 0,612] | 0,524 ± 0,029 |
| Mendeley 600, AUROC | 0,597 | 0,585 ± 0,010 |
| Mendeley tanpa penanda bantahan (n=280), macro-F1 | 0,493 | 0,459 ± 0,031 |
| Mendeley tanpa penanda bantahan, AUROC | 0,493 | 0,498 ± 0,007 |
| Mendeley dengan penanda bantahan (n=320), macro-F1 | 0,551 | 0,526 ± 0,012 |

- Fine-tuning tidak memperbaiki transfer lintas-sumber. Skor in-domain hampir sama dengan baseline (0,987 vs 0,984), dan pada Mendeley IndoBERT tidak lebih baik; pada irisan tanpa penanda bantahan AUROC 0,498 ± 0,007, setara tebakan acak. Model yang lebih kuat mempelajari pembeda sumber/gaya yang sama dengan model TF-IDF.
- Perbandingan memakai satu pengaturan hiperparameter (belum disetel), teks dipotong 256 token, dan TF-IDF hanya satu kali jalan. Selisih TF-IDF vs IndoBERT pada Mendeley tidak diuji secara berpasangan, jadi jangan dibaca sebagai bukti IndoBERT lebih buruk.
- Recall hoaks pada Mendeley sangat bervariasi antar seed (0,429 ± 0,120): keputusan pada ambang 0,5 tidak stabil di luar domain.
- Waktu: sekitar 254 detik per epoch pada T4, sekitar 41 menit untuk 3 seed.
- Model tidak disimpan (`--save-dir` tidak dipakai); skor per seed tersimpan di folder hasil Drive untuk kalibrasi dan analisis galat. Probe klaim pendek tidak dijalankan pada run ini.

### IndoNLI — stance (pra-latih)

- **URL:** https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl
- **Lisensi:** CC BY-SA 4.0.
- **Ukuran:** train 10.330 / val 2.197 / test_lay 2.201 / test_expert 2.984. Train: entailment 3.476, neutral 3.415, contradiction 3.439. Premis rata-rata 21,1 kata, hipotesis 7,6 kata.
- **Format:** JSONL `premise`, `hypothesis`, `label` (e/n/c), `annotator_type`, `sentence_size`.
- **Catatan:** **bias negasi** — 53% hipotesis contradiction di train memuat "tidak/bukan/belum" (1.835 dari 3.439), dibanding 22% keseluruhan. test_expert lebih sulit (premis 31 kata, negasi 12,5%). Laporkan test_expert, bukan hanya test_lay.

### nlp-brin-id/fakenews-mafindo (BRIN) — pasangan stance

- **URL:** https://huggingface.co/datasets/nlp-brin-id/fakenews-mafindo
- **Akses:** gated di HF (perlu persetujuan); versi baru di dataverse BRIN (hdl:20.500.12690/RIN/7QBRKQ).
- **Ukuran:** 8.940 baris turnbackhoax 2018–2023; kolom `Title`, `Content`, `Fact`, `References`, `Classification`; label HOAX 1 / NON-HOAX 0.
- **Pemakaian:** pasangan (Title, Fact) sebagai `refutes`, parafrasa judul sebagai `supports`. `Classification` (Fabricated, False Context, Misleading, Satire, ...) untuk error analysis per jenis hoaks.
- **Catatan:** penafian BRIN: representasi NON-HOAX tidak seimbang.

## Dataset cadangan

| Dataset | Pemakaian |
|---|---|
| Rifky/indonesian-hoax-news (HF) | Hanya judul; perluas basis klaim ke sebelum Des 2024 bila perlu (isi harus di-scrape ulang) |
| Indonesian Fact and Hoax Political News (Kaggle) | Bukan data latih (tanpa lisensi); opsional sebagai uji domain politik |
| Cahyo dkk. (2025), NER atas klarifikasi Komdigi | Tidak dipakai sebagai data (lisensi tidak disebut, tim tidak meminta izin); hanya dikutip sebagai literatur |
| IndoLEM NER (NER-UI, NER-UGM) | Opsional bila F1 NERGrit kurang |

Ditolak: COVID-19 Indonesian misinformation tweets (hanya ID tweet), MultiClaim/SemEval-2025 Task 7 (protokol evaluasinya diadopsi, datanya tidak).

## Data buatan sendiri

| File | Isi | Pemilik |
|---|---|---|
| `data/annotations/ner/` | Seed set ±1.500 kalimat CLAIM/TIME: train label otomatis, dev 50 + test 100 dikoreksi manual | Jonathan |
| `data/annotations/claim_verification/` | 300 pasangan stance | Danendra (verifikasi dua orang) |
| `data/short_claims.jsonl` | 300 klaim pendek untuk uji klasifikasi | Dzaky, diverifikasi Danendra |
| `data/demo_cases.txt` | Kasus uji demo CLI | Semua |
| `data/splits/` | Hash file split yang dibekukan | Semua |

## Etika

- Sistem adalah alat bantu penyaringan awal, bukan pemutus akhir.
- Tidak melakukan scraping ulang komdigi.go.id; cukup unduhan Kaggle (Apache 2.0).
- Cek lisensi tiap dataset sebelum mempublikasikan turunan data.
