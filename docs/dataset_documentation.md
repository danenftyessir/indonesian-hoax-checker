# Dataset Documentation

Sumber angka: *Resolusi Diskusi dan Audit Dataset* (11 Nov 2026). Angka NER dan IndoNLI dihitung dari file mentah; angka Kaggle dan Mendeley dari halaman dataset. Angka bertanda **(cek ulang)** harus diverifikasi setelah tim mengunduh sendiri.

Dataset tidak di-commit (`data/raw/`, `data/interim/`, `data/processed/` ada di `.gitignore`). Anotasi sendiri di `data/annotations/` dilacak Git.

## Ringkasan per task

| Task | Latih | Validasi / uji in-domain | Uji buatan sendiri |
|---|---|---|---|
| 1 NER | NERGrit train (1.672, cased) + seed ±1.350 (label otomatis); NERP dedup (6.397) hanya varian lowercase | NERGrit valid (209), seed dev 50 | Seed test 100 (dikoreksi manual) + 30 kalimat informal |
| 2 Klasifikasi | Mafindo-2024 train (23.641) | Mafindo-2024 test (5.911) | Mendeley 600, 300 klaim pendek, 200 klaim Komdigi netral |
| 3a Retrieval | — (BM25 / embedding siap pakai) | Basis Komdigi 3.043 klaim | 300 kueri: 100 judul asli, 100 parafrasa, 100 tanpa padanan |
| 3b Stance | IndoNLI train (10.330) + pasangan BRIN | IndoNLI val (2.197), test_expert (2.984) | 300 pasangan beranotasi tim |

## Dataset utama

### Indonesian Hoax News Dataset (Kaggle, ireddragonicy) — basis klaim terbantah

- **URL:** https://www.kaggle.com/datasets/ireddragonicy/indonesian-hoax-news-dataset
- **Lisensi:** Apache 2.0. Diperbarui harian dari API resmi Komdigi.
- **Ukuran:** 3.043 record (versi 396, 10 Nov 2026), artikel 28 Des 2024 – 10 Nov 2026. File `komdigi_hoaks.csv`, 10,07 MB. **(cek ulang)**
- **Kolom:** `id`, `url`, `title`, `slug`, `published_at`, `view_count`, `excerpt`, `body_html`, `body_text`, `main_image_url`, `category`, `tags`, `topics`.
- **Catatan:**
  - Angka "15.000 record" di deskripsi tidak berlaku lagi (sisa portal lama kominfo.go.id).
  - Isi adalah **klarifikasi**, bukan teks hoaks. Klaim harus diekstraksi dari judul `[HOAKS] ...` dan kalimat "Beredar ... yang mengeklaim ...".
  - `excerpt` kosong 61%; kategori tunggal "Klarifikasi Hoaks".
  - `tags`/`topics` dipakai untuk hasil per kategori (kesehatan, politik, penipuan tautan). Dari 50 judul terbaru: 9 penipuan "Tautan ...", sekitar 10 foto/video/audio rekayasa AI.
  - Butuh login Kaggle untuk mengunduh.
- **Unduhan tercatat:** 11 Nov 2026 (file bertanggal 10 Nov 2026), versi Kaggle `___` (isi dari halaman dataset). 3.048 record, `published_at` 27 Des 2024 – 9 Nov 2026, 343 artikel sejak 1 Sep 2026. File diawali BOM (baca dengan `utf-8-sig`). Unduh ulang dan perbarui baris ini saat pembekuan test set.

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
- **Catatan:** berita benar dari Tempo, Kompas, dll.; hoaks dari CekFakta/turnbackhoax. Kelas hoaks adalah artikel cek fakta, bukan teks hoaks asli → risiko konfound sumber ([0001](decisions/0001-sumber-kelas-valid.md)). Deskripsi menyebut >15k tapi file 29,6k baris. Tidak ada tanggal. Periksa duplikat judul.

### Indonesian Hoax News Detection Dataset (Mendeley, Rahutomo dkk.) — uji lintas-domain

- **URL:** https://data.mendeley.com/datasets/p3hfgr5j3m
- **Lisensi:** CC BY 4.0.
- **Ukuran:** 600 berita (228 hoaks, 372 valid), tahun 2017, dipublikasikan 15 Okt 2018; `600 news with valid hoax label.csv`. **(cek ulang)**
- **Catatan:** terlalu kecil dan tua untuk fine-tuning; hanya dipakai sebagai set uji lintas-domain.

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
