# Draf laporan: Task 1 — Named Entity Recognition

Draf bagian NER untuk laporan akhir. Angka berasal dari `reports/tables/ner_*.md` (dibuat ulang dengan `python -m indonesian_hoax_checker.ner.report`) dan `docs/experiment_protocol.md`. F1 = entity-level seqeval strict; ± = simpangan baku 3 seed (13, 42, 2026) kecuali disebut lain.

## 1. Tujuan dan label

NER menandai entitas yang dipakai tahap lain: **PER**, **ORG**, **LOC**, **TIME**, dan **CLAIM** (frasa predikat-objek yang bisa dicek kebenarannya, maksimal satu per kalimat; [0003](../docs/decisions/0003-definisi-claim.md)). `core_claim` (span CLAIM pertama) dipakai sebagai hypothesis pada stance detection; entitas dipakai untuk fitur dan penyempitan kandidat retrieval.

## 2. Data

| Data | Isi | Dipakai untuk |
|---|---|---|
| NERGrit (cased) | train 1.505, dev 167 (dipisah dari train), test 209 kalimat; PER/ORG/LOC | Tahap 1 |
| NERP (dedup, lowercase) | 323 duplikat train dibuang; 5.757 / 640 / 754 kalimat; PER/LOC | Varian lowercase (tidak dipakai model akhir) |
| Seed set CLAIM/TIME | 1.500 kalimat Komdigi dan Mafindo: train 1.430 (label otomatis/silver), dev 50 dan test 100 (dikoreksi manusia) | Tahap 2 |
| Uji informal | 30 kalimat gaya pesan berantai tulisan tim (dikoreksi); seed test versi medsos sintetis (100) | RQ1 |

Label otomatis seed set: PER/ORG/LOC dari model tahap 1, TIME dari aturan (`time_rules.py`), CLAIM dari aturan (`claim_rules.py`: penanda klaim, judul artikel Komdigi, struktur predikat). Test seed = artikel Komdigi sejak 1 Sep 2026 dan Mafindo, terpisah dari train. Pedoman anotasi: `docs/annotation_guidelines.md`.

## 3. Model

- **Baseline:** BiLSTM-CRF (embedding kata + karakter, dilatih dari nol).
- **Model kuat:** IndoBERT-base-p1 token classification.
- **Dua tahap:** tahap 1 di NERGrit, tahap 2 dilanjutkan dari checkpoint tahap 1 di seed set.
- **Dua layer** ([0012](../docs/decisions/0012-layer-entitas-dan-claim-terpisah.md)): model entitas (PER/ORG/LOC/TIME) dan model CLAIM dilatih terpisah, lalu digabung saat prediksi. Pada model satu layer, CLAIM menimpa entitas di dalamnya, sehingga F1 entitas turun tajam.
- **Pasca-proses CLAIM** (`ner/claim_trim.py`): span CLAIM dipotong di TIME (dan kata depan waktunya) serta sebelum "?"/"!", sesuai pedoman.
- Training: lr 3e-5, batch 16, maks 20 epoch, early stopping di dev (patience 5), max_length 256, GPU T4 (Colab).

## 4. Hasil

### 4.1 Tahap 1: NERGrit

| Model | F1 | PER | ORG | LOC |
|---|---|---|---|---|
| BiLSTM-CRF | 0,676 ± 0,004 | 0,676 | 0,469 | 0,750 |
| IndoBERT-base-p1 | **0,783 ± 0,012** | 0,817 | 0,581 | 0,841 |

IndoBERT unggul 10,7 poin; ORG paling sulit untuk keduanya. Gambar: `reports/figures/ner_baseline.png`.

### 4.2 Tahap 2: satu layer vs dua layer (seed test)

| Model | PER | ORG | LOC | TIME | CLAIM | NERGrit test |
|---|---|---|---|---|---|---|
| BiLSTM-CRF satu layer | 0,768 | 0,721 | 0,704 | 0,522 | 0,605 | 0,291 |
| IndoBERT satu layer | 0,847 | 0,700 | 0,823 | 0,683 | 0,567 | 0,420 |
| IndoBERT dua layer | **0,873** | **0,728** | **0,865** | **0,783** | **0,630** | **0,705** |

Dengan dua layer, F1 NERGrit test tidak lagi jatuh (0,420 → 0,705), TIME naik 10 poin, dan CLAIM naik 6 poin. Model dua layer menjadi model terpasang.

### 4.3 RQ1: normalisasi dan augmentasi teks informal

Pertanyaan: apakah normalisasi (kamus slang → baku) dan augmentasi (salinan data latih bergaya medsos: huruf kecil, singkatan, typo, "!!!") memperbaiki NER pada teks informal?

| Layer | Varian | Dev | Test formal | Tulisan tangan (30) | Sintetis |
|---|---|---|---|---|---|
| CLAIM | base | 0,752 | 0,630 ± 0,024 | 0,101 ± 0,021 | 0,348 ± 0,015 |
| CLAIM | norm | 0,775 | 0,629 ± 0,017 | 0,089 ± 0,005 | 0,351 ± 0,018 |
| CLAIM | aug | 0,761 | 0,647 ± 0,029 | 0,091 ± 0,003 | 0,624 ± 0,029 |
| CLAIM | aug3 | **0,802** | **0,654 ± 0,034** | 0,106 ± 0,052 | **0,638 ± 0,053** |
| Entitas | base | 0,778 | **0,803 ± 0,012** | 0,616 ± 0,087 | 0,717 ± 0,019 |
| Entitas | norm | **0,782** | 0,804 ± 0,007 | 0,600 ± 0,093 | 0,726 ± 0,024 |
| Entitas | aug | 0,776 | 0,792 ± 0,007 | 0,630 ± 0,072 | 0,770 ± 0,010 |

Jawaban RQ1:

1. **Augmentasi** menaikkan F1 pada teks informal sintetis (CLAIM 0,35 → 0,64; entitas 0,72 → 0,77) tanpa menurunkan teks formal. Namun uji sintetis dibuat dengan generator yang sama, jadi sebagian kenaikan bersifat dalam-distribusi.
2. **Normalisasi tidak membantu**: semua selisih di dalam simpangan baku.
3. Pada **30 kalimat tulisan tangan**, CLAIM tetap sekitar 0,10 di semua varian. Penyebabnya dibahas di §5.

Varian dipilih dari dev: CLAIM memakai **aug3** (dev +0,05), entitas tetap **base** (dev tidak berbeda; kenaikan hanya di uji sintetis). Gambar: `reports/figures/ner_rq1_claim.png`, `ner_rq1_entity.png`.

### 4.4 Model terpasang

Entitas: IndoBERT dua layer `base` seed 42. CLAIM: `aug3` seed 42 + pasca-proses.

| | Dev | Test formal | Sintetis |
|---|---|---|---|
| CLAIM tanpa pasca-proses | 0,816 | 0,618 | 0,577 |
| CLAIM + pasca-proses (time, punct) | 0,816 | **0,647** | **0,595** |

Pasca-proses dipilih di dev (tidak menurunkan dev). Aturan lain yang diuji dan ditolak karena menurunkan dev: buang keterangan tempat "di <LOC>" (dev 0,796) dan buang subjek bukan entitas (dev 0,673).

### 4.5 Dampak NER ke retrieval

Pada 351 kueri kalimat klaim Komdigi split test (BM25, indeks judul): kueri teks penuh success@1 **0,841**; `core_claim` saja 0,735; `core_claim` + entitas 0,786. Filter keras entitas menurunkan success@1 ke 0,746 (klaim emas tersingkir pada 17% kueri); boost skor bila berbagi entitas tidak berpengaruh (0,838). Karena itu pipeline memakai **teks penuh** sebagai kueri retrieval dan `core_claim` sebagai hypothesis stance.

## 5. Error analysis

**Batas span, bukan deteksi.** Model CLAIM menemukan letak klaim dengan baik; yang meleset adalah batasnya (`reports/figures/ner_claim_strict_vs_overlap.png`):

| Uji | F1 strict | F1 tumpang tindih |
|---|---|---|
| Seed test formal | 0,618 | 0,933 |
| Informal sintetis | 0,577 | 0,950 |
| Tulisan tangan (30) | 0,164 | 0,870 |

Pada tulisan tangan, 26 dari 28 klaim tersentuh, tetapi 21 salah batas: 16 prediksi lebih panjang (keterangan waktu/tempat dan obrolan di akhir: "bakal guncang padang mggu dpn kata bmkg", "... ? ? hati2 ya bun"), 5 lebih pendek atau bergeser. Span gold tulisan tangan rata-rata 3,6 token, sedangkan seed set 12,6 token.

**Label yang tidak konsisten.** Ditemukan dua masalah label:

- Label 30 kalimat tulisan tangan ditulis sebagai satu layer, sehingga CLAIM terpotong tepat sebelum entitas ("bakal guncang | padang"). 8 label diperbaiki sesuai pedoman (F1 strict 0,164 → 0,230 karena label saja). Perbaikan dibuat setelah melihat prediksi, jadi uji ini tidak lagi independen.
- Pedoman (contoh 5) mengeluarkan subjek bukan entitas dari CLAIM, tetapi label seed set formal sering menyertakannya ("seorang kakek penjual keripik ditendang oleh ..."). Aturan "buang subjek" menurunkan dev 0,14, yang mengonfirmasi ketidakkonsistenan ini. Model belajar dari label tersebut, sehingga ia ikut menyertakan subjek pada teks informal.

**Kasus demo** (`data/demo_cases.txt`, 8 kategori; hasil lengkap `reports/tables/ner_demo_cases.md`):

| Kategori | Contoh | Masalah |
|---|---|---|
| Informal | "pemerintah stop bansos mulai bsk gaes!!!" | CLAIM salah letak ("mulai bsk gaes") |
| Informal | "kemenag bagi2 dana hibah arab saudi gaes" | "kemenag" huruf kecil tidak dikenali ORG; "gaes" ikut CLAIM |
| Entitas | "Kementerian Agama", "Bank Indonesia" | Batas ORG terpotong ("Kementerian", "Indonesia") |
| Entitas | "PLN", "Kominfo" | Singkatan lembaga terlewat |
| TIME | "hingga akhir tahun", "selama libur Natal" | TIME tidak dikenali, sehingga ikut ke CLAIM |
| Satir | "kucing ... dilantik jadi ketua RT ... (satir)" | NER tetap memberi CLAIM; penanda satir harus ditangani tahap lain |
| Campur bahasa | "the government says bansos will be stopped next month" | CLAIM berbahasa Inggris ikut terambil, entitas tidak |
| Penyangkalan | "Tidak benar bahwa air nanas panas dapat menyembuhkan kanker" | CLAIM tanpa "tidak benar", sesuai pedoman; stance yang menilai arahnya |

Yang berjalan baik: klaim Komdigi asli dan parafrasanya, negasi di dalam klaim ("tidak membatasi pembelian BBM"), perubahan angka (Rp13 ribu vs Rp15 ribu tetap di CLAIM, tanggal di TIME), input kosong dan sangat pendek (tanpa entitas), serta teks beberapa paragraf (CLAIM diambil dari kalimat yang tepat).

## 6. Keterbatasan

- CLAIM pada teks informal nyata masih lemah; augmentasi hanya terbukti pada data sintetis.
- Data latih tahap 2 berlabel otomatis (silver); hanya dev/test yang diperiksa manusia, oleh satu annotator (tanpa kappa).
- Uji tulisan tangan hanya 30 kalimat dan sudah dipakai untuk error analysis.
- Angka model terpasang dengan pasca-proses berasal dari satu seed (42); perbandingan 3 seed ada di §4.3.
- Perbaikan yang lebih besar memerlukan pelabelan ulang seed set agar konsisten soal subjek, lalu training ulang.
