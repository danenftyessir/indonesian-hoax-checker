# Annotation Guidelines

Pedoman anotasi untuk label **CLAIM** dan **TIME** (task 1) serta **pasangan stance** (task 3b). Versi: **1.1**. Seed set dibuat dengan **label otomatis lalu dikoreksi satu annotator** (Jonathan); lihat [0007](decisions/0007-seed-set-dan-annotator.md). Pemilik: Jonathan.

Contoh kalimat di dokumen ini dibuat untuk ilustrasi, bukan kutipan dari dataset.

## Bagian A — NER (CLAIM dan TIME)

### Label

Skema IOB2. Tipe: `PER`, `ORG`, `LOC`, `TIME`, `CLAIM`. PER/ORG/LOC mengikuti konvensi NERGrit; dokumen ini fokus pada dua label baru.

### CLAIM

**Definisi:** frasa predikat-objek yang **bisa dicek kebenarannya**, yaitu apa yang dikatakan terjadi atau akan terjadi ([0003](decisions/0003-definisi-claim.md)).

Aturan:

1. **Span, bukan kalimat.** Mulai dari predikat (kata kerja utama), sampai akhir objek/pelengkapnya.
2. **Maksimal satu CLAIM per kalimat.** Jika ada lebih dari satu, pilih yang menjadi inti berita (biasanya yang dibantah).
3. **Subjek tidak masuk** bila sudah berlabel PER/ORG/LOC.
4. **TIME tidak masuk** ke dalam CLAIM; beri label TIME sendiri.
5. **Negasi ikut masuk** bila menjadi bagian dari isi klaim ("tidak akan mencairkan ...").
6. **Kata pengantar tidak masuk:** "beredar", "viral", "disebutkan bahwa", "mengeklaim", "katanya".
7. Kalimat tanpa klaim yang bisa dicek (opini, sapaan, ajakan) → tanpa CLAIM.

### TIME

**Definisi:** ungkapan waktu, absolut atau relatif.

- Masuk: tanggal ("1 Desember 2026"), hari, bulan, tahun, jam, dan ungkapan relatif ("bulan depan", "besok", "mulai pekan ini").
- Kata depan seperti "pada", "mulai", "sejak" **tidak masuk** span.
- Durasi tanpa titik waktu ("selama tiga hari") → masuk TIME.

### Contoh

| # | Kalimat dan anotasi | Catatan |
|---|---|---|
| 1 | **[Pemerintah]ORG** **[menghentikan seluruh bantuan Makan Bergizi Gratis]CLAIM** mulai **[bulan depan]TIME** | Contoh laporan desain |
| 2 | Beredar pesan bahwa **[Kominfo]ORG** **[akan memutus internet nasional]CLAIM** **[selama tiga hari]TIME** | "Beredar pesan bahwa" tidak masuk |
| 3 | **[Bank Indonesia]ORG** **[tidak akan menerbitkan uang pecahan Rp1.000 lagi]CLAIM** | Negasi bagian dari klaim |
| 4 | Video yang mengeklaim **[Presiden]PER** **[meresmikan bandara baru]CLAIM** di **[Bali]LOC** | "yang mengeklaim" tidak masuk |
| 5 | Harga Pertalite **[naik menjadi Rp13 ribu per liter]CLAIM** mulai **[1 Desember]TIME** | Subjek bukan entitas tetap di luar CLAIM; span mulai dari predikat |
| 6 | **[BPJS Kesehatan]ORG** **[membagikan saldo Rp500 ribu lewat tautan pendaftaran]CLAIM** dan meminta data KTP | Klaim ganda: pilih yang inti |
| 7 | Faktanya, **[Kementerian Kesehatan]ORG** **[tidak pernah mengeluarkan surat edaran tersebut]CLAIM** | Kalimat klarifikasi tetap dianotasi |
| 8 | Ayo sebarkan ke grup keluarga! | Tanpa CLAIM |
| 9 | pemerintah **[stop bansos]CLAIM** **[bsk]TIME** gaes!!! | Teks informal: singkatan tetap dilabel |
| 10 | Menurut **[Komdigi]ORG**, informasi tersebut tidak benar | Pernyataan bantahan umum, bukan klaim spesifik → tanpa CLAIM |

Contoh 5 dan 10 adalah kasus batas; ikuti anotasi di tabel ini.

### Alur: label otomatis, lalu koreksi manual

Label Studio dijalankan lokal (`pipx install label-studio`, lalu `label-studio start`, buka `http://localhost:8080`).

1. **Pilih kalimat:** `python -m indonesian_hoax_checker.ner.seed_sentences`
   Menulis `seed_sentences.jsonl` dengan field `split`: test 100 (60 Komdigi sejak 1 Sep 2026 + 40 Mafindo), dev 50, dan kandidat train 1.545. Tiap kalimat menyimpan `title` (judul artikel) sebagai petunjuk CLAIM. Butuh CSV Komdigi dari Kaggle (`configs/data_sources.yaml`).
2. **Label otomatis semua kalimat:**
   `python -m indonesian_hoax_checker.ner.annotation auto --checkpoint artifacts/checkpoints/ner/ner_indobert_seed42`
   PER/ORG/LOC dari model NER tahap 1, TIME dari `ner/time_rules.py`, CLAIM dari `ner/claim_rules.py` (penanda klaim, judul artikel Komdigi, dan struktur predikat; lihat komentar di file itu). Hasil: `seed_auto.jsonl`.
3. **Buat tugas koreksi:** `python -m indonesian_hoax_checker.ner.annotation tasks`
   Hanya kalimat dev dan test (150) yang masuk `label_studio_tasks.json`; train tidak dikoreksi.
4. **Koreksi di Label Studio:** buat proyek, tempel `configs/annotation/label_studio_ner.xml` di *Labeling Setup*, impor `label_studio_tasks.json`. Label otomatis tampil sebagai prediksi; perbaiki yang salah (tambah, hapus, geser batas) sesuai Bagian A, lalu **Submit** tiap kalimat. Jangan menerima label otomatis tanpa membaca kalimatnya.
5. **Ekspor** hasil sebagai JSON ke `data/annotations/ner/export_<tanggal>.json`.

### Proses setelah koreksi

1. **Laporan koreksi:** `python -m indonesian_hoax_checker.ner.annotation report --export <ekspor>`
   Membandingkan label otomatis dengan hasil koreksi (hasil koreksi = kunci jawaban): persentase kalimat yang tidak diubah, dan precision/recall/F1 label otomatis per tipe. Angka ini menjadi perkiraan kualitas data latih (silver) dan dilaporkan sebagai bagian dari eksperimen.
2. **Gabungkan:** `python -m indonesian_hoax_checker.ner.annotation build --export <ekspor>`
   Bila semua 150 kalimat dev/test sudah diperiksa dan tidak ada yang diubah, ekspor tidak diperlukan: `build --accept-auto` memakai label otomatis apa adanya sebagai `quality: human`.
   Hasil `seed_adjudicated.jsonl`: dev/test dari hasil koreksi (`quality: human`), train dari label otomatis (`quality: silver`). Kalimat train yang tidak mendapat CLAIM dari aturan dibuang (label O di kalimat itu kemungkinan besar salah). Kalimat dev/test yang belum dikoreksi tidak dipakai dan dicetak sebagai peringatan.
3. **Split:** `python -m indonesian_hoax_checker.ner.annotation split` menulis `seed_{train,dev,test}.conll` (dan `seed_informal_test.conll` bila ada kalimat informal). Dijalankan otomatis di notebook Colab.
4. Bila ada beberapa annotator, `kappa --export <ekspor> [<ekspor lain> ...]` dan `build --adjudicator` tetap tersedia.
5. Catat setiap perubahan aturan di bagian *Riwayat* di bawah.

**Batasan:** dengan satu annotator, tidak ada kesepakatan antar-annotator (kappa). Ini dicatat sebagai keterbatasan di laporan.

## Bagian B — Pasangan stance

Arah pasangan ([0002](decisions/0002-arah-pasangan-stance.md)): **premise** = klaim yang sudah dibantah, **hypothesis** = klaim input.

| Label | Kapan dipakai | Contoh hypothesis (premise: "Pemerintah menghentikan bantuan MBG bulan depan") |
|---|---|---|
| `supports` | Hypothesis mengulang atau memparafrasakan klaim terbantah | "Program makan bergizi gratis disetop mulai bulan depan" |
| `refutes` | Hypothesis menyangkal klaim terbantah | "Bantuan MBG tetap berjalan, tidak ada penghentian" |
| `unrelated` | Topik berbeda, atau hanya berbagi entitas | "Pemerintah meresmikan bendungan baru di Jawa Barat" |

Aturan:

- Nilai hanya arah hypothesis terhadap premise, **bukan** kebenaran faktual.
- Perubahan angka, tanggal, atau entitas yang mengubah isi klaim → `unrelated`, kecuali jelas menyangkal.
- Target: 300 pasangan (100 per label), diverifikasi dua orang; simpan di `data/annotations/claim_verification/`.

## Riwayat

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0 | 2026-11-11 | Versi pertama |
| 1.1 | 2026-11-12 | Label otomatis + koreksi satu annotator; seed 1.500 kalimat |
