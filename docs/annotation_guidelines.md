# Annotation Guidelines

Pedoman anotasi untuk label **CLAIM** dan **TIME** (task 1) serta **pasangan stance** (task 3b). Status: **draf v0.1**, direvisi setelah kalibrasi 100 kalimat. Pemilik: Jonathan.

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

Contoh 5 dan 10 adalah kasus batas yang perlu dibahas saat kalibrasi.

### Proses

1. **Kalibrasi:** ketiga anggota menganotasi 100 kalimat yang sama. Hitung Cohen's kappa per label (pasangan annotator, lalu rata-rata).
2. Bila kappa CLAIM **< 0,6**: bahas kasus berbeda, revisi pedoman ini (naikkan versi), ulangi pada sampel baru.
3. Bila **≥ 0,6**: sisa kalimat dibagi, dua annotator per kalimat.
4. **Adjudikasi:** perbedaan diselesaikan oleh annotator ketiga; hasil di `data/annotations/ner/seed_adjudicated.conll`.
5. Catat setiap perubahan aturan di bagian *Riwayat* di bawah.

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
| 0.1 | 2026-11-11 | Draf awal |
