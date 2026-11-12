# 0007 — Ukuran seed set dan annotator

**Status:** Final (direvisi 2026-11-12)

## Konteks
NERGrit dan NERP tidak punya label CLAIM dan TIME. Rencana awal (400 kalimat, tiga annotator, kalibrasi kappa) membutuhkan waktu Dzaky dan Danendra, sedangkan task NER perlu diselesaikan lebih dulu.

## Keputusan
- **Satu annotator:** Jonathan. Anggota lain tidak ikut menganotasi.
- **Label otomatis untuk semua kalimat:** PER/ORG/LOC dari model NER tahap 1, TIME dari `ner/time_rules.py`, CLAIM dari `ner/claim_rules.py`. CLAIM memakai empat petunjuk berurutan: kata penanda kuat ("mengeklaim bahwa X"), judul artikel Komdigi (judul = ringkasan klaim, dicocokkan termasuk bentuk berimbuhan), penanda lemah ("dengan narasi X"), dan struktur kalimat (kata kerja utama berawalan me-/ber-/di-/ter- atau kata bantu akan/telah/tidak). Sekitar 89% kalimat mendapat CLAIM.
- **Koreksi manual hanya untuk data evaluasi:** dev 50 dan test 100, di Label Studio.
- **Komposisi (±1.500 kalimat terpakai):**

| Split | Jumlah | Isi | Label |
|---|---|---|---|
| train | ±1.350 | Komdigi sebelum 1 Sep 2026 + Mafindo | Otomatis (silver) |
| dev | 50 | 28 Komdigi sebelum 1 Sep 2026 + 22 Mafindo | Dikoreksi manual |
| test | 100 | 60 Komdigi sejak 1 Sep 2026 + 40 Mafindo | Dikoreksi manual |

- Kandidat train dipilih 1.545 kalimat; yang tidak mendapat CLAIM dari aturan dibuang (sekitar 160), karena hampir semua kalimat seed memuat klaim sehingga label O-nya kemungkinan besar salah. Total terpakai sekitar 1.530.
- **Pasangan stance (task 3b) tidak berubah:** 300 pasangan (100 supports parafrasa, 100 refutes dari judul vs Fact, 100 unrelated), diverifikasi dua orang.
- **Pencegahan kebocoran:** kalimat Komdigi periode test (sejak 1 Sep 2026) hanya masuk test; sisanya tidak dipakai.

## Alasan
Label otomatis murah, jadi data latih bisa jauh lebih besar dari 400. Data evaluasi tetap diperiksa manusia supaya F1 mengukur kemampuan model mengenali klaim, bukan kemiripannya dengan aturan.

## Hasil pemeriksaan (2026-11-12)
Jonathan memeriksa 150 kalimat dev/test dan tidak menemukan yang perlu diubah, sehingga label otomatis dipakai apa adanya (`annotation build --accept-auto`). Karena tidak ada koreksi, `annotation report` tidak menghasilkan perbandingan label otomatis vs manusia.

## Konsekuensi
- Tidak ada kappa antar-annotator; dicatat sebagai keterbatasan.
- Kualitas label otomatis diukur dengan `annotation report` (label otomatis vs hasil koreksi pada 150 kalimat) dan dilaporkan.
- Model bisa ikut meniru kesalahan aturan pada pola tertentu; error analysis tahap 2 memeriksa ini.
- Anotasi disimpan di `data/annotations/ner/` (dilacak Git).
