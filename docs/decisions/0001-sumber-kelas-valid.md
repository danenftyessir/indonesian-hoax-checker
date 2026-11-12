# 0001 — Sumber kelas valid dan data latih klasifikasi

**Status:** Final

## Konteks
Mendeley (Rahutomo dkk.) hanya 600 berita tahun 2017 (228 hoaks, 372 valid), terlalu kecil untuk fine-tuning IndoBERT. Semua dataset hoaks Indonesia yang besar memuat artikel cek fakta sebagai kelas hoaks, bukan teks hoaks aslinya, sehingga classifier bisa belajar membedakan sumber atau gaya, bukan kebenaran.

## Keputusan
- Latih pada **Mafindo-2024** (`umn-cyber/mafindo-2024`, 29.552 artikel, MIT): berita Tempo/Kompas sebagai `valid`, artikel Mafindo sebagai `hoax`.
- Uji lintas-sumber: **Mendeley 600**, **300 klaim pendek** (`data/short_claims.jsonl`), dan **200 klaim Komdigi** yang ditulis ulang netral tanpa kata "klarifikasi", "faktanya", "[HOAKS]".

## Alasan
Konfound sumber harus diukur, bukan diabaikan. Selisih besar antara F1 in-domain dan lintas-sumber adalah bukti konfound dan masuk error analysis.

## Konsekuensi
F1 in-domain dan lintas-sumber selalu dilaporkan berdampingan. Mafindo-2024 tidak punya tanggal, jadi split-nya stratified setelah dedup judul, bukan split waktu.
