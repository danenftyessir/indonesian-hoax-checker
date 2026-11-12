# 0010 — Klasifikasi dan retrieval dijalankan paralel

**Status:** Final (keputusan tambahan, tidak ada di daftar lama)

## Konteks
Di laporan desain, klasifikasi berjalan sebelum pencarian bukti dan skornya menentukan seberapa dalam pencarian dilakukan.

## Keputusan
Klasifikasi dan retrieval dijalankan paralel di orchestrator, lalu digabung di `pipeline/confidence.py`. Retrieval selalu dijalankan, bukan hanya saat skor hoaks tinggi.

## Alasan
Retrieval hanya butuh teks dan entitas. Menjalankannya selalu memudahkan kasus "berita benar yang mirip hoaks": stance `refutes` terhadap klaim terbantah justru menjadi bukti bahwa input benar.

## Konsekuensi
Laporan dan diagram BEAM perlu disesuaikan satu kalimat.
