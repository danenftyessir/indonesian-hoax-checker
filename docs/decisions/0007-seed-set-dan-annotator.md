# 0007 — Ukuran seed set dan annotator

**Status:** Final

## Keputusan
- **Seed set NER:** 400 kalimat = 300 kalimat "Beredar ... yang mengeklaim ..." dari Komdigi (satu per artikel, acak dari seluruh rentang tanggal) + 100 kalimat berita benar. Split 300/50/50.
- **Kalibrasi:** 100 kalimat pertama dianotasi ketiga anggota; hitung Cohen's kappa per label. Lanjut bila kappa CLAIM ≥ 0,6, selain itu revisi pedoman dan ulangi pada sampel baru. Sisanya dua annotator per kalimat.
- **Pasangan stance:** 300 (100 supports parafrasa, 100 refutes dari judul vs Fact, 100 unrelated), diverifikasi dua orang.

## Alasan
Cukup untuk fine-tuning tambahan dan evaluasi; sekitar 1–2 jam per orang per 100 kalimat. Kalimat "Beredar ..." selalu memuat klaim, jadi kepadatan label CLAIM tinggi.

- **Pencegahan kebocoran:** kalimat Komdigi dari artikel periode test (sejak 1 Sep 2026) hanya masuk split test seed set, supaya NER tidak dilatih pada klaim yang diuji di retrieval.

## Konsekuensi
Anotasi disimpan di `data/annotations/` (dilacak Git).
