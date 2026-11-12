# 0008 — Near-duplicate dan pembekuan test set

**Status:** Final

## Konteks
Hoaks berulang (tautan bansos, BLT, lowongan kerja) sangat sering di Komdigi, jadi split waktu saja tidak mencegah kebocoran.

## Keputusan
- Sebelum split waktu: cosine TF-IDF karakter 3–5-gram antar judul Komdigi; pasangan ≥ 0,8 digabung ke satu klaster. Seluruh klaster masuk split yang sama, ditentukan tanggal terbarunya.
- Ambang 0,8 diverifikasi manual pada 50 pasangan sebelum dipakai.
- Test set = artikel sejak **1 September 2026** (sekitar 340 klaim), dibekukan **18 November 2026**.
- Simpan hash file split di `data/splits/`.

## Konsekuensi
Unduh ulang Kaggle tepat sebelum pembekuan dan catat nomor versinya di `docs/dataset_documentation.md`. Semua ablasi dinilai pada test set yang sama.
