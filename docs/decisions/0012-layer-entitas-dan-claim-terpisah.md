# 0012 — Layer entitas dan CLAIM dilatih terpisah

**Status:** Final (usulan 2026-11-12, mengubah [0004](0004-skema-ner-dan-nerp.md) dan [0007](0007-seed-set-dan-annotator.md) bagian pelabelan)

## Konteks
Pada label IOB2 tunggal, CLAIM diberi prioritas dan menimpa PER/ORG/LOC/TIME di dalam span klaim. Pada seed set, 1.649 dari sekitar 3.400 entitas PER/ORG/LOC di train (49%), 64 dari 118 di dev, dan 101 dari 250 di test tertimpa. Model gabungan belajar bahwa isi klaim tidak berisi entitas, padahal RQ3 memakai entitas untuk menyempitkan kandidat retrieval. Pemeriksaan ulang 150 kalimat dev/test juga menemukan 74 kalimat dengan label yang perlu diperbaiki (batas CLAIM atau entitas); kesepakatan aturan CLAIM dengan hasil pemeriksaan: F1 span-exact 0,83.

## Keputusan
- Tiap kalimat seed menyimpan dua layer: `entity_tags` (PER, ORG, LOC, TIME; entitas lengkap) dan `claim_tags` (hanya CLAIM). Label gabungan `tags` tetap ada untuk model gabungan sebagai pembanding.
- Dua model IndoBERT-p1 dilatih dari model NERGrit: `ner_indobert_entity` dan `ner_indobert_claim`. `NERPredictor(claim_checkpoint=...)` menggabungkan keluaran keduanya ke satu `NERResult`; entitas dan CLAIM boleh tumpang tindih. Kontrak `schemas.py` tidak berubah.
- Aturan CLAIM diperbaiki (`ner/claim_rules.py`): kata pengantar "klaim/informasi bahwa", "adanya" dibuang, kalimat klarifikasi ("Faktanya") dipotong, kurung penutup dijaga, "terpilih", "meter", dan satuan angka bukan predikat, batas posisi predikat dinaikkan ke 24 token.
- Koreksi dev/test disimpan di `data/annotations/ner/review_patches.json` dan diterapkan dengan `annotation patch`, supaya bisa ditinjau dan diulang.

## Alasan
Memisahkan layer mempertahankan entitas untuk retrieval dan membuat evaluasi CLAIM tidak bercampur dengan evaluasi entitas.

## Konsekuensi
Model gabungan (`ner_indobert_claim_time`) tetap dilatih sebagai pembanding. Bila model terpisah tidak lebih baik pada F1 CLAIM, F1 entitas, `nergrit_test`, dan `seed_informal_test`, keputusan ini ditinjau ulang. Angka label otomatis vs pemeriksaan hanya pada 150 kalimat dev/test; label train tetap silver.
