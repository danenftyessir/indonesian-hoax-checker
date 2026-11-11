# 0005 — Entitas untuk task 2 dan 3

**Status:** Usulan

## Konteks
Melatih task 2 dan 3 dengan entitas gold terlalu optimis, tetapi NER final bergantung pada anotasi CLAIM yang belum ada.

## Keputusan
Pakai entitas **prediksi** dari NER. Selama NER final belum jadi: entitas gold hanya pada seed set, dan model NER baseline cased NERGrit untuk data lain.

## Alasan
Baseline NERGrit tidak bergantung pada anotasi CLAIM, jadi bisa dilatih minggu ini dan task 2 dan 3 bisa mulai dengan entitas prediksi sungguhan.

## Konsekuensi
Evaluasi akhir RQ2 dan RQ3 memakai entitas prediksi dari NER final.
