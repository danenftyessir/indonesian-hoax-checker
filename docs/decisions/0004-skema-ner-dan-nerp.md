# 0004 — Skema NER dan pemakaian NERP

**Status:** Final

## Konteks
NERGrit (PERSON, ORGANISATION, PLACE; cased), NERP (PPL, PLC, IND, EVT, FNB; seluruhnya huruf kecil, 323 kalimat duplikat di train), dan target tim (PER, ORG, LOC, TIME, CLAIM) berbeda skema.

## Keputusan
- Skema final: **PER, ORG, LOC, TIME, CLAIM**, IOB2.
- NERGrit: PERSON→PER, ORGANISATION→ORG, PLACE→LOC; dipakai cased sebagai data utama.
- NERP: PPL→PER, PLC→LOC, IND/EVT/FNB→O; hanya untuk eksperimen lowercase (RQ1), setelah dedup.
- Model: `indobenchmark/indobert-base-p1` (p1, karena NERGrit F1 p1 lebih tinggi di IndoNLU).

## Alasan
Casing adalah fitur penting NER Indonesia. Mencampur NERP yang lowercase ke model cased merusak sinyal itu; justru NERP cocok mensimulasikan teks informal.

Catatan: tokenizer `indobert-base-p1` mengubah teks ke huruf kecil, jadi alasan casing berlaku untuk BiLSTM-CRF (fitur karakter), tidak untuk IndoBERT. Aturan pemakaian NERP tetap sama supaya varian lowercase RQ1 terpisah jelas.

## Konsekuensi
TIME dan CLAIM hanya berasal dari anotasi sendiri (seed set, lihat 0007).
