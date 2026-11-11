# 0004 — Skema NER dan pemakaian NERP

**Status:** Usulan

## Konteks
NERGrit (PERSON, ORGANISATION, PLACE; cased), NERP (PPL, PLC, IND, EVT, FNB; seluruhnya huruf kecil, 323 kalimat duplikat di train), dan target tim (PER, ORG, LOC, TIME, CLAIM) berbeda skema.

## Keputusan
- Skema final: **PER, ORG, LOC, TIME, CLAIM**, IOB2.
- NERGrit: PERSON→PER, ORGANISATION→ORG, PLACE→LOC; dipakai cased sebagai data utama.
- NERP: PPL→PER, PLC→LOC, IND/EVT/FNB→O; hanya untuk eksperimen lowercase (RQ1), setelah dedup.
- Model: `indobenchmark/indobert-base-p1` (p1, karena NERGrit F1 p1 lebih tinggi di IndoNLU).

## Alasan
Casing adalah fitur penting NER Indonesia. Mencampur NERP yang lowercase ke model cased merusak sinyal itu; justru NERP cocok mensimulasikan teks informal.

## Konsekuensi
TIME dan CLAIM hanya berasal dari anotasi sendiri (seed set, lihat 0007).
