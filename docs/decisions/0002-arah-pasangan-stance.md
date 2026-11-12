# 0002 — Arah pasangan stance

**Status:** Final

## Konteks
Laporan mendefinisikan `supports` sebagai "input mengulang klaim yang sudah dibantah" (kredibilitas rendah). `configs/labels.yaml` versi awal memakai premise = klarifikasi, sehingga entailment berarti input benar: kebalikan dari definisi itu.

## Keputusan
- premise = **klaim yang dibantah** (ekstraksi dari Komdigi/Mafindo), hypothesis = **klaim input**.
- `supports` = input mengulang klaim terbantah (→ kredibilitas rendah), `refutes` = input menyangkalnya, `unrelated`.
- IndoNLI: entailment → supports, contradiction → refutes, neutral → unrelated.
- Data latih tambahan dari dataset BRIN Mafindo: Title vs Fact berlabel `refutes`, parafrasa judul berlabel `supports`.

## Alasan
Konsisten dengan laporan dan contoh slide ("stance: supports → low credibility").

## Konsekuensi
`configs/labels.yaml` dan `docs/label_mapping.md` sudah diperbarui. Parafrasa untuk pasangan `supports` ditulis manual ([0011](0011-keputusan-teknis-tambahan.md)).
