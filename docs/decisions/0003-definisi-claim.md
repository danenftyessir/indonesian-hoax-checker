# 0003 — Definisi CLAIM

**Status:** Final

## Konteks
Belum diputuskan apakah CLAIM berupa span atau label kalimat.

## Keputusan
CLAIM adalah **span**: frasa predikat-objek yang bisa dicek (contoh: "menghentikan seluruh bantuan Makan Bergizi Gratis"), **maksimal satu per kalimat**, tanpa subjek yang sudah berlabel entitas.

## Alasan
Span bisa langsung dipakai sebagai kueri retrieval; label kalimat hanya mengulang tugas klasifikasi. Cahyo dkk. (2025) memakai label ACT (tindakan) yang serupa dan mencapai F1 87 dengan BERT.

## Konsekuensi
Pedoman anotasi (`docs/annotation_guidelines.md`) harus memuat kasus negasi dan klaim ganda. Untuk input multi-kalimat, `NERResult.core_claim` = span CLAIM pertama ([0011](0011-keputusan-teknis-tambahan.md)).
