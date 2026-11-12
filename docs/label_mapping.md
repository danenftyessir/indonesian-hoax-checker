# Label Mapping

Sumber: *Resolusi Diskusi dan Audit Dataset* (11 Nov 2026). Nilai mesin ada di `configs/labels.yaml`; dokumen ini menjelaskan alasannya. Status: **Final**.

## Klasifikasi (task 2)

| Label di kode | Istilah di laporan | Arti |
|---|---|---|
| `hoax` | hoaks | Teks mengandung informasi yang terindikasi hoaks |
| `valid` | valid | Tidak terindikasi hoaks |

"Fakta" tidak dipakai karena menyiratkan verifikasi positif yang tidak dilakukan sistem ([0009](decisions/0009-nama-label-klasifikasi.md)).

## NER (task 1)

Skema IOB2, tipe `PER`, `ORG`, `LOC`, `TIME`, `CLAIM` ([0004](decisions/0004-skema-ner-dan-nerp.md)).

| Sumber | Label asli | Label proyek |
|---|---|---|
| NERGrit | PERSON / ORGANISATION / PLACE | PER / ORG / LOC |
| NERP | PPL / PLC | PER / LOC |
| NERP | IND / EVT / FNB | O |
| Anotasi sendiri | — | TIME, CLAIM |

- NERGrit dipakai cased. NERP seluruhnya huruf kecil, jadi hanya masuk varian lowercase (RQ1), setelah 323 kalimat duplikat dibuang.
- `CLAIM` adalah span frasa predikat-objek yang bisa dicek, maksimal satu per kalimat, tanpa subjek yang sudah berlabel entitas ([0003](decisions/0003-definisi-claim.md)).

Contoh:

```
Pemerintah  menghentikan seluruh  bantuan Makan   Bergizi Gratis  mulai bulan  depan
B-ORG       B-CLAIM      I-CLAIM  I-CLAIM I-CLAIM I-CLAIM I-CLAIM O     B-TIME I-TIME
```

## Stance (task 3b)

Arah pasangan ([0002](decisions/0002-arah-pasangan-stance.md)):

- **premise** = klaim yang sudah dibantah (diekstraksi dari Komdigi/Mafindo)
- **hypothesis** = klaim input

| Label | Arti | Dampak ke kredibilitas |
|---|---|---|
| `supports` | Input mengulang klaim yang sudah dibantah | Rendah, sertakan bantahan resmi |
| `refutes` | Input menyangkal klaim yang sudah dibantah | Bukti bahwa input kemungkinan benar |
| `unrelated` | Tidak berkaitan | Tidak ada bukti dari basis klaim |

Pemetaan IndoNLI: `entailment` → `supports`, `contradiction` → `refutes`, `neutral` → `unrelated`.

Versi sebelumnya memakai premise = klarifikasi. Dengan arah itu, entailment berarti input selaras dengan klarifikasi (input benar), kebalikan dari definisi "supports" di laporan. Arah itu tidak dipakai lagi.

Catatan: 53% hipotesis contradiction di IndoNLI train memuat "tidak/bukan/belum". Evaluasi akhir stance harus memakai `test_expert` dan pasangan klaim buatan tim, bukan hanya `test_lay`.
